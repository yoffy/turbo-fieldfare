import Darwin
import Foundation

public struct LocalStreamingRepackOptions: Sendable {
    public let sourceDir: String
    public let outputDir: String
    public let requireKnownSource: Bool
    public let rangeChunkBytes: Int
    public let writeTileBytes: Int
    public let minFreeReserveBytes: UInt64
    public let overwrite: Bool

    public init(sourceDir: String,
                outputDir: String,
                requireKnownSource: Bool = true,
                rangeChunkBytes: Int = RemoteChunkPolicy.defaultBytes,
                writeTileBytes: Int = WriterCore.tileBytes,
                minFreeReserveBytes: UInt64 = 1 * 1024 * 1024 * 1024,
                overwrite: Bool = false) {
        self.sourceDir = sourceDir
        self.outputDir = outputDir
        self.requireKnownSource = requireKnownSource
        self.rangeChunkBytes = rangeChunkBytes
        self.writeTileBytes = writeTileBytes
        self.minFreeReserveBytes = minFreeReserveBytes
        self.overwrite = overwrite
    }
}

public struct LocalStreamingRepackResult: Sendable {
    public let outputDir: String
    public let modelID: String?
    public let sourceIndexSHA256: String
    public let rangeRequestCount: Int
    public let copiedBytes: UInt64
    public let outputBytes: UInt64
}

/// Repacks a checkpoint already on local disk into `.gturbo` form. Planning,
/// destination layout, tokenizer sidecars, and manifest encoding match the
/// remote installer so the same source snapshot yields byte-identical
/// `model_weights.bin` / `packed_experts/layer_*.bin` / `layout.json` files.
public final class LocalStreamingRepacker {
    private let options: LocalStreamingRepackOptions
    private let audit: RepackAudit
    private let startTime = Date()

    public init(options: LocalStreamingRepackOptions,
                audit: RepackAudit = RepackAudit()) {
        self.options = options
        self.audit = audit
    }

    public func run(progress: @escaping @Sendable (ModelInstallProgress) -> Void = { _ in }) async throws
        -> LocalStreamingRepackResult {
        try validateOptions()
        let installLock = try InstallLock.acquire(outputDirectory: options.outputDir)
        defer { withExtendedLifetime(installLock) {} }
        let paths = installLock.paths
        if try Posix.entryKind(paths.finalDirectory) == .directory, !options.overwrite {
            throw RepackError.configurationInvalid(detail:
                "output directory already exists: \(paths.finalDirectory)")
        }
        // Local installs do not support resume; refuse to mix with a saved
        // remote download instead of silently reusing it.
        let hasPartial = try Posix.entryKind(paths.partialDirectory) != .absent
        let hasCheckpoint = try Posix.entryKind(paths.checkpointFile) != .absent
        if hasPartial || hasCheckpoint {
            throw RepackError.installStateIncompatible(
                detail: "saved download exists; resume or discard it before a local repack")
        }
        do {
            return try await runPrepared(paths: paths, progress: progress)
        } catch {
            try? FileManager.default.removeItem(atPath: paths.partialDirectory)
            throw error
        }
    }

    private func runPrepared(paths: RemoteInstallPaths,
                             progress: @escaping @Sendable (ModelInstallProgress) -> Void) async throws
        -> LocalStreamingRepackResult {
        try Task.checkCancellation()
        progress(.downloadingMetadata)
        let snapshot = try LocalSnapshotLoader.load(sourceDir: options.sourceDir,
                                                    requireKnownSource: options.requireKnownSource)
        try Task.checkCancellation()
        let plan = try RepackPlanner.plan(meta: snapshot.metadata,
                                          arch: snapshot.arch,
                                          shardHeaders: snapshot.shardHeaders,
                                          outputDir: paths.partialDirectory)
        let rangePlan = try RangeCopyPlanner.plan(repackPlan: plan,
                                                  rangeChunkBytes: options.rangeChunkBytes,
                                                  layoutMode: "identity",
                                                  layoutOrderSha256: nil)
        let outputBytes = plan.resident.totalSize
            + plan.layers.reduce(UInt64(0)) { $0 + $1.fileSize }
            + plan.pleShards.reduce(UInt64(0)) { $0 + $1.fileSize }
        progress(.planning(downloadBytes: rangePlan.remoteBytesToDownload,
                           outputBytes: outputBytes))
        let diskRequirement = try DiskSpaceChecker.requireAvailable(
            path: paths.parentDirectory,
            bytes: outputBytes + UInt64(options.rangeChunkBytes),
            reserveBytes: options.minFreeReserveBytes)
        progress(.checkingDisk(diskRequirement))
        try Task.checkCancellation()

        audit.sourceSnapshotSha256 = snapshot.metadata.indexSha256Hex
        audit.bitWidthOverridesHonored = snapshot.metadata.bitsOverrides.count
        audit.tensorsDroppedMultimodal = plan.excludedMultimodalTensorNames
        audit.packedExpertLayoutMode = "identity"

        progress(.reservingOutput(bytes: outputBytes))
        try createOutputFiles(plan: plan, paths: paths)

        let provider = LocalFileSourceByteProvider(sourceDir: snapshot.sourceDir,
                                                   writeTileBytes: options.writeTileBytes)
        progress(.copyingPayload(reusedBytes: 0,
                                 downloadedThisRunBytes: 0,
                                 totalBytes: rangePlan.remoteBytesToDownload))
        try await provider.copyBatch(
            rangePlan.coalescedCopies,
            completedRangeIDs: Set<String>(),
            partialDirectory: paths.partialDirectory,
            temporaryPath: paths.rangeTemporaryFile,
            audit: audit,
            progress: { copiedBytes in
                progress(.copyingPayload(reusedBytes: 0,
                                         downloadedThisRunBytes: copiedBytes,
                                         totalBytes: rangePlan.remoteBytesToDownload))
            },
            commit: { _ in })

        try recordOutputFile(relativePath: "model_weights.bin",
                             path: plan.resident.path,
                             progress: progress)
        for layer in plan.layers where layer.expertsPerLayer > 0 {
            try Task.checkCancellation()
            let rel = "packed_experts/" + (layer.path as NSString).lastPathComponent
            try recordOutputFile(relativePath: rel, path: layer.path, progress: progress)
        }
        for shard in plan.pleShards {
            try Task.checkCancellation()
            let rel = "packed_ple/" + (shard.path as NSString).lastPathComponent
            try recordOutputFile(relativePath: rel, path: shard.path, progress: progress)
        }

        let layoutPath = ((paths.partialDirectory as NSString)
            .appendingPathComponent("packed_experts") as NSString)
            .appendingPathComponent("layout.json")
        let expertStride = plan.layers.first(where: { $0.expertsPerLayer > 0 })?.expertStride ?? 0
        let layoutData = try GTurboJSON.encodeLayout(plan: plan, expertStride: expertStride)
        try writeSmall(path: layoutPath, data: layoutData)
        try GTurboLayoutValidator.validate(path: layoutPath, plan: plan)
        try recordOutputFile(relativePath: "packed_experts/layout.json",
                             path: layoutPath,
                             progress: progress)

        if !plan.pleShards.isEmpty {
            let pleLayoutPath = ((paths.partialDirectory as NSString)
                .appendingPathComponent("packed_ple") as NSString)
                .appendingPathComponent("layout.json")
            let pleLayoutData = try GTurboJSON.encodePleLayout(plan: plan)
            try writeSmall(path: pleLayoutPath, data: pleLayoutData)
            try recordOutputFile(relativePath: "packed_ple/layout.json",
                                 path: pleLayoutPath,
                                 progress: progress)
        }

        try Task.checkCancellation()
        try copyLocalMetadataSidecars(snapshot: snapshot,
                                      partialDir: paths.partialDirectory,
                                      progress: progress)
        try? FileManager.default.removeItem(atPath: paths.rangeTemporaryFile)
        progress(.finalizing)
        try Task.checkCancellation()
        try writeManifest(plan: plan,
                          partialDir: paths.partialDirectory,
                          metadata: snapshot.metadata)

        try Task.checkCancellation()
        if try Posix.entryKind(paths.finalDirectory) == .directory {
            try Posix.renameSwap(paths.partialDirectory, paths.finalDirectory)
            try Posix.fsyncDirectory(paths.parentDirectory)
            try? FileManager.default.removeItem(atPath: paths.partialDirectory)
        } else {
            try Posix.rename(from: paths.partialDirectory, to: paths.finalDirectory)
            try Posix.fsyncDirectory(paths.parentDirectory)
        }

        audit.wallTimeSeconds = Date().timeIntervalSince(startTime)
        audit.wholeFileHeapBuffers = false

        return LocalStreamingRepackResult(
            outputDir: options.outputDir,
            modelID: plan.matchedModelID,
            sourceIndexSHA256: snapshot.metadata.indexSha256Hex,
            rangeRequestCount: rangePlan.coalescedCopies.count,
            copiedBytes: rangePlan.remoteBytesToDownload,
            outputBytes: outputBytes)
    }

    private func validateOptions() throws {
        guard options.rangeChunkBytes > 0,
              options.rangeChunkBytes <= RemoteChunkPolicy.maxBytes else {
            throw RepackError.configurationInvalid(detail: "bad range chunk bytes \(options.rangeChunkBytes)")
        }
        guard options.writeTileBytes > 0,
              options.writeTileBytes <= BoundedScratch.defaultLimitBytes else {
            throw RepackError.configurationInvalid(detail: "bad write tile bytes \(options.writeTileBytes)")
        }
    }

    private func createOutputFiles(plan: RepackPlan,
                                   paths: RemoteInstallPaths) throws {
        try Posix.mkdirP((paths.partialDirectory as NSString)
            .appendingPathComponent("packed_experts"))
        if !plan.pleShards.isEmpty {
            try Posix.mkdirP((paths.partialDirectory as NSString)
                .appendingPathComponent("packed_ple"))
        }
        let resident = try ResidentWriter.createAndWriteIndex(
            plan: plan.resident,
            audit: audit)
        try Posix.fsync(resident, path: plan.resident.path)
        close(resident)
        for layer in plan.layers where layer.expertsPerLayer > 0 {
            try Task.checkCancellation()
            let descriptor = try Posix.openCreateRW(layer.path)
            try Posix.ftruncate(descriptor, path: layer.path, size: layer.fileSize)
            try Posix.fsync(descriptor, path: layer.path)
            close(descriptor)
        }
        for shard in plan.pleShards {
            try Task.checkCancellation()
            let descriptor = try Posix.openCreateRW(shard.path)
            try Posix.ftruncate(descriptor, path: shard.path, size: shard.fileSize)
            try Posix.fsync(descriptor, path: shard.path)
            close(descriptor)
        }
        try Posix.fsyncDirectory(paths.partialDirectory)
    }

    private func recordOutputFile(relativePath: String,
                                  path: String,
                                  progress: @Sendable (ModelInstallProgress) -> Void) throws {
        progress(.hashingOutput(relativePath))
        try Task.checkCancellation()
        let fd = try Posix.openRead(path)
        defer { close(fd) }
        let size = try Posix.fileSize(fd: fd, path: path)
        let sha = try WriterCore.hashEntireFile(path: path,
                                                size: size,
                                                audit: audit,
                                                cancellationCheck: Task.checkCancellation)
        audit.outputFiles.append(.init(relativePath: relativePath, size: size, sha256: sha))
    }

    private func writeSmall(path: String, data: Data) throws {
        try Posix.mkdirP((path as NSString).deletingLastPathComponent)
        try data.write(to: URL(fileURLWithPath: path), options: [.atomic])
        audit.recordWrite(bytes: data.count)
    }

    private func copyLocalMetadataSidecars(snapshot: LocalSnapshot,
                                           partialDir: String,
                                           progress: @Sendable (ModelInstallProgress) -> Void) throws {
        let tokenizerDir = (partialDir as NSString).appendingPathComponent("tokenizer")
        // config.json ships with the snapshot; record it under tokenizer/ to
        // match the remote installer layout.
        let configSrc = (snapshot.sourceDir as NSString).appendingPathComponent("config.json")
        if FileManager.default.fileExists(atPath: configSrc) {
            try Posix.mkdirP(tokenizerDir)
            let dst = (tokenizerDir as NSString).appendingPathComponent("config.json")
            if FileManager.default.fileExists(atPath: dst) {
                try FileManager.default.removeItem(atPath: dst)
            }
            try FileManager.default.copyItem(atPath: configSrc, toPath: dst)
            try recordOutputFile(relativePath: "tokenizer/config.json",
                                 path: dst,
                                 progress: progress)
        }

        let tokenizerFiles: [(name: String, required: Bool)] = [
            ("tokenizer.json", true),
            ("tokenizer_config.json", true),
            ("special_tokens_map.json", false),
            ("chat_template.jinja", false),
            ("chat_template.json", false),
        ]
        for file in tokenizerFiles {
            try Task.checkCancellation()
            let src = (snapshot.sourceDir as NSString).appendingPathComponent(file.name)
            guard FileManager.default.fileExists(atPath: src) else {
                if file.required {
                    throw RepackError.configurationInvalid(detail: "missing required sidecar: \(file.name)")
                }
                continue
            }
            try Posix.mkdirP(tokenizerDir)
            let dst = (tokenizerDir as NSString).appendingPathComponent(file.name)
            if FileManager.default.fileExists(atPath: dst) {
                try FileManager.default.removeItem(atPath: dst)
            }
            try FileManager.default.copyItem(atPath: src, toPath: dst)
            try recordOutputFile(relativePath: "tokenizer/\(file.name)",
                                 path: dst,
                                 progress: progress)
        }
    }

    private func writeManifest(plan: RepackPlan,
                               partialDir: String,
                               metadata: IndexLoader.SourceMetadata) throws {
        var bits = GTurboJSON.QuantBitWidths(
            embedding: 4,
            attention: 4,
            router: 8,
            sharedExpert: 8,
            routedExpert: 4)
        for e in plan.resident.entries {
            if e.name == "language_model.model.embed_tokens.weight", let s = e.quantSpec {
                bits.embedding = s.bits
                bits.embeddingGroup = s.groupSize
            }
            if e.name.hasSuffix(".self_attn.q_proj.weight")
                || e.name.hasSuffix(".linear_attn.in_proj_qkv.weight"),
               let s = e.quantSpec {
                bits.attention = s.bits
                bits.attentionGroup = s.groupSize
            }
            if e.name.hasSuffix(".router.proj.weight")
                || e.name.hasSuffix(".mlp.gate.weight"),
               let s = e.quantSpec {
                bits.router = s.bits
                bits.routerGroup = s.groupSize
            }
            if e.name.hasSuffix(".mlp.gate_proj.weight")
                || e.name.hasSuffix(".mlp.shared_expert.gate_proj.weight"),
               let s = e.quantSpec {
                bits.sharedExpert = s.bits
                bits.sharedExpertGroup = s.groupSize
            }
        }
        if let layer = plan.layers.first(where: { !$0.subTensors.isEmpty }),
           let routedBits = layer.subTensors.first?.bitsForWeights {
            bits.routedExpert = routedBits
            // The routed-expert group travels with the base: qwen38 packs
            // experts 32-wide, older families 64-wide.
            bits.routedExpertGroup = plan.baseGroupSize
        }
        let files = audit.outputFiles.map {
            ($0.relativePath, GTurboJSON.FileEntry(size: $0.size, sha256: $0.sha256))
        }
        let data = try GTurboJSON.encodeManifest(
            plan: plan,
            modelID: plan.matchedModelID ?? "unknown/snapshot",
            sourceSnapshotHash: "sha256:" + metadata.indexSha256Hex,
            files: files,
            expertsPerLayer: plan.layers.first(where: { $0.expertsPerLayer > 0 })?.expertsPerLayer ?? 0,
            numLayers: plan.arch.numLayers,
            expertStride: plan.layers.first(where: { $0.expertsPerLayer > 0 })?.expertStride ?? 0,
            bitWidths: bits)
        let tmp = (partialDir as NSString).appendingPathComponent("manifest.json.tmp")
        let final = (partialDir as NSString).appendingPathComponent("manifest.json")
        try writeSmall(path: tmp, data: data)
        try Posix.rename(from: tmp, to: final)
        let manifestSha = try Sha256Stream.hashFile(path: final)
        let receipt = try VerifiedInstallReceiptWriter.encode(
            outputDir: options.outputDir,
            manifestSha256: manifestSha,
            manifestSize: UInt64(data.count),
            sourceRepoID: plan.matchedModelID ?? "local-snapshot",
            sourceRevision: "local:" + metadata.indexSha256Hex,
            files: audit.outputFiles)
        let receiptPath = (partialDir as NSString)
            .appendingPathComponent(VerifiedInstallReceiptWriter.fileName)
        try writeSmall(path: receiptPath, data: receipt)
    }
}
