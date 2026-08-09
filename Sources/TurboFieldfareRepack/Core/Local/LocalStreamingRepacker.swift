import Foundation

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

        let inputDir = options.inputLocal
        let outputDir = options.outputDir
        // Remove existing output dir if --overwrite
        if try Posix.entryKind(outputDir) == .directory, options.overwrite {
            try FileManager.default.removeItem(atPath: outputDir)
        }


          // Check output directory conflict
        if try Posix.entryKind(outputDir) == .directory, !options.overwrite {
            throw RepackError.configurationInvalid(
                detail: "output directory already exists: \(outputDir)")
          }

         // Validate input directory
        guard FileManager.default.fileExists(atPath: inputDir) else {
            throw RepackError.localInputNotFound(path: inputDir)
          }
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: inputDir, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            throw RepackError.localInputNotDirectory(path: inputDir)
          }

         // Verify required files exist
        let indexPath = (inputDir as NSString).appendingPathComponent("model.safetensors.index.json")
        let configPath = (inputDir as NSString).appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: indexPath) else {
            throw RepackError.indexJsonInvalid(path: indexPath, detail: "index.json not found in input directory")
          }
        guard FileManager.default.fileExists(atPath: configPath) else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "config.json not found in input directory")
          }

         // Load index, metadata, and architecture from local files
        progress(.downloadingMetadata)
        let meta = try IndexLoader.load(snapshotDir: inputDir)
        let arch = try ArchInfo.load(configPath: configPath)
        let shardHeaders: [Safetensors.Header] = try loadShardHeaders(from: inputDir, meta: meta)

         // Debug: print shard header summary
        let debugPath = ProcessInfo.processInfo.environment["TURBOFIELDFARE_DEBUG_SHARDS"]
        if let debugPath {
            var lines: [String] = []
            for header in shardHeaders {
                for tensor in header.tensors {
                    let shapeStr = tensor.shape.map { String($0) }.joined(separator: ", ")
                    lines.append(
                        "\(tensor.name) shard=\(tensor.shardPath) offset=\(tensor.absoluteOffset) "
                        + "size=\(tensor.sizeBytes) dtype=\(tensor.dtype.rawValue) shape=\(shapeStr)")
                }
            }
            let text = lines.joined(separator: "\n") + "\n"
            try Data(text.utf8).write(to: URL(fileURLWithPath: debugPath), options: .atomic)
        }

         // Plan repack
        let plan = try RepackPlanner.plan(meta: meta,
                                          arch: arch,
                                          shardHeaders: shardHeaders,
                                          outputDir: outputDir)
        let rangePlan = try RangeCopyPlanner.plan(repackPlan: plan,
                                                  rangeChunkBytes: options.rangeChunkBytes,
                                                  layoutMode: "identity",
                                                  layoutOrderSha256: nil)

        let outputBytes = plan.resident.totalSize
              + plan.layers.reduce(UInt64(0)) { $0 + $1.fileSize }
        progress(.planning(downloadBytes: rangePlan.remoteBytesToDownload,
                           outputBytes: outputBytes))

         // Dry-run space check
        if options.dryRunSpaceCheck {
            return LocalStreamingRepackResult(outputDir: options.outputDir,
                                              plan: plan,
                                              localBytesRead: 0,
                                              dryRun: true)
          }

         // Check disk space
        let diskRequirement = try DiskSpaceChecker.requireAvailable(
            path: outputDir,
            bytes: outputBytes,
            reserveBytes: options.minFreeReserveBytes)
        progress(.checkingDisk(diskRequirement))
        try Task.checkCancellation()

        audit.packedExpertLayoutMode = "identity"

         // Create output files
        progress(.reservingOutput(bytes: outputBytes))
        try createOutputFiles(plan: plan, outputDir: outputDir)

         // Copy bytes from local file
        let shardPathFor: (String) -> String = { filename in
            if filename.hasPrefix("/") { return filename }
            return (inputDir as NSString).appendingPathComponent(filename)
            }
        let provider = LocalByteProvider(inputDir: inputDir,
                                         shardPathFor: shardPathFor,
                                         writeTileBytes: options.writeTileBytes)
        progress(.copyingPayload(
            reusedBytes: 0,
            downloadedThisRunBytes: 0,
            totalBytes: rangePlan.remoteBytesToDownload))
        try await provider.copyBatch(
            rangePlan.coalescedCopies,
            completedRangeIDs: [],
            partialDirectory: outputDir,
            temporaryPath: "",
            audit: audit,
            progress: { downloadedBytes in
                progress(.copyingPayload(
                    reusedBytes: 0,
                    downloadedThisRunBytes: downloadedBytes,
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

        // Write layout.json
        let packedExpertsDir = (outputDir as NSString).appendingPathComponent("packed_experts")
        let layoutPath = (packedExpertsDir as NSString).appendingPathComponent("layout.json")
        let expertStride = plan.layers.first(where: { $0.expertsPerLayer > 0 })?.expertStride ?? 0
        let layoutData = try GTurboJSON.encodeLayout(plan: plan, expertStride: expertStride)
        try writeSmall(path: layoutPath, data: layoutData)
        try GTurboLayoutValidator.validate(path: layoutPath, plan: plan)
        try recordOutputFile(relativePath: "packed_experts/layout.json",
                             path: layoutPath,
                             progress: progress)

        try Task.checkCancellation()
        try copyLocalMetadataSidecars(inputDir: inputDir,
                                      outputDir: outputDir,
                                      progress: progress)
        try Task.checkCancellation()
        progress(.finalizing)
        try writeManifest(plan: plan,
                          outputDir: outputDir,
                          metadata: meta,
                          expertStride: expertStride)

        audit.wallTimeSeconds = Date().timeIntervalSince(startTime)
        audit.wholeFileHeapBuffers = false

        return LocalStreamingRepackResult(outputDir: options.outputDir,
                                          plan: plan,
                                          localBytesRead: audit.sourceBytesRead,
                                          dryRun: false)
     }

     // MARK: - Helpers

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

    private func copyLocalMetadataSidecars(inputDir: String,
                                          outputDir: String,
                                          progress: @Sendable (ModelInstallProgress) -> Void) throws {
        let tokenizerDir = (outputDir as NSString).appendingPathComponent("tokenizer")
        try Posix.mkdirP(tokenizerDir)
        let sidecarFiles = ["config.json",
                             "tokenizer.json",
                             "tokenizer_config.json",
                             "special_tokens_map.json",
                             "chat_template.jinja",
                             "chat_template.json"]
        for filename in sidecarFiles {
            let src = (inputDir as NSString).appendingPathComponent(filename)
            guard FileManager.default.fileExists(atPath: src) else { continue }
            let dst = (tokenizerDir as NSString).appendingPathComponent(filename)
            if FileManager.default.fileExists(atPath: dst) {
                try FileManager.default.removeItem(atPath: dst)
            }
            try FileManager.default.copyItem(atPath: src, toPath: dst)
            let size = try FileManager.default.attributesOfItem(atPath: dst)[.size] as? UInt64 ?? 0
            audit.recordWrite(bytes: Int(size))
        }
    }

    private func loadShardHeaders(from inputDir: String,
                                  meta: IndexLoader.SourceMetadata) throws -> [Safetensors.Header] {
        var headers: [Safetensors.Header] = []
        headers.reserveCapacity(meta.shardFilenames.count)
        for shardFilename in meta.shardFilenames {
            let shardPath = (inputDir as NSString).appendingPathComponent(shardFilename)
            let fd = try Posix.openReadNoFollow(shardPath)
            defer { close(fd) }
            let fileSize = try Posix.fileSize(fd: fd, path: shardPath)
            let data = try Data(contentsOf: URL(fileURLWithPath: shardPath),
                                options: [.alwaysMapped])
            guard data.count >= 8 else {
                throw RepackError.safetensorsHeaderInvalid(path: shardPath, detail: "file too small for header")
              }
            let headerSizeBytes = data.subdata(in: 0..<8)
            let headerSize = headerSizeBytes.withUnsafeBytes { ptr -> UInt64 in
                ptr.loadUnaligned(fromByteOffset: 0, as: UInt64.self)
              }
            guard headerSize <= Safetensors.maxHeaderBytes else {
                throw RepackError.safetensorsHeaderTooLarge(path: shardPath, size: headerSize)
              }
            guard data.count >= 8 + headerSize else {
                throw RepackError.safetensorsHeaderInvalid(path: shardPath, detail: "file smaller than declared header")
              }
            let headerData = data.subdata(in: Int(8)..<Int(8 + headerSize))
            let header = try Safetensors.parseHeaderBytes(path: shardPath,
                                                          fileSize: fileSize,
                                                          headerBytes: headerData)
            headers.append(header)
          }
        return headers
      }

    private func createOutputFiles(plan: RepackPlan, outputDir: String) throws {
        let packedExpertsDir = (outputDir as NSString).appendingPathComponent("packed_experts")
        try Posix.mkdirP(packedExpertsDir)
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
        try Posix.fsyncDirectory(outputDir)
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

    private func writeManifest(plan: RepackPlan,
                               outputDir: String,
                               metadata: IndexLoader.SourceMetadata,
                               expertStride: UInt64) throws {
        var bits = GTurboJSON.QuantBitWidths(
            embedding: 4,
            attention: 4,
            router: 8,
            sharedExpert: 8,
            routedExpert: 4)
        for e in plan.resident.entries {
            if e.name == "language_model.model.embed_tokens.weight", let s = e.quantSpec {
                bits.embedding = s.bits
              }
            if e.name.hasSuffix(".self_attn.q_proj.weight")
                  || e.name.hasSuffix(".linear_attn.in_proj_qkv.weight"),
               let s = e.quantSpec {
                bits.attention = s.bits
              }
            if e.name.hasSuffix(".router.proj.weight")
                  || e.name.hasSuffix(".mlp.gate.weight"),
               let s = e.quantSpec {
                bits.router = s.bits
              }
            if e.name.hasSuffix(".mlp.gate_proj.weight")
                  || e.name.hasSuffix(".mlp.shared_expert.gate_proj.weight"),
               let s = e.quantSpec {
                bits.sharedExpert = s.bits
              }
          }
        if let layer = plan.layers.first(where: { !$0.subTensors.isEmpty }),
           let routedBits = layer.subTensors.first?.bitsForWeights {
            bits.routedExpert = routedBits
          }
        let files = audit.outputFiles.map {
            ($0.relativePath, GTurboJSON.FileEntry(size: $0.size, sha256: $0.sha256))
          }
        let data = try GTurboJSON.encodeManifest(
            plan: plan,
            modelID: plan.matchedModelID ?? "local/snapshot",
            sourceSnapshotHash: "sha256:" + metadata.indexSha256Hex,
            files: files,
            expertsPerLayer: plan.layers.first(where: { $0.expertsPerLayer > 0 })?.expertsPerLayer ?? 0,
            numLayers: plan.arch.numLayers,
            expertStride: expertStride,
            bitWidths: bits)
        let final = (outputDir as NSString).appendingPathComponent("manifest.json")
        try writeSmall(path: final, data: data)
     }
}
