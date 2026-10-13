import Foundation

/// On-disk page alignment unit for `.gturbo` files. Fixed at 16 KB regardless
/// of host page size — the format is the contract, not the kernel.
enum Layout {
    static let pageBytes: UInt64 = 16_384
}

// MARK: - Plan data types

struct ResidentEntry: Sendable {
    let name: String
    /// dtype byte for IndexEntry: 0 = U32, 1 = BF16, 2 = FP16, 3 = FP32.
    let dtype: UInt8
    /// Logical shape after dequant (max rank 4; trailing zeros).
    let logicalShape4: [UInt32]
    /// File offset where the (packed) weight bytes start.
    let fileOffset: UInt64
    /// Size in bytes of the weight bytes.
    let sizeBytes: UInt64
    /// Offset where BF16 scales start (0 if none).
    let scaleOffset: UInt64
    let scaleSize: UInt64
    /// Offset where BF16 biases start (0 if none).
    let biasOffset: UInt64
    let biasSize: UInt64
    /// Quantization spec (nil for unquantized scalars/norms).
    let quantSpec: QuantSpec?

    /// Source tensors that supply this entry's bytes.
    let sourceWeight: SourceTensor
    let sourceScales: SourceTensor?
    let sourceBiases: SourceTensor?
}

struct ResidentFilePlan: Sendable {
    let path: String
    let entries: [ResidentEntry]
    let stringTable: [UInt8]
    let stringTableOffsets: [UInt32]   // per-entry offsets into the table
    let indexSize: UInt64              // header + entries + table + padding
    let residentSize: UInt64           // tensor payload region
    var totalSize: UInt64 { indexSize + residentSize }
}

struct PerExpertTensorSlice: Sendable {
    let role: String                   // "gate" | "up" | "down"
    let component: String              // "weights" | "scales" | "biases"
    let dtype: UInt8                   // 0=U32, 1=BF16
    let logicalShape: [UInt64]         // per-expert logical shape
    let offsetInExpertBlob: UInt64     // within each expert blob
    let sizeInExpertBlob: UInt64
    /// For each expert e (0..<expertsPerLayer): source byte offset & size.
    let sourceOffsetPerExpert: UInt64  // stride per expert in source
    let sourceTensor: SourceTensor
    let bitsForWeights: Int?           // 4 for routed expert weight; nil for scales/biases
}

struct LayerFilePlan: Sendable {
    let layerIndex: Int
    let path: String
    let expertsPerLayer: Int
    let expertStride: UInt64
    let subTensors: [PerExpertTensorSlice]  // 9 entries: gate/up/down × {weights, scales, biases}
    var fileSize: UInt64 { UInt64(expertsPerLayer) * expertStride }

    func physicalRank(for logicalExpert: Int) -> Int {
        logicalExpert
    }

    init(layerIndex: Int,
                path: String,
                expertsPerLayer: Int,
                expertStride: UInt64,
                subTensors: [PerExpertTensorSlice]) {
        self.layerIndex = layerIndex
        self.path = path
        self.expertsPerLayer = expertsPerLayer
        self.expertStride = expertStride
        self.subTensors = subTensors
    }
}

struct RepackPlan: Sendable {
    let arch: ArchInfo
    let baseMode: String                  // "affine"
    let baseGroupSize: Int                // 64 (32 for qwen38)
    let bitsOverrideCount: Int
    let resident: ResidentFilePlan
    let layers: [LayerFilePlan]
    let matchedModelID: String?
    let excludedMultimodalTensorNames: [String]
    /// SSD-streamed PLE n-gram shards (qwen38 only; empty otherwise).
    /// Each shard is one file under `packed_ple/` holding the quantized
    /// weight bytes followed by BF16 scales then biases.
    let pleShards: [PleShardPlan]
}

/// One SSD-streamed PLE n-gram embedding shard.
struct PleShardPlan: Sendable {
    let layerIndex: Int
    let shardIndex: Int
    let path: String
    let weightBytes: UInt64
    let scalesBytes: UInt64
    let biasesBytes: UInt64
    var fileSize: UInt64 { weightBytes + scalesBytes + biasesBytes }
    /// Logical (dequantized) shape of the shard weight.
    let logicalShape: [UInt64]
    let bits: Int
    let groupSize: Int
    let sourceWeight: SourceTensor
    let sourceScales: SourceTensor
    let sourceBiases: SourceTensor
}

// MARK: - Planner

enum RepackPlanner {

    /// Classify a tensor name. Routed-expert tensors split off the LM bucket;
    /// qwen38 PLE n-gram shards split into their own SSD-streamed bucket.
    enum Bucket: Equatable {
        case lmResident
        case routedExpert(role: String, layer: Int)   // role = "gate"|"up"|"down"
        case pleShard(layer: Int, shard: Int)
        case excludedMultimodal
        case unknown
    }

    static func classify(_ name: String, numLayers: Int,
                         family: RepackModelFamily) -> Bucket {
        if name.hasPrefix("language_model.") {
            // PLE n-gram shards stream from SSD, never resident.
            if family == .qwen38,
               let (layer, shard) = pleShardIndices(in: name),
               layer >= 0 && layer < numLayers {
                return .pleShard(layer: layer, shard: shard)
            }
            // Routed expert?
            if let role = routedExpertRole(in: name, family: family),
               let layer = layerIndex(in: name),
               layer >= 0 && layer < numLayers {
                return .routedExpert(role: role, layer: layer)
            }
            return .lmResident
        }
        if isMultimodalTensorName(name) {
            return .excludedMultimodal
        }
        return .unknown
    }

    private static func routedExpertRole(in name: String,
                                         family: RepackModelFamily) -> String? {
        let routedContainer: String
        switch family {
        case .gemma4: routedContainer = ".experts.switch_glu."
        case .qwen36, .qwen35, .qwen38: routedContainer = ".mlp.switch_mlp."
        }
        guard name.contains(routedContainer) else { return nil }
        if name.contains(".gate_proj.") { return "gate" }
        if name.contains(".up_proj.")   { return "up" }
        if name.contains(".down_proj.") { return "down" }
        return nil
    }

    /// Parses `...layers.<L>.ple.ple_embedding.ngram_embedding.shards.<S>.{weight,scales,biases}`.
    private static func pleShardIndices(in name: String) -> (Int, Int)? {
        guard name.contains(".ple.ple_embedding.ngram_embedding.shards.") else { return nil }
        guard let layer = layerIndex(in: name) else { return nil }
        guard let r = name.range(of: ".shards.") else { return nil }
        let tail = name[r.upperBound...]
        guard let dot = tail.firstIndex(of: ".") else { return nil }
        guard let shard = Int(tail[tail.startIndex..<dot]) else { return nil }
        return (layer, shard)
    }

    private static func layerIndex(in name: String) -> Int? {
        // matches "...layers.<N>...."
        guard let r = name.range(of: ".layers.") else { return nil }
        let tail = name[r.upperBound...]
        guard let dot = tail.firstIndex(of: ".") else { return nil }
        return Int(tail[tail.startIndex..<dot])
    }

    /// Build the plan from parsed shard headers + source metadata.
    /// - throws: classification + companion + override count failures.
    static func plan(meta: IndexLoader.SourceMetadata,
                            arch: ArchInfo,
                            shardHeaders: [Safetensors.Header],
                            outputDir: String) throws -> RepackPlan {

        // Companion tensors may live in different shards, so resolve them
        // through one global registry.
        var registry: [String: SourceTensor] = [:]
        registry.reserveCapacity(meta.weightMap.count)
        for h in shardHeaders {
            for t in h.tensors { registry[t.name] = t }
        }

        // Source allowlisting owns exact fingerprint validation. Preserve the
        // declared override count for the output manifest audit.
        let bitsOverrideCount = meta.bitsOverrides.count

        var lmResidentBases: [String] = []
        var excludedMultimodalNames: [String] = []
        var routedByLayerAndRole: [Int: [String: String]] = [:]
        var pleByLayerAndShard: [Int: [Int: String]] = [:]
        for (name, _) in registry {
            if isMultimodalTensorName(name) {
                excludedMultimodalNames.append(name)
            }
            if name.hasSuffix(".scales") || name.hasSuffix(".biases") { continue }
            let b = classify(name, numLayers: arch.numLayers, family: arch.family)
            switch b {
            case .lmResident:                   lmResidentBases.append(name)
            case .routedExpert(let role, let layer):
                var byRole = routedByLayerAndRole[layer] ?? [:]
                if byRole[role] != nil {
                    throw RepackError.configurationInvalid(detail:
                        "two routed-expert tensors for layer \(layer) role \(role)")
                }
                byRole[role] = name
                routedByLayerAndRole[layer] = byRole
            case .pleShard(let layer, let shard):
                var byShard = pleByLayerAndShard[layer] ?? [:]
                if byShard[shard] != nil {
                    throw RepackError.configurationInvalid(detail:
                        "two PLE shard tensors for layer \(layer) shard \(shard)")
                }
                byShard[shard] = name
                pleByLayerAndShard[layer] = byShard
            case .excludedMultimodal:           continue
            case .unknown:                      throw RepackError.unknownTensorPrefix(name: name)
            }
        }

        // Sort deterministically. The LM order follows a fixed template.
        lmResidentBases.sort(by: lmResidentOrdering(family: arch.family))
        excludedMultimodalNames.sort()

        let residentPath = (outputDir as NSString).appendingPathComponent("model_weights.bin")
        let resident = try planResidentFile(path: residentPath,
                                            baseNames: lmResidentBases,
                                            registry: registry, meta: meta)

        let layersDir = (outputDir as NSString).appendingPathComponent("packed_experts")
        var layerPlans: [LayerFilePlan] = []
        layerPlans.reserveCapacity(arch.numLayers)
        for layer in 0..<arch.numLayers {
            let bundle = routedByLayerAndRole[layer] ?? [:]
            // Synthetic snapshots may legitimately have no routed experts.
            guard let gName = bundle["gate"], let uName = bundle["up"], let dName = bundle["down"] else {
                if bundle.isEmpty {
                    layerPlans.append(LayerFilePlan(layerIndex: layer,
                                                    path: (layersDir as NSString).appendingPathComponent("layer_\(String(format: "%02d", layer)).bin"),
                                                    expertsPerLayer: 0,
                                                    expertStride: 0,
                                                    subTensors: []))
                    continue
                }
                throw RepackError.configurationInvalid(detail:
                    "layer \(layer) routed-expert bundle incomplete: \(bundle)")
            }
            let path = (layersDir as NSString)
                .appendingPathComponent("layer_\(String(format: "%02d", layer)).bin")
            let lp = try planLayerFile(path: path, layer: layer,
                                       gateName: gName, upName: uName, downName: dName,
                                       registry: registry, meta: meta, arch: arch)
            layerPlans.append(lp)
        }

        let matched = SourceFingerprint.modelID(forIndexSha256: meta.indexSha256Hex)

        var pleShards: [PleShardPlan] = []
        let pleDir = (outputDir as NSString).appendingPathComponent("packed_ple")
        for layer in pleByLayerAndShard.keys.sorted() {
            for shard in (pleByLayerAndShard[layer] ?? [:]).keys.sorted() {
                let base = pleByLayerAndShard[layer]![shard]!
                let path = (pleDir as NSString)
                    .appendingPathComponent("shard_\(String(format: "%03d", shard)).bin")
                pleShards.append(try planPleShard(
                    path: path,
                    layer: layer, shard: shard, baseName: base,
                    registry: registry, meta: meta))
            }
        }

        return RepackPlan(arch: arch,
                          baseMode: meta.baseMode,
                          baseGroupSize: meta.baseGroupSize,
                          bitsOverrideCount: bitsOverrideCount,
                          resident: resident,
                          layers: layerPlans,
                          matchedModelID: matched,
                          excludedMultimodalTensorNames: excludedMultimodalNames,
                          pleShards: pleShards)
    }

    private static func isMultimodalTensorName(_ name: String) -> Bool {
        name.hasPrefix("vision_tower.") ||
            name.hasPrefix("embed_vision.") ||
            name.hasPrefix("audio_tower.")
    }

    // MARK: - Resident planning

    private static func planResidentFile(path: String,
                                         baseNames: [String],
                                         registry: [String: SourceTensor],
                                         meta: IndexLoader.SourceMetadata) throws
                                        -> ResidentFilePlan {
        let entryCount = baseNames.count

        var stringTable: [UInt8] = []
        var offsets: [UInt32] = []
        offsets.reserveCapacity(entryCount)
        for n in baseNames {
            offsets.append(UInt32(stringTable.count))
            stringTable.append(contentsOf: n.utf8)
        }

        // Index size includes the fixed header, fixed-width entries, and the
        // string table, padded to a 16 KB page boundary.
        let rawIdx = UInt64(GTurboBinary.indexHeaderBytes
            + entryCount * GTurboBinary.indexEntryBytes
            + stringTable.count)
        let indexSize = roundUpToPage(rawIdx)

        var fileCursor = indexSize
        var entries: [ResidentEntry] = []
        entries.reserveCapacity(entryCount)

        for name in baseNames {
            guard let weight = registry[name] else {
                throw RepackError.missingTensor(name: name)
            }
            let dtype = ietnyDtype(weight.dtype)
            let isQuantizedPacked = (weight.dtype == .u32) && name.hasSuffix(".weight")

            if isQuantizedPacked {
                let base = String(name.dropLast(".weight".count))
                guard let scales = registry[base + ".scales"] else {
                    throw RepackError.missingScalesCompanion(name: name)
                }
                guard let biases = registry[base + ".biases"] else {
                    throw RepackError.missingBiasesCompanion(name: name)
                }
                if scales.dtype != .bf16 || biases.dtype != .bf16 {
                    throw RepackError.dtypeMismatch(name: name,
                        detail: "expected BF16 scales/biases, got \(scales.dtype)/\(biases.dtype)")
                }
                let spec = IndexLoader.quantSpec(forTensor: name, meta: meta)
                let logical = logicalShape(forPackedSource: weight.shape, bits: spec.bits)

                let wOff = fileCursor
                let wSize = weight.sizeBytes
                let sOff = wOff + wSize
                let sSize = scales.sizeBytes
                let bOff = sOff + sSize
                let bSize = biases.sizeBytes
                fileCursor = bOff + bSize

                entries.append(ResidentEntry(
                    name: name, dtype: 0,
                    logicalShape4: padTo4(logical),
                    fileOffset: wOff, sizeBytes: wSize,
                    scaleOffset: sOff, scaleSize: sSize,
                    biasOffset: bOff, biasSize: bSize,
                    quantSpec: spec,
                    sourceWeight: weight, sourceScales: scales, sourceBiases: biases))
            } else {
                // Unquantized (BF16 norm / scalar) — no companions.
                let off = fileCursor
                let size = weight.sizeBytes
                fileCursor = off + size

                entries.append(ResidentEntry(
                    name: name, dtype: dtype,
                    logicalShape4: padTo4(weight.shape),
                    fileOffset: off, sizeBytes: size,
                    scaleOffset: 0, scaleSize: 0,
                    biasOffset: 0, biasSize: 0,
                    quantSpec: nil,
                    sourceWeight: weight, sourceScales: nil, sourceBiases: nil))
            }
        }

        let residentSize = fileCursor - indexSize

        return ResidentFilePlan(path: path,
                                entries: entries,
                                stringTable: stringTable,
                                stringTableOffsets: offsets,
                                indexSize: indexSize,
                                residentSize: residentSize)
    }

    // MARK: - Layer planning

    private static func planLayerFile(path: String, layer: Int,
                                      gateName: String, upName: String, downName: String,
                                      registry: [String: SourceTensor],
                                      meta: IndexLoader.SourceMetadata,
                                      arch: ArchInfo) throws -> LayerFilePlan {
        let expertCount = arch.numExperts
        let roles: [(role: String, name: String)] = [
            ("gate", gateName), ("up", upName), ("down", downName)
        ]
        var subs: [PerExpertTensorSlice] = []
        subs.reserveCapacity(9)
        var blobCursor: UInt64 = 0

        for (role, name) in roles {
            guard let w = registry[name] else { throw RepackError.missingTensor(name: name) }
            if w.dtype != .u32 || w.shape.count != 3 || Int(w.shape[0]) != expertCount {
                throw RepackError.shapeMismatch(name: name,
                    detail: "expected U32 rank-3 with leading \(expertCount), got \(w.dtype) \(w.shape)")
            }
            let base = name.hasSuffix(".weight") ? String(name.dropLast(".weight".count)) : name
            guard let s = registry[base + ".scales"] else { throw RepackError.missingScalesCompanion(name: name) }
            guard let b = registry[base + ".biases"] else { throw RepackError.missingBiasesCompanion(name: name) }
            if s.dtype != .bf16 || b.dtype != .bf16 {
                throw RepackError.dtypeMismatch(name: name,
                    detail: "expected BF16 scales/biases, got \(s.dtype)/\(b.dtype)")
            }

            let perExpertWeightSize = w.sizeBytes / UInt64(expertCount)
            let perExpertScaleSize  = s.sizeBytes / UInt64(expertCount)
            let perExpertBiasSize   = b.sizeBytes / UInt64(expertCount)
            if perExpertWeightSize * UInt64(expertCount) != w.sizeBytes ||
               perExpertScaleSize  * UInt64(expertCount) != s.sizeBytes ||
               perExpertBiasSize   * UInt64(expertCount) != b.sizeBytes {
                throw RepackError.shapeMismatch(name: name,
                    detail: "source bytes not evenly divisible by \(expertCount) experts")
            }

            let spec = IndexLoader.quantSpec(forTensor: name, meta: meta)
            let perExpertSourceShape = Array(w.shape.dropFirst())
            let logicalPerExpert = logicalShape(forPackedSource: perExpertSourceShape, bits: spec.bits)
            let scalesLogical = Array(s.shape.dropFirst())
            let biasesLogical = Array(b.shape.dropFirst())

            let wSlice = PerExpertTensorSlice(
                role: role, component: "weights", dtype: 0,
                logicalShape: logicalPerExpert,
                offsetInExpertBlob: blobCursor, sizeInExpertBlob: perExpertWeightSize,
                sourceOffsetPerExpert: perExpertWeightSize, sourceTensor: w,
                bitsForWeights: spec.bits)
            blobCursor += perExpertWeightSize
            let sSlice = PerExpertTensorSlice(
                role: role, component: "scales", dtype: 1,
                logicalShape: scalesLogical,
                offsetInExpertBlob: blobCursor, sizeInExpertBlob: perExpertScaleSize,
                sourceOffsetPerExpert: perExpertScaleSize, sourceTensor: s,
                bitsForWeights: nil)
            blobCursor += perExpertScaleSize
            let bSlice = PerExpertTensorSlice(
                role: role, component: "biases", dtype: 1,
                logicalShape: biasesLogical,
                offsetInExpertBlob: blobCursor, sizeInExpertBlob: perExpertBiasSize,
                sourceOffsetPerExpert: perExpertBiasSize, sourceTensor: b,
                bitsForWeights: nil)
            blobCursor += perExpertBiasSize

            subs.append(wSlice); subs.append(sSlice); subs.append(bSlice)
        }

        let expertStride = roundUpToPage(blobCursor)
        return LayerFilePlan(layerIndex: layer, path: path,
                             expertsPerLayer: expertCount,
                             expertStride: expertStride,
                             subTensors: subs)
    }

    /// Plans one SSD-streamed PLE n-gram shard file. The file holds the
    /// quantized weight bytes followed by BF16 scales then biases; the
    /// companion shapes must tile the shard's row groups exactly.
    private static func planPleShard(path: String, layer: Int, shard: Int,
                                     baseName: String,
                                     registry: [String: SourceTensor],
                                     meta: IndexLoader.SourceMetadata) throws -> PleShardPlan {
        let weightName = baseName.hasSuffix(".weight") ? baseName : baseName + ".weight"
        guard let w = registry[weightName] else { throw RepackError.missingTensor(name: weightName) }
        guard w.dtype == .u32 else {
            throw RepackError.shapeMismatch(name: weightName,
                detail: "expected U32 packed PLE shard, got \(w.dtype)")
        }
        let base = String(weightName.dropLast(".weight".count))
        guard let s = registry[base + ".scales"] else { throw RepackError.missingScalesCompanion(name: weightName) }
        guard let b = registry[base + ".biases"] else { throw RepackError.missingBiasesCompanion(name: weightName) }
        if s.dtype != .bf16 || b.dtype != .bf16 {
            throw RepackError.dtypeMismatch(name: weightName,
                detail: "expected BF16 scales/biases, got \(s.dtype)/\(b.dtype)")
        }
        let spec = IndexLoader.quantSpec(forTensor: weightName, meta: meta)
        let logical = logicalShape(forPackedSource: w.shape, bits: spec.bits)
        return PleShardPlan(
            layerIndex: layer,
            shardIndex: shard,
            path: path,
            weightBytes: w.sizeBytes,
            scalesBytes: s.sizeBytes,
            biasesBytes: b.sizeBytes,
            logicalShape: logical,
            bits: spec.bits,
            groupSize: spec.groupSize,
            sourceWeight: w,
            sourceScales: s,
            sourceBiases: b)
    }

    // MARK: - Helpers

    private static func ietnyDtype(_ d: SourceTensor.Dtype) -> UInt8 {
        switch d { case .u32: 0; case .bf16: 1; case .fp16: 2; case .fp32: 3; case .i64: 4 }
    }

    private static func roundUpToPage(_ v: UInt64) -> UInt64 {
        let p = Layout.pageBytes
        return ((v + p - 1) / p) * p
    }

    private static func padTo4(_ s: [UInt64]) -> [UInt32] {
        var out: [UInt32] = []
        out.reserveCapacity(4)
        for v in s.prefix(4) { out.append(UInt32(v)) }
        while out.count < 4 { out.append(0) }
        return out
    }

    /// Logical shape of a packed quantized tensor whose source is `[D0,..,Dn-1, Dn/factor]`.
    private static func logicalShape(forPackedSource source: [UInt64], bits: Int) -> [UInt64] {
        let factor = UInt64(32 / bits)
        guard !source.isEmpty else { return source }
        var out = source
        out[out.count - 1] = source[source.count - 1] * factor
        return out
    }

    /// Stable order for the resident LM tensor list. Embedding first, then
    /// per-layer groups in layer index order, then the final norm (and, for
    /// families with an untied head, `lm_head` last).
    private static func lmResidentOrdering(family: RepackModelFamily)
        -> (String, String) -> Bool {
        // Compute a sort key per name; we order by (group rank, layer, slot rank, name).
        func key(_ n: String) -> (Int, Int, Int, String) {
            if n == "language_model.model.embed_tokens.weight" { return (0, 0, 0, n) }
            if n == "language_model.model.hyper_connection_mixer.hc_norm.weight" { return (3, 0, 0, n) }
            if n == "language_model.model.hyper_connection_mixer.input_mix_weight_down.weight" { return (3, 0, 1, n) }
            if n == "language_model.model.hyper_connection_mixer.input_mix_weight_up.weight" { return (3, 0, 2, n) }
            if n == "language_model.model.norm.weight"          { return (3, 0, 0, n) }
            if n == "language_model.lm_head.weight"             { return (4, 0, 0, n) }
            if let li = layerIndex(in: n) {
                let slot: Int
                switch family {
                case .gemma4: slot = slotRank(in: n)
                case .qwen36, .qwen35: slot = qwenSlotRank(in: n)
                case .qwen38: slot = qwen38SlotRank(in: n)
                }
                return (1, li, slot, n)
            }
            return (2, 0, 0, n)
        }
        return { a, b in
            let ka = key(a), kb = key(b)
            if ka.0 != kb.0 { return ka.0 < kb.0 }
            if ka.1 != kb.1 { return ka.1 < kb.1 }
            if ka.2 != kb.2 { return ka.2 < kb.2 }
            return ka.3 < kb.3
        }
    }

    /// Within-layer slot order for the Qwen 3.5/3.6 family: full-attention
    /// projections/norms, then the gated-DeltaNet linear-attention bundle,
    /// then router, shared-expert gate and MLP, then the two layer norms.
    private static func qwenSlotRank(in n: String) -> Int {
        if n.contains(".self_attn.q_proj.weight")   { return 0 }
        if n.contains(".self_attn.k_proj.weight")   { return 1 }
        if n.contains(".self_attn.v_proj.weight")   { return 2 }
        if n.contains(".self_attn.o_proj.weight")   { return 3 }
        if n.contains(".self_attn.q_norm.weight")   { return 4 }
        if n.contains(".self_attn.k_norm.weight")   { return 5 }
        if n.contains(".linear_attn.in_proj_qkv.weight") { return 6 }
        if n.contains(".linear_attn.in_proj_z.weight")   { return 7 }
        if n.contains(".linear_attn.in_proj_a.weight")   { return 8 }
        if n.contains(".linear_attn.in_proj_b.weight")   { return 9 }
        if n.contains(".linear_attn.conv1d.weight")      { return 10 }
        if n.hasSuffix(".linear_attn.A_log")             { return 11 }
        if n.hasSuffix(".linear_attn.dt_bias")           { return 12 }
        if n.contains(".linear_attn.norm.weight")        { return 13 }
        if n.contains(".linear_attn.out_proj.weight")    { return 14 }
        if n.contains(".mlp.gate.weight")                { return 15 }
        if n.contains(".mlp.shared_expert_gate.weight")  { return 16 }
        if n.contains(".mlp.shared_expert.gate_proj.weight") { return 17 }
        if n.contains(".mlp.shared_expert.up_proj.weight")   { return 18 }
        if n.contains(".mlp.shared_expert.down_proj.weight") { return 19 }
        if n.hasSuffix(".input_layernorm.weight")        { return 20 }
        if n.hasSuffix(".post_attention_layernorm.weight") { return 21 }
        return 100
    }

    /// Within-layer slot order for the Qwen 3.8 family: QSA/full-attention
    /// projections plus the indexer bundle, then the gated-DeltaNet bundle,
    /// then router, shared-expert gate and MLP, then the per-sub-layer
    /// gated-residual (hyper-connection) weights, then the PLE projections
    /// (layer 1 only). There are no classic layer norms: the residual stream
    /// is carried by the hyper-connections and closed by
    /// `hyper_connection_mixer`.
    private static func qwen38SlotRank(in n: String) -> Int {
        if n.contains(".self_attn.q_proj.weight")   { return 0 }
        if n.contains(".self_attn.k_proj.weight")   { return 1 }
        if n.contains(".self_attn.v_proj.weight")   { return 2 }
        if n.contains(".self_attn.o_proj.weight")   { return 3 }
        if n.contains(".self_attn.q_norm.weight")   { return 4 }
        if n.contains(".self_attn.k_norm.weight")   { return 5 }
        if n.contains(".self_attn.indexer.index_qk_proj.weight") { return 6 }
        if n.contains(".self_attn.indexer.q_layernorm.weight")   { return 7 }
        if n.contains(".self_attn.indexer.k_layernorm.weight")   { return 8 }
        if n.contains(".linear_attn.in_proj_qkv.weight") { return 9 }
        if n.contains(".linear_attn.in_proj_z.weight")   { return 10 }
        if n.contains(".linear_attn.in_proj_a.weight")   { return 11 }
        if n.contains(".linear_attn.in_proj_b.weight")   { return 12 }
        if n.contains(".linear_attn.conv1d.weight")      { return 13 }
        if n.hasSuffix(".linear_attn.A_log")             { return 14 }
        if n.hasSuffix(".linear_attn.dt_bias")           { return 15 }
        if n.contains(".linear_attn.norm.weight")        { return 16 }
        if n.contains(".linear_attn.out_proj.weight")    { return 17 }
        if n.contains(".mlp.gate.weight")                { return 18 }
        if n.contains(".mlp.shared_expert_gate.weight")  { return 19 }
        if n.contains(".mlp.shared_expert.gate_proj.weight") { return 20 }
        if n.contains(".mlp.shared_expert.up_proj.weight")   { return 21 }
        if n.contains(".mlp.shared_expert.down_proj.weight") { return 22 }
        if n.contains(".attn_hyper_connection.hc_norm.weight") { return 23 }
        if n.contains(".attn_hyper_connection.input_mix_weight_down.weight") { return 24 }
        if n.contains(".attn_hyper_connection.input_mix_weight_up.weight")   { return 25 }
        if n.contains(".attn_hyper_connection.block_inject_weight.weight")   { return 26 }
        if n.contains(".mlp_hyper_connection.hc_norm.weight") { return 27 }
        if n.contains(".mlp_hyper_connection.input_mix_weight_down.weight") { return 28 }
        if n.contains(".mlp_hyper_connection.input_mix_weight_up.weight")   { return 29 }
        if n.contains(".mlp_hyper_connection.block_inject_weight.weight")   { return 30 }
        if n.hasSuffix(".ple_embedding.layer_multipliers") { return 31 }
        if n.hasSuffix(".ple_embedding.ngram_heads_offsets") { return 32 }
        if n.hasSuffix(".ple_embedding.ngram_heads_vocab_sizes") { return 33 }
        if n.contains(".ple.key_proj.weight")   { return 34 }
        if n.contains(".ple.value_proj.weight") { return 35 }
        if n.contains(".ple.conv1d.weight")     { return 36 }
        if n.contains(".ple.norm_conv.weight")  { return 37 }
        if n.contains(".ple.norm_key.weight")   { return 38 }
        if n.contains(".ple.norm_query.weight") { return 39 }
        return 100
    }

    /// Within-layer slot order. Mirrors the per-layer description in the
    /// architecture doc.
    private static func slotRank(in n: String) -> Int {
        if n.contains(".self_attn.q_proj.weight") { return 0 }
        if n.contains(".self_attn.k_proj.weight") { return 1 }
        if n.contains(".self_attn.v_proj.weight") { return 2 }
        if n.contains(".self_attn.o_proj.weight") { return 3 }
        if n.contains(".self_attn.q_norm.weight") { return 4 }
        if n.contains(".self_attn.k_norm.weight") { return 5 }
        if n.contains(".router.proj.weight")      { return 6 }
        if n.contains(".router.scale")            { return 7 }
        if n.contains(".router.per_expert_scale") { return 8 }
        if n.contains(".mlp.gate_proj.weight")    { return 9 }
        if n.contains(".mlp.up_proj.weight")      { return 10 }
        if n.contains(".mlp.down_proj.weight")    { return 11 }
        if n.hasSuffix(".input_layernorm.weight") { return 12 }
        if n.hasSuffix(".post_attention_layernorm.weight") { return 13 }
        if n.hasSuffix(".pre_feedforward_layernorm.weight") { return 14 }
        if n.hasSuffix(".pre_feedforward_layernorm_2.weight") { return 15 }
        if n.hasSuffix(".post_feedforward_layernorm.weight") { return 16 }
        if n.hasSuffix(".post_feedforward_layernorm_1.weight") { return 17 }
        if n.hasSuffix(".post_feedforward_layernorm_2.weight") { return 18 }
        if n.hasSuffix(".layer_scalar")           { return 19 }
        return 100
    }
}
