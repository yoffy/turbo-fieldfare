import Foundation
@testable import TurboFieldfare
@testable import TurboFieldfareRepackCore

/// Synthetic Qwen 3.8 toy fixture: a tiny runnable `.gturbo/` directory with
/// the qwen38 tensor-name contract (hyper-connections on every sub-layer,
/// QSA indexer on mask-1 layers, GDN on mask-2 layers, PLE tables on layer 1,
/// 4-bit group-32 base with 8-bit group-64 router/scalar gate, untied
/// lm_head, SSD-streamed `packed_ple/` shards). Mirrors `QwenToySynthetic`
/// for the 3.6 family.
enum Qwen38ToySynthetic {

    static let pleShardRows = 128
    static let pleHeadDim = 32
    static let pleHeads = 4
    /// Per-head table rows and base offsets of the toy n-gram geometry.
    static let pleHeadVocabs = [64, 64, 64, 64]
    static let pleHeadOffsets = [0, 64, 128, 192]

    /// Build the toy directory in a temp dir and return its URL.
    static func write() throws -> URL {
        let toy = ArchConfig.qwen38Toy()
        let hcHidden = toy.hcCount * toy.hiddenSize
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-qwen38-toy-\(UUID().uuidString)")
        let exp = dir.appendingPathComponent("packed_experts")
        let pleDir = dir.appendingPathComponent("packed_ple")
        try FileManager.default.createDirectory(at: exp, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: pleDir, withIntermediateDirectories: true)

        struct ResidentSpec {
            let name: String
            let dtype: UInt8
            let shape: [UInt32]
            let weightBytes: UInt64
            let scaleBytes: UInt64
            let biasBytes: UInt64
            let groupSize: Int
        }

        let d = toy.hiddenSize
        let u16 = MemoryLayout<UInt16>.stride

        func int4Spec(_ name: String, rows: Int, cols: Int, group: Int = 32) -> ResidentSpec {
            let groups = cols / group
            let auxBytes = UInt64(rows * groups * u16)
            return ResidentSpec(name: name, dtype: 0,
                                shape: [UInt32(rows), UInt32(cols), 0, 0],
                                weightBytes: UInt64(rows * cols / 2),
                                scaleBytes: auxBytes, biasBytes: auxBytes,
                                groupSize: group)
        }

        func int8Spec(_ name: String, rows: Int, cols: Int) -> ResidentSpec {
            let groups = cols / 64
            let auxBytes = UInt64(rows * groups * u16)
            return ResidentSpec(name: name, dtype: 0,
                                shape: [UInt32(rows), UInt32(cols), 0, 0],
                                weightBytes: UInt64(rows * cols),
                                scaleBytes: auxBytes, biasBytes: auxBytes,
                                groupSize: 64)
        }

        func bf16Spec(_ name: String, shape: [UInt32], count: Int) -> ResidentSpec {
            ResidentSpec(name: name, dtype: 1, shape: shape,
                         weightBytes: UInt64(count * u16),
                         scaleBytes: 0, biasBytes: 0, groupSize: 64)
        }

        func i64Spec(_ name: String, count: Int) -> ResidentSpec {
            ResidentSpec(name: name, dtype: 4,
                         shape: [UInt32(count), 0, 0, 0],
                         weightBytes: UInt64(count * 8),
                         scaleBytes: 0, biasBytes: 0, groupSize: 64)
        }

        let la = toy.linearAttention
        let qDim = toy.numHeads * toy.fullHeadDim
        let kvDim = toy.numFullKVHeads * toy.fullHeadDim
        let idxRows = (toy.indexerNHeads + toy.indexerKvHeads) * toy.indexerHeadDim

        // 1. Resident specs.
        var specs: [ResidentSpec] = [
            int4Spec("language_model.model.embed_tokens.weight",
                     rows: toy.vocabSize, cols: d),
            int4Spec("language_model.lm_head.weight",
                     rows: toy.vocabSize, cols: d),
            bf16Spec("language_model.model.hyper_connection_mixer.hc_norm.weight",
                     shape: [UInt32(hcHidden), 0, 0, 0], count: hcHidden),
            int4Spec("language_model.model.hyper_connection_mixer.input_mix_weight_down.weight",
                     rows: toy.hcLowrank, cols: hcHidden),
            int4Spec("language_model.model.hyper_connection_mixer.input_mix_weight_up.weight",
                     rows: hcHidden, cols: toy.hcLowrank),
        ]
        for L in 0..<toy.numLayers {
            let prefix = "language_model.model.layers.\(L)"
            // MoE trunk (router + scalar gate are 8-bit group-64 overrides).
            specs.append(int8Spec("\(prefix).mlp.gate.weight",
                                  rows: toy.numExperts, cols: d))
            specs.append(int8Spec("\(prefix).mlp.shared_expert_gate.weight",
                                  rows: 1, cols: d))
            specs.append(int4Spec("\(prefix).mlp.shared_expert.gate_proj.weight",
                                  rows: toy.intermediateSize, cols: d))
            specs.append(int4Spec("\(prefix).mlp.shared_expert.up_proj.weight",
                                  rows: toy.intermediateSize, cols: d))
            specs.append(int4Spec("\(prefix).mlp.shared_expert.down_proj.weight",
                                  rows: d, cols: toy.intermediateSize))
            // Hyper-connections (both sub-layers).
            for hcName in ["attn_hyper_connection", "mlp_hyper_connection"] {
                specs.append(bf16Spec("\(prefix).\(hcName).hc_norm.weight",
                                      shape: [UInt32(hcHidden), 0, 0, 0],
                                      count: hcHidden))
                specs.append(int4Spec("\(prefix).\(hcName).input_mix_weight_down.weight",
                                      rows: toy.hcLowrank, cols: hcHidden))
                specs.append(int4Spec("\(prefix).\(hcName).input_mix_weight_up.weight",
                                      rows: hcHidden, cols: toy.hcLowrank))
                specs.append(int4Spec("\(prefix).\(hcName).block_inject_weight.weight",
                                      rows: toy.hcCount, cols: hcHidden))
            }
            if toy.layerIsLinear(L) {
                specs.append(int4Spec("\(prefix).linear_attn.in_proj_qkv.weight",
                                      rows: la.qkvDim, cols: d))
                specs.append(int4Spec("\(prefix).linear_attn.in_proj_z.weight",
                                      rows: la.valueDim, cols: d))
                specs.append(int4Spec("\(prefix).linear_attn.in_proj_a.weight",
                                      rows: la.numVHeads, cols: d))
                specs.append(int4Spec("\(prefix).linear_attn.in_proj_b.weight",
                                      rows: la.numVHeads, cols: d))
                specs.append(int4Spec("\(prefix).linear_attn.out_proj.weight",
                                      rows: d, cols: la.valueDim))
                specs.append(bf16Spec("\(prefix).linear_attn.conv1d.weight",
                                      shape: [UInt32(la.qkvDim), UInt32(la.convKernelSize), 1, 0],
                                      count: la.qkvDim * la.convKernelSize))
                specs.append(bf16Spec("\(prefix).linear_attn.A_log",
                                      shape: [UInt32(la.numVHeads), 0, 0, 0],
                                      count: la.numVHeads))
                specs.append(bf16Spec("\(prefix).linear_attn.dt_bias",
                                      shape: [UInt32(la.numVHeads), 0, 0, 0],
                                      count: la.numVHeads))
                specs.append(bf16Spec("\(prefix).linear_attn.norm.weight",
                                      shape: [UInt32(la.valueHeadDim), 0, 0, 0],
                                      count: la.valueHeadDim))
            } else {
                specs.append(int4Spec("\(prefix).self_attn.q_proj.weight",
                                      rows: 2 * qDim, cols: d))
                specs.append(int4Spec("\(prefix).self_attn.k_proj.weight",
                                      rows: kvDim, cols: d))
                specs.append(int4Spec("\(prefix).self_attn.v_proj.weight",
                                      rows: kvDim, cols: d))
                specs.append(int4Spec("\(prefix).self_attn.o_proj.weight",
                                      rows: d, cols: qDim))
                specs.append(bf16Spec("\(prefix).self_attn.q_norm.weight",
                                      shape: [UInt32(toy.fullHeadDim), 0, 0, 0],
                                      count: toy.fullHeadDim))
                specs.append(bf16Spec("\(prefix).self_attn.k_norm.weight",
                                      shape: [UInt32(toy.fullHeadDim), 0, 0, 0],
                                      count: toy.fullHeadDim))
                specs.append(int4Spec("\(prefix).self_attn.indexer.index_qk_proj.weight",
                                      rows: idxRows, cols: d))
                specs.append(bf16Spec("\(prefix).self_attn.indexer.q_layernorm.weight",
                                      shape: [UInt32(toy.indexerHeadDim), 0, 0, 0],
                                      count: toy.indexerHeadDim))
                specs.append(bf16Spec("\(prefix).self_attn.indexer.k_layernorm.weight",
                                      shape: [UInt32(toy.indexerHeadDim), 0, 0, 0],
                                      count: toy.indexerHeadDim))
            }
            // PLE resident tables on layer 1 only.
            if L == 1 {
                specs.append(i64Spec("\(prefix).ple.ple_embedding.layer_multipliers",
                                     count: toy.pleNgramSize))
                specs.append(i64Spec("\(prefix).ple.ple_embedding.ngram_heads_offsets",
                                     count: pleHeads))
                specs.append(i64Spec("\(prefix).ple.ple_embedding.ngram_heads_vocab_sizes",
                                     count: pleHeads))
                specs.append(int4Spec("\(prefix).ple.key_proj.weight",
                                      rows: hcHidden, cols: toy.pleEmbedDim))
                specs.append(int4Spec("\(prefix).ple.value_proj.weight",
                                      rows: d, cols: toy.pleEmbedDim))
                specs.append(bf16Spec("\(prefix).ple.conv1d.weight",
                                      shape: [UInt32(hcHidden), 4, 1, 0],
                                      count: hcHidden * 4))
                specs.append(bf16Spec("\(prefix).ple.norm_conv.weight",
                                      shape: [UInt32(hcHidden), 0, 0, 0],
                                      count: hcHidden))
                specs.append(bf16Spec("\(prefix).ple.norm_key.weight",
                                      shape: [UInt32(hcHidden), 0, 0, 0],
                                      count: hcHidden))
                specs.append(bf16Spec("\(prefix).ple.norm_query.weight",
                                      shape: [UInt32(hcHidden), 0, 0, 0],
                                      count: hcHidden))
            }
        }

        // 2. Serialize the resident index + payload.
        let names = specs.map(\.name)
        let stringTable = names.joined().data(using: .utf8)!
        let headerBytes = GTurboBinary.indexHeaderBytes
        let entryBytes = GTurboBinary.indexEntryBytes
        let entriesBase = headerBytes
        let stringTableBase = entriesBase + names.count * entryBytes
        var nameAbsOffsets: [UInt32] = []
        var cursor = 0
        for n in names {
            nameAbsOffsets.append(UInt32(stringTableBase + cursor))
            cursor += n.utf8.count
        }
        let indexBytes = UInt64(stringTableBase + stringTable.count)

        var entries: [ResidentEntry] = []
        entries.reserveCapacity(specs.count)
        var payloadCursor = indexBytes
        for spec in specs {
            let weightOffset = payloadCursor
            let scaleOffset = spec.scaleBytes > 0 ? weightOffset + spec.weightBytes : 0
            let biasOffset = spec.biasBytes > 0 ? scaleOffset + spec.scaleBytes : 0
            entries.append(ResidentEntry(
                name: spec.name,
                dtype: spec.dtype,
                logicalShape4: spec.shape,
                fileOffset: weightOffset,
                sizeBytes: spec.weightBytes,
                scaleOffset: scaleOffset,
                scaleSize: spec.scaleBytes,
                biasOffset: biasOffset,
                biasSize: spec.biasBytes,
                quantSpec: nil,
                sourceWeight: ModelLoaderTests.dummySource(spec.name),
                sourceScales: nil,
                sourceBiases: nil))
            payloadCursor += spec.weightBytes + spec.scaleBytes + spec.biasBytes
        }
        let residentSize = payloadCursor - indexBytes

        let totalBytes = Int(indexBytes + residentSize)
        var fileBuf = [UInt8](repeating: 0, count: totalBytes)
        fileBuf.withUnsafeMutableBytes { raw in
            let base = raw.baseAddress!
            GTurboBinary.writeIndexHeader(into: base,
                                          indexSize: indexBytes,
                                          residentSize: residentSize,
                                          entryCount: UInt64(entries.count))
            for (i, e) in entries.enumerated() {
                let dst = base.advanced(by: entriesBase + i * entryBytes)
                GTurboBinary.writeIndexEntry(into: dst, entry: e,
                                             nameOffset: nameAbsOffsets[i])
            }
            _ = stringTable.withUnsafeBytes { sb in
                memcpy(base.advanced(by: stringTableBase), sb.baseAddress!, stringTable.count)
            }
            for (spec, entry) in zip(specs, entries) {
                if entry.dtype == 0 {
                    memset(base.advanced(by: Int(entry.fileOffset)), 0x11, Int(entry.sizeBytes))
                    if entry.scaleSize > 0 {
                        let scales = base.advanced(by: Int(entry.scaleOffset))
                            .assumingMemoryBound(to: UInt16.self)
                        for i in 0..<(Int(entry.scaleSize) / u16) {
                            scales[i] = Quantization.bf16Bits(0.01)
                        }
                    }
                } else if entry.dtype == 1 {
                    let dst = base.advanced(by: Int(entry.fileOffset))
                        .assumingMemoryBound(to: UInt16.self)
                    for i in 0..<(Int(entry.sizeBytes) / u16) {
                        dst[i] = Quantization.bf16Bits(1.0)
                    }
                } else if entry.dtype == 4 {
                    // I64 tables: deterministic small values per tensor kind.
                    let dst = base.advanced(by: Int(entry.fileOffset))
                        .assumingMemoryBound(to: UInt64.self)
                    let count = Int(entry.sizeBytes) / 8
                    if spec.name.hasSuffix(".layer_multipliers") {
                        let mults: [UInt64] = [11, 101, 1009]
                        for i in 0..<count { dst[i] = mults[i % mults.count] }
                    } else if spec.name.hasSuffix(".ngram_heads_offsets") {
                        for i in 0..<count { dst[i] = UInt64(pleHeadOffsets[i % pleHeadOffsets.count]) }
                    } else {
                        for i in 0..<count { dst[i] = UInt64(pleHeadVocabs[i % pleHeadVocabs.count]) }
                    }
                }
            }
        }
        let weightsURL = dir.appendingPathComponent("model_weights.bin")
        try Data(fileBuf).write(to: weightsURL)
        let weightsSha = try Sha256Verifier.hashFile(at: weightsURL)

        // 3. Packed experts: int4 group-32 gate/up/down per expert.
        func appendU16(_ values: [UInt16], to bytes: inout [UInt8]) {
            for value in values {
                bytes.append(UInt8(truncatingIfNeeded: value))
                bytes.append(UInt8(truncatingIfNeeded: value >> 8))
            }
        }

        func toyExpertRows(rows: Int, cols: Int, expert: Int, role: Int) -> [[Float]] {
            (0..<rows).map { row in
                (0..<cols).map { col in
                    Float(expert + 1) * 0.001
                        + Float(role + 1) * 0.003
                        + Float((row % 7) - 3) * 0.0004
                        + Float((col % 11) - 5) * 0.0002
                }
            }
        }

        func toyExpertBlob(expert: Int) -> (bytes: [UInt8], tensors: [String: [String: Any]]) {
            var bytes: [UInt8] = []
            var tensors: [String: [String: Any]] = [:]

            func addProjection(prefix: String, rows: Int, cols: Int, role: Int) {
                let projectionRows = toyExpertRows(rows: rows, cols: cols,
                                                   expert: expert, role: role)
                let quantized = projectionRows.map {
                    Quantization.quantizeInt4Affine($0, groupSize: 32)
                }
                let packedOffset = bytes.count
                for row in quantized { bytes.append(contentsOf: row.packed) }
                tensors[prefix] = [
                    "offset": packedOffset, "size": bytes.count - packedOffset,
                    "dtype": "U32", "shape": [rows, cols],
                    "bits": 4,
                ]
                let scalesOffset = bytes.count
                for row in quantized { appendU16(row.scales, to: &bytes) }
                tensors["\(prefix)_scales"] = [
                    "offset": scalesOffset, "size": bytes.count - scalesOffset,
                    "dtype": "BF16", "shape": [rows, cols / 32],
                ]
                let biasesOffset = bytes.count
                for row in quantized { appendU16(row.biases, to: &bytes) }
                tensors["\(prefix)_biases"] = [
                    "offset": biasesOffset, "size": bytes.count - biasesOffset,
                    "dtype": "BF16", "shape": [rows, cols / 32],
                ]
            }

            addProjection(prefix: "gate", rows: toy.moeIntermediateSize, cols: d, role: 0)
            addProjection(prefix: "up", rows: toy.moeIntermediateSize, cols: d, role: 1)
            addProjection(prefix: "down", rows: d, cols: toy.moeIntermediateSize, role: 2)
            return (bytes, tensors)
        }

        let expertStride: UInt64 = 16384
        let layerBytes = Int(expertStride) * toy.numExperts
        for L in 0..<toy.numLayers {
            var payload = Data(count: layerBytes)
            for E in 0..<toy.numExperts {
                let blob = toyExpertBlob(expert: E).bytes
                let baseB = E * Int(expertStride)
                precondition(blob.count <= Int(expertStride),
                             "toy expert blob exceeds stride")
                for (i, byte) in blob.enumerated() {
                    payload[baseB + i] = byte
                }
            }
            let url = exp.appendingPathComponent(String(format: "layer_%02d.bin", L))
            try payload.write(to: url)
        }
        var layerShaByName: [String: String] = [:]
        for L in 0..<toy.numLayers {
            let basename = String(format: "layer_%02d.bin", L)
            let url = exp.appendingPathComponent(basename)
            layerShaByName["packed_experts/\(basename)"] = try Sha256Verifier.hashFile(at: url)
        }

        // 4. layout.json
        var layersArr: [[String: Any]] = []
        for L in 0..<toy.numLayers {
            var experts: [[String: Any]] = []
            for E in 0..<toy.numExperts {
                let blob = toyExpertBlob(expert: E)
                experts.append([
                    "expert": E,
                    "offset": UInt64(E) * expertStride,
                    "size":   expertStride,
                    "tensors": blob.tensors,
                ])
            }
            layersArr.append([
                "layer": L,
                "file": String(format: "layer_%02d.bin", L),
                "experts": experts,
            ])
        }
        let layoutRoot: [String: Any] = [
            "expertStride": expertStride,
            "numLayers": toy.numLayers,
            "expertsPerLayer": toy.numExperts,
            "layers": layersArr,
        ]
        let layoutData = try JSONSerialization.data(
            withJSONObject: layoutRoot, options: [.sortedKeys])
        let layoutURL = exp.appendingPathComponent("layout.json")
        try layoutData.write(to: layoutURL)
        let layoutSha = try Sha256Verifier.hashFile(at: layoutURL)

        // 5. packed_ple shards: int4 group-32 [shardRows, headDim] tables.
        let pleHeadsTotal = pleHeadVocabs.reduce(0, +)
        precondition(pleHeadsTotal == pleShardRows * toy.pleSplitParts)
        var pleFiles: [String: [String: Any]] = [:]
        var pleShardsArr: [[String: Any]] = []
        for shard in 0..<toy.pleSplitParts {
            // File layout is weight | scales | biases (reader contract).
            var packed: [UInt8] = []
            var scalesBytes: [UInt8] = []
            var biasesBytes: [UInt8] = []
            for row in 0..<pleShardRows {
                let raw = (0..<pleHeadDim).map { col in
                    Float(shard + 1) * 0.01
                        + Float(row % 13) * 0.001
                        + Float(col % 7) * 0.0007
                }
                let q = Quantization.quantizeInt4Affine(raw, groupSize: 32)
                packed.append(contentsOf: q.packed)
                appendU16(q.scales, to: &scalesBytes)
                appendU16(q.biases, to: &biasesBytes)
            }
            let weightSize = packed.count
            let scalesSize = scalesBytes.count
            let fileBytes = packed + scalesBytes + biasesBytes
            let basename = String(format: "shard_%03d.bin", shard)
            try Data(fileBytes).write(to: pleDir.appendingPathComponent(basename))
            let sha = try Sha256Verifier.hashFile(at: pleDir.appendingPathComponent(basename))
            pleFiles["packed_ple/\(basename)"] = [
                "size": fileBytes.count, "sha256": sha,
            ]
            pleShardsArr.append([
                "shard": shard,
                "file": basename,
                "weightOffset": 0,
                "weightSize": weightSize,
                "scalesOffset": weightSize,
                "scalesSize": scalesSize,
                "biasesOffset": weightSize + scalesSize,
                "biasesSize": biasesBytes.count,
                "logicalShape": [pleShardRows, pleHeadDim],
                "bits": 4,
                "groupSize": 32,
            ])
        }
        let pleLayoutRoot: [String: Any] = [
            "layer": 1,
            "shardCount": toy.pleSplitParts,
            "shards": pleShardsArr,
        ]
        let pleLayoutData = try JSONSerialization.data(
            withJSONObject: pleLayoutRoot, options: [.sortedKeys])
        try pleLayoutData.write(to: pleDir.appendingPathComponent("layout.json"))
        let pleLayoutSha = try Sha256Verifier.hashFile(
            at: pleDir.appendingPathComponent("layout.json"))
        pleFiles["packed_ple/layout.json"] = [
            "size": pleLayoutData.count, "sha256": pleLayoutSha,
        ]

        // 6. manifest.json (arch v2 with the qwen38 family fields).
        var files: [String: [String: Any]] = [
            "model_weights.bin": ["size": Int(totalBytes), "sha256": weightsSha],
            "packed_experts/layout.json": ["size": layoutData.count, "sha256": layoutSha],
        ]
        for (rel, sha) in layerShaByName {
            files[rel] = ["size": layerBytes, "sha256": sha]
        }
        for (rel, entry) in pleFiles {
            files[rel] = entry
        }

        let la2 = toy.linearAttention
        let archDict: [String: Any] = [
            "hiddenSize": toy.hiddenSize, "ffnIntermediate": toy.intermediateSize,
            "moeIntermediateSize": toy.moeIntermediateSize,
            "numHeads": toy.numHeads, "numKVHeads": toy.numKVHeads,
            "numFullKVHeads": toy.numFullKVHeads,
            "headDim": toy.headDim, "fullHeadDim": toy.fullHeadDim,
            "vocabSize": toy.vocabSize, "slidingWindow": toy.slidingWindow,
            "finalLogitSoftcap": toy.finalLogitSoftcap,
            "ropeTheta": toy.ropeTheta, "fullRopeTheta": toy.fullRopeTheta,
            "partialRotaryFactor": toy.partialRotaryFactor,
            "numLayers": toy.numLayers, "numExperts": toy.numExperts,
            "topKExperts": toy.topKExperts,
            "tieWordEmbeddings": toy.tieWordEmbeddings,
            "attentionKEqV": toy.attentionKEqV,
            "hiddenActivation": toy.hiddenActivation,
            "fullAttentionLayerMask": toy.fullAttentionLayerMask.map { Int($0) },
            "family": toy.family.rawValue,
            "attnOutputGate": toy.attnOutputGate,
            "attentionScale": toy.attentionScale,
            "embeddingScaledBySqrtHidden": toy.embeddingScaledBySqrtHidden,
            "routerScaled": toy.routerScaled,
            "ffnSandwichNorms": toy.ffnSandwichNorms,
            "sharedExpertGated": toy.sharedExpertGated,
            "ropeNeoxSubdim": toy.ropeNeoxSubdim,
            "linearNumKHeads": la2.numKHeads,
            "linearNumVHeads": la2.numVHeads,
            "linearKeyHeadDim": la2.keyHeadDim,
            "linearValueHeadDim": la2.valueHeadDim,
            "linearConvKernelSize": la2.convKernelSize,
            "hcCount": toy.hcCount,
            "hcLowrank": toy.hcLowrank,
            "pleEmbedDim": toy.pleEmbedDim,
            "pleNgramSize": toy.pleNgramSize,
            "pleSplitParts": toy.pleSplitParts,
            "pleHeadsPerNgram": toy.pleHeadsPerNgram,
            "indexerBudget": toy.indexerBudget,
            "indexerCompressRatio": toy.indexerCompressRatio,
            "indexerHeadDim": toy.indexerHeadDim,
            "indexerKvHeads": toy.indexerKvHeads,
            "indexerNHeads": toy.indexerNHeads,
        ]
        let quantDict: [String: Any] = [
            "embedding": ["weightBits": 4, "scheme": "affine",
                          "scaleType": "bf16", "biasType": "bf16", "groupSize": 32],
            "attention": ["weightBits": 4, "scheme": "affine",
                          "scaleType": "bf16", "biasType": "bf16", "groupSize": 32],
            "router": ["weightBits": 8, "scheme": "affine",
                       "scaleType": "bf16", "biasType": "bf16", "groupSize": 64],
            "sharedExpert": ["weightBits": 4, "scheme": "affine",
                             "scaleType": "bf16", "biasType": "bf16", "groupSize": 32],
            "routedExpert": ["weightBits": 4, "scheme": "affine",
                             "scaleType": "bf16", "biasType": "bf16", "groupSize": 32],
        ]
        let manifestRoot: [String: Any] = [
            "magic": "GTURBO",
            "versionMajor": 1,
            "versionMinor": 0,
            "flags": ["streamingPresent": true, "turboQuantKV": false, "aneSharedExpert": false],
            "modelID": "qwen38-toy",
            "arch": archDict,
            "quant": quantDict,
            "files": files,
            "expertsPerLayer": toy.numExperts,
            "numLayers": toy.numLayers,
            "expertStride": expertStride,
            "pleShards": toy.pleSplitParts,
        ]
        let manifestData = try JSONSerialization.data(withJSONObject: manifestRoot,
            options: [.sortedKeys, .withoutEscapingSlashes])
        try manifestData.write(to: dir.appendingPathComponent("manifest.json"))
        return dir
    }
}

extension ArchConfig {
    /// Tiny Qwen 3.8 baseline: 4 layers alternating gated-DeltaNet linear
    /// (mask 2) and QSA sparse (mask 1), 16 experts (top-10 to exercise the
    /// k10 path), 4-branch hyper-connections, PLE on layer 1 (2 shards),
    /// 4-bit group-32 base with 8-bit group-64 router/scalar gate. Dimensions
    /// respect every kernel constraint (D % 64 for the INT8 router, N % 32
    /// for all INT4 GEMVs, keyHeadDim % 32, even rotaryDim).
    static func qwen38Toy() -> ArchConfig {
        ArchConfig(
            hiddenSize: 128,
            intermediateSize: 64,
            moeIntermediateSize: 64,
            numHeads: 4,
            numKVHeads: 2,
            numFullKVHeads: 2,
            headDim: 64,
            fullHeadDim: 64,
            vocabSize: 256,
            slidingWindow: 0,
            finalLogitSoftcap: 0.0,
            ropeTheta: 10_000_000.0,
            fullRopeTheta: 10_000_000.0,
            partialRotaryFactor: 0.25,
            numLayers: 4,
            numExperts: 16,
            topKExperts: 10,
            tieWordEmbeddings: false,
            attentionKEqV: false,
            fullAttentionLayerMask: [2, 1, 2, 1],
            hiddenActivation: "silu",
            family: .qwen38,
            attnOutputGate: true,
            attentionScale: 0.125,
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: true,
            ropeNeoxSubdim: true,
            linearAttention: LinearAttentionConfig(
                numKHeads: 2, numVHeads: 4,
                keyHeadDim: 32, valueHeadDim: 32,
                convKernelSize: 4),
            hcCount: 2,
            hcLowrank: 32,
            pleEmbedDim: 128,
            pleNgramSize: 3,
            pleSplitParts: 2,
            pleHeadsPerNgram: 2,
            indexerBudget: 8,
            indexerCompressRatio: 2,
            indexerHeadDim: 32,
            indexerKvHeads: 1,
            indexerNHeads: 2
        )
    }
}
