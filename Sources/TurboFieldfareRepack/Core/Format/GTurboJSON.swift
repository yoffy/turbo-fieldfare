import Foundation

/// JSON encoders for `manifest.json` and `packed_experts/layout.json`. The
/// files are small (kilobytes), so we use Foundation's `JSONSerialization`
/// rather than streaming.
enum GTurboJSON {

    static let magic = "GTURBO"
    static let versionMajor = 1
    static let versionMinor = 0

    struct FileEntry {
        let size: UInt64
        let sha256: String
    }

    struct QuantBitWidths {
        var embedding: Int
        var attention: Int
        var router: Int
        var sharedExpert: Int
        var routedExpert: Int
        /// Per-slot row-group widths. Older checkpoints are uniformly 64;
        /// Qwen 3.8 mixes a 32-wide base with 64-wide gate overrides.
        var embeddingGroup: Int = 64
        var attentionGroup: Int = 64
        var routerGroup: Int = 64
        var sharedExpertGroup: Int = 64
        var routedExpertGroup: Int = 64
    }

    static func encodeManifest(plan: RepackPlan,
                                      modelID: String,
                                      sourceSnapshotHash: String,
                                      files: [(relativePath: String, info: FileEntry)],
                                      expertsPerLayer: Int,
                                      numLayers: Int,
                                      expertStride: UInt64,
                                      bitWidths: QuantBitWidths) throws -> Data {
        let arch = plan.arch
        var archDict: [String: Any] = [
            "hiddenSize": arch.hiddenSize,
            "ffnIntermediate": arch.intermediateSize,
            "moeIntermediateSize": arch.moeIntermediateSize,
            "numHeads": arch.numHeads,
            "numKVHeads": arch.numKVHeads,
            "numFullKVHeads": arch.numFullKVHeads,
            "headDim": arch.headDim,
            "fullHeadDim": arch.fullHeadDim,
            "vocabSize": arch.vocabSize,
            "slidingWindow": arch.slidingWindow,
            "finalLogitSoftcap": arch.finalLogitSoftcap,
            "ropeTheta": arch.ropeTheta,
            "fullRopeTheta": arch.fullRopeTheta,
            "partialRotaryFactor": arch.partialRotaryFactor,
            "numLayers": arch.numLayers,
            "numExperts": arch.numExperts,
            "topKExperts": arch.topKExperts,
            "tieWordEmbeddings": arch.tieWordEmbeddings,
            "attentionKEqV": arch.attentionKEqV,
            "hiddenActivation": arch.hiddenActivation,
            "fullAttentionLayerMask": arch.fullAttentionLayerMask.map { Int($0) }
        ]
        // Family extension fields. Gemma manifests omit them (byte-identical
        // to the pre-family format); the reader treats absence as the Gemma
        // defaults.
        if arch.family != .gemma4 {
            archDict["family"] = arch.family.rawValue
            archDict["attnOutputGate"] = arch.attnOutputGate
            archDict["attentionScale"] = arch.attentionScale
            archDict["embeddingScaledBySqrtHidden"] = arch.embeddingScaledBySqrtHidden
            archDict["routerScaled"] = arch.routerScaled
            archDict["ffnSandwichNorms"] = arch.ffnSandwichNorms
            archDict["sharedExpertGated"] = arch.sharedExpertGated
            archDict["ropeNeoxSubdim"] = arch.ropeNeoxSubdim
            archDict["linearNumKHeads"] = arch.linearNumKHeads
            archDict["linearNumVHeads"] = arch.linearNumVHeads
            archDict["linearKeyHeadDim"] = arch.linearKeyHeadDim
            archDict["linearValueHeadDim"] = arch.linearValueHeadDim
            archDict["linearConvKernelSize"] = arch.linearConvKernelSize
        }
        if arch.family == .qwen38 {
            archDict["hcCount"] = arch.hcCount
            archDict["hcLowrank"] = arch.hcLowrank
            archDict["pleEmbedDim"] = arch.pleEmbedDim
            archDict["pleNgramSize"] = arch.pleNgramSize
            archDict["pleSplitParts"] = arch.pleSplitParts
            archDict["pleHeadsPerNgram"] = arch.pleHeadsPerNgram
            archDict["indexerBudget"] = arch.indexerBudget
            archDict["indexerCompressRatio"] = arch.indexerCompressRatio
            archDict["indexerHeadDim"] = arch.indexerHeadDim
            archDict["indexerKvHeads"] = arch.indexerKvHeads
            archDict["indexerNHeads"] = arch.indexerNHeads
        }
        let quantBits: [(String, Int, Int)] = [
            ("embedding", bitWidths.embedding, bitWidths.embeddingGroup),
            ("attention", bitWidths.attention, bitWidths.attentionGroup),
            ("router", bitWidths.router, bitWidths.routerGroup),
            ("sharedExpert", bitWidths.sharedExpert, bitWidths.sharedExpertGroup),
            ("routedExpert", bitWidths.routedExpert, bitWidths.routedExpertGroup),
        ]
        var quantDict: [String: Any] = [:]
        for (slot, bits, group) in quantBits {
            quantDict[slot] = [
                "weightBits": bits,
                "scheme": plan.baseMode,
                "scaleType": "BF16",
                "biasType": "BF16",
                "groupSize": group
            ]
        }

        var filesDict: [String: Any] = [:]
        for (path, info) in files {
            filesDict[path] = ["size": info.size, "sha256": info.sha256]
        }

        var manifest: [String: Any] = [
            "magic": GTurboJSON.magic,
            "versionMajor": GTurboJSON.versionMajor,
            "versionMinor": GTurboJSON.versionMinor,
            "flags": [
                "streamingPresent": true,
                "turboQuantKV": false,
                "aneSharedExpert": false
            ],
            "modelID": modelID,
            "sourceSnapshotHash": sourceSnapshotHash,
            "arch": archDict,
            "quant": quantDict,
            "files": filesDict,
            "expertsPerLayer": expertsPerLayer,
            "numLayers": numLayers,
            "expertStride": expertStride,
            "bitWidthOverridesHonored": plan.bitsOverrideCount
        ]
        // Only qwen38 installs carry PLE shards; older manifests stay
        // byte-identical to the pre-PLE format.
        if !plan.pleShards.isEmpty {
            manifest["pleShards"] = plan.pleShards.count
        }
        return try JSONSerialization.data(withJSONObject: manifest,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    }

    static func encodeLayout(plan: RepackPlan,
                                    expertStride: UInt64) throws -> Data {
        let arch = plan.arch
        var layersArr: [[String: Any]] = []
        layersArr.reserveCapacity(plan.layers.count)
        for lp in plan.layers {
            let layerFile = (lp.path as NSString).lastPathComponent
            var experts: [[String: Any]] = []
            experts.reserveCapacity(lp.expertsPerLayer)
            for e in 0..<lp.expertsPerLayer {
                let base = UInt64(e) * lp.expertStride
                var tensors: [String: Any] = [:]
                for slice in lp.subTensors {
                    let key: String
                    switch slice.component {
                    case "weights": key = slice.role
                    case "scales":  key = slice.role + "_scales"
                    case "biases":  key = slice.role + "_biases"
                    default:        key = slice.role + "_" + slice.component
                    }
                    var t: [String: Any] = [
                        "offset": slice.offsetInExpertBlob,
                        "size":   slice.sizeInExpertBlob,
                        "dtype":  slice.dtype == 0 ? "U32" : "BF16",
                        "shape":  slice.logicalShape.map { Int($0) }
                    ]
                    if let bits = slice.bitsForWeights { t["bits"] = bits }
                    tensors[key] = t
                }
                let expertEntry: [String: Any] = [
                    "expert": e,
                    "offset": base,
                    "size":   lp.expertStride,
                    "tensors": tensors
                ]
                experts.append(expertEntry)
            }
            layersArr.append([
                "layer": lp.layerIndex,
                "file":  layerFile,
                "experts": experts
            ])
        }
        let obj: [String: Any] = [
            "expertStride": expertStride,
            "numLayers": arch.numLayers,
            "expertsPerLayer": plan.layers.first?.expertsPerLayer ?? 0,
            "layers": layersArr
        ]
        return try JSONSerialization.data(withJSONObject: obj,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    }

    /// Encodes `packed_ple/layout.json`: one entry per SSD-streamed PLE
    /// n-gram shard. Each shard file holds the quantized weight bytes
    /// followed by BF16 scales then biases; the runtime PLE streamer (a
    /// later inference PR) addresses shards through this index.
    static func encodePleLayout(plan: RepackPlan) throws -> Data {
        var shardsArr: [[String: Any]] = []
        shardsArr.reserveCapacity(plan.pleShards.count)
        for shard in plan.pleShards {
            shardsArr.append([
                "shard": shard.shardIndex,
                "file": (shard.path as NSString).lastPathComponent,
                "weightOffset": 0,
                "weightSize": shard.weightBytes,
                "scalesOffset": shard.weightBytes,
                "scalesSize": shard.scalesBytes,
                "biasesOffset": shard.weightBytes + shard.scalesBytes,
                "biasesSize": shard.biasesBytes,
                "logicalShape": shard.logicalShape.map { Int($0) },
                "bits": shard.bits,
                "groupSize": shard.groupSize,
            ])
        }
        let obj: [String: Any] = [
            "layer": plan.pleShards.first?.layerIndex ?? 0,
            "shardCount": plan.pleShards.count,
            "shards": shardsArr
        ]
        return try JSONSerialization.data(withJSONObject: obj,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    }
}
