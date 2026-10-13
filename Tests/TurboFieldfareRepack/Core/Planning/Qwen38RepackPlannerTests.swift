import Darwin
import Foundation
import Testing
@testable import TurboFieldfareRepackCore

@Suite
struct Qwen38RepackPlannerTests {

    @Test func qwen38ArchInfoLoadsFromSyntheticConfig() throws {
        let snapshotDir = temporaryRoot("qwen38-arch")
        defer { try? FileManager.default.removeItem(atPath: snapshotDir) }
        _ = try SyntheticSnapshot.buildQwen38(at: snapshotDir)

        let arch = try ArchInfo.load(
            configPath: (snapshotDir as NSString).appendingPathComponent("config.json"))

        #expect(arch.family == .qwen38)
        #expect(arch.hiddenSize == 128)
        #expect(arch.numLayers == 4)
        #expect(arch.fullAttentionLayerMask == [2, 2, 2, 1])
        #expect(arch.numExperts == 2)
        #expect(arch.topKExperts == 2)
        #expect(arch.moeIntermediateSize == 64)
        #expect(arch.intermediateSize == 64)
        #expect(arch.numHeads == 2)
        #expect(arch.tieWordEmbeddings == false)
        #expect(arch.attentionKEqV == false)
        #expect(arch.hiddenActivation == "silu")
        #expect(arch.attnOutputGate == true)
        #expect(arch.attentionScale == 0.125)   // 64^-0.5
        #expect(arch.embeddingScaledBySqrtHidden == false)
        #expect(arch.routerScaled == false)
        #expect(arch.ffnSandwichNorms == false)
        #expect(arch.sharedExpertGated == true)
        #expect(arch.ropeNeoxSubdim == true)
        #expect(arch.linearNumKHeads == 2)
        #expect(arch.linearNumVHeads == 4)
        #expect(arch.linearKeyHeadDim == 32)
        #expect(arch.linearValueHeadDim == 32)
        #expect(arch.linearConvKernelSize == 4)
        #expect(arch.hcCount == 2)
        #expect(arch.hcLowrank == 32)
        #expect(arch.pleEmbedDim == 64)
        #expect(arch.pleNgramSize == 3)
        #expect(arch.pleSplitParts == 4)
        #expect(arch.pleHeadsPerNgram == 8)
        #expect(arch.indexerBudget == 2048)
        #expect(arch.indexerCompressRatio == 4)
        #expect(arch.indexerHeadDim == 32)
        #expect(arch.indexerKvHeads == 1)
        #expect(arch.indexerNHeads == 2)
    }

    @Test func productionQwen38ConfigParsesAndCrossChecks() throws {
        let root = temporaryRoot("qwen38-prod")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let configPath = (root as NSString).appendingPathComponent("config.json")
        try writeProductionQwen38Config(to: configPath, mutate: { _ in })

        let arch = try ArchInfo.load(configPath: configPath)
        #expect(arch.family == .qwen38)
        #expect(arch.hiddenSize == 2560)
        #expect(arch.numLayers == 48)
        #expect(arch.numHeads == 24)
        #expect(arch.vocabSize == 248_320)
        #expect(arch.numExperts == 512)
        #expect(arch.topKExperts == 10)
        #expect(arch.intermediateSize == 640)
        #expect(arch.moeIntermediateSize == 640)
        #expect(arch.attentionScale == 0.0625) // 256^-0.5
        #expect(arch.linearNumKHeads == 16)
        #expect(arch.linearNumVHeads == 48)
        #expect(arch.linearKeyHeadDim == 128)
        #expect(arch.linearValueHeadDim == 128)
        #expect(arch.linearConvKernelSize == 4)
        #expect(arch.hcCount == 4)
        #expect(arch.hcLowrank == 320)
        #expect(arch.pleEmbedDim == 2560)
        #expect(arch.pleNgramSize == 3)
        #expect(arch.pleSplitParts == 128)
        #expect(arch.pleHeadsPerNgram == 8)
        #expect(arch.indexerBudget == 2048)
        #expect(arch.indexerCompressRatio == 4)
        #expect(arch.indexerHeadDim == 128)
        #expect(arch.indexerKvHeads == 1)
        #expect(arch.indexerNHeads == 4)
        #expect(arch.fullAttentionLayerMask.count == 48)
        #expect(arch.fullAttentionLayerMask.filter { $0 == 1 }.count == 12)
        for (i, v) in arch.fullAttentionLayerMask.enumerated() {
            #expect(v == ((i + 1) % 4 == 0 ? 1 : 2))
        }
    }

    @Test func productionQwen38ConfigMismatchIsRejected() throws {
        let root = temporaryRoot("qwen38-prod-bad")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let configPath = (root as NSString).appendingPathComponent("config.json")
        try writeProductionQwen38Config(to: configPath, mutate: { tc in
            tc["num_experts"] = 256
        })

        #expect(throws: RepackError.self) {
            _ = try ArchInfo.load(configPath: configPath)
        }
    }

    @Test func qwen38SourceFingerprintIsKnown() {
        #expect(SourceFingerprint.modelID(forIndexSha256:
            "3581f8d40a330d40009d0417359f5f75b7cf79e9f8fc48ba0b1461eabd43dd5f")
            == "qwen3.8-flash-next-4bit")
        #expect(SupportedModelSource.named("qwen38")?.repoID
            == "mlx-community/Qwen3.8-Flash-Next-4bit")
    }

    @Test func qwen38ClassificationBucketsNames() {
        let family = RepackModelFamily.qwen38
        #expect(RepackPlanner.classify(
            "language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight",
            numLayers: 4, family: family)
            == .routedExpert(role: "gate", layer: 1))
        // PLE n-gram shards stream from SSD, never resident.
        #expect(RepackPlanner.classify(
            "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0.weight",
            numLayers: 4, family: family)
            == .pleShard(layer: 1, shard: 0))
        #expect(RepackPlanner.classify(
            "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.127.scales",
            numLayers: 48, family: family)
            == .pleShard(layer: 1, shard: 127))
        // Out-of-range layers do not classify as PLE.
        #expect(RepackPlanner.classify(
            "language_model.model.layers.9.ple.ple_embedding.ngram_embedding.shards.0.weight",
            numLayers: 4, family: family) == .lmResident)
        // Gated residuals, the indexer, and the mixer stay resident.
        #expect(RepackPlanner.classify(
            "language_model.model.layers.0.attn_hyper_connection.block_inject_weight.weight",
            numLayers: 4, family: family) == .lmResident)
        #expect(RepackPlanner.classify(
            "language_model.model.layers.3.self_attn.indexer.index_qk_proj.weight",
            numLayers: 4, family: family) == .lmResident)
        #expect(RepackPlanner.classify(
            "language_model.model.hyper_connection_mixer.input_mix_weight_up.weight",
            numLayers: 4, family: family) == .lmResident)
        // Shared expert and its gate stay resident.
        #expect(RepackPlanner.classify(
            "language_model.model.layers.0.mlp.shared_expert.gate_proj.weight",
            numLayers: 4, family: family) == .lmResident)
        // Vision is excluded; unknown prefixes stay unknown.
        #expect(RepackPlanner.classify(
            "vision_tower.blocks.0.norm1.weight", numLayers: 4, family: family)
            == .excludedMultimodal)
        #expect(RepackPlanner.classify(
            "model.layers.0.mlp.switch_mlp.gate_proj.weight",
            numLayers: 4, family: family) == .unknown)
    }

    @Test func qwen38PlanOrdersResidentsStreamsPleAndSlicesExperts() throws {
        let snapshotDir = temporaryRoot("qwen38-plan")
        let outputDir = temporaryRoot("qwen38-plan-out")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDir)
            try? FileManager.default.removeItem(atPath: outputDir)
        }
        let snapshot = try SyntheticSnapshot.buildQwen38(at: snapshotDir)
        let metadata = try IndexLoader.load(snapshotDir: snapshotDir)
        // Mixed 32-wide base with 64-wide gate overrides must load.
        #expect(metadata.baseGroupSize == 32)
        #expect(metadata.bitsOverrides.count == 8)
        let arch = try ArchInfo.load(
            configPath: (snapshotDir as NSString).appendingPathComponent("config.json"))
        let header = try parseHeader(path: snapshot.shardPath)

        let plan = try RepackPlanner.plan(
            meta: metadata,
            arch: arch,
            shardHeaders: [header],
            outputDir: outputDir)

        let names = plan.resident.entries.map(\.name)
        // Embedding first; mixer trio then the untied lm_head last.
        // There is no `model.norm` in this family.
        #expect(names.first == "language_model.model.embed_tokens.weight")
        #expect(names.last == "language_model.lm_head.weight")
        #expect(!names.contains("language_model.model.norm.weight"))
        let tail4 = Array(names.suffix(4))
        #expect(tail4 == [
            "language_model.model.hyper_connection_mixer.hc_norm.weight",
            "language_model.model.hyper_connection_mixer.input_mix_weight_down.weight",
            "language_model.model.hyper_connection_mixer.input_mix_weight_up.weight",
            "language_model.lm_head.weight",
        ])

        // Layer 0 is linear: DeltaNet bundle, router + shared expert, then
        // the two gated residuals. No classic norms, no PLE outside layer 1.
        let expectedLayer0 = [
            "linear_attn.in_proj_qkv.weight",
            "linear_attn.in_proj_z.weight",
            "linear_attn.in_proj_a.weight",
            "linear_attn.in_proj_b.weight",
            "linear_attn.conv1d.weight",
            "linear_attn.A_log",
            "linear_attn.dt_bias",
            "linear_attn.norm.weight",
            "linear_attn.out_proj.weight",
            "mlp.gate.weight",
            "mlp.shared_expert_gate.weight",
            "mlp.shared_expert.gate_proj.weight",
            "mlp.shared_expert.up_proj.weight",
            "mlp.shared_expert.down_proj.weight",
            "attn_hyper_connection.hc_norm.weight",
            "attn_hyper_connection.input_mix_weight_down.weight",
            "attn_hyper_connection.input_mix_weight_up.weight",
            "attn_hyper_connection.block_inject_weight.weight",
            "mlp_hyper_connection.hc_norm.weight",
            "mlp_hyper_connection.input_mix_weight_down.weight",
            "mlp_hyper_connection.input_mix_weight_up.weight",
            "mlp_hyper_connection.block_inject_weight.weight",
        ].map { "language_model.model.layers.0." + $0 }
        let layer0 = names.filter { $0.contains(".layers.0.") }
        #expect(layer0 == expectedLayer0)

        // Layer 3 is the QSA layer: projections, indexer, MoE, residuals.
        let expectedLayer3Prefix = [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "self_attn.indexer.index_qk_proj.weight",
            "self_attn.indexer.q_layernorm.weight",
            "self_attn.indexer.k_layernorm.weight",
        ].map { "language_model.model.layers.3." + $0 }
        let layer3 = names.filter { $0.contains(".layers.3.") }
        #expect(Array(layer3.prefix(9)) == expectedLayer3Prefix)

        // Layer 1 carries the resident PLE projections after the residuals.
        let layer1 = names.filter { $0.contains(".layers.1.") }
        #expect(layer1.contains(
            "language_model.model.layers.1.ple.key_proj.weight"))
        #expect(layer1.contains(
            "language_model.model.layers.1.ple.ple_embedding.layer_multipliers"))
        #expect(!layer1.contains {
            $0.contains(".shards.")
        })

        // PLE integer metadata (I64) stays resident, verbatim, unquantized.
        for suffix in ["ple.ple_embedding.layer_multipliers",
                       "ple.ple_embedding.ngram_heads_offsets",
                       "ple.ple_embedding.ngram_heads_vocab_sizes"] {
            let entry = try #require(plan.resident.entries.first {
                $0.name == "language_model.model.layers.1." + suffix
            })
            #expect(entry.dtype == 4)
            #expect(entry.quantSpec == nil)
            #expect(entry.sourceScales == nil)
            #expect(entry.sourceBiases == nil)
        }
        // The four PLE shards stream to packed_ple, never resident.
        #expect(plan.pleShards.count == 4)
        for (i, shard) in plan.pleShards.enumerated() {
            #expect(shard.layerIndex == 1)
            #expect(shard.shardIndex == i)
            #expect((shard.path as NSString).lastPathComponent
                == String(format: "shard_%03d.bin", i))
            #expect(shard.bits == 4)
            #expect(shard.groupSize == 32)
            #expect(shard.fileSize == shard.weightBytes + shard.scalesBytes + shard.biasesBytes)
        }
        #expect(!names.contains { $0.contains(".shards.") })

        // Router gate keeps its 64-wide 8-bit override; attention is 32-wide.
        let router = try #require(plan.resident.entries.first {
            $0.name == "language_model.model.layers.0.mlp.gate.weight"
        })
        #expect(router.quantSpec?.bits == 8)
        #expect(router.quantSpec?.groupSize == 64)
        let qkv = try #require(plan.resident.entries.first {
            $0.name == "language_model.model.layers.0.linear_attn.in_proj_qkv.weight"
        })
        #expect(qkv.quantSpec?.bits == 4)
        #expect(qkv.quantSpec?.groupSize == 32)

        // Every layer slices two experts into the fixed 9-slice blob.
        #expect(plan.layers.count == 4)
        for lp in plan.layers {
            #expect(lp.expertsPerLayer == 2)
            #expect(lp.subTensors.count == 9)
            #expect(lp.expertStride % 16_384 == 0)
        }

        // Vision-tower tensors are dropped, not planned.
        #expect(plan.excludedMultimodalTensorNames.sorted() == [
            "vision_tower.blocks.0.norm1.weight",
            "vision_tower.patch_embed.proj.weight",
        ])
    }

    @Test func qwen38ManifestCarriesFamilyGroupsAndPle() throws {
        let snapshotDir = temporaryRoot("qwen38-manifest")
        let outputDir = temporaryRoot("qwen38-manifest-out")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDir)
            try? FileManager.default.removeItem(atPath: outputDir)
        }
        let snapshot = try SyntheticSnapshot.buildQwen38(at: snapshotDir)
        let metadata = try IndexLoader.load(snapshotDir: snapshotDir)
        let arch = try ArchInfo.load(
            configPath: (snapshotDir as NSString).appendingPathComponent("config.json"))
        let header = try parseHeader(path: snapshot.shardPath)
        let plan = try RepackPlanner.plan(
            meta: metadata, arch: arch, shardHeaders: [header], outputDir: outputDir)

        let data = try GTurboJSON.encodeManifest(
            plan: plan,
            modelID: "qwen3.8-flash-next-4bit",
            sourceSnapshotHash: "sha256:0",
            files: [],
            expertsPerLayer: 2,
            numLayers: arch.numLayers,
            expertStride: 16_384,
            bitWidths: GTurboJSON.QuantBitWidths(
                embedding: 4, attention: 4, router: 8,
                sharedExpert: 4, routedExpert: 4,
                embeddingGroup: 32, attentionGroup: 32, routerGroup: 64,
                sharedExpertGroup: 32, routedExpertGroup: 32))
        let obj = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let archDict = obj["arch"] as! [String: Any]
        #expect(archDict["family"] as? String == "qwen38")
        #expect(archDict["hcCount"] as? Int == 2)
        #expect(archDict["pleSplitParts"] as? Int == 4)
        #expect(archDict["indexerBudget"] as? Int == 2048)
        #expect(obj["pleShards"] as? Int == 4)
        let quant = obj["quant"] as! [String: [String: Any]]
        #expect(quant["router"]?["weightBits"] as? Int == 8)
        #expect(quant["router"]?["groupSize"] as? Int == 64)
        #expect(quant["attention"]?["weightBits"] as? Int == 4)
        #expect(quant["attention"]?["groupSize"] as? Int == 32)
        #expect(quant["routedExpert"]?["groupSize"] as? Int == 32)

        let pleLayout = try GTurboJSON.encodePleLayout(plan: plan)
        let pleObj = try JSONSerialization.jsonObject(with: pleLayout) as! [String: Any]
        #expect(pleObj["shardCount"] as? Int == 4)
        #expect((pleObj["shards"] as? [[String: Any]])?.count == 4)
    }

    // MARK: - Helpers

    private func writeProductionQwen38Config(
        to path: String,
        mutate: (inout [String: Any]) -> Void) throws {
        var layerTypes: [String] = []
        for i in 0..<48 {
            layerTypes.append((i + 1) % 4 == 0 ? "full_attention" : "linear_attention")
        }
        var tc: [String: Any] = [
            "hidden_size": 2560,
            "moe_intermediate_size": 640,
            "shared_expert_intermediate_size": 640,
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "vocab_size": 248_320,
            "num_hidden_layers": 48,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "layer_types": layerTypes,
            "rope_parameters": [
                "rope_theta": 10_000_000.0,
                "rope_type": "default",
                "partial_rotary_factor": 0.25
            ],
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "tie_word_embeddings": false,
            "hidden_act": "silu",
            "hc_count": 4,
            "hc_lowrank": 320,
            "ple_embed_dim": 2560,
            "ngram_size": 3,
            "split_ngram_parts": 128,
            "heads_per_ngram": 8,
            "indexer_budget": 2048,
            "indexer_compress_ratio": 4,
            "indexer_head_dim": 128,
            "indexer_kv_heads": 1,
            "indexer_n_heads": 4
        ]
        mutate(&tc)
        var quant: [String: Any] = ["bits": 4, "group_size": 32, "mode": "affine"]
        for li in 0..<48 {
            let prefix = "language_model.model.layers.\(li)"
            for k in ["mlp.gate", "mlp.shared_expert_gate"] {
                quant[prefix + "." + k] = ["bits": 8, "group_size": 64]
            }
        }
        let config: [String: Any] = [
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "model_type": "qwen4_exp",
            "quantization": quant,
            "text_config": tc
        ]
        let data = try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
        try data.write(to: URL(fileURLWithPath: path))
    }

    private func temporaryRoot(_ tag: String) -> String {
        let path = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("turbofieldfare-qwen38-plan-\(tag)-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            atPath: path,
            withIntermediateDirectories: true)
        return path
    }

    private func parseHeader(path: String) throws -> Safetensors.Header {
        let fd = try Posix.openRead(path)
        defer { close(fd) }
        var headerSize: UInt64 = 0
        try withUnsafeMutableBytes(of: &headerSize) {
            try Posix.preadAll(
                fd: fd,
                path: path,
                buf: $0.baseAddress!,
                count: 8,
                offset: 0)
        }
        headerSize = UInt64(littleEndian: headerSize)
        var headerData = Data(count: Int(headerSize))
        try headerData.withUnsafeMutableBytes {
            try Posix.preadAll(
                fd: fd,
                path: path,
                buf: $0.baseAddress!,
                count: $0.count,
                offset: 8)
        }
        return try Safetensors.parseHeaderBytes(
            path: path,
            fileSize: try Posix.fileSize(fd: fd, path: path),
            headerBytes: headerData)
    }
}
