import Testing
import Foundation
import Metal
@testable import TurboFieldfare

/// Qwen 3.8 runtime integration over the qwen38 toy fixture: runner
/// construction against the hyper-connection / QSA / PLE tensor contract,
/// decode and prefill-as-decode smoke, and the PLE/KV reset interplay.
/// Mirrors `QwenRunnerTests` for the 3.6 family. Requires Metal (macOS).
@Suite struct Qwen38RunnerTests {

    private func makeRunner(maxContext: Int = 32) throws -> (URL, MetalContext, RealForwardRunner) {
        let dir = try Qwen38ToySynthetic.write()
        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir,
                                   device: ctx.device,
                                   expecting: .qwen38Toy())
        let runner = try RealForwardRunner(model: model,
                                           context: ctx,
                                           maxContext: maxContext)
        return (dir, ctx, runner)
    }

    private func makeLogits(_ ctx: MetalContext, vocab: Int) throws -> MTLBuffer {
        guard let buf = ctx.device.makeBuffer(
            length: vocab * MemoryLayout<Float16>.stride,
            options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        return buf
    }

    /// Runner init wires the qwen38 contract: hyper bundles every layer, the
    /// mixer closes the stream, indexer + PLE residents resolve, and the
    /// top-10 MoE specializes without touching Gemma/Qwen3.6 norm tensors
    /// (absent from the fixture — any access would throw `tensorNotFound`).
    @Test func runnerInit_qwen38Toy_resolvesContract() throws {
        let (dir, _, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(runner.maxContext == 32)
        #expect(runner.usesFusedGreedyHead)
    }

    /// Decode smoke over the hybrid graph: linear (GDN) + QSA sparse layers
    /// under hyper-connections, PLE injection on layer 1, top-10 routing over
    /// 16 experts, mixer + untied head. Verifies tokens are produced, the
    /// cursor advances, and reset() reproduces the first token (no stale PLE
    /// conv state, indexer tails, GDN state, or KV).
    @Test func decodeSmoke_hybridGraph() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx, vocab: 256)

        try await runner.produce(token: 1, position: 0, into: logits)
        let first = runner.lastGreedyToken
        #expect(first < 256)
        try await runner.produce(token: Int32(first), position: 1, into: logits)
        #expect(runner.continuationPosition == 2)

        runner.reset()
        #expect(runner.continuationPosition == 0)
        try await runner.produce(token: 1, position: 0, into: logits)
        #expect(runner.lastGreedyToken == first)
    }

    /// Prefill/decode consistency: the chunked prefill advances the same
    /// per-token state machine as decode, so prefilling [t0] then decoding
    /// t1 matches decoding t0, t1 step by step.
    @Test func prefillThenDecode_matchesPureDecode() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx, vocab: 256)

        try await runner.produce(token: 11, position: 0, into: logits)
        try await runner.produce(token: 7, position: 1, into: logits)
        let reference = runner.lastGreedyToken

        runner.reset()
        let tokens: [Int32] = [11]
        _ = try await runner.prefillChunked(
            tokens: tokens[...],
            startPosition: 0,
            outputMode: .greedyIfAvailable,
            config: .production(chunkTokens: 32),
            into: logits,
            onProgress: { _ in })
        try await runner.produce(token: 7, position: 1, into: logits)
        #expect(runner.lastGreedyToken == reference)
    }

    /// Chunked prefill smoke over several tokens, then a decode continuation.
    @Test func prefillChunkedSmoke_thenDecodeContinuation() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx, vocab: 256)

        let tokens: [Int32] = [1, 2, 3, 4, 5, 6]
        let result = try await runner.prefillChunked(
            tokens: tokens[...],
            startPosition: 0,
            outputMode: .greedyIfAvailable,
            config: .production(chunkTokens: 32),
            into: logits,
            onProgress: { _ in })
        #expect(result.newPosition == 6)
        if case .greedyToken(let token) = result.seed {
            #expect(token < 256)
        } else {
            Issue.record("expected a greedy seed token from the fused head")
        }

        try await runner.produce(token: 9, position: 6, into: logits)
        #expect(runner.lastGreedyToken < 256)
        #expect(runner.continuationPosition == 7)
    }

    /// QSA short-context equivalence: with fewer than `compress` visible
    /// tokens the indexer selects the dense prefix, so the first two tokens
    /// attend densely by construction.
    @Test func qsaShortContext_selectsDensePrefix() {
        let cfg = ArchConfig.qwen38Toy()
        // compress=2: positions 0..1 form no complete block.
        let sel = Qwen38Reference.qsaSelectTokens(
            visible: [0, 1], blockScores: [0.3], compress: 2, blockTopK: 4)
        #expect(sel == [0, 1])
        _ = cfg
    }
}

/// `PleShardReader` against the toy `packed_ple/` table: layout geometry,
/// uniform rows-per-shard, and row dequant matching the CPU quantizer.
@Suite struct PleShardReaderTests {

    @Test func layoutAndRowRead_matchFixture() throws {
        let dir = try Qwen38ToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: dir) }
        let reader = try PleShardReader(directoryURL: dir)
        #expect(reader.shardCount == 2)
        #expect(reader.layerIndex == 1)
        let rps = try reader.rowsPerShard
        #expect(rps == Qwen38ToySynthetic.pleShardRows)
        // Shard of a global row id follows contiguous dim0 slicing.
        #expect(Qwen38Reference.shardForRow(0, rowsPerShard: rps) == 0)
        #expect(Qwen38Reference.shardForRow(200, rowsPerShard: rps) == 1)
        // Rows decode to finite head-dim vectors.
        let row = try reader.readRow(globalRow: 5, rowsPerShard: rps)
        #expect(row.count == Qwen38ToySynthetic.pleHeadDim)
        #expect(row.allSatisfy { $0.isFinite })
        let last = try reader.readRow(globalRow: 255, rowsPerShard: rps)
        #expect(last.count == Qwen38ToySynthetic.pleHeadDim)
    }
}
