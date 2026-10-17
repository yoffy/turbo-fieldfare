import Testing
import Foundation
@testable import TurboFieldfare

/// CPU reference math for Qwen 3.8 (`Qwen38Reference`, ported from
/// transformers `modular_qwen4_exp.py`). Expected vectors below were computed
/// with an independent Python mirror of the same equations (see task notes),
/// so they guard the Swift port rather than echoing it.
@Suite struct Qwen38ReferenceTests {

    @Test func rmsNormOnePlus() {
        let y = Qwen38Reference.rmsNorm([1, 2, 3, 4],
                                        weight: [0, 0, 0, 0], eps: 1e-6)
        let expect: [Float] = [0.36514835, 0.73029669, 1.09544504, 1.46059339]
        for (a, b) in zip(y, expect) {
            #expect(abs(a - b) < 1e-5)
        }
        // Nonzero weight scales by (1 + w), not w.
        let y2 = Qwen38Reference.rmsNorm([1, 2, 3, 4],
                                         weight: [1, 1, 1, 1], eps: 1e-6)
        for (a, b) in zip(y2, expect) {
            #expect(abs(a - 2 * b) < 1e-5)
        }
        // Grouped form equals per-group norms.
        let g = Qwen38Reference.rmsNorm([1, 2, 3, 4],
                                        weight: [0, 0, 0, 0],
                                        groupSize: 2, eps: 1e-6)
        #expect(abs(g[0] - 0.63245541) < 1e-5)
        #expect(abs(g[1] - 1.26491081) < 1e-5)
        #expect(abs(g[2] - 0.84852810) < 1e-5)
        #expect(abs(g[3] - 1.13137080) < 1e-5)
    }

    @Test func gatedResidualMatchesReference() {
        // hc=2, hidden=2, lowrank=2. Identity-ish down/up, no inject.
        let r = Qwen38Reference.gatedResidual(
            hyperInput: [1, 2, 3, 4], hcCount: 2, hidden: 2, lowrank: 2,
            hcNormWeight: [0, 0, 0, 0],
            downWeight: [1, 0, 0, 0,
                         0, 0, 0, 1],
            upWeight: [1, 0,
                       0, 0,
                       0, 0,
                       0, 1],
            blockInjectWeight: nil, eps: 1e-6)
        #expect(r.injection == nil)
        #expect(abs(r.mixed[0] - 0.38466581) < 1e-5)
        #expect(abs(r.mixed[1] - 0.64954546) < 1e-5)
        // Inject add: hyper + sub * inj per branch.
        let added = Qwen38Reference.injectAdd(
            hyperInput: [1, 2, 3, 4], subOutput: [10, 20],
            injection: [1, 2], hcCount: 2, hidden: 2)
        #expect(added == [11, 22, 23, 44])
    }

    @Test func routerTopKMatchesSoftmaxRenorm() {
        let r = Qwen38Reference.routerTopK(logits: [1, 2, 3, 0.5], topK: 2)
        #expect(r.indices == [2, 1])
        #expect(abs(r.weights[0] - 0.73105858) < 1e-5)
        #expect(abs(r.weights[1] - 0.26894142) < 1e-5)
        // Ties break to the smaller expert index.
        let t = Qwen38Reference.routerTopK(logits: [1, 1, 0], topK: 2)
        #expect(t.indices == [0, 1])
        #expect(abs(t.weights[0] - 0.5) < 1e-6)
        // Per-expert scale multiplies after renormalization.
        let s = Qwen38Reference.routerTopK(logits: [1, 2, 3, 0.5], topK: 2,
                                           perExpertScale: [1, 1, 2, 1])
        #expect(abs(s.weights[0] - 2 * 0.73105858) < 1e-5)
    }

    @Test func qsaScoresAndSelection() {
        let scores = Qwen38Reference.qsaBlockScores(
            queryHeads: [[1, 0], [0, 1]],
            blockKeys: [[1, 0], [0, 2]], headDim: 2)
        #expect(abs(scores[0] - 0.70710678) < 1e-5)
        #expect(abs(scores[1] - 1.41421356) < 1e-5)
        let sel = Qwen38Reference.qsaSelectTokens(
            visible: [0, 1, 2, 3, 4, 5],
            blockScores: [0.1, 0.9, 0.5], compress: 2, blockTopK: 1)
        #expect(sel == [2, 3])
        // Incomplete tail always passes through.
        let selTail = Qwen38Reference.qsaSelectTokens(
            visible: [0, 1, 2, 3, 4, 5, 6],
            blockScores: [0.1, 0.9, 0.5], compress: 2, blockTopK: 1)
        #expect(selTail == [2, 3, 6])
        // Short context: dense passthrough.
        let dense = Qwen38Reference.qsaSelectTokens(
            visible: [0, 1], blockScores: [], compress: 4, blockTopK: 512)
        #expect(dense == [0, 1])
        #expect(Qwen38Reference.qsaAllowMask(selected: [2, 3], kvLen: 5)
                == [false, false, true, true, false])
    }

    @Test func qsaSelectionIsTheDensePrefixInsideTheBlockBudget() {
        // The decode fast path skips block scoring, top-k and the gather while
        // `position + 1 <= blockTopK * compress`; inside that window the
        // reference selection must be exactly the causal prefix, otherwise the
        // shortcut would change which tokens attention sees.
        let compress = 4
        let blockTopK = 512
        let limit = blockTopK * compress          // 2048 visible tokens
        for visibleCount in [1, 2, 5, compress, compress + 1, limit - 1, limit] {
            let visible = Array(0..<visibleCount)
            var scores = [Float](repeating: 0, count: visibleCount / compress)
            for i in scores.indices { scores[i] = Float((i % 3) - 1) }
            let selected = Qwen38Reference.qsaSelectTokens(
                visible: visible, blockScores: scores,
                compress: compress, blockTopK: blockTopK)
            #expect(selected == visible)
        }
        // The last dense window is `limit` visible tokens; one full block past
        // it there are `blockTopK + 1` complete blocks, so exactly one block
        // (the lowest score) drops out and the selection stops being the whole
        // prefix. That is the first position the sparse path must handle.
        let overVisible = Array(0...(limit + compress - 1))   // 2052 tokens
        var overScores = [Float](repeating: 0, count: overVisible.count / compress)
        for i in overScores.indices { overScores[i] = Float(i) }
        let overSelected = Qwen38Reference.qsaSelectTokens(
            visible: overVisible, blockScores: overScores,
            compress: compress, blockTopK: blockTopK)
        #expect(overSelected.count == blockTopK * compress)
        #expect(overSelected.first == compress)
        #expect(overSelected.last == limit + compress - 1)
        #expect(overSelected != overVisible)
        // Ties are broken by smaller block index, so with `blockTopK + 1`
        // complete blocks the *highest* index loses the last slot and the
        // output stays index-ascending (reproducible selection).
        var tieScores = [Float](repeating: 1, count: overVisible.count / compress)
        tieScores[7] = 2
        let tieSelected = Qwen38Reference.qsaSelectTokens(
            visible: overVisible, blockScores: tieScores,
            compress: compress, blockTopK: blockTopK)
        #expect(tieSelected.count == blockTopK * compress)
        #expect(tieSelected.first == 0)
        #expect(tieSelected.contains(7 * compress))
        #expect(!tieSelected.contains(blockTopK * compress))
        #expect(tieSelected.last == blockTopK * compress - 1)
    }

    @Test func splitmixAndMultipliersMatchUpstream() {
        // `_build_layer_multipliers(248320, 3, ple_layer_index, seed=1234)`.
        let m0 = Qwen38Reference.buildLayerMultipliers(
            unigramVocab: 248320, ngramSize: 3, pleLayerIndex: 0, seed: 1234)
        #expect(m0 == [23703573157769, 20109073645365, 8052911324071])
        let m1 = Qwen38Reference.buildLayerMultipliers(
            unigramVocab: 248320, ngramSize: 3, pleLayerIndex: 1, seed: 1234)
        #expect(m1 == [3352040966061, 19752442147475, 10742103100073])
        #expect(Qwen38Reference.isPrime(20000003))
        #expect(!Qwen38Reference.isPrime(20000000))
        #expect(Qwen38Reference.nthPrimeAfter(20000000 - 1, count: 1) == 20000003)
    }

    @Test func ngramGeometryIsConsistent() {
        let g = Qwen38Reference.ngramGeometry(
            ngramHeads: 16, pleLayerIndex: 0, vocabBase: 20_000_000,
            divisibleBy: 128, splitParts: 128)
        #expect(g.headVocabSizes.count == 16)
        #expect(g.headOffsets[0] == 0)
        #expect(g.totalVocab == g.headVocabSizes.reduce(0, +))
        #expect(g.paddedVocab % 128 == 0)
        #expect(g.paddedVocab >= g.totalVocab)
        #expect(g.rowsPerShard * 128 == g.paddedVocab)
        // Offsets tile the table contiguously.
        for h in 1..<16 {
            #expect(g.headOffsets[h] == g.headOffsets[h - 1] + g.headVocabSizes[h - 1])
        }
    }

    @Test func shiftRightIgnoreEosMatchesReference() {
        #expect(Qwen38Reference.shiftRightIgnoreEos(
            tokens: [5, 6, 7, 248044, 8, 9], shift: 1, eosId: 248044)
            == [248044, 5, 6, 7, 248044, 8])
        #expect(Qwen38Reference.shiftRightIgnoreEos(
            tokens: [5, 6, 7, 248044, 8, 9], shift: 2, eosId: 248044)
            == [248044, 248044, 5, 6, 248044, 248044])
    }

    @Test func ngramRowIdsMatchReference() {
        let rows = Qwen38Reference.ngramRowIds(
            history: [7, 8, 9], seqLen: 1, ngramSize: 3, headsPerNgram: 1,
            multipliers: [11, 101, 1009],
            headVocabSizes: [100, 100], headOffsets: [0, 100], eosId: 0)
        #expect(rows.count == 1 && rows[0].count == 2)
        #expect(rows[0][0] == 43)
        #expect(rows[0][1] == 164)
        #expect(Qwen38Reference.shardForRow(43, rowsPerShard: 128) == 0)
        #expect(Qwen38Reference.shardForRow(164, rowsPerShard: 128) == 1)
    }

    @Test func ropeNeoxCpuMatchesKernelConvention() {
        // pos=1, theta=1e4, rotaryDim=4: pair 0 rotates by angle 1.
        let y = Qwen38Reference.applyRopeNeoxCPU(
            [1, 0, 0, 0], position: 1, headDim: 4, rotaryDim: 4, theta: 10000)
        #expect(abs(y[0] - 0.54030231) < 1e-6)
        #expect(abs(y[2] - 0.84147098) < 1e-6)
        #expect(abs(y[1]) < 1e-7 && abs(y[3]) < 1e-7)
        // Position 0 is the identity; tail past rotaryDim passes through.
        let z = Qwen38Reference.applyRopeNeoxCPU(
            [1, 2, 3, 4, 5, 6], position: 0, headDim: 6, rotaryDim: 4,
            theta: 10000)
        #expect(z == [1, 2, 3, 4, 5, 6])
    }

    @Test func pleGateMatchesReference() {
        let (gated, gate) = Qwen38Reference.pleGate(
            keyStreams: [[1, 0]], queryStreams: [[1, 0]],
            value: [5, 6], hidden: 2)
        #expect(abs(gate[0] - 0.84089642) < 1e-5)
        #expect(abs(gated[0][0] - 3.49326989) < 1e-4)
        #expect(abs(gated[0][1] - 4.19192387) < 1e-4)
    }

    @Test func quantGroupSizes() {
        #expect(Quantization.isSupportedGroupSize(32))
        #expect(Quantization.isSupportedGroupSize(64))
        #expect(!Quantization.isSupportedGroupSize(16))
        #expect(Quantization.groupsPerRow(n: 128, groupSize: 32) == 4)
    }

    @Test func quantGroup32Roundtrip() {
        let row = (0..<128).map { Float($0) * 0.01 - 0.5 }
        for g in [32, 64] {
            let q = Quantization.quantizeInt4Affine(row, groupSize: g)
            #expect(q.scales.count == 128 / g)
            let back = Quantization.dequantizeInt4Affine(q, n: 128, groupSize: g)
            var maxErr: Float = 0
            for (a, b) in zip(row, back) { maxErr = max(maxErr, abs(a - b)) }
            // One LSB of the per-group range plus BF16 rounding.
            #expect(maxErr < 0.09)
        }
        let q8 = Quantization.quantizeInt8Affine(row, groupSize: 32)
        #expect(q8.scales.count == 4)
        let back8 = Quantization.dequantizeInt8Affine(q8, n: 128, groupSize: 32)
        var maxErr8: Float = 0
        for (a, b) in zip(row, back8) { maxErr8 = max(maxErr8, abs(a - b)) }
        #expect(maxErr8 < 0.01)
    }
}
