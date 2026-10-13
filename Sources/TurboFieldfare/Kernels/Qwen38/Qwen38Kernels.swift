import Foundation
import Metal

/// Swift wrappers for the Qwen 3.8 small kernels (`qwen38.metal`): (1+w)
/// RMSNorms, hyper-connection elementwise, QSA indexer helpers, and PLE
/// helpers. Shared INT4 GEMVs (hyper down/up/inject, indexer qk, PLE
/// key/value, QSA q/k/v/o) go through `DequantInt4GEMV` with group 32.
final class Qwen38Kernels {
    private let normGroupedPSO: MTLComputePipelineState
    private let hcRepeatPSO: MTLComputePipelineState
    private let hcMixMeanPSO: MTLComputePipelineState
    private let injectAddPSO: MTLComputePipelineState
    private let siluDivPSO: MTLComputePipelineState
    private let sigmoidPSO: MTLComputePipelineState
    private let sigmoid2DivPSO: MTLComputePipelineState
    private let blockScorePSO: MTLComputePipelineState
    private let gatherRowsPSO: MTLComputePipelineState
    private let pleGateDotPSO: MTLComputePipelineState
    private let pleScaleRowsPSO: MTLComputePipelineState
    private let pleShortConvPSO: MTLComputePipelineState

    init(context: MetalContext) throws {
        normGroupedPSO = try context.pipeline("qwen38_rmsnorm_1pw_grouped")
        hcRepeatPSO = try context.pipeline("qwen38_hc_repeat")
        hcMixMeanPSO = try context.pipeline("qwen38_hc_mix_mean")
        injectAddPSO = try context.pipeline("qwen38_inject_add")
        siluDivPSO = try context.pipeline("qwen38_silu_div")
        sigmoidPSO = try context.pipeline("qwen38_sigmoid")
        sigmoid2DivPSO = try context.pipeline("qwen38_sigmoid2_div")
        blockScorePSO = try context.pipeline("qsa_block_score")
        gatherRowsPSO = try context.pipeline("qsa_gather_rows")
        pleGateDotPSO = try context.pipeline("ple_gate_dot")
        pleScaleRowsPSO = try context.pipeline("ple_scale_rows")
        pleShortConvPSO = try context.pipeline("ple_short_conv_step")
    }

    // MARK: - Norms

    /// (1+w) RMSNorm over `groups` rows with a full `[groups, dim]` weight
    /// (hc_norm, ple norm_key/query/conv, mixer norm).
    func encodeNormGrouped(commandBuffer: MTLCommandBuffer,
                           x: MTLBuffer, xOffset: Int = 0,
                           weight: MTLBuffer, weightOffset: Int = 0,
                           out: MTLBuffer, outOffset: Int = 0,
                           dim: UInt32, groups: Int,
                           eps: Float) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(normGroupedPSO)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(weight, offset: weightOffset, index: 1)
        enc.setBuffer(out, offset: outOffset, index: 2)
        var d = dim
        var epsVar = eps
        enc.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&epsVar, length: MemoryLayout<Float>.size, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    // MARK: - Hyper-connection elementwise

    func encodeRepeat(commandBuffer: MTLCommandBuffer,
                      x: MTLBuffer, xOffset: Int = 0,
                      y: MTLBuffer, yOffset: Int = 0,
                      d: Int, h: Int) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(hcRepeatPSO)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(y, offset: yOffset, index: 1)
        var dd = UInt32(d), hh = UInt32(h)
        enc.setBytes(&dd, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&hh, length: MemoryLayout<UInt32>.size, index: 3)
        dispatch1D(enc, pso: hcRepeatPSO, threads: d * h)
        enc.endEncoding()
    }

    func encodeMixMean(commandBuffer: MTLCommandBuffer,
                       mix: MTLBuffer, mixOffset: Int = 0,
                       normed: MTLBuffer, normedOffset: Int = 0,
                       mixed: MTLBuffer, mixedOffset: Int = 0,
                       d: Int, h: Int) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(hcMixMeanPSO)
        enc.setBuffer(mix, offset: mixOffset, index: 0)
        enc.setBuffer(normed, offset: normedOffset, index: 1)
        enc.setBuffer(mixed, offset: mixedOffset, index: 2)
        var dd = UInt32(d), hh = UInt32(h)
        enc.setBytes(&dd, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&hh, length: MemoryLayout<UInt32>.size, index: 4)
        dispatch1D(enc, pso: hcMixMeanPSO, threads: d)
        enc.endEncoding()
    }

    func encodeInjectAdd(commandBuffer: MTLCommandBuffer,
                         hyper: MTLBuffer, hyperOffset: Int = 0,
                         sub: MTLBuffer, subOffset: Int = 0,
                         inj: MTLBuffer, injOffset: Int = 0,
                         d: Int, h: Int) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(injectAddPSO)
        enc.setBuffer(hyper, offset: hyperOffset, index: 0)
        enc.setBuffer(sub, offset: subOffset, index: 1)
        enc.setBuffer(inj, offset: injOffset, index: 2)
        var dd = UInt32(d), hh = UInt32(h)
        enc.setBytes(&dd, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&hh, length: MemoryLayout<UInt32>.size, index: 4)
        dispatch1D(enc, pso: injectAddPSO, threads: d * h)
        enc.endEncoding()
    }

    func encodeSiluDiv(commandBuffer: MTLCommandBuffer,
                       x: MTLBuffer, xOffset: Int = 0,
                       y: MTLBuffer, yOffset: Int = 0,
                       div: Float, count: Int) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(siluDivPSO)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(y, offset: yOffset, index: 1)
        var divVar = div
        var n = UInt32(count)
        enc.setBytes(&divVar, length: MemoryLayout<Float>.size, index: 2)
        enc.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 3)
        dispatch1D(enc, pso: siluDivPSO, threads: count)
        enc.endEncoding()
    }

    func encodeSigmoid(commandBuffer: MTLCommandBuffer,
                       x: MTLBuffer, xOffset: Int = 0,
                       y: MTLBuffer, yOffset: Int = 0,
                       count: Int) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(sigmoidPSO)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(y, offset: yOffset, index: 1)
        var n = UInt32(count)
        enc.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 2)
        dispatch1D(enc, pso: sigmoidPSO, threads: count)
        enc.endEncoding()
    }

    func encodeSigmoid2Div(commandBuffer: MTLCommandBuffer,
                           x: MTLBuffer, xOffset: Int = 0,
                           y: MTLBuffer, yOffset: Int = 0,
                           div: Float, count: Int) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(sigmoid2DivPSO)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(y, offset: yOffset, index: 1)
        var divVar = div
        var n = UInt32(count)
        enc.setBytes(&divVar, length: MemoryLayout<Float>.size, index: 2)
        enc.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 3)
        dispatch1D(enc, pso: sigmoid2DivPSO, threads: count)
        enc.endEncoding()
    }

    // MARK: - QSA

    /// Score `blockCount` pooled keys against the index query heads.
    func encodeBlockScore(commandBuffer: MTLCommandBuffer,
                          q: MTLBuffer, qOffset: Int = 0,
                          k: MTLBuffer, kOffset: Int = 0,
                          scores: MTLBuffer, scoresOffset: Int = 0,
                          headDim: Int, nHeads: Int, blockCount: Int) {
        guard blockCount > 0 else { return }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(blockScorePSO)
        enc.setBuffer(q, offset: qOffset, index: 0)
        enc.setBuffer(k, offset: kOffset, index: 1)
        enc.setBuffer(scores, offset: scoresOffset, index: 2)
        var hd = UInt32(headDim), nh = UInt32(nHeads)
        enc.setBytes(&hd, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&nh, length: MemoryLayout<UInt32>.size, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: blockCount, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding()
    }

    func encodeGatherRows(commandBuffer: MTLCommandBuffer,
                          src: MTLBuffer, srcOffset: Int = 0,
                          sel: MTLBuffer, selOffset: Int = 0,
                          dst: MTLBuffer, dstOffset: Int = 0,
                          rowElems: Int, rows: Int) {
        guard rows > 0 else { return }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(gatherRowsPSO)
        enc.setBuffer(src, offset: srcOffset, index: 0)
        enc.setBuffer(sel, offset: selOffset, index: 1)
        enc.setBuffer(dst, offset: dstOffset, index: 2)
        var re = UInt32(rowElems), rr = UInt32(rows)
        enc.setBytes(&re, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&rr, length: MemoryLayout<UInt32>.size, index: 4)
        dispatch1D(enc, pso: gatherRowsPSO, threads: rows * rowElems)
        enc.endEncoding()
    }

    // MARK: - PLE

    func encodeGateDot(commandBuffer: MTLCommandBuffer,
                       key: MTLBuffer, keyOffset: Int = 0,
                       query: MTLBuffer, queryOffset: Int = 0,
                       dots: MTLBuffer, dotsOffset: Int = 0,
                       d: Int, h: Int) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pleGateDotPSO)
        enc.setBuffer(key, offset: keyOffset, index: 0)
        enc.setBuffer(query, offset: queryOffset, index: 1)
        enc.setBuffer(dots, offset: dotsOffset, index: 2)
        var dd = UInt32(d)
        enc.setBytes(&dd, length: MemoryLayout<UInt32>.size, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: h, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    func encodeScaleRows(commandBuffer: MTLCommandBuffer,
                         value: MTLBuffer, valueOffset: Int = 0,
                         gate: MTLBuffer, gateOffset: Int = 0,
                         out: MTLBuffer, outOffset: Int = 0,
                         d: Int, h: Int) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pleScaleRowsPSO)
        enc.setBuffer(value, offset: valueOffset, index: 0)
        enc.setBuffer(gate, offset: gateOffset, index: 1)
        enc.setBuffer(out, offset: outOffset, index: 2)
        var dd = UInt32(d), hh = UInt32(h)
        enc.setBytes(&dd, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&hh, length: MemoryLayout<UInt32>.size, index: 4)
        dispatch1D(enc, pso: pleScaleRowsPSO, threads: d * h)
        enc.endEncoding()
    }

    func encodeShortConvStep(commandBuffer: MTLCommandBuffer,
                             state: MTLBuffer, stateOffset: Int = 0,
                             input: MTLBuffer, inputOffset: Int = 0,
                             weight: MTLBuffer, weightOffset: Int = 0,
                             out: MTLBuffer, outOffset: Int = 0,
                             channels: Int, kernel: Int,
                             dilation: Int, stateLen: Int) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pleShortConvPSO)
        enc.setBuffer(state, offset: stateOffset, index: 0)
        enc.setBuffer(input, offset: inputOffset, index: 1)
        enc.setBuffer(weight, offset: weightOffset, index: 2)
        enc.setBuffer(out, offset: outOffset, index: 3)
        var cc = UInt32(channels), kk = UInt32(kernel)
        var dl = UInt32(dilation), sl = UInt32(stateLen)
        enc.setBytes(&cc, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&kk, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&dl, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&sl, length: MemoryLayout<UInt32>.size, index: 7)
        dispatch1D(enc, pso: pleShortConvPSO, threads: channels)
        enc.endEncoding()
    }

    // MARK: - Private

    private func dispatch1D(_ enc: MTLComputeCommandEncoder,
                            pso: MTLComputePipelineState,
                            threads: Int) {
        guard threads > 0 else { return }
        let width = min(pso.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }
}
