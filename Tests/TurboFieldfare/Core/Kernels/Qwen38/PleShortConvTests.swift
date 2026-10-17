import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// Qwen 3.8 PLE dilated depthwise short conv (`Qwen4ExpTextPLELayer._short_conv`,
/// dilation = ngram_size). `F.conv1d` pads `(K-1)*dilation` zeros on the left
/// and cross-correlates, so tap `t` weights the lag `(K-1-t)*dilation` and the
/// current token carries `w[K-1]`. One decode step shifts the state window left
/// and appends the current normalized row.
@Suite struct PleShortConvTests {

    private static let channels = 8
    private static let kernel = 4
    private static let dilation = 3
    private static let stateLen = (kernel - 1) * dilation   // 9

    private static func silu(_ x: Float) -> Float { x / (1 + expf(-x)) }

    /// Tap `t` reads the `history` row at lag `(K-1-t)*dilation`; rows before
    /// the first token are the left zero padding.
    private static func crossCorrelate(history: [[Float]], at step: Int,
                                       channel c: Int,
                                       weights: [[Float]]) -> Float {
        var acc: Float = 0
        for t in 0..<kernel {
            let src = step - (kernel - 1 - t) * dilation
            if src >= 0 { acc += weights[c][t] * history[src][c] }
        }
        return acc
    }

    @Test func decodeStepsMatchCrossCorrelation() throws {
        var rng = SeedTree(0x51C0).key("ple-short-conv")
        let weights = (0..<Self.channels).map { _ in
            (0..<Self.kernel).map { _ in
                Quantization.bf16ToFloat(Quantization.bf16Bits(rng.uniform(-1.0, 1.0)))
            }
        }
        let tokens = (0..<4).map { _ in
            (0..<Self.channels).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        }

        let ctx = try MetalContext()
        let qwen = try Qwen38Kernels(context: ctx)
        guard let stateBuf = Fp16Buffer.make(
                  ctx.device, count: Self.stateLen * Self.channels),
              let outBuf = Fp16Buffer.make(ctx.device, count: Self.channels),
              let wBuf = ctx.device.makeBuffer(
                  length: Self.channels * Self.kernel * 2,
                  options: .storageModeShared) else {
            Issue.record("Failed to allocate buffers"); return
        }
        memset(stateBuf.contents(), 0, stateBuf.length)
        let wPtr = wBuf.contents().bindMemory(
            to: UInt16.self, capacity: Self.channels * Self.kernel)
        for c in 0..<Self.channels {
            for t in 0..<Self.kernel { wPtr[c * Self.kernel + t] =
                Quantization.bf16Bits(weights[c][t]) }
        }

        var history: [[Float]] = []
        for (step, token) in tokens.enumerated() {
            guard let inBuf = Fp16Buffer.make(ctx.device, halves: token),
                  let cb = ctx.queue.makeCommandBuffer() else {
                Issue.record("no command buffer"); return
            }
            qwen.encodeShortConvStep(commandBuffer: cb, state: stateBuf,
                                     input: inBuf, weight: wBuf,
                                     weightOffset: 0, out: outBuf,
                                     channels: Self.channels,
                                     kernel: Self.kernel,
                                     dilation: Self.dilation,
                                     stateLen: Self.stateLen)
            cb.commit()
            cb.waitUntilCompleted()
            history.append(token.map { Float($0) })

            let got = Fp16Buffer.read(outBuf, count: Self.channels)
            for c in 0..<Self.channels {
                let acc = Self.crossCorrelate(history: history, at: step,
                                              channel: c, weights: weights)
                let want = Float(Float16(Self.silu(acc)))
                #expect(abs(got[c] - want) <= 1e-3,
                        "step \(step) channel \(c): got \(got[c]), want \(want)")
            }
            // State window = the last `stateLen` rows of the zero-padded
            // history, oldest first.
            let stateNow = Fp16Buffer.readHalf(
                stateBuf, count: Self.stateLen * Self.channels)
            for r in 0..<Self.stateLen {
                let src = step - (Self.stateLen - 1) + r
                for c in 0..<Self.channels {
                    let want: Float16 = src >= 0 ? Float16(history[src][c]) : 0
                    let got = stateNow[r * Self.channels + c]
                    #expect(got == want,
                            "step \(step) state row \(r) channel \(c): "
                                + "\(got) vs \(want)")
                }
            }
        }
    }
}
