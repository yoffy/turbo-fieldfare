import Foundation
import Metal

/// 4-bit affine embedding lookup with fused output scale.
///
/// Gemma 4 ships a tied 4-bit MLX-affine embedding table (`embed_tokens.weight`
/// is `U32`/packed 4-bit with `weightBits=4` in the manifest). The same table
/// also drives the lm_head GEMV via `DequantInt4GEMV` over the transposed
/// access pattern. The `outScale` parameter fuses the `sqrt(hidden_size)`
/// post-embedding scale so the per-token dequant + scale
/// is one pass.
final class EmbedLookupInt4 {
    private let pso: MTLComputePipelineState
    private let psoG32: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("embed_lookup_int4")
        self.psoG32 = try context.pipeline(
            "embed_lookup_int4",
            constants: [
                MetalFunctionConstant(index: 27, value: .uint32(32)),
                MetalFunctionConstant(index: 28, value: .bool(true)),
            ])
    }

    /// Encodes the lookup. `table`, `scales`, `biases` typically live inside
    /// one resident blob — pass that buffer with the per-region offsets.
    /// Pass `outScale = 1.0` to write the raw dequantized row. `groupSize`
    /// selects the manifest slot's recorded width (64 legacy, 32 Qwen 3.8).
    func encode(commandBuffer: MTLCommandBuffer,
                       table:  MTLBuffer, tableOffset:  Int = 0,
                       scales: MTLBuffer, scalesOffset: Int = 0,
                       biases: MTLBuffer, biasesOffset: Int = 0,
                       out:    MTLBuffer, outOffset: Int = 0,
                       tokenId: UInt32,
                       d: UInt32,
                       outScale: Float,
                       groupSize: Int = 64) {
        precondition(groupSize == 32 || groupSize == 64,
                     "groupSize must be 32 or 64, got \(groupSize)")
        precondition(d % UInt32(groupSize) == 0,
                     "D must be a multiple of \(groupSize)")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(groupSize == 32 ? psoG32 : pso)
        enc.setBuffer(table,  offset: tableOffset,  index: 0)
        enc.setBuffer(scales, offset: scalesOffset, index: 1)
        enc.setBuffer(biases, offset: biasesOffset, index: 2)
        enc.setBuffer(out,    offset: outOffset,    index: 3)
        var tokenVar = tokenId
        var dVar     = d
        var sVar     = outScale
        enc.setBytes(&tokenVar, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&dVar,     length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&sVar,     length: MemoryLayout<Float>.size,  index: 6)

        let threadsPerGroup = min(Int(pso.maxTotalThreadsPerThreadgroup), 256)
        let gridSize = MTLSize(width: Int(d), height: 1, depth: 1)
        let tgSize   = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        enc.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        enc.endEncoding()
    }
}
