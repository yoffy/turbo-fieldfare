import Foundation
import Metal

final class FusedQKVGEMV {
    private struct Shape: Hashable {
        var qRows: UInt32
        var kvRows: UInt32
        var n: UInt32
    }

    private let pso: MTLComputePipelineState
    private let specializedPSOs: [Shape: MTLComputePipelineState]
    private let psoG32: MTLComputePipelineState
    private let specializedPSOsG32: [Shape: MTLComputePipelineState]

    private static let realDecodeShapes: [Shape] = [
        Shape(qRows: 4096, kvRows: 2048, n: 2816),
        Shape(qRows: 8192, kvRows: 1024, n: 2816),
    ]

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("dequant_int4_qkv_gemv_simd",
                                        constants: [],
                                        maxTotalThreadsPerThreadgroup: 512)
        // Qwen 3.8 QSA projections (4-bit, group 32). The QKV body is shared
        // with the decode GEMV, so the 32-wide path rides the same function
        // constants (27/28).
        let g32Constants: [MetalFunctionConstant] = [
            MetalFunctionConstant(index: 27, value: .uint32(32)),
            MetalFunctionConstant(index: 28, value: .bool(true)),
        ]
        self.psoG32 = try context.pipeline("dequant_int4_qkv_gemv_simd",
                                           constants: g32Constants,
                                           maxTotalThreadsPerThreadgroup: 512)
        var variants: [Shape: MTLComputePipelineState] = [:]
        for shape in Self.realDecodeShapes {
            variants[shape] = try context.pipeline(
                "dequant_int4_qkv_gemv_simd",
                constants: [
                    MetalFunctionConstant(index: 23, value: .uint32(shape.qRows)),
                    MetalFunctionConstant(index: 24, value: .uint32(shape.kvRows)),
                    MetalFunctionConstant(index: 25, value: .uint32(shape.n)),
                    MetalFunctionConstant(index: 26, value: .bool(true)),
                ],
                maxTotalThreadsPerThreadgroup: 512)
        }
        self.specializedPSOs = variants
        var g32: [Shape: MTLComputePipelineState] = [:]
        for shape in Self.realDecodeShapes {
            g32[shape] = try context.pipeline(
                "dequant_int4_qkv_gemv_simd",
                constants: [
                    MetalFunctionConstant(index: 23, value: .uint32(shape.qRows)),
                    MetalFunctionConstant(index: 24, value: .uint32(shape.kvRows)),
                    MetalFunctionConstant(index: 25, value: .uint32(shape.n)),
                    MetalFunctionConstant(index: 26, value: .bool(true)),
                ] + g32Constants,
                maxTotalThreadsPerThreadgroup: 512)
        }
        self.specializedPSOsG32 = g32
    }

    func encode(commandBuffer: MTLCommandBuffer,
                       qWeights: MTLBuffer, qWeightsOffset: Int = 0,
                       qScales: MTLBuffer, qScalesOffset: Int = 0,
                       qBiases: MTLBuffer, qBiasesOffset: Int = 0,
                       kWeights: MTLBuffer, kWeightsOffset: Int = 0,
                       kScales: MTLBuffer, kScalesOffset: Int = 0,
                       kBiases: MTLBuffer, kBiasesOffset: Int = 0,
                       vWeights: MTLBuffer, vWeightsOffset: Int = 0,
                       vScales: MTLBuffer, vScalesOffset: Int = 0,
                       vBiases: MTLBuffer, vBiasesOffset: Int = 0,
                       x: MTLBuffer,
                       qOut: MTLBuffer, qOutOffset: Int = 0,
                       kOut: MTLBuffer, kOutOffset: Int = 0,
                       vOut: MTLBuffer, vOutOffset: Int = 0,
                       qRows: UInt32,
                       kvRows: UInt32,
                       n: UInt32,
                       groupSize: Int = 64) {
        precondition(groupSize == 32 || groupSize == 64,
                     "groupSize must be 32 or 64, got \(groupSize)")
        precondition(n % UInt32(groupSize) == 0,
                     "N must be a multiple of \(groupSize)")
        precondition(qWeightsOffset % 2 == 0 &&
                     kWeightsOffset % 2 == 0 &&
                     vWeightsOffset % 2 == 0,
                     "FusedQKVGEMV needs 2-aligned weights offsets")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        let shape = Shape(qRows: qRows, kvRows: kvRows, n: n)
        if groupSize == 32 {
            enc.setComputePipelineState(specializedPSOsG32[shape] ?? psoG32)
        } else {
            enc.setComputePipelineState(specializedPSOs[shape] ?? pso)
        }
        enc.setBuffer(qWeights, offset: qWeightsOffset, index: 0)
        enc.setBuffer(qScales, offset: qScalesOffset, index: 1)
        enc.setBuffer(qBiases, offset: qBiasesOffset, index: 2)
        enc.setBuffer(kWeights, offset: kWeightsOffset, index: 3)
        enc.setBuffer(kScales, offset: kScalesOffset, index: 4)
        enc.setBuffer(kBiases, offset: kBiasesOffset, index: 5)
        enc.setBuffer(vWeights, offset: vWeightsOffset, index: 6)
        enc.setBuffer(vScales, offset: vScalesOffset, index: 7)
        enc.setBuffer(vBiases, offset: vBiasesOffset, index: 8)
        enc.setBuffer(x, offset: 0, index: 9)
        enc.setBuffer(qOut, offset: qOutOffset, index: 10)
        enc.setBuffer(kOut, offset: kOutOffset, index: 11)
        enc.setBuffer(vOut, offset: vOutOffset, index: 12)
        var qVar = qRows
        var kvVar = kvRows
        var nVar = n
        enc.setBytes(&qVar, length: MemoryLayout<UInt32>.size, index: 13)
        enc.setBytes(&kvVar, length: MemoryLayout<UInt32>.size, index: 14)
        enc.setBytes(&nVar, length: MemoryLayout<UInt32>.size, index: 15)
        let totalRows = Int(qRows + 2 * kvRows)
        enc.dispatchThreadgroups(MTLSize(width: (totalRows + 7) / 8,
                                         height: 1,
                                         depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256,
                                                                 height: 1,
                                                                 depth: 1))
        enc.endEncoding()
    }
}
