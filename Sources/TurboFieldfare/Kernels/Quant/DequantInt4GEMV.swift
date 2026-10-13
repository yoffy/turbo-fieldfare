import Metal

/// MLX-affine INT4 matrix-vector multiplication.
/// Eight SIMD groups process eight output rows per threadgroup.
final class DequantInt4GEMV {
    private struct Shape: Hashable {
        var m: UInt32
        var n: UInt32
        var g: UInt32
    }

    private static let rowsPerThreadgroup = 8
    private static let realDecodeShapes: [Shape] = [
        Shape(m: 4096, n: 2816, g: 64),
        Shape(m: 2048, n: 2816, g: 64),
        Shape(m: 2816, n: 4096, g: 64),
        Shape(m: 8192, n: 2816, g: 64),
        Shape(m: 1024, n: 2816, g: 64),
        Shape(m: 2816, n: 8192, g: 64),
    ]

    private let pipeline: MTLComputePipelineState
    private let specializedPipelines: [Shape: MTLComputePipelineState]

    /// Group-32 pipelines for the Qwen 3.8 base (`FC_INT4_GROUP=32`). The
    /// 64-wide pipelines above stay bit-identical; the g32 set folds the
    /// same M/N constants with the 32-wide dequant path.
    private let pipelineG32: MTLComputePipelineState
    private let specializedPipelinesG32: [Shape: MTLComputePipelineState]

    private static func groupConstants(groupSize: Int) -> [MetalFunctionConstant] {
        guard groupSize == 32 else { return [] }
        return [
            MetalFunctionConstant(index: 27, value: .uint32(32)),
            MetalFunctionConstant(index: 28, value: .bool(true)),
        ]
    }

    private static func specializedConstants(shape: Shape) -> [MetalFunctionConstant] {
        [
            MetalFunctionConstant(index: 20, value: .uint32(shape.m)),
            MetalFunctionConstant(index: 21, value: .uint32(shape.n)),
            MetalFunctionConstant(index: 22, value: .bool(true)),
        ] + groupConstants(groupSize: Int(shape.g))
    }

    /// `additionalShapes` compiles extra constant-folded variants for a
    /// non-Gemma model's decode shapes. Measured: an unspecialized
    /// 4096x2048 GEMV runs at 102 GB/s, the specialized one at 141 GB/s.
    /// Shapes are compiled for both group widths; the runner selects by the
    /// manifest quant slot's recorded groupSize.
    init(context: MetalContext,
         additionalShapes: [(m: Int, n: Int)] = []) throws {
        self.pipeline = try context.pipeline(
            "dequant_int4_gemv_simd",
            constants: [],
            maxTotalThreadsPerThreadgroup: 512)
        self.pipelineG32 = try context.pipeline(
            "dequant_int4_gemv_simd",
            constants: Self.groupConstants(groupSize: 32),
            maxTotalThreadsPerThreadgroup: 512)

        let shapes = Self.realDecodeShapes
            + additionalShapes.map { Shape(m: UInt32($0.m), n: UInt32($0.n), g: 64) }
        var specializedPipelines: [Shape: MTLComputePipelineState] = [:]
        for shape in shapes {
            specializedPipelines[shape] = try context.pipeline(
                "dequant_int4_gemv_simd",
                constants: Self.specializedConstants(shape: shape),
                maxTotalThreadsPerThreadgroup: 512)
        }
        self.specializedPipelines = specializedPipelines

        var g32: [Shape: MTLComputePipelineState] = [:]
        let g32Shapes = Self.realDecodeShapes.map {
            Shape(m: $0.m, n: $0.n, g: 32)
        } + additionalShapes.map { Shape(m: UInt32($0.m), n: UInt32($0.n), g: 32) }
        for shape in g32Shapes {
            g32[shape] = try context.pipeline(
                "dequant_int4_gemv_simd",
                constants: Self.specializedConstants(shape: shape),
                maxTotalThreadsPerThreadgroup: 512)
        }
        self.specializedPipelinesG32 = g32
    }

    func encode(commandBuffer: MTLCommandBuffer,
                weights: MTLBuffer,
                weightsOffset: Int = 0,
                scales: MTLBuffer,
                scalesOffset: Int = 0,
                biases: MTLBuffer,
                biasesOffset: Int = 0,
                x: MTLBuffer,
                xOffset: Int = 0,
                y: MTLBuffer,
                yOffset: Int = 0,
                m: UInt32,
                n: UInt32,
                groupSize: Int = 64) {
        precondition(groupSize == 32 || groupSize == 64,
                     "groupSize must be 32 or 64, got \(groupSize)")
        precondition(n % UInt32(groupSize) == 0,
                     "N must be a multiple of \(groupSize)")
        // The kernel reads packed weights through a `ushort*`; the repacker
        // guarantees two-byte sub-tensor alignment but not four-byte alignment.
        precondition(weightsOffset % 2 == 0,
                     "dequant_int4_gemv_simd needs a 2-aligned weightsOffset, got \(weightsOffset)")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        let key = Shape(m: m, n: n, g: UInt32(groupSize))
        if groupSize == 32 {
            encoder.setComputePipelineState(
                specializedPipelinesG32[key] ?? pipelineG32)
        } else {
            encoder.setComputePipelineState(
                specializedPipelines[key] ?? pipeline)
        }
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(x, offset: xOffset, index: 3)
        encoder.setBuffer(y, offset: yOffset, index: 4)
        var mValue = m
        var nValue = n
        encoder.setBytes(&mValue, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&nValue, length: MemoryLayout<UInt32>.size, index: 6)

        let threadgroupSize = MTLSize(
            width: 32 * Self.rowsPerThreadgroup,
            height: 1,
            depth: 1)
        let threadgroupCount = MTLSize(
            width: (Int(m) + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup,
            height: 1,
            depth: 1)
        encoder.dispatchThreadgroups(threadgroupCount,
                                     threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }
}
