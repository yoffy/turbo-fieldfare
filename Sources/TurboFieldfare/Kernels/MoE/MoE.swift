import Foundation
import Metal

@frozen
public struct MoEExpertOffsets {
    public var gateWOff: UInt32
    public var gateSOff: UInt32
    public var gateBOff: UInt32
    public var upWOff: UInt32
    public var upSOff: UInt32
    public var upBOff: UInt32
    public var downWOff: UInt32
    public var downSOff: UInt32
    public var downBOff: UInt32

    public init(gateWOff: UInt32, gateSOff: UInt32, gateBOff: UInt32,
                upWOff: UInt32, upSOff: UInt32, upBOff: UInt32,
                downWOff: UInt32, downSOff: UInt32, downBOff: UInt32) {
        self.gateWOff = gateWOff
        self.gateSOff = gateSOff
        self.gateBOff = gateBOff
        self.upWOff = upWOff
        self.upSOff = upSOff
        self.upBOff = upBOff
        self.downWOff = downWOff
        self.downSOff = downSOff
        self.downBOff = downBOff
    }
}

final class MoE {
    /// Maximum top-k the streamer + kernels can carry in one decode step.
    /// Qwen 3.8 selects 10 of 512 experts; older families select 8 of
    /// 128/256. `validate` accepts exactly the supported top-k values so a
    /// misconfigured arch fails loudly instead of silently dropping experts.
    static let maxStreamedExperts = 10
    static let supportedTopK: [UInt32] = [8, 10]
    /// Maximum router width (Qwen 3.8: 512 experts).
    static let maxNumExperts = 512

    private let realDecodeD: UInt32
    private let realDecodeF: UInt32
    private let realDecodeTopK: UInt32
    private let realDecodeNumExperts: UInt32

    private let routerGemvPSO: MTLComputePipelineState
    private let routerGemvSpecializedPSO: MTLComputePipelineState
    private let routerGemvInt4PSO: MTLComputePipelineState
    private let routerGemvInt4SpecializedPSO: MTLComputePipelineState
    private let routerSelectK8PSO: MTLComputePipelineState
    private let routerSelectK8SpecializedPSO: MTLComputePipelineState
    private let routerSelectK10PSO: MTLComputePipelineState
    private let routerSelectK10SpecializedPSO: MTLComputePipelineState
    private let routerLogits: MTLBuffer
    private let phase1U16PSO: MTLComputePipelineState
    private let phase1U16SpecializedPSO: MTLComputePipelineState
    private let phase1SubsetU16PSO: MTLComputePipelineState
    private let phase1SubsetU16SpecializedPSO: MTLComputePipelineState
    private let phase2ReduceK8PSO: MTLComputePipelineState
    private let phase2ReduceK8SpecializedPSO: MTLComputePipelineState
    private let phase2ReduceK10PSO: MTLComputePipelineState
    private let phase2ReduceK10SpecializedPSO: MTLComputePipelineState
    /// 32-wide routed-expert variants (Qwen 3.8). Same kernels with
    /// `FC_MOE_GROUP=32` folded; the 64-wide PSOs above stay bit-identical.
    private let phase1U16G32PSO: MTLComputePipelineState
    private let phase1U16SpecializedG32PSO: MTLComputePipelineState
    private let phase1SubsetU16G32PSO: MTLComputePipelineState
    private let phase1SubsetU16SpecializedG32PSO: MTLComputePipelineState
    private let phase2ReduceK8G32PSO: MTLComputePipelineState
    private let phase2ReduceK8SpecializedG32PSO: MTLComputePipelineState
    private let phase2ReduceK10G32PSO: MTLComputePipelineState
    private let phase2ReduceK10SpecializedG32PSO: MTLComputePipelineState
    /// Quantized group width of the routed-expert weights (64 legacy, 32 for
    /// the Qwen 3.8 base). Router/scalar-gate GEMVs always read group 64.
    private let routedGroupSize: Int
    private let routedArgEncoder: MTLArgumentEncoder
    private let reusableRoutedArgBuffer: MTLBuffer

    /// `specializedD`/`specializedF`/`specializedNumExperts` describe the
    /// production shape this instance specializes for (Gemma 4 by default;
    /// Qwen 3.6 passes 2048/512/256, Qwen 3.8 passes 2560/640/512).
    /// `siluActivation` selects the expert FFN activation (false =
    /// gelu_pytorch_tanh, true = silu). `specializedTopK` selects the decode
    /// top-k the specialized pipelines fold (8 legacy, 10 for Qwen 3.8).
    init(context: MetalContext,
         siluActivation: Bool = false,
         specializedD: UInt32 = 2816,
         specializedF: UInt32 = 704,
         specializedNumExperts: UInt32 = 128,
         specializedTopK: UInt32 = 8,
         routedGroupSize: Int = 64) throws {
        precondition(Self.supportedTopK.contains(specializedTopK),
                     "unsupported specializedTopK \(specializedTopK)")
        precondition(routedGroupSize == 32 || routedGroupSize == 64,
                     "unsupported routedGroupSize \(routedGroupSize)")
        self.routedGroupSize = routedGroupSize
        self.realDecodeD = specializedD
        self.realDecodeF = specializedF
        self.realDecodeTopK = specializedTopK
        self.realDecodeNumExperts = specializedNumExperts
        let activationConstants: [MetalFunctionConstant] = siluActivation
            ? [MetalFunctionConstant(index: 4, value: .bool(true))]
            : []
        let moeConstants: [MetalFunctionConstant] = [
            MetalFunctionConstant(index: 0, value: .uint32(specializedD)),
            MetalFunctionConstant(index: 1, value: .uint32(specializedF)),
            MetalFunctionConstant(index: 2, value: .uint32(specializedTopK)),
            MetalFunctionConstant(index: 3, value: .bool(true)),
        ] + activationConstants
        let routerConstants: [MetalFunctionConstant] = [
            MetalFunctionConstant(index: 40, value: .uint32(specializedNumExperts)),
            MetalFunctionConstant(index: 41, value: .uint32(specializedD)),
            MetalFunctionConstant(index: 42, value: .uint32(specializedTopK)),
            MetalFunctionConstant(index: 43, value: .bool(true)),
        ]
        let routerName = "router_gemv_gemma4_r4"
        self.routerGemvPSO = try context.pipeline(
            routerName,
            constants: [],
            maxTotalThreadsPerThreadgroup: 512)
        self.routerGemvSpecializedPSO = try context.pipeline(
            routerName,
            constants: routerConstants,
            maxTotalThreadsPerThreadgroup: 512)
        // Qwen 3.5's 4-bit router: same bindings and top-k tail, nibble-packed
        // weights (see `router_gemv_int4` in moe.metal).
        self.routerGemvInt4PSO = try context.pipeline(
            "router_gemv_int4",
            constants: [],
            maxTotalThreadsPerThreadgroup: 512)
        self.routerGemvInt4SpecializedPSO = try context.pipeline(
            "router_gemv_int4",
            constants: routerConstants,
            maxTotalThreadsPerThreadgroup: 512)
        self.routerSelectK8PSO = try context.pipeline("router_topk_select_k8")
        self.routerSelectK8SpecializedPSO = try context.pipeline(
            "router_topk_select_k8",
            constants: routerConstants)
        // Qwen 3.8 top-10 tail: same insertion-sort + renormalized top-k
        // softmax as k8, over 10 slots (see `router_topk_select_k10`).
        self.routerSelectK10PSO = try context.pipeline("router_topk_select_k10")
        self.routerSelectK10SpecializedPSO = try context.pipeline(
            "router_topk_select_k10",
            constants: routerConstants)
        self.phase1U16PSO = try context.pipeline(
            "moe_phase1_gate_up_act_u16load", constants: activationConstants)
        self.phase1U16SpecializedPSO = try context.pipeline(
            "moe_phase1_gate_up_act_u16load",
            constants: moeConstants)
        self.phase1SubsetU16PSO = try context.pipeline(
            "moe_phase1_gate_up_act_subset_u16load", constants: activationConstants)
        self.phase1SubsetU16SpecializedPSO = try context.pipeline(
            "moe_phase1_gate_up_act_subset_u16load",
            constants: moeConstants)
        self.phase2ReduceK8PSO = try context.pipeline("moe_phase2_down_reduce_k8")
        self.phase2ReduceK8SpecializedPSO = try context.pipeline(
            "moe_phase2_down_reduce_k8",
            constants: moeConstants)
        self.phase2ReduceK10PSO = try context.pipeline("moe_phase2_down_reduce_k10")
        self.phase2ReduceK10SpecializedPSO = try context.pipeline(
            "moe_phase2_down_reduce_k10",
            constants: moeConstants)
        let g32Extra: [MetalFunctionConstant] = [
            MetalFunctionConstant(index: 5, value: .uint32(32)),
            MetalFunctionConstant(index: 6, value: .bool(true)),
        ]
        self.phase1U16G32PSO = try context.pipeline(
            "moe_phase1_gate_up_act_u16load",
            constants: activationConstants + g32Extra)
        self.phase1U16SpecializedG32PSO = try context.pipeline(
            "moe_phase1_gate_up_act_u16load",
            constants: moeConstants + g32Extra)
        self.phase1SubsetU16G32PSO = try context.pipeline(
            "moe_phase1_gate_up_act_subset_u16load",
            constants: activationConstants + g32Extra)
        self.phase1SubsetU16SpecializedG32PSO = try context.pipeline(
            "moe_phase1_gate_up_act_subset_u16load",
            constants: moeConstants + g32Extra)
        self.phase2ReduceK8G32PSO = try context.pipeline(
            "moe_phase2_down_reduce_k8", constants: g32Extra)
        self.phase2ReduceK8SpecializedG32PSO = try context.pipeline(
            "moe_phase2_down_reduce_k8",
            constants: moeConstants + g32Extra)
        self.phase2ReduceK10G32PSO = try context.pipeline(
            "moe_phase2_down_reduce_k10", constants: g32Extra)
        self.phase2ReduceK10SpecializedG32PSO = try context.pipeline(
            "moe_phase2_down_reduce_k10",
            constants: moeConstants + g32Extra)

        guard let logits = context.device.makeBuffer(
            length: Self.maxNumExperts * MemoryLayout<Float>.stride,
            options: .storageModeShared),
              let phase1Function = context.library.makeFunction(
                name: "moe_phase1_gate_up_act_u16load") else {
            throw MetalError.noDevice
        }
        self.routerLogits = logits
        self.routedArgEncoder = phase1Function.makeArgumentEncoder(bufferIndex: 0)
        guard let reusable = context.device.makeBuffer(
            length: routedArgEncoder.encodedLength,
            options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        self.reusableRoutedArgBuffer = reusable
    }

    func encodeRouterGemma4(commandBuffer: MTLCommandBuffer,
                                   weights: MTLBuffer, weightsOffset: Int = 0,
                                   scales: MTLBuffer, scalesOffset: Int = 0,
                                   biases: MTLBuffer, biasesOffset: Int = 0,
                                   hidden: MTLBuffer,
                                   effectiveScale: MTLBuffer, effectiveScaleOffset: Int = 0,
                                   perExpertScale: MTLBuffer, perExpertScaleOffset: Int = 0,
                                   outIndices: MTLBuffer,
                                   outWeights: MTLBuffer,
                                   numExperts: UInt32,
                                   d: UInt32,
                                   topK: UInt32,
                                   weightBits: Int = 8) {
        precondition(d.isMultiple(of: 32),
                     "router hidden dim \(d) must be a multiple of 32 (supports group 32/64)")
        precondition(numExperts <= UInt32(Self.maxNumExperts))
        precondition(Self.supportedTopK.contains(topK),
                     "router topK must be 8 or 10, got \(topK)")
        precondition(weightBits == 8 || weightBits == 4,
                     "router weight bits must be 8 or 4")

        var expertCount = numExperts
        var dimension = d
        let useSpecialized = numExperts == realDecodeNumExperts
            && d == realDecodeD
            && topK == realDecodeTopK
        let gemvPSO = weightBits == 4
            ? (useSpecialized ? routerGemvInt4SpecializedPSO : routerGemvInt4PSO)
            : (useSpecialized ? routerGemvSpecializedPSO : routerGemvPSO)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(gemvPSO)
            encoder.setBuffer(weights, offset: weightsOffset, index: 0)
            encoder.setBuffer(scales, offset: scalesOffset, index: 1)
            encoder.setBuffer(biases, offset: biasesOffset, index: 2)
            encoder.setBuffer(hidden, offset: 0, index: 3)
            encoder.setBuffer(effectiveScale, offset: effectiveScaleOffset, index: 4)
            encoder.setBuffer(routerLogits, offset: 0, index: 5)
            encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 6)
            encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 7)
            encoder.dispatchThreadgroups(
                MTLSize(width: (Int(numExperts) + 3) / 4, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            encoder.endEncoding()
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            let selectPSO: MTLComputePipelineState
            switch topK {
            case 8:
                selectPSO = useSpecialized
                    ? routerSelectK8SpecializedPSO : routerSelectK8PSO
            case 10:
                selectPSO = useSpecialized
                    ? routerSelectK10SpecializedPSO : routerSelectK10PSO
            default:
                fatalError("unreachable: topK validated above")
            }
            encoder.setComputePipelineState(selectPSO)
            encoder.setBuffer(routerLogits, offset: 0, index: 0)
            encoder.setBuffer(perExpertScale, offset: perExpertScaleOffset, index: 1)
            encoder.setBuffer(outIndices, offset: 0, index: 2)
            encoder.setBuffer(outWeights, offset: 0, index: 3)
            encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 4)
            encoder.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            encoder.endEncoding()
        }
    }

    func makeRoutedArgumentBuffer(routedBlobs: [MTLBuffer],
                                         topK: UInt32) -> MTLBuffer? {
        validate(routedBlobs: routedBlobs, topK: topK)
        guard let buffer = routedBlobs.first?.device.makeBuffer(
            length: routedArgEncoder.encodedLength,
            options: .storageModeShared) else {
            return nil
        }
        encodeRoutedArgumentBuffer(buffer, routedBlobs: routedBlobs)
        return buffer
    }

    func makeReusedRoutedArgumentBuffer(routedBlobs: [MTLBuffer],
                                               topK: UInt32) -> MTLBuffer {
        validate(routedBlobs: routedBlobs, topK: topK)
        encodeRoutedArgumentBuffer(reusableRoutedArgBuffer, routedBlobs: routedBlobs)
        return reusableRoutedArgBuffer
    }

    func encodeRoutedPersistentPhase1U16Load(
        commandBuffer: MTLCommandBuffer,
        routedArgBuffer: MTLBuffer,
        routedBlobs: [MTLBuffer],
        routedOffsets: MoEExpertOffsets,
        x: MTLBuffer,
        acts: MTLBuffer,
        d: UInt32,
        f: UInt32,
        topK: UInt32
    ) {
        validate(routedBlobs: routedBlobs, topK: topK)
        var dimension = d
        var intermediate = f
        var expertCount = topK
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        let g32 = routedGroupSize == 32
        let specialized = useRealDecodeConstants(d: d, f: f)
        let pso1: MTLComputePipelineState
        if g32 {
            pso1 = specialized ? phase1U16SpecializedG32PSO : phase1U16G32PSO
        } else {
            pso1 = specialized ? phase1U16SpecializedPSO : phase1U16PSO
        }
        encoder.setComputePipelineState(pso1)
        encoder.setBuffer(routedArgBuffer, offset: 0, index: 0)
        for buffer in routedBlobs { encoder.useResource(buffer, usage: .read) }
        var offsets = routedOffsets
        encoder.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        encoder.setBuffer(x, offset: 0, index: 2)
        encoder.setBuffer(acts, offset: 0, index: 3)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(topK * f) + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encodeRoutedPersistentPhase1SubsetU16Load(
        commandBuffer: MTLCommandBuffer,
        routedArgBuffer: MTLBuffer,
        routedBlobs: [MTLBuffer],
        routedOffsets: MoEExpertOffsets,
        x: MTLBuffer,
        acts: MTLBuffer,
        activeSlots: MTLBuffer,
        activeSlotIndices: [UInt32],
        activeCount: UInt32,
        d: UInt32,
        f: UInt32,
        topK: UInt32
    ) {
        guard activeCount > 0 else { return }
        validate(routedBlobs: routedBlobs, topK: topK)
        precondition(activeSlotIndices.count == Int(activeCount))
        var dimension = d
        var intermediate = f
        var expertCount = topK
        var active = activeCount
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        let g32Subset = routedGroupSize == 32
        let specializedSubset = useRealDecodeConstants(d: d, f: f)
        let psoSubset: MTLComputePipelineState
        if g32Subset {
            psoSubset = specializedSubset
                ? phase1SubsetU16SpecializedG32PSO : phase1SubsetU16G32PSO
        } else {
            psoSubset = specializedSubset
                ? phase1SubsetU16SpecializedPSO : phase1SubsetU16PSO
        }
        encoder.setComputePipelineState(psoSubset)
        encoder.setBuffer(routedArgBuffer, offset: 0, index: 0)
        for slot in activeSlotIndices {
            encoder.useResource(routedBlobs[Int(slot)], usage: .read)
        }
        var offsets = routedOffsets
        encoder.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        encoder.setBuffer(x, offset: 0, index: 2)
        encoder.setBuffer(acts, offset: 0, index: 3)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBuffer(activeSlots, offset: 0, index: 7)
        encoder.setBytes(&active, length: MemoryLayout<UInt32>.stride, index: 8)
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(activeCount * f) + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encodeRoutedPersistentPhase2Reduce(
        commandBuffer: MTLCommandBuffer,
        routedArgBuffer: MTLBuffer,
        routedBlobs: [MTLBuffer],
        routedOffsets: MoEExpertOffsets,
        acts: MTLBuffer,
        routingWeights: MTLBuffer,
        residual: MTLBuffer,
        y: MTLBuffer,
        d: UInt32,
        f: UInt32,
        topK: UInt32
    ) {
        validate(routedBlobs: routedBlobs, topK: topK)
        var dimension = d
        var intermediate = f
        var expertCount = topK
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        let specialized = useRealDecodeConstants(d: d, f: f, topK: topK)
        let g32Reduce = routedGroupSize == 32
        switch topK {
        case 8:
            encoder.setComputePipelineState(
                g32Reduce
                    ? (specialized ? phase2ReduceK8SpecializedG32PSO : phase2ReduceK8G32PSO)
                    : (specialized ? phase2ReduceK8SpecializedPSO : phase2ReduceK8PSO))
        case 10:
            encoder.setComputePipelineState(
                g32Reduce
                    ? (specialized ? phase2ReduceK10SpecializedG32PSO : phase2ReduceK10G32PSO)
                    : (specialized ? phase2ReduceK10SpecializedPSO : phase2ReduceK10PSO))
        default:
            fatalError("unreachable: topK validated")
        }
        encoder.setBuffer(routedArgBuffer, offset: 0, index: 0)
        for buffer in routedBlobs { encoder.useResource(buffer, usage: .read) }
        var offsets = routedOffsets
        encoder.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        encoder.setBuffer(acts, offset: 0, index: 2)
        encoder.setBuffer(routingWeights, offset: 0, index: 3)
        encoder.setBuffer(residual, offset: 0, index: 4)
        encoder.setBuffer(y, offset: 0, index: 5)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 7)
        // The k10 kernel binds the top-k count at index 8 (k8 hardcodes 8
        // and leaves the slot unset).
        if topK == 10 {
            encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 8)
        }
        // k8 uses 8 SIMDs (256 threads); k10 uses 10 SIMDs (320 threads).
        let threadsPerTG = topK == 10 ? 320 : 256
        encoder.dispatchThreadgroups(
            MTLSize(width: Int(d), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadsPerTG, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func validate(routedBlobs: [MTLBuffer], topK: UInt32) {
        precondition(Self.supportedTopK.contains(topK),
                     "MoE topK must be 8 or 10, got \(topK)")
        precondition(routedBlobs.count == Int(topK))
    }

    private func encodeRoutedArgumentBuffer(_ buffer: MTLBuffer,
                                            routedBlobs: [MTLBuffer]) {
        routedArgEncoder.setArgumentBuffer(buffer, offset: 0)
        for (index, blob) in routedBlobs.enumerated() {
            routedArgEncoder.setBuffer(blob, offset: 0, index: index)
        }
    }

    private func useRealDecodeConstants(d: UInt32, f: UInt32) -> Bool {
        d == realDecodeD && f == realDecodeF
    }

    private func useRealDecodeConstants(d: UInt32, f: UInt32, topK: UInt32) -> Bool {
        d == realDecodeD && f == realDecodeF && topK == realDecodeTopK
    }
}
