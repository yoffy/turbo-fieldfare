import Foundation
import Metal

public enum RDAdvicePolicyMode: String, Codable, Sendable, Equatable {
    case `default`
    case off
    case bounded
    case adaptive

    public static func parse(_ raw: String?) -> RDAdvicePolicyMode {
        switch raw?.lowercased() {
        case "off", "none", "disabled":
            return .off
        case "bounded":
            return .bounded
        case "adaptive":
            return .adaptive
        default:
            return .default
        }
    }
}

public struct RDAdviceAdaptivePolicyConfig: Sendable, Equatable {
    public var missCap: Int
    public var byteCap: UInt64
    public var slowCallNanos: UInt64

    public init(missCap: Int,
                byteCap: UInt64,
                slowCallNanos: UInt64) {
        self.missCap = missCap
        self.byteCap = byteCap
        self.slowCallNanos = slowCallNanos
    }

    public static let conservative = RDAdviceAdaptivePolicyConfig(
        missCap: 12,
        byteCap: 384 * 1_048_576,
        slowCallNanos: 1_000_000)
}

struct RDAdviceAdaptivePolicyState: Sendable, Equatable {
    var config: RDAdviceAdaptivePolicyConfig
    private var skipUntilPosition: Int = -1
    private(set) var recentSlowCallNanos: UInt64 = 0

    init(config: RDAdviceAdaptivePolicyConfig = .conservative) {
        self.config = config
    }

    mutating func reset() {
        skipUntilPosition = -1
        recentSlowCallNanos = 0
    }

    func shouldSkip(position: Int,
                    requestedMisses: Int,
                    estimatedBytes: UInt64,
                    canOverlapUsefulGPUWork: Bool) -> Bool {
        position <= skipUntilPosition ||
        !canOverlapUsefulGPUWork ||
        requestedMisses > config.missCap ||
        estimatedBytes > config.byteCap
    }

    mutating func update(after result: ExpertIOAdviceResult,
                                position: Int) {
        recentSlowCallNanos = max(recentSlowCallNanos, result.maxCallNanos)
        if result.maxCallNanos >= config.slowCallNanos {
            skipUntilPosition = max(skipUntilPosition, position)
        }
    }
}

/// Gemma 4 real-forward decode pass.
///
/// Composes the production kernels against the `.gturbo` model:
///
///   embed_lookup_int4(token) * sqrt(H)
///   for L in 0..<30:
///     a = rmsnorm_bf16w(h, input_layernorm)
///     Q = q_proj(a)    K = k_proj(a)    V = (SWA) v_proj(a) | (full) k_proj(a)
///     per-head q/k_norm (bf16w), per-head v_norm (no_scale)
///     NeoX RoPE on Q + K (default for SWA, proportional for full)
///     write K and V into separate cache slots
///     attn = attention(scale=1.0, SWA window or full causal)
///     attn = o_proj(attn)
///     h = h + rmsnorm_bf16w(attn, post_attention_layernorm)
///     h1 = rmsnorm_bf16w(h, pre_feedforward_layernorm)
///     h1 = SharedExpertInt8(h1)
///     h1 = rmsnorm_bf16w(h1, post_feedforward_layernorm_1)
///     // router + routed branch
///     xr   = rmsnorm_no_scale(h)
///     idx, w = router_topk_gemma4(xr, effective_scale[L], per_expert_scale[L])
///     h2 = rmsnorm_bf16w(h, pre_feedforward_layernorm_2)
///     h2 = moe_fused_ffn_streamed_routed(h2, residual=0, routedBlobs=fetch(idx), w)
///     h2 = rmsnorm_bf16w(h2, post_feedforward_layernorm_2)
///     h = h + rmsnorm_bf16w(h1 + h2, post_feedforward_layernorm)
///     h = h * layer_scalar[L]
///   logits = DequantInt4GEMV(rmsnorm_bf16w(h, model.norm), embed_table^T)
///   // final softcap and softmax happen in the Sampler.
///
/// Direct against `Model`; this is the only production decode forward path.
internal enum PrefillProjectionFamily: Sendable, Equatable {
    case q
    case kv
    case o
    case shared
    case routed
}

internal enum PrefillProjectionDispatch: Sendable, Equatable {
    case repeatedGEMV
    case qmm
}

internal enum PrefillProjectionDispatchPolicy {
    static func selectedDispatch(for family: PrefillProjectionFamily,
                                 chunkTokens: Int) -> PrefillProjectionDispatch {
        guard chunkTokens >= 32 else {
            return .repeatedGEMV
        }
        switch family {
        case .q:
            return .repeatedGEMV
        case .kv, .o, .shared, .routed:
            return .qmm
        }
    }
}

public final class RealForwardRunner: ChunkedPrefillRunner, ContextWindowReporting, ContinuableLogitProducer, @unchecked Sendable {
    private struct LayerSharedExpertProjections {
        let gate: SharedExpertInt8Proj
        let up: SharedExpertInt8Proj
        let down: SharedExpertInt8Proj
        /// Gemma-only post_feedforward_layernorm_1; nil when the arch has no
        /// FFN sandwich norms.
        let postF1: TensorView?
        /// Qwen-only [1, hidden] scalar gate on the shared expert branch.
        let scalarGate: TensorView?
    }

    private let model: Model
    private let ctx: MetalContext
    private let kv: KVCacheManager?
    private let cfg: ArchConfig

    // Kernels
    private let embedInt4: EmbedLookupInt4
    private let rms: RMSNorm
    private let int4: DequantInt4GEMV
    private let attention: Attention
    private let shared: SharedExpertRuntime
    private let moe: MoE
    private let fusionHead: LMHeadChainInt4
    private let fusedQKVGEMV: FusedQKVGEMV
    private let fusedQKVEpilogue: FusedQKVEpilogue
    private let fusedPostAttentionSetup: FusedPostAttentionSetup
    private let fusedTail: FusedLayerTail

    // Qwen 3.6 kernels. Nil on architectures that never dispatch them.
    private let elementwise: Elementwise?
    private let gdn: GDN?
    private let gdnState: GDNStateManager?
    private let rope: RoPE?
    private let int8ScalarGate: DequantInt8GEMV?

    // Prefill kernels. These are initialized once per runner so the chunk path
    // cannot accidentally rebuild PSOs inside a per-layer loop.
    private let prefillEmbed: PrefillEmbedLookupInt4
    private let prefillRMS: PrefillRMSNorm
    private let prefillQMM: PrefillInt4QMM
    private let prefillMPPAffineInt4: MPPPrefillInt4QMM?
    private let prefillQKVEpilogue: PrefillQKVEpilogue
    private let prefillAttention: PrefillAttention
    private let prefillPostAttention: PrefillPostAttentionSetup
    private let prefillRouter: PrefillRouter
    private let prefillSharedExpert: PrefillSharedExpert
    private let prefillGroupedMoE: PrefillGroupedRoutedMoE
    private let prefillMoE: PrefillMoE
    private let prefillLayerTail: PrefillLayerTail
    private let prefillFinalRowHead: PrefillFinalRowHeadInt4

    // Scratch — preallocated per spec'd D / F / vocab.
    private let hidden: MTLBuffer        // [D] FP16
    private let normed: MTLBuffer        // [D] FP16
    private let attnOut: MTLBuffer       // [N_HEADS * head_dim] FP16
    private let qScratch: MTLBuffer      // [N_HEADS * head_dim] FP16
    private let kStage: MTLBuffer        // [max KV heads * head_dim] FP16, current token
    private let vStage: MTLBuffer        // [max KV heads * head_dim] FP16, current token
    private let oOut: MTLBuffer          // [D] FP16
    private let h1Buf: MTLBuffer         // [D] FP16 (dense MLP output)
    private let h2Buf: MTLBuffer         // [D] FP16 (routed output)
    private let routedX: MTLBuffer       // [D] FP16 (pre_feedforward_layernorm_2 output)
    private let denseX: MTLBuffer        // [D] FP16 (pre_feedforward_layernorm output)
    private let denseScratchGate: MTLBuffer // [F=2112] FP16
    private let denseScratchUp: MTLBuffer   // [F=2112] FP16
    private let denseScratchAct: MTLBuffer  // [F=2112] FP16
    private let routerInput: MTLBuffer   // [D] FP16 (rmsnorm_no_scale(h))
    private let zeroResidual: MTLBuffer  // [D] FP16 zeros — for routed branch base
    private let outIndices: MTLBuffer    // [topK] UInt32
    private let outWeights: MTLBuffer    // [topK] FP16
    // Persistent MoE scratch, allocated once; about 56 KiB at production shape.
    private let moeActs: MTLBuffer       // [topK * FmoE] FP16
    private let moeHitActiveSlots: MTLBuffer // [topK] UInt32
    private let moeMissActiveSlots: MTLBuffer // [topK] UInt32
    private let greedyTokenBuf: MTLBuffer // 4 B UInt32 fused-head output
    // Qwen 3.6 decode scratch (nil on architectures that never use it).
    private let qPackedScratch: MTLBuffer?   // [2 * N_HEADS * head_dim] packed [q ; gate]
    private let attnGateScratch: MTLBuffer?  // [N_HEADS * head_dim]
    private let gdnQKVRaw: MTLBuffer?        // [qkvDim] raw in_proj_qkv output
    private let gdnConvOut: MTLBuffer?       // [qkvDim] conv + SiLU output
    private let gdnZ: MTLBuffer?             // [valueDim]
    private let gdnA: MTLBuffer?             // [numVHeads]
    private let gdnB: MTLBuffer?             // [numVHeads]
    private let gdnY: MTLBuffer?             // [valueDim] delta-rule output
    private let gdnOut: MTLBuffer?           // [valueDim] gated-norm output
    private let sharedScalarGateBuf: MTLBuffer? // [1] shared-expert gate logit
    /// BF16 ones over [numExperts]; neutral per_expert_scale when the router
    /// has no auxiliary scale tensors.
    private let onesPerExpertScale: MTLBuffer?
    private var prefillChunkState = PrefillChunkCommitState()
    private var prefillScratch: PrefillChunkScratchBuffers?
    /// Debug: recurrent state snapshot taken just before each GDN delta step.
    private var debugStateBefore: MTLBuffer?
    private var debugTailBefore: MTLBuffer?

    /// Debug: when `TURBO_RMS_DUMP=1`, the running `hidden` residual is
    /// read back to the CPU after every layer and its RMS + maxAbs are
    /// printed, so a corrupted layer can be located by the first RMS blow-up.
    private var debugRmsDump: Bool {
        ProcessInfo.processInfo.environment["TURBO_RMS_DUMP"] == "1"
    }

    private static let rdadviseBoundedMissCap = 12
    private static let rdadviseBoundedMaxCallNanos: UInt64 = 250_000
    private static let rdadviseAdaptiveMissCap = 12
    private static let rdadviseAdaptiveByteCap: UInt64 = 384 * 1_048_576
    private static let rdadviseAdaptiveSlowCallNanos: UInt64 = 1_000_000
    private static let prefillRoutedTileSchedulerConfig = PrefillRoutedTileSchedulerConfig()

    /// Per-layer `router.scale * D^-0.5` pre-folded into one BF16 buffer
    /// allocation per layer. ~168 KB total at 30 layers × 2816 BF16 — bounded
    /// host work done once at init.
    private let effectiveScaleBuffers: [MTLBuffer]
    private let sharedExpertProjections: [LayerSharedExpertProjections]

    public let maxContext: Int

    /// Per-instance head and RDADVISE modes. The fused head (default) skips the
    /// 512 KB logits write and leaves a greedy argmax in `lastGreedyToken`;
    /// callers that sample from the logits buffer (non-greedy configs) must pass
    /// `forceLogitsHead: true` or they read a never-written buffer.
    private let useFusedGreedyHead: Bool
    private let prefillAttentionPath: RuntimePrefillAttentionPath
    public let rdadviseEnabled: Bool
    public let rdadvisePolicyMode: RDAdvicePolicyMode
    private var rdadviseSkipUntilPosition: Int = -1
    private var rdadviseAdaptiveState: RDAdviceAdaptivePolicyState
    private var rdadviseAdaptivePosition: Int = -1
    private var rdadviseAdaptivePositionBytes: UInt64 = 0
    public init(model: Model, context: MetalContext, maxContext: Int,
                runtimeConfiguration: RuntimeConfiguration = .production) throws {
        self.model = model
        self.ctx = context
        self.cfg = model.config
        self.maxContext = maxContext
        self.useFusedGreedyHead = runtimeConfiguration.headPath == .fusedRows
        self.prefillAttentionPath = runtimeConfiguration.prefillAttentionPath
        let useFP16Ring = runtimeConfiguration.fp16RingEnabled
        self.rdadvisePolicyMode = runtimeConfiguration.rdadvisePolicy
        self.rdadviseAdaptiveState = RDAdviceAdaptivePolicyState(
            config: RDAdviceAdaptivePolicyConfig(
                missCap: Self.rdadviseAdaptiveMissCap,
                byteCap: Self.rdadviseAdaptiveByteCap,
                slowCallNanos: Self.rdadviseAdaptiveSlowCallNanos))
        self.rdadviseEnabled = runtimeConfiguration.rdadviseEnabled
        self.kv = try KVCacheManager(device: context.device,
                                     config: cfg,
                                     maxContext: maxContext,
                                     fp16RingEnabled: useFP16Ring,
                                     slidingWindow: cfg.slidingWindow,
                                     maxPrefillChunkTokens: PrefillRuntimeConfig.maxChunkTokens)

        let silu = cfg.hiddenActivation == "silu"
        self.embedInt4 = try EmbedLookupInt4(context: context)
        self.rms       = try RMSNorm(context: context)
        self.int4      = try DequantInt4GEMV(
            context: context,
            additionalShapes: cfg.decodeInt4GEMVShapes)
        self.attention = try Attention(context: context)
        self.shared    = try SharedExpertRuntime(context: context,
                                                  weightBits: model.sharedExpertWeightBits,
                                                  siluActivation: silu)
        self.moe       = try MoE(context: context,
                                 siluActivation: silu,
                                 specializedD: UInt32(cfg.hiddenSize),
                                 specializedF: UInt32(cfg.moeIntermediateSize),
                                 specializedNumExperts: UInt32(cfg.numExperts))
        self.fusionHead = try LMHeadChainInt4(context: context,
                                              maxD: cfg.hiddenSize,
                                              maxVocab: cfg.vocabSize)
        self.fusedQKVGEMV = try FusedQKVGEMV(context: context)
        self.fusedQKVEpilogue = try FusedQKVEpilogue(context: context)
        self.fusedPostAttentionSetup = try FusedPostAttentionSetup(context: context)
        self.fusedTail = try FusedLayerTail(context: context)
        self.prefillEmbed = try PrefillEmbedLookupInt4(context: context)
        self.prefillRMS = try PrefillRMSNorm(context: context)
        self.prefillQMM = try PrefillInt4QMM(context: context)
        self.prefillMPPAffineInt4 = MPPPrefillInt4QMM(context: context)
        self.prefillQKVEpilogue = try PrefillQKVEpilogue(context: context)
        self.prefillAttention = try PrefillAttention(context: context)
        self.prefillPostAttention = try PrefillPostAttentionSetup(context: context)
        self.prefillRouter = try PrefillRouter(context: context)
        self.prefillSharedExpert = try PrefillSharedExpert(
            context: context,
            weightBits: model.sharedExpertWeightBits,
            siluActivation: silu)
        self.prefillGroupedMoE = try PrefillGroupedRoutedMoE(context: context,
                                                             siluActivation: silu)
        self.prefillMoE = try PrefillMoE(context: context)
        self.prefillLayerTail = try PrefillLayerTail(context: context)
        self.prefillFinalRowHead = try PrefillFinalRowHeadInt4(context: context,
                                                               maxD: cfg.hiddenSize)

        // Qwen 3.6 kernels, keyed off the data flags so architectures that
        // never dispatch them pay no PSO compile cost.
        let needsElementwise = cfg.attnOutputGate
            || cfg.sharedExpertGated
            || !cfg.ffnSandwichNorms
            || cfg.hasLinearAttentionLayers
        self.elementwise = needsElementwise ? try Elementwise(context: context) : nil
        if cfg.hasLinearAttentionLayers {
            self.gdn = try GDN(context: context, config: cfg.linearAttention,
                               specializedHiddenSize: cfg.hiddenSize)
            self.gdnState = try GDNStateManager(device: context.device, config: cfg)
        } else {
            self.gdn = nil
            self.gdnState = nil
        }
        self.rope = cfg.ropeNeoxSubdim ? try RoPE(context: context) : nil
        self.int8ScalarGate = cfg.sharedExpertGated
            ? try DequantInt8GEMV(context: context,
                                  additionalShapes: cfg.decodeInt8GEMVShapes)
            : nil

        let device = context.device
        let D = cfg.hiddenSize
        let F = cfg.intermediateSize
        let maxQ = cfg.numHeads * max(cfg.headDim, cfg.fullHeadDim)

        func buf(_ count: Int, _ stride: Int = MemoryLayout<Float16>.size) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: max(count, 1) * stride,
                                            options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            return b
        }
        self.hidden        = try buf(D)
        self.normed        = try buf(D)
        self.attnOut       = try buf(maxQ)
        self.qScratch      = try buf(maxQ)
        self.kStage        = try buf(max(cfg.numKVHeads * cfg.headDim,
                                         cfg.numFullKVHeads * cfg.fullHeadDim))
        self.vStage        = try buf(max(cfg.numKVHeads * cfg.headDim,
                                         cfg.numFullKVHeads * cfg.fullHeadDim))
        self.oOut          = try buf(D)
        self.h1Buf         = try buf(D)
        self.h2Buf         = try buf(D)
        self.routedX       = try buf(D)
        self.denseX        = try buf(D)
        self.denseScratchGate = try buf(F)
        self.denseScratchUp   = try buf(F)
        self.denseScratchAct  = try buf(F)
        self.routerInput   = try buf(D)
        self.zeroResidual  = try buf(D)
        // The routed MoE kernel seeds y[d] = residual[d]; pinning this buffer
        // to zero once at init makes the routed branch's residual contribution
        // exactly zero (it's combined with the dense MLP downstream).
        memset(self.zeroResidual.contents(), 0, self.zeroResidual.length)
        self.outIndices    = try buf(cfg.topKExperts, MemoryLayout<UInt32>.size)
        self.outWeights    = try buf(cfg.topKExperts)
        self.moeActs       = try buf(cfg.topKExperts * cfg.moeIntermediateSize)
        self.moeHitActiveSlots = try buf(cfg.topKExperts, MemoryLayout<UInt32>.size)
        self.moeMissActiveSlots = try buf(cfg.topKExperts, MemoryLayout<UInt32>.size)
        guard let tok = device.makeBuffer(length: MemoryLayout<UInt32>.size,
                                          options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        self.greedyTokenBuf = tok

        // Qwen 3.6 decode scratch — allocated once here, never in the hot path.
        if cfg.attnOutputGate {
            self.qPackedScratch = try buf(2 * maxQ)
            self.attnGateScratch = try buf(maxQ)
        } else {
            self.qPackedScratch = nil
            self.attnGateScratch = nil
        }
        if cfg.hasLinearAttentionLayers {
            let la = cfg.linearAttention
            self.gdnQKVRaw = try buf(la.qkvDim)
            self.gdnConvOut = try buf(la.qkvDim)
            self.gdnZ = try buf(la.valueDim)
            self.gdnA = try buf(la.numVHeads)
            self.gdnB = try buf(la.numVHeads)
            self.gdnY = try buf(la.valueDim)
            self.gdnOut = try buf(la.valueDim)
        } else {
            self.gdnQKVRaw = nil
            self.gdnConvOut = nil
            self.gdnZ = nil
            self.gdnA = nil
            self.gdnB = nil
            self.gdnY = nil
            self.gdnOut = nil
        }
        self.sharedScalarGateBuf = cfg.sharedExpertGated ? try buf(1) : nil

        func sharedProj(_ view: TensorView, rows: UInt32, cols: UInt32) -> SharedExpertProjection {
            SharedExpertProjection(weights: view.buffer,
                                 scales: view.buffer,
                                 biases: view.buffer,
                                 weightsOffset: Int(view.offset),
                                 scalesOffset: Int(view.scaleOffset),
                                 biasesOffset: Int(view.biasOffset),
                                 rows: rows,
                                 cols: cols)
        }
        var sharedViews: [LayerSharedExpertProjections] = []
        sharedViews.reserveCapacity(cfg.numLayers)
        for L in 0..<cfg.numLayers {
            let gate = try model.sharedExpertGate(layer: L)
            let up = try model.sharedExpertUp(layer: L)
            let down = try model.sharedExpertDown(layer: L)
            sharedViews.append(LayerSharedExpertProjections(
                gate: sharedProj(gate, rows: UInt32(F), cols: UInt32(D)),
                up: sharedProj(up, rows: UInt32(F), cols: UInt32(D)),
                down: sharedProj(down, rows: UInt32(D), cols: UInt32(F)),
                postF1: cfg.ffnSandwichNorms ? try model.postFFN1(layer: L) : nil,
                scalarGate: cfg.sharedExpertGated
                    ? try model.sharedExpertScalarGate(layer: L) : nil))
        }
        self.sharedExpertProjections = sharedViews

        func bf16OnesBuffer(count: Int, label: String) throws -> MTLBuffer {
            guard let buf = device.makeBuffer(length: count * MemoryLayout<UInt16>.size,
                                              options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            let dst = buf.contents().assumingMemoryBound(to: UInt16.self)
            for i in 0..<count { dst[i] = 0x3F80 }  // BF16 1.0
            buf.label = label
            return buf
        }

        if cfg.routerScaled {
            // Pre-fold 1/sqrt(D) into router.scale per layer. Each layer gets
            // its own BF16 [D] buffer — the kernel reads `effective_scale[i]`
            // and we pay for the multiply once per generation, not per token.
            var perLayer: [MTLBuffer] = []
            perLayer.reserveCapacity(cfg.numLayers)
            let invSqrtD = Float(1.0) / Float(D).squareRoot()
            let dInts = D
            for L in 0..<cfg.numLayers {
                let scaleView = try model.routerScale(layer: L)
                guard let buf = device.makeBuffer(length: dInts * MemoryLayout<UInt16>.size,
                                                  options: .storageModeShared) else {
                    throw ModelError.residentBufferWrapFailed
                }
                let src = scaleView.buffer.contents()
                    .advanced(by: Int(scaleView.offset))
                    .assumingMemoryBound(to: UInt16.self)
                let dst = buf.contents().assumingMemoryBound(to: UInt16.self)
                for i in 0..<dInts {
                    let v = Quantization.bf16ToFloat(src[i]) * invSqrtD
                    dst[i] = Quantization.bf16Bits(v)
                }
                buf.label = "effective_scale.L\(L)"
                perLayer.append(buf)
            }
            self.effectiveScaleBuffers = perLayer
            self.onesPerExpertScale = nil
        } else {
            // Plain linear router (Qwen): one shared BF16 ones buffer keeps
            // the router kernel's effective_scale multiply neutral, and a ones
            // per_expert_scale keeps the top-k weights untouched. (Softmax
            // over top-k then renormalize equals Qwen's softmax over all
            // experts then renormalize the selected top-k.)
            let ones = try bf16OnesBuffer(count: D, label: "effective_scale.ones")
            self.effectiveScaleBuffers = [MTLBuffer](repeating: ones,
                                                     count: cfg.numLayers)
            self.onesPerExpertScale = try bf16OnesBuffer(count: cfg.numExperts,
                                                         label: "per_expert_scale.ones")
        }

        // DEBUG: verify q_proj shape for gated-attention models
        if cfg.attnOutputGate {
            for L in 0..<cfg.numLayers {
                if cfg.fullAttentionLayerMask[L] == 1 {
                    let q = try model.qProj(layer: L)
                    let expectedRows = UInt32(2 * cfg.numHeads * cfg.fullHeadDim)
                    let actualRows = q.shape.0
                    print("[DEBUG-qproj] layer=\(L) family=\(cfg.family.rawValue) "
                          + "shape=(\(q.shape.0), \(q.shape.1), \(q.shape.2), \(q.shape.3)) "
                          + "dtype=\(q.dtype) expectedRows=\(expectedRows) actualRows=\(actualRows) "
                          + "length=\(q.length) scaleLen=\(q.scaleLength) biasLen=\(q.biasLength)")
                    break
                }
            }
        }
    }

    public func reset() {
        kv?.reset()
        gdnState?.reset()
        resetTransientState()
    }

    public var continuationPosition: Int {
        kv?.position ?? 0
    }

    public func prepareForContinuation(expectedPosition: Int) throws {
        guard let kv else {
            throw PrefillError.prefillCursorMismatch(
                "continuation requires an initialized KV cache")
        }
        guard expectedPosition > 0, kv.position == expectedPosition else {
            throw PrefillError.prefillCursorMismatch(
                "continuation expected KV position \(expectedPosition), current \(kv.position)")
        }
        resetTransientState()
    }

    private func resetTransientState() {
        prefillChunkState.reset()
        rdadviseSkipUntilPosition = -1
        rdadviseAdaptiveState.reset()
        rdadviseAdaptivePosition = -1
        rdadviseAdaptivePositionBytes = 0
    }

    public private(set) var totalIoNanos: UInt64 = 0
    public private(set) var totalCb1Nanos: UInt64 = 0
    public private(set) var totalCb2Nanos: UInt64 = 0
    public private(set) var totalHeadNanos: UInt64 = 0
    public private(set) var totalHeadFusedNanos: UInt64 = 0
    public private(set) var lastGreedyToken: UInt32 = 0
    public var usesFusedGreedyHead: Bool { useFusedGreedyHead }
    public private(set) var totalRDAdviseNanos: UInt64 = 0
    public private(set) var totalRDAdviseCalls: UInt64 = 0
    public private(set) var totalRDAdviseBytes: UInt64 = 0
    public private(set) var totalRDAdviseFailures: UInt64 = 0
    public private(set) var totalRDAdviseSkipped: UInt64 = 0

    private func recordRDAdvice(_ result: ExpertIOAdviceResult, wallNanos: UInt64) {
        totalRDAdviseNanos &+= wallNanos
        totalRDAdviseCalls &+= UInt64(result.calls)
        totalRDAdviseBytes &+= result.bytes
        totalRDAdviseFailures &+= UInt64(result.failed)
        totalRDAdviseSkipped &+= UInt64(result.skipped)
    }

    private func shouldSkipRDAdvice(position: Int,
                                    requestedMisses: Int,
                                    estimatedBytes: UInt64,
                                    canOverlapUsefulGPUWork: Bool) -> ExpertIOAdviceResult? {
        switch rdadvisePolicyMode {
        case .bounded:
            if position <= rdadviseSkipUntilPosition {
                return ExpertIOAdviceResult.skipped(requested: requestedMisses,
                                                    bytes: estimatedBytes)
            }
            if requestedMisses > Self.rdadviseBoundedMissCap {
                return ExpertIOAdviceResult.skipped(requested: requestedMisses,
                                                    bytes: estimatedBytes)
            }
            return nil
        case .adaptive:
            if position != rdadviseAdaptivePosition {
                rdadviseAdaptivePosition = position
                rdadviseAdaptivePositionBytes = 0
            }
            let cumulativeEstimatedBytes = rdadviseAdaptivePositionBytes &+ estimatedBytes
            let shouldSkip = rdadviseAdaptiveState.shouldSkip(
                position: position,
                requestedMisses: requestedMisses,
                estimatedBytes: cumulativeEstimatedBytes,
                canOverlapUsefulGPUWork: canOverlapUsefulGPUWork)
            rdadviseAdaptivePositionBytes = cumulativeEstimatedBytes
            guard shouldSkip else { return nil }
            return ExpertIOAdviceResult.skipped(requested: requestedMisses,
                                                bytes: estimatedBytes)
        case .default, .off:
            return nil
        }
    }

    private func updateRDAdvicePolicy(after result: ExpertIOAdviceResult,
                                      position: Int) {
        switch rdadvisePolicyMode {
        case .bounded:
            if result.maxCallNanos > Self.rdadviseBoundedMaxCallNanos {
                rdadviseSkipUntilPosition = max(rdadviseSkipUntilPosition, position + 1)
            }
        case .adaptive:
            rdadviseAdaptiveState.update(after: result, position: position)
        case .default, .off:
            break
        }
    }

    public func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
        try prefillChunkState.requireClean(operation: "produce")
        try await produceToken(token: token,
                               position: position,
                               into: logits,
                               emitHead: true,
                               outputMode: .greedyIfAvailable)
    }

    public func prefillChunked(tokens: ArraySlice<Int32>,
                               startPosition: Int,
                               outputMode: PrefillOutputMode,
                               config: PrefillRuntimeConfig,
                               into logits: MTLBuffer,
                               onProgress: (Int) -> Void) async throws -> PrefillResult {
        try prefillChunkState.requireClean(operation: "prefillChunked")
        guard config.mode == .chunked else {
            throw PrefillError.chunkedUnsupported(
                "prefillChunked requires PrefillRuntimeConfig.mode == .chunked")
        }
        guard startPosition >= 0 else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill startPosition must be non-negative")
        }
        let kvPosition = kv?.position ?? 0
        guard kvPosition == startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill cursor \(kvPosition) != startPosition \(startPosition)")
        }
        guard tokens.count <= maxContext - startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill range starting at \(startPosition) with \(tokens.count) tokens exceeds maxContext \(maxContext)")
        }
        guard !tokens.isEmpty else {
            return PrefillResult(newPosition: startPosition, seed: .logitsWritten)
        }

        let scratch = try ensurePrefillScratch(config: config)
        let spans = PrefillChunkPlanner.spans(tokenCount: tokens.count,
                                              startPosition: startPosition,
                                              config: config)
        for (spanIndex, span) in spans.enumerated() {
            let lower = tokens.index(tokens.startIndex, offsetBy: span.tokenOffset)
            let upper = tokens.index(lower, offsetBy: span.tokenCount)
            try await executePrefillChunk(
                tokens: tokens[lower..<upper],
                startPosition: span.startPosition,
                outputMode: outputMode,
                logits: logits,
                scratch: scratch,
                config: config,
                writeFinalHead: spanIndex == spans.count - 1)
            onProgress(span.completedCount)
        }
        if outputMode == .greedyIfAvailable, useFusedGreedyHead {
            return PrefillResult(newPosition: startPosition + tokens.count,
                                 seed: .greedyToken(lastGreedyToken))
        }
        return PrefillResult(newPosition: startPosition + tokens.count,
                             seed: .logitsWritten)
    }

    @discardableResult
    private func ensurePrefillScratch(config: PrefillRuntimeConfig) throws -> PrefillChunkScratchBuffers {
        let layout = PrefillChunkScratchLayout(config: cfg, runtime: config)
        if let scratch = prefillScratch, scratch.layout == layout {
            return scratch
        }
        let scratch = try PrefillChunkScratchBuffers.allocate(device: ctx.device, layout: layout)
        prefillScratch = scratch
        return scratch
    }

    private func executePrefillChunk(tokens: ArraySlice<Int32>,
                                     startPosition: Int,
                                     outputMode: PrefillOutputMode,
                                     logits: MTLBuffer,
                                     scratch: PrefillChunkScratchBuffers,
                                     config: PrefillRuntimeConfig,
                                     writeFinalHead: Bool) async throws {
        guard !tokens.isEmpty else { return }
        guard kv != nil else {
            throw PrefillError.chunkedUnsupported("chunked prefill attention requires FP16 KV")
        }
        let kvPosition = kv?.position ?? 0
        guard kvPosition == startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill cursor \(kvPosition) != startPosition \(startPosition)")
        }
        guard startPosition >= 0, startPosition + tokens.count <= maxContext else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill range [\(startPosition), \(startPosition + tokens.count)) exceeds maxContext \(maxContext)")
        }
        guard tokens.count <= scratch.layout.chunkTokens else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill token count \(tokens.count) exceeds scratch chunk size \(scratch.layout.chunkTokens)")
        }
        if let kv, kv.fp16RingEnabled, let ringLayer = (0..<cfg.numLayers).first(where: {
            kv.ringCapacity(layer: $0) > 0
        }) {
            let requiredCapacity = min(maxContext, cfg.slidingWindow + config.chunkTokens)
            let ringCapacity = kv.ringCapacity(layer: ringLayer)
            guard requiredCapacity <= ringCapacity else {
                throw PrefillError.chunkedUnsupported(
                    "FP16 KV ring capacity \(ringCapacity) cannot hold required capacity \(requiredCapacity) for maxContext \(maxContext), slidingWindow \(cfg.slidingWindow), and prefillChunkTokens \(config.chunkTokens)")
            }
        }

        struct LayerPrefillQKVViews {
            let inputNorm: TensorView
            let postAttention: TensorView
            let router: TensorView
            // Softmax-attention layers only (nil on linear-attention layers).
            let q: TensorView?
            let k: TensorView?
            let v: TensorView?
            let o: TensorView?
            let qNorm: TensorView?
            let kNorm: TensorView?
            // Gemma FFN sandwich only.
            let preFFN: TensorView?
            let preFFN2: TensorView?
            let postFFN2: TensorView?
            let postFFN: TensorView?
            let layerScalar: TensorView?
            let routerPerExpertScale: TensorView?
            // Gated-DeltaNet linear-attention layers only.
            let linQKV: TensorView?
            let linZ: TensorView?
            let linA: TensorView?
            let linB: TensorView?
            let linOut: TensorView?
            let linConv: TensorView?
            let linALog: TensorView?
            let linDtBias: TensorView?
            let linNorm: TensorView?
        }

        let layerViews = try (0..<cfg.numLayers).map { L in
            let isFull = cfg.fullAttentionLayerMask[L] == 1
            let isLinear = cfg.layerIsLinear(L)
            let sandwich = cfg.ffnSandwichNorms
            return LayerPrefillQKVViews(
                inputNorm: try model.inputNorm(layer: L),
                postAttention: try model.postAttnNorm(layer: L),
                router: try model.router(layer: L),
                q: isLinear ? nil : try model.qProj(layer: L),
                k: isLinear ? nil : try model.kProj(layer: L),
                v: isLinear ? nil
                    : ((isFull && cfg.attentionKEqV)
                        ? (try model.kProj(layer: L))
                        : (try model.vProj(layer: L))),
                o: isLinear ? nil : try model.oProj(layer: L),
                qNorm: isLinear ? nil : try model.qNorm(layer: L),
                kNorm: isLinear ? nil : try model.kNorm(layer: L),
                preFFN: sandwich ? try model.preFFN(layer: L) : nil,
                preFFN2: sandwich ? try model.preFFN2(layer: L) : nil,
                postFFN2: sandwich ? try model.postFFN2(layer: L) : nil,
                postFFN: sandwich ? try model.postFFN(layer: L) : nil,
                layerScalar: sandwich ? try model.layerScalar(layer: L) : nil,
                routerPerExpertScale: cfg.routerScaled
                    ? try model.routerPerExpertScale(layer: L) : nil,
                linQKV: isLinear ? try model.linearInProjQKV(layer: L) : nil,
                linZ: isLinear ? try model.linearInProjZ(layer: L) : nil,
                linA: isLinear ? try model.linearInProjA(layer: L) : nil,
                linB: isLinear ? try model.linearInProjB(layer: L) : nil,
                linOut: isLinear ? try model.linearOutProj(layer: L) : nil,
                linConv: isLinear ? try model.linearConv1d(layer: L) : nil,
                linALog: isLinear ? try model.linearALog(layer: L) : nil,
                linDtBias: isLinear ? try model.linearDtBias(layer: L) : nil,
                linNorm: isLinear ? try model.linearNorm(layer: L) : nil)
        }

        let tokenIDs = tokens.map { UInt32(bitPattern: $0) }
        guard let tokenBuffer = ctx.device.makeBuffer(bytes: tokenIDs,
                                                      length: tokenIDs.count * MemoryLayout<UInt32>.stride,
                                                      options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        let D = cfg.hiddenSize
        let eps: Float = 1e-6
        let embedOutScale = cfg.embeddingScaledBySqrtHidden
            ? Float(D).squareRoot()
            : 1.0
        let t = tokens.count
        let emb = model.embedding

        func encodeInt4Projection(commandBuffer: MTLCommandBuffer,
                                  family: PrefillProjectionFamily,
                                  weights: TensorView,
                                  x: MTLBuffer,
                                  y: MTLBuffer,
                                  rows: Int,
                                  columns: Int,
                                  tokenCount: Int,
                                  xStrideElements: Int,
                                  yStrideElements: Int) {
            if tokenCount >= 32,
               family == .q || family == .kv || family == .o,
               let candidate = prefillMPPAffineInt4 {
                let path = candidate.encode(
                    commandBuffer: commandBuffer,
                    weights: weights.buffer,
                    weightsOffset: Int(weights.offset),
                    scales: weights.buffer,
                    scalesOffset: Int(weights.scaleOffset),
                    biases: weights.buffer,
                    biasesOffset: Int(weights.biasOffset),
                    x: x,
                    y: y,
                    m: tokenCount,
                    n: rows,
                    k: columns)
                if path == .affineThreadgroupF16 {
                    return
                }
            }
            if PrefillProjectionDispatchPolicy.selectedDispatch(for: family,
                                                                chunkTokens: tokenCount) == .qmm {
                prefillQMM.encode(commandBuffer: commandBuffer,
                                  weights: weights.buffer,
                                  weightsOffset: Int(weights.offset),
                                  scales: weights.buffer,
                                  scalesOffset: Int(weights.scaleOffset),
                                  biases: weights.buffer,
                                  biasesOffset: Int(weights.biasOffset),
                                  x: x,
                                  y: y,
                                  t: tokenCount,
                                  n: rows,
                                  k: columns)
                return
            }
            for row in 0..<tokenCount {
                int4.encode(commandBuffer: commandBuffer,
                            weights: weights.buffer,
                            weightsOffset: Int(weights.offset),
                            scales: weights.buffer,
                            scalesOffset: Int(weights.scaleOffset),
                            biases: weights.buffer,
                            biasesOffset: Int(weights.biasOffset),
                            x: x,
                            xOffset: row * xStrideElements * MemoryLayout<Float16>.stride,
                            y: y,
                            yOffset: row * yStrideElements * MemoryLayout<Float16>.stride,
                            m: UInt32(rows),
                            n: UInt32(columns))
            }
        }

        func copyPrefillKV(commandBuffer: MTLCommandBuffer,
                           source: MTLBuffer,
                           destination: (buffer: MTLBuffer, offset: Int, stride: Int),
                           sourceTokenOffset: Int,
                           tokenCount: Int,
                           bytesPerToken: Int) throws {
            guard tokenCount > 0 else { return }
            guard let blit = commandBuffer.makeBlitCommandEncoder() else {
                throw ModelError.residentBufferWrapFailed
            }
            blit.copy(from: source,
                      sourceOffset: sourceTokenOffset * bytesPerToken,
                      to: destination.buffer,
                      destinationOffset: destination.offset,
                      size: tokenCount * bytesPerToken)
            blit.endEncoding()
        }

        func copyPrefillKVToCache(commandBuffer: MTLCommandBuffer,
                                  kv: KVCacheManager,
                                  layer: Int,
                                  startPosition: Int,
                                  tokenCount: Int,
                                  keySource: MTLBuffer,
                                  valueSource: MTLBuffer,
                                  bytesPerToken: Int) throws {
            let capacity = kv.capacity(layer: layer)
            let physicalStart = startPosition % capacity
            let firstSpan = min(tokenCount, capacity - physicalStart)
            let keyFirst = kv.kRange(layer: layer, start: startPosition, count: firstSpan)
            let valueFirst = kv.vRange(layer: layer, start: startPosition, count: firstSpan)
            try copyPrefillKV(commandBuffer: commandBuffer,
                              source: keySource,
                              destination: keyFirst,
                              sourceTokenOffset: 0,
                              tokenCount: firstSpan,
                              bytesPerToken: bytesPerToken)
            try copyPrefillKV(commandBuffer: commandBuffer,
                              source: valueSource,
                              destination: valueFirst,
                              sourceTokenOffset: 0,
                              tokenCount: firstSpan,
                              bytesPerToken: bytesPerToken)
            guard firstSpan < tokenCount else { return }

            let secondCount = tokenCount - firstSpan
            let secondStart = startPosition + firstSpan
            let keySecond = kv.kRange(layer: layer, start: secondStart, count: secondCount)
            let valueSecond = kv.vRange(layer: layer, start: secondStart, count: secondCount)
            try copyPrefillKV(commandBuffer: commandBuffer,
                              source: keySource,
                              destination: keySecond,
                              sourceTokenOffset: firstSpan,
                              tokenCount: secondCount,
                              bytesPerToken: bytesPerToken)
            try copyPrefillKV(commandBuffer: commandBuffer,
                              source: valueSource,
                              destination: valueSecond,
                              sourceTokenOffset: firstSpan,
                              tokenCount: secondCount,
                              bytesPerToken: bytesPerToken)
        }

        prefillChunkState.markDirty(startPosition: startPosition, tokenCount: tokens.count)

        guard var cb = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        prefillEmbed.encode(commandBuffer: cb,
                            table: emb.buffer,
                            tableOffset: Int(emb.offset),
                            scales: emb.buffer,
                            scalesOffset: Int(emb.scaleOffset),
                            biases: emb.buffer,
                            biasesOffset: Int(emb.biasOffset),
                            tokens: tokenBuffer,
                            out: scratch.hidden,
                            t: UInt32(t),
                            d: UInt32(D),
                            outScale: embedOutScale)

        for L in 0..<cfg.numLayers {
            model.beginOpeningRoutedExpertStreamer(layer: L)
            let views = layerViews[L]
            let isLinear = cfg.layerIsLinear(L)
            let isFull = cfg.fullAttentionLayerMask[L] == 1
            let headDim = isFull ? cfg.fullHeadDim : cfg.headDim
            let numKVHeads = isFull ? cfg.numFullKVHeads : cfg.numKVHeads
            let qDim = cfg.numHeads * headDim
            let kvDim = numKVHeads * headDim

            prefillRMS.encodeBF16W(commandBuffer: cb,
                                   x: scratch.hidden,
                                   weight: views.inputNorm.buffer,
                                   weightOffset: Int(views.inputNorm.offset),
                                   out: scratch.normed,
                                   t: UInt32(t),
                                   d: UInt32(D),
                                   eps: eps)
            if isLinear {
                // Gated-DeltaNet linear attention over the chunk: batched
                // projections, causal conv (+ tail carry), delta-rule
                // recurrence, gated norm, out_proj. No KV writes, no
                // attention, no blit.
                guard let gdn, let gdnState else {
                    preconditionFailure("linear-attention layer without GDN kernels")
                }
                let la = cfg.linearAttention
                encodeInt4Projection(commandBuffer: cb,
                                     family: .q,
                                     weights: views.linQKV!,
                                     x: scratch.normed,
                                     y: scratch.q,
                                     rows: la.qkvDim,
                                     columns: D,
                                     tokenCount: t,
                                     xStrideElements: D,
                                     yStrideElements: la.qkvDim)
                encodeInt4Projection(commandBuffer: cb,
                                     family: .kv,
                                     weights: views.linZ!,
                                     x: scratch.normed,
                                     y: scratch.gdnZ,
                                     rows: la.valueDim,
                                     columns: D,
                                     tokenCount: t,
                                     xStrideElements: D,
                                     yStrideElements: la.valueDim)
                encodeInt4Projection(commandBuffer: cb,
                                     family: .kv,
                                     weights: views.linA!,
                                     x: scratch.normed,
                                     y: scratch.gdnA,
                                     rows: la.numVHeads,
                                     columns: D,
                                     tokenCount: t,
                                     xStrideElements: D,
                                     yStrideElements: la.numVHeads)
                encodeInt4Projection(commandBuffer: cb,
                                     family: .kv,
                                     weights: views.linB!,
                                     x: scratch.normed,
                                     y: scratch.gdnB,
                                     rows: la.numVHeads,
                                     columns: D,
                                     tokenCount: t,
                                     xStrideElements: D,
                                     yStrideElements: la.numVHeads)
                let convW = views.linConv!
                let tail = gdnState.convTailBuffer(layer: L)
                gdn.encodeConvPrefill(commandBuffer: cb,
                                      tail: tail,
                                      qkvRows: scratch.q,
                                      convWeight: convW.buffer,
                                      convWeightOffset: Int(convW.offset),
                                      out: scratch.gdnConvOut,
                                      rows: t)
                gdn.encodeConvTailUpdate(commandBuffer: cb,
                                         tail: tail,
                                         qkvRows: scratch.q,
                                         rows: t)
                gdn.encodeQKNorm(commandBuffer: cb,
                                 convOut: scratch.gdnConvOut,
                                 rows: t)
                let aLog = views.linALog!
                let dtBias = views.linDtBias!
                gdn.encodeDeltaStepPrefill(commandBuffer: cb,
                                           convOut: scratch.gdnConvOut,
                                           aProj: scratch.gdnA,
                                           bProj: scratch.gdnB,
                                           aLog: aLog.buffer,
                                           aLogOffset: Int(aLog.offset),
                                           dtBias: dtBias.buffer,
                                           dtBiasOffset: Int(dtBias.offset),
                                           state: gdnState.stateBuffer(layer: L),
                                           y: scratch.gdnY,
                                           rows: t)
                let gatedNormW = views.linNorm!
                gdn.encodeGatedNorm(commandBuffer: cb,
                                    y: scratch.gdnY,
                                    z: scratch.gdnZ,
                                    weight: gatedNormW.buffer,
                                    weightOffset: Int(gatedNormW.offset),
                                    out: scratch.attentionOutput,
                                    rows: t)
                encodeInt4Projection(commandBuffer: cb,
                                     family: .o,
                                     weights: views.linOut!,
                                     x: scratch.attentionOutput,
                                     y: scratch.h1,
                                     rows: D,
                                     columns: la.valueDim,
                                     tokenCount: t,
                                     xStrideElements: la.valueDim,
                                     yStrideElements: D)
            } else {
                let qProjRows = cfg.attnOutputGate ? 2 * qDim : qDim
                encodeInt4Projection(commandBuffer: cb,
                                     family: .q,
                                     weights: views.q!,
                                     x: scratch.normed,
                                     y: scratch.q,
                                     rows: qProjRows,
                                     columns: D,
                                     tokenCount: t,
                                     xStrideElements: D,
                                     yStrideElements: qProjRows)
                encodeInt4Projection(commandBuffer: cb,
                                     family: .kv,
                                     weights: views.k!,
                                     x: scratch.normed,
                                     y: scratch.kStage,
                                     rows: kvDim,
                                     columns: D,
                                     tokenCount: t,
                                     xStrideElements: D,
                                     yStrideElements: kvDim)
                encodeInt4Projection(commandBuffer: cb,
                                     family: .kv,
                                     weights: views.v!,
                                     x: scratch.normed,
                                     y: scratch.vStage,
                                     rows: kvDim,
                                     columns: D,
                                     tokenCount: t,
                                     xStrideElements: D,
                                     yStrideElements: kvDim)

                // The attention input Q: the packed q_proj output is split
                // into per-head query/gate halves for gated architectures.
                let attnQ: MTLBuffer
                if cfg.attnOutputGate {
                    elementwise!.encodeSplitQGate(commandBuffer: cb,
                                                  packed: scratch.q,
                                                  q: scratch.attnQ,
                                                  gate: scratch.attnGate,
                                                  heads: cfg.numHeads,
                                                  dim: headDim,
                                                  rows: t)
                    attnQ = scratch.attnQ
                } else {
                    attnQ = scratch.q
                }

                if cfg.ropeNeoxSubdim {
                    let rotaryDim = UInt32(Double(headDim) * cfg.partialRotaryFactor)
                    prefillQKVEpilogue.encodeNeoxSubdimNoVNorm(
                        commandBuffer: cb,
                        q: attnQ,
                        k: scratch.kStage,
                        qWeight: views.qNorm!.buffer,
                        qWeightOffset: Int(views.qNorm!.offset),
                        kWeight: views.kNorm!.buffer,
                        kWeightOffset: Int(views.kNorm!.offset),
                        startPosition: UInt32(startPosition),
                        queryCount: UInt32(t),
                        headDim: UInt32(headDim),
                        numQHeads: UInt32(cfg.numHeads),
                        numKVHeads: UInt32(numKVHeads),
                        qTokenStrideElements: UInt32(qDim),
                        kvTokenStrideElements: UInt32(kvDim),
                        theta: Float(cfg.fullRopeTheta),
                        rotaryDim: rotaryDim,
                        eps: eps)
                } else {
                    let rotatedPairs = isFull
                        ? UInt32(Double(cfg.fullHeadDim) * cfg.partialRotaryFactor / 2.0)
                        : UInt32(headDim / 2)
                    prefillQKVEpilogue.encode(commandBuffer: cb,
                                               q: attnQ,
                                               k: scratch.kStage,
                                               v: scratch.vStage,
                                               qWeight: views.qNorm!.buffer,
                                               qWeightOffset: Int(views.qNorm!.offset),
                                               kWeight: views.kNorm!.buffer,
                                               kWeightOffset: Int(views.kNorm!.offset),
                                               startPosition: UInt32(startPosition),
                                               queryCount: UInt32(t),
                                               headDim: UInt32(headDim),
                                               numQHeads: UInt32(cfg.numHeads),
                                               numKVHeads: UInt32(numKVHeads),
                                               qTokenStrideElements: UInt32(qDim),
                                               kvTokenStrideElements: UInt32(kvDim),
                                               theta: isFull ? Float(cfg.fullRopeTheta) : Float(cfg.ropeTheta),
                                               rotatedPairs: rotatedPairs,
                                               eps: eps)
                }

                if let kv {
                    let bytes = t * kvDim * MemoryLayout<Float16>.stride
                    try copyPrefillKVToCache(commandBuffer: cb,
                                             kv: kv,
                                             layer: L,
                                             startPosition: startPosition,
                                             tokenCount: t,
                                             keySource: scratch.kStage,
                                             valueSource: scratch.vStage,
                                             bytesPerToken: bytes / t)
                }
                let params = PrefillAttentionParams(
                        startPosition: UInt32(startPosition),
                        queryCount: UInt32(t),
                        headDim: UInt32(headDim),
                        numQHeads: UInt32(cfg.numHeads),
                        numKVHeads: UInt32(numKVHeads),
                        kvValidCount: UInt32(startPosition + t),
                        slidingWindow: isFull ? UInt32(startPosition + t) : UInt32(cfg.slidingWindow),
                        kvTokenStrideElements: UInt32(kvDim),
                        qTokenStrideElements: UInt32(qDim),
                        oTokenStrideElements: UInt32(qDim),
                        scale: Float(cfg.attentionScale))
                if let kv {
                        let keyBuffer = kv.keyBuffer(layer: L, validTokenCount: startPosition + t)
                        let valueBuffer = kv.valueBuffer(layer: L, validTokenCount: startPosition + t)
                        let ringCapacity = kv.ringCapacity(layer: L)
                        let activeRingCapacity = ringCapacity > 0 && startPosition + t > ringCapacity
                            ? UInt32(ringCapacity)
                            : 0
                        prefillAttention.encodeCausal(commandBuffer: cb,
                                                      q: attnQ,
                                                      k: keyBuffer,
                                                      v: valueBuffer,
                                                      out: scratch.attentionOutput,
                                                      params: params,
                                                      kvRingCapacity: activeRingCapacity,
                                                      path: prefillAttentionPath)
                } else {
                    throw PrefillError.chunkedUnsupported(
                        "chunked prefill attention requires FP16 KV")
                }
                if cfg.attnOutputGate {
                    elementwise!.encodeSigmoidGateMul(commandBuffer: cb,
                                                      out: scratch.attentionOutput,
                                                      gate: scratch.attnGate,
                                                      count: t * qDim)
                }
                encodeInt4Projection(commandBuffer: cb,
                                         family: .o,
                                         weights: views.o!,
                                         x: scratch.attentionOutput,
                                         y: scratch.h1,
                                         rows: D,
                                         columns: qDim,
                                         tokenCount: t,
                                         xStrideElements: qDim,
                                         yStrideElements: D)
            }
            if cfg.ffnSandwichNorms {
                prefillPostAttention.encode(commandBuffer: cb,
                                                hidden: scratch.hidden,
                                                attn: scratch.h1,
                                                denseX: scratch.denseX,
                                                routedX: scratch.routedX,
                                                routerX: scratch.routerX,
                                                postAttentionWeight: views.postAttention.buffer,
                                                postAttentionWeightOffset: Int(views.postAttention.offset),
                                                preFFNWeight: views.preFFN!.buffer,
                                                preFFNWeightOffset: Int(views.preFFN!.offset),
                                                preFFN2Weight: views.preFFN2!.buffer,
                                                preFFN2WeightOffset: Int(views.preFFN2!.offset),
                                                queryCount: UInt32(t),
                                                d: UInt32(D),
                                                hiddenStrideElements: UInt32(D),
                                                attnStrideElements: UInt32(D),
                                                denseStrideElements: UInt32(D),
                                                routedStrideElements: UInt32(D),
                                                routerStrideElements: UInt32(D),
                                                eps: eps)
            } else {
                // Plain pre-norm residual block: hidden += attention branch,
                // then one post-attention norm feeds router, shared expert,
                // and routed phase 1 (routedX doubles as moeX).
                elementwise!.encodeResidualAdd(commandBuffer: cb,
                                               hidden: scratch.hidden,
                                               delta: scratch.h1,
                                               count: t * D)
                prefillRMS.encodeBF16W(commandBuffer: cb,
                                       x: scratch.hidden,
                                       weight: views.postAttention.buffer,
                                       weightOffset: Int(views.postAttention.offset),
                                       out: scratch.routedX,
                                       t: UInt32(t),
                                       d: UInt32(D),
                                       eps: eps)
            }
            let perExpertScale: (buffer: MTLBuffer, offset: Int)
            if cfg.routerScaled {
                let view = views.routerPerExpertScale!
                perExpertScale = (view.buffer, Int(view.offset))
            } else {
                perExpertScale = (onesPerExpertScale!, 0)
            }
            prefillRouter.encodeGemma4Block(
                        commandBuffer: cb,
                        weights: views.router.buffer,
                        weightsOffset: Int(views.router.offset),
                        scales: views.router.buffer,
                        scalesOffset: Int(views.router.scaleOffset),
                        biases: views.router.buffer,
                        biasesOffset: Int(views.router.biasOffset),
                        hidden: cfg.ffnSandwichNorms ? scratch.routerX : scratch.routedX,
                        effectiveScale: effectiveScaleBuffers[L],
                        perExpertScale: perExpertScale.buffer,
                        perExpertScaleOffset: perExpertScale.offset,
                        outIndices: scratch.routeIDs,
                        outWeights: scratch.routeWeights,
                        queryCount: UInt32(t),
                        numExperts: UInt32(cfg.numExperts),
                        d: UInt32(D),
                        topK: UInt32(cfg.topKExperts),
                        hiddenStrideElements: UInt32(D),
                        weightBits: model.routerWeightBits)

                    cb.commit()
                    waitForCompletion(cb)
                    if let error = cb.error {
                        throw error
                    }

                    let routeCount = t * cfg.topKExperts
                    let idPtr = scratch.routeIDs.contents()
                        .bindMemory(to: UInt32.self, capacity: routeCount)
                    let weightPtr = scratch.routeWeights.contents()
                        .bindMemory(to: Float16.self, capacity: routeCount)
                    var routeIDs = [UInt32]()
                    routeIDs.reserveCapacity(routeCount)
                    var routeWeights = [Float16]()
                    routeWeights.reserveCapacity(routeCount)
                    for i in 0..<routeCount {
                        routeIDs.append(min(idPtr[i], UInt32(cfg.numExperts - 1)))
                        routeWeights.append(weightPtr[i])
                    }
                    let pairs = PrefillRouter.makeTokenExpertPairs(indices: routeIDs,
                                                                   weights: routeWeights,
                                                                   queryCount: t,
                                                                   topK: cfg.topKExperts)
                    let schedulerConfig = Self.prefillRoutedTileSchedulerConfig
                    let routeTileExpertCount: Int
                    if let slotCount = model.routedExpertCacheSlotCount(layer: L) {
                        guard schedulerConfig.fitsSlotBudget(slotCount: slotCount) else {
                            throw PrefillError.chunkedUnsupported(
                                "prefill routed tile depth \(schedulerConfig.maxPendingDepth) with \(schedulerConfig.tileExperts) experts/tile needs \((schedulerConfig.maxPendingDepth + 1) * schedulerConfig.tileExperts) slots, has \(slotCount)")
                        }
                        routeTileExpertCount = min(schedulerConfig.tileExperts, slotCount)
                    } else {
                        routeTileExpertCount = schedulerConfig.tileExperts
                    }
                    let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
                        pairs,
                        queryCount: t,
                        topK: cfg.topKExperts,
                        numExperts: cfg.numExperts,
                        tileExpertCount: routeTileExpertCount,
                        expertSortKeys: model.routedExpertPhysicalOffsets(layer: L))

                    guard let sharedCB = ctx.queue.makeCommandBuffer() else {
                        throw ModelError.residentBufferWrapFailed
                    }
                    let sharedProj = sharedExpertProjections[L]
                    try prefillSharedExpert.encodeBlock(commandBuffer: sharedCB,
                                                        x: cfg.ffnSandwichNorms
                                                            ? scratch.denseX
                                                            : scratch.routedX,
                                                        y: scratch.h1,
                                                        gate: sharedProj.gate,
                                                        up: sharedProj.up,
                                                        down: sharedProj.down,
                                                        scratchGate: scratch.sharedGateScratch,
                                                        scratchUp: scratch.sharedUpScratch,
                                                        scratchAct: scratch.sharedActScratch,
                                                        queryCount: t,
                                                        d: D,
                                                        intermediate: cfg.intermediateSize,
                                                        xStrideElements: D,
                                                        yStrideElements: D)
                    if cfg.ffnSandwichNorms {
                        let postF1 = sharedProj.postF1!
                        prefillRMS.encodeBF16W(commandBuffer: sharedCB,
                                               x: scratch.h1,
                                               weight: postF1.buffer,
                                               weightOffset: Int(postF1.offset),
                                               out: scratch.h1,
                                               t: UInt32(t),
                                               d: UInt32(D),
                                               eps: eps)
                    } else if cfg.sharedExpertGated {
                        // out = sigmoid(shared_expert_gate(moeX)) * shared_mlp(moeX),
                        // per chunk row.
                        let gateView = sharedProj.scalarGate!
                        let halfBytes = MemoryLayout<Float16>.stride
                        for row in 0..<t {
                            int8ScalarGate!.encode(
                                commandBuffer: sharedCB,
                                weights: gateView.buffer,
                                weightsOffset: Int(gateView.offset),
                                scales: gateView.buffer,
                                scalesOffset: Int(gateView.scaleOffset),
                                biases: gateView.buffer,
                                biasesOffset: Int(gateView.biasOffset),
                                x: scratch.routedX,
                                xOffset: row * D * halfBytes,
                                y: scratch.sharedScalarGate,
                                yOffset: row * halfBytes,
                                m: 1, n: UInt32(D))
                        }
                        for row in 0..<t {
                            elementwise!.encodeSigmoidScalarMul(
                                commandBuffer: sharedCB,
                                y: scratch.h1,
                                yOffset: row * D * halfBytes,
                                gate: scratch.sharedScalarGate,
                                gateOffset: row * halfBytes,
                                count: D)
                        }
                    }
                    sharedCB.commit()
                    waitForCompletion(sharedCB)
                    if let error = sharedCB.error {
                        throw error
                    }

                    let metadata = try prefillGroupedMoE.makeStreamedMetadataBuffers(
                        device: ctx.device,
                        routes: routes)
                    let routedOffsets = model.routedExpertOffsets(layer: L)
                    struct PendingPrefillTile {
                        let tileIndex: Int
                        let commandBuffer: MTLCommandBuffer
                        let fetch: PrefillStreamedTileFetchResult
                        let argumentBuffer: PrefillStreamedTileArgumentBuffer
                    }
                    var pendingTiles: [PendingPrefillTile] = []
                    var tileLifetime = PrefillStreamedTileSlotLifetime()
                    func drainOldestPendingTile() throws {
                        guard !pendingTiles.isEmpty else { return }
                        let pending = pendingTiles.removeFirst()
                        withExtendedLifetime((pending.fetch, pending.argumentBuffer)) {
                            waitForCompletion(pending.commandBuffer)
                        }
                        if let error = pending.commandBuffer.error {
                            throw error
                        }
                        if !pending.fetch.plannedMissSlots.isEmpty {
                            try tileLifetime.complete(tileIndex: pending.tileIndex)
                        }
                    }

                    let routedTileScheduler = PrefillRoutedTileScheduler(config: schedulerConfig)
                    for (tileIndex, tile) in routes.tiles.enumerated() {
                        let expertIDs = try PrefillStreamedTileBinding.expertIDs(
                            forTile: tileIndex,
                            routes: routes)
                        var plannedFetch: RoutedExpertFetchPlan?
                        if !pendingTiles.isEmpty {
                            let pendingAssignedSlots = pendingTiles.flatMap(\.fetch.plannedAssignedSlots)
                            if !pendingAssignedSlots.isEmpty {
                                let pendingSlots = Set(pendingAssignedSlots)
                                let plan = try model.planRoutedExpertsIfPossible(
                                    layer: L,
                                    experts: expertIDs,
                                    avoidingSlots: pendingSlots)
                                let decision = routedTileScheduler.decide(
                                    PrefillRoutedTileSchedulerInput(
                                        hasPendingTile: true,
                                        pendingDepth: pendingTiles.count,
                                        pendingAssignedSlots: pendingAssignedSlots,
                                        avoidingSlotPlanAvailable: plan != nil))
                                switch decision {
                                case .prefetchNext:
                                    guard let plan else {
                                        throw ModelError.indexCorrupt(
                                            detail: "routed tile scheduler requested missing plan")
                                    }
                                    plannedFetch = plan
                                case .drainBeforeIssue:
                                    try drainOldestPendingTile()
                                case .issueWithoutPending:
                                    throw ModelError.indexCorrupt(
                                        detail: "routed tile scheduler ignored pending tile")
                                }
                            } else {
                                let decision = routedTileScheduler.decide(
                                    PrefillRoutedTileSchedulerInput(
                                        hasPendingTile: true,
                                        pendingDepth: pendingTiles.count,
                                        pendingAssignedSlots: [],
                                        avoidingSlotPlanAvailable: false))
                                switch decision {
                                case .drainBeforeIssue:
                                    try drainOldestPendingTile()
                                case .issueWithoutPending, .prefetchNext:
                                    throw ModelError.indexCorrupt(
                                        detail: "routed tile scheduler failed to drain empty-slot pending tile")
                                }
                            }
                        } else {
                            let decision = routedTileScheduler.decide(
                                PrefillRoutedTileSchedulerInput(
                                    hasPendingTile: false,
                                    pendingAssignedSlots: [],
                                    avoidingSlotPlanAvailable: false))
                            switch decision {
                            case .issueWithoutPending:
                                break
                            case .prefetchNext, .drainBeforeIssue:
                                throw ModelError.indexCorrupt(
                                    detail: "routed tile scheduler requested pending action without pending tile")
                            }
                        }
                        let fetch = try await PrefillStreamedTileBinding.fetchBindingForTile(
                            model: model,
                            layer: L,
                            tileIndex: tileIndex,
                            routes: routes,
                            plannedFetch: plannedFetch,
                            avoidingSlots: Set(pendingTiles.flatMap(\.fetch.plannedAssignedSlots)))
                        try fetch.binding.validateCoversPairs(routes.sortedPairs,
                                                              pairStart: Int(tile.pairStart),
                                                              pairCount: Int(tile.pairCount))
                        if !fetch.plannedMissSlots.isEmpty {
                            try tileLifetime.begin(tileIndex: tileIndex,
                                                   plannedSlots: fetch.plannedMissSlots)
                        }
                        let argumentBuffer = try prefillGroupedMoE.makeStreamedArgumentBuffer(
                            device: ctx.device,
                            binding: fetch.binding)
                        let streamedParams = PrefillGroupedRoutedMoEStreamedParams(
                            pairStart: tile.pairStart,
                            pairCount: tile.pairCount,
                            d: UInt32(D),
                            routedIntermediate: UInt32(cfg.moeIntermediateSize),
                            topK: UInt32(cfg.topKExperts),
                            hiddenStrideElements: UInt32(D),
                            binding: fetch.binding,
                            offsets: routedOffsets)
                        guard let tileCB = ctx.queue.makeCommandBuffer() else {
                            throw ModelError.residentBufferWrapFailed
                        }
                        _ = prefillGroupedMoE.encodeStreamedBatched(
                            commandBuffer: tileCB,
                            hidden: scratch.routedX,
                            sortedPairs: metadata.sortedPairs,
                            routePartials: scratch.routePartials,
                            gateUpActScratch: scratch.routedGateUpActScratch,
                            downScratch: scratch.routedDownScratch,
                            argumentBuffer: argumentBuffer,
                            binding: fetch.binding,
                            params: streamedParams,
                            pairMicrobatchRows: scratch.layout.routedPairMicrobatchRows)
                        tileCB.commit()
                        pendingTiles.append(PendingPrefillTile(tileIndex: tileIndex,
                                                               commandBuffer: tileCB,
                                                               fetch: fetch,
                                                               argumentBuffer: argumentBuffer))
                        while pendingTiles.count > schedulerConfig.maxPendingDepth {
                            try drainOldestPendingTile()
                        }
                    }
                    while !pendingTiles.isEmpty {
                        try drainOldestPendingTile()
                    }
                    guard let tailCB = ctx.queue.makeCommandBuffer() else {
                        throw ModelError.residentBufferWrapFailed
                    }
                    prefillMoE.encodeReduceTokenMajor(commandBuffer: tailCB,
                                                      routePartials: scratch.routePartials,
                                                      routeWeights: scratch.routeWeights,
                                                      h2: scratch.h2,
                                                      queryCount: UInt32(t),
                                                      topK: UInt32(cfg.topKExperts),
                                                      d: UInt32(D))
                    if cfg.ffnSandwichNorms {
                        let layerScalarView = views.layerScalar!
                        let scalarBits = layerScalarView.buffer.contents()
                            .advanced(by: Int(layerScalarView.offset))
                            .assumingMemoryBound(to: UInt16.self)[0]
                        prefillLayerTail.encode(commandBuffer: tailCB,
                                                h2: scratch.h2,
                                                h1: scratch.h1,
                                                hidden: scratch.hidden,
                                                postFFN2Weight: views.postFFN2!.buffer,
                                                postFFN2WeightOffset: Int(views.postFFN2!.offset),
                                                postFFNWeight: views.postFFN!.buffer,
                                                postFFNWeightOffset: Int(views.postFFN!.offset),
                                                queryCount: UInt32(t),
                                                d: UInt32(D),
                                                h2StrideElements: UInt32(D),
                                                h1StrideElements: UInt32(D),
                                                hiddenStrideElements: UInt32(D),
                                                eps: eps,
                                                layerScalar: Quantization.bf16ToFloat(scalarBits))
                    } else {
                        // Plain pre-norm tail: hidden += gated shared branch
                        // + routed branch.
                        elementwise!.encodeResidualAdd(commandBuffer: tailCB,
                                                       hidden: scratch.hidden,
                                                       delta: scratch.h1,
                                                       count: t * D)
                        elementwise!.encodeResidualAdd(commandBuffer: tailCB,
                                                       hidden: scratch.hidden,
                                                       delta: scratch.h2,
                                                       count: t * D)
                    }
                    tailCB.commit()
                    withExtendedLifetime(metadata) {
                        waitForCompletion(tailCB)
                    }
                    if let error = tailCB.error {
                        throw error
                    }
                    if debugRmsDump {
                        // `scratch.hidden` is a private GPU buffer: blit the
                        // chunk rows out to a shared scratch buffer before the
                        // CPU readback in `dumpHiddenRMS`.
                        let rmsBytes = t * D * MemoryLayout<Float16>.stride
                        guard let rmsDst = ctx.device.makeBuffer(
                            length: rmsBytes, options: .storageModeShared) else {
                            throw ModelError.residentBufferWrapFailed
                        }
                        guard let rmsCB = ctx.queue.makeCommandBuffer(),
                              let rmsBlit = rmsCB.makeBlitCommandEncoder() else {
                            throw ModelError.residentBufferWrapFailed
                        }
                        rmsBlit.copy(from: scratch.hidden,
                                     sourceOffset: 0,
                                     to: rmsDst,
                                     destinationOffset: 0,
                                     size: rmsBytes)
                        rmsBlit.endEncoding()
                        rmsCB.commit()
                        waitForCompletion(rmsCB)
                        dumpHiddenRMS(label: "prefill", layer: L,
                                      buffer: rmsDst, count: t * D)
                        if cfg.layerIsLinear(L) {
                            dumpGDNProbe(label: "prefill", layer: L,
                                         includeShared: false)
                            let la = cfg.linearAttention
                            // `scratch.gdnZ` / `scratch.attentionOutput` are
                            // private GPU buffers: blit before CPU readback.
                            let gdnBytes = t * la.valueDim * MemoryLayout<Float16>.stride
                            if let dst = ctx.device.makeBuffer(
                                length: gdnBytes, options: .storageModeShared),
                               let cbuf = ctx.queue.makeCommandBuffer(),
                               let blit = cbuf.makeBlitCommandEncoder() {
                                blit.copy(from: scratch.gdnZ, sourceOffset: 0,
                                          to: dst, destinationOffset: 0, size: gdnBytes)
                                blit.endEncoding()
                                cbuf.commit()
                                waitForCompletion(cbuf)
                                dumpGDNBuffer(label: "prefill", layer: L, name: "z",
                                              buffer: dst, count: t * la.valueDim)
                            }
                            if let dst = ctx.device.makeBuffer(
                                length: gdnBytes, options: .storageModeShared),
                               let cbuf = ctx.queue.makeCommandBuffer(),
                               let blit = cbuf.makeBlitCommandEncoder() {
                                blit.copy(from: scratch.attentionOutput, sourceOffset: 0,
                                          to: dst, destinationOffset: 0, size: gdnBytes)
                                blit.endEncoding()
                                cbuf.commit()
                                waitForCompletion(cbuf)
                                dumpGDNBuffer(label: "prefill", layer: L, name: "gdnOut",
                                              buffer: dst, count: t * la.valueDim)
                            }
                        }
                    }
                    if L + 1 < cfg.numLayers {
                        guard let nextCB = ctx.queue.makeCommandBuffer() else {
                            throw ModelError.residentBufferWrapFailed
                        }
                        cb = nextCB
                    }
                    continue
        }

        if writeFinalHead {
            let finalNorm = model.finalNorm
            let lm = model.lmHead
            guard let finalCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            if outputMode == .greedyIfAvailable, useFusedGreedyHead {
                fusionHead.encodeGreedyDecode(
                    commandBuffer: finalCB,
                    hidden: scratch.hidden,
                    hiddenOffset: (t - 1) * D * MemoryLayout<Float16>.stride,
                    normWeight: finalNorm.buffer,
                    normOffset: Int(finalNorm.offset),
                    weights: lm.buffer,
                    weightsOffset: Int(lm.offset),
                    scales: lm.buffer,
                    scalesOffset: Int(lm.scaleOffset),
                    biases: lm.buffer,
                    biasesOffset: Int(lm.biasOffset),
                    outToken: greedyTokenBuf,
                    d: UInt32(D),
                    vocab: UInt32(cfg.vocabSize),
                    rmsEps: eps)
            } else {
                prefillFinalRowHead.encodeLogits(commandBuffer: finalCB,
                                                 hiddenBlock: scratch.hidden,
                                                 row: t - 1,
                                                 rowStrideElements: D,
                                                 normWeight: finalNorm.buffer,
                                                 normWeightOffset: Int(finalNorm.offset),
                                                 weights: lm.buffer,
                                                 weightsOffset: Int(lm.offset),
                                                 scales: lm.buffer,
                                                 scalesOffset: Int(lm.scaleOffset),
                                                 biases: lm.buffer,
                                                 biasesOffset: Int(lm.biasOffset),
                                                 logits: logits,
                                                 d: UInt32(D),
                                                 vocab: UInt32(cfg.vocabSize),
                                                 rmsEps: eps)
            }
            finalCB.commit()
            waitForCompletion(finalCB)
            if let error = finalCB.error {
                throw error
            }
            if outputMode == .greedyIfAvailable, useFusedGreedyHead {
                lastGreedyToken = greedyTokenBuf.contents().load(as: UInt32.self)
            }
        }

        kv?.advance(by: tokens.count)
        prefillChunkState.markCommitted()
    }

    private func produceToken(token: Int32,
                              position: Int,
                              into logits: MTLBuffer,
                              emitHead: Bool,
                              outputMode: PrefillOutputMode) async throws {
        let kvPosition = kv?.position ?? 0
        guard kvPosition == position else {
            throw PrefillError.prefillCursorMismatch(
                "produce cursor \(kvPosition) != position \(position)")
        }
        guard position < maxContext else {
            throw PrefillError.prefillCursorMismatch(
                "produce position \(position) exceeds maxContext \(maxContext)")
        }
        let D    = UInt32(cfg.hiddenSize)
        let FmoE = UInt32(cfg.moeIntermediateSize)
        let eps: Float = 1e-6
           // DEBUG: one-time config dump at position 0
        if position == 0 {
            print("[DEBUG-decode] START: family=\(cfg.family.rawValue) "
                       + "hidden=\(cfg.hiddenSize) numHeads=\(cfg.numHeads) "
                       + "numKVHeads=\(cfg.numKVHeads) numFullKVHeads=\(cfg.numFullKVHeads) "
                       + "headDim=\(cfg.headDim) fullHeadDim=\(cfg.fullHeadDim) "
                       + "attnOutputGate=\(cfg.attnOutputGate) attnScale=\(cfg.attentionScale) "
                       + "ropeNeox=\(cfg.ropeNeoxSubdim) partialRotary=\(cfg.partialRotaryFactor) "
                       + "layers=\(cfg.numLayers) experts=\(cfg.numExperts) topK=\(cfg.topKExperts)")
            var maskDesc = ""
            for i in 0..<cfg.numLayers {
                let ch: Character
                switch cfg.fullAttentionLayerMask[i] {
                case 1: ch = "F"
                case 2: ch = "L"
                default: ch = "S"
                   }
                maskDesc += "\(i):\(ch) "
               }
            print("[DEBUG-decode] layerMask: \(maskDesc)")
           }
        let embedOutScale = cfg.embeddingScaledBySqrtHidden
            ? Float(cfg.hiddenSize).squareRoot()
            : 1.0
        struct PendingRoutedCommand {
            let cb: MTLCommandBuffer
            let sharedCB: MTLCommandBuffer?
            let phase1HitCB: MTLCommandBuffer?
            let encodeAndCommitNanos: UInt64
        }
        var pendingRoutedCommand: PendingRoutedCommand?

        func finishPendingRoutedCommand(_ pending: PendingRoutedCommand,
                                        waitIfNeeded: Bool) {
            if waitIfNeeded {
                func wait(_ cb: MTLCommandBuffer) {
                    waitForCompletion(cb)
                }
                if let sharedCB = pending.sharedCB {
                    wait(sharedCB)
                }
                if let phase1HitCB = pending.phase1HitCB {
                    wait(phase1HitCB)
                }
                wait(pending.cb)
            } else if let err = pending.cb.error {
                print("CB error: \(err)")
            }
            if let sharedCB = pending.sharedCB {
                if let err = sharedCB.error {
                    print("CB error: \(err)")
                }
            }
            if let phase1HitCB = pending.phase1HitCB,
               let err = phase1HitCB.error {
                print("CB error: \(err)")
            }
            totalCb2Nanos &+= pending.encodeAndCommitNanos
        }

        func writeActiveSlots(_ slots: [UInt32], into buffer: MTLBuffer) {
            let ptr = buffer.contents().assumingMemoryBound(to: UInt32.self)
            for i in 0..<slots.count { ptr[i] = slots[i] }
        }

        // Embed lookup + sqrt(H) fused.
        let emb = model.embedding
        do {
            runSync { cb in
                embedInt4.encode(commandBuffer: cb,
                                 table:  emb.buffer, tableOffset:  Int(emb.offset),
                                 scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                                 biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                                 out: hidden,
                                 tokenId: UInt32(bitPattern: token),
                                 d: D,
                                 outScale: embedOutScale)
            }
        }

        for L in 0..<cfg.numLayers {
            let isLinear = cfg.layerIsLinear(L)
            let isFull = cfg.fullAttentionLayerMask[L] == 1
            let headDimL = isFull ? cfg.fullHeadDim : cfg.headDim
            let numKVL   = isFull ? cfg.numFullKVHeads : cfg.numKVHeads
            let qDim     = UInt32(cfg.numHeads * headDimL)
            let kvDim    = UInt32(numKVL * headDimL)
            let seqLen   = UInt32(position + 1)

            let inNorm   = try model.inputNorm(layer: L)
            let postAttn = try model.postAttnNorm(layer: L)
            let sharedProj = sharedExpertProjections[L]
            let routerW  = try model.router(layer: L)
            let perExpertScale: (buffer: MTLBuffer, offset: Int)
            if cfg.routerScaled {
                let view = try model.routerPerExpertScale(layer: L)
                perExpertScale = (view.buffer, Int(view.offset))
            } else {
                perExpertScale = (onesPerExpertScale!, 0)
            }

            let tCb1Start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            // Everything up to and including the router runs in a single CB:
            // the only reason to break is the CPU readback of router indices
            // needed to issue I/O for the routed-expert blobs.
            let cb = ctx.queue.makeCommandBuffer()!
            rms.encodeBF16W(commandBuffer: cb,
                            x: hidden,
                            weight: inNorm.buffer, weightOffset: Int(inNorm.offset),
                            out: normed,
                            d: D, eps: eps)

            if isLinear {
                // Gated-DeltaNet linear attention: no KV slots, no RoPE — a
                // fixed-size recurrent state updated in place.
                try encodeLinearAttentionDecode(cb, layer: L)
            } else if cfg.attnOutputGate {
                // Qwen full attention: packed [query ; gate] q_proj, real
                // v_proj, no V norm, NeoX sub-dim RoPE, sigmoid output gate.
                try encodeGatedFullAttentionDecode(cb, layer: L,
                                                   position: position,
                                                   seqLen: seqLen)
            } else {
                let kSlot = kv?.kSlot(layer: L, position: position) ?? (buffer: kStage, offset: 0)
                let vSlot = kv?.vSlot(layer: L, position: position) ?? (buffer: vStage, offset: 0)
                let q     = try model.qProj(layer: L)
                let k     = try model.kProj(layer: L)
                // Under the K=V quirk full layers reuse k_proj; otherwise
                // v_proj is a real tensor.
                let vProj = (isFull && cfg.attentionKEqV) ? k : (try model.vProj(layer: L))
                let o     = try model.oProj(layer: L)
                let qNorm = try model.qNorm(layer: L)
                let kNorm = try model.kNorm(layer: L)

                fusedQKVGEMV.encode(commandBuffer: cb,
                                    qWeights: q.buffer, qWeightsOffset: Int(q.offset),
                                    qScales: q.buffer, qScalesOffset: Int(q.scaleOffset),
                                    qBiases: q.buffer, qBiasesOffset: Int(q.biasOffset),
                                    kWeights: k.buffer, kWeightsOffset: Int(k.offset),
                                    kScales: k.buffer, kScalesOffset: Int(k.scaleOffset),
                                    kBiases: k.buffer, kBiasesOffset: Int(k.biasOffset),
                                    vWeights: vProj.buffer, vWeightsOffset: Int(vProj.offset),
                                    vScales: vProj.buffer, vScalesOffset: Int(vProj.scaleOffset),
                                    vBiases: vProj.buffer, vBiasesOffset: Int(vProj.biasOffset),
                                    x: normed,
                                    qOut: qScratch,
                                    kOut: kSlot.buffer, kOutOffset: kSlot.offset,
                                    vOut: vSlot.buffer, vOutOffset: vSlot.offset,
                                    qRows: qDim,
                                    kvRows: kvDim,
                                    n: D)

                let rotated = isFull
                    ? UInt32(Double(cfg.fullHeadDim) * cfg.partialRotaryFactor / 2.0)
                    : UInt32(headDimL / 2)
                fusedQKVEpilogue.encode(commandBuffer: cb,
                                        q: qScratch,
                                        k: kSlot.buffer,
                                        kOffset: kSlot.offset,
                                        v: vSlot.buffer,
                                        vOffset: vSlot.offset,
                                        qWeight: qNorm.buffer,
                                        qWeightOffset: Int(qNorm.offset),
                                        kWeight: kNorm.buffer,
                                        kWeightOffset: Int(kNorm.offset),
                                        headDim: UInt32(headDimL),
                                        numQHeads: UInt32(cfg.numHeads),
                                        numKVHeads: UInt32(numKVL),
                                        position: UInt32(position),
                                        theta: isFull ? Float(cfg.fullRopeTheta) : Float(cfg.ropeTheta),
                                        rotatedPairs: rotated,
                                        eps: eps)

                guard kv != nil else {
                    preconditionFailure("FP16 attention requires an FP16 KV cache")
                }
                if isFull {
                    attention.encodeFull(commandBuffer: cb,
                                         q: qScratch,
                                         k: kSlot.buffer, kOffset: 0,
                                         v: vSlot.buffer, vOffset: 0,
                                         out: attnOut,
                                         headDim: UInt32(headDimL),
                                         numQHeads: UInt32(cfg.numHeads),
                                         numKVHeads: UInt32(numKVL),
                                         seqLen: seqLen,
                                         scale: Float(cfg.attentionScale))
                } else {
                    let ringCapacity = kv?.ringCapacity(layer: L) ?? 0
                    let activeRingCapacity = ringCapacity > 0 && Int(seqLen) > ringCapacity
                        ? UInt32(ringCapacity)
                        : 0
                    attention.encodeSWA(commandBuffer: cb,
                                        q: qScratch,
                                        k: kSlot.buffer, kOffset: 0,
                                        v: vSlot.buffer, vOffset: 0,
                                        out: attnOut,
                                        headDim: UInt32(headDimL),
                                        numQHeads: UInt32(cfg.numHeads),
                                        numKVHeads: UInt32(numKVL),
                                        seqLen: seqLen,
                                        window: UInt32(cfg.slidingWindow),
                                        scale: Float(cfg.attentionScale),
                                        ringCapacity: activeRingCapacity)
                }
                int4.encode(commandBuffer: cb,
                            weights: o.buffer, weightsOffset: Int(o.offset),
                            scales:  o.buffer, scalesOffset:  Int(o.scaleOffset),
                            biases:  o.buffer, biasesOffset:  Int(o.biasOffset),
                            x: attnOut, y: oOut, m: D, n: qDim)
            }

            if cfg.ffnSandwichNorms {
                let preFFN   = try model.preFFN(layer: L)
                let preFFN2  = try model.preFFN2(layer: L)
                fusedPostAttentionSetup.encode(commandBuffer: cb,
                                               hidden: hidden,
                                               attn: oOut,
                                               denseX: denseX,
                                               routedX: routedX,
                                               routerX: routerInput,
                                               postAttentionWeight: postAttn.buffer,
                                               postAttentionWeightOffset: Int(postAttn.offset),
                                               preFFNWeight: preFFN.buffer,
                                               preFFNWeightOffset: Int(preFFN.offset),
                                               preFFN2Weight: preFFN2.buffer,
                                               preFFN2WeightOffset: Int(preFFN2.offset),
                                               d: D,
                                               eps: eps)
            } else {
                // Plain pre-norm residual block: hidden += attention branch,
                // then one post-attention norm feeds router, shared expert,
                // and routed phase 1 (routedX doubles as moeX).
                elementwise!.encodeResidualAdd(commandBuffer: cb,
                                               hidden: hidden,
                                               delta: oOut,
                                               count: cfg.hiddenSize)
                rms.encodeBF16W(commandBuffer: cb,
                                x: hidden,
                                weight: postAttn.buffer,
                                weightOffset: Int(postAttn.offset),
                                out: routedX,
                                d: D, eps: eps)
            }

            moe.encodeRouterGemma4(commandBuffer: cb,
                weights: routerW.buffer, weightsOffset: Int(routerW.offset),
                scales:  routerW.buffer, scalesOffset:  Int(routerW.scaleOffset),
                biases:  routerW.buffer, biasesOffset:  Int(routerW.biasOffset),
                hidden: cfg.ffnSandwichNorms ? routerInput : routedX,
                effectiveScale: effectiveScaleBuffers[L],
                perExpertScale: perExpertScale.buffer,
                perExpertScaleOffset: perExpertScale.offset,
                outIndices: outIndices, outWeights: outWeights,
                numExperts: UInt32(cfg.numExperts), d: D, topK: UInt32(cfg.topKExperts),
                weightBits: model.routerWeightBits)
            cb.commit()
            let tWait = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            waitForCompletion(cb)
            let waitNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tWait
            if let pending = pendingRoutedCommand {
                finishPendingRoutedCommand(pending, waitIfNeeded: false)
                pendingRoutedCommand = nil
            }
            totalCb1Nanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tCb1Start - waitNanos

            if debugRmsDump {
                // At this point (cb done, MoE not yet folded in):
                //   hidden = input + attention/GDN branch
                //   oOut   = attention/GDN branch output (D elements)
                //   GDN state + conv tail + intermediates are this layer's
                //   final values for the step.
                dumpHiddenRMS(label: "decodeAttnOut p\(position)", layer: L,
                              buffer: oOut, count: cfg.hiddenSize)
                dumpHiddenRMS(label: "decodeAttn p\(position)", layer: L,
                              buffer: hidden, count: cfg.hiddenSize)
                if cfg.layerIsLinear(L) {
                    dumpGDNProbe(label: "decode p\(position)", layer: L,
                                 includeShared: true)
                }
            }

            // CPU readback to fetch routed-expert blobs from disk.
            let idxPtr = outIndices.contents().bindMemory(to: UInt32.self,
                                                          capacity: cfg.topKExperts)
            var experts = [Int](repeating: 0, count: cfg.topKExperts)
            for i in 0..<cfg.topKExperts {
                experts[i] = min(Int(idxPtr[i]), cfg.numExperts - 1)
            }

            let routedOffsets = model.routedExpertOffsets(layer: L)
            let topK = UInt32(cfg.topKExperts)
            let canPlanPhase1HitSplit =
                cfg.topKExperts <= MoE.maxStreamedExperts
            let plannedFetch = canPlanPhase1HitSplit
                ? try model.planRoutedExperts(layer: L, experts: experts)
                : nil
            var phase1HitCB: MTLCommandBuffer?
            var phase1HitSplitArgBuf: MTLBuffer?
            var phase1HitSplitRoutedBufs: [MTLBuffer] = []
            var phase1HitSlots: [UInt32] = []
            var phase1MissSlots: [UInt32] = []

            if let plan = plannedFetch {
                let missSet = Set(plan.misses)
                phase1HitSlots = (0..<cfg.topKExperts)
                    .filter { !missSet.contains($0) }
                    .map { UInt32($0) }
                phase1MissSlots = plan.misses.map { UInt32($0) }
            }
            func encodeRoutedPhase1Full(
                _ cb: MTLCommandBuffer,
                argBuf: MTLBuffer,
                routedBufs: [MTLBuffer]
            ) {
                moe.encodeRoutedPersistentPhase1U16Load(commandBuffer: cb,
                                                        routedArgBuffer: argBuf,
                                                        routedBlobs: routedBufs,
                                                        routedOffsets: routedOffsets,
                                                        x: routedX,
                                                        acts: moeActs,
                                                        d: D,
                                                        f: FmoE,
                                                        topK: topK)
            }

            func encodeRoutedPhase1Subset(
                _ cb: MTLCommandBuffer,
                argBuf: MTLBuffer,
                routedBufs: [MTLBuffer],
                activeSlots: MTLBuffer,
                activeSlotIndices: [UInt32],
                activeCount: UInt32
            ) {
                moe.encodeRoutedPersistentPhase1SubsetU16Load(
                    commandBuffer: cb,
                    routedArgBuffer: argBuf,
                    routedBlobs: routedBufs,
                    routedOffsets: routedOffsets,
                    x: routedX,
                    acts: moeActs,
                    activeSlots: activeSlots,
                    activeSlotIndices: activeSlotIndices,
                    activeCount: activeCount,
                    d: D,
                    f: FmoE,
                    topK: topK)
            }

            if let plan = plannedFetch,
               plan.hits > 0,
               !plan.misses.isEmpty {
                let plannedBlobs = try model.routedExpertBuffers(for: plan)
                phase1HitSplitRoutedBufs = plannedBlobs.map { $0.buffer }
                phase1HitSplitArgBuf = moe.makeRoutedArgumentBuffer(
                    routedBlobs: phase1HitSplitRoutedBufs,
                    topK: topK)
                if let argBuf = phase1HitSplitArgBuf, plan.hits > 0, !plan.misses.isEmpty {
                    writeActiveSlots(phase1HitSlots, into: moeHitActiveSlots)
                    let cb = ctx.queue.makeCommandBuffer()!
                    encodeRoutedPhase1Subset(
                        cb,
                        argBuf: argBuf,
                        routedBufs: phase1HitSplitRoutedBufs,
                        activeSlots: moeHitActiveSlots,
                        activeSlotIndices: phase1HitSlots,
                        activeCount: UInt32(phase1HitSlots.count))
                    phase1HitCB = cb
                }
            }

            // The shared dense MLP depends only on its normed input, not on
            // the routed experts. Commit it without waiting so its GPU work
            // overlaps the routed-expert pread. The routed CB follows it on
            // the same queue, so the combine sees h1Buf.
            let sharedCB = ctx.queue.makeCommandBuffer()!
            try! shared.encode(commandBuffer: sharedCB,
                               x: cfg.ffnSandwichNorms ? denseX : routedX,
                               gate: sharedProj.gate,
                               up: sharedProj.up,
                               down: sharedProj.down,
                               y: h1Buf,
                               scratchGate: denseScratchGate,
                               scratchUp: denseScratchUp,
                               scratchAct: denseScratchAct)
            if cfg.ffnSandwichNorms {
                let postF1 = sharedProj.postF1!
                rms.encodeBF16W(commandBuffer: sharedCB, x: h1Buf,
                                weight: postF1.buffer,
                                weightOffset: Int(postF1.offset),
                                out: h1Buf, d: D, eps: eps)
            } else if cfg.sharedExpertGated {
                // out = sigmoid(shared_expert_gate(moeX)) * shared_mlp(moeX)
                let gateView = sharedProj.scalarGate!
                int8ScalarGate!.encode(commandBuffer: sharedCB,
                                       weights: gateView.buffer,
                                       weightsOffset: Int(gateView.offset),
                                       scales: gateView.buffer,
                                       scalesOffset: Int(gateView.scaleOffset),
                                       biases: gateView.buffer,
                                       biasesOffset: Int(gateView.biasOffset),
                                       x: routedX,
                                       y: sharedScalarGateBuf!,
                                       m: 1, n: D)
                elementwise!.encodeSigmoidScalarMul(commandBuffer: sharedCB,
                                                    y: h1Buf,
                                                    gate: sharedScalarGateBuf!,
                                                    count: cfg.hiddenSize)
            }
            sharedCB.commit()
            if let cb = phase1HitCB {
                cb.commit()
            }
            if rdadviseEnabled && rdadvisePolicyMode != .off {
                let requestedMisses = plannedFetch?.misses.count ?? experts.count
                let estimatedAdviceBytes = try model.routedExpertAdviceByteEstimate(
                    layer: L,
                    missCount: requestedMisses)
                if let skipped = shouldSkipRDAdvice(position: position,
                                                    requestedMisses: requestedMisses,
                                                    estimatedBytes: estimatedAdviceBytes,
                                                    canOverlapUsefulGPUWork: true) {
                    recordRDAdvice(skipped, wallNanos: 0)
                } else {
                    let tAdvice = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                    let result: ExpertIOAdviceResult
                    if let plannedFetch {
                        result = try model.adviseRoutedExperts(plan: plannedFetch)
                    } else {
                        result = try model.adviseRoutedExperts(layer: L, experts: experts)
                    }
                    let wallNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tAdvice
                    recordRDAdvice(result, wallNanos: wallNanos)
                    updateRDAdvicePolicy(after: result, position: position)
                }
            }

            // Routed-expert pread — overlaps the shared MLP GPU work above.
            let tIoStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let blobs: [TensorView]
            if let plannedFetch {
                blobs = try await model.fetchRoutedExperts(plan: plannedFetch)
            } else {
                blobs = try await model.fetchRoutedExperts(layer: L, experts: experts)
            }
            let layerIo = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tIoStart
            totalIoNanos &+= layerIo
            let routedBufs = blobs.map { $0.buffer }
            let tCb2Start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let gTail: (MTLCommandBuffer) -> Void
            if cfg.ffnSandwichNorms {
                let postF2 = try model.postFFN2(layer: L)
                let postF = try model.postFFN(layer: L)
                let layerScalarView = try model.layerScalar(layer: L)
                let scalarPtr = layerScalarView.buffer.contents()
                    .advanced(by: Int(layerScalarView.offset))
                    .assumingMemoryBound(to: UInt16.self)
                let layerScalar = Quantization.bf16ToFloat(scalarPtr[0])
                gTail = { [self] cb in
                    fusedTail.encode(commandBuffer: cb,
                                     h2: h2Buf,
                                     h1: h1Buf,
                                     hidden: hidden,
                                     postFFN2Weight: postF2.buffer,
                                     postFFN2WeightOffset: Int(postF2.offset),
                                     postFFNWeight: postF.buffer,
                                     postFFNWeightOffset: Int(postF.offset),
                                     d: D,
                                     eps: eps,
                                     layerScalar: layerScalar)
                }
            } else {
                // The phase-2 reduce already folded the shared branch (h1Buf
                // as its residual); the tail is a plain residual add.
                gTail = { [self] cb in
                    elementwise!.encodeResidualAdd(commandBuffer: cb,
                                                   hidden: hidden,
                                                   delta: h2Buf,
                                                   count: cfg.hiddenSize)
                }
            }
            let routedCB = ctx.queue.makeCommandBuffer()!
            let splitArgBuf = phase1HitCB != nil && !phase1MissSlots.isEmpty
                ? phase1HitSplitArgBuf
                : nil
            let argBuf = splitArgBuf ?? moe.makeReusedRoutedArgumentBuffer(
                routedBlobs: routedBufs,
                topK: topK)
            if splitArgBuf != nil {
                writeActiveSlots(phase1MissSlots, into: moeMissActiveSlots)
                encodeRoutedPhase1Subset(
                    routedCB,
                    argBuf: argBuf,
                    routedBufs: routedBufs,
                    activeSlots: moeMissActiveSlots,
                    activeSlotIndices: phase1MissSlots,
                    activeCount: UInt32(phase1MissSlots.count))
            } else {
                encodeRoutedPhase1Full(routedCB,
                                       argBuf: argBuf,
                                       routedBufs: routedBufs)
            }
            moe.encodeRoutedPersistentPhase2Reduce(commandBuffer: routedCB,
                                                   routedArgBuffer: argBuf,
                                                   routedBlobs: routedBufs,
                                                   routedOffsets: routedOffsets,
                                                   acts: moeActs,
                                                   routingWeights: outWeights,
                                                   residual: cfg.ffnSandwichNorms ? zeroResidual : h1Buf,
                                                   y: h2Buf,
                                                   d: D,
                                                   f: FmoE,
                                                   topK: topK)
            gTail(routedCB)
            routedCB.commit()
            precondition(pendingRoutedCommand == nil,
                         "routed command-buffer pipeline drained before queuing the next layer")
            pendingRoutedCommand = PendingRoutedCommand(
                cb: routedCB,
                sharedCB: sharedCB,
                phase1HitCB: phase1HitCB,
                encodeAndCommitNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tCb2Start)
            if debugRmsDump {
                // `hidden` (this layer's output) is final once routedCB + the
                // overlapped sharedCB / phase1HitCB have completed; wait then
                // read the running residual back and report its RMS so the
                // corrupted layer shows up as the first blow-up.
                waitForCompletion(routedCB)
                waitForCompletion(sharedCB)
                if let phase1HitCB { waitForCompletion(phase1HitCB) }
                dumpHiddenRMS(label: "decode p\(position)", layer: L,
                              buffer: hidden, count: cfg.hiddenSize)
                // MoE branch: h2Buf = routed(top-k weighted) + gated shared.
                // hidden_after = hidden_before + oOut(attention/GDN) + h2Buf.
                dumpHiddenRMS(label: "moe p\(position)", layer: L,
                              buffer: h2Buf, count: cfg.hiddenSize)
                // Router decision for this layer: which experts, with what
                // weights. Garbage expert selection (4-bit router) shows up
                // here as uniform/near-random weights and indices.
                let idxRaw = outIndices.contents()
                let wRaw = outWeights.contents()
                var idxDesc = [String]()
                var wDesc = [String]()
                for i in 0..<cfg.topKExperts {
                    idxDesc.append(String(idxRaw.load(fromByteOffset: i * 4, as: UInt32.self)))
                    wDesc.append(String(format: "%.3f", Self.float16ToDouble(
                        wRaw.load(fromByteOffset: i * 2, as: UInt16.self))))
                }
                print("[ROUTER-\(position)-L\(L)] idx=[\(idxDesc.joined(separator: ","))] w=[\(wDesc.joined(separator: ","))]")

                // CPU reference reimplementation of the decode router GEMV
                // (router_gemv_gemma4_body) + top-k (router_topk_select_k8),
                // to separate "the 4-bit dequant/top-k path is buggy" from
                // "4-bit quantization itself flattens the logits". Mirrors the
                // ztest nibble/affine dequant exactly. effective_scale and
                // per_expert_scale are all-ones for Qwen but are applied here
                // so the path stays correct for the Gemma routerScaled case.
                let Di = Int(D)
                let groups = Di / 64
                let esRaw = effectiveScaleBuffers[L].contents()
                let pesRaw = perExpertScale.buffer.contents().advanced(by: perExpertScale.offset)
                let xRaw = routedX.contents()
                var xv = [Double](repeating: 0, count: Di)
                for k in 0..<Di {
                    let es = Double(Quantization.bf16ToFloat(
                        esRaw.load(fromByteOffset: k * 2, as: UInt16.self)))
                    xv[k] = Self.float16ToDouble(xRaw.load(fromByteOffset: k * 2, as: UInt16.self)) * es
                }
                let rwRaw = routerW.buffer.contents().advanced(by: Int(routerW.offset))
                let rwS = routerW.buffer.contents().advanced(by: Int(routerW.scaleOffset))
                let rwB = routerW.buffer.contents().advanced(by: Int(routerW.biasOffset))
                var logits = [Double](repeating: 0, count: cfg.numExperts)
                for e in 0..<cfg.numExperts {
                    var acc = 0.0
                    let rowBase = e * (Di / 2)
                    for g in 0..<groups {
                        let scale = Double(Quantization.bf16ToFloat(
                            rwS.load(fromByteOffset: (e * groups + g) * 2, as: UInt16.self)))
                        let bias = Double(Quantization.bf16ToFloat(
                            rwB.load(fromByteOffset: (e * groups + g) * 2, as: UInt16.self)))
                        var dot = 0.0
                        var sumx = 0.0
                        for j in 0..<64 {
                            let col = g * 64 + j
                            let wbyte = rwRaw.load(fromByteOffset: rowBase + g * 32 + j / 2,
                                                   as: UInt8.self)
                            let nib = (j % 2 == 0) ? Double(wbyte & 0x0F)
                                                   : Double((wbyte >> 4) & 0x0F)
                            dot += nib * xv[col]
                            sumx += xv[col]
                        }
                        acc += scale * dot + bias * sumx
                    }
                    logits[e] = acc
                }
                // CPU top-8 (desc by logit, ties → lower index) — matches the
                // kernel's `s == top_score[i] && e < top_idx[i]` tie-break.
                let top8 = (0..<cfg.numExperts).sorted { a, b in
                    if logits[a] != logits[b] { return logits[a] > logits[b] }
                    return a < b
                }.prefix(8)
                let lmax = logits[top8[0]]
                let esum = top8.reduce(0.0) { $0 + exp(logits[$1] - lmax) }
                var cpuIdx = [Int]()
                var cpuW = [Double]()
                for e in top8 {
                    cpuIdx.append(e)
                    let pes = Double(Quantization.bf16ToFloat(
                        pesRaw.load(fromByteOffset: e * 2, as: UInt16.self)))
                    cpuW.append(exp(logits[e] - lmax) / esum * pes)
                }
                let lmean = logits.reduce(0, +) / Double(logits.count)
                let lstd = (logits.reduce(0.0) { $0 + ($1 - lmean) * ($1 - lmean) }
                            / Double(logits.count)).squareRoot()
                let lmin = logits.min()!
                let lmx = logits.max()!
                var kIdx = [Int]()
                var kW = [Double]()
                for i in 0..<8 {
                    kIdx.append(Int(idxRaw.load(fromByteOffset: i * 4, as: UInt32.self)))
                    kW.append(Self.float16ToDouble(wRaw.load(fromByteOffset: i * 2, as: UInt16.self)))
                }
                let idxMatch = (0..<8).allSatisfy { cpuIdx[$0] == kIdx[$0] }
                let wMatch = (0..<8).allSatisfy {
                    abs(cpuW[$0] - kW[$0]) <= max(1e-3, 0.02 * kW[$0])
                }
                print("[ROUTERLOGIT-\(position)-L\(L)] logitMax=\(String(format: "%.4f", lmx)) "
                      + "logitMin=\(String(format: "%.4f", lmin)) logitStd=\(String(format: "%.4f", lstd)) "
                      + "top8L=\(top8.map { String(format: "%.3f", logits[$0]) }.joined(separator: ","))")
                print("[ROUTERLOGIT-\(position)-L\(L)] cpuIdx=[\(cpuIdx.map { String($0) }.joined(separator: ","))] "
                      + "cpuW=\(cpuW.map { String(format: "%.4f", $0) }.joined(separator: ","))")
                print("[ROUTERLOGIT-\(position)-L\(L)] kernIdx=[\(kIdx.map { String($0) }.joined(separator: ","))] "
                      + "kernW=\(kW.map { String(format: "%.4f", $0) }.joined(separator: ","))")
                print("[ROUTERLOGIT-\(position)-L\(L)] match=\(idxMatch && wMatch ? "YES" : "NO") "
                      + "idx=\(idxMatch ? "YES" : "NO") w=\(wMatch ? "YES" : "NO")")
            }
            continue
        }
        if let pending = pendingRoutedCommand {
            finishPendingRoutedCommand(pending, waitIfNeeded: true)
            pendingRoutedCommand = nil
        }

        // The fused head skips the vocab buffer and leaves a greedy token in
        // greedyTokenBuf; the logits path writes the complete vector.
        let fNorm = model.finalNorm
        let lm    = model.lmHead
        let gFinalNorm: (MTLCommandBuffer) -> Void = { cb in
            self.rms.encodeBF16W(commandBuffer: cb, x: self.hidden,
                                 weight: fNorm.buffer, weightOffset: Int(fNorm.offset),
                                 out: self.normed, d: D, eps: eps)
        }
        let gLmHead: (MTLCommandBuffer) -> Void = { cb in
            self.int4.encode(commandBuffer: cb,
                             weights: lm.buffer, weightsOffset: Int(lm.offset),
                             scales:  lm.buffer, scalesOffset:  Int(lm.scaleOffset),
                             biases:  lm.buffer, biasesOffset:  Int(lm.biasOffset),
                             x: self.normed, y: logits, m: UInt32(self.cfg.vocabSize), n: D)
        }
        let gFusionHead: (MTLCommandBuffer) -> Void = { cb in
            self.fusionHead.encodeGreedyDecode(
                commandBuffer: cb,
                hidden: self.hidden,
                normWeight: fNorm.buffer, normOffset: Int(fNorm.offset),
                weights: lm.buffer, weightsOffset: Int(lm.offset),
                scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                outToken: self.greedyTokenBuf,
                d: D, vocab: UInt32(self.cfg.vocabSize),
                rmsEps: eps)
        }
        if emitHead {
            let useFusedHeadForThisToken = useFusedGreedyHead && outputMode == .greedyIfAvailable
            let tHead = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            if useFusedHeadForThisToken {
                runSync(gFusionHead)
                totalHeadFusedNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tHead
                lastGreedyToken = greedyTokenBuf.contents().load(as: UInt32.self)
            } else {
                runSync { cb in
                    gFinalNorm(cb)
                    gLmHead(cb)
                }
                totalHeadNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tHead
            }
        }

        kv?.advance()
    }

    /// Gated-DeltaNet linear attention (layer mask 2), one decode step.
    /// Reads `normed`, updates the layer's recurrent state + conv tail in
    /// place, and leaves the attention-branch output in `oOut`.
    private func encodeLinearAttentionDecode(_ cb: MTLCommandBuffer, layer L: Int) throws {
        guard let gdn, let gdnState, let gdnQKVRaw, let gdnConvOut,
              let gdnZ, let gdnA, let gdnB, let gdnY, let gdnOut else {
            preconditionFailure("linear-attention layer without GDN kernels")
        }
        let la = cfg.linearAttention
        let D = UInt32(cfg.hiddenSize)
        let qkvW = try model.linearInProjQKV(layer: L)
        let zW = try model.linearInProjZ(layer: L)
        let aW = try model.linearInProjA(layer: L)
        let bW = try model.linearInProjB(layer: L)
        let outW = try model.linearOutProj(layer: L)
        let convW = try model.linearConv1d(layer: L)
        let aLog = try model.linearALog(layer: L)
        let dtBias = try model.linearDtBias(layer: L)
        let gatedNormW = try model.linearNorm(layer: L)

        // One dispatch over the concatenated qkv/z/a/b row space instead of four
        // separate GEMVs (a and b were 4 threadgroups each).
        gdn.encodeInputProjections(commandBuffer: cb,
                                   x: normed,
                                   qkv: qkvW, qkvOut: gdnQKVRaw,
                                   z: zW, zOut: gdnZ,
                                   a: aW, aOut: gdnA,
                                   b: bW, bOut: gdnB,
                                   hiddenSize: cfg.hiddenSize)

        if debugRmsDump {
             // Snapshot the conv tail BEFORE this step consumes/shifts it, so the
             // CPU conv reference (convtest) can replay the same inputs the
             // gdn_conv_mix_decode kernel saw. Queued before `cb` (not yet
             // committed), so it runs first — mirrors the state snapshot above.
            let tailBytes = max(0, la.convKernelSize - 1) * la.qkvDim
                           * MemoryLayout<Float16>.stride
            if debugTailBefore == nil {
                debugTailBefore = ctx.device.makeBuffer(
                    length: tailBytes, options: .storageModeShared)
             }
            if let dst = debugTailBefore,
               let blitCB = ctx.queue.makeCommandBuffer(),
               let blit = blitCB.makeBlitCommandEncoder() {
                blit.copy(from: gdnState.convTailBuffer(layer: L), sourceOffset: 0,
                          to: dst, destinationOffset: 0, size: tailBytes)
                blit.endEncoding()
                blitCB.commit()
             }
         }
        gdn.encodeConvDecode(commandBuffer: cb,
                             tail: gdnState.convTailBuffer(layer: L),
                             qkv: gdnQKVRaw,
                             convWeight: convW.buffer,
                             convWeightOffset: Int(convW.offset),
                             out: gdnConvOut)
        gdn.encodeQKNorm(commandBuffer: cb, convOut: gdnConvOut)
        if debugRmsDump {
            // Snapshot the recurrent state BEFORE this step's update so the
            // CPU reference delta-rule can be replayed against the kernel's y.
            // Queued before `cb` (not yet committed), so it runs first.
            let bytes = la.numVHeads * la.valueHeadDim * la.keyHeadDim
                * MemoryLayout<Float>.stride
            if debugStateBefore == nil {
                debugStateBefore = ctx.device.makeBuffer(
                    length: bytes, options: .storageModeShared)
            }
            if let dst = debugStateBefore,
               let blitCB = ctx.queue.makeCommandBuffer(),
               let blit = blitCB.makeBlitCommandEncoder() {
                blit.copy(from: gdnState.stateBuffer(layer: L), sourceOffset: 0,
                          to: dst, destinationOffset: 0, size: bytes)
                blit.endEncoding()
                blitCB.commit()
            }
        }
        gdn.encodeDeltaStepDecode(commandBuffer: cb,
                                  convOut: gdnConvOut,
                                  aProj: gdnA,
                                  bProj: gdnB,
                                  aLog: aLog.buffer, aLogOffset: Int(aLog.offset),
                                  dtBias: dtBias.buffer, dtBiasOffset: Int(dtBias.offset),
                                  state: gdnState.stateBuffer(layer: L),
                                  y: gdnY)
        gdn.encodeGatedNorm(commandBuffer: cb,
                            y: gdnY,
                            z: gdnZ,
                            weight: gatedNormW.buffer,
                            weightOffset: Int(gatedNormW.offset),
                            out: gdnOut)
        int4.encode(commandBuffer: cb,
                    weights: outW.buffer, weightsOffset: Int(outW.offset),
                    scales: outW.buffer, scalesOffset: Int(outW.scaleOffset),
                    biases: outW.buffer, biasesOffset: Int(outW.biasOffset),
                    x: gdnOut, y: oOut, m: D, n: UInt32(la.valueDim))
    }

    /// Qwen full attention (attn_output_gate), one decode step: packed
    /// [query ; gate] q_proj split per head, weighted per-head q/k norms
    /// (no V norm), NeoX sub-dim RoPE, full attention with the configured
    /// scale, sigmoid output gate, then o_proj into `oOut`.
    private func encodeGatedFullAttentionDecode(_ cb: MTLCommandBuffer,
                                                layer L: Int,
                                                position: Int,
                                                seqLen: UInt32) throws {
        guard let elementwise, let rope, let qPackedScratch, let attnGateScratch else {
            preconditionFailure("attn_output_gate layer without gate kernels")
        }
        guard let kv else {
            preconditionFailure("FP16 attention requires an FP16 KV cache")
        }
        let D = UInt32(cfg.hiddenSize)
        let eps: Float = 1e-6
        let headDim = cfg.fullHeadDim
        let numKV = cfg.numFullKVHeads
        let qDim = UInt32(cfg.numHeads * headDim)
        let kvDim = UInt32(numKV * headDim)
        let kSlot = kv.kSlot(layer: L, position: position)
        let vSlot = kv.vSlot(layer: L, position: position)
        let q = try model.qProj(layer: L)
        let k = try model.kProj(layer: L)
        let v = try model.vProj(layer: L)
        let o = try model.oProj(layer: L)
        let qNormW = try model.qNorm(layer: L)
        let kNormW = try model.kNorm(layer: L)
        let rotaryDim = UInt32(Double(headDim) * cfg.partialRotaryFactor)

         // DEBUG: one-time config dump for the first full-attn layer
        if L == 0 || cfg.fullAttentionLayerMask[L - 1] != 1 {
            print("[DEBUG-gated-attn] CONFIG: family=\(cfg.family.rawValue) "
                 + "numHeads=\(cfg.numHeads) numKVHeads=\(numKV) "
                 + "headDim=\(headDim) qDim=\(qDim) kvDim=\(kvDim) "
                 + "qRows=\(2 * qDim) attnScale=\(cfg.attentionScale) "
                 + "ropeNeox=\(cfg.ropeNeoxSubdim) rotaryDim=\(rotaryDim)")
        }
         // DEBUG: print weight shapes for first full-attn layer
        if L == 3 {  // first full-attn layer for Qwen3.5
            print("[DEBUG-gated-attn] L=\(L) q_proj shape=(\(q.shape.0),\(q.shape.1)) "
                 + "k_proj shape=(\(k.shape.0),\(k.shape.1)) "
                 + "v_proj shape=(\(v.shape.0),\(v.shape.1)) "
                 + "o_proj shape=(\(o.shape.0),\(o.shape.1)) "
                 + "q_norm=\(qNormW.shape.0) k_norm=\(kNormW.shape.0)")
        }

        fusedQKVGEMV.encode(commandBuffer: cb,
                            qWeights: q.buffer, qWeightsOffset: Int(q.offset),
                            qScales: q.buffer, qScalesOffset: Int(q.scaleOffset),
                            qBiases: q.buffer, qBiasesOffset: Int(q.biasOffset),
                            kWeights: k.buffer, kWeightsOffset: Int(k.offset),
                            kScales: k.buffer, kScalesOffset: Int(k.scaleOffset),
                            kBiases: k.buffer, kBiasesOffset: Int(k.biasOffset),
                            vWeights: v.buffer, vWeightsOffset: Int(v.offset),
                            vScales: v.buffer, vScalesOffset: Int(v.scaleOffset),
                            vBiases: v.buffer, vBiasesOffset: Int(v.biasOffset),
                            x: normed,
                            qOut: qPackedScratch,
                            kOut: kSlot.buffer, kOutOffset: kSlot.offset,
                            vOut: vSlot.buffer, vOutOffset: vSlot.offset,
                            qRows: 2 * qDim,
                            kvRows: kvDim,
                            n: D)
        elementwise.encodeSplitQGate(commandBuffer: cb,
                                     packed: qPackedScratch,
                                     q: qScratch,
                                     gate: attnGateScratch,
                                     heads: cfg.numHeads,
                                     dim: headDim)
        rms.encodeBF16WPerHead(commandBuffer: cb,
                               x: qScratch,
                               weight: qNormW.buffer,
                               weightOffset: Int(qNormW.offset),
                               out: qScratch,
                               headDim: UInt32(headDim),
                               numHeads: cfg.numHeads,
                               eps: eps)
        rms.encodeBF16WPerHead(commandBuffer: cb,
                               x: kSlot.buffer, xOffset: kSlot.offset,
                               weight: kNormW.buffer,
                               weightOffset: Int(kNormW.offset),
                               out: kSlot.buffer, outOffset: kSlot.offset,
                               headDim: UInt32(headDim),
                               numHeads: numKV,
                               eps: eps)
        rope.encodeNeoxSubdim(commandBuffer: cb,
                              data: qScratch,
                              position: UInt32(position),
                              headDim: UInt32(headDim),
                              numHeads: UInt32(cfg.numHeads),
                              rotaryDim: rotaryDim,
                              theta: Float(cfg.fullRopeTheta))
        rope.encodeNeoxSubdim(commandBuffer: cb,
                              data: kSlot.buffer,
                              dataOffset: kSlot.offset,
                              position: UInt32(position),
                              headDim: UInt32(headDim),
                              numHeads: UInt32(numKV),
                              rotaryDim: rotaryDim,
                              theta: Float(cfg.fullRopeTheta))
        attention.encodeFull(commandBuffer: cb,
                             q: qScratch,
                             k: kSlot.buffer, kOffset: 0,
                             v: vSlot.buffer, vOffset: 0,
                             out: attnOut,
                             headDim: UInt32(headDim),
                             numQHeads: UInt32(cfg.numHeads),
                             numKVHeads: UInt32(numKV),
                             seqLen: seqLen,
                             scale: Float(cfg.attentionScale))
        elementwise.encodeSigmoidGateMul(commandBuffer: cb,
                                         out: attnOut,
                                         gate: attnGateScratch,
                                         count: Int(qDim))
        int4.encode(commandBuffer: cb,
                    weights: o.buffer, weightsOffset: Int(o.offset),
                    scales: o.buffer, scalesOffset: Int(o.scaleOffset),
                    biases: o.buffer, biasesOffset: Int(o.biasOffset),
                    x: attnOut, y: oOut, m: D, n: qDim)
    }

    private func runSync(_ body: (MTLCommandBuffer) -> Void) {
        let cb = ctx.queue.makeCommandBuffer()!
        body(cb)
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error {
            print("CB error: \(err)")
        }
    }

    private nonisolated func waitForCompletion(_ cb: MTLCommandBuffer) {
        cb.waitUntilCompleted()
        if let err = cb.error {
            print("CB error: \(err)")
        }
    }

    // MARK: - Debug: GDN intermediate probe

    /// Read back a linear-attention layer's GDN intermediates and state
    /// (all shared-mode buffers, so a direct CPU load is safe): the
    /// FP32 recurrent state, the FP16 conv tail, the per-row conv output,
    /// the delta-rule output `y`, and the a/b projections. Used to split
    /// "state carried in from prefill is corrupt" from "the decode
    /// delta-rule step corrupts it".
    /// `includeShared` gates the per-row intermediates (convOut/y/a/b), which
    /// live in shared scratch and only hold layer-`L`'s values in the decode
    /// path. Prefill calls with `includeShared: false` (state + tail only,
    /// which are per-layer and stable).
    private func dumpGDNProbe(label: String, layer: Int, includeShared: Bool) {
        guard cfg.layerIsLinear(layer), let gdnState else { return }
        let la = cfg.linearAttention

        let state = gdnState.stateBuffer(layer: layer)
        let stateCount = la.numVHeads * la.valueHeadDim * la.keyHeadDim
        let raw = state.contents()
        var sumSq = 0.0
        var maxAbs = 0.0
        for i in 0..<stateCount {
            let v = raw.load(fromByteOffset: i * MemoryLayout<Float>.stride,
                             as: Float.self)
            if v.isNaN || v.isInfinite { continue }
            let dv = Double(v)
            sumSq += dv * dv
            if abs(dv) > maxAbs { maxAbs = abs(dv) }
        }
        print("[GDN-\(label)-L\(layer)] state rms=\(String(format: "%.5f", (sumSq / Double(stateCount)).squareRoot())) maxAbs=\(String(format: "%.5f", maxAbs))")

        let tail = gdnState.convTailBuffer(layer: layer)
        dumpGDNBuffer(label: label, layer: layer, name: "tail",
                      buffer: tail, count: max(0, la.convKernelSize - 1) * la.qkvDim)

        // The gated-norm weight (BF16, [valueHeadDim]) — out = rmsnorm(y) *
        // weight * silu(z), so gdnOut's scale is set by this weight and z.
        if let view = try? model.linearNorm(layer: layer) {
            let count = la.valueHeadDim
            let raw = view.buffer.contents().advanced(by: Int(view.offset))
            var sumSq = 0.0
            var maxAbs = 0.0
            var minV = Double.greatestFiniteMagnitude
            var maxV = -Double.greatestFiniteMagnitude
            var first8 = [String]()
            for i in 0..<count {
                let bits = raw.load(fromByteOffset: i * MemoryLayout<UInt16>.stride,
                                    as: UInt16.self)
                let v = Double(Quantization.bf16ToFloat(bits))
                if v.isNaN || v.isInfinite { continue }
                sumSq += v * v
                if abs(v) > maxAbs { maxAbs = abs(v) }
                if v < minV { minV = v }
                if v > maxV { maxV = v }
                if i < 8 { first8.append(String(format: "%.4f", v)) }
            }
            print("[GDN-\(label)-L\(layer)] normW rms=\(String(format: "%.5f", (sumSq / Double(count)).squareRoot())) "
                  + "min=\(String(format: "%.4f", minV)) max=\(String(format: "%.4f", maxV)) first8=[\(first8.joined(separator: ","))]")
        }

        // Decisive alignment check: recompute gdnOut = (y/rms(y_h)) * W *
        // silu(z) per head from the read-back inputs and compare against the
        // kernel's actual output. If predicted matches actual (both large),
        // the blow-up is data-driven (y's energy lands on z's open dims);
        // if predicted is small but actual large, the kernel/buffers differ
        // from the data we see.
        if includeShared, let gdnY, let gdnZ, let gdnOut,
           let view = try? model.linearNorm(layer: layer) {
            let dv = la.valueHeadDim
            let Hv = la.numVHeads
            let yRaw = gdnY.contents()
            let zRaw = gdnZ.contents()
            let oRaw = gdnOut.contents()
            let wRaw = view.buffer.contents().advanced(by: Int(view.offset))
            func yv(_ i: Int) -> Double {
                Double(Self.float16ToDouble(yRaw.load(fromByteOffset: i * 2, as: UInt16.self)))
            }
            func zv(_ i: Int) -> Double {
                Self.float16ToDouble(zRaw.load(fromByteOffset: i * 2, as: UInt16.self))
            }
            func ov(_ i: Int) -> Double {
                Self.float16ToDouble(oRaw.load(fromByteOffset: i * 2, as: UInt16.self))
            }
            func wv(_ i: Int) -> Double {
                Double(Quantization.bf16ToFloat(wRaw.load(fromByteOffset: i * 2, as: UInt16.self)))
            }
            var maxPred = 0.0
            var maxAct = 0.0
            var predHead = 0
            var actHead = 0
            for h in 0..<Hv {
                var sumSq = 0.0
                for i in 0..<dv {
                    let v = yv(h * dv + i)
                    sumSq += v * v
                }
                let rms = (sumSq / Double(dv)).squareRoot()
                guard rms > 0 else { continue }
                var pSq = 0.0
                var aSq = 0.0
                for i in 0..<dv {
                    let w = (yv(h * dv + i) / rms) * wv(i) * Self.siluDouble(zv(h * dv + i))
                    pSq += w * w
                    let a = ov(h * dv + i)
                    aSq += a * a
                }
                let p = (pSq / Double(dv)).squareRoot()
                let a = (aSq / Double(dv)).squareRoot()
                if p > maxPred { maxPred = p; predHead = h }
                if a > maxAct { maxAct = a; actHead = h }
            }
            // Top-5 z spikes (global) with the y value at each spike.
            var spikes = [(idx: Int, z: Double, y: Double)]()
            for i in 0..<(Hv * dv) {
                let z = zv(i)
                if spikes.count < 5 || z > spikes.last!.z {
                    spikes.append((i, z, yv(i)))
                    spikes.sort { $0.z > $1.z }
                    if spikes.count > 5 { spikes.removeLast() }
                }
            }
            let spikeDesc = spikes.map { s -> String in
                "[\(s.idx)/h\(s.idx / dv) z=\(String(format: "%.3f", s.z)) y=\(String(format: "%.3f", s.y))]"
            }.joined(separator: " ")
            // Element-level truth: top-5 |gdnOut| elements with every factor,
            // so we can see exactly what the kernel computed at its largest
            // outputs.
            var topOut = [(idx: Int, o: Double)]()
            for i in 0..<(Hv * dv) {
                let o = ov(i)
                if topOut.count < 5 || abs(o) > abs(topOut.last!.o) {
                    topOut.append((i, o))
                    topOut.sort { abs($0.o) > abs($1.o) }
                    if topOut.count > 5 { topOut.removeLast() }
                }
            }
            let topDesc = topOut.map { e -> String in
                let h = e.idx / dv
                let i = e.idx % dv
                let sumSq = (0..<dv).reduce(0.0) { acc, j in
                    let v = yv(h * dv + j)
                    return acc + v * v
                }
                let rms = (sumSq / Double(dv)).squareRoot()
                let formula = (rms > 0 ? (yv(e.idx) / rms) : 0) * wv(i) * Self.siluDouble(zv(e.idx))
                return "[i\(e.idx)/h\(h) out=\(String(format: "%.4f", e.o)) "
                    + "y=\(String(format: "%.4f", yv(e.idx))) z=\(String(format: "%.3f", zv(e.idx))) "
                    + "W=\(String(format: "%.3f", wv(i))) rms=\(String(format: "%.4f", rms)) "
                    + "formula=\(String(format: "%.4f", formula))]"
            }.joined(separator: " ")
            print("[GDN-\(label)-L\(layer)] align maxPredHead\(predHead)=\(String(format: "%.4f", maxPred)) "
                  + "maxActHead\(actHead)=\(String(format: "%.4f", maxAct)) spikes \(spikeDesc)")
            print("[GDN-\(label)-L\(layer)] topout \(topDesc)")

            // Reference delta-rule replay for the top-output positions:
            //   S   = S_before[h][dv] * g
            //   kv  = S[h][dv] . k[hk]
            //   dlt = (v[h][dv] - kv) * beta
            //   S'  = S + k[hk] * dlt
            //   y   = S'[h][dv] . q[hk]
            // Uses the pre-step state snapshot (debugStateBefore), q/k/v from
            // convOut (post qk-norm), and a/b/A_log/dt_bias.
            if let stateBefore = debugStateBefore,
               let convOut = gdnConvOut,
               let aProj = gdnA, let bProj = gdnB,
               let aLogV = try? model.linearALog(layer: layer),
               let dtV = try? model.linearDtBias(layer: layer) {
                let Dk = la.keyHeadDim
                let Dv = la.valueHeadDim
                let Hk = la.numKHeads
                let Hv = la.numVHeads
                let ratio = Hv / Hk
                func cval(_ i: Int) -> Double {
                    Self.float16ToDouble(convOut.contents().load(fromByteOffset: i * 2, as: UInt16.self))
                }
                let sRaw2 = stateBefore.contents()
                func sval(_ h: Int, _ dv: Int, _ j: Int) -> Double {
                    Double(sRaw2.load(fromByteOffset: ((h * Dv + dv) * Dk + j) * 4, as: Float.self))
                }
                func aval(_ h: Int) -> Double {
                    Self.float16ToDouble(aProj.contents().load(fromByteOffset: h * 2, as: UInt16.self))
                }
                func bval(_ h: Int) -> Double {
                    Self.float16ToDouble(bProj.contents().load(fromByteOffset: h * 2, as: UInt16.self))
                }
                let aLogRaw = aLogV.buffer.contents().advanced(by: Int(aLogV.offset))
                let dtRaw = dtV.buffer.contents().advanced(by: Int(dtV.offset))
                func aLogval(_ h: Int) -> Double {
                    Double(Quantization.bf16ToFloat(aLogRaw.load(fromByteOffset: h * 2, as: UInt16.self)))
                }
                func dtval(_ h: Int) -> Double {
                    Double(Quantization.bf16ToFloat(dtRaw.load(fromByteOffset: h * 2, as: UInt16.self)))
                }
                for e in topOut {
                    let i = e.idx
                    let h = i / Dv
                    let dv = i % Dv
                    let hk = h / ratio
                    let a = aval(h)
                    let b = bval(h)
                    let Alog = aLogval(h)
                    let dtb = dtval(h)
                    let softplus = (a + dtb) > 20.0 ? (a + dtb) : log(1.0 + exp(a + dtb))
                    let g = exp(-exp(Alog) * softplus)
                    let beta = 1.0 / (1.0 + exp(-b))
                    // convOut offsets (match gdn_delta_step_decode exactly):
                    //   q[hk][j] = convOut[hk*Dk + j]
                    //   k[hk][j] = convOut[Hk*Dk + hk*Dk + j]
                    //   v[h][dv] = convOut[2*Hk*Dk + h*Dv + dv] = convOut[2*Hk*Dk + i]
                    let qOff = hk * Dk
                    let kOff = Hk * Dk + hk * Dk
                    let vVal = cval(2 * Hk * Dk + i)
                    // kernel: s[i]=S_before*g ; kv += s[i]*k  =>  kv = g * Σ S_before*k
                    var kv = 0.0
                    for j in 0..<Dk { kv += (sval(h, dv, j) * g) * cval(kOff + j) }
                    let delta = (vVal - kv) * beta
                    // kernel: s[i]=s[i]+k[i]*delta ; out += s[i]*q
                    //        => y = Σ (S_before*g + k*delta) * q
                    var yRec = 0.0
                    for j in 0..<Dk {
                        let sNew = sval(h, dv, j) * g + cval(kOff + j) * delta
                        yRec += sNew * cval(qOff + j)
                    }
                    let yKernel = yv(i)
                    let rel = (yRec != 0) ? abs(yRec - yKernel) / max(abs(yRec), 1e-9) : abs(yRec - yKernel)
                    print("[GDN-\(label)-L\(layer)] ytest i\(i)/h\(h)/dv\(dv) yK=\(String(format: "%.5f", yKernel)) "
                          + "yR=\(String(format: "%.5f", yRec)) g=\(String(format: "%.4f", g)) "
                          + "beta=\(String(format: "%.4f", beta)) kv=\(String(format: "%.4f", kv)) "
                          + "delta=\(String(format: "%.4f", delta)) v=\(String(format: "%.4f", vVal)) "
                          + "match=\(rel <= 0.05 ? "YES" : "NO")")
                }
            }

            // Definitive z test: recompute z[i] = normed @ W_z^T (4-bit affine)
            // in the CPU for the top-output positions and compare to the
            // kernel's z. If they disagree, the in_proj_z GEMV (decode fused
            // path) is producing a wrong z; if they agree, z is correct and
            // the anomaly is in y (the delta-rule output).
            if let zView = try? model.linearInProjZ(layer: layer) {
                let D = cfg.hiddenSize
                let groups = D / 64
                let normRaw = normed.contents()
                func nx(_ j: Int) -> Double {
                    Self.float16ToDouble(normRaw.load(fromByteOffset: j * 2, as: UInt16.self))
                }
                let wRaw = zView.buffer.contents().advanced(by: Int(zView.offset))
                let sRaw = zView.buffer.contents().advanced(by: Int(zView.scaleOffset))
                let bRaw = zView.buffer.contents().advanced(by: Int(zView.biasOffset))
                for e in topOut {
                    let i = e.idx
                    var zrec = 0.0
                    for g in 0..<groups {
                        let scale = Double(Quantization.bf16ToFloat(
                            sRaw.load(fromByteOffset: (i * groups + g) * 2, as: UInt16.self)))
                        let bias = Double(Quantization.bf16ToFloat(
                            bRaw.load(fromByteOffset: (i * groups + g) * 2, as: UInt16.self)))
                        var dot = 0.0
                        var sumx = 0.0
                        for j in 0..<64 {
                            let col = g * 64 + j
                            let wbyte = wRaw.load(fromByteOffset: i * (D / 2) + col / 2,
                                                  as: UInt8.self)
                            let nib = (j % 2 == 0) ? Double(wbyte & 0x0F)
                                                   : Double((wbyte >> 4) & 0x0F)
                            let x = nx(col)
                            dot += nib * x
                            sumx += x
                        }
                        zrec += scale * dot + bias * sumx
                    }
                    let zk = zv(i)
                    let ok = abs(zrec - zk) <= max(1e-3, 0.02 * abs(zrec))
                    print("[GDN-\(label)-L\(layer)] ztest i\(i) zKernel=\(String(format: "%.4f", zk)) "
                          + "zRecomputed=\(String(format: "%.4f", zrec)) match=\(ok ? "YES" : "NO")")
                }
            }

            // Same test for the qkv projection, row i (v[h][dv] lives at
            // convOut[2*Hk*Dk + i] pre-conv). If the in_proj_qkv GEMV is
            // producing a wrong v, the y spike (which comes from state·q /
            // the delta update with this v) is explained at the source.
            if let qkvView = try? model.linearInProjQKV(layer: layer),
               let qkvRaw = gdnQKVRaw {
                let D = cfg.hiddenSize
                let groups = D / 64
                let Hk = la.numKHeads
                let Dk = la.keyHeadDim
                let normRaw = normed.contents()
                func nx(_ j: Int) -> Double {
                    Self.float16ToDouble(normRaw.load(fromByteOffset: j * 2, as: UInt16.self))
                }
                let wRaw = qkvView.buffer.contents().advanced(by: Int(qkvView.offset))
                let sRaw = qkvView.buffer.contents().advanced(by: Int(qkvView.scaleOffset))
                let bRaw = qkvView.buffer.contents().advanced(by: Int(qkvView.biasOffset))
                for e in topOut {
                    let row = 2 * Hk * Dk + e.idx
                    var vrec = 0.0
                    for g in 0..<groups {
                        let scale = Double(Quantization.bf16ToFloat(
                            sRaw.load(fromByteOffset: (row * groups + g) * 2, as: UInt16.self)))
                        let bias = Double(Quantization.bf16ToFloat(
                            bRaw.load(fromByteOffset: (row * groups + g) * 2, as: UInt16.self)))
                        var dot = 0.0
                        var sumx = 0.0
                        for j in 0..<64 {
                            let col = g * 64 + j
                            let wbyte = wRaw.load(fromByteOffset: row * (D / 2) + col / 2,
                                                  as: UInt8.self)
                            let nib = (j % 2 == 0) ? Double(wbyte & 0x0F)
                                                   : Double((wbyte >> 4) & 0x0F)
                            let x = nx(col)
                            dot += nib * x
                            sumx += x
                        }
                        vrec += scale * dot + bias * sumx
                    }
                    let vk = Self.float16ToDouble(qkvRaw.contents().load(
                        fromByteOffset: row * 2, as: UInt16.self))
                    let ok = abs(vrec - vk) <= max(1e-3, 0.02 * abs(vrec))
                    print("[GDN-\(label)-L\(layer)] vtest row\(row) vKernel=\(String(format: "%.4f", vk)) "
                          + "vRecomputed=\(String(format: "%.4f", vrec)) match=\(ok ? "YES" : "NO")")
                }
            }

                  // convtest: replay gdn_conv_mix_decode + gdn_qk_norm in the CPU
                  // and compare to the kernel's gdnConvOut.
                  // gdn_conv_mix_decode:171 applies silu to ALL channels (q/k/v),
                  // so gdnConvOut holds silu(conv_acc) everywhere; gdn_qk_norm
                  // then norms only the q and k slices (v passes through).
                  //   v slice [2*Hk*Dk, qkvDim):            convOut[c] = silu(conv_acc[c])
                  //   q slice [0, Hk*Dk):                   convOut[q] = silu(conv_acc) * invRms * (1/Dk)
                  //   k slice [Hk*Dk, 2*Hk*Dk):            convOut[k] = silu(conv_acc) * invRms * rsqrt(Dk)
                  //   conv_acc[c] = qkvRaw[c]*convW[c,K-1] + sum_{j=0..K-2} tail[j*C+c]*convW[c,j]
                  //   invRms = rsqrt(mean( silu(conv_acc)^2 ) + eps), over the Dk of that head.
                  //       (per gdn_conv_mix_decode:153-179, gdn_qk_norm:276-320).
                  //   tail is the pre-step snapshot in debugTailBefore;
                  //   convW is BF16 [qkvDim, K] (K = convKernelSize).
            if let convView = try? model.linearConv1d(layer: layer),
               let convOut = gdnConvOut,
               let tailSnap = debugTailBefore,
               let qkvRaw = gdnQKVRaw {
              let C = la.qkvDim
              let K = la.convKernelSize
              let Dk = la.keyHeadDim
              let Hk = la.numKHeads
              let rmsEps: Double = 1e-6
              func convWv(_ c: Int, _ j: Int) -> Double {
                Double(Quantization.bf16ToFloat(
                  convView.buffer.contents().advanced(by: Int(convView.offset))
                          .load(fromByteOffset: (c * K + j) * 2, as: UInt16.self)))
              }
              func qkvRawV(_ c: Int) -> Double {
                Self.float16ToDouble(qkvRaw.contents().load(
                  fromByteOffset: c * 2, as: UInt16.self))
              }
              func tailV(_ j: Int, _ c: Int) -> Double {
                Self.float16ToDouble(tailSnap.contents().load(
                  fromByteOffset: (j * C + c) * 2, as: UInt16.self))
              }
                // conv_acc per gdn_conv_mix_decode:167-170
              func convAcc(_ c: Int) -> Double {
                var acc = qkvRawV(c) * convWv(c, K - 1)
                for j in 0..<(K - 1) {
                  acc += tailV(j, c) * convWv(c, j)
                }
                return acc
              }
                // gdn_conv_mix_decode:171 applies silu to every channel output
              func convOutRec(_ c: Int) -> Double {
                Self.siluDouble(convAcc(c))
              }
              func convOutK(_ c: Int) -> Double {
                Self.float16ToDouble(convOut.contents().load(
                  fromByteOffset: c * 2, as: UInt16.self))
              }
                // 1) v slice: topOut idx are v-slice elements; compare
                //    silu(conv_acc) to the kernel's gdnConvOut.
              for e in topOut {
                let c = 2 * Hk * Dk + e.idx
                let rec = convOutRec(c)
                let kv = convOutK(c)
                let ok = abs(rec - kv) <= max(1e-3, 0.02 * abs(rec))
                print("[GDN-\(label)-L\(layer)] convtest-v c\(c) convK=\(String(format: "%.4f", kv)) "
                      + "convR=\(String(format: "%.4f", rec)) match=\(ok ? "YES" : "NO")")
              }
                // 2) q slice: head 0, first Dk channels.
                //    qk_norm: x*invRms*(1/Dk), where x = silu(conv_acc) and
                //    invRms = rsqrt(mean(x^2)+eps) over those Dk channels.
              do {
                let base = 0
                var sumSq = 0.0
                for i in 0..<Dk {
                  let x = convOutRec(base + i)
                  sumSq += x * x
                }
                let invRms = 1.0 / sqrt(sumSq / Double(Dk) + rmsEps)
                var maxErr = 0.0
                for i in 0..<Dk {
                  let rec = convOutRec(base + i) * invRms * (1.0 / Double(Dk))
                  let kv = convOutK(base + i)
                  let er = abs(rec - kv)
                  if er > maxErr { maxErr = er }
                }
                let ok = maxErr <= max(1e-3, 0.02)
                print("[GDN-\(label)-L\(layer)] convtest-q h0 Dk\(Dk) maxErr=\(String(format: "%.4f", maxErr)) "
                      + "match=\(ok ? "YES" : "NO")")
              }
                // 3) k slice: head 0, channels [Hk*Dk, 2*Hk*Dk).
                //    qk_norm: x*invRms*rsqrt(Dk), x = silu(conv_acc).
              do {
                let base = Hk * Dk
                var sumSq = 0.0
                for i in 0..<Dk {
                  let x = convOutRec(base + i)
                  sumSq += x * x
                }
                let invRms = 1.0 / sqrt(sumSq / Double(Dk) + rmsEps)
                var maxErr = 0.0
                for i in 0..<Dk {
                  let rec = convOutRec(base + i) * invRms * (1.0 / sqrt(Double(Dk)))
                  let kv = convOutK(base + i)
                  let er = abs(rec - kv)
                  if er > maxErr { maxErr = er }
                }
                let ok = maxErr <= max(1e-3, 0.02)
                print("[GDN-\(label)-L\(layer)] convtest-k h0 Dk\(Dk) maxErr=\(String(format: "%.4f", maxErr)) "
                      + "match=\(ok ? "YES" : "NO")")
              }
             }
        }

        guard includeShared, let gdnConvOut, let gdnY, let gdnA, let gdnB,
              let gdnZ, let gdnOut else { return }
        dumpGDNBuffer(label: label, layer: layer, name: "convOut",
                      buffer: gdnConvOut, count: la.qkvDim)
        dumpGDNBuffer(label: label, layer: layer, name: "y",
                      buffer: gdnY, count: la.valueDim)
        dumpGDNBuffer(label: label, layer: layer, name: "z",
                      buffer: gdnZ, count: la.valueDim)
        dumpGDNBuffer(label: label, layer: layer, name: "gdnOut",
                      buffer: gdnOut, count: la.valueDim)
        dumpGDNBuffer(label: label, layer: layer, name: "a",
                      buffer: gdnA, count: la.numVHeads)
        dumpGDNBuffer(label: label, layer: layer, name: "b",
                      buffer: gdnB, count: la.numVHeads)
    }

    private func dumpGDNBuffer(label: String, layer: Int, name: String,
                               buffer: MTLBuffer, count: Int) {
        let raw = buffer.contents()
        var sumSq = 0.0
        var maxAbs = 0.0
        var nans = 0
        for i in 0..<count {
            let bits = raw.load(fromByteOffset: i * MemoryLayout<Float16>.stride,
                                as: UInt16.self)
            let v = Self.float16ToDouble(bits)
            if v.isNaN || v.isInfinite { nans &+= 1; continue }
            sumSq += v * v
            if abs(v) > maxAbs { maxAbs = abs(v) }
        }
        print("[GDN-\(label)-L\(layer)] \(name) rms=\(String(format: "%.5f", (sumSq / Double(count)).squareRoot())) maxAbs=\(String(format: "%.5f", maxAbs)) nanInf=\(nans)")
    }

    /// Read back a hidden residual (FP16, [count]) and print its RMS and
    /// max-abs value. Driven by `debugRmsDump` (`TURBO_RMS_DUMP=1`) to locate
    /// a corrupted layer by the first RMS blow-up when a repacked model emits
    /// garbage. The read is raw-byte based so it does not depend on a
    /// `Float16.bitPattern` API that is absent on some toolchains.
    private func dumpHiddenRMS(label: String, layer: Int,
                               buffer: MTLBuffer, count: Int) {
        let raw = buffer.contents()
        var sumSq = 0.0
        var maxAbs = 0.0
        var nans = 0
        for i in 0..<count {
            let bits = raw.load(fromByteOffset: i * MemoryLayout<Float16>.stride,
                                as: UInt16.self)
            let v = Self.float16ToDouble(bits)
            if v.isNaN { nans &+= 1; continue }
            sumSq += v * v
            if abs(v) > maxAbs { maxAbs = abs(v) }
        }
        let rms = (sumSq / Double(count)).squareRoot()
        let kind = cfg.layerIsLinear(layer) ? "GDN"
                 : (cfg.fullAttentionLayerMask[layer] == 1 ? "FULL" : "SWA")
        print("[RMS-\(label)-L\(layer)/\(kind)] rms=\(String(format: "%.5f", rms)) "
             + "maxAbs=\(String(format: "%.5f", maxAbs)) nan=\(nans)")
    }

    private static func siluDouble(_ x: Double) -> Double {
        x / (1.0 + exp(-x))
    }

    /// Decode an IEEE-754 half-precision (Float16) bit pattern to Double.
    /// `Float16` has no stable `Double(_:)` initializer across toolchains, so
    /// the 1/5/10-bit layout is decoded directly to guarantee a correct,
    /// portable conversion for the debug readback.
    private static func float16ToDouble(_ bits: UInt16) -> Double {
        let sign = (bits & 0x8000) != 0 ? -1.0 : 1.0
        let exp = Int((bits >> 10) & 0x1F)
        let mant = Int(bits & 0x3FF)
        if exp == 0x1F {
            return mant == 0 ? sign * Double.infinity : Double.nan
        }
        if exp == 0 {
            // Subnormal: value = mant * 2^-24.
            return sign * Double(mant) * 0x1.0p-24
        }
        // Normal: value = (1 + mant/1024) * 2^(exp-15).
        return sign * (1.0 + Double(mant) / 1024.0) * pow(2.0, Double(exp) - 15.0)
    }

}
