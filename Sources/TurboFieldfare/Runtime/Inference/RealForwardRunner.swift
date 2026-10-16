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
    /// Router (`mlp.gate`) weight bits from the manifest. 8 on Qwen 3.6, 4 on
    /// Qwen 3.5. Selects the INT8 vs INT4 router GEMV kernels and the scalar
    /// shared-expert gate path (the scalar gate shares the router's width in
    /// both pinned checkpoints).
    private let routerWeightBits: Int

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
    // Qwen 3.8 decode state (nil on other families). Hyper-connection stream
    // buffers are [hc * hidden] FP16; QSA pooled keys persist per QSA layer
    // across tokens ([maxBlocks, indexerHeadDim] FP16, position-indexed).
    private let q38: Qwen38Kernels?
    private let pleReader: PleShardReader?
    private let q38Hyper: MTLBuffer?
    private let q38HyperNormed: MTLBuffer?
    private let q38Low: MTLBuffer?
    private let q38Mix: MTLBuffer?
    private let q38AttnIn: MTLBuffer?
    private let q38MlpIn: MTLBuffer?
    private let q38Inj: MTLBuffer?
    private let q38IdxQK: MTLBuffer?
    private let q38IdxQ: MTLBuffer?
    private let q38PooledKeys: [MTLBuffer]
    private let q38Scores: MTLBuffer?
    private let q38Sel: MTLBuffer?
    private let q38CompactK: MTLBuffer?
    private let q38CompactV: MTLBuffer?
    private let q38PleEmbed: MTLBuffer?
    private let q38PleKey: MTLBuffer?
    private let q38PleValue: MTLBuffer?
    private let q38PleQuery: MTLBuffer?
    private let q38PleDots: MTLBuffer?
    private let q38PleGate: MTLBuffer?
    private let q38PleGated: MTLBuffer?
    private let q38PleConvOut: MTLBuffer?
    private let q38PleConvState: MTLBuffer?
    private let q38PleEos: Int32
    private var q38PleHistory: [Int32]
    private var q38IdxTails: [[Float]]
    private var prefillChunkState = PrefillChunkCommitState()
    private var prefillScratch: PrefillChunkScratchBuffers?

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
                runtimeConfiguration: RuntimeConfiguration = .production,
                pleEosTokenId: Int = 248046) throws {
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
        self.routerWeightBits = model.routerWeightBits
        self.embedInt4 = try EmbedLookupInt4(context: context)
        self.rms       = try RMSNorm(context: context)
        // A 4-bit router (Qwen 3.5) runs through the INT4 GEMV, so its
        // shapes join the specialization set alongside the scalar gate's.
        var int4Shapes = cfg.decodeInt4GEMVShapes
        if model.routerWeightBits == 4 {
            int4Shapes.append((m: cfg.numExperts, n: cfg.hiddenSize))
            int4Shapes.append((m: 1, n: cfg.hiddenSize))
        }
        self.int4      = try DequantInt4GEMV(
            context: context,
            additionalShapes: int4Shapes)
        self.attention = try Attention(context: context)
        self.shared    = try SharedExpertRuntime(context: context,
                                                  weightBits: model.sharedExpertWeightBits,
                                                  siluActivation: silu)
        self.moe       = try MoE(context: context,
                                 siluActivation: silu,
                                 specializedD: UInt32(cfg.hiddenSize),
                                 specializedF: UInt32(cfg.moeIntermediateSize),
                                 specializedNumExperts: UInt32(cfg.numExperts),
                                 specializedTopK: UInt32(cfg.topKExperts),
                                 routedGroupSize: model.routedExpertGroupSize)
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
                               specializedHiddenSize: cfg.hiddenSize,
                               projGroupSize: model.attentionGroupSize)
            self.gdnState = try GDNStateManager(device: context.device, config: cfg)
        } else {
            self.gdn = nil
            self.gdnState = nil
        }
        self.rope = cfg.ropeNeoxSubdim ? try RoPE(context: context) : nil
        // The scalar shared-expert gate shares the router's quantization
        // width (8-bit on Qwen 3.6, 4-bit on Qwen 3.5); the 4-bit gate runs
        // through the INT4 GEMV instead.
        self.int8ScalarGate = (cfg.sharedExpertGated && model.routerWeightBits == 8)
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

        // Qwen 3.8 decode scratch — allocated once here, never in the hot path.
        if cfg.family == .qwen38 {
            let hc = cfg.hcCount
            let hcHidden = hc * D
            let lowrank = cfg.hcLowrank
            let idxDim = cfg.indexerHeadDim
            let idxHeads = cfg.indexerNHeads
            let idxKvHeads = cfg.indexerKvHeads
            let idxRows = idxHeads * idxDim + idxKvHeads * idxDim
            let compress = max(cfg.indexerCompressRatio, 1)
            let maxBlocks = (maxContext + compress - 1) / compress
            let selMax = cfg.indexerBudget + compress
            let blockTopK = cfg.indexerBudget / compress
            let compactRows = blockTopK * compress + compress
            let kvElems = cfg.numFullKVHeads * cfg.fullHeadDim
            // PLE short-conv state: (kernel - 1) * dilation rows of the
            // gated values. Production geometry (k=4, dilation=ngram_size=3)
            // holds 9 rows; the kernel takes the lengths as parameters.
            let pleConvStateLen = 9
            self.q38 = try Qwen38Kernels(context: context)
            self.pleReader = try PleShardReader(directoryURL: model.directoryURL)
            self.q38Hyper = try buf(hcHidden)
            self.q38HyperNormed = try buf(hcHidden)
            self.q38Low = try buf(lowrank)
            self.q38Mix = try buf(hcHidden)
            self.q38AttnIn = try buf(D)
            self.q38MlpIn = try buf(D)
            self.q38Inj = try buf(hc)
            self.q38IdxQK = try buf(idxRows)
            self.q38IdxQ = try buf(idxHeads * idxDim)
            var pooled: [MTLBuffer] = []
            pooled.reserveCapacity(cfg.numLayers)
            for L in 0..<cfg.numLayers {
                if cfg.layerIsFull(L) {
                    pooled.append(try buf(maxBlocks * idxDim))
                } else {
                    // Non-QSA layers carry no pooled index keys; keep the
                    // array layer-indexed with a 1-element placeholder.
                    pooled.append(try buf(1))
                }
            }
            self.q38PooledKeys = pooled
            self.q38Scores = try buf(maxBlocks, MemoryLayout<Float>.stride)
            self.q38Sel = try buf(selMax, MemoryLayout<UInt32>.stride)
            self.q38CompactK = try buf(compactRows * kvElems)
            self.q38CompactV = try buf(compactRows * kvElems)
            self.q38PleEmbed = try buf(cfg.pleEmbedDim)
            self.q38PleKey = try buf(hcHidden)
            self.q38PleValue = try buf(D)
            self.q38PleQuery = try buf(hcHidden)
            self.q38PleDots = try buf(hc, MemoryLayout<Float>.stride)
            self.q38PleGate = try buf(hc, MemoryLayout<Float>.stride)
            self.q38PleGated = try buf(hcHidden)
            self.q38PleConvOut = try buf(hcHidden)
            self.q38PleConvState = try buf(pleConvStateLen * hcHidden)
            memset(self.q38PleConvState!.contents(), 0, self.q38PleConvState!.length)
            self.q38PleEos = Int32(pleEosTokenId)
            self.q38PleHistory = []
            self.q38IdxTails = [[Float]](repeating: [], count: cfg.numLayers)
        } else {
            self.q38 = nil
            self.pleReader = nil
            self.q38Hyper = nil
            self.q38HyperNormed = nil
            self.q38Low = nil
            self.q38Mix = nil
            self.q38AttnIn = nil
            self.q38MlpIn = nil
            self.q38Inj = nil
            self.q38IdxQK = nil
            self.q38IdxQ = nil
            self.q38PooledKeys = []
            self.q38Scores = nil
            self.q38Sel = nil
            self.q38CompactK = nil
            self.q38CompactV = nil
            self.q38PleEmbed = nil
            self.q38PleKey = nil
            self.q38PleValue = nil
            self.q38PleQuery = nil
            self.q38PleDots = nil
            self.q38PleGate = nil
            self.q38PleGated = nil
            self.q38PleConvOut = nil
            self.q38PleConvState = nil
            self.q38PleEos = 0
            self.q38PleHistory = []
            self.q38IdxTails = []
        }

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
    }

    public func reset() {
        kv?.reset()
        gdnState?.reset()
        resetTransientState()
        // Qwen 3.8: PLE token history, indexer raw tails, and the PLE short-
        // conv state restart. Pooled index keys are position-indexed, so stale
        // rows are simply overwritten as blocks complete again.
        q38PleHistory = []
        q38IdxTails = [[Float]](repeating: [], count: cfg.numLayers)
        if let st = q38PleConvState {
            memset(st.contents(), 0, st.length)
        }
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

        // Qwen 3.8 correctness-first staging: chunked prefill advances the
        // same per-token state machine as decode (KV + GDN recurrence + PLE
        // history + indexer pools), one token at a time. Batched QSA/hyper
        // prefill kernels are a follow-up optimization; state equivalence
        // with pure decode holds by construction.
        if cfg.family == .qwen38 {
            var pos = startPosition
            let lastIndex = tokens.index(tokens.startIndex, offsetBy: tokens.count - 1)
            for idx in tokens.indices {
                let isLast = (idx == lastIndex)
                try await produceTokenQwen38(
                    token: tokens[idx], position: pos, into: logits,
                    emitHead: isLast,
                    outputMode: isLast ? outputMode : .greedyIfAvailable)
                pos += 1
                onProgress(pos - startPosition)
            }
            if outputMode == .greedyIfAvailable, useFusedGreedyHead {
                return PrefillResult(newPosition: pos, seed: .greedyToken(lastGreedyToken))
            }
            return PrefillResult(newPosition: pos, seed: .logitsWritten)
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
                        weightBits: routerWeightBits)

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
                            encodeScalarGate(sharedCB,
                                             gateView: gateView,
                                             x: scratch.routedX,
                                             xOffset: row * D * halfBytes,
                                             y: scratch.sharedScalarGate,
                                             yOffset: row * halfBytes,
                                             d: D)
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
        // Qwen 3.8 runs the hyper-connection / QSA / PLE graph instead of the
        // Gemma/Qwen3.5-3.6 residual graph below.
        if cfg.family == .qwen38 {
            try await produceTokenQwen38(token: token, position: position,
                                         into: logits, emitHead: emitHead,
                                         outputMode: outputMode)
            return
        }
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
                weightBits: routerWeightBits)
            cb.commit()
            let tWait = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            waitForCompletion(cb)
            let waitNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tWait
            if let pending = pendingRoutedCommand {
                finishPendingRoutedCommand(pending, waitIfNeeded: false)
                pendingRoutedCommand = nil
            }
            totalCb1Nanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tCb1Start - waitNanos

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
                encodeScalarGate(sharedCB,
                                 gateView: gateView,
                                 x: routedX, xOffset: 0,
                                 y: sharedScalarGateBuf!, yOffset: 0,
                                 d: cfg.hiddenSize)
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
    private func encodeLinearAttentionDecode(_ cb: MTLCommandBuffer, layer L: Int,
                                             from input: MTLBuffer? = nil,
                                             outGroupSize: Int = 64) throws {
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
                                   x: input ?? normed,
                                   qkv: qkvW, qkvOut: gdnQKVRaw,
                                   z: zW, zOut: gdnZ,
                                   a: aW, aOut: gdnA,
                                   b: bW, bOut: gdnB,
                                   hiddenSize: cfg.hiddenSize)

        gdn.encodeConvDecode(commandBuffer: cb,
                             tail: gdnState.convTailBuffer(layer: L),
                             qkv: gdnQKVRaw,
                             convWeight: convW.buffer,
                             convWeightOffset: Int(convW.offset),
                             out: gdnConvOut)
        gdn.encodeQKNorm(commandBuffer: cb, convOut: gdnConvOut)
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
                    x: gdnOut, y: oOut, m: D, n: UInt32(la.valueDim),
                    groupSize: outGroupSize)
    }

    /// Scalar shared-expert gate GEMV (`mlp.shared_expert_gate`, [1, D]).
    /// The gate shares the router's quantization width: INT8 on Qwen 3.6,
    /// INT4 on Qwen 3.5.
    private func encodeScalarGate(_ cb: MTLCommandBuffer,
                                  gateView: TensorView,
                                  x: MTLBuffer, xOffset: Int,
                                  y: MTLBuffer, yOffset: Int,
                                  d: Int) {
        if routerWeightBits == 4 {
            int4.encode(commandBuffer: cb,
                        weights: gateView.buffer,
                        weightsOffset: Int(gateView.offset),
                        scales: gateView.buffer,
                        scalesOffset: Int(gateView.scaleOffset),
                        biases: gateView.buffer,
                        biasesOffset: Int(gateView.biasOffset),
                        x: x, xOffset: xOffset,
                        y: y, yOffset: yOffset,
                        m: 1, n: UInt32(d))
        } else {
            int8ScalarGate!.encode(commandBuffer: cb,
                                   weights: gateView.buffer,
                                   weightsOffset: Int(gateView.offset),
                                   scales: gateView.buffer,
                                   scalesOffset: Int(gateView.scaleOffset),
                                   biases: gateView.buffer,
                                   biasesOffset: Int(gateView.biasOffset),
                                   x: x, xOffset: xOffset,
                                   y: y, yOffset: yOffset,
                                   m: 1, n: UInt32(d))
        }
    }

    /// Gated full attention (Qwen): packed
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


    // MARK: - Qwen 3.8 decode (hyper-connection / QSA / PLE)

    @inline(__always)
    private static func q38F16(_ x: Float) -> UInt16 { Float16(x).bitPattern }

    @inline(__always)
    private static func q38F32(_ b: UInt16) -> Float { Float(Float16(bitPattern: b)) }

    /// Temporary env-gated probe (TURBO_FIELDFARE_DEBUG_QWEN38=1): min/max/
    /// meanabs plus NaN/Inf flags of an FP16 shared buffer, to stderr.
    /// Zero overhead when the flag is unset.
    private func q38DebugDump(_ label: String, _ msg: String) {
        if let d = ("[q38dbg] " + label + " " + msg + "\n").data(using: .utf8) {
            FileHandle.standardError.write(d)
        }
    }

    /// Dump resident INT4 scales/biases (BF16) stats: count check plus
    /// min/max/mean. Catches misaligned aux reads (zeros/garbage).
    private func q38DebugDumpQuant(_ label: String, _ view: TensorView,
                                  rows: Int, groupsPerRow: Int) {
        let expect = rows * groupsPerRow
        let haveS = Int(view.scaleLength) / 2
        let haveB = Int(view.biasLength) / 2
        let sPtr = view.buffer.contents().advanced(by: Int(view.scaleOffset))
            .assumingMemoryBound(to: UInt16.self)
        let bPtr = view.buffer.contents().advanced(by: Int(view.biasOffset))
            .assumingMemoryBound(to: UInt16.self)
        var sMin = Float.infinity, sMax = -Float.infinity, sAcc: Float = 0
        var bMin = Float.infinity, bMax = -Float.infinity, bAcc: Float = 0
        for i in 0..<min(expect, min(haveS, haveB)) {
            let sv = Quantization.bf16ToFloat(sPtr[i])
            let bv = Quantization.bf16ToFloat(bPtr[i])
            sMin = min(sMin, sv); sMax = max(sMax, sv); sAcc += sv
            bMin = min(bMin, bv); bMax = max(bMax, bv); bAcc += bv
        }
        let c = Float(max(1, min(expect, min(haveS, haveB))))
        var first = ""
        for i in 0..<min(4, min(expect, min(haveS, haveB))) {
            first += "s\(Quantization.bf16ToFloat(sPtr[i]))/b\(Quantization.bf16ToFloat(bPtr[i])) "
        }
        q38DebugDump(label, "expScales=\(expect) haveS=\(haveS) haveB=\(haveB) "
            + "sMin=\(sMin) sMax=\(sMax) sMean=\(sAcc/c) "
            + "bMin=\(bMin) bMax=\(bMax) bMean=\(bAcc/c) first=[\(first)]")
    }

    /// Dump resident BF16 weight stats (norms should sit near 0 = scale ~1).
    private func q38DebugDumpBF16(_ label: String, _ view: TensorView, count: Int) {
        let have = Int(view.length) / 2
        let ptr = view.buffer.contents().advanced(by: Int(view.offset))
            .assumingMemoryBound(to: UInt16.self)
        var mn = Float.infinity, mx = -Float.infinity, acc: Float = 0
        for i in 0..<min(count, have) {
            let v = Quantization.bf16ToFloat(ptr[i])
            mn = min(mn, v); mx = max(mx, v); acc += v
        }
        let c = Float(max(1, min(count, have)))
        q38DebugDump(label, "exp=\(count) have=\(have) min=\(mn) max=\(mx) mean=\(acc/c)")
    }

    /// Env-gated group-32 GEMV self-test (synthetic, production widths).
    /// Row 0: q=1, s=0.01, b=0, x=1 → y = N*0.01. Row 1: q=2, s=0.01,
    /// b=0.5, x=1 → y = N*0.52. N=2560 covers the fast path (80 groups);
    /// N=96 covers the remainder path (3 groups). Prints PASS/FAIL.
    private func q38DebugSelfTestGEMV() {
        func runCase(n: Int, m: Int, q: UInt8, s: Float, b: Float) -> (Float, Float) {
            let g = 32
            let groups = n / g
            let wBytes = m * n / 2
            let aBytes = m * groups * 2
            let dev = ctx.device
            guard let wBuf = dev.makeBuffer(length: wBytes, options: .storageModeShared),
                  let sBuf = dev.makeBuffer(length: aBytes, options: .storageModeShared),
                  let bBuf = dev.makeBuffer(length: aBytes, options: .storageModeShared),
                  let xBuf = dev.makeBuffer(length: n * 2, options: .storageModeShared),
                  let yBuf = dev.makeBuffer(length: m * 2, options: .storageModeShared) else {
                return (Float.nan, Float.nan)
            }
            let fill: UInt8 = (q & 0x0F) | ((q & 0x0F) << 4)
            memset(wBuf.contents(), Int32(fill), wBytes)
            let sBits = Quantization.bf16Bits(s)
            let bBits = Quantization.bf16Bits(b)
            let sPtr = sBuf.contents().assumingMemoryBound(to: UInt16.self)
            let bPtr = bBuf.contents().assumingMemoryBound(to: UInt16.self)
            for i in 0..<(m * groups) { sPtr[i] = sBits; bPtr[i] = bBits }
            let xPtr = xBuf.contents().assumingMemoryBound(to: UInt16.self)
            let one = Float16(1.0).bitPattern
            for i in 0..<n { xPtr[i] = one }
            let yPtr = yBuf.contents().assumingMemoryBound(to: UInt16.self)
            for i in 0..<m { yPtr[i] = 0 }
            runSync { cb in
                self.int4.encode(commandBuffer: cb,
                                 weights: wBuf, weightsOffset: 0,
                                 scales: sBuf, scalesOffset: 0,
                                 biases: bBuf, biasesOffset: 0,
                                 x: xBuf, y: yBuf,
                                 m: UInt32(m), n: UInt32(n),
                                 groupSize: 32)
            }
            let oPtr = yBuf.contents().assumingMemoryBound(to: UInt16.self)
            let y0 = Float(Float16(bitPattern: oPtr[0]))
            let expected = Float(n) * (Float(q) * s + b)
            return (y0, expected)
        }
        let (yA, eA) = runCase(n: 2560, m: 2, q: 1, s: 0.01, b: 0.0)
        let (yB, eB) = runCase(n: 96, m: 2, q: 2, s: 0.01, b: 0.5)
        let okA = abs(yA - eA) / max(1e-6, abs(eA)) < 0.02
        let okB = abs(yB - eB) / max(1e-6, abs(eB)) < 0.02
        q38DebugDump("gemv32", "fast y=\(yA) exp=\(eA) \(okA ? "PASS" : "FAIL") "
            + "rem y=\(yB) exp=\(eB) \(okB ? "PASS" : "FAIL")")
    }

    /// Env-gated CPU-vs-GPU check for the L0 shared-expert gate GEMV with
    /// REAL weights: dequantizes gate_proj on CPU (same formula as the
    /// Metal kernel) and compares against `denseScratchGate`. Catches
    /// nibble-order / scale-layout mismatches with production data that
    /// synthetic self-tests cannot see.
    private func q38DebugRefCheckGate(mlpIn: MTLBuffer, layer L: Int) {
        let proj = sharedExpertProjections[L]
        let rows = Int(proj.gate.rows)
        let cols = Int(proj.gate.cols)
        let g = 32
        guard cols % g == 0 else {
            q38DebugDump("refGate", "cols \(cols) not multiple of 32")
            return
        }
        let groups = cols / g
        let xPtr = mlpIn.contents().assumingMemoryBound(to: UInt16.self)
        var x = [Float](repeating: 0, count: cols)
        for i in 0..<cols { x[i] = Self.q38F32(xPtr[i]) }
        let wBase = proj.gate.weights.contents().advanced(by: proj.gate.weightsOffset)
        let sBase = proj.gate.scales.contents().advanced(by: proj.gate.scalesOffset)
        let bBase = proj.gate.biases.contents().advanced(by: proj.gate.biasesOffset)
        let rowBytes = cols / 2
        var maxDiff: Float = 0
        var sumDiff: Float = 0
        var sumAbs: Float = 0
        let yPtr = denseScratchGate.contents().assumingMemoryBound(to: UInt16.self)
        for r in 0..<rows {
            let wRow = wBase.advanced(by: r * rowBytes).assumingMemoryBound(to: UInt8.self)
            let sRow = sBase.advanced(by: r * groups * 2).assumingMemoryBound(to: UInt16.self)
            let bRow = bBase.advanced(by: r * groups * 2).assumingMemoryBound(to: UInt16.self)
            var packed = [UInt8](repeating: 0, count: rowBytes)
            var sc = [UInt16](repeating: 0, count: groups)
            var bi = [UInt16](repeating: 0, count: groups)
            for i in 0..<rowBytes { packed[i] = wRow[i] }
            for i in 0..<groups { sc[i] = sRow[i]; bi[i] = bRow[i] }
            let w = Quantization.dequantizeInt4Affine(
                Quantization.Int4AffineRow(packed: packed, scales: sc, biases: bi),
                n: cols, groupSize: g)
            var acc: Float = 0
            for i in 0..<cols { acc += w[i] * x[i] }
            let yGPU = Self.q38F32(yPtr[r])
            let d = abs(acc - yGPU)
            maxDiff = max(maxDiff, d)
            sumDiff += d
            sumAbs += abs(acc)
        }
        let meanDiff = sumDiff / Float(max(1, rows))
        let meanAbs = sumAbs / Float(max(1, rows))
        let rel = meanDiff / max(1e-6, meanAbs)
        q38DebugDump("refGate", "rows=\(rows) cols=\(cols) meanDiff=\(meanDiff) "
            + "maxDiff=\(maxDiff) rel=\(rel) \(rel < 0.05 ? "PASS" : "FAIL")")
    }

    /// Env-gated CPU-vs-GPU check for the L3 fused QKV projection with REAL
    /// weights: dequantizes q_proj on CPU and compares against
    /// `qPackedScratch` (raw packed [query;gate], pre-split). The fused QKV
    /// kernel is a different code path from the proven `int4.encode` and
    /// was never validated with production qwen38 dims (24Q/2KV, head 256).
    private func q38DebugRefCheckQPack(attnIn: MTLBuffer, qPacked: MTLBuffer,
                                      layer L: Int) {
        do {
            let q = try model.qProj(layer: L)
            let rows = 2 * cfg.numHeads * cfg.fullHeadDim
            let cols = cfg.hiddenSize
            let g = model.attentionGroupSize
            guard cols % g == 0 else {
                q38DebugDump("refQPack", "cols not multiple of group")
                return
            }
            let groups = cols / g
            let xPtr = attnIn.contents().assumingMemoryBound(to: UInt16.self)
            var x = [Float](repeating: 0, count: cols)
            for i in 0..<cols { x[i] = Self.q38F32(xPtr[i]) }
            let wBase = q.buffer.contents().advanced(by: Int(q.offset))
            let sBase = q.buffer.contents().advanced(by: Int(q.scaleOffset))
            let bBase = q.buffer.contents().advanced(by: Int(q.biasOffset))
            let rowBytes = cols / 2
            var maxDiff: Float = 0
            var sumDiff: Float = 0
            var sumAbs: Float = 0
            let checkRows = min(rows, 256)
            let yPtr = qPacked.contents().assumingMemoryBound(to: UInt16.self)
            for r in 0..<checkRows {
                let wRow = wBase.advanced(by: r * rowBytes).assumingMemoryBound(to: UInt8.self)
                let sRow = sBase.advanced(by: r * groups * 2).assumingMemoryBound(to: UInt16.self)
                let bRow = bBase.advanced(by: r * groups * 2).assumingMemoryBound(to: UInt16.self)
                var packed = [UInt8](repeating: 0, count: rowBytes)
                var sc = [UInt16](repeating: 0, count: groups)
                var bi = [UInt16](repeating: 0, count: groups)
                for i in 0..<rowBytes { packed[i] = wRow[i] }
                for i in 0..<groups { sc[i] = sRow[i]; bi[i] = bRow[i] }
                let w = Quantization.dequantizeInt4Affine(
                    Quantization.Int4AffineRow(packed: packed, scales: sc, biases: bi),
                    n: cols, groupSize: g)
                var acc: Float = 0
                for i in 0..<cols { acc += w[i] * x[i] }
                let yGPU = Self.q38F32(yPtr[r])
                let d = abs(acc - yGPU)
                maxDiff = max(maxDiff, d)
                sumDiff += d
                sumAbs += abs(acc)
            }
            let meanDiff = sumDiff / Float(max(1, checkRows))
            let meanAbs = sumAbs / Float(max(1, checkRows))
            let rel = meanDiff / max(1e-6, meanAbs)
            q38DebugDump("refQPack", "rows=\(checkRows)/\(rows) cols=\(cols) meanDiff=\(meanDiff) "
                + "maxDiff=\(maxDiff) rel=\(rel) \(rel < 0.05 ? "PASS" : "FAIL")")
        } catch {
            q38DebugDump("refQPack", "error \(error)")
        }
    }

    /// Shared CPU INT4-g32 GEMV helper for reference checks: dequantizes
    /// `rows` of `view` ([rows, cols], group 32) and dots each with `x`.
    private func q38DebugGEMV4(view: TensorView, rows: Int, cols: Int,
                              x: [Float]) -> [Float] {
        let g = 32
        let groups = cols / g
        let rowBytes = cols / 2
        let wBase = view.buffer.contents().advanced(by: Int(view.offset))
        let sBase = view.buffer.contents().advanced(by: Int(view.scaleOffset))
        let bBase = view.buffer.contents().advanced(by: Int(view.biasOffset))
        var y = [Float](repeating: 0, count: rows)
        for r in 0..<rows {
            let wRow = wBase.advanced(by: r * rowBytes).assumingMemoryBound(to: UInt8.self)
            let sRow = sBase.advanced(by: r * groups * 2).assumingMemoryBound(to: UInt16.self)
            let bRow = bBase.advanced(by: r * groups * 2).assumingMemoryBound(to: UInt16.self)
            var packed = [UInt8](repeating: 0, count: rowBytes)
            var sc = [UInt16](repeating: 0, count: groups)
            var bi = [UInt16](repeating: 0, count: groups)
            for i in 0..<rowBytes { packed[i] = wRow[i] }
            for i in 0..<groups { sc[i] = sRow[i]; bi[i] = bRow[i] }
            let w = Quantization.dequantizeInt4Affine(
                Quantization.Int4AffineRow(packed: packed, scales: sc, biases: bi),
                n: cols, groupSize: g)
            var acc: Float = 0
            for i in 0..<cols { acc += w[i] * x[i] }
            y[r] = acc
        }
        return y
    }

    /// L0 router check: CPU INT8-g64 logits + reference top-k vs the GPU
    /// `outIndices`/`outWeights`. Wrong expert selection destroys content
    /// while preserving magnitudes (softmax renormalizes anything).
    private func q38DebugRefCheckRouter(mlpIn: MTLBuffer, layer L: Int) {
        do {
            let routerW = try model.router(layer: L)
            let nExp = cfg.numExperts
            let d = cfg.hiddenSize
            let groups = d / 64
            let xPtr = mlpIn.contents().assumingMemoryBound(to: UInt16.self)
            var x = [Float](repeating: 0, count: d)
            for i in 0..<d { x[i] = Self.q38F32(xPtr[i]) }
            let wBase = routerW.buffer.contents().advanced(by: Int(routerW.offset))
            let sBase = routerW.buffer.contents().advanced(by: Int(routerW.scaleOffset))
            let bBase = routerW.buffer.contents().advanced(by: Int(routerW.biasOffset))
            var logits = [Float](repeating: 0, count: nExp)
            for r in 0..<nExp {
                let wRow = wBase.advanced(by: r * d).assumingMemoryBound(to: UInt8.self)
                let sRow = sBase.advanced(by: r * groups * 2).assumingMemoryBound(to: UInt16.self)
                let bRow = bBase.advanced(by: r * groups * 2).assumingMemoryBound(to: UInt16.self)
                var packed = [UInt8](repeating: 0, count: d)
                var sc = [UInt16](repeating: 0, count: groups)
                var bi = [UInt16](repeating: 0, count: groups)
                for i in 0..<d { packed[i] = wRow[i] }
                for i in 0..<groups { sc[i] = sRow[i]; bi[i] = bRow[i] }
                let w = Quantization.dequantizeInt8Affine(
                    Quantization.Int8AffineRow(packed: packed, scales: sc, biases: bi),
                    n: d, groupSize: 64)
                var acc: Float = 0
                for i in 0..<d { acc += w[i] * x[i] }
                logits[r] = acc
            }
            let ref = Qwen38Reference.routerTopK(logits: logits, topK: cfg.topKExperts)
            let iPtr = outIndices.contents().assumingMemoryBound(to: UInt32.self)
            let wPtr = outWeights.contents().assumingMemoryBound(to: UInt16.self)
            var idxOK = true
            var maxWDiff: Float = 0
            for i in 0..<cfg.topKExperts {
                if Int(iPtr[i]) != ref.indices[i] { idxOK = false }
                maxWDiff = max(maxWDiff, abs(Self.q38F32(wPtr[i]) - ref.weights[i]))
            }
            q38DebugDump("refRouter", "idxMatch=\(idxOK) maxWDiff=\(maxWDiff) "
                + "gpu0=\(iPtr[0]) ref0=\(ref.indices[0]) "
                + "\((idxOK && maxWDiff < 0.02) ? "PASS" : "FAIL")")
        } catch {
            q38DebugDump("refRouter", "error \(error)")
        }
    }

    /// CPU mirror of the GDN conv tail + recurrent state per layer, for the
    /// position-1 carry check. Saved during the position-0 step, consumed
    /// (and re-saved) at position 1. Debug-only.
    private var q38DebugGDNMirror: [Int: (tail: [[Float]], s: [[[Float]]])] = [:]

    /// L0 GDN check: CPU single-step (fresh state, position 0) reference vs
    /// GPU `gdnConvOut` (isolates in-proj + conv) and `oOut` (full delta
    /// rule + gated norm + out_proj). The delta rule for one fresh token is
    /// `out = beta * (q.k) * v` after L2 norms; straight from the reference.
    private func q38DebugRefCheckGDN(attnIn: MTLBuffer, oOut: MTLBuffer,
                                       layer L: Int, position p: Int) {
        do {
            let nK = cfg.linearAttention.numKHeads
            let nV = cfg.linearAttention.numVHeads
            let hK = cfg.linearAttention.keyHeadDim
            let hV = cfg.linearAttention.valueHeadDim
            let keyDim = nK * hK
            let valDim = nV * hV
            let qkvDim = keyDim * 2 + valDim
            let kernel = cfg.linearAttention.convKernelSize
            let d = cfg.hiddenSize
            let xPtr = attnIn.contents().assumingMemoryBound(to: UInt16.self)
            var x = [Float](repeating: 0, count: d)
            for i in 0..<d { x[i] = Self.q38F32(xPtr[i]) }
            let qkv = q38DebugGEMV4(view: try model.linearInProjQKV(layer: L),
                                    rows: qkvDim, cols: d, x: x)
            if let qraw = gdnQKVRaw {
                let qp = qraw.contents().assumingMemoryBound(to: UInt16.self)
                var mdq: Float = 0, mxq: Float = 0, saq: Float = 0
                for i in 0..<qkvDim {
                    let dd = abs(qkv[i] - Self.q38F32(qp[i]))
                    mdq += dd; mxq = max(mxq, dd); saq += abs(qkv[i])
                }
                let relq = (mdq / Float(qkvDim)) / max(1e-6, saq / Float(qkvDim))
                q38DebugDump("refGDNqkv.L\(L)", "rel=\(relq) \(relq < 0.05 ? "PASS" : "FAIL")")
            }
            let z = q38DebugGEMV4(view: try model.linearInProjZ(layer: L),
                                  rows: valDim, cols: d, x: x)
            let b = q38DebugGEMV4(view: try model.linearInProjB(layer: L),
                                  rows: nV, cols: d, x: x)
            // Conv window: zero past at position 0, mirror tail otherwise.
            var window: [[Float]] = []
            if p == 0 {
                let zq = [Float](repeating: 0, count: qkvDim)
                window = [zq, zq, zq].suffix(kernel - 1).map { $0 }
                if window.count < kernel - 1 {
                    window = [[Float]](repeating: zq, count: kernel - 1)
                }
            } else {
                guard let m = q38DebugGDNMirror[L] else {
                    q38DebugDump("refGDNpos1", "no mirror; run position 0 first")
                    return
                }
                window = m.tail
            }
            window.append(qkv)
            while window.count > kernel { window.removeFirst() }
            while window.count < kernel {
                window.insert([Float](repeating: 0, count: qkvDim), at: 0)
            }
            let convW = try model.linearConv1d(layer: L)
            let cwPtr = convW.buffer.contents().advanced(by: Int(convW.offset))
                .assumingMemoryBound(to: UInt16.self)
            var conv = [Float](repeating: 0, count: qkvDim)
            for c in 0..<qkvDim {
                var acc: Float = 0
                for t in 0..<kernel {
                    acc += Quantization.bf16ToFloat(cwPtr[c * kernel + t]) * window[t][c]
                }
                conv[c] = acc / (1.0 + exp(-acc)) // silu
            }
            // Save tail for the next position (last kernel-1 windows).
            let newTail = Array(window.suffix(kernel - 1))
            if p == 0, let gco = gdnConvOut {
                let gp = gco.contents().assumingMemoryBound(to: UInt16.self)
                var md: Float = 0, mx: Float = 0, sa: Float = 0
                for i in 0..<qkvDim {
                    let dd = abs(conv[i] - Self.q38F32(gp[i]))
                    md += dd; mx = max(mx, dd); sa += abs(conv[i])
                }
                let rel = (md / Float(qkvDim)) / max(1e-6, sa / Float(qkvDim))
                q38DebugDump("refGDNconv", "rel=\(rel) \(rel < 0.05 ? "PASS" : "FAIL")")
            }
            let q = Array(conv[0..<keyDim])
            let k = Array(conv[keyDim..<(2 * keyDim)])
            let v = Array(conv[(2 * keyDim)..<qkvDim])
            func l2(_ r: [Float]) -> [Float] {
                var s: Float = 0
                for e in r { s += e * e }
                let inv = 1.0 / sqrt(s + 1e-6)
                return r.map { $0 * inv }
            }
            let qScale = 1.0 / sqrt(Float(hK))
            var qH: [[Float]] = [], kH: [[Float]] = []
            for h in 0..<nK {
                let qr = l2(Array(q[(h * hK)..<((h + 1) * hK)]))
                qH.append(qr.map { $0 * qScale })
                kH.append(l2(Array(k[(h * hK)..<((h + 1) * hK)])))
            }
            var vH: [[Float]] = [], zH: [[Float]] = []
            for h in 0..<nV {
                vH.append(Array(v[(h * hV)..<((h + 1) * hV)]))
                zH.append(Array(z[(h * hV)..<((h + 1) * hV)]))
            }
            let rep = nV / nK
            // Recurrent step: fresh (S=0) at position 0, mirror state after.
            var sState: [[[Float]]]
            var beta = [Float](repeating: 0, count: nV)
            for h in 0..<nV { beta[h] = 1.0 / (1.0 + exp(-b[h])) }
            if p == 0 {
                sState = [[[Float]]](
                    repeating: [[Float]](
                        repeating: [Float](repeating: 0, count: hV), count: hK),
                    count: nV)
            } else {
                guard let m = q38DebugGDNMirror[L] else {
                    q38DebugDump("refGDNpos1", "no mirror; run position 0 first")
                    return
                }
                sState = m.s
                let a = q38DebugGEMV4(view: try model.linearInProjA(layer: L),
                                      rows: nV, cols: d, x: x)
                let aLogV = try model.linearALog(layer: L)
                let dtV = try model.linearDtBias(layer: L)
                let ap = aLogV.buffer.contents().advanced(by: Int(aLogV.offset))
                    .assumingMemoryBound(to: UInt16.self)
                let dp = dtV.buffer.contents().advanced(by: Int(dtV.offset))
                    .assumingMemoryBound(to: UInt16.self)
                for h in 0..<nV {
                    let av = a[h] + Quantization.bf16ToFloat(dp[h])
                    let g = -exp(Quantization.bf16ToFloat(ap[h]))
                        * log(1.0 + exp(av))
                    let dec = exp(g)
                    for i in 0..<hK {
                        for j in 0..<hV { sState[h][i][j] *= dec }
                    }
                }
            }
            var core = [Float](repeating: 0, count: valDim)
            for h in 0..<nV {
                let qh = qH[h / rep], kh = kH[h / rep]
                if p == 0 {
                    var dot: Float = 0
                    for j in 0..<hK { dot += qh[j] * kh[j] }
                    for j in 0..<hV { core[h * hV + j] = beta[h] * dot * vH[h][j] }
                    for i in 0..<hK {
                        for j in 0..<hV {
                            sState[h][i][j] = kh[i] * (beta[h] * vH[h][j])
                        }
                    }
                } else {
                    var kvMem = [Float](repeating: 0, count: hV)
                    for i in 0..<hK {
                        for j in 0..<hV { kvMem[j] += sState[h][i][j] * kh[i] }
                    }
                    var delta = [Float](repeating: 0, count: hV)
                    for j in 0..<hV { delta[j] = (vH[h][j] - kvMem[j]) * beta[h] }
                    for i in 0..<hK {
                        for j in 0..<hV { sState[h][i][j] += kh[i] * delta[j] }
                    }
                    for j in 0..<hV {
                        var o: Float = 0
                        for i in 0..<hK { o += sState[h][i][j] * qh[i] }
                        core[h * hV + j] = o
                    }
                }
            }
            q38DebugGDNMirror[L] = (tail: newTail, s: sState)
            let gnW = try model.linearNorm(layer: L)
            let gnPtr = gnW.buffer.contents().advanced(by: Int(gnW.offset))
                .assumingMemoryBound(to: UInt16.self)
            var gnWf = [Float](repeating: 0, count: hV)
            for j in 0..<hV { gnWf[j] = Quantization.bf16ToFloat(gnPtr[j]) }
            var gated = [Float](repeating: 0, count: valDim)
            for h in 0..<nV {
                var ms: Float = 0
                for j in 0..<hV { ms += core[h * hV + j] * core[h * hV + j] }
                let inv = 1.0 / sqrt(ms / Float(hV) + 1e-6)
                for j in 0..<hV {
                    let sz: Float = zH[h][j] / (1.0 + exp(-zH[h][j]))
                    gated[h * hV + j] = core[h * hV + j] * inv * gnWf[j] * sz
                }
            }
            let o = q38DebugGEMV4(view: try model.linearOutProj(layer: L),
                                 rows: d, cols: valDim, x: gated)
            // Intermediate localizations (no extra GEMVs): z (post in-proj),
            // core (post delta rule), gated (post gated-norm).
            if let zb = gdnZ {
                let zp = zb.contents().assumingMemoryBound(to: UInt16.self)
                var mdz: Float = 0, mxz: Float = 0, saz: Float = 0
                for i in 0..<valDim {
                    let dd = abs(z[i] - Self.q38F32(zp[i]))
                    mdz += dd; mxz = max(mxz, dd); saz += abs(z[i])
                }
                let relz = (mdz / Float(valDim)) / max(1e-6, saz / Float(valDim))
                q38DebugDump("refGDNz.L\(L)", "rel=\(relz) \(relz < 0.05 ? "PASS" : "FAIL")")
            }
            if let bb = gdnB {
                let bp = bb.contents().assumingMemoryBound(to: UInt16.self)
                var mdb: Float = 0, mxb: Float = 0, sab: Float = 0
                for i in 0..<nV {
                    let dd = abs(b[i] - Self.q38F32(bp[i]))
                    mdb += dd; mxb = max(mxb, dd); sab += abs(b[i])
                }
                let relb = (mdb / Float(nV)) / max(1e-6, sab / Float(nV))
                q38DebugDump("refGDNb.L\(L)", "rel=\(relb) \(relb < 0.05 ? "PASS" : "FAIL")")
            }
            if let yb = gdnY {
                let yp = yb.contents().assumingMemoryBound(to: UInt16.self)
                var mdy: Float = 0, mxy: Float = 0, say: Float = 0
                for i in 0..<valDim {
                    let dd = abs(core[i] - Self.q38F32(yp[i]))
                    mdy += dd; mxy = max(mxy, dd); say += abs(core[i])
                }
                let rely = (mdy / Float(valDim)) / max(1e-6, say / Float(valDim))
                var worst = ""
                if rely >= 0.05 {
                    var diffs: [(Int, Float, Float)] = []
                    for i in 0..<valDim {
                        let dd = abs(core[i] - Self.q38F32(yp[i]))
                        if dd > mxy * 0.5 { diffs.append((i, core[i], Self.q38F32(yp[i]))) }
                        if diffs.count >= 8 { break }
                    }
                    worst = " worst=" + diffs.map { "[\($0.0):cpu\(String(format: "%.4f", $0.1))/gpu\(String(format: "%.4f", $0.2))]" }.joined()
                }
                q38DebugDump("refGDNcore.L\(L)", "rel=\(rely) \(rely < 0.05 ? "PASS" : "FAIL")\(worst)")
            }
            if let go = gdnOut {
                let gp = go.contents().assumingMemoryBound(to: UInt16.self)
                var mdg2: Float = 0, mxg2: Float = 0, sag2: Float = 0
                for i in 0..<valDim {
                    let dd = abs(gated[i] - Self.q38F32(gp[i]))
                    mdg2 += dd; mxg2 = max(mxg2, dd); sag2 += abs(gated[i])
                }
                let relg2 = (mdg2 / Float(valDim)) / max(1e-6, sag2 / Float(valDim))
                q38DebugDump("refGDNout.L\(L)", "rel=\(relg2) \(relg2 < 0.05 ? "PASS" : "FAIL")")
            }
            let oPtr = oOut.contents().assumingMemoryBound(to: UInt16.self)
            var md: Float = 0, mx: Float = 0, sa: Float = 0
            for i in 0..<d {
                let dd = abs(o[i] - Self.q38F32(oPtr[i]))
                md += dd; mx = max(mx, dd); sa += abs(o[i])
            }
            let rel = (md / Float(d)) / max(1e-6, sa / Float(d))
            q38DebugDump("refGDNoOut.p\(p)", "rel=\(rel) maxDiff=\(mx) \(rel < 0.08 ? "PASS" : "FAIL")")
        } catch {
            q38DebugDump("refGDN", "error \(error)")
        }
    }
    /// L3 QSA check (position 0, single KV): CPU V/gate/o_proj path vs GPU
    /// `oOut`. At position 0 attention is trivial (weight 1.0 on the only
    /// key) and RoPE is the identity, so this validates V + gate + o_proj +
    /// norms with real weights; scores/softmax/RoPE need a later position.
    /// At position 1 the full two-key path (cache slots, RoPE'd Q, scores,
    /// softmax, repeat) is verified against the KV cache.
    private func q38DebugRefCheckQSA(attnIn: MTLBuffer, oOut: MTLBuffer,
                                    compactV: MTLBuffer, layer L: Int,
                                    position p: Int, kv cache: KVCacheManager) {
        do {
            let nQ = cfg.numHeads
            let nKV = cfg.numFullKVHeads
            let hd = cfg.fullHeadDim
            let qDim = nQ * hd
            let kvDim = nKV * hd
            let d = cfg.hiddenSize
            let eps: Float = 1e-6
            let xPtr = attnIn.contents().assumingMemoryBound(to: UInt16.self)
            var x = [Float](repeating: 0, count: d)
            for i in 0..<d { x[i] = Self.q38F32(xPtr[i]) }
            let qq = q38DebugGEMV4(view: try model.qProj(layer: L),
                                   rows: 2 * qDim, cols: d, x: x)
            // Packed q_proj is per-head interleaved [q0,g0,q1,g1,...] (viewed
            // as [heads, 2*headDim] then chunked on the last dim), NOT two
            // contiguous halves.
            var query = [Float](repeating: 0, count: qDim)
            var gate = [Float](repeating: 0, count: qDim)
            for h in 0..<nQ {
                for j in 0..<hd {
                    query[h * hd + j] = qq[h * 2 * hd + j]
                    gate[h * hd + j] = qq[h * 2 * hd + hd + j]
                }
            }
            // Note: K is unused at position 0 (single-key attention has
            // weight 1.0 by construction); Q·K scores need a later position.
            let vv = q38DebugGEMV4(view: try model.vProj(layer: L),
                                   rows: kvDim, cols: d, x: x)
            let qNW = try model.qNorm(layer: L)
            let kNW = try model.kNorm(layer: L)
            let qwPtr = qNW.buffer.contents().advanced(by: Int(qNW.offset))
                .assumingMemoryBound(to: UInt16.self)
            let kwPtr = kNW.buffer.contents().advanced(by: Int(kNW.offset))
                .assumingMemoryBound(to: UInt16.self)
            var qw = [Float](repeating: 0, count: hd)
            var kw = [Float](repeating: 0, count: hd)
            for i in 0..<hd {
                qw[i] = Quantization.bf16ToFloat(qwPtr[i])
                kw[i] = Quantization.bf16ToFloat(kwPtr[i])
            }
            let rep = nQ / nKV
            var attn = [Float](repeating: 0, count: qDim)
            for h in 0..<nQ {
                let qr = Qwen38Reference.rmsNorm(Array(query[(h * hd)..<((h + 1) * hd)]),
                                                weight: qw, eps: eps)
                _ = qr // normed Q unused at pos 0 (single-key attention)
                let vh = Array(vv[((h / rep) * hd)..<(((h / rep) + 1) * hd)])
                let gr = Array(gate[(h * hd)..<((h + 1) * hd)])
                for j in 0..<hd {
                    attn[h * hd + j] = vh[j] / (1.0 + exp(-gr[j]))
                }
            }
            let o = q38DebugGEMV4(view: try model.oProj(layer: L),
                                 rows: d, cols: qDim, x: attn)
            let oPtr = oOut.contents().assumingMemoryBound(to: UInt16.self)
            var md: Float = 0, mx: Float = 0, sa: Float = 0
            for i in 0..<d {
                let dd = abs(o[i] - Self.q38F32(oPtr[i]))
                md += dd; mx = max(mx, dd); sa += abs(o[i])
            }
            let rel = (md / Float(d)) / max(1e-6, sa / Float(d))
            if p == 0 {
                q38DebugDump("refQSAoOut", "rel=\(rel) maxDiff=\(mx) \(rel < 0.08 ? "PASS" : "FAIL")")
            }
            // V-only isolation: post-gather compactV vs CPU V (catches K/V
            // dequant, cache write/read, gather).
            if p == 0 {
                do {
                    let cp = compactV.contents().assumingMemoryBound(to: UInt16.self)
                    var mdv: Float = 0, mxv: Float = 0, sav: Float = 0
                    for i in 0..<kvDim {
                        let dd = abs(vv[i] - Self.q38F32(cp[i]))
                        mdv += dd; mxv = max(mxv, dd); sav += abs(vv[i])
                    }
                    let relv = (mdv / Float(kvDim)) / max(1e-6, sav / Float(kvDim))
                    q38DebugDump("refQSAv", "rel=\(relv) maxDiff=\(mxv) \(relv < 0.05 ? "PASS" : "FAIL")")
                }
            }
            // Pre-o_proj isolation: attnOut (V repeated + gate) vs CPU.
            if p == 0 {
                do {
                    let ap = self.attnOut.contents().assumingMemoryBound(to: UInt16.self)
                    var mda: Float = 0, mxa: Float = 0, saa: Float = 0
                    for i in 0..<qDim {
                        let dd = abs(attn[i] - Self.q38F32(ap[i]))
                        mda += dd; mxa = max(mxa, dd); saa += abs(attn[i])
                    }
                    let rela = (mda / Float(qDim)) / max(1e-6, saa / Float(qDim))
                    q38DebugDump("refQSAattn", "rel=\(rela) maxDiff=\(mxa) \(rela < 0.05 ? "PASS" : "FAIL")")
                }
            }
            // Gate-only isolation: GPU post-split raw gate vs CPU strided
            // gate. The fused-QKV check covered query rows only; gate rows
            // (odd 256-blocks) were never verified.
            if p == 0, let ags = attnGateScratch {
                let gp = ags.contents().assumingMemoryBound(to: UInt16.self)
                var mdg: Float = 0, mxg: Float = 0, sag: Float = 0
                for i in 0..<qDim {
                    let dd = abs(gate[i] - Self.q38F32(gp[i]))
                    mdg += dd; mxg = max(mxg, dd); sag += abs(gate[i])
                }
                let relg = (mdg / Float(qDim)) / max(1e-6, sag / Float(qDim))
                q38DebugDump("refQSAgate", "rel=\(relg) maxDiff=\(mxg) \(relg < 0.05 ? "PASS" : "FAIL")")
            }
            if p >= 1 {
                // Multi-key attention from the KV cache: normed + RoPE'd Q
                // at pos p against cached (already RoPE'd) K, softmax over
                // p+1 keys, repeat V. Covers KV carry, nonzero RoPE, scores.
                let rotaryDim = Int(Double(hd) * cfg.partialRotaryFactor)
                var qN: [[Float]] = []
                for h in 0..<nQ {
                    let nrm = Qwen38Reference.rmsNorm(
                        Array(query[(h * hd)..<((h + 1) * hd)]), weight: qw, eps: eps)
                    qN.append(Qwen38Reference.applyRopeNeoxCPU(
                        nrm, position: p, headDim: hd, rotaryDim: rotaryDim,
                        theta: cfg.ropeTheta))
                }
                var kk: [[Float]] = [], vvc: [[Float]] = []
                for t in 0...p {
                    let ks = cache.kSlot(layer: L, position: t)
                    let vs = cache.vSlot(layer: L, position: t)
                    let kp = ks.buffer.contents().advanced(by: ks.offset)
                        .assumingMemoryBound(to: UInt16.self)
                    let vp = vs.buffer.contents().advanced(by: vs.offset)
                        .assumingMemoryBound(to: UInt16.self)
                    var kr = [Float](repeating: 0, count: kvDim)
                    var vr = [Float](repeating: 0, count: kvDim)
                    for i in 0..<kvDim {
                        kr[i] = Self.q38F32(kp[i])
                        vr[i] = Self.q38F32(vp[i])
                    }
                    kk.append(kr)
                    vvc.append(vr)
                }
                var attn1 = [Float](repeating: 0, count: qDim)
                for h in 0..<nQ {
                    var sc = [Float](repeating: 0, count: p + 1)
                    for t in 0...p {
                        var dt: Float = 0
                        for j in 0..<hd { dt += qN[h][j] * kk[t][(h / rep) * hd + j] }
                        sc[t] = dt * Float(cfg.attentionScale)
                    }
                    let mx = sc.max() ?? 0
                    var ex = sc.map { exp($0 - mx) }
                    let dn = ex.reduce(0, +)
                    ex = ex.map { $0 / dn }
                    let gr = Array(gate[(h * hd)..<((h + 1) * hd)])
                    for j in 0..<hd {
                        var ov: Float = 0
                        for t in 0...p { ov += ex[t] * vvc[t][(h / rep) * hd + j] }
                        attn1[h * hd + j] = ov / (1.0 + exp(-gr[j]))
                    }
                }
                let o1 = q38DebugGEMV4(view: try model.oProj(layer: L),
                                      rows: d, cols: qDim, x: attn1)
                let oPtr1 = oOut.contents().assumingMemoryBound(to: UInt16.self)
                var md1: Float = 0, mx1: Float = 0, sa1: Float = 0
                for i in 0..<d {
                    let dd = abs(o1[i] - Self.q38F32(oPtr1[i]))
                    md1 += dd; mx1 = max(mx1, dd); sa1 += abs(o1[i])
                }
                let rel1 = (md1 / Float(d)) / max(1e-6, sa1 / Float(d))
                q38DebugDump("refQSAoOut.p\(p)", "rel=\(rel1) maxDiff=\(mx1) \(rel1 < 0.08 ? "PASS" : "FAIL")")
            }
        } catch {
            q38DebugDump("refQSA", "error \(error)")
        }
    }

    /// Head consistency check (position 0): non-fused full-logits argmax vs
    /// fused greedy-head id, both from the same `normed` hidden state. Same
    /// weights + same input must agree; disagreement isolates the fused
    /// head (the production greedy path) without a 250k-row CPU GEMV.
    private func q38DebugRefCheckHead() {
        let lm = model.lmHead
        let d = cfg.hiddenSize
        let vocab = cfg.vocabSize
        let eps: Float = 1e-6
        let attnGroup = model.attentionGroupSize
        guard let logitsBuf = ctx.device.makeBuffer(length: vocab * 2,
                                                    options: .storageModeShared),
              let tokBuf = ctx.device.makeBuffer(length: 4,
                                                 options: .storageModeShared) else {
            q38DebugDump("refHead", "alloc failed")
            return
        }
        runSync { cb in
            self.int4.encode(commandBuffer: cb,
                             weights: lm.buffer, weightsOffset: Int(lm.offset),
                             scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                             biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                             x: self.normed, y: logitsBuf,
                             m: UInt32(vocab), n: UInt32(d),
                             groupSize: attnGroup)
        }
        let lPtr = logitsBuf.contents().assumingMemoryBound(to: UInt16.self)
        var best = 0
        var bestV = -Float.infinity
        for i in 0..<vocab {
            let v = Self.q38F32(lPtr[i])
            if v > bestV { bestV = v; best = i }
        }
        runSync { cb in
            self.fusionHead.encodeGreedyDecode(
                commandBuffer: cb, hidden: self.normed,
                normWeight: self.normed, normOffset: 0,
                weights: lm.buffer, weightsOffset: Int(lm.offset),
                scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                outToken: tokBuf,
                d: UInt32(d), vocab: UInt32(vocab),
                rmsEps: eps, groupSize: attnGroup, skipNorm: true)
        }
        let fused = Int(tokBuf.contents().load(as: UInt32.self))
        var top: [(Int, Float)] = []
        for i in 0..<vocab {
            let v = Self.q38F32(lPtr[i])
            top.append((i, v))
            if top.count > 5 {
                top.sort { $0.1 < $1.1 }
                top.removeFirst()
            }
        }
        top.sort { $0.1 > $1.1 }
        let desc = top.map { "\($0.0):\(String(format: "%.2f", $0.1))" }.joined(separator: " ")
        q38DebugDump("refHead", "nonfused=\(best) fused=\(fused) "
            + "\(best == fused ? "PASS" : "FAIL") top5=[\(desc)]")
    }

    /// RoPE unit check (no model data): synthetic ramp vector through
    /// `encodeNeoxSubdim` at position 5 vs the CPU reference. Position 0 is
    /// the identity and verifies nothing; every prefill token after the
    /// first depends on nonzero-position RoPE. Rotation preserves
    /// magnitudes, so a RoPE bug destroys content while keeping all
    /// magnitude probes green -- exactly the observed pattern.
    private func q38DebugRefCheckRope() {
        guard let ropeK = rope else {
            q38DebugDump("refRope", "no rope kernel")
            return
        }
        let hd = cfg.fullHeadDim
        let nH = 4
        let rotaryDim = Int(Double(hd) * cfg.partialRotaryFactor)
        let count = nH * hd
        guard let buf = ctx.device.makeBuffer(length: count * 2,
                                             options: .storageModeShared) else {
            q38DebugDump("refRope", "alloc failed")
            return
        }
        let ptr = buf.contents().assumingMemoryBound(to: UInt16.self)
        var cpu = [Float](repeating: 0, count: count)
        for h in 0..<nH {
            for i in 0..<hd {
                let v = Float(h * hd + i) / Float(count)
                cpu[h * hd + i] = v
                ptr[h * hd + i] = Self.q38F16(v)
            }
        }
        runSync { cb in
            ropeK.encodeNeoxSubdim(commandBuffer: cb, data: buf,
                                   position: 5,
                                   headDim: UInt32(hd),
                                   numHeads: UInt32(nH),
                                   rotaryDim: UInt32(rotaryDim),
                                   theta: Float(cfg.fullRopeTheta))
        }
        let oPtr = buf.contents().assumingMemoryBound(to: UInt16.self)
        var md: Float = 0, mx: Float = 0, sa: Float = 0
        for h in 0..<nH {
            let row = Array(cpu[(h * hd)..<((h + 1) * hd)])
            let ref = Qwen38Reference.applyRopeNeoxCPU(
                row, position: 5, headDim: hd, rotaryDim: rotaryDim,
                theta: cfg.ropeTheta)
            for i in 0..<hd {
                let dd = abs(ref[i] - Self.q38F32(oPtr[h * hd + i]))
                md += dd; mx = max(mx, dd); sa += abs(ref[i])
            }
        }
        let rel = (md / Float(count)) / max(1e-6, sa / Float(count))
        q38DebugDump("refRope", "rel=\(rel) maxDiff=\(mx) \(rel < 0.02 ? "PASS" : "FAIL")")
    }

    /// Expert-identity check (L0): memcmp the delivered blob bytes for
    /// `experts[0]` against the layer FILE bytes at that expert's recorded
    /// file offset. Catches streamer cache-slot confusion (right kernel,
    /// wrong expert's weights) without any dequant math.
    private func q38DebugRefCheckExpertMap(layer L: Int, experts: [Int],
                                          blobs: [TensorView]) {
        do {
            guard !experts.isEmpty, !blobs.isEmpty else { return }
            let want = experts[0]
            let phys = model.routedExpertPhysicalOffsets(layer: L)
            guard want >= 0 && want < phys.count else {
                q38DebugDump("refExpMap", "expert \(want) out of range")
                return
            }
            let offs = model.routedExpertOffsets(layer: L)
            let fileBase = model.packedExpertsLayout.layers[L].file
            let url = model.directoryURL
                .appendingPathComponent("packed_experts")
                .appendingPathComponent(fileBase)
            let fh = try FileHandle(forReadingFrom: url)
            defer { try? fh.close() }
            try fh.seek(toOffset: phys[want] + UInt64(offs.gateWOff))
            let fileBytes = try fh.read(upToCount: 4096) ?? Data()
            let b = blobs[0]
            let bp = b.buffer.contents().advanced(by: Int(b.offset))
                .assumingMemoryBound(to: UInt8.self)
            var same = fileBytes.count == 4096
            if same {
                for i in 0..<4096 {
                    if fileBytes[i] != bp[i] { same = false; break }
                }
            }
            q38DebugDump("refExpMap", "expert=\(want) \(same ? "MATCH" : "MISMATCH")")
        } catch {
            q38DebugDump("refExpMap", "error \(error)")
        }
    }

    /// Full routed-MoE check (L0): CPU SwiGLU over all 10 selected experts
    /// with real blob bytes vs GPU `h2Buf`. Validates phase-1/phase-2,
    /// per-expert dequant, and the top-k weighted reduction. Mapping itself
    /// is proven by refExpMap; this proves the compute.
    private func q38DebugRefCheckRouted(mlpIn: MTLBuffer, h1: MTLBuffer,
                                       h2: MTLBuffer, layer L: Int,
                                       experts: [Int], blobs: [TensorView]) {
        let f = cfg.moeIntermediateSize
        let d = cfg.hiddenSize
        let g = model.routedExpertGroupSize
        let topK = cfg.topKExperts
        guard experts.count == topK && blobs.count == topK else {
            q38DebugDump("refRouted", "count mismatch")
            return
        }
        let offs = model.routedExpertOffsets(layer: L)
        let xPtr = mlpIn.contents().assumingMemoryBound(to: UInt16.self)
        var x = [Float](repeating: 0, count: d)
        for i in 0..<d { x[i] = Self.q38F32(xPtr[i]) }
        let wPtr = outWeights.contents().assumingMemoryBound(to: UInt16.self)
        var rw = [Float](repeating: 0, count: topK)
        for i in 0..<topK { rw[i] = Self.q38F32(wPtr[i]) }
        func gemv(base: UnsafeRawPointer, wOff: Int, sOff: Int, bOff: Int,
                  rows: Int, cols: Int) -> [Float] {
            let groups = cols / g
            let rowBytes = cols / 2
            var y = [Float](repeating: 0, count: rows)
            for r in 0..<rows {
                let wRow = base.advanced(by: wOff + r * rowBytes)
                    .assumingMemoryBound(to: UInt8.self)
                let sRow = base.advanced(by: sOff + r * groups * 2)
                    .assumingMemoryBound(to: UInt16.self)
                let bRow = base.advanced(by: bOff + r * groups * 2)
                    .assumingMemoryBound(to: UInt16.self)
                var packed = [UInt8](repeating: 0, count: rowBytes)
                var sc = [UInt16](repeating: 0, count: groups)
                var bi = [UInt16](repeating: 0, count: groups)
                for i in 0..<rowBytes { packed[i] = wRow[i] }
                for i in 0..<groups { sc[i] = sRow[i]; bi[i] = bRow[i] }
                let w = Quantization.dequantizeInt4Affine(
                    Quantization.Int4AffineRow(packed: packed, scales: sc, biases: bi),
                    n: cols, groupSize: g)
                var acc: Float = 0
                for i in 0..<cols { acc += w[i] * x[i] }
                y[r] = acc
            }
            return y
        }
        // Down proj needs a different input per expert; do it inline below.
        let h1Ptr = h1.contents().assumingMemoryBound(to: UInt16.self)
        var h2cpu = [Float](repeating: 0, count: d)
        for i in 0..<d { h2cpu[i] = Self.q38F32(h1Ptr[i]) }
        for s in 0..<topK {
            let base = blobs[s].buffer.contents().advanced(by: Int(blobs[s].offset))
            let go = Int(offs.gateWOff), gs = Int(offs.gateSOff), gb = Int(offs.gateBOff)
            let uo = Int(offs.upWOff), us = Int(offs.upSOff), ub = Int(offs.upBOff)
            let dwn = Int(offs.downWOff), ds = Int(offs.downSOff), db = Int(offs.downBOff)
            let gv = gemv(base: base, wOff: go, sOff: gs, bOff: gb, rows: f, cols: d)
            let uv = gemv(base: base, wOff: uo, sOff: us, bOff: ub, rows: f, cols: d)
            var act = [Float](repeating: 0, count: f)
            for i in 0..<f {
                let sg = gv[i] / (1.0 + exp(-gv[i]))
                act[i] = sg * uv[i]
            }
            let groupsD = f / g
            let rowBytesD = f / 2
            for r in 0..<d {
                let wRow = base.advanced(by: dwn + r * rowBytesD)
                    .assumingMemoryBound(to: UInt8.self)
                let sRow = base.advanced(by: ds + r * groupsD * 2)
                    .assumingMemoryBound(to: UInt16.self)
                let bRow = base.advanced(by: db + r * groupsD * 2)
                    .assumingMemoryBound(to: UInt16.self)
                var packed = [UInt8](repeating: 0, count: rowBytesD)
                var sc = [UInt16](repeating: 0, count: groupsD)
                var bi = [UInt16](repeating: 0, count: groupsD)
                for i in 0..<rowBytesD { packed[i] = wRow[i] }
                for i in 0..<groupsD { sc[i] = sRow[i]; bi[i] = bRow[i] }
                let w = Quantization.dequantizeInt4Affine(
                    Quantization.Int4AffineRow(packed: packed, scales: sc, biases: bi),
                    n: f, groupSize: g)
                var acc: Float = 0
                for i in 0..<f { acc += w[i] * act[i] }
                h2cpu[r] += rw[s] * acc
            }
        }
        let h2Ptr = h2.contents().assumingMemoryBound(to: UInt16.self)
        var md: Float = 0, mx: Float = 0, sa: Float = 0
        for i in 0..<d {
            let dd = abs(h2cpu[i] - Self.q38F32(h2Ptr[i]))
            md += dd; mx = max(mx, dd); sa += abs(h2cpu[i])
        }
        let rel = (md / Float(d)) / max(1e-6, sa / Float(d))
        q38DebugDump("refRouted", "rel=\(rel) maxDiff=\(mx) \(rel < 0.08 ? "PASS" : "FAIL")")
    }

    /// Mixer check: CPU `Qwen38Reference.gatedResidual` (no combine) with
    /// real weights vs GPU `normed`. Covers hc_norm/down/up/silu/sigmoid/
    /// mean -- the same machinery every layer's hyper-mix uses, verified
    /// here once with 48-deep real hyper state.
    private func q38DebugRefCheckMixer(hyper: MTLBuffer, mixed: MTLBuffer) {
        let hc = cfg.hcCount
        let d = cfg.hiddenSize
        let hcHidden = hc * d
        let lowrank = cfg.hcLowrank
        let hp = hyper.contents().assumingMemoryBound(to: UInt16.self)
        var hyperIn = [Float](repeating: 0, count: hcHidden)
        for i in 0..<hcHidden { hyperIn[i] = Self.q38F32(hp[i]) }
        func debf16(_ view: TensorView) -> [Float] {
            let p = view.buffer.contents().advanced(by: Int(view.offset))
                .assumingMemoryBound(to: UInt16.self)
            let n = Int(view.length) / 2
            var o = [Float](repeating: 0, count: n)
            for i in 0..<n { o[i] = Quantization.bf16ToFloat(p[i]) }
            return o
        }
        let normW = debf16(model.mixerHcNorm)
        let downW = q38DebugGEMV4Raw(view: model.mixerDown, rows: lowrank, cols: hcHidden)
        let upW = q38DebugGEMV4Raw(view: model.mixerUp, rows: hcHidden, cols: lowrank)
        let ref = Qwen38Reference.gatedResidual(
            hyperInput: hyperIn, hcCount: hc, hidden: d, lowrank: lowrank,
            hcNormWeight: normW, downWeight: downW, upWeight: upW,
            blockInjectWeight: nil)
        let mp = mixed.contents().assumingMemoryBound(to: UInt16.self)
        var md: Float = 0, mx: Float = 0, sa: Float = 0
        for i in 0..<d {
            let dd = abs(ref.mixed[i] - Self.q38F32(mp[i]))
            md += dd; mx = max(mx, dd); sa += abs(ref.mixed[i])
        }
        let rel = (md / Float(d)) / max(1e-6, sa / Float(d))
        q38DebugDump("refMixer", "rel=\(rel) maxDiff=\(mx) \(rel < 0.08 ? "PASS" : "FAIL")")
    }

    /// Raw INT4-g32 dequant of a full view to row-major [Float] (for the
    /// reference hyper-mix check).
    private func q38DebugGEMV4Raw(view: TensorView, rows: Int, cols: Int) -> [Float] {
        let g = 32
        let groups = cols / g
        let rowBytes = cols / 2
        let wBase = view.buffer.contents().advanced(by: Int(view.offset))
        let sBase = view.buffer.contents().advanced(by: Int(view.scaleOffset))
        let bBase = view.buffer.contents().advanced(by: Int(view.biasOffset))
        var out = [Float](repeating: 0, count: rows * cols)
        for r in 0..<rows {
            let wRow = wBase.advanced(by: r * rowBytes).assumingMemoryBound(to: UInt8.self)
            let sRow = sBase.advanced(by: r * groups * 2).assumingMemoryBound(to: UInt16.self)
            let bRow = bBase.advanced(by: r * groups * 2).assumingMemoryBound(to: UInt16.self)
            var packed = [UInt8](repeating: 0, count: rowBytes)
            var sc = [UInt16](repeating: 0, count: groups)
            var bi = [UInt16](repeating: 0, count: groups)
            for i in 0..<rowBytes { packed[i] = wRow[i] }
            for i in 0..<groups { sc[i] = sRow[i]; bi[i] = bRow[i] }
            let w = Quantization.dequantizeInt4Affine(
                Quantization.Int4AffineRow(packed: packed, scales: sc, biases: bi),
                n: cols, groupSize: g)
            for i in 0..<cols { out[r * cols + i] = w[i] }
        }
        return out
    }

    /// PLE read-path check (position 0): CPU n-gram rows + SSD reads +
    /// dequant vs GPU `pleEmbed`, then key/value GEMVs + grouped norms vs
    /// GPU `pleKey`/`pleValue` (post-norm, in-place). Covers row selection,
    /// sharding, and uploads with real tokens -- the highest-risk PLE part.
    private func q38DebugRefCheckPLE(token: Int32, pleLayer L: Int) {
        do {
            guard let pleReader, let pleEmbed = q38PleEmbed,
                  let pleKey = q38PleKey else { return }
            let ngramSize = cfg.pleNgramSize
            let headsPerNgram = cfg.pleHeadsPerNgram
            let ngramHeads = (ngramSize - 1) * headsPerNgram
            let contextLen = ngramSize - 1
            var window = q38PleHistory.suffix(contextLen).map { Int($0) }
            while window.count < contextLen { window.insert(Int(q38PleEos), at: 0) }
            window.append(Int(token))
            let multView = try model.pleMultipliers(layer: L)
            let offView = try model.pleHeadsOffsets(layer: L)
            let vszView = try model.pleHeadsVocabSizes(layer: L)
            let multPtr = multView.buffer.contents().advanced(by: Int(multView.offset))
                .assumingMemoryBound(to: UInt64.self)
            let offPtr = offView.buffer.contents().advanced(by: Int(offView.offset))
                .assumingMemoryBound(to: Int64.self)
            let vszPtr = vszView.buffer.contents().advanced(by: Int(vszView.offset))
                .assumingMemoryBound(to: Int64.self)
            let multipliers = (0..<ngramSize).map { multPtr[$0] }
            let headOffsets = (0..<ngramHeads).map { Int(offPtr[$0]) }
            let headVocabSizes = (0..<ngramHeads).map { Int(vszPtr[$0]) }
            let rows = Qwen38Reference.ngramRowIds(
                history: window, seqLen: 1, ngramSize: ngramSize,
                headsPerNgram: headsPerNgram, multipliers: multipliers,
                headVocabSizes: headVocabSizes, headOffsets: headOffsets,
                eosId: Int(q38PleEos))
            let rps = try pleReader.rowsPerShard
            let headDim = cfg.pleEmbedDim / ngramHeads
            let embPtr = pleEmbed.contents().assumingMemoryBound(to: UInt16.self)
            var md: Float = 0, mx: Float = 0, sa: Float = 0
            for h in 0..<ngramHeads {
                let vec = try pleReader.readRow(globalRow: rows[0][h], rowsPerShard: rps)
                for dd in 0..<headDim {
                    let a = vec[dd]
                    let b = Self.q38F32(embPtr[h * headDim + dd])
                    let dff = abs(a - b)
                    md += dff; mx = max(mx, dff); sa += abs(a)
                }
            }
            let n = Float(ngramHeads * headDim)
            let rel = (md / n) / max(1e-6, sa / n)
            q38DebugDump("refPLEembed", "rel=\(rel) maxDiff=\(mx) \(rel < 0.05 ? "PASS" : "FAIL")")
            // Key/value projections + grouped norms.
            let eps: Float = 1e-6
            let hc = cfg.hcCount
            let d = cfg.hiddenSize
            var emb = [Float](repeating: 0, count: cfg.pleEmbedDim)
            for i in 0..<cfg.pleEmbedDim { emb[i] = Self.q38F32(embPtr[i]) }
            let keyW = try model.pleKeyProj(layer: L)
            let valW = try model.pleValueProj(layer: L)
            let nKeyW = try model.pleNormKey(layer: L)
            let kk = q38DebugGEMV4(view: keyW, rows: hc * d, cols: cfg.pleEmbedDim, x: emb)
            let vv2 = q38DebugGEMV4(view: valW, rows: d, cols: cfg.pleEmbedDim, x: emb)
            func readW(_ v: TensorView) -> [Float] {
                let p = v.buffer.contents().advanced(by: Int(v.offset))
                    .assumingMemoryBound(to: UInt16.self)
                let c = Int(v.length) / 2
                var o = [Float](repeating: 0, count: c)
                for i in 0..<c { o[i] = Quantization.bf16ToFloat(p[i]) }
                return o
            }
            let nkw = readW(nKeyW)
            var keyN = [Float](repeating: 0, count: hc * d)
            for h in 0..<hc {
                let nr = Qwen38Reference.rmsNorm(Array(kk[(h * d)..<((h + 1) * d)]),
                                                weight: Array(nkw[(h * d)..<((h + 1) * d)]),
                                                groupSize: d, eps: eps)
                for j in 0..<d { keyN[h * d + j] = nr[j] }
            }
            let kp = pleKey.contents().assumingMemoryBound(to: UInt16.self)
            var mdk: Float = 0, mxk: Float = 0, sak: Float = 0
            for i in 0..<(hc * d) {
                let dd = abs(keyN[i] - Self.q38F32(kp[i]))
                mdk += dd; mxk = max(mxk, dd); sak += abs(keyN[i])
            }
            let relk = (mdk / Float(hc * d)) / max(1e-6, sak / Float(hc * d))
            q38DebugDump("refPLEkey", "rel=\(relk) maxDiff=\(mxk) \(relk < 0.05 ? "PASS" : "FAIL")")
            _ = vv2
        } catch {
            q38DebugDump("refPLE", "error \(error)")
        }
    }

    /// PLE gate/conv/compose check (position 0, fresh conv state): CPU dots
    /// + signed-sqrt gate + scaleRows + grouped norm + dilated depthwise
    /// conv + residual adds vs GPU `pleGated` (final). Needs pre-PLE hyper?
    /// No -- query norm output `pleQuery` is still in its buffer; dots are
    /// recomputed from the (intact) normed key/query buffers.
    private func q38DebugRefCheckPLEGateConv(pleLayer L: Int) {
        do {
            guard let pleKey = q38PleKey, let pleQuery = q38PleQuery,
                  let pleValue = q38PleValue, let pleDots = q38PleDots,
                  let pleGate = q38PleGate, let pleGated = q38PleGated,
                  let pleConvState = q38PleConvState else { return }
            let hc = cfg.hcCount
            let d = cfg.hiddenSize
            let hcHidden = hc * d
            let eps: Float = 1e-6
            let kPtr = pleKey.contents().assumingMemoryBound(to: UInt16.self)
            let qPtr = pleQuery.contents().assumingMemoryBound(to: UInt16.self)
            let vPtr = pleValue.contents().assumingMemoryBound(to: UInt16.self)
            // Dots (GPU wrote pleDots as FP32; recompute and compare).
            var dots = [Float](repeating: 0, count: hc)
            for h in 0..<hc {
                var acc: Float = 0
                for j in 0..<d {
                    acc += Self.q38F32(kPtr[h * d + j]) * Self.q38F32(qPtr[h * d + j])
                }
                dots[h] = acc / sqrt(Float(d))
            }
            let dPtr = pleDots.contents().assumingMemoryBound(to: Float.self)
            var md0: Float = 0
            for h in 0..<hc { md0 = max(md0, abs(dots[h] - dPtr[h])) }
            // Gate (same CPU code as production) + scaleRows.
            var gate = [Float](repeating: 0, count: hc)
            for h in 0..<hc {
                let mag = sqrt(max(1e-6, abs(dots[h])))
                gate[h] = (dots[h] >= 0 ? 1.0 : -1.0) * mag
            }
            var scaled = [Float](repeating: 0, count: hcHidden)
            for h in 0..<hc {
                let s: Float = 1.0 / (1.0 + exp(-gate[h]))
                for j in 0..<d { scaled[h * d + j] = s * Self.q38F32(vPtr[j]) }
            }
            // Grouped norm (1+w) with norm_conv.
            let nConvW = try model.pleNormConv(layer: L)
            let nwPtr = nConvW.buffer.contents().advanced(by: Int(nConvW.offset))
                .assumingMemoryBound(to: UInt16.self)
            var normed = [Float](repeating: 0, count: hcHidden)
            for h in 0..<hc {
                var ms: Float = 0
                for j in 0..<d { ms += scaled[h * d + j] * scaled[h * d + j] }
                let inv = 1.0 / sqrt(ms / Float(d) + eps)
                for j in 0..<d {
                    let w = Quantization.bf16ToFloat(nwPtr[h * d + j])
                    normed[h * d + j] = scaled[h * d + j] * inv * (1.0 + w)
                }
            }
            // Dilated depthwise short conv (kernel 4, dilation 3, fresh zero
            // state) + silu, then two residual adds (gated + conv).
            let convW = try model.pleConv1d(layer: L)
            let cwPtr = convW.buffer.contents().advanced(by: Int(convW.offset))
                .assumingMemoryBound(to: UInt16.self)
            let kernel = 4 // matches the encodeShortConvStep call in q38PLE
            var convOut = [Float](repeating: 0, count: hcHidden)
            for c in 0..<hcHidden {
                // Fresh zero state: only the newest tap (t = kernel-1,
                // delay 0) sees the current input.
                let wNew = Quantization.bf16ToFloat(cwPtr[c * kernel + (kernel - 1)])
                let acc = wNew * normed[c]
                convOut[c] = acc / (1.0 + exp(-acc)) // silu
            }
            var expect = [Float](repeating: 0, count: hcHidden)
            for i in 0..<hcHidden { expect[i] = normed[i] + convOut[i] }
            let gPtr = pleGated.contents().assumingMemoryBound(to: UInt16.self)
            var md: Float = 0, mx: Float = 0, sa: Float = 0
            for i in 0..<hcHidden {
                let dd = abs(expect[i] - Self.q38F32(gPtr[i]))
                md += dd; mx = max(mx, dd); sa += abs(expect[i])
            }
            let rel = (md / Float(hcHidden)) / max(1e-6, sa / Float(hcHidden))
            q38DebugDump("refPLEgate", "dotsDiff=\(md0) rel=\(rel) maxDiff=\(mx) "
                + "\((md0 < 1e-3 && rel < 0.08) ? "PASS" : "FAIL")")
            _ = pleConvState
            _ = pleGate
        } catch {
            q38DebugDump("refPLEgate", "error \(error)")
        }
    }

    private func q38DebugProbe(_ label: String, _ buf: MTLBuffer, _ count: Int) {        let ptr = buf.contents().assumingMemoryBound(to: UInt16.self)
        var mn = Float.infinity
        var mx = -Float.infinity
        var acc: Float = 0
        var nan = false
        var inf = false
        for i in 0..<count {
            let v = Self.q38F32(ptr[i])
            if v.isNaN { nan = true; continue }
            if v.isInfinite { inf = true; continue }
            mn = min(mn, v)
            mx = max(mx, v)
            acc += abs(v)
        }
        let mean = count > 0 ? acc / Float(count) : 0
        let msg = "[q38dbg] \(label) n=\(count) min=\(mn) max=\(mx) meanabs=\(mean) nan=\(nan) inf=\(inf)\n"
        if let d = msg.data(using: .utf8) { FileHandle.standardError.write(d) }
    }

    /// One gated-residual mix (`Qwen4ExpTextGatedResidual`): grouped (1+w)
    /// norm, low-rank down/up mixer with silu/sigmoid, branch-mean. With
    /// `injectW != nil` also emits `2*sigmoid(block_inject/ hc)` injection
    /// weights. All GEMVs read the attention-slot group width (base 4-bit).
    private func q38HyperMix(_ cb: MTLCommandBuffer,
                             hyper: MTLBuffer,
                             normW: TensorView,
                             downW: TensorView,
                             upW: TensorView,
                             injectW: TensorView?,
                             mixedOut: MTLBuffer,
                             injOut: MTLBuffer?,
                             groupSize: Int) throws {
        guard let q38, let hyperNormed = q38HyperNormed, let low = q38Low,
              let mix = q38Mix else {
            preconditionFailure("qwen38 hyper-connection without scratch")
        }
        let D = cfg.hiddenSize
        let hc = cfg.hcCount
        let hcHidden = hc * D
        let lowrank = cfg.hcLowrank
        let eps: Float = 1e-6
        q38.encodeNormGrouped(commandBuffer: cb,
                              x: hyper, weight: normW.buffer,
                              weightOffset: Int(normW.offset),
                              out: hyperNormed, dim: UInt32(D), groups: hc,
                              eps: eps)
        int4.encode(commandBuffer: cb,
                    weights: downW.buffer, weightsOffset: Int(downW.offset),
                    scales: downW.buffer, scalesOffset: Int(downW.scaleOffset),
                    biases: downW.buffer, biasesOffset: Int(downW.biasOffset),
                    x: hyperNormed, y: low,
                    m: UInt32(lowrank), n: UInt32(hcHidden),
                    groupSize: groupSize)
        q38.encodeSiluDiv(commandBuffer: cb, x: low, y: low,
                          div: Float(hc), count: lowrank)
        int4.encode(commandBuffer: cb,
                    weights: upW.buffer, weightsOffset: Int(upW.offset),
                    scales: upW.buffer, scalesOffset: Int(upW.scaleOffset),
                    biases: upW.buffer, biasesOffset: Int(upW.biasOffset),
                    x: low, y: mix,
                    m: UInt32(hcHidden), n: UInt32(lowrank),
                    groupSize: groupSize)
        q38.encodeSigmoid(commandBuffer: cb, x: mix, y: mix, count: hcHidden)
        q38.encodeMixMean(commandBuffer: cb, mix: mix, normed: hyperNormed,
                          mixed: mixedOut, d: D, h: hc)
        if let injectW, let injOut {
            int4.encode(commandBuffer: cb,
                        weights: injectW.buffer, weightsOffset: Int(injectW.offset),
                        scales: injectW.buffer, scalesOffset: Int(injectW.scaleOffset),
                        biases: injectW.buffer, biasesOffset: Int(injectW.biasOffset),
                        x: hyperNormed, y: injOut,
                        m: UInt32(hc), n: UInt32(hcHidden),
                        groupSize: groupSize)
            q38.encodeSigmoid2Div(commandBuffer: cb, x: injOut, y: injOut,
                                  div: Float(hc), count: hc)
        }
    }

    /// CPU half of the QSA indexer for one decode step. Reads back the
    /// `index_qk_proj` row (shared buffer, no copy), splits query/raw key,
    /// applies the reference layernorm + RoPE on CPU, appends the raw key to
    /// the layer tail, and finalizes newly completed `compress`-token blocks
    /// straight into the GPU pooled-key buffer (shared write). Returns the
    /// completed-block count visible to this query.
    private func q38IndexerPost(layer L: Int, position p: Int) throws -> Int {
        guard let idxQK = q38IdxQK, let idxQ = q38IdxQ else {
            preconditionFailure("qwen38 indexer without scratch")
        }
        let idxDim = cfg.indexerHeadDim
        let idxHeads = cfg.indexerNHeads
        let compress = cfg.indexerCompressRatio
        let rotaryDim = Int(Double(cfg.headDim) * cfg.partialRotaryFactor)
        let eps: Float = 1e-6
        // Read back the projected row (FP16 shared buffer).
        let src = idxQK.contents().assumingMemoryBound(to: UInt16.self)
        var qk = [Float](repeating: 0, count: (idxHeads + cfg.indexerKvHeads) * idxDim)
        for i in 0..<qk.count { qk[i] = Self.q38F32(src[i]) }
        let qFlat = Array(qk[0..<(idxHeads * idxDim)])
        let rawKey = Array(qk[(idxHeads * idxDim)...])
        // Query layernorm (shared [headDim] weight) + RoPE at p.
        let qNormW = try model.indexerQNorm(layer: L)
        let qw = qNormW.buffer.contents().advanced(by: Int(qNormW.offset))
            .assumingMemoryBound(to: UInt16.self)
        var qWeight = [Float](repeating: 0, count: idxDim)
        for i in 0..<idxDim { qWeight[i] = Quantization.bf16ToFloat(qw[i]) }
        var qHeads: [[Float]] = []
        qHeads.reserveCapacity(idxHeads)
        for h in 0..<idxHeads {
            let row = Array(qFlat[(h * idxDim)..<((h + 1) * idxDim)])
            let normed = Qwen38Reference.rmsNorm(row, weight: qWeight, eps: eps)
            qHeads.append(Qwen38Reference.applyRopeNeoxCPU(
                normed, position: p, headDim: idxDim, rotaryDim: rotaryDim,
                theta: cfg.ropeTheta))
        }
        // Publish the query heads to the GPU score buffer input.
        let qDst = idxQ.contents().assumingMemoryBound(to: UInt16.self)
        for h in 0..<idxHeads {
            for d in 0..<idxDim {
                qDst[h * idxDim + d] = Self.q38F16(qHeads[h][d])
            }
        }
        // Tail append + block finalization (mean -> k layernorm -> RoPE at
        // the group start -> shared write into the pooled cache).
        var tail = q38IdxTails[L]
        tail.append(contentsOf: rawKey)
        let visible = p + 1
        let completed = visible / compress
        let kNormW = try model.indexerKNorm(layer: L)
        let kw = kNormW.buffer.contents().advanced(by: Int(kNormW.offset))
            .assumingMemoryBound(to: UInt16.self)
        var kWeight = [Float](repeating: 0, count: idxDim)
        for i in 0..<idxDim { kWeight[i] = Quantization.bf16ToFloat(kw[i]) }
        let poolDst = q38PooledKeys[L].contents().assumingMemoryBound(to: UInt16.self)
        while tail.count >= compress * idxDim {
            let blockRows = Array(tail.prefix(compress * idxDim))
            tail.removeFirst(compress * idxDim)
            var mean = [Float](repeating: 0, count: idxDim)
            for t in 0..<compress {
                for d in 0..<idxDim { mean[d] += blockRows[t * idxDim + d] }
            }
            for d in 0..<idxDim { mean[d] /= Float(compress) }
            let normed = Qwen38Reference.rmsNorm(mean, weight: kWeight, eps: eps)
            // Group-start position of the block just completed.
            let blockIndex = (visible - tail.count / idxDim) / compress - 1
            let start = blockIndex * compress
            let rotated = Qwen38Reference.applyRopeNeoxCPU(
                normed, position: start, headDim: idxDim, rotaryDim: rotaryDim,
                theta: cfg.ropeTheta)
            for d in 0..<idxDim {
                poolDst[blockIndex * idxDim + d] = Self.q38F16(rotated[d])
            }
        }
        q38IdxTails[L] = tail
        return completed
    }

    /// CPU top-k over the GPU block scores. Reads back `completed` FP32
    /// scores (shared buffer), runs the reference block selection + tail
    /// passthrough, and publishes the selected token indices. Short contexts
    /// (no complete block) return the dense visible prefix, matching exact
    /// attention by construction.
    private func q38SelectTokens(layer L: Int, position p: Int,
                                 completed: Int) -> Int {
        guard let scores = q38Scores, let sel = q38Sel else {
            preconditionFailure("qwen38 QSA without scratch")
        }
        let compress = cfg.indexerCompressRatio
        let blockTopK = cfg.indexerBudget / compress
        let visible = Array(0...(p))
        let selected: [Int]
        if completed > 0 {
            let sPtr = scores.contents().assumingMemoryBound(to: Float.self)
            var blockScores = [Float](repeating: 0, count: completed)
            for b in 0..<completed { blockScores[b] = sPtr[b] }
            selected = Qwen38Reference.qsaSelectTokens(
                visible: visible, blockScores: blockScores,
                compress: compress, blockTopK: blockTopK)
        } else {
            selected = visible
        }
        let dst = sel.contents().assumingMemoryBound(to: UInt32.self)
        for (i, tok) in selected.enumerated() { dst[i] = UInt32(tok) }
        return selected.count
    }

    /// PLE injection for the PLE layer: CPU n-gram ids + SSD row reads +
    /// shared upload, then GPU key/value projections, gated mix, dilated
    /// short conv, and add-back into the hyper stream.
    private func q38PLE(token: Int32, pleLayer L: Int) throws {
        guard let q38, let pleReader, let elementwise,
              let pleEmbed = q38PleEmbed, let pleKey = q38PleKey,
              let pleValue = q38PleValue, let pleQuery = q38PleQuery,
              let pleDots = q38PleDots, let pleGate = q38PleGate,
              let pleGated = q38PleGated, let pleConvOut = q38PleConvOut,
              let pleConvState = q38PleConvState, let hyper = q38Hyper else {
            preconditionFailure("qwen38 PLE without scratch")
        }
        let D = cfg.hiddenSize
        let hc = cfg.hcCount
        let hcHidden = hc * D
        let ngramSize = cfg.pleNgramSize
        let headsPerNgram = cfg.pleHeadsPerNgram
        let ngramHeads = (ngramSize - 1) * headsPerNgram
        let eps: Float = 1e-6
        let attnGroup = model.attentionGroupSize
        // CPU: n-gram row ids for the current token from the trailing
        // context window (EOS-padded at stream start, like the reference's
        // conv_states[2] init).
        let contextLen = ngramSize - 1
        var window = q38PleHistory.suffix(contextLen).map { Int($0) }
        while window.count < contextLen { window.insert(Int(q38PleEos), at: 0) }
        window.append(Int(token))
        let multView = try model.pleMultipliers(layer: L)
        let offView = try model.pleHeadsOffsets(layer: L)
        let vszView = try model.pleHeadsVocabSizes(layer: L)
        let multPtr = multView.buffer.contents().advanced(by: Int(multView.offset))
            .assumingMemoryBound(to: UInt64.self)
        let offPtr = offView.buffer.contents().advanced(by: Int(offView.offset))
            .assumingMemoryBound(to: Int64.self)
        let vszPtr = vszView.buffer.contents().advanced(by: Int(vszView.offset))
            .assumingMemoryBound(to: Int64.self)
        let multipliers = (0..<ngramSize).map { multPtr[$0] }
        let headOffsets = (0..<ngramHeads).map { Int(offPtr[$0]) }
        let headVocabSizes = (0..<ngramHeads).map { Int(vszPtr[$0]) }
        let rows = Qwen38Reference.ngramRowIds(
            history: window, seqLen: 1, ngramSize: ngramSize,
            headsPerNgram: headsPerNgram, multipliers: multipliers,
            headVocabSizes: headVocabSizes, headOffsets: headOffsets,
            eosId: Int(q38PleEos))
        let rps = try pleReader.rowsPerShard
        // SSD row reads (one per head) + dequant + shared upload.
        let embDst = pleEmbed.contents().assumingMemoryBound(to: UInt16.self)
        let headDim = cfg.pleEmbedDim / ngramHeads
        for h in 0..<ngramHeads {
            let vec = try pleReader.readRow(globalRow: rows[0][h], rowsPerShard: rps)
            precondition(vec.count == headDim)
            for d in 0..<headDim {
                embDst[h * headDim + d] = Self.q38F16(vec[d])
            }
        }
        // GPU compose.
        let keyW = try model.pleKeyProj(layer: L)
        let valW = try model.pleValueProj(layer: L)
        let convW = try model.pleConv1d(layer: L)
        let nConvW = try model.pleNormConv(layer: L)
        let nKeyW = try model.pleNormKey(layer: L)
        let nQueryW = try model.pleNormQuery(layer: L)
        runSync { cb in
            self.int4.encode(commandBuffer: cb,
                             weights: keyW.buffer, weightsOffset: Int(keyW.offset),
                             scales: keyW.buffer, scalesOffset: Int(keyW.scaleOffset),
                             biases: keyW.buffer, biasesOffset: Int(keyW.biasOffset),
                             x: pleEmbed, y: pleKey,
                             m: UInt32(hcHidden), n: UInt32(cfg.pleEmbedDim),
                             groupSize: attnGroup)
            self.int4.encode(commandBuffer: cb,
                             weights: valW.buffer, weightsOffset: Int(valW.offset),
                             scales: valW.buffer, scalesOffset: Int(valW.scaleOffset),
                             biases: valW.buffer, biasesOffset: Int(valW.biasOffset),
                             x: pleEmbed, y: pleValue,
                             m: UInt32(D), n: UInt32(cfg.pleEmbedDim),
                             groupSize: attnGroup)
            q38.encodeNormGrouped(commandBuffer: cb, x: pleKey,
                                  weight: nKeyW.buffer,
                                  weightOffset: Int(nKeyW.offset),
                                  out: pleKey, dim: UInt32(D), groups: hc,
                                  eps: eps)
            q38.encodeNormGrouped(commandBuffer: cb, x: hyper,
                                  weight: nQueryW.buffer,
                                  weightOffset: Int(nQueryW.offset),
                                  out: pleQuery, dim: UInt32(D), groups: hc,
                                  eps: eps)
            q38.encodeGateDot(commandBuffer: cb, key: pleKey, query: pleQuery,
                              dots: pleDots, d: D, h: hc)
        }
        // CPU: signed-sqrt gate transform (4 floats, shared buffer).
        let dotPtr = pleDots.contents().assumingMemoryBound(to: Float.self)
        let gatePtr = pleGate.contents().assumingMemoryBound(to: Float.self)
        for h in 0..<hc {
            let dot = dotPtr[h]
            let mag = sqrt(max(1e-6, abs(dot)))
            gatePtr[h] = (dot >= 0 ? 1.0 : -1.0) * mag
        }
        runSync { cb in
            q38.encodeScaleRows(commandBuffer: cb, value: pleValue,
                                gate: pleGate, out: pleGated, d: D, h: hc)
            q38.encodeNormGrouped(commandBuffer: cb, x: pleGated,
                                  weight: nConvW.buffer,
                                  weightOffset: Int(nConvW.offset),
                                  out: pleGated, dim: UInt32(D), groups: hc,
                                  eps: eps)
            q38.encodeShortConvStep(
                commandBuffer: cb, state: pleConvState, input: pleGated,
                weight: convW.buffer, weightOffset: Int(convW.offset),
                out: pleConvOut, channels: hcHidden, kernel: 4,
                dilation: ngramSize, stateLen: 9)
            elementwise.encodeResidualAdd(commandBuffer: cb,
                                                hidden: pleGated,
                                                delta: pleConvOut,
                                                count: hcHidden)
            elementwise.encodeResidualAdd(commandBuffer: cb,
                                                hidden: hyper,
                                                delta: pleGated,
                                                count: hcHidden)
        }
        if Foundation.getenv("TURBO_FIELDFARE_DEBUG_QWEN38") != nil {
            q38DebugProbe("pleEmbed", pleEmbed, cfg.pleEmbedDim)
            q38DebugProbe("pleValue", pleValue, D)
            q38DebugProbe("pleGated", pleGated, hcHidden)
            q38DebugProbe("pleConvOut", pleConvOut, hcHidden)
            if q38PleHistory.isEmpty {
                q38DebugRefCheckPLE(token: token, pleLayer: L)
                q38DebugRefCheckPLEGateConv(pleLayer: L)
            }
        }
    }




    /// Qwen 3.8 per-token decode. Dataflow per token (reference:
    /// Qwen4ExpTextModel / DecoderLayer):
    /// embed -> repeat x4 -> [PLE add] -> per layer {attn hyper-mix ->
    /// (GDN | QSA sparse) -> inject -> mlp hyper-mix -> MoE(top-10+shared) ->
    /// inject} -> mixer -> head. QSA layers extend the KV cache like full
    /// attention; GDN layers use the recurrent state. PLE history and indexer
    /// tails are CPU-side; pooled index keys persist per QSA layer.
    private func produceTokenQwen38(token: Int32,
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
        guard let q38, let elementwise, let rope,
              let hyper = q38Hyper, let hyperNormed = q38HyperNormed,
              let attnIn = q38AttnIn, let mlpIn = q38MlpIn,
              let inj = q38Inj, let idxQK = q38IdxQK, let idxQ = q38IdxQ,
              let scores = q38Scores, let sel = q38Sel,
              let compactK = q38CompactK, let compactV = q38CompactV,
              let qPackedScratch, let attnGateScratch,
              let kv, let onesPerExpert = onesPerExpertScale else {
            preconditionFailure("qwen38 decode without kernels/scratch")
        }
        _ = hyperNormed; _ = idxQ; _ = scores; _ = sel
        let D = UInt32(cfg.hiddenSize)
        let FmoE = UInt32(cfg.moeIntermediateSize)
        let eps: Float = 1e-6
        let attnGroup = model.attentionGroupSize
        let sharedGroup = model.sharedExpertGroupSize
        let hc = cfg.hcCount
        let headDim = cfg.fullHeadDim
        let numKV = cfg.numFullKVHeads
        let qDim = UInt32(cfg.numHeads * headDim)
        let kvDim = UInt32(numKV * headDim)
        let idxDim = cfg.indexerHeadDim
        let idxHeads = cfg.indexerNHeads
        let idxRows = UInt32((idxHeads + cfg.indexerKvHeads) * idxDim)
        let rotaryDim = UInt32(Double(cfg.fullHeadDim) * cfg.partialRotaryFactor)
        let topK = UInt32(cfg.topKExperts)
        var pendingQ38: [MTLCommandBuffer] = []
        let dbgQ38 = Foundation.getenv("TURBO_FIELDFARE_DEBUG_QWEN38") != nil
        // Deep-cache verification position (default: last prefill token of
        // the 13-token Hello prompt). Exercises full KV/GDN carry.
        let deepPos: Int = {
            guard let c = Foundation.getenv("TURBO_FIELDFARE_Q38_DEEPPOS") else { return 12 }
            return Int(String(cString: c)) ?? 12
        }()
        // GDN-deep: advance the CPU mirror through every prefill position
        // (needed to verify recurrence at depth). Slow (~2 min); opt-in.
        let gdnDeep = Foundation.getenv("TURBO_FIELDFARE_Q38_GDNDEEP") != nil
        // All-layer sweep for single-tensor corruption (~6 min); opt-in.
        let sweepQ38 = Foundation.getenv("TURBO_FIELDFARE_Q38_SWEEP") != nil

        // Embed + 4-branch expand.
        let emb = model.embedding
        runSync { cb in
            self.embedInt4.encode(commandBuffer: cb,
                                  table: emb.buffer, tableOffset: Int(emb.offset),
                                  scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                                  biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                                  out: self.hidden,
                                  tokenId: UInt32(bitPattern: token),
                                  d: D, outScale: 1.0,
                                  groupSize: self.model.embeddingGroupSize)
            q38.encodeRepeat(commandBuffer: cb, x: self.hidden, y: hyper,
                             d: self.cfg.hiddenSize, h: hc)
        }
        if dbgQ38 && position == 0 {
            q38DebugProbe("embed", self.hidden, self.cfg.hiddenSize)
        }

        let pleLayer = model.pleLayer()
        let sharedProjViews = sharedExpertProjections
        if dbgQ38 && position == 0 {
            do {
                let inj0 = try model.attnHcBlockInject(layer: 0)
                q38DebugDumpQuant("inj0", inj0, rows: hc,
                                  groupsPerRow: hc * cfg.hiddenSize / 32)
                let sg0 = try model.sharedExpertGate(layer: 0)
                q38DebugDumpQuant("sharedGate0", sg0, rows: cfg.intermediateSize,
                                  groupsPerRow: cfg.hiddenSize / 32)
                if let ple = pleLayer {
                    let pv = try model.pleValueProj(layer: ple)
                    q38DebugDumpQuant("pleValue", pv, rows: cfg.hiddenSize,
                                      groupsPerRow: cfg.pleEmbedDim / 32)
                    let nc = try model.pleNormConv(layer: ple)
                    q38DebugDumpBF16("pleNormConv", nc, count: hc * cfg.hiddenSize)
                }
                let hn = try model.attnHcNorm(layer: 0)
                q38DebugDumpBF16("hcNorm0", hn, count: hc * cfg.hiddenSize)
                q38DebugSelfTestGEMV()
            } catch {
                q38DebugDump("dumpError", "\(error)")
            }
        }

        for L in 0..<cfg.numLayers {
            // Temporary kill-switch (TURBO_FIELDFARE_DISABLE_PLE=1) to isolate
            // PLE as the garbage-output suspect. No-op when unset.
            if let ple = pleLayer, ple == L,
               Foundation.getenv("TURBO_FIELDFARE_DISABLE_PLE") == nil {
                try q38PLE(token: token, pleLayer: ple)
            }
            let routerW = try model.router(layer: L)
            let isLinear = cfg.layerIsLinear(L)

            if isLinear {
                // One CB through the router: attn hyper-mix (with inject),
                // GDN recurrence, attn inject, mlp hyper-mix (with inject).
                let cb = ctx.queue.makeCommandBuffer()!
                try q38HyperMix(cb, hyper: hyper,
                                normW: try model.attnHcNorm(layer: L),
                                downW: try model.attnHcMixDown(layer: L),
                                upW: try model.attnHcMixUp(layer: L),
                                injectW: try model.attnHcBlockInject(layer: L),
                                mixedOut: attnIn, injOut: inj,
                                groupSize: attnGroup)
                try encodeLinearAttentionDecode(cb, layer: L, from: attnIn,
                                                outGroupSize: attnGroup)
                q38.encodeInjectAdd(commandBuffer: cb, hyper: hyper,
                                    sub: oOut, inj: inj,
                                    d: cfg.hiddenSize, h: hc)
                try q38HyperMix(cb, hyper: hyper,
                                normW: try model.mlpHcNorm(layer: L),
                                downW: try model.mlpHcMixDown(layer: L),
                                upW: try model.mlpHcMixUp(layer: L),
                                injectW: try model.mlpHcBlockInject(layer: L),
                                mixedOut: mlpIn, injOut: inj,
                                groupSize: attnGroup)
                moe.encodeRouterGemma4(
                    commandBuffer: cb,
                    weights: routerW.buffer, weightsOffset: Int(routerW.offset),
                    scales: routerW.buffer, scalesOffset: Int(routerW.scaleOffset),
                    biases: routerW.buffer, biasesOffset: Int(routerW.biasOffset),
                    hidden: mlpIn,
                    effectiveScale: effectiveScaleBuffers[L],
                    perExpertScale: onesPerExpert, perExpertScaleOffset: 0,
                    outIndices: outIndices, outWeights: outWeights,
                    numExperts: UInt32(cfg.numExperts), d: D, topK: topK,
                    weightBits: routerWeightBits)
                cb.commit()
                waitForCompletion(cb)
            } else {
                // CB1: attn hyper-mix (with inject), packed q/k/v, per-head
                // norms, RoPE, indexer projection.
                let cb1 = ctx.queue.makeCommandBuffer()!
                try q38HyperMix(cb1, hyper: hyper,
                                normW: try model.attnHcNorm(layer: L),
                                downW: try model.attnHcMixDown(layer: L),
                                upW: try model.attnHcMixUp(layer: L),
                                injectW: try model.attnHcBlockInject(layer: L),
                                mixedOut: attnIn, injOut: inj,
                                groupSize: attnGroup)
                let q = try model.qProj(layer: L)
                let k = try model.kProj(layer: L)
                let v = try model.vProj(layer: L)
                let kSlot = kv.kSlot(layer: L, position: position)
                let vSlot = kv.vSlot(layer: L, position: position)
                fusedQKVGEMV.encode(
                    commandBuffer: cb1,
                    qWeights: q.buffer, qWeightsOffset: Int(q.offset),
                    qScales: q.buffer, qScalesOffset: Int(q.scaleOffset),
                    qBiases: q.buffer, qBiasesOffset: Int(q.biasOffset),
                    kWeights: k.buffer, kWeightsOffset: Int(k.offset),
                    kScales: k.buffer, kScalesOffset: Int(k.scaleOffset),
                    kBiases: k.buffer, kBiasesOffset: Int(k.biasOffset),
                    vWeights: v.buffer, vWeightsOffset: Int(v.offset),
                    vScales: v.buffer, vScalesOffset: Int(v.scaleOffset),
                    vBiases: v.buffer, vBiasesOffset: Int(v.biasOffset),
                    x: attnIn,
                    qOut: qPackedScratch,
                    kOut: kSlot.buffer, kOutOffset: kSlot.offset,
                    vOut: vSlot.buffer, vOutOffset: vSlot.offset,
                    qRows: 2 * qDim, kvRows: kvDim, n: D,
                    groupSize: attnGroup)
                elementwise.encodeSplitQGate(
                    commandBuffer: cb1, packed: qPackedScratch,
                    q: qScratch, gate: attnGateScratch,
                    heads: cfg.numHeads, dim: headDim)
                let qNormW = try model.qNorm(layer: L)
                let kNormW = try model.kNorm(layer: L)
                // Qwen 3.8 Q/K norms are zeros-init (1 + w); older families
                // are ones-init (w). The 1-centered kernel is QSA-only.
                if cfg.family == .qwen38 {
                    rms.encodeOnePlusBF16WPerHead(
                        commandBuffer: cb1, x: qScratch,
                        weight: qNormW.buffer, weightOffset: Int(qNormW.offset),
                        out: qScratch, headDim: UInt32(headDim),
                        numHeads: cfg.numHeads, eps: eps)
                    rms.encodeOnePlusBF16WPerHead(
                        commandBuffer: cb1,
                        x: kSlot.buffer, xOffset: kSlot.offset,
                        weight: kNormW.buffer, weightOffset: Int(kNormW.offset),
                        out: kSlot.buffer, outOffset: kSlot.offset,
                        headDim: UInt32(headDim), numHeads: numKV, eps: eps)
                } else {
                    rms.encodeBF16WPerHead(
                        commandBuffer: cb1, x: qScratch,
                        weight: qNormW.buffer, weightOffset: Int(qNormW.offset),
                        out: qScratch, headDim: UInt32(headDim),
                        numHeads: cfg.numHeads, eps: eps)
                    rms.encodeBF16WPerHead(
                        commandBuffer: cb1,
                        x: kSlot.buffer, xOffset: kSlot.offset,
                        weight: kNormW.buffer, weightOffset: Int(kNormW.offset),
                        out: kSlot.buffer, outOffset: kSlot.offset,
                        headDim: UInt32(headDim), numHeads: numKV, eps: eps)
                }
                rope.encodeNeoxSubdim(
                    commandBuffer: cb1, data: qScratch,
                    position: UInt32(position),
                    headDim: UInt32(headDim),
                    numHeads: UInt32(cfg.numHeads),
                    rotaryDim: rotaryDim,
                    theta: Float(cfg.fullRopeTheta))
                rope.encodeNeoxSubdim(
                    commandBuffer: cb1, data: kSlot.buffer,
                    dataOffset: kSlot.offset,
                    position: UInt32(position),
                    headDim: UInt32(headDim),
                    numHeads: UInt32(numKV),
                    rotaryDim: rotaryDim,
                    theta: Float(cfg.fullRopeTheta))
                let iqk = try model.indexerQKProj(layer: L)
                int4.encode(commandBuffer: cb1,
                            weights: iqk.buffer, weightsOffset: Int(iqk.offset),
                            scales: iqk.buffer, scalesOffset: Int(iqk.scaleOffset),
                            biases: iqk.buffer, biasesOffset: Int(iqk.biasOffset),
                            x: attnIn, y: idxQK, m: idxRows, n: D,
                            groupSize: attnGroup)
                cb1.commit()
                waitForCompletion(cb1)
                if dbgQ38 && position == 0 && L == 3 {
                    q38DebugProbe("qNormed.L3", qScratch, cfg.numHeads * headDim)
                    q38DebugRefCheckQPack(attnIn: attnIn, qPacked: qPackedScratch, layer: L)
                }
                // CPU: query finalize + block finalize into the pooled cache.
                let completed = try q38IndexerPost(layer: L, position: position)
                if completed > 0 {
                    runSync { cb in
                        q38.encodeBlockScore(
                            commandBuffer: cb, q: idxQ, k: self.q38PooledKeys[L],
                            scores: scores, headDim: idxDim, nHeads: idxHeads,
                            blockCount: completed)
                    }
                }
                // CPU: top-k + tail -> sel.
                let selCount = q38SelectTokens(layer: L, position: position,
                                               completed: completed)
                // CB2: gather subset -> dense attention -> gate -> o_proj ->
                // attn inject -> mlp hyper-mix -> router.
                let cb2 = ctx.queue.makeCommandBuffer()!
                let base = kv.gatherBase(layer: L)
                q38.encodeGatherRows(commandBuffer: cb2, src: base.k, sel: sel,
                                     dst: compactK, rowElems: Int(kvDim),
                                     rows: selCount)
                q38.encodeGatherRows(commandBuffer: cb2, src: base.v, sel: sel,
                                     dst: compactV, rowElems: Int(kvDim),
                                     rows: selCount)
                attention.encodeFull(
                    commandBuffer: cb2, q: qScratch,
                    k: compactK, v: compactV, out: attnOut,
                    headDim: UInt32(headDim),
                    numQHeads: UInt32(cfg.numHeads),
                    numKVHeads: UInt32(numKV),
                    seqLen: UInt32(selCount),
                    scale: Float(cfg.attentionScale))
                elementwise.encodeSigmoidGateMul(
                    commandBuffer: cb2, out: attnOut, gate: attnGateScratch,
                    count: Int(qDim))
                let o = try model.oProj(layer: L)
                int4.encode(commandBuffer: cb2,
                            weights: o.buffer, weightsOffset: Int(o.offset),
                            scales: o.buffer, scalesOffset: Int(o.scaleOffset),
                            biases: o.buffer, biasesOffset: Int(o.biasOffset),
                            x: attnOut, y: oOut, m: D, n: qDim,
                            groupSize: attnGroup)
                q38.encodeInjectAdd(commandBuffer: cb2, hyper: hyper,
                                    sub: oOut, inj: inj,
                                    d: cfg.hiddenSize, h: hc)
                try q38HyperMix(cb2, hyper: hyper,
                                normW: try model.mlpHcNorm(layer: L),
                                downW: try model.mlpHcMixDown(layer: L),
                                upW: try model.mlpHcMixUp(layer: L),
                                injectW: try model.mlpHcBlockInject(layer: L),
                                mixedOut: mlpIn, injOut: inj,
                                groupSize: attnGroup)
                moe.encodeRouterGemma4(
                    commandBuffer: cb2,
                    weights: routerW.buffer, weightsOffset: Int(routerW.offset),
                    scales: routerW.buffer, scalesOffset: Int(routerW.scaleOffset),
                    biases: routerW.buffer, biasesOffset: Int(routerW.biasOffset),
                    hidden: mlpIn,
                    effectiveScale: effectiveScaleBuffers[L],
                    perExpertScale: onesPerExpert, perExpertScaleOffset: 0,
                    outIndices: outIndices, outWeights: outWeights,
                    numExperts: UInt32(cfg.numExperts), d: D, topK: topK,
                    weightBits: routerWeightBits)
                cb2.commit()
                waitForCompletion(cb2)
                if dbgQ38 && position == 0 && L == 3 {
                    q38DebugProbe("compactV.L3", compactV, Int(kvDim))
                    q38DebugProbe("attnOut.L3", self.attnOut, cfg.numHeads * headDim)
                    q38DebugDump("selCount.L3", "\(selCount)")
                    q38DebugRefCheckQSA(attnIn: attnIn, oOut: oOut, compactV: compactV, layer: L,
                                         position: position, kv: kv)
                }
            }

            // Router readback -> SSD fetch (overlaps the shared expert below).
            let idxPtr = outIndices.contents().bindMemory(to: UInt32.self,
                                                          capacity: cfg.topKExperts)
            var experts = [Int](repeating: 0, count: cfg.topKExperts)
            for i in 0..<cfg.topKExperts {
                experts[i] = min(Int(idxPtr[i]), cfg.numExperts - 1)
            }
            let routedOffsets = model.routedExpertOffsets(layer: L)
            let sharedCB = ctx.queue.makeCommandBuffer()!
            let proj = sharedProjViews[L]
            try! shared.encode(commandBuffer: sharedCB, x: mlpIn,
                               gate: proj.gate, up: proj.up, down: proj.down,
                               y: h1Buf,
                               scratchGate: denseScratchGate,
                               scratchUp: denseScratchUp,
                               scratchAct: denseScratchAct,
                               groupSize: sharedGroup)
            let gateView = proj.scalarGate!
            if routerWeightBits == 4 {
                int4.encode(commandBuffer: sharedCB,
                            weights: gateView.buffer,
                            weightsOffset: Int(gateView.offset),
                            scales: gateView.buffer,
                            scalesOffset: Int(gateView.scaleOffset),
                            biases: gateView.buffer,
                            biasesOffset: Int(gateView.biasOffset),
                            x: mlpIn, y: sharedScalarGateBuf!,
                            m: 1, n: D,
                            groupSize: attnGroup)
            } else {
                int8ScalarGate!.encode(
                    commandBuffer: sharedCB,
                    weights: gateView.buffer,
                    weightsOffset: Int(gateView.offset),
                    scales: gateView.buffer,
                    scalesOffset: Int(gateView.scaleOffset),
                    biases: gateView.buffer,
                    biasesOffset: Int(gateView.biasOffset),
                    x: mlpIn, y: sharedScalarGateBuf!,
                    m: 1, n: D)
            }
            elementwise.encodeSigmoidScalarMul(
                commandBuffer: sharedCB, y: h1Buf,
                gate: sharedScalarGateBuf!, count: cfg.hiddenSize)
            sharedCB.commit()
            if rdadviseEnabled, rdadvisePolicyMode != .off {
                do {
                    let estimated = try model.routedExpertAdviceByteEstimate(
                        layer: L, missCount: experts.count)
                    if let skipped = shouldSkipRDAdvice(
                        position: position, requestedMisses: experts.count,
                        estimatedBytes: estimated, canOverlapUsefulGPUWork: true) {
                        recordRDAdvice(skipped, wallNanos: 0)
                    } else {
                        let tAdvice = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                        let result = try model.adviseRoutedExperts(
                            layer: L, experts: experts)
                        recordRDAdvice(result,
                                       wallNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tAdvice)
                        updateRDAdvicePolicy(after: result, position: position)
                    }
                } catch {
                    // Advice is best-effort; a failure must not fail decode.
                }
            }
            let tIoStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let blobs = try await model.fetchRoutedExperts(layer: L, experts: experts)
            totalIoNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tIoStart
            let routedBufs = blobs.map { $0.buffer }
            let routedCB = ctx.queue.makeCommandBuffer()!
            let argBuf = moe.makeReusedRoutedArgumentBuffer(
                routedBlobs: routedBufs, topK: topK)
            moe.encodeRoutedPersistentPhase1U16Load(
                commandBuffer: routedCB, routedArgBuffer: argBuf,
                routedBlobs: routedBufs, routedOffsets: routedOffsets,
                x: mlpIn, acts: moeActs, d: D, f: FmoE, topK: topK)
            moe.encodeRoutedPersistentPhase2Reduce(
                commandBuffer: routedCB, routedArgBuffer: argBuf,
                routedBlobs: routedBufs, routedOffsets: routedOffsets,
                acts: moeActs, routingWeights: outWeights,
                residual: h1Buf, y: h2Buf, d: D, f: FmoE, topK: topK)
            q38.encodeInjectAdd(commandBuffer: routedCB, hyper: hyper,
                                sub: h2Buf, inj: inj,
                                d: cfg.hiddenSize, h: hc)
            routedCB.commit()
            pendingQ38.append(sharedCB)
            pendingQ38.append(routedCB)
            if dbgQ38 && position == 0 {
                waitForCompletion(routedCB)
                q38DebugProbe("hyper.L\(L)", hyper, hc * cfg.hiddenSize)
                q38DebugProbe("attnIn.L\(L)", attnIn, cfg.hiddenSize)
                q38DebugProbe("oOut.L\(L)", oOut, cfg.hiddenSize)
                q38DebugProbe("mlpIn.L\(L)", mlpIn, cfg.hiddenSize)
                q38DebugProbe("h2.L\(L)", h2Buf, cfg.hiddenSize)
                q38DebugProbe("inj.L\(L)", inj, hc)
                q38DebugProbe("h1.L\(L)", h1Buf, cfg.hiddenSize)
                if L == 0 {
                    q38DebugRefCheckGate(mlpIn: mlpIn, layer: L)
                    q38DebugRefCheckRouter(mlpIn: mlpIn, layer: L)
                    q38DebugRefCheckGDN(attnIn: attnIn, oOut: oOut, layer: L,
                                        position: position)
                    q38DebugRefCheckExpertMap(layer: L, experts: experts, blobs: blobs)
                    q38DebugRefCheckRouted(mlpIn: mlpIn, h1: h1Buf, h2: h2Buf, layer: L,
                                           experts: experts, blobs: blobs)
                    let wPtr = outWeights.contents().assumingMemoryBound(to: UInt16.self)
                    var wSum: Float = 0
                    for i in 0..<cfg.topKExperts { wSum += Self.q38F32(wPtr[i]) }
                    let iPtr = outIndices.contents().assumingMemoryBound(to: UInt32.self)
                    var iMax: UInt32 = 0
                    var seen = Set<UInt32>()
                    for i in 0..<cfg.topKExperts {
                        iMax = max(iMax, iPtr[i])
                        seen.insert(iPtr[i])
                    }
                    let msg = "[q38dbg] router wsum=\(wSum) imax=\(iMax) uniq=\(seen.count)\n"
                    if let d = msg.data(using: .utf8) { FileHandle.standardError.write(d) }
                }
                // All-layer sweep (opt-in): GDN-oOut for linear layers,
                // QSA-oOut for full layers. Catches single-tensor repack
                // corruption hiding in unchecked layers.
                if sweepQ38 && L > 0 {
                    if cfg.layerIsLinear(L) {
                        q38DebugRefCheckGDN(attnIn: attnIn, oOut: oOut, layer: L,
                                            position: position)
                    } else {
                        q38DebugRefCheckQSA(attnIn: attnIn, oOut: oOut, compactV: compactV, layer: L,
                                            position: position, kv: kv)
                    }
                }
            }
            if dbgQ38 && (position == 1 || (gdnDeep && position > 1 && position <= 13)) && L == 0 {
                waitForCompletion(routedCB)
                q38DebugRefCheckGDN(attnIn: attnIn, oOut: oOut, layer: L,
                                    position: position)
            }
            if dbgQ38 && position == 1 && L == 3 {
                q38DebugRefCheckQSA(attnIn: attnIn, oOut: oOut, compactV: compactV, layer: L,
                                    position: position, kv: kv)
            }
            if dbgQ38 && position == deepPos && L == 0 {
                waitForCompletion(routedCB)
                q38DebugRefCheckGate(mlpIn: mlpIn, layer: L)
                q38DebugRefCheckRouter(mlpIn: mlpIn, layer: L)
            }
            if dbgQ38 && position == deepPos && L == 3 {
                // cb2 waited unconditionally in the QSA path above.
                q38DebugRefCheckQSA(attnIn: attnIn, oOut: oOut, compactV: compactV, layer: L,
                                    position: position, kv: kv)
            }
        }
        // Drain per-layer work (serial queue: waiting the last covers all).
        if let last = pendingQ38.last { waitForCompletion(last) }
        for cb in pendingQ38 {
            if let err = cb.error { print("CB error: \(err)") }
        }

        // Mixer closes the stream (no block inject, no final norm).
        let mixerNorm = model.mixerHcNorm
        let mixerDown = model.mixerDown
        let mixerUp = model.mixerUp
        runSync { cb in
            try! self.q38HyperMix(cb, hyper: hyper,
                                  normW: mixerNorm,
                                  downW: mixerDown,
                                  upW: mixerUp,
                                  injectW: nil, mixedOut: self.normed,
                                  injOut: nil,
                                  groupSize: attnGroup)
        }
        if dbgQ38 && position == 0 {
            q38DebugProbe("mixer", self.normed, self.cfg.hiddenSize)
            q38DebugRefCheckMixer(hyper: hyper, mixed: self.normed)
            q38DebugRefCheckHead()
            q38DebugRefCheckRope()
        }

        if emitHead {
            let useFusedHeadForThisToken = useFusedGreedyHead && outputMode == .greedyIfAvailable
            let lm = model.lmHead
            if useFusedHeadForThisToken {
                let tHead = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                runSync { cb in
                    self.fusionHead.encodeGreedyDecode(
                        commandBuffer: cb, hidden: self.normed,
                        normWeight: self.normed, normOffset: 0,
                        weights: lm.buffer, weightsOffset: Int(lm.offset),
                        scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                        biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                        outToken: self.greedyTokenBuf,
                        d: D, vocab: UInt32(self.cfg.vocabSize),
                        rmsEps: eps, groupSize: attnGroup, skipNorm: true)
                }
                totalHeadFusedNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tHead
                lastGreedyToken = greedyTokenBuf.contents().load(as: UInt32.self)
                if dbgQ38 && position == 0 {
                    let msg = "[q38dbg] token=\(lastGreedyToken)\n"
                    if let d = msg.data(using: .utf8) { FileHandle.standardError.write(d) }
                }
            } else {
                let tHead = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                runSync { cb in
                    self.int4.encode(commandBuffer: cb,
                                     weights: lm.buffer,
                                     weightsOffset: Int(lm.offset),
                                     scales: lm.buffer,
                                     scalesOffset: Int(lm.scaleOffset),
                                     biases: lm.buffer,
                                     biasesOffset: Int(lm.biasOffset),
                                     x: self.normed, y: logits,
                                     m: UInt32(self.cfg.vocabSize), n: D,
                                     groupSize: attnGroup)
                }
                totalHeadNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tHead
            }
        }

        kv.advance()
        q38PleHistory.append(token)
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

}
