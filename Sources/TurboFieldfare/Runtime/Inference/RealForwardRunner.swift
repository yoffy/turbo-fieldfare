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
    private func q38DebugProbe(_ label: String, _ buf: MTLBuffer, _ count: Int) {
        let ptr = buf.contents().assumingMemoryBound(to: UInt16.self)
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

        for L in 0..<cfg.numLayers {
            if let ple = pleLayer, ple == L {
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
            if dbgQ38 && position == 0 && L <= 3 {
                waitForCompletion(routedCB)
                q38DebugProbe("hyper.L\(L)", hyper, hc * cfg.hiddenSize)
                q38DebugProbe("attnIn.L\(L)", attnIn, cfg.hiddenSize)
                q38DebugProbe("oOut.L\(L)", oOut, cfg.hiddenSize)
                q38DebugProbe("mlpIn.L\(L)", mlpIn, cfg.hiddenSize)
                q38DebugProbe("h2.L\(L)", h2Buf, cfg.hiddenSize)
                q38DebugProbe("inj.L\(L)", inj, hc)
                q38DebugProbe("h1.L\(L)", h1Buf, cfg.hiddenSize)
                if L == 0 {
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
