import Foundation

/// CPU reference math for Qwen3.8-Flash-Next (`qwen4_exp` text), ported
/// exactly from `transformers` `modular_qwen4_exp.py` /
/// `modeling_qwen4_exp.py`.
///
/// Purpose: give the Metal kernels and the `RealForwardRunner` wiring a
/// single auditable source of truth that synthetic unit tests can compare
/// against without downloading the 111 GB checkpoint. All functions operate
/// on plain `[Float]` vectors (FP32) and mirror the reference's FP32
/// accumulations; the GPU kernels stay in FP16/BF16 and are allowed the
/// usual small rounding slack.
///
/// Naming mirrors the reference:
/// `Qwen4ExpTextGatedResidual`, `Qwen4ExpTextQSAIndexer`,
/// `Qwen4ExpTextNGramEmbedding`, `Qwen4ExpTextPLELayer`,
/// `Qwen4ExpTextTopKRouter`, `Qwen4ExpTextRMSNorm`.
public enum Qwen38Reference {

    // MARK: - Activations

    @inline(__always)
    public static func sigmoid(_ x: Float) -> Float {
        1.0 / (1.0 + exp(-x))
    }

    @inline(__always)
    public static func silu(_ x: Float) -> Float {
        x * sigmoid(x)
    }

    // MARK: - RMSNorm (reference: weight zeros, scale is 1 + w)

    /// `Qwen4ExpTextRMSNorm`: grouped when `groupSize != nil`.
    /// `weight` holds the raw zeros-init parameter (scale = 1 + w).
    public static func rmsNorm(_ x: [Float], weight: [Float],
                               groupSize: Int? = nil,
                               eps: Float = 1e-6) -> [Float] {
        precondition(weight.count == x.count)
        if let g = groupSize {
            precondition(x.count % g == 0)
            var out = [Float](repeating: 0, count: x.count)
            var base = 0
            while base < x.count {
                var mean: Float = 0
                for k in 0..<g { mean += x[base + k] * x[base + k] }
                mean /= Float(g)
                let inv = 1.0 / sqrt(mean + eps)
                for k in 0..<g {
                    out[base + k] = x[base + k] * inv * (1.0 + weight[base + k])
                }
                base += g
            }
            return out
        }
        var mean: Float = 0
        for v in x { mean += v * v }
        mean /= Float(x.count)
        let inv = 1.0 / sqrt(mean + eps)
        return zip(x, weight).map { $0 * inv * (1.0 + $1) }
    }

    // MARK: - Gated residual (hyper-connection)

    public struct GatedResidualResult {
        /// Mixed `[hidden]` input to the sub-layer (attn or MLP).
        public let mixed: [Float]
        /// Normed hyper input `[hc * hidden]` (kept for the inject add).
        public let hyperNormed: [Float]
        /// Injection weights `[hc]` (nil when `useCombine == false`, i.e.
        /// the final mixer).
        public let injection: [Float]?
    }

    /// Exact port of `Qwen4ExpTextGatedResidual.forward`.
    ///
    /// - `hyperInput`: `[hcCount * hidden]`.
    /// - `hcNormWeight`: `[hcCount * hidden]` (zeros-init).
    /// - `downWeight`: `[lowrank, hcHidden]` row-major.
    /// - `upWeight`: `[hcHidden, lowrank]` row-major.
    /// - `blockInjectWeight`: `[hcCount, hcHidden]` row-major, nil for mixer.
    public static func gatedResidual(hyperInput: [Float],
                                     hcCount: Int,
                                     hidden: Int,
                                     lowrank: Int,
                                     hcNormWeight: [Float],
                                     downWeight: [Float],
                                     upWeight: [Float],
                                     blockInjectWeight: [Float]?,
                                     eps: Float = 1e-6) -> GatedResidualResult {
        let hcHidden = hcCount * hidden
        precondition(hyperInput.count == hcHidden)
        precondition(hcNormWeight.count == hcHidden)
        precondition(downWeight.count == lowrank * hcHidden)
        precondition(upWeight.count == hcHidden * lowrank)
        let normed = rmsNorm(hyperInput, weight: hcNormWeight,
                             groupSize: hidden, eps: eps)
        // down: [hcHidden] -> [lowrank], divided by hcCount, SiLU.
        var low = [Float](repeating: 0, count: lowrank)
        for r in 0..<lowrank {
            var acc: Float = 0
            for c in 0..<hcHidden { acc += downWeight[r * hcHidden + c] * normed[c] }
            low[r] = silu(acc / Float(hcCount))
        }
        // up: [lowrank] -> [hcHidden], sigmoid.
        var mix = [Float](repeating: 0, count: hcHidden)
        for r in 0..<hcHidden {
            var acc: Float = 0
            for c in 0..<lowrank { acc += upWeight[r * lowrank + c] * low[c] }
            mix[r] = sigmoid(acc)
        }
        // mean over hc branches of mix * normed.
        var mixed = [Float](repeating: 0, count: hidden)
        for h in 0..<hcCount {
            for d in 0..<hidden {
                mixed[d] += mix[h * hidden + d] * normed[h * hidden + d]
            }
        }
        for d in 0..<hidden { mixed[d] /= Float(hcCount) }
        guard let bij = blockInjectWeight else {
            return GatedResidualResult(mixed: mixed, hyperNormed: normed,
                                       injection: nil)
        }
        precondition(bij.count == hcCount * hcHidden)
        var injection = [Float](repeating: 0, count: hcCount)
        for h in 0..<hcCount {
            var acc: Float = 0
            for c in 0..<hcHidden { acc += bij[h * hcHidden + c] * normed[c] }
            injection[h] = 2.0 * sigmoid(acc / Float(hcCount))
        }
        return GatedResidualResult(mixed: mixed, hyperNormed: normed,
                                   injection: injection)
    }

    /// The inject add that closes each sub-layer:
    /// `hyper + flatten(subOut[h] * injection[h])`.
    public static func injectAdd(hyperInput: [Float],
                                 subOutput: [Float],
                                 injection: [Float],
                                 hcCount: Int,
                                 hidden: Int) -> [Float] {
        precondition(hyperInput.count == hcCount * hidden)
        precondition(subOutput.count == hidden)
        precondition(injection.count == hcCount)
        var out = hyperInput
        for h in 0..<hcCount {
            for d in 0..<hidden {
                out[h * hidden + d] += subOutput[d] * injection[h]
            }
        }
        return out
    }

    // MARK: - MoE router (top-k, norm_topk_prob = true)

    public struct RouterTopK {
        public let indices: [Int]
        public let weights: [Float]
    }

    /// Exact port of `Qwen4ExpTextTopKRouter` with `norm_topk_prob = true`:
    /// full softmax, deterministic top-k (higher score wins; ties break to
    /// the smaller expert index, matching the Metal insertion sort), then
    /// renormalize over the selected top-k and multiply by per-expert scale.
    public static func routerTopK(logits: [Float], topK: Int,
                                  perExpertScale: [Float]? = nil) -> RouterTopK {
        precondition(topK >= 1 && topK <= logits.count)
        let maxLogit = logits.max() ?? 0
        let exps = logits.map { exp($0 - maxLogit) }
        let denom = exps.reduce(0, +)
        let probs = exps.map { $0 / denom }
        let order = probs.indices.sorted {
            if probs[$0] != probs[$1] { return probs[$0] > probs[$1] }
            return $0 < $1
        }
        let top = Array(order.prefix(topK))
        let topSum = top.reduce(Float(0)) { $0 + probs[$1] }
        var weights = top.map { probs[$0] / topSum }
        if let s = perExpertScale {
            precondition(s.count == logits.count)
            weights = zip(top, weights).map { $1 * s[$0] }
        }
        return RouterTopK(indices: top, weights: weights)
    }

    // MARK: - QSA indexer (sparse mask)

    /// One query's block scores, ported from `Qwen4ExpTextQSAIndexer`:
    /// `relu(q_h . k_b)` summed over the `nHeads` index query heads,
    /// divided by `sqrt(headDim)`. Inputs are post-layernorm/post-RoPE.
    public static func qsaBlockScores(queryHeads: [[Float]],
                                      blockKeys: [[Float]],
                                      headDim: Int) -> [Float] {
        precondition(!queryHeads.isEmpty && !blockKeys.isEmpty)
        var scores = [Float](repeating: 0, count: blockKeys.count)
        for (b, key) in blockKeys.enumerated() {
            precondition(key.count == headDim)
            var s: Float = 0
            for q in queryHeads {
                precondition(q.count == headDim)
                var dot: Float = 0
                for d in 0..<headDim { dot += q[d] * key[d] }
                s += max(0, dot)
            }
            scores[b] = s / sqrt(Float(headDim))
        }
        return scores
    }

    /// Token selection for one query, ported from the reference loop:
    /// complete `compress`-token blocks are scored and top-`blockTopK`
    /// blocks expand back to tokens; the incomplete tail always passes
    /// through. Short contexts (fewer than `compress` visible tokens) return
    /// the dense visible set, so short-context inference matches exact
    /// attention by construction.
    ///
    /// - `visible`: sorted visible KV positions for this query (causal).
    /// - `blockScores`: one score per complete block, in block order.
    /// - `blockTopK`: `budget / compress`.
    public static func qsaSelectTokens(visible: [Int],
                                       blockScores: [Float],
                                       compress: Int,
                                       blockTopK: Int) -> [Int] {
        precondition(compress >= 1 && blockTopK >= 1)
        let numComplete = visible.count / compress
        precondition(blockScores.count == numComplete)
        let k = min(blockTopK, numComplete)
        // Deterministic top-k over blocks (ties → smaller block index).
        let blockOrder = blockScores.indices.sorted {
            if blockScores[$0] != blockScores[$1] {
                return blockScores[$0] > blockScores[$1]
            }
            return $0 < $1
        }
        let chosen = blockOrder.prefix(k).sorted()
        var out: [Int] = []
        out.reserveCapacity(k * compress + compress)
        for b in chosen {
            for t in 0..<compress {
                out.append(visible[b * compress + t])
            }
        }
        out.append(contentsOf: visible[(numComplete * compress)...])
        return out
    }

    /// Scatter selected indices into a boolean allow-mask of length `kvLen`,
    /// ported from the reference's `scatter(..., True)[..., :kvLen]`.
    public static func qsaAllowMask(selected: [Int], kvLen: Int) -> [Bool] {
        precondition(kvLen >= 0)
        var mask = [Bool](repeating: false, count: kvLen)
        for idx in selected where idx >= 0 && idx < kvLen {
            mask[idx] = true
        }
        return mask
    }

    // MARK: - RoPE (NeoX sub-dim, CPU mirror)

    /// CPU mirror of `rope_neox_subdim` / `apply_neox_pair` in rope.metal:
    /// pairs (i, rotaryDim/2 + i) for i < rotaryDim/2 rotate by
    /// `position * theta^(-2i/rotaryDim)`; elements past rotaryDim pass
    /// through. Used for indexer block-key finalization on CPU.
    public static func applyRopeNeoxCPU(_ x: [Float], position: Int,
                                        headDim: Int, rotaryDim: Int,
                                        theta: Double) -> [Float] {
        precondition(x.count == headDim)
        precondition(rotaryDim % 2 == 0 && rotaryDim <= headDim)
        var out = x
        let half = rotaryDim / 2
        // Float32 throughout to match the GPU kernel's precision.
        let thetaF = Float(theta)
        for i in 0..<half {
            let exponent = -Float(2 * i) / Float(rotaryDim)
            let angle = Float(position) * pow(thetaF, exponent)
            let c = cos(angle), sn = sin(angle)
            let x0 = x[i], x1 = x[half + i]
            out[i] = x0 * c - x1 * sn
            out[half + i] = x0 * sn + x1 * c
        }
        return out
    }

    // MARK: - PLE hashing (splitmix64 + multipliers + primes)

    static let mask64: UInt64 = 0xFFFF_FFFF_FFFF_FFFF
    static let splitmixGamma: UInt64 = 0x9E3779B97F4A7C15
    static let splitmixM1: UInt64 = 0xBF58476D1CE4E5B9
    static let splitmixM2: UInt64 = 0x94D049BB133111EB
    static let prime1: Int = 10_007

    public static func splitmix64(_ value: UInt64) -> UInt64 {
        var v = value &+ splitmixGamma
        v = ((v ^ (v >> 30)) &* splitmixM1) & mask64
        v = ((v ^ (v >> 27)) &* splitmixM2) & mask64
        return (v ^ (v >> 31)) & mask64
    }

    /// Port of `_build_layer_multipliers`.
    public static func buildLayerMultipliers(unigramVocab: Int,
                                            ngramSize: Int,
                                            pleLayerIndex: Int,
                                            seed: Int) -> [UInt64] {
        let maxLong: UInt64 = (UInt64(1) << 63) - 1
        let multiplierMax = maxLong / UInt64(max(1, unigramVocab))
        let halfBound = max(UInt64(1), multiplierMax / 2)
        let baseSeed = UInt64(bitPattern: Int64(seed + prime1 * pleLayerIndex))
        return (0..<ngramSize).map { index in
            let value = (baseSeed &+ splitmixGamma &* UInt64(index + 1)) & mask64
            return 2 &* (splitmix64(value) % halfBound) &+ 1
        }
    }

    public static func isPrime(_ value: Int) -> Bool {
        if value < 2 { return false }
        if value % 2 == 0 { return value == 2 }
        var d = 3
        while d * d <= value {
            if value % d == 0 { return false }
            d += 2
        }
        return true
    }

    public static func nthPrimeAfter(_ start: Int, count: Int) -> Int {
        var prime = start
        for _ in 0..<count {
            prime += 1
            while !isPrime(prime) { prime += 1 }
        }
        return prime
    }

    public struct NgramGeometry {
        public let headVocabSizes: [Int]
        public let headOffsets: [Int]
        public let totalVocab: Int
        public let paddedVocab: Int
        public let rowsPerShard: Int
    }

    /// Port of `Qwen4ExpTextNGramEmbedding.__init__` geometry.
    public static func ngramGeometry(ngramHeads: Int,
                                     pleLayerIndex: Int,
                                     vocabBase: Int,
                                     divisibleBy: Int,
                                     splitParts: Int) -> NgramGeometry {
        var sizes: [Int] = []
        var offsets: [Int] = []
        var total = 0
        for head in 0..<ngramHeads {
            let globalHead = pleLayerIndex * ngramHeads + head
            let size = nthPrimeAfter(vocabBase - 1, count: globalHead + 1)
            sizes.append(size)
            offsets.append(total)
            total += size
        }
        let padded = ((total + divisibleBy - 1) / divisibleBy) * divisibleBy
        precondition(padded % splitParts == 0,
                     "padded ngram vocab \(padded) must split evenly over \(splitParts) parts")
        return NgramGeometry(headVocabSizes: sizes, headOffsets: offsets,
                             totalVocab: total, paddedVocab: padded,
                             rowsPerShard: padded / splitParts)
    }

    /// Shift-right with EOS flooding across segment restarts, ported from
    /// `_shift_right_ignore_eos`. `tokens` is the full history
    /// `[previousContext..., inputIds...]`; returns the shifted sequence of
    /// the same length.
    public static func shiftRightIgnoreEos(tokens: [Int], shift: Int,
                                           eosId: Int) -> [Int] {
        if shift == 0 { return tokens }
        let n = tokens.count
        var out = [Int](repeating: eosId, count: n)
        // Segment starts: position after the most recent EOS at or before i.
        var segStart = [Int](repeating: 0, count: n)
        var lastEos = -1
        for i in 0..<n {
            segStart[i] = lastEos + 1
            if tokens[i] == eosId { lastEos = i }
        }
        for i in 0..<n {
            let src = i - shift
            if src >= 0 && (i - segStart[i]) >= shift {
                out[i] = tokens[src]
            } else {
                out[i] = eosId
            }
        }
        return out
    }

    /// N-gram row ids for the current chunk, ported from
    /// `Qwen4ExpTextNGramEmbedding.forward` (embedding lookup excluded).
    /// Returns `[seqLen, ngramHeads]` global row ids.
    public static func ngramRowIds(history: [Int],
                                   seqLen: Int,
                                   ngramSize: Int,
                                   headsPerNgram: Int,
                                   multipliers: [UInt64],
                                   headVocabSizes: [Int],
                                   headOffsets: [Int],
                                   eosId: Int) -> [[Int]] {
        precondition(multipliers.count == ngramSize)
        let ngramHeads = (ngramSize - 1) * headsPerNgram
        precondition(headVocabSizes.count == ngramHeads)
        precondition(headOffsets.count == ngramHeads)
        let shifted = (0..<ngramSize).map {
            shiftRightIgnoreEos(tokens: history, shift: $0, eosId: eosId)
        }
        let start = history.count - seqLen
        precondition(start >= 0)
        var rows: [[Int]] = []
        rows.reserveCapacity(seqLen)
        for i in start..<history.count {
            var row: [Int] = []
            row.reserveCapacity(ngramHeads)
            for ngram in 2...ngramSize {
                var mixed = UInt64(bitPattern: Int64(shifted[0][i])) &* multipliers[0]
                for pos in 1..<ngram {
                    mixed ^= UInt64(bitPattern: Int64(shifted[pos][i])) &* multipliers[pos]
                }
                let headStart = (ngram - 2) * headsPerNgram
                for h in 0..<headsPerNgram {
                    let v = headVocabSizes[headStart + h]
                    let id = Int(mixed % UInt64(v)) + headOffsets[headStart + h]
                    row.append(id)
                }
            }
            rows.append(row)
        }
        return rows
    }

    /// Contiguous dim0 shard for a global n-gram row id (checkpoint shards
    /// are contiguous slices of the padded table).
    public static func shardForRow(_ row: Int, rowsPerShard: Int) -> Int {
        precondition(row >= 0 && rowsPerShard > 0)
        return row / rowsPerShard
    }

    // MARK: - PLE gate (lexical injection)

    /// Port of the `Qwen4ExpTextPLELayer.forward` gate:
    /// `gate = sign-preserving sqrt(|dot|)` then `sigmoid(gate) * value`.
    /// Inputs are post-projection/post-norm vectors; `key`/`query` are
    /// `[hc, hidden]` unflattened, `value` is `[hidden]`.
    public static func pleGate(keyStreams: [[Float]],
                               queryStreams: [[Float]],
                               value: [Float],
                               hidden: Int) -> (gated: [[Float]], gate: [Float]) {
        precondition(!keyStreams.isEmpty)
        precondition(keyStreams.count == queryStreams.count)
        precondition(value.count == hidden)
        let hc = keyStreams.count
        var gate = [Float](repeating: 0, count: hc)
        for h in 0..<hc {
            precondition(keyStreams[h].count == hidden)
            precondition(queryStreams[h].count == hidden)
            var dot: Float = 0
            for d in 0..<hidden { dot += keyStreams[h][d] * queryStreams[h][d] }
            dot /= sqrt(Float(hidden))
            let mag = sqrt(max(1e-6, abs(dot)))
            gate[h] = (dot >= 0 ? 1.0 : -1.0) * mag
        }
        let gated = gate.map { g in value.map { sigmoid(g) * $0 } }
        return (gated, gate)
    }
}
