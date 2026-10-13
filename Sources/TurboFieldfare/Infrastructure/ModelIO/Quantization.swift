import Foundation

public enum Quantization {

    public static let groupSize: Int = 64

    /// Quantized group widths the runtime can decode. Qwen 3.8 mixes a
    /// 32-wide 4-bit base with 64-wide 8-bit gate overrides; older families
    /// are uniformly 64-wide. Manifest slots record the per-slot width and
    /// `Model.load` rejects anything outside this set.
    public static let supportedGroupSizes: [Int] = [32, 64]

    @inline(__always)
    public static func isSupportedGroupSize(_ g: Int) -> Bool {
        g == 32 || g == 64
    }

    @inline(__always)
    public static func groupsPerRow(n: Int, groupSize g: Int) -> Int {
        precondition(isSupportedGroupSize(g), "unsupported groupSize \(g)")
        precondition(n % g == 0, "row length \(n) is not a multiple of \(g)")
        return n / g
    }

    // MARK: - BF16 helpers
    //
    // BF16 = top 16 bits of FP32 (8-bit exponent, 7-bit mantissa). Stored on
    // disk and on the GPU as the native `bfloat` type; in Swift we carry the
    // raw bits as `UInt16` and convert via this pair. Round-half-to-even on
    // encode matches the IEEE-754 default. No NaN/Inf special-case: fixtures
    // are bounded and finite, and the .gturbo importer copies BF16 bytes
    // through unchanged so we never call the encoder on weight data.

    @inline(__always)
    public static func bf16Bits(_ x: Float) -> UInt16 {
        let bits = x.bitPattern
        let lsb  = (bits >> 16) & 1
        let roundingBias: UInt32 = 0x7FFF &+ lsb
        let rounded = (bits &+ roundingBias) >> 16
        return UInt16(truncatingIfNeeded: rounded)
    }

    @inline(__always)
    public static func bf16ToFloat(_ bits: UInt16) -> Float {
        Float(bitPattern: UInt32(bits) << 16)
    }

    // MARK: - INT4 affine

    /// MLX `affine` 4-bit row. Packed unsigned nibbles, BF16 scale + bias per
    /// group of 64. `scales` and `biases` carry BF16 bit patterns as `UInt16`
    /// so the same buffer can be uploaded to a Metal `device const bfloat*`.
    public struct Int4AffineRow {
        public let packed: [UInt8]   // N / 2 bytes; low nibble = even index, high = odd
        public let scales: [UInt16]  // N / 64 BF16 bits
        public let biases: [UInt16]  // N / 64 BF16 bits

        public init(packed: [UInt8], scales: [UInt16], biases: [UInt16]) {
            self.packed = packed
            self.scales = scales
            self.biases = biases
        }
    }

    /// Affine 4-bit quantize: `q ∈ [0..15]`, `w ≈ q * scale + bias`.
    /// Scale and bias are computed from per-group min/max, then rounded to BF16.
    /// Test-fixture only — the runtime importer never calls this.
    /// `groupSize` selects the row-group the scales/biases quantize with
    /// (64 legacy, 32 for the Qwen 3.8 base). Defaults to 64 so existing
    /// callers keep byte-identical output.
    public static func quantizeInt4Affine(_ row: [Float], groupSize g: Int = 64) -> Int4AffineRow {
        precondition(isSupportedGroupSize(g), "unsupported groupSize \(g)")
        precondition(row.count % g == 0,
                     "row length \(row.count) is not a multiple of \(g)")

        let nGroups = row.count / g
        var packed = [UInt8](repeating: 0, count: row.count / 2)
        var scales = [UInt16](repeating: 0, count: nGroups)
        var biases = [UInt16](repeating: 0, count: nGroups)

        for gIdx in 0..<nGroups {
            var wmin: Float =  .infinity
            var wmax: Float = -.infinity
            for k in 0..<g {
                let w = row[gIdx * g + k]
                if w < wmin { wmin = w }
                if w > wmax { wmax = w }
            }
            // Constant group: scale=1, bias=value preserves exact reconstruction.
            let scaleF: Float
            let biasF:  Float
            if wmax == wmin {
                scaleF = 1
                biasF  = wmin
            } else {
                scaleF = (wmax - wmin) / 15.0
                biasF  = wmin
            }
            // Round through BF16 first, then quantize against the rounded
            // values so the runtime decode (which reads BF16) reproduces the
            // same q the fixture stored.
            let sBits = bf16Bits(scaleF)
            let bBits = bf16Bits(biasF)
            scales[gIdx] = sBits
            biases[gIdx] = bBits
            let scale = bf16ToFloat(sBits)
            let bias  = bf16ToFloat(bBits)
            let invScale = scale == 0 ? Float(0) : 1.0 / scale

            for k in 0..<g {
                let w = row[gIdx * g + k]
                var q = Int(((w - bias) * invScale).rounded())
                q = max(0, min(15, q))
                let nibble = UInt8(q) & 0x0F
                let byteIdx = gIdx * (g / 2) + (k / 2)
                if (k & 1) == 0 {
                    packed[byteIdx] = (packed[byteIdx] & 0xF0) | nibble
                } else {
                    packed[byteIdx] = (packed[byteIdx] & 0x0F) | (nibble << 4)
                }
            }
        }
        return Int4AffineRow(packed: packed, scales: scales, biases: biases)
    }

    public static func dequantizeInt4Affine(_ r: Int4AffineRow, n: Int, groupSize g: Int = 64) -> [Float] {
        precondition(isSupportedGroupSize(g), "unsupported groupSize \(g)")
        precondition(n == r.packed.count * 2)
        precondition(n % g == 0, "row length \(n) is not a multiple of \(g)")
        precondition(r.scales.count == n / g && r.biases.count == n / g,
                     "scales/biases count \(r.scales.count)/\(r.biases.count) != groups \(n / g)")
        var out = [Float](repeating: 0, count: n)
        let nGroups = n / g
        for gIdx in 0..<nGroups {
            let scale = bf16ToFloat(r.scales[gIdx])
            let bias  = bf16ToFloat(r.biases[gIdx])
            for k in 0..<g {
                let byteIdx = gIdx * (g / 2) + (k / 2)
                let b = r.packed[byteIdx]
                let nibble: Int = (k & 1) == 0 ? Int(b & 0x0F) : Int(b >> 4)
                out[gIdx * g + k] = Float(nibble) * scale + bias
            }
        }
        return out
    }

    // MARK: - INT8 affine

    public struct Int8AffineRow {
        public let packed: [UInt8]   // N unsigned bytes
        public let scales: [UInt16]  // N / 64 BF16 bits
        public let biases: [UInt16]  // N / 64 BF16 bits

        public init(packed: [UInt8], scales: [UInt16], biases: [UInt16]) {
            self.packed = packed
            self.scales = scales
            self.biases = biases
        }
    }

    public static func quantizeInt8Affine(_ row: [Float], groupSize g: Int = 64) -> Int8AffineRow {
        precondition(isSupportedGroupSize(g), "unsupported groupSize \(g)")
        precondition(row.count % g == 0,
                     "row length \(row.count) is not a multiple of \(g)")

        let nGroups = row.count / g
        var packed = [UInt8](repeating: 0, count: row.count)
        var scales = [UInt16](repeating: 0, count: nGroups)
        var biases = [UInt16](repeating: 0, count: nGroups)

        for gIdx in 0..<nGroups {
            var wmin: Float =  .infinity
            var wmax: Float = -.infinity
            for k in 0..<g {
                let w = row[gIdx * g + k]
                if w < wmin { wmin = w }
                if w > wmax { wmax = w }
            }
            let scaleF: Float
            let biasF:  Float
            if wmax == wmin {
                scaleF = 1
                biasF  = wmin
            } else {
                scaleF = (wmax - wmin) / 255.0
                biasF  = wmin
            }
            let sBits = bf16Bits(scaleF)
            let bBits = bf16Bits(biasF)
            scales[gIdx] = sBits
            biases[gIdx] = bBits
            let scale = bf16ToFloat(sBits)
            let bias  = bf16ToFloat(bBits)
            let invScale = scale == 0 ? Float(0) : 1.0 / scale

            for k in 0..<g {
                let w = row[gIdx * g + k]
                var q = Int(((w - bias) * invScale).rounded())
                q = max(0, min(255, q))
                packed[gIdx * g + k] = UInt8(q)
            }
        }
        return Int8AffineRow(packed: packed, scales: scales, biases: biases)
    }

    public static func dequantizeInt8Affine(_ r: Int8AffineRow, n: Int, groupSize g: Int = 64) -> [Float] {
        precondition(isSupportedGroupSize(g), "unsupported groupSize \(g)")
        precondition(n == r.packed.count)
        precondition(n % g == 0, "row length \(n) is not a multiple of \(g)")
        precondition(r.scales.count == n / g && r.biases.count == n / g,
                     "scales/biases count mismatch for groups \(n / g)")
        var out = [Float](repeating: 0, count: n)
        let nGroups = n / g
        for gIdx in 0..<nGroups {
            let scale = bf16ToFloat(r.scales[gIdx])
            let bias  = bf16ToFloat(r.biases[gIdx])
            for k in 0..<g {
                out[gIdx * g + k] = Float(r.packed[gIdx * g + k]) * scale + bias
            }
        }
        return out
    }
}
