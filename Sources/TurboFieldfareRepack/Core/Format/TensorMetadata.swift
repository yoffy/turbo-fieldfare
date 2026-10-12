import Foundation

/// One physical tensor in a safetensors shard. Coordinates are absolute file
/// offsets so the writer can map and copy a single tensor without re-parsing
/// the shard header.
struct SourceTensor: Sendable, Hashable {
    enum Dtype: UInt8, Sendable, Hashable {
        case u32  = 0
        case bf16 = 1
        case fp16 = 2
        case fp32 = 3

        var elementBytes: Int {
            switch self { case .u32: 4; case .bf16: 2; case .fp16: 2; case .fp32: 4 }
        }
    }

    let name: String
    let shardPath: String
    let dtype: Dtype
    let shape: [UInt64]
    let absoluteOffset: UInt64
    let sizeBytes: UInt64
}

/// Bit-width override resolved from `config.json -> quantization`.
/// `groupSize` records the row-group the scales/biases were quantized with.
/// Qwen 3.8 mixes a 32-wide base with 64-wide gate overrides; older
/// checkpoints are uniformly 64-wide.
struct QuantSpec: Sendable, Hashable {
    let bits: Int
    let groupSize: Int

    init(bits: Int, groupSize: Int = 64) {
        self.bits = bits
        self.groupSize = groupSize
    }
}
