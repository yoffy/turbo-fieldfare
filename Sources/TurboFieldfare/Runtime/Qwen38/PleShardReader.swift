import Darwin
import Foundation

/// Row-granular reader for the SSD-streamed PLE n-gram table
/// (`packed_ple/shard_*.bin` + `layout.json`).
///
/// Unlike routed experts (whole 2 MB-aligned blobs into slot caches), PLE
/// inference touches at most `ngramHeads` rows per token (one per hashed
/// head, each `pleEmbedDim / ngramHeads` wide), so the reader `pread`s only
/// those rows (weight nibbles + BF16 scales/biases, ~100 B/row at production
/// geometry) and dequantizes them on CPU. The 24 GB table never enters GPU
/// memory; per-token I/O is a few KB.
///
/// Shard selection is contiguous dim0 slicing of the padded table
/// (`Qwen38Reference.shardForRow`), matching the checkpoint's sharded layout.
public struct PleShardEntry: Sendable {
    public let shard: Int
    public let file: String
    public let weightOffset: UInt64
    public let weightSize: UInt64
    public let scalesOffset: UInt64
    public let scalesSize: UInt64
    public let biasesOffset: UInt64
    public let biasesSize: UInt64
    public let rows: Int
    public let cols: Int
    public let bits: Int
    public let groupSize: Int
}

public final class PleShardReader: @unchecked Sendable {
    public let directoryURL: URL
    public let layerIndex: Int
    public let shards: [PleShardEntry]

    private var fds: [Int: Int32] = [:]
    private let lock = NSLock()

    public init(directoryURL: URL) throws {
        self.directoryURL = directoryURL
        let layoutURL = directoryURL
            .appendingPathComponent("packed_ple")
            .appendingPathComponent("layout.json")
        guard FileManager.default.fileExists(atPath: layoutURL.path) else {
            throw ModelError.missingFile(name: "packed_ple/layout.json")
        }
        let data = try Data(contentsOf: layoutURL)
        guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let shardsArr = obj["shards"] as? [[String: Any]] else {
            throw ModelError.indexCorrupt(detail: "packed_ple/layout.json is not a shard index")
        }
        self.layerIndex = (obj["layer"] as? Int) ?? 0
        var entries: [PleShardEntry] = []
        entries.reserveCapacity(shardsArr.count)
        for item in shardsArr {
            guard let shard = item["shard"] as? Int,
                  let file = item["file"] as? String,
                  let shape = item["logicalShape"] as? [Int], shape.count >= 2,
                  let bits = item["bits"] as? Int,
                  let group = item["groupSize"] as? Int else {
                throw ModelError.indexCorrupt(detail: "packed_ple/layout.json entry malformed")
            }
            func u64(_ key: String) -> UInt64 {
                if let n = item[key] as? NSNumber { return n.uint64Value }
                if let v = item[key] as? Int { return UInt64(v) }
                return 0
            }
            let rows = shape[0], cols = shape[1]
            let entry = PleShardEntry(
                shard: shard, file: file,
                weightOffset: u64("weightOffset"), weightSize: u64("weightSize"),
                scalesOffset: u64("scalesOffset"), scalesSize: u64("scalesSize"),
                biasesOffset: u64("biasesOffset"), biasesSize: u64("biasesSize"),
                rows: rows, cols: cols, bits: bits, groupSize: group)
            // Shape/size cross-check: weight = rows*cols/2 (4-bit), aux =
            // rows*(cols/group) BF16 each.
            if bits == 4 {
                let groupsPerRow = cols / group
                let expectW = UInt64(rows * cols / 2)
                let expectAux = UInt64(rows * groupsPerRow * 2)
                guard entry.weightSize == expectW,
                      entry.scalesSize == expectAux,
                      entry.biasesSize == expectAux else {
                    throw ModelError.indexCorrupt(
                        detail: "packed_ple shard \(shard) size/shape mismatch")
                }
            }
            entries.append(entry)
        }
        self.shards = entries.sorted { $0.shard < $1.shard }
    }

    deinit {
        for fd in fds.values { close(fd) }
    }

    public var shardCount: Int { shards.count }

    /// Contiguous dim0 rows per shard (the padded table splits evenly; the
    /// reference geometry guarantees `paddedVocab % splitParts == 0`).
    public var rowsPerShard: Int {
        get throws {
            guard let first = shards.first else {
                throw ModelError.indexCorrupt(detail: "packed_ple has no shards")
            }
            for shard in shards where shard.rows != first.rows {
                throw ModelError.indexCorrupt(
                    detail: "packed_ple shard \(shard.shard) rows \(shard.rows) != \(first.rows)")
            }
            return first.rows
        }
    }

    /// Dequantized embedding row (length `cols`) for a global padded-table
    /// row id. Shard + local row derive from contiguous dim0 slicing.
    public func readRow(globalRow row: Int, rowsPerShard: Int) throws -> [Float] {
        precondition(rowsPerShard > 0)
        let shardIdx = row / rowsPerShard
        let localRow = row % rowsPerShard
        guard shardIdx >= 0 && shardIdx < shards.count else {
            throw ModelError.indexCorrupt(detail: "PLE row \(row) maps past shard \(shardIdx)")
        }
        return try readLocalRow(shard: shards[shardIdx], row: localRow)
    }

    public func readLocalRow(shard: PleShardEntry, row: Int) throws -> [Float] {
        guard row >= 0 && row < shard.rows else {
            throw ModelError.indexCorrupt(detail: "PLE shard \(shard.shard) row \(row) out of range")
        }
        guard shard.bits == 4 else {
            throw ModelError.indexCorrupt(detail: "PLE shard \(shard.shard) bits \(shard.bits) != 4")
        }
        let g = shard.groupSize
        guard g == 32 || g == 64 else {
            throw ModelError.indexCorrupt(detail: "PLE shard \(shard.shard) group \(g)")
        }
        let cols = shard.cols
        let groupsPerRow = cols / g
        let rowWeightBytes = cols / 2
        var packed = [UInt8](repeating: 0, count: rowWeightBytes)
        var scales = [UInt16](repeating: 0, count: groupsPerRow)
        var biases = [UInt16](repeating: 0, count: groupsPerRow)
        let fd = try fdForShard(shard)
        try preadInto(fd: fd, fileOffset: shard.weightOffset + UInt64(row * rowWeightBytes),
                      bytes: &packed, count: rowWeightBytes)
        try preadU16Into(fd: fd, fileOffset: shard.scalesOffset + UInt64(row * groupsPerRow * 2),
                         values: &scales, count: groupsPerRow)
        try preadU16Into(fd: fd, fileOffset: shard.biasesOffset + UInt64(row * groupsPerRow * 2),
                         values: &biases, count: groupsPerRow)
        let row = Quantization.Int4AffineRow(packed: packed, scales: scales, biases: biases)
        return Quantization.dequantizeInt4Affine(row, n: cols, groupSize: g)
    }

    // MARK: - Private I/O

    private func fdForShard(_ shard: PleShardEntry) throws -> Int32 {
        lock.lock()
        defer { lock.unlock() }
        if let fd = fds[shard.shard] { return fd }
        let url = directoryURL
            .appendingPathComponent("packed_ple")
            .appendingPathComponent(shard.file)
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else {
            throw ModelError.missingFile(name: "packed_ple/\(shard.file)")
        }
        fds[shard.shard] = fd
        return fd
    }

    private func preadInto(fd: Int32, fileOffset: UInt64,
                           bytes: inout [UInt8], count: Int) throws {
        var filled = 0
        while filled < count {
            let n = bytes.withUnsafeMutableBytes { raw -> Int in
                pread(fd, raw.baseAddress!.advanced(by: filled),
                      count - filled, off_t(fileOffset) + off_t(filled))
            }
            if n < 0 { throw ModelError.posixFailed(call: "pread", errno: errno) }
            if n == 0 {
                throw ModelError.indexCorrupt(detail: "short read on packed_ple shard")
            }
            filled += n
        }
    }

    private func preadU16Into(fd: Int32, fileOffset: UInt64,
                              values: inout [UInt16], count: Int) throws {
        var filledBytes = 0
        let totalBytes = count * 2
        while filledBytes < totalBytes {
            let n = values.withUnsafeMutableBytes { raw -> Int in
                pread(fd, raw.baseAddress!.advanced(by: filledBytes),
                      totalBytes - filledBytes, off_t(fileOffset) + off_t(filledBytes))
            }
            if n < 0 { throw ModelError.posixFailed(call: "pread", errno: errno) }
            if n == 0 {
                throw ModelError.indexCorrupt(detail: "short read on packed_ple shard")
            }
            filledBytes += n
        }
    }
}
