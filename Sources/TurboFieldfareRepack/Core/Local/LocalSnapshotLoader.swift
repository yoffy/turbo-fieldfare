import Darwin
import Foundation

/// A safetensors snapshot already present on local disk (for example a
/// Hugging Face checkpoint cloned with `git lfs` or downloaded by another
/// tool). Mirrors `RemoteSnapshot` but carries the source directory instead
/// of remote file metadata so the planner and copy stages can stay shared.
struct LocalSnapshot: Sendable {
    let metadata: IndexLoader.SourceMetadata
    let arch: ArchInfo
    let shardHeaders: [Safetensors.Header]
    let sourceDir: String
}

/// Loads `model.safetensors.index.json` + `config.json` + per-shard headers
/// from a local directory. The index SHA-256 fingerprint contract is the same
/// as the remote installer: when `requireKnownSource` is set, unknown
/// snapshots are rejected before any output is written.
enum LocalSnapshotLoader {
    static func load(sourceDir: String,
                     requireKnownSource: Bool) throws -> LocalSnapshot {
        let standardized = URL(fileURLWithPath: sourceDir).standardizedFileURL.path
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: standardized, isDirectory: &isDir),
              isDir.boolValue else {
            throw RepackError.configurationInvalid(detail: "source directory not found: \(sourceDir)")
        }

        let metadata = try IndexLoader.load(snapshotDir: standardized)
        if requireKnownSource,
           SourceFingerprint.modelID(forIndexSha256: metadata.indexSha256Hex) == nil {
            throw RepackError.sourceFingerprintRejected(path: metadata.indexPath,
                                                        sha256: metadata.indexSha256Hex)
        }
        let arch = try ArchInfo.load(configPath: metadata.configPath)

        var headers: [Safetensors.Header] = []
        headers.reserveCapacity(metadata.shardFilenames.count)
        for shard in metadata.shardFilenames {
            try validateShardFilename(shard)
            let fullPath = (standardized as NSString).appendingPathComponent(shard)
            guard FileManager.default.fileExists(atPath: fullPath) else {
                throw RepackError.configurationInvalid(detail: "missing shard file: \(fullPath)")
            }
            let fileSize = try localFileSize(path: fullPath)
            let headerSize = try readHeaderSize(path: fullPath)
            if headerSize > Safetensors.maxHeaderBytes || headerSize > fileSize - 8 {
                throw RepackError.safetensorsHeaderTooLarge(path: shard, size: headerSize)
            }
            let headerData = try readBytes(path: fullPath, offset: 8, count: Int(headerSize))
            headers.append(try Safetensors.parseHeaderBytes(path: shard,
                                                            fileSize: fileSize,
                                                            headerBytes: headerData))
        }

        return LocalSnapshot(metadata: metadata,
                             arch: arch,
                             shardHeaders: headers,
                             sourceDir: standardized)
    }

    private static func validateShardFilename(_ filename: String) throws {
        guard !filename.isEmpty,
              !filename.hasPrefix("/"),
              !filename.contains(".."),
              !filename.contains("?"),
              !filename.contains("#") else {
            throw RepackError.configurationInvalid(detail: "invalid shard filename \(filename)")
        }
    }

    private static func localFileSize(path: String) throws -> UInt64 {
        let fd = try Posix.openReadNoFollow(path)
        defer { close(fd) }
        return try Posix.fileSize(fd: fd, path: path)
    }

    private static func readHeaderSize(path: String) throws -> UInt64 {
        let fd = try Posix.openReadNoFollow(path)
        defer { close(fd) }
        var prefix = Data(count: 8)
        try prefix.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress else { return }
            try Posix.preadAll(fd: fd, path: path, buf: base, count: 8, offset: 0)
        }
        return prefix.withUnsafeBytes { raw -> UInt64 in
            var value: UInt64 = 0
            for i in 0..<8 {
                value |= UInt64(raw[i]) << UInt64(i * 8)
            }
            return value
        }
    }

    private static func readBytes(path: String, offset: UInt64, count: Int) throws -> Data {
        let fd = try Posix.openReadNoFollow(path)
        defer { close(fd) }
        var data = Data(count: count)
        try data.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress else { return }
            try Posix.preadAll(fd: fd, path: path, buf: base, count: count, offset: offset)
        }
        return data
    }
}
