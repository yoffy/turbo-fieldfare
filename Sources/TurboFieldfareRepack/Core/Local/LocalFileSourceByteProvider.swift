import Darwin
import Foundation

/// Copies planned byte ranges from shards already on local disk. The
/// destination layout, coalescing, and digest scheme are identical to the
/// HTTP range provider so local and remote installs produce byte-identical
/// `.gturbo` outputs for the same source snapshot.
public final class LocalFileSourceByteProvider: SourceByteProvider {
    private let sourceDir: String
    private let writeTileBytes: Int

    public init(sourceDir: String,
                writeTileBytes: Int = WriterCore.tileBytes) {
        self.sourceDir = sourceDir
        self.writeTileBytes = writeTileBytes
    }

    public func copyBatch(
        _ copies: [CoalescedRangeCopy],
        completedRangeIDs: Set<String>,
        partialDirectory: String,
        temporaryPath: String,
        audit: RepackAudit,
        progress: @escaping @Sendable (UInt64) -> Void,
        commit: (RemoteCompletedRange) throws -> Void
    ) async throws {
        _ = temporaryPath
        let scratch = UnsafeMutableRawBufferPointer.allocate(
            byteCount: writeTileBytes,
            alignment: 16_384)
        defer { scratch.deallocate() }
        audit.largestScratchBytes = max(audit.largestScratchBytes, scratch.count)

        var outputFDs: [String: Int32] = [:]
        defer { outputFDs.values.forEach { close($0) } }
        var sourceFDs: [String: Int32] = [:]
        defer { sourceFDs.values.forEach { close($0) } }
        var copied: UInt64 = 0

        func sourceFD(for shardID: String) throws -> Int32 {
            if let existing = sourceFDs[shardID] { return existing }
            let fullPath = try shardPath(for: shardID)
            let fd = try Posix.openReadNoFollow(fullPath)
            sourceFDs[shardID] = fd
            return fd
        }

        for copy in copies where !completedRangeIDs.contains(copy.id) {
            try Task.checkCancellation()
            let fd = try sourceFD(for: copy.shardID)
            let shardPath = try shardPath(for: copy.shardID)
            for destination in copy.destinations {
                let destinationFD: Int32
                if let existing = outputFDs[destination.destinationPath] {
                    destinationFD = existing
                } else {
                    destinationFD = try Posix.openExistingRW(destination.destinationPath)
                    outputFDs[destination.destinationPath] = destinationFD
                }
                try copyBytes(
                    sourceFD: fd,
                    sourcePath: shardPath,
                    sourceOffset: destination.sourceOffset,
                    destinationFD: destinationFD,
                    destinationPath: destination.destinationPath,
                    destinationOffset: destination.destinationOffset,
                    size: destination.size,
                    scratch: scratch,
                    audit: audit)
            }
            copied += copy.size
            progress(copied)

            try Task.checkCancellation()
            for destination in copy.destinations {
                if let descriptor = outputFDs[destination.destinationPath] {
                    try Posix.fsync(descriptor, path: destination.destinationPath)
                }
            }
            let digest = try HTTPRangeSourceByteProvider.destinationDigest(
                copy,
                partialDirectory: partialDirectory,
                scratch: scratch)
            try commit(RemoteCompletedRange(
                id: copy.id,
                destinationDigest: digest,
                sourceBytes: copy.size,
                destinationBytes: copy.destinations.reduce(0) { $0 + $1.size }))
            try Task.checkCancellation()
        }
    }

    private func shardPath(for shardID: String) throws -> String {
        guard !shardID.isEmpty,
              !shardID.hasPrefix("/"),
              !shardID.contains("..") else {
            throw RepackError.configurationInvalid(detail: "invalid shard id \(shardID)")
        }
        return (sourceDir as NSString).appendingPathComponent(shardID)
    }

    private func copyBytes(
        sourceFD: Int32,
        sourcePath: String,
        sourceOffset: UInt64,
        destinationFD: Int32,
        destinationPath: String,
        destinationOffset: UInt64,
        size: UInt64,
        scratch: UnsafeMutableRawBufferPointer,
        audit: RepackAudit
    ) throws {
        var remaining = size
        var source = sourceOffset
        var destination = destinationOffset
        while remaining > 0 {
            try Task.checkCancellation()
            let count = min(Int(remaining), scratch.count)
            try Posix.preadAll(
                fd: sourceFD,
                path: sourcePath,
                buf: scratch.baseAddress!,
                count: count,
                offset: source)
            try Posix.pwriteAll(
                fd: destinationFD,
                path: destinationPath,
                buf: scratch.baseAddress!,
                count: count,
                offset: destination)
            audit.recordTile(bytes: count)
            audit.recordRead(bytes: count)
            audit.recordWrite(bytes: count)
            remaining -= UInt64(count)
            source += UInt64(count)
            destination += UInt64(count)
        }
    }
}
