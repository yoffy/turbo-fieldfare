import Darwin
import Foundation

public final class LocalByteProvider: SourceByteProvider {
    private let inputDir: String
    private let shardPathFor: (String) -> String
    private let writeTileBytes: Int

    public init(inputDir: String,
                shardPathFor: @escaping (String) -> String,
                writeTileBytes: Int = WriterCore.tileBytes) {
        self.inputDir = inputDir
        self.shardPathFor = shardPathFor
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
        let scratch = UnsafeMutableRawBufferPointer.allocate(
            byteCount: writeTileBytes,
            alignment: 16_384)
        defer { scratch.deallocate() }
        audit.largestScratchBytes = max(audit.largestScratchBytes, scratch.count)

        var outputFDs: [String: Int32] = [:]
        defer { outputFDs.values.forEach { close($0) } }

        var downloaded: UInt64 = 0

        for copy in copies where !completedRangeIDs.contains(copy.id) {
            try Task.checkCancellation()

            let shardFile = copy.shardID
            let shardPath = shardPathFor(shardFile)
            var sourceFD: Int32?
            do {
                sourceFD = try Posix.openReadNoFollow(shardPath)
             } catch {
                var stat = stat()
                var fileType: String?
                if lstat(shardPath, &stat) == 0 {
                    switch stat.st_mode & S_IFMT {
                    case S_IFREG:   fileType = "regular"
                    case S_IFLNK:   fileType = "symlink"
                    case S_IFDIR:   fileType = "directory"
                    case S_IFCHR:   fileType = "character device"
                    case S_IFBLK:   fileType = "block device"
                    case S_IFIFO:   fileType = "FIFO/pipe"
                    case S_IFSOCK:  fileType = "socket"
                    default:        fileType = "unknown"
                    }
                }
                throw RepackError.localInputReadFailed(path: shardPath, errno: 0, fileType: fileType)
             }
            defer { if let fd = sourceFD { close(fd) } }

            var touched = Set<String>()

            do {
                for destination in copy.destinations {
                    let destinationFD: Int32
                    if let existing = outputFDs[destination.destinationPath] {
                        destinationFD = existing
                      } else {
                        destinationFD = try Posix.openExistingRW(destination.destinationPath)
                        outputFDs[destination.destinationPath] = destinationFD
                      }
                    touched.insert(destination.destinationPath)
                    try copyBytes(
                        sourceFD: sourceFD!,
                        sourcePath: shardPath,
                        sourceOffset: destination.sourceOffset - copy.sourceOffset,
                        destinationFD: destinationFD,
                        destinationPath: destination.destinationPath,
                        destinationOffset: destination.destinationOffset,
                        size: destination.size,
                        scratch: scratch,
                        audit: audit)
                  }
                close(sourceFD!)
              } catch {
                close(sourceFD!)
                throw error
              }

            downloaded += copy.size
            progress(downloaded)

            for path in touched {
                if let descriptor = outputFDs[path] {
                    try Posix.fsync(descriptor, path: path)
                  }
              }

            let digest = try computeDigest(copy: copy, partialDirectory: partialDirectory, scratch: scratch)
            try commit(RemoteCompletedRange(
                id: copy.id,
                destinationDigest: digest,
                sourceBytes: copy.size,
                destinationBytes: copy.destinations.reduce(0) { $0 + $1.size }))
            progress(downloaded)
            try Task.checkCancellation()
          }
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

    private func computeDigest(
        copy: CoalescedRangeCopy,
        partialDirectory: String,
        scratch suppliedScratch: UnsafeMutableRawBufferPointer? = nil
     ) throws -> String {
        let scratch = suppliedScratch ?? UnsafeMutableRawBufferPointer.allocate(
            byteCount: WriterCore.tileBytes,
            alignment: 16_384)
        defer {
            if suppliedScratch == nil { scratch.deallocate() }
          }

        var digest = DestinationDigestLocal(copy: copy)
        for destination in copy.destinations {
            digest.append(try RangeCopyPlanner.normalizedRelativePath(
                destination.destinationPath,
                root: partialDirectory))
            digest.append(destination.destinationOffset)
            digest.append(destination.sourceOffset - copy.sourceOffset)
            digest.append(destination.size)

            let descriptor = try Posix.openReadNoFollow(destination.destinationPath)
            defer { close(descriptor) }
            var remaining = destination.size
            var offset = destination.destinationOffset
            while remaining > 0 {
                let count = min(Int(remaining), scratch.count)
                try Posix.preadAll(
                    fd: descriptor,
                    path: destination.destinationPath,
                    buf: scratch.baseAddress!,
                    count: count,
                    offset: offset)
                digest.append(UnsafeRawBufferPointer(
                    start: scratch.baseAddress,
                    count: count))
                remaining -= UInt64(count)
                offset += UInt64(count)
              }
          }
        return digest.finalize()
       }
}

// MARK: - Local digest helper (mirrors Remote DestinationDigest)

private struct DestinationDigestLocal {
    private var stream = Sha256Stream()

    init(copy: CoalescedRangeCopy) {
        append("TurboFieldfare.RemoteRangeDestination.v1")
        append(copy.id)
        append(UInt64(copy.destinations.count))
      }

    mutating func append(_ value: UInt64) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { stream.update($0) }
      }

    mutating func append(_ value: String) {
        let data = Data(value.utf8)
        append(UInt64(data.count))
        data.withUnsafeBytes { stream.update($0) }
      }

    mutating func append(_ bytes: UnsafeRawBufferPointer) {
        stream.update(bytes)
      }

    func finalize() -> String {
        stream.finalizeHexString()
      }
}
