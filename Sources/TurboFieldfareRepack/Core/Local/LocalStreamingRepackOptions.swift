import Foundation

public struct LocalStreamingRepackOptions: Sendable {
    public let inputLocal: String
    public let outputDir: String
    public let overwrite: Bool
    public let rangeChunkBytes: Int
    public let writeTileBytes: Int
    public let minFreeReserveBytes: UInt64
    public let dryRunSpaceCheck: Bool

    public init(inputLocal: String,
                outputDir: String,
                overwrite: Bool = false,
                rangeChunkBytes: Int = RemoteChunkPolicy.defaultBytes,
                writeTileBytes: Int = WriterCore.tileBytes,
                minFreeReserveBytes: UInt64 = 1 * 1024 * 1024 * 1024,
                dryRunSpaceCheck: Bool = false) {
        self.inputLocal = inputLocal
        self.outputDir = outputDir
        self.overwrite = overwrite
        self.rangeChunkBytes = rangeChunkBytes
        self.writeTileBytes = writeTileBytes
        self.minFreeReserveBytes = minFreeReserveBytes
        self.dryRunSpaceCheck = dryRunSpaceCheck
    }
}
