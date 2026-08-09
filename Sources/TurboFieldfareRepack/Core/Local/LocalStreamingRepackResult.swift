import Foundation

public struct LocalStreamingRepackResult: Sendable {
    public let outputDir: String
    let plan: RepackPlan
    public let localBytesRead: UInt64
    public let dryRun: Bool
}
