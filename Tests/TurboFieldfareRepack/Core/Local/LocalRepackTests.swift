import Foundation
import Testing

@testable import TurboFieldfareRepackCore

@Suite(.serialized)
struct LocalRepackTests {
    @Test func qwenLocalRepackMatchesRemoteLayout() async throws {
        let snapshotDir = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("turbofieldfare-local-snap-\(UUID().uuidString)")
        let output = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("turbofieldfare-local-out-\(UUID().uuidString).gturbo")
        defer {
            for path in [snapshotDir, output, output + ".partial", output + ".install.lock"] {
                try? FileManager.default.removeItem(atPath: path)
            }
        }
        _ = try SyntheticSnapshot.buildQwen(at: snapshotDir)
        // Synthetic snapshots only ship config.json; the local repacker
        // requires the tokenizer sidecars, so stage minimal ones.
        try Data(#"{"tokenizer_class":"PreTrainedTokenizerFast"}"#.utf8).write(
            to: URL(fileURLWithPath: (snapshotDir as NSString).appendingPathComponent("tokenizer.json")))
        try Data(#"{"tokenizer_class":"PreTrainedTokenizerFast"}"#.utf8).write(
            to: URL(fileURLWithPath: (snapshotDir as NSString).appendingPathComponent("tokenizer_config.json")))

        let result = try await LocalStreamingRepacker(
            options: LocalStreamingRepackOptions(
                sourceDir: snapshotDir,
                outputDir: output,
                requireKnownSource: false,
                rangeChunkBytes: 4096,
                minFreeReserveBytes: 0,
                overwrite: true)
        ).run()

        #expect(result.copiedBytes > 0)
        for relativePath in [
            "model_weights.bin",
            "packed_experts/layout.json",
            "packed_experts/layer_00.bin",
            "packed_experts/layer_01.bin",
            "packed_experts/layer_02.bin",
            "packed_experts/layer_03.bin",
            "manifest.json",
            "tokenizer/config.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
        ] {
            let path = (output as NSString).appendingPathComponent(relativePath)
            #expect(FileManager.default.fileExists(atPath: path), "missing \(relativePath)")
        }

        let verify = try VerifiedInstallTool.run(options: VerifyInstallOptions(inputGTurbo: output))
        #expect(verify.unexpectedEntries.isEmpty)
        #expect(verify.fileCount > 0)
    }

    @Test func localRepackRejectsUnknownSourceWhenRequired() async throws {
        let snapshotDir = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("turbofieldfare-local-unknown-\(UUID().uuidString)")
        let output = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("turbofieldfare-local-unknown-out-\(UUID().uuidString).gturbo")
        defer {
            for path in [snapshotDir, output, output + ".partial", output + ".install.lock"] {
                try? FileManager.default.removeItem(atPath: path)
            }
        }
        _ = try SyntheticSnapshot.buildQwen(at: snapshotDir)
        await #expect(throws: RepackError.self) {
            _ = try await LocalStreamingRepacker(
                options: LocalStreamingRepackOptions(
                    sourceDir: snapshotDir,
                    outputDir: output,
                    requireKnownSource: true,
                    rangeChunkBytes: 4096,
                    minFreeReserveBytes: 0,
                    overwrite: true)
            ).run()
        }
    }

    @Test func localRepackRefusesExistingOutputWithoutOverwrite() async throws {
        let snapshotDir = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("turbofieldfare-local-exists-\(UUID().uuidString)")
        let output = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("turbofieldfare-local-exists-out-\(UUID().uuidString).gturbo")
        defer {
            for path in [snapshotDir, output, output + ".partial", output + ".install.lock"] {
                try? FileManager.default.removeItem(atPath: path)
            }
        }
        _ = try SyntheticSnapshot.buildQwen(at: snapshotDir)
        try Data(#"{}"#.utf8).write(
            to: URL(fileURLWithPath: (snapshotDir as NSString).appendingPathComponent("tokenizer.json")))
        try Data(#"{}"#.utf8).write(
            to: URL(fileURLWithPath: (snapshotDir as NSString).appendingPathComponent("tokenizer_config.json")))
        try FileManager.default.createDirectory(atPath: output, withIntermediateDirectories: true)
        await #expect(throws: RepackError.self) {
            _ = try await LocalStreamingRepacker(
                options: LocalStreamingRepackOptions(
                    sourceDir: snapshotDir,
                    outputDir: output,
                    requireKnownSource: false,
                    rangeChunkBytes: 4096,
                    minFreeReserveBytes: 0,
                    overwrite: false)
            ).run()
        }
    }
}
