import Foundation
import Testing

@testable import TurboFieldfareRepackCore

extension RemotePayloadCopyTests {
    @Test func qwen38RemoteInstallStreamsPleShards() async throws {
        let snapshotDir = tmpDirForRemote("qwen38-snap")
        let remoteOutput = tmpPathForRemote("qwen38-remote")
        defer { cleanUpRemote([snapshotDir, remoteOutput]) }
        let snapshot = try SyntheticSnapshot.buildQwen38(
            at: snapshotDir,
            seed: 0x38F1A5E500000002)

        resetFakeHF()
        FakeHFURLProtocol.files = try remoteFiles(
            snapshotDir: snapshotDir,
            snap: snapshot,
            includeRequiredTokenizer: true,
            includeOptionalTokenizer: true)
        let recorder = InstallProgressRecorder()

        let result = try await RemoteStreamingRepacker(
            options: remoteOptions(
                outputDir: remoteOutput,
                session: fakeHFSession())
        ).run { recorder.append($0) }

        #expect(result.reusedBytes == 0)
        #expect(result.downloadedThisRunBytes == result.remoteBytesToDownload)
        // 4 experts layers + 4 PLE shards + layouts + manifest.
        for relativePath in [
            "model_weights.bin",
            "packed_experts/layout.json",
            "packed_experts/layer_00.bin",
            "packed_experts/layer_01.bin",
            "packed_experts/layer_02.bin",
            "packed_experts/layer_03.bin",
            "packed_ple/layout.json",
            "packed_ple/shard_000.bin",
            "packed_ple/shard_001.bin",
            "packed_ple/shard_002.bin",
            "packed_ple/shard_003.bin",
            "manifest.json",
        ] {
            let remote = (remoteOutput as NSString).appendingPathComponent(relativePath)
            #expect(FileManager.default.fileExists(atPath: remote), "\(relativePath) exists")
        }
        #expect(recorder.values.contains(.finalizing))

        let manifestData = try Data(contentsOf: URL(fileURLWithPath:
            (remoteOutput as NSString).appendingPathComponent("manifest.json")))
        let manifest = try JSONSerialization.jsonObject(with: manifestData) as! [String: Any]
        let arch = manifest["arch"] as! [String: Any]
        #expect(arch["family"] as? String == "qwen38")
        #expect(arch["hcCount"] as? Int == 2)
        #expect(arch["pleSplitParts"] as? Int == 4)
        #expect(arch["indexerNHeads"] as? Int == 2)
        #expect(manifest["pleShards"] as? Int == 4)
        let quant = manifest["quant"] as! [String: [String: Any]]
        #expect(quant["router"]?["weightBits"] as? Int == 8)
        #expect(quant["router"]?["groupSize"] as? Int == 64)
        #expect(quant["attention"]?["weightBits"] as? Int == 4)
        #expect(quant["attention"]?["groupSize"] as? Int == 32)
        #expect(quant["routedExpert"]?["groupSize"] as? Int == 32)

        let pleLayoutData = try Data(contentsOf: URL(fileURLWithPath:
            (remoteOutput as NSString)
                .appendingPathComponent("packed_ple/layout.json")))
        let pleLayout = try JSONSerialization.jsonObject(with: pleLayoutData) as! [String: Any]
        #expect(pleLayout["shardCount"] as? Int == 4)

        // The finished install passes post-hoc verification.
        let verify = try VerifiedInstallTool.run(
            options: VerifyInstallOptions(inputGTurbo: remoteOutput))
        #expect(verify.unexpectedEntries.isEmpty)
        #expect(verify.fileCount > 0)
    }
}
