import Foundation
import TurboFieldfareRepackCore

private let usage = """
Usage:
  TurboFieldfareRepack [--model <gemma4|qwen35|qwen36>] --output <model.gturbo> [--overwrite] [--resume]
  TurboFieldfareRepack --input-local <dir> --output <model.gturbo> [--overwrite]
  TurboFieldfareRepack --discard-partial --output <model.gturbo>
  TurboFieldfareRepack --verify-install --input-gturbo <model.gturbo>
  TurboFieldfareRepack --help

The installer streams the selected checkpoint (default: the supported Gemma 4
checkpoint) from Hugging Face and repackages it without materializing the
source checkpoint on disk. Set HF_TOKEN only if Hugging Face requests
authentication. A cancelled or interrupted download can be continued with
--resume or removed with --discard-partial.

--input-local repacks a local safetensors directory (containing
model.safetensors.index.json and config.json) into .gturbo format.
"""

private struct Arguments {
    var model = SupportedModelSource.default
    var inputLocal: String?
    var output: String?
    var overwrite = false
    var resume = false
    var discardPartial = false
    var verifyInstall = false
    var inputGTurbo: String?

    static func parse(_ values: [String]) throws -> Arguments {
        var parsed = Arguments()
        var index = 1
        while index < values.count {
            let flag = values[index]
            switch flag {
            case "--help":
                throw ParseError.help
            case "--overwrite":
                parsed.overwrite = true
                index += 1
            case "--resume":
                parsed.resume = true
                index += 1
            case "--discard-partial":
                parsed.discardPartial = true
                index += 1
            case "--verify-install":
                parsed.verifyInstall = true
                index += 1
            case "--input-local":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                 }
                parsed.inputLocal = values[index + 1]
                index += 2
            case "--model":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                 }
                guard let source = SupportedModelSource.named(values[index + 1]) else {
                    throw ParseError.invalidMode(
                         "unknown model \"\(values[index + 1])\"; supported: "
                         + SupportedModelSource.all.map(\.name).joined(separator: ", "))
                 }
                parsed.model = source
                index += 2
            case "--output", "--input-gturbo":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                 }
                if flag == "--output" {
                    parsed.output = values[index + 1]
                 } else {
                    parsed.inputGTurbo = values[index + 1]
                 }
                index += 2
            default:
                throw ParseError.unknown(flag)
             }
         }

        guard !(parsed.resume && parsed.discardPartial) else {
            throw ParseError.invalidMode("--resume and --discard-partial are mutually exclusive")
         }
        guard !(parsed.resume && parsed.verifyInstall) else {
            throw ParseError.invalidMode("--resume and --verify-install are mutually exclusive")
         }
        guard !(parsed.discardPartial && parsed.verifyInstall) else {
            throw ParseError.invalidMode("--discard-partial and --verify-install are mutually exclusive")
         }

         // --input-local mode
        if let _ = parsed.inputLocal {
            guard parsed.output != nil else {
                throw ParseError.missingRequired("--output")
             }
            guard parsed.inputGTurbo == nil else {
                throw ParseError.invalidMode("--input-local and --input-gturbo are mutually exclusive")
             }
            // --resume, --discard-partial, --verify-install are incompatible with --input-local
            guard !parsed.resume else {
                throw ParseError.invalidMode("--input-local does not support --resume")
             }
            guard !parsed.discardPartial else {
                throw ParseError.invalidMode("--input-local does not support --discard-partial")
             }
            guard !parsed.verifyInstall else {
                throw ParseError.invalidMode("--input-local does not support --verify-install")
             }
             // --model is optional for local repack (arch is loaded from config.json)
             // but if provided it must be valid (already validated above)
        } else {
            // Remote repack mode
            guard parsed.output != nil else {
                throw ParseError.missingRequired("--output")
             }
            guard parsed.inputLocal == nil else {
                throw ParseError.invalidMode("--output requires --input-local or remote download")
             }
            guard parsed.inputGTurbo == nil else {
                throw ParseError.invalidMode("--input-gturbo requires --verify-install")
             }
        }
        return parsed
     }
}

private enum ParseError: Error, CustomStringConvertible {
    case help
    case unknown(String)
    case missingValue(String)
    case missingRequired(String)
    case invalidMode(String)

    var description: String {
        switch self {
        case .help: return "help"
        case .unknown(let flag): return "unknown argument: \(flag)"
        case .missingValue(let flag): return "missing value for \(flag)"
        case .missingRequired(let flag): return "required argument: \(flag)"
        case .invalidMode(let message): return message
         }
      }
}

private func printError(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

private func run(_ values: [String]) async -> Int32 {
    let arguments: Arguments
    do {
        arguments = try Arguments.parse(values)
     } catch ParseError.help {
        print(usage)
        return 0
     } catch {
        printError("error: \(error)\n\n\(usage)")
        return 2
     }

    if arguments.discardPartial, let output = arguments.output {
        do {
            try RemoteStreamingRepacker.discardPartial(outputDirectory: output)
            print("Discarded saved download for \(output)")
            return 0
         } catch {
            printError("discard failed: \(error)")
            return 1
         }
    }

    if arguments.verifyInstall, let input = arguments.inputGTurbo {
        do {
            let result = try VerifiedInstallTool.run(
                options: VerifyInstallOptions(inputGTurbo: input))
            print("Verified \(result.fileCount) files (\(result.bytesVerified) bytes)")
            print("Receipt: \(result.receiptPath)")
            return 0
         } catch {
            printError("verification failed: \(error)")
            return 1
         }
    }

     // Local repack mode
    if let inputLocal = arguments.inputLocal, let output = arguments.output {
        let options = LocalStreamingRepackOptions(
            inputLocal: inputLocal,
            outputDir: output,
            overwrite: arguments.overwrite)
        do {
            let result = try await LocalStreamingRepacker(options: options).run()
            print("Repacked local model")
            print("Model: \(result.outputDir)")
            print("Bytes read: \(result.localBytesRead)")
            return 0
         } catch {
            printError("repack failed: \(error)")
            return 1
         }
    }

    guard let output = arguments.output else { return 2 }
    let source = arguments.model
    let options = source.installOptions(
        outputDirectory: URL(fileURLWithPath: output),
        overwrite: arguments.overwrite,
        token: ProcessInfo.processInfo.environment["HF_TOKEN"],
        resume: arguments.resume)
    do {
        let result = try await RemoteStreamingRepacker(options: options).run()
        print("Installed \(source.displayName)")
        print("Source revision: \(result.resolvedCommit)")
        print("Model: \(result.outputDir)")
        return 0
     } catch {
        printError("install failed: \(error)")
        return 1
     }
}

exit(await run(CommandLine.arguments))
