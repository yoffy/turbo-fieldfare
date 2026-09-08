import Darwin
import Foundation
import TurboFieldfareServerCore

private func isLoopbackHost(_ host: String) -> Bool {
    let normalized = host.lowercased()
    return normalized == "localhost"
        || normalized == "127.0.0.1"
        || normalized == "::1"
        || normalized.hasPrefix("127.")
}

let arguments: ServerArguments
do {
    arguments = try ServerArguments.parse(Array(CommandLine.arguments.dropFirst()))
} catch ServerArgumentError.help {
    print(ServerArguments.usage)
    exit(0)
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n\n\(ServerArguments.usage)\n".utf8))
    exit(2)
}

do {
    let signals = ServerTerminationSignals()
    let modelURL = URL(fileURLWithPath: arguments.model).standardizedFileURL
    let backend = try await ServerModelSession.load(
        modelDirectory: modelURL,
        maxContext: arguments.maxContext,
        promptCacheMode: arguments.promptCacheMode)
    let modelID = arguments.modelIDOverride ?? backend.defaultModelID
    let server = TurboFieldfareHTTPServer(
        modelID: modelID,
        queueLimit: arguments.queueLimit,
        backend: backend,
        chatDialect: backend.chatDialect)
    _ = try await server.start(port: arguments.port, host: arguments.host)
    print("TurboFieldfareServer ready at http://\(arguments.host):\(arguments.port) model=\(modelID) context=\(arguments.maxContext) prompt_cache=\(arguments.promptCacheMode.rawValue)")
    if !isLoopbackHost(arguments.host) {
        FileHandle.standardError.write(Data("warning: binding to non-loopback address '\(arguments.host)' exposes the server without authentication or TLS\n".utf8))
    }

    _ = await signals.wait()
    try await server.shutdown()
    await signals.cancel()
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}
