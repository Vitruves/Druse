// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import AppKit

/// Installs the `druse` terminal command and its shell completion scripts for the current user
/// (Settings ▸ Command Line / Shell Completions, and the `cli` console command).
///
/// The installed `druse` is a small bash wrapper that runs this app bundle's executable with
/// `--headless`; it is written with mode 0755, so it works without a manual `chmod +x`. It knows
/// the bundle path it was installed from: after moving Druse.app, installing again updates it
/// (`status` reports such a wrapper as `.stale`).
///
/// Every location derives from `home`, so the installer can be exercised against a temporary
/// directory (`HOME=/tmp/x druse cli install`); `.current` uses `$HOME` and this app bundle.
struct CommandLineInstaller {

    /// Where the `druse` command is installed.
    enum Location: String, CaseIterable, Sendable {
        /// `~/.local/bin`: no administrator rights needed; may need adding to PATH.
        case user
        /// `/usr/local/bin`: on the default PATH of every shell; needs an administrator password.
        case system
    }

    /// State of the `druse` command at one location.
    enum ToolStatus: Equatable {
        case notInstalled
        /// Our wrapper, pointing at this app bundle.
        case installed
        /// Our wrapper, pointing at another (moved or older) Druse.app.
        case stale(appPath: String)
        /// Some other file (a symlink to a source checkout's Tools/druse, another tool…); left alone.
        case foreign
    }

    /// State of one shell's completion script.
    enum CompletionStatus: Equatable {
        case notInstalled
        case current
        /// Installed but generated from an older command registry.
        case outdated
        /// A file Druse did not generate (no `ShellCompletions.signature`); never replaced or removed.
        case foreign
    }

    struct InstallError: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    /// First line after the shebang of every wrapper we write; identifies files we may replace.
    private static let marker = "# Installed by Druse.app (Settings ▸ Command Line)."

    let home: URL
    let appBundle: URL

    /// `AppSettings.shellHome` (`$HOME`) and the running app bundle.
    static var current: CommandLineInstaller {
        CommandLineInstaller(home: AppSettings.shellHome, appBundle: Bundle.main.bundleURL)
    }

    // MARK: - The druse command

    func directory(for location: Location) -> URL {
        switch location {
        case .user: home.appendingPathComponent(".local/bin", isDirectory: true)
        case .system: URL(fileURLWithPath: "/usr/local/bin", isDirectory: true)
        }
    }

    func toolURL(for location: Location) -> URL {
        directory(for: location).appendingPathComponent("druse")
    }

    /// The wrapper's text: runs `<bundle>/Contents/MacOS/Druse --headless "$@"`, honouring `$DRUSE_APP`.
    var wrapperScript: String {
        """
        #!/bin/bash
        \(Self.marker)
        # Runs DruseScript headless: 'druse --help'. Reinstall from Druse ▸ Settings ▸ Command Line after moving Druse.app.
        app="${DRUSE_APP:-\(quotedAppPath)}"
        if [[ ! -x "$app/Contents/MacOS/Druse" ]]; then
            echo "druse: Druse.app not found at $app; reinstall from Druse ▸ Settings ▸ Command Line, or set DRUSE_APP" >&2
            exit 2
        fi
        exec "$app/Contents/MacOS/Druse" --headless "$@"

        """
    }

    /// The bundle path as written inside the wrapper's double quotes.
    private var quotedAppPath: String {
        var quoted = ""
        for character in appBundle.path {
            if "\\\"$`".contains(character) { quoted.append("\\") }
            quoted.append(character)
        }
        return quoted
    }

    func status(at location: Location) -> ToolStatus {
        let url = toolURL(for: location)
        guard FileManager.default.fileExists(atPath: url.path) || isSymlink(url) else { return .notInstalled }
        guard let text = try? String(contentsOf: url, encoding: .utf8), text.contains(Self.marker) else { return .foreign }
        guard let appPath = Self.appPath(inWrapper: text) else { return .foreign }
        return appPath == quotedAppPath ? .installed : .stale(appPath: appPath)
    }

    /// Writes the wrapper to `~/.local/bin/druse` (creating the folder); returns its path.
    @discardableResult
    func installForUser() throws -> URL {
        let url = toolURL(for: .user)
        if status(at: .user) == .foreign {
            throw InstallError(message: "\(url.path) already exists and was not installed by Druse; remove it first")
        }
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? FileManager.default.removeItem(at: url)
        guard FileManager.default.createFile(atPath: url.path, contents: Data(wrapperScript.utf8),
                                             attributes: [.posixPermissions: 0o755]) else {
            throw InstallError(message: "cannot write \(url.path)")
        }
        return url
    }

    /// Writes the wrapper to `/usr/local/bin/druse` through the system administrator prompt.
    @MainActor
    func installForAllUsers() throws {
        if status(at: .system) == .foreign {
            throw InstallError(message: "/usr/local/bin/druse already exists and was not installed by Druse; remove it first")
        }
        let staging = FileManager.default.temporaryDirectory.appendingPathComponent("druse-\(UUID().uuidString)")
        try Data(wrapperScript.utf8).write(to: staging)
        defer { try? FileManager.default.removeItem(at: staging) }
        let target = toolURL(for: .system).path
        try Self.runAsAdministrator(
            "mkdir -p /usr/local/bin && install -m 0755 \(Self.shellQuote(staging.path)) \(Self.shellQuote(target))")
    }

    /// Removes a wrapper Druse installed (never a foreign file). `.system` asks for an administrator password.
    @MainActor
    func uninstall(from location: Location) throws {
        switch status(at: location) {
        case .notInstalled:
            return
        case .foreign:
            throw InstallError(message: "\(toolURL(for: location).path) was not installed by Druse; left in place")
        case .installed, .stale:
            switch location {
            case .user: try FileManager.default.removeItem(at: toolURL(for: .user))
            case .system: try Self.runAsAdministrator("rm -f \(Self.shellQuote(toolURL(for: .system).path))")
            }
        }
    }

    // MARK: - PATH

    /// The shell the user logs in with (from the account database, not `$SHELL`, which GUI apps do
    /// not reliably inherit); nil for shells Druse has no completions for.
    static var loginShell: CompletionShell? {
        guard let entry = getpwuid(getuid()), let shell = entry.pointee.pw_shell else { return nil }
        return CompletionShell(rawValue: URL(fileURLWithPath: String(cString: shell)).lastPathComponent)
    }

    /// PATH as a login shell of `shell` sets it up (GUI apps get launchd's minimal PATH). Runs the
    /// shell once, off the main thread; nil when it fails or takes over 10 s.
    ///
    /// The shell is a login shell but not interactive (`-l`, no `-i`): an interactive shell sources
    /// the whole rc, may grab the terminal's foreground process group (suspending a headless REPL
    /// with SIGTTIN) and can start background helpers. PATH set only in `~/.zshrc` or `~/.bashrc` is
    /// therefore not seen; fish reads `config.fish` in every mode. Output is read after the shell
    /// exits, without blocking, so a background child that keeps the pipe open cannot hang the call.
    static func shellPATH(for shell: CompletionShell) async -> [String]? {
        let executable: String?
        let script: String
        switch shell {
        case .fish:
            executable = fishExecutable
            script = "printf '__DRUSE_PATH__%s__END__' (string join : $PATH)"
        case .zsh:
            executable = "/bin/zsh"
            script = "printf '__DRUSE_PATH__%s__END__' \"$PATH\""
        case .bash:
            executable = "/bin/bash"
            script = "printf '__DRUSE_PATH__%s__END__' \"$PATH\""
        }
        guard let executable else { return nil }
        return await Task.detached(priority: .utility) { () -> [String]? in
            let process = Process()
            process.executableURL = URL(fileURLWithPath: executable)
            process.arguments = ["-l", "-c", script]
            let pipe = Pipe()
            process.standardOutput = pipe
            process.standardError = FileHandle.nullDevice
            process.standardInput = FileHandle.nullDevice
            do { try process.run() } catch { return nil }
            let deadline = DispatchTime.now() + .seconds(10)
            DispatchQueue.global().asyncAfter(deadline: deadline) { if process.isRunning { process.terminate() } }
            process.waitUntilExit()
            let output = String(decoding: readAvailable(pipe.fileHandleForReading), as: UTF8.self)
            guard let start = output.range(of: "__DRUSE_PATH__"),
                  let end = output.range(of: "__END__", range: start.upperBound..<output.endIndex) else { return nil }
            return output[start.upperBound..<end.lowerBound].split(separator: ":").map(String.init)
        }.value
    }

    /// Everything already buffered in `handle`, without waiting for the writer to close it.
    private static func readAvailable(_ handle: FileHandle) -> Data {
        let fd = handle.fileDescriptor
        _ = fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK)
        var data = Data()
        var buffer = [UInt8](repeating: 0, count: 16_384)
        while true {
            let count = buffer.withUnsafeMutableBytes { read(fd, $0.baseAddress, $0.count) }
            guard count > 0 else { break }
            data.append(contentsOf: buffer[0..<count])
        }
        return data
    }

    private static var fishExecutable: String? {
        ["/opt/homebrew/bin/fish", "/usr/local/bin/fish", "/usr/bin/fish"].first {
            FileManager.default.isExecutableFile(atPath: $0)
        }
    }

    /// The line that puts `directory` on PATH in `shell`, and the file it goes in.
    func pathSetup(for shell: CompletionShell, directory: URL) -> (line: String, file: String) {
        let display = displayPath(directory)
        switch shell {
        case .fish: return ("fish_add_path \(display)", "run once in fish (it is remembered)")
        case .zsh: return ("export PATH=\"\(display.replacingOccurrences(of: "~", with: "$HOME")):$PATH\"", "~/.zshrc")
        case .bash: return ("export PATH=\"\(display.replacingOccurrences(of: "~", with: "$HOME")):$PATH\"", "~/.bash_profile")
        }
    }

    // MARK: - Shell completions

    /// The standard per-user location each shell loads completions from.
    func completionURL(for shell: CompletionShell) -> URL {
        switch shell {
        case .fish: home.appendingPathComponent(".config/fish/completions/druse.fish")
        case .zsh: home.appendingPathComponent(".zfunc/_druse")
        case .bash: home.appendingPathComponent(".local/share/bash-completion/completions/druse")
        }
    }

    /// Compares the installed script with `script` (freshly generated from the registry).
    func completionStatus(for shell: CompletionShell, script: String) -> CompletionStatus {
        guard let installed = try? String(contentsOf: completionURL(for: shell), encoding: .utf8) else {
            return FileManager.default.fileExists(atPath: completionURL(for: shell).path) ? .foreign : .notInstalled
        }
        if installed == script { return .current }
        return installed.contains(ShellCompletions.signature) ? .outdated : .foreign
    }

    /// Writes `script` to the shell's completion file. Refuses to replace a file Druse did not generate.
    @discardableResult
    func installCompletions(for shell: CompletionShell, script: String) throws -> URL {
        let url = completionURL(for: shell)
        try refuseForeignCompletion(shell, at: url)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(script.utf8).write(to: url, options: .atomic)
        return url
    }

    /// Removes the shell's completion file if Druse generated it; a foreign file is left in place.
    func uninstallCompletions(for shell: CompletionShell) throws {
        let url = completionURL(for: shell)
        guard FileManager.default.fileExists(atPath: url.path) else { return }
        try refuseForeignCompletion(shell, at: url)
        try FileManager.default.removeItem(at: url)
    }

    private func refuseForeignCompletion(_ shell: CompletionShell, at url: URL) throws {
        guard completionStatus(for: shell, script: "") == .foreign else { return }
        throw InstallError(message: "\(displayPath(url)) was not generated by Druse; move it away first")
    }

    /// What the user still has to do for the shell to load the installed script, or nil when it is
    /// picked up automatically (fish).
    func completionSetupHint(for shell: CompletionShell) -> String? {
        switch shell {
        case .fish:
            return nil
        case .zsh:
            return "Add to ~/.zshrc, before any compinit call:  fpath=(~/.zfunc $fpath); autoload -Uz compinit && compinit"
        case .bash:
            return "Loaded automatically with bash-completion 2 (brew install bash-completion@2); otherwise add to ~/.bash_profile:  source \(displayPath(completionURL(for: .bash)))"
        }
    }

    /// `path` with the home folder written as `~`.
    func displayPath(_ url: URL) -> String {
        let path = url.path
        return path.hasPrefix(home.path + "/") ? "~" + path.dropFirst(home.path.count) : path
    }

    // MARK: - Helpers

    private func isSymlink(_ url: URL) -> Bool {
        (try? FileManager.default.destinationOfSymbolicLink(atPath: url.path)) != nil
    }

    private static func appPath(inWrapper text: String) -> String? {
        guard let line = text.split(separator: "\n").first(where: { $0.hasPrefix("app=\"${DRUSE_APP:-") }) else { return nil }
        return String(line.dropFirst("app=\"${DRUSE_APP:-".count).dropLast("}\"".count))
    }

    private static func shellQuote(_ text: String) -> String {
        "'" + text.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }

    /// Runs a shell command through the macOS administrator authorization prompt.
    @MainActor
    private static func runAsAdministrator(_ command: String) throws {
        let escaped = command.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"")
        let source = "do shell script \"\(escaped)\" with administrator privileges"
        var error: NSDictionary?
        guard let script = NSAppleScript(source: source) else {
            throw InstallError(message: "cannot prepare the administrator prompt")
        }
        script.executeAndReturnError(&error)
        if let error {
            let message = error[NSAppleScript.errorMessage] as? String ?? "authorization failed"
            throw InstallError(message: message)
        }
    }
}
