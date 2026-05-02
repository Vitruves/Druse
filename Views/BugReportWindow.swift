// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI
import AppKit
import Metal
import ScreenCaptureKit

struct BugReportWindow: View {
    @Environment(\.dismiss) private var dismiss

    @State private var title: String = ""
    @State private var steps: String = ""
    @State private var reporterEmail: String = ""
    @State private var savedScreenshots: [URL] = []
    @State private var includeLog: Bool = true
    @State private var status: Status = .idle

    /// Tail of the activity log to inline into the message body, in bytes.
    /// The form endpoint rejects file uploads on its free tier, so we ship
    /// the log as text. 32 KB is enough to capture recent context without
    /// blowing out the request body or the inbox.
    private static let inlineLogMaxBytes = 32 * 1024

    private enum Status: Equatable {
        case idle
        case sending
        case sent
        case error(String)
    }

    private static let endpoint = URL(string: "https://formspree.io/f/mvzdrakq")!
    fileprivate static let replyAddress = "johan.natter@gmail.com"
    private let placeholder = """
    Steps to reproduce:
    1.
    2.
    3.

    Expected:
    Actual:
    """

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Report a Bug")
                .font(.title2).bold()
            Text("Your message goes directly to the developer. No data is sent until you press Send.")
                .font(.caption)
                .foregroundStyle(.secondary)

            GroupBox("Summary") {
                TextField("Short title (e.g. \"Crash when loading PDB 1ABC\")", text: $title)
                    .textFieldStyle(.roundedBorder)
                    .padding(.vertical, 2)
            }

            GroupBox("Description") {
                ZStack(alignment: .topLeading) {
                    if steps.isEmpty {
                        Text(placeholder)
                            .foregroundStyle(.secondary.opacity(0.7))
                            .font(.system(.body, design: .monospaced))
                            .padding(6)
                            .allowsHitTesting(false)
                    }
                    TextEditor(text: $steps)
                        .font(.system(.body, design: .monospaced))
                        .frame(minHeight: 160)
                        .scrollContentBackground(.hidden)
                }
            }

            GroupBox("Reply-to (optional)") {
                TextField("your@email.com — so the developer can reply", text: $reporterEmail)
                    .textFieldStyle(.roundedBorder)
                    .padding(.vertical, 2)
            }

            GroupBox("Diagnostics") {
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Toggle("Include recent activity log", isOn: $includeLog)
                            .toggleStyle(.checkbox)
                        Spacer()
                        Button {
                            Task { await captureMainWindow() }
                        } label: {
                            Label("Save Screenshot to Desktop", systemImage: "camera.viewfinder")
                        }
                    }
                    Text("Screenshots can't be auto-uploaded — save one to your Desktop and email it to \(Self.replyAddress) with this report's summary line as the subject.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    if !savedScreenshots.isEmpty {
                        ForEach(savedScreenshots, id: \.self) { url in
                            HStack(spacing: 4) {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundStyle(.green)
                                Text("Saved: \(url.lastPathComponent)")
                                    .lineLimit(1)
                                    .truncationMode(.middle)
                                Spacer()
                                Button("Reveal") {
                                    NSWorkspace.shared.activateFileViewerSelecting([url])
                                }
                                .buttonStyle(.plain)
                                .foregroundStyle(.blue)
                            }
                            .font(.caption)
                        }
                    }
                }
                .padding(.vertical, 2)
            }

            Spacer(minLength: 0)

            HStack {
                statusView
                Spacer()
                Button("Cancel") { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Button {
                    Task { await send() }
                } label: {
                    if status == .sending {
                        ProgressView().controlSize(.small)
                    } else {
                        Text("Send")
                    }
                }
                .keyboardShortcut(.defaultAction)
                .disabled(!canSend)
            }
        }
        .padding(16)
        .frame(minWidth: 560, minHeight: 560)
        .onAppear {
            // Drop a marker so the inlined log tail makes it obvious when the
            // user reached for the bug report, separating "what they were
            // doing" from "what they typed in the form".
            ActivityLog.shared.info("--- User opened Bug Report ---", category: .system)
        }
    }

    @ViewBuilder
    private var statusView: some View {
        switch status {
        case .idle:
            EmptyView()
        case .sending:
            Label("Sending…", systemImage: "paperplane")
                .font(.caption)
                .foregroundStyle(.secondary)
        case .sent:
            Label("Sent. Thank you!", systemImage: "checkmark.circle.fill")
                .font(.caption)
                .foregroundStyle(.green)
        case .error(let msg):
            Label(msg, systemImage: "exclamationmark.triangle.fill")
                .font(.caption)
                .foregroundStyle(.red)
                .lineLimit(2)
        }
    }

    private var canSend: Bool {
        guard status != .sending else { return false }
        return !title.trimmingCharacters(in: .whitespaces).isEmpty
            && !steps.trimmingCharacters(in: .whitespaces).isEmpty
    }

    // MARK: - Capture window

    @MainActor
    private func captureMainWindow() async {
        guard let nsWindow = NSApp.keyWindow ?? NSApp.mainWindow else { return }
        let windowID = CGWindowID(nsWindow.windowNumber)

        do {
            let content = try await SCShareableContent.excludingDesktopWindows(
                false,
                onScreenWindowsOnly: true
            )
            guard let scWindow = content.windows.first(where: { $0.windowID == windowID }) else {
                status = .error("Could not find window to capture.")
                return
            }

            let filter = SCContentFilter(desktopIndependentWindow: scWindow)
            let config = SCStreamConfiguration()
            config.width = Int(filter.contentRect.width * CGFloat(filter.pointPixelScale))
            config.height = Int(filter.contentRect.height * CGFloat(filter.pointPixelScale))
            config.showsCursor = false

            let image = try await SCScreenshotManager.captureImage(
                contentFilter: filter,
                configuration: config
            )

            let rep = NSBitmapImageRep(cgImage: image)
            guard let data = rep.representation(using: .png, properties: [:]) else {
                status = .error("Could not encode screenshot.")
                return
            }

            let desktop = FileManager.default.urls(for: .desktopDirectory, in: .userDomainMask).first
                ?? FileManager.default.temporaryDirectory
            let url = desktop.appendingPathComponent("druse-bug-screenshot-\(Int(Date().timeIntervalSince1970)).png")
            try data.write(to: url)
            savedScreenshots.append(url)
        } catch {
            status = .error("Could not capture screenshot: \(error.localizedDescription)")
        }
    }

    // MARK: - Submit

    private func send() async {
        status = .sending
        do {
            let request = try buildRequest()
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse else {
                throw BugReportError.badResponse("No HTTP response")
            }
            guard (200..<300).contains(http.statusCode) else {
                let body = String(data: data, encoding: .utf8) ?? ""
                throw BugReportError.badResponse("HTTP \(http.statusCode): \(body.prefix(120))")
            }
            status = .sent
            try? await Task.sleep(nanoseconds: 1_200_000_000)
            dismiss()
        } catch {
            status = .error(error.localizedDescription)
        }
    }

    private func buildRequest() throws -> URLRequest {
        // Use application/x-www-form-urlencoded — text fields only, no file
        // uploads, since the Formspree free tier rejects them with HTTP 400
        // ("File Uploads Not Permitted"). The activity log is inlined into
        // the message body instead of attached.
        var request = URLRequest(url: Self.endpoint)
        request.httpMethod = "POST"
        request.setValue("application/x-www-form-urlencoded; charset=utf-8", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.timeoutInterval = 30

        var fields: [(String, String)] = []
        fields.append(("_subject", "[Druse Bug] \(title)"))
        fields.append(("title", title))
        fields.append(("message", composedMessage()))
        if !reporterEmail.trimmingCharacters(in: .whitespaces).isEmpty {
            fields.append(("email", reporterEmail))
            fields.append(("_replyto", reporterEmail))
        }
        fields.append(("diagnostics", diagnostics()))

        var allowed = CharacterSet.urlQueryAllowed
        allowed.remove(charactersIn: "&=+")
        let encoded = fields.map { (k, v) in
            let key = k.addingPercentEncoding(withAllowedCharacters: allowed) ?? k
            let val = v.addingPercentEncoding(withAllowedCharacters: allowed) ?? v
            return "\(key)=\(val)"
        }.joined(separator: "&")

        request.httpBody = encoded.data(using: .utf8)
        return request
    }

    /// Build the message body that gets submitted: the user's description,
    /// followed by the tail of the activity log if requested. Inlining is
    /// required because the form endpoint refuses file attachments.
    private func composedMessage() -> String {
        var parts: [String] = [steps]
        if !savedScreenshots.isEmpty {
            let names = savedScreenshots.map { $0.lastPathComponent }.joined(separator: ", ")
            parts.append("\nScreenshots saved to Desktop: \(names) — please email these separately to \(Self.replyAddress).")
        }
        if includeLog,
           let logURL = ActivityLog.shared.currentLogFileURL,
           let data = try? Data(contentsOf: logURL) {
            let tailData = data.suffix(Self.inlineLogMaxBytes)
            if let tail = String(data: tailData, encoding: .utf8), !tail.isEmpty {
                let truncatedNote = data.count > tailData.count
                    ? "[log truncated — last \(Self.inlineLogMaxBytes / 1024) KB of \(data.count / 1024) KB]\n"
                    : ""
                parts.append("\n--- Activity Log ---\n\(truncatedNote)\(tail)")
            }
        }
        return parts.joined(separator: "\n")
    }

    private func diagnostics() -> String {
        let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "?"
        let build = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "?"
        let os = ProcessInfo.processInfo.operatingSystemVersionString
        let gpu = MTLCreateSystemDefaultDevice()?.name ?? "unknown"
        let locale = Locale.current.identifier
        return """
        Druse \(version) (\(build))
        macOS: \(os)
        GPU: \(gpu)
        Locale: \(locale)
        """
    }
}

private enum BugReportError: LocalizedError {
    case badResponse(String)
    var errorDescription: String? {
        switch self {
        case .badResponse(let s): return "Send failed — \(s)"
        }
    }
}

private extension Data {
    mutating func append(_ string: String) {
        if let data = string.data(using: .utf8) { append(data) }
    }
}
