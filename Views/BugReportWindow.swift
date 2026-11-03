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
        DSSheetScaffold(
            title: "Report a Bug",
            subtitle: "Your message goes directly to the developer. Nothing is sent until you press Send.",
            systemImage: "ladybug"
        ) {
            Form {
                Section("Summary") {
                    TextField("Title", text: $title,
                              prompt: Text("e.g. Crash when loading PDB 1ABC"))
                }

                Section {
                    ZStack(alignment: .topLeading) {
                        if steps.isEmpty {
                            Text(placeholder)
                                .font(DS.Font.mono)
                                .foregroundStyle(.tertiary)
                                .padding(.top, DS.Space.xxs)
                                .padding(.leading, DS.Space.xs + DS.Space.xxs)
                                .allowsHitTesting(false)
                                .accessibilityHidden(true)
                        }
                        TextEditor(text: $steps)
                            .font(DS.Font.mono)
                            .scrollContentBackground(.hidden)
                            .frame(minHeight: 120)
                            .accessibilityLabel("Description")
                    }
                } header: {
                    Text("Description")
                } footer: {
                    Text("What did you do, what did you expect, and what happened instead?")
                }

                Section {
                    TextField("Reply-to", text: $reporterEmail, prompt: Text("your@email.com"))
                        .textContentType(.emailAddress)
                } header: {
                    Text("Contact")
                } footer: {
                    Text("Optional — lets the developer reply to you.")
                }

                Section {
                    Toggle("Include recent activity log", isOn: $includeLog)
                    LabeledContent("Screenshot") {
                        Button {
                            Task { await captureMainWindow() }
                        } label: {
                            Label("Save to Desktop", systemImage: "camera.viewfinder")
                        }
                        .help("Capture the main Druse window as a PNG on your Desktop")
                    }
                    ForEach(savedScreenshots, id: \.self) { url in
                        LabeledContent {
                            Button("Reveal in Finder") {
                                NSWorkspace.shared.activateFileViewerSelecting([url])
                            }
                            .buttonStyle(.link)
                        } label: {
                            Label {
                                Text(url.lastPathComponent)
                                    .lineLimit(1)
                                    .truncationMode(.middle)
                            } icon: {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundStyle(DS.Color.success)
                            }
                            .font(DS.Font.label)
                        }
                    }
                    DisclosureGroup("System details sent with the report") {
                        ForEach(Self.systemDetails, id: \.label) { detail in
                            LabeledContent(detail.label) {
                                Text(detail.value)
                                    .font(DS.Font.monoSmall)
                                    .monospacedDigit()
                                    .textSelection(.enabled)
                            }
                        }
                    }
                } header: {
                    Text("Diagnostics")
                } footer: {
                    Text("Screenshots can't be uploaded automatically — email them to \(Self.replyAddress) with this report's title as the subject.")
                }
            }
            .formStyle(.grouped)
        } footer: {
            statusView
                .frame(maxWidth: .infinity, alignment: .leading)
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
            .buttonStyle(.borderedProminent)
            .keyboardShortcut(.defaultAction)
            .disabled(!canSend)
        }
        .frame(minWidth: 540, idealWidth: 600, maxWidth: 720, minHeight: 600, idealHeight: 700)
        // The scaffold header already says "Report a Bug"; a second title in the
        // window's title bar only repeats it.
        .toolbar(removing: .title)
        .toolbarBackgroundVisibility(.hidden, for: .windowToolbar)
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
                .font(DS.Font.secondary)
                .foregroundStyle(.secondary)
        case .sent:
            Label("Sent. Thank you!", systemImage: "checkmark.circle.fill")
                .font(DS.Font.secondary)
                .foregroundStyle(DS.Color.success)
        case .error(let msg):
            Label(msg, systemImage: "exclamationmark.triangle.fill")
                .font(DS.Font.secondary)
                .foregroundStyle(DS.Color.danger)
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

    /// System details shown in the form and sent with every report.
    private static let systemDetails: [(label: String, value: String)] = {
        let info = Bundle.main.infoDictionary
        let version = info?["CFBundleShortVersionString"] as? String ?? "?"
        let build = info?["CFBundleVersion"] as? String ?? "?"
        return [
            ("Druse", "\(version) (\(build))"),
            ("macOS", ProcessInfo.processInfo.operatingSystemVersionString),
            ("GPU", MTLCreateSystemDefaultDevice()?.name ?? "unknown"),
            ("Locale", Locale.current.identifier),
        ]
    }()

    private func diagnostics() -> String {
        Self.systemDetails
            .map { $0.label == "Druse" ? "Druse \($0.value)" : "\($0.label): \($0.value)" }
            .joined(separator: "\n")
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
