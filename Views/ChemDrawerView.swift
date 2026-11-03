// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI
import WebKit

// MARK: - Chemical Structure Drawer (Ketcher)

/// Embeds the Ketcher chemical structure editor in a WKWebView.
///
/// Communication:
/// - Ketcher posts `{ eventType: "init" }` when ready.
/// - Swift calls `ketcher.getSmiles()` / `ketcher.getSmarts()` / `ketcher.setMolecule()` via JS.
/// - An `onStructureChanged` callback fires whenever the user draws or modifies a structure.
///
/// Usage:
/// ```swift
/// ChemDrawerView(initialSmiles: "c1ccccc1", mode: .scaffold) { smiles, smarts in
///     print(smiles, smarts)
/// }
/// ```
struct ChemDrawerView: NSViewRepresentable {
    /// Optional SMILES to pre-load into the editor.
    var initialSmiles: String?

    /// Controls which format to export.
    enum Mode {
        /// General molecule drawing — exports SMILES.
        case molecule
        /// Scaffold / query drawing — exports SMARTS.
        case scaffold
    }

    var mode: Mode = .molecule

    /// Called when the drawn structure changes. Provides (smiles, smarts).
    var onStructureChanged: ((String, String) -> Void)?

    /// Called when the drawn structure changes, with the raw mol block included.
    /// Unlike `onStructureChanged`, this preserves R-group / attachment-point
    /// information that is lost by Ketcher's `getSmiles()` on aromatic atoms.
    /// Used by the scaffold analog generator.
    var onScaffoldChanged: ((_ smiles: String, _ smarts: String, _ molfile: String) -> Void)?

    func makeNSView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.preferences.setValue(true, forKey: "allowFileAccessFromFileURLs")

        let handler = context.coordinator
        config.userContentController.add(handler, name: "ketcherReady")
        config.userContentController.add(handler, name: "ketcherChanged")

        // Inject a bridge script that:
        // 1. Listens for Ketcher init event and notifies Swift.
        // 2. Polls for structure changes (Ketcher has no built-in onChange callback).
        let bridgeScript = WKUserScript(source: Self.bridgeJS, injectionTime: .atDocumentEnd, forMainFrameOnly: true)
        config.userContentController.addUserScript(bridgeScript)

        let webView = WKWebView(frame: .zero, configuration: config)
        webView.setValue(false, forKey: "drawsBackground")
        handler.webView = webView

        // Load the local Ketcher standalone build
        if let ketcherURL = Bundle.main.url(forResource: "index", withExtension: "html",
                                             subdirectory: "standalone") {
            webView.loadFileURL(ketcherURL, allowingReadAccessTo: ketcherURL.deletingLastPathComponent())
        }

        return webView
    }

    func updateNSView(_ webView: WKWebView, context: Context) {
        context.coordinator.mode = mode
        context.coordinator.onStructureChanged = onStructureChanged
        context.coordinator.onScaffoldChanged = onScaffoldChanged
        // If initialSmiles changed after Ketcher is ready, reload
        if let smiles = initialSmiles, smiles != context.coordinator.lastLoadedSmiles,
           context.coordinator.isReady {
            context.coordinator.setMolecule(smiles)
        }
    }

    func makeCoordinator() -> Coordinator {
        let c = Coordinator(initialSmiles: initialSmiles, mode: mode, onChanged: onStructureChanged)
        c.onScaffoldChanged = onScaffoldChanged
        return c
    }

    // MARK: - Coordinator

    @MainActor
    final class Coordinator: NSObject, WKScriptMessageHandler {
        weak var webView: WKWebView?
        var mode: Mode
        var onStructureChanged: ((String, String) -> Void)?
        var onScaffoldChanged: ((String, String, String) -> Void)?
        var isReady = false
        var lastLoadedSmiles: String?
        private let initialSmiles: String?

        init(initialSmiles: String?, mode: Mode, onChanged: ((String, String) -> Void)?) {
            self.initialSmiles = initialSmiles
            self.mode = mode
            self.onStructureChanged = onChanged
        }

        func userContentController(_ userContentController: WKUserContentController,
                                   didReceive message: WKScriptMessage) {
            switch message.name {
            case "ketcherReady":
                isReady = true
                if let smiles = initialSmiles, !smiles.isEmpty {
                    setMolecule(smiles)
                }

            case "ketcherChanged":
                guard let body = message.body as? [String: String] else { return }
                let smiles = body["smiles"] ?? ""
                let smarts = body["smarts"] ?? ""
                let molfile = body["molfile"] ?? ""
                // Track what Ketcher reported so updateNSView won't echo it back via setMolecule()
                lastLoadedSmiles = smiles
                onStructureChanged?(smiles, smarts)
                onScaffoldChanged?(smiles, smarts, molfile)

            default:
                break
            }
        }

        func setMolecule(_ smiles: String) {
            lastLoadedSmiles = smiles
            let escaped = smiles.replacingOccurrences(of: "\\", with: "\\\\")
                                .replacingOccurrences(of: "'", with: "\\'")
            webView?.evaluateJavaScript("""
                (async () => {
                    try { await window.ketcher.setMolecule('\(escaped)'); } catch(e) { console.error(e); }
                })();
            """)
        }

        /// Retrieve the current structure. Completion provides (smiles, smarts).
        func getStructure(completion: @escaping (String, String) -> Void) {
            guard isReady, let wv = webView else { completion("", ""); return }
            wv.evaluateJavaScript("""
                (async () => {
                    try {
                        const smiles = await window.ketcher.getSmiles();
                        const smarts = await window.ketcher.getSmarts();
                        return JSON.stringify({ smiles, smarts });
                    } catch(e) { return JSON.stringify({ smiles: '', smarts: '' }); }
                })();
            """) { result, _ in
                guard let json = result as? String,
                      let data = json.data(using: .utf8),
                      let dict = try? JSONSerialization.jsonObject(with: data) as? [String: String] else {
                    completion("", ""); return
                }
                completion(dict["smiles"] ?? "", dict["smarts"] ?? "")
            }
        }
    }

    // MARK: - Bridge JavaScript

    /// JS injected into Ketcher's page to bridge with Swift via WKScriptMessageHandler.
    private static let bridgeJS = """
    (function() {
        // Wait for Ketcher to fire the init event
        window.addEventListener('message', function(event) {
            if (event.data && event.data.eventType === 'init') {
                window.webkit.messageHandlers.ketcherReady.postMessage('ready');
                startPolling();
            }
        });

        // Also check if ketcher is already available (in case init fired before our listener)
        let initCheck = setInterval(function() {
            if (window.ketcher) {
                clearInterval(initCheck);
                window.webkit.messageHandlers.ketcherReady.postMessage('ready');
                startPolling();
            }
        }, 200);

        // Poll for changes every 800ms — Ketcher lacks an onChange callback.
        // We ratchet on (smiles + molfile) so R-group / attachment-point edits
        // that don't alter the SMILES still fire a change event (needed by the
        // scaffold analog generator, which relies on the molfile).
        let lastKey = '';
        function startPolling() {
            clearInterval(initCheck);
            setInterval(async function() {
                try {
                    if (!window.ketcher) return;
                    const smiles = await window.ketcher.getSmiles();
                    let molfile = '';
                    try { molfile = await window.ketcher.getMolfile(); } catch(e) {}
                    const key = smiles + '\\u0001' + molfile;
                    if (key !== lastKey) {
                        lastKey = key;
                        let smarts = '';
                        try { smarts = await window.ketcher.getSmarts(); } catch(e) {}
                        window.webkit.messageHandlers.ketcherChanged.postMessage({
                            smiles: smiles,
                            smarts: smarts,
                            molfile: molfile
                        });
                    }
                } catch(e) {}
            }, 800);
        }
    })();
    """;
}

// MARK: - Standalone Drawer Sheet

/// A modal sheet wrapping ChemDrawerView with confirm/cancel buttons.
/// Used from both the pharmacophore editor and the ligand database.
struct ChemDrawerSheet: View {
    @Environment(\.dismiss) private var dismiss

    let title: String
    let initialSmiles: String
    let mode: ChemDrawerView.Mode
    let onConfirm: (String, String) -> Void

    @State private var currentSmiles: String = ""
    @State private var currentSmarts: String = ""

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack(spacing: DS.Space.m) {
                Image(systemName: "pencil.and.outline")
                    .font(.title3)
                    .symbolRenderingMode(.hierarchical)
                    .foregroundStyle(DS.Color.accent)
                Text(title)
                    .font(DS.Font.stepTitle)
                    .lineLimit(1)
                Spacer(minLength: DS.Space.l)

                if !currentSmiles.isEmpty {
                    Text(currentSmiles)
                        .font(DS.Font.monoSmall)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .textSelection(.enabled)
                        .frame(maxWidth: 360, alignment: .trailing)
                        .help(currentSmiles)
                }
            }
            .padding(.horizontal, DS.Space.xl)
            .padding(.vertical, DS.Space.m)

            Divider()

            // Ketcher editor
            ChemDrawerView(
                initialSmiles: initialSmiles.isEmpty ? nil : initialSmiles,
                mode: mode
            ) { smiles, smarts in
                currentSmiles = smiles
                currentSmarts = smarts
            }

            Divider()

            // Footer
            HStack(spacing: DS.Space.m) {
                if mode == .scaffold && !currentSmarts.isEmpty {
                    HStack(spacing: DS.Space.xs) {
                        Text("SMARTS")
                            .font(DS.Font.eyebrow)
                            .tracking(0.6)
                            .foregroundStyle(.tertiary)
                        Text(currentSmarts)
                            .font(DS.Font.monoSmall)
                            .foregroundStyle(DS.Color.pocket)
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .textSelection(.enabled)
                            .frame(maxWidth: 320, alignment: .leading)
                            .help(currentSmarts)
                    }
                }
                Spacer()
                Button("Cancel") { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Button(mode == .scaffold ? "Use as Scaffold" : "Use Structure") {
                    onConfirm(currentSmiles, currentSmarts)
                    dismiss()
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(currentSmiles.isEmpty)
            }
            .padding(DS.Space.l)
        }
        .frame(minWidth: 900, minHeight: 650)
        .onAppear {
            currentSmiles = initialSmiles
        }
    }
}
