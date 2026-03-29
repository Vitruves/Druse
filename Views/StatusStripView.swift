import SwiftUI

/// Persistent bottom status strip showing current operation, contextual hints, and system metrics.
struct StatusStripView: View {
    @Environment(AppViewModel.self) private var viewModel
    @Binding var showConsole: Bool
    @Binding var consoleHeight: CGFloat
    let log = ActivityLog.shared

    @State private var filterLevel: LogLevel? = nil
    @State private var scrollToBottom = true

    @GestureState private var dragStartHeight: CGFloat = 0

    // Console selection state
    @State private var selectedEntryIDs: Set<UUID> = []
    @State private var lastClickedID: UUID? = nil

    var filteredEntries: [LogEntry] {
        if let level = filterLevel {
            return log.entries.filter { $0.level == level }
        }
        return log.entries
    }

    // MARK: - Computed Status

    private var progressFraction: Float {
        if viewModel.docking.isBatchDocking {
            let bp = viewModel.docking.batchProgress
            return bp.total > 0 ? Float(bp.current) / Float(bp.total) : 0
        }
        if viewModel.docking.screeningProgress > 0 && viewModel.docking.screeningProgress < 1 {
            return viewModel.docking.screeningProgress
        }
        return 0
    }

    private var isScreeningActive: Bool {
        if case .idle = viewModel.docking.screeningState { return false }
        return true
    }

    private var isActive: Bool {
        viewModel.docking.isDocking || viewModel.docking.isBatchDocking || viewModel.molecules.isMinimizing
            || viewModel.workspace.isGeneratingSurface || viewModel.workspace.isLoading || viewModel.workspace.isSearching
            || isScreeningActive || viewModel.leadOpt.isGenerating || viewModel.leadOpt.isDocking
            || viewModel.workspace.statusMessage.hasSuffix("...")
    }

    private var statusInfo: (icon: String, message: String, color: Color) {
        if viewModel.docking.isBatchDocking {
            let bp = viewModel.docking.batchProgress
            return ("bolt.fill", "Docking \(bp.current)/\(bp.total)", .cyan)
        }
        if viewModel.docking.isDocking {
            return ("bolt.fill", "Docking...", .cyan)
        }
        if viewModel.molecules.isMinimizing {
            return ("waveform.path.ecg", "Structure cleanup...", .orange)
        }
        if viewModel.workspace.isGeneratingSurface {
            return ("drop.halffull", "Generating surface...", .blue)
        }
        if viewModel.workspace.isLoading {
            return ("arrow.down.circle", viewModel.workspace.loadingMessage, .blue)
        }
        if viewModel.workspace.isSearching {
            return ("magnifyingglass", "Searching RCSB...", .blue)
        }
        // Use statusMessage (updated immediately) instead of log.entries.last (throttled)
        let msg = viewModel.workspace.statusMessage
        if msg != "Ready", let last = log.entries.last {
            return (last.level.icon, msg, last.level.color)
        }
        if let last = log.entries.last {
            return (last.level.icon, last.message, last.level.color)
        }
        return ("checkmark.circle", "Ready", .green)
    }

    /// Contextual hint based on current app state — guides the user to the next logical step.
    private var contextualHint: String? {
        if isActive { return nil }
        if viewModel.molecules.protein == nil {
            return "Load a protein from Search or drag a PDB file"
        }
        if !viewModel.molecules.proteinPrepared && viewModel.molecules.protein != nil {
            return "Tip: Run Preparation to clean the structure"
        }
        if viewModel.molecules.ligand == nil && viewModel.molecules.protein != nil {
            return "Add a ligand from the Ligands tab to dock"
        }
        if viewModel.molecules.ligand != nil && viewModel.docking.selectedPocket == nil && viewModel.docking.dockingResults.isEmpty {
            return "Detect a binding site in the Docking tab"
        }
        if viewModel.docking.selectedPocket != nil && viewModel.docking.dockingResults.isEmpty {
            return "Ready to dock — configure and run"
        }
        if !viewModel.docking.dockingResults.isEmpty {
            let method = viewModel.docking.scoringMethod
            let score = viewModel.docking.dockingResults.first?.displayScore(method: method) ?? 0
            return String(format: "Best pose: %.2f %@ — check Results", score, method.unitLabel)
        }
        return nil
    }

    var body: some View {
        VStack(spacing: 0) {
            // Drag handle for resizing console
            if showConsole {
                Rectangle()
                    .fill(Color.clear)
                    .frame(height: 6)
                    .contentShape(Rectangle())
                    .onHover { hovering in
                        if hovering { NSCursor.resizeUpDown.push() } else { NSCursor.pop() }
                    }
                    .gesture(
                        DragGesture(minimumDistance: 1)
                            .updating($dragStartHeight) { _, state, _ in
                                if state == 0 { state = consoleHeight }
                            }
                            .onChanged { value in
                                let initial = dragStartHeight != 0 ? dragStartHeight : consoleHeight
                                consoleHeight = max(60, min(500, initial - value.translation.height))
                            }
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 1.5)
                            .fill(Color.secondary.opacity(0.4))
                            .frame(width: 36, height: 3)
                    )
                    .background(Color(nsColor: .windowBackgroundColor))
            }

            Divider()

            // Status strip — always visible
            HStack(alignment: .center, spacing: 6) {
                // Expand/collapse console
                Button(action: { withAnimation(.easeInOut(duration: 0.2)) { showConsole.toggle() } }) {
                    Image(systemName: showConsole ? "chevron.down" : "chevron.up")
                        .font(.caption2.weight(.bold))
                        .frame(width: 20, height: 20)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .help(showConsole ? "Collapse console" : "Expand console")
                .plainButtonAccessibility(AccessibilityID.statusToggleConsole)

                // Status indicator
                let status = statusInfo
                HStack(spacing: 4) {
                    if isActive {
                        ProgressView()
                            .controlSize(.mini)
                            .scaleEffect(0.7)
                            .frame(width: 14, height: 14)
                    } else {
                        Image(systemName: status.icon)
                            .font(.caption2)
                            .foregroundStyle(status.color)
                            .frame(width: 14, height: 14)
                    }

                    Text(status.message)
                        .font(.footnote.monospaced())
                        .foregroundStyle(.primary)
                        .lineLimit(1)
                }

                // Progress bar + stop button for active operations
                if isActive {
                    let progress = progressFraction
                    if progress > 0 {
                        GeometryReader { geo in
                            ZStack(alignment: .leading) {
                                RoundedRectangle(cornerRadius: 2)
                                    .fill(Color.primary.opacity(0.08))
                                RoundedRectangle(cornerRadius: 2)
                                    .fill(Color.cyan)
                                    .frame(width: geo.size.width * CGFloat(progress))
                            }
                        }
                        .frame(width: 80, height: 4)

                    }

                    Button(action: stopCurrentOperation) {
                        Image(systemName: "stop.fill")
                            .font(.caption2)
                            .foregroundStyle(.red)
                            .frame(width: 20, height: 20)
                            .background(Color.red.opacity(0.12))
                            .clipShape(RoundedRectangle(cornerRadius: 4))
                    }
                    .buttonStyle(.plain)
                    .help(stopButtonHint)
                    .plainButtonAccessibility(AccessibilityID.statusStop)
                }

                // Contextual hint (when not actively working)
                if !isActive, let hint = contextualHint {
                    Divider().frame(height: 12)
                    HStack(spacing: 4) {
                        Image(systemName: "lightbulb.fill")
                            .font(.caption2)
                            .foregroundStyle(.yellow.opacity(0.7))
                            .frame(width: 14, height: 14)
                        Text(hint)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                }

                Spacer(minLength: 4)

                if showConsole {
                    Picker("", selection: $filterLevel) {
                        Text("All").tag(nil as LogLevel?)
                        ForEach(LogLevel.allCases, id: \.self) { level in
                            Label(level.rawValue, systemImage: level.icon).tag(level as LogLevel?)
                        }
                    }
                    .pickerStyle(.menu)
                    .frame(width: 60)
                    .help("Filter console by log level")

                    if !selectedEntryIDs.isEmpty {
                        Button(action: copySelected) {
                            HStack(spacing: 4) {
                                Image(systemName: "doc.on.doc")
                                    .font(.caption2)
                                Text("\(selectedEntryIDs.count)")
                                    .font(.footnote.monospaced())
                            }
                            .frame(height: 20)
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(.secondary)
                        .help("Copy selected lines")
                        .plainButtonAccessibility(AccessibilityID.statusCopySelected)
                    }

                    statusBarButton(icon: "doc.on.clipboard", help: "Copy all console output",
                                    accessibilityID: AccessibilityID.statusCopyAll) { copyAll() }

                    statusBarButton(icon: "trash", help: "Clear all log entries",
                                    accessibilityID: AccessibilityID.statusClearLog) {
                        log.clear()
                        selectedEntryIDs.removeAll()
                        lastClickedID = nil
                    }

                    if log.currentLogFileURL != nil {
                        statusBarButton(icon: "doc.text.magnifyingglass", help: "Reveal log file in Finder",
                                        accessibilityID: AccessibilityID.statusRevealLog) { log.revealLogInFinder() }
                    }
                }

                if !showConsole {
                    let errorCount = log.entries.filter { $0.level == .error }.count
                    if errorCount > 0 {
                        HStack(spacing: 3) {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .font(.caption2)
                            Text("\(errorCount)")
                                .font(.caption2.weight(.medium).monospaced())
                        }
                        .padding(.horizontal, 4)
                        .padding(.vertical, 2)
                        .background(Color.red.opacity(0.15))
                        .foregroundStyle(.red)
                        .clipShape(Capsule())
                    }
                }

                Divider().frame(height: 12)

                // Console toggle
                Button(action: { withAnimation(.easeInOut(duration: 0.2)) { showConsole.toggle() } }) {
                    HStack(spacing: 4) {
                        Image(systemName: "terminal")
                            .font(.caption2)
                        if !showConsole {
                            Text("Console")
                                .font(.footnote)
                        }
                    }
                    .foregroundStyle(showConsole ? .primary : .secondary)
                    .padding(.horizontal, 6)
                    .frame(height: 20)
                    .background(showConsole ? Color.accentColor.opacity(0.12) : Color.clear)
                    .clipShape(RoundedRectangle(cornerRadius: 4))
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help(showConsole ? "Hide console" : "Show console")
            }
            .padding(.horizontal, 10)
            .frame(height: 26)
            .background(Color(nsColor: .windowBackgroundColor))

            // Expanded console
            if showConsole {
                Divider()
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 1) {
                            ForEach(filteredEntries) { entry in
                                logEntryRow(entry)
                                    .id(entry.id)
                            }
                        }
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                    }
                    .onChange(of: log.entries.count) { _, _ in
                        if scrollToBottom, let last = filteredEntries.last {
                            proxy.scrollTo(last.id, anchor: .bottom)
                        }
                    }
                    .onChange(of: showConsole) { _, visible in
                        if visible, let last = filteredEntries.last {
                            DispatchQueue.main.async {
                                proxy.scrollTo(last.id, anchor: .bottom)
                            }
                        }
                    }
                    .onKeyPress(.escape) {
                        selectedEntryIDs.removeAll()
                        lastClickedID = nil
                        return .handled
                    }
                }
                .frame(height: consoleHeight)
                .background(Color(nsColor: .controlBackgroundColor))
                .copyable(formatEntries(selectedEntryIDs.isEmpty ? filteredEntries : selectedEntries))
            }
        }
    }

    // MARK: - Status Bar Button

    private func statusBarButton(icon: String, help: String, accessibilityID: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: icon)
                .font(.caption2)
                .frame(width: 20, height: 20)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .foregroundStyle(.secondary)
        .help(help)
        .plainButtonAccessibility(accessibilityID)
    }

    // MARK: - Log Entry Row

    @ViewBuilder
    private func logEntryRow(_ entry: LogEntry) -> some View {
        let isSelected = selectedEntryIDs.contains(entry.id)
        HStack(alignment: .top, spacing: 8) {
            Text(entry.formattedTime)
                .font(.footnote.monospaced())
                .foregroundStyle(.secondary)
                .frame(width: 80, alignment: .leading)
            Image(systemName: entry.level.icon)
                .font(.footnote)
                .foregroundStyle(entry.level.color)
                .frame(width: 14)
            Text(entry.category.rawValue)
                .font(.footnote.weight(.medium).monospaced())
                .padding(.horizontal, 4)
                .padding(.vertical, 1)
                .background(Color.accentColor.opacity(0.15))
                .clipShape(RoundedRectangle(cornerRadius: 4))
            Text(entry.message)
                .font(.subheadline.monospaced())
                .foregroundStyle(.primary)
                .lineLimit(2)
                .textSelection(.enabled)
            Spacer()
        }
        .padding(.vertical, 3)
        .padding(.horizontal, 4)
        .background(isSelected ? Color.accentColor.opacity(0.18) : Color.clear)
        .clipShape(RoundedRectangle(cornerRadius: 4))
        .contentShape(Rectangle())
        .overlay(alignment: .bottom) {
            Divider().padding(.leading, 8)
        }
        .onTapGesture {
            handleEntryClick(entry, shiftKey: NSEvent.modifierFlags.contains(.shift))
        }
    }

    // MARK: - Selection & Copy

    private var selectedEntries: [LogEntry] {
        let entries = filteredEntries
        return entries.filter { selectedEntryIDs.contains($0.id) }
    }

    private func handleEntryClick(_ entry: LogEntry, shiftKey: Bool) {
        let entries = filteredEntries
        if shiftKey, let anchorID = lastClickedID,
           let anchorIdx = entries.firstIndex(where: { $0.id == anchorID }),
           let clickIdx = entries.firstIndex(where: { $0.id == entry.id }) {
            // Shift-click: select range
            let range = min(anchorIdx, clickIdx)...max(anchorIdx, clickIdx)
            selectedEntryIDs = Set(entries[range].map(\.id))
        } else {
            // Single click: toggle or select
            if selectedEntryIDs.contains(entry.id) && selectedEntryIDs.count == 1 {
                selectedEntryIDs.removeAll()
                lastClickedID = nil
            } else {
                selectedEntryIDs = [entry.id]
                lastClickedID = entry.id
            }
        }
    }

    private func formatEntries(_ entries: [LogEntry]) -> [String] {
        let text = entries.map { entry in
            "\(entry.formattedTime)  [\(entry.level.rawValue)]  [\(entry.category.rawValue)]  \(entry.message)"
        }.joined(separator: "\n")
        return [text]
    }

    private func copySelected() {
        let text = formatEntries(selectedEntries).first ?? ""
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }

    private func copyAll() {
        let text = formatEntries(filteredEntries).first ?? ""
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }

    private func saveProject() {
        let panel = NSSavePanel()
        panel.title = "Save Druse Project"
        panel.allowedContentTypes = [.folder]
        panel.nameFieldStringValue = "\(viewModel.molecules.protein?.name ?? "Untitled").druse"
        panel.canCreateDirectories = true
        panel.begin { response in
            guard response == .OK, let url = panel.url else { return }
            Task { @MainActor in
                do {
                    try DruseProjectIO.save(to: url, viewModel: viewModel)
                } catch {
                    viewModel.log.error("Save failed: \(error.localizedDescription)", category: .system)
                }
            }
        }
    }

    // MARK: - Stop Operations

    private var stopButtonHint: String {
        if viewModel.docking.isBatchDocking { return "Stop batch docking" }
        if viewModel.docking.isDocking { return "Stop docking" }
        if isScreeningActive { return "Stop virtual screening" }
        if viewModel.leadOpt.isGenerating || viewModel.leadOpt.isDocking { return "Stop lead optimization" }
        if viewModel.molecules.isMinimizing { return "Stop structure cleanup" }
        if viewModel.workspace.isGeneratingSurface { return "Stop surface generation" }
        if viewModel.workspace.isLoading { return "Cancel loading" }
        if viewModel.workspace.statusMessage.hasSuffix("...") { return "Cancel current operation" }
        return "Stop"
    }

    private func stopCurrentOperation() {
        if viewModel.docking.isBatchDocking {
            viewModel.cancelBatchDocking()
        } else if viewModel.docking.isDocking {
            viewModel.stopDocking()
        } else if isScreeningActive {
            viewModel.cancelScreening()
        } else if viewModel.leadOpt.isGenerating || viewModel.leadOpt.isDocking {
            viewModel.cancelLeadOptimization()
        } else if viewModel.molecules.isMinimizing {
            viewModel.molecules.isMinimizing = false
        } else if viewModel.workspace.isGeneratingSurface {
            viewModel.workspace.isGeneratingSurface = false
        } else if viewModel.workspace.isLoading {
            viewModel.workspace.isLoading = false
        }
        // Reset status for operations tracked only by statusMessage
        if viewModel.workspace.statusMessage.hasSuffix("...") {
            viewModel.workspace.statusMessage = "Cancelled"
        }
    }

    private func openProject() {
        let panel = NSOpenPanel()
        panel.title = "Open Druse Project"
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.begin { response in
            guard response == .OK, let url = panel.url else { return }
            Task { @MainActor in
                do {
                    try await DruseProjectIO.load(from: url, into: viewModel)
                } catch {
                    viewModel.log.error("Load failed: \(error.localizedDescription)", category: .system)
                }
            }
        }
    }
}
