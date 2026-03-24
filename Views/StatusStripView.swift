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
            let best = viewModel.docking.dockingResults.first
            if viewModel.docking.scoringMethod == .druseAffinity, let pKd = best?.mlPKd {
                let display = viewModel.docking.affinityDisplayUnit.format(pKd)
                let unit = viewModel.docking.affinityDisplayUnit.unitLabel
                return "Best pose: \(display) \(unit) — check Results"
            }
            let energy = best?.energy ?? 0
            return String(format: "Best pose: %.2f kcal/mol — check Results", energy)
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
                    .background(.ultraThinMaterial)
            }

            Divider()

            // Status strip — always visible
            HStack(spacing: 6) {
                // Expand/collapse console
                Button(action: { withAnimation(.easeInOut(duration: 0.2)) { showConsole.toggle() } }) {
                    Image(systemName: showConsole ? "chevron.down" : "chevron.up")
                        .font(.system(size: 9, weight: .bold))
                        .frame(width: 16, height: 16)
                }
                .buttonStyle(.plain)
                .foregroundStyle(.tertiary)

                // Status indicator
                let status = statusInfo
                HStack(spacing: 5) {
                    if isActive {
                        ProgressView()
                            .controlSize(.mini)
                            .scaleEffect(0.7)
                    } else {
                        Image(systemName: status.icon)
                            .font(.system(size: 10))
                            .foregroundStyle(status.color)
                    }

                    Text(status.message)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(isActive ? .primary : .secondary)
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
                            .font(.system(size: 8))
                            .foregroundStyle(.red)
                            .frame(width: 16, height: 16)
                            .background(Color.red.opacity(0.12))
                            .clipShape(RoundedRectangle(cornerRadius: 3))
                    }
                    .buttonStyle(.plain)
                    .help(stopButtonHint)
                }

                // Contextual hint (when not actively working)
                if !isActive, let hint = contextualHint {
                    Divider().frame(height: 14)
                    HStack(spacing: 3) {
                        Image(systemName: "lightbulb.fill")
                            .font(.system(size: 8))
                            .foregroundStyle(.yellow.opacity(0.7))
                        Text(hint)
                            .font(.system(size: 10))
                            .foregroundStyle(.tertiary)
                            .lineLimit(1)
                    }
                }

                Spacer()

                // Project save/load
                Button(action: { saveProject() }) {
                    Image(systemName: "square.and.arrow.down")
                        .font(.system(size: 10))
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .help("Save Project (Cmd+S)")

                Button(action: { openProject() }) {
                    Image(systemName: "folder")
                        .font(.system(size: 10))
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .help("Open Project (Cmd+Shift+O)")

                Divider().frame(height: 14)

                if showConsole {
                    Picker("", selection: $filterLevel) {
                        Text("All").tag(nil as LogLevel?)
                        ForEach(LogLevel.allCases, id: \.self) { level in
                            Label(level.rawValue, systemImage: level.icon).tag(level as LogLevel?)
                        }
                    }
                    .pickerStyle(.menu)
                    .frame(width: 100)

                    if !selectedEntryIDs.isEmpty {
                        Button(action: copySelected) {
                            HStack(spacing: 3) {
                                Image(systemName: "doc.on.doc")
                                    .font(.system(size: 9))
                                Text("\(selectedEntryIDs.count)")
                                    .font(.system(size: 9, design: .monospaced))
                            }
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(.secondary)
                        .help("Copy selected lines")
                    }

                    Button(action: copyAll) {
                        Image(systemName: "doc.on.clipboard")
                            .font(.system(size: 10))
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)
                    .help("Copy all console output")

                    Button(action: {
                        log.clear()
                        selectedEntryIDs.removeAll()
                        lastClickedID = nil
                    }) {
                        Image(systemName: "trash")
                            .font(.system(size: 10))
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)

                    if log.currentLogFileURL != nil {
                        Divider().frame(height: 10)

                        Button(action: { log.revealLogInFinder() }) {
                            Image(systemName: "doc.text.magnifyingglass")
                                .font(.system(size: 10))
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(.secondary)
                        .help("Reveal log file in Finder")
                    }
                }

                if !showConsole {
                    let errorCount = log.entries.filter { $0.level == .error }.count
                    if errorCount > 0 {
                        HStack(spacing: 3) {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .font(.system(size: 9))
                            Text("\(errorCount)")
                                .font(.system(size: 9, weight: .medium, design: .monospaced))
                        }
                        .padding(.horizontal, 5)
                        .padding(.vertical, 2)
                        .background(Color.red.opacity(0.15))
                        .foregroundStyle(.red)
                        .clipShape(Capsule())
                    }
                }

                // System metrics removed (caused performance interference)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 4)
            .frame(height: 26)
            .background(.ultraThinMaterial)

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
                    .onKeyPress(.escape) {
                        selectedEntryIDs.removeAll()
                        lastClickedID = nil
                        return .handled
                    }
                }
                .frame(height: consoleHeight)
                .background(Color(nsColor: .controlBackgroundColor).opacity(0.3))
                .copyable(formatEntries(selectedEntryIDs.isEmpty ? filteredEntries : selectedEntries))
            }
        }
    }

    // MARK: - Log Entry Row

    @ViewBuilder
    private func logEntryRow(_ entry: LogEntry) -> some View {
        let isSelected = selectedEntryIDs.contains(entry.id)
        HStack(alignment: .top, spacing: 6) {
            Text(entry.formattedTime)
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(.tertiary)
                .frame(width: 80, alignment: .leading)
            Image(systemName: entry.level.icon)
                .font(.system(size: 10))
                .foregroundStyle(entry.level.color)
                .frame(width: 14)
            Text(entry.category.rawValue)
                .font(.system(size: 9, weight: .medium, design: .monospaced))
                .padding(.horizontal, 4)
                .padding(.vertical, 1)
                .background(Color.accentColor.opacity(0.15))
                .clipShape(RoundedRectangle(cornerRadius: 3))
            Text(entry.message)
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(.primary)
                .lineLimit(2)
                .textSelection(.enabled)
            Spacer()
        }
        .padding(.vertical, 2)
        .padding(.horizontal, 4)
        .background(isSelected ? Color.accentColor.opacity(0.18) : Color.clear)
        .clipShape(RoundedRectangle(cornerRadius: 3))
        .contentShape(Rectangle())
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
