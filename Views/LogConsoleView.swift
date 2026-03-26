import SwiftUI

struct LogConsoleView: View {
    @Binding var isExpanded: Bool
    @Binding var consoleHeight: CGFloat
    let log = ActivityLog.shared
    @State private var filterLevel: LogLevel? = nil
    @State private var scrollToBottom = true
    @GestureState private var dragStartHeight: CGFloat = 0

    var filteredEntries: [LogEntry] {
        if let level = filterLevel {
            return log.entries.filter { $0.level == level }
        }
        return log.entries
    }

    var body: some View {
        VStack(spacing: 0) {
            // Drag handle for resizing console
            if isExpanded {
                Rectangle()
                    .fill(Color.clear)
                    .frame(height: 6)
                    .contentShape(Rectangle())
                    .onHover { hovering in
                        if hovering {
                            NSCursor.resizeUpDown.push()
                        } else {
                            NSCursor.pop()
                        }
                    }
                    .gesture(
                        DragGesture(minimumDistance: 1)
                            .updating($dragStartHeight) { _, state, _ in
                                if state == 0 { state = consoleHeight }
                            }
                            .onChanged { value in
                                let initial = dragStartHeight != 0 ? dragStartHeight : consoleHeight
                                let maxH = (NSScreen.main?.visibleFrame.height ?? 800) * 0.6
                                consoleHeight = max(60, min(maxH, initial - value.translation.height))
                            }
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 1.5)
                            .fill(Color.secondary.opacity(0.4))
                            .frame(width: 36, height: 3)
                    )
                    .background(Color(nsColor: .windowBackgroundColor))
            }

            // Toggle bar
            HStack(spacing: 8) {
                Button(action: { withAnimation(.easeInOut(duration: 0.2)) { isExpanded.toggle() } }) {
                    Image(systemName: isExpanded ? "chevron.down" : "chevron.up")
                        .font(.footnote.weight(.bold))
                }
                .buttonStyle(.plain)

                Image(systemName: "terminal.fill")
                    .font(.subheadline)
                    .foregroundStyle(.primary)

                Text("Console")
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(.primary)

                Text("\(log.entries.count)")
                    .font(.footnote.weight(.medium).monospaced())
                    .padding(.horizontal, 4)
                    .padding(.vertical, 1)
                    .background(.quaternary)
                    .clipShape(Capsule())

                Spacer()

                if isExpanded {
                    // Level filter
                    Picker("", selection: $filterLevel) {
                        Text("All").tag(nil as LogLevel?)
                        ForEach(LogLevel.allCases, id: \.self) { level in
                            Label(level.rawValue, systemImage: level.icon).tag(level as LogLevel?)
                        }
                    }
                    .pickerStyle(.menu)
                    .frame(width: 100)

                    Button(action: { log.clear() }) {
                        Image(systemName: "trash")
                            .font(.subheadline)
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)
                }

                if !isExpanded, let last = log.entries.last {
                    HStack(spacing: 4) {
                        Image(systemName: last.level.icon)
                            .foregroundStyle(last.level.color)
                            .font(.footnote)
                        Text(last.message)
                            .font(.subheadline.monospaced())
                            .foregroundStyle(.primary)
                            .lineLimit(1)
                    }
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(Color(nsColor: .windowBackgroundColor))

            Divider()

            if isExpanded {
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
                }
                .frame(height: consoleHeight)
                .background(Color(nsColor: .controlBackgroundColor))
            }
        }
    }

    @ViewBuilder
    private func logEntryRow(_ entry: LogEntry) -> some View {
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

            Spacer()
        }
        .padding(.vertical, 2)
    }
}
