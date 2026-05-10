// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

// MARK: - Layout constants

enum PanelStyle {
    static let cardCornerRadius: CGFloat = 10
    static let cardPadding: CGFloat = 12
    static let cardSpacing: CGFloat = 10
    static let sectionSpacing: CGFloat = 12
    static let rowSpacing: CGFloat = 6
    static let labelColumnWidth: CGFloat = 88
    static let valueColumnWidth: CGFloat = 60
    static let numberFieldWidth: CGFloat = 64
    static let buttonHeight: CGFloat = 24
    static let runButtonHeight: CGFloat = 32
    static let cardFillOpacity: Double = 0.045
    static let chipFillOpacity: Double = 0.06

    static let titleFont: Font = .system(size: 13, weight: .semibold)
    static let bodyFont: Font = .system(size: 12)
    static let smallFont: Font = .system(size: 11)
    static let captionFont: Font = .system(size: 10)
    static let monoSmall: Font = .system(size: 11, design: .monospaced)
    static let monoBody: Font = .system(size: 12, design: .monospaced)
}

// MARK: - Card

struct PanelCard<Content: View>: View {
    let title: String
    let icon: String
    let accessory: AnyView?
    @ViewBuilder let content: () -> Content

    init(_ title: String, icon: String,
         @ViewBuilder content: @escaping () -> Content) {
        self.title = title
        self.icon = icon
        self.accessory = nil
        self.content = content
    }

    init<A: View>(_ title: String, icon: String,
                  @ViewBuilder accessory: () -> A,
                  @ViewBuilder content: @escaping () -> Content) {
        self.title = title
        self.icon = icon
        self.accessory = AnyView(accessory())
        self.content = content
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 6) {
                Image(systemName: icon)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.secondary)
                    .frame(width: 16, height: 16)
                Text(title)
                    .font(PanelStyle.titleFont)
                Spacer(minLength: 4)
                if let accessory { accessory }
            }
            content()
        }
        .padding(PanelStyle.cardPadding)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: PanelStyle.cardCornerRadius, style: .continuous)
                .fill(Color.primary.opacity(PanelStyle.cardFillOpacity))
        )
    }
}

// MARK: - Subheader (uppercase caption used inside a card)

struct PanelSubheader: View {
    let title: String
    var icon: String? = nil
    var body: some View {
        HStack(spacing: 4) {
            if let icon {
                Image(systemName: icon).font(.caption2)
            }
            Text(title)
                .font(.caption.weight(.semibold))
                .textCase(.uppercase)
        }
        .foregroundStyle(.secondary)
    }
}

// MARK: - Slider row

struct PanelSliderRow: View {
    let label: String
    @Binding var value: Float
    let range: ClosedRange<Float>
    var step: Float = 0.01
    var format: (Float) -> String = { String(format: "%.2f", $0) }
    var labelWidth: CGFloat = PanelStyle.labelColumnWidth
    var valueWidth: CGFloat = PanelStyle.valueColumnWidth
    var valueColor: Color = .secondary
    var onChange: ((Float) -> Void)? = nil

    var body: some View {
        HStack(spacing: 8) {
            Text(label)
                .font(PanelStyle.smallFont)
                .foregroundStyle(.secondary)
                .frame(width: labelWidth, alignment: .leading)
                .lineLimit(1)
            Slider(value: $value, in: range, step: step)
                .controlSize(.mini)
                .onChange(of: value) { _, newValue in onChange?(newValue) }
            Text(format(value))
                .font(PanelStyle.monoSmall)
                .foregroundStyle(valueColor)
                .frame(width: valueWidth, alignment: .trailing)
        }
    }
}

// MARK: - Single-axis slider (XYZ grid rows)

struct PanelAxisSlider: View {
    let axis: String
    @Binding var value: Float
    let range: ClosedRange<Float>
    var step: Float = 0.5
    var format: (Float) -> String = { String(format: "%.1f Å", $0) }
    var onChange: ((Float) -> Void)? = nil

    var body: some View {
        HStack(spacing: 6) {
            Text(axis)
                .font(.system(size: 10, weight: .bold, design: .monospaced))
                .foregroundStyle(.secondary)
                .frame(width: 10, alignment: .trailing)
            Slider(value: $value, in: range, step: step)
                .controlSize(.mini)
                .onChange(of: value) { _, newValue in onChange?(newValue) }
            Text(format(value))
                .font(PanelStyle.monoSmall)
                .foregroundStyle(.secondary)
                .frame(width: 56, alignment: .trailing)
        }
    }
}

// MARK: - Number field

struct PanelNumberField: View {
    @Binding var value: Int
    var width: CGFloat = PanelStyle.numberFieldWidth
    var minimum: Int = 1

    var body: some View {
        TextField("", value: Binding(
            get: { value },
            set: { value = max(minimum, $0) }
        ), format: .number)
            .textFieldStyle(.roundedBorder)
            .controlSize(.small)
            .font(PanelStyle.monoSmall)
            .multilineTextAlignment(.trailing)
            .frame(width: width)
    }
}

// MARK: - LabeledRow (label left, content right)

struct PanelLabeledRow<Content: View>: View {
    let label: String
    var help: String? = nil
    @ViewBuilder let content: () -> Content

    init(_ label: String, help: String? = nil,
         @ViewBuilder content: @escaping () -> Content) {
        self.label = label
        self.help = help
        self.content = content
    }

    var body: some View {
        HStack(spacing: 8) {
            Text(label)
                .font(PanelStyle.bodyFont)
            Spacer(minLength: 4)
            content()
        }
        .help(help ?? "")
    }
}

// MARK: - Choice button (segment-style)

struct PanelChoiceButton: View {
    let title: String
    var icon: String? = nil
    var badge: String? = nil
    let isSelected: Bool
    var isDisabled: Bool = false
    var help: String? = nil
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 4) {
                if let icon {
                    Image(systemName: icon)
                        .font(.system(size: 11, weight: .medium))
                }
                Text(title)
                    .font(.system(size: 12, weight: isSelected ? .semibold : .regular))
                    .lineLimit(1)
                if let badge {
                    Text(badge)
                        .font(.system(size: 8, weight: .bold))
                        .foregroundStyle(.orange)
                        .padding(.horizontal, 3)
                        .padding(.vertical, 1)
                        .background(Color.orange.opacity(0.15))
                        .clipShape(RoundedRectangle(cornerRadius: 2))
                }
            }
            .frame(maxWidth: .infinity)
            .frame(height: PanelStyle.buttonHeight)
            .background(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill(isSelected
                          ? Color.accentColor.opacity(0.18)
                          : Color.primary.opacity(PanelStyle.chipFillOpacity))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .strokeBorder(isSelected ? Color.accentColor.opacity(0.55) : Color.clear,
                                  lineWidth: 1)
            )
            .foregroundStyle(isSelected ? Color.accentColor : Color.primary)
            .opacity(isDisabled ? 0.4 : 1.0)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(isDisabled)
        .help(help ?? "")
    }
}

// MARK: - Choice grid wrapper

struct PanelChoiceGrid<Content: View>: View {
    var columns: Int = 2
    var spacing: CGFloat = 6
    @ViewBuilder let content: () -> Content

    var body: some View {
        LazyVGrid(
            columns: Array(repeating: GridItem(.flexible(), spacing: spacing), count: columns),
            spacing: spacing
        ) {
            content()
        }
    }
}

// MARK: - Toggle row

struct PanelToggleRow: View {
    let title: String
    var subtitle: String? = nil
    var icon: String? = nil
    @Binding var isOn: Bool

    var body: some View {
        HStack(alignment: .center, spacing: 8) {
            if let icon {
                Image(systemName: icon)
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                    .frame(width: 14)
            }
            VStack(alignment: .leading, spacing: 1) {
                Text(title)
                    .font(PanelStyle.bodyFont)
                if let subtitle {
                    Text(subtitle)
                        .font(PanelStyle.smallFont)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer()
            Toggle("", isOn: $isOn)
                .toggleStyle(.switch)
                .controlSize(.mini)
                .labelsHidden()
        }
    }
}

// MARK: - Hint / inline message

struct PanelHint: View {
    let text: String
    var icon: String? = nil
    var color: Color = .secondary
    var body: some View {
        HStack(alignment: .top, spacing: 4) {
            if let icon {
                Image(systemName: icon)
                    .font(.system(size: 10))
                    .foregroundStyle(color)
                    .padding(.top, 1)
            }
            Text(text)
                .font(PanelStyle.smallFont)
                .foregroundStyle(color)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

// MARK: - Stat cell (dashboard-style number + caption)

struct PanelStat: View {
    let label: String
    let value: String
    var unit: String? = nil
    var color: Color = .primary

    var body: some View {
        VStack(spacing: 2) {
            HStack(alignment: .firstTextBaseline, spacing: 2) {
                Text(value)
                    .font(.system(size: 13, weight: .medium, design: .monospaced))
                    .foregroundStyle(color)
                if let unit {
                    Text(unit)
                        .font(PanelStyle.captionFont)
                        .foregroundStyle(.secondary)
                }
            }
            Text(label)
                .font(PanelStyle.captionFont)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - Run button (primary action)

struct PanelRunButton: View {
    let title: String
    var icon: String = "play.fill"
    var color: Color = .accentColor
    var isDisabled: Bool = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 6) {
                Image(systemName: icon)
                    .font(.system(size: 12, weight: .semibold))
                Text(title)
                    .font(.system(size: 13, weight: .semibold))
            }
            .frame(maxWidth: .infinity)
            .frame(height: PanelStyle.runButtonHeight)
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(color.opacity(isDisabled ? 0.35 : 1.0))
            )
            .foregroundStyle(.white)
        }
        .buttonStyle(.plain)
        .disabled(isDisabled)
    }
}

// MARK: - Secondary button (compact, neutral)

struct PanelSecondaryButton: View {
    let title: String
    var icon: String? = nil
    var tint: Color? = nil
    var isDisabled: Bool = false
    var help: String? = nil
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 4) {
                if let icon {
                    Image(systemName: icon)
                        .font(.system(size: 11, weight: .medium))
                }
                Text(title)
                    .font(PanelStyle.bodyFont)
                    .lineLimit(1)
            }
            .frame(maxWidth: .infinity)
            .frame(height: PanelStyle.buttonHeight)
            .background(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill(Color.primary.opacity(PanelStyle.chipFillOpacity))
            )
            .foregroundStyle(tint ?? .primary)
            .opacity(isDisabled ? 0.4 : 1.0)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(isDisabled)
        .help(help ?? "")
    }
}

// MARK: - Labeled divider (uppercase label with thin horizontal line)

/// Thin horizontal rule with an uppercase caption label on the left.
/// Used to separate non-essential / advanced sections inside a card without
/// hiding them behind a disclosure.
struct PanelLabeledDivider: View {
    let title: String
    var icon: String? = nil

    var body: some View {
        HStack(spacing: 6) {
            if let icon {
                Image(systemName: icon).font(.caption2)
            }
            Text(title)
                .font(.caption.weight(.semibold))
                .textCase(.uppercase)
            Rectangle()
                .fill(Color.secondary.opacity(0.25))
                .frame(height: 0.5)
        }
        .foregroundStyle(.secondary)
        .padding(.top, 2)
    }
}

// MARK: - Disclosure section (collapsible group inside a card)

struct PanelDisclosure<Content: View>: View {
    let title: String
    var icon: String? = nil
    @Binding var isExpanded: Bool
    @ViewBuilder let content: () -> Content

    init(_ title: String,
         icon: String? = nil,
         isExpanded: Binding<Bool>,
         @ViewBuilder content: @escaping () -> Content) {
        self.title = title
        self.icon = icon
        self._isExpanded = isExpanded
        self.content = content
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button(action: {
                withAnimation(.easeInOut(duration: 0.15)) { isExpanded.toggle() }
            }) {
                HStack(spacing: 4) {
                    Image(systemName: "chevron.right")
                        .font(.system(size: 9, weight: .semibold))
                        .rotationEffect(.degrees(isExpanded ? 90 : 0))
                    if let icon {
                        Image(systemName: icon).font(.caption2)
                    }
                    Text(title)
                        .font(.caption.weight(.semibold))
                        .textCase(.uppercase)
                    Spacer()
                }
                .foregroundStyle(.secondary)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            if isExpanded {
                content()
            }
        }
    }
}

// MARK: - Highlight box (used for batch ligand info, single-ligand chip, etc.)

struct PanelHighlightRow<Content: View>: View {
    let color: Color
    @ViewBuilder let content: () -> Content

    var body: some View {
        content()
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill(color.opacity(0.10))
            )
    }
}

// MARK: - Tag chip (residue names, etc.)

struct PanelChip: View {
    let text: String
    var color: Color = .accentColor
    var body: some View {
        Text(text)
            .font(.system(size: 11, weight: .medium, design: .monospaced))
            .foregroundStyle(color)
            .padding(.horizontal, 5)
            .padding(.vertical, 2)
            .background(
                RoundedRectangle(cornerRadius: 4, style: .continuous)
                    .fill(color.opacity(0.14))
            )
    }
}

// MARK: - Requirement label (red x + text)

struct PanelRequirement: View {
    let text: String
    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: "xmark.circle.fill")
                .font(.system(size: 11))
                .foregroundStyle(.red.opacity(0.75))
            Text(text)
                .font(PanelStyle.smallFont)
                .foregroundStyle(.secondary)
        }
    }
}
