// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import AppKit
import SwiftUI

extension ScriptHighlightKind {
    /// Colour of the span in the command line and in echoed commands (nil = terminal foreground).
    @MainActor var nsColor: NSColor? {
        switch self {
        case .verb:        DS.nsColor(DS.Color.accent)
        case .argument:    nil
        case .optionKey:   DS.nsColor(DS.Color.pocket)
        case .optionValue: DS.nsColor(DS.Color.ligand)
        case .string:      DS.nsColor(DS.Color.success)
        case .variable:    DS.nsColor(DS.Color.info)
        case .comment:     TerminalStyle.dim
        case .punctuation: TerminalStyle.dim
        }
    }
}

/// Fonts and colours of the terminal: a light and a dark theme built from DS tokens, with the
/// DS status colours standing in for the ANSI green / yellow / red / blue a shell would use.
@MainActor
enum TerminalStyle {
    static let font = NSFont.monospacedSystemFont(ofSize: NSFont.preferredFont(forTextStyle: .callout).pointSize, weight: .regular)
    static let boldFont = NSFont.monospacedSystemFont(ofSize: font.pointSize, weight: .semibold)

    static let background = DS.nsColor(DS.dynamic(light: 0xFBFBFA, dark: 0x111417))
    static let foreground = DS.nsColor(DS.dynamic(light: 0x1F2328, dark: 0xE6EDF3))
    /// Secondary text: banner, `^C`, comments, ghost suggestions, the ⌃R search label.
    static let dim = DS.nsColor(DS.dynamic(light: 0x6E7781, dark: 0x8B949E))
    static let prompt = DS.nsColor(DS.Color.accent)
    static let success = DS.nsColor(DS.Color.success)
    static let warning = DS.nsColor(DS.Color.warning)
    static let error = DS.nsColor(DS.Color.danger)
    static let progress = DS.nsColor(DS.Color.info)
    /// Surface of the completion list floating over the scrollback, one step above the background.
    static let popup = DS.nsColor(DS.dynamic(light: 0xFFFFFF, dark: 0x1B2026))

    static var base: [NSAttributedString.Key: Any] { [.font: font, .foregroundColor: foreground] }

    /// Space above each command line, so every command and its output read as one block.
    static let commandSpacing: CGFloat = DS.Space.s

    /// The text box of the line holding `glyph`, without the paragraph spacing above it: from the
    /// font's ascender above the baseline to its descender below.
    static func textLineRect(of glyph: Int, in layoutManager: NSLayoutManager) -> NSRect {
        let fragment = layoutManager.lineFragmentRect(forGlyphAt: glyph, effectiveRange: nil)
        let baseline = fragment.minY + layoutManager.location(forGlyphAt: glyph).y
        return NSRect(x: fragment.minX, y: baseline - font.ascender, width: fragment.width,
                      height: font.ascender - font.descender)
    }

    /// Width of `druse› `, the indent of continuation lines of a multi-line command.
    static let promptWidth = ceil((ConsoleSession.prompt + " " as NSString).size(withAttributes: [.font: boldFont]).width)
}

/// Where the word being completed sits on screen (terminal view coordinates, top-left origin):
/// the completion list is drawn against it, like an IDE or shell completion pager.
@Observable
@MainActor
final class TerminalCaretAnchor {
    /// The caret's line, starting at the word being completed; nil while no prompt is shown.
    var wordLine: CGRect?
}

/// What the last line of the terminal shows, below the scrollback.
enum TerminalLiveLine: Equatable {
    /// The prompt and the command line being edited, with the ghost suggestion after it.
    case prompt(input: String, suggestion: String)
    /// A command is running: its live progress line, or an empty line holding the cursor.
    case running(progress: String?)
    /// ⌃R reverse search in place of the prompt.
    case search(ConsoleHistorySearch)

    @MainActor
    init(_ session: ConsoleSession) {
        if session.isRunning {
            self = .running(progress: session.progressText)
        } else if let search = session.historySearch {
            self = .search(search)
        } else {
            self = .prompt(input: session.input, suggestion: session.suggestion)
        }
    }
}

/// The terminal: one continuous, selectable, monospaced `NSTextView` holding the scrollback
/// followed by the live line (prompt and input inline, a progress line while a command runs,
/// or the ⌃R search). Only the input after the prompt is editable.
///
/// An AppKit text view is used rather than SwiftUI text because a terminal needs what only it
/// gives together: selection and copy across the whole scrollback, a real blinking caret inline
/// at the end of the text, text drawn after the caret (fish-style suggestions), per-key command
/// routing, and incremental appends to a long scrollback without re-laying out every line.
/// Lines wrap at the panel width, as in a terminal emulator.
struct TerminalTextView: NSViewRepresentable {
    let session: ConsoleSession
    /// Passed by value (with `clearCount` and `live`) so SwiftUI updates the view when they change.
    let transcript: [ConsoleEntry]
    let clearCount: Int
    let live: TerminalLiveLine
    let anchor: TerminalCaretAnchor

    func makeCoordinator() -> Coordinator { Coordinator(session: session, anchor: anchor) }

    func makeNSView(context: Context) -> NSScrollView {
        let textView = TerminalNSTextView(usingTextLayoutManager: false)
        textView.delegate = context.coordinator
        textView.coordinator = context.coordinator
        textView.font = TerminalStyle.font
        textView.textColor = TerminalStyle.foreground
        textView.backgroundColor = TerminalStyle.background
        textView.insertionPointColor = TerminalStyle.prompt
        textView.drawsBackground = true
        textView.isRichText = true
        textView.importsGraphics = false
        textView.allowsUndo = false
        textView.usesFontPanel = false
        textView.usesFindBar = true
        textView.isIncrementalSearchingEnabled = true
        textView.isAutomaticQuoteSubstitutionEnabled = false
        textView.isAutomaticDashSubstitutionEnabled = false
        textView.isAutomaticTextReplacementEnabled = false
        textView.isAutomaticSpellingCorrectionEnabled = false
        textView.isAutomaticLinkDetectionEnabled = false
        textView.isContinuousSpellCheckingEnabled = false
        textView.isGrammarCheckingEnabled = false
        textView.smartInsertDeleteEnabled = false
        textView.textContainerInset = NSSize(width: DS.Space.l, height: DS.Space.m)
        textView.layoutManager?.allowsNonContiguousLayout = true
        // Wrap at the view width: the container follows the view, which follows the scroll view.
        textView.isHorizontallyResizable = false
        textView.isVerticallyResizable = true
        textView.minSize = .zero
        textView.maxSize = NSSize(width: CGFloat.greatestFiniteMagnitude, height: .greatestFiniteMagnitude)
        textView.autoresizingMask = [.width]
        textView.textContainer?.widthTracksTextView = true
        textView.textContainer?.containerSize = NSSize(width: 0, height: CGFloat.greatestFiniteMagnitude)
        textView.typingAttributes = TerminalStyle.base
        textView.setAccessibilityIdentifier("console.terminal")
        textView.setAccessibilityLabel("Terminal")

        let scrollView = TerminalScrollView()
        scrollView.drawsBackground = true
        scrollView.backgroundColor = TerminalStyle.background
        scrollView.hasVerticalScroller = true
        scrollView.hasHorizontalScroller = false
        scrollView.horizontalScrollElasticity = .none
        scrollView.autohidesScrollers = true
        scrollView.borderType = .noBorder
        scrollView.documentView = textView
        scrollView.contentView.postsBoundsChangedNotifications = true
        context.coordinator.textView = textView
        context.coordinator.observeScrolling(of: scrollView)
        return scrollView
    }

    func updateNSView(_ scrollView: NSScrollView, context: Context) {
        context.coordinator.session = session
        context.coordinator.render(transcript: transcript, clearCount: clearCount, live: live)
    }

    /// Keeps the text view in step with the session (incremental appends, live-line updates)
    /// and routes typing and key commands from the text view back to the session.
    @MainActor
    final class Coordinator: NSObject, NSTextViewDelegate {
        var session: ConsoleSession
        weak var textView: TerminalNSTextView?
        private let anchor: TerminalCaretAnchor
        private var anchorUpdatePending = false

        /// Scrollback rendered so far: ids and lengths (UTF-16) of the entries, in order.
        private var renderedIDs: [UUID] = []
        private var renderedLengths: [Int] = []
        /// Length of the rendered scrollback; the live line starts here.
        private var committedLength = 0
        private var renderedClearCount = -1
        private var renderedLive: TerminalLiveLine?

        init(session: ConsoleSession, anchor: TerminalCaretAnchor) {
            self.session = session
            self.anchor = anchor
        }

        /// Scrolling and resizing move the prompt, so the completion list's anchor follows them.
        func observeScrolling(of scrollView: NSScrollView) {
            NotificationCenter.default.addObserver(self, selector: #selector(viewportChanged),
                                                   name: NSView.boundsDidChangeNotification, object: scrollView.contentView)
            NotificationCenter.default.addObserver(self, selector: #selector(viewportChanged),
                                                   name: NSView.frameDidChangeNotification, object: scrollView.contentView)
        }

        @objc private func viewportChanged(_ notification: Notification) {
            scheduleAnchorUpdate()
        }

        /// Recomputes the anchor after the current layout pass (it is observed by SwiftUI, which
        /// must not be mutated while it updates the view).
        private func scheduleAnchorUpdate() {
            guard !anchorUpdatePending else { return }
            anchorUpdatePending = true
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                anchorUpdatePending = false
                let line = wordLineRect()
                if anchor.wordLine != line { anchor.wordLine = line }
            }
        }

        /// The caret's line fragment, starting at the first character of the word being typed,
        /// in the scroll view's top-left coordinates.
        private func wordLineRect() -> CGRect? {
            guard case .prompt = renderedLive, let textView, let scrollView = textView.enclosingScrollView,
                  let layoutManager = textView.layoutManager, let container = textView.textContainer,
                  let storage = textView.textStorage, storage.length > 0 else { return nil }
            let length = storage.length
            let start = inputStart
            let input = (storage.string as NSString).substring(from: min(start, length)) as NSString
            let lastBreak = input.rangeOfCharacter(from: .whitespacesAndNewlines, options: .backwards)
            let wordStart = start + (lastBreak.location == NSNotFound ? 0 : lastBreak.location + lastBreak.length)
            layoutManager.ensureLayout(forCharacterRange: NSRange(location: max(0, start - 1), length: length - max(0, start - 1)))

            let lastGlyph = layoutManager.glyphIndexForCharacter(at: length - 1)
            let line: NSRect = layoutManager.extraLineFragmentTextContainer != nil
                ? layoutManager.extraLineFragmentUsedRect
                : TerminalStyle.textLineRect(of: lastGlyph, in: layoutManager)
            let x: CGFloat
            if wordStart < length {
                let glyph = layoutManager.glyphIndexForCharacter(at: wordStart)
                x = layoutManager.boundingRect(forGlyphRange: NSRange(location: glyph, length: 1), in: container).minX
            } else {
                x = layoutManager.boundingRect(forGlyphRange: NSRange(location: lastGlyph, length: 1), in: container).maxX
            }
            let origin = textView.textContainerOrigin
            let rect = NSRect(x: origin.x + x, y: origin.y + line.minY, width: max(1, line.maxX - x), height: line.height)
            let converted = textView.convert(rect, to: scrollView)
            return CGRect(x: converted.minX, y: scrollView.bounds.height - converted.maxY,
                          width: converted.width, height: converted.height)
        }

        /// True while the prompt is shown and nothing runs: the only state in which typing edits.
        var acceptsInput: Bool {
            if case .prompt = renderedLive { return !session.isRunning }
            return false
        }

        /// Where the editable command line begins (just after the prompt).
        var inputStart: Int {
            guard case .prompt = renderedLive else { return textLength }
            return committedLength + promptLength
        }

        private var textLength: Int { textView?.textStorage?.length ?? 0 }
        private var promptLength: Int { (ConsoleSession.prompt + " " as NSString).length }

        // MARK: Rendering

        /// Brings the text up to date: drops what a clear or the scrollback cap removed, appends new
        /// entries, then redraws the live line unless the user's own typing already matches it.
        func render(transcript: [ConsoleEntry], clearCount: Int, live: TerminalLiveLine) {
            guard let textView, let storage = textView.textStorage else { return }
            let wasAtBottom = isScrolledToBottom
            let selection = textView.selectedRange()
            let oldCommitted = committedLength
            var removedPrefix = 0

            storage.beginEditing()
            if clearCount != renderedClearCount || !isRenderedPrefix(of: transcript) {
                removedPrefix = committedLength
                storage.deleteCharacters(in: NSRange(location: 0, length: committedLength))
                renderedIDs = []
                renderedLengths = []
                committedLength = 0
                renderedClearCount = clearCount
            }
            if let first = transcript.first?.id, renderedIDs.first != first,
               let dropped = renderedIDs.firstIndex(of: first) {
                let length = renderedLengths[..<dropped].reduce(0, +)
                storage.deleteCharacters(in: NSRange(location: 0, length: length))
                renderedIDs.removeFirst(dropped)
                renderedLengths.removeFirst(dropped)
                committedLength -= length
                removedPrefix += length
            }
            if transcript.count > renderedIDs.count {
                let added = NSMutableAttributedString()
                for entry in transcript[renderedIDs.count...] {
                    let text = Self.attributed(entry)
                    added.append(text)
                    renderedIDs.append(entry.id)
                    renderedLengths.append(text.length)
                }
                storage.insert(added, at: committedLength)
                committedLength += added.length
            }

            let typedInputMatches: Bool = {
                guard case .prompt(let input, _) = live, case .prompt = renderedLive else { return false }
                return currentInput(in: storage) == input
            }()
            let liveChanged = !typedInputMatches && live != renderedLive
            if typedInputMatches {
                highlightInput()
            } else if liveChanged {
                let liveRange = NSRange(location: committedLength, length: storage.length - committedLength)
                storage.replaceCharacters(in: liveRange, with: Self.attributed(live))
            }
            renderedLive = live
            storage.endEditing()

            // A new live line takes the caret only when it was already on the live line, so a
            // selection being made in the scrollback survives progress updates and new output.
            let moveCaretToEnd = liveChanged && selection.location >= oldCommitted
            if moveCaretToEnd {
                textView.setSelectedRange(NSRange(location: storage.length, length: 0))
            } else {
                textView.setSelectedRange(Self.shifted(selection, oldCommitted: oldCommitted,
                                                       newCommitted: committedLength, removedPrefix: removedPrefix,
                                                       length: storage.length))
            }
            textView.typingAttributes = TerminalStyle.base
            if case .prompt(_, let suggestion) = live {
                textView.ghostText = suggestion
            } else {
                textView.ghostText = ""
            }
            if wasAtBottom || moveCaretToEnd {
                textView.scrollRangeToVisible(NSRange(location: storage.length, length: 0))
            }
            scheduleAnchorUpdate()
        }

        /// True when what is on screen is a prefix of `transcript` once entries the cap dropped
        /// from its front are removed; otherwise the scrollback is rebuilt.
        private func isRenderedPrefix(of transcript: [ConsoleEntry]) -> Bool {
            guard let first = transcript.first else { return renderedIDs.isEmpty }
            let start = renderedIDs.firstIndex(of: first.id) ?? (renderedIDs.isEmpty ? 0 : -1)
            guard start >= 0 else { return false }
            let kept = renderedIDs.count - start
            guard kept <= transcript.count else { return false }
            return kept == 0 || transcript[kept - 1].id == renderedIDs.last
        }

        /// A selection made before an update, moved to where the same text is afterwards.
        private static func shifted(_ range: NSRange, oldCommitted: Int, newCommitted: Int,
                                    removedPrefix: Int, length: Int) -> NSRange {
            let location: Int
            if range.location >= oldCommitted {
                location = range.location - oldCommitted + newCommitted
            } else if range.location >= removedPrefix {
                location = range.location - removedPrefix
            } else {
                return NSRange(location: length, length: 0)
            }
            let clamped = min(location, length)
            return NSRange(location: clamped, length: min(range.length, length - clamped))
        }

        private var isScrolledToBottom: Bool {
            guard let textView, let clip = textView.enclosingScrollView?.contentView else { return true }
            return clip.bounds.maxY >= textView.frame.height - TerminalStyle.font.pointSize
        }

        private func currentInput(in storage: NSTextStorage) -> String {
            let start = committedLength + promptLength
            guard start <= storage.length else { return "" }
            return (storage.string as NSString).substring(from: start)
        }

        /// Re-colours the command line after the user typed (attributes only; text and caret stay).
        private func highlightInput() {
            guard let textView, let storage = textView.textStorage, !textView.hasMarkedText() else { return }
            let start = inputStart
            guard start <= storage.length else { return }
            let range = NSRange(location: start, length: storage.length - start)
            let input = (storage.string as NSString).substring(with: range)
            let styled = Self.highlighted(input)
            styled.enumerateAttributes(in: NSRange(location: 0, length: styled.length)) { attributes, subrange, _ in
                storage.setAttributes(attributes, range: NSRange(location: start + subrange.location, length: subrange.length))
            }
        }

        // MARK: Attributed text

        /// One scrollback entry, ending with a line break.
        private static func attributed(_ entry: ConsoleEntry) -> NSAttributedString {
            let text = NSMutableAttributedString()
            switch entry.style {
            case .command:
                text.append(promptText)
                text.append(highlighted(entry.text))
                if entry.interrupted { text.append(NSAttributedString(string: "^C", attributes: colored(TerminalStyle.dim))) }
                separateFromPreviousOutput(text)
            case .output:
                text.append(NSAttributedString(string: entry.text, attributes: TerminalStyle.base))
            case .success:
                text.append(NSAttributedString(string: entry.text, attributes: colored(TerminalStyle.success)))
            case .warning:
                text.append(NSAttributedString(string: "warning: ", attributes: colored(TerminalStyle.warning, bold: true)))
                text.append(NSAttributedString(string: entry.text, attributes: colored(TerminalStyle.warning)))
            case .error:
                text.append(NSAttributedString(string: "error: ", attributes: colored(TerminalStyle.error, bold: true)))
                text.append(NSAttributedString(string: entry.text, attributes: colored(TerminalStyle.error)))
            case .note:
                text.append(NSAttributedString(string: entry.text, attributes: colored(TerminalStyle.dim)))
            }
            text.append(NSAttributedString(string: "\n", attributes: TerminalStyle.base))
            return text
        }

        /// The live line (no trailing line break: the caret sits at its end).
        private static func attributed(_ live: TerminalLiveLine) -> NSAttributedString {
            switch live {
            case .prompt(let input, _):
                let text = NSMutableAttributedString(attributedString: promptText)
                text.append(highlighted(input))
                separateFromPreviousOutput(text)
                return text
            case .running(let progress):
                return NSAttributedString(string: progress ?? "", attributes: colored(TerminalStyle.progress))
            case .search(let search):
                let label = "(\(search.failed ? "failing " : "")reverse-i-search)`\(search.query)': "
                let text = NSMutableAttributedString(string: label, attributes: colored(TerminalStyle.dim))
                text.append(highlighted(search.match))
                return text
            }
        }

        /// Adds `commandSpacing` above the first line of a command (its continuation lines keep
        /// their own indent style), so each command starts a visibly new block.
        private static func separateFromPreviousOutput(_ text: NSMutableAttributedString) {
            let firstLine = (text.string as NSString).lineRange(for: NSRange(location: 0, length: 0))
            let style = NSMutableParagraphStyle()
            style.paragraphSpacingBefore = TerminalStyle.commandSpacing
            text.addAttribute(.paragraphStyle, value: style, range: firstLine)
        }

        private static var promptText: NSAttributedString {
            NSAttributedString(string: ConsoleSession.prompt + " ", attributes: colored(TerminalStyle.prompt, bold: true))
        }

        private static func colored(_ color: NSColor, bold: Bool = false) -> [NSAttributedString.Key: Any] {
            [.font: bold ? TerminalStyle.boldFont : TerminalStyle.font, .foregroundColor: color]
        }

        /// DruseScript syntax colours; lines after the first are indented under the prompt.
        private static func highlighted(_ source: String) -> NSMutableAttributedString {
            let text = NSMutableAttributedString(string: source, attributes: TerminalStyle.base)
            for span in DruseScriptHighlighter.highlight(source) {
                guard let color = span.kind.nsColor else { continue }
                text.addAttribute(.foregroundColor, value: color, range: NSRange(span.range, in: source))
            }
            let firstBreak = (source as NSString).range(of: "\n")
            if firstBreak.location != NSNotFound {
                let continuation = NSMutableParagraphStyle()
                continuation.firstLineHeadIndent = TerminalStyle.promptWidth
                continuation.headIndent = TerminalStyle.promptWidth
                let start = firstBreak.location + 1
                text.addAttribute(.paragraphStyle, value: continuation, range: NSRange(location: start, length: text.length - start))
            }
            return text
        }

        // MARK: Typing

        func textView(_ textView: NSTextView, shouldChangeTextIn range: NSRange, replacementString: String?) -> Bool {
            acceptsInput && range.location >= inputStart
        }

        func textDidChange(_ notification: Notification) {
            guard acceptsInput, let storage = textView?.textStorage else { return }
            let input = currentInput(in: storage)
            renderedLive = .prompt(input: input, suggestion: "")
            highlightInput()
            session.edit(input)
        }

        func textViewDidChangeSelection(_ notification: Notification) {
            textView?.needsDisplay = true
        }

        /// Typing, pasting or deleting while the selection is in the scrollback edits the command
        /// line instead: the caret jumps to its end first, as in a terminal.
        func moveCaretIntoInputIfNeeded() {
            guard acceptsInput, let textView, textView.selectedRange().location < inputStart else { return }
            textView.setSelectedRange(NSRange(location: textLength, length: 0))
        }

        // MARK: Keys

        /// Keys a text view would not route as commands: control chords, ⇧↩ and the keys of a
        /// ⌃R search. Returns true when handled.
        func handleKeyDown(_ event: NSEvent) -> Bool {
            let flags = event.modifierFlags.intersection([.command, .control, .option, .shift])
            let key = event.charactersIgnoringModifiers?.lowercased() ?? ""
            if flags == .control {
                switch key {
                case "c":
                    session.interrupt()
                    return true
                case "r":
                    session.searchHistoryBackward()
                    return true
                case "g" where session.historySearch != nil:
                    session.cancelHistorySearch()
                    return true
                case "l":
                    session.clearScrollback()
                    return true
                default:
                    break
                }
            }
            if let search = session.historySearch {
                return handleSearchKey(event, search: search, flags: flags)
            }
            if flags == .shift, event.keyCode == KeyCode.return || event.keyCode == KeyCode.enter {
                insertLineBreak()
                return true
            }
            return false
        }

        /// While searching, printable keys edit the query, ⌫ shortens it, Return runs the match,
        /// and any other key puts the match on the command line for editing.
        private func handleSearchKey(_ event: NSEvent, search: ConsoleHistorySearch, flags: NSEvent.ModifierFlags) -> Bool {
            switch event.keyCode {
            case KeyCode.return, KeyCode.enter:
                session.acceptHistorySearch(run: true)
            case KeyCode.delete:
                session.updateHistorySearch(query: String(search.query.dropLast()))
            default:
                if flags.isDisjoint(with: [.command, .control, .option]), let characters = event.characters,
                   !characters.isEmpty, characters.unicodeScalars.allSatisfy(Self.isPrintable) {
                    session.updateHistorySearch(query: search.query + characters)
                } else if flags.contains(.command) {
                    return false
                } else {
                    session.acceptHistorySearch(run: false)
                }
            }
            return true
        }

        private static func isPrintable(_ scalar: Unicode.Scalar) -> Bool {
            // 0xF700… are AppKit's function-key characters (arrows, F-keys, Home, End).
            !CharacterSet.controlCharacters.contains(scalar) && !(0xF700...0xF8FF).contains(scalar.value)
        }

        /// ⇧↩ / ⌥↩: continues the command on a new line (the whole input then runs as one script).
        private func insertLineBreak() {
            guard acceptsInput, let textView else { return }
            moveCaretIntoInputIfNeeded()
            textView.insertText("\n", replacementRange: textView.selectedRange())
        }

        func textView(_ textView: NSTextView, doCommandBy selector: Selector) -> Bool {
            let session = self.session
            let listOpen = !session.completionItems.isEmpty
            let length = textLength
            let selection = textView.selectedRange()
            let caretAtEnd = selection == NSRange(location: length, length: 0)
            let start = inputStart
            let inFirstInputLine = acceptsInput && selection.location >= start
                && !(textView.string as NSString).substring(with: NSRange(location: start, length: selection.location - start)).contains("\n")
            switch selector {
            case #selector(NSResponder.insertNewline(_:)):
                guard acceptsInput else { return true }
                if listOpen { session.acceptCompletion() } else { session.submitInput() }
                return true
            case #selector(NSResponder.insertNewlineIgnoringFieldEditor(_:)), #selector(NSResponder.insertLineBreak(_:)):
                insertLineBreak()
                return true
            case #selector(NSResponder.insertTab(_:)):
                if acceptsInput, caretAtEnd { session.complete() }
                return true
            case #selector(NSResponder.insertBacktab(_:)):
                session.moveCompletionSelection(by: -1)
                return true
            case #selector(NSResponder.cancelOperation(_:)):
                guard listOpen else { return false }
                session.dismissCompletions()
                return true
            case #selector(NSResponder.moveUp(_:)):
                return moveVertically(by: -1, listOpen: listOpen)
            case #selector(NSResponder.moveDown(_:)):
                return moveVertically(by: 1, listOpen: listOpen)
            case #selector(NSResponder.moveLeft(_:)):
                // The caret never walks back into the prompt.
                return acceptsInput && selection.length == 0 && selection.location == start
            case #selector(NSResponder.moveToBeginningOfLine(_:)),
                 #selector(NSResponder.moveToBeginningOfParagraph(_:)),
                 #selector(NSResponder.moveToLeftEndOfLine(_:)):
                guard inFirstInputLine else { return false }
                textView.setSelectedRange(NSRange(location: start, length: 0))
                return true
            case #selector(NSResponder.deleteToBeginningOfLine(_:)),
                 #selector(NSResponder.deleteToBeginningOfParagraph(_:)):
                guard inFirstInputLine, selection.location > start else { return inFirstInputLine }
                textView.insertText("", replacementRange: NSRange(location: start, length: selection.location - start))
                return true
            case #selector(NSResponder.moveRight(_:)),
                 #selector(NSResponder.moveToEndOfLine(_:)),
                 #selector(NSResponder.moveToRightEndOfLine(_:)),
                 #selector(NSResponder.moveToEndOfParagraph(_:)),
                 #selector(NSResponder.moveToEndOfDocument(_:)),
                 #selector(NSResponder.scrollToEndOfDocument(_:)):
                return caretAtEnd && session.acceptSuggestion(wordOnly: false)
            case #selector(NSResponder.moveWordRight(_:)):
                return caretAtEnd && session.acceptSuggestion(wordOnly: true)
            default:
                return false
            }
        }

        /// ↑/↓ move in the open completion list, else through history; inside a multi-line
        /// command they move the caret as usual.
        private func moveVertically(by offset: Int, listOpen: Bool) -> Bool {
            if listOpen {
                session.moveCompletionSelection(by: offset)
                return true
            }
            guard acceptsInput else { return false }
            guard !session.input.contains("\n") else { return false }
            if offset < 0 { session.showPreviousHistory() } else { session.showNextHistory() }
            return true
        }
    }
}

/// Virtual key codes the terminal reacts to independently of the keyboard layout.
private enum KeyCode {
    static let `return`: UInt16 = 36
    static let enter: UInt16 = 76
    static let delete: UInt16 = 51
}

/// Keeps the text view exactly as wide as the visible area, so lines always wrap at the panel
/// edge (a terminal never scrolls sideways), whatever size the view had when it was created.
final class TerminalScrollView: NSScrollView {
    override func tile() {
        super.tile()
        guard let documentView, documentView.frame.width != contentSize.width else { return }
        documentView.setFrameSize(NSSize(width: contentSize.width, height: documentView.frame.height))
    }
}

/// The terminal's text view: forwards special keys to the coordinator, keeps typing inside the
/// command line, copies and pastes plain text only, draws the ghost suggestion after the caret,
/// and takes keyboard focus as soon as it is placed in a window.
final class TerminalNSTextView: NSTextView {
    weak var coordinator: TerminalTextView.Coordinator?

    /// Fish-style suggestion drawn after the caret while it sits at the end of the command line.
    var ghostText = "" {
        didSet { if ghostText != oldValue { needsDisplay = true } }
    }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        if let window { window.makeFirstResponder(self) }
    }

    override func keyDown(with event: NSEvent) {
        if coordinator?.handleKeyDown(event) == true { return }
        super.keyDown(with: event)
    }

    /// ⌘K clears the scrollback while the terminal has focus.
    override func performKeyEquivalent(with event: NSEvent) -> Bool {
        if window?.firstResponder === self,
           event.modifierFlags.intersection([.command, .control, .option, .shift]) == .command,
           event.charactersIgnoringModifiers?.lowercased() == "k" {
            coordinator?.session.clearScrollback()
            return true
        }
        return super.performKeyEquivalent(with: event)
    }

    override func insertText(_ string: Any, replacementRange: NSRange) {
        if replacementRange.location == NSNotFound { coordinator?.moveCaretIntoInputIfNeeded() }
        super.insertText(string, replacementRange: replacementRange)
    }

    /// Pasted text lands on the command line as plain text; several lines stay one multi-line
    /// command that Return runs as a script (like bracketed paste in zsh or fish).
    override func paste(_ sender: Any?) {
        coordinator?.moveCaretIntoInputIfNeeded()
        pasteAsPlainText(sender)
    }

    override func deleteBackward(_ sender: Any?) {
        coordinator?.moveCaretIntoInputIfNeeded()
        super.deleteBackward(sender)
    }

    override var writablePasteboardTypes: [NSPasteboard.PasteboardType] { [.string] }
    override var readablePasteboardTypes: [NSPasteboard.PasteboardType] { [.string] }

    override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)
        let length = (string as NSString).length
        guard !ghostText.isEmpty, length > 0, selectedRange() == NSRange(location: length, length: 0),
              let layoutManager, let textContainer else { return }
        let lastGlyph = layoutManager.glyphIndexForCharacter(at: length - 1)
        let glyphRect = layoutManager.boundingRect(forGlyphRange: NSRange(location: lastGlyph, length: 1), in: textContainer)
        let lineRect = TerminalStyle.textLineRect(of: lastGlyph, in: layoutManager)
        let origin = textContainerOrigin
        let point = NSPoint(x: origin.x + glyphRect.maxX, y: origin.y + lineRect.minY)
        (ghostText as NSString).draw(at: point, withAttributes: [.font: TerminalStyle.font, .foregroundColor: TerminalStyle.dim])
    }
}
