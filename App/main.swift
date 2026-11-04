// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Foundation

// Entry point: decide between the SwiftUI app and headless DruseScript execution
// before any scene, window or NSApplication is created.
MainActor.assumeIsolated {
    let arguments = Array(CommandLine.arguments.dropFirst())
    if HeadlessRunner.isRequested(by: arguments) {
        HeadlessRunner.start(arguments: arguments)
    } else {
        DruseApp.main()
    }
}
