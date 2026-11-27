# Repository Guidelines

## Project Structure

Druse is a SwiftUI molecular docking app for Apple Silicon. `App/main.swift` selects the GUI or headless entry point. Both use the interpreter in `Console/`. Put DruseScript commands in `Console/Commands/` so both interfaces can use them.

- `Views/` contains SwiftUI screens and `ViewModels/` owns application state.
- `Chemistry/` handles parsing and preparation. `Docking/` coordinates docking runs, and `CppCore/` exposes native chemistry routines through `druse_core.h`.
- `Renderer/` drives 3D rendering. `Metal/` contains GPU kernels and shared shader types.
- `Resources/` holds bundled assets. `Examples/` contains runnable DruseScript files.

## Build and Run

Use Apple Silicon with macOS 26+, Xcode 26+, CMake, and Homebrew. Run these commands from the repository root.

- `make deps` installs the Homebrew libraries and XcodeGen.
- `make` builds the C++ core, generates the Xcode project, and builds the Release app.
- `make run CONFIG=Debug` builds and launches a debug app.
- `make eval CMD="help"` runs the command interpreter without opening a window.

Edit target settings in `project.yml`. Register new top-level source directories there, then run `make project` to regenerate the Xcode project.

## Coding Style

Indent Swift with four spaces. Use UpperCamelCase for types and lowerCamelCase for methods and properties. Name feature extensions after the type and responsibility, as in `AppViewModel+Docking.swift`. Match surrounding formatting. The repository has no configured Swift formatter or linter.

## Testing

Use XCTest in `Tests/` for unit and integration tests and `UITests/` for UI workflows. Name test classes `<Feature>Tests` and methods after observable behavior, such as `testLastDuplicateOptionWins`.

Run `make test` for the `DruseTests` target. Ask before launching tests from an agent session because the host app opens a window. Set `DRUSE_INTEGRATION_TESTS=1` in the test environment to enable slow integration cases. The repository specifies no minimum coverage percentage.

## Commits and Pull Requests

Follow the history’s short, descriptive commit subjects, such as “Fixed build from source.” Release commits often include the version. In a pull request, describe the behavior change and how you verified it. Link a related issue when available and include a screenshot for visible UI changes.
