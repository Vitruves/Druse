# Installation

Installing Druse from the release DMG, clearing Gatekeeper on an unsigned build, and building the whole stack from source.

Contents: [System requirements](#system-requirements) · [Install from the DMG](#install-from-the-dmg) · [Gatekeeper](#gatekeeper) · [Build from source](#build-from-source) · [Running the app you built](#running-the-app-you-built) · [Packaging a portable DMG](#packaging-a-portable-dmg) · [Version convention](#version-convention)

## System requirements

| Requirement | Value |
|---|---|
| Operating system | macOS 26.0 or later (`MACOSX_DEPLOYMENT_TARGET: "26.0"` in `project.yml`) |
| Hardware | Apple Silicon only — the app is built `arm64`-only (`ARCHS: arm64`, `CMAKE_OSX_ARCHITECTURES "arm64"`) and needs a Metal device |
| Xcode (to build) | 26.0 or later (`options.xcodeVersion: "26.0"`) |
| XcodeGen (to build) | 2.35 or later (`options.minimumXcodeGenVersion: "2.35"`) |
| Swift (to build) | 5.9 language mode, with `SWIFT_STRICT_CONCURRENCY: complete` |

A release DMG built by `scripts/bundle_dmg.sh` carries its own copies of every non-system dynamic library, so an installed app needs no Homebrew packages. Homebrew is only required to build from source.

## Install from the DMG

1. Download `Druse-<version>.dmg` from the [releases page](https://github.com/Vitruves/Druse/releases).
2. Open the DMG and drag `Druse.app` onto the `Applications` symlink inside it.
3. Launch Druse from Applications (see [Gatekeeper](#gatekeeper) for the first launch).

The DMG contains the app and an `/Applications` symlink; nothing else is installed, and the app writes only under your home folder (`~/.druse` for the PDB cache, logs and the terminal history — see [Settings](Settings.md)).

## Gatekeeper

Druse is distributed unsigned (the bundle is ad-hoc signed, not Developer ID signed and not notarized), so macOS blocks it on first launch. Two ways around it:

- **Right-click to open.** Control-click `Druse.app` in Finder, choose **Open**, then **Open** again in the confirmation dialog. macOS remembers the decision for that copy of the app.
- **Strip the quarantine attribute.** In Terminal:

```bash
xattr -cr /Applications/Druse.app
```

Both are needed only once per copy of the app. Moving or replacing the bundle re-applies the quarantine flag to the new copy.

## Build from source

### 1. Install the toolchain and dependencies

```bash
brew install rdkit boost eigen nanoflann tbb
brew install xcodegen
```

`CppCore/CMakeLists.txt` expects Homebrew at `/opt/homebrew` (`RDKIT_ROOT`, `BOOST_ROOT=/opt/homebrew/opt/boost`, `/opt/homebrew/include/eigen3`, `/opt/homebrew/include` for nanoflann); override `-DRDKIT_ROOT=` / `-DBOOST_ROOT=` if yours lives elsewhere. Two more dependencies are fetched by CMake and need no Homebrew formula:

| Dependency | How it is obtained | Version |
|---|---|---|
| [gemmi](https://github.com/project-gemmi/gemmi) | `FetchContent` at configure time, built as a static library | `v0.7.0` |
| [LBFGSpp](https://github.com/yixuan/LBFGSpp) | `FetchContent`, header-only | `v0.3.0` |
| [OpenMM](https://github.com/openmm/openmm) | Cloned, configured, built and installed into the CMake deps tree on the first configure | `8.5.0` |

OpenMM is controlled by the `DRUSE_ENABLE_OPENMM` option (`ON` by default). Its first build takes several minutes; afterwards CMake reuses the installed copy. If the OpenMM build fails, CMake prints a warning and continues without OpenMM refinement — but the Xcode target links `-lOpenMM`, so the app link will then fail; configure with `-DDRUSE_ENABLE_OPENMM=OFF` only if you also remove that flag from `project.yml`.

GFN2-xTB is implemented inside the C++ core (`CppCore/druse_xtb.cpp`) with a Metal accelerator; no external xTB package is linked.

### 2. Clone the repository

```bash
git clone https://github.com/Vitruves/Druse.git
cd Druse
```

### 3. Build the C++ core

```bash
cd CppCore
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)
cd ../..
```

This produces `CppCore/build/libdruse_core.a` and `CppCore/build/libgemmi_cpp.a`, plus `CppCore/build/_deps/openmm-install/` — exactly the paths `project.yml` lists in `LIBRARY_SEARCH_PATHS`, so the build directory must stay where it is. Link-time optimisation is deliberately forced off for the core and for gemmi: Xcode links the app with Apple's libLTO, and bitcode from a non-Apple LLVM cannot be parsed by it.

### 4. Generate the Xcode project

```bash
xcodegen generate
```

Re-run this after adding or removing source files, and after adding a top-level source directory — new directories must also be listed under the `Druse` target's `sources` in `project.yml`.

### 5. Build the app

```bash
xcodebuild -project Druse.xcodeproj -scheme Druse -configuration Release build
```

To keep the build products inside the checkout instead of Xcode's shared `DerivedData`, add `-derivedDataPath build`; the app then lands at `build/Build/Products/Release/Druse.app`, which is also where `Tools/druse` looks for a locally built bundle (see [Console](Console.md)).

```bash
xcodebuild -project Druse.xcodeproj -scheme Druse -configuration Release -derivedDataPath build build
```

The project defines three targets: `Druse` (the app), `DruseTests` (unit tests, which also compile `Benchmark/`) and `DruseUITests`. Running the test scheme opens a GUI window.

## Running the app you built

```bash
open build/Build/Products/Release/Druse.app
```

With the default `DerivedData` location, ask `xcodebuild` where the product went:

```bash
open "$(xcodebuild -project Druse.xcodeproj -scheme Druse -configuration Release -showBuildSettings 2>/dev/null \
  | awk -F' = ' '/ BUILT_PRODUCTS_DIR /{print $2}')/Druse.app"
```

A source checkout can also drive the headless interpreter through the `Tools/druse` wrapper, which finds a built or installed bundle by itself:

```bash
chmod +x Tools/druse
./Tools/druse --help
```

See [Console](Console.md) for headless mode, and for installing a `druse` command from an installed app.

## Packaging a portable DMG

`scripts/bundle_dmg.sh` turns a built app into a self-contained bundle and a compressed DMG:

```bash
./scripts/bundle_dmg.sh                                         # builds Release into build/derived first
./scripts/bundle_dmg.sh build/Build/Products/Release/Druse.app  # bundles an app you already built
```

What it does, in order:

1. Builds the Release app (into `build/derived`) unless an existing `.app` is given as the first argument.
2. Checks that the required resources are present in the bundle: `Resources/dun2010bbdep.bin` (Dunbrack rotamer library) and `Resources/standalone/` (the Ketcher editor's `index.html` and `static/js`). It stops if any is missing.
3. Walks the dynamic-library dependency graph of the executable recursively, skipping `/usr/lib`, `/System` and `@executable_path` entries, and copies every remaining dylib into `Contents/Frameworks`.
4. Rewrites all load paths to `@rpath/...`, adds `@executable_path/../Frameworks` to the executable's rpath, then makes a second pass over the bundled dylibs to catch cross-references.
5. Ad-hoc codesigns each bundled dylib, the CoreML model (`PocketDetector.mlmodelc`) when present, and the app. Override the identity with `CODESIGN_IDENTITY=` (default `-`, ad-hoc).
6. Verifies that no `/opt/homebrew` reference remains in the executable or in any bundled dylib, and reports any that do.
7. Creates `build/Druse-<VERSION>.dmg` with `hdiutil` (UDZO, zlib level 9), volume name `Druse <VERSION>`, containing the app and an `/Applications` symlink.

The version in the DMG name and volume label is read from the repository's `VERSION` file. The script finishes by printing the `xcrun notarytool submit` / `xcrun stapler staple` commands needed to notarize the DMG, which requires a Developer ID.

## Version convention

`VERSION` at the repository root is the single source of truth (for example `0.1.1`, or `0.1.30-beta`). Two places mirror it:

| File | Field | Form |
|---|---|---|
| `project.yml` | `MARKETING_VERSION` under the `Druse` target | the version with any `-suffix` stripped, since `CFBundleShortVersionString` only accepts dotted integers |
| `README.md` | version badge, download button label, and the `.dmg` release URL | the full version; the badge escapes `-` as `--` for shields.io |

`./update_readme_version.sh` syncs both from `VERSION` and does **not** bump anything by itself:

```bash
./update_readme_version.sh                          # sync README.md and project.yml from the current VERSION
./update_readme_version.sh --version 0.1.30-beta    # write VERSION, then sync
```

The `--version` argument must match `MAJOR.MINOR.PATCH` with an optional alphanumeric suffix (`0.1.30` or `0.1.30-beta`); anything else is rejected. After a bump, re-run `xcodegen generate` so the new `MARKETING_VERSION` reaches the Xcode project.

---

Next: [Preparation](Preparation.md) · [Console](Console.md) · [Settings](Settings.md) · [README](../README.md)
