#!/bin/bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────
# bundle_dmg.sh — Create a portable DMG for Druse
#
# Copies all Homebrew/build dylibs into Druse.app/Contents/Frameworks,
# rewrites load paths so the app is fully self-contained.
#
# Usage:  ./scripts/bundle_dmg.sh [path/to/Druse.app]
#
# If no .app path given, builds Release first.
# ─────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VERSION=$(cat "$PROJECT_DIR/VERSION")
BUILD_DIR="$PROJECT_DIR/build"
DMG_DIR="$BUILD_DIR/dmg"
FRAMEWORKS_DIR=""  # set after we know APP_PATH

# Temp file for collecting dylib paths
DYLIB_LIST=$(mktemp)
trap "rm -f $DYLIB_LIST" EXIT

# ── Codesign identity (set to "-" for ad-hoc, or your Developer ID) ──
CODESIGN_IDENTITY="${CODESIGN_IDENTITY:--}"

# ── Resolve or build the .app ──
if [ -n "${1:-}" ] && [ -d "$1" ]; then
    APP_PATH="$1"
else
    echo "==> Building Druse Release..."
    xcodebuild -project "$PROJECT_DIR/Druse.xcodeproj" \
        -scheme Druse -configuration Release \
        -derivedDataPath "$BUILD_DIR/derived" \
        build 2>&1 | tail -5
    APP_PATH=$(find "$BUILD_DIR/derived" -name "Druse.app" -type d | head -1)
    if [ -z "$APP_PATH" ]; then
        echo "ERROR: Could not find Druse.app after build"
        exit 1
    fi
fi

echo "==> Using app: $APP_PATH"
FRAMEWORKS_DIR="$APP_PATH/Contents/Frameworks"
mkdir -p "$FRAMEWORKS_DIR"

# ─────────────────────────────────────────────────────────────
# 1. Collect all dylibs that need bundling
# ─────────────────────────────────────────────────────────────

HOMEBREW="/opt/homebrew"
OPENMM_LIB="$PROJECT_DIR/CppCore/build/_deps/openmm-install/lib"

# Helper: get the real file behind a path (follow symlinks)
realfile() {
    python3 -c "import os,sys; print(os.path.realpath(sys.argv[1]))" "$1"
}

# Helper: add a dylib to the list (resolves symlinks, deduplicates later)
add_dylib() {
    local path="$1"
    if [ -f "$path" ] || [ -L "$path" ]; then
        realfile "$path" >> "$DYLIB_LIST"
    else
        echo "  WARNING: not found: $path"
    fi
}

echo "==> Collecting dylibs..."

# RDKit libs (from project.yml)
for lib in \
    RDKitSmilesParse RDKitGraphMol RDKitRDGeneral \
    RDKitDistGeomHelpers RDKitDistGeometry \
    RDKitForceFieldHelpers RDKitForceField \
    RDKitPartialCharges RDKitDescriptors \
    RDKitMolTransforms RDKitSubstructMatch \
    RDKitEigenSolvers RDKitAlignment \
    RDKitRDGeometryLib RDKitDataStructs \
    RDKitDepictor RDKitFileParsers \
    RDKitChemTransforms RDKitFingerprints \
    RDKitGenericGroups RDKitMolDraw2D \
    RDKitCIPLabeler RDKitRingDecomposerLib \
    RDKitOptimizer RDKitRDStreams \
    RDKitCatalogs RDKitSubgraphs \
    RDKitMMPA RDKitMolChemicalFeatures \
    RDKitFMCS RDKitMolStandardize \
; do
    add_dylib "$HOMEBREW/lib/lib${lib}.dylib"
done

# coordgen + maeparser
add_dylib "$HOMEBREW/opt/coordgen/lib/libcoordgen.dylib"
add_dylib "$HOMEBREW/opt/maeparser/lib/libmaeparser.dylib"

# TBB
add_dylib "$HOMEBREW/opt/tbb/lib/libtbb.dylib"

# Boost (serialization, iostreams, random, regex)
for lib in libboost_serialization libboost_iostreams libboost_random libboost_regex; do
    add_dylib "$HOMEBREW/opt/boost/lib/${lib}.dylib"
done

# ICU (needed by boost_iostreams)
for lib in libicudata.78 libicui18n.78 libicuuc.78; do
    add_dylib "$HOMEBREW/opt/icu4c@78/lib/${lib}.dylib"
done

# lzma (needed by boost_iostreams)
add_dylib "$HOMEBREW/opt/xz/lib/liblzma.5.dylib"

# OpenMM
if [ -d "$OPENMM_LIB" ]; then
    for f in "$OPENMM_LIB"/libOpenMM*.dylib; do
        add_dylib "$f"
    done
    # libomp (needed by OpenMM)
    add_dylib "$HOMEBREW/opt/llvm/lib/libomp.dylib"
    # LLVM's libc++ (needed by OpenMM)
    if [ -f "$HOMEBREW/opt/llvm/lib/c++/libc++.1.dylib" ]; then
        add_dylib "$HOMEBREW/opt/llvm/lib/c++/libc++.1.dylib"
    fi
fi

# Deduplicate
sort -u "$DYLIB_LIST" > "${DYLIB_LIST}.dedup"
mv "${DYLIB_LIST}.dedup" "$DYLIB_LIST"

DYLIB_COUNT=$(wc -l < "$DYLIB_LIST" | tr -d ' ')
echo "  Found $DYLIB_COUNT dylibs to bundle"

# ─────────────────────────────────────────────────────────────
# 2. Copy dylibs into Frameworks/
# ─────────────────────────────────────────────────────────────

echo "==> Copying dylibs to Frameworks/..."
while IFS= read -r dylib; do
    cp -f "$dylib" "$FRAMEWORKS_DIR/"
    chmod 755 "$FRAMEWORKS_DIR/$(basename "$dylib")"
    echo "  $(basename "$dylib")"
done < "$DYLIB_LIST"

# ─────────────────────────────────────────────────────────────
# 3. Rewrite load paths with install_name_tool
# ─────────────────────────────────────────────────────────────

echo "==> Rewriting dylib load paths..."

# Helper: rewrite all non-system references in a Mach-O binary
rewrite_paths() {
    local binary="$1"

    # Change the install name of the binary itself (if it's a dylib)
    local current_id
    current_id=$(otool -D "$binary" 2>/dev/null | tail -1)
    if [ -n "$current_id" ] && [ "$current_id" != "$binary" ]; then
        case "$current_id" in
            *"not an object"*) ;;
            *)
                local basename_id
                basename_id=$(basename "$current_id")
                install_name_tool -id "@rpath/$basename_id" "$binary" 2>/dev/null || true
                ;;
        esac
    fi

    # Rewrite all dependency paths
    otool -L "$binary" 2>/dev/null | tail -n +2 | awk '{print $1}' | while IFS= read -r dep; do
        local dep_basename
        dep_basename=$(basename "$dep")

        # Skip system libs
        case "$dep" in
            /usr/lib/*|/System/*) continue ;;
        esac

        # If this dep is one of our bundled dylibs, rewrite to @rpath
        if [ -f "$FRAMEWORKS_DIR/$dep_basename" ]; then
            install_name_tool -change "$dep" "@rpath/$dep_basename" "$binary" 2>/dev/null || true
        fi

        # Handle @rpath references with versioned names (e.g. libRDKitFoo.1.dylib)
        case "$dep" in
            @rpath/*)
                if [ ! -f "$FRAMEWORKS_DIR/$dep_basename" ]; then
                    # Try to find the actual file (strip version number)
                    local unversioned
                    unversioned=$(echo "$dep_basename" | sed 's/\.[0-9]*\.dylib/.dylib/')
                    if [ -f "$FRAMEWORKS_DIR/$unversioned" ]; then
                        ln -sf "$unversioned" "$FRAMEWORKS_DIR/$dep_basename" 2>/dev/null || true
                    fi
                fi
                ;;
        esac

        # Handle @loader_path references (e.g., Boost internal refs)
        case "$dep" in
            @loader_path/*)
                if [ -f "$FRAMEWORKS_DIR/$dep_basename" ]; then
                    install_name_tool -change "$dep" "@rpath/$dep_basename" "$binary" 2>/dev/null || true
                fi
                ;;
        esac
    done
}

# Rewrite paths in all bundled dylibs
for dylib in "$FRAMEWORKS_DIR"/*.dylib; do
    [ -L "$dylib" ] && continue  # skip symlinks
    rewrite_paths "$dylib"
done

# Rewrite paths in the main executable
MAIN_EXEC="$APP_PATH/Contents/MacOS/Druse"
if [ -f "$MAIN_EXEC" ]; then
    rewrite_paths "$MAIN_EXEC"

    # Add @executable_path/../Frameworks as rpath if not already present
    install_name_tool -add_rpath "@executable_path/../Frameworks" "$MAIN_EXEC" 2>/dev/null || true
fi

# ─────────────────────────────────────────────────────────────
# 4. Create versioned symlinks for @rpath references
# ─────────────────────────────────────────────────────────────

echo "==> Creating versioned symlinks..."
cd "$FRAMEWORKS_DIR"
for dylib in *.dylib; do
    [ -L "$dylib" ] && continue  # skip existing symlinks

    # RDKit: full-version name like libRDKitFoo.2025.09.6.dylib -> .1.dylib + unversioned
    case "$dylib" in
        libRDKit*.*.*.*.dylib)
            short_name=$(echo "$dylib" | sed 's/\.[0-9][0-9]*\.[0-9]*\.[0-9]*\.dylib/.1.dylib/')
            unversioned=$(echo "$dylib" | sed 's/\.[0-9][0-9]*\.[0-9]*\.[0-9]*\.dylib/.dylib/')
            ln -sf "$dylib" "$short_name" 2>/dev/null || true
            ln -sf "$dylib" "$unversioned" 2>/dev/null || true
            echo "  $short_name -> $dylib"
            ;;
    esac

    # TBB versioned symlink
    case "$dylib" in
        libtbb.12.*.dylib)
            ln -sf "$dylib" "libtbb.12.dylib" 2>/dev/null || true
            ln -sf "$dylib" "libtbb.dylib" 2>/dev/null || true
            ;;
    esac

    # coordgen versioned symlink
    case "$dylib" in
        libcoordgen.3.*.dylib)
            ln -sf "$dylib" "libcoordgen.3.dylib" 2>/dev/null || true
            ln -sf "$dylib" "libcoordgen.dylib" 2>/dev/null || true
            ;;
    esac

    # maeparser versioned symlink
    case "$dylib" in
        libmaeparser.1.*.dylib)
            ln -sf "$dylib" "libmaeparser.1.dylib" 2>/dev/null || true
            ln -sf "$dylib" "libmaeparser.dylib" 2>/dev/null || true
            ;;
    esac
done
cd "$PROJECT_DIR"

# ─────────────────────────────────────────────────────────────
# 5. Ad-hoc codesign everything
# ─────────────────────────────────────────────────────────────

echo "==> Codesigning..."
# Sign frameworks first, then the app
for dylib in "$FRAMEWORKS_DIR"/*.dylib; do
    [ -L "$dylib" ] && continue  # skip symlinks
    codesign --force --sign "$CODESIGN_IDENTITY" "$dylib" 2>/dev/null || true
done
codesign --force --deep --sign "$CODESIGN_IDENTITY" "$APP_PATH" 2>/dev/null || true

# ─────────────────────────────────────────────────────────────
# 6. Verify
# ─────────────────────────────────────────────────────────────

echo "==> Verifying bundle..."
otool -L "$MAIN_EXEC" 2>/dev/null | tail -n +2 | awk '{print $1}' | while IFS= read -r dep; do
    case "$dep" in
        /usr/lib/*|/System/*|@rpath/*|@executable_path/*) continue ;;
        /opt/homebrew/*) echo "  WARNING: unbundled dependency: $dep" ;;
    esac
done

# Also verify a bundled dylib
echo "  Checking a sample bundled dylib..."
SAMPLE=$(ls "$FRAMEWORKS_DIR"/libRDKitSmilesParse*.dylib 2>/dev/null | head -1)
if [ -n "$SAMPLE" ]; then
    UNBUNDLED=$(otool -L "$SAMPLE" | awk '{print $1}' | grep -c "/opt/homebrew" || true)
    if [ "$UNBUNDLED" -gt 0 ]; then
        echo "  WARNING: $UNBUNDLED unbundled references in $(basename "$SAMPLE")"
    else
        echo "  OK — no Homebrew references in bundled dylibs"
    fi
fi

# ─────────────────────────────────────────────────────────────
# 7. Create DMG
# ─────────────────────────────────────────────────────────────

echo "==> Creating DMG..."
rm -rf "$DMG_DIR"
mkdir -p "$DMG_DIR"
cp -R "$APP_PATH" "$DMG_DIR/"
ln -s /Applications "$DMG_DIR/Applications"

DMG_PATH="$BUILD_DIR/Druse-${VERSION}.dmg"
rm -f "$DMG_PATH"

hdiutil create -volname "Druse ${VERSION}" \
    -srcfolder "$DMG_DIR" \
    -ov -format UDZO \
    -imagekey zlib-level=9 \
    "$DMG_PATH"

rm -rf "$DMG_DIR"

DMG_SIZE=$(du -sh "$DMG_PATH" | awk '{print $1}')
echo ""
echo "==> Done! DMG created:"
echo "    $DMG_PATH ($DMG_SIZE)"
echo ""
echo "    To notarize (requires Developer ID):"
echo "    xcrun notarytool submit $DMG_PATH --apple-id YOU --team-id TEAM --password APP_PWD --wait"
echo "    xcrun stapler staple $DMG_PATH"
