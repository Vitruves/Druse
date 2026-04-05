#!/bin/bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────
# bundle_dmg.sh — Create a fully portable DMG for Druse
#
# Recursively discovers all non-system dylib dependencies,
# copies them into Druse.app/Contents/Frameworks, and rewrites
# all load paths so the app needs zero Homebrew deps.
#
# Usage:  ./scripts/bundle_dmg.sh [path/to/Druse.app]
# Env:    CODESIGN_IDENTITY="-" (default ad-hoc)
# ─────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VERSION=$(cat "$PROJECT_DIR/VERSION")
BUILD_DIR="$PROJECT_DIR/build"
DMG_DIR="$BUILD_DIR/dmg"

CODESIGN_IDENTITY="${CODESIGN_IDENTITY:--}"

# Temp files
SEEN_FILE=$(mktemp)
QUEUE_FILE=$(mktemp)
trap "rm -f $SEEN_FILE $QUEUE_FILE" EXIT

# ── Build or locate the .app ──────────────────────────────────

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
MAIN_EXEC="$APP_PATH/Contents/MacOS/Druse"
RESOURCES_DIR="$APP_PATH/Contents/Resources"
mkdir -p "$FRAMEWORKS_DIR"

# ── Verify required bundled resources ────────────────────────
echo "==> Checking bundled resources..."
MISSING=0
if [ ! -f "$RESOURCES_DIR/dun2010bbdep.bin" ]; then
    echo "  MISSING: dun2010bbdep.bin (Dunbrack rotamer library)"
    MISSING=1
fi
if [ ! -f "$RESOURCES_DIR/standalone/index.html" ]; then
    echo "  MISSING: standalone/index.html (Ketcher chemical editor)"
    MISSING=1
fi
if [ ! -d "$RESOURCES_DIR/standalone/static/js" ]; then
    echo "  MISSING: standalone/static/js/ (Ketcher JS bundles)"
    MISSING=1
fi
if [ "$MISSING" -eq 1 ]; then
    echo ""
    echo "  ERROR: Required resources are missing from the app bundle."
    echo "  Ensure Resources/dun2010bbdep.bin and Resources/Ketcher/standalone/ exist"
    echo "  in the source tree before building. See README.md for details."
    exit 1
fi
echo "  OK — all required resources present"

# ── Helper: resolve a path to its real file ───────────────────

realfile() {
    python3 -c "import os,sys; print(os.path.realpath(sys.argv[1]))" "$1"
}

# ── Helper: check if a dep is a system lib we should skip ─────

is_system_lib() {
    case "$1" in
        /usr/lib/*) return 0 ;;
        /System/*) return 0 ;;
        @executable_path/*) return 0 ;;
    esac
    return 1
}

# LLVM's libc++ / libc++abi / libunwind should NOT be bundled.
# OpenMM links against them, but macOS ships compatible system versions.
# We'll rewrite OpenMM to use system libc++ instead.
is_llvm_cxx_lib() {
    case "$1" in
        */llvm/lib/c++/*) return 0 ;;
        */libc++.1*|*/libc++abi*|*/libunwind*)
            # Only skip if it's LLVM's copy, not the system one
            case "$1" in
                /usr/lib/*) return 1 ;;
                *llvm*|*homebrew*) return 0 ;;
            esac
            ;;
    esac
    # Also check @rpath refs that resolve to LLVM's libc++
    case "$1" in
        @rpath/libc++.1*|@rpath/libc++abi*|@rpath/libunwind*) return 0 ;;
    esac
    return 1
}

# Known rpath search directories (where @rpath and @loader_path resolve to)
RPATH_DIRS="/opt/homebrew/lib /opt/homebrew/opt/rdkit/lib /opt/homebrew/opt/boost/lib /opt/homebrew/opt/coordgen/lib /opt/homebrew/opt/maeparser/lib /opt/homebrew/opt/tbb/lib /opt/homebrew/opt/llvm/lib /opt/homebrew/opt/llvm/lib/c++ /opt/homebrew/opt/icu4c@78/lib /opt/homebrew/opt/xz/lib /opt/homebrew/opt/zstd/lib /opt/homebrew/opt/freetype/lib /opt/homebrew/opt/cairo/lib /opt/homebrew/opt/libpng/lib /opt/homebrew/opt/fontconfig/lib /opt/homebrew/opt/libx11/lib /opt/homebrew/opt/libxext/lib /opt/homebrew/opt/libxrender/lib /opt/homebrew/opt/libxcb/lib /opt/homebrew/opt/pixman/lib /opt/homebrew/opt/libxau/lib /opt/homebrew/opt/libxdmcp/lib /opt/homebrew/opt/gettext/lib $PROJECT_DIR/CppCore/build/_deps/openmm-install/lib"

# ── Helper: resolve a dependency path to a real file ──────────
# Handles /opt/homebrew paths, build paths, etc.

resolve_dep_path() {
    local dep="$1"

    # Direct absolute path
    if [ -f "$dep" ] || [ -L "$dep" ]; then
        realfile "$dep"
        return 0
    fi

    # @rpath or @loader_path — search known directories
    case "$dep" in
        @rpath/*|@loader_path/*)
            local basename_dep
            basename_dep=$(basename "$dep")
            for dir in $RPATH_DIRS; do
                if [ -f "$dir/$basename_dep" ] || [ -L "$dir/$basename_dep" ]; then
                    realfile "$dir/$basename_dep"
                    return 0
                fi
            done
            ;;
    esac

    return 1
}

# ─────────────────────────────────────────────────────────────
# 1. Recursively discover all non-system dylib dependencies
# ─────────────────────────────────────────────────────────────

echo "==> Discovering dependencies recursively..."

# Seed the queue with the main executable
echo "$MAIN_EXEC" > "$QUEUE_FILE"
> "$SEEN_FILE"

collect_deps() {
    local binary="$1"

    otool -L "$binary" 2>/dev/null | tail -n +2 | awk '{print $1}' | while IFS= read -r dep; do
        # Skip system libs
        if is_system_lib "$dep"; then
            continue
        fi

        # Skip LLVM's libc++ chain (will rewrite to system libc++ later)
        if is_llvm_cxx_lib "$dep"; then
            continue
        fi

        # Resolve to real file
        local real=""
        real=$(resolve_dep_path "$dep" 2>/dev/null || true)
        if [ -z "$real" ]; then
            continue
        fi

        # Skip if already seen
        if grep -qxF "$real" "$SEEN_FILE" 2>/dev/null; then
            continue
        fi

        echo "$real" >> "$SEEN_FILE"
        echo "$real"
    done
}

# BFS: process queue until empty
iteration=0
while true; do
    iteration=$((iteration + 1))
    new_queue=$(mktemp)
    found_new=0

    while IFS= read -r binary; do
        for dep in $(collect_deps "$binary"); do
            echo "$dep" >> "$new_queue"
            found_new=1
        done
    done < "$QUEUE_FILE"

    if [ "$found_new" -eq 0 ]; then
        rm -f "$new_queue"
        break
    fi

    mv "$new_queue" "$QUEUE_FILE"

    if [ "$iteration" -gt 20 ]; then
        echo "WARNING: dependency resolution exceeded 20 iterations, stopping"
        break
    fi
done

DYLIB_COUNT=$(wc -l < "$SEEN_FILE" | tr -d ' ')
echo "  Found $DYLIB_COUNT dylibs to bundle"

# ─────────────────────────────────────────────────────────────
# 2. Copy dylibs into Frameworks/
# ─────────────────────────────────────────────────────────────

echo "==> Copying dylibs to Frameworks/..."
while IFS= read -r dylib; do
    dest_name=$(basename "$dylib")
    cp -f "$dylib" "$FRAMEWORKS_DIR/$dest_name"
    chmod 755 "$FRAMEWORKS_DIR/$dest_name"
    echo "  $dest_name"
done < "$SEEN_FILE"

# ─────────────────────────────────────────────────────────────
# 3. Create versioned symlinks BEFORE rewriting paths
#    (so install_name_tool -change can match the target names)
# ─────────────────────────────────────────────────────────────

echo "==> Creating symlinks..."
cd "$FRAMEWORKS_DIR"
for dylib in *.dylib; do
    [ -L "$dylib" ] && continue

    # Extract the install name to figure out what symlinks are needed
    install_id=$(otool -D "$FRAMEWORKS_DIR/$dylib" 2>/dev/null | tail -1)
    if [ -n "$install_id" ]; then
        id_base=$(basename "$install_id")
        if [ "$id_base" != "$dylib" ] && [ ! -e "$id_base" ]; then
            ln -sf "$dylib" "$id_base"
            echo "  $id_base -> $dylib"
        fi
    fi

    # Also create unversioned symlinks (e.g. libRDKitFoo.dylib -> libRDKitFoo.2025.09.6.dylib)
    # Strip version numbers: lib<name>.<version>.dylib -> lib<name>.dylib
    case "$dylib" in
        libRDKit*.*.*.*.dylib)
            unversioned=$(echo "$dylib" | sed 's/\.[0-9][0-9]*\.[0-9]*\.[0-9]*\.dylib/.dylib/')
            if [ ! -e "$unversioned" ]; then
                ln -sf "$dylib" "$unversioned"
            fi
            ;;
    esac
done
cd "$PROJECT_DIR"

# ─────────────────────────────────────────────────────────────
# 4. Rewrite load paths with install_name_tool
# ─────────────────────────────────────────────────────────────

echo "==> Rewriting load paths..."

# Build a mapping of original dep path -> @rpath/bundled_name
# We need to handle cases where the dep is referenced by various paths
# (absolute, @rpath, @loader_path) but the bundled file may have a
# different name (versioned vs unversioned).

rewrite_binary() {
    local binary="$1"

    # Change install name of the dylib itself
    local current_id
    current_id=$(otool -D "$binary" 2>/dev/null | tail -1 || true)
    case "$current_id" in
        ""|*"not an object"*|*"is not an object"*) ;;
        *)
            install_name_tool -id "@rpath/$(basename "$current_id")" "$binary" 2>/dev/null || true
            ;;
    esac

    # Rewrite each dependency
    otool -L "$binary" 2>/dev/null | tail -n +2 | awk '{print $1}' | while IFS= read -r dep; do
        case "$dep" in
            /usr/lib/*|/System/*) continue ;;
        esac

        # Rewrite LLVM's libc++ references to system libc++
        if is_llvm_cxx_lib "$dep"; then
            local dep_base
            dep_base=$(basename "$dep")
            case "$dep_base" in
                libc++.1*|libc++.dylib)
                    install_name_tool -change "$dep" "/usr/lib/libc++.1.dylib" "$binary" 2>/dev/null || true
                    ;;
                libc++abi*)
                    install_name_tool -change "$dep" "/usr/lib/libc++abi.dylib" "$binary" 2>/dev/null || true
                    ;;
                libunwind*)
                    # System libunwind is part of libSystem, no explicit ref needed
                    # but rewrite to system path just in case
                    install_name_tool -change "$dep" "/usr/lib/libunwind.dylib" "$binary" 2>/dev/null || true
                    ;;
            esac
            continue
        fi

        local dep_base
        dep_base=$(basename "$dep")

        # Check if we have this file (or a symlink to it) in Frameworks
        if [ -e "$FRAMEWORKS_DIR/$dep_base" ]; then
            install_name_tool -change "$dep" "@rpath/$dep_base" "$binary" 2>/dev/null || true
        else
            # Try to find it by resolving the original path and matching basenames
            local real_base=""
            if [ -f "$dep" ] || [ -L "$dep" ]; then
                real_base=$(basename "$(realfile "$dep")")
            fi
            if [ -n "$real_base" ] && [ -e "$FRAMEWORKS_DIR/$real_base" ]; then
                # Create a symlink for the versioned name too
                if [ ! -e "$FRAMEWORKS_DIR/$dep_base" ]; then
                    ln -sf "$real_base" "$FRAMEWORKS_DIR/$dep_base"
                fi
                install_name_tool -change "$dep" "@rpath/$dep_base" "$binary" 2>/dev/null || true
            fi
        fi
    done
}

# Rewrite the main executable
rewrite_binary "$MAIN_EXEC"
install_name_tool -add_rpath "@executable_path/../Frameworks" "$MAIN_EXEC" 2>/dev/null || true

# Rewrite all bundled dylibs
for dylib in "$FRAMEWORKS_DIR"/*.dylib; do
    [ -L "$dylib" ] && continue
    rewrite_binary "$dylib"
done

# ── Second pass: ensure all @rpath refs within dylibs also point
#    to bundled files (catches cross-references between bundled libs)

echo "==> Verifying cross-references (second pass)..."
for dylib in "$FRAMEWORKS_DIR"/*.dylib; do
    [ -L "$dylib" ] && continue
    otool -L "$dylib" 2>/dev/null | tail -n +2 | awk '{print $1}' | while IFS= read -r dep; do
        # Skip LLVM libc++ refs (already rewritten to system paths)
        if is_llvm_cxx_lib "$dep"; then
            continue
        fi
        case "$dep" in
            @rpath/*)
                dep_base=$(basename "$dep")
                if [ ! -e "$FRAMEWORKS_DIR/$dep_base" ]; then
                    echo "  WARNING: missing @rpath target: $dep_base (from $(basename "$dylib"))"
                fi
                ;;
            /opt/homebrew/*)
                dep_base=$(basename "$dep")
                echo "  FIXING: $dep in $(basename "$dylib")"
                if [ -e "$FRAMEWORKS_DIR/$dep_base" ]; then
                    install_name_tool -change "$dep" "@rpath/$dep_base" "$dylib" 2>/dev/null || true
                else
                    # The file might be under a different versioned name
                    if [ -f "$dep" ] || [ -L "$dep" ]; then
                        real_base=$(basename "$(realfile "$dep")")
                        if [ -e "$FRAMEWORKS_DIR/$real_base" ]; then
                            ln -sf "$real_base" "$FRAMEWORKS_DIR/$dep_base" 2>/dev/null || true
                            install_name_tool -change "$dep" "@rpath/$dep_base" "$dylib" 2>/dev/null || true
                        fi
                    fi
                fi
                ;;
        esac
    done
done

# ─────────────────────────────────────────────────────────────
# 5. Ad-hoc codesign
# ─────────────────────────────────────────────────────────────

echo "==> Codesigning..."
for dylib in "$FRAMEWORKS_DIR"/*.dylib; do
    [ -L "$dylib" ] && continue
    codesign --force --sign "$CODESIGN_IDENTITY" "$dylib" 2>/dev/null || true
done

# Sign CoreML model if present
if [ -d "$RESOURCES_DIR/PocketDetector.mlmodelc" ]; then
    codesign --force --sign "$CODESIGN_IDENTITY" "$RESOURCES_DIR/PocketDetector.mlmodelc" 2>/dev/null || true
fi

# Sign the app (not --deep; Ketcher WASM/JS resources are data, not code)
codesign --force --sign "$CODESIGN_IDENTITY" "$APP_PATH" 2>/dev/null || true

# ─────────────────────────────────────────────────────────────
# 6. Final verification
# ─────────────────────────────────────────────────────────────

echo "==> Final verification..."
PROBLEMS=0

# Check main executable
BAD=$(otool -L "$MAIN_EXEC" 2>/dev/null | awk '{print $1}' | grep "/opt/homebrew" || true)
if [ -n "$BAD" ]; then
    echo "  FAIL: main executable still references Homebrew:"
    echo "$BAD" | sed 's/^/    /'
    PROBLEMS=1
fi

# Check all bundled dylibs
for dylib in "$FRAMEWORKS_DIR"/*.dylib; do
    [ -L "$dylib" ] && continue
    BAD=$(otool -L "$dylib" 2>/dev/null | awk '{print $1}' | grep "/opt/homebrew" || true)
    if [ -n "$BAD" ]; then
        echo "  FAIL: $(basename "$dylib") still references Homebrew:"
        echo "$BAD" | sed 's/^/    /'
        PROBLEMS=1
    fi
done

if [ "$PROBLEMS" -eq 0 ]; then
    echo "  OK — no Homebrew references remain"
else
    echo ""
    echo "  WARNING: Some Homebrew references remain. The DMG may not be portable."
    echo "  Re-run to attempt fixing, or inspect manually."
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
FWCOUNT=$(ls -1 "$FRAMEWORKS_DIR"/*.dylib 2>/dev/null | wc -l | tr -d ' ')
RESSIZE=$(du -sh "$RESOURCES_DIR" 2>/dev/null | awk '{print $1}')
echo "    Bundled $FWCOUNT dylibs in Frameworks/"
echo "    Resources: $RESSIZE (Ketcher, rotamer library, ML models)"
echo ""
echo "    To notarize (requires Developer ID):"
echo "    xcrun notarytool submit $DMG_PATH --apple-id YOU --team-id TEAM --password APP_PWD --wait"
echo "    xcrun stapler staple $DMG_PATH"
