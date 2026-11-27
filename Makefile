# Druse build entry points. See Docs/Installation.md for the full build-from-source guide.
#
#   make            build the C++ core, generate the Xcode project, build the app (Release)
#   make help       list every target

CONFIG      ?= Release
DERIVED     := build
APP         := $(DERIVED)/Build/Products/$(CONFIG)/Druse.app
JOBS        ?= $(shell sysctl -n hw.ncpu)
CORE_BUILD  := CppCore/build
XCODEBUILD  := xcodebuild -project Druse.xcodeproj -scheme Druse -configuration $(CONFIG) -derivedDataPath $(DERIVED)

.DEFAULT_GOAL := all
.PHONY: all help deps core project app run headless eval test dmg version bump clean clean-core clean-all

all: app ## Build everything (core, Xcode project, app)

help: ## List targets
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  \033[1m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "  Variables: CONFIG=$(CONFIG) (Release|Debug)  JOBS=$(JOBS)  SCRIPT=file.druse  CMD=\"cmd; cmd\"  V=X.Y.Z-beta"

deps: ## Install Homebrew dependencies
	brew install rdkit boost eigen nanoflann tbb xcodegen

$(CORE_BUILD)/CMakeCache.txt:
	cmake -S CppCore -B $(CORE_BUILD) -DCMAKE_BUILD_TYPE=Release

core: $(CORE_BUILD)/CMakeCache.txt ## Build the C++ core (first run also builds OpenMM, several minutes)
	cmake --build $(CORE_BUILD) -j$(JOBS)

project: ## Generate Druse.xcodeproj with XcodeGen (skipped when project.yml and sources are unchanged)
	xcodegen generate --use-cache

app: core project ## Build Druse.app into build/Build/Products/<CONFIG>
	$(XCODEBUILD) build

run: app ## Build and launch the app
	open "$(APP)"

headless: app ## Run a DruseScript file without a window: make headless SCRIPT=file.druse
	@test -n "$(SCRIPT)" || { echo "usage: make headless SCRIPT=file.druse" >&2; exit 2; }
	"$(APP)/Contents/MacOS/Druse" --headless "$(SCRIPT)"

eval: app ## Run DruseScript commands without a window: make eval CMD="cmd; cmd"
	@test -n "$(CMD)" || { echo "usage: make eval CMD=\"command; command\"" >&2; exit 2; }
	"$(APP)/Contents/MacOS/Druse" --headless --eval "$(CMD)"

test: core project ## Run the unit tests (opens a GUI window)
	xcodebuild -project Druse.xcodeproj -scheme Druse -derivedDataPath $(DERIVED) -only-testing:DruseTests test

dmg: ## Build a Release app and package a portable DMG into build/
	$(MAKE) app CONFIG=Release
	./scripts/bundle_dmg.sh "$(DERIVED)/Build/Products/Release/Druse.app"

version: ## Sync README.md and project.yml from VERSION
	./update_readme_version.sh
	xcodegen generate

bump: ## Bump the version: make bump V=0.1.31-beta
	@test -n "$(V)" || { echo "usage: make bump V=X.Y.Z[-suffix]" >&2; exit 2; }
	./update_readme_version.sh --version $(V)
	xcodegen generate

clean: ## Remove the app build products (keeps the C++ core)
	rm -rf $(DERIVED)

clean-core: ## Remove the C++ core build (next build reconfigures and rebuilds OpenMM)
	rm -rf $(CORE_BUILD)

clean-all: clean clean-core ## Remove all build products
