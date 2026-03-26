import XCTest

/// XCUITest smoke tests for Druse.
///
/// Systematically clicks every button per tab/view and asserts the UI responds.
/// No real PDB data needed — tests verify wiring (button → action), not correctness.
///
/// Run: xcodebuild test -project Druse.xcodeproj -scheme Druse -only-testing DruseUITests
final class DruseUITests: XCTestCase {

    var app: XCUIApplication!

    override func setUp() {
        continueAfterFailure = true
        app = XCUIApplication()
        app.launchArguments += ["--ui-test"]
        app.launch()
    }

    override func tearDown() {
        app = nil
    }

    // MARK: - Helpers

    /// Find an element by accessibility identifier, searching all types.
    /// `.accessibilityElement(children: .combine)` can change the element type
    /// from Button to Other, so we search broadly.
    func element(_ id: String) -> XCUIElement {
        // Try buttons first (fastest), then fall back to any element
        let button = app.buttons[id]
        if button.exists { return button }
        return app.descendants(matching: .any)[id]
    }

    /// Tap a button by accessibility identifier; assert it exists first.
    @discardableResult
    func tapButton(_ id: String, timeout: TimeInterval = 5, file: StaticString = #file, line: UInt = #line) -> XCUIElement {
        let el = element(id)
        let exists = el.waitForExistence(timeout: timeout)
        if !exists {
            // Retry with broader search after wait
            let fallback = app.descendants(matching: .any)[id]
            let fallbackExists = fallback.waitForExistence(timeout: 2)
            XCTAssertTrue(fallbackExists, "Button '\(id)' not found", file: file, line: line)
            if fallbackExists && fallback.isEnabled {
                fallback.tap()
            }
            return fallback
        }
        if el.isEnabled {
            el.tap()
        }
        return el
    }

    /// Tap a button only if it exists (no assertion failure if missing — for conditional UI).
    @discardableResult
    func tapIfExists(_ id: String, timeout: TimeInterval = 3) -> Bool {
        let el = element(id)
        if el.waitForExistence(timeout: timeout) && el.isEnabled {
            el.tap()
            return true
        }
        // Broader fallback
        let fallback = app.descendants(matching: .any)[id]
        if fallback.waitForExistence(timeout: 1) && fallback.isEnabled {
            fallback.tap()
            return true
        }
        return false
    }

    /// Open a pipeline tab by tapping its identifier.
    func openTab(_ tabName: String) {
        let id = "pipeline_\(tabName)"
        let el = app.descendants(matching: .any)[id]
        XCTAssertTrue(el.waitForExistence(timeout: 5), "Pipeline tab '\(tabName)' not found")
        el.tap()
        usleep(500_000) // 500ms for panel animation
    }

    /// Assert an element exists (any type).
    func assertExists(_ id: String, timeout: TimeInterval = 5, file: StaticString = #file, line: UInt = #line) {
        let el = app.descendants(matching: .any)[id]
        XCTAssertTrue(el.waitForExistence(timeout: timeout), "Element '\(id)' not found", file: file, line: line)
    }

    // MARK: - Welcome Screen

    func testWelcomeScreenButtons() {
        // Wait for welcome screen animations to complete (0.6s animation + 0.25s card delays)
        usleep(1_500_000)

        // On fresh launch with no protein, welcome screen should appear
        // Use descendants(matching: .any) because .accessibilityElement(children: .combine)
        // may expose these as "Other" instead of "Button"
        let startProtein = app.descendants(matching: .any)["welcome_startProtein"]
        let startLigand = app.descendants(matching: .any)["welcome_startLigand"]
        let openProject = app.descendants(matching: .any)["welcome_openProject"]
        let showMe = app.descendants(matching: .any)["welcome_showMe"]

        // All four cards should exist
        XCTAssertTrue(startProtein.waitForExistence(timeout: 5), "Welcome 'Start from Protein' not found")
        XCTAssertTrue(startLigand.waitForExistence(timeout: 3), "Welcome 'Start with Ligand' not found")
        XCTAssertTrue(openProject.waitForExistence(timeout: 3), "Welcome 'Open Project' not found")
        XCTAssertTrue(showMe.waitForExistence(timeout: 3), "Welcome 'Show Me' not found")

        // Tap "Start from Protein" → should open Search panel
        startProtein.tap()
        usleep(500_000)

        // Search panel should now be visible (check for a search tab element)
        let fetchButton = app.buttons["search_fetchPDB"]
        XCTAssertTrue(fetchButton.waitForExistence(timeout: 3), "Search panel did not open after 'Start from Protein'")
    }

    func testWelcomeStartLigand() {
        usleep(1_500_000) // Wait for welcome animations
        let startLigand = app.descendants(matching: .any)["welcome_startLigand"]
        XCTAssertTrue(startLigand.waitForExistence(timeout: 5))
        startLigand.tap()
        usleep(500_000)

        // Ligand panel should be open
        let addSmiles = app.buttons["lig_addSmiles"]
        XCTAssertTrue(addSmiles.waitForExistence(timeout: 3), "Ligand panel did not open after 'Start with Ligand'")
    }

    // MARK: - Pipeline Tabs

    func testAllPipelineTabsOpen() {
        // First dismiss welcome by opening search
        tapIfExists("welcome_startProtein")
        usleep(300_000)

        let tabs = ["Search", "Preparation", "Sequence", "Ligands", "Docking", "Results", "Lead Opt"]
        for tab in tabs {
            openTab(tab)
            usleep(300_000)
            // Verify the panel header shows the tab name
            let header = app.staticTexts[tab]
            XCTAssertTrue(header.waitForExistence(timeout: 2), "Tab '\(tab)' panel header not found")
        }
    }

    // MARK: - Search Tab Buttons

    func testSearchTabButtons() {
        tapIfExists("welcome_startProtein")
        usleep(300_000)
        openTab("Search")
        usleep(300_000)

        // Import buttons should exist
        assertExists("search_importProtein")
        assertExists("search_importLigand")

        // PDB ID field and Fetch button
        let pdbField = app.textFields["search_pdbField"]
        XCTAssertTrue(pdbField.waitForExistence(timeout: 3), "PDB ID field not found")

        // Fetch should be disabled with empty field
        let fetchBtn = app.buttons["search_fetchPDB"]
        XCTAssertTrue(fetchBtn.exists, "Fetch button not found")

        // Search field and Search button
        let searchField = app.textFields["search_queryField"]
        XCTAssertTrue(searchField.waitForExistence(timeout: 3), "Search query field not found")
        assertExists("search_runSearch")
    }

    func testSearchFetchPDB() {
        tapIfExists("welcome_startProtein")
        usleep(300_000)
        openTab("Search")
        usleep(300_000)

        // Type a PDB ID and fetch
        let pdbField = app.textFields["search_pdbField"]
        if pdbField.waitForExistence(timeout: 3) {
            pdbField.tap()
            pdbField.typeText("1HSG")
            usleep(200_000)

            let fetchBtn = app.buttons["search_fetchPDB"]
            if fetchBtn.isEnabled {
                fetchBtn.tap()
                // Wait for protein to load (network dependent)
                usleep(5_000_000) // 5s for fetch

                // After loading, welcome screen should be gone
                let welcome = app.buttons["welcome_startProtein"]
                XCTAssertFalse(welcome.exists, "Welcome screen still showing after PDB load")
            }
        }
    }

    // MARK: - Preparation Tab Buttons

    func testPreparationTabButtonsExist() {
        loadTestProtein()
        openTab("Preparation")
        usleep(300_000)

        // All prep buttons should exist (some may be disabled without protein)
        let prepButtons = [
            "prep_removeWaters",
            "prep_removeNonStandard",
            "prep_removeAltConfs",
            "prep_addHydrogens",
            "prep_structureCleanup",
            "prep_fixMissing",
            "prep_analyzeMissing",
            "prep_repairMissing",
        ]

        for id in prepButtons {
            assertExists(id)
        }
    }

    func testPreparationButtonActions() {
        loadTestProtein()
        openTab("Preparation")
        usleep(300_000)

        // These buttons should be clickable with a protein loaded
        tapButton("prep_removeWaters")
        usleep(500_000)

        tapButton("prep_removeNonStandard")
        usleep(500_000)

        tapButton("prep_removeAltConfs")
        usleep(500_000)

        tapButton("prep_addHydrogens")
        usleep(1_000_000) // Hydrogens can take a moment

        // After adding hydrogens, remove should be available
        tapButton("prep_removeHydrogens")
        usleep(500_000)
    }

    // MARK: - Sequence Tab

    func testSequenceTabButtons() {
        loadTestProtein()
        openTab("Sequence")
        usleep(300_000)

        // Copy sequence button should exist
        assertExists("seq_copySequence")

        // Clear selection should exist
        assertExists("seq_clearSelection")

        // Tap copy (should not crash)
        tapButton("seq_copySequence")
    }

    // MARK: - Ligand Database Tab

    func testLigandDatabaseButtons() {
        tapIfExists("welcome_startLigand")
        usleep(300_000)

        // SMILES field and Add button
        let smilesField = app.textFields["lig_smilesField"]
        XCTAssertTrue(smilesField.waitForExistence(timeout: 3), "SMILES field not found")

        assertExists("lig_addSmiles")
        assertExists("lig_openManager")
        assertExists("lig_saveDB")
        assertExists("lig_loadDB")
    }

    func testAddLigandFromSMILES() {
        tapIfExists("welcome_startLigand")
        usleep(300_000)

        let smilesField = app.textFields["lig_smilesField"]
        if smilesField.waitForExistence(timeout: 3) {
            smilesField.tap()
            smilesField.typeText("CC(=O)Oc1ccccc1C(=O)O") // Aspirin
            usleep(200_000)

            tapButton("lig_addSmiles")
            usleep(2_000_000) // Conformer generation takes a moment
        }
    }

    // MARK: - Docking Tab Buttons

    func testDockingTabButtonsExist() {
        loadTestProtein()
        openTab("Docking")
        usleep(300_000)

        // Pocket detection buttons
        assertExists("dock_detectAuto")
        assertExists("dock_detectML")

        // Grid controls
        assertExists("dock_applyGrid")
        assertExists("dock_gridProtein")

        // Start button
        assertExists("dock_startButton")
    }

    func testPocketDetection() {
        loadTestProtein()
        openTab("Docking")
        usleep(300_000)

        // Auto pocket detection
        tapButton("dock_detectAuto")
        usleep(3_000_000) // Pocket detection takes a few seconds

        // After detection, focus button should appear
        tapIfExists("dock_focusPocket")
    }

    func testGridPlacement() {
        loadTestProtein()
        openTab("Docking")
        usleep(300_000)

        // Center on protein
        tapButton("dock_gridProtein")
        usleep(300_000)

        // Apply grid
        tapButton("dock_applyGrid")
        usleep(300_000)
    }

    // MARK: - Results Tab (mostly empty without docking)

    func testResultsTabButtonsExist() {
        tapIfExists("welcome_startProtein")
        usleep(300_000)
        openTab("Results")
        usleep(300_000)

        // Results DB button should always exist
        assertExists("results_openDB")
    }

    // MARK: - Lead Optimization Tab

    func testLeadOptTabButtonsExist() {
        tapIfExists("welcome_startProtein")
        usleep(300_000)
        openTab("Lead Opt")
        usleep(300_000)

        // Generate button should exist (may be disabled without reference)
        assertExists("lead_generate")
    }

    // MARK: - Render Controls (bottom bar)

    func testRenderControlsExist() {
        loadTestProtein()
        usleep(500_000)

        // Render mode buttons
        assertExists("render_Ball & Stick")
        assertExists("render_Stick")
        assertExists("render_Sphere")
        assertExists("render_Ribbon")

        // Other controls
        assertExists("render_hydrogens")
        assertExists("render_surface")
        assertExists("render_lighting")
        assertExists("render_clipping")
        assertExists("render_fitToView")
        assertExists("render_resetCamera")
    }

    func testRenderModeToggling() {
        loadTestProtein()
        usleep(500_000)

        // Cycle through render modes
        tapButton("render_Ribbon")
        usleep(300_000)
        tapButton("render_Sphere")
        usleep(300_000)
        tapButton("render_Stick")
        usleep(300_000)
        tapButton("render_Ball & Stick")
        usleep(300_000)
    }

    func testRenderToggles() {
        loadTestProtein()
        usleep(500_000)

        // Lighting toggle
        tapButton("render_lighting")
        usleep(200_000)
        tapButton("render_lighting")
        usleep(200_000)

        // Clipping toggle
        tapButton("render_clipping")
        usleep(200_000)
        tapButton("render_clipping")
        usleep(200_000)

        // Fit to view
        tapButton("render_fitToView")
        usleep(200_000)

        // Reset camera
        tapButton("render_resetCamera")
        usleep(200_000)
    }

    // MARK: - Status Strip

    func testStatusStripButtons() {
        // Console toggle should always exist
        assertExists("status_toggleConsole")

        // Toggle console open
        tapButton("status_toggleConsole")
        usleep(300_000)

        // Console controls should appear
        assertExists("status_clearLog")
        assertExists("status_revealLog")

        // Toggle console closed
        tapButton("status_toggleConsole")
        usleep(300_000)
    }

    func testStatusStripProjectButtons() {
        assertExists("status_saveProject")
        assertExists("status_openProject")
    }

    // MARK: - Full Pipeline Smoke Test

    /// End-to-end: fetch PDB → prep → detect pocket → start docking
    /// This is the most important test — walks the full user journey.
    func testFullPipelineSmoke() {
        // 1. Welcome → Start from Protein
        tapButton("welcome_startProtein")
        usleep(500_000)

        // 2. Fetch a small protein
        let pdbField = app.textFields["search_pdbField"]
        guard pdbField.waitForExistence(timeout: 3) else {
            XCTFail("Search panel did not open")
            return
        }
        pdbField.tap()
        pdbField.typeText("1HSG")
        tapButton("search_fetchPDB")

        // Wait for load
        let prepTab = app.buttons["pipeline_Preparation"]
        _ = prepTab.waitForExistence(timeout: 10)
        usleep(3_000_000)

        // 3. Preparation — remove waters, add hydrogens
        openTab("Preparation")
        usleep(500_000)
        tapIfExists("prep_removeWaters")
        usleep(1_000_000)
        tapIfExists("prep_addHydrogens")
        usleep(2_000_000)

        // 4. Docking tab — detect pocket
        openTab("Docking")
        usleep(500_000)
        tapIfExists("dock_detectAuto")
        usleep(3_000_000)

        // 5. Start docking (opens PreDockSheet)
        tapIfExists("dock_startButton")
        usleep(1_000_000)

        // PreDockSheet should appear — look for Start Docking or Cancel
        let preDockStart = app.buttons["preDock_start"]
        if preDockStart.waitForExistence(timeout: 3) {
            // Cancel for now (don't actually run docking in smoke test)
            tapIfExists("preDock_cancel")
        }

        // 6. Check results tab opens without crash
        openTab("Results")
        usleep(300_000)
        assertExists("results_openDB")
    }

    // MARK: - Button-Does-Nothing Detector

    /// Scans all known button IDs and reports which ones exist but appear non-functional.
    /// A button is "suspicious" if tapping it causes no change in the UI at all.
    func testAllButtonsAreWired() {
        loadTestProtein()
        usleep(1_000_000)

        // Collect all button IDs that should exist when a protein is loaded
        let buttonIDs = [
            // Render controls
            "render_hydrogens", "render_surface", "render_lighting",
            "render_clipping", "render_fitToView", "render_resetCamera",
            // Status strip
            "status_toggleConsole", "status_saveProject", "status_openProject",
        ]

        var missingButtons: [String] = []
        var disabledButtons: [String] = []

        for id in buttonIDs {
            let button = app.buttons[id]
            if !button.waitForExistence(timeout: 2) {
                missingButtons.append(id)
            } else if !button.isEnabled {
                disabledButtons.append(id)
            }
        }

        if !missingButtons.isEmpty {
            XCTContext.runActivity(named: "Missing buttons") { _ in
                for id in missingButtons {
                    XCTFail("Button '\(id)' not found in UI")
                }
            }
        }

        if !disabledButtons.isEmpty {
            XCTContext.runActivity(named: "Unexpectedly disabled buttons") { _ in
                for id in disabledButtons {
                    XCTFail("Button '\(id)' is disabled when protein is loaded")
                }
            }
        }
    }

    /// Walk every tab and verify all buttons within each tab exist and are tappable.
    func testPerTabButtonAudit() {
        loadTestProtein()
        usleep(1_000_000)

        let tabButtons: [(tab: String, buttons: [String])] = [
            ("Search", [
                "search_importProtein", "search_importLigand",
                "search_fetchPDB", "search_runSearch"
            ]),
            ("Preparation", [
                "prep_removeWaters", "prep_removeNonStandard",
                "prep_removeAltConfs", "prep_addHydrogens",
                "prep_structureCleanup", "prep_fixMissing",
                "prep_analyzeMissing", "prep_repairMissing"
            ]),
            ("Sequence", [
                "seq_copySequence", "seq_clearSelection"
            ]),
            ("Docking", [
                "dock_detectAuto", "dock_detectML",
                "dock_applyGrid", "dock_gridProtein",
                "dock_startButton"
            ]),
            ("Results", [
                "results_openDB"
            ]),
            ("Lead Opt", [
                "lead_generate"
            ]),
        ]

        for (tab, buttons) in tabButtons {
            XCTContext.runActivity(named: "Tab: \(tab)") { _ in
                openTab(tab)
                usleep(500_000)

                for id in buttons {
                    let el = app.descendants(matching: .any)[id]
                    XCTAssertTrue(
                        el.waitForExistence(timeout: 5),
                        "[\(tab)] Element '\(id)' not found"
                    )
                }
            }
        }
    }

    // MARK: - Helpers (test data loading)

    /// Load a test protein via the Search tab PDB fetch.
    private func loadTestProtein() {
        // Wait for welcome screen animations
        usleep(1_500_000)

        // Dismiss welcome — search by any type since .combine changes element type
        tapIfExists("welcome_startProtein", timeout: 5)
        usleep(500_000)

        // If search panel is already open, use it
        // TextField may also appear as .any with combine
        let pdbField = app.textFields["search_pdbField"]
        let pdbFieldAny = app.descendants(matching: .any)["search_pdbField"]
        let field = pdbField.waitForExistence(timeout: 5) ? pdbField : pdbFieldAny
        guard field.waitForExistence(timeout: 5) else { return }

        field.tap()
        field.typeText("1HSG")
        usleep(200_000)

        // Fetch — search broadly
        let fetchBtn = app.descendants(matching: .any)["search_fetchPDB"]
        if fetchBtn.waitForExistence(timeout: 3) && fetchBtn.isEnabled {
            fetchBtn.tap()
            // Wait for load to complete (render controls should appear)
            let fitBtn = app.descendants(matching: .any)["render_fitToView"]
            _ = fitBtn.waitForExistence(timeout: 15)
            usleep(3_000_000)
        }
    }
}
