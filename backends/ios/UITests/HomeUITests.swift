import XCTest

/// HOMEREV1 (DESIGN.md): the home by a real finger (scripts/home-uitest.sh runs it). The first screen; a slide right from
/// the left edge shows the drawer; a slide left on the chat's edge brings the chat back; the menu button opens it too;
/// its Map card opens the Map app; Voice is one tap from the first screen and back.
/// HOMEREV2: the drawer's switcher (Chat · Voice · Camera · Notes · Work) takes a thumb to Notes, and the edge slide brings
/// it back. Shots go to $HOME_SHOTS (TEST_RUNNER_HOME_SHOTS). ICONREVERT: one icon set (the original glyphs), so no
/// `-homeIcons` / `-homeHero` any more.
@MainActor
final class HomeUITests: XCTestCase {
    private var app: XCUIApplication!
    private let env = ProcessInfo.processInfo.environment

    override func setUp() async throws {
        continueAfterFailure = false
        app = XCUIApplication()
        app.launchArguments += ["-noModelSetup"]
    }

    private func shot(_ name: String) {
        let png = XCUIScreen.main.screenshot().pngRepresentation
        let attachment = XCTAttachment(data: png, uniformTypeIdentifier: "public.png")
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
        if let dir = env["HOME_SHOTS"], !dir.isEmpty {
            try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
            try? png.write(to: URL(fileURLWithPath: dir).appendingPathComponent("\(name).png"))
        }
    }

    private func button(_ label: String) -> XCUIElement { app.buttons[label].firstMatch }
    private func hittable(_ e: XCUIElement, _ seconds: Double = 5) -> Bool {
        XCTWaiter().wait(for: [expectation(for: NSPredicate(format: "hittable == true"), evaluatedWith: e)], timeout: seconds) == .completed
    }
    /// A thumb's slide across the middle of the screen, from `from` to `to` (fractions of the width).
    private func slide(_ from: CGFloat, _ to: CGFloat) {
        let start = app.coordinate(withNormalizedOffset: CGVector(dx: from, dy: 0.5))
        let end = app.coordinate(withNormalizedOffset: CGVector(dx: to, dy: 0.52))
        start.press(forDuration: 0.05, thenDragTo: end, withVelocity: .default, thenHoldForDuration: 0.05)
    }

    func testHomeDrawerChatCardVoice() throws {
        app.launch()
        let voice = button("Voice: talk with Eureka")
        XCTAssertTrue(voice.waitForExistence(timeout: 30), "the home never showed (no Voice disc in the composer)")
        sleep(4) // the first launch's one-time nudge, then rest
        shot("sim-home")

        // Slide right: the drawer. Its foot is the switcher (Camera among its tabs); the chat's edge is the way back.
        slide(0.03, 0.85)
        let back = button("Back to the chat")
        XCTAssertTrue(hittable(back), "the slide right did not open the drawer")
        XCTAssertTrue(app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Camera:'")).firstMatch.exists, "no Camera in the drawer")
        sleep(1)
        shot("sim-drawer")

        // HOMEREV2: the switcher's Notes tab is the Core, one thumb away; its back button brings the chat back.
        let notes = app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Core:'")).firstMatch
        XCTAssertTrue(hittable(notes), "no Notes tab in the switcher")
        notes.tap()
        sleep(1)
        shot("sim-switch-notes")
        let leave = button("Back to chat")
        XCTAssertTrue(hittable(leave), "the Notes tab did not open the Core")
        leave.tap()
        XCTAssertTrue(hittable(voice, 5), "the Core's back button did not bring the chat back")
        slide(0.03, 0.85)
        XCTAssertTrue(hittable(back), "the slide right did not open the drawer again")

        // Slide left on the chat's edge: the chat, in one motion.
        slide(0.95, 0.15)
        XCTAssertTrue(hittable(voice), "the slide left did not bring the chat back")

        // The menu button opens the same drawer (its label carries alarm state: "Menu: 1 alarm ringing"); its Map card
        // opens the Map app (UX2-IOSNAMES: the old map-card chat prefill is gone), and back.
        app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Menu:'")).firstMatch.tap()
        let map = button("Map: find places, build a trip")
        XCTAssertTrue(hittable(map), "the menu did not open the drawer")
        map.tap()
        let toChat = button("Back to Chat")
        XCTAssertTrue(hittable(toChat, 10), "the Map card did not open the Map app")
        XCTAssertTrue(button("Use my location").exists, "the Map app's Use my location")
        sleep(1)
        shot("sim-map")
        toChat.tap()

        // Voice: one tap from the chat (Hands-free / Hold), and back.
        button("New chat").tap()
        XCTAssertTrue(hittable(voice, 10))
        voice.tap()
        sleep(2)
        shot("sim-voice")
    }
}
