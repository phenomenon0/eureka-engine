import XCTest

/// BENDFIX-1: the owner's first-usage report, by a real finger on the phone (scripts/bendfix-device.sh runs these).
///
///   testSurfaceVoice    drawer → Calendar → the small lumen: a TAP (BENDFIX_GESTURE=tap, what the owner did) or a HOLD
///                       (=hold, 2.6 s); then the answer is waited for and shot. BENDFIX_ARGS adds launch arguments
///                       (`-surfaceFixture bendfix/thursday.wav`: the lumen hears that clip instead of the room).
///   testAnswersInPlace  drawer → Calendar → a typed question it can't read ("any tips for staying focused?") is
///                       answered in the answer card; Send → the calendar card in the chat → Open ▸ → the panel → an ask
///                       there is answered in the panel, and the panel is still up.
///
/// Shots are XCTAttachments (kept): scripts/bendfix-device.sh exports them from the .xcresult. Env vars come in as
/// TEST_RUNNER_BENDFIX_*.
@MainActor
final class BendfixUITests: XCTestCase {
    private var app: XCUIApplication!
    private let env = ProcessInfo.processInfo.environment

    override func setUp() async throws {
        continueAfterFailure = false
        app = XCUIApplication()
        app.launchArguments += ["-noModelSetup"]
        app.launchEnvironment["BEND_CONSOLE_FILE"] = "documents:bendfix-console.txt"
        if let extra = env["BENDFIX_ARGS"], !extra.isEmpty { app.launchArguments += extra.split(separator: " ").map(String.init) }
    }

    private func shot(_ name: String) {
        let attachment = XCTAttachment(data: XCUIScreen.main.screenshot().pngRepresentation, uniformTypeIdentifier: "public.png")
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    private func button(_ label: String) -> XCUIElement { app.buttons[label].firstMatch }
    private func starting(_ prefix: String) -> XCUIElement { app.buttons.matching(NSPredicate(format: "label BEGINSWITH %@", prefix)).firstMatch }
    private func hittable(_ e: XCUIElement, _ seconds: Double = 8) -> Bool {
        XCTWaiter().wait(for: [expectation(for: NSPredicate(format: "hittable == true"), evaluatedWith: e)], timeout: seconds) == .completed
    }
    private func texts(_ fragment: String) -> XCUIElement {
        app.staticTexts.matching(NSPredicate(format: "label CONTAINS[c] %@", fragment)).firstMatch
    }

    /// The chat, then the drawer's Calendar tile: the person's own calendar over the chat.
    private func openCalendar() {
        app.launch()
        let menu = starting("Menu")
        XCTAssertTrue(menu.waitForExistence(timeout: 45), "the home never showed (no menu button)")
        sleep(3)
        shot("1-chat")
        menu.tap()
        let tile = starting("Calendar:")
        XCTAssertTrue(hittable(tile), "no Calendar in the drawer")
        tile.tap()
        XCTAssertTrue(starting("Back to").waitForExistence(timeout: 8), "the Calendar did not open")
        sleep(2)
    }

    func testSurfaceVoice() throws {
        openCalendar()
        // r8 called it "Hold to ask the calendar"; BENDFIX-1 "Tap or hold to ask the calendar".
        let lumen = app.buttons.matching(NSPredicate(format: "label CONTAINS 'ask the calendar'")).firstMatch
        XCTAssertTrue(hittable(lumen, 20), "the calendar's small lumen never became ready")
        shot("2-calendar")
        let hold = env["BENDFIX_GESTURE"] == "hold"
        if hold { lumen.press(forDuration: 2.6) } else { lumen.tap() }
        sleep(1)
        shot("3-after-\(hold ? "hold" : "tap")")
        sleep(UInt32(env["BENDFIX_WAIT"].flatMap { UInt32($0) } ?? 9))
        shot("4-answer")
        sleep(6)
        shot("5-later")
        XCTAssertTrue(starting("Back to").exists, "the Calendar is no longer up")
    }

    func testAnswersInPlace() throws {
        openCalendar()
        let field = app.textFields["Ask your calendar"].firstMatch
        XCTAssertTrue(hittable(field), "no ask strip")
        field.tap()
        field.typeText("any tips for staying focused?\n")
        sleep(1)
        shot("2-calendar-thinking")
        sleep(UInt32(env["BENDFIX_WAIT"].flatMap { UInt32($0) } ?? 14))
        shot("3-calendar-answered-in-place")
        XCTAssertTrue(starting("Back to").exists, "the Calendar left (it should answer in place)")

        // Send the day to the chat (explicit: the person posts it), then open the card's room and ask there.
        starting("Send ").tap()
        let open = app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Open ' AND NOT (label BEGINSWITH 'Open in')")).firstMatch
        XCTAssertTrue(hittable(open, 10), "no Open ▸ on the calendar card in the chat")
        sleep(1)
        shot("4-chat-card")
        open.tap()
        sleep(2)
        shot("5-panel")
        let ask = app.textFields.matching(NSPredicate(format: "placeholderValue BEGINSWITH 'Ask about'")).firstMatch
        XCTAssertTrue(hittable(ask), "no ask bar in the panel")
        ask.tap()
        ask.typeText("which of these is the earliest?\n")
        sleep(1)
        shot("6-panel-asked")
        sleep(UInt32(env["BENDFIX_WAIT"].flatMap { UInt32($0) } ?? 14))
        shot("7-panel-answered")
        XCTAssertTrue(app.otherElements["object-panel"].exists || app.buttons["Back to the chat"].exists, "the panel closed (it should answer in place)")
    }
}
