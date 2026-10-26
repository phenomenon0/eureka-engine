import XCTest

/// VOICESHOP: the voice picker and the voice shop by real taps (scripts/voices-uitest.sh runs it): ▶ on a bundled voice,
/// tap another to use it (the checkmark moves), More voices → Download a catalog voice → it installs → Use this voice →
/// ✓ In use → back in the picker it is ticked. Shots go to $VOICES_SHOTS (TEST_RUNNER_VOICES_SHOTS).
@MainActor
final class VoicesUITests: XCTestCase {
    private var app: XCUIApplication!
    private let env = ProcessInfo.processInfo.environment

    override func setUp() async throws {
        continueAfterFailure = false
        app = XCUIApplication()
        app.launchArguments += ["-noModelSetup", "-voiceSilent", "-voices", "-voicesCatalogURL", env["VOICES_CATALOG"] ?? "http://127.0.0.1:8961/catalog.json"]
    }

    private func shot(_ name: String) {
        let png = XCUIScreen.main.screenshot().pngRepresentation
        let attachment = XCTAttachment(data: png, uniformTypeIdentifier: "public.png")
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
        if let dir = env["VOICES_SHOTS"], !dir.isEmpty {
            try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
            try? png.write(to: URL(fileURLWithPath: dir).appendingPathComponent("\(name).png"))
        }
    }

    private func button(_ id: String) -> XCUIElement { app.buttons[id].firstMatch }

    private func waitSelected(_ id: String, _ seconds: Double = 30) -> Bool {
        let p = NSPredicate(format: "value == 'selected'")
        return XCTWaiter().wait(for: [expectation(for: p, evaluatedWith: button(id))], timeout: seconds) == .completed
    }

    func testPickPreviewAndShop() throws {
        app.launch()
        XCTAssertTrue(app.staticTexts["voices.title"].waitForExistence(timeout: 20), "Speech settings never showed")
        for id in ["af_heart", "af_bella", "af_nicole", "bf_emma", "am_michael", "am_fenrir", "am_puck", "bm_george"] {
            XCTAssertTrue(button("voices.use.kokoro:\(id)").exists, "bundled voice \(id) missing from the picker")
        }
        XCTAssertTrue(waitSelected("voices.use.kokoro:af_bella", 10), "a fresh install should start on the house voice, Bella")
        shot("ui-01-picker")

        // ▶ on Heart: the same demo line, rendered on the phone (the row's button turns to ■ while it plays)
        button("voices.preview.kokoro:af_heart").tap()
        sleep(3)
        shot("ui-02-preview-heart")

        // tap Michael: the Speaker swaps (no restart), the tick moves
        button("voices.use.kokoro:am_michael").tap()
        XCTAssertTrue(waitSelected("voices.use.kokoro:am_michael"), "Michael never became the voice")
        shot("ui-03-picked-michael")

        // More voices → Download Sarah → installed → Use this voice → ✓ In use
        let more = button("voices.more")
        while !more.isHittable { app.swipeUp() }
        // UX2-CARRY: SOUND-DESIGN's ambience row under More voices (a Toggle: XCUITest's switch, on or off).
        let ambience = app.switches["voices.ambience"].firstMatch
        if !ambience.waitForExistence(timeout: 5), let dir = env["VOICES_SHOTS"], !dir.isEmpty {
            try? app.debugDescription.write(toFile: "\(dir)/debug-no-ambience.txt", atomically: true, encoding: .utf8)
        }
        XCTAssertTrue(ambience.exists, "the Sound — ambience row is missing")
        XCTAssertTrue(ambience.label.contains("ambience"), "the ambience row says: \(ambience.label)")
        XCTAssertTrue(["0", "1"].contains(ambience.value as? String ?? ""), "the ambience switch reads \(String(describing: ambience.value))")
        for _ in 0..<3 where !ambience.isHittable { app.swipeUp() }
        if let dir = env["VOICES_SHOTS"], !dir.isEmpty {
            try? "label: \(ambience.label)\nvalue: \(ambience.value ?? "")\nhittable: \(ambience.isHittable)\n".write(toFile: "\(dir)/ambience.txt", atomically: true, encoding: .utf8)
        }
        shot("ui-03b-ambience")
        for _ in 0..<3 where !more.isHittable { app.swipeDown() }
        more.tap()
        let download = button("shop.download.af_sarah")
        XCTAssertTrue(download.waitForExistence(timeout: 20), "the catalog never listed Sarah")
        shot("ui-04-shop")
        download.tap()
        let use = button("shop.use.af_sarah")
        XCTAssertTrue(use.waitForExistence(timeout: 60), "Sarah never finished installing")
        shot("ui-05-sarah-installed")
        use.tap()
        XCTAssertTrue(button("shop.inuse.af_sarah").waitForExistence(timeout: 30), "Sarah never became the voice")
        shot("ui-06-sarah-in-use")

        // back: the picker lists Sarah (from the shop), ticked
        app.navigationBars.buttons.element(boundBy: 0).tap()
        let sarah = button("voices.use.kokoro:af_sarah")
        XCTAssertTrue(sarah.waitForExistence(timeout: 10), "the picker doesn't list Sarah")
        while !sarah.isHittable { app.swipeUp() }
        XCTAssertTrue(waitSelected("voices.use.kokoro:af_sarah", 5), "Sarah isn't ticked in the picker")
        shot("ui-07-picker-sarah")
    }
}
