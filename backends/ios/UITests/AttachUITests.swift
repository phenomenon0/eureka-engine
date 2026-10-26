import XCTest

/// ATTACHVIS: the photo path in the simulator, by real taps (scripts/attach-sim.sh runs it). The web composer's
/// attach button → its sheet → "Photo library" → iOS's own picker → the chip → Send → the photo in the bubble → the
/// look's reply; and the camera surface's "Send to chat" → the chip → Send → the reply. With `-hostEngines` (the
/// script's default) the reply comes from OUR engine through the Mac harness; without it, from the demo engines.
/// Screenshots go to $ATTACH_SHOTS (the script passes TEST_RUNNER_ATTACH_SHOTS).
@MainActor
final class AttachUITests: XCTestCase {
    private var app: XCUIApplication!
    private let env = ProcessInfo.processInfo.environment

    override func setUp() async throws {
        continueAfterFailure = false
        app = XCUIApplication()
        if let host = env["ATTACH_HOST_ENGINES"], !host.isEmpty {
            app.launchArguments += ["-hostEngines"] + host.split(separator: " ").map(String.init)
            // CAMCAP: the camera card captions through the same Mac harness (the simulator has no Metal for our engine).
            app.launchArguments += ["-camCaptionHost"] + host.split(separator: " ").map(String.init)
        }
        app.launchEnvironment["BEND_PORT_BASE"] = env["ATTACH_PORT_BASE"] ?? "9970"
        // IMAGETRUE: the camera is Look's viewfinder now; these tests drive the CAMCAP/ATTACHVIS surface (Describe, Send to chat).
        app.launchArguments += ["-camVerdict"]
        // UX2-IOSNAMES: the first-run model sheet (a simulator has no models) covers the home and its drawer; not what these cover.
        app.launchArguments += ["-noModelSetup"]
    }

    private func shot(_ name: String) {
        let png = XCUIScreen.main.screenshot().pngRepresentation
        let attachment = XCTAttachment(data: png, uniformTypeIdentifier: "public.png")
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
        if let dir = env["ATTACH_SHOTS"], !dir.isEmpty {
            try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
            try? png.write(to: URL(fileURLWithPath: dir).appendingPathComponent("\(name).png"))
        }
    }

    /// The tree as XCUITest sees it, next to the screenshots (for a step that didn't find its element).
    private func dump(_ name: String) {
        shot(name)
        if let dir = env["ATTACH_SHOTS"], !dir.isEmpty {
            try? app.debugDescription.write(toFile: "\(dir)/\(name).txt", atomically: true, encoding: .utf8)
        }
    }

    /// The keyboard down before a reply's screenshot (the composer takes focus back when a reply lands): WebKit's ✓.
    private func hideKeyboard() {
        let done = app.toolbars.buttons["Done"].exists ? app.toolbars.buttons["Done"] : app.buttons["Done"]
        if app.keyboards.count > 0, done.exists { done.tap() }
        Thread.sleep(forTimeInterval: 0.8)
    }

    private func ready() -> XCUIElement {
        let attach = app.buttons["Add a photo"]
        XCTAssertTrue(attach.waitForExistence(timeout: 30), "the composer's photo button")
        let enabled = NSPredicate(format: "isEnabled == true")
        expectation(for: enabled, evaluatedWith: attach)
        waitForExpectations(timeout: 60)
        return attach
    }

    /// UX2-IOSNAMES: the header's old camera button is gone (HOMEREV1/2): the camera surface is
    /// the drawer switcher's Camera tab now ("Camera: Shoot, then ask"). The menu's label carries alarm state, so prefix-match.
    private func cameraTab() -> XCUIElement {
        let menu = app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Menu:'")).firstMatch
        if menu.waitForExistence(timeout: 10) { menu.tap() }
        return app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Camera:'")).firstMatch
    }

    /// UX2-CARRY (OPENFRAME): the thread prints no model tag any more (time on gaps only), so "· vision" never shows. A
    /// look's answer is its card ("Answer: <question>"), whose Copy verb is there once the answer has landed: one per reply.
    private var landedAnswers: Int {
        app.otherElements.matching(NSPredicate(format: "label BEGINSWITH 'Answer:'")).allElementsBoundByIndex.filter { $0.buttons["Copy"].exists }.count
    }

    /// The thread's newest photo object (its card is "Photo: <title>"; the thread keeps earlier runs' objects above it).
    private var newestObject: XCUIElement {
        app.otherElements.matching(NSPredicate(format: "label BEGINSWITH 'Photo: '")).allElementsBoundByIndex.last ?? app.otherElements["no photo object"]
    }

    /// A new look reply has landed (one more landed answer than `before`); an error reply fails the test with its words.
    private func waitForReply(after before: Int, timeout: TimeInterval) {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if landedAnswers > before { return }
            // UX2-CARRY: + the one-Kev harness's words ("Camera vision needs Two models in Settings").
            let refused = app.staticTexts.matching(NSPredicate(format: "label CONTAINS[c] 'vision model' OR label CONTAINS[c] 'can\\'t look' OR label CONTAINS[c] 'Two models'")).firstMatch
            if refused.exists { XCTFail("the look was refused: \(refused.label)"); return }
            Thread.sleep(forTimeInterval: 0.5)
        }
        dump("debug-no-reply")
        XCTFail("no look reply within \(timeout) s")
    }

    private func send(_ text: String) {
        let box = app.textViews.firstMatch
        XCTAssertTrue(box.waitForExistence(timeout: 10))
        box.tap()
        box.typeText(text)
        app.buttons["Send"].firstMatch.tap()
    }

    func testPickChipSendReply() {
        app.launch()
        let attach = ready()
        shot("sim-attach-01-composer")
        attach.tap()
        // UX2-IOSNAMES (OPENFRAME): no web sheet any more; "Add a photo" is WebKit's file input, so iOS's own menu
        // (Photo Library / Choose Files) comes straight up.
        let menu = app.buttons["Photo Library"]
        XCTAssertTrue(menu.waitForExistence(timeout: 5), "iOS's Photo Library")
        shot("sim-attach-02-sheet")
        menu.tap()
        // iOS's photo picker (out of process): the first photo is the newest, the fixture added last.
        let photo = app.images.matching(identifier: "PXGGridLayout-Info").firstMatch
        XCTAssertTrue(photo.waitForExistence(timeout: 15), "iOS's photo picker")
        photo.tap()
        // iOS 27's picker confirms a selection with its ✓ ("Done"); older ones say "Add", or return on the tap.
        let done = app.buttons["Done"]
        if done.waitForExistence(timeout: 4) { done.tap() } else if app.buttons["Add"].exists { app.buttons["Add"].tap() }
        // Posted on pick as the thread's photo object, the ask bar bound to it (no composer chip any more).
        let object = app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Open the photo'")).firstMatch
        if !object.waitForExistence(timeout: 20) { dump("debug-no-object"); XCTFail("the photo object") }
        XCTAssertTrue(app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Asking about'")).firstMatch.exists, "the ask bar bound to it")
        hideKeyboard()
        shot("sim-attach-03-posted")
        guard env["ATTACH_HOST_ENGINES"]?.isEmpty == false else { return }
        let before = landedAnswers
        send(env["ATTACH_QUESTION"] ?? "What is in this photo?")
        waitForReply(after: before, timeout: 180)
        hideKeyboard()
        shot("sim-attach-06-reply")
    }

    /// Run against a shell without the tower (scripts/attach-sim.sh MISSING=1): the reply is the models-missing words,
    /// with Retry, within seconds. Skipped otherwise.
    func testMissingVision() throws {
        try XCTSkipUnless(env["ATTACH_EXPECT"] == "missing", "only against a shell without the vision tower")
        app.launch()
        let attach = ready()
        attach.tap()
        app.buttons["Photo library"].tap()
        let menu = app.buttons["Photo Library"]
        if menu.waitForExistence(timeout: 4) { menu.tap() }
        let photo = app.images.matching(identifier: "PXGGridLayout-Info").firstMatch
        XCTAssertTrue(photo.waitForExistence(timeout: 15), "iOS's photo picker")
        photo.tap()
        let done = app.buttons["Done"]
        if done.waitForExistence(timeout: 4) { done.tap() } else if app.buttons["Add"].exists { app.buttons["Add"].tap() }
        XCTAssertTrue(app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Preview the photo'")).firstMatch.waitForExistence(timeout: 20))
        let t0 = Date()
        send("Is there a cat?")
        let missing = app.staticTexts.matching(NSPredicate(format: "label CONTAINS[c] 'vision model isn'")).firstMatch
        XCTAssertTrue(missing.waitForExistence(timeout: 20), "the models-missing words")
        XCTAssertLessThan(Date().timeIntervalSince(t0), 20)
        XCTAssertTrue(app.buttons.matching(NSPredicate(format: "label CONTAINS 'Retry'")).firstMatch.exists, "Retry")
        hideKeyboard()
        shot("sim-attach-07-missing")
    }

    func testCameraSendToChat() {
        if let fixtures = env["ATTACH_CAM_FIXTURES"], !fixtures.isEmpty { app.launchArguments += ["-camFixtures", fixtures] }
        app.launch()
        _ = ready()
        let camera = cameraTab()
        XCTAssertTrue(camera.waitForExistence(timeout: 10), "the drawer's Camera tab")
        camera.tap()
        let sendToChat = app.buttons["camera.sendToChat"]
        XCTAssertTrue(sendToChat.waitForExistence(timeout: 15), "the camera surface's Send to chat")
        Thread.sleep(forTimeInterval: 2.5)  // a fixture frame on screen
        shot("sim-camera-01-surface")
        sendToChat.tap()
        // UX2-IOSNAMES (OPENFRAME): the shot is posted on send as the thread's photo object, the ask bar bound to it; no
        // composer chip any more. Its glance answer is the first look; with the harness, a question gets a vision reply.
        let photo = app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Open the photo'")).firstMatch
        XCTAssertTrue(photo.waitForExistence(timeout: 20), "the shot as the thread's photo object")
        XCTAssertTrue(app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Asking about'")).firstMatch.exists, "the ask bar bound to it")
        hideKeyboard()
        shot("sim-camera-02-posted")
        guard env["ATTACH_HOST_ENGINES"]?.isEmpty == false else { return }
        let before = landedAnswers
        send(env["ATTACH_CAM_QUESTION"] ?? "What do you see?")
        waitForReply(after: before, timeout: 180)
        hideKeyboard()
        shot("sim-camera-03-reply")
    }

    /// CAMCAP: the caption path by real taps: Describe on a frozen frame → the card (loading → the caption) → "Send photo +
    /// caption" → the photo and its caption in the thread, no look run. Needs the Mac harness (skipped on the demo engines).
    func testCameraCaptionToChat() throws {
        try XCTSkipUnless(env["ATTACH_HOST_ENGINES"]?.isEmpty == false, "the caption comes from OUR engine through the Mac harness")
        if let fixtures = env["ATTACH_CAM_FIXTURES"], !fixtures.isEmpty { app.launchArguments += ["-camFixtures", fixtures] }
        app.launch()
        _ = ready()
        let camera = cameraTab()
        XCTAssertTrue(camera.waitForExistence(timeout: 10), "the drawer's Camera tab")
        camera.tap()
        let describe = app.buttons["camera.describe"]
        XCTAssertTrue(describe.waitForExistence(timeout: 15), "the surface's Describe")
        Thread.sleep(forTimeInterval: 2.5)  // a fixture frame on screen
        shot("camcap-sim-01-live-frame")
        describe.tap()
        XCTAssertTrue(app.otherElements["camera.caption"].waitForExistence(timeout: 5) || app.staticTexts["camera.caption.loading"].exists, "the card appears at once")
        shot("camcap-sim-02-loading")
        let meta = app.staticTexts["camera.caption.meta"]
        XCTAssertTrue(meta.waitForExistence(timeout: 240), "the caption finished (its time shows)")
        let caption = app.staticTexts["camera.caption.text"].label
        XCTAssertFalse(caption.isEmpty, "the caption has words")
        XCTAssertFalse(app.staticTexts["camera.caption.error"].exists, "no error card")
        shot("camcap-sim-03-caption-card")
        try? caption.write(toFile: "\(env["ATTACH_SHOTS"] ?? "/tmp")/camcap-sim-caption.txt", atomically: true, encoding: .utf8)
        let send = app.buttons["camera.sendToChat"]
        XCTAssertTrue(send.exists && send.label.contains("caption"), "Send says it carries the caption: \(send.label)")
        send.tap()
        // UX2-CARRY (OPENFRAME): the shot is the thread's photo object and the caption is its glance, inside the card (no
        // model tag is printed any more); it neither waits on a look nor runs one.
        let bubble = app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Open the photo'")).firstMatch
        XCTAssertTrue(bubble.waitForExistence(timeout: 20), "the photo in the thread")
        let posted = app.staticTexts[caption]
        XCTAssertTrue(posted.waitForExistence(timeout: 10), "the caption as the reply: \(caption)")
        let object = newestObject
        if !object.staticTexts[caption].exists { dump("debug-caption-object") }
        XCTAssertTrue(object.staticTexts[caption].exists, "the caption is the shot's glance, in its object: \(object.label)")
        XCTAssertFalse(object.staticTexts["Waiting"].exists, "the object doesn't wait on a look")
        XCTAssertFalse(object.staticTexts["Looking at the photo"].exists, "no look runs for a captioned shot")
        hideKeyboard()
        shot("camcap-sim-04-chat-photo-and-caption")
    }

    /// CAMCAP: a build that can't caption says so on the card (the simulator without the harness), the shot still goes to
    /// the chat as before. Skipped when the harness is on.
    func testCameraCaptionUnavailable() throws {
        try XCTSkipUnless(env["ATTACH_HOST_ENGINES"]?.isEmpty != false, "only without the Mac harness")
        if let fixtures = env["ATTACH_CAM_FIXTURES"], !fixtures.isEmpty { app.launchArguments += ["-camFixtures", fixtures] }
        app.launch()
        _ = ready()
        let camera = cameraTab()
        XCTAssertTrue(camera.waitForExistence(timeout: 10))
        camera.tap()
        let describe = app.buttons["camera.describe"]
        XCTAssertTrue(describe.waitForExistence(timeout: 15))
        Thread.sleep(forTimeInterval: 2.5)
        describe.tap()
        let why = app.staticTexts["camera.caption.unavailable"]
        XCTAssertTrue(why.waitForExistence(timeout: 5), "the card says why")
        XCTAssertTrue(why.label.contains("Simulator"), why.label)
        XCTAssertEqual(app.buttons["camera.sendToChat"].label, "Send to chat", "no caption to carry: the plain send")
        shot("camcap-sim-05-unavailable")
    }
}
