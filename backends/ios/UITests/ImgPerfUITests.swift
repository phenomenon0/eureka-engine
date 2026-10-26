import XCTest

/// IMGPERF: the photo conversation the owner found slow, in the simulator by real taps, with the web's frame probe on
/// (-perfProbe) and the console lines in $IMGPERF_CONSOLE (BEND engine web timing …: per turn send→first token,
/// send→done, frames / gaps > 50 ms / worst gap; the photo's prepare). The engine is OUR engine on the Mac harness
/// (-hostEngines; the simulator's Metal can't run it). scripts/img-sim.sh runs it.
///   testPhotoConversation  pick (iOS's picker: the newest photo) → posted as the photo object, the ask bar bound →
///                          question → Send → reply; Open ▸ (the photo's panel) and back → a second question through the
///                          bound ask bar → reply; the photo let go → a plain text follow-up → reply
///   testTextControl        three text turns in a fresh chat (the no-image control)
@MainActor
final class ImgPerfUITests: XCTestCase {
    private var app: XCUIApplication!
    private let env = ProcessInfo.processInfo.environment

    override func setUp() async throws {
        continueAfterFailure = false
        app = XCUIApplication()
        // UX2-TESTS: as AttachUITests (UX2-IOSNAMES): a model-less sim's first-run sheet would cover the home.
        app.launchArguments += ["-perfProbe", "-noModelSetup"]
        if env["IMGPERF_OFF"] == "1" { app.launchArguments += ["-imgperfOff"] }  // the before: ATTACHVIS's photo wire
        if let host = env["IMGPERF_HOST_ENGINES"], !host.isEmpty {
            app.launchArguments += ["-hostEngines"] + host.split(separator: " ").map(String.init)
        }
        app.launchEnvironment["BEND_PORT_BASE"] = env["IMGPERF_PORT_BASE"] ?? "9995"
        if let console = env["IMGPERF_CONSOLE"], !console.isEmpty { app.launchEnvironment["BEND_CONSOLE_FILE"] = console }
    }

    private func shot(_ name: String) {
        let png = XCUIScreen.main.screenshot().pngRepresentation
        if let dir = env["IMGPERF_SHOTS"], !dir.isEmpty {
            try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
            try? png.write(to: URL(fileURLWithPath: dir).appendingPathComponent("\(name).png"))
        }
    }

    /// The accessibility tree, next to the shots (the probe: what the objects thread exposes at each stage).
    private func dump(_ name: String) {
        if let dir = env["IMGPERF_SHOTS"], !dir.isEmpty {
            try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
            try? app.debugDescription.write(toFile: "\(dir)/\(name).tree.txt", atomically: true, encoding: .utf8)
        }
    }

    private func ready() -> XCUIElement {
        let attach = app.buttons["Add a photo"]
        XCTAssertTrue(attach.waitForExistence(timeout: 30), "the composer's photo button")
        expectation(for: NSPredicate(format: "isEnabled == true"), evaluatedWith: attach)
        waitForExpectations(timeout: 60)
        return attach
    }

    /// Replies that have landed: the web's own `… send→done` timing lines in the console file (one per finished turn).
    private var finished: Int {
        guard let path = env["IMGPERF_CONSOLE"], let text = try? String(contentsOfFile: path, encoding: .utf8) else { return 0 }
        return text.components(separatedBy: "send→done").count - 1
    }

    private func waitForReply(after before: Int, timeout: TimeInterval = 180) {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if finished > before { Thread.sleep(forTimeInterval: 0.5); return }
            Thread.sleep(forTimeInterval: 0.25)
        }
        shot("debug-no-reply")
        if let dir = env["IMGPERF_SHOTS"] { try? app.debugDescription.write(toFile: "\(dir)/debug-no-reply.txt", atomically: true, encoding: .utf8) }
        XCTFail("no reply within \(timeout) s")
    }

    private func send(_ text: String) {
        let box = app.textViews.firstMatch
        XCTAssertTrue(box.waitForExistence(timeout: 10))
        box.tap()
        box.typeText(text)
        app.buttons["Send"].firstMatch.tap()
    }

    private func newChat() {
        let new = app.buttons["New chat"]
        if new.waitForExistence(timeout: 3) { new.tap() }
    }

    func testPhotoConversation() {
        app.launch()
        newChat()
        let attach = ready()
        attach.tap()
        // UX2-TESTS (OPENFRAME): no web sheet any more; "Add a photo" is WebKit's file input, so iOS's own menu
        // (Photo Library / Choose Files) comes straight up.
        let menu = app.buttons["Photo Library"]
        XCTAssertTrue(menu.waitForExistence(timeout: 5), "iOS's Photo Library")
        menu.tap()
        let photo = app.images.matching(identifier: "PXGGridLayout-Info").firstMatch
        XCTAssertTrue(photo.waitForExistence(timeout: 15), "iOS's photo picker")
        photo.tap()
        let done = app.buttons["Done"]
        if done.waitForExistence(timeout: 4) { done.tap() } else if app.buttons["Add"].exists { app.buttons["Add"].tap() }
        // Posted on pick as the thread's photo object, the ask bar bound to it (no composer chip any more).
        let object = app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Open the photo'")).firstMatch
        if !object.waitForExistence(timeout: 20) { dump("debug-no-object"); XCTFail("the photo object") }
        let bound = app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Asking about'")).firstMatch
        XCTAssertTrue(bound.exists, "the ask bar bound to it")
        dump("img-sim-00-posted")
        var before = finished
        send(env["IMGPERF_Q1"] ?? "What is the headline?")
        waitForReply(after: before)
        shot("img-sim-01-look")
        dump("img-sim-01-look")
        // The photo's room (Open ▸: the object panel, which took over from the sent photo's viewer), then back.
        object.tap()
        let back = app.buttons["Back to the chat"].firstMatch
        if !back.waitForExistence(timeout: 10) { dump("debug-no-panel"); XCTFail("the photo's panel") }
        shot("img-sim-01b-panel")
        dump("img-sim-01b-panel")
        back.tap()
        // A second question about the same photo: through the bound ask bar (a tap on the photo binds it again
        // if the first answer let it go; WebKit exposes that aria-pressed photo as a switch, not a button).
        let photoButton = app.switches.matching(NSPredicate(format: "label BEGINSWITH 'Ask about ' OR label BEGINSWITH 'Asking about '")).firstMatch
        if !bound.waitForExistence(timeout: 5) {
            XCTAssertTrue(photoButton.waitForExistence(timeout: 5), "the photo in its object")
            photoButton.tap()
        }
        if !bound.waitForExistence(timeout: 10) { dump("debug-not-bound"); XCTFail("the photo bound again") }
        before = finished
        send(env["IMGPERF_Q2"] ?? "What newspaper is this, and what is the date?")
        waitForReply(after: before)
        shot("img-sim-02-again")
        dump("img-sim-02-again")
        // A plain text follow-up: the photo let go first, so this is a chat turn, not a third look.
        let stop = app.buttons["Stop asking about this photo"]
        if stop.waitForExistence(timeout: 3) { stop.tap() }
        before = finished
        send(env["IMGPERF_F"] ?? "Why was that such big news at the time?")
        waitForReply(after: before)
        shot("img-sim-03-followup")
    }

    func testTextControl() {
        app.launch()
        newChat()
        _ = ready()
        for (i, q) in ["What is a good way to start running if I have never done it?", "How many times a week should I go at first?",
                       "Why do runners get side stitches?"].enumerated() {
            let before = finished
            send(q)
            waitForReply(after: before)
            if i == 2 { shot("img-sim-04-control") }
        }
    }
}
