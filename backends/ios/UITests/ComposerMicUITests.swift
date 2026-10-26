import XCTest

/// UX-COMPOSER-MIC (G-5, RE-11): the composer's mic by a real finger (scripts/composer-mic-uitest.sh runs it). Idle: the
/// mic holds Send's place, enabled. With no recognizer (this simulator) a hold is a tap: the lumen opens and says why
/// (the script also screenshots the simulator mid-hold, on the "HOLD BEGIN" line).
/// Shots go to $MIC_SHOTS (TEST_RUNNER_MIC_SHOTS).
@MainActor
final class ComposerMicUITests: XCTestCase {
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
        if let dir = env["MIC_SHOTS"], !dir.isEmpty {
            try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
            try? png.write(to: URL(fileURLWithPath: dir).appendingPathComponent("\(name).png"))
            try? app.debugDescription.write(toFile: "\(dir)/\(name).txt", atomically: true, encoding: .utf8)
        }
    }

    func testComposerMicHoldAndTap() throws {
        app.launch()
        let mic = app.buttons["Voice: talk with Eureka"].firstMatch
        let found = mic.waitForExistence(timeout: 60)
        if !found { shot("sim-0-no-mic") }
        XCTAssertTrue(found, "no mic in the composer")
        expectation(for: NSPredicate(format: "isEnabled == true AND hittable == true"), evaluatedWith: mic)
        waitForExpectations(timeout: 90)
        sleep(1)
        shot("sim-1-idle")

        // The simulator has no speech recognizer (Voice.swift KnownText): the mic says so in its description, and a hold
        // is a tap there, so it opens the lumen, which tells you why. Hearing a hold is a device step (REPORT).
        print("COMPOSER-MIC HOLD BEGIN")
        mic.press(forDuration: 2.0)
        print("COMPOSER-MIC HOLD END")
        sleep(3)
        shot("sim-3-hold-no-recognizer")
    }
}
