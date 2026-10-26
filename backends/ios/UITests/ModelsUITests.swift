import XCTest

final class ModelsUITests: XCTestCase {
    func testDefaultAndSavedTwoModelChoice() {
        let app = XCUIApplication()
        app.launchArguments = ["-modelSetup"]
        app.launch()

        XCTAssertTrue(app.staticTexts["Chat and decisions"].waitForExistence(timeout: 15))
        XCTAssertTrue(app.staticTexts["Kev-4B"].exists)
        XCTAssertFalse(app.staticTexts["Qwen3.5-4B"].exists)
        XCTAssertTrue(app.buttons["models.mode.one"].exists)
        XCTAssertTrue(app.buttons["models.mode.two"].exists)

        app.buttons["models.mode.two"].tap()
        XCTAssertTrue(app.staticTexts["Reopen Eureka to apply this model choice."].exists)
        app.terminate()
        app.launch()

        XCTAssertTrue(app.staticTexts["Qwen3.5-4B"].waitForExistence(timeout: 15))
        XCTAssertTrue(app.staticTexts["Kev-4B"].exists)
        XCTAssertTrue(app.staticTexts["Chat"].exists)
        XCTAssertFalse(app.staticTexts["Chat and decisions"].exists)
        // UX2-IOSNAMES: put the default (one model) back, so a second run on the same simulator starts where this one did.
        app.buttons["models.mode.one"].tap()
    }
}
