#if BEND_LLB
import SwiftUI
import WebKit

/// PHONEUX: never a black screen. From the first SwiftUI frame until the web app has painted (it posts bendOS
/// `ui.ready` after its first frame), this sits over the web view: the wordmark, "Starting…" with the seconds since
/// launch, and, on the first screen, why the engine can't run here when it can't ("Engine files not set up …"). It
/// fades out on `ui.ready`. A page that hasn't painted after `slowAfter` seconds gets a Reload, and the overlay never
/// waits on the engine: warming shows on the page itself (src/app/BootPanel.tsx).
@MainActor @Observable
final class BootModel {
    static let shared = BootModel()
    static let slowAfter: Double = 8

    private(set) var pageUp = false
    /// The web view, for Reload.
    weak var webView: WKWebView?

    func pageReady(_ detail: String) {
        guard !pageUp else { return }
        pageUp = true
        EngineManager.console(String(format: "BEND boot ui.ready at %.0f ms since process start (%@)", Boot.sinceStartMs(), detail))
    }

    func reload() {
        EngineManager.console("BEND boot reload asked (the page had not painted)")
        webView?.reload()
    }

    /// One line under "Starting…" when the engine can't run on this phone, so the first screen says so.
    static var engineLine: String? {
        let missing = (Boot.setup()["missing"] as? [String]) ?? []
        let chatMissing = !EngineManager.hasModel(.chat)
        if EngineManager.shared.hostEngines { return "Engine: the Mac harness on the host (-hostEngines)." }
        #if targetEnvironment(simulator)
        let where_ = "Simulator: the engine needs an iPhone's GPU, so replies here are demo replies."
        return chatMissing ? "Engine files not set up — run the one-time setup.\n\(where_)" : where_
        #else
        if chatMissing { return "Engine files not set up — run the one-time setup.\n\(missing.count) file(s) missing in Documents/models." }
        return nil
        #endif
    }
}

struct BootOverlay: View {
    let model: BootModel
    private let line = BootModel.engineLine

    var body: some View {
        TimelineView(.periodic(from: .now, by: 0.1)) { context in
            let seconds = context.date.timeIntervalSince(Boot.processStart)
            // The wordmark at the screen's centre, where the launch screen and the page's own splash put it, so the
            // hand-over from one to the next doesn't move it; what's going on sits below it.
            ZStack {
                Color.bendBackground
                Text("Eureka")
                    .font(.system(size: 34, weight: .medium))
                    .tracking(-1.2)
                    .foregroundStyle(Color(white: 0.93))
                VStack(spacing: 16) {
                HStack(spacing: 10) {
                    ProgressView().controlSize(.small).tint(Color(white: 0.6))
                    Text(String(format: "Starting… %.1f s", seconds))
                        .font(.system(size: 13, weight: .medium).monospacedDigit())
                        .foregroundStyle(Color(white: 0.6))
                }
                .accessibilityIdentifier("boot.starting")
                if let line {
                    Text(line)
                        .font(.system(size: 13))
                        .multilineTextAlignment(.center)
                        .foregroundStyle(Color(white: 0.75))
                        .padding(.horizontal, 32)
                        .accessibilityIdentifier("boot.engine")
                }
                if seconds > BootModel.slowAfter {
                    Button("The screen is slow to load · Reload") { model.reload() }
                        .font(.system(size: 13, weight: .semibold))
                        .tint(Color(white: 0.85))
                        .accessibilityIdentifier("boot.reload")
                }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
                .padding(.top, UIScreen.main.bounds.height / 2 + 30)
            }
        }
        .ignoresSafeArea()
    }
}

/// What the web app reads at document start, always (engines on or not): the platform, the engine files, the
/// surfaces this build has, and the process's start, so the first screen can say what is going on. The engines'
/// own bridge (`__BEND_NATIVE__`, EngineManager.bridgeScript) is separate and only there when they run.
@MainActor
enum ShellInfo {
    static var script: String {
        #if targetEnvironment(simulator)
        let platform = "simulator"
        #else
        let platform = "device"
        #endif
        #if BEND_CAMERA
        let camera = true
        #else
        let camera = false
        #endif
        let engines = EngineManager.shared.enabled ? "on" : (EngineManager.shared.hostEngines ? "host" : "off")
        let info: [String: Any] = [
            "platform": platform,
            "engines": engines,
            "setup": Boot.setup(),
            // The camera surface and its photo source (PHONECAM), and voice mode (PHONEVOICE: ios/Bend/Voice, the web's
            // Voice screen), all in this merged build.
            "surfaces": ["camera": camera, "photo": camera, "voice": true],
            "processStart": Boot.processStart.timeIntervalSince1970 * 1000,
        ]
        let json = (try? JSONSerialization.data(withJSONObject: info)).map { String(decoding: $0, as: UTF8.self) } ?? "{}"
        return "window.__BEND_SHELL__ = Object.freeze(\(json));"
    }
}
#endif
