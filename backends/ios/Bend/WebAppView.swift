import SwiftUI
import WebKit

/// Full-screen WKWebView showing the bundled web app (web/index.html in the app bundle).
struct WebAppView: UIViewRepresentable {
    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeUIView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.allowsInlineMediaPlayback = true
        configuration.preferences.isTextInteractionEnabled = true
        // Tells the web app where the on-device engines listen (src/engines/native.ts). Absent → demo engines.
        #if BEND_LLB
        // PHONEUX: always, engines or not: platform, engine files, surfaces, process start (BootOverlay.swift).
        configuration.userContentController.addUserScript(WKUserScript(source: ShellInfo.script, injectionTime: .atDocumentStart, forMainFrameOnly: true))
        #endif
        if let bridge = EngineManager.shared.bridgeScript {
            configuration.userContentController.addUserScript(WKUserScript(source: bridge, injectionTime: .atDocumentStart, forMainFrameOnly: true))
        }
        // PHONEVOICE: where voice mode's port is (src/voice/native.ts).
        configuration.userContentController.addUserScript(WKUserScript(source: VoiceServer.shared.bridgeScript, injectionTime: .atDocumentStart, forMainFrameOnly: true))
        #if BEND_CAMERA
        // PHONEONE: the web header's Camera button shows only where the native camera surface exists (src/app/os.ts).
        configuration.userContentController.addUserScript(WKUserScript(source: "window.__BEND_CAMERA__ = true;", injectionTime: .atDocumentStart, forMainFrameOnly: true))
        #endif
        #if BEND_FETCH
        // IOSFETCH: the web's "No models on this phone" line opens the setup card (src/app/os.ts).
        configuration.userContentController.addUserScript(WKUserScript(source: "window.__BEND_MODELS__ = true;", injectionTime: .atDocumentStart, forMainFrameOnly: true))
        // VOICESHOP: the shell has the voice picker + shop (bendOS voices.open)
        configuration.userContentController.addUserScript(WKUserScript(source: "window.__BEND_VOICES__ = true;", injectionTime: .atDocumentStart, forMainFrameOnly: true))
        #endif
        // BENDSDK: the Dev menu's builder examples (src/app/dev). Debug builds and `-dev` show it; `-example <name>` opens one,
        // `-exampleCheck` has it run its self-check (the sim receipts, scripts/sdk-sim.sh).
        if let dev = Self.devScript {
            configuration.userContentController.addUserScript(WKUserScript(source: dev, injectionTime: .atDocumentStart, forMainFrameOnly: true))
        }
        // R15: notifications, Shortcuts, background refresh (window.webkit.messageHandlers.bendOS, with replies).
        configuration.userContentController.addScriptMessageHandler(OSBridge.shared, contentWorld: .page, name: OSBridge.handlerName)
        if let check = OSBridge.shared.userScript { configuration.userContentController.addUserScript(check) }
        // WORKBENCH2: `-workbenchCheck <scene>` (session | editor | files) drives the Workbench to a receipt state
        // (src/app/workbench/check.ts); its report comes back as `BEND agent …` lines through check.report.
        let args = ProcessInfo.processInfo.arguments
        if let at = args.firstIndex(of: "-workbenchCheck") {
            let scene = at + 1 < args.count ? args[at + 1].filter { $0.isLetter } : "session"
            configuration.userContentController.addUserScript(WKUserScript(source: "window.__BEND_WORKBENCH__ = { check: \"\(scene)\" };", injectionTime: .atDocumentStart, forMainFrameOnly: true))
        }
        // HOMEREV1 (DESIGN.md): `-homeDrawer open|peek` drives the drawer at launch through the page's own drag (receipts);
        // src/app/home/flags.ts reads it. ICONREVERT: `-homeIcons` / `-homeHero` went with the icon sets (archive/).
        func value(_ flag: String) -> String? { args.firstIndex(of: flag).flatMap { $0 + 1 < args.count ? String(args[$0 + 1].filter { $0.isLetter }) : nil } }
        if let drawer = value("-homeDrawer") {
            configuration.userContentController.addUserScript(WKUserScript(source: "window.__BEND_HOME__ = { drawer: \"\(drawer)\" };", injectionTime: .atDocumentStart, forMainFrameOnly: true))
        }
        // ICONREVERT (receipts): `-homeTheme day|night` pins the page's theme (the app is pinned night; day is the
        // foundation's :root), held against the page's own writes, so the boards can show the home both ways.
        if let theme = value("-homeTheme"), theme == "day" || theme == "night" {
            let t = theme == "day" ? "light" : "dark"
            let script = "(() => { const r = document.documentElement; const pin = () => { if (r.dataset.theme !== \"\(t)\") r.dataset.theme = \"\(t)\"; }; pin(); new MutationObserver(pin).observe(r, { attributes: true, attributeFilter: [\"data-theme\"] }); })();"
            configuration.userContentController.addUserScript(WKUserScript(source: script, injectionTime: .atDocumentStart, forMainFrameOnly: true))
        }
        // VOICESHADE: `-voiceShade` turns on the shader voice surface (src/app/voice/shade); `-voiceShadeTheme day|night` pins
        // its theme (default: the system appearance); `-voiceShadeAmbient <ms>` shortens the idle → ambient wait (receipts).
        if args.contains("-voiceShade") {
            let value = { (flag: String) -> String in
                args.firstIndex(of: flag).flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil }?.filter { $0.isLetter || $0.isNumber } ?? ""
            }
            // The app is pinned dark (project.yml UIUserInterfaceStyle, BendApp .preferredColorScheme), which also hides the
            // system's appearance from the page and from this process's traits: the surface follows the app (night) unless pinned.
            let theme = value("-voiceShadeTheme")
            let script = "window.__BEND_VOICESHADE__ = { on: true, theme: \"\(theme)\", ambient: \"\(value("-voiceShadeAmbient"))\", open: \(args.contains("-voiceShadeOpen")) };"
            configuration.userContentController.addUserScript(WKUserScript(source: script, injectionTime: .atDocumentStart, forMainFrameOnly: true))
        }

        let webView = WKWebView(frame: .zero, configuration: configuration)
        OSBridge.shared.webView = webView
        #if BEND_LLB
        BootModel.shared.webView = webView
        #endif
        let background = UIColor(Color.bendBackground)
        webView.isOpaque = false
        webView.backgroundColor = background
        webView.underPageBackgroundColor = background
        webView.scrollView.backgroundColor = background
        // The page scrolls its own message list; the outer view stays put and never zooms.
        webView.scrollView.bounces = false
        webView.scrollView.contentInsetAdjustmentBehavior = .never
        webView.scrollView.delegate = context.coordinator
        webView.navigationDelegate = context.coordinator
        #if DEBUG
        webView.isInspectable = true // Safari ▸ Develop ▸ Simulator / device
        #endif

        if let index = Bundle.main.url(forResource: "index", withExtension: "html", subdirectory: "web") {
            webView.loadFileURL(index, allowingReadAccessTo: index.deletingLastPathComponent())
        } else {
            webView.loadHTMLString(Self.missingBundleHTML, baseURL: nil)
        }
        return webView
    }

    func updateUIView(_ webView: WKWebView, context: Context) {}

    private static var devScript: String? {
        let args = ProcessInfo.processInfo.arguments
        #if DEBUG
        let on = true
        #else
        let on = args.contains("-dev")
        #endif
        guard on else { return nil }
        let example = args.firstIndex(of: "-example").flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil } ?? ""
        let safe = example.filter { $0.isLetter || $0.isNumber }
        return "window.__BEND_DEV__ = Object.freeze({ example: \"\(safe)\", check: \(args.contains("-exampleCheck")) });"
    }

    /// Shown if the app was built without running scripts/build-ios.sh first.
    private static let missingBundleHTML = """
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <body style="margin:0;display:grid;place-items:center;height:100vh;background:#141414;color:#aaa;font:15px -apple-system;text-align:center;padding:24px;box-sizing:border-box">
    <p>The web app is missing from this build.<br>Run <code>scripts/build-ios.sh</code> and rebuild.</p>
    </body>
    """

    final class Coordinator: NSObject, WKNavigationDelegate, UIScrollViewDelegate {
        func viewForZooming(in scrollView: UIScrollView) -> UIView? { nil }

        /// Links that leave the bundle open in Safari instead of replacing the app.
        func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction) async -> WKNavigationActionPolicy {
            guard let url = action.request.url, url.scheme == "http" || url.scheme == "https" else { return .allow }
            await UIApplication.shared.open(url)
            return .cancel
        }

        /// The web UI is up: the engine may prime its compiled program now (R17, EngineManager.pageReady).
        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            MainActor.assumeIsolated {
                EngineManager.shared.pageReady()
                #if BEND_LLB
                EngineManager.console(String(format: "BEND boot web didFinish at %.0f ms since process start", Boot.sinceStartMs()))
                // A page built before PHONEUX never posts ui.ready: give it 3 s from its load, then lift the overlay.
                Task { @MainActor in
                    try? await Task.sleep(for: .seconds(3))
                    BootModel.shared.pageReady("no ui.ready 3 s after didFinish")
                }
                #endif
            }
        }

        /// WebKit can kill the content process under memory pressure; reload instead of showing a blank page.
        func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
            webView.reload()
        }
    }
}
