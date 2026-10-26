import BackgroundTasks
import ImageIO
import UIKit
import UserNotifications
import WebKit

/// R15's OS bridges (docs/AGENT.md): one WKScriptMessageHandlerWithReply, `bendOS`, so each web call
/// (`window.webkit.messageHandlers.bendOS.postMessage({op, …})`) resolves with a reply. Chosen over a control-port route:
/// no key, no loopback hop, both targets get it, and the web view is the only caller there is.
///
///   notify.sync {alarms: [{id, at (epoch ms), label, repeat: once|daily}]}
///       asks for notification permission the first time there's an alarm to schedule, then replaces every pending
///       `bend-alarm-*` request with the set (so add, edit, off and delete are one idempotent call).
///       → {permission, scheduled}. Denied → nothing scheduled; the web's in-app banner still rings.
///   notify.status                         → {permission}
///   shortcuts.run {name, input?}          opens shortcuts://x-callback-url/run-shortcut?name=…, with x-success /
///       x-cancel / x-error back to bend://shortcut. iOS switches to Shortcuts and back; nothing silent. → {opened}
///   refresh.status                        → {status, scheduled, lastRun, lastResult}
///   timing {line}                         R17: one `BEND engine web timing …` line (send → first token, tap → card).
///   sdk.log {line}                        BENDSDK: an SDK trace line on the console (`BEND engine sdk …`).
///   check.report {report}                 `-agentCheck`: prints the web's runtime check as `BEND agent …` lines.
///   check.say {line}                      `-workCheck` (WORKFIX): prints one `BEND work …` progress line.
///
/// Background refresh: one BGAppRefreshTask (`com.example.bend.refresh`, in Info.plist). When iOS grants a
/// window, the handler calls the page's `window.__bendBackground()` (tick: ring alarms, start due jobs, settle) and
/// completes. Opportunistic: iOS picks the time (earliest 15 min out, often hours), and only while the app is in
/// memory with its web view; a launch straight into the background without a scene has no page and completes false.
@MainActor
final class OSBridge: NSObject, WKScriptMessageHandlerWithReply, UNUserNotificationCenterDelegate {
    static let shared = OSBridge()
    static let handlerName = "bendOS"
    nonisolated static let refreshID = "com.example.bend.refresh"
    nonisolated static let alarmPrefix = "bend-alarm-"

    /// The page, for background refresh. Set by WebAppView.
    weak var webView: WKWebView?
    private var refreshScheduled = false

    nonisolated static func say(_ line: String) {
        print("BEND agent \(line)")
        fflush(stdout)
        NSLog("BEND agent %@", line)
    }

    // MARK: launch

    /// Before the app finishes launching (BGTaskScheduler's rule): the refresh handler and the notification delegate.
    func launch() {
        UNUserNotificationCenter.current().delegate = self
        let registered = BGTaskScheduler.shared.register(forTaskWithIdentifier: Self.refreshID, using: .main) { task in
            guard let task = task as? BGAppRefreshTask else { return task.setTaskCompleted(success: false) }
            MainActor.assumeIsolated { OSBridge.shared.refresh(task) }
        }
        if !registered { Self.say("refresh: not registered (is \(Self.refreshID) in BGTaskSchedulerPermittedIdentifiers?)") }
        NotificationCenter.default.addObserver(forName: UIApplication.didEnterBackgroundNotification, object: nil, queue: .main) { _ in
            MainActor.assumeIsolated { OSBridge.shared.scheduleRefresh() }
        }
    }

    /// Bridge script: `-agentCheck` asks the page to run its runtime check once the Core is open.
    /// OPENUI-R1: `-openuiCheck <variant>` (e.g. `chart`, `map+offline`) has the page send one canned prompt through the
    /// demo engines and draw the generated view, for sim receipts (src/openui/check.ts). Letters, `+` and `-` only.
    /// WORKFIX: `-workCheck keep|revert` asks it to drive the Work e2e (src/app/work/workCheck.ts).
    /// PHONEUX: `-uxTap camera|photo|voice` has the page press that surface's button itself (HOMEREV1: in the drawer) once it's up (receipts: the
    /// main screen's entry → bendOS → the native surface, with no hand on the simulator).
    var userScript: WKUserScript? {
        let args = ProcessInfo.processInfo.arguments
        var source = ""
        if args.contains("-agentCheck") { source += "window.__bendAgentCheck = true;" }
        if let at = args.firstIndex(of: "-openuiCheck"), at + 1 < args.count {
            let variant = args[at + 1].filter { $0.isLetter || $0 == "+" || $0 == "-" }
            source += "window.__bendOpenUICheck = \"\(variant)\";"
        }
        if let i = args.firstIndex(of: "-workCheck"), i + 1 < args.count { source += "window.__bendWorkCheck = \"\(args[i + 1])\";" }
        // IMAGETRUE: `-lookCheck <clip,clip,…>` (receipts): once Look has seen its photo, it asks each clip out loud in turn
        // (hold, the clip as the mic, let go), as a person would. Clip paths are the voice port's (the Mac harness's files).
        if let i = args.firstIndex(of: "-lookCheck"), i + 1 < args.count,
           let json = try? JSONSerialization.data(withJSONObject: args[i + 1].split(separator: ",").map(String.init)),
           let list = String(data: json, encoding: .utf8) {
            source += "window.__bendLookCheck = \(list);"
        }
        // BENDFIX-1: `-surfaceFixture <clip>` (receipts): a sub-app's small lumen hears this clip instead of the mic, on the
        // real touch: a hold plays it as the held mic, a tap opens the synthetic line and says it there. Voice-port paths
        // (`look/x.wav` = Documents/look/x.wav).
        if let i = args.firstIndex(of: "-surfaceFixture"), i + 1 < args.count,
           let json = try? JSONSerialization.data(withJSONObject: [args[i + 1]]), let one = String(data: json, encoding: .utf8) {
            source += "window.__bendSurfaceFixture = \(one)[0];"
        }
        if let i = args.firstIndex(of: "-uxTap"), i + 1 < args.count, ["camera", "photo", "voice"].contains(args[i + 1]) {
            source += "window.__bendUxTap = \"\(args[i + 1])\";"
        }
        return source.isEmpty ? nil : WKUserScript(source: source, injectionTime: .atDocumentStart, forMainFrameOnly: true)
    }

    // MARK: messages

    func userContentController(_ controller: WKUserContentController, didReceive message: WKScriptMessage) async -> (Any?, String?) {
        guard let body = message.body as? [String: Any], let op = body["op"] as? String else { return (nil, "bendOS: a message needs an op") }
        switch op {
        case "ui.ready":  // PHONEUX: the page painted its first frame; the boot overlay goes (BootOverlay.swift)
            #if BEND_LLB
            BootModel.shared.pageReady("page")
            #endif
            return (["ok": true], nil)
        #if BEND_CAMERA
        case "camera.open":  // PHONECAM: the native camera surface over the page; PHONEUX: {source: "photo"} opens it on a photo
            #if BEND_FETCH && !targetEnvironment(simulator)
            // IOSFETCH: no vision tower yet: the setup card with the camera row chosen, not a surface with no verdicts.
            if let vision = ModelStore.shared.visionItem, !ModelStore.shared.present(vision) {
                ModelStore.shared.present(vision: true)
                return (["ok": true, "models": true], nil)
            }
            #endif
            CameraPresenter.shared.open(source: (body["source"] as? String) == "photo" ? .photo : .camera)
            return (["ok": true], nil)
        #endif
        #if BEND_FETCH
        case "models.open":  // IOSFETCH: the setup card (the web's "No models on this phone" line)
            ModelStore.shared.present(vision: body["vision"] as? Bool ?? false)
            return (["ok": true], nil)
        case "voices.open":  // VOICESHOP: Speech settings (the voice picker); {shop: true} opens More voices
            VoicePicker.shared.present(shop: body["shop"] as? Bool ?? false)
            return (["ok": true], nil)
        #endif
        case "notify.sync":
            return (await syncAlarms(body["alarms"] as? [[String: Any]] ?? []), nil)
        case "notify.status":
            return (["permission": Self.permissionName(await UNUserNotificationCenter.current().notificationSettings().authorizationStatus)], nil)
        case "shortcuts.run":
            guard let name = (body["name"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines), !name.isEmpty else {
                return (nil, "shortcuts.run needs a name")
            }
            guard let url = Self.shortcutURL(name: name, input: body["input"] as? String) else { return (nil, "bad shortcut name") }
            let opened = await UIApplication.shared.open(url)
            Self.say("shortcuts.run \(name) opened=\(opened)")
            return (["opened": opened], nil)
        case "refresh.status":
            return (refreshStatus(), nil)
        case "timing":
            // R17: the web's stopwatch (src/app/timing.ts), on the engine's console so the device receipt has it.
            EngineManager.log("web timing \(body["line"] as? String ?? "?")")
            return (["ok": true], nil)
        case "check.say":
            // WORKBENCH2: a progress line from the -workbenchCheck run ("SHOT session-review"), for the receipt script.
            // WORKFIX: under -workCheck, a `BEND work …` line instead (scripts/work-sim-e2e.sh shoots on these).
            if ProcessInfo.processInfo.arguments.contains("-workCheck") {
                print("BEND work \(body["line"] as? String ?? "")")
                fflush(stdout)
            } else {
                Self.say("workbench \(body["line"] as? String ?? "?")")
            }
            return (["ok": true], nil)
        case "sdk.log":
            // BENDSDK: an example's trace line (src/sdk/bend.ts `trace`), on the engine's console for the receipts.
            EngineManager.log("sdk \(body["line"] as? String ?? "?")")
            return (["ok": true], nil)
        case "check.report":
            report(body["report"])
            return (["ok": true], nil)
        default:
            return (nil, "bendOS: unknown op \(op)")
        }
    }

    // MARK: camera → chat (ATTACHVIS)

    /// The camera surface's shot into the chat composer: JPEG (long edge ≤ 1280, as the web's own prep) as a data URL to
    /// the page's `window.__bendAttach` (src/app/images/useAttachment.ts), or into `__bendAttachQueue` if the page isn't
    /// listening yet. False when there is no page or the image won't encode.
    @discardableResult
    func attach(_ image: CGImage, name: String, caption: String? = nil) -> Bool {
        guard let webView, let data = Self.jpeg(image, edge: 1280, quality: 0.85) else { return false }
        var shot = ["dataUrl": "data:image/jpeg;base64,\(data.base64EncodedString())", "name": name]
        // CAMCAP: with a caption the page posts the photo and the caption into the thread (no look); without, the chip.
        if let caption, !caption.isEmpty { shot["caption"] = caption }
        guard let json = try? JSONSerialization.data(withJSONObject: shot), let arg = String(data: json, encoding: .utf8) else { return false }
        webView.evaluateJavaScript("(function(s){ if (window.__bendAttach) return window.__bendAttach(s); (window.__bendAttachQueue = window.__bendAttachQueue || []).push(s); return true; })(\(arg))")
        EngineManager.log("camera send-to-chat \(image.width)x\(image.height) jpeg_bytes=\(data.count) caption=\(caption?.count ?? 0)ch")
        return true
    }

    /// IMAGETRUE: the camera's shot → the page's Look surface (`window.__bendLook`, src/app/look/hook.ts): the JPEG (≤1280 px,
    /// made off the main thread by the caller), its size, the key the shell holds its canvas by (VisionLook.hold; nil when
    /// this process can't look), and the shutter's epoch ms. Queued in `__bendLookQueue` if the page isn't listening yet.
    func look(jpeg data: Data, width: Int, height: Int, key: String?, shotAt: Double) -> Bool {
        guard let webView else { return false }
        var shot: [String: Any] = ["dataUrl": "data:image/jpeg;base64,\(data.base64EncodedString())", "width": width, "height": height, "shotAt": shotAt]
        if let key { shot["key"] = key }
        guard let json = try? JSONSerialization.data(withJSONObject: shot), let arg = String(data: json, encoding: .utf8) else { return false }
        webView.evaluateJavaScript("(function(s){ if (window.__bendLook) return window.__bendLook(s); (window.__bendLookQueue = window.__bendLookQueue || []).push(s); return true; })(\(arg))")
        EngineManager.log(String(format: "look handed %dx%d jpeg_bytes=%d key=%@ since_shot_ms=%.0f", width, height, data.count, key ?? "-",
                                 Date().timeIntervalSince1970 * 1000 - shotAt))
        return true
    }

    nonisolated static func jpeg(_ image: CGImage, edge: Int, quality: Double) -> Data? {
        let scale = min(1, Double(edge) / Double(max(image.width, image.height)))
        let (w, h) = (max(1, Int((Double(image.width) * scale).rounded())), max(1, Int((Double(image.height) * scale).rounded())))
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        guard let scaled = ctx.makeImage() else { return nil }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out, "public.jpeg" as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, scaled, [kCGImageDestinationLossyCompressionQuality: quality] as CFDictionary)
        return CGImageDestinationFinalize(dest) ? out as Data : nil
    }

    // MARK: notifications

    nonisolated static func permissionName(_ status: UNAuthorizationStatus) -> String {
        switch status {
        case .authorized: "granted"
        case .denied: "denied"
        case .notDetermined: "notDetermined"
        case .provisional: "provisional"
        case .ephemeral: "ephemeral"
        @unknown default: "unknown"
        }
    }

    private func syncAlarms(_ alarms: [[String: Any]]) async -> [String: Any] {
        let center = UNUserNotificationCenter.current()
        var status = await center.notificationSettings().authorizationStatus
        var cameraUp = false
        #if BEND_CAMERA
        // PHONECAM: never prompt over the full-screen camera; the next sync (the page on screen again) asks
        cameraUp = CameraPresenter.shared.isPresented
        #endif
        #if BEND_FETCH
        // IOSFETCH: nor over the first-run setup card (one thing at a time; the next sync asks)
        cameraUp = cameraUp || ModelStore.shared.isPresented
        cameraUp = cameraUp || VoicePicker.shared.isPresented  // VOICESHOP: nor over Speech settings
        #endif
        // PHONEONE: nor over the voice self-test (-voiceCheck: a receipt run, nobody there to answer it)
        // WORKBENCH2: nor over the Workbench receipt run (-workbenchCheck); WORKFIX: nor the Work e2e (-workCheck); OPENUI-R1 too
        let voiceCheck = ["-voiceCheck", "-openuiCheck", "-workbenchCheck", "-workCheck"].contains { ProcessInfo.processInfo.arguments.contains($0) }
        if status == .notDetermined && !alarms.isEmpty && !cameraUp && !voiceCheck {
            // First use: the system prompt, once. A refusal is final here; Settings is the way back.
            _ = try? await center.requestAuthorization(options: [.alert, .sound])
            status = await center.notificationSettings().authorizationStatus
        }
        let ours = await center.pendingNotificationRequests().map(\.identifier).filter { $0.hasPrefix(Self.alarmPrefix) }
        center.removePendingNotificationRequests(withIdentifiers: ours)
        let allowed = status == .authorized || status == .provisional || status == .ephemeral
        var scheduled = 0
        if allowed {
            for alarm in alarms {
                guard let request = Self.request(for: alarm) else { continue }
                do {
                    try await center.add(request)
                    scheduled += 1
                } catch {
                    Self.say("notify: \(request.identifier) failed: \(error.localizedDescription)")
                }
            }
        }
        Self.say("notify.sync \(alarms.count) alarm(s) → \(scheduled) scheduled, permission \(Self.permissionName(status)), removed \(ours.count)")
        return ["permission": Self.permissionName(status), "scheduled": scheduled]
    }

    /// Once: a calendar trigger at the exact minute (skipped if it's past). Daily: hour and minute, repeating.
    nonisolated static func request(for alarm: [String: Any], now: Date = Date()) -> UNNotificationRequest? {
        guard let id = alarm["id"] as? String, let ms = (alarm["at"] as? NSNumber)?.doubleValue else { return nil }
        let at = Date(timeIntervalSince1970: ms / 1000)
        let daily = (alarm["repeat"] as? String) == "daily"
        let calendar = Calendar.current
        let trigger: UNCalendarNotificationTrigger
        if daily {
            trigger = UNCalendarNotificationTrigger(dateMatching: calendar.dateComponents([.hour, .minute], from: at), repeats: true)
        } else {
            guard at > now else { return nil }
            trigger = UNCalendarNotificationTrigger(dateMatching: calendar.dateComponents([.year, .month, .day, .hour, .minute, .second], from: at), repeats: false)
        }
        let content = UNMutableNotificationContent()
        content.title = (alarm["label"] as? String).flatMap { $0.isEmpty ? nil : $0 } ?? "Reminder"
        content.body = daily ? "Eureka · daily" : "Eureka"
        content.sound = .default
        content.threadIdentifier = "bend-alarms"
        return UNNotificationRequest(identifier: alarmPrefix + id, content: content, trigger: trigger)
    }

    /// In the foreground the page's banner rings (within a 5 s tick), so the system stays quiet: Notification Center only.
    nonisolated func userNotificationCenter(_ center: UNUserNotificationCenter, willPresent notification: UNNotification) async -> UNNotificationPresentationOptions {
        [.list]
    }

    // MARK: shortcuts

    nonisolated static func shortcutURL(name: String, input: String?) -> URL? {
        // Query values encoded strictly (a "+" or "&" in a name must survive), so build the query by hand.
        var allowed = CharacterSet.alphanumerics
        allowed.insert(charactersIn: "-._~")
        func enc(_ s: String) -> String { s.addingPercentEncoding(withAllowedCharacters: allowed) ?? s }
        var query = ["name=\(enc(name))"]
        if let input { query += ["input=text", "text=\(enc(input))"] }
        query += ["x-success=\(enc("bend://shortcut?status=done"))", "x-cancel=\(enc("bend://shortcut?status=cancelled"))", "x-error=\(enc("bend://shortcut?status=error"))"]
        return URL(string: "shortcuts://x-callback-url/run-shortcut?\(query.joined(separator: "&"))")
    }

    // MARK: background refresh

    func scheduleRefresh() {
        let request = BGAppRefreshTaskRequest(identifier: Self.refreshID)
        request.earliestBeginDate = Date(timeIntervalSinceNow: 15 * 60)
        Task { @MainActor in
            do {
                try await BGTaskScheduler.shared.submitTaskRequest(request)
                refreshScheduled = true
            } catch {
                // The simulator refuses (unavailable); a device refuses when Background App Refresh is off for Bend.
                refreshScheduled = false
                Self.say("refresh: submit failed: \(error.localizedDescription)")
            }
        }
    }

    private func refresh(_ task: BGAppRefreshTask) {
        scheduleRefresh() // the next window, whatever happens to this one
        let defaults = UserDefaults.standard
        defaults.set(Date().timeIntervalSince1970, forKey: "bend.refresh.lastRun")
        guard let webView else {
            defaults.set("no web view (launched without a scene)", forKey: "bend.refresh.lastResult")
            return task.setTaskCompleted(success: false)
        }
        var done = false
        let finish = { (ok: Bool, why: String) in
            guard !done else { return }
            done = true
            defaults.set(why, forKey: "bend.refresh.lastResult")
            Self.say("refresh: \(why)")
            task.setTaskCompleted(success: ok)
        }
        let work = Task { @MainActor in
            do {
                let result = try await webView.callAsyncJavaScript("return window.__bendBackground ? await window.__bendBackground() : null", contentWorld: .page)
                finish(result != nil, result.map { "ran: \($0)" } ?? "the page has no background hook yet")
            } catch {
                finish(false, "failed: \(error.localizedDescription)")
            }
        }
        task.expirationHandler = {
            MainActor.assumeIsolated {
                work.cancel()
                finish(false, "expired before the page finished")
            }
        }
    }

    private func refreshStatus() -> [String: Any] {
        let status: String = switch UIApplication.shared.backgroundRefreshStatus {
        case .available: "available"
        case .denied: "denied"
        case .restricted: "restricted"
        @unknown default: "unknown"
        }
        let defaults = UserDefaults.standard
        return [
            "status": status,
            "scheduled": refreshScheduled,
            "lastRun": defaults.double(forKey: "bend.refresh.lastRun") * 1000,
            "lastResult": defaults.string(forKey: "bend.refresh.lastResult") ?? "never",
        ]
    }

    // MARK: -agentCheck

    private func report(_ value: Any?) {
        guard let report = value as? [String: Any] else { return Self.say("RESULT FAIL (no report)") }
        for case let check as [String: Any] in report["checks"] as? [Any] ?? [] {
            Self.say("\((check["ok"] as? Bool) == true ? "ok  " : "FAIL") \(check["name"] ?? "?") \(check["detail"] ?? "")")
        }
        if let stats = report["stats"] { Self.say("stats \(stats)") }
        Self.say("RESULT \((report["ok"] as? Bool) == true ? "PASS" : "FAIL")")
        if ProcessInfo.processInfo.arguments.contains("-agentCheckExit") { exit((report["ok"] as? Bool) == true ? 0 : 1) }
    }
}
