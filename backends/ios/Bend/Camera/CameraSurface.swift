import SwiftUI

/// The surface's state (the iOS view and the Mac's rendered shots share it).
@MainActor @Observable
final class CameraSurfaceModel {
    enum Phase: Equatable {
        case starting
        case ready
        case unavailable(String)
        case failed(String)
    }

    var verdict: CameraVerdict?
    var verdictOrigin: CameraPipeline.Origin?
    var stats = CameraPipeline.Stats()
    var phase: Phase = .starting
    var live = false
    /// Tap-to-freeze: the frame being judged / last judged by a tap; nil = the live view.
    var frozen: Canvas?
    /// The injected sources' latest frame (the real camera shows its preview layer instead).
    var lastFrame: Canvas?
    var question = CameraConfig.defaultQuestion
    var sourceLabel = "rear camera"
    /// The frame the verdict was made on (shown small in the corner while live).
    var judged: Canvas?

    /// CAMCAP: the full-resolution frame taken when the view froze (the canvas is only the tower's 320 px). Set by the
    /// iOS view on the real camera; Describe and Send to chat use it over the canvas.
    var frozenShot: CGImage?
    /// The real camera delivers a full-resolution shot a frame after the freeze: Describe waits for it.
    var expectsShot = false
    let pipeline: CameraPipeline
    let caption: CaptionController
    private var latest: Canvas?

    init(engine: VerdictEngine, captioner: Captioner? = nil) {
        pipeline = CameraPipeline(engine: engine)
        caption = CaptionController(captioner: captioner)
        let pipe = pipeline
        caption.onBusy = { pipe.hold($0) }
        pipeline.onVerdict = { [weak self] v, origin, canvas in
            Task { @MainActor in
                self?.verdict = v
                self?.verdictOrigin = origin
                self?.judged = canvas
                EngineManager.log(v.logLine(origin.rawValue))
            }
        }
        pipeline.onStats = { [weak self] s in Task { @MainActor in self?.stats = s } }
        pipeline.onReady = { [weak self] r in
            Task { @MainActor in
                switch r {
                case .success: self?.phase = .ready
                case .failure(let e):
                    let why = e.localizedDescription
                    self?.phase = CameraEngine.unavailableReason != nil ? .unavailable(why) : .failed(why)
                }
            }
        }
    }

    func start() {
        if let why = CameraEngine.unavailableReason, pipeline.engine is CameraEngine {
            phase = .unavailable(why)
        } else {
            pipeline.prepare()
        }
    }

    /// A frame from the source (any queue).
    nonisolated func deliver(_ f: CameraFrame, showFrame: Bool) {
        pipeline.frame(f.canvas, tMs: f.tMs)
        Task { @MainActor [weak self] in
            guard let self else { return }
            self.latest = f.canvas
            if showFrame { self.lastFrame = f.canvas }
            if let q = f.question, q != self.question { self.setQuestion(q) }
        }
    }

    func setLive(_ on: Bool) {
        live = on
        if on { frozen = nil; frozenShot = nil; describing?.cancel(); caption.cancel() }
        pipeline.setLive(on)
    }

    func setQuestion(_ q: String) {
        question = q
        pipeline.setQuestion(q)
    }

    /// PHONEUX: a still (a picked photo): letterboxed into the canvas like a camera frame, frozen, and judged now (the
    /// tap path). Live mode is switched off; the next photo, or a tap, moves on.
    func judgePhoto(data: Data) {
        guard let image = Canvas.decode(data) else { return photoFailed("Couldn't open that photo") }
        let canvas = Canvas.letterbox(image)
        setLive(false)
        sourceLabel = "photo"
        verdict = nil
        latest = canvas
        lastFrame = canvas
        frozen = canvas
        describing?.cancel()
        caption.cancel()
        frozenShot = image
        pipeline.check(canvas)
    }

    /// PHONEUX: the camera can't start (no rear camera, access denied): say so, point at the photo button, and keep
    /// the reason the engine can't judge here if there is one (the simulator; engine files not set up).
    func cameraFailed(_ why: String) {
        let next = "\(why) · pick a photo instead"
        if case .unavailable(let reason) = phase { phase = .unavailable("\(next)\n\(reason)") } else { phase = .failed(next) }
    }

    func photoFailed(_ why: String) {
        sourceLabel = "photo"
        if case .unavailable = phase { return }
        phase = .failed(why)
    }

    /// Tap: freeze this frame and judge it; tap again: back to the live view.
    func tapFreeze() {
        if frozen != nil { unfreeze(); return }
        guard let c = latest else { return }
        frozen = c
        describing?.cancel()
        caption.cancel()
        pipeline.check(c)
    }

    private func unfreeze() {
        frozen = nil
        frozenShot = nil
        describing?.cancel()
        caption.cancel()
        pipeline.touched()
    }

    /// CAMCAP: Describe freezes the frame (if the view is live; no verdict is asked) and captions it. Already frozen:
    /// captions again (the card's Retry). It reads the full-resolution shot when the camera gives one.
    func describe() {
        if frozen == nil {
            guard let c = latest else { return }
            frozen = c
            live = false
            pipeline.setLive(false)
            pipeline.markJudged(c)
        }
        guard let canvas = frozen else { return }
        describing?.cancel()
        caption.expect()
        describing = Task { @MainActor [weak self] in
            for _ in 0..<32 where self?.expectsShot == true && self?.frozenShot == nil {
                try? await Task.sleep(for: .milliseconds(50))
            }
            guard let self, !Task.isCancelled, self.frozen != nil else { return }
            guard let image = self.frozenShot ?? canvas.cgImage() else { self.caption.fail("No frame to describe."); return }
            self.caption.ask(image)
        }
    }

    private var describing: Task<Void, Never>?
}

/// The big verdict: YES / NO, confidence, frame→verdict ms; or what the engine is doing.
struct VerdictOverlay: View {
    let model: CameraSurfaceModel

    var body: some View {
        VStack(spacing: 6) {
            if let v = model.verdict {
                Text(v.yes ? "YES" : "NO")
                    .font(.system(size: 104, weight: .black, design: .rounded))
                    .foregroundStyle(v.yes ? Color.green : Color.red)
                    .accessibilityIdentifier("camera.verdict")
                HStack(spacing: 14) {
                    Text(String(format: "%.0f%%", v.confidence * 100)).accessibilityIdentifier("camera.confidence")
                    Text(String(format: "%.0f ms", v.frameMs)).accessibilityIdentifier("camera.ms")
                }
                .font(.system(size: 30, weight: .bold, design: .rounded).monospacedDigit())
                .foregroundStyle(.white)
                Text(String(format: "%@ · tower %.0f · llm %.0f ms · score %+.2f", model.verdictOrigin?.rawValue ?? "",
                            v.towerMs, v.llmMs, v.score))
                    .font(.system(size: 13, weight: .medium).monospacedDigit())
                    .foregroundStyle(.white.opacity(0.75))
            } else {
                Text(placeholder)
                    .font(.system(size: 22, weight: .semibold, design: .rounded))
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.white.opacity(0.9))
                    .accessibilityIdentifier("camera.placeholder")
            }
        }
        .padding(.vertical, 14)
        .padding(.horizontal, 22)
        .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 22))
    }

    private var placeholder: String {
        switch model.phase {
        case .starting: "Warming the engine…"
        case .ready: model.live ? "Hold still on something" : "Tap to check"
        case .unavailable(let why): "No verdicts here\n\(why)"
        case .failed(let why): "Can't judge\n\(why)"
        }
    }
}

/// CAMCAP: the caption card, in the weave's language (the thread's look reply: words, a "· vision" tag, Retry on a
/// failure): "Looking…" until the first words, the caption as it streams, then its time. No state is a dead end.
struct CaptionCard: View {
    let caption: CaptionController
    var onRetry: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Text("CAPTION").font(.system(size: 11, weight: .heavy)).tracking(1.2).foregroundStyle(.white.opacity(0.6))
                Spacer()
                if case .done(_, let ms) = caption.state {
                    Text(String(format: "vision · %.1f s", ms / 1000)).font(.system(size: 12, weight: .medium).monospacedDigit())
                        .foregroundStyle(.white.opacity(0.6))
                        .accessibilityIdentifier("camera.caption.meta")
                }
            }
            switch caption.state {
            case .idle:
                EmptyView()
            case .loading:
                HStack(spacing: 8) {
                    ProgressView().tint(.white)
                    Text("Looking at the frame…").foregroundStyle(.white.opacity(0.85))
                }
                .font(.system(size: 18, weight: .medium))
                .accessibilityIdentifier("camera.caption.loading")
            case .streaming(let t):
                Text("\(t) ▍")
                    .font(.system(size: 20, weight: .semibold))
                    .foregroundStyle(.white)
                    .accessibilityIdentifier("camera.caption.text")
            case .done(let t, _):
                Text(t).font(.system(size: 20, weight: .semibold)).foregroundStyle(.white)
                    .accessibilityIdentifier("camera.caption.text")
            case .failed(let why):
                Text(why).font(.system(size: 16, weight: .medium)).foregroundStyle(.white.opacity(0.9))
                    .accessibilityIdentifier("camera.caption.error")
                retry
            case .unavailable(let why):
                Text("Eureka can't caption photos here\n\(why)").font(.system(size: 16, weight: .medium))
                    .foregroundStyle(.white.opacity(0.9))
                    .accessibilityIdentifier("camera.caption.unavailable")
            }
            if case .done = caption.state { retry }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.vertical, 14)
        .padding(.horizontal, 18)
        .background(.black.opacity(0.62), in: RoundedRectangle(cornerRadius: 22))
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("camera.caption")
    }

    private var retry: some View {
        Button(action: onRetry) {
            Label("Again", systemImage: "arrow.clockwise").font(.system(size: 14, weight: .semibold))
                .foregroundStyle(.white.opacity(0.85))
        }
        .accessibilityIdentifier("camera.caption.retry")
    }
}

/// One status line: mode, gate, counts, engine duty.
struct CameraStatusLine: View {
    let model: CameraSurfaceModel

    var body: some View {
        let s = model.stats
        let skip = s.frames > 0 ? Double(s.skipped) / Double(max(1, s.fires + s.skipped)) * 100 : 0
        Text(String(format: "%@ · %@ · gate %@ · fires %d · skip %.0f%% · duty %.0f%%%@", model.frozen != nil ? (model.sourceLabel == "photo" ? "PHOTO" : "FROZEN") : (model.live ? "LIVE" : "TAP"),
                    model.sourceLabel, s.gateState.rawValue, s.fires, skip, s.duty * 100,
                    s.engineBusy ? " · judging…" : ""))
            .font(.system(size: 12, weight: .semibold).monospacedDigit())
            .foregroundStyle(.white.opacity(0.85))
            .padding(.horizontal, 10).padding(.vertical, 5)
            .background(.black.opacity(0.5), in: Capsule())
            .accessibilityIdentifier("camera.status")
    }
}

/// The surface without the camera hardware: the frame (frozen / injected), the question, the verdict, the status and
/// the controls. The iOS view puts the live preview layer under it; the Mac harness renders it to PNG as it stands.
struct CameraSurface<Preview: View>: View {
    @Bindable var model: CameraSurfaceModel
    let preview: Preview
    var onClose: (() -> Void)?
    /// PHONEUX: the photo button (pick a photo to judge); nil hides it (the Mac's rendered shots).
    var onPhoto: (() -> Void)? = nil
    /// false: the question as a label (ImageRenderer, the Mac's shots, can't draw a TextField)
    var editableQuestion = true
    /// ATTACHVIS: "Send to chat": the shot (the frozen frame, else the view now) goes to the chat composer as its photo.
    var onSendToChat: (() -> Void)? = nil
    /// The shot is on its way (the button shows it, and can't be pressed twice).
    var sending = false
    /// CAMCAP: the shot goes with its caption (the button says so).
    var sendsCaption: Bool { model.caption.text != nil }

    var body: some View {
        ZStack {
            Color.black
            Group {
                if let f = model.frozen ?? (model.lastFrame), let img = f.cgImage() {
                    Image(decorative: img, scale: 1).resizable().interpolation(.high).scaledToFit()
                } else {
                    preview
                }
            }
            .contentShape(Rectangle())
            .onTapGesture { model.tapFreeze() }
            .accessibilityIdentifier("camera.preview")
            VStack(spacing: 12) {
                HStack(spacing: 10) {
                    Group {
                        if editableQuestion {
                            TextField("Ask a yes/no question", text: Binding(get: { model.question }, set: { model.setQuestion($0) }))
                                .textFieldStyle(.plain)
                        } else {
                            Text(model.question).lineLimit(1).frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                        .font(.system(size: 17, weight: .semibold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 14).padding(.vertical, 10)
                        .background(.black.opacity(0.55), in: Capsule())
                        .accessibilityIdentifier("camera.question")
                    if let onClose {
                        Button(action: onClose) {
                            Image(systemName: "xmark").font(.system(size: 17, weight: .bold)).foregroundStyle(.white)
                                .frame(width: 40, height: 40).background(.black.opacity(0.55), in: Circle())
                        }
                        .accessibilityIdentifier("camera.close")
                    }
                }
                CameraStatusLine(model: model)
                Spacer()
                if model.caption.state == .idle {
                    VerdictOverlay(model: model)
                } else {
                    CaptionCard(caption: model.caption, onRetry: { model.describe() })
                }
                Spacer().frame(height: 8)
                if let onSendToChat {
                    Button(action: onSendToChat) {
                        Label(sending ? "Sending…" : (sendsCaption ? "Send photo + caption" : "Send to chat"), systemImage: "arrow.up.message")
                            .font(.system(size: 16, weight: .semibold))
                            .foregroundStyle(.black)
                            .padding(.horizontal, 20)
                            .frame(minHeight: 48)
                            .background(Color.white, in: Capsule())
                    }
                    .disabled(sending)
                    .accessibilityIdentifier("camera.sendToChat")
                    .accessibilityHint("\(sendsCaption ? "Posts this shot and its caption in the chat" : "Puts this shot in the chat's message box, to ask about it")")
                }
                HStack(spacing: onPhoto == nil ? 20 : 12) {
                    if let onPhoto {
                        Button(action: onPhoto) {
                            Image(systemName: "photo.on.rectangle")
                                .font(.system(size: 18, weight: .semibold)).foregroundStyle(.white)
                                .frame(width: 52, height: 44).background(.black.opacity(0.55), in: Capsule())
                        }
                        .accessibilityLabel("Pick a photo")
                        .accessibilityIdentifier("camera.photo")
                    }
                    Button { model.setLive(!model.live) } label: {
                        Text(model.live ? "LIVE" : "Live")
                            .font(.system(size: 16, weight: .heavy))
                            .foregroundStyle(model.live ? .black : .white)
                            .frame(width: 76, height: 44)
                            .background(model.live ? Color.yellow : Color.black.opacity(0.55), in: Capsule())
                    }
                    .accessibilityIdentifier("camera.live")
                    Button { model.tapFreeze() } label: {
                        Circle().strokeBorder(.white, lineWidth: 5)
                            .background(Circle().fill(model.frozen != nil ? Color.white.opacity(0.85) : Color.white.opacity(0.25)))
                            .frame(width: 78, height: 78)
                    }
                    .accessibilityIdentifier("camera.check")
                    Button { model.describe() } label: {
                        Text(model.caption.isBusy ? "…" : "Describe")
                            .font(.system(size: 15, weight: .heavy))
                            .foregroundStyle(.white)
                            .frame(width: 88, height: 44)
                            .background(model.caption.state == .idle ? Color.black.opacity(0.55) : Color.blue.opacity(0.75), in: Capsule())
                    }
                    .disabled(model.caption.isBusy)
                    .accessibilityIdentifier("camera.describe")
                    .accessibilityHint("Freezes the frame and writes a caption for it")
                }
                .padding(.bottom, 18)
            }
            .padding(.horizontal, 16)
            .padding(.top, 54)
        }
    }
}
