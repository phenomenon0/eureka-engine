#if os(iOS)
import AVFoundation
import PhotosUI
import SwiftUI

/// IMAGETRUE: the camera, calm. A full-bleed viewfinder with one action (the shutter), tap-to-focus, the photo library
/// beside it, and a way out. The shot goes straight to the page's Look surface (src/app/look), where Bend says what it sees
/// and you can talk to it: the shutter grabs the full-resolution frame, holds its look canvas by a key and starts the tower
/// on it at once (VisionLook.hold), then hands the JPEG over (OSBridge.look) and closes. While the viewfinder is up, chat
/// warms and the tower opens, so the first look pays neither.
///
/// No verdicts, no gate, no engine work while framing (the yes/no surface is `-camVerdict`). Receipts: `-camPhoto <file>`
/// shows a photo as the camera's view (the simulator has no camera), `-camFixtures` / `-camSynthetic` feed frames as
/// before, `-camShoot <s>` presses the shutter after s seconds.
@MainActor @Observable
final class ViewfinderModel {
    /// What an injected source shows (the simulator, receipts); the real camera draws its preview layer instead.
    var frame: CGImage?
    /// The camera couldn't start: said calmly, with the library as the way on.
    var problem: String?
}

struct LookViewfinder: View {
    @State private var model = ViewfinderModel()
    @State private var capture = CameraCapture()
    @State private var player = FramePlayer()
    @State private var realCamera = false
    @State private var still: CGImage?
    @State private var flash = false
    @State private var focus: (point: CGPoint, id: Int)?
    @State private var picking = false
    @State private var picked: PhotosPickerItem?
    @State private var handed = false
    @State private var shooting = false
    @State private var shots = 0
    @Environment(\.dismiss) private var dismiss

    /// The lumen's warm light (VoiceShade's --vs-warm): the person's own marks (the focus ring).
    private static let warm = Color(red: 1, green: 0.886, blue: 0.749)

    var body: some View {
        ZStack {
            Color.black
            viewport
                .ignoresSafeArea()
            if let f = focus {
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .strokeBorder(Self.warm.opacity(0.9), lineWidth: 1.5)
                    .frame(width: 76, height: 76)
                    .position(f.point)
                    .transition(.scale(scale: 1.35).combined(with: .opacity))
                    .id(f.id)
                    .allowsHitTesting(false)
                    .ignoresSafeArea()
            }
            Color.white.opacity(flash ? 0.85 : 0).ignoresSafeArea().allowsHitTesting(false)
            chrome
        }
        .preferredColorScheme(.dark)
        .onAppear(perform: begin)
        .onDisappear(perform: stop)
        .photosPicker(isPresented: $picking, selection: $picked, matching: .images)
        .onChange(of: picked) { _, item in
            guard let item else { return }
            picked = nil
            Task {
                guard let data = try? await item.loadTransferable(type: Data.self), let image = Canvas.decode(data) else {
                    model.problem = "Couldn't open that photo"
                    return
                }
                handOff(image, shotAt: Date().timeIntervalSince1970 * 1000)
            }
        }
        .accessibilityIdentifier("look.viewfinder")
    }

    /// The camera's view (or the shot, frozen while it goes): fills the screen; tap to focus.
    @ViewBuilder private var viewport: some View {
        GeometryReader { geo in
            ZStack {
                if let still {
                    Image(decorative: still, scale: 1).resizable()
                        .aspectRatio(contentMode: realCamera ? .fill : .fit)
                        .frame(width: geo.size.width, height: geo.size.height).clipped()
                } else if let frame = model.frame {
                    // An injected source (a photo, fixtures: the simulator, receipts) is shown whole.
                    Image(decorative: frame, scale: 1).resizable().interpolation(.high).scaledToFit()
                        .frame(width: geo.size.width, height: geo.size.height)
                        .contentShape(Rectangle())
                        .onTapGesture(coordinateSpace: .local) { p in ring(at: p) }
                } else if realCamera {
                    CameraPreview(session: capture.session, gravity: .resizeAspectFill) { view, device in
                        ring(at: view)
                        capture.focus(at: device)
                    }
                }
            }
        }
        .accessibilityIdentifier("look.preview")
    }

    private var chrome: some View {
        VStack(spacing: 0) {
            HStack {
                Button(action: close) {
                    Image(systemName: "xmark").font(.system(size: 16, weight: .regular)).foregroundStyle(.white)
                        .frame(width: 44, height: 44).background(.ultraThinMaterial, in: Circle())
                }
                .accessibilityLabel("Close the camera")
                .accessibilityIdentifier("look.close")
                Spacer()
            }
            .padding(.horizontal, 18)
            .padding(.top, 6)
            Spacer()
            if let problem = model.problem {
                Text(problem)
                    .font(.system(size: 15, weight: .regular))
                    .foregroundStyle(.white.opacity(0.85))
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 18).padding(.vertical, 12)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
                    .padding(.bottom, 18)
                    .accessibilityIdentifier("look.problem")
            }
            Text(hint)
                .font(.system(size: 13, weight: .regular))
                .foregroundStyle(.white.opacity(0.62))
                .frame(height: 18)
                .padding(.bottom, 16)
                .accessibilityIdentifier("look.hint")
            HStack(alignment: .center) {
                Button { picking = true } label: {
                    Image(systemName: "photo.on.rectangle").font(.system(size: 20, weight: .light)).foregroundStyle(.white)
                        .frame(width: 52, height: 52)
                        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 15, style: .continuous))
                }
                .accessibilityLabel("Pick a photo")
                .accessibilityIdentifier("look.library")
                Spacer()
                Button(action: shoot) {
                    ZStack {
                        Circle().strokeBorder(.white, lineWidth: 4).frame(width: 78, height: 78)
                        Circle().fill(.white.opacity(shooting ? 0.55 : 1)).frame(width: 64, height: 64)
                    }
                }
                .buttonStyle(ShutterStyle())
                .disabled(shooting || !canShoot)
                .opacity(canShoot ? 1 : 0.4)
                .accessibilityLabel("Take the photo and look at it")
                .accessibilityIdentifier("look.shutter")
                Spacer()
                Color.clear.frame(width: 52, height: 52)
            }
            .padding(.horizontal, 34)
            .padding(.bottom, 22)
        }
        .background(alignment: .bottom) {
            LinearGradient(colors: [.clear, .black.opacity(0.5)], startPoint: .top, endPoint: .bottom)
                .frame(height: 230).ignoresSafeArea().allowsHitTesting(false)
        }
    }

    private var canShoot: Bool { realCamera || model.frame != nil }

    /// One small guidance line: what to do, never how it works.
    private var hint: String {
        if shooting { return "Looking…" }
        if !canShoot { return model.problem == nil ? "Starting the camera…" : "Pick a photo to look at" }
        return realCamera ? "Tap to focus · press to look" : "Press to look"
    }

    private func ring(at p: CGPoint) {
        let id = (focus?.id ?? 0) + 1
        withAnimation(.spring(response: 0.32, dampingFraction: 0.7)) { focus = (p, id) }
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) {
            if focus?.id == id { withAnimation(.easeOut(duration: 0.3)) { focus = nil } }
        }
    }

    /// The shutter: the next full-resolution frame (or the injected view), frozen, to Look.
    private func shoot() {
        guard !shooting, canShoot else { return }
        shooting = true
        shots += 1
        let shotAt = Date().timeIntervalSince1970 * 1000
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
        withAnimation(.easeOut(duration: 0.06)) { flash = true }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.08) { withAnimation(.easeOut(duration: 0.28)) { flash = false } }
        if realCamera {
            capture.grab { image in
                Task { @MainActor in
                    guard let image else { shooting = false; model.problem = "The camera didn't give a frame. Try again."; return }
                    handOff(image, shotAt: shotAt)
                }
            }
        } else if let frame = model.frame {
            handOff(frame, shotAt: shotAt)
        }
    }

    /// The shot to the page's Look surface: held by key with its tower started, the JPEG made off the main thread.
    private func handOff(_ image: CGImage, shotAt: Double) {
        shooting = true
        still = image
        Task.detached(priority: .userInitiated) {
            let key = VisionLook.hold(image)
            let scale = min(1, 1280 / Double(max(image.width, image.height)))
            let (w, h) = (max(1, Int((Double(image.width) * scale).rounded())), max(1, Int((Double(image.height) * scale).rounded())))
            let jpeg = OSBridge.jpeg(image, edge: 1280, quality: 0.85)
            await MainActor.run {
                guard let jpeg, OSBridge.shared.look(jpeg: jpeg, width: w, height: h, key: key, shotAt: shotAt) else {
                    shooting = false
                    still = nil
                    model.problem = "Couldn't hand the photo over. Try again."
                    return
                }
                handed = true
                stop()
                dismiss()
            }
        }
    }

    private func close() {
        stop()
        dismiss()
    }

    private func begin() {
        let args = ProcessInfo.processInfo.arguments
        func value(_ flag: String) -> String? { args.firstIndex(of: flag).flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil } }
        VisionLook.cameraHeld = true
        handed = false
        // While the person frames: chat warm and the tower open (the shot's look then pays neither).
        if VisionLook.unavailableReason == nil {
            EngineManager.onEngineThread("bend.engine.warm") {
                let t0 = Date()
                do {
                    try CameraEngine.shared.ensureReady(tower: true, ahead: true)
                    EngineManager.log(String(format: "look viewfinder warm ms=%.0f footprint_mb=%d", Date().timeIntervalSince(t0) * 1000, EngineManager.footprint()))
                } catch {
                    EngineManager.log("look viewfinder warm failed: \(error.localizedDescription)")
                }
            }
        }
        let m = model
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        if let file = value("-camPhoto") {
            let url = URL(fileURLWithPath: file, relativeTo: documents)
            if let data = try? Data(contentsOf: url), let image = Canvas.decode(data) { model.frame = image } else { model.problem = "No photo at \(file)" }
        } else if let dir = value("-camFixtures") {
            let set = FixtureSet.load(URL(fileURLWithPath: dir, relativeTo: documents))
            guard !set.canvases.isEmpty else { model.problem = "No fixture canvases in \(dir)"; return }
            let perPose = 15 * 6
            player.start(fps: 15, count: perPose * set.canvases.count, frame: { i in
                CameraFrame(canvas: set.canvases[i / perPose], tMs: Double(i) / 15 * 1000, question: nil, index: i, pose: i / perPose)
            }, deliver: { f in
                let image = f.canvas.cgImage()
                Task { @MainActor in m.frame = image }
            })
        } else if args.contains("-camSynthetic") {
            let seq = SyntheticSequence(poses: SyntheticSequence.patterns(value("-camSynthetic").flatMap(Int.init) ?? 6))
            player.start(fps: seq.fps, count: seq.count, frame: { seq.frame($0) }, deliver: { f in
                let image = f.canvas.cgImage()
                Task { @MainActor in m.frame = image }
            })
        } else {
            realCamera = true
            capture.letterbox = false
            capture.start { _ in } failed: { why in
                Task { @MainActor in
                    m.problem = why == "no rear camera" ? "No camera here · pick a photo" : "The camera can't start (\(why)) · pick a photo"
                }
            }
        }
        if CameraPresenter.shared.source == .photo { picking = true }
        if let s = value("-camShoot").flatMap(Double.init) {
            DispatchQueue.main.asyncAfter(deadline: .now() + s) { if shots == 0 { shoot() } }
        }
    }

    private func stop() {
        VisionLook.cameraHeld = false
        player.stop()
        capture.stop()
        // A shot went to Look: its look keeps the tower (the idle close frees it after). Closed without one: free it now.
        if !handed {
            DispatchQueue.global().async { if !VisionLook.shared.inFlight { CameraEngine.shared.close() } }
        }
    }
}

/// The shutter gives under the thumb.
private struct ShutterStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.92 : 1)
            .animation(.spring(response: 0.22, dampingFraction: 0.6), value: configuration.isPressed)
    }
}
#endif
