#if os(iOS)
import AVFoundation
import PhotosUI
import SwiftUI

/// Opens the camera surface: `-camera` at launch, `bend://camera`, or the web app's bendOS "camera.open" message.
/// PHONEUX: on a photo instead of the camera (`-cameraPhoto`, `bend://camera?source=photo`, `{op: "camera.open",
/// source: "photo"}`): the surface opens with the photo picker up, and the picked photo is judged like a tapped frame.
@MainActor @Observable
final class CameraPresenter {
    enum Source { case camera, photo }
    static let shared = CameraPresenter()
    var isPresented = ProcessInfo.processInfo.arguments.contains("-camera") || ProcessInfo.processInfo.arguments.contains("-cameraPhoto")
    var source: Source = ProcessInfo.processInfo.arguments.contains("-cameraPhoto") ? .photo : .camera

    func open(source: Source) {
        self.source = source
        isPresented = true
    }
}

/// IMAGETRUE: the camera is a viewfinder for the Look surface (LookViewfinder, below): shoot, and the photo opens in Look,
/// where Bend says what it sees and you talk to it. The yes/no verdict surface (PHONECAM) stays for its receipts and tests
/// behind `-camVerdict`.
struct CameraView: View {
    static let verdict = ProcessInfo.processInfo.arguments.contains("-camVerdict")

    var body: some View {
        if Self.verdict { VerdictCameraView() } else { LookViewfinder() }
    }
}

/// PHONECAM: the camera surface. The rear camera (AVCaptureSession, 1280×720 BGRA, 15 fps) → each frame letterboxed into
/// the canvas → CameraPipeline (gate, keep-warm, one request in flight) → OUR engine in process → the big verdict.
/// Injected sources, for the simulator and the receipts (no camera there): `-camFixtures <dir>` (canvas PPMs +
/// questions.tsv, held 6 s each) or `-camSynthetic [k]` (k generated poses, vlive1 G4's clip recipe); `-camLive` starts
/// in live mode; `-camQuestion "<q>"`. IMAGETRUE: only under `-camVerdict`.
struct VerdictCameraView: View {
    @State private var model = CameraSurfaceModel(engine: CameraEngine.shared, captioner: VerdictCameraView.captioner())
    @State private var capture = CameraCapture()
    @State private var player = FramePlayer()
    @State private var sending = false
    @State private var realCamera = false
    @State private var picking = false
    @State private var picked: PhotosPickerItem?
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        CameraSurface(model: model, preview: CameraPreview(session: capture.session), onClose: {
            stop()
            dismiss()
        }, onPhoto: { picking = true }, onSendToChat: sendToChat, sending: sending)
        .ignoresSafeArea()
        .preferredColorScheme(.dark)
        .onAppear(perform: begin)
        .onDisappear(perform: stop)
        .onChange(of: model.frozen != nil) { _, frozen in
            // PHONEUX: a picked photo is its own full-resolution shot (judgePhoto set it); don't grab the camera's.
            if model.sourceLabel == "photo", frozen, model.frozenShot != nil { return }
            model.frozenShot = nil
            guard frozen, realCamera else { return }
            capture.grab { image in Task { @MainActor in if model.frozen != nil { model.frozenShot = image } } }
        }
        // PHONEUX: a photo from the library, judged like a tapped frame (the same canvas, gate and engine call).
        .photosPicker(isPresented: $picking, selection: $picked, matching: .images)
        .onChange(of: picked) { _, item in
            guard let item else { return }
            picked = nil
            Task {
                guard let data = try? await item.loadTransferable(type: Data.self) else {
                    model.photoFailed("Couldn't open that photo")
                    return
                }
                model.judgePhoto(data: data)
            }
        }
    }

    /// CAMCAP: the caption's engine. The phone: VisionLook on the in-process tower. The simulator has no Metal for it:
    /// `-camCaptionHost <base> <key>` captions through the Mac's engine harness (HostCaptioner), else the card says why.
    static func captioner() -> Captioner {
        #if targetEnvironment(simulator)
        let args = ProcessInfo.processInfo.arguments
        if let i = args.firstIndex(of: "-camCaptionHost"), i + 2 < args.count, let base = Int(args[i + 1]) {
            return HostCaptioner(base: base, key: args[i + 2])
        }
        #endif
        return VisionCaptioner()
    }

    /// ATTACHVIS: camera → chat. The frozen frame (full resolution on the real camera), else the view as it is now; JPEG
    /// to the page's composer (OSBridge.attach), and the surface closes onto the chat with the photo in its chip.
    private func sendToChat() {
        guard !sending else { return }
        sending = true
        // CAMCAP: a finished caption rides the shot (the thread gets the photo and its caption); else the plain shot.
        let caption = model.caption.text
        let finish: @Sendable (CGImage?) -> Void = { image in
            Task { @MainActor in
                guard let image, OSBridge.shared.attach(image, name: "Camera", caption: caption) else {
                    // No frame came (an interruption): the button is usable again, the surface carries on.
                    sending = false
                    EngineManager.log("camera send-to-chat: no shot")
                    return
                }
                stop()
                dismiss()
            }
        }
        if let frozen = model.frozen {
            finish(model.frozenShot ?? frozen.cgImage())
        } else if realCamera {
            capture.grab(finish)
        } else {
            finish((model.lastFrame ?? model.judged)?.cgImage())
        }
    }

    private func begin() {
        let args = ProcessInfo.processInfo.arguments
        func value(_ flag: String) -> String? { args.firstIndex(of: flag).flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil } }
        VisionLook.cameraHeld = true
        model.expectsShot = false
        if let q = value("-camQuestion") { model.setQuestion(q) }
        model.start()
        let m = model
        if let file = value("-camPhoto") {
            // Receipts: a photo file (under Documents, or absolute) as if picked, no picker.
            let url = URL(fileURLWithPath: file, relativeTo: FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0])
            if let data = try? Data(contentsOf: url) { model.judgePhoto(data: data) } else { model.photoFailed("no photo at \(file)") }
        } else if CameraPresenter.shared.source == .photo {
            model.sourceLabel = "photo"
            picking = true
        } else if let dir = value("-camFixtures") {
            let set = FixtureSet.load(URL(fileURLWithPath: dir, relativeTo: FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]))
            model.sourceLabel = "fixtures \(set.canvases.count)"
            guard !set.canvases.isEmpty else { model.phase = .failed("no fixture canvases in \(dir)"); return }
            let fps = 15.0, hold = 6.0
            let perPose = Int(fps * hold)
            player.start(fps: fps, count: perPose * set.canvases.count, frame: { i in
                let k = i / perPose
                return CameraFrame(canvas: set.canvases[k], tMs: Double(i) / fps * 1000, question: set.questions[k], index: i, pose: k)
            }, deliver: { m.deliver($0, showFrame: true) })
        } else if args.contains("-camSynthetic") {
            let k = value("-camSynthetic").flatMap(Int.init) ?? 6
            let seq = SyntheticSequence(poses: SyntheticSequence.patterns(k))
            model.sourceLabel = "synthetic \(k)"
            player.start(fps: seq.fps, count: seq.count, frame: { seq.frame($0) }, deliver: { m.deliver($0, showFrame: true) })
        } else {
            realCamera = true
            model.expectsShot = true
            model.sourceLabel = "rear camera"
            capture.start { m.deliver($0, showFrame: false) } failed: { why in
                Task { @MainActor in m.cameraFailed(why) }
            }
        }
        if args.contains("-camLive") { model.setLive(true) }
    }

    private func stop() {
        VisionLook.cameraHeld = false
        player.stop()
        capture.stop()
        model.setLive(false)
        let engine = model.pipeline.engine
        // ATTACHVIS: a chat look streaming under the surface keeps the tower; its idle close frees it after.
        DispatchQueue.global().async { if !VisionLook.shared.inFlight { (engine as? CameraEngine)?.close() } }
    }
}

/// The rear camera's frames as canvases (letterboxed), with their capture timestamps.
final class CameraCapture: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "bend.camera.capture", qos: .userInitiated)
    private var deliver: (@Sendable (CameraFrame) -> Void)?
    private var index = 0
    private var configured = false
    // ATTACHVIS: full-resolution copies of the next frame wanted (Send to chat, a freeze), by id. Capture queue only.

    private var grabID = 0
    private var grabs: [Int: @Sendable (CGImage?) -> Void] = [:]
    private var device: AVCaptureDevice?
    /// IMAGETRUE: false = no canvas per frame (the Look viewfinder judges nothing; only grabs are served).
    var letterbox = true

    /// IMAGETRUE: tap to focus: focus and exposure at `point` (the capture device's 0…1 space), then back to continuous
    /// once the scene changes (subject-area monitoring).
    func focus(at point: CGPoint) {
        queue.async {
            guard let d = self.device, (try? d.lockForConfiguration()) != nil else { return }
            defer { d.unlockForConfiguration() }
            if d.isFocusPointOfInterestSupported, d.isFocusModeSupported(.autoFocus) { d.focusPointOfInterest = point; d.focusMode = .autoFocus }
            if d.isExposurePointOfInterestSupported, d.isExposureModeSupported(.autoExpose) { d.exposurePointOfInterest = point; d.exposureMode = .autoExpose }
            d.isSubjectAreaChangeMonitoringEnabled = true
        }
    }

    /// Back to continuous focus and exposure (the scene moved on after a tap-to-focus).
    @objc func subjectChanged() {
        queue.async {
            guard let d = self.device, (try? d.lockForConfiguration()) != nil else { return }
            defer { d.unlockForConfiguration() }
            if d.isFocusModeSupported(.continuousAutoFocus) { d.focusMode = .continuousAutoFocus }
            if d.isExposureModeSupported(.continuousAutoExposure) { d.exposureMode = .continuousAutoExposure }
            d.isSubjectAreaChangeMonitoringEnabled = false
        }
    }

    /// The next frame, full resolution (1280×720 rotated to portrait), as an image of its own; nil if no frame comes
    /// within 1.5 s (an interruption) or the camera stops first.
    func grab(_ done: @escaping @Sendable (CGImage?) -> Void) {
        queue.async {
            guard self.session.isRunning else { return done(nil) }
            self.grabID += 1
            let id = self.grabID
            self.grabs[id] = done
            self.queue.asyncAfter(deadline: .now() + 1.5) { self.grabs.removeValue(forKey: id)?(nil) }
        }
    }

    func start(_ deliver: @escaping @Sendable (CameraFrame) -> Void, failed: @escaping @Sendable (String) -> Void) {
        self.deliver = deliver
        AVCaptureDevice.requestAccess(for: .video) { ok in
            guard ok else { return failed("camera access denied (Settings ▸ Eureka ▸ Camera)") }
            self.queue.async {
                if !self.configured {
                    if let why = self.configure() { return failed(why) }
                    self.configured = true
                }
                self.session.startRunning()
            }
        }
    }

    func stop() {
        queue.async {
            if self.session.isRunning { self.session.stopRunning() }
            let pending = self.grabs.values
            self.grabs.removeAll()
            pending.forEach { $0(nil) }
        }
    }

    private func configure() -> String? {
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device) else { return "no rear camera" }
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        session.sessionPreset = .hd1280x720
        guard session.canAddInput(input) else { return "cannot use the rear camera" }
        session.addInput(input)
        self.device = device
        NotificationCenter.default.addObserver(self, selector: #selector(subjectChanged), name: AVCaptureDevice.subjectAreaDidChangeNotification, object: device)
        if (try? device.lockForConfiguration()) != nil {
            if device.isFocusModeSupported(.continuousAutoFocus) { device.focusMode = .continuousAutoFocus }
            device.unlockForConfiguration()
        }
        let out = AVCaptureVideoDataOutput()
        out.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        out.alwaysDiscardsLateVideoFrames = true
        out.setSampleBufferDelegate(self, queue: queue)
        guard session.canAddOutput(out) else { return "cannot read camera frames" }
        session.addOutput(out)
        if let c = out.connection(with: .video), c.isVideoRotationAngleSupported(90) { c.videoRotationAngle = 90 }
        if (try? device.lockForConfiguration()) != nil {
            let d = CMTime(value: 1, timescale: CameraConfig.captureFPS)
            device.activeVideoMinFrameDuration = d
            device.activeVideoMaxFrameDuration = d
            device.unlockForConfiguration()
        }
        return nil
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let tMs = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer)) * 1000
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb), bpr = CVPixelBufferGetBytesPerRow(pb)
        guard let base = CVPixelBufferGetBaseAddress(pb),
              let ctx = CGContext(data: base, width: w, height: h, bitsPerComponent: 8, bytesPerRow: bpr,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue),
              let image = ctx.makeImage() else { return }
        if !grabs.isEmpty {
            // Drawn into memory of its own while the pixel buffer is still locked.
            let copy = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                                 bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
            copy?.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
            let shot = copy?.makeImage()
            let pending = grabs.values
            grabs.removeAll()
            pending.forEach { $0(shot) }
        }
        guard letterbox else { return }
        let canvas = Canvas.letterbox(image)
        deliver?(CameraFrame(canvas: canvas, tMs: tMs, question: nil, index: index, pose: nil))
        index += 1
    }
}

/// The live preview (AVCaptureVideoPreviewLayer, aspect fit so the view shows what the canvas gets).
struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    /// IMAGETRUE: the Look viewfinder fills the screen (aspect fill); the verdict surface shows what the canvas gets (fit).
    var gravity: AVLayerVideoGravity = .resizeAspect
    /// IMAGETRUE: a tap on the preview: where (in the view) and the same point in the capture device's 0…1 space.
    var onTap: ((_ view: CGPoint, _ device: CGPoint) -> Void)? = nil

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var preview: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
        var onTap: ((CGPoint, CGPoint) -> Void)?

        @objc func tapped(_ g: UITapGestureRecognizer) {
            let p = g.location(in: self)
            onTap?(p, preview.captureDevicePointConverted(fromLayerPoint: p))
        }
    }

    func makeUIView(context: Context) -> PreviewView {
        let v = PreviewView()
        v.preview.session = session
        v.preview.videoGravity = gravity
        v.backgroundColor = .black
        v.onTap = onTap
        if onTap != nil { v.addGestureRecognizer(UITapGestureRecognizer(target: v, action: #selector(PreviewView.tapped(_:)))) }
        return v
    }

    func updateUIView(_ uiView: PreviewView, context: Context) { uiView.onTap = onTap }
}
#endif
