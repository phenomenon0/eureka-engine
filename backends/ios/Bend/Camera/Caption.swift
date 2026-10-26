import CoreGraphics
import Foundation
import Observation

/// CAMCAP: the camera's open caption (docs/CAMERA.md § Caption). The verdict path answers one yes/no; the caption path
/// asks the engine's look path (VisionLook: greedy decoding over llb_classify_rgb, the same path chat photos use) to
/// describe the frozen frame in a sentence, streamed into a card. The engine is behind `Captioner`, so the surface, the
/// tests and the simulator (which has no Metal for our engine) share one controller.
enum CaptionConfig {
    /// What the card asks. One short sentence: every id costs a tower + prefill on the b0ffd9af engine.
    static let question = "Caption this photo in one short sentence."
    static let maxNew = 48
}

struct CaptionResult: Sendable, Equatable {
    var text: String
    var firstMs: Double
    var totalMs: Double
    var finish: String
}

protocol Captioner: AnyObject, Sendable {
    /// Why this build can't caption (the verdict's reasons: simulator Metal, -kevChat, the model, the tower), or nil.
    var unavailableReason: String? { get }
    /// One caption of `image`, each piece through `onPiece` as it lands (return false: stop now).
    func caption(_ image: CGImage, onPiece: @escaping @Sendable (String) -> Bool) async throws -> CaptionResult
}

/// One caption run's stop flag and its text so far (pieces arrive on the engine's thread, the card reads on main).
final class CaptionRun: @unchecked Sendable {
    private let lock = NSLock()
    private var stopped = false
    private var text = ""
    var isStopped: Bool { lock.withLock { stopped } }
    func stop() { lock.withLock { stopped = true } }
    func append(_ piece: String) -> String { lock.withLock { text += piece; return text } }
}

/// The caption card's state and its run. Main actor; the engine work is off it.
@MainActor @Observable
final class CaptionController {
    enum State: Equatable {
        case idle
        /// Asked; the engine is warming or reading the frame (no words yet).
        case loading
        case streaming(String)
        case done(String, ms: Double)
        /// The engine said no, or nothing came: words and a way to retry (never a dead end).
        case failed(String)
        /// This build can't caption at all (the simulator's Metal, -kevChat, a missing model): the verdict path's words.
        case unavailable(String)
    }

    private(set) var state: State = .idle
    let captioner: Captioner?
    /// The pipeline pauses its own fires/warm-ups while a caption runs (they would queue between the look's ids).
    var onBusy: (@MainActor (Bool) -> Void)?
    /// The caption was finalised (for the voice path).
    var onDone: (@MainActor (String) -> Void)?
    private var run: CaptionRun?

    init(captioner: Captioner?) { self.captioner = captioner }

    var isBusy: Bool { state == .loading || { if case .streaming = state { return true } else { return false } }() }
    /// The finished caption, or nil.
    var text: String? { if case .done(let t, _) = state { return t } else { return nil } }

    func ask(_ image: CGImage) {
        cancel()
        guard let captioner else { state = .unavailable("No engine here"); return }
        if let why = captioner.unavailableReason { state = .unavailable(why); return }
        let mine = CaptionRun()
        run = mine
        state = .loading
        onBusy?(true)
        let show: @Sendable (String) -> Void = { [weak self] soFar in
            Task { @MainActor in self?.piece(soFar, of: mine) }
        }
        Task {
            let result: Result<CaptionResult, Error>
            do {
                result = .success(try await captioner.caption(image) { piece in
                    show(mine.append(piece))
                    return !mine.isStopped
                })
            } catch { result = .failure(error) }
            finish(result, of: mine)
        }
    }

    /// Asked, the frame not in hand yet: the card says so at once.
    func expect() { cancel(); state = .loading }

    func fail(_ why: String) { cancel(); state = .failed(why) }

    /// Back to the live view (or a new frame): the run stops, the card goes.
    func cancel() {
        guard let mine = run else { state = .idle; return }
        mine.stop()
        run = nil
        state = .idle
        onBusy?(false)
    }

    private func piece(_ soFar: String, of mine: CaptionRun) {
        guard run === mine else { return }
        if case .streaming(let shown) = state, shown.count > soFar.count { return }  // a late hop never shortens the card
        state = .streaming(soFar)
    }

    private func finish(_ result: Result<CaptionResult, Error>, of mine: CaptionRun) {
        guard run === mine else { return }
        run = nil
        onBusy?(false)
        switch result {
        case .success(let r):
            let text = r.text.trimmingCharacters(in: .whitespacesAndNewlines)
            if text.isEmpty {
                state = .failed("Eureka had nothing to say about this frame. Try again.")
            } else {
                state = .done(text, ms: r.totalMs)
                onDone?(text)
            }
        case .failure(let e):
            let why = e.localizedDescription
            state = why.hasPrefix("no-vision:") || why.hasPrefix("no-models:") ? .unavailable(why) : .failed(why)
        }
    }
}
