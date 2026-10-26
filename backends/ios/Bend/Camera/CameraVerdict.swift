import Foundation

/// One verdict: the readout and where its time went.
struct CameraVerdict: Sendable, Equatable {
    var yes: Bool
    /// logit(Yes) − logit(No) at the last prompt position.
    var score: Float
    /// max(p, 1 − p), p = sigmoid(score): the softmax over the two readout logits (camprod's; not calibrated).
    var confidence: Double
    var yesBits: UInt32
    var noBits: UInt32
    /// The engine's argmax at that position (the one new id): Yes/No when the model answers in-format.
    var firstID: UInt32
    var question: String
    var towerMs: Double = 0
    var llmMs: Double = 0
    /// llb_classify_rgb, whole call.
    var engineMs: Double = 0
    /// Frame in (or tap) → verdict out, measured by the pipeline.
    var frameMs: Double = 0

    init(score: Float, yesBits: UInt32, noBits: UInt32, firstID: UInt32, question: String) {
        self.score = score
        self.yes = score > 0
        let p = 1 / (1 + exp(-Double(score)))
        self.confidence = max(p, 1 - p)
        self.yesBits = yesBits; self.noBits = noBits; self.firstID = firstID; self.question = question
    }
}

extension CameraVerdict {
    /// The console line every verdict gets (`BEND engine camera verdict …`; scripts/cam-compare.py reads it).
    func logLine(_ origin: String) -> String {
        String(format: "camera verdict %@ %@ score=%+.4f conf=%.3f frame_ms=%.1f engine_ms=%.1f tower_ms=%.1f llm_ms=%.1f bits=%08x/%08x q=\"%@\"",
               origin, yes ? "YES" : "NO", score, confidence, frameMs, engineMs, towerMs, llmMs, yesBits, noBits, question)
    }
}

/// What the pipeline needs from an engine (CameraEngine; a fake in the tests).
protocol VerdictEngine: AnyObject, Sendable {
    /// Blocks until the engine can classify (the model resident, the tower loaded), or throws why it can't.
    func ensureReady() throws
    /// Blocks for one verdict.
    func classify(_ canvas: Canvas, question: String) throws -> CameraVerdict
}
