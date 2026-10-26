import Foundation

/// PHONECAM: the camera surface's fixed numbers, one place (docs/CAMERA.md).
///
/// The loop is vlive1's (the vlive1 report) and camprod's (the camprod report), moved in
/// process: frame → S×S canvas → pixel gate → (fire) → OUR engine on the resident 4B session: the Metal tower on the
/// canvas, its rows spliced into the prompt, one prefill, the head's Yes/No logits at the last prompt position → verdict.
enum CameraConfig {
    /// The canvas the tower sees: side × side RGB, letterboxed on black. 320 → (320/32)² = 100 image tokens (Qwen3.5's
    /// patch 16, merge 2). vlive1 G0: 49 / 100 / 196 tokens at 224 / 320 / 448; 100 is the camprod product size.
    static let canvasSide = 320
    static var imageTokens: Int { (canvasSide / 32) * (canvasSide / 32) }

    /// Rows the engine's session image area holds (llb_vision_area, reserved once per session): any canvas up to 1024
    /// tokens (1024 px), so changing canvasSide never needs an engine change. ~10 MB at the 4B's width.
    static let imageArea: Int32 = 1024
    /// The 4B backbone width the tower projects to (the mmproj's projection_dim).
    static let backboneWidth: Int32 = 2560

    /// The readout: logit("Yes") − logit("No") at the last prompt position (vlive1's ids; Qwen3.5's tokenizer).
    static let yesID: UInt32 = 9175
    static let noID: UInt32 = 2665

    /// vlive1's prompt, unchanged: the 33-id prefix (…<|vision_start|>), the image's pads, then the question turn.
    static let system = "You are a careful visual inspector. Look at the photo and answer the question with a single word: Yes or No."
    static var prefixText: String { "<|im_start|>system\n\(system)<|im_end|>\n<|im_start|>user\n<|vision_start|>" }
    static func suffixText(_ question: String) -> String {
        "<|vision_end|>\(question)<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    }
    /// vlive1 receipts/g0/prompt.json prefix_ids: the prefix must tokenize to exactly these (checked at prepare).
    static let vlive1Prefix: [UInt32] = [248045, 8678, 198, 2523, 513, 264, 16097, 8851, 43241, 13, 8919, 506, 279, 6345,
                                         321, 4087, 279, 3296, 440, 264, 3074, 3299, 25, 7179, 466, 2233, 13, 248046,
                                         198, 248045, 846, 198, 248053]
    static let defaultQuestion = "Is there a person in this image?"

    /// The pixel gate (vlive1 PixelGate, camprod's auto mode): gray block means on a rows × cols grid, mean abs diff.
    static let gateRows = 36
    static let gateCols = 64
    static let gateMotion: Float = 4
    static let gateChange: Float = 6
    static let gateSettleMs: Double = 300

    /// Keep-warm (camprod's finding, reproduced on this engine: receipts/idle): after ≥ ~1.5 s idle a frame costs ~2×
    /// (Mac 148 → ~290 ms). While the surface is live, a frame the gate skips runs one discarded engine request when
    /// the engine has idled longer than this. Manual mode does the same while the person touched it in the last minute.
    static let keepWarmAfter: TimeInterval = 0.8
    static let keepWarmTouchWindow: TimeInterval = 60

    /// The camera's frame pace into the gate (frames are dropped while the gate is still on the previous one).
    static let captureFPS: Int32 = 15

    /// Documents/models/<dir>/<file>: the 4B's vision tower (unsloth/Qwen3.5-4B-GGUF mmproj-F16, sha256 cd88edcf…).
    static let mmprojDir = "qwen35-4b-mmproj"
    static let mmprojFile = "mmproj-F16.gguf"
    static let mmprojSHA256Prefix = "cd88edcf"
}
