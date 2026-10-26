import Foundation

/// vlive1's PixelGate (tools/vl1client.py), ported line for line: judge a frame once the view has changed from the
/// last judged frame AND has then been still for `settleMs`. Gray (0.299 R + 0.587 G + 0.114 B, Float32), block-
/// averaged to a rows × cols grid (the frame cropped to a multiple of the grid first, as the numpy reshape does), mean
/// abs diff. "Still" is measured against the frame that opened the still window (the anchor), so a slow drift whose
/// every step is under the motion bar still counts as motion. Timestamps are the frames' capture times, not wall time.
struct PixelGate: Sendable {
    enum State: String, Sendable { case moving, `static`, fire }
    struct Step: Sendable {
        var fire: Bool
        var state: State
        var dAnchor: Float
        var dRef: Float?
    }

    let rows: Int, cols: Int
    let motion: Float, change: Float, settleMs: Double
    private(set) var anchor: [Float]?
    private(set) var ref: [Float]?
    private var stillSince: Double = 0

    init(rows: Int = CameraConfig.gateRows, cols: Int = CameraConfig.gateCols, motion: Float = CameraConfig.gateMotion,
         change: Float = CameraConfig.gateChange, settleMs: Double = CameraConfig.gateSettleMs) {
        self.rows = rows; self.cols = cols; self.motion = motion; self.change = change; self.settleMs = settleMs
    }

    /// The grid of block means of an RGB frame (w × h × 3 bytes, row stride w*3).
    func small(_ rgb: UnsafeBufferPointer<UInt8>, width w: Int, height h: Int) -> [Float] {
        let bh = (h - h % rows) / rows, bw = (w - w % cols) / cols
        var out = [Float](repeating: 0, count: rows * cols)
        let n = Double(bh * bw)
        for gy in 0..<rows {
            for gx in 0..<cols {
                var sum = 0.0
                for y in (gy * bh)..<(gy * bh + bh) {
                    var p = (y * w + gx * bw) * 3
                    for _ in 0..<bw {
                        let g: Float = Float(rgb[p]) * 0.299 + Float(rgb[p + 1]) * 0.587 + Float(rgb[p + 2]) * 0.114
                        sum += Double(g)
                        p += 3
                    }
                }
                out[gy * cols + gx] = Float(sum / n)
            }
        }
        return out
    }

    static func meanAbs(_ a: [Float], _ b: [Float]) -> Float {
        var s = 0.0
        for i in a.indices { s += Double(abs(a[i] - b[i])) }
        return Float(s / Double(a.count))
    }

    mutating func step(_ rgb: UnsafeBufferPointer<UInt8>, width: Int, height: Int, tMs: Double) -> Step {
        let s = small(rgb, width: width, height: height)
        let dAnchor = anchor.map { Self.meanAbs(s, $0) } ?? 1e9
        if dAnchor > motion {
            anchor = s
            stillSince = tMs
            return Step(fire: false, state: .moving, dAnchor: dAnchor, dRef: nil)
        }
        let dRef = ref.map { Self.meanAbs(s, $0) } ?? 1e9
        if tMs - stillSince >= settleMs && dRef > change {
            ref = s
            return Step(fire: true, state: .fire, dAnchor: dAnchor, dRef: dRef)
        }
        return Step(fire: false, state: .static, dAnchor: dAnchor, dRef: dRef)
    }

    /// camprod: a manual check marks its view as judged (the gate won't fire again on the same view).
    mutating func markJudged(_ rgb: UnsafeBufferPointer<UInt8>, width: Int, height: Int) {
        ref = small(rgb, width: width, height: height)
    }

    /// camprod: a new question re-arms the gate (the current view gets judged again once it is still).
    mutating func rearm() { ref = nil }
}
