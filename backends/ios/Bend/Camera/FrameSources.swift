import Foundation

/// One frame into the pipeline: the canvas, its capture time (ms, monotonic within the source), and, for fixtures,
/// the question that goes with it.
struct CameraFrame: Sendable {
    var canvas: Canvas
    var tMs: Double
    var question: String?
    var index: Int
    /// Which pose (synthetic / fixtures) the frame belongs to, or nil in a transition (the receipts' ground truth).
    var pose: Int?
}

/// Synthetic "camera" (no human, no camera): K poses, each held, with sensor noise, slide/crossfade transitions
/// between them and a short hand shake in some — vlive1 G4's clip recipe (tools/mkclip.py), generated in process.
/// Deterministic (a fixed LCG): the same index gives the same frame.
struct SyntheticSequence: Sendable {
    let poses: [Canvas]
    let fps: Double
    let holdS: Double
    let transitionS: Double
    let shakePoses: Set<Int>
    let noise: Int  // uniform ±noise per channel (±3: sigma 2, as G4)

    init(poses: [Canvas], fps: Double = 15, holdS: Double = 5, transitionS: Double = 1.5, shakePoses: Set<Int> = [2, 4], noise: Int = 3) {
        self.poses = poses; self.fps = fps; self.holdS = holdS; self.transitionS = transitionS
        self.shakePoses = shakePoses; self.noise = noise
    }

    var period: Double { holdS + transitionS }
    var count: Int { Int((Double(poses.count) * period - transitionS) * fps) }

    /// Generated poses when no fixtures are at hand: flat fields + bars + a disc, each clearly different.
    static func patterns(_ k: Int, side: Int = CameraConfig.canvasSide) -> [Canvas] {
        (0..<k).map { i in
            var rgb = [UInt8](repeating: 0, count: side * side * 3)
            let base: [(UInt8, UInt8, UInt8)] = [(200, 60, 50), (40, 160, 70), (50, 80, 200), (220, 200, 60), (150, 60, 170), (60, 190, 190)]
            let (r, g, b) = base[i % base.count]
            for y in 0..<side {
                for x in 0..<side {
                    let bar = ((x / (12 + 6 * i)) + i) % 2 == 0
                    let dx = x - side / 2 - (i * 23) % 80 + 40, dy = y - side / 2
                    let disc = dx * dx + dy * dy < (40 + 8 * i) * (40 + 8 * i)
                    let p = (y * side + x) * 3
                    rgb[p] = disc ? 255 - r : bar ? r : r / 2
                    rgb[p + 1] = disc ? 255 - g : bar ? g : g / 2
                    rgb[p + 2] = disc ? 255 - b : bar ? b : b / 2
                }
            }
            return Canvas(side: side, rgb: rgb)
        }
    }

    func frame(_ i: Int) -> CameraFrame {
        let t = Double(i) / fps
        let k = min(Int(t / period), poses.count - 1)
        let u = t - Double(k) * period
        let side = poses[0].side
        var rgb: [UInt8]
        var pose: Int?
        if u < holdS || k == poses.count - 1 {
            pose = k
            var dx = 0, dy = 0
            // the hand shake: ±4 px for 0.6 s in the middle of the hold
            if shakePoses.contains(k), u > 2.0, u < 2.6 {
                var r = UInt64(i) &* 6364136223846793005 &+ 1442695040888963407
                r ^= r >> 33
                dx = Int(r % 9) - 4; dy = Int((r >> 8) % 9) - 4
            }
            rgb = Self.shift(poses[k].rgb, side: side, dx: dx, dy: dy)
        } else {
            // slide + crossfade into the next pose
            let a = (u - holdS) / transitionS
            let from = Self.shift(poses[k].rgb, side: side, dx: -Int(a * Double(side) / 3), dy: 0)
            let to = poses[k + 1].rgb
            rgb = [UInt8](repeating: 0, count: from.count)
            for j in rgb.indices { rgb[j] = UInt8((Double(from[j]) * (1 - a) + Double(to[j]) * a).rounded()) }
        }
        if noise > 0 {
            var s = UInt64(i + 1) &* 0x9E3779B97F4A7C15
            let span = UInt64(2 * noise + 1)
            for j in rgb.indices {
                s = s &* 6364136223846793005 &+ 1442695040888963407
                let n = Int((s >> 33) % span) - noise
                rgb[j] = UInt8(clamping: Int(rgb[j]) + n)
            }
        }
        return CameraFrame(canvas: Canvas(side: side, rgb: rgb), tMs: t * 1000, question: nil, index: i, pose: pose)
    }

    static func shift(_ src: [UInt8], side: Int, dx: Int, dy: Int) -> [UInt8] {
        if dx == 0 && dy == 0 { return src }
        var out = [UInt8](repeating: 0, count: src.count)
        for y in 0..<side {
            let sy = y - dy
            guard sy >= 0, sy < side else { continue }
            for x in 0..<side {
                let sx = x - dx
                guard sx >= 0, sx < side else { continue }
                let o = (y * side + x) * 3, s = (sy * side + sx) * 3
                out[o] = src[s]; out[o + 1] = src[s + 1]; out[o + 2] = src[s + 2]
            }
        }
        return out
    }
}

/// Fixture "camera": canvases from a folder (`<name>.ppm`, P6 at the canvas size, byte for byte), each held still for
/// holdS, with its question from `questions.tsv` (`<name>\t<question>` lines) when there is one. The vlive1 battery
/// canvases go in as they are, so the verdict bits can be checked against the references.
struct FixtureSet: Sendable {
    var names: [String]
    var canvases: [Canvas]
    var questions: [String?]

    static func load(_ dir: URL, side: Int = CameraConfig.canvasSide) -> FixtureSet {
        let fm = FileManager.default
        var q: [String: String] = [:]
        if let t = try? String(contentsOf: dir.appendingPathComponent("questions.tsv"), encoding: .utf8) {
            for line in t.split(separator: "\n") {
                let p = line.split(separator: "\t", maxSplits: 1)
                if p.count == 2 { q[String(p[0])] = String(p[1]) }
            }
        }
        let files = ((try? fm.contentsOfDirectory(atPath: dir.path)) ?? []).filter { $0.hasSuffix(".ppm") }.sorted()
        var set = FixtureSet(names: [], canvases: [], questions: [])
        for f in files {
            guard let d = fm.contents(atPath: dir.appendingPathComponent(f).path), let c = Canvas.ppm(d, side: side) else { continue }
            let name = String(f.dropLast(4))
            set.names.append(name); set.canvases.append(c); set.questions.append(q[name])
        }
        return set
    }
}

/// Plays a frame function at its fps in real time on a timer (the surface's injected sources).
final class FramePlayer: @unchecked Sendable {
    private let queue = DispatchQueue(label: "bend.camera.player", qos: .userInitiated)
    private var timer: DispatchSourceTimer?
    private var i = 0

    func start(fps: Double, count: Int?, frame: @escaping @Sendable (Int) -> CameraFrame, deliver: @escaping @Sendable (CameraFrame) -> Void) {
        stop()
        i = 0
        let t = DispatchSource.makeTimerSource(queue: queue)
        t.schedule(deadline: .now(), repeating: 1 / fps, leeway: .milliseconds(2))
        t.setEventHandler { [weak self] in
            guard let self else { return }
            let n = count ?? Int.max
            deliver(frame(self.i % n))
            self.i += 1
        }
        timer = t
        t.resume()
    }

    func stop() {
        timer?.cancel()
        timer = nil
    }
}
