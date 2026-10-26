import Foundation

/// IOSFETCH: what Bend downloads on first run, and from where (docs/MODELS.md).
///
/// The file set is exactly what the engine opens: per model dir, the GGUF + its llb sidecar (`manifest.json`,
/// `table.bin`, `tokenizer.bin`, `chat.flags`), the layout `scripts/engine-device-run.sh models` pushes; and the 4B's
/// vision tower (the camera upgrade, opt-in). Every file has its byte count and sha256: a download is kept only when
/// both match. The URL of a file is `<base>/<dir>/<name>`, so a host is the models dir as it sits on the phone.
struct ModelFile: Sendable, Hashable {
    let dir: String
    let name: String
    let bytes: Int64
    let sha256: String

    var path: String { "\(dir)/\(name)" }
}

struct ModelItem: Sendable, Identifiable, Hashable {
    enum ID: String, Sendable { case chat, decision, vision }
    let id: ID
    /// The row's title and what it adds, as the setup card says them.
    let title: String
    let detail: String
    let files: [ModelFile]
    /// Required items are the engines (no chat without them); the rest are opt-in upgrades.
    let required: Bool

    var bytes: Int64 { files.reduce(0) { $0 + $1.bytes } }
    var dir: String { files[0].dir }
}

enum ModelManifest {
    /// THE policy point: where Bend's models come from. One constant; nothing else in the app names a host.
    /// Public HF mirror of the models tree (jethrovic/bend-models; GGUFs = Apache-2.0 upstream bytes, docs/MODELS.md). Relocate
    /// the host by changing this one string. Unreachable host: a fresh install shows the setup
    /// card and a clear "can't reach the model host" with Retry, never a dead end.
    static let BASE_URL = "https://huggingface.co/jethrovic/bend-models/resolve/main"

    /// The host this launch uses. The debug override, in order: the `-modelsBaseURL <url>` launch argument (it is also
    /// remembered, DEBUG builds only, so a later plain launch from the home screen keeps it), BEND_MODELS_BASE_URL
    /// (the Mac harness), the remembered one, then BASE_URL. Release builds read BASE_URL only.
    static var baseURL: URL {
        #if DEBUG
        let defaults = UserDefaults.standard
        let args = ProcessInfo.processInfo.arguments
        if let i = args.firstIndex(of: "-modelsBaseURL"), i + 1 < args.count, let url = URL(string: args[i + 1]) {
            defaults.set(args[i + 1], forKey: overrideKey)
            return url
        }
        if let env = ProcessInfo.processInfo.environment["BEND_MODELS_BASE_URL"], let url = URL(string: env) { return url }
        if let saved = defaults.string(forKey: overrideKey), let url = URL(string: saved) { return url }
        #endif
        return URL(string: BASE_URL)!
    }

    static var overridden: Bool { baseURL.absoluteString != BASE_URL }
    static let overrideKey = "bend.models.baseURL"

    static func url(_ file: ModelFile, base: URL = baseURL) -> URL {
        base.appendingPathComponent(file.dir).appendingPathComponent(file.name)
    }

    // MARK: the file set (sha256 + bytes measured from the files the phone runs today: docs/MODELS.md)

    static let qwen35_4b = [
        ModelFile(dir: "qwen35-4b-q4km", name: "manifest.json", bytes: 77_564, sha256: "b7d3f1dcddfad51632bfc35ec02a19112f54470aa141b62ba31df3ac2cbf8205"),
        ModelFile(dir: "qwen35-4b-q4km", name: "table.bin", bytes: 4_096, sha256: "c769fc9a24d6d10618a4afe524eb4e03b7e49f29adb3f9a0ce548bfd6caf6e04"),
        ModelFile(dir: "qwen35-4b-q4km", name: "tokenizer.bin", bytes: 7_460_363, sha256: "de71a648a71960a539d4d660ebff804a63a691ba74aea89e3e09dc7bf4553ad9"),
        ModelFile(dir: "qwen35-4b-q4km", name: "chat.flags", bytes: 8, sha256: "1e64839b42edb4947530d0b2de55948bd3b2872f879e3e16482919b114d75702"),
        // Last: the engine counts a model as present once its GGUF is there (EngineManager.hasModel).
        ModelFile(dir: "qwen35-4b-q4km", name: "Qwen3.5-4B-Q4_K_M.gguf", bytes: 2_740_937_888, sha256: "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4"),
    ]

    static let kev_4b = [
        ModelFile(dir: "kev-4b-q4km", name: "manifest.json", bytes: 77_903, sha256: "75afba88188b33937beab56d9383c4773ac9522689896e95115ccd719436b83e"),
        ModelFile(dir: "kev-4b-q4km", name: "table.bin", bytes: 4_096, sha256: "32a1ed1017c666b05e0841b46b6fbbd14fdc9c27cb59d2e0007f292bb500a042"),
        ModelFile(dir: "kev-4b-q4km", name: "tokenizer.bin", bytes: 7_460_363, sha256: "26a365388799957aaca11961f6106c7bb2f0954ef6255c545f98970a9a41c8d5"),
        ModelFile(dir: "kev-4b-q4km", name: "chat.flags", bytes: 8, sha256: "783d7805356f611b21330624c0568b9df7b51ecae255eb190603cf341c1cf9ce"),
        ModelFile(dir: "kev-4b-q4km", name: "Kev-4B-Q4_K_M.gguf", bytes: 3_033_489_824, sha256: "33ae6b18926502b2209a1bf7d3b61a350d65938441515c686bb87d226a19eff9"),
    ]

    /// IOS-POCKET: the 0.8B chat fixture (Kev-0.8B-Q8_0, d 1024, 24 layers) that the chat cell declares. Staged by
    /// ios-5of5 (IOS-ASSETS-STAGED.md) with the engine's own gguf2pack sidecar, oracle-checked byte-for-byte against
    /// kev-4b's shipped sidecar first. The 0.8B pocket port (EngineID.dir) is still open: nothing selects this dir
    /// yet, so it is a staged asset, not a setup item, and items() does not offer it.
    static let kev_0_8b = [
        ModelFile(dir: "kev-0.8b-q8", name: "manifest.json", bytes: 54_182, sha256: "0ba72c546a5ab19958b1d8a51a0d277ce0a301dcb02a9901ac990b09ea9a10ca"),
        ModelFile(dir: "kev-0.8b-q8", name: "table.bin", bytes: 3_072, sha256: "2d032a796ba44f985757d43d6b5e3a6023067de2872f41b8d48ab11a244e30a4"),
        ModelFile(dir: "kev-0.8b-q8", name: "tokenizer.bin", bytes: 7_460_363, sha256: "26a365388799957aaca11961f6106c7bb2f0954ef6255c545f98970a9a41c8d5"),
        ModelFile(dir: "kev-0.8b-q8", name: "chat.flags", bytes: 8, sha256: "783d7805356f611b21330624c0568b9df7b51ecae255eb190603cf341c1cf9ce"),
        // Last: the engine counts a model as present once its GGUF is there (EngineManager.hasModel).
        ModelFile(dir: "kev-0.8b-q8", name: "Kev-0.8B-Q8_0.gguf", bytes: 812_406_304, sha256: "27278f34eb3273bceea4c053dc50dd61a5161da21a718c4aacdf8fd5830771d0"),
    ]

    /// unsloth/Qwen3.5-4B-GGUF mmproj-F16 (CameraConfig.mmprojDir/mmprojFile; 641 MiB).
    static let vision4b = [
        ModelFile(dir: "qwen35-4b-mmproj", name: "mmproj-F16.gguf", bytes: 672_423_616, sha256: "cd88edcf8d031894960bb0c9c5b9b7e1fea6ebee02b9f7ce925a00d12891f864"),
    ]

    static func files(dir: String) -> [ModelFile] {
        switch dir {
        case "kev-4b-q4km": kev_4b
        case "kev-0.8b-q8": kev_0_8b
        case "qwen35-4b-mmproj": vision4b
        default: qwen35_4b
        }
    }

    /// The items this build needs, in download order. Camera vision needs Qwen's backbone, so it is offered only
    /// with the optional two-model pairing. `chatDir`/`decisionDir` are EngineID's dirs.
    static func items(chatDir: String, decisionDir: String) -> [ModelItem] {
        var items = [ModelItem(id: .chat, title: chatDir == "kev-4b-q4km" ? "Kev-4B" : "Qwen3.5-4B",
                               detail: chatDir == decisionDir ? "Chat and decisions" : "Chat",
                               files: files(dir: chatDir), required: true)]
        if decisionDir != chatDir {
            items.append(ModelItem(id: .decision, title: "Kev-4B", detail: "Decisions: ranks your options",
                                   files: files(dir: decisionDir), required: true))
        }
        if chatDir == "qwen35-4b-q4km" {
            items.append(ModelItem(id: .vision, title: "Camera vision", detail: "Yes/no answers about what the camera sees",
                                   files: vision4b, required: false))
        }
        return items
    }
}
