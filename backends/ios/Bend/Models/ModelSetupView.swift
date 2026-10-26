#if os(iOS)
import SwiftUI

extension ModelStore {
    /// The app's store: the engine's own model dirs (EngineID) under Documents/models (EngineManager.modelsDir).
    static let shared: ModelStore = {
        let store = ModelStore(items: ModelManifest.items(chatDir: EngineID.chat.dir, decisionDir: EngineID.decision.dir),
                               modelsDir: EngineManager.modelsDir)
        let args = ProcessInfo.processInfo.arguments
        store.wantVision = args.contains("-modelsVision")
        // Shown on a launch with a required model missing; `-modelSetup` shows it anyway, `-noModelSetup` never.
        store.isPresented = !args.contains("-noModelSetup") && (store.needsSetup || args.contains("-modelSetup"))
        return store
    }()

    /// The web's status line, bend://models, bendOS models.open, or the camera without its tower.
    func present(vision: Bool = false) {
        if vision { wantVision = true }
        refresh()
        isPresented = true
    }
}

/// IOSFETCH: the first-run setup card. The web app's language (docs/WEAVE.md): neutral greys on oklch(19%), one hairline
/// card on the surface, the status pill with its dot (the only hue), pills for buttons. Every state has a way on:
/// Download → progress (bytes, speed, time left) with Pause → done (Open Bend); an error says what happened and offers
/// Retry; "Not now" opens the app on demo replies, and the card comes back from the web's status line.
struct ModelSetupView: View {
    @Bindable var store: ModelStore
    @AppStorage(EngineID.modeKey) private var oneModel = true
    var onClose: () -> Void

    var body: some View {
        ZStack {
            Color.bendBackground.ignoresSafeArea()
            ScrollView {
                VStack(spacing: 24) {
                    Spacer(minLength: 56)
                    VStack(spacing: 8) {
                        Text("Eureka")
                            .font(.system(size: 34, weight: .medium))
                            .tracking(-1.2)
                            .foregroundStyle(Palette.foreground)
                        Text("Talk it through. When it’s a choice, Eureka ranks it.")
                            .font(.system(size: 15))
                            .foregroundStyle(Palette.secondary)
                            .multilineTextAlignment(.center)
                    }
                    card
                    Spacer(minLength: 24)
                }
                .padding(.horizontal, 16)
                .frame(maxWidth: 520)
                .frame(maxWidth: .infinity)
            }
            .scrollBounceBehavior(.basedOnSize)
        }
        .preferredColorScheme(.dark)
        .onAppear {
            store.refresh()
            if ProcessInfo.processInfo.arguments.contains("-modelsAutoDownload"), store.phase == .idle, !store.wanted.isEmpty { store.start() }
        }
    }

    // MARK: the card

    private var card: some View {
        VStack(alignment: .leading, spacing: 14) {
            VStack(alignment: .leading, spacing: 6) {
                Text(title)
                    .font(.system(size: 19, weight: .semibold))
                    .tracking(-0.3)
                    .foregroundStyle(Palette.foreground)
                    .accessibilityIdentifier("models.title")
                Text(subtitle)
                    .font(.system(size: 13))
                    .foregroundStyle(Palette.muted)
                    .fixedSize(horizontal: false, vertical: true)
            }
            StatusPill(state: pillState, text: pillText)
                .accessibilityIdentifier("models.fit")
            modePicker
            VStack(spacing: 0) {
                ForEach(store.items) { item in
                    row(item)
                    if item.id != store.items.last?.id { Hairline() }
                }
            }
            .background(Palette.surfaceMuted.opacity(0.35), in: RoundedRectangle(cornerRadius: 12))
            .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(Palette.borderSubtle))
            if showsProgress { progress }
            if let error { errorBox(error) }
            buttons
        }
        .padding(16)
        .background(Palette.surface, in: RoundedRectangle(cornerRadius: 16))
        .overlay(RoundedRectangle(cornerRadius: 16).strokeBorder(Palette.border))
    }

    private var modePicker: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Model mode")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(Palette.secondary)
            modeChoice(true, "One model — chat & decisions · Kev-4B")
            modeChoice(false, "Two models — best quality")
            if oneModel != EngineID.kevChat {
                Text("Reopen Eureka to apply this model choice.")
                    .font(.system(size: 12))
                    .foregroundStyle(Palette.secondary)
                    .accessibilityIdentifier("models.mode.pending")
            }
        }
    }

    private func modeChoice(_ one: Bool, _ label: String) -> some View {
        Button {
            oneModel = one
        } label: {
            HStack(spacing: 10) {
                Image(systemName: oneModel == one ? "largecircle.fill.circle" : "circle")
                Text(label).multilineTextAlignment(.leading)
                Spacer(minLength: 0)
            }
            .font(.system(size: 14))
            .foregroundStyle(Palette.foreground)
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Palette.surfaceMuted.opacity(0.35), in: RoundedRectangle(cornerRadius: 8))
        }
        .buttonStyle(.plain)
        .disabled(store.isBusy)
        .accessibilityIdentifier(one ? "models.mode.one" : "models.mode.two")
        .accessibilityAddTraits(oneModel == one ? .isSelected : [])
    }

    private var title: String {
        switch store.phase {
        case .done: store.needsSetup ? "Download the models" : "Eureka is ready"
        case .running, .retrying: "Downloading the models"
        case .paused: "Download paused"
        case .failed: "The download stopped"
        case .idle: store.needsSetup ? "Download the models" : "Models on this \(ModelStore.deviceName)"
        }
    }

    private var subtitle: String {
        let wanted = store.wanted
        let size = ModelStore.size(wanted.reduce(0) { $0 + $1.bytes })
        let host = ModelManifest.baseURL.host() ?? "the model host"
        switch store.phase {
        case .done where !store.needsSetup:
            return "Everything runs on this \(ModelStore.deviceName): no account, nothing sent anywhere. "
                + (EngineManager.shared.enabled ? "The chat engine is warming." : "(The engines run on a phone, not in the simulator.)")
        default:
            if wanted.isEmpty { return "All set. Everything runs on this \(ModelStore.deviceName)." }
            return "Eureka answers on this \(ModelStore.deviceName), with no cloud. Once, \(size) from \(host)\(ModelManifest.overridden ? " (debug host)" : ""). Best on Wi-Fi."
        }
    }

    private var fit: ModelStore.Fit { store.fit() }

    private var pillState: StatusPill.State {
        switch store.phase {
        case .running, .retrying: .waiting
        case .failed: .retry
        case .done where !store.needsSetup: .ready
        default: fit.ok ? .ready : .retry
        }
    }

    private var pillText: String {
        switch store.phase {
        case .running: store.currentFile.map { "Getting \(($0 as NSString).lastPathComponent)" } ?? "Starting…"
        case .retrying(_, let attempt): "Connection trouble · trying again (\(attempt) of 3)"
        case .done where !store.needsSetup: "Verified (sha256) · on this \(ModelStore.deviceName)"
        default: fit.line
        }
    }

    private func row(_ item: ModelItem) -> some View {
        let state = store.rows[item.id] ?? .missing
        return HStack(spacing: 12) {
            rowIcon(item, state)
                .frame(width: 22)
            VStack(alignment: .leading, spacing: 2) {
                Text(item.required ? item.title : "\(item.title) — \(ModelStore.size(item.bytes))")
                    .font(.system(size: 15, weight: .medium))
                    .foregroundStyle(Palette.foreground)
                Text(item.required ? item.detail : "\(item.detail) · optional")
                    .font(.system(size: 12))
                    .foregroundStyle(Palette.muted)
            }
            Spacer(minLength: 8)
            rowTrailing(item, state)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 11)
        .accessibilityIdentifier("models.row.\(item.id.rawValue)")
    }

    @ViewBuilder private func rowIcon(_ item: ModelItem, _ state: ModelStore.Row) -> some View {
        switch state {
        case .present: Image(systemName: "checkmark.circle.fill").foregroundStyle(Palette.ready)
        case .failed: Image(systemName: "exclamationmark.circle").foregroundStyle(Palette.offline)
        case .downloading, .checking: ProgressView().controlSize(.small).tint(Palette.secondary)
        case .queued: Image(systemName: "arrow.down.circle").foregroundStyle(Palette.secondary)
        case .missing: Image(systemName: item.id == .vision ? "camera" : "arrow.down.circle").foregroundStyle(Palette.muted)
        }
    }

    @ViewBuilder private func rowTrailing(_ item: ModelItem, _ state: ModelStore.Row) -> some View {
        switch state {
        case .present:
            Text("Ready").font(.system(size: 12)).foregroundStyle(Palette.muted)
        case .downloading(let f):
            Text("\(Int(f * 100))%").font(.system(size: 13, weight: .medium).monospacedDigit()).foregroundStyle(Palette.secondary)
        case .checking(let f):
            Text("checking \(Int(f * 100))%").font(.system(size: 12).monospacedDigit()).foregroundStyle(Palette.muted)
        default:
            if item.required {
                Text(ModelStore.size(item.bytes)).font(.system(size: 13).monospacedDigit()).foregroundStyle(Palette.secondary)
            } else {
                Toggle("", isOn: $store.wantVision)
                    .labelsHidden()
                    .tint(Palette.ready)
                    .disabled(store.isBusy)
                    .accessibilityIdentifier("models.vision.toggle")
            }
        }
    }

    private var showsProgress: Bool {
        switch store.phase {
        case .running, .retrying, .paused: true
        case .failed: store.bytesDone > 0 && store.bytesTotal > 0
        default: false
        }
    }

    private var progress: some View {
        VStack(alignment: .leading, spacing: 6) {
            ProgressView(value: Double(store.bytesDone), total: Double(max(store.bytesTotal, 1)))
                .tint(Palette.foreground)
                .accessibilityIdentifier("models.progress")
            Text(progressLine)
                .font(.system(size: 12).monospacedDigit())
                .foregroundStyle(Palette.muted)
                .accessibilityIdentifier("models.progress.line")
        }
    }

    private var progressLine: String {
        var parts = ["\(ModelStore.size(store.bytesDone)) of \(ModelStore.size(store.bytesTotal))"]
        if store.phase == .running {
            if store.speed > 0 { parts.append("\(ModelStore.size(Int64(store.speed)))/s") }
            if let eta = store.eta { parts.append("\(ModelStore.duration(eta)) left") } else { parts.append("measuring speed…") }
        } else if case .paused = store.phase {
            parts.append("kept · Resume continues from here")
        }
        return parts.joined(separator: " · ")
    }

    private var error: String? {
        switch store.phase {
        case .failed(let m): m
        case .retrying(let m, _): m
        case .paused(let m) where m != "Paused": m
        default: nil
        }
    }

    private func errorBox(_ text: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "exclamationmark.triangle").foregroundStyle(Palette.offline).font(.system(size: 13))
            Text(text)
                .font(.system(size: 13))
                .foregroundStyle(Palette.secondary)
                .fixedSize(horizontal: false, vertical: true)
                .accessibilityIdentifier("models.error")
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Palette.offline.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(Palette.offline.opacity(0.25)))
    }

    // MARK: buttons

    @ViewBuilder private var buttons: some View {
        VStack(spacing: 10) {
            switch store.phase {
            case .running, .retrying:
                PillButton(title: "Pause", primary: false) { store.pause() }
                    .accessibilityIdentifier("models.pause")
            case .paused:
                PillButton(title: "Resume", primary: true) { store.start() }
                    .accessibilityIdentifier("models.resume")
                notNow
            case .failed:
                PillButton(title: "Retry", primary: true) { store.start() }
                    .accessibilityIdentifier("models.retry")
                notNow
            case .done where !store.needsSetup, .idle where store.wanted.isEmpty:
                PillButton(title: "Open Eureka", primary: true, action: onClose)
                    .accessibilityIdentifier("models.open")
            default:
                PillButton(title: "Download · \(ModelStore.size(store.remaining(store.wanted)))", primary: true) { store.start() }
                    .accessibilityIdentifier("models.download")
                notNow
            }
        }
        .padding(.top, 2)
    }

    private var notNow: some View {
        Button(action: onClose) {
            Text(store.needsSetup ? "Not now · use demo replies" : "Not now")
                .font(.system(size: 14))
                .foregroundStyle(Palette.muted)
                .frame(maxWidth: .infinity, minHeight: 32)
        }
        .accessibilityIdentifier("models.notnow")
    }
}

// MARK: the web's tokens (src/index.css, the arc foundation), as sRGB

enum Palette {
    static let surface = Color(white: 25 / 255)          // --surface oklch(21.5% 0 0)
    static let surfaceMuted = Color(white: 36 / 255)     // --surface-muted oklch(26% 0 0)
    static let foreground = Color(white: 242 / 255)      // --foreground oklch(96% 0 0)
    static let secondary = Color(white: 180 / 255)       // --text-secondary oklch(77% 0 0)
    static let muted = Color(white: 140 / 255)           // --text-muted oklch(64% 0 0)
    static let border = Color.white.opacity(0.09)
    static let borderSubtle = Color.white.opacity(0.06)
    static let ready = Color(red: 111 / 255, green: 176 / 255, blue: 125 / 255)    // --status-ready oklch(70% .1 150)
    static let offline = Color(red: 210 / 255, green: 115 / 255, blue: 108 / 255)  // --status-offline oklch(66% .12 25)
}

/// EmptyState's status line: a hairline pill, a 6 pt dot (green ready, red retry, pulsing while waiting).
struct StatusPill: View {
    enum State { case ready, waiting, retry }
    let state: State
    let text: String
    @SwiftUI.State private var dim = false

    var body: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(state == .ready ? Palette.ready : state == .retry ? Palette.offline : Palette.muted)
                .frame(width: 6, height: 6)
                .opacity(state == .waiting && dim ? 0.35 : 1)
                .animation(state == .waiting ? .easeInOut(duration: 0.6).repeatForever() : nil, value: dim)
                .onAppear { dim = state == .waiting }
                .id(state)  // a new dot per state: the pulse never carries the last state's colour
            Text(text)
                .font(.system(size: 12).monospacedDigit())
                .foregroundStyle(Palette.muted)
                .lineLimit(2)
        }
        .padding(.leading, 10).padding(.trailing, 12).padding(.vertical, 5)
        .overlay(Capsule().strokeBorder(Palette.borderSubtle))
    }
}

struct Hairline: View {
    var body: some View { Rectangle().fill(Palette.borderSubtle).frame(height: 1) }
}

struct PillButton: View {
    let title: String
    let primary: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(primary ? Color.bendBackground : Palette.foreground)
                .frame(maxWidth: .infinity, minHeight: 48)
                .background(primary ? Palette.foreground : Palette.surfaceMuted, in: Capsule())
                .overlay(Capsule().strokeBorder(primary ? .clear : Palette.border))
        }
        .buttonStyle(.plain)
    }
}
#endif
