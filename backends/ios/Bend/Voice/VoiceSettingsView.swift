#if os(iOS) && BEND_FETCH
import AVFoundation
import SwiftUI

/// VOICESHOP: the Speech settings sheet (docs/VOICES.md): the voices on this phone (bundled, from the shop, or dropped
/// into Files › Bend › voices), each with ▶ (the same demo line in every voice) and a tap to use it: the Speaker swaps
/// at the next sentence, no restart, and the choice is remembered. "More voices" opens the shop.
/// Opened from the Voice surface's voice line (bendOS `voices.open`), bend://voices, or `-voices` / `-voicesShop` at launch.
@MainActor @Observable
final class VoicePicker {
    static let shared = VoicePicker()

    var isPresented = false
    var showShop = false
    private(set) var packs: [VoicePack] = []
    private(set) var choice = VoiceLibrary.shared.choice
    private(set) var speaking: String?
    /// The voice being loaded / rendered right now (its row shows a spinner).
    private(set) var busy: String?
    private(set) var previewing: String?
    private(set) var error: String?

    init() {
        let args = ProcessInfo.processInfo.arguments
        isPresented = args.contains("-voices") || args.contains("-voicesShop")
        showShop = args.contains("-voicesShop")
        refresh()
        NotificationCenter.default.addObserver(forName: Speaker.voiceChanged, object: nil, queue: .main) { _ in
            MainActor.assumeIsolated { VoicePicker.shared.refresh() }
        }
    }

    func present(shop: Bool = false) {
        refresh()
        showShop = shop
        isPresented = true
    }

    func refresh() {
        packs = VoiceLibrary.shared.packs(refresh: true)
        choice = VoiceLibrary.shared.choice
        speaking = Voice.shared.speaker.packID
    }

    func use(_ id: String) {
        guard busy == nil else { return }
        busy = id
        error = nil
        Task {
            do {
                let r = try await Voice.shared.speaker.select(id)
                EngineManager.console("BEND voices picker select \(id) ms=\(r["ms"] ?? "?") voice=\(r["voice"] ?? "?")")
            } catch {
                self.error = "\(VoiceLibrary.shared.pack(id)?.name ?? id) couldn’t load: \(error.localizedDescription)"
            }
            busy = nil
            refresh()
        }
    }

    func preview(_ id: String) {
        if previewing == id { Voice.shared.speaker.stopPreview(); previewing = nil; return }
        previewing = id
        Task {
            do {
                let r = try await Voice.shared.speaker.preview(id)
                let ms = (r["audioMs"] as? Double) ?? 4000
                try? await Task.sleep(for: .milliseconds(Int(max(ms, 1500))))
            } catch {
                self.error = error.localizedDescription
            }
            if previewing == id { previewing = nil }
        }
    }
}

struct VoiceSettingsView: View {
    @Bindable var picker: VoicePicker
    var shop: VoiceShop
    var onClose: () -> Void
    @State private var ambience = Ambience.enabled

    var body: some View {
        NavigationStack {
            ZStack {
                Color.bendBackground.ignoresSafeArea()
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Speech")
                                .font(.system(size: 28, weight: .semibold)).tracking(-0.6)
                                .foregroundStyle(Palette.foreground)
                                .accessibilityIdentifier("voices.title")
                            Text("Every voice runs on this \(ModelStore.deviceName). Tap one to use it; ▶ plays the same line in each.")
                                .font(.system(size: 13)).foregroundStyle(Palette.muted)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        StatusPill(state: Voice.shared.speaker.fallback == nil ? .ready : .retry, text: speakingLine)
                            .accessibilityIdentifier("voices.speaking")
                        list
                        if let error = picker.error { errorBox(error) }
                        moreVoices
                        sound
                        Text("Add your own: copy a Kokoro voice file (.pt as published in hexgrad/Kokoro-82M, or a 510×256 .bin) into Files › On My \(ModelStore.deviceName) › Eureka › voices. It appears here.")
                            .font(.system(size: 12)).foregroundStyle(Palette.muted)
                            .fixedSize(horizontal: false, vertical: true)
                            .accessibilityIdentifier("voices.dropin")
                    }
                    .padding(16)
                    .frame(maxWidth: 560)
                    .frame(maxWidth: .infinity)
                }
            }
            .navigationDestination(isPresented: $picker.showShop) { VoiceShopView(shop: shop, picker: picker) }
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done", action: onClose).foregroundStyle(Palette.foreground).accessibilityIdentifier("voices.done")
                }
            }
            .toolbarBackground(Color.bendBackground, for: .navigationBar)
        }
        .preferredColorScheme(.dark)
        .onAppear { picker.refresh() }
    }

    private var speakingLine: String {
        if let why = Voice.shared.speaker.fallback { return "\(Voice.shared.speaker.name) (Apple) · \(why)" }
        let name = picker.packs.first { $0.id == picker.speaking }?.name ?? picker.packs.first { $0.id == picker.choice }?.name ?? "—"
        return "Speaking: \(name) · on device"
    }

    private var list: some View {
        VStack(spacing: 0) {
            ForEach(picker.packs) { pack in
                row(id: pack.id, title: pack.name, subtitle: subtitle(pack), canPreview: true)
                Hairline()
            }
            row(id: Speaker.appleID, title: "System voice", subtitle: "Apple · \(Voice.shared.speaker.voice.name)", canPreview: false)
        }
        .background(Palette.surface, in: RoundedRectangle(cornerRadius: 14))
        .overlay(RoundedRectangle(cornerRadius: 14).strokeBorder(Palette.border))
    }

    private func subtitle(_ p: VoicePack) -> String {
        var parts = p.id == VoiceLibrary.defaultID ? ["House voice", p.summary] : [p.summary]
        switch p.origin {
        case .bundled: break
        case .shop: parts.append("from the shop")
        case .dropped: parts.append("added from Files")
        }
        if p.engine != "kokoro" { parts.append(VoiceLibrary.shared.adapter(p.engine)?.title ?? p.engine) }
        return parts.joined(separator: " · ")
    }

    private func row(id: String, title: String, subtitle: String, canPreview: Bool) -> some View {
        let chosen = picker.choice == id
        return HStack(spacing: 12) {
            if canPreview {
                Button { picker.preview(id) } label: {
                    Image(systemName: picker.previewing == id ? "stop.fill" : "play.fill")
                        .font(.system(size: 12))
                        .foregroundStyle(Palette.foreground)
                        .frame(width: 32, height: 32)
                        .background(Palette.surfaceMuted, in: Circle())
                        .overlay(Circle().strokeBorder(Palette.border))
                }
                .buttonStyle(.plain)
                .accessibilityIdentifier("voices.preview.\(id)")
            } else {
                Image(systemName: "apple.logo").font(.system(size: 13)).foregroundStyle(Palette.muted).frame(width: 32, height: 32)
            }
            Button { picker.use(id) } label: {
                HStack(spacing: 8) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(title).font(.system(size: 15, weight: .medium)).foregroundStyle(Palette.foreground)
                        Text(subtitle).font(.system(size: 12)).foregroundStyle(Palette.muted).lineLimit(1)
                    }
                    Spacer(minLength: 8)
                    if picker.busy == id {
                        ProgressView().controlSize(.small).tint(Palette.secondary)
                    } else if chosen {
                        Image(systemName: "checkmark").font(.system(size: 14, weight: .semibold)).foregroundStyle(Palette.ready)
                    }
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier("voices.use.\(id)")
            .accessibilityValue(chosen ? "selected" : "")
        }
        .padding(.horizontal, 12).padding(.vertical, 9)
    }

    private func errorBox(_ text: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "exclamationmark.triangle").foregroundStyle(Palette.offline).font(.system(size: 13))
            Text(text).font(.system(size: 13)).foregroundStyle(Palette.secondary).fixedSize(horizontal: false, vertical: true)
                .accessibilityIdentifier("voices.error")
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Palette.offline.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(Palette.offline.opacity(0.25)))
    }

    private var moreVoices: some View {
        Button { picker.showShop = true } label: {
            HStack(spacing: 12) {
                Image(systemName: "sparkles").foregroundStyle(Palette.secondary).frame(width: 32)
                VStack(alignment: .leading, spacing: 2) {
                    Text("More voices").font(.system(size: 15, weight: .medium)).foregroundStyle(Palette.foreground)
                    Text(shopLine).font(.system(size: 12)).foregroundStyle(Palette.muted).lineLimit(1)
                }
                Spacer()
                Image(systemName: "chevron.right").font(.system(size: 13)).foregroundStyle(Palette.muted)
            }
            .padding(.horizontal, 12).padding(.vertical, 12)
            .background(Palette.surface, in: RoundedRectangle(cornerRadius: 14))
            .overlay(RoundedRectangle(cornerRadius: 14).strokeBorder(Palette.border))
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("voices.more")
    }

    /// SOUND-DESIGN: Sound — ambience. On: a faint room tone under a voice session and three tiny cues. Off: silence.
    private var sound: some View {
        Toggle(isOn: $ambience) {
            HStack(spacing: 12) {
                Image(systemName: "waveform").foregroundStyle(Palette.secondary).frame(width: 32)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Sound — ambience").font(.system(size: 15, weight: .medium)).foregroundStyle(Palette.foreground)
                    Text(ambience ? "A faint room tone and soft cues while you talk" : "Off · no background, no cues")
                        .font(.system(size: 12)).foregroundStyle(Palette.muted).lineLimit(1)
                }
            }
        }
        .tint(Palette.ready)
        .onChange(of: ambience) { _, on in Ambience.enabled = on }
        .padding(.horizontal, 12).padding(.vertical, 10)
        .background(Palette.surface, in: RoundedRectangle(cornerRadius: 14))
        .overlay(RoundedRectangle(cornerRadius: 14).strokeBorder(Palette.border))
        .accessibilityIdentifier("voices.ambience")
    }

    private var shopLine: String {
        switch shop.listing {
        case .ready: "\(shop.entries.count) curated voices · about 0.5 MB each"
        case .offline: "The shop is offline · your voices still work"
        default: "Curated voice packs · about 0.5 MB each"
        }
    }
}

/// VOICESHOP: More voices. One card per catalog voice: ▶ Preview (the shop's clip of the demo line, streamed), then
/// Download with progress → Pause / Resume (Range) → sha256-checked → installed → Use this voice. Offline: says so, shows
/// the last list it saw, and the bundled voices are untouched. Every state has a way on (Retry), as IOSFETCH's card.
struct VoiceShopView: View {
    var shop: VoiceShop
    @Bindable var picker: VoicePicker
    @State private var player: AVPlayer?
    @State private var playing: String?

    var body: some View {
        ZStack {
            Color.bendBackground.ignoresSafeArea()
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("More voices")
                            .font(.system(size: 28, weight: .semibold)).tracking(-0.6)
                            .foregroundStyle(Palette.foreground)
                            .accessibilityIdentifier("shop.title")
                        Text("Curated voice packs. Each is about half a megabyte, checked (sha256) before it’s used, and runs on this \(ModelStore.deviceName), offline once it’s here.")
                            .font(.system(size: 13)).foregroundStyle(Palette.muted)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    StatusPill(state: pillState, text: pillText).accessibilityIdentifier("shop.status")
                    Text("House voice: \(VoiceLibrary.shared.pack(shop.houseVoice)?.name ?? "Bella") · on this \(ModelStore.deviceName) already, no download")
                        .font(.system(size: 12)).foregroundStyle(Palette.muted)
                        .accessibilityIdentifier("shop.house")
                    if case .offline(let why, let cached) = shop.listing {
                        offlineBox(why, cached: cached)
                    }
                    ForEach(shop.entries) { card($0) }
                }
                .padding(16)
                .frame(maxWidth: 560)
                .frame(maxWidth: .infinity)
            }
            .refreshable { await shop.load() }
        }
        .navigationBarTitleDisplayMode(.inline)
        .task {
            if shop.listing == .idle || shop.entries.isEmpty { await shop.load() } else { shop.refreshRows() }
            let args = ProcessInfo.processInfo.arguments
            if let i = args.firstIndex(of: "-voicesAutoDownload"), i + 1 < args.count, let e = shop.entries.first(where: { $0.id == args[i + 1] }),
               shop.rows[e.id] != .installed {
                shop.download(e)
            }
        }
        .onDisappear { player?.pause(); playing = nil }
    }

    private var pillState: StatusPill.State {
        switch shop.listing {
        case .ready: .ready
        case .offline: .retry
        default: .waiting
        }
    }

    private var pillText: String {
        switch shop.listing {
        case .idle, .loading: "Reading the catalog from \(shop.host)…"
        case .ready: "From \(shop.host)\(VoiceCatalog.overridden ? " (debug host)" : "") · \(shop.entries.count) voices"
        case .offline(_, let cached): cached ? "Offline · the last list Eureka saw" : "Offline"
        }
    }

    private func offlineBox(_ why: String, cached: Bool) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: "wifi.slash").foregroundStyle(Palette.offline).font(.system(size: 13))
                Text("\(why) The \(picker.packs.count) voices on this \(ModelStore.deviceName) still work.\(cached ? " Downloads start once the shop is back." : "")")
                    .font(.system(size: 13)).foregroundStyle(Palette.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityIdentifier("shop.offline")
            }
            PillButton(title: "Retry", primary: false) { Task { await shop.load() } }
                .accessibilityIdentifier("shop.retry")
        }
        .padding(10)
        .background(Palette.offline.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(Palette.offline.opacity(0.25)))
    }

    private func card(_ e: VoiceCatalog.Entry) -> some View {
        let row = shop.rows[e.id] ?? .available
        return VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(e.name).font(.system(size: 17, weight: .semibold)).foregroundStyle(Palette.foreground)
                    Text(e.summary).font(.system(size: 12)).foregroundStyle(Palette.muted)
                }
                Spacer()
                Text(VoiceLibrary.size(e.bytes)).font(.system(size: 12).monospacedDigit()).foregroundStyle(Palette.muted)
            }
            if let d = e.detail { Text(d).font(.system(size: 13)).foregroundStyle(Palette.secondary) }
            Text("\(e.license) · \(e.credit) · \(VoiceLibrary.shared.adapter(e.engine)?.title ?? e.engine)")
                .font(.system(size: 11)).foregroundStyle(Palette.muted)
            switch row {
            case .downloading(let f):
                VStack(alignment: .leading, spacing: 4) {
                    ProgressView(value: f).tint(Palette.foreground).accessibilityIdentifier("shop.progress.\(e.id)")
                    Text("\(VoiceLibrary.size(shop.bytesDone[e.id] ?? 0)) of \(VoiceLibrary.size(e.bytes)) · \(Int(f * 100))%")
                        .font(.system(size: 12).monospacedDigit()).foregroundStyle(Palette.muted)
                }
            case .paused(let have):
                Text("Paused at \(VoiceLibrary.size(have)) of \(VoiceLibrary.size(e.bytes)) · Resume continues from here")
                    .font(.system(size: 12).monospacedDigit()).foregroundStyle(Palette.muted)
            case .failed(let m):
                Text(m).font(.system(size: 12)).foregroundStyle(Palette.offline).fixedSize(horizontal: false, vertical: true)
                    .accessibilityIdentifier("shop.error.\(e.id)")
            default: EmptyView()
            }
            HStack(spacing: 10) {
                if e.preview_url != nil || row == .installed {
                    PillButton(title: playing == e.id ? "Stop" : "▶ Preview", primary: false) { preview(e, installed: row == .installed) }
                        .accessibilityIdentifier("shop.preview.\(e.id)")
                }
                action(e, row)
            }
        }
        .padding(14)
        .background(Palette.surface, in: RoundedRectangle(cornerRadius: 14))
        .overlay(RoundedRectangle(cornerRadius: 14).strokeBorder(Palette.border))
        .accessibilityElement(children: .contain)  // the card's id must not replace its buttons' ids
        .accessibilityIdentifier("shop.card.\(e.id)")
    }

    @ViewBuilder private func action(_ e: VoiceCatalog.Entry, _ row: VoiceShop.Row) -> some View {
        switch row {
        case .available:
            PillButton(title: "Download", primary: true) { shop.download(e) }.accessibilityIdentifier("shop.download.\(e.id)")
        case .downloading:
            PillButton(title: "Pause", primary: false) { shop.pause(e) }.accessibilityIdentifier("shop.pause.\(e.id)")
        case .paused:
            PillButton(title: "Resume", primary: true) { shop.download(e) }.accessibilityIdentifier("shop.resume.\(e.id)")
        case .failed:
            PillButton(title: "Retry", primary: true) { shop.download(e) }.accessibilityIdentifier("shop.retry.\(e.id)")
        case .installed:
            if picker.choice == e.packID {
                PillButton(title: "✓ In use", primary: false) {}.disabled(true).accessibilityIdentifier("shop.inuse.\(e.id)")
            } else {
                PillButton(title: picker.busy == e.packID ? "Loading…" : "Use this voice", primary: true) { picker.use(e.packID) }
                    .accessibilityIdentifier("shop.use.\(e.id)")
            }
        }
    }

    /// Before download: the shop's clip (streamed). After: the phone renders it itself, as the picker does.
    private func preview(_ e: VoiceCatalog.Entry, installed: Bool) {
        if playing == e.id { player?.pause(); Voice.shared.speaker.stopPreview(); playing = nil; return }
        playing = e.id
        if installed { picker.preview(e.packID); return }
        guard let ref = e.preview_url, let url = VoiceCatalog.resolve(ref, against: VoiceCatalog.url) else { return }
        try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .spokenAudio)
        player = AVPlayer(url: url)
        player?.play()
        VoiceShop.log("preview stream id=\(e.id) url=\(url.absoluteString)")
    }
}
#endif
