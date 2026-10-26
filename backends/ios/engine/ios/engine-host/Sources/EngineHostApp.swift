import SwiftUI

@main
struct EngineHostApp: App {
  // -headless 1: the probe without UI, then exit (for `simctl spawn`, which runs the
  // app binary in the simulator without SpringBoard/launch services)
  init() {
    if UserDefaults.standard.bool(forKey: "headless") {
      _ = EngineProbe.run()
      exit(0)
    }
  }

  var body: some Scene {
    WindowGroup { ContentView() }
  }
}

struct ContentView: View {
  @State private var lines: [String] = []
  @State private var running = false
  @State private var done = false

  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      Text("Bend engine (4B) on iOS").font(.headline)
      Button(running ? "running…" : done ? "done (relaunch to run again)" : "run the engine probe") {
        probe()
      }
      .buttonStyle(.borderedProminent)
      .disabled(running || done)
      ScrollView {
        Text(lines.joined(separator: "\n"))
          .font(.system(.caption, design: .monospaced))
          .textSelection(.enabled)
          .frame(maxWidth: .infinity, alignment: .leading)
      }
    }
    .padding()
    .onAppear {
      if UserDefaults.standard.bool(forKey: "autorun") { probe() }
    }
  }

  private func probe() {
    running = true
    lines = ["running \(EngineProbe.platform)…"]
    let exitAfter = UserDefaults.standard.bool(forKey: "exit")
    Thread.detachNewThread {
      let out = EngineProbe.run()
      DispatchQueue.main.async {
        lines = out
        running = false
        done = true
        if exitAfter { exit(0) }
      }
    }
  }
}
