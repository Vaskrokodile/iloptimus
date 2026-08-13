import AppKit
import WebKit

final class AppDelegate: NSObject, NSApplicationDelegate, WKNavigationDelegate {
    private var window: NSWindow!
    private var webView: WKWebView!
    private var server: Process?
    private let localURL = URL(string: "http://127.0.0.1:7860")!

    func applicationDidFinishLaunching(_ notification: Notification) {
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .default()
        webView = WKWebView(frame: .zero, configuration: configuration)
        webView.navigationDelegate = self

        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1440, height: 920),
            styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        window.title = "IL Optimus"
        window.titlebarAppearsTransparent = true
        window.center()
        window.contentView = webView
        window.makeKeyAndOrderFront(nil)

        startServer()
        waitForServer(attempt: 0)
    }

    private func startServer() {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/zsh")
        process.arguments = ["-lc", "if command -v iloptimus >/dev/null; then exec iloptimus serve --no-browser; elif [ -x \"$HOME/.local/bin/iloptimus\" ]; then exec \"$HOME/.local/bin/iloptimus\" serve --no-browser; else exit 127; fi"]
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        do {
            try process.run()
            server = process
        } catch {
            showStartupError("Could not start the local IL Optimus service: \(error.localizedDescription)")
        }
    }

    private func waitForServer(attempt: Int) {
        var request = URLRequest(url: localURL.appendingPathComponent("api/health"))
        request.timeoutInterval = 1
        URLSession.shared.dataTask(with: request) { [weak self] _, response, _ in
            DispatchQueue.main.async {
                guard let self else { return }
                if let http = response as? HTTPURLResponse, (200..<500).contains(http.statusCode) {
                    self.webView.load(URLRequest(url: self.localURL))
                } else if attempt < 80 {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) {
                        self.waitForServer(attempt: attempt + 1)
                    }
                } else {
                    self.showStartupError("The local service did not become ready. Run ‘iloptimus doctor’ in Terminal for details.")
                }
            }
        }.resume()
    }

    private func showStartupError(_ message: String) {
        let escaped = message
            .replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
        webView.loadHTMLString("<style>body{background:#111216;color:#fff;font:16px -apple-system;padding:70px}h1{font-size:32px}p{color:#aaa;line-height:1.6}</style><h1>IL Optimus could not start</h1><p>\(escaped)</p>", baseURL: nil)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    func applicationWillTerminate(_ notification: Notification) {
        if let server, server.isRunning { server.terminate() }
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.regular)
let delegate = AppDelegate()
app.delegate = delegate
app.activate(ignoringOtherApps: true)
app.run()
