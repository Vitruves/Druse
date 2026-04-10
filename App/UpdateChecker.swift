import Foundation

/// Checks GitHub Releases for a newer version of Druse.
@MainActor
@Observable
final class UpdateChecker {
    static let shared = UpdateChecker()

    private(set) var latestVersion: String?
    private(set) var downloadURL: URL?
    private(set) var updateAvailable = false

    private let repo = "Vitruves/Druse"
    private let checkInterval: TimeInterval = 24 * 60 * 60 // once per day

    private var hasChecked = false

    func checkIfNeeded() {
        guard !hasChecked else { return }
        hasChecked = true

        // Respect last-check timestamp
        let lastCheck = UserDefaults.standard.double(forKey: "lastUpdateCheck")
        if lastCheck > 0, Date().timeIntervalSince1970 - lastCheck < checkInterval {
            // Restore cached result
            if let cached = UserDefaults.standard.string(forKey: "cachedLatestVersion") {
                latestVersion = cached
                downloadURL = UserDefaults.standard.url(forKey: "cachedDownloadURL")
                updateAvailable = isNewer(cached, than: currentVersion)
            }
            return
        }

        Task {
            await check()
        }
    }

    func check() async {
        guard let url = URL(string: "https://api.github.com/repos/\(repo)/releases/latest") else { return }

        var request = URLRequest(url: url)
        request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        request.timeoutInterval = 10

        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else { return }

            guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let tagName = json["tag_name"] as? String else { return }

            // Strip leading "v" if present
            let version = tagName.hasPrefix("v") ? String(tagName.dropFirst()) : tagName

            // Find .dmg asset download URL
            var dmgURL: URL?
            if let assets = json["assets"] as? [[String: Any]] {
                for asset in assets {
                    if let name = asset["name"] as? String, name.hasSuffix(".dmg"),
                       let urlStr = asset["browser_download_url"] as? String {
                        dmgURL = URL(string: urlStr)
                        break
                    }
                }
            }

            // Fallback to the release page
            let fallbackURL = URL(string: "https://github.com/\(repo)/releases/tag/\(tagName)")

            latestVersion = version
            downloadURL = dmgURL ?? fallbackURL
            updateAvailable = isNewer(version, than: currentVersion)

            // Cache
            UserDefaults.standard.set(Date().timeIntervalSince1970, forKey: "lastUpdateCheck")
            UserDefaults.standard.set(version, forKey: "cachedLatestVersion")
            if let dl = downloadURL {
                UserDefaults.standard.set(dl, forKey: "cachedDownloadURL")
            }
        } catch {
            // Silently ignore — network may be unavailable
        }
    }

    // MARK: - Version comparison

    var currentVersion: String {
        Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "0.0.0"
    }

    /// Returns true if `remote` is newer than `local`, handling beta suffixes.
    /// e.g. "0.1.22-beta" > "0.1.21-beta", "0.2.0" > "0.1.99-beta"
    private func isNewer(_ remote: String, than local: String) -> Bool {
        let rParts = versionComponents(remote)
        let lParts = versionComponents(local)

        for i in 0..<max(rParts.count, lParts.count) {
            let r = i < rParts.count ? rParts[i] : 0
            let l = i < lParts.count ? lParts[i] : 0
            if r > l { return true }
            if r < l { return false }
        }
        return false
    }

    /// Extracts numeric version components, stripping suffixes like "-beta".
    private func versionComponents(_ version: String) -> [Int] {
        let numeric = version.split(separator: "-").first ?? Substring(version)
        return numeric.split(separator: ".").compactMap { Int($0) }
    }
}
