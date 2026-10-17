import Foundation
#if canImport(os)
import os
#endif

/// JSON-lines telemetry with fsync per record: a run that is killed mid-way
/// still leaves every record written before the kill (what was shown to the
/// user, when the trigger fired, which phase was running).
public final class Telemetry {
    private let fd: Int32
    public init?(url: URL) {
        fd = open(url.path, O_WRONLY | O_CREAT | O_APPEND, 0o644)
        if fd < 0 { return nil }
    }
    deinit { close(fd) }

    public func write(_ dict: [String: Any]) {
        guard JSONSerialization.isValidJSONObject(dict),
              var d = try? JSONSerialization.data(withJSONObject: dict, options: [.sortedKeys])
        else { return }
        d.append(0x0A)
        d.withUnsafeBytes { _ = Darwin.write(fd, $0.baseAddress, $0.count) }
        fsync(fd)
    }
}

/// os_signpost intervals named like the Python phases, for Instruments
/// (Points of Interest / os_signpost instrument next to Allocations + VM Tracker).
public enum Signposts {
    #if canImport(os)
    public static let log = OSLog(subsystem: "dev.molt.bench", category: .pointsOfInterest)
    public static func event(_ name: StaticString) { os_signpost(.event, log: log, name: name) }
    #endif
}
