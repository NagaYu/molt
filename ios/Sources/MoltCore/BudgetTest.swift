import Darwin
import Foundation

/// Memory Budget Test: measure this device's per-app limit by allocating until
/// the system terminates the app, persisting a record (with fsync) before every
/// step so the last record survives the kill.  The *next* launch reads the log
/// and reports the limit.
///
/// Each step allocates `chunkBytes` with `malloc` and writes one byte per 16 KiB
/// page — Apple's docs: memory becomes part of the footprint once it is
/// written ("dirty"); clean allocations do not count.
public final class BudgetTest {
    public struct Record: Codable, Sendable {
        public let step: Int
        public let allocatedMiB: Double
        public let footprintMiB: Double
        public let availableMiB: Double
        public let peakMiB: Double
        public let tSeconds: Double
    }

    public struct Summary: Codable, Sendable {
        public let device: String
        public let os: String
        public let steps: Int
        public let lastFootprintMiB: Double
        public let lastAvailableMiB: Double
        /// footprint + available at the start: the limit as the OS reports it
        public let reportedLimitMiB: Double
        /// last persisted footprint + one chunk: the limit as observed by dying
        public let observedLimitUpperMiB: Double
        public let killed: Bool
    }

    let url: URL
    let probe: MemoryProbe
    let chunkBytes: Int
    private var blocks: [UnsafeMutableRawPointer] = []

    public init(logURL: URL, probe: MemoryProbe = MemoryProbe(), chunkMiB: Int = 16) {
        self.url = logURL
        self.probe = probe
        self.chunkBytes = chunkMiB << 20
    }

    private func append(_ r: Record) {
        let line = (try? JSONEncoder().encode(r)).map { $0 + Data([0x0A]) } ?? Data()
        let fd = open(url.path, O_WRONLY | O_CREAT | O_APPEND, 0o644)
        guard fd >= 0 else { return }
        line.withUnsafeBytes { _ = write(fd, $0.baseAddress, $0.count) }
        fsync(fd)
        close(fd)
    }

    /// Allocate until killed (or `maxSteps`).  On a device this never returns
    /// normally: the last fsynced record is the measurement.
    public func run(maxSteps: Int = 100_000, stepDelay: Double = 0.01,
                    stopWhenAvailableBelow: UInt64? = nil) {
        let t0 = Date()
        for step in 0..<maxSteps {
            let avail = probe.availableMemory()
            append(Record(step: step, allocatedMiB: Double(blocks.count * chunkBytes) / 1048576,
                          footprintMiB: Double(probe.physFootprint()) / 1048576,
                          availableMiB: avail == UInt64.max ? -1 : Double(avail) / 1048576,
                          peakMiB: Double(probe.peakFootprint()) / 1048576,
                          tSeconds: Date().timeIntervalSince(t0)))
            if let stop = stopWhenAvailableBelow, avail < stop { return }
            guard let p = malloc(chunkBytes) else { return }
            var off = 0
            while off < chunkBytes { p.storeBytes(of: UInt8(1), toByteOffset: off, as: UInt8.self); off += 16384 }
            blocks.append(p)
            if stepDelay > 0 { Thread.sleep(forTimeInterval: stepDelay) }
        }
    }

    public func release() { blocks.forEach { free($0) }; blocks.removeAll() }

    /// Read a previous run's log.  `killed` is true when the log does not end
    /// with a clean stop (the caller writes a marker file on clean exit).
    public static func summarize(logURL: URL, device: String, os: String,
                                 cleanExit: Bool) -> Summary? {
        guard let text = try? String(contentsOf: logURL, encoding: .utf8) else { return nil }
        let recs = text.split(separator: "\n").compactMap {
            try? JSONDecoder().decode(Record.self, from: Data($0.utf8)) }
        guard let first = recs.first, let last = recs.last else { return nil }
        let chunk = recs.count > 1 ? recs[1].allocatedMiB - recs[0].allocatedMiB : 0
        return Summary(device: device, os: os, steps: recs.count,
                       lastFootprintMiB: last.footprintMiB, lastAvailableMiB: last.availableMiB,
                       reportedLimitMiB: first.availableMiB < 0 ? -1 : first.footprintMiB + first.availableMiB,
                       observedLimitUpperMiB: last.footprintMiB + chunk, killed: !cleanExit)
    }
}
