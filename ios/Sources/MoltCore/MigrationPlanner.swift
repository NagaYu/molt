import Dispatch
import Foundation

/// The model-side operations a migration is made of.  MoltMLX implements this
/// over MLX-Swift; the self-test implements it over a simulated allocator.
public protocol MigrationOps: AnyObject {
    var sourceWeightBytes: UInt64 { get }
    var mapperBytes: UInt64 { get }
    /// Keep the source KV cache (as arrays) and the RoPE parameters; nothing else
    /// of the source model is needed afterwards.
    func snapshotSourceKV() throws
    /// Drop every reference to the source weights.
    func dropSource()
    /// Return freed buffers to the OS (MLX: synchronize, then clear the cache).
    func returnFreedMemory()
    func loadMapper() async throws
    /// Map the snapshot to the destination's geometry; drops the source KV.
    func mapKV() async throws
    func dropMapper()
    func loadDestination() async throws
    /// Native sink/tail recompute and feeding of the already-emitted token.
    func anchorAndResume() async throws
}

public enum Schedule: String, Sendable, CaseIterable {
    /// Destination (+mapper) loaded while the source is resident — the
    /// prefill-handoff shape.  Peak ≈ source + destination.
    case colocated
    /// Drop + return + **wait until phys_footprint has actually fallen** before
    /// loading anything.  Peak ≈ max(source, kv + mapper, destination).
    case ordered
}

public struct PhaseRecord: Sendable {
    public let name: String
    public let startNs: UInt64
    public let endNs: UInt64
    public let footprintAfter: UInt64
}

public struct MigrationTrace: Sendable {
    public var phases: [PhaseRecord] = []
    public var releaseWaitsMs: [Double] = []
    public var releaseReached: [Bool] = []
    public var peakFootprint: UInt64 = 0
    public var seconds: Double { phases.isEmpty ? 0 :
        Double(phases.last!.endNs - phases.first!.startNs) / 1e9 }
}

public enum MigrationError: Error { case releaseTimedOut(String) }

/// Anything that can report the footprint (the real probe, or a simulation).
public protocol FootprintReading {
    func physFootprint() -> UInt64
    func peakFootprint() -> UInt64
}

extension MemoryProbe: FootprintReading {}

/// Runs a migration in the memory order of the chosen schedule.  Mirrors
/// `mlx/molt_mlx/migrate.py` phase for phase (same phase names), so device
/// traces and Mac traces line up.
public struct MigrationPlanner {
    public var probe: any FootprintReading
    public var schedule: Schedule
    public var releaseTimeout: Double = 3.0
    public var pollInterval: Double = 0.002
    /// Optional per-phase hook (os_signpost, phase file for an external sampler).
    public var onPhase: ((String) -> Void)?
    /// `true`: abort when memory fails to come back (never load on top of it).
    public var strict: Bool = true

    public init(probe: any FootprintReading, schedule: Schedule) {
        self.probe = probe
        self.schedule = schedule
    }

    private func awaitFootprint(below target: UInt64, ops: MigrationOps,
                                trace: inout MigrationTrace, what: String) async throws {
        let t0 = DispatchTime.now().uptimeNanoseconds
        ops.returnFreedMemory()
        var fp = probe.physFootprint()
        while fp > target,
              Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9 < releaseTimeout {
            if pollInterval > 0 { try await Task.sleep(nanoseconds: UInt64(pollInterval * 1e9)) }
            ops.returnFreedMemory()
            fp = probe.physFootprint()
        }
        trace.releaseWaitsMs.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6)
        trace.releaseReached.append(fp <= target)
        if fp > target && strict { throw MigrationError.releaseTimedOut(what) }
    }

    public func run(_ ops: MigrationOps) async throws -> MigrationTrace {
        var trace = MigrationTrace()
        var current = ""
        var start: UInt64 = 0
        func begin(_ name: String) {
            onPhase?(name)
            current = name
            start = DispatchTime.now().uptimeNanoseconds
        }
        func end() {
            let fp = probe.physFootprint()
            trace.peakFootprint = max(trace.peakFootprint, fp, probe.peakFootprint())
            trace.phases.append(PhaseRecord(name: current, startNs: start,
                                            endNs: DispatchTime.now().uptimeNanoseconds,
                                            footprintAfter: fp))
        }
        begin("snapshot"); try ops.snapshotSourceKV(); end()
        switch schedule {
        case .colocated:
            begin("load_dst"); try await ops.loadDestination(); end()
            begin("load_mapper"); try await ops.loadMapper(); end()
            begin("map"); try await ops.mapKV(); end()
            begin("transplant"); try await ops.anchorAndResume(); end()
            begin("release_src"); ops.dropSource(); ops.dropMapper(); ops.returnFreedMemory(); end()
        case .ordered:
            begin("release_src")
            let fp0 = probe.physFootprint()
            ops.dropSource()
            try await awaitFootprint(below: fp0 - min(fp0, UInt64(Double(ops.sourceWeightBytes) * 0.9)),
                                     ops: ops, trace: &trace, what: "source")
            end()
            begin("load_mapper"); try await ops.loadMapper(); end()
            begin("map")
            let fp1 = probe.physFootprint()
            let mb = ops.mapperBytes
            try await ops.mapKV()
            ops.dropMapper()
            if mb > 0 {
                let ref = min(fp1, probe.physFootprint())
                try await awaitFootprint(below: ref - min(ref, UInt64(Double(mb) * 0.9)),
                                         ops: ops, trace: &trace, what: "mapper")
            }
            end()
            begin("load_dst"); try await ops.loadDestination(); end()
            begin("transplant"); try await ops.anchorAndResume(); end()
        }
        onPhase?("resumed")
        trace.peakFootprint = max(trace.peakFootprint, probe.peakFootprint())
        return trace
    }
}
