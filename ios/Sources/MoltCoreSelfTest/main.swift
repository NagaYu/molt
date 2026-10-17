// swift run MoltCoreSelfTest — self-test of every MoltCore component.
// Exits non-zero on the first failed check.
import Foundation
import MoltCore

var failures = 0
func check(_ cond: Bool, _ msg: String) {
    if cond { print("  ok   \(msg)") } else { print("  FAIL \(msg)"); failures += 1 }
}

// ---------------------------------------------------------------- MemoryProbe
do {
    let p = MemoryProbe()
    let before = p.physFootprint()
    let n = 128 << 20
    let buf = malloc(n)!
    var off = 0
    while off < n { buf.storeBytes(of: UInt8(1), toByteOffset: off, as: UInt8.self); off += 16384 }
    let during = p.physFootprint()
    check(during > before + (100 << 20), "phys_footprint sees 128 MiB of touched malloc (+\((during - before) >> 20) MiB)")
    free(buf)
    let after = p.physFootprint()
    check(p.peakFootprint() >= during, "ledger peak remembers the 128 MiB after free (now +\((Int(after) - Int(before)) >> 20) MiB)")
    var capped = MemoryProbe(emulatedCapBytes: after + (64 << 20))
    check(MemoryProbe.hasNativeLimit || (capped.availableMemory() <= 64 << 20 && capped.availableMemory() > 32 << 20),
          "availableMemory: native=\(MemoryProbe.hasNativeLimit), emulated headroom \(capped.availableMemory() >> 20) MiB")
    capped.emulatedCapBytes = 1
    check(MemoryProbe.hasNativeLimit || capped.availableMemory() == 0, "available is 0 once over the limit (as documented)")
}

// ------------------------------------------------------------- PressureMonitor
do {
    let probe = MemoryProbe(emulatedCapBytes: MemoryProbe().physFootprint() + (100 << 20))
    var got: [PressureSignal] = []
    let mon = PressureMonitor(probe: probe) { s, _ in got.append(s) }
    mon.start()
    check(!mon.poll(needBytes: 10 << 20), "poll: 100 MiB headroom absorbs a 10 MiB need")
    check(mon.poll(needBytes: 96 << 20, nextDemandBytes: 16 << 20), "poll: fires when need + next demand > headroom")
    if case .headroomBelow? = got.last { check(true, "headroomBelow signal recorded") } else { check(false, "signal") }
    mon.stop()
}

// ----------------------------------------------- MigrationPlanner (simulated)
/// A simulated allocator with MLX-like freeing: dropped buffers are reclaimed
/// only after `returnFreedMemory()` *and* `reclaimPolls` further footprint reads.
final class SimHeap: FootprintReading {
    var live: UInt64 = 50 << 20
    var pending: [(UInt64, Int)] = []
    var returned = false
    var peak: UInt64 = 0
    let reclaimPolls: Int
    init(reclaimPolls: Int) { self.reclaimPolls = reclaimPolls; peak = live }
    func alloc(_ b: UInt64) { live += b; peak = max(peak, live) }
    func drop(_ b: UInt64) { pending.append((b, reclaimPolls)) }
    func physFootprint() -> UInt64 {
        if returned {
            pending = pending.compactMap { b, n in
                if n <= 0 { live -= b; return nil }
                return (b, n - 1)
            }
        }
        return live
    }
    func peakFootprint() -> UInt64 { peak }
}

final class SimOps: MigrationOps {
    let heap: SimHeap
    let src: UInt64 = 1700 << 20, dst: UInt64 = 870 << 20, mapper: UInt64 = 60 << 20, kv: UInt64 = 40 << 20
    var order: [String] = []
    init(_ h: SimHeap) { heap = h; h.alloc(src); h.alloc(kv) }
    var sourceWeightBytes: UInt64 { src }
    var mapperBytes: UInt64 { mapper }
    func snapshotSourceKV() { order.append("snapshot") }
    func dropSource() { heap.drop(src); order.append("drop_src") }
    func returnFreedMemory() { heap.returned = true }
    func loadMapper() { heap.alloc(mapper); order.append("load_mapper") }
    func mapKV() { heap.alloc(kv / 2); heap.drop(kv); order.append("map") }
    func dropMapper() { heap.drop(mapper) }
    func loadDestination() { heap.alloc(dst); order.append("load_dst") }
    func anchorAndResume() { order.append("resume") }
}

do {
    let base: UInt64 = 50 << 20
    let srcResident = base + (1700 << 20) + (40 << 20)
    let cap = srcResident + (64 << 20)            // 64 MiB of headroom over the big model
    for sched in Schedule.allCases {
        let heap = SimHeap(reclaimPolls: 40)
        let ops = SimOps(heap)
        var planner = MigrationPlanner(probe: heap, schedule: sched)
        planner.pollInterval = 0
        let trace = try await planner.run(ops)
        let peakMiB = heap.peak >> 20
        if sched == .ordered {
            check(heap.peak <= cap, "ordered: simulated peak \(peakMiB) MiB ≤ cap \(cap >> 20) MiB with asynchronous reclaim")
            check(ops.order.firstIndex(of: "load_dst")! > ops.order.firstIndex(of: "drop_src")!, "ordered: destination loads only after the source is dropped")
            check(trace.releaseReached.allSatisfy { $0 }, "ordered: every release reached its footprint target (waits \(trace.releaseWaitsMs.map { Int($0) }) ms)")
        } else {
            check(heap.peak > cap, "colocated: simulated peak \(peakMiB) MiB exceeds cap \(cap >> 20) MiB")
        }
    }
    // a release that never comes back must abort, not load on top of it
    let stuck = SimHeap(reclaimPolls: Int.max)
    let ops = SimOps(stuck)
    var planner = MigrationPlanner(probe: stuck, schedule: .ordered)
    planner.pollInterval = 0.0005; planner.releaseTimeout = 0.05
    do { _ = try await planner.run(ops); check(false, "strict planner should throw") }
    catch { check(!ops.order.contains("load_dst"), "strict: a release that never lands aborts before loading") }
} catch { check(false, "planner threw \(error)") }

// ---------------------------------------------------------------- BudgetTest
do {
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent("molt-budget-\(getpid())")
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    let url = dir.appendingPathComponent("budget.jsonl")
    let probe = MemoryProbe(emulatedCapBytes: MemoryProbe().physFootprint() + (96 << 20))
    let bt = BudgetTest(logURL: url, probe: probe, chunkMiB: 16)
    bt.run(maxSteps: 50, stepDelay: 0, stopWhenAvailableBelow: 16 << 20)
    bt.release()
    let s = BudgetTest.summarize(logURL: url, device: "mac", os: "macOS", cleanExit: true)
    check(s != nil && s!.steps >= 4, "budget log persisted \(s?.steps ?? 0) fsynced records")
    if let s { check(abs(s.reportedLimitMiB - s.observedLimitUpperMiB) < 48, "budget: reported limit \(Int(s.reportedLimitMiB)) MiB ≈ observed \(Int(s.observedLimitUpperMiB)) MiB") }
}

// ---------------------------------------------------------------- Telemetry
do {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("molt-tel-\(getpid()).jsonl")
    let t = Telemetry(url: url)!
    t.write(["ev": "trigger", "n": 48])
    t.write(["ev": "token", "i": 49])
    let lines = (try? String(contentsOf: url, encoding: .utf8))?.split(separator: "\n") ?? []
    check(lines.count == 2, "telemetry wrote \(lines.count) fsynced lines")
}

print(failures == 0 ? "MoltCoreSelfTest: ok" : "MoltCoreSelfTest: \(failures) failure(s)")
exit(failures == 0 ? 0 : 1)
