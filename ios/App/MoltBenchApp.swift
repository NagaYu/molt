// MoltBenchApp — the on-device harness.  One run per launch (process isolation
// on iOS = a fresh launch), selected by launch arguments, results written to
// Documents/ and pulled with `xcrun devicectl device copy from`.
//
//   -mode budget                          Memory Budget Test (allocate until killed)
//   -mode budget-report                   summarise the previous budget run
//   -mode bench -pair 1.5b-0.5b -cond D -prompt open-01 [-ballastMiB 600]
//
// STATUS: NOT compiled on the authoring machine — see ios/README.md.
import SwiftUI
import MoltCore

@main
struct MoltBenchApp: App {
    @State private var status = "idle"
    var body: some Scene {
        WindowGroup {
            VStack(alignment: .leading, spacing: 8) {
                Text("molt bench").font(.headline)
                Text(status).font(.system(.footnote, design: .monospaced))
            }
            .padding()
            .task { status = await Runner.run(arguments: ProcessInfo.processInfo.arguments) }
        }
    }
}

enum Runner {
    static var docs: URL { FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0] }

    static func arg(_ name: String, _ args: [String]) -> String? {
        guard let i = args.firstIndex(of: "-" + name), i + 1 < args.count else { return nil }
        return args[i + 1]
    }

    static func deviceString() -> String {
        var sys = utsname(); uname(&sys)
        let machine = withUnsafePointer(to: &sys.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 256) { String(cString: $0) } }
        return machine
    }

    static func run(arguments args: [String]) async -> String {
        let probe = MemoryProbe()
        switch arg("mode", args) ?? "none" {
        case "budget":
            let url = docs.appendingPathComponent("budget.jsonl")
            try? FileManager.default.removeItem(at: url)
            try? FileManager.default.removeItem(at: docs.appendingPathComponent("budget.clean"))
            let bt = BudgetTest(logURL: url, probe: probe, chunkMiB: Int(arg("chunkMiB", args) ?? "16") ?? 16)
            bt.run()                         // returns only if never killed
            FileManager.default.createFile(atPath: docs.appendingPathComponent("budget.clean").path, contents: nil)
            return "budget run ended without a kill (unexpected)"
        case "budget-report":
            let clean = FileManager.default.fileExists(atPath: docs.appendingPathComponent("budget.clean").path)
            guard let s = BudgetTest.summarize(logURL: docs.appendingPathComponent("budget.jsonl"),
                                               device: deviceString(),
                                               os: UIDevice.current.systemVersion, cleanExit: clean)
            else { return "no budget log" }
            if let d = try? JSONEncoder().encode(s) {
                try? d.write(to: docs.appendingPathComponent("budget-summary.json"))
            }
            return "limit: reported \(Int(s.reportedLimitMiB)) MiB, observed ≤ \(Int(s.observedLimitUpperMiB)) MiB, killed=\(s.killed)"
        case "bench":
            return await BenchRun.run(args: args, probe: probe)
        default:
            return "available \(probe.availableMemory() >> 20) MiB, footprint \(probe.physFootprint() >> 20) MiB"
        }
    }
}
