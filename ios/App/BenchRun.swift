// BenchRun — one benchmark condition for one prompt, on the device.
// Mirrors bench/worker.py: greedy decode; from token `pressure_at` a foreign
// demand arrives in 64 MiB chunks; before every token the loop polls
// os_proc_available_memory() (primary trigger, see MoltCore.PressureMonitor) and
// migrates once headroom < (this schedule's need + next chunk).  Every token,
// signal and phase is written (fsync) to Documents/bench-*.jsonl, so a run that
// jetsam kills still leaves its record.
//
// STATUS: NOT compiled on the authoring machine — see ios/README.md.
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MoltCore

struct PromptItem: Decodable {
    let id: String, kind: String, text: String
    let answer: String?
    let max_new: Int, pressure_at: Int
}

struct MapperManifest: Decodable {
    struct Anchor: Decodable { let sink: Int; let tail: Int }
    let selected: String
    let molt_anchor: Anchor
    /// written by bench/export_device.py from the destination's safetensors sizes
    let dst_weight_bytes: UInt64
}

enum BenchRun {
    static let tiers = ["3b": "mlx-community/Qwen2.5-3B-Instruct-4bit",
                        "1.5b": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
                        "0.5b": "mlx-community/Qwen2.5-0.5B-Instruct-4bit"]
    static let chunkMiB: UInt64 = 64
    static let marginMiB: UInt64 = 96

    static func load(_ tier: String) async throws -> ModelContext {
        try await LLMModelFactory.shared.load(configuration: ModelConfiguration(id: tiers[tier]!))
    }

    static func argmax(_ logits: MLXArray) -> Int {
        logits.reshaped([-1]).argMax().item(Int.self)
    }

    static func weightBytes(_ m: any LanguageModel) -> UInt64 {
        UInt64(m.parameters().flattened().reduce(0) { $0 + $1.1.nbytes })
    }

    static func run(args: [String], probe: MemoryProbe) async -> String {
        let pair = Runner.arg("pair", args) ?? "1.5b-0.5b"
        let cond = Runner.arg("cond", args) ?? "D"
        let promptID = Runner.arg("prompt", args) ?? "open-01"
        let ballastMiB = UInt64(Runner.arg("ballastMiB", args) ?? "0") ?? 0
        let tiersOf = pair.split(separator: "-").map(String.init)
        let (src, dst) = (tiersOf[0], tiersOf[1])
        guard let tel = Telemetry(url: Runner.docs.appendingPathComponent("bench-\(pair)-\(cond)-\(promptID).jsonl"))
        else { return "cannot open telemetry" }
        let t0 = DispatchTime.now().uptimeNanoseconds
        func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9 }
        tel.write(["ev": "start", "cond": cond, "prompt": promptID, "pair": pair,
                   "device": Runner.deviceString(), "t": now(),
                   "available_mib": Int(probe.availableMemory() >> 20),
                   "footprint_mib": Int(probe.physFootprint() >> 20)])
        let monitor = PressureMonitor(probe: probe) { s, t in
            tel.write(["ev": "signal", "signal": "\(s)", "t_ns": Int(t)])
        }
        monitor.start()
        defer { monitor.stop() }
        MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)   // the value mlx-swift-examples use

        do {
            let bundle = Bundle.main
            let items = try JSONDecoder().decode([PromptItem].self,
                from: Data(contentsOf: bundle.url(forResource: "prompts", withExtension: "json")!))
            guard let item = items.first(where: { $0.id == promptID }) else { return "no prompt \(promptID)" }
            let mdir = Runner.docs.appendingPathComponent("mappers/\(pair)")
            let man = try JSONDecoder().decode(MapperManifest.self,
                from: Data(contentsOf: mdir.appendingPathComponent("manifest.json")))
            let usesMapper = ["C", "D", "C_ordered", "D_colocated"].contains(cond)
            let anchors = cond.hasPrefix("D")
            let schedule: Schedule = (cond == "C" || cond == "D_colocated") ? .colocated : .ordered
            let migrates = !["B", "big", "static_big"].contains(cond)
            let mapperURL = usesMapper ? mdir.appendingPathComponent("\(man.selected).safetensors") : nil
            let mapperMiB = UInt64((try? mapperURL.map { try FileManager.default.attributesOfItem(atPath: $0.path)[.size] as? Int ?? 0 } ?? 0) ?? 0) >> 20

            var ctx: ModelContext? = try await load(cond == "B" ? dst : src)
            let tokenizer = ctx!.tokenizer
            let eos = Set([tokenizer.eosTokenId].compactMap { $0 })
            let promptIDs = try tokenizer.applyChatTemplate(messages: [["role": "user", "content": item.text]])
            var model: (any LanguageModel)? = ctx!.model
            ctx = nil                                     // only the model + tokenizer stay
            tel.write(["ev": "loaded", "t": now(), "footprint_mib": Int(probe.physFootprint() >> 20)])
            let dstWeightMiB = man.dst_weight_bytes >> 20
            var cache = model!.newCache(parameters: nil)
            let first = model!(LMInput.Text(tokens: MLXArray(promptIDs.map { Int32($0) })[.newAxis]),
                               cache: cache, state: nil)
            var pending = argmax(first.logits[0..., -1, 0...])
            var tokens = promptIDs
            var emitted = [pending]
            tel.write(["ev": "token", "i": 0, "t": now()])
            var ballast: [UnsafeMutableRawPointer] = []
            var migrated = false
            let need = (marginMiB + (schedule == .colocated ? dstWeightMiB + mapperMiB : 0)) << 20

            while emitted.count < item.max_new && !eos.contains(pending) {
                let n = emitted.count
                let pressure = n >= item.pressure_at
                if pressure, UInt64(ballast.count) * chunkMiB < ballastMiB, let p = malloc(Int(chunkMiB << 20)) {
                    var off = 0
                    while off < Int(chunkMiB << 20) { p.storeBytes(of: UInt8(1), toByteOffset: off, as: UInt8.self); off += 16384 }
                    ballast.append(p)
                }
                let nextChunk: UInt64 = UInt64(ballast.count) * chunkMiB < ballastMiB ? chunkMiB << 20 : 0
                if pressure && migrates && !migrated && monitor.poll(needBytes: need, nextDemandBytes: nextChunk) {
                    tel.write(["ev": "trigger", "t": now(), "n_emitted": n,
                               "available_mib": Int(probe.availableMemory() >> 20),
                               "footprint_mib": Int(probe.physFootprint() >> 20)])
                    let restart = cond == "A"
                    let ops = MLXMigrationOps(
                        source: model!, cache: cache, tokens: restart ? Array(promptIDs.dropLast()) : tokens,
                        pending: restart ? promptIDs.last! : pending,
                        sourceWeightBytes: weightBytes(model!), mapperURL: mapperURL,
                        sink: anchors ? man.molt_anchor.sink : 0,
                        tail: (cond == "A" || cond == "A2") ? Int.max / 2 : (anchors ? man.molt_anchor.tail : 0),
                        loadDestination: { try await load(dst).model })
                    model = nil
                    cache = []
                    var planner = MigrationPlanner(probe: probe, schedule: schedule)
                    planner.onPhase = { name in tel.write(["ev": "phase", "name": name, "t": now(),
                                                           "footprint_mib": Int(probe.physFootprint() >> 20)]) }
                    let trace = try await planner.run(ops)
                    tel.write(["ev": "migrated", "t": now(), "seconds": trace.seconds,
                               "peak_mib": Int(trace.peakFootprint >> 20),
                               "release_waits_ms": trace.releaseWaitsMs])
                    model = ops.destination
                    cache = ops.destinationCache ?? []
                    if restart {
                        tel.write(["ev": "retracted", "n": emitted.count])
                        tokens = promptIDs
                        emitted = []
                    } else {
                        tokens.append(pending)
                    }
                    pending = argmax(ops.nextLogits!)
                    emitted.append(pending)
                    tel.write(["ev": "token", "i": emitted.count - 1, "t": now()])
                    migrated = true
                    continue
                }
                let out = model!(LMInput.Text(tokens: MLXArray([Int32(pending)])[.newAxis]), cache: cache, state: nil)
                tokens.append(pending)
                pending = argmax(out.logits[0..., -1, 0...])
                emitted.append(pending)
                tel.write(["ev": "token", "i": emitted.count - 1, "t": now()])
            }
            let text = tokenizer.decode(tokens: emitted)
            tel.write(["ev": "done", "t": now(), "text": text, "generated": emitted,
                       "peak_mib": Int(probe.peakFootprint() >> 20)])
            ballast.forEach { free($0) }
            return "done: \(emitted.count) tokens, peak \(probe.peakFootprint() >> 20) MiB"
        } catch {
            tel.write(["ev": "error", "error": "\(error)", "t": now()])
            return "error: \(error)"
        }
    }
}
