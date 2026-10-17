// MLXMigrationOps.swift — MoltCore.MigrationOps over MLX-Swift + MLXLMCommon.
//
// The memory order lives in MoltCore.MigrationPlanner (self-tested on macOS);
// this file only supplies the operations.  Freeing follows the facts measured
// in mlx/molt_mlx/footprint.py: drop references, synchronize the GPU stream
// (buffers return to MLX's cache only after the GPU is done with them), then
// clear MLX's cache; the planner then polls phys_footprint until it has fallen.
//
// STATUS: NOT compiled on the authoring machine — see KVMapper.swift.
import Foundation
import MLX
import MLXLMCommon
import MoltCore

public final class MLXMigrationOps: MigrationOps {
    var source: (any LanguageModel)?
    var sourceCache: [KVCache]?
    var snapshot: [(MLXArray, MLXArray)]?
    var mapped: [(MLXArray, MLXArray)]?
    var mapper: KVMapper?
    public private(set) var destination: (any LanguageModel)?
    public private(set) var destinationCache: [KVCache]?
    public private(set) var nextLogits: MLXArray?

    let loadDst: () async throws -> any LanguageModel
    let mapperURL: URL?
    let tokens: [Int]          // everything the source cache covers
    let pending: Int           // emitted to the user, not yet fed
    let sink: Int, tail: Int
    public let sourceWeightBytes: UInt64
    public var mapperBytes: UInt64 { UInt64(mapper?.nbytes ?? 0) }

    public init(source: any LanguageModel, cache: [KVCache], tokens: [Int], pending: Int,
                sourceWeightBytes: UInt64, mapperURL: URL?, sink: Int, tail: Int,
                loadDestination: @escaping () async throws -> any LanguageModel) {
        self.source = source; self.sourceCache = cache; self.tokens = tokens
        self.pending = pending; self.sourceWeightBytes = sourceWeightBytes
        self.mapperURL = mapperURL; self.sink = sink; self.tail = tail
        self.loadDst = loadDestination
    }

    public func snapshotSourceKV() throws {
        snapshot = sourceCache!.map { c in let s = c.state; return (s[0], s[1]) }
        sourceCache = nil
    }

    public func dropSource() { source = nil }

    public func returnFreedMemory() {
        Stream.gpu.synchronize()
        MLX.GPU.clearCache()
    }

    public func loadMapper() async throws {
        if let u = mapperURL { mapper = try KVMapper(url: u) }
    }

    public func mapKV() async throws {
        guard let m = mapper, let snap = snapshot else { return }   // re-prefill: no mapper
        let keep = max(0, tokens.count - tail)
        let cut = snap.map { ($0.0[0..., 0..., ..<keep, 0...], $0.1[0..., 0..., ..<keep, 0...]) }
        let out = m.apply(cut, offset: 0)
        eval(out.flatMap { [$0.0, $0.1] })
        mapped = out
        snapshot = nil
    }

    public func dropMapper() { mapper = nil }

    public func loadDestination() async throws {
        let d = try await loadDst()
        eval(d)
        destination = d
    }

    /// Native sink positions (exact: causal), mapped middle, native tail, then
    /// the pending token — `transplant()` in mlx/molt_mlx/migrate.py.
    public func anchorAndResume() async throws {
        guard let d = destination else { return }
        let keep = max(0, tokens.count - tail)
        var caches = d.newCache(parameters: nil)
        if let mp = mapped, keep > 0 {
            var pairs = mp
            let s = min(sink, keep)
            if s > 0 {
                let native = d.newCache(parameters: nil)
                _ = d(LMInput.Text(tokens: MLXArray(tokens[..<s].map { Int32($0) })[.newAxis]),
                      cache: native, state: nil)
                pairs = zip(native, mp).map { n, m in
                    let st = n.state
                    return (concatenated([st[0], m.0[0..., 0..., s..., 0...]], axis: 2),
                            concatenated([st[1], m.1[0..., 0..., s..., 0...]], axis: 2))
                }
            }
            for (c, p) in zip(caches, pairs) { c.state = [p.0, p.1] }
        }
        let feed = (keep > 0 && mapped != nil ? Array(tokens[keep...]) : tokens) + [pending]
        let out = d(LMInput.Text(tokens: MLXArray(feed.map { Int32($0) })[.newAxis]),
                    cache: caches, state: nil)
        nextLogits = out.logits[0..., -1, 0...]
        eval(nextLogits!)
        destinationCache = caches
        mapped = nil
        caches = []
    }
}
