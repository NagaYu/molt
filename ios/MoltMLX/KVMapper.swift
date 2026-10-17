// KVMapper.swift — load and apply a molt-kv-mapper/1 file with MLX-Swift.
//
// Mirrors mlx/molt_mlx/mappers.py::KVMapper.apply exactly: de-rotate source keys
// with the source RoPE (MLXFast.RoPE with negated scale = exact inverse), concat
// the selected source layers' heads, x·W + b in float32, re-rotate with the
// destination RoPE at the same positions.  The mapping itself is prior work
// (Heo et al., arXiv:2608.03893).
//
// STATUS: written against mlx-swift 0.25 / mlx-swift-examples 2.25 (MLXLMCommon)
// APIs; NOT compiled on the authoring machine (Xcode license not accepted, no
// device).  See ios/README.md.
import Foundation
import MLX

public struct RopeSpec: Codable, Sendable {
    public let dims: Int
    public let base: Float
    public let traditional: Bool
    public let scale: Float

    func apply(_ k: MLXArray, offset: Int) -> MLXArray {
        MLXFast.RoPE(k, dimensions: dims, traditional: traditional, base: base, scale: scale, offset: offset)
    }
    func remove(_ k: MLXArray, offset: Int) -> MLXArray {
        MLXFast.RoPE(k, dimensions: dims, traditional: traditional, base: base, scale: -scale, offset: offset)
    }
}

public struct Geometry: Codable, Sendable {
    public let n_layers: Int
    public let n_kv_heads: Int
    public let head_dim: Int
    public var width: Int { n_kv_heads * head_dim }
}

public final class KVMapper {
    public struct LayerMap { let sources: [Int]; let wk, bk, wv, bv: MLXArray }

    public let src: Geometry, dst: Geometry
    public let srcRope: RopeSpec, dstRope: RopeSpec
    public let policy: String
    public let layers: [LayerMap]
    public let nbytes: Int

    public init(url: URL) throws {
        let (arrays, meta) = try loadArraysAndMetadata(url: url)
        let dec = JSONDecoder()
        func j<T: Decodable>(_ k: String, _ t: T.Type) throws -> T {
            try dec.decode(T.self, from: Data((meta[k] ?? "").utf8))
        }
        src = try j("src", Geometry.self); dst = try j("dst", Geometry.self)
        srcRope = try j("src_rope", RopeSpec.self); dstRope = try j("dst_rope", RopeSpec.self)
        policy = meta["policy"] ?? "?"
        let sources = try j("sources", [[Int]].self)
        var ls: [LayerMap] = []
        var total = 0
        for l in 0..<dst.n_layers {
            let m = LayerMap(sources: sources[l], wk: arrays["layer\(l).Wk"]!, bk: arrays["layer\(l).bk"]!,
                             wv: arrays["layer\(l).Wv"]!, bv: arrays["layer\(l).bv"]!)
            total += m.wk.nbytes + m.bk.nbytes + m.wv.nbytes + m.bv.nbytes
            ls.append(m)
        }
        layers = ls
        nbytes = total
        eval(layers.flatMap { [$0.wk, $0.bk, $0.wv, $0.bv] })
    }

    /// `pairs[l] = (K, V)`, K post-RoPE `[B, n_kv_s, T, d_s]`.
    public func apply(_ pairs: [(MLXArray, MLXArray)], offset: Int = 0,
                      outType: DType? = nil) -> [(MLXArray, MLXArray)] {
        let B = pairs[0].0.dim(0), T = pairs[0].0.dim(2)
        let dt = outType ?? pairs[0].0.dtype
        var fk: [Int: MLXArray] = [:], fv: [Int: MLXArray] = [:]
        func flatK(_ s: Int) -> MLXArray {
            if let x = fk[s] { return x }
            let k = srcRope.remove(pairs[s].0.asType(.float32), offset: offset)
            let x = k.transposed(0, 2, 1, 3).reshaped([B, T, src.width]); fk[s] = x; return x
        }
        func flatV(_ s: Int) -> MLXArray {
            if let x = fv[s] { return x }
            let x = pairs[s].1.asType(.float32).transposed(0, 2, 1, 3).reshaped([B, T, src.width])
            fv[s] = x; return x
        }
        return layers.map { lm in
            let xk = concatenated(lm.sources.map(flatK), axis: -1)
            let xv = concatenated(lm.sources.map(flatV), axis: -1)
            let yk = matmul(xk, lm.wk.asType(.float32)) + lm.bk.asType(.float32)
            let yv = matmul(xv, lm.wv.asType(.float32)) + lm.bv.asType(.float32)
            var k = yk.reshaped([B, T, dst.n_kv_heads, dst.head_dim]).transposed(0, 2, 1, 3)
            let v = yv.reshaped([B, T, dst.n_kv_heads, dst.head_dim]).transposed(0, 2, 1, 3)
            k = dstRope.apply(k, offset: offset)
            return (k.asType(dt), v.asType(dt))
        }
    }
}
