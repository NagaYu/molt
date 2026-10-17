// swift-tools-version:5.9
// MoltCore: the platform layer of molt — memory probe, pressure signals, the
// memory-ordered migration planner and the Memory Budget Test.  No dependencies,
// so it builds with `swift build` on macOS and is self-tested with
// `swift run MoltCoreSelfTest`.  The MLX-Swift parts (ios/MoltMLX) and the app
// (ios/App) are built with Xcode; see ios/README.md.
import PackageDescription

let package = Package(
    name: "MoltCore",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [.library(name: "MoltCore", targets: ["MoltCore"])],
    targets: [
        .target(name: "MoltCore"),
        .executableTarget(name: "MoltCoreSelfTest", dependencies: ["MoltCore"]),
    ]
)
