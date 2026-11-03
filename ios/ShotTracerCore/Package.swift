// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "ShotTracerCore",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
    ],
    products: [
        .library(name: "ShotTracerCore", targets: ["ShotTracerCore"]),
        .library(name: "ShotTracerUI", targets: ["ShotTracerUI"]),
    ],
    targets: [
        .target(name: "ShotTracerCore"),
        .target(name: "ShotTracerUI", dependencies: ["ShotTracerCore"]),
        .testTarget(name: "ShotTracerCoreTests", dependencies: ["ShotTracerCore"]),
    ]
)
