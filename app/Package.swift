// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "ShotTracer",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    targets: [
        // Detection, tracking, smoothing, and rendering logic. No UI dependency —
        // consumed by both the CLI harness (macOS) and the future iOS app (ADR 0006).
        .target(name: "TracingCore"),

        // Phase 0/1 offline harness: clip in, annotated video + JSON sidecar out.
        .executableTarget(
            name: "tracer-harness",
            dependencies: ["TracingCore"],
            path: "Tools/tracer-harness"
        ),
    ]
)
