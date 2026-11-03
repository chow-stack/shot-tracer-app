import Foundation

public enum ShotProfile: String, Codable, CaseIterable, Sendable {
    case wood
    case iron
    case wedge
}

struct PhysicsTemplateSpec: Sendable {
    let identifier: String
    let profile: ShotProfile
    let shape: String
    let launchSpeedMetersPerSecond: Double
    let launchAngleDegrees: Double
    let backspinRPM: Double
    let sidespinRPM: Double
}

struct PhysicsTemplate: Sendable {
    let specification: PhysicsTemplateSpec
    let simulation: FlightSimulation
}

enum PhysicsTemplateBank {
    static let version = "club-profile-bank-v1"

    private static let variants: [ShotProfile: [(Double, Double, Double)]] = [
        .wood: [(58, 13, 3_000), (66, 11, 2_500)],
        .iron: [(42, 22, 5_600), (49, 18, 4_700)],
        .wedge: [(27, 34, 8_500), (33, 29, 7_500)],
    ]
    private static let shapes: [(String, Double)] = [
        ("draw", -350), ("straight", 0), ("fade", 350),
    ]

    static func specifications(profile: ShotProfile) -> [PhysicsTemplateSpec] {
        (variants[profile] ?? []).enumerated().flatMap { variantIndex, values in
            shapes.map { shape, sidespin in
                PhysicsTemplateSpec(
                    identifier: "\(version):\(profile.rawValue):\(variantIndex + 1):\(shape)",
                    profile: profile,
                    shape: shape,
                    launchSpeedMetersPerSecond: values.0,
                    launchAngleDegrees: values.1,
                    backspinRPM: values.2,
                    sidespinRPM: sidespin
                )
            }
        }
    }

    static func templates(profile: ShotProfile) -> [PhysicsTemplate] {
        specifications(profile: profile).map { specification in
            let angle = specification.launchAngleDegrees * Double.pi / 180
            let rpmToRadiansPerSecond = 2 * Double.pi / 60
            let initialState = FlightState(
                positionMeters: Vector3(x: 0, y: 0, z: 0.01),
                velocityMetersPerSecond: Vector3(
                    x: specification.launchSpeedMetersPerSecond * cos(angle),
                    y: 0,
                    z: specification.launchSpeedMetersPerSecond * sin(angle)
                ),
                spinRadiansPerSecond: Vector3(
                    x: 0,
                    y: specification.backspinRPM * rpmToRadiansPerSecond,
                    z: specification.sidespinRPM * rpmToRadiansPerSecond
                )
            )
            return PhysicsTemplate(
                specification: specification,
                simulation: PhysicsSimulator.simulate(initialState: initialState)
            )
        }
    }
}
