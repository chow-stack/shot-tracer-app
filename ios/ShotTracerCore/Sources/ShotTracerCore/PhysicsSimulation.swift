import Foundation

struct Vector3: Sendable, Equatable {
    var x: Double
    var y: Double
    var z: Double

    static let zero = Vector3(x: 0, y: 0, z: 0)

    var isFinite: Bool { x.isFinite && y.isFinite && z.isFinite }
    var magnitude: Double { sqrt(x * x + y * y + z * z) }

    static func + (left: Self, right: Self) -> Self {
        Self(x: left.x + right.x, y: left.y + right.y, z: left.z + right.z)
    }

    static func - (left: Self, right: Self) -> Self {
        Self(x: left.x - right.x, y: left.y - right.y, z: left.z - right.z)
    }

    static prefix func - (value: Self) -> Self {
        Self(x: -value.x, y: -value.y, z: -value.z)
    }

    static func * (left: Self, right: Double) -> Self {
        Self(x: left.x * right, y: left.y * right, z: left.z * right)
    }

    static func / (left: Self, right: Double) -> Self {
        Self(x: left.x / right, y: left.y / right, z: left.z / right)
    }

    static func cross(_ left: Self, _ right: Self) -> Self {
        Self(
            x: left.y * right.z - left.z * right.y,
            y: left.z * right.x - left.x * right.z,
            z: left.x * right.y - left.y * right.x
        )
    }
}

struct FlightState: Sendable, Equatable {
    var positionMeters: Vector3
    var velocityMetersPerSecond: Vector3
    var spinRadiansPerSecond: Vector3

    var isFinite: Bool {
        positionMeters.isFinite && velocityMetersPerSecond.isFinite
            && spinRadiansPerSecond.isFinite
    }

    static func + (left: Self, right: Self) -> Self {
        Self(
            positionMeters: left.positionMeters + right.positionMeters,
            velocityMetersPerSecond: left.velocityMetersPerSecond + right.velocityMetersPerSecond,
            spinRadiansPerSecond: left.spinRadiansPerSecond + right.spinRadiansPerSecond
        )
    }

    static func * (left: Self, right: Double) -> Self {
        Self(
            positionMeters: left.positionMeters * right,
            velocityMetersPerSecond: left.velocityMetersPerSecond * right,
            spinRadiansPerSecond: left.spinRadiansPerSecond * right
        )
    }
}

struct AerodynamicConfiguration: Sendable {
    var airDensityKilogramsPerCubicMeter = 1.225
    var windMetersPerSecond = Vector3.zero
    var enablesDrag = true
    var enablesLift = true
    var spinDecaySeconds: Double? = 18
}

struct FlightSimulation: Sendable {
    let timesSeconds: [Double]
    let states: [FlightState]
    let terminatedAtGround: Bool

    var flightSeconds: Double { timesSeconds[timesSeconds.count - 1] }
}

enum PhysicsSimulator {
    static let aerodynamicModelVersion = "golf-flight-envelope-v1"
    static let gravityMetersPerSecondSquared = 9.80665
    static let ballMassKilograms = 0.04593
    static let ballDiameterMeters = 0.04267
    static let ballRadiusMeters = 0.5 * ballDiameterMeters
    static let ballAreaSquareMeters = Double.pi * ballRadiusMeters * ballRadiusMeters

    static func dragCoefficient(speed: Double) -> Double {
        min(0.34, max(0.18, 0.22 + 0.07 * exp(-speed / 28)))
    }

    static func liftCoefficient(speed: Double, spin: Vector3) -> Double {
        guard speed > 1e-9 else { return 0 }
        let spinRatio = ballRadiusMeters * spin.magnitude / speed
        return min(0.34, max(0, 1.45 * spinRatio))
    }

    static func derivative(
        _ state: FlightState,
        configuration: AerodynamicConfiguration
    ) -> FlightState {
        let relativeVelocity = state.velocityMetersPerSecond
            - configuration.windMetersPerSecond
        let speed = relativeVelocity.magnitude
        var acceleration = Vector3(
            x: 0, y: 0, z: -gravityMetersPerSecondSquared
        )
        if speed > 1e-9 {
            let pressureAreaOverMass = 0.5
                * configuration.airDensityKilogramsPerCubicMeter
                * ballAreaSquareMeters / ballMassKilograms
            if configuration.enablesDrag {
                acceleration = acceleration
                    + relativeVelocity
                    * (-pressureAreaOverMass * dragCoefficient(speed: speed) * speed)
            }
            if configuration.enablesLift, state.spinRadiansPerSecond.magnitude > 1e-9 {
                let direction = Vector3.cross(relativeVelocity, state.spinRadiansPerSecond)
                let directionMagnitude = direction.magnitude
                if directionMagnitude > 1e-12 {
                    acceleration = acceleration + direction * (
                        pressureAreaOverMass
                            * liftCoefficient(speed: speed, spin: state.spinRadiansPerSecond)
                            * speed * speed / directionMagnitude
                    )
                }
            }
        }
        let spinDerivative = configuration.spinDecaySeconds.map {
            -state.spinRadiansPerSecond / $0
        } ?? .zero
        return FlightState(
            positionMeters: state.velocityMetersPerSecond,
            velocityMetersPerSecond: acceleration,
            spinRadiansPerSecond: spinDerivative
        )
    }

    static func rk4Step(
        _ state: FlightState,
        timestepSeconds: Double,
        configuration: AerodynamicConfiguration
    ) -> FlightState {
        let k1 = derivative(state, configuration: configuration)
        let k2 = derivative(
            state + k1 * (0.5 * timestepSeconds), configuration: configuration
        )
        let k3 = derivative(
            state + k2 * (0.5 * timestepSeconds), configuration: configuration
        )
        let k4 = derivative(state + k3 * timestepSeconds, configuration: configuration)
        return state + (k1 + k2 * 2 + k3 * 2 + k4) * (timestepSeconds / 6)
    }

    static func simulate(
        initialState: FlightState,
        timestepSeconds: Double = 0.02,
        maximumSeconds: Double = 15,
        configuration: AerodynamicConfiguration = AerodynamicConfiguration()
    ) -> FlightSimulation {
        var states = [initialState]
        var times = [0.0]
        var state = initialState
        let maximumSteps = Int(ceil(maximumSeconds / timestepSeconds))
        for _ in 0..<maximumSteps {
            let advanced = rk4Step(
                state, timestepSeconds: timestepSeconds, configuration: configuration
            )
            if state.positionMeters.z > 0, advanced.positionMeters.z <= 0 {
                let denominator = state.positionMeters.z - advanced.positionMeters.z
                let fraction = denominator > 0 ? state.positionMeters.z / denominator : 1
                var intersection = state + (advanced + state * -1) * fraction
                intersection.positionMeters.z = 0
                states.append(intersection)
                times.append(times[times.count - 1] + fraction * timestepSeconds)
                return FlightSimulation(
                    timesSeconds: times, states: states, terminatedAtGround: true
                )
            }
            states.append(advanced)
            times.append(times[times.count - 1] + timestepSeconds)
            state = advanced
        }
        return FlightSimulation(
            timesSeconds: times, states: states, terminatedAtGround: false
        )
    }
}
