import Foundation
import Testing
@testable import ShotTracerCore

@Test func frozenTemplateBankIsSmallCompleteAndGroundTerminated() {
    let templates = ShotProfile.allCases.flatMap {
        PhysicsTemplateBank.templates(profile: $0)
    }
    #expect(templates.count == 18)
    #expect(Set(templates.map(\.specification.identifier)).count == 18)
    #expect(templates.allSatisfy { $0.simulation.terminatedAtGround })
    #expect(templates.allSatisfy { template in
        template.simulation.states.allSatisfy { $0.isFinite }
    })
}

@Test func vacuumSimulationMatchesAnalyticParabola() {
    let initial = FlightState(
        positionMeters: Vector3(x: 0, y: 0, z: 1),
        velocityMetersPerSecond: Vector3(x: 10, y: 0, z: 10),
        spinRadiansPerSecond: .zero
    )
    let configuration = AerodynamicConfiguration(
        enablesDrag: false,
        enablesLift: false,
        spinDecaySeconds: nil
    )
    let result = PhysicsSimulator.simulate(
        initialState: initial,
        timestepSeconds: 0.005,
        maximumSeconds: 5,
        configuration: configuration
    )
    let expectedTime = (10 + sqrt(
        100 + 2 * PhysicsSimulator.gravityMetersPerSecondSquared
    )) / PhysicsSimulator.gravityMetersPerSecondSquared
    #expect(result.terminatedAtGround)
    #expect(abs(result.flightSeconds - expectedTime) < 1e-5)
    #expect(abs(result.states.last!.positionMeters.x - 10 * result.flightSeconds) < 1e-9)
    #expect(result.states.last!.positionMeters.z == 0)
}

@Test func dragOpposesMotionAndMagnusLiftIsPerpendicular() {
    let state = FlightState(
        positionMeters: Vector3(x: 0, y: 0, z: 1),
        velocityMetersPerSecond: Vector3(x: 50, y: 0, z: 0),
        spinRadiansPerSecond: Vector3(x: 0, y: 100, z: 0)
    )
    let dragOnly = PhysicsSimulator.derivative(
        state,
        configuration: AerodynamicConfiguration(enablesDrag: true, enablesLift: false)
    ).velocityMetersPerSecond
    #expect(dragOnly.x < 0)

    let liftDerivative = PhysicsSimulator.derivative(
        state,
        configuration: AerodynamicConfiguration(enablesDrag: false, enablesLift: true)
    ).velocityMetersPerSecond
    let liftOnly = liftDerivative - Vector3(
        x: 0, y: 0, z: -PhysicsSimulator.gravityMetersPerSecondSquared
    )
    let dotProduct = liftOnly.x * state.velocityMetersPerSecond.x
        + liftOnly.y * state.velocityMetersPerSecond.y
        + liftOnly.z * state.velocityMetersPerSecond.z
    #expect(abs(dotProduct) < 1e-12)
}
