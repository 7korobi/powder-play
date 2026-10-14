import { describe, expect, it } from 'vitest';
import {
	createMpmGate4Scene,
	createMpmGate4StepConfig,
	GATE4_FIXED_STEP_SECONDS,
	GATE4_FRACTURE_BOND_COUNT,
	GATE4_FRACTURE_PARTICLE_COUNT,
	GATE4_LIQUID_PARTICLE_AREA,
	GATE4_OIL_PARTICLE_COUNT,
	GATE4_PARTICLE_COUNT,
	GATE4_PEBBLE_COUNT,
	GATE4_PEBBLE_PARTICLE_COUNT,
	GATE4_SCENE_SEED,
	GATE4_WATER_PARTICLE_COUNT
} from '../scenes/gate4.js';
import { MpmCpuReferenceSimulation } from './cpu-simulation.js';

describe('MpmCpuReferenceSimulation', () => {
	it('exposes the shared Gate 4 workload and liquid particle area', () => {
		const scene = createMpmGate4Scene(GATE4_SCENE_SEED);
		const simulation = new MpmCpuReferenceSimulation(scene);

		expect(simulation.particleCount).toBe(GATE4_PARTICLE_COUNT);
		expect(simulation.waterParticleCount).toBe(GATE4_WATER_PARTICLE_COUNT);
		expect(simulation.oilParticleCount).toBe(GATE4_OIL_PARTICLE_COUNT);
		expect(simulation.pebbleNodeCount).toBe(GATE4_PEBBLE_PARTICLE_COUNT);
		expect(simulation.triangleBodyCount).toBe(GATE4_PEBBLE_COUNT);
		expect(simulation.fractureNodeCount).toBe(GATE4_FRACTURE_PARTICLE_COUNT);
		expect(simulation.fractureBondCount).toBe(GATE4_FRACTURE_BOND_COUNT);
		expect(
			simulation.particles
				.filter((particle) => particle.material?.model === 'water' || particle.material?.model === 'oil')
				.every((particle) => particle.volume === GATE4_LIQUID_PARTICLE_AREA)
		).toBe(true);
		const snapshot = simulation.createRenderSnapshot();
		const renderCopy = new MpmCpuReferenceSimulation(scene);
		renderCopy.applyRenderSnapshot(snapshot);
		expect(renderCopy.particles.map(({ position }) => position)).toEqual(
			simulation.particles.map(({ position }) => position)
		);

		simulation.dispose();
		renderCopy.dispose();
		expect(() => simulation.particles).toThrow(/disposed/);
	});

	it('updates its particle and fracture state through the shared MPM stepper', () => {
		const scene = createMpmGate4Scene(GATE4_SCENE_SEED);
		const simulation = new MpmCpuReferenceSimulation(scene);
		const result = simulation.step(
			createMpmGate4StepConfig(
				{ x: 0, y: 0 },
				GATE4_FIXED_STEP_SECONDS / 8,
				scene.fractureBonds
			)
		);

		expect(simulation.particles).toBe(result.particles);
		expect(simulation.fractureBonds).toBe(result.fractureBonds);
		expect(simulation.particles).toHaveLength(GATE4_PARTICLE_COUNT);
		expect(simulation.activeFractureBondCount).toBe(GATE4_FRACTURE_BOND_COUNT);
	});
});
