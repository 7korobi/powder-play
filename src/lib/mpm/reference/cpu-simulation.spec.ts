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
		expect(snapshot.currentLiquidAreas[0]).toBe(GATE4_LIQUID_PARTICLE_AREA);
		const renderCopy = new MpmCpuReferenceSimulation(scene);
		renderCopy.applyRenderSnapshot(snapshot);
		expect(renderCopy.particles.map(({ position }) => position)).toEqual(
			simulation.particles.map(({ position }) => position)
		);

		simulation.dispose();
		renderCopy.dispose();
		expect(() => simulation.particles).toThrow(/disposed/);
	});

	it('renders current liquid area from the latest worker snapshot', () => {
		const scene = createMpmGate4Scene(GATE4_SCENE_SEED);
		const deformedScene = {
			...scene,
			particles: scene.particles.map((particle, index) =>
				index === 0
					? {
							...particle,
							F: { m00: 2, m01: 0, m10: 0, m11: 2 }
						}
					: particle
			)
		};
		const workerSimulation = new MpmCpuReferenceSimulation(deformedScene);
		const renderSimulation = new MpmCpuReferenceSimulation(scene);
		const snapshot = workerSimulation.createRenderSnapshot();
		const particleRadii: number[] = [];
		const context = {
			fillRect() {},
			save() {},
			beginPath() {},
			rect() {},
			clip() {},
			fill() {},
			strokeRect() {},
			restore() {},
			moveTo() {},
			arc(_x: number, _y: number, radius: number) {
				particleRadii.push(radius);
			},
			closePath() {},
			lineTo() {},
			stroke() {}
		} as unknown as CanvasRenderingContext2D;

		renderSimulation.applyRenderSnapshot(snapshot);
		renderSimulation.draw(context, 2_200, 1_500, 'surface');
		expect(particleRadii[0]).toBeCloseTo(Math.sqrt(GATE4_LIQUID_PARTICLE_AREA * 4) * 0.675);

		particleRadii.length = 0;
		renderSimulation.draw(context, 2_200, 1_500, 'particles');
		expect(particleRadii[0]).toBeCloseTo(Math.sqrt(GATE4_LIQUID_PARTICLE_AREA * 4) * 0.24);

		workerSimulation.dispose();
		renderSimulation.dispose();
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
