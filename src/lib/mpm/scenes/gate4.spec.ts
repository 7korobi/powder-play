import { describe, expect, it } from 'vitest';
import {
	createMpmCpuParitySample,
	createMpmGate4Scene,
	GATE4_CLONE_OFFSET,
	GATE4_GRID,
	GATE4_LIQUID_INITIAL_VELOCITY_COMPONENT_MAX,
	GATE4_LIQUID_BULK_MODULUS_SCALE,
	GATE4_LIQUID_PARTICLE_AREA,
	GATE4_LIQUID_PARTICLE_AREA_SCALE,
	GATE4_LIQUID_PARTICLE_SPACING,
	GATE4_LIQUID_START_X,
	GATE4_OIL_PARTICLE_COUNT,
	GATE4_OIL_START_Y,
	GATE4_OIL_PARTICLE_COLUMNS,
	GATE4_OIL_PARTICLE_ROWS,
	GATE4_PARTICLE_AREA,
	GATE4_PARTICLE_CAPACITY,
	GATE4_PEBBLE_EDGE_LENGTH,
	GATE4_PEBBLE_INITIAL_VELOCITY_COMPONENT_MAX,
	GATE4_PEBBLE_PARTICLE_COUNT,
	GATE4_PEBBLE_PARTICLES_PER_OBJECT,
	GATE4_PEBBLE_COUNT,
	GATE4_PEBBLE_COLUMNS,
	GATE4_PEBBLE_ROWS,
	GATE4_PEBBLE_SPACING,
	GATE4_FRACTURE_BOND_COUNT,
	GATE4_FRACTURE_COLUMNS,
	GATE4_FRACTURE_FAILURE_STRAIN,
	GATE4_FRACTURE_INITIAL_VELOCITY_COMPONENT_MAX,
	GATE4_FRACTURE_OBJECT_ID,
	GATE4_FRACTURE_PARTICLE_COUNT,
	GATE4_FRACTURE_ROWS,
	GATE4_FRACTURE_SPACING,
	GATE4_PARTICLE_COUNT,
	GATE4_SOLID_PARTICLE_MASS,
	GATE4_WATER_PARTICLE_COLUMNS,
	GATE4_WATER_PARTICLE_COUNT,
	GATE4_WATER_START_Y
} from './gate4.js';
import { stepMpm } from '../reference/solver.js';

describe('Gate 4 mixed-material sample scene', () => {
	const sceneSeed = 0x4d504d34;
	const scene = createMpmGate4Scene(sceneSeed);
	const LIQUID_PARTICLE_COUNT = GATE4_WATER_PARTICLE_COUNT + GATE4_OIL_PARTICLE_COUNT;
	const PEBBLE_START = LIQUID_PARTICLE_COUNT;
	const PEBBLE_END = PEBBLE_START + GATE4_PEBBLE_PARTICLE_COUNT;

	it('creates quarter-count liquid and solid bodies with one connected fracture solid', () => {
		expect(scene.particles).toHaveLength(3_674);
		expect(scene.particles).toHaveLength(14_696 / 4);
		expect(scene.particles).toHaveLength(GATE4_PARTICLE_COUNT);
		expect(GATE4_PARTICLE_CAPACITY).toBe(1_000_000);
		expect(GATE4_GRID).toEqual({ cellsX: 220, cellsY: 150, cellSize: 10 });
		expect(GATE4_PEBBLE_ROWS).toBe(
			Math.ceil(GATE4_PEBBLE_COUNT / GATE4_PEBBLE_COLUMNS)
		);
		expect(GATE4_PEBBLE_COLUMNS * GATE4_PEBBLE_ROWS).toBeGreaterThanOrEqual(
			GATE4_PEBBLE_COUNT
		);
		expect(GATE4_PEBBLE_PARTICLE_COUNT).toBe(
			GATE4_PEBBLE_COUNT * GATE4_PEBBLE_PARTICLES_PER_OBJECT
		);
		expect(scene.parityProbeIndices).toHaveLength(3);
		expect(
			scene.particles
				.slice(0, GATE4_WATER_PARTICLE_COUNT)
				.every((particle) => particle.objectId === 1)
		).toBe(true);
		expect(
			scene.particles
				.slice(GATE4_WATER_PARTICLE_COUNT, LIQUID_PARTICLE_COUNT)
				.every((particle) => particle.objectId === 2)
		).toBe(true);
		const pebbleParticles = scene.particles.slice(PEBBLE_START, PEBBLE_END);
		expect(pebbleParticles).toHaveLength(GATE4_PEBBLE_PARTICLE_COUNT);
		const fractureParticles = scene.particles.slice(PEBBLE_END);
		expect(fractureParticles).toHaveLength(GATE4_FRACTURE_PARTICLE_COUNT);
		expect(fractureParticles.every((particle) => particle.objectId === GATE4_FRACTURE_OBJECT_ID)).toBe(true);
		expect(fractureParticles.every((particle) => particle.bondedTriangleVertex === undefined)).toBe(true);
		expect(GATE4_FRACTURE_BOND_COUNT).toBe(
			GATE4_FRACTURE_ROWS * (GATE4_FRACTURE_COLUMNS - 1) +
				(GATE4_FRACTURE_ROWS - 1) * GATE4_FRACTURE_COLUMNS +
				(GATE4_FRACTURE_ROWS - 1) * (GATE4_FRACTURE_COLUMNS - 1)
		);
		expect(scene.fractureBonds).toHaveLength(GATE4_FRACTURE_BOND_COUNT);
		expect(scene.fractureBonds.every((bond) => bond.failureStrain === GATE4_FRACTURE_FAILURE_STRAIN)).toBe(true);
		expect(scene.fractureBonds.every((bond) =>
			bond.firstParticle >= PEBBLE_END &&
			bond.secondParticle < GATE4_PARTICLE_COUNT
		)).toBe(true);
		expect(scene.fractureBonds.some((bond) =>
			Math.abs(bond.restLength - GATE4_FRACTURE_SPACING * Math.SQRT2) < 1e-12
		)).toBe(true);

		const particlesPerObject = new Map<number, number>();
		for (const particle of pebbleParticles) {
			const objectId = particle.objectId;
			if (objectId === undefined) {
				throw new Error('Every pebble particle must belong to an object.');
			}
			particlesPerObject.set(
				objectId,
				(particlesPerObject.get(objectId) ?? 0) + 1
			);
		}
		expect([...particlesPerObject.keys()]).toEqual(
			Array.from({ length: GATE4_PEBBLE_COUNT }, (_, index) => index + 3)
		);
		expect([...particlesPerObject.values()].every((count) => count === 3)).toBe(true);

		const materialCounts = scene.particles.reduce(
			(counts, particle) => {
				const model = particle.material?.model;
				if (model === 'water' || model === 'oil') {
					counts[model] += 1;
				} else if (model === 'neo-hookean') {
					counts[
						particle.objectId === GATE4_FRACTURE_OBJECT_ID
							? 'fractureSolid'
							: 'elasticPebble'
					] += 1;
				}
				return counts;
			},
			{ water: 0, oil: 0, elasticPebble: 0, fractureSolid: 0 }
		);
		expect(materialCounts).toEqual({
			water: GATE4_WATER_PARTICLE_COUNT,
			oil: GATE4_OIL_PARTICLE_COUNT,
			elasticPebble: GATE4_PEBBLE_PARTICLE_COUNT,
			fractureSolid: GATE4_FRACTURE_PARTICLE_COUNT
		});
		expect(
			scene.particles
				.slice(0, LIQUID_PARTICLE_COUNT)
				.every((particle) => particle.volume === GATE4_LIQUID_PARTICLE_AREA)
		).toBe(true);
		expect(
			scene.particles
				.slice(LIQUID_PARTICLE_COUNT)
				.every((particle) => particle.volume === GATE4_PARTICLE_AREA)
		).toBe(true);
		expect(GATE4_PARTICLE_AREA).toBe(4 * 25);
		expect(GATE4_LIQUID_PARTICLE_AREA_SCALE).toBe(4);
		expect(GATE4_LIQUID_PARTICLE_AREA).toBe(
			GATE4_PARTICLE_AREA * GATE4_LIQUID_PARTICLE_AREA_SCALE
		);
		expect(GATE4_LIQUID_BULK_MODULUS_SCALE).toBe(4);
		expect(scene.particles[0].material?.lambda).toBe(160_000);
		expect(
			scene.particles[GATE4_WATER_PARTICLE_COUNT].material?.lambda
		).toBe(120_000);
		expect(
			GATE4_WATER_PARTICLE_COUNT * GATE4_LIQUID_PARTICLE_AREA * 0.02
		).toBe(
			4 * GATE4_WATER_PARTICLE_COUNT * GATE4_PARTICLE_AREA * 0.02
		);
		expect(
			GATE4_OIL_PARTICLE_COUNT * GATE4_LIQUID_PARTICLE_AREA * 0.016
		).toBe(
			4 * GATE4_OIL_PARTICLE_COUNT * GATE4_PARTICLE_AREA * 0.016
		);
		expect(
			GATE4_WATER_PARTICLE_COUNT * GATE4_LIQUID_PARTICLE_AREA * 0.02
		).toBe(16_000);
		expect(
			GATE4_OIL_PARTICLE_COUNT * GATE4_LIQUID_PARTICLE_AREA * 0.016
		).toBe(9_600);
		expect(GATE4_PEBBLE_PARTICLE_COUNT * GATE4_SOLID_PARTICLE_MASS).toBe(600);
		expect(GATE4_FRACTURE_PARTICLE_COUNT * GATE4_SOLID_PARTICLE_MASS).toBe(96);
	});

	it('spaces the initial particles farther apart without changing the particle count', () => {
		expect(scene.particles).toHaveLength(GATE4_PARTICLE_COUNT);
		expect(GATE4_LIQUID_PARTICLE_SPACING).toBe(18);
		expect(scene.particles[1].position.x - scene.particles[0].position.x).toBe(
			GATE4_LIQUID_PARTICLE_SPACING
		);
		expect(
			scene.particles[GATE4_WATER_PARTICLE_COLUMNS].position.y -
				scene.particles[0].position.y
		).toBe(GATE4_LIQUID_PARTICLE_SPACING);
	});

	it('keeps initial liquid positions while using fourfold reference area', () => {
		expect(scene.particles[0].position).toEqual({
			x: GATE4_LIQUID_START_X,
			y: GATE4_WATER_START_Y
		});
		expect(scene.particles[GATE4_WATER_PARTICLE_COUNT].position).toEqual({
			x: GATE4_LIQUID_START_X,
			y: GATE4_OIL_START_Y
		});
		expect(scene.particles[0].volume).toBe(GATE4_LIQUID_PARTICLE_AREA);
		expect(scene.particles[GATE4_WATER_PARTICLE_COUNT].volume).toBe(
			GATE4_LIQUID_PARTICLE_AREA
		);
	});

	it('gives every particle bounded random initial velocity and gives pebbles angular momentum', () => {
		expect(GATE4_LIQUID_INITIAL_VELOCITY_COMPONENT_MAX).toBe(64);
		expect(GATE4_PEBBLE_INITIAL_VELOCITY_COMPONENT_MAX).toBe(240);
		expect(GATE4_FRACTURE_INITIAL_VELOCITY_COMPONENT_MAX).toBe(80);
		const repeatedScene = createMpmGate4Scene(sceneSeed);
		expect(
			repeatedScene.particles.map(({ velocity }) => velocity)
		).toEqual(scene.particles.map(({ velocity }) => velocity));

		const liquidParticles = scene.particles.slice(0, LIQUID_PARTICLE_COUNT);
		expect(
			liquidParticles.every(
				({ velocity }) =>
					Math.abs(velocity.x) <= GATE4_LIQUID_INITIAL_VELOCITY_COMPONENT_MAX &&
					Math.abs(velocity.y) <= GATE4_LIQUID_INITIAL_VELOCITY_COMPONENT_MAX
			)
		).toBe(true);
		expect(
			liquidParticles.some(({ velocity }) => velocity.x !== 0 || velocity.y !== 0)
		).toBe(true);
		expect(
			liquidParticles
				.slice(0, GATE4_WATER_PARTICLE_COUNT)
				.some(({ velocity }) => velocity.x !== 0 || velocity.y !== 0)
		).toBe(true);
		expect(
			liquidParticles
				.slice(GATE4_WATER_PARTICLE_COUNT)
				.some(({ velocity }) => velocity.x !== 0 || velocity.y !== 0)
		).toBe(true);

		const pebbleParticles = scene.particles.slice(PEBBLE_START, PEBBLE_END);
		expect(
			pebbleParticles.every(
				({ velocity }) =>
					Math.abs(velocity.x) <= GATE4_PEBBLE_INITIAL_VELOCITY_COMPONENT_MAX &&
					Math.abs(velocity.y) <= GATE4_PEBBLE_INITIAL_VELOCITY_COMPONENT_MAX
			)
		).toBe(true);

		const firstPebble = pebbleParticles.slice(0, GATE4_PEBBLE_PARTICLES_PER_OBJECT);
		const pebbles = Array.from({ length: GATE4_PEBBLE_COUNT }, (_, index) =>
			pebbleParticles.slice(
				index * GATE4_PEBBLE_PARTICLES_PER_OBJECT,
				(index + 1) * GATE4_PEBBLE_PARTICLES_PER_OBJECT
			)
		);
		expect(
			pebbles.every(
				(pebble) =>
					new Set(pebble.map(({ velocity }) => `${velocity.x},${velocity.y}`)).size === 3
			)
		).toBe(true);
		const center = {
			x: firstPebble.reduce((sum, { position }) => sum + position.x, 0) / firstPebble.length,
			y: firstPebble.reduce((sum, { position }) => sum + position.y, 0) / firstPebble.length
		};
		const angularMomentum = firstPebble.reduce(
			(sum, particle) =>
				sum +
				(particle.position.x - center.x) * particle.velocity.y -
				(particle.position.y - center.y) * particle.velocity.x,
			0
		);
		expect(Math.abs(angularMomentum)).toBeGreaterThan(1);
		const fractureParticles = scene.particles.slice(PEBBLE_END);
		expect(
			fractureParticles.every(
				({ velocity }) =>
					Math.abs(velocity.x) <= GATE4_FRACTURE_INITIAL_VELOCITY_COMPONENT_MAX &&
					Math.abs(velocity.y) <= GATE4_FRACTURE_INITIAL_VELOCITY_COMPONENT_MAX
			)
		).toBe(true);
		expect(new Set(fractureParticles.map(({ velocity }) => `${velocity.x},${velocity.y}`)).size).toBe(1);
	});

	it('rejects scene seeds outside the unsigned 32-bit range', () => {
		expect(() => createMpmGate4Scene(-1)).toThrow(RangeError);
		expect(() => createMpmGate4Scene(0x1_0000_0000)).toThrow(RangeError);
	});

	it('uses fourfold liquid particle area and stronger EOS with Neo-Hookean pebbles', () => {
		expect(scene.particles).toHaveLength(GATE4_PARTICLE_COUNT);
		expect(scene.particles[0].material).toMatchObject({
			model: 'water',
			lambda: 160_000,
			density: 0.02
		});
		expect(scene.particles[0].volume).toBe(GATE4_LIQUID_PARTICLE_AREA);
		expect(scene.particles[GATE4_WATER_PARTICLE_COUNT].material).toMatchObject({
			model: 'oil',
			lambda: 120_000,
			density: 0.016
		});
		expect(scene.particles[GATE4_WATER_PARTICLE_COUNT].volume).toBe(
			GATE4_LIQUID_PARTICLE_AREA
		);
		expect(scene.particles[PEBBLE_START].material).toMatchObject({
			model: 'neo-hookean',
			lambda: 20,
			mu: 10
		});
	});

	it('builds 50 undeformed equilateral pebbles with 6%-width edges and spacing', () => {
		expect(GATE4_PEBBLE_COUNT).toBe(200 / 4);
		expect(GATE4_PEBBLE_COUNT * GATE4_PEBBLE_EDGE_LENGTH ** 2).toBe(
			200 * 66 ** 2
		);
		expect(GATE4_PEBBLE_EDGE_LENGTH).toBeCloseTo(
			GATE4_GRID.cellsX * GATE4_GRID.cellSize * 0.06,
			12
		);
		expect(GATE4_PEBBLE_SPACING).toBeGreaterThan(GATE4_PEBBLE_EDGE_LENGTH);

		const pebbleParticles = scene.particles.slice(PEBBLE_START, PEBBLE_END);
		const pebbleCenters = [];
		for (let pebbleIndex = 0; pebbleIndex < GATE4_PEBBLE_COUNT; pebbleIndex += 1) {
			const start = pebbleIndex * GATE4_PEBBLE_PARTICLES_PER_OBJECT;
			const pebble = pebbleParticles.slice(
				start,
				start + GATE4_PEBBLE_PARTICLES_PER_OBJECT
			);
			const edgeLengths = [
				Math.hypot(
					pebble[1].position.x - pebble[0].position.x,
					pebble[1].position.y - pebble[0].position.y
				),
				Math.hypot(
					pebble[2].position.x - pebble[1].position.x,
					pebble[2].position.y - pebble[1].position.y
				),
				Math.hypot(
					pebble[0].position.x - pebble[2].position.x,
					pebble[0].position.y - pebble[2].position.y
				)
			];
			const minX = Math.min(...pebble.map(({ position }) => position.x));
			const maxX = Math.max(...pebble.map(({ position }) => position.x));

			expect(maxX - minX).toBeCloseTo(GATE4_PEBBLE_EDGE_LENGTH, 12);
			for (const edgeLength of edgeLengths) {
				expect(edgeLength).toBeCloseTo(GATE4_PEBBLE_EDGE_LENGTH, 12);
			}
			expect(pebble.map((particle) => particle.objectId)).toEqual([
				pebbleIndex + 3,
				pebbleIndex + 3,
				pebbleIndex + 3
			]);
			expect(
				pebble.every(
					(particle) =>
						particle.C.m00 === 0 &&
						particle.C.m01 === 0 &&
						particle.C.m10 === 0 &&
						particle.C.m11 === 0
				)
			).toBe(true);
			expect(
				pebble.every(
					(particle) =>
						particle.F.m00 === 1 &&
						particle.F.m01 === 0 &&
						particle.F.m10 === 0 &&
						particle.F.m11 === 1
				)
			).toBe(true);
			pebbleCenters.push({
				x: pebble.reduce((sum, { position }) => sum + position.x, 0) / pebble.length,
				y: pebble.reduce((sum, { position }) => sum + position.y, 0) / pebble.length
			});
		}
		expect(pebbleCenters[1].x - pebbleCenters[0].x).toBe(GATE4_PEBBLE_SPACING);
		expect(pebbleCenters[1].y - pebbleCenters[0].y).toBe(0);
		expect(pebbleCenters[GATE4_PEBBLE_COLUMNS].x - pebbleCenters[0].x).toBe(0);
		expect(pebbleCenters[GATE4_PEBBLE_COLUMNS].y - pebbleCenters[0].y).toBe(
			GATE4_PEBBLE_SPACING
		);
		const lastRowStart = (GATE4_PEBBLE_ROWS - 1) * GATE4_PEBBLE_COLUMNS;
		const lastRow = pebbleCenters.slice(lastRowStart);
		expect(lastRow).toHaveLength(
			GATE4_PEBBLE_COUNT - lastRowStart
		);
		expect(lastRow[0].y).toBe(lastRow[lastRow.length - 1].y);
		expect(lastRow[0].x + lastRow[lastRow.length - 1].x).toBeCloseTo(
			pebbleCenters[0].x + pebbleCenters[GATE4_PEBBLE_COLUMNS - 1].x,
			12
		);
		const liquidOverlapCount = pebbleParticles.filter(({ position }) =>
			position.x >= GATE4_LIQUID_START_X &&
			position.x <= GATE4_LIQUID_START_X +
				(GATE4_WATER_PARTICLE_COLUMNS - 1) * GATE4_LIQUID_PARTICLE_SPACING &&
			(
				(position.y >= GATE4_OIL_START_Y &&
					position.y <= GATE4_OIL_START_Y +
						(GATE4_OIL_PARTICLE_ROWS - 1) * GATE4_LIQUID_PARTICLE_SPACING) ||
				(position.y >= GATE4_WATER_START_Y &&
					position.y <= GATE4_WATER_START_Y +
						(GATE4_WATER_PARTICLE_COUNT / GATE4_WATER_PARTICLE_COLUMNS - 1) *
							GATE4_LIQUID_PARTICLE_SPACING)
			)
		);
		expect(liquidOverlapCount.length).toBeGreaterThan(pebbleParticles.length * 0.6);
	});

	it('keeps all starting particles inside the grid and gathers only local parity neighborhoods', () => {
		const sample = createMpmCpuParitySample(
			scene.particles,
			scene.parityProbeIndices,
			GATE4_GRID.cellSize
		);
		const wallCushion = 8 * GATE4_GRID.cellSize;

		expect(
			scene.particles.every(
				({ position }) =>
					position.x >= wallCushion &&
					position.x <= GATE4_GRID.cellsX * GATE4_GRID.cellSize - wallCushion &&
					position.y >= wallCushion &&
					position.y <= GATE4_GRID.cellsY * GATE4_GRID.cellSize - wallCushion &&
					position.x + GATE4_CLONE_OFFSET.x <=
						GATE4_GRID.cellsX * GATE4_GRID.cellSize - wallCushion &&
					position.y + GATE4_CLONE_OFFSET.y <=
						GATE4_GRID.cellsY * GATE4_GRID.cellSize - wallCushion
			)
		).toBe(true);
		expect(sample.particles.length).toBeGreaterThan(110);
		expect(sample.particles.length).toBeLessThan(scene.particles.length);
		expect(sample.sourceIndices).toHaveLength(sample.particles.length);
		expect(sample.probeSampleIndices).toHaveLength(3);
		for (const sourceIndex of sample.sourceIndices) {
			const vertex = scene.particles[sourceIndex].bondedTriangleVertex;
			if (vertex === undefined) continue;
			const bodyStart = sourceIndex - vertex;
			expect(sample.sourceIndices).toContain(bodyStart);
			expect(sample.sourceIndices).toContain(bodyStart + 1);
			expect(sample.sourceIndices).toContain(bodyStart + 2);
		}
	});

	it('produces the same CPU step for each probe as the complete reduced scene', () => {
		const config = {
			dt: 1 / 120,
			grid: GATE4_GRID,
			particleMass: GATE4_SOLID_PARTICLE_MASS,
			particleVolume: GATE4_PARTICLE_AREA,
			bondRestLength: GATE4_PEBBLE_EDGE_LENGTH,
			material: { lambda: 20, mu: 10, model: 'neo-hookean' as const },
			gravity: { x: 0, y: 90 }
		};
		const sample = createMpmCpuParitySample(
			scene.particles,
			scene.parityProbeIndices,
			GATE4_GRID.cellSize
		);
		const fullStep = stepMpm(scene.particles, config).particles;
		const sampleStep = stepMpm(sample.particles, config).particles;

		for (let index = 0; index < scene.parityProbeIndices.length; index += 1) {
			expect(sampleStep[sample.probeSampleIndices[index]]).toEqual(
				fullStep[scene.parityProbeIndices[index]]
			);
		}
	});

	it('preserves parity probes across three CFL microsteps', () => {
		const config = {
			dt: 1 / 720,
			grid: GATE4_GRID,
			particleMass: GATE4_SOLID_PARTICLE_MASS,
			particleVolume: GATE4_PARTICLE_AREA,
			bondRestLength: GATE4_PEBBLE_EDGE_LENGTH,
			material: { lambda: 20, mu: 10, model: 'neo-hookean' as const },
			gravity: { x: 0, y: 90 }
		};
		const sample = createMpmCpuParitySample(
			scene.particles,
			scene.parityProbeIndices,
			GATE4_GRID.cellSize
		);
		let fullParticles = scene.particles;
		let sampleParticles = sample.particles;
		const numericValues = (particle: (typeof fullParticles)[number]) => [
			particle.position.x,
			particle.position.y,
			particle.velocity.x,
			particle.velocity.y,
			particle.C.m00,
			particle.C.m01,
			particle.C.m10,
			particle.C.m11,
			particle.F.m00,
			particle.F.m01,
			particle.F.m10,
			particle.F.m11
		];
		for (let microstep = 0; microstep < 3; microstep += 1) {
			fullParticles = stepMpm(fullParticles, config).particles;
			sampleParticles = stepMpm(sampleParticles, config).particles;
			for (let index = 0; index < scene.parityProbeIndices.length; index += 1) {
				const fullValues = numericValues(fullParticles[scene.parityProbeIndices[index]]);
				const sampleValues = numericValues(sampleParticles[sample.probeSampleIndices[index]]);
				const differences = fullValues.map((value, valueIndex) =>
					Math.abs(value - sampleValues[valueIndex])
				);
				const maximumDifference = Math.max(...differences);
				const maximumDifferenceIndex = differences.indexOf(maximumDifference);
				expect(
					maximumDifference,
					`probe ${index}, microstep ${microstep + 1}, value ${maximumDifferenceIndex}, ` +
						`full=${fullValues[maximumDifferenceIndex]}, sample=${sampleValues[maximumDifferenceIndex]}`
				).toBeLessThanOrEqual(1e-4);
			}
		}
	});
});
