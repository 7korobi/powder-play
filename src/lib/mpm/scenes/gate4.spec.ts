import { describe, expect, it } from 'vitest';
import {
	createMpmCpuParitySample,
	createMpmGate4Scene,
	GATE4_CLONE_OFFSET,
	GATE4_GRID,
	GATE4_PARTICLE_CAPACITY,
	GATE4_PEBBLE_EDGE_LENGTH,
	GATE4_PEBBLE_PARTICLE_COUNT,
	GATE4_PEBBLE_PARTICLES_PER_OBJECT,
	GATE4_PEBBLE_COUNT,
	GATE4_PEBBLE_COLUMNS,
	GATE4_PEBBLE_ROWS,
	GATE4_PEBBLE_SPACING,
	GATE4_PARTICLE_COUNT
} from './gate4.js';
import { stepMpm } from '../reference/solver.js';

describe('Gate 4 mixed-material sample scene', () => {
	const scene = createMpmGate4Scene();

	it('creates two liquid regions and 200 contiguous three-particle elastic pebbles', () => {
		expect(scene.particles).toHaveLength(14_600);
		expect(scene.particles).toHaveLength(GATE4_PARTICLE_COUNT);
		expect(GATE4_PARTICLE_CAPACITY).toBe(1_000_000);
		expect(GATE4_GRID).toEqual({ cellsX: 220, cellsY: 150, cellSize: 10 });
		expect(GATE4_PEBBLE_COLUMNS * GATE4_PEBBLE_ROWS).toBe(GATE4_PEBBLE_COUNT);
		expect(GATE4_PEBBLE_PARTICLE_COUNT).toBe(
			GATE4_PEBBLE_COUNT * GATE4_PEBBLE_PARTICLES_PER_OBJECT
		);
		expect(scene.parityProbeIndices).toHaveLength(3);
		expect(scene.particles.slice(0, 8_000).every((particle) => particle.objectId === 1)).toBe(true);
		expect(
			scene.particles.slice(8_000, 14_000).every((particle) => particle.objectId === 2)
		).toBe(true);
		const pebbleParticles = scene.particles.slice(14_000);
		expect(pebbleParticles).toHaveLength(GATE4_PEBBLE_PARTICLE_COUNT);

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
					counts.elasticPebble += 1;
				}
				return counts;
			},
			{ water: 0, oil: 0, elasticPebble: 0 }
		);
		expect(materialCounts).toEqual({ water: 8_000, oil: 6_000, elasticPebble: 600 });
	});

	it('spaces the initial particles farther apart without changing the particle count', () => {
		expect(scene.particles).toHaveLength(GATE4_PARTICLE_COUNT);
		expect(scene.particles[1].position.x - scene.particles[0].position.x).toBe(9);
		expect(scene.particles[100].position.y - scene.particles[0].position.y).toBe(9);
	});

	it('uses half liquid densities, tenfold liquid stiffness, and Neo-Hookean pebbles', () => {
		expect(scene.particles).toHaveLength(GATE4_PARTICLE_COUNT);
		expect(scene.particles[0].material).toMatchObject({
			model: 'water',
			lambda: 40_000,
			density: 0.02
		});
		expect(scene.particles[8_000].material).toMatchObject({
			model: 'oil',
			lambda: 30_000,
			density: 0.016
		});
		expect(scene.particles[14_000].material).toMatchObject({
			model: 'neo-hookean',
			lambda: 20,
			mu: 10
		});
	});

	it('builds 200 undeformed equilateral pebbles with 0.5%-width edges on a spaced grid', () => {
		expect(GATE4_PEBBLE_EDGE_LENGTH).toBeCloseTo(
			GATE4_GRID.cellsX * GATE4_GRID.cellSize * 0.005,
			12
		);

		const pebbleParticles = scene.particles.slice(14_000);
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
		expect(sample.particles).toHaveLength(83);
		expect(sample.sourceIndices).toHaveLength(83);
		expect(sample.probeSampleIndices).toHaveLength(3);
	});

	it('produces the same CPU step for each probe as the complete 14,600-particle scene', () => {
		const config = {
			dt: 1 / 120,
			grid: GATE4_GRID,
			particleMass: 1,
			particleVolume: 25,
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
});
