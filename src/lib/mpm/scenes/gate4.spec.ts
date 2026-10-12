import { describe, expect, it } from 'vitest';
import { createMpmCpuParitySample, createMpmGate4Scene, GATE4_GRID } from './gate4.js';
import { stepMpm } from '../reference/solver.js';

describe('Gate 4 mixed-material sample scene', () => {
	const scene = createMpmGate4Scene();

	it('creates exactly 20,000 particles in three contiguous material objects', () => {
		expect(scene.particles).toHaveLength(20_000);
		expect(scene.parityProbeIndices).toHaveLength(3);
		expect(scene.particles.slice(0, 8_000).every((particle) => particle.objectId === 1)).toBe(true);
		expect(
			scene.particles.slice(8_000, 14_000).every((particle) => particle.objectId === 2)
		).toBe(true);
		expect(scene.particles.slice(14_000).every((particle) => particle.objectId === 3)).toBe(true);

		const materialCounts = scene.particles.reduce(
			(counts, particle) => {
				const model = particle.material?.model;
				if (model === 'water' || model === 'oil' || model === 'powder') {
					counts[model] += 1;
				}
				return counts;
			},
			{ water: 0, oil: 0, powder: 0 }
		);
		expect(materialCounts).toEqual({ water: 8_000, oil: 6_000, powder: 6_000 });
	});

	it('keeps all starting particles inside the grid and gathers only local parity neighborhoods', () => {
		const sample = createMpmCpuParitySample(
			scene.particles,
			scene.parityProbeIndices,
			GATE4_GRID.cellSize
		);

		expect(
			scene.particles.every(
				({ position }) =>
					position.x >= 0 &&
					position.x <= GATE4_GRID.cellsX * GATE4_GRID.cellSize &&
					position.y >= 0 &&
					position.y <= GATE4_GRID.cellsY * GATE4_GRID.cellSize
			)
		).toBe(true);
		expect(sample.particles).toHaveLength(147);
		expect(sample.sourceIndices).toHaveLength(147);
		expect(sample.probeSampleIndices).toHaveLength(3);
	});

	it('produces the same CPU step for each probe as the complete 20,000-particle scene', () => {
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
