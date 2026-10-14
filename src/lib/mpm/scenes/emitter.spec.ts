import { describe, expect, it } from 'vitest';
import {
	GATE4_LIQUID_BULK_MODULUS_SCALE,
	GATE4_LIQUID_PARTICLE_AREA
} from './gate4.js';
import { createMpmEmitterParticles } from './emitter.js';

const BASE_CONFIG = {
	count: 24,
	seed: 0x51eed,
	center: { x: 24, y: 24 },
	shape: 'circle' as const,
	spacing: 1,
	velocity: { x: 2, y: -3 },
	material: 'water' as const,
	grid: { cellsX: 64, cellsY: 64, cellSize: 1 }
};

describe('deterministic GPU MPM emitter', () => {
	it('creates repeatable phase-specific particles with the requested velocity', () => {
		const first = createMpmEmitterParticles(BASE_CONFIG);
		const second = createMpmEmitterParticles(BASE_CONFIG);

		expect(first).toEqual(second);
		expect(first).toHaveLength(BASE_CONFIG.count);
		expect(first.every((particle) => particle.material?.model === 'water')).toBe(true);
		expect(
			first.every(
				(particle) => particle.material?.lambda === 40_000 * GATE4_LIQUID_BULK_MODULUS_SCALE
			)
		).toBe(true);
		expect(first.every((particle) => particle.volume === 25)).toBe(true);
		expect(first.every((particle) => particle.velocity.x === 2 && particle.velocity.y === -3)).toBe(true);
		expect(first.every((particle) => particle.position.x >= 0 && particle.position.y >= 0)).toBe(true);
	});

	it('supports distinct oil, solid, and rectangle source definitions', () => {
		const water = createMpmEmitterParticles({
			...BASE_CONFIG,
			particleVolume: GATE4_LIQUID_PARTICLE_AREA
		});
		const oil = createMpmEmitterParticles({
			...BASE_CONFIG,
			material: 'oil',
			particleVolume: GATE4_LIQUID_PARTICLE_AREA
		});
		const solid = createMpmEmitterParticles({
			...BASE_CONFIG,
			material: 'neo-hookean',
			shape: 'rectangle',
			particleVolume: 100
		});

		expect(water.every((particle) => particle.volume === GATE4_LIQUID_PARTICLE_AREA)).toBe(true);
		expect(oil.every((particle) => particle.material?.model === 'oil')).toBe(true);
		expect(
			oil.every(
				(particle) => particle.material?.lambda === 30_000 * GATE4_LIQUID_BULK_MODULUS_SCALE
			)
		).toBe(true);
		expect(solid.every((particle) => particle.material?.model === 'neo-hookean')).toBe(true);
		expect(solid.every((particle) => particle.volume === 100)).toBe(true);
		expect(oil.every((particle) => particle.volume === GATE4_LIQUID_PARTICLE_AREA)).toBe(true);
		expect(new Set(solid.map(({ position }) => `${position.x}:${position.y}`)).size).toBe(
			BASE_CONFIG.count
		);
	});

	it('rejects unsafe count, seed, and positions rather than emitting partial batches', () => {
		expect(() =>
			createMpmEmitterParticles({ ...BASE_CONFIG, count: 0 })
		).toThrow(/count/);
		expect(() =>
			createMpmEmitterParticles({ ...BASE_CONFIG, seed: -1 })
		).toThrow(/seed/);
		expect(() =>
			createMpmEmitterParticles({ ...BASE_CONFIG, particleVolume: 0 })
		).toThrow(/particleVolume/);
		expect(() =>
			createMpmEmitterParticles({
				...BASE_CONFIG,
				center: { x: 0, y: 0 },
				count: 16
			})
		).toThrow(/outside the simulation domain/);
	});
});
