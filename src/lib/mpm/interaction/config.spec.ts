import { describe, expect, it } from 'vitest';
import type { MpmLabConfig } from '#lib/mpm/contracts.js';
import { buildMpmInteractionLayout, MAX_INTERACTION_PARTICLES } from './config.js';

const baseConfig: MpmLabConfig = {
	worldWidth: 100,
	worldHeight: 80,
	capacity: 8,
	particles: [
		{ position: { x: 10, y: 12 }, materialId: 1, objectId: 9 },
		{ position: { x: 30, y: 14 }, materialId: 2, objectId: 4 },
		{
			position: { x: 11, y: 13 },
			velocity: { x: 2, y: -3 },
			materialId: 3,
			objectId: 9
		}
	]
};

describe('MPM interaction config layout', () => {
	it('validates input and packs each object into one contiguous range', () => {
		const layout = buildMpmInteractionLayout(baseConfig);

		expect(layout.worldWidth).toBe(100);
		expect(layout.worldHeight).toBe(80);
		expect(layout.objects).toEqual([
			{ objectId: 9, start: 0, count: 2 },
			{ objectId: 4, start: 2, count: 1 }
		]);
		expect([...layout.particleState]).toEqual([
			10, 12, 0, 0,
			11, 13, 2, -3,
			30, 14, 0, 0
		]);
		expect([...layout.materialIds]).toEqual([1, 3, 2]);
	});

	it('rejects invalid dimensions, positions, ids, and capacity', () => {
		expect(() => buildMpmInteractionLayout({ ...baseConfig, worldWidth: Number.NaN })).toThrow();
		expect(() =>
			buildMpmInteractionLayout({
				...baseConfig,
				particles: [{ ...baseConfig.particles[0], position: { x: 101, y: 0 } }]
			})
		).toThrow(/world bounds/);
		expect(() =>
			buildMpmInteractionLayout({
				...baseConfig,
				particles: [{ ...baseConfig.particles[0], objectId: -1 }]
			})
		).toThrow(/objectId/);
		expect(() => buildMpmInteractionLayout({ ...baseConfig, capacity: 2 })).toThrow(
			/particle count/
		);
		expect(() =>
			buildMpmInteractionLayout({
				...baseConfig,
				capacity: MAX_INTERACTION_PARTICLES + 1
			})
		).toThrow(/capacity/);
	});
});
