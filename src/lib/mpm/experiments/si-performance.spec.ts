import { describe, expect, it } from 'vitest';
import type { MpmParticle } from '../reference/solver.js';
import { summarizeSiWater } from './si-performance.js';

const particle: MpmParticle = {
	position: { x: 0.1, y: 0.1 },
	velocity: { x: 3, y: 4 },
	C: { m00: 0, m01: 0, m10: 0, m11: 0 },
	F: { m00: 1, m01: 0, m10: 0, m11: 1 },
	volume: 0.001
};

describe('SI performance quality diagnostics', () => {
	it('measures current area, density error and speed rather than reference area alone', () => {
		const result = summarizeSiWater([
			particle,
			{ ...particle, F: { m00: 0.98, m01: 0, m10: 0, m11: 1 } }
		], 0.002);
		expect(result.areaRatio).toBeCloseTo(0.99);
		expect(result.areaDriftPercent).toBeCloseTo(-1);
		expect(result.densityErrorPercentP95).toBeCloseTo(100 * (1 / 0.98 - 1));
		expect(result.maximumSpeed).toBe(5);
	});

	it('rejects empty, invalid and non-positive states', () => {
		expect(() => summarizeSiWater([], 1)).toThrow();
		expect(() => summarizeSiWater([particle], 0)).toThrow();
		expect(() => summarizeSiWater([{ ...particle, volume: undefined }], 1)).toThrow();
		expect(() => summarizeSiWater([{ ...particle, velocity: { x: NaN, y: 0 } }], 1)).toThrow();
		expect(() => summarizeSiWater([{
			...particle, F: { m00: -1, m01: 0, m10: 0, m11: 1 }
		}], 1)).toThrow();
	});
});
