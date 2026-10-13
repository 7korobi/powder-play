import { describe, expect, it } from 'vitest';
import {
	calculateMpmCflSafeDt,
	calculateMpmCflVelocityLimit,
	calculateMpmCflMicrosteps,
	DEFAULT_MPM_COURANT_LIMIT,
	type MpmCflMicrostepInput
} from './cfl.js';

const baseInput: MpmCflMicrostepInput = {
	macroDt: 0.025,
	cellSize: 0.1,
	waveSpeedBound: 2,
	velocityBound: 1,
	maxMicrosteps: 3
};

describe('MPM CFL microstep policy', () => {
	it('calculates the maximum safe duration for one CFL microstep', () => {
		expect(calculateMpmCflSafeDt(baseInput)).toBeCloseTo(
			(DEFAULT_MPM_COURANT_LIMIT * baseInput.cellSize) /
				(baseInput.waveSpeedBound + baseInput.velocityBound)
		);
	});

	it('uses the default Courant limit and returns a complete microstep plan', () => {
		const plan = calculateMpmCflMicrosteps(baseInput);

		expect(DEFAULT_MPM_COURANT_LIMIT).toBe(0.3);
		expect(plan.stepCount).toBe(3);
		expect(plan.safeDt).toBeCloseTo((0.3 * baseInput.cellSize) / 3);
		expect(plan.microDt).toBeCloseTo(baseInput.macroDt / plan.stepCount);
		expect(plan.microDt).toBeLessThanOrEqual(plan.safeDt);
	});

	it('accepts a certified minimum count for a finer CFL plan', () => {
		const plan = calculateMpmCflMicrosteps({
			...baseInput,
			maxMicrosteps: 6,
			minimumStepCount: 5
		});

		expect(plan.stepCount).toBe(5);
		expect(plan.microDt).toBeCloseTo(baseInput.macroDt / 5);
		expect(plan.microDt).toBeLessThanOrEqual(plan.safeDt);
		expect(() =>
			calculateMpmCflMicrosteps({
				...baseInput,
				maxMicrosteps: 4,
				minimumStepCount: 5
			})
		).toThrow(/required microstep count .* exceeds maxMicrosteps/);
	});

	it('accepts a custom positive Courant limit', () => {
		const plan = calculateMpmCflMicrosteps({
			macroDt: 0.25,
			cellSize: 1,
			waveSpeedBound: 2,
			velocityBound: 3,
			courantLimit: 0.5,
			maxMicrosteps: 3
		});

		expect(plan).toEqual({
			stepCount: 3,
			microDt: 0.25 / 3,
			safeDt: 0.1
		});
	});

	it('keeps the float32 microstep within its safe timestep', () => {
		const plan = calculateMpmCflMicrosteps({
			macroDt: 1,
			cellSize: 1,
			waveSpeedBound: 4,
			velocityBound: 6,
			courantLimit: 1,
			maxMicrosteps: 11
		});

		expect(plan.stepCount).toBe(11);
		expect(Math.fround(plan.microDt)).toBeLessThanOrEqual(plan.safeDt);
	});

	it('derives the conservative actual-speed allowance from float32 microDt', () => {
		const microDt = 0.001;
		const velocityLimit = calculateMpmCflVelocityLimit({
			cellSize: 10,
			waveSpeedBound: 1_400,
			microDt,
			courantLimit: 0.3
		});

		expect(Math.fround(velocityLimit)).toBe(velocityLimit);
		expect(
			(velocityLimit + Math.fround(1_400)) * Math.fround(microDt)
		).toBeLessThanOrEqual(0.3 * Math.fround(10));
		expect(velocityLimit).toBeGreaterThan(1_599);
	});

	it('rejects invalid inputs and microsteps that exceed the acoustic CFL limit', () => {
		expect(() =>
			calculateMpmCflVelocityLimit({
				cellSize: 10,
				waveSpeedBound: 400,
				microDt: 0.01
			})
		).toThrow(/no finite non-negative CFL velocity allowance/);
		expect(() =>
			calculateMpmCflVelocityLimit({
				cellSize: 0,
				waveSpeedBound: 1,
				microDt: 0.01
			})
		).toThrow(RangeError);
		expect(() =>
			calculateMpmCflVelocityLimit({
				cellSize: 10,
				waveSpeedBound: Number.POSITIVE_INFINITY,
				microDt: 0.01
			})
		).toThrow(RangeError);
		expect(() =>
			calculateMpmCflVelocityLimit({
				cellSize: 10,
				waveSpeedBound: 1,
				microDt: 0
			})
		).toThrow(RangeError);
	});

	it('uses one complete step when both speed bounds are zero', () => {
		const plan = calculateMpmCflMicrosteps({
			...baseInput,
			macroDt: 0.1,
			waveSpeedBound: 0,
			velocityBound: 0,
			maxMicrosteps: 1
		});

		expect(plan).toEqual({
			stepCount: 1,
			microDt: 0.1,
			safeDt: Number.POSITIVE_INFINITY
		});
	});

	it('throws instead of clamping when the required count exceeds the limit', () => {
		expect(() =>
			calculateMpmCflMicrosteps({
				...baseInput,
				maxMicrosteps: 2
			})
		).toThrow(/required microstep count .* exceeds maxMicrosteps/);
	});

	it('handles finite bounds whose intermediate sum or product would overflow', () => {
		const sumOverflowPlan = calculateMpmCflMicrosteps({
			macroDt: 0.1,
			cellSize: Number.MAX_VALUE,
			waveSpeedBound: Number.MAX_VALUE,
			velocityBound: Number.MAX_VALUE,
			maxMicrosteps: 1
		});
		const productOverflowPlan = calculateMpmCflMicrosteps({
			macroDt: 1e30,
			cellSize: Number.MAX_VALUE,
			waveSpeedBound: 1,
			velocityBound: 1,
			courantLimit: 2,
			maxMicrosteps: 1
		});

		expect(sumOverflowPlan.safeDt).toBeCloseTo(0.15, 14);
		expect(sumOverflowPlan.microDt).toBeLessThanOrEqual(sumOverflowPlan.safeDt);
		expect(productOverflowPlan.safeDt).toBe(Number.MAX_VALUE);
		expect(productOverflowPlan.microDt).toBeLessThanOrEqual(productOverflowPlan.safeDt);
	});

	it('rejects CFL intervals that cannot be represented by finite float32 microsteps', () => {
		expect(() =>
			calculateMpmCflMicrosteps({
				macroDt: Number.MAX_VALUE,
				cellSize: 1,
				waveSpeedBound: 0,
				velocityBound: 0,
				maxMicrosteps: Number.MAX_SAFE_INTEGER
			})
		).toThrow(/required float32 microstep count .* exceeds maxMicrosteps/);
	});

	it('rejects invalid macro dt, cell size, speed bounds, limit, and step cap', () => {
		const expectInvalid = (overrides: Partial<MpmCflMicrostepInput>) => {
			expect(() =>
				calculateMpmCflMicrosteps({
					...baseInput,
					...overrides
				})
			).toThrow(RangeError);
		};

		expectInvalid({ macroDt: 0 });
		expectInvalid({ macroDt: Number.NaN });
		expectInvalid({ macroDt: Number.POSITIVE_INFINITY });
		expectInvalid({ cellSize: 0 });
		expectInvalid({ cellSize: Number.NaN });
		expectInvalid({ waveSpeedBound: -1 });
		expectInvalid({ waveSpeedBound: Number.POSITIVE_INFINITY });
		expectInvalid({ velocityBound: -1 });
		expectInvalid({ velocityBound: Number.NaN });
		expectInvalid({ courantLimit: 0 });
		expectInvalid({ courantLimit: Number.POSITIVE_INFINITY });
		expectInvalid({ maxMicrosteps: 0 });
		expectInvalid({ maxMicrosteps: 1.5 });
		expectInvalid({ maxMicrosteps: Number.MAX_SAFE_INTEGER + 1 });
	});
});
