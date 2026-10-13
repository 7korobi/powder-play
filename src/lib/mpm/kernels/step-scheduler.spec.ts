import { describe, expect, it } from 'vitest';
import { calculateMpmStepBatch } from './step-scheduler.js';

const FIXED_STEP_SECONDS = 0.025;
const CFL_BOUNDS = {
	cellSize: 0.1,
	waveSpeedBound: 2,
	velocityBound: 1
};

describe('MPM macrostep and CFL microstep scheduling', () => {
	it('plans complete microsteps while preserving requested macro time', () => {
		const batch = calculateMpmStepBatch({
			accumulatorSeconds: 0,
			elapsedSeconds: FIXED_STEP_SECONDS * 2,
			simulationSpeed: 1,
			fixedStepSeconds: FIXED_STEP_SECONDS,
			...CFL_BOUNDS,
			maxMicrosteps: 6
		});

		expect(batch.macroStepCount).toBe(2);
		expect(batch.microstepPlan.stepCount).toBe(3);
		expect(batch.microStepCount).toBe(6);
		expect(batch.simulatedSeconds).toBe(FIXED_STEP_SECONDS * 2);
		expect(batch.microstepPlan.microDt).toBeLessThanOrEqual(batch.microstepPlan.safeDt);
		expect(batch.microStepCount * batch.microstepPlan.microDt).toBeCloseTo(
			batch.simulatedSeconds
		);
		expect(batch.remainderSeconds).toBeCloseTo(0);
	});

	it.each([
		[0.5, 2],
		[1, 4],
		[2, 8],
		[4, 16]
	])('preserves %s× acceleration as %i macrosteps per 60 Hz frame', (speed, expected) => {
		const batch = calculateMpmStepBatch({
			accumulatorSeconds: 0,
			elapsedSeconds: 1 / 60,
			simulationSpeed: speed,
			fixedStepSeconds: 1 / 240,
			cellSize: 0.1,
			waveSpeedBound: 20,
			velocityBound: 1,
			maxMicrosteps: 64
		});

		expect(batch.macroStepCount).toBe(expected);
		expect(batch.microstepPlan.stepCount).toBe(3);
		expect(batch.microStepCount).toBe(expected * 3);
		expect(batch.simulatedSeconds).toBeCloseTo((1 / 60) * speed);
	});

	it('preserves fractional macrostep time across frames', () => {
		const halfStep = FIXED_STEP_SECONDS / 2;
		const firstBatch = calculateMpmStepBatch({
			accumulatorSeconds: 0,
			elapsedSeconds: halfStep,
			simulationSpeed: 1,
			fixedStepSeconds: FIXED_STEP_SECONDS,
			...CFL_BOUNDS,
			maxMicrosteps: 6
		});
		const secondBatch = calculateMpmStepBatch({
			accumulatorSeconds: firstBatch.remainderSeconds,
			elapsedSeconds: halfStep,
			simulationSpeed: 1,
			fixedStepSeconds: FIXED_STEP_SECONDS,
			...CFL_BOUNDS,
			maxMicrosteps: 6
		});

		expect(firstBatch.macroStepCount).toBe(0);
		expect(firstBatch.microStepCount).toBe(0);
		expect(firstBatch.remainderSeconds).toBeCloseTo(halfStep);
		expect(secondBatch.macroStepCount).toBe(1);
		expect(secondBatch.microStepCount).toBe(3);
		expect(secondBatch.remainderSeconds).toBeCloseTo(0);
	});

	it('throws rather than dropping requested time when the multiplied work cap is exceeded', () => {
		expect(() =>
			calculateMpmStepBatch({
				accumulatorSeconds: 0,
				elapsedSeconds: FIXED_STEP_SECONDS * 2,
				simulationSpeed: 1,
				fixedStepSeconds: FIXED_STEP_SECONDS,
				...CFL_BOUNDS,
				maxMicrosteps: 5
			})
		).toThrow(/requested microstep count .* exceeds maxMicrosteps/);
	});

	it('throws when one macrostep alone requires more than the frame cap', () => {
		expect(() =>
			calculateMpmStepBatch({
				accumulatorSeconds: 0,
				elapsedSeconds: 0,
				simulationSpeed: 1,
				fixedStepSeconds: FIXED_STEP_SECONDS,
				cellSize: 0.1,
				waveSpeedBound: 100,
				velocityBound: 0,
				maxMicrosteps: 64
			})
		).toThrow(/required microstep count .* exceeds maxMicrosteps/);
	});

	it('rejects invalid time, speed, and cap inputs', () => {
		const baseInput = {
			accumulatorSeconds: 0,
			elapsedSeconds: 1 / 60,
			simulationSpeed: 1,
			fixedStepSeconds: FIXED_STEP_SECONDS,
			...CFL_BOUNDS,
			maxMicrosteps: 6
		};

		expect(() =>
			calculateMpmStepBatch({ ...baseInput, accumulatorSeconds: Number.NaN })
		).toThrow(RangeError);
		expect(() =>
			calculateMpmStepBatch({ ...baseInput, elapsedSeconds: Number.POSITIVE_INFINITY })
		).toThrow(RangeError);
		expect(() =>
			calculateMpmStepBatch({ ...baseInput, simulationSpeed: -1 })
		).toThrow(RangeError);
		expect(() =>
			calculateMpmStepBatch({ ...baseInput, fixedStepSeconds: 0 })
		).toThrow(RangeError);
		expect(() =>
			calculateMpmStepBatch({ ...baseInput, maxMicrosteps: 0 })
		).toThrow(RangeError);
		expect(() =>
			calculateMpmStepBatch({ ...baseInput, velocityBound: -1 })
		).toThrow(RangeError);
	});
});
