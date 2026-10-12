import { describe, expect, it } from 'vitest';
import { calculateMpmStepBatch } from './step-scheduler.js';

const FIXED_STEP_SECONDS = 1 / 240;
const MAX_STEPS_PER_FRAME = 128;

describe('MPM fixed-step batch scheduling', () => {
	it.each([
		[0.5, 2],
		[1, 4],
		[2, 8],
		[4, 16],
		[8, 32],
		[16, 64],
		[32, 128]
	])('schedules %s× as %i steps per 60 Hz frame', (simulationSpeed, expectedSteps) => {
		const batch = calculateMpmStepBatch({
			accumulatorSeconds: 0,
			elapsedSeconds: 1 / 60,
			simulationSpeed,
			fixedStepSeconds: FIXED_STEP_SECONDS,
			maxSteps: MAX_STEPS_PER_FRAME
		});

		expect(batch.stepCount).toBe(expectedSteps);
		expect(batch.remainderSeconds).toBeCloseTo(0);
		expect(batch.discardedSeconds).toBe(0);
	});

	it('preserves fractional fixed-step time across frames', () => {
		const halfStep = FIXED_STEP_SECONDS / 2;
		const firstBatch = calculateMpmStepBatch({
			accumulatorSeconds: 0,
			elapsedSeconds: halfStep,
			simulationSpeed: 1,
			fixedStepSeconds: FIXED_STEP_SECONDS,
			maxSteps: MAX_STEPS_PER_FRAME
		});
		const secondBatch = calculateMpmStepBatch({
			accumulatorSeconds: firstBatch.remainderSeconds,
			elapsedSeconds: halfStep,
			simulationSpeed: 1,
			fixedStepSeconds: FIXED_STEP_SECONDS,
			maxSteps: MAX_STEPS_PER_FRAME
		});

		expect(firstBatch.stepCount).toBe(0);
		expect(firstBatch.remainderSeconds).toBeCloseTo(halfStep);
		expect(secondBatch.stepCount).toBe(1);
		expect(secondBatch.remainderSeconds).toBeCloseTo(0);
	});

	it('bounds a delayed frame and reports discarded simulation time', () => {
		const batch = calculateMpmStepBatch({
			accumulatorSeconds: 0,
			elapsedSeconds: 0.1,
			simulationSpeed: 32,
			fixedStepSeconds: FIXED_STEP_SECONDS,
			maxSteps: MAX_STEPS_PER_FRAME
		});

		expect(batch.stepCount).toBe(MAX_STEPS_PER_FRAME);
		expect(batch.remainderSeconds).toBe(0);
		expect(batch.discardedSeconds).toBeCloseTo(
			3.2 - MAX_STEPS_PER_FRAME * FIXED_STEP_SECONDS
		);
	});

	it('rejects invalid batch limits and fixed-step values', () => {
		expect(() =>
			calculateMpmStepBatch({
				accumulatorSeconds: 0,
				elapsedSeconds: 1 / 60,
				simulationSpeed: 1,
				fixedStepSeconds: 0,
				maxSteps: MAX_STEPS_PER_FRAME
			})
		).toThrow(RangeError);
		expect(() =>
			calculateMpmStepBatch({
				accumulatorSeconds: 0,
				elapsedSeconds: 1 / 60,
				simulationSpeed: 1,
				fixedStepSeconds: FIXED_STEP_SECONDS,
				maxSteps: 0
			})
		).toThrow(RangeError);
	});
});
