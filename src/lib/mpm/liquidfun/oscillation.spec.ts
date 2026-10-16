import { describe, expect, it } from 'vitest';
import {
	HORIZONTAL_OSCILLATION_FREQUENCY_HZ,
	meanSinusoidalAcceleration
} from '../kernels/oscillation.js';
import { LIQUIDFUN_DT_SECONDS } from './workload.js';
import { calculateLiquidFunStepGravity } from './oscillation.js';

describe('calculateLiquidFunStepGravity', () => {
	it('preserves the manual gravity when horizontal oscillation is disabled', () => {
		const manualGravity = { x: 15, y: 90 };

		expect(calculateLiquidFunStepGravity(manualGravity, 2, 0)).toEqual(manualGravity);
	});

	it('adds the step-averaged oscillation to X without changing Y', () => {
		const manualGravity = { x: 15, y: 90 };
		const amplitude = 240;
		const oscillatingGravity = calculateLiquidFunStepGravity(
			manualGravity,
			0,
			amplitude
		);

		expect(oscillatingGravity.x).toBeCloseTo(
			manualGravity.x +
				meanSinusoidalAcceleration(0, LIQUIDFUN_DT_SECONDS, amplitude),
			12
		);
		expect(oscillatingGravity.y).toBe(manualGravity.y);
	});

	it('has no net horizontal acceleration over a complete oscillation cycle', () => {
		const stepCount = Math.round(
			1 /
				(HORIZONTAL_OSCILLATION_FREQUENCY_HZ * LIQUIDFUN_DT_SECONDS)
		);
		let accumulatedVelocityChange = 0;

		for (let step = 0; step < stepCount; step += 1) {
			const stepGravity = calculateLiquidFunStepGravity(
				{ x: 0, y: 0 },
				step * LIQUIDFUN_DT_SECONDS,
				240
			);
			accumulatedVelocityChange += stepGravity.x * LIQUIDFUN_DT_SECONDS;
		}

		expect(accumulatedVelocityChange).toBeCloseTo(0, 10);
	});
});
