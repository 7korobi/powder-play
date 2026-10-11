import { describe, expect, it } from 'vitest';
import {
	blendVector,
	screenAccelerationForce,
	screenGravityFromAcceleration,
	STANDARD_GRAVITY
} from './sensors';

describe('device motion mapping', () => {
	it.each([
		[0, 0, STANDARD_GRAVITY],
		[90, STANDARD_GRAVITY, 0],
		[180, 0, -STANDARD_GRAVITY],
		[270, -STANDARD_GRAVITY, 0]
	])('maps a stationary device at screen angle %i to downward gravity', (angle, ax, ay) => {
		const gravity = screenGravityFromAcceleration(ax, ay, angle);
		expect(gravity.x).toBeCloseTo(0);
		expect(gravity.y).toBeCloseTo(1);
	});

	it('falls back to screen-down when the device has no in-plane gravity', () => {
		expect(screenGravityFromAcceleration(0, 0, 0)).toEqual({ x: 0, y: 1 });
		expect(screenGravityFromAcceleration(null, null, 0)).toEqual({ x: 0, y: 1 });
	});

	it('maps linear acceleration into a bounded inertial force', () => {
		expect(screenAccelerationForce(-STANDARD_GRAVITY, 0, 0)).toEqual({ x: 1, y: 0 });
		expect(screenAccelerationForce(0, -STANDARD_GRAVITY * 4, 0)).toEqual({ x: 0, y: -1 });
	});

	it('smooths vectors without mutating the input', () => {
		const previous = { x: 0, y: 1 };
		expect(blendVector(previous, { x: 1, y: 0 }, 0.25)).toEqual({ x: 0.25, y: 0.75 });
		expect(previous).toEqual({ x: 0, y: 1 });
	});
});
