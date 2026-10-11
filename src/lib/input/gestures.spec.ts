import { describe, expect, it } from 'vitest';
import {
	classifyTwoFingerGesture,
	modeForDialRotation,
	shortestAngleDelta
} from './gestures';

describe('two-finger gestures', () => {
	it('classifies camera movement and dial rotation at their thresholds', () => {
		expect(classifyTwoFingerGesture(0, 0.02, 8)).toBeNull();
		expect(classifyTwoFingerGesture(0, 0.1, 0)).toBe('camera');
		expect(classifyTwoFingerGesture(0.4, 0, 0)).toBe('dial');
	});

	it('uses the strongest normalized movement when gestures overlap', () => {
		expect(classifyTwoFingerGesture(0.3, 0.2, 20)).toBe('camera');
	});

	it('moves the dial between all modes and wraps in either direction', () => {
		expect(modeForDialRotation('select', (2 * Math.PI) / 3)).toBe('place');
		expect(modeForDialRotation('select', (-2 * Math.PI) / 3)).toBe('flow');
		expect(modeForDialRotation('flow', (2 * Math.PI) / 3)).toBe('select');
	});

	it('unwraps angle changes across the -pi/pi boundary', () => {
		expect(shortestAngleDelta(Math.PI - 0.1, -Math.PI + 0.1)).toBeCloseTo(0.2);
	});
});
