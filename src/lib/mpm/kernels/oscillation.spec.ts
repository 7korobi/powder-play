import { describe, expect, it } from 'vitest';
import {
	DEFAULT_HORIZONTAL_OSCILLATION_AMPLITUDE,
	HORIZONTAL_OSCILLATION_FREQUENCY_HZ,
	meanSinusoidalAcceleration
} from './oscillation.js';

describe('meanSinusoidalAcceleration', () => {
	const periodSeconds = 1 / HORIZONTAL_OSCILLATION_FREQUENCY_HZ;

	it('integrates the requested acceleration over a complete oscillation', () => {
		expect(
			meanSinusoidalAcceleration(
				0,
				periodSeconds,
				DEFAULT_HORIZONTAL_OSCILLATION_AMPLITUDE
			)
		).toBeCloseTo(0, 12);
	});

	it('uses the time-average acceleration over a batch interval', () => {
		expect(
			meanSinusoidalAcceleration(
				0,
				periodSeconds / 2,
				DEFAULT_HORIZONTAL_OSCILLATION_AMPLITUDE
			)
		).toBeCloseTo((2 * DEFAULT_HORIZONTAL_OSCILLATION_AMPLITUDE) / Math.PI, 12);
	});

	it('returns zero when the oscillation amplitude is disabled', () => {
		expect(meanSinusoidalAcceleration(2, 1, 0)).toBe(0);
	});

	it('rejects invalid time intervals and frequencies', () => {
		expect(() => meanSinusoidalAcceleration(0, -1, 1)).toThrow(RangeError);
		expect(() => meanSinusoidalAcceleration(0, 1, 1, -0.1)).toThrow(RangeError);
	});
});
