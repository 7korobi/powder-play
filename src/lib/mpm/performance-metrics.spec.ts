import { describe, expect, it } from 'vitest';
import { MpmRollingMetric } from './performance-metrics.js';

describe('MPM rolling performance metrics', () => {
	it('summarizes the median, nearest-rank p95, bounds, and sample count', () => {
		const metric = new MpmRollingMetric(1_000);
		metric.record(4, 100);
		metric.record(1, 200);
		metric.record(3, 300);
		metric.record(2, 400);

		expect(metric.summarize(400)).toEqual({
			median: 2.5,
			p95: 4,
			min: 1,
			max: 4,
			samples: 4
		});
	});

	it('expires samples outside the rolling time window', () => {
		const metric = new MpmRollingMetric(1_000);
		metric.record(10, 100);
		metric.record(20, 500);
		metric.record(30, 1_100);

		expect(metric.summarize(1_100)).toEqual({
			median: 20,
			p95: 30,
			min: 10,
			max: 30,
			samples: 3
		});
		expect(metric.summarize(1_501)).toEqual({
			median: 30,
			p95: 30,
			min: 30,
			max: 30,
			samples: 1
		});
		expect(metric.summarize(2_101)).toBeNull();
	});

	it('clears collected samples', () => {
		const metric = new MpmRollingMetric();
		metric.record(1, 1);
		metric.clear();

		expect(metric.summarize(2)).toBeNull();
	});

	it('rejects invalid windows, samples, and timestamps', () => {
		expect(() => new MpmRollingMetric(0)).toThrow(RangeError);
		expect(() => new MpmRollingMetric(Number.NaN)).toThrow(RangeError);

		const metric = new MpmRollingMetric();
		expect(() => metric.record(Number.NaN, 1)).toThrow(RangeError);
		expect(() => metric.record(1, -1)).toThrow(RangeError);
		metric.record(1, 2);
		expect(() => metric.record(2, 1)).toThrow(RangeError);
		expect(() => metric.summarize(Number.POSITIVE_INFINITY)).toThrow(RangeError);
	});
});
