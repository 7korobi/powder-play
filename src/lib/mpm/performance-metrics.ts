export interface MpmMetricDistribution {
	readonly median: number;
	readonly p95: number;
	readonly min: number;
	readonly max: number;
	readonly samples: number;
}

interface TimedMetricSample {
	readonly timestampMs: number;
	readonly value: number;
}

export const MPM_LIVE_METRIC_WINDOW_MS = 5_000;

export class MpmRollingMetric {
	private readonly samples: TimedMetricSample[] = [];

	constructor(private readonly windowMs = MPM_LIVE_METRIC_WINDOW_MS) {
		if (!Number.isFinite(windowMs) || windowMs <= 0) {
			throw new RangeError('windowMs must be finite and greater than zero.');
		}
	}

	record(value: number, timestampMs: number): void {
		if (!Number.isFinite(value) || value < 0) {
			throw new RangeError('Metric values must be finite and non-negative.');
		}
		if (!Number.isFinite(timestampMs) || timestampMs < 0) {
			throw new RangeError('Metric timestamps must be finite and non-negative.');
		}
		const previous = this.samples.at(-1);
		if (previous && timestampMs < previous.timestampMs) {
			throw new RangeError('Metric timestamps must be monotonic.');
		}

		this.samples.push({ value, timestampMs });
		this.discardExpired(timestampMs);
	}

	summarize(nowMs: number): MpmMetricDistribution | null {
		if (!Number.isFinite(nowMs) || nowMs < 0) {
			throw new RangeError('Summary timestamps must be finite and non-negative.');
		}
		this.discardExpired(nowMs);
		if (this.samples.length === 0) return null;

		const sorted = this.samples
			.map(({ value }) => value)
			.sort((first, second) => first - second);
		const midpoint = sorted.length / 2;
		const median =
			sorted.length % 2 === 0
				? ((sorted[midpoint - 1] ?? 0) + (sorted[midpoint] ?? 0)) / 2
				: (sorted[Math.floor(midpoint)] ?? 0);
		const p95Index = Math.ceil(sorted.length * 0.95) - 1;

		return {
			median,
			p95: sorted[p95Index] ?? 0,
			min: sorted[0] ?? 0,
			max: sorted[sorted.length - 1] ?? 0,
			samples: sorted.length
		};
	}

	clear(): void {
		this.samples.length = 0;
	}

	private discardExpired(nowMs: number): void {
		const cutoffMs = nowMs - this.windowMs;
		while (
			this.samples.length > 0 &&
			(this.samples[0]?.timestampMs ?? 0) < cutoffMs
		) {
			this.samples.shift();
		}
	}
}
