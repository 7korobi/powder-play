export const DEFAULT_HORIZONTAL_OSCILLATION_AMPLITUDE = 240;
export const INITIAL_HORIZONTAL_OSCILLATION_AMPLITUDE = 0;
export const MAX_HORIZONTAL_OSCILLATION_AMPLITUDE = 360;
export const HORIZONTAL_OSCILLATION_FREQUENCY_HZ = 0.16;

export function meanSinusoidalAcceleration(
	startSeconds: number,
	durationSeconds: number,
	amplitude: number,
	frequencyHz = HORIZONTAL_OSCILLATION_FREQUENCY_HZ
): number {
	if (
		!Number.isFinite(startSeconds) ||
		!Number.isFinite(durationSeconds) ||
		durationSeconds < 0 ||
		!Number.isFinite(amplitude) ||
		!Number.isFinite(frequencyHz) ||
		frequencyHz < 0
	) {
		throw new RangeError('Oscillation inputs must be finite, with non-negative duration and frequency.');
	}

	if (amplitude === 0 || frequencyHz === 0) return 0;

	const angularFrequency = 2 * Math.PI * frequencyHz;
	const halfPhase = (angularFrequency * durationSeconds) / 2;
	const sinc = halfPhase === 0 ? 1 : Math.sin(halfPhase) / halfPhase;
	return (
		amplitude *
		Math.sin(angularFrequency * (startSeconds + durationSeconds / 2)) *
		sinc
	);
}
