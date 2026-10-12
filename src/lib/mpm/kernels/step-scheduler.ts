export interface MpmStepBatchInput {
	readonly accumulatorSeconds: number;
	readonly elapsedSeconds: number;
	readonly simulationSpeed: number;
	readonly fixedStepSeconds: number;
	readonly maxSteps: number;
}

export interface MpmStepBatch {
	readonly stepCount: number;
	readonly remainderSeconds: number;
	readonly discardedSeconds: number;
}

export function calculateMpmStepBatch({
	accumulatorSeconds,
	elapsedSeconds,
	simulationSpeed,
	fixedStepSeconds,
	maxSteps
}: MpmStepBatchInput): MpmStepBatch {
	if (!Number.isFinite(accumulatorSeconds) || accumulatorSeconds < 0) {
		throw new RangeError('accumulatorSeconds must be finite and non-negative.');
	}
	if (!Number.isFinite(elapsedSeconds) || elapsedSeconds < 0) {
		throw new RangeError('elapsedSeconds must be finite and non-negative.');
	}
	if (!Number.isFinite(simulationSpeed) || simulationSpeed < 0) {
		throw new RangeError('simulationSpeed must be finite and non-negative.');
	}
	if (!Number.isFinite(fixedStepSeconds) || fixedStepSeconds <= 0) {
		throw new RangeError('fixedStepSeconds must be positive and finite.');
	}
	if (!Number.isSafeInteger(maxSteps) || maxSteps < 1) {
		throw new RangeError('maxSteps must be a positive safe integer.');
	}

	const accumulatedSeconds = accumulatorSeconds + elapsedSeconds * simulationSpeed;
	if (!Number.isFinite(accumulatedSeconds)) {
		throw new RangeError('accumulated simulation time is outside the supported range.');
	}

	const requestedSteps = Math.floor(accumulatedSeconds / fixedStepSeconds + 1e-9);
	if (!Number.isSafeInteger(requestedSteps)) {
		throw new RangeError('requested simulation step count is outside the supported range.');
	}

	const stepCount = Math.min(requestedSteps, maxSteps);
	const unconsumedSeconds = Math.max(
		0,
		accumulatedSeconds - stepCount * fixedStepSeconds
	);
	const discardedSeconds = requestedSteps > maxSteps ? unconsumedSeconds : 0;

	return {
		stepCount,
		remainderSeconds: discardedSeconds > 0 ? 0 : unconsumedSeconds,
		discardedSeconds
	};
}
