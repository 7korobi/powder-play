import {
	calculateMpmCflMicrosteps,
	type MpmCflMicrostepInput,
	type MpmCflMicrostepPlan
} from './cfl.js';
import {
	calculateMpmCflVelocityBound,
	type MpmCflVelocityEnvelopeInput
} from './cfl-bounds.js';

export interface MpmStepBatchInput
	extends Omit<MpmCflMicrostepInput, 'macroDt' | 'maxMicrosteps'> {
	readonly accumulatorSeconds: number;
	readonly elapsedSeconds: number;
	readonly simulationSpeed: number;
	readonly fixedStepSeconds: number;
	readonly maxMicrosteps: number;
}

export interface MpmStepBatch {
	readonly macroStepCount: number;
	readonly microStepCount: number;
	readonly simulatedSeconds: number;
	readonly remainderSeconds: number;
	readonly microstepPlan: MpmCflMicrostepPlan;
}

export interface MpmCflStepBatchInput
	extends Omit<MpmStepBatchInput, 'waveSpeedBound' | 'velocityBound'>,
		Pick<MpmCflVelocityEnvelopeInput, 'gravityMagnitude' | 'oscillationAmplitude' | 'dragRadius' | 'dragStiffness' | 'dragDamping'> {
	readonly waveSpeedBound: number;
	readonly currentVelocityBound: number;
}

export interface MpmCflStepBatch {
	readonly batch: MpmStepBatch;
	readonly velocityBound: number;
}

export function calculateMpmStepBatch({
	accumulatorSeconds,
	elapsedSeconds,
	simulationSpeed,
	fixedStepSeconds,
	cellSize,
	waveSpeedBound,
	velocityBound,
	courantLimit,
	maxMicrosteps
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

	const accumulatedSeconds = accumulatorSeconds + elapsedSeconds * simulationSpeed;
	if (!Number.isFinite(accumulatedSeconds)) {
		throw new RangeError('accumulated simulation time is outside the supported range.');
	}

	const microstepPlan = calculateMpmCflMicrosteps({
		macroDt: fixedStepSeconds,
		cellSize,
		waveSpeedBound,
		velocityBound,
		courantLimit,
		maxMicrosteps
	});
	const requestedMacroSteps = Math.floor(accumulatedSeconds / fixedStepSeconds);
	if (!Number.isSafeInteger(requestedMacroSteps)) {
		throw new RangeError('requested macrostep count is outside the supported range.');
	}

	const microStepCount = requestedMacroSteps * microstepPlan.stepCount;
	if (!Number.isSafeInteger(microStepCount) || microStepCount > maxMicrosteps) {
		throw new RangeError(
			`requested microstep count (${microStepCount}) exceeds maxMicrosteps (${maxMicrosteps}).`
		);
	}

	const simulatedSeconds = requestedMacroSteps * fixedStepSeconds;
	if (!Number.isFinite(simulatedSeconds)) {
		throw new RangeError('requested simulation duration is outside the supported range.');
	}
	const remainderSeconds = Math.max(0, accumulatedSeconds - simulatedSeconds);

	return {
		macroStepCount: requestedMacroSteps,
		microStepCount,
		simulatedSeconds,
		remainderSeconds,
		microstepPlan
	};
}

export function calculateMpmCflStepBatch({
	currentVelocityBound,
	gravityMagnitude,
	oscillationAmplitude,
	dragRadius,
	dragStiffness,
	dragDamping,
	...batchInput
}: MpmCflStepBatchInput): MpmCflStepBatch {
	const baseInput: MpmStepBatchInput = {
		...batchInput,
		velocityBound: currentVelocityBound
	};
	const preview = calculateMpmStepBatch(baseInput);
	const velocityBound = calculateMpmCflVelocityBound({
		currentVelocityBound,
		durationSeconds: preview.simulatedSeconds,
		gravityMagnitude,
		oscillationAmplitude,
		dragRadius,
		dragStiffness,
		dragDamping
	});
	const batch = calculateMpmStepBatch({ ...baseInput, velocityBound });
	const gpuMicroDt = Math.fround(batch.microstepPlan.microDt);
	const gpuDragDamping = Math.fround(dragDamping);
	if (
		batch.macroStepCount > 0 &&
		dragRadius > 0 &&
		(!Number.isFinite(gpuMicroDt) ||
			gpuMicroDt <= 0 ||
			!Number.isFinite(gpuDragDamping) ||
			gpuMicroDt * gpuDragDamping > 1)
	) {
		throw new RangeError(
			'CFL microDt is too large for the explicit drag damping update to remain dissipative.'
		);
	}

	return { batch, velocityBound };
}
