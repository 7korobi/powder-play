import {
	calculateMpmCflSafeDt,
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
	readonly allowPartialBatch?: boolean;
}

export interface MpmStepBatch {
	readonly macroStepCount: number;
	readonly macroDt: number;
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

function isMicrostepCapacityError(error: unknown): boolean {
	return (
		error instanceof RangeError &&
		(/exceeds maxMicrosteps|microDt is too small to represent/.test(error.message))
	);
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
	maxMicrosteps,
	allowPartialBatch = false
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

	let macroDt = fixedStepSeconds;
	let microstepPlan: MpmCflMicrostepPlan;
	try {
		microstepPlan = calculateMpmCflMicrosteps({
			macroDt,
			cellSize,
			waveSpeedBound,
			velocityBound,
			courantLimit,
			maxMicrosteps
		});
	} catch (error) {
		if (!allowPartialBatch || !isMicrostepCapacityError(error)) throw error;
		const safeDt = calculateMpmCflSafeDt({
			cellSize,
			waveSpeedBound,
			velocityBound,
			courantLimit
		});
		const reservedStepCount = Math.max(
			1,
			maxMicrosteps - (maxMicrosteps > 1 ? 1 : 0)
		);
		macroDt = Math.min(
			fixedStepSeconds,
			safeDt * reservedStepCount * (1 - 1e-6)
		);
		if (!Number.isFinite(macroDt) || macroDt <= 0) {
			throw new RangeError(
				`No positive CFL-safe macrostep duration fits the ${maxMicrosteps}-microstep batch cap.`
			);
		}
		microstepPlan = calculateMpmCflMicrosteps({
			macroDt,
			cellSize,
			waveSpeedBound,
			velocityBound,
			courantLimit,
			maxMicrosteps
		});
	}
	const requestedMacroSteps = Math.floor(accumulatedSeconds / fixedStepSeconds);
	if (!Number.isSafeInteger(requestedMacroSteps)) {
		throw new RangeError('requested macrostep count is outside the supported range.');
	}

	const macroStepCount =
		macroDt < fixedStepSeconds
			? requestedMacroSteps > 0
				? 1
				: 0
			: allowPartialBatch
				? Math.min(
						requestedMacroSteps,
						Math.floor(maxMicrosteps / microstepPlan.stepCount)
					)
				: requestedMacroSteps;
	const microStepCount = macroStepCount * microstepPlan.stepCount;
	if (!Number.isSafeInteger(microStepCount) || microStepCount > maxMicrosteps) {
		throw new RangeError(
			`requested microstep count (${microStepCount}) exceeds maxMicrosteps (${maxMicrosteps}).`
		);
	}

	const simulatedSeconds = macroStepCount * macroDt;
	if (!Number.isFinite(simulatedSeconds)) {
		throw new RangeError('requested simulation duration is outside the supported range.');
	}
	const remainderSeconds = Math.max(0, accumulatedSeconds - simulatedSeconds);

	return {
		macroStepCount,
		macroDt,
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
