export const DEFAULT_MPM_COURANT_LIMIT = 0.3;

export type MpmCflMicrostepInput = {
	readonly macroDt: number;
	readonly cellSize: number;
	readonly waveSpeedBound: number;
	readonly velocityBound: number;
	readonly maxMicrosteps: number;
	readonly courantLimit?: number;
};

export type MpmCflMicrostepPlan = {
	readonly stepCount: number;
	readonly microDt: number;
	readonly safeDt: number;
};

export type MpmCflVelocityLimitInput = {
	readonly cellSize: number;
	readonly waveSpeedBound: number;
	readonly microDt: number;
	readonly courantLimit?: number;
};

const MAX_FLOAT32 = 3.4028234663852886e38;

type BinaryParts = {
	readonly mantissa: number;
	readonly exponent: number;
};

// Represent a positive finite number as mantissa * 2 ** exponent.
function decomposePositive(value: number): BinaryParts {
	let exponent = Math.floor(Math.log2(value)) + 1;
	exponent = Math.min(1024, Math.max(-1073, exponent));

	let mantissa = (value / 2 ** (exponent - 1)) / 2;
	while (mantissa >= 1) {
		mantissa *= 0.5;
		exponent += 1;
	}
	while (mantissa < 0.5) {
		mantissa *= 2;
		exponent -= 1;
	}

	return { mantissa, exponent };
}

function scaleByPowerOfTwo(mantissa: number, exponent: number): number {
	if (exponent > 1024) {
		return Number.POSITIVE_INFINITY;
	}
	if (exponent < -1074) {
		return 0;
	}
	if (exponent === 1024) {
		return (mantissa * 2 ** 1023) * 2;
	}
	return mantissa * 2 ** exponent;
}

function calculateSafeDt(
	courantLimit: number,
	cellSize: number,
	waveSpeedBound: number,
	velocityBound: number
): number {
	const speedBound = waveSpeedBound + velocityBound;
	if (speedBound === 0) {
		return Number.POSITIVE_INFINITY;
	}

	const courantDistance = courantLimit * cellSize;
	if (
		Number.isFinite(speedBound) &&
		Number.isFinite(courantDistance) &&
		courantDistance > 0
	) {
		const safeDt = courantDistance / speedBound;
		if (safeDt > 0) {
			return safeDt;
		}
	}

	// Scale the sum and ratio so finite inputs do not overflow in intermediate products.
	const speedScale = Math.max(waveSpeedBound, velocityBound);
	const scaledSpeedBound = waveSpeedBound / speedScale + velocityBound / speedScale;
	const limitParts = decomposePositive(courantLimit);
	const cellParts = decomposePositive(cellSize);
	const speedScaleParts = decomposePositive(speedScale);
	const scaledSpeedParts = decomposePositive(scaledSpeedBound);

	let mantissa =
		(limitParts.mantissa * cellParts.mantissa) /
		(speedScaleParts.mantissa * scaledSpeedParts.mantissa);
	let exponent =
		limitParts.exponent +
		cellParts.exponent -
		speedScaleParts.exponent -
		scaledSpeedParts.exponent;

	while (mantissa >= 1) {
		mantissa *= 0.5;
		exponent += 1;
	}
	while (mantissa < 0.5) {
		mantissa *= 2;
		exponent -= 1;
	}

	return scaleByPowerOfTwo(mantissa, exponent);
}

export function calculateMpmCflMicrosteps(
	input: MpmCflMicrostepInput
): MpmCflMicrostepPlan {
	if (typeof input !== 'object' || input === null) {
		throw new RangeError('input must be an object.');
	}

	const {
		macroDt,
		cellSize,
		waveSpeedBound,
		velocityBound,
		maxMicrosteps,
		courantLimit = DEFAULT_MPM_COURANT_LIMIT
	} = input;

	if (!Number.isFinite(macroDt) || macroDt <= 0) {
		throw new RangeError('macroDt must be positive and finite.');
	}
	if (!Number.isFinite(cellSize) || cellSize <= 0) {
		throw new RangeError('cellSize must be positive and finite.');
	}
	if (!Number.isFinite(waveSpeedBound) || waveSpeedBound < 0) {
		throw new RangeError('waveSpeedBound must be finite and non-negative.');
	}
	if (!Number.isFinite(velocityBound) || velocityBound < 0) {
		throw new RangeError('velocityBound must be finite and non-negative.');
	}
	if (!Number.isFinite(courantLimit) || courantLimit <= 0) {
		throw new RangeError('courantLimit must be positive and finite.');
	}
	if (!Number.isSafeInteger(maxMicrosteps) || maxMicrosteps < 1) {
		throw new RangeError('maxMicrosteps must be a positive safe integer.');
	}

	const safeDt = calculateSafeDt(courantLimit, cellSize, waveSpeedBound, velocityBound);
	if (Number.isNaN(safeDt) || safeDt <= 0) {
		throw new RangeError('safeDt is outside the positive representable numeric range.');
	}

	let stepCount = Math.max(1, Math.ceil(macroDt / safeDt));
	if (stepCount > maxMicrosteps) {
		throw new RangeError(
			`required microstep count (${stepCount}) exceeds maxMicrosteps (${maxMicrosteps}).`
		);
	}

	let microDt = macroDt / stepCount;
	let gpuMicroDt = Math.fround(microDt);
	if (!Number.isFinite(gpuMicroDt)) {
		const minimumFloat32StepCount = Math.ceil(macroDt / MAX_FLOAT32);
		if (
			!Number.isSafeInteger(minimumFloat32StepCount) ||
			minimumFloat32StepCount > maxMicrosteps
		) {
			throw new RangeError(
				`required float32 microstep count (${minimumFloat32StepCount}) exceeds maxMicrosteps (${maxMicrosteps}).`
			);
		}
		stepCount = Math.max(stepCount, minimumFloat32StepCount);
		microDt = macroDt / stepCount;
		gpuMicroDt = Math.fround(microDt);
	}
	while (
		microDt > safeDt ||
		gpuMicroDt > safeDt
	) {
		if (gpuMicroDt === 0) {
			throw new RangeError('microDt is too small to represent as a positive float32 value.');
		}
		stepCount += 1;
		if (stepCount > maxMicrosteps) {
			throw new RangeError(
				`required microstep count (${stepCount}) exceeds maxMicrosteps (${maxMicrosteps}).`
			);
		}
		microDt = macroDt / stepCount;
		gpuMicroDt = Math.fround(microDt);
	}
	if (gpuMicroDt <= 0) {
		throw new RangeError('microDt is too small to represent as a positive float32 value.');
	}
	if (!Number.isFinite(gpuMicroDt)) {
		throw new RangeError('microDt cannot be represented as a finite float32 value.');
	}

	return { stepCount, microDt, safeDt };
}

export function calculateMpmCflVelocityLimit({
	cellSize,
	waveSpeedBound,
	microDt,
	courantLimit = DEFAULT_MPM_COURANT_LIMIT
}: MpmCflVelocityLimitInput): number {
	if (!Number.isFinite(cellSize) || cellSize <= 0) {
		throw new RangeError('cellSize must be positive and finite.');
	}
	if (!Number.isFinite(waveSpeedBound) || waveSpeedBound < 0) {
		throw new RangeError('waveSpeedBound must be finite and non-negative.');
	}
	if (!Number.isFinite(microDt) || microDt <= 0) {
		throw new RangeError('microDt must be positive and finite.');
	}
	if (!Number.isFinite(courantLimit) || courantLimit <= 0) {
		throw new RangeError('courantLimit must be positive and finite.');
	}

	const gpuCellSize = roundDownFloat32(cellSize);
	const gpuWaveSpeedBound = roundUpFloat32(waveSpeedBound);
	const gpuMicroDt = Math.fround(microDt);
	if (gpuCellSize <= 0 || !Number.isFinite(gpuMicroDt) || gpuMicroDt <= 0) {
		throw new RangeError('CFL inputs must be representable as positive float32 values.');
	}
	const courantDistance = courantLimit * gpuCellSize;
	const maximumTotalSpeed = courantDistance / gpuMicroDt;
	const velocityLimit = maximumTotalSpeed - gpuWaveSpeedBound;
	if (
		!Number.isFinite(courantDistance) ||
		courantDistance <= 0 ||
		!Number.isFinite(maximumTotalSpeed) ||
		!Number.isFinite(velocityLimit) ||
		velocityLimit < 0
	) {
		throw new RangeError('microDt leaves no finite non-negative CFL velocity allowance.');
	}

	return roundDownFloat32(velocityLimit);
}

function roundDownFloat32(value: number): number {
	if (!Number.isFinite(value) || value < 0) {
		throw new RangeError('CFL velocity limits must be finite and non-negative.');
	}
	if (value === 0) return 0;

	const data = new DataView(new ArrayBuffer(Float32Array.BYTES_PER_ELEMENT));
	data.setFloat32(0, value, true);
	let bits = data.getUint32(0, true);
	let rounded = data.getFloat32(0, true);
	if (!Number.isFinite(rounded)) {
		throw new RangeError('CFL velocity limit cannot be represented as a finite float32 value.');
	}
	if (rounded > value) {
		bits -= 1;
		data.setUint32(0, bits, true);
		rounded = data.getFloat32(0, true);
	}
	return rounded;
}

function roundUpFloat32(value: number): number {
	if (!Number.isFinite(value) || value < 0) {
		throw new RangeError('CFL wave-speed bounds must be finite and non-negative.');
	}
	if (value === 0) return 0;

	const data = new DataView(new ArrayBuffer(Float32Array.BYTES_PER_ELEMENT));
	data.setFloat32(0, value, true);
	let bits = data.getUint32(0, true);
	let rounded = data.getFloat32(0, true);
	if (!Number.isFinite(rounded)) {
		throw new RangeError('CFL wave-speed bound cannot be represented as a finite float32 value.');
	}
	if (rounded < value) {
		bits += 1;
		data.setUint32(0, bits, true);
		rounded = data.getFloat32(0, true);
	}
	return rounded;
}
