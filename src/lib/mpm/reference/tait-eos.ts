import { ATMOSPHERIC_PRESSURE_PA } from './gas-state.js';
import type { MpmMaterialParameters } from './solver.js';

export interface TaitEosParameters {
	readonly referenceDensityKgPerM3: number;
	readonly exponent: number;
	readonly pressureCoefficientPa: number;
	readonly ambientPressurePa?: number;
}

export interface TaitEosEvaluation {
	readonly densityKgPerM3: number;
	readonly gaugePressurePa: number;
	readonly absolutePressurePa: number;
	readonly bulkModulusPa: number;
	readonly soundSpeedMPerS: number;
	readonly referenceSoundSpeedMPerS: number;
}

export interface TaitFluidStressEvaluation {
	readonly volumeRatio: number;
	readonly pressure: TaitEosEvaluation;
	readonly stressTimesFtIsotropicPa: number;
}

export function createMpmTaitLiquidMaterial(
	parameters: TaitEosParameters,
	extrusionDepthM: number,
	model: 'water' | 'oil' = 'water'
): Extract<MpmMaterialParameters, { readonly model: 'water' | 'oil' }> {
	validateParameters(parameters);
	assertPositiveFinite(extrusionDepthM, 'extrusionDepthM');
	const density = parameters.referenceDensityKgPerM3 * extrusionDepthM;
	const lambda = parameters.pressureCoefficientPa * extrusionDepthM;
	assertPositiveFinite(density, 'calculated areal reference density');
	assertPositiveFinite(lambda, 'calculated areal pressure coefficient');
	return {
		model,
		lambda,
		mu: 0,
		density,
		equationOfState: { model: 'tait', exponent: parameters.exponent }
	};
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function assertFiniteNumber(value: unknown, name: string): asserts value is number {
	if (typeof value !== 'number' || !Number.isFinite(value)) {
		throw new RangeError(`${name} must be a finite number.`);
	}
}

function assertPositiveFinite(value: unknown, name: string): asserts value is number {
	assertFiniteNumber(value, name);
	if (value <= 0) {
		throw new RangeError(`${name} must be a finite positive number.`);
	}
}

function validateParameters(value: unknown): asserts value is TaitEosParameters {
	if (!isRecord(value)) throw new TypeError('parameters must be an object.');
	assertPositiveFinite(value.referenceDensityKgPerM3, 'referenceDensityKgPerM3');
	assertPositiveFinite(value.exponent, 'exponent');
	assertPositiveFinite(value.pressureCoefficientPa, 'pressureCoefficientPa');
	if (value.ambientPressurePa !== undefined) {
		assertPositiveFinite(value.ambientPressurePa, 'ambientPressurePa');
	}
}

export function deriveTaitPressureCoefficientPa(
	referenceDensityKgPerM3: number,
	exponent: number,
	soundSpeedMPerS: number
): number {
	assertPositiveFinite(referenceDensityKgPerM3, 'referenceDensityKgPerM3');
	assertPositiveFinite(exponent, 'exponent');
	assertPositiveFinite(soundSpeedMPerS, 'soundSpeedMPerS');

	const pressureCoefficientPa =
		(referenceDensityKgPerM3 * soundSpeedMPerS * soundSpeedMPerS) / exponent;
	assertPositiveFinite(pressureCoefficientPa, 'calculated pressureCoefficientPa');
	return pressureCoefficientPa;
}

export function evaluateTaitEos(
	densityKgPerM3: number,
	parameters: TaitEosParameters
): TaitEosEvaluation {
	assertPositiveFinite(densityKgPerM3, 'densityKgPerM3');
	validateParameters(parameters);

	const ambientPressurePa =
		parameters.ambientPressurePa === undefined
			? ATMOSPHERIC_PRESSURE_PA
			: parameters.ambientPressurePa;
	const densityRatio = densityKgPerM3 / parameters.referenceDensityKgPerM3;
	assertPositiveFinite(densityRatio, 'calculated densityRatio');

	const densityRatioPower = densityRatio ** parameters.exponent;
	assertPositiveFinite(densityRatioPower, 'calculated density ratio power');

	const referenceBulkModulusPa =
		parameters.exponent * parameters.pressureCoefficientPa;
	assertPositiveFinite(referenceBulkModulusPa, 'calculated referenceBulkModulusPa');
	const referenceSoundSpeedMPerS = Math.sqrt(
		referenceBulkModulusPa / parameters.referenceDensityKgPerM3
	);
	assertPositiveFinite(
		referenceSoundSpeedMPerS,
		'calculated referenceSoundSpeedMPerS'
	);

	const gaugePressurePa =
		parameters.pressureCoefficientPa * (densityRatioPower - 1);
	assertFiniteNumber(gaugePressurePa, 'calculated gaugePressurePa');
	const absolutePressurePa = ambientPressurePa + gaugePressurePa;
	assertFiniteNumber(absolutePressurePa, 'calculated absolutePressurePa');
	if (absolutePressurePa <= 0) {
		throw new RangeError(
			'calculated absolutePressurePa must be positive; low-pressure or cavitation handling is required.'
		);
	}

	const bulkModulusPa = referenceBulkModulusPa * densityRatioPower;
	assertPositiveFinite(bulkModulusPa, 'calculated bulkModulusPa');
	const soundSpeedMPerS = Math.sqrt(
		(bulkModulusPa / densityKgPerM3)
	);
	assertPositiveFinite(soundSpeedMPerS, 'calculated soundSpeedMPerS');

	return {
		densityKgPerM3,
		gaugePressurePa,
		absolutePressurePa,
		bulkModulusPa,
		soundSpeedMPerS,
		referenceSoundSpeedMPerS
	};
}

export function evaluateTaitFluidStressFromVolumeRatio(
	volumeRatio: number,
	parameters: TaitEosParameters
): TaitFluidStressEvaluation {
	assertPositiveFinite(volumeRatio, 'volumeRatio');
	validateParameters(parameters);

	const densityKgPerM3 = parameters.referenceDensityKgPerM3 / volumeRatio;
	assertPositiveFinite(densityKgPerM3, 'calculated densityKgPerM3');
	const pressure = evaluateTaitEos(densityKgPerM3, parameters);
	const rawStressTimesFtIsotropicPa = -volumeRatio * pressure.gaugePressurePa;
	assertFiniteNumber(
		rawStressTimesFtIsotropicPa,
		'calculated stressTimesFtIsotropicPa'
	);
	const stressTimesFtIsotropicPa =
		rawStressTimesFtIsotropicPa === 0 ? 0 : rawStressTimesFtIsotropicPa;

	return { volumeRatio, pressure, stressTimesFtIsotropicPa };
}
