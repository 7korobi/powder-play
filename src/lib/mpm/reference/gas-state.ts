import type { MpmParticle } from './solver.js';

export const ATMOSPHERIC_PRESSURE_PA = 101_325;
export const REFERENCE_GAS_TEMPERATURE_K = 293.15;
export const DRY_AIR_SPECIFIC_GAS_CONSTANT_J_PER_KG_K = 287;

export interface IsothermalGasCompartmentState {
	readonly massKg: number;
	readonly volumeM3: number;
	readonly temperatureK: number;
	readonly specificGasConstantJPerKgK: number;
}

export interface GasCompartmentGeometry {
	readonly innerWidthM: number;
	readonly innerHeightM: number;
	readonly extrusionDepthM: number;
}

export interface IsothermalGasCompartmentConfig {
	readonly volumeM3: number;
	readonly absolutePressurePa?: number;
	readonly temperatureK?: number;
	readonly specificGasConstantJPerKgK?: number;
}

export interface GasCompartmentPressure {
	readonly absolutePressurePa: number;
	readonly gaugePressurePa: number;
}

export interface IsothermalGasCompartmentEvaluation {
	readonly state: IsothermalGasCompartmentState;
	readonly occupiedAreaM2: number;
	readonly pressure: GasCompartmentPressure;
}

export interface MpmParticleOccupiedAreaConfig {
	/** Default reference area in the MPM coordinate system's squared units. */
	readonly defaultParticleArea: number;
	/** Metres represented by one MPM coordinate length unit. */
	readonly coordinateUnitMeters: number;
}

export interface GasPressureBoundaryForce {
	readonly xN: number;
	readonly yN: number;
}

export interface FluidSurfaceOutwardNormal {
	readonly x: number;
	readonly y: number;
}

export interface GasPressureBoundarySegment {
	readonly lengthM: number;
	readonly fluidOutwardNormal: FluidSurfaceOutwardNormal;
}

export interface GasCompartmentBoundaryEvaluation extends IsothermalGasCompartmentEvaluation {
	readonly boundaryForces: readonly GasPressureBoundaryForce[];
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

function assertFiniteNonNegative(
	value: unknown,
	name: string
): asserts value is number {
	assertFiniteNumber(value, name);
	if (value < 0) {
		throw new RangeError(`${name} must be a finite non-negative number.`);
	}
}

function validateGasState(state: unknown): asserts state is IsothermalGasCompartmentState {
	if (!isRecord(state)) throw new RangeError('state must be an object.');

	assertPositiveFinite(state.massKg, 'massKg');
	assertPositiveFinite(state.volumeM3, 'volumeM3');
	assertPositiveFinite(state.temperatureK, 'temperatureK');
	assertPositiveFinite(
		state.specificGasConstantJPerKgK,
		'specificGasConstantJPerKgK'
	);
}

export function calculateGasCompartmentVolume(
	geometry: GasCompartmentGeometry,
	occupiedAreaM2: number
): number {
	if (typeof geometry !== 'object' || geometry === null) {
		throw new RangeError('geometry must be an object.');
	}

	assertPositiveFinite(geometry.innerWidthM, 'innerWidthM');
	assertPositiveFinite(geometry.innerHeightM, 'innerHeightM');
	assertPositiveFinite(geometry.extrusionDepthM, 'extrusionDepthM');
	assertFiniteNonNegative(occupiedAreaM2, 'occupiedAreaM2');

	const innerAreaM2 = geometry.innerWidthM * geometry.innerHeightM;
	assertPositiveFinite(innerAreaM2, 'calculated innerAreaM2');
	if (occupiedAreaM2 > innerAreaM2) {
		throw new RangeError('occupiedAreaM2 must not exceed the chamber interior area.');
	}

	const innerVolumeM3 = innerAreaM2 * geometry.extrusionDepthM;
	assertPositiveFinite(innerVolumeM3, 'calculated innerVolumeM3');
	const gasVolumeM3 = (innerAreaM2 - occupiedAreaM2) * geometry.extrusionDepthM;
	assertFiniteNonNegative(gasVolumeM3, 'calculated gasVolumeM3');
	return gasVolumeM3;
}

/** Sums MPM particle material areas, rather than computing their geometric union. */
export function calculateMpmParticleOccupiedAreaM2(
	particles: readonly Pick<MpmParticle, 'F' | 'volume'>[],
	config: MpmParticleOccupiedAreaConfig
): number {
	if (!Array.isArray(particles)) throw new TypeError('particles must be an array.');
	if (!isRecord(config)) throw new TypeError('config must be an object.');

	assertPositiveFinite(config.defaultParticleArea, 'defaultParticleArea');
	assertPositiveFinite(config.coordinateUnitMeters, 'coordinateUnitMeters');
	const areaScaleM2 = config.coordinateUnitMeters * config.coordinateUnitMeters;
	assertPositiveFinite(areaScaleM2, 'calculated areaScaleM2');

	let occupiedAreaM2 = 0;
	for (let index = 0; index < particles.length; index += 1) {
		const value: unknown = particles[index];
		if (!isRecord(value)) {
			throw new TypeError(`particles[${index}] must be an object.`);
		}
		if (!isRecord(value.F)) {
			throw new TypeError(`particles[${index}].F must be an object.`);
		}

		const matrix = value.F;
		const m00 = matrix.m00;
		const m01 = matrix.m01;
		const m10 = matrix.m10;
		const m11 = matrix.m11;
		assertFiniteNumber(m00, `particles[${index}].F.m00`);
		assertFiniteNumber(m01, `particles[${index}].F.m01`);
		assertFiniteNumber(m10, `particles[${index}].F.m10`);
		assertFiniteNumber(m11, `particles[${index}].F.m11`);

		const determinant = m00 * m11 - m01 * m10;
		assertPositiveFinite(determinant, `particles[${index}].F determinant`);

		const referenceArea =
			value.volume === undefined ? config.defaultParticleArea : value.volume;
		assertPositiveFinite(referenceArea, `particles[${index}].volume`);
		const currentArea = referenceArea * determinant;
		assertPositiveFinite(currentArea, `particles[${index}] current area`);
		const currentAreaM2 = currentArea * areaScaleM2;
		assertPositiveFinite(currentAreaM2, `particles[${index}] current area in m²`);

		occupiedAreaM2 += currentAreaM2;
		assertFiniteNonNegative(occupiedAreaM2, 'total occupied area in m²');
	}

	return occupiedAreaM2;
}

export function createIsothermalGasCompartment(
	config: IsothermalGasCompartmentConfig
): IsothermalGasCompartmentState {
	if (typeof config !== 'object' || config === null) {
		throw new RangeError('config must be an object.');
	}

	const absolutePressurePa =
		config.absolutePressurePa === undefined
			? ATMOSPHERIC_PRESSURE_PA
			: config.absolutePressurePa;
	const temperatureK =
		config.temperatureK === undefined
			? REFERENCE_GAS_TEMPERATURE_K
			: config.temperatureK;
	const specificGasConstantJPerKgK =
		config.specificGasConstantJPerKgK === undefined
			? DRY_AIR_SPECIFIC_GAS_CONSTANT_J_PER_KG_K
			: config.specificGasConstantJPerKgK;

	assertPositiveFinite(config.volumeM3, 'volumeM3');
	assertPositiveFinite(absolutePressurePa, 'absolutePressurePa');
	assertPositiveFinite(temperatureK, 'temperatureK');
	assertPositiveFinite(specificGasConstantJPerKgK, 'specificGasConstantJPerKgK');

	const massKg =
		(absolutePressurePa * config.volumeM3) /
		(specificGasConstantJPerKgK * temperatureK);
	assertPositiveFinite(massKg, 'calculated massKg');

	return {
		massKg,
		volumeM3: config.volumeM3,
		temperatureK,
		specificGasConstantJPerKgK
	};
}

export function initializeIsothermalGasCompartmentFromMpmParticles(
	geometry: GasCompartmentGeometry,
	particles: readonly Pick<MpmParticle, 'F' | 'volume'>[],
	particleAreaConfig: MpmParticleOccupiedAreaConfig,
	gasConfig: Omit<IsothermalGasCompartmentConfig, 'volumeM3'> = {}
): IsothermalGasCompartmentEvaluation {
	const occupiedAreaM2 = calculateMpmParticleOccupiedAreaM2(
		particles,
		particleAreaConfig
	);
	const volumeM3 = calculateGasCompartmentVolume(geometry, occupiedAreaM2);
	const state = createIsothermalGasCompartment({ ...gasConfig, volumeM3 });
	const pressure = calculateIsothermalGasPressure(state);

	return { state, occupiedAreaM2, pressure };
}

export function updateIsothermalGasCompartmentVolume(
	state: IsothermalGasCompartmentState,
	volumeM3: number
): IsothermalGasCompartmentState {
	validateGasState(state);
	assertPositiveFinite(volumeM3, 'volumeM3');
	return { ...state, volumeM3 };
}

export function calculateIsothermalGasPressure(
	state: IsothermalGasCompartmentState,
	referencePressurePa = ATMOSPHERIC_PRESSURE_PA
): GasCompartmentPressure {
	validateGasState(state);
	assertPositiveFinite(referencePressurePa, 'referencePressurePa');

	const absolutePressurePa =
		(state.massKg * state.specificGasConstantJPerKgK * state.temperatureK) /
		state.volumeM3;
	assertPositiveFinite(absolutePressurePa, 'calculated absolutePressurePa');

	return {
		absolutePressurePa,
		gaugePressurePa: absolutePressurePa - referencePressurePa
	};
}

export function calculateGasPressureForceOnFluidBoundary(
	gaugePressurePa: number,
	interfaceLengthM: number,
	extrusionDepthM: number,
	fluidOutwardNormal: FluidSurfaceOutwardNormal
): GasPressureBoundaryForce {
	assertFiniteNumber(gaugePressurePa, 'gaugePressurePa');
	assertPositiveFinite(interfaceLengthM, 'interfaceLengthM');
	assertPositiveFinite(extrusionDepthM, 'extrusionDepthM');
	if (!isRecord(fluidOutwardNormal)) {
		throw new TypeError('fluidOutwardNormal must be an object.');
	}

	const normalX = fluidOutwardNormal.x;
	const normalY = fluidOutwardNormal.y;
	assertFiniteNumber(normalX, 'fluidOutwardNormal.x');
	assertFiniteNumber(normalY, 'fluidOutwardNormal.y');
	const normalLength = Math.hypot(normalX, normalY);
	assertPositiveFinite(normalLength, 'fluidOutwardNormal magnitude');

	const interfaceAreaM2 = interfaceLengthM * extrusionDepthM;
	assertPositiveFinite(interfaceAreaM2, 'calculated interfaceAreaM2');
	const pressureForceN = -gaugePressurePa * interfaceAreaM2;
	assertFiniteNumber(pressureForceN, 'calculated pressureForceN');
	if (pressureForceN === 0) return { xN: 0, yN: 0 };

	const rawXn = (pressureForceN * normalX) / normalLength;
	const rawYn = (pressureForceN * normalY) / normalLength;
	assertFiniteNumber(rawXn, 'calculated pressure force x');
	assertFiniteNumber(rawYn, 'calculated pressure force y');
	return {
		xN: rawXn === 0 ? 0 : rawXn,
		yN: rawYn === 0 ? 0 : rawYn
	};
}

export function evaluateIsothermalGasCompartment(
	state: IsothermalGasCompartmentState,
	geometry: GasCompartmentGeometry,
	particles: readonly Pick<MpmParticle, 'F' | 'volume'>[],
	particleAreaConfig: MpmParticleOccupiedAreaConfig,
	referencePressurePa = ATMOSPHERIC_PRESSURE_PA
): IsothermalGasCompartmentEvaluation {
	const occupiedAreaM2 = calculateMpmParticleOccupiedAreaM2(
		particles,
		particleAreaConfig
	);
	const volumeM3 = calculateGasCompartmentVolume(geometry, occupiedAreaM2);
	const updatedState = updateIsothermalGasCompartmentVolume(state, volumeM3);
	const pressure = calculateIsothermalGasPressure(updatedState, referencePressurePa);

	return {
		state: updatedState,
		occupiedAreaM2,
		pressure
	};
}

export function evaluateIsothermalGasBoundary(
	state: IsothermalGasCompartmentState,
	geometry: GasCompartmentGeometry,
	particles: readonly Pick<MpmParticle, 'F' | 'volume'>[],
	particleAreaConfig: MpmParticleOccupiedAreaConfig,
	boundarySegments: readonly GasPressureBoundarySegment[],
	referencePressurePa = ATMOSPHERIC_PRESSURE_PA
): GasCompartmentBoundaryEvaluation {
	if (!Array.isArray(boundarySegments)) {
		throw new TypeError('boundarySegments must be an array.');
	}

	const evaluation = evaluateIsothermalGasCompartment(
		state,
		geometry,
		particles,
		particleAreaConfig,
		referencePressurePa
	);
	const boundaryForces = boundarySegments.map((segment, index) => {
		if (!isRecord(segment)) {
			throw new TypeError(`boundarySegments[${index}] must be an object.`);
		}
		const lengthM = segment.lengthM;
		assertPositiveFinite(lengthM, `boundarySegments[${index}].lengthM`);
		if (!isRecord(segment.fluidOutwardNormal)) {
			throw new TypeError(
				`boundarySegments[${index}].fluidOutwardNormal must be an object.`
			);
		}
		const normalX = segment.fluidOutwardNormal.x;
		const normalY = segment.fluidOutwardNormal.y;
		assertFiniteNumber(normalX, `boundarySegments[${index}].fluidOutwardNormal.x`);
		assertFiniteNumber(normalY, `boundarySegments[${index}].fluidOutwardNormal.y`);

		return calculateGasPressureForceOnFluidBoundary(
			evaluation.pressure.gaugePressurePa,
			lengthM,
			geometry.extrusionDepthM,
			{ x: normalX, y: normalY }
		);
	});

	return { ...evaluation, boundaryForces };
}
