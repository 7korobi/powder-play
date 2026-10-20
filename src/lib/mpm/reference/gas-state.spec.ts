import { describe, expect, it } from 'vitest';
import {
	ATMOSPHERIC_PRESSURE_PA,
	createIsothermalGasCompartment,
	calculateIsothermalGasPressure,
	calculateGasPressureForceOnFluidBoundary,
	calculateGasCompartmentVolume,
	calculateMpmParticleOccupiedAreaM2,
	DRY_AIR_SPECIFIC_GAS_CONSTANT_J_PER_KG_K,
	evaluateIsothermalGasBoundary,
	evaluateIsothermalGasCompartment,
	initializeIsothermalGasCompartmentFromMpmParticles,
	REFERENCE_GAS_TEMPERATURE_K,
	updateIsothermalGasCompartmentVolume,
	type GasCompartmentGeometry,
	type MpmParticleOccupiedAreaConfig,
	type IsothermalGasCompartmentConfig
} from './gas-state.js';
import { MPM_SI_BOX_DOMAIN } from '../si-domain.js';

const REFERENCE_CHAMBER: GasCompartmentGeometry = MPM_SI_BOX_DOMAIN;
const IDENTITY = { m00: 1, m01: 0, m10: 0, m11: 1 } as const;
const SI_PARTICLE_AREA: MpmParticleOccupiedAreaConfig = {
	defaultParticleArea: 0.02,
	coordinateUnitMeters: 1
};

describe('isothermal ideal-gas compartment reference', () => {
	it('initializes dry air at the reference atmospheric state', () => {
		const compartment = createIsothermalGasCompartment({ volumeM3: 0.0008 });
		const pressure = calculateIsothermalGasPressure(compartment);

		expect(compartment.massKg).toBeCloseTo(
			(ATMOSPHERIC_PRESSURE_PA * compartment.volumeM3) /
				(DRY_AIR_SPECIFIC_GAS_CONSTANT_J_PER_KG_K * REFERENCE_GAS_TEMPERATURE_K),
			12
		);
		expect(compartment.temperatureK).toBe(REFERENCE_GAS_TEMPERATURE_K);
		expect(compartment.specificGasConstantJPerKgK).toBe(
			DRY_AIR_SPECIFIC_GAS_CONSTANT_J_PER_KG_K
		);
		expect(pressure.absolutePressurePa).toBeCloseTo(ATMOSPHERIC_PRESSURE_PA, 8);
		expect(pressure.gaugePressurePa).toBeCloseTo(0, 8);
	});

	it('calculates remaining gas volume from the 200 mm chamber and 20 mm depth', () => {
		const innerAreaM2 =
			REFERENCE_CHAMBER.innerWidthM * REFERENCE_CHAMBER.innerHeightM;

		expect(calculateGasCompartmentVolume(REFERENCE_CHAMBER, 0)).toBeCloseTo(
			0.0008,
			14
		);
		expect(calculateGasCompartmentVolume(REFERENCE_CHAMBER, 0.02)).toBeCloseTo(
			0.0004,
			14
		);
		expect(calculateGasCompartmentVolume(REFERENCE_CHAMBER, innerAreaM2)).toBe(0);
	});

	it('keeps the 50 mm exterior wall outside the viewport-sized fluid interior', () => {
		expect(MPM_SI_BOX_DOMAIN.exteriorWallThicknessM).toBe(0.05);
		expect(Object.values(MPM_SI_BOX_DOMAIN.boundaries)).toEqual([
			'closed',
			'closed',
			'closed',
			'closed'
		]);
		expect(calculateGasCompartmentVolume(MPM_SI_BOX_DOMAIN, 0)).toBeCloseTo(
			0.0008,
			14
		);
	});

	it('keeps mass fixed and makes pressure inversely proportional to volume', () => {
		const compartment = createIsothermalGasCompartment({ volumeM3: 0.0008 });
		const compressed = updateIsothermalGasCompartmentVolume(
			compartment,
			compartment.volumeM3 / 2
		);
		const pressure = calculateIsothermalGasPressure(compressed);

		expect(compressed.massKg).toBe(compartment.massKg);
		expect(compressed.temperatureK).toBe(compartment.temperatureK);
		expect(pressure.absolutePressurePa).toBeCloseTo(
			ATMOSPHERIC_PRESSURE_PA * 2,
			8
		);
		expect(pressure.gaugePressurePa).toBeCloseTo(ATMOSPHERIC_PRESSURE_PA, 8);
	});

	it('links chamber fill volume to ideal-gas compression without changing gas mass', () => {
		const initialParticles = [{ F: IDENTITY, volume: 0.02 }];
		const compressedParticles = [
			{ F: { m00: 1.5, m01: 0, m10: 0, m11: 1 }, volume: 0.02 }
		];
		const initial = initializeIsothermalGasCompartmentFromMpmParticles(
			REFERENCE_CHAMBER,
			initialParticles,
			SI_PARTICLE_AREA
		);
		const compressed = evaluateIsothermalGasCompartment(
			initial.state,
			REFERENCE_CHAMBER,
			compressedParticles,
			SI_PARTICLE_AREA
		);
		const pressure = compressed.pressure;

		expect(initial.occupiedAreaM2).toBeCloseTo(0.02, 14);
		expect(initial.pressure.gaugePressurePa).toBeCloseTo(0, 8);
		expect(compressed.occupiedAreaM2).toBeCloseTo(0.03, 14);
		expect(compressed.state.volumeM3).toBeCloseTo(0.0002, 14);
		expect(compressed.state.massKg).toBe(initial.state.massKg);
		expect(pressure.absolutePressurePa).toBeCloseTo(
			ATMOSPHERIC_PRESSURE_PA * 2,
			8
		);
	});

	it('evaluates the sealed-gas pressure and force on known interface segments together', () => {
		const initial = initializeIsothermalGasCompartmentFromMpmParticles(
			REFERENCE_CHAMBER,
			[{ F: IDENTITY, volume: 0.02 }],
			SI_PARTICLE_AREA
		);
		const evaluation = evaluateIsothermalGasBoundary(
			initial.state,
			REFERENCE_CHAMBER,
			[{ F: { m00: 1.5, m01: 0, m10: 0, m11: 1 }, volume: 0.02 }],
			SI_PARTICLE_AREA,
			[
				{ lengthM: 0.2, fluidOutwardNormal: { x: 0, y: -1 } },
				{ lengthM: 0.1, fluidOutwardNormal: { x: 1, y: 0 } }
			]
		);

		expect(evaluation.state.massKg).toBe(initial.state.massKg);
		expect(evaluation.pressure.gaugePressurePa).toBeCloseTo(
			ATMOSPHERIC_PRESSURE_PA,
			8
		);
		expect(evaluation.boundaryForces).toHaveLength(2);
		expect(evaluation.boundaryForces[0]?.xN).toBe(0);
		expect(evaluation.boundaryForces[0]?.yN).toBeCloseTo(405.3, 8);
		expect(evaluation.boundaryForces[1]?.xN).toBeCloseTo(-202.65, 8);
		expect(evaluation.boundaryForces[1]?.yN).toBe(0);
	});

	it('rejects an evaluation that leaves no positive gas volume', () => {
		const initial = createIsothermalGasCompartment({ volumeM3: 0.0008 });
		const fullyOccupiedAreaM2 =
			REFERENCE_CHAMBER.innerWidthM * REFERENCE_CHAMBER.innerHeightM;

		expect(() =>
			evaluateIsothermalGasCompartment(
				initial,
				REFERENCE_CHAMBER,
				[{ F: IDENTITY, volume: fullyOccupiedAreaM2 }],
				SI_PARTICLE_AREA
			)
		).toThrow(/volumeM3/);
	});

	it('preserves gas state when particle sampling is empty only when input occupancy is absent', () => {
		const initial = createIsothermalGasCompartment({ volumeM3: 0.0008 });
		const evaluation = evaluateIsothermalGasCompartment(
			initial,
			REFERENCE_CHAMBER,
			[],
			SI_PARTICLE_AREA
		);

		expect(evaluation.occupiedAreaM2).toBe(0);
		expect(evaluation.state.volumeM3).toBeCloseTo(0.0008, 14);
		expect(evaluation.state.massKg).toBe(initial.massKg);
		expect(evaluation.pressure.absolutePressurePa).toBeCloseTo(
			ATMOSPHERIC_PRESSURE_PA,
			8
		);
	});

	it('sums deformed particle area and converts the coordinate scale to square metres', () => {
		const occupiedAreaM2 = calculateMpmParticleOccupiedAreaM2(
			[
				{ F: IDENTITY, volume: 1.5 },
				{ F: { m00: -2, m01: 0, m10: 0, m11: -1 } }
			],
			{ defaultParticleArea: 0.5, coordinateUnitMeters: 0.1 }
		);

		expect(occupiedAreaM2).toBeCloseTo(0.025, 14);
		expect(calculateMpmParticleOccupiedAreaM2([], SI_PARTICLE_AREA)).toBe(0);
	});

	it('calculates gauge pressure against the supplied reference pressure', () => {
		const compartment = createIsothermalGasCompartment({
			volumeM3: 0.5,
			absolutePressurePa: 200_000,
			temperatureK: 310,
			specificGasConstantJPerKgK: 300
		});
		const pressure = calculateIsothermalGasPressure(compartment, 100_000);

		expect(pressure.absolutePressurePa).toBeCloseTo(200_000, 8);
		expect(pressure.gaugePressurePa).toBeCloseTo(100_000, 8);
	});

	it('converts gauge pressure to force on a 2D fluid boundary segment', () => {
		const compressed = calculateGasPressureForceOnFluidBoundary(
			100,
			0.5,
			0.02,
			{ x: 3, y: 4 }
		);
		const ambient = calculateGasPressureForceOnFluidBoundary(
			0,
			0.5,
			0.02,
			{ x: 0, y: 1 }
		);
		const subatmospheric = calculateGasPressureForceOnFluidBoundary(
			-100,
			0.5,
			0.02,
			{ x: 0, y: 1 }
		);

		expect(compressed.xN).toBeCloseTo(-0.6, 14);
		expect(compressed.yN).toBeCloseTo(-0.8, 14);
		expect(ambient).toEqual({ xN: 0, yN: 0 });
		expect(subatmospheric).toEqual({ xN: 0, yN: 1 });
	});

	it('rejects non-finite or non-positive initial conditions', () => {
		const expectInvalid = (overrides: Partial<IsothermalGasCompartmentConfig>) => {
			expect(() =>
				createIsothermalGasCompartment({ volumeM3: 1, ...overrides })
			).toThrow(RangeError);
		};

		for (const invalid of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
			expectInvalid({ volumeM3: invalid });
			expectInvalid({ absolutePressurePa: invalid });
			expectInvalid({ temperatureK: invalid });
			expectInvalid({ specificGasConstantJPerKgK: invalid });
		}
	});

	it('rejects explicit null values instead of replacing them with defaults', () => {
		const config = {
			volumeM3: 1,
			absolutePressurePa: null
		} as unknown as IsothermalGasCompartmentConfig;

		expect(() => createIsothermalGasCompartment(config)).toThrow(/absolutePressurePa/);
	});

	it('rejects invalid chamber geometry, negative occupancy, and overfill', () => {
		for (const invalid of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
			expect(() =>
				calculateGasCompartmentVolume(
					{ ...REFERENCE_CHAMBER, innerWidthM: invalid },
					0
				)
			).toThrow(RangeError);
			expect(() =>
				calculateGasCompartmentVolume(
					{ ...REFERENCE_CHAMBER, innerHeightM: invalid },
					0
				)
			).toThrow(RangeError);
			expect(() =>
				calculateGasCompartmentVolume(
					{ ...REFERENCE_CHAMBER, extrusionDepthM: invalid },
					0
				)
			).toThrow(RangeError);
		}
		for (const invalid of [-1, Number.NaN, Number.POSITIVE_INFINITY]) {
			expect(() => calculateGasCompartmentVolume(REFERENCE_CHAMBER, invalid)).toThrow(
				RangeError
			);
		}
		expect(() => calculateGasCompartmentVolume(REFERENCE_CHAMBER, 0.0401)).toThrow(
			/occupiedAreaM2/
		);
	});

	it('rejects invalid particle areas, deformation, and unit scales', () => {
		const expectInvalidParticle = (particle: unknown) => {
			expect(() =>
				calculateMpmParticleOccupiedAreaM2(
					[particle] as Parameters<typeof calculateMpmParticleOccupiedAreaM2>[0],
					SI_PARTICLE_AREA
				)
			).toThrow(RangeError);
		};

		for (const invalid of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
			expectInvalidParticle({ F: IDENTITY, volume: invalid });
			expect(() =>
				calculateMpmParticleOccupiedAreaM2([], {
					...SI_PARTICLE_AREA,
					defaultParticleArea: invalid
				})
			).toThrow(RangeError);
			expect(() =>
				calculateMpmParticleOccupiedAreaM2([], {
					...SI_PARTICLE_AREA,
					coordinateUnitMeters: invalid
				})
			).toThrow(RangeError);
		}
		expectInvalidParticle({ F: { m00: 1, m01: 0, m10: 0, m11: -1 } });
		expectInvalidParticle({ F: { m00: 1, m01: 0, m10: 0, m11: 0 } });
		expectInvalidParticle({ F: { m00: Number.NaN, m01: 0, m10: 0, m11: 1 } });
	});

	it('rejects non-positive volume updates', () => {
		const compartment = createIsothermalGasCompartment({ volumeM3: 0.0008 });

		for (const invalid of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
			expect(() => updateIsothermalGasCompartmentVolume(compartment, invalid)).toThrow(
				RangeError
			);
		}
	});

	it('rejects invalid interface geometry and normals', () => {
		for (const invalid of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
			expect(() =>
				calculateGasPressureForceOnFluidBoundary(1, invalid, 1, { x: 1, y: 0 })
			).toThrow(RangeError);
			expect(() =>
				calculateGasPressureForceOnFluidBoundary(1, 1, invalid, { x: 1, y: 0 })
			).toThrow(RangeError);
		}
		for (const invalid of [Number.NaN, Number.POSITIVE_INFINITY]) {
			expect(() =>
				calculateGasPressureForceOnFluidBoundary(invalid, 1, 1, { x: 1, y: 0 })
			).toThrow(RangeError);
		}
		expect(() =>
			calculateGasPressureForceOnFluidBoundary(1, 1, 1, { x: 0, y: 0 })
		).toThrow(/magnitude/);
		expect(() =>
			calculateGasPressureForceOnFluidBoundary(1, 1, 1, {
				x: Number.NaN,
				y: 1
			})
		).toThrow(/fluidOutwardNormal.x/);
	});

	it('rejects malformed gas boundary segment input', () => {
		const state = createIsothermalGasCompartment({ volumeM3: 0.0008 });
		const particles = [{ F: IDENTITY, volume: 0.02 }];
		const expectInvalidSegments = (segments: unknown) => {
			expect(() =>
				evaluateIsothermalGasBoundary(
					state,
					REFERENCE_CHAMBER,
					particles,
					SI_PARTICLE_AREA,
					segments as Parameters<typeof evaluateIsothermalGasBoundary>[4]
				)
			).toThrow();
		};

		expectInvalidSegments(null);
		expectInvalidSegments([null]);
		expectInvalidSegments([{ lengthM: 0, fluidOutwardNormal: { x: 1, y: 0 } }]);
		expectInvalidSegments([{ lengthM: 1, fluidOutwardNormal: { x: 0, y: 0 } }]);
	});

	it('rejects invalid states and reference pressures', () => {
		const compartment = createIsothermalGasCompartment({ volumeM3: 0.0008 });

		for (const invalid of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
			expect(() =>
				calculateIsothermalGasPressure({ ...compartment, massKg: invalid })
			).toThrow(RangeError);
			expect(() =>
				calculateIsothermalGasPressure({ ...compartment, volumeM3: invalid })
			).toThrow(RangeError);
			expect(() =>
				calculateIsothermalGasPressure({ ...compartment, temperatureK: invalid })
			).toThrow(RangeError);
			expect(() =>
				calculateIsothermalGasPressure({
					...compartment,
					specificGasConstantJPerKgK: invalid
				})
			).toThrow(RangeError);
			expect(() =>
				calculateIsothermalGasPressure(compartment, invalid)
			).toThrow(RangeError);
		}
	});
});
