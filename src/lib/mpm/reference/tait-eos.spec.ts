import { describe, expect, it } from 'vitest';
import { ATMOSPHERIC_PRESSURE_PA } from './gas-state.js';
import {
	createMpmTaitLiquidMaterial,
	deriveTaitPressureCoefficientPa,
	evaluateTaitEos,
	evaluateTaitFluidStressFromVolumeRatio,
	type TaitEosParameters
} from './tait-eos.js';

const WATER_REFERENCE = {
	referenceDensityKgPerM3: 998.2,
	exponent: 7.15
} as const;

describe('Tait equation-of-state reference', () => {
	it('derives the pressure coefficient from the reference sound speed', () => {
		const pressureCoefficientPa = deriveTaitPressureCoefficientPa(
			WATER_REFERENCE.referenceDensityKgPerM3,
			WATER_REFERENCE.exponent,
			1_482
		);

		expect(pressureCoefficientPa).toBeCloseTo(306_625_260, -4);

		const evaluation = evaluateTaitEos(WATER_REFERENCE.referenceDensityKgPerM3, {
			...WATER_REFERENCE,
			pressureCoefficientPa
		});
		expect(evaluation.gaugePressurePa).toBe(0);
		expect(evaluation.absolutePressurePa).toBe(ATMOSPHERIC_PRESSURE_PA);
		expect(evaluation.referenceSoundSpeedMPerS).toBeCloseTo(1_482, 10);
		expect(evaluation.soundSpeedMPerS).toBeCloseTo(1_482, 10);
		expect(evaluation.bulkModulusPa).toBeCloseTo(
			WATER_REFERENCE.exponent * pressureCoefficientPa,
			8
		);
	});

	it('evaluates compression and expansion pressure without clamping tension', () => {
		const pressureCoefficientPa = deriveTaitPressureCoefficientPa(
			WATER_REFERENCE.referenceDensityKgPerM3,
			WATER_REFERENCE.exponent,
			20
		);
		const parameters: TaitEosParameters = {
			...WATER_REFERENCE,
			pressureCoefficientPa
		};
		const compression = evaluateTaitEos(
			WATER_REFERENCE.referenceDensityKgPerM3 * 1.01,
			parameters
		);
		const expansion = evaluateTaitEos(
			WATER_REFERENCE.referenceDensityKgPerM3 * 0.99,
			parameters
		);

		expect(compression.gaugePressurePa).toBeGreaterThan(0);
		expect(compression.absolutePressurePa).toBeGreaterThan(ATMOSPHERIC_PRESSURE_PA);
		expect(compression.soundSpeedMPerS).toBeGreaterThan(20);
		expect(expansion.gaugePressurePa).toBeLessThan(0);
		expect(expansion.gaugePressurePa).not.toBe(0);
		expect(expansion.absolutePressurePa).toBeGreaterThan(0);
		expect(expansion.soundSpeedMPerS).toBeLessThan(20);
	});

	it('rejects a non-positive absolute pressure instead of clamping it', () => {
		const pressureCoefficientPa = deriveTaitPressureCoefficientPa(
			WATER_REFERENCE.referenceDensityKgPerM3,
			WATER_REFERENCE.exponent,
			1_482
		);

		expect(() =>
			evaluateTaitEos(
				WATER_REFERENCE.referenceDensityKgPerM3 * 0.9,
				{ ...WATER_REFERENCE, pressureCoefficientPa }
			)
		).toThrow(/low-pressure or cavitation handling is required/);
	});

	it('maps Tait pressure to the isotropic MPM P times F-transpose term', () => {
		const pressureCoefficientPa = deriveTaitPressureCoefficientPa(
			WATER_REFERENCE.referenceDensityKgPerM3,
			WATER_REFERENCE.exponent,
			20
		);
		const parameters = { ...WATER_REFERENCE, pressureCoefficientPa };
		const equilibrium = evaluateTaitFluidStressFromVolumeRatio(1, parameters);
		const compressed = evaluateTaitFluidStressFromVolumeRatio(0.99, parameters);
		const expanded = evaluateTaitFluidStressFromVolumeRatio(1.01, parameters);

		expect(equilibrium.pressure.gaugePressurePa).toBe(0);
		expect(equilibrium.stressTimesFtIsotropicPa).toBe(0);
		expect(compressed.pressure.gaugePressurePa).toBeGreaterThan(0);
		expect(compressed.stressTimesFtIsotropicPa).toBeCloseTo(
			-0.99 * compressed.pressure.gaugePressurePa,
			10
		);
		expect(compressed.stressTimesFtIsotropicPa).toBeLessThan(0);
		expect(expanded.pressure.gaugePressurePa).toBeLessThan(0);
		expect(expanded.stressTimesFtIsotropicPa).toBeGreaterThan(0);
	});

	it('converts SI Tait parameters to consistent 2D areal material values', () => {
		const pressureCoefficientPa = deriveTaitPressureCoefficientPa(
			WATER_REFERENCE.referenceDensityKgPerM3,
			WATER_REFERENCE.exponent,
			20
		);
		const material = createMpmTaitLiquidMaterial(
			{ ...WATER_REFERENCE, pressureCoefficientPa },
			0.02
		);

		expect(material).toEqual({
			model: 'water',
			lambda: pressureCoefficientPa * 0.02,
			mu: 0,
			density: WATER_REFERENCE.referenceDensityKgPerM3 * 0.02,
			equationOfState: { model: 'tait', exponent: WATER_REFERENCE.exponent }
		});
	});

	it('supports an explicit non-atmospheric reference pressure', () => {
		const pressureCoefficientPa = deriveTaitPressureCoefficientPa(1_000, 7, 20);
		const evaluation = evaluateTaitEos(1_010, {
			referenceDensityKgPerM3: 1_000,
			exponent: 7,
			pressureCoefficientPa,
			ambientPressurePa: 90_000
		});

		expect(evaluation.absolutePressurePa).toBeCloseTo(
			90_000 + pressureCoefficientPa * (1.01 ** 7 - 1),
			8
		);
	});

	it('rejects invalid density, parameters, and derived values', () => {
		const validParameters: TaitEosParameters = {
			...WATER_REFERENCE,
			pressureCoefficientPa: 55_000
		};

		for (const invalid of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
			expect(() => evaluateTaitEos(invalid, validParameters)).toThrow(RangeError);
			expect(() =>
				deriveTaitPressureCoefficientPa(invalid, 7.15, 20)
			).toThrow(RangeError);
			expect(() =>
				deriveTaitPressureCoefficientPa(1_000, invalid, 20)
			).toThrow(RangeError);
			expect(() =>
				deriveTaitPressureCoefficientPa(1_000, 7, invalid)
			).toThrow(RangeError);
		}

		expect(() => evaluateTaitEos(1_000, null as unknown as TaitEosParameters)).toThrow(
			/parameters must be an object/
		);
		expect(() =>
			evaluateTaitEos(1_000, {
				...validParameters,
				pressureCoefficientPa: Number.MAX_VALUE
			})
		).toThrow(/referenceBulkModulusPa/);
	});
});
