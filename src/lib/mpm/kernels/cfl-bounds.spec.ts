import { describe, expect, it } from 'vitest';
import {
	createMpmGate4Scene,
	GATE4_GRID,
	GATE4_PARTICLE_AREA,
	GATE4_SOLID_PARTICLE_MASS
} from '../scenes/gate4.js';
import {
	calculateMpmCflVelocityBound,
	deriveMpmMlsGpuCflBounds,
	type MpmCflVelocityEnvelopeInput
} from './cfl-bounds.js';
import { calculateMpmCflStepBatch } from './step-scheduler.js';

const gate4Config = {
	grid: GATE4_GRID,
	particleMass: GATE4_SOLID_PARTICLE_MASS,
	particleVolume: GATE4_PARTICLE_AREA,
	material: { lambda: 20, mu: 10, model: 'neo-hookean' as const }
};

describe('GPU MPM CFL bound derivation', () => {
	it('derives wave and velocity bounds from the actual seeded Gate 4 state', () => {
		const scene = createMpmGate4Scene(0x4c414234);
		const bounds = deriveMpmMlsGpuCflBounds(scene.particles, gate4Config);
		const actualInitialSpeed = Math.max(
			...scene.particles.map((particle) =>
				Math.hypot(
					Math.fround(particle.velocity.x),
					Math.fround(particle.velocity.y)
				)
			)
		);

		expect(bounds).not.toBeNull();
		expect(bounds?.waveSpeedBound).toBeGreaterThanOrEqual(Math.sqrt(160_000 / 0.02));
		expect(bounds?.waveSpeedBound).toBeCloseTo(Math.sqrt(160_000 / 0.02), 2);
		expect(bounds?.velocityBound).toBeGreaterThanOrEqual(actualInitialSpeed);
		expect(bounds?.velocityBound).toBeLessThan(actualInitialSpeed + 0.01);
	});

	it('derives Neo-Hookean acoustic bounds from the current deformation tensor', () => {
		const particle = {
			position: { x: 1, y: 1 },
			velocity: { x: 0, y: 0 },
			C: { m00: 0, m01: 0, m10: 0, m11: 0 },
			F: { m00: 1, m01: 0, m10: 0, m11: 1 }
		};
		const reference = deriveMpmMlsGpuCflBounds([particle], gate4Config);
		const stretched = deriveMpmMlsGpuCflBounds(
			[{ ...particle, F: { m00: 2, m01: 0, m10: 0, m11: 1 } }],
			gate4Config
		);

		expect(reference?.waveSpeedBound).toBeCloseTo(Math.sqrt(40 / 0.04), 4);
		expect(stretched?.waveSpeedBound).toBeGreaterThan(reference!.waveSpeedBound);
	});

	it('derives Tait sound speed from the exponent and current compression', () => {
		const material = {
			lambda: ((998.2 * 20 * 20) / 7.15) * 0.02,
			mu: 0 as const,
			model: 'water' as const,
			density: 998.2 * 0.02,
			equationOfState: { model: 'tait' as const, exponent: 7.15 }
		};
		const particle = {
			position: { x: 1, y: 1 },
			velocity: { x: 0, y: 0 },
			C: { m00: 0, m01: 0, m10: 0, m11: 0 },
			F: { m00: 1, m01: 0, m10: 0, m11: 1 },
			material
		};
		const reference = deriveMpmMlsGpuCflBounds([particle], gate4Config);
		const compressed = deriveMpmMlsGpuCflBounds(
			[{ ...particle, F: { m00: 0.9, m01: 0, m10: 0, m11: 1 } }],
			gate4Config
		);

		expect(reference?.waveSpeedBound).toBeCloseTo(20, 2);
		expect(compressed?.waveSpeedBound).toBeGreaterThan(reference!.waveSpeedBound);
	});

	it('returns null rather than guessing a bound for unsupported constitutive models', () => {
		const unsupported = {
			...gate4Config,
			material: {
				lambda: 4,
				mu: 2,
				model: 'powder' as const,
				frictionCoefficient: 0.5
			}
		};

		expect(deriveMpmMlsGpuCflBounds([], unsupported)).toEqual({
			waveSpeedBound: 0,
			velocityBound: 0
		});
		expect(
			deriveMpmMlsGpuCflBounds(
				[
					{
						position: { x: 0, y: 0 },
						velocity: { x: 0, y: 0 },
						C: { m00: 0, m01: 0, m10: 0, m11: 0 },
						F: { m00: 1, m01: 0, m10: 0, m11: 1 }
					}
				],
				unsupported
			)
		).toBeNull();
	});

	it('rejects invalid state/material inputs instead of deriving a success-shaped bound', () => {
		const particle = {
			position: { x: 0, y: 0 },
			velocity: { x: 0, y: 0 },
			C: { m00: 0, m01: 0, m10: 0, m11: 0 },
			F: { m00: 1, m01: 0, m10: 0, m11: -1 }
		};

		expect(() => deriveMpmMlsGpuCflBounds([particle], gate4Config)).toThrow(
			/positive density and determinant/
		);
		expect(() =>
			deriveMpmMlsGpuCflBounds(
				[
					{
						...particle,
						F: { m00: 1, m01: 0, m10: 0, m11: 1 },
						velocity: { x: Number.POSITIVE_INFINITY, y: 0 }
					}
				],
				gate4Config
			)
		).toThrow(/non-finite initial velocity/);
	});
});

describe('GPU MPM CFL velocity envelope', () => {
	const baseInput: MpmCflVelocityEnvelopeInput = {
		currentVelocityBound: 12,
		durationSeconds: 0.25,
		gravityMagnitude: 3,
		oscillationAmplitude: 4,
		dragRadius: 2,
		dragStiffness: 5,
		dragDamping: 8
	};

	it('includes gravity, oscillation and bounded pointer spring over the interval', () => {
		const accelerationBound = Math.hypot(3, 4) + 2 * 5;
		const expected = 12 + accelerationBound * 0.25;

		expect(calculateMpmCflVelocityBound(baseInput)).toBeGreaterThanOrEqual(expected);
		expect(calculateMpmCflVelocityBound(baseInput)).toBeLessThan(expected + 0.01);
		expect(
			calculateMpmCflVelocityBound({ ...baseInput, dragDamping: 0 })
		).toBe(calculateMpmCflVelocityBound(baseInput));
	});

	it('preserves the current maximum for a zero-duration batch and rejects invalid inputs', () => {
		const zeroDuration = calculateMpmCflVelocityBound({
			...baseInput,
			durationSeconds: 0
		});
		expect(zeroDuration).toBeGreaterThanOrEqual(baseInput.currentVelocityBound);

		for (const invalid of [
			{ currentVelocityBound: -1 },
			{ durationSeconds: Number.NaN },
			{ gravityMagnitude: Number.POSITIVE_INFINITY },
			{ oscillationAmplitude: -1 },
			{ dragRadius: Number.NaN },
			{ dragStiffness: -1 },
			{ dragDamping: Number.POSITIVE_INFINITY }
		]) {
			expect(() =>
				calculateMpmCflVelocityBound({ ...baseInput, ...invalid })
			).toThrow(RangeError);
		}
	});

	it('plans a bounded default 1x Gate 4 frame without discarding macro time', () => {
		const scene = createMpmGate4Scene(0x4c414234);
		const bounds = deriveMpmMlsGpuCflBounds(scene.particles, gate4Config);
		expect(bounds).not.toBeNull();

		const frameInput = {
			accumulatorSeconds: 0,
			elapsedSeconds: 1 / 60,
			simulationSpeed: 1,
			fixedStepSeconds: 1 / 240,
			cellSize: GATE4_GRID.cellSize,
			waveSpeedBound: bounds!.waveSpeedBound,
			currentVelocityBound: bounds!.velocityBound,
			gravityMagnitude: 90,
			oscillationAmplitude: 240,
			dragRadius: 0,
			dragStiffness: 20,
			dragDamping: 0,
			maxMicrosteps: 512
		};
		const frame = calculateMpmCflStepBatch(frameInput);

		expect(frame.batch.macroStepCount).toBe(4);
		expect(frame.batch.microStepCount).toBe(
			frame.batch.macroStepCount * frame.batch.microstepPlan.stepCount
		);
		expect(frame.batch.microStepCount).toBeLessThanOrEqual(512);
		expect(frame.batch.simulatedSeconds).toBeCloseTo(1 / 60);
		expect(frame.velocityBound).toBeGreaterThan(bounds!.velocityBound);
		expect(() =>
			calculateMpmCflStepBatch({
				...frameInput,
				maxMicrosteps: frame.batch.microStepCount - 1
			})
		).toThrow(/requested microstep count .* exceeds maxMicrosteps/);
	});

	it('rejects pointer drag plans whose float32 microstep could reverse grid velocity', () => {
		expect(() =>
			calculateMpmCflStepBatch({
				accumulatorSeconds: 0,
				elapsedSeconds: 1,
				simulationSpeed: 1,
				fixedStepSeconds: 1,
				cellSize: 1,
				waveSpeedBound: 0,
				currentVelocityBound: 0,
				gravityMagnitude: 0,
				oscillationAmplitude: 0,
				dragRadius: 1,
				dragStiffness: 0,
				dragDamping: 2,
				maxMicrosteps: 1
			})
		).toThrow(/explicit drag damping update to remain dissipative/);
	});
});
