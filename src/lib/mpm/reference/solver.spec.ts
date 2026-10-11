import { describe, expect, it } from 'vitest';
import { stepMpm, type MpmMatrix2, type MpmParticle, type MpmStepConfig } from './solver.js';

const ZERO_MATRIX: MpmMatrix2 = { m00: 0, m01: 0, m10: 0, m11: 0 };
const IDENTITY: MpmMatrix2 = { m00: 1, m01: 0, m10: 0, m11: 1 };

function makeConfig(overrides: Partial<MpmStepConfig> = {}): MpmStepConfig {
	return {
		dt: 0.001,
		grid: { cellsX: 16, cellsY: 16, cellSize: 1 },
		particleMass: 0.1,
		particleVolume: 0.01,
		material: { lambda: 20, mu: 10 },
		gravity: { x: 0, y: 0 },
		...overrides
	};
}

function makeParticle(overrides: Partial<MpmParticle> = {}): MpmParticle {
	return {
		position: { x: 8, y: 8 },
		velocity: { x: 0, y: 0 },
		C: ZERO_MATRIX,
		F: IDENTITY,
		...overrides
	};
}

function closeEnough(actual: number, expected: number, tolerance: number): void {
	expect(Math.abs(actual - expected)).toBeLessThanOrEqual(tolerance);
}

describe('CPU MLS-MPM reference', () => {
	it('returns zero diagnostics for an empty particle set', () => {
		const result = stepMpm([], makeConfig());

		expect(result.particles).toEqual([]);
		expect(result.diagnostics).toEqual({
			inputMass: 0,
			gridMass: 0,
			outputMass: 0,
			inputMomentum: { x: 0, y: 0 },
			gridMomentum: { x: 0, y: 0 },
			outputMomentum: { x: 0, y: 0 }
		});
	});

	it('rejects invalid or omitted step parameters and particle state', () => {
		expect(() => stepMpm([], makeConfig({ dt: 0 }))).toThrow(/dt/);
		expect(() =>
			stepMpm([], { ...makeConfig(), dt: undefined } as unknown as MpmStepConfig)
		).toThrow(/dt/);
		expect(() =>
			stepMpm([], makeConfig({ grid: { cellsX: 0, cellsY: 16, cellSize: 1 } }))
		).toThrow(/cellsX/);
		expect(() =>
			stepMpm([makeParticle({ position: { x: -0.01, y: 8 } })], makeConfig())
		).toThrow(/grid domain/);
		expect(() =>
			stepMpm(
				[makeParticle({ F: { m00: -1, m01: 0, m10: 0, m11: 1 } })],
				makeConfig()
			)
		).toThrow(/determinant/);
	});

	it('keeps a valid deformed state finite during an explicit step', () => {
		const result = stepMpm(
			[
				makeParticle({
					position: { x: 8.25, y: 7.75 },
					velocity: { x: 0.2, y: -0.3 },
					C: { m00: 0.01, m01: 0.02, m10: -0.01, m11: 0.005 },
					F: { m00: 1.02, m01: 0.01, m10: 0, m11: 0.99 }
				})
			],
			makeConfig({ dt: 0.0001, gravity: { x: 0, y: -9.81 } })
		);
		const output = result.particles[0];
		const values = [
			output.position.x,
			output.position.y,
			output.velocity.x,
			output.velocity.y,
			output.C.m00,
			output.C.m01,
			output.C.m10,
			output.C.m11,
			output.F.m00,
			output.F.m01,
			output.F.m10,
			output.F.m11
		];

		expect(values.every(Number.isFinite)).toBe(true);
		expect(output.position.x).toBeGreaterThanOrEqual(0);
		expect(output.position.x).toBeLessThanOrEqual(16);
		expect(output.position.y).toBeGreaterThanOrEqual(0);
		expect(output.position.y).toBeLessThanOrEqual(16);
	});

	it('collides with each domain wall without leaving the domain', () => {
		const cases = [
			{
				particle: makeParticle({ position: { x: 0, y: 4 }, velocity: { x: -10, y: 0 } }),
				position: { x: 0, y: 4 },
				normalVelocity: { x: 0, y: 0 }
			},
			{
				particle: makeParticle({ position: { x: 8, y: 4 }, velocity: { x: 10, y: 0 } }),
				position: { x: 8, y: 4 },
				normalVelocity: { x: 0, y: 0 }
			},
			{
				particle: makeParticle({ position: { x: 4, y: 0 }, velocity: { x: 0, y: -10 } }),
				position: { x: 4, y: 0 },
				normalVelocity: { x: 0, y: 0 }
			},
			{
				particle: makeParticle({ position: { x: 4, y: 8 }, velocity: { x: 0, y: 10 } }),
				position: { x: 4, y: 8 },
				normalVelocity: { x: 0, y: 0 }
			}
		];
		const config = makeConfig({
			dt: 0.1,
			grid: { cellsX: 8, cellsY: 8, cellSize: 1 }
		});

		for (const collision of cases) {
			const result = stepMpm([collision.particle], config);
			const [output] = result.particles;

			expect(output.position).toEqual(collision.position);
			expect(output.velocity).toEqual(collision.normalVelocity);
			expect(output.position.x).toBeGreaterThanOrEqual(0);
			expect(output.position.x).toBeLessThanOrEqual(8);
			expect(output.position.y).toBeGreaterThanOrEqual(0);
			expect(output.position.y).toBeLessThanOrEqual(8);
		}
	});

	it('conserves mass and momentum away from gravity and active boundaries', () => {
		const particles = [
			makeParticle({
				position: { x: 24.2, y: 21.7 },
				velocity: { x: 0.5, y: -0.2 }
			}),
			makeParticle({
				position: { x: 25.1, y: 21.9 },
				velocity: { x: -0.1, y: 0.4 }
			}),
			makeParticle({
				position: { x: 24.6, y: 22.8 },
				velocity: { x: 0.2, y: 0.1 }
			})
		];
		const mass = 0.1;
		const config = makeConfig({
			dt: 0.001,
			grid: { cellsX: 64, cellsY: 64, cellSize: 1 },
			particleMass: mass
		});
		const initialMomentum = {
			x: particles.reduce((sum, particle) => sum + mass * particle.velocity.x, 0),
			y: particles.reduce((sum, particle) => sum + mass * particle.velocity.y, 0)
		};
		const result = stepMpm(particles, config);
		const { diagnostics } = result;
		const massTolerance = 1e-12;
		const momentumTolerance = 1e-10;

		closeEnough(diagnostics.gridMass, diagnostics.inputMass, massTolerance);
		closeEnough(diagnostics.outputMass, diagnostics.inputMass, massTolerance);
		closeEnough(diagnostics.gridMomentum.x, initialMomentum.x, momentumTolerance);
		closeEnough(diagnostics.gridMomentum.y, initialMomentum.y, momentumTolerance);
		closeEnough(diagnostics.outputMomentum.x, initialMomentum.x, momentumTolerance);
		closeEnough(diagnostics.outputMomentum.y, initialMomentum.y, momentumTolerance);
	});

	it('keeps Neo-Hookean and snow objects distinct in one particle array', () => {
		const particles = [
			makeParticle({
				position: { x: 7.5, y: 8 },
				objectId: 1,
				material: { lambda: 20, mu: 10, model: 'neo-hookean' }
			}),
			makeParticle({
				position: { x: 8.5, y: 8 },
				F: { m00: 0.9, m01: 0, m10: 0, m11: 0.9 },
				objectId: 2,
				material: { lambda: 20, mu: 10, model: 'snow' }
			})
		];
		const result = stepMpm(
			particles,
			makeConfig({ dt: 1e-9, gravity: { x: 0, y: 0 } })
		);
		const [elastic, plastic] = result.particles;

		expect(elastic.objectId).toBe(1);
		expect(elastic.material?.model).toBe('neo-hookean');
		expect(elastic.plasticVolume).toBe(1);
		expect(elastic.F.m00).toBeCloseTo(1, 5);
		expect(plastic.objectId).toBe(2);
		expect(plastic.material?.model).toBe('snow');
		expect(plastic.F.m00).toBeCloseTo(0.975, 6);
		expect(plastic.F.m11).toBeCloseTo(0.975, 6);
		expect(plastic.plasticVolume).toBeCloseTo((0.9 * 0.9) / (0.975 * 0.975), 6);
	});

	it('projects rotated Snow stretches while preserving the principal rotation', () => {
		const angle = 0.37;
		const cosine = Math.cos(angle);
		const sine = Math.sin(angle);
		const result = stepMpm(
			[
				makeParticle({
					position: { x: 8, y: 8 },
					F: {
						m00: cosine * 0.9,
						m01: -sine * 1.1,
						m10: sine * 0.9,
						m11: cosine * 1.1
					},
					material: { lambda: 20, mu: 10, model: 'snow' }
				})
			],
			makeConfig({ dt: 1e-9, gravity: { x: 0, y: 0 } })
		);
		const [particle] = result.particles;

		expect(particle.F.m00).toBeCloseTo(cosine * 0.975, 6);
		expect(particle.F.m01).toBeCloseTo(-sine * 1.0075, 6);
		expect(particle.F.m10).toBeCloseTo(sine * 0.975, 6);
		expect(particle.F.m11).toBeCloseTo(cosine * 1.0075, 6);
	});

	it('rejects unsupported per-particle constitutive models and plastic history', () => {
		expect(() =>
			stepMpm(
				[
					makeParticle({
						material: { lambda: 20, mu: 10, model: 'unsupported' as 'snow' }
					})
				],
				makeConfig()
			)
		).toThrow(/model/);
		expect(() =>
			stepMpm([makeParticle({ plasticVolume: 0 })], makeConfig())
		).toThrow(/plasticVolume/);
	});

	it('keeps snow deformation and plastic history finite through repeated steps', () => {
		let particles = [
			makeParticle({
				position: { x: 7.8, y: 8 },
				F: { m00: 0.9, m01: 0, m10: 0, m11: 0.9 },
				objectId: 4,
				material: { lambda: 20, mu: 10, model: 'snow' }
			}),
			makeParticle({
				position: { x: 8.2, y: 8 },
				F: { m00: 0.9, m01: 0, m10: 0, m11: 0.9 },
				objectId: 4,
				material: { lambda: 20, mu: 10, model: 'snow' }
			})
		];
		const config = makeConfig({
			dt: 1e-5,
			gravity: { x: 0, y: 0 }
		});

		for (let step = 0; step < 100; step += 1) {
			particles = [...stepMpm(particles, config).particles];
		}

		for (const particle of particles) {
			expect(particle.plasticVolume).toBeGreaterThanOrEqual(0.6);
			expect(particle.plasticVolume).toBeLessThanOrEqual(20);
			expect(particle.F.m00 * particle.F.m11 - particle.F.m01 * particle.F.m10)
				.toBeGreaterThan(0);
			expect([
				particle.position.x,
				particle.position.y,
				particle.velocity.x,
				particle.velocity.y,
				particle.plasticVolume
			].every(Number.isFinite)).toBe(true);
		}
	});
});
