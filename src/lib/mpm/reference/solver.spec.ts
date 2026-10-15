import { describe, expect, it } from 'vitest';
import {
	stepMpm,
	type MpmMaterialParameters,
	type MpmMatrix2,
	type MpmParticle,
	type MpmStepConfig
} from './solver.js';

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

function triangleEdgeLengthError(
	particles: readonly MpmParticle[],
	restLength: number
): number {
	const positions = particles.map(({ position }) => position);
	return (
		Math.abs(Math.hypot(positions[1].x - positions[0].x, positions[1].y - positions[0].y) - restLength) +
		Math.abs(Math.hypot(positions[2].x - positions[1].x, positions[2].y - positions[1].y) - restLength) +
		Math.abs(Math.hypot(positions[0].x - positions[2].x, positions[0].y - positions[2].y) - restLength)
	);
}

function principalLogStretchDifference(F: MpmMatrix2): number {
	const c00 = F.m00 * F.m00 + F.m10 * F.m10;
	const c01 = F.m00 * F.m01 + F.m10 * F.m11;
	const c11 = F.m01 * F.m01 + F.m11 * F.m11;
	const radius = Math.hypot(c00 - c11, 2 * c01);
	const sigma1 = Math.sqrt((c00 + c11 + radius) * 0.5);
	const sigma2 = (F.m00 * F.m11 - F.m01 * F.m10) / sigma1;
	return Math.log(sigma1) - Math.log(sigma2);
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
		expect(() => stepMpm([], makeConfig({ bondReferenceDt: 0 }))).toThrow(
			/bondReferenceDt/
		);
		expect(() =>
			stepMpm([], { ...makeConfig(), dt: undefined } as unknown as MpmStepConfig)
		).toThrow(/dt/);
		expect(() =>
			stepMpm([], makeConfig({ grid: { cellsX: 0, cellsY: 16, cellSize: 1 } }))
		).toThrow(/cellsX/);
		expect(() =>
			stepMpm(
				[],
				makeConfig({
					grid: {
						cellsX: Number.MAX_SAFE_INTEGER - 4,
						cellsY: 1,
						cellSize: 1
					}
				})
			)
		).toThrow(/safe node-index range/);
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

	it('substeps a large velocity gradient before it can invert the deformation', () => {
		const result = stepMpm(
			[
				makeParticle({
					C: { m00: 0, m01: 0, m10: 0, m11: -100 }
				})
			],
			makeConfig({ dt: 0.1 })
		);
		const deformation = result.particles[0].F;

		expect(deformation.m00 * deformation.m11 - deformation.m01 * deformation.m10).toBeGreaterThan(0);
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

	it('keeps all material models stable under sustained acceleration into domain walls', () => {
		const materials: readonly MpmMaterialParameters[] = [
			{ lambda: 8, mu: 0, model: 'water', density: 0.04 },
			{ lambda: 6, mu: 0, model: 'oil', density: 0.032 },
			{ lambda: 20, mu: 10, model: 'powder', frictionCoefficient: 0.55 }
		];
		const collisions = [
			{ startX: 8.2, startY: 15.2, gravity: { x: -90, y: 0 } },
			{ startX: 23.2, startY: 15.2, gravity: { x: 90, y: 0 } },
			{ startX: 15.2, startY: 8.2, gravity: { x: 0, y: -90 } },
			{ startX: 15.2, startY: 23.2, gravity: { x: 0, y: 90 } }
		] as const;
		const config = (material: MpmMaterialParameters, gravity: MpmStepConfig['gravity']) =>
			makeConfig({
				dt: 1 / 240,
				grid: { cellsX: 32, cellsY: 32, cellSize: 1 },
				particleVolume: 0.16,
				material,
				gravity
			});

		for (const material of materials) {
			for (const collision of collisions) {
				let particles: readonly MpmParticle[] = Array.from({ length: 9 }, (_, index) =>
					makeParticle({
						position: {
							x: collision.startX + (index % 3) * 0.4,
							y: collision.startY + Math.floor(index / 3) * 0.4
						},
						material
					})
				);
				const stepConfig = config(material, collision.gravity);

				for (let step = 0; step < 480; step += 1) {
					particles = stepMpm(particles, stepConfig).particles;
				}

				for (const particle of particles) {
					const jacobian = particle.F.m00 * particle.F.m11 - particle.F.m01 * particle.F.m10;
					expect(jacobian).toBeGreaterThan(0);
					expect(particle.position.x).toBeGreaterThanOrEqual(0);
					expect(particle.position.x).toBeLessThanOrEqual(32);
					expect(particle.position.y).toBeGreaterThanOrEqual(0);
					expect(particle.position.y).toBeLessThanOrEqual(32);
				}
			}
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

	it('keeps solid fracture localized, conserves mass and momentum, and lets broken fragments move independently', () => {
		const particles = [
			makeParticle({
				position: { x: 10, y: 20 },
				velocity: { x: 1, y: 0 },
				objectId: 4
			}),
			makeParticle({
				position: { x: 25, y: 20 },
				velocity: { x: -1, y: 0 },
				objectId: 4
			}),
			makeParticle({
				position: { x: 35, y: 20 },
				objectId: 4
			})
		];
		const fractureBonds = [
			{
				firstParticle: 0,
				secondParticle: 1,
				restLength: 10,
				failureStrain: 0.1
			}
		] as const;
		const config = makeConfig({
			dt: 0.001,
			grid: { cellsX: 64, cellsY: 64, cellSize: 1 },
			particleMass: 1,
			fractureBonds
		});
		const result = stepMpm(particles, config);
		const next = stepMpm(result.particles, {
			...config,
			fractureBonds: result.fractureBonds
		});

		expect(result.fractureBonds?.[0]).toMatchObject({ damage: 1, broken: true });
		expect(result.particles.map(({ damage }) => damage ?? 0)).toEqual([1, 1, 0]);
		expect(result.diagnostics.inputMass).toBe(3);
		expect(result.diagnostics.gridMass).toBeCloseTo(3, 12);
		expect(result.diagnostics.outputMass).toBe(3);
		expect(result.diagnostics.outputMomentum.x).toBeCloseTo(0, 12);
		expect(next.fractureBonds?.[0]).toMatchObject({ damage: 1, broken: true });
		expect(next.particles[0].position.x).toBeGreaterThan(result.particles[0].position.x);
		expect(next.particles[1].position.x).toBeLessThan(result.particles[1].position.x);
		expect(next.particles[2].damage).toBeUndefined();
		for (const particle of [...result.particles, ...next.particles]) {
			expect(particle.position.x).toBeGreaterThanOrEqual(0);
			expect(particle.position.x).toBeLessThanOrEqual(config.grid.cellsX);
			expect(particle.position.y).toBeGreaterThanOrEqual(0);
			expect(particle.position.y).toBeLessThanOrEqual(config.grid.cellsY);
		}
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

	it('bounds a collapsing Neo-Hookean principal stretch before it becomes singular', () => {
		const material = { lambda: 20, mu: 10, model: 'neo-hookean' } as const;
		const result = stepMpm(
			[
				makeParticle({
					F: { m00: 1, m01: 0, m10: 0, m11: 1e-8 },
					material
				})
			],
			makeConfig({ dt: 1e-8, gravity: { x: 0, y: 0 } })
		);
		const deformation = result.particles[0].F;
		const determinant =
			deformation.m00 * deformation.m11 - deformation.m01 * deformation.m10;

		expect(determinant).toBeGreaterThanOrEqual(0.05 * 0.05 - 1e-10);
		expect(deformation.m11).toBeCloseTo(0.05, 6);

		const extreme = stepMpm(
			[
				makeParticle({
					F: { m00: 1_000, m01: 0, m10: 0, m11: 1e-8 },
					material
				})
			],
			makeConfig({ dt: 1e-12, gravity: { x: 0, y: 0 } })
		).particles[0].F;
		expect(extreme.m00).toBeCloseTo(20, 5);
		expect(extreme.m11).toBeCloseTo(0.05, 5);
		expect(extreme.m00 * extreme.m11).toBeGreaterThan(0);
	});

	it('uses per-particle water and oil density without introducing liquid shear stress', () => {
		const shear = { m00: 1, m01: 0.4, m10: 0, m11: 1 };
		const particles = [
			makeParticle({
				position: { x: 7.5, y: 8 },
				velocity: { x: 0.5, y: -0.25 },
				F: shear,
				volume: 0.01,
				material: { lambda: 8, mu: 0, model: 'water', density: 1 }
			}),
			makeParticle({
				position: { x: 8.5, y: 8 },
				velocity: { x: 0.5, y: -0.25 },
				F: shear,
				volume: 0.02,
				material: { lambda: 6, mu: 0, model: 'oil', density: 0.8 }
			})
		];
		const result = stepMpm(
			particles,
			makeConfig({ dt: 1e-12, gravity: { x: 0, y: 0 } })
		);

		const waterDeformation = result.particles[0].F;
		const oilDeformation = result.particles[1].F;
		expect(waterDeformation.m00).toBeCloseTo(waterDeformation.m11, 12);
		expect(waterDeformation.m01).toBeCloseTo(-waterDeformation.m10, 12);
		expect(waterDeformation.m00 * waterDeformation.m11 - waterDeformation.m01 * waterDeformation.m10).toBeCloseTo(1, 11);
		expect(oilDeformation.m00).toBeCloseTo(waterDeformation.m00, 12);
		expect(oilDeformation.m01).toBeCloseTo(waterDeformation.m01, 12);
		expect(oilDeformation.m10).toBeCloseTo(waterDeformation.m10, 12);
		expect(oilDeformation.m11).toBeCloseTo(waterDeformation.m11, 12);
		expect(result.particles[0].material?.model).toBe('water');
		expect(result.particles[1].material?.model).toBe('oil');
		expect(result.diagnostics.inputMass).toBeCloseTo(0.026, 12);
		expect(result.diagnostics.gridMass).toBeCloseTo(0.026, 12);
		expect(result.diagnostics.outputMass).toBeCloseTo(0.026, 12);
		expect(result.diagnostics.inputMomentum.x).toBeCloseTo(0.013, 14);
		expect(result.diagnostics.inputMomentum.y).toBeCloseTo(-0.0065, 14);
		expect(result.diagnostics.outputMomentum.x).toBeCloseTo(0.013, 14);
		expect(result.diagnostics.outputMomentum.y).toBeCloseTo(-0.0065, 14);
	});

	it('restores compressed liquid volume more strongly with a stiffer EOS', () => {
		const initialParticle = makeParticle({
			position: { x: 8.37, y: 8.23 },
			F: { m00: 0.8, m01: 0, m10: 0, m11: 0.8 },
			volume: 0.25,
			material: { lambda: 8, mu: 0, model: 'water', density: 1 }
		});
		const config = makeConfig({
			dt: 0.0001,
			grid: { cellsX: 16, cellsY: 16, cellSize: 1 },
			particleVolume: 0.25,
			gravity: { x: 0, y: 0 }
		});
		const simulate = (lambda: number): number => {
			let particle: MpmParticle = {
				...initialParticle,
				material: { lambda, mu: 0 as const, model: 'water' as const, density: 1 }
			};
			for (let step = 0; step < 100; step += 1) {
				particle = stepMpm([particle], config).particles[0];
			}
			return particle.F.m00 * particle.F.m11 - particle.F.m01 * particle.F.m10;
		};

		const baselineDeterminant = simulate(8);
		const stiffDeterminant = simulate(32);

		expect(baselineDeterminant).toBeGreaterThan(0.64);
		expect(stiffDeterminant).toBeGreaterThan(baselineDeterminant);
		expect(stiffDeterminant).toBeLessThanOrEqual(1);
	});

	it('moves denser liquid particles downward relative to lighter liquids in a mixed region', () => {
		const particles = [
			makeParticle({
				position: { x: 8, y: 8 },
				volume: 1,
				material: { lambda: 8, mu: 0, model: 'water', density: 2 }
			}),
			makeParticle({
				position: { x: 8, y: 8 },
				volume: 1,
				material: { lambda: 8, mu: 0, model: 'oil', density: 1 }
			})
		];
		const result = stepMpm(
			particles,
			makeConfig({ dt: 0.01, gravity: { x: 0, y: 10 } })
		);

		expect(result.particles[0].velocity.y).toBeGreaterThan(result.particles[1].velocity.y);
		expect(result.particles[0].position.y).toBeGreaterThan(result.particles[1].position.y);
	});

	it('moves denser water down through lighter oil when their layers start inverted', () => {
		let particles = [
			makeParticle({
				position: { x: 8, y: 7.5 },
				volume: 1,
				material: { lambda: 8, mu: 0, model: 'water', density: 2 }
			}),
			makeParticle({
				position: { x: 8, y: 8.5 },
				volume: 1,
				material: { lambda: 8, mu: 0, model: 'oil', density: 1 }
			})
		];
		const config = makeConfig({ dt: 0.001, gravity: { x: 0, y: 10 } });

		for (let step = 0; step < 500; step += 1) {
			particles = [...stepMpm(particles, config).particles];
		}

		expect(particles[0].position.y).toBeGreaterThan(particles[1].position.y);
	});

	it('orients the interface for the denser oil phase when oil is heavier', () => {
		const particles = [
			makeParticle({
				position: { x: 8, y: 8.5 },
				volume: 1,
				material: { lambda: 8, mu: 0, model: 'water', density: 1 }
			}),
			makeParticle({
				position: { x: 8, y: 7.5 },
				volume: 1,
				material: { lambda: 8, mu: 0, model: 'oil', density: 2 }
			})
		];
		const result = stepMpm(
			particles,
			makeConfig({ dt: 0.001, gravity: { x: 0, y: 10 } })
		);

		expect(result.particles[1].velocity.y).toBeGreaterThan(
			result.particles[0].velocity.y
		);
	});

	it.each([
		{ dt: 0.0001, speed: 0.5 },
		{ dt: 0.001, speed: 0.5 },
		{ dt: 0.01, speed: 0.5 },
		{ dt: 0.001, speed: 64 }
	])(
		'resolves water-oil contact with equal and opposite normal impulses ($dt s, $speed units/s)',
		({ dt, speed }) => {
			const particles = [
			makeParticle({
				position: { x: 7.5, y: 8 },
				velocity: { x: speed, y: 0 },
				objectId: 1,
				volume: 0.25,
				material: { lambda: 8, mu: 0, model: 'water', density: 1 }
			}),
			makeParticle({
				position: { x: 8.5, y: 8 },
				velocity: { x: -speed, y: 0 },
				objectId: 2,
				volume: 0.25,
				material: { lambda: 8, mu: 0, model: 'oil', density: 1 }
			})
			];
			const result = stepMpm(
				particles,
				makeConfig({
					dt,
					particleVolume: 0.25,
					gravity: { x: 0, y: 0 }
				})
			);

			expect(result.particles[0].velocity.x).toBeLessThan(speed);
			expect(result.particles[1].velocity.x).toBeGreaterThan(-speed);
			expect(
				result.particles[1].velocity.x - result.particles[0].velocity.x
			).toBeGreaterThanOrEqual(0);
			expect(
				result.particles[1].velocity.x - result.particles[0].velocity.x
			).toBeLessThanOrEqual(Math.max(2 * speed, 5) + 1e-8);
			expect(result.particles[0].material?.model).toBe('water');
			expect(result.particles[1].material?.model).toBe('oil');
			expect(result.particles[0].position.x).toBeLessThanOrEqual(
				result.particles[1].position.x
			);
			expect(result.diagnostics.inputMass).toBe(0.5);
			expect(result.diagnostics.outputMass).toBe(0.5);
			expect(
				result.particles
					.filter((particle) => particle.material?.model === 'water')
					.length * 0.25
			).toBe(0.25);
			expect(
				result.particles
					.filter((particle) => particle.material?.model === 'oil')
					.length * 0.25
			).toBe(0.25);
			expect(result.diagnostics.outputMomentum.x).toBeCloseTo(
				result.diagnostics.inputMomentum.x,
				10
			);
			expect(
				result.particles.every(
					(particle) =>
						Number.isFinite(particle.velocity.x) &&
						Number.isFinite(particle.position.x)
				)
			).toBe(true);
			if (speed > 1) {
				const outputKineticEnergy = result.particles.reduce(
					(energy, particle) =>
						energy +
						0.5 *
							0.25 *
							(particle.velocity.x ** 2 + particle.velocity.y ** 2),
					0
				);
				expect(outputKineticEnergy).toBeLessThan(0.25 * speed ** 2);
			}
		}
	);

	it('returns liquid contact reaction to a bonded triangle and preserves vertex markers', () => {
		const particles: MpmParticle[] = [
			makeParticle({
				position: { x: 7.5, y: 8 },
				velocity: { x: 4, y: 0 },
				objectId: 1,
				volume: 0.25,
				material: { lambda: 8, mu: 0, model: 'water', density: 1 }
			}),
			makeParticle({
				position: { x: 8, y: 8 },
				objectId: 2,
				bondedTriangleVertex: 0
			}),
			makeParticle({
				position: { x: 8 + Math.sqrt(3), y: 7 },
				objectId: 2,
				bondedTriangleVertex: 1
			}),
			makeParticle({
				position: { x: 8 + Math.sqrt(3), y: 9 },
				objectId: 2,
				bondedTriangleVertex: 2
			})
		];
		const result = stepMpm(
			particles,
			makeConfig({
				dt: 0.001,
				particleVolume: 0.25,
				bondRestLength: 2,
				gravity: { x: 0, y: 0 }
			})
		);
		const water = result.particles[0];
		const triangle = result.particles.slice(1);
		const triangleMomentumX = triangle.reduce(
			(momentum, particle) => momentum + 0.1 * particle.velocity.x,
			0
		);

		expect(water.velocity.x).toBeLessThan(4);
		expect(triangleMomentumX).toBeGreaterThan(0);
		expect(triangle.map(({ bondedTriangleVertex }) => bondedTriangleVertex)).toEqual([
			0, 1, 2
		]);
		expect(result.diagnostics.outputMomentum.x).toBeCloseTo(
			result.diagnostics.inputMomentum.x,
			8
		);
	});

	it('transfers tangential liquid motion to bonded triangles without gaining energy', () => {
		const particles: MpmParticle[] = [
			makeParticle({
				position: { x: 7.5, y: 8 },
				velocity: { x: 0, y: 4 },
				objectId: 1,
				volume: 0.25,
				material: { lambda: 8, mu: 0, model: 'water', density: 1 }
			}),
			makeParticle({
				position: { x: 8, y: 7.5 },
				objectId: 2,
				bondedTriangleVertex: 0
			}),
			makeParticle({
				position: { x: 8, y: 8.5 },
				objectId: 2,
				bondedTriangleVertex: 1
			}),
			makeParticle({
				position: { x: 8 + Math.sqrt(3) / 2, y: 8 },
				objectId: 2,
				bondedTriangleVertex: 2
			})
		];
		const result = stepMpm(
			particles,
			makeConfig({
				dt: 0.01,
				particleVolume: 0.25,
				bondRestLength: 1,
				gravity: { x: 0, y: 0 }
			})
		);
		const triangle = result.particles.slice(1);
		const triangleMomentumY = triangle.reduce(
			(momentum, particle) => momentum + 0.1 * particle.velocity.y,
			0
		);
		const inputKineticEnergy = 0.5 * 0.25 * 4 ** 2;
		const outputKineticEnergy = result.particles.reduce(
			(energy, particle, index) =>
				energy +
				0.5 *
					(index === 0 ? 0.25 : 0.1) *
					(particle.velocity.x ** 2 + particle.velocity.y ** 2),
			0
		);

		expect(result.particles[0].velocity.y).toBeLessThan(4);
		expect(triangleMomentumY).toBeGreaterThan(0);
		expect(outputKineticEnergy).toBeLessThanOrEqual(inputKineticEnergy + 1e-8);
		expect(result.diagnostics.outputMomentum.y).toBeCloseTo(
			result.diagnostics.inputMomentum.y,
			8
		);
	});

	it('restores a stretched bonded triangle toward its configured edge length', () => {
		const particles: MpmParticle[] = [
			makeParticle({
				position: { x: 8, y: 8 },
				objectId: 4,
				bondedTriangleVertex: 0
			}),
			makeParticle({
				position: { x: 10, y: 8 },
				objectId: 4,
				bondedTriangleVertex: 1
			}),
			makeParticle({
				position: { x: 9, y: 9.5 },
				objectId: 4,
				bondedTriangleVertex: 2
			})
		];
		const initialError = triangleEdgeLengthError(particles, 1);
		const result = stepMpm(
			particles,
			makeConfig({
				dt: 0.001,
				bondRestLength: 1,
				bondStiffness: 1,
				gravity: { x: 0, y: 0 }
			})
		);

		expect(triangleEdgeLengthError(result.particles, 1)).toBeLessThan(initialError);
		expect(result.particles.map(({ bondedTriangleVertex }) => bondedTriangleVertex)).toEqual([
			0, 1, 2
		]);
	});

	it('bounds velocity impulses across simultaneous bonded-triangle contacts', () => {
		const makeTriangle = (
			centerX: number,
			velocity: { readonly x: number; readonly y: number },
			objectId: number
		): MpmParticle[] => {
			const centerY = 8;
			const topY = centerY - Math.sqrt(3) / 6;
			const bottomY = centerY + Math.sqrt(3) / 3;
			return [
				makeParticle({
					position: { x: centerX - 0.5, y: topY },
					velocity,
					objectId,
					bondedTriangleVertex: 0
				}),
				makeParticle({
					position: { x: centerX + 0.5, y: topY },
					velocity,
					objectId,
					bondedTriangleVertex: 1
				}),
				makeParticle({
					position: { x: centerX, y: bottomY },
					velocity,
					objectId,
					bondedTriangleVertex: 2
				})
			];
		};
		const particles = [
			...makeTriangle(7, { x: 2, y: 0 }, 4),
			...makeTriangle(7.75, { x: 0, y: 0 }, 5),
			...makeTriangle(8.5, { x: -2, y: 0 }, 6)
		];
		const result = stepMpm(
			particles,
			makeConfig({
				dt: 0.001,
				bondRestLength: 1,
				gravity: { x: 0, y: 0 }
			})
		);
		const inputKineticEnergy = particles.reduce(
			(energy, { velocity }) =>
				energy + 0.5 * 0.1 * (velocity.x ** 2 + velocity.y ** 2),
			0
		);
		const outputKineticEnergy = result.particles.reduce(
			(energy, { velocity }) =>
				energy + 0.5 * 0.1 * (velocity.x ** 2 + velocity.y ** 2),
			0
		);

		expect(outputKineticEnergy).toBeLessThanOrEqual(inputKineticEnergy + 1e-6);
		expect(result.diagnostics.outputMomentum.x).toBeCloseTo(
			result.diagnostics.inputMomentum.x,
			8
		);
	});

	it('does not translate stationary bonded triangles during pair contacts', () => {
		const makeTriangle = (centerX: number, objectId: number): MpmParticle[] => {
			const centerY = 8;
			const topY = centerY - Math.sqrt(3) / 6;
			const bottomY = centerY + Math.sqrt(3) / 3;
			return [
				makeParticle({
					position: { x: centerX - 0.5, y: topY },
					objectId,
					bondedTriangleVertex: 0
				}),
				makeParticle({
					position: { x: centerX + 0.5, y: topY },
					objectId,
					bondedTriangleVertex: 1
				}),
				makeParticle({
					position: { x: centerX, y: bottomY },
					objectId,
					bondedTriangleVertex: 2
				})
			];
		};
		const particles = [...makeTriangle(7, 4), ...makeTriangle(7.75, 5)];
		const result = stepMpm(
			particles,
			makeConfig({
				dt: 0.001,
				bondRestLength: 1,
				gravity: { x: 0, y: 0 }
			})
		);

		expect(result.particles.map(({ position }) => position)).toEqual(
			particles.map(({ position }) => position)
		);
	});

	it('scales bonded-triangle velocity corrections with CFL microstep duration', () => {
		const particles: MpmParticle[] = [
			makeParticle({
				position: { x: 8, y: 8 },
				objectId: 4,
				bondedTriangleVertex: 0
			}),
			makeParticle({
				position: { x: 10, y: 8 },
				objectId: 4,
				bondedTriangleVertex: 1
			}),
			makeParticle({
				position: { x: 9, y: 9.5 },
				objectId: 4,
				bondedTriangleVertex: 2
			})
		];
		const referenceDt = 0.001;
		const config = {
			bondRestLength: 1,
			bondStiffness: 0.35,
			bondReferenceDt: referenceDt,
			gravity: { x: 0, y: 0 }
		} as const;
		const fullStep = stepMpm(
			particles,
			makeConfig({ ...config, dt: referenceDt })
		);
		const halfStep = stepMpm(
			particles,
			makeConfig({ ...config, dt: referenceDt / 2 })
		);
		const twoHalfSteps = stepMpm(
			halfStep.particles,
			makeConfig({ ...config, dt: referenceDt / 2 })
		);

		const fullStepSpeed = Math.hypot(
			fullStep.particles[0].velocity.x,
			fullStep.particles[0].velocity.y
		);
		const halfStepSpeed = Math.hypot(
			halfStep.particles[0].velocity.x,
			halfStep.particles[0].velocity.y
		);
		const twoHalfStepSpeed = Math.hypot(
			twoHalfSteps.particles[0].velocity.x,
			twoHalfSteps.particles[0].velocity.y
		);

		expect(fullStepSpeed).toBeGreaterThan(0);
		expect(halfStepSpeed).toBeLessThan(fullStepSpeed * 0.75);
		expect(twoHalfStepSpeed).toBeLessThan(fullStepSpeed * 1.3);
	});

	it('applies liquid bulk pressure only to compression', () => {
		const material = { lambda: 8, mu: 0, model: 'water', density: 1 } as const;
		const compressed = stepMpm(
			[
				makeParticle({
					F: { m00: 0.9, m01: 0, m10: 0, m11: 0.9 },
					material
				})
			],
			makeConfig({ dt: 0.001, gravity: { x: 0, y: 0 } })
		).particles[0];
		const expanded = stepMpm(
			[
				makeParticle({
					F: { m00: 1.1, m01: 0, m10: 0, m11: 1.1 },
					material
				})
			],
			makeConfig({ dt: 0.001, gravity: { x: 0, y: 0 } })
		).particles[0];

		expect(compressed.C.m00).toBeGreaterThan(0);
		expect(compressed.F.m00).toBeGreaterThan(0.9);
		expect(expanded.C.m00).toBe(0);
		expect(expanded.F.m00).toBe(1.1);
	});

	it('bounds liquid volume expansion to keep deformation determinants finite', () => {
		const particles = [
			makeParticle({
				F: { m00: 3, m01: 0, m10: 0, m11: 3 },
				material: { lambda: 8, mu: 0, model: 'water', density: 1 }
			})
		];
		const config = makeConfig({ dt: 0.001, gravity: { x: 0, y: 0 } });
		let state = particles;

		for (let step = 0; step < 20; step += 1) {
			state = [...stepMpm(state, config).particles];
			const deformation = state[0].F;
			const determinant =
				deformation.m00 * deformation.m11 - deformation.m01 * deformation.m10;

			expect(Number.isFinite(determinant)).toBe(true);
			expect(determinant).toBeGreaterThanOrEqual(0.5 - 1e-10);
			expect(determinant).toBeLessThanOrEqual(2 + 1e-10);
		}
	});

	it('bounds powder volume collapse to prevent excessive compression', () => {
		const particle = makeParticle({
			F: { m00: 0.1, m01: 0, m10: 0, m11: 1 },
			material: {
				lambda: 20,
				mu: 10,
				model: 'powder',
				frictionCoefficient: 0.55
			}
		});
		const result = stepMpm(
			[particle],
			makeConfig({ dt: 1e-12, gravity: { x: 0, y: 0 } })
		).particles[0];
		const determinant =
			result.F.m00 * result.F.m11 - result.F.m01 * result.F.m10;

		expect(determinant).toBeCloseTo(0.5, 10);
	});

	it('caps powder shear by pressure-dependent friction', () => {
		const compressedShear = { m00: 0.8, m01: 0.8, m10: 0, m11: 0.8 };
		const result = stepMpm(
			[
				makeParticle({
					position: { x: 6, y: 8 },
					F: compressedShear,
					material: {
						lambda: 20,
						mu: 10,
						model: 'powder',
						frictionCoefficient: 0
					}
				}),
				makeParticle({
					position: { x: 12, y: 8 },
					F: compressedShear,
					material: {
						lambda: 20,
						mu: 10,
						model: 'powder',
						frictionCoefficient: 0.55
					}
				})
			],
			makeConfig({ dt: 1e-12, gravity: { x: 0, y: 0 } })
		);
		const zeroFrictionStretch = principalLogStretchDifference(result.particles[0].F);
		const frictionalStretch = principalLogStretchDifference(result.particles[1].F);
		const frictionalMaterial = result.particles[1].material;

		expect(Math.abs(zeroFrictionStretch)).toBeLessThan(1e-8);
		expect(frictionalStretch).toBeGreaterThan(0.5);
		expect(result.particles[0].material?.model).toBe('powder');
		expect(frictionalMaterial).toMatchObject({ model: 'powder', frictionCoefficient: 0.55 });
	});

	it('preserves an unyielded powder deformation during projection', () => {
		const material = {
			lambda: 20,
			mu: 10,
			model: 'powder',
			frictionCoefficient: 0.55
		} as const;
		const result = stepMpm(
			[makeParticle({ material })],
			makeConfig({ gravity: { x: 0, y: 0 } })
		);

		expect(result.particles[0].F).toEqual(IDENTITY);
	});

	it('projects liquid shear to rotation while preserving volume', () => {
		for (const model of ['water', 'oil'] as const) {
			const result = stepMpm(
				[
					makeParticle({
						F: { m00: 1, m01: 1, m10: 0, m11: 1 },
						material: { lambda: 20, mu: 0, model, density: 0.04 }
					})
				],
				makeConfig({ gravity: { x: 0, y: 0 } })
			);
			const deformation = result.particles[0].F;

			expect(deformation.m00).toBeCloseTo(deformation.m11, 6);
			expect(deformation.m01).toBeCloseTo(-deformation.m10, 6);
			expect(deformation.m00 * deformation.m11 - deformation.m01 * deformation.m10).toBeCloseTo(1, 6);
		}
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
		expect(() =>
			stepMpm(
				[
					makeParticle({
						material: {
							lambda: 8,
							mu: 1,
							model: 'water',
							density: 1
						} as MpmParticle['material']
					})
				],
				makeConfig()
			)
		).toThrow(/mu must be zero/);
		expect(() =>
			stepMpm(
				[
					makeParticle({
						material: {
							lambda: 20,
							mu: 10,
							model: 'powder'
						} as MpmParticle['material']
					})
				],
				makeConfig()
			)
		).toThrow(/frictionCoefficient/);
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
