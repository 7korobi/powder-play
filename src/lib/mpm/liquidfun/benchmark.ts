import { createLiquidFunSimulation } from './simulation.js';
import {
	LIQUIDFUN_BOX2D_DOMAIN_UNITS,
	LIQUIDFUN_BOX2D_GRAVITY_UNITS_PER_SECOND_SQUARED,
	LIQUIDFUN_BOX2D_UNITS_PER_METER,
	LIQUIDFUN_DT_SECONDS,
	LIQUIDFUN_GATE4_DOMAIN,
	LIQUIDFUN_GATE4_ACCELERATION_UNITS_PER_METER_PER_SECOND_SQUARED,
	LIQUIDFUN_GATE4_GRAVITY,
	LIQUIDFUN_GATE4_UNITS_PER_METER,
	LIQUIDFUN_FRACTURE_BOND_FAILURE_STRAIN,
	LIQUIDFUN_GRAVITY_WARMUP_STEPS,
	LIQUIDFUN_ELASTIC_BALL_PARTICLES_PER_GROUP,
	LIQUIDFUN_ELASTIC_STRENGTH,
	LIQUIDFUN_PACKAGE_VERSION,
	LIQUIDFUN_PARTICLE_RADIUS_METERS,
	LIQUIDFUN_PARTICLE_SPACING_METERS,
	LIQUIDFUN_PRESSURE_STRENGTH,
	LIQUIDFUN_FLUID_PARTICLE_COUNT,
	LIQUIDFUN_INK_PARTICLE_COUNT,
	LIQUIDFUN_OIL_PARTICLE_COUNT,
	LIQUIDFUN_SAMPLE_BATCHES,
	LIQUIDFUN_SCENE_SEED,
	LIQUIDFUN_SHARED_PARTICLE_DENSITY,
	LIQUIDFUN_SHARED_PARTICLE_MASS,
	LIQUIDFUN_SOIL_PARTICLE_COUNT,
	LIQUIDFUN_SPRING_STRENGTH,
	LIQUIDFUN_SPRING_VINE_LINK_COUNT,
	LIQUIDFUN_SPRING_VINE_LENGTH_METERS,
	LIQUIDFUN_SPRING_VINE_PARTICLES_PER_GROUP,
	LIQUIDFUN_SPRING_VINE_SPAN_METERS,
	LIQUIDFUN_STATIONARY_WARMUP_MS,
	LIQUIDFUN_STEPS_PER_BATCH,
	LIQUIDFUN_TOTAL_PARTICLE_COUNT,
	LIQUIDFUN_VELOCITY_ITERATIONS,
	LIQUIDFUN_POSITION_ITERATIONS,
	LIQUIDFUN_PHYSICAL_DOMAIN_METERS,
	LIQUIDFUN_WATER_PARTICLE_COUNT,
	summarizeSamples
} from './workload.js';
import { GATE4_GRID } from '../scenes/gate4.js';

export interface LiquidFunBenchmarkInput {
	readonly packageVersion: string;
}

export async function runLiquidFunBenchmark({
	packageVersion
}: LiquidFunBenchmarkInput) {
	if (packageVersion !== LIQUIDFUN_PACKAGE_VERSION) {
		throw new Error(
			`Benchmark package version mismatch: expected ${LIQUIDFUN_PACKAGE_VERSION}, received ${packageVersion}.`
		);
	}
	if (typeof navigator === 'undefined' || typeof performance === 'undefined') {
		throw new Error('LiquidFun benchmark requires a browser performance clock.');
	}

	const simulation = await createLiquidFunSimulation();
	try {
		simulation.setGravityFromGate4({ x: 0, y: 0 });
		const stationaryWarmupUntil =
			performance.now() + LIQUIDFUN_STATIONARY_WARMUP_MS;
		let stationaryWarmupSteps = 0;
		while (performance.now() < stationaryWarmupUntil) {
			simulation.step();
			stationaryWarmupSteps += 1;
		}

		simulation.setGravityFromGate4(LIQUIDFUN_GATE4_GRAVITY);
		for (let step = 0; step < LIQUIDFUN_GRAVITY_WARMUP_STEPS; step += 1) {
			simulation.step();
		}

		const samples: number[] = [];
		for (let batch = 0; batch < LIQUIDFUN_SAMPLE_BATCHES; batch += 1) {
			const startedAt = performance.now();
			for (let step = 0; step < LIQUIDFUN_STEPS_PER_BATCH; step += 1) {
				simulation.step();
			}
			samples.push(performance.now() - startedAt);
		}

		const batchDistribution = summarizeSamples(samples);
		return {
			engine: {
				name: 'liquidfun-wasm',
				packageVersion,
				provenance: 'Birch-san/box2d-wasm LiquidFun-derived Box2D 2.4.1-based WebAssembly fork'
			},
			browser: {
				userAgent: navigator.userAgent,
				hardwareConcurrency: Number.isSafeInteger(navigator.hardwareConcurrency)
					? navigator.hardwareConcurrency
					: null
			},
			config: {
				scene: 'LiquidFun mixed-material workbench',
				sourceScene: 'Gate 4',
				seed: LIQUIDFUN_SCENE_SEED,
				seedHex: `0x${LIQUIDFUN_SCENE_SEED.toString(16)}`,
				sourceMpmNodeCount: simulation.sourceMpmNodeCount,
				liquidFunParticleCount: simulation.particleCount,
				totalParticleCount: LIQUIDFUN_TOTAL_PARTICLE_COUNT,
				fluidParticleCount: LIQUIDFUN_FLUID_PARTICLE_COUNT,
				waterParticleCount: simulation.waterParticleCount,
				oilParticleCount: simulation.oilParticleCount,
				inkParticleCount: simulation.inkParticleCount,
				soilParticleCount: simulation.soilParticleCount,
				configuredFluidCounts: {
					water: LIQUIDFUN_WATER_PARTICLE_COUNT,
					oil: LIQUIDFUN_OIL_PARTICLE_COUNT,
					ink: LIQUIDFUN_INK_PARTICLE_COUNT,
					soil: LIQUIDFUN_SOIL_PARTICLE_COUNT
				},
				elasticBallCount: simulation.elasticBallCount,
				elasticBallParticlesPerGroup: LIQUIDFUN_ELASTIC_BALL_PARTICLES_PER_GROUP,
				elasticParticleCount: simulation.elasticParticleCount,
				elasticTriadCount: simulation.elasticTriadCount,
				elasticStrength: LIQUIDFUN_ELASTIC_STRENGTH,
				springVineCount: simulation.springVineCount,
				springVineParticlesPerGroup: LIQUIDFUN_SPRING_VINE_PARTICLES_PER_GROUP,
				springVineParticleCount: simulation.springVineParticleCount,
				springPairCount: simulation.springPairCount,
				expectedSpringPairCount: LIQUIDFUN_SPRING_VINE_LINK_COUNT,
				springVineTargetLengthMeters: LIQUIDFUN_SPRING_VINE_LENGTH_METERS,
				springVineSpanMeters: LIQUIDFUN_SPRING_VINE_SPAN_METERS,
				springStrength: LIQUIDFUN_SPRING_STRENGTH,
				fractureNodeCount: simulation.fractureNodeCount,
				fractureBondCount: simulation.fractureBondCount,
				activeFractureBondCountAfterMeasurement:
					simulation.activeFractureBondCount,
				fractureBondFailureStrain: LIQUIDFUN_FRACTURE_BOND_FAILURE_STRAIN,
				dynamicSolidBodyCount: simulation.fractureNodeCount,
				staticWallBodyCount: simulation.staticWallBodyCount,
				grid: {
					cellsX: GATE4_GRID.cellsX,
					cellsY: GATE4_GRID.cellsY,
					cellSizeGate4Units: GATE4_GRID.cellSize
				},
				domainGate4Units: LIQUIDFUN_GATE4_DOMAIN,
				domainMeters: LIQUIDFUN_PHYSICAL_DOMAIN_METERS,
				domainBox2dUnits: LIQUIDFUN_BOX2D_DOMAIN_UNITS,
				gate4UnitsPerMeter: LIQUIDFUN_GATE4_UNITS_PER_METER,
				box2dUnitsPerMeter: LIQUIDFUN_BOX2D_UNITS_PER_METER,
				gate4AccelerationUnitsPerMeterPerSecondSquared:
					LIQUIDFUN_GATE4_ACCELERATION_UNITS_PER_METER_PER_SECOND_SQUARED,
				dtSeconds: LIQUIDFUN_DT_SECONDS,
				gravityGate4UnitsPerSecondSquared: LIQUIDFUN_GATE4_GRAVITY,
				gravityBox2dUnitsPerSecondSquared:
					LIQUIDFUN_BOX2D_GRAVITY_UNITS_PER_SECOND_SQUARED,
				gravityMetersPerSecondSquared: {
					x:
						LIQUIDFUN_GATE4_GRAVITY.x /
						LIQUIDFUN_GATE4_ACCELERATION_UNITS_PER_METER_PER_SECOND_SQUARED,
					y:
						-LIQUIDFUN_GATE4_GRAVITY.y /
						LIQUIDFUN_GATE4_ACCELERATION_UNITS_PER_METER_PER_SECOND_SQUARED
				},
				particleRadiusMeters: LIQUIDFUN_PARTICLE_RADIUS_METERS,
				particleSpacingMeters: LIQUIDFUN_PARTICLE_SPACING_METERS,
				pressureStrength: LIQUIDFUN_PRESSURE_STRENGTH,
				sharedParticleMass: LIQUIDFUN_SHARED_PARTICLE_MASS,
				sharedParticleSystemDensity: LIQUIDFUN_SHARED_PARTICLE_DENSITY,
				velocityIterations: LIQUIDFUN_VELOCITY_ITERATIONS,
				positionIterations: LIQUIDFUN_POSITION_ITERATIONS,
				particleIterations: simulation.particleIterations,
				stationaryWarmupMs: LIQUIDFUN_STATIONARY_WARMUP_MS,
				stationaryWarmupSteps,
				gravityWarmupSteps: LIQUIDFUN_GRAVITY_WARMUP_STEPS,
				sampleBatches: LIQUIDFUN_SAMPLE_BATCHES,
				stepsPerBatch: LIQUIDFUN_STEPS_PER_BATCH,
				rendered: false
			},
			measurement: {
				scope: 'Synchronous CPU wall-clock time around eight LiquidFun simulation steps per batch, including world.Step(), mixed-material particle contacts, elastic-ball triads, spring-vine pairs, and fracture bond checks.',
				clock: 'performance.now()',
				includes: 'JavaScript-to-WASM call overhead, LiquidFun mixed-material particle contact, elastic-triad and spring-pair solving, and fracture bond strain checks.',
				excludes: 'Canvas2D drawing, UI work, GPU timestamps, GPU queue completion/wait, and world construction.',
				simulationStepBatchCpuMs: batchDistribution,
				simulationStepCpuMsPerStep: summarizeSamples(
					samples.map((milliseconds) => milliseconds / LIQUIDFUN_STEPS_PER_BATCH)
				),
				simulationStepsPerSecondAtMedian:
					(LIQUIDFUN_STEPS_PER_BATCH * 1_000) / batchDistribution.median
			}
		};
	} finally {
		simulation.dispose();
	}
}
