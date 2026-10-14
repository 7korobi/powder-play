import { createLiquidFunSimulation } from './simulation.js';
import {
	LIQUIDFUN_BOX2D_DOMAIN_METERS,
	LIQUIDFUN_BOX2D_GRAVITY,
	LIQUIDFUN_DT_SECONDS,
	LIQUIDFUN_GATE4_DOMAIN,
	LIQUIDFUN_GATE4_GRAVITY,
	LIQUIDFUN_GATE4_UNITS_PER_METER,
	LIQUIDFUN_GRAVITY_WARMUP_STEPS,
	LIQUIDFUN_PACKAGE_VERSION,
	LIQUIDFUN_SAMPLE_BATCHES,
	LIQUIDFUN_SCENE_SEED,
	LIQUIDFUN_SHARED_PARTICLE_DENSITY,
	LIQUIDFUN_STATIONARY_WARMUP_MS,
	LIQUIDFUN_STEPS_PER_BATCH,
	LIQUIDFUN_VELOCITY_ITERATIONS,
	LIQUIDFUN_POSITION_ITERATIONS,
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
				scene: 'Gate 4',
				seed: LIQUIDFUN_SCENE_SEED,
				seedHex: `0x${LIQUIDFUN_SCENE_SEED.toString(16)}`,
				sourceMpmNodeCount: simulation.sourceMpmNodeCount,
				liquidFunParticleCount: simulation.particleCount,
				waterParticleCount: simulation.waterParticleCount,
				oilViscousParticleCount: simulation.oilParticleCount,
				dynamicTriangleBodyCount: simulation.triangleBodyCount,
				staticWallBodyCount: simulation.staticWallBodyCount,
				omittedFractureNodeCount: simulation.omittedFractureNodeCount,
				grid: {
					cellsX: GATE4_GRID.cellsX,
					cellsY: GATE4_GRID.cellsY,
					cellSizeGate4Units: GATE4_GRID.cellSize
				},
				domainGate4Units: LIQUIDFUN_GATE4_DOMAIN,
				domainMeters: LIQUIDFUN_BOX2D_DOMAIN_METERS,
				gate4UnitsPerMeter: LIQUIDFUN_GATE4_UNITS_PER_METER,
				dtSeconds: LIQUIDFUN_DT_SECONDS,
				gravityGate4UnitsPerSecondSquared: LIQUIDFUN_GATE4_GRAVITY,
				gravityBox2dMetersPerSecondSquared: LIQUIDFUN_BOX2D_GRAVITY,
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
				scope: 'Synchronous CPU wall-clock time around eight LiquidFun world.Step() calls per batch.',
				clock: 'performance.now()',
				includes: 'JavaScript-to-WASM call overhead and LiquidFun/Box2D world.Step() work.',
				excludes: 'Canvas2D drawing, UI work, GPU timestamps, GPU queue completion/wait, and world construction.',
				worldStepBatchCpuMs: batchDistribution,
				worldStepCpuMsPerStep: summarizeSamples(
					samples.map((milliseconds) => milliseconds / LIQUIDFUN_STEPS_PER_BATCH)
				),
				stepsPerSecondAtMedian:
					(LIQUIDFUN_STEPS_PER_BATCH * 1_000) / batchDistribution.median
			}
		};
	} finally {
		simulation.dispose();
	}
}
