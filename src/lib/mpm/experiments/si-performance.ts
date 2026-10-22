import { calculateMpmCflStepBatch } from '../kernels/step-scheduler.js';
import { deriveMpmMlsGpuCflBounds } from '../kernels/cfl-bounds.js';
import {
	createExperimentalMpmMlsAtomicGpuSolver,
	createMpmMlsGpuSolver,
	type MpmMlsGpuSolver
} from '../kernels/solver.js';
import type { MpmParticle } from '../reference/solver.js';
import {
	createSiFluidScene,
	SI_FLUID_FIXED_STEP_SECONDS,
	SI_FLUID_GRID,
	SI_FLUID_MAX_MICROSTEPS_PER_BATCH,
	SI_FLUID_PARTICLE_SPACING_M
} from '../scenes/si-fluid.js';

const BUFFER_USAGE = { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, QUERY_RESOLVE: 0x0200 };

export function summarizeSiWater(particles: readonly MpmParticle[], referenceArea: number) {
	if (particles.length === 0 || !Number.isFinite(referenceArea) || referenceArea <= 0) {
		throw new RangeError('Water diagnostics require particles and a positive reference area.');
	}
	const determinants: number[] = [];
	let area = 0;
	let maximumSpeed = 0;
	for (const particle of particles) {
		const { F, velocity, position, C } = particle;
		const determinant = F.m00 * F.m11 - F.m01 * F.m10;
		const values = [
			...Object.values(F), ...Object.values(C), position.x, position.y,
			velocity.x, velocity.y, particle.volume
		];
		if (
			values.some((value) => typeof value !== 'number' || !Number.isFinite(value)) ||
			!Number.isFinite(determinant) ||
			determinant <= 0 ||
			particle.volume === undefined ||
			particle.volume <= 0
		) {
			throw new Error('Non-finite or non-positive water state in SI diagnostics.');
		}
		determinants.push(determinant);
		area += particle.volume * determinant;
		maximumSpeed = Math.max(maximumSpeed, Math.hypot(velocity.x, velocity.y));
	}
	return {
		areaRatio: area / referenceArea,
		areaDriftPercent: 100 * (area / referenceArea - 1),
		determinantMin: Math.min(...determinants),
		determinantMax: Math.max(...determinants),
		densityErrorPercentP95: distribution(
			determinants.map((value) => 100 * Math.abs(1 / value - 1))
		).p95,
		maximumSpeed
	};
}

function distribution(values: readonly number[]) {
	if (values.length === 0 || values.some((value) => !Number.isFinite(value))) {
		throw new Error('A timing distribution requires finite samples.');
	}
	const sorted = [...values].sort((a, b) => a - b);
	const quantile = (fraction: number) =>
		sorted[Math.ceil(fraction * sorted.length) - 1];
	return {
		median: quantile(0.5),
		p95: quantile(0.95),
		min: sorted[0],
		max: sorted[sorted.length - 1],
		samples: sorted.length
	};
}

export interface SiPerformanceOptions {
	readonly courantLimit?: number;
	readonly simulationSeconds?: number;
	readonly render?: boolean;
	readonly transfer?: 'gather' | 'atomic';
	readonly particleRefinement?: 1 | 2 | 4;
}

/**
 * Run in a dedicated localhost browser tab, never in a running simulation.
 * GPU timestamps cover the normal batch; split-pass attribution is not used.
 */
export async function runSiPerformanceExperiment({
	courantLimit = 0.3,
	simulationSeconds = 3,
	render = false,
	transfer = 'gather',
	particleRefinement = 1
}: SiPerformanceOptions = {}, createSolver?: typeof createMpmMlsGpuSolver) {
	if (
		!Number.isFinite(courantLimit) || courantLimit <= 0 || courantLimit > 1 ||
		!Number.isFinite(simulationSeconds) || simulationSeconds < 0.1 ||
		simulationSeconds > 60 || typeof render !== 'boolean' ||
		(transfer !== 'gather' && transfer !== 'atomic') ||
		![1, 2, 4].includes(particleRefinement)
	) {
		throw new RangeError('Invalid SI performance experiment options.');
	}
	if (!navigator.gpu) throw new Error('WebGPU is required.');
	const adapter = await navigator.gpu.requestAdapter();
	if (!adapter || adapter.info.isFallbackAdapter || !adapter.features.has('timestamp-query')) {
		throw new Error('A hardware WebGPU adapter with timestamp-query is required.');
	}
	const canvas = document.createElement('canvas');
	canvas.width = 724;
	canvas.height = 724;
	document.body.replaceChildren(canvas);
	const requestDevice = GPUAdapter.prototype.requestDevice;
	let instrumentedDevice: GPUDevice | undefined;
	let restoreDevice: (() => void) | undefined;
	let querySet: GPUQuerySet | undefined;
	let resolved: GPUBuffer | undefined;
	let staging: GPUBuffer | undefined;
	let capture: { queryCount: number; dispatches: number } | null = null;
	let solver: MpmMlsGpuSolver | undefined;
	let allocatedBytes = 0;
	const gpuSamples: number[] = [];
	const batchSamples: number[] = [];
	const renderSamples: number[] = [];
	const stepSamples: number[] = [];
	const snapshots: ReturnType<typeof summarizeSiWater>[] = [];
	let retryCount = 0;
	let microsteps = 0;
	let timedSimulationSeconds = 0;
	let timedGpuSeconds = 0;
	let timedDispatchCount = 0;
	let timedGpuMilliseconds = 0;
	GPUAdapter.prototype.requestDevice = async function (descriptor = {}) {
		if (instrumentedDevice) throw new Error('The SI experiment must own exactly one GPU device.');
		const device = await requestDevice.call(this, {
			...descriptor,
			requiredFeatures: [...new Set<GPUFeatureName>([
				...(descriptor.requiredFeatures ?? []), 'timestamp-query'
			])]
		});
		instrumentedDevice = device;
		const createBuffer = device.createBuffer.bind(device);
		const createEncoder = device.createCommandEncoder.bind(device);
		querySet = device.createQuerySet({ type: 'timestamp', count: 128 });
		resolved = createBuffer({
			size: 1024, usage: BUFFER_USAGE.QUERY_RESOLVE | BUFFER_USAGE.COPY_SRC
		});
		staging = createBuffer({
			size: 1024, usage: BUFFER_USAGE.COPY_DST | BUFFER_USAGE.MAP_READ
		});
		device.createBuffer = (descriptor) => {
			allocatedBytes += descriptor.size;
			return createBuffer(descriptor);
		};
		device.createCommandEncoder = (descriptor) => {
			const encoder = createEncoder(descriptor);
			const recording = capture;
			if (!recording) return encoder;
			const begin = encoder.beginComputePass.bind(encoder);
			const finish = encoder.finish.bind(encoder);
			const queryStart = recording.queryCount;
			encoder.beginComputePass = (descriptor) => {
				if (descriptor?.label !== 'MLS-MPM simulation batch') return begin(descriptor);
				const index = recording.queryCount;
				if (index + 2 > 128) throw new Error('SI timestamp capacity exceeded.');
				recording.queryCount += 2;
				const pass = begin({
					...descriptor,
					timestampWrites: {
						querySet: querySet!,
						beginningOfPassWriteIndex: index,
						endOfPassWriteIndex: index + 1
					}
				});
				const dispatch = pass.dispatchWorkgroups.bind(pass);
				pass.dispatchWorkgroups = (...args) => {
					recording.dispatches += 1;
					dispatch(...args);
				};
				return pass;
			};
			encoder.finish = (descriptor) => {
				const count = recording.queryCount - queryStart;
				if (count > 0) {
					// Resolve offsets must be 256-byte aligned; copy only this encoder's queries.
					encoder.resolveQuerySet(querySet!, queryStart, count, resolved!, 0);
					encoder.copyBufferToBuffer(resolved!, 0, staging!, queryStart * 8, count * 8);
				}
				return finish(descriptor);
			};
			return encoder;
		};
		restoreDevice = () => {
			device.createBuffer = createBuffer;
			device.createCommandEncoder = createEncoder;
		};
		return device;
	};
	try {
		const initial = createSiFluidScene();
		const particlesPerInitial = particleRefinement ** 2;
		const scene = {
			config: {
				...initial.config,
				particleMass: initial.config.particleMass / particlesPerInitial,
				particleVolume: initial.config.particleVolume / particlesPerInitial
			},
			particles: initial.particles.flatMap((particle) =>
				Array.from({ length: particlesPerInitial }, (_, index) => ({
					...particle,
					volume: (particle.volume ?? initial.config.particleVolume) / particlesPerInitial,
					position: {
						x: particle.position.x + SI_FLUID_PARTICLE_SPACING_M *
							((index % particleRefinement + 0.5) / particleRefinement - 0.5),
						y: particle.position.y + SI_FLUID_PARTICLE_SPACING_M *
							((Math.floor(index / particleRefinement) + 0.5) / particleRefinement - 0.5)
					}
				}))
			)
		};
		const referenceArea = scene.particles.reduce(
			(sum, particle) => sum + (particle.volume ?? scene.config.particleVolume), 0
		);
		const bounds = deriveMpmMlsGpuCflBounds(scene.particles, scene.config);
		if (!bounds) throw new Error('Missing derived SI CFL bounds.');
		const factory = createSolver ?? (
			transfer === 'atomic' ? createExperimentalMpmMlsAtomicGpuSolver : createMpmMlsGpuSolver
		);
		solver = await factory(canvas, scene.particles, scene.config, {
			...bounds, courantLimit
		});
		const initialAllocatedBytes = allocatedBytes;
		let simulationTime = 0;
		let latestVelocityBound = 0;
		let firstTimedAt = 0;
		let lastTimedAt = 0;
		const requestedMacrosteps = Math.ceil(simulationSeconds / SI_FLUID_FIXED_STEP_SECONDS);
		const warmupMacrosteps = 12;
		const timestampEvery = Math.max(1, Math.floor(requestedMacrosteps / 36));
		let nextSnapshotTime = 1;
		for (let index = 0; index < requestedMacrosteps + warmupMacrosteps; index += 1) {
			const measuring = index >= warmupMacrosteps;
			const gravityPhase = Math.floor(simulationTime / 0.75) % 4;
			const gravity = gravityPhase === 1
				? { x: 9.81, y: 0 }
				: gravityPhase === 2 ? { x: -9.81, y: 0 } : { x: 0, y: 9.81 };
			const bounds = solver.cflBounds;
			if (!bounds) throw new Error('Missing derived SI CFL bounds.');
			const { batch, velocityBound } = calculateMpmCflStepBatch({
				accumulatorSeconds: SI_FLUID_FIXED_STEP_SECONDS,
				elapsedSeconds: 0,
				simulationSpeed: 1,
				fixedStepSeconds: SI_FLUID_FIXED_STEP_SECONDS,
				cellSize: SI_FLUID_GRID.cellSize,
				waveSpeedBound: bounds.waveSpeedBound,
				currentVelocityBound: Math.max(bounds.velocityBound, latestVelocityBound),
				courantLimit,
				maxMicrosteps: SI_FLUID_MAX_MICROSTEPS_PER_BATCH,
				gravityMagnitude: 9.81,
				oscillationAmplitude: 0,
				dragRadius: 0,
				dragStiffness: 0,
				dragDamping: 0
			});
			if (batch.macroStepCount !== 1) throw new Error('One SI macrostep must fit this experiment.');
			capture = measuring && (index - warmupMacrosteps) % timestampEvery === 0 &&
				gpuSamples.length < 36
				? { queryCount: 0, dispatches: 0 } : null;
			const started = performance.now();
			if (firstTimedAt === 0 && measuring) firstTimedAt = started;
			const measurement = await solver.stepCflBatchAdaptive(
				1, batch.macroDt, batch.microstepPlan, velocityBound,
				SI_FLUID_MAX_MICROSTEPS_PER_BATCH, gravity
			);
			const completedAt = performance.now();
			const recording = capture;
			capture = null;
			if (measurement.remainingDuration > 1e-12) {
				throw new Error(`Incomplete SI macrostep: ${measurement.remainingDuration} seconds remain.`);
			}
			latestVelocityBound = measurement.recommendedVelocityBound;
			simulationTime += measurement.simulatedDuration;
			if (render) {
				solver.render('surface');
				await solver.measureQueueCompletionMs();
			}
			lastTimedAt = performance.now();
			if (measuring) {
				timedSimulationSeconds += measurement.simulatedDuration;
				batchSamples.push(completedAt - started);
				renderSamples.push(lastTimedAt - started);
				stepSamples.push(measurement.attemptedMicroStepCount);
				retryCount += measurement.retryCount;
				microsteps += measurement.attemptedMicroStepCount;
			}
			if (recording) {
				if (recording.queryCount === 0) throw new Error('No simulation GPU timestamps were captured.');
				await staging!.mapAsync(1, 0, recording.queryCount * 8);
				const values = new BigUint64Array(staging!.getMappedRange(0, recording.queryCount * 8).slice(0));
				staging!.unmap();
				let milliseconds = 0;
				for (let query = 0; query < values.length; query += 2) {
					if (values[query + 1] <= values[query]) throw new Error('Invalid or unresolved GPU timestamps.');
					milliseconds += Number(values[query + 1] - values[query]) / 1e6;
				}
				gpuSamples.push(milliseconds);
				timedGpuMilliseconds += milliseconds;
				timedGpuSeconds += measurement.simulatedDuration;
				timedDispatchCount += recording.dispatches;
			}
			if (simulationTime >= nextSnapshotTime) {
				snapshots.push(summarizeSiWater(await solver.readParticles(), referenceArea));
				nextSnapshotTime += 1;
			}
		}
		await solver.checkNumericalStatus();
		const particles = await solver.readParticles();
		const quality = summarizeSiWater(particles, referenceArea);
		return {
			environment: {
				userAgent: navigator.userAgent,
				adapter: {
					vendor: adapter.info.vendor, architecture: adapter.info.architecture,
					isFallbackAdapter: adapter.info.isFallbackAdapter
				},
				timestampQuery: true,
				canvas: [canvas.width, canvas.height]
			},
			conditions: {
				particles: particles.length, courantLimit, render, transfer, particleRefinement,
				warmupMacrosteps, timedSimulationSeconds,
				gravity: '0.75s down/right/left/down in simulation time',
				initialAllocatedBytes
			},
			gpuBatchMs: distribution(gpuSamples),
			batchAndReadbackMs: distribution(batchSamples),
			batchReadbackAndRenderMs: distribution(renderSamples),
			attemptedMicrostepsPerBatch: distribution(stepSamples),
			gpuMillisecondsPerSimulationSecond: timedGpuMilliseconds / timedGpuSeconds,
			dispatchesPerSimulationSecond: timedDispatchCount / timedGpuSeconds,
			wholeRunSimulationWallRatio: timedSimulationSeconds / ((lastTimedAt - firstTimedAt) / 1000),
			timedBatchSimulationWallRatio: timedSimulationSeconds / (
				renderSamples.reduce((sum, value) => sum + value, 0) / 1000
			),
			retryCount, microsteps, finalCflBounds: solver.cflBounds,
			quality, snapshots,
			particleState: particles
		};
	} finally {
		capture = null;
		GPUAdapter.prototype.requestDevice = requestDevice;
		restoreDevice?.();
		try {
			solver?.dispose();
		} finally {
			staging?.destroy();
			resolved?.destroy();
			querySet?.destroy();
			canvas.remove();
		}
	}
}
