import { DEFAULT_MPM_COURANT_LIMIT } from '../kernels/cfl.js';
import { deriveMpmMlsGpuCflBounds } from '../kernels/cfl-bounds.js';
import { calculateMpmCflStepBatch } from '../kernels/step-scheduler.js';
import { meanSinusoidalAcceleration } from '../kernels/oscillation.js';
import {
	createMpmGate4Scene,
	createMpmGate4StepConfig,
	GATE4_FIXED_STEP_SECONDS,
	GATE4_GRID,
	GATE4_SCENE_SEED
} from '../scenes/gate4.js';
import type { MpmVector2 } from './solver.js';
import {
	MpmCpuReferenceSimulation,
	MPM_CPU_MAX_CFL_MICROSTEPS_PER_MANUAL_STEP,
	type MpmCpuWorkerRequest,
	type MpmCpuWorkerResponse
} from './cpu-simulation.js';

interface CpuSimulationWorkerScope {
	addEventListener(
		type: 'message',
		listener: (event: MessageEvent<MpmCpuWorkerRequest>) => void
	): void;
	postMessage(message: MpmCpuWorkerResponse): void;
}

const cpuWorkerScope = globalThis as unknown as CpuSimulationWorkerScope;
let simulation: MpmCpuReferenceSimulation | null = null;
let simulatedTimeSeconds = 0;

function calculateBatch(
	accumulatorSeconds: number,
	elapsedSeconds: number,
	simulationSpeed: number,
	maxMicrosteps: number,
	gravity: MpmVector2,
	oscillationAmplitude: number
): ReturnType<typeof calculateMpmCflStepBatch> {
	if (!simulation) {
		throw new Error('CPU MLS-MPM simulation is not initialized.');
	}
	const config = createMpmGate4StepConfig(
		gravity,
		GATE4_FIXED_STEP_SECONDS,
		simulation.fractureBonds
	);
	const cflBounds = deriveMpmMlsGpuCflBounds(simulation.particles, config);
	if (cflBounds === null) {
		throw new RangeError('The Gate 4 scene has no supported CFL wave-speed bound.');
	}
	return calculateMpmCflStepBatch({
		accumulatorSeconds,
		elapsedSeconds,
		simulationSpeed,
		fixedStepSeconds: GATE4_FIXED_STEP_SECONDS,
		cellSize: GATE4_GRID.cellSize,
		waveSpeedBound: cflBounds.waveSpeedBound,
		currentVelocityBound: cflBounds.velocityBound,
		gravityMagnitude: Math.hypot(gravity.x, gravity.y),
		oscillationAmplitude,
		dragRadius: 0,
		dragStiffness: 0,
		dragDamping: 0,
		courantLimit: DEFAULT_MPM_COURANT_LIMIT,
		maxMicrosteps,
		allowPartialBatch: true
	});
}

function advanceBatch(
	batch: ReturnType<typeof calculateMpmCflStepBatch>['batch'],
	gravity: MpmVector2,
	oscillationAmplitude: number
): void {
	if (!simulation) {
		throw new Error('CPU MLS-MPM simulation is not initialized.');
	}
	const startTimeSeconds = simulatedTimeSeconds;
	for (let macroStep = 0; macroStep < batch.macroStepCount; macroStep += 1) {
		for (
			let microstep = 0;
			microstep < batch.microstepPlan.stepCount;
			microstep += 1
		) {
			const microDt = batch.microstepPlan.microDt;
			const offsetSeconds = macroStep * batch.macroDt + microstep * microDt;
			const oscillation = meanSinusoidalAcceleration(
				startTimeSeconds + offsetSeconds,
				microDt,
				oscillationAmplitude
			);
			simulation.step(
				createMpmGate4StepConfig(
					{ x: gravity.x + oscillation, y: gravity.y },
					microDt,
					simulation.fractureBonds
				)
			);
		}
	}
	simulatedTimeSeconds += batch.simulatedSeconds;
}

function errorText(error: unknown): string {
	return error instanceof Error && error.message.trim()
		? error.message
		: 'CPU MLS-MPM worker failed with an unknown error.';
}

cpuWorkerScope.addEventListener('message', ({ data: request }) => {
	try {
		if (request.type === 'initialize') {
			simulation?.dispose();
			simulatedTimeSeconds = 0;
			simulation = new MpmCpuReferenceSimulation(
				createMpmGate4Scene(GATE4_SCENE_SEED, {
					liquidParticleDiameter: request.liquidParticleDiameter
				})
			);
			const initial = calculateBatch(
				0,
				GATE4_FIXED_STEP_SECONDS,
				1,
				MPM_CPU_MAX_CFL_MICROSTEPS_PER_MANUAL_STEP,
				request.gravity,
				0
			);
			if (
				initial.batch.macroStepCount !== 1 ||
				initial.batch.macroDt !== GATE4_FIXED_STEP_SECONDS
			) {
				throw new RangeError('The initial Gate 4 macrostep could not fit the CFL batch.');
			}
			advanceBatch(initial.batch, request.gravity, 0);
			cpuWorkerScope.postMessage({
				type: 'ready',
				generation: request.generation,
				snapshot: simulation.createRenderSnapshot(),
				remainderSeconds: initial.batch.remainderSeconds
			});
			return;
		}

		const startedAt = performance.now();
		const planned =
			request.type === 'advance'
				? calculateBatch(
						request.accumulatorSeconds,
						0,
						1,
						request.maxMicrosteps,
						request.gravity,
						request.oscillationAmplitude
					)
				: calculateBatch(
						0,
						GATE4_FIXED_STEP_SECONDS,
						1,
						request.maxMicrosteps,
						request.gravity,
						request.oscillationAmplitude
					);
		if (request.type === 'step' && planned.batch.macroStepCount !== 1) {
			throw new RangeError('A CFL-safe Gate 4 step could not be planned.');
		}
		advanceBatch(
			planned.batch,
			request.gravity,
			request.oscillationAmplitude
		);
		if (!simulation) {
			throw new Error('CPU MLS-MPM simulation was disposed during a worker batch.');
		}
		const workerMilliseconds = performance.now() - startedAt;
		const snapshot = simulation.createRenderSnapshot();
		cpuWorkerScope.postMessage({
			type: request.type === 'advance' ? 'advanced' : 'stepped',
			generation: request.generation,
			snapshot,
			simulatedSeconds: planned.batch.simulatedSeconds,
			remainderSeconds: planned.batch.remainderSeconds,
			microStepCount: planned.batch.microStepCount,
			workerMilliseconds
		});
	} catch (error) {
		cpuWorkerScope.postMessage({
			type: 'error',
			generation: request.generation,
			message: errorText(error)
		});
	}
});
