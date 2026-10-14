import type {
	MpmMatrix2,
	MpmMaterialParameters,
	MpmParticle,
	MpmVector2
} from '../reference/solver.js';
import { decodeMpmConstitutiveModel } from '../materials.js';
import { createMpmWebGpuSession } from '../platform/webgpu.js';
import {
	calculateMpmCflSafeDt,
	calculateMpmCflVelocityLimit,
	calculateMpmCflMicrosteps,
	type MpmCflMicrostepInput,
	type MpmCflMicrostepPlan
} from './cfl.js';
import {
	deriveMpmMlsGpuCflBounds,
	type MpmMlsGpuCflBounds as DerivedMpmMlsGpuCflBounds
} from './cfl-bounds.js';
import {
	calculateMpmMlsGpuBufferLayout,
	COMPUTE_WORKGROUP_SIZE,
	NUMERICAL_STATUS_BYTES,
	NUMERICAL_STATUS_WORDS,
	OBJECT_ID_STRIDE_BYTES,
	PARTICLE_FLOATS,
	packMpmMlsGpuStepParameters,
	PARTICLE_STRIDE_BYTES,
	RENDER_PARAMS_BYTES,
	SURFACE_RENDER_PARAMS_BYTES,
	SIMULATION_PARAMS_BYTES,
	validateMpmMlsGpuInput,
	type MpmMlsGpuBufferLayout,
	type MpmMlsGpuConfig,
	type MpmMlsGpuObjectRange,
	type ValidatedMpmMlsGpuInput
} from './layout.js';
import { MLS_MPM_COMPUTE_WGSL, MLS_MPM_RENDER_WGSL } from './shaders.js';

export type { MpmMlsGpuConfig } from './layout.js';

export interface MpmMlsGpuSelection {
	readonly particleIndex: number;
	readonly objectId: number;
	readonly anchor: MpmVector2;
}

export interface MpmMlsGpuSpeedMeasurement {
	readonly maximumSpeed: number;
	readonly sequence: number;
}

export interface MpmMlsGpuCflFailureDetails {
	readonly particleSpeed: number;
	readonly waveSpeed: number;
	readonly velocityLimit: number;
	readonly waveSpeedLimit: number;
}

export interface MpmMlsGpuCflBatchMeasurement extends MpmMlsGpuSpeedMeasurement {
	readonly recommendedVelocityBound: number;
	readonly attemptedMicroStepCount: number;
	readonly retryCount: number;
	readonly retrySubmissionMilliseconds: number;
	readonly readbackMilliseconds: number;
	readonly simulatedDuration: number;
	readonly remainingDuration: number;
}

const ERROR_DEFORMATION = 2;
const ERROR_DEFORMATION_SUBSTEP_LIMIT = 32;
const ERROR_CFL_BOUND = 16_384;
const MAX_CFL_STEP_RETRIES = 4;
const CFL_RECOVERY_BOUND_SAFETY_FACTOR = 1.25;

function isCflMicrostepCapacityError(error: unknown): boolean {
	return (
		error instanceof RangeError &&
		(/exceeds maxMicrosteps|microDt is too small to represent/.test(error.message))
	);
}

export interface MpmMlsGpuAppendResult {
	readonly requested: number;
	readonly accepted: number;
	readonly refused: number;
	readonly dropped: number;
	readonly activeParticleCount: number;
	readonly allocatedParticleCount: number;
	readonly capacity: number;
	readonly velocityBound: number;
	readonly reason?: string;
}

export interface MpmMlsGpuOutletBounds {
	readonly minX: number;
	readonly minY: number;
	readonly maxX: number;
	readonly maxY: number;
}

export type MpmMlsGpuCflBounds = DerivedMpmMlsGpuCflBounds &
	Pick<MpmCflMicrostepInput, 'courantLimit'>;

export type MpmMlsGpuRenderMode = 'surface' | 'particles';

export interface MpmMlsGpuSolver {
	readonly deviceLost: Promise<GPUDeviceLostInfo>;
	readonly particleCount: number;
	readonly allocatedParticleCount: number;
	readonly particleCapacity: number;
	readonly cflBounds: MpmMlsGpuCflBounds | null;
	step(dt: number, gravity: MpmVector2): void;
	stepBatch(stepCount: number, dt: number, gravity: MpmVector2): void;
	stepCflBatch(
		macroStepCount: number,
		macroDt: number,
		microstepPlan: MpmCflMicrostepPlan,
		velocityBound: number,
		maxMicrosteps: number,
		gravity: MpmVector2
	): void;
	stepCflBatchAdaptive(
		macroStepCount: number,
		macroDt: number,
		microstepPlan: MpmCflMicrostepPlan,
		velocityBound: number,
		maxMicrosteps: number,
		gravity: MpmVector2,
		gravityForDuration?: (offsetSeconds: number, durationSeconds: number) => MpmVector2
	): Promise<MpmMlsGpuCflBatchMeasurement>;
	render(mode?: MpmMlsGpuRenderMode): void;
	readParticles(particleIndices?: readonly number[]): Promise<readonly MpmParticle[]>;
	measureMaximumParticleSpeed(): Promise<MpmMlsGpuSpeedMeasurement>;
	checkNumericalStatus(): Promise<void>;
	measureQueueCompletionMs(): Promise<number>;
	appendParticles(particles: readonly MpmParticle[]): MpmMlsGpuAppendResult;
	removeParticlesInOutlet(bounds: MpmMlsGpuOutletBounds): Promise<number>;
	pick(point: MpmVector2, radius: number): Promise<MpmMlsGpuSelection | null>;
	beginDrag(selection: MpmMlsGpuSelection, target: MpmVector2, radius: number): void;
	updateDrag(target: MpmVector2): void;
	endDrag(): void;
	clone(selection: MpmMlsGpuSelection, offset: MpmVector2): boolean;
	dispose(): void;
}

export class MpmMlsGpuNumericalError extends Error {
	constructor(
		readonly flags: number,
		readonly particleIndex?: number,
		readonly diagnostics?: {
			readonly determinant: number;
			readonly deformationIncrement: number;
			readonly precedingDeterminant: number;
		},
		readonly particleState?: MpmParticle,
		readonly projectedDeformation?: MpmMatrix2,
		readonly cflDetails?: MpmMlsGpuCflFailureDetails,
		readonly completedMicrosteps?: number,
		readonly measuredMaximumSpeed?: number
	) {
		const details = [
			[1, 'invalid particle stencil or linked-list state'],
			[2, 'non-positive trial deformation determinant'],
			[4, 'P2G validation failed'],
			[8, 'non-finite grid state'],
			[16, 'non-finite G2P particle state'],
			[32, 'deformation update exceeded the adaptive substep limit'],
			[64, 'non-positive projected deformation determinant'],
			[128, 'invalid P2G material parameters'],
			[256, 'non-finite particle stress or momentum'],
			[1024, 'non-finite P2G node accumulation'],
			[2048, 'non-finite P2G node output'],
			[4096, 'invalid triangle contact field or reaction'],
			[8192, 'invalid bonded triangle state'],
			[16384, 'particle exceeded the enforced CFL wave-speed or velocity bound'],
			[32768, 'invalid persistent solid-fracture bond state']
		] as const;
		const message = details
			.filter(([flag]) => (flags & flag) !== 0)
			.map(([, description]) => description)
			.join('; ');
		const particleContext = particleIndex === undefined ? '' : ` at particle ${particleIndex}`;
		const diagnosticContext =
			diagnostics === undefined
				? ''
				: ` (det=${diagnostics.determinant}, deformation increment=${diagnostics.deformationIncrement}, preceding det=${diagnostics.precedingDeterminant})`;
		const stateContext =
			particleState === undefined
				? ''
				: ` (last valid state: position=(${particleState.position.x},${particleState.position.y}), velocity=(${particleState.velocity.x},${particleState.velocity.y}), C=(${particleState.C.m00},${particleState.C.m01},${particleState.C.m10},${particleState.C.m11}), F=(${particleState.F.m00},${particleState.F.m01},${particleState.F.m10},${particleState.F.m11}))`;
		const projectionContext =
			projectedDeformation === undefined
				? ''
				: ` (projected F=(${projectedDeformation.m00},${projectedDeformation.m01},${projectedDeformation.m10},${projectedDeformation.m11}))`;
		const cflContext =
			cflDetails === undefined
				? ''
				: ` (particle speed=${cflDetails.particleSpeed}, wave speed=${cflDetails.waveSpeed}, limits=(${cflDetails.velocityLimit}, ${cflDetails.waveSpeedLimit}))`;
		super(
			`GPU MLS-MPM rejected a step (${message || `status 0x${flags.toString(16)}`}${particleContext}${diagnosticContext}${stateContext}${projectionContext}${cflContext}); ` +
				'all particles were rolled back to the last valid state.'
		);
		this.name = 'MpmMlsGpuNumericalError';
	}
}

class MpmMlsGpuSurfaceError extends Error {
	constructor(readonly flags: number) {
		const details = [
			[1, 'invalid free-surface bucket state'],
			[2, 'invalid particle area or deformation determinant'],
			[4, 'non-finite free-surface occupancy accumulation']
		] as const;
		const message = details
			.filter(([flag]) => (flags & flag) !== 0)
			.map(([, description]) => description)
			.join('; ');
		super(
			`GPU free-surface occupancy generation failed (${message || `status 0x${flags.toString(16)}`}); ` +
				'particle state was not modified.'
		);
		this.name = 'MpmMlsGpuSurfaceError';
	}
}

interface MpmMlsGpuCflReadbackDiagnostics {
	readonly completedMicrosteps: number;
	readonly measuredMaximumSpeed: number;
}

function decodeNumericalError(
	snapshot: ArrayBuffer,
	offset: number,
	particleAt?: (particleIndex: number) => MpmParticle | undefined,
	cflReadback?: MpmMlsGpuCflReadbackDiagnostics
): MpmMlsGpuNumericalError | null {
	const status = new Uint32Array(snapshot, offset, NUMERICAL_STATUS_WORDS);
	if (status[0] === 0) return null;

	const particleIndex = status[1] === MAX_U32 ? undefined : status[1];
	const values = new Float32Array(
		snapshot,
		offset + 2 * Uint32Array.BYTES_PER_ELEMENT,
		4
	);
	const diagnostics =
		status[6] !== 1
			? undefined
			: {
					determinant: values[0],
					deformationIncrement: values[1],
					precedingDeterminant: values[2]
				};
	const projectedDeformation =
		status[6] !== 2
			? undefined
			: {
					m00: values[0],
					m01: values[1],
					m10: values[2],
					m11: values[3]
				};
	const cflDetails =
		status[6] !== 3
			? undefined
			: {
					particleSpeed: values[0],
					waveSpeed: values[1],
					velocityLimit: values[2],
					waveSpeedLimit: values[3]
				};

	return new MpmMlsGpuNumericalError(
		status[0],
		particleIndex,
		diagnostics,
		particleIndex === undefined ? undefined : particleAt?.(particleIndex),
		projectedDeformation,
		cflDetails,
		cflReadback?.completedMicrosteps,
		cflReadback?.measuredMaximumSpeed
	);
}

function decodeSurfaceError(
	snapshot: ArrayBuffer,
	offset: number
): MpmMlsGpuSurfaceError | null {
	const surfaceFlags = new Uint32Array(snapshot, offset, NUMERICAL_STATUS_WORDS)[
		SURFACE_STATUS_WORD_INDEX
	];
	return surfaceFlags === 0 ? null : new MpmMlsGpuSurfaceError(surfaceFlags);
}

function isRecoverableCflError(error: MpmMlsGpuNumericalError): boolean {
	if (error.flags === ERROR_CFL_BOUND) {
		const details = error.cflDetails;
		return (
			details !== undefined &&
			Number.isFinite(details.particleSpeed) &&
			Number.isFinite(details.waveSpeed) &&
			Number.isFinite(details.velocityLimit) &&
			Number.isFinite(details.waveSpeedLimit)
		);
	}
	return (
		(error.flags === ERROR_DEFORMATION ||
			error.flags === ERROR_DEFORMATION_SUBSTEP_LIMIT) &&
		Number.isFinite(error.measuredMaximumSpeed)
	);
}

interface MpmMlsGpuPipelines {
	readonly clearGrid: GPUComputePipeline;
	readonly clearSurfaceOccupancy: GPUComputePipeline;
	readonly prepareParticles: GPUComputePipeline;
	readonly buildBuckets: GPUComputePipeline;
	readonly buildTriangleFieldCandidates: GPUComputePipeline;
	readonly buildTriangleField: GPUComputePipeline;
	readonly gatherSurfaceOccupancy: GPUComputePipeline;
	readonly p2g: GPUComputePipeline;
	readonly updateGrid: GPUComputePipeline;
	readonly g2p: GPUComputePipeline;
	readonly validateCflParticles: GPUComputePipeline;
	readonly measureParticleSpeedPartials: GPUComputePipeline;
	readonly reduceParticleSpeedPartials: GPUComputePipeline;
	readonly applyTriangleContactImpulses: GPUComputePipeline;
	readonly solveBondedTriangles: GPUComputePipeline;
	readonly gatherTriangleContactStates: GPUComputePipeline;
	readonly gatherTrianglePairContacts: GPUComputePipeline;
	readonly applyTrianglePairContacts: GPUComputePipeline;
	readonly updateFractureBonds: GPUComputePipeline;
	readonly applyFractureParticleDamage: GPUComputePipeline;
	readonly removeParticlesInOutlet: GPUComputePipeline;
	readonly commitParticles: GPUComputePipeline;
	readonly pick: GPUComputePipeline;
	readonly cloneObject: GPUComputePipeline;
	readonly render: GPURenderPipeline;
	readonly surfaceRender: GPURenderPipeline;
}

interface MpmMlsGpuStepBindings {
	readonly gatherSurfaceOccupancy: GPUBindGroup;
	readonly prepareParticles: GPUBindGroup;
	readonly buildBuckets: GPUBindGroup;
	buildTriangleFieldCandidates: GPUBindGroup;
	buildTriangleField: GPUBindGroup;
	readonly p2g: GPUBindGroup;
	readonly updateGrid: GPUBindGroup;
	readonly g2p: GPUBindGroup;
	readonly validateCflParticles: GPUBindGroup;
	readonly measureParticleSpeedPartials: GPUBindGroup;
	readonly applyTriangleContactImpulses: GPUBindGroup;
	readonly solveBondedTriangles: GPUBindGroup;
	readonly gatherTriangleContactStates: GPUBindGroup;
	readonly gatherTrianglePairContacts: GPUBindGroup;
	readonly applyTrianglePairContacts: GPUBindGroup;
	readonly updateFractureBonds: GPUBindGroup;
	readonly applyFractureParticleDamage: GPUBindGroup;
	readonly removeParticlesInOutlet: GPUBindGroup;
	readonly commitParticles: GPUBindGroup;
	readonly pick: GPUBindGroup;
	readonly render: GPUBindGroup;
}

const INTERACTION_PARAMS_BYTES = 32;
const PICK_PARAMS_BYTES = 32;
const PICK_RESULT_BYTES = 16;
const CLONE_PARAMS_BYTES = 48;
const OUTLET_PARAMS_BYTES = 16;
const OUTLET_COUNTER_BYTES = Uint32Array.BYTES_PER_ELEMENT;
const SURFACE_STATUS_WORD_INDEX = 7;
const SURFACE_PARAMETER_DT = 1;
const MAX_U32 = 0xffff_ffff;
const TRIANGLE_FIELD_CANDIDATE_BODY_WORD_BITS = 32;
export const MPM_MLS_GPU_DRAG_STIFFNESS = 20;
export const MPM_MLS_GPU_DRAG_DAMPING = 8;
const BUFFER_USAGE = {
	MAP_READ: 0x0001,
	COPY_SRC: 0x0004,
	COPY_DST: 0x0008,
	UNIFORM: 0x0040,
	STORAGE: 0x0080
} as const satisfies Record<string, GPUBufferUsageFlags>;
const MAP_MODE_READ = 0x0001;

export async function createMpmMlsGpuSolver(
	canvas: HTMLCanvasElement,
	particles: readonly MpmParticle[],
	config: MpmMlsGpuConfig,
	cflBounds?: MpmMlsGpuCflBounds
): Promise<MpmMlsGpuSolver> {
	if (canvas === null || typeof canvas?.getContext !== 'function') {
		throw new TypeError('canvas must be an HTML canvas with getContext()');
	}
	const initialState = validateMpmMlsGpuInput(particles, config);
	const derivedCflBounds = deriveMpmMlsGpuCflBounds(particles, initialState.config);
	const certifiedCflBounds = validateCflBounds(
		cflBounds,
		initialState,
		derivedCflBounds
	);
	const session = await createMpmWebGpuSession(canvas);
	const ownedBuffers = new Set<GPUBuffer>();

	try {
		const layout = calculateMpmMlsGpuBufferLayout(initialState, session.device.limits);
		const runtime = await withDeviceSetupScopes(
			session.device,
			'GPU MLS-MPM pipeline and buffer setup',
			() =>
				createRuntime(
					session.device,
					session.format,
					initialState,
					layout,
					ownedBuffers
				)
		);
		return createSolver(
			session,
			initialState,
			layout,
			runtime,
			ownedBuffers,
			certifiedCflBounds
		);
	} catch (cause) {
		const cleanupErrors = destroyBuffers(ownedBuffers);
		try {
			session.dispose();
		} catch (cleanupError) {
			cleanupErrors.push(cleanupError);
		}
		if (cleanupErrors.length > 0) {
			throw new AggregateError(
				[cause, ...cleanupErrors],
				'GPU MLS-MPM setup failed and cleanup also encountered errors.'
			);
		}
		throw new Error(
			`Failed to initialize the GPU MLS-MPM solver: ${describeError(cause)}`,
			{ cause }
		);
	}
}

function validateCflBounds(
	bounds: MpmMlsGpuCflBounds | undefined,
	state: ValidatedMpmMlsGpuInput,
	derived: DerivedMpmMlsGpuCflBounds | null
): MpmMlsGpuCflBounds | null {
	if (bounds === undefined) return derived;
	if (typeof bounds !== 'object' || bounds === null) {
		throw new RangeError('cflBounds must be an object of valid speed bounds.');
	}
	if (derived === null) {
		throw new RangeError(
			'this solver configuration has no derived CFL bounds for its constitutive material.'
		);
	}
	if (!Number.isFinite(bounds.waveSpeedBound) || bounds.waveSpeedBound < 0) {
		throw new RangeError('cflBounds.waveSpeedBound must be finite and non-negative.');
	}
	if (!Number.isFinite(bounds.velocityBound) || bounds.velocityBound < 0) {
		throw new RangeError('cflBounds.velocityBound must be finite and non-negative.');
	}
	if (
		bounds.courantLimit !== undefined &&
		(!Number.isFinite(bounds.courantLimit) || bounds.courantLimit <= 0)
	) {
		throw new RangeError('cflBounds.courantLimit must be positive and finite.');
	}
	if (bounds.waveSpeedBound < derived.waveSpeedBound) {
		throw new RangeError(
			'cflBounds.waveSpeedBound must not be lower than the material-derived initial bound.'
		);
	}
	if (bounds.velocityBound < derived.velocityBound) {
		throw new RangeError(
			'cflBounds.velocityBound must not be lower than the configured initial-state bound.'
		);
	}

	for (let particleIndex = 0; particleIndex < state.particleCount; particleIndex += 1) {
		const offset = particleIndex * PARTICLE_FLOATS;
		const speed = Math.hypot(
			state.packedParticles[offset + 2],
			state.packedParticles[offset + 3]
		);
		if (speed > bounds.velocityBound) {
			throw new RangeError(
				`initial velocity at particle ${particleIndex} exceeds cflBounds.velocityBound.`
			);
		}
	}

	return Object.freeze({
		waveSpeedBound: bounds.waveSpeedBound,
		velocityBound: bounds.velocityBound,
		...(bounds.courantLimit === undefined
			? {}
			: { courantLimit: bounds.courantLimit })
	});
}

interface MpmMlsGpuRuntime {
	readonly buffers: {
		readonly simulationParams: GPUBuffer;
		readonly interactionParams: GPUBuffer;
		readonly renderParams: GPUBuffer;
		readonly surfaceRenderParams: GPUBuffer;
		readonly stateError: GPUBuffer;
		readonly bucketHeads: GPUBuffer;
		readonly bucketNext: GPUBuffer;
		readonly gridNodes: GPUBuffer;
		readonly surfaceFields: GPUBuffer;
		triangleFieldCandidates: GPUBuffer;
		readonly triangleFieldNodes: GPUBuffer;
		readonly bondedBodyStarts: GPUBuffer;
		readonly triangleContactStates: GPUBuffer;
		readonly trianglePairContactCache: GPUBuffer;
		readonly trianglePairContactCandidates: GPUBuffer;
		readonly fractureBonds: GPUBuffer;
		readonly fractureBondTopology: GPUBuffer;
		readonly preparedAffines: GPUBuffer;
		readonly particleObjectIds: GPUBuffer;
		readonly outletParams: GPUBuffer;
		readonly outletRemovalCounter: GPUBuffer;
		readonly pickParams: GPUBuffer;
		readonly pickResult: GPUBuffer;
		readonly cloneParams: GPUBuffer;
		readonly cflSpeedPartials: GPUBuffer;
		readonly cflMaximumSpeed: GPUBuffer;
		readonly cflStepProgress: GPUBuffer;
		readonly particles: readonly [GPUBuffer, GPUBuffer];
	};
	readonly pipelines: MpmMlsGpuPipelines;
	readonly clearGrid: GPUBindGroup;
	readonly clearSurfaceOccupancy: GPUBindGroup;
	readonly surfaceRender: GPUBindGroup;
	readonly bindings: [MpmMlsGpuStepBindings, MpmMlsGpuStepBindings];
	readonly cloneBindings: readonly [GPUBindGroup, GPUBindGroup];
	readonly reduceParticleSpeedPartials: GPUBindGroup;
}

function createTriangleFieldBindings(
	device: GPUDevice,
	pipelines: MpmMlsGpuPipelines,
	simulationParams: GPUBuffer,
	input: GPUBuffer,
	stateError: GPUBuffer,
	triangleFieldNodes: GPUBuffer,
	bondedBodyStarts: GPUBuffer,
	triangleFieldCandidates: GPUBuffer
): Pick<
	MpmMlsGpuStepBindings,
	'buildTriangleFieldCandidates' | 'buildTriangleField'
> {
	return {
		buildTriangleFieldCandidates: createBindGroup(
			device,
			pipelines.buildTriangleFieldCandidates,
			[
				bufferEntry(0, simulationParams),
				bufferEntry(1, input),
				bufferEntry(17, bondedBodyStarts),
				bufferEntry(28, triangleFieldCandidates)
			]
		),
		buildTriangleField: createBindGroup(device, pipelines.buildTriangleField, [
			bufferEntry(0, simulationParams),
			bufferEntry(1, input),
			bufferEntry(6, stateError),
			bufferEntry(16, triangleFieldNodes),
			bufferEntry(17, bondedBodyStarts),
			bufferEntry(28, triangleFieldCandidates)
		])
	};
}

async function createRuntime(
	device: GPUDevice,
	format: GPUTextureFormat,
	state: ValidatedMpmMlsGpuInput,
	layout: MpmMlsGpuBufferLayout,
	ownedBuffers: Set<GPUBuffer>
): Promise<MpmMlsGpuRuntime> {
	const computeModule = device.createShaderModule({
		label: 'MLS-MPM compute shaders',
		code: MLS_MPM_COMPUTE_WGSL
	});
	const renderModule = device.createShaderModule({
		label: 'MLS-MPM particle renderer',
		code: MLS_MPM_RENDER_WGSL
	});
	await assertShaderModuleCompiles(computeModule, 'MLS-MPM compute shaders');
	await assertShaderModuleCompiles(renderModule, 'MLS-MPM particle renderer');

	const pipelines: MpmMlsGpuPipelines = {
		clearGrid: await createComputePipeline(device, computeModule, 'clearGrid'),
		clearSurfaceOccupancy: await createComputePipeline(
			device,
			computeModule,
			'clearSurfaceOccupancy'
		),
		prepareParticles: await createComputePipeline(device, computeModule, 'prepareParticles'),
		buildBuckets: await createComputePipeline(device, computeModule, 'buildBuckets'),
		buildTriangleFieldCandidates: await createComputePipeline(
			device,
			computeModule,
			'buildTriangleFieldCandidates'
		),
		buildTriangleField: await createComputePipeline(
			device,
			computeModule,
			'buildTriangleField'
		),
		gatherSurfaceOccupancy: await createComputePipeline(
			device,
			computeModule,
			'gatherSurfaceOccupancy'
		),
		p2g: await createComputePipeline(device, computeModule, 'p2g'),
		updateGrid: await createComputePipeline(device, computeModule, 'updateGrid'),
		g2p: await createComputePipeline(device, computeModule, 'g2p'),
		validateCflParticles: await createComputePipeline(
			device,
			computeModule,
			'validateCflParticles'
		),
		measureParticleSpeedPartials: await createComputePipeline(
			device,
			computeModule,
			'measureParticleSpeedPartials'
		),
		reduceParticleSpeedPartials: await createComputePipeline(
			device,
			computeModule,
			'reduceParticleSpeedPartials'
		),
		applyTriangleContactImpulses: await createComputePipeline(
			device,
			computeModule,
			'applyTriangleContactImpulses'
		),
		solveBondedTriangles: await createComputePipeline(
			device,
			computeModule,
			'solveBondedTriangles'
		),
		gatherTriangleContactStates: await createComputePipeline(
			device,
			computeModule,
			'gatherTriangleContactStates'
		),
		gatherTrianglePairContacts: await createComputePipeline(
			device,
			computeModule,
			'gatherTrianglePairContacts'
		),
		applyTrianglePairContacts: await createComputePipeline(
			device,
			computeModule,
			'applyTrianglePairContacts'
		),
		updateFractureBonds: await createComputePipeline(
			device,
			computeModule,
			'updateFractureBonds'
		),
		applyFractureParticleDamage: await createComputePipeline(
			device,
			computeModule,
			'applyFractureParticleDamage'
		),
		removeParticlesInOutlet: await createComputePipeline(
			device,
			computeModule,
			'removeParticlesInOutlet'
		),
		commitParticles: await createComputePipeline(device, computeModule, 'commitParticles'),
		pick: await createComputePipeline(device, computeModule, 'pickParticle'),
		cloneObject: await createComputePipeline(device, computeModule, 'cloneObject'),
		render: await createRenderPipeline(device, renderModule, format),
		surfaceRender: await createRenderPipeline(device, renderModule, format, {
			label: 'MLS-MPM water/oil surface render pipeline',
			vertexEntryPoint: 'surfaceVertexMain',
			fragmentEntryPoint: 'surfaceFragmentMain'
		})
	};

	const createBuffer = (label: string, size: number, usage: GPUBufferUsageFlags) => {
		const buffer = device.createBuffer({ label, size, usage });
		ownedBuffers.add(buffer);
		return buffer;
	};
	const particleBufferUsage =
		BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_SRC | BUFFER_USAGE.COPY_DST;
	const cflSpeedPartialCount = Math.max(
		1,
		Math.ceil(layout.particleCapacity / COMPUTE_WORKGROUP_SIZE)
	);

	const buffers = {
		simulationParams: createBuffer(
			'MLS-MPM simulation parameters',
			SIMULATION_PARAMS_BYTES,
			BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST
		),
		interactionParams: createBuffer(
			'MLS-MPM interaction parameters',
			INTERACTION_PARAMS_BYTES,
			BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST
		),
		renderParams: createBuffer(
			'MLS-MPM render parameters',
			RENDER_PARAMS_BYTES,
			BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST
		),
		surfaceRenderParams: createBuffer(
			'MLS-MPM free-surface render parameters',
			SURFACE_RENDER_PARAMS_BYTES,
			BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST
		),
		stateError: createBuffer(
			'MLS-MPM numerical status',
			NUMERICAL_STATUS_BYTES,
			BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_SRC | BUFFER_USAGE.COPY_DST
		),
		bucketHeads: createBuffer(
			'MLS-MPM linked-list bucket heads',
			layout.bucketHeadBytes,
			BUFFER_USAGE.STORAGE
		),
		bucketNext: createBuffer(
			'MLS-MPM linked-list entries',
			layout.bucketNextBytes,
			BUFFER_USAGE.STORAGE
		),
		gridNodes: createBuffer(
			'MLS-MPM halo grid state',
			layout.gridStateBytes,
			BUFFER_USAGE.STORAGE
		),
		surfaceFields: createBuffer(
			'MLS-MPM water/oil occupancy fields',
			layout.surfaceFieldBytes,
			BUFFER_USAGE.STORAGE
		),
		triangleFieldCandidates: createBuffer(
			'MLS-MPM triangle field candidate bitset',
			layout.triangleFieldCandidateBytes,
			BUFFER_USAGE.STORAGE
		),
		triangleFieldNodes: createBuffer(
			'MLS-MPM triangle contact field',
			layout.triangleFieldBytes,
			BUFFER_USAGE.STORAGE
		),
		bondedBodyStarts: createBuffer(
			'MLS-MPM bonded triangle body indices',
			layout.bondedBodyStartBytes,
			BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_DST
		),
		triangleContactStates: createBuffer(
			'MLS-MPM triangle-pair contact states',
			layout.triangleContactStateBytes,
			BUFFER_USAGE.STORAGE
		),
		trianglePairContactCache: createBuffer(
			'MLS-MPM triangle-pair contact cache',
			layout.trianglePairContactCacheBytes,
			BUFFER_USAGE.STORAGE
		),
		trianglePairContactCandidates: createBuffer(
			'MLS-MPM triangle-pair contact candidate bitset',
			layout.trianglePairContactCandidateBytes,
			BUFFER_USAGE.STORAGE
		),
		fractureBonds: createBuffer(
			'MLS-MPM persistent fracture bond state',
			layout.fractureBondBufferBytes,
			BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_DST
		),
		fractureBondTopology: createBuffer(
			'MLS-MPM fracture bond adjacency',
			layout.fractureBondTopologyBufferBytes,
			BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_DST
		),
		preparedAffines: createBuffer(
			'MLS-MPM prepared affine momentum',
			layout.preparedAffineBytes,
			BUFFER_USAGE.STORAGE
		),
		particleObjectIds: createBuffer(
			'MLS-MPM particle object ids',
			layout.objectIdBufferBytes,
			particleBufferUsage
		),
		outletParams: createBuffer(
			'MLS-MPM outlet bounds',
			OUTLET_PARAMS_BYTES,
			BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST
		),
		outletRemovalCounter: createBuffer(
			'MLS-MPM outlet removal count',
			OUTLET_COUNTER_BYTES,
			BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_SRC | BUFFER_USAGE.COPY_DST
		),
		pickParams: createBuffer(
			'MLS-MPM on-demand particle pick parameters',
			PICK_PARAMS_BYTES,
			BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST
		),
		pickResult: createBuffer(
			'MLS-MPM on-demand particle pick result',
			PICK_RESULT_BYTES,
			BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_SRC | BUFFER_USAGE.COPY_DST
		),
		cloneParams: createBuffer(
			'MLS-MPM GPU object clone parameters',
			CLONE_PARAMS_BYTES,
			BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST
		),
		cflSpeedPartials: createBuffer(
			'MLS-MPM CFL maximum-speed partials',
			cflSpeedPartialCount * Float32Array.BYTES_PER_ELEMENT,
			BUFFER_USAGE.STORAGE
		),
		cflMaximumSpeed: createBuffer(
			'MLS-MPM CFL maximum-speed scalar',
			Float32Array.BYTES_PER_ELEMENT,
			BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_SRC
		),
		cflStepProgress: createBuffer(
			'MLS-MPM CFL committed microstep count',
			Uint32Array.BYTES_PER_ELEMENT,
			BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_SRC | BUFFER_USAGE.COPY_DST
		),
		particles: [
			createBuffer(
				'MLS-MPM particle state A',
				layout.particleBufferBytes,
				particleBufferUsage
			),
			createBuffer(
				'MLS-MPM particle state B',
				layout.particleBufferBytes,
				particleBufferUsage
			)
		] as const
	};

	device.queue.writeBuffer(
		buffers.stateError,
		0,
		new Uint32Array([0, MAX_U32, 0, 0, 0, 0, 0, 0])
	);
	// Surface bucketing needs the grid/count fields before the first simulation step.
	// This valid placeholder is replaced by the caller's full physics parameters before stepping.
	device.queue.writeBuffer(
		buffers.simulationParams,
		0,
		packMpmMlsGpuStepParameters(state, layout, SURFACE_PARAMETER_DT, { x: 0, y: 0 })
	);
	device.queue.writeBuffer(buffers.interactionParams, 0, new Float32Array(8));
	device.queue.writeBuffer(buffers.pickResult, 0, new Uint32Array([MAX_U32, 0, 0, 0]));
	device.queue.writeBuffer(
		buffers.outletRemovalCounter,
		0,
		new Uint32Array([0])
	);
	device.queue.writeBuffer(
		buffers.renderParams,
		0,
		createRenderParameters(state, 'particles')
	);
	device.queue.writeBuffer(
		buffers.surfaceRenderParams,
		0,
		createSurfaceRenderParameters(state, layout)
	);
	if (layout.particleStateBytes > 0) {
		device.queue.writeBuffer(buffers.particles[0], 0, state.packedParticles);
		device.queue.writeBuffer(buffers.particleObjectIds, 0, state.packedObjectIds);
	}
	if (state.bondedBodyStarts.length > 0) {
		device.queue.writeBuffer(
			buffers.bondedBodyStarts,
			0,
			state.bondedBodyStarts
		);
	}
	if (state.packedFractureBonds.byteLength > 0) {
		device.queue.writeBuffer(buffers.fractureBonds, 0, state.packedFractureBonds);
		device.queue.writeBuffer(
			buffers.fractureBondTopology,
			0,
			state.packedFractureBondTopology
		);
	}

	const clearGrid = createBindGroup(device, pipelines.clearGrid, [
		bufferEntry(0, buffers.simulationParams),
		bufferEntry(3, buffers.bucketHeads),
		bufferEntry(5, buffers.gridNodes),
		bufferEntry(16, buffers.triangleFieldNodes)
	]);
	const clearSurfaceOccupancy = createBindGroup(
		device,
		pipelines.clearSurfaceOccupancy,
		[
			bufferEntry(0, buffers.simulationParams),
			bufferEntry(3, buffers.bucketHeads),
			bufferEntry(15, buffers.surfaceFields)
		]
	);
	const surfaceRender = createBindGroup(device, pipelines.surfaceRender, [
		bufferEntry(2, buffers.surfaceFields),
		bufferEntry(3, buffers.surfaceRenderParams)
	]);
	const createStepBindings = (inputIndex: 0 | 1): MpmMlsGpuStepBindings => {
		const inputBuffers = buffers.particles;
		const input = inputBuffers[inputIndex];
		const output = inputBuffers[1 - inputIndex];
		const prepareParticles = createBindGroup(device, pipelines.prepareParticles, [
			bufferEntry(0, buffers.simulationParams),
			bufferEntry(1, input),
			bufferEntry(6, buffers.stateError),
			bufferEntry(14, buffers.preparedAffines)
		]);
		const buildBuckets = createBindGroup(device, pipelines.buildBuckets, [
			bufferEntry(0, buffers.simulationParams),
			bufferEntry(1, input),
			bufferEntry(3, buffers.bucketHeads),
			bufferEntry(4, buffers.bucketNext),
			bufferEntry(6, buffers.stateError)
		]);
		const triangleFieldBindings = createTriangleFieldBindings(
			device,
			pipelines,
			buffers.simulationParams,
			input,
			buffers.stateError,
			buffers.triangleFieldNodes,
			buffers.bondedBodyStarts,
			buffers.triangleFieldCandidates
		);
		const gatherSurfaceOccupancy = createBindGroup(
			device,
			pipelines.gatherSurfaceOccupancy,
			[
				bufferEntry(0, buffers.simulationParams),
				bufferEntry(1, input),
				bufferEntry(3, buffers.bucketHeads),
				bufferEntry(4, buffers.bucketNext),
				bufferEntry(6, buffers.stateError),
				bufferEntry(15, buffers.surfaceFields)
			]
		);
		const p2g = createBindGroup(device, pipelines.p2g, [
			bufferEntry(0, buffers.simulationParams),
			bufferEntry(1, input),
			bufferEntry(3, buffers.bucketHeads),
			bufferEntry(4, buffers.bucketNext),
			bufferEntry(5, buffers.gridNodes),
			bufferEntry(6, buffers.stateError),
			bufferEntry(14, buffers.preparedAffines)
		]);
		const updateGrid = createBindGroup(device, pipelines.updateGrid, [
			bufferEntry(0, buffers.simulationParams),
			bufferEntry(1, input),
			bufferEntry(5, buffers.gridNodes),
			bufferEntry(6, buffers.stateError),
			bufferEntry(7, buffers.interactionParams),
			bufferEntry(16, buffers.triangleFieldNodes)
		]);
		const g2p = createBindGroup(device, pipelines.g2p, [
			bufferEntry(0, buffers.simulationParams),
			bufferEntry(1, input),
			bufferEntry(2, output),
			bufferEntry(5, buffers.gridNodes),
			bufferEntry(6, buffers.stateError),
			bufferEntry(16, buffers.triangleFieldNodes)
		]);
		const validateCflParticles = createBindGroup(
			device,
			pipelines.validateCflParticles,
			[
				bufferEntry(0, buffers.simulationParams),
				bufferEntry(1, output),
				bufferEntry(6, buffers.stateError)
			]
		);
		const measureParticleSpeedPartials = createBindGroup(
			device,
			pipelines.measureParticleSpeedPartials,
			[
				bufferEntry(0, buffers.simulationParams),
				bufferEntry(1, input),
				bufferEntry(18, buffers.cflSpeedPartials)
			]
		);
		const applyTriangleContactImpulses = createBindGroup(
			device,
			pipelines.applyTriangleContactImpulses,
			[
				bufferEntry(0, buffers.simulationParams),
				bufferEntry(1, input),
				bufferEntry(2, output),
				bufferEntry(6, buffers.stateError),
				bufferEntry(16, buffers.triangleFieldNodes),
				bufferEntry(17, buffers.bondedBodyStarts)
			]
		);
		const solveBondedTriangles = createBindGroup(
			device,
			pipelines.solveBondedTriangles,
			[
				bufferEntry(0, buffers.simulationParams),
				bufferEntry(1, input),
				bufferEntry(2, output),
				bufferEntry(6, buffers.stateError),
				bufferEntry(17, buffers.bondedBodyStarts)
			]
		);
		const gatherTriangleContactStates = createBindGroup(
			device,
			pipelines.gatherTriangleContactStates,
			[
				bufferEntry(0, buffers.simulationParams),
				bufferEntry(1, input),
				bufferEntry(2, output),
				bufferEntry(6, buffers.stateError),
				bufferEntry(8, buffers.particleObjectIds),
				bufferEntry(17, buffers.bondedBodyStarts),
				bufferEntry(25, buffers.triangleContactStates),
				bufferEntry(27, buffers.trianglePairContactCandidates)
			]
		);
		const gatherTrianglePairContacts = createBindGroup(
			device,
			pipelines.gatherTrianglePairContacts,
			[
				bufferEntry(0, buffers.simulationParams),
				bufferEntry(1, input),
				bufferEntry(6, buffers.stateError),
				bufferEntry(8, buffers.particleObjectIds),
				bufferEntry(17, buffers.bondedBodyStarts),
				bufferEntry(25, buffers.triangleContactStates),
				bufferEntry(26, buffers.trianglePairContactCache),
				bufferEntry(27, buffers.trianglePairContactCandidates)
			]
		);
		const applyTrianglePairContacts = createBindGroup(
			device,
			pipelines.applyTrianglePairContacts,
			[
				bufferEntry(0, buffers.simulationParams),
				bufferEntry(1, input),
				bufferEntry(2, output),
				bufferEntry(6, buffers.stateError),
				bufferEntry(8, buffers.particleObjectIds),
				bufferEntry(17, buffers.bondedBodyStarts),
				bufferEntry(25, buffers.triangleContactStates),
				bufferEntry(26, buffers.trianglePairContactCache),
				bufferEntry(27, buffers.trianglePairContactCandidates)
			]
		);
		const updateFractureBonds = createBindGroup(
			device,
			pipelines.updateFractureBonds,
			[
				bufferEntry(0, buffers.simulationParams),
				bufferEntry(2, output),
				bufferEntry(6, buffers.stateError),
				bufferEntry(20, buffers.fractureBonds)
			]
		);
		const applyFractureParticleDamage = createBindGroup(
			device,
			pipelines.applyFractureParticleDamage,
			[
				bufferEntry(0, buffers.simulationParams),
				bufferEntry(2, output),
				bufferEntry(6, buffers.stateError),
				bufferEntry(20, buffers.fractureBonds),
				bufferEntry(21, buffers.fractureBondTopology)
			]
		);
		const removeParticlesInOutlet = createBindGroup(
			device,
			pipelines.removeParticlesInOutlet,
			[
				bufferEntry(0, buffers.simulationParams),
				bufferEntry(1, input),
				bufferEntry(2, output),
				bufferEntry(22, buffers.outletParams),
				bufferEntry(23, buffers.outletRemovalCounter)
			]
		);
		const commitParticles = createBindGroup(device, pipelines.commitParticles, [
			bufferEntry(0, buffers.simulationParams),
			bufferEntry(1, input),
			bufferEntry(2, output),
			bufferEntry(6, buffers.stateError),
			bufferEntry(24, buffers.cflStepProgress)
		]);
		const render = createBindGroup(device, pipelines.render, [
			bufferEntry(0, input),
			bufferEntry(1, buffers.renderParams)
		]);
		const pick = createBindGroup(device, pipelines.pick, [
			bufferEntry(1, input),
			bufferEntry(8, buffers.particleObjectIds),
			bufferEntry(9, buffers.pickParams),
			bufferEntry(10, buffers.pickResult)
		]);
		return {
			gatherSurfaceOccupancy,
			prepareParticles,
			buildBuckets,
			...triangleFieldBindings,
			p2g,
			updateGrid,
			g2p,
			validateCflParticles,
			measureParticleSpeedPartials,
			applyTriangleContactImpulses,
			solveBondedTriangles,
			gatherTriangleContactStates,
			gatherTrianglePairContacts,
			applyTrianglePairContacts,
			updateFractureBonds,
			applyFractureParticleDamage,
			removeParticlesInOutlet,
			commitParticles,
			pick,
			render
		};
	};
	const createCloneBinding = (inputIndex: 0 | 1): GPUBindGroup =>
		createBindGroup(device, pipelines.cloneObject, [
			bufferEntry(11, buffers.particles[inputIndex]),
			bufferEntry(12, buffers.particleObjectIds),
			bufferEntry(13, buffers.cloneParams),
			bufferEntry(17, buffers.bondedBodyStarts)
		]);
	const bindings: [MpmMlsGpuStepBindings, MpmMlsGpuStepBindings] = [
		createStepBindings(0),
		createStepBindings(1)
	];
	const cloneBindings: readonly [GPUBindGroup, GPUBindGroup] = [
		createCloneBinding(0),
		createCloneBinding(1)
	];
	const reduceParticleSpeedPartials = createBindGroup(
		device,
		pipelines.reduceParticleSpeedPartials,
		[
			bufferEntry(0, buffers.simulationParams),
			bufferEntry(18, buffers.cflSpeedPartials),
			bufferEntry(19, buffers.cflMaximumSpeed)
		]
	);

	return {
		buffers,
		pipelines,
		clearGrid,
		clearSurfaceOccupancy,
		surfaceRender,
		bindings,
		cloneBindings,
		reduceParticleSpeedPartials
	};
}

function createSolver(
	session: Awaited<ReturnType<typeof createMpmWebGpuSession>>,
	state: ValidatedMpmMlsGpuInput,
	layout: MpmMlsGpuBufferLayout,
	runtime: MpmMlsGpuRuntime,
	ownedBuffers: Set<GPUBuffer>,
	cflBounds: MpmMlsGpuCflBounds | null
): MpmMlsGpuSolver {
	let currentBuffer = 0;
	let particleSlotCount = state.particleCount;
	let activeParticleCount = state.particleCount;
	let disposed = false;
	let lostInfo: GPUDeviceLostInfo | null = null;
	let pickPending = false;
	let removalPending = false;
	let hasRemovedParticles = false;
	let cflSpeedReadbacksPending = 0;
	let activeDrag: { readonly radius: number } | null = null;
	let cflSpeedMeasurementSubmitted = false;
	let cflStateCflUsable = cflBounds !== null;
	let lastScheduledVelocityBound = cflBounds?.velocityBound ?? null;
	let latestCflBatchSequence = 0;
	let latestCflBatchMicrostepCount = 0;
	let latestCompletedCflMeasurementSequence = 0;
	let currentRenderMode: MpmMlsGpuRenderMode = 'particles';
	let bondedBodyCount = state.bondedBodyStarts.length;
	let triangleFieldCandidateWordsPerTile =
		layout.triangleFieldCandidateWordsPerTile;
	const temporaryReadbacks = new Set<GPUBuffer>();
	const objectRanges = new Map<number, MpmMlsGpuObjectRange>(
		state.objectRanges.map((range) => [range.objectId, range])
	);
	const bondedBodyCountsByObjectId = new Map(state.bondedBodyCountsByObjectId);
	const fractureBondCountsByObjectId = new Map(state.fractureBondCountsByObjectId);
	let nextObjectId = 0;
	for (const range of state.objectRanges) {
		nextObjectId = Math.max(nextObjectId, range.objectId + 1);
	}

	function ensureTriangleFieldCandidateCapacity(bodyCount: number): void {
		const wordsPerTile = Math.ceil(
			Math.min(bodyCount, layout.triangleFieldCandidateBodyLimit) /
				TRIANGLE_FIELD_CANDIDATE_BODY_WORD_BITS
		);
		if (wordsPerTile <= triangleFieldCandidateWordsPerTile) return;

		const wordCount = layout.triangleFieldCandidateTileCount * wordsPerTile;
		const bytes = wordCount * Uint32Array.BYTES_PER_ELEMENT;
		if (!Number.isSafeInteger(wordCount) || !Number.isSafeInteger(bytes)) {
			throw new RangeError('triangle field candidate bitset size exceeds the safe integer range');
		}

		const nextBuffer = session.device.createBuffer({
			label: 'MLS-MPM triangle field candidate bitset',
			size: Math.max(Uint32Array.BYTES_PER_ELEMENT, bytes),
			usage: BUFFER_USAGE.STORAGE
		});
		ownedBuffers.add(nextBuffer);
		try {
			const nextCandidateBindings0 = createTriangleFieldBindings(
				session.device,
				runtime.pipelines,
				runtime.buffers.simulationParams,
				runtime.buffers.particles[0],
				runtime.buffers.stateError,
				runtime.buffers.triangleFieldNodes,
				runtime.buffers.bondedBodyStarts,
				nextBuffer
			);
			const nextCandidateBindings1 = createTriangleFieldBindings(
				session.device,
				runtime.pipelines,
				runtime.buffers.simulationParams,
				runtime.buffers.particles[1],
				runtime.buffers.stateError,
				runtime.buffers.triangleFieldNodes,
				runtime.buffers.bondedBodyStarts,
				nextBuffer
			);
			const previousBuffer = runtime.buffers.triangleFieldCandidates;
			const previousWorkComplete = session.device.queue.onSubmittedWorkDone();

			runtime.buffers.triangleFieldCandidates = nextBuffer;
			runtime.bindings[0] = {
				...runtime.bindings[0],
				...nextCandidateBindings0
			};
			runtime.bindings[1] = {
				...runtime.bindings[1],
				...nextCandidateBindings1
			};
			triangleFieldCandidateWordsPerTile = wordsPerTile;

			void previousWorkComplete
				.then(() => {
					if (!ownedBuffers.has(previousBuffer)) return;
					try {
						previousBuffer.destroy();
						ownedBuffers.delete(previousBuffer);
					} catch {
						// Keep the old buffer tracked so dispose can retry cleanup.
					}
				})
				.catch(() => {
					// Device teardown will release the retired buffer if the queue is lost.
				});
		} catch (cause) {
			try {
				nextBuffer.destroy();
				ownedBuffers.delete(nextBuffer);
			} catch {
				// Keep the buffer tracked so dispose can retry cleanup.
			}
			throw cause;
		}
	}

	const deviceLost = session.lost;
	void deviceLost.then((info) => {
		lostInfo = info;
	});

	function assertActive(): void {
		if (disposed) throw new Error('GPU MLS-MPM solver has been disposed.');
		if (lostInfo) {
			throw new Error(
				`GPU MLS-MPM device was lost (${lostInfo.reason}): ${lostInfo.message}`
			);
		}
	}

	function resolveSelection(selection: MpmMlsGpuSelection): MpmMlsGpuObjectRange {
		if (typeof selection !== 'object' || selection === null) {
			throw new TypeError('selection must identify an MPM particle.');
		}
		if (
			!Number.isSafeInteger(selection.particleIndex) ||
			selection.particleIndex < 0 ||
			selection.particleIndex >= particleSlotCount ||
			!Number.isSafeInteger(selection.objectId) ||
			selection.objectId < 0 ||
			selection.objectId > MAX_U32
		) {
			throw new RangeError('selection must identify an active MPM particle.');
		}
		validateMpmWorldPoint(selection.anchor, state, 'selection anchor');
		const range = objectRanges.get(selection.objectId);
		if (
			!range ||
			selection.particleIndex < range.start ||
			selection.particleIndex >= range.start + range.count
		) {
			throw new RangeError('selection objectId does not match the selected particle.');
		}
		return range;
	}

	function writeInteractionParameters(target: MpmVector2, radius: number): void {
		const parameters = new Float32Array([
			target.x,
			target.y,
			radius,
			MPM_MLS_GPU_DRAG_STIFFNESS,
			MPM_MLS_GPU_DRAG_DAMPING,
			0,
			0,
			0
		]);
		session.device.queue.writeBuffer(runtime.buffers.interactionParams, 0, parameters);
	}

	function step(dt: number, gravity: MpmVector2): void {
		stepBatch(1, dt, gravity);
	}

	function stepBatch(
		stepCount: number,
		dt: number,
		gravity: MpmVector2,
		cflVelocityBound?: number,
		bondReferenceDt?: number
	): void {
		assertActive();
		if (!Number.isSafeInteger(stepCount) || stepCount < 1) {
			throw new RangeError('stepCount must be a positive safe integer.');
		}
		let cflGuard: Pick<MpmMlsGpuCflBounds, 'waveSpeedBound' | 'velocityBound'> | null =
			null;
		if (cflVelocityBound !== undefined) {
			if (cflBounds === null) {
				throw new RangeError('this solver configuration has no safely derived CFL bounds.');
			}
			if (!Number.isFinite(cflVelocityBound) || cflVelocityBound < 0) {
				throw new RangeError('velocityBound must be finite and non-negative.');
			}
			cflGuard = {
				waveSpeedBound: cflBounds.waveSpeedBound,
				velocityBound: cflVelocityBound
			};
		}
		const parameters = packMpmMlsGpuStepParameters(
			state,
			layout,
			dt,
			gravity,
			particleSlotCount,
			bondedBodyCount,
			cflGuard,
			bondReferenceDt
		);
		let inputIndex = currentBuffer;
		try {
			if (cflGuard !== null) {
				session.device.queue.writeBuffer(
					runtime.buffers.cflStepProgress,
					0,
					new Uint32Array([0])
				);
			}
			session.device.queue.writeBuffer(runtime.buffers.simulationParams, 0, parameters);
			const encoder = session.device.createCommandEncoder({
				label:
					stepCount === 1
						? 'MLS-MPM simulation step'
						: `MLS-MPM simulation batch (${stepCount} steps)`
			});
			const pass = encoder.beginComputePass({ label: 'MLS-MPM simulation batch' });
			for (let stepIndex = 0; stepIndex < stepCount; stepIndex += 1) {
				const groups = runtime.bindings[inputIndex];
				dispatchCompute(
					pass,
					runtime.pipelines.clearGrid,
					runtime.clearGrid,
					layout.nodeCount
				);
				dispatchCompute(
					pass,
					runtime.pipelines.prepareParticles,
					groups.prepareParticles,
					particleSlotCount
				);
				dispatchCompute(
					pass,
					runtime.pipelines.buildBuckets,
					groups.buildBuckets,
					particleSlotCount
				);
				dispatchCompute(pass, runtime.pipelines.p2g, groups.p2g, layout.nodeCount);
				if (bondedBodyCount > 0) {
					if (bondedBodyCount <= layout.triangleFieldCandidateBodyLimit) {
						dispatchCompute(
							pass,
							runtime.pipelines.buildTriangleFieldCandidates,
							groups.buildTriangleFieldCandidates,
							layout.triangleFieldCandidateTileCount *
								Math.ceil(
									bondedBodyCount /
										TRIANGLE_FIELD_CANDIDATE_BODY_WORD_BITS
								)
						);
					}
					dispatchCompute(
						pass,
						runtime.pipelines.buildTriangleField,
						groups.buildTriangleField,
						layout.nodeCount
					);
				}
				dispatchCompute(
					pass,
					runtime.pipelines.updateGrid,
					groups.updateGrid,
					layout.nodeCount
				);
				dispatchCompute(
					pass,
					runtime.pipelines.g2p,
					groups.g2p,
					particleSlotCount
				);
				if (bondedBodyCount > 0) {
					dispatchCompute(
						pass,
						runtime.pipelines.applyTriangleContactImpulses,
						groups.applyTriangleContactImpulses,
						bondedBodyCount
					);
					dispatchCompute(
						pass,
						runtime.pipelines.solveBondedTriangles,
						groups.solveBondedTriangles,
						bondedBodyCount
					);
					dispatchCompute(
						pass,
						runtime.pipelines.gatherTriangleContactStates,
						groups.gatherTriangleContactStates,
						bondedBodyCount
					);
					if (bondedBodyCount <= layout.trianglePairContactCacheCapacity) {
						dispatchCompute(
							pass,
							runtime.pipelines.gatherTrianglePairContacts,
							groups.gatherTrianglePairContacts,
							bondedBodyCount * bondedBodyCount
						);
					}
					dispatchCompute(
						pass,
						runtime.pipelines.applyTrianglePairContacts,
						groups.applyTrianglePairContacts,
						bondedBodyCount
					);
				}
				if (cflGuard !== null) {
					dispatchCompute(
						pass,
						runtime.pipelines.validateCflParticles,
						groups.validateCflParticles,
						particleSlotCount
					);
				}
				if (state.fractureBondCount > 0) {
					dispatchCompute(
						pass,
						runtime.pipelines.updateFractureBonds,
						groups.updateFractureBonds,
						state.fractureBondCount
					);
					dispatchCompute(
						pass,
						runtime.pipelines.applyFractureParticleDamage,
						groups.applyFractureParticleDamage,
						particleSlotCount
					);
				}
				dispatchCompute(
					pass,
					runtime.pipelines.commitParticles,
					groups.commitParticles,
					particleSlotCount
				);
				inputIndex = 1 - inputIndex;
			}
			if (cflGuard !== null) {
				const finalGroups = runtime.bindings[inputIndex];
				dispatchCompute(
					pass,
					runtime.pipelines.measureParticleSpeedPartials,
					finalGroups.measureParticleSpeedPartials,
					particleSlotCount
				);
				dispatchCompute(
					pass,
					runtime.pipelines.reduceParticleSpeedPartials,
					runtime.reduceParticleSpeedPartials,
					1
				);
			}
			pass.end();
			session.device.queue.submit([encoder.finish()]);
			currentBuffer = inputIndex;
			cflSpeedMeasurementSubmitted = cflGuard !== null;
			if (cflGuard === null) cflStateCflUsable = false;
		} catch (cause) {
			cflStateCflUsable = false;
			cflSpeedMeasurementSubmitted = false;
			throw new Error(
				`Failed to encode or submit an MLS-MPM batch of ${stepCount} step(s).`,
				{ cause }
			);
		}
	}

	function stepCflBatch(
		macroStepCount: number,
		macroDt: number,
		microstepPlan: MpmCflMicrostepPlan,
		velocityBound: number,
		maxMicrosteps: number,
		gravity: MpmVector2,
		bondReferenceDt?: number
	): void {
		assertActive();
		if (cflBounds === null) {
			throw new RangeError(
				'CFL stepping requires a material-derived wave-speed bound; this solver configuration has no safely derived CFL bounds.'
			);
		}
		if (!cflStateCflUsable) {
			throw new Error(
				'a direct non-CFL GPU step or a failed CFL measurement invalidated the tracked velocity bound; recreate the solver before CFL stepping.'
			);
		}
		if (cflSpeedMeasurementSubmitted) {
			throw new Error(
				'queue measureMaximumParticleSpeed() after the previous CFL batch before submitting another one.'
			);
		}
		if (
			lastScheduledVelocityBound !== null &&
			velocityBound < lastScheduledVelocityBound
		) {
			throw new RangeError(
				'velocityBound must cover the last submitted GPU maximum-particle envelope.'
			);
		}
		if (!Number.isSafeInteger(macroStepCount) || macroStepCount < 1) {
			throw new RangeError('macroStepCount must be a positive safe integer.');
		}
		if (!Number.isSafeInteger(maxMicrosteps) || maxMicrosteps < 1) {
			throw new RangeError('maxMicrosteps must be a positive safe integer.');
		}

		const expectedPlan = calculateMpmCflMicrosteps({
			macroDt,
			cellSize: state.config.grid.cellSize,
			waveSpeedBound: cflBounds.waveSpeedBound,
			velocityBound,
			courantLimit: cflBounds.courantLimit,
			maxMicrosteps
		});
		const certifiedPlan =
			typeof microstepPlan === 'object' &&
			microstepPlan !== null &&
			Number.isSafeInteger(microstepPlan.stepCount) &&
			microstepPlan.stepCount >= expectedPlan.stepCount
				? calculateMpmCflMicrosteps({
						macroDt,
						cellSize: state.config.grid.cellSize,
						waveSpeedBound: cflBounds.waveSpeedBound,
						velocityBound,
						courantLimit: cflBounds.courantLimit,
						maxMicrosteps,
						minimumStepCount: microstepPlan.stepCount
					})
				: null;
		if (
			certifiedPlan === null ||
			microstepPlan.stepCount !== certifiedPlan.stepCount ||
			microstepPlan.microDt !== certifiedPlan.microDt ||
			microstepPlan.safeDt !== certifiedPlan.safeDt
		) {
			throw new RangeError(
				'microstepPlan does not match this solver configuration or is not a certified CFL refinement.'
			);
		}

		const microStepCount = macroStepCount * certifiedPlan.stepCount;
		if (!Number.isSafeInteger(microStepCount) || microStepCount > maxMicrosteps) {
			throw new RangeError(
				`requested microstep count (${microStepCount}) exceeds maxMicrosteps (${maxMicrosteps}).`
			);
		}
		const representedMacroDt = certifiedPlan.microDt * certifiedPlan.stepCount;
		const durationTolerance =
			Math.max(macroDt, Number.MIN_VALUE) * Number.EPSILON * 4;
		if (Math.abs(representedMacroDt - macroDt) > durationTolerance) {
			throw new RangeError('microstepPlan does not preserve the requested macro dt.');
		}
		const gpuMicroDt = Math.fround(certifiedPlan.microDt);
		if (
			activeDrag !== null &&
			(!Number.isFinite(gpuMicroDt) ||
				gpuMicroDt <= 0 ||
				gpuMicroDt * MPM_MLS_GPU_DRAG_DAMPING > 1)
		) {
			throw new RangeError(
				'CFL microDt is too large for the explicit drag damping update to remain dissipative.'
			);
		}

		const runtimeVelocityLimit = calculateMpmCflVelocityLimit({
			cellSize: state.config.grid.cellSize,
			waveSpeedBound: cflBounds.waveSpeedBound,
			microDt: certifiedPlan.microDt,
			courantLimit: cflBounds.courantLimit
		});
		if (runtimeVelocityLimit < velocityBound) {
			throw new RangeError(
				'float32 microDt does not preserve the planned CFL velocity bound; increase the microstep count.'
			);
		}

		// stepBatch encodes every full P2G -> grid update -> G2P sequence using microDt.
		stepBatch(
			microStepCount,
			certifiedPlan.microDt,
			gravity,
			runtimeVelocityLimit,
			bondReferenceDt ?? state.bondReferenceDtF32 ?? macroDt
		);
		latestCflBatchSequence += 1;
		latestCflBatchMicrostepCount = microStepCount;
		lastScheduledVelocityBound = velocityBound;
	}

	async function stepCflBatchAdaptive(
		macroStepCount: number,
		macroDt: number,
		microstepPlan: MpmCflMicrostepPlan,
		velocityBound: number,
		maxMicrosteps: number,
		gravity: MpmVector2,
		gravityForDuration?: (offsetSeconds: number, durationSeconds: number) => MpmVector2
	): Promise<MpmMlsGpuCflBatchMeasurement> {
		if (cflBounds === null) {
			throw new RangeError(
				'CFL stepping requires a material-derived wave-speed bound; this solver configuration has no safely derived CFL bounds.'
			);
		}
		if (!Number.isSafeInteger(macroStepCount) || macroStepCount < 1) {
			throw new RangeError('macroStepCount must be a positive safe integer.');
		}
		if (!Number.isFinite(macroDt) || macroDt <= 0) {
			throw new RangeError('macroDt must be positive and finite.');
		}
		const totalDuration = macroStepCount * macroDt;
		if (!Number.isFinite(totalDuration)) {
			throw new RangeError('CFL batch duration is outside the supported range.');
		}
		const bondReferenceDt = state.bondReferenceDtF32 ?? macroDt;

		let remainingDuration = totalDuration;
		let attemptMacroStepCount = macroStepCount;
		let attemptMacroDt = macroDt;
		let attemptPlan = microstepPlan;
		let attemptVelocityBound = velocityBound;
		let consumedMicrostepBudget = 0;
		let retryCount = 0;
		let adaptiveSubmissionCount = 0;
		let retrySubmissionMilliseconds = 0;
		let readbackMilliseconds = 0;
		const durationTolerance = Math.max(totalDuration, Number.MIN_VALUE) * Number.EPSILON * 4;
		const makeBatchMeasurement = (
			measurement: MpmMlsGpuSpeedMeasurement,
			recommendedVelocityBound = measurement.maximumSpeed
		): MpmMlsGpuCflBatchMeasurement => ({
			...measurement,
			recommendedVelocityBound,
			attemptedMicroStepCount: consumedMicrostepBudget,
			retryCount,
			retrySubmissionMilliseconds,
			readbackMilliseconds,
			simulatedDuration: Math.max(0, totalDuration - remainingDuration),
			remainingDuration
		});

		while (true) {
			const attemptMicroStepCount = attemptMacroStepCount * attemptPlan.stepCount;
			const attemptDuration = attemptMacroStepCount * attemptMacroDt;
			const remainingMicrostepBudget = maxMicrosteps - consumedMicrostepBudget;
			if (
				!Number.isSafeInteger(attemptMicroStepCount) ||
				attemptMicroStepCount < 1 ||
				!Number.isFinite(attemptDuration) ||
				attemptDuration <= 0 ||
				attemptMicroStepCount > remainingMicrostepBudget
			) {
				throw new RangeError(
					`adaptive CFL retry requires ${attemptMicroStepCount} microsteps with ${remainingMicrostepBudget} remaining.`
				);
			}

			const submissionStartedAt = performance.now();
			const attemptGravity =
				gravityForDuration?.(totalDuration - remainingDuration, attemptDuration) ?? gravity;
			stepCflBatch(
				attemptMacroStepCount,
				attemptMacroDt,
				attemptPlan,
				attemptVelocityBound,
				remainingMicrostepBudget,
				attemptGravity,
				bondReferenceDt
			);
			if (adaptiveSubmissionCount > 0) {
				retrySubmissionMilliseconds += performance.now() - submissionStartedAt;
			}
			adaptiveSubmissionCount += 1;

			const readbackStartedAt = performance.now();
			let measurement: MpmMlsGpuSpeedMeasurement;
			try {
				measurement = await measureMaximumParticleSpeed();
			} catch (error) {
				readbackMilliseconds += performance.now() - readbackStartedAt;
				if (!(error instanceof MpmMlsGpuNumericalError) || !isRecoverableCflError(error)) {
					throw error;
				}

				const completedMicrosteps = error.completedMicrosteps;
				if (
					!Number.isSafeInteger(completedMicrosteps) ||
					completedMicrosteps === undefined ||
					completedMicrosteps < 0 ||
					completedMicrosteps >= attemptMicroStepCount
				) {
					throw new Error(
						`Adaptive CFL recovery could not determine completed microsteps: ${error.message}`,
						{ cause: error }
					);
				}

				const completedDuration = completedMicrosteps * attemptPlan.microDt;
				if (!Number.isFinite(completedDuration) || completedDuration >= remainingDuration) {
					throw new Error(
						`Adaptive CFL recovery received invalid simulation progress: ${error.message}`,
						{ cause: error }
					);
				}
				remainingDuration -= completedDuration;
				consumedMicrostepBudget += attemptMicroStepCount;

				const stopRecovery = (reason: string): never => {
					throw new Error(
						`Adaptive CFL recovery stopped after ${retryCount} retry attempt(s) (${reason}); last GPU error: ${error.message}`,
						{ cause: error }
					);
				};

				const observedMaximumSpeed = Math.max(
					error.measuredMaximumSpeed ?? 0,
					error.cflDetails?.particleSpeed ?? 0
				);
				const retryVelocityBound = Math.max(
					attemptVelocityBound,
					lastScheduledVelocityBound ?? 0,
					observedMaximumSpeed * CFL_RECOVERY_BOUND_SAFETY_FACTOR
				);
				if (!Number.isFinite(retryVelocityBound)) {
					stopRecovery('non-finite velocity bound');
				}
				lastScheduledVelocityBound = retryVelocityBound;
				if (error.cflDetails !== undefined) {
					const retryWaveSpeedBound =
						error.cflDetails.waveSpeed * CFL_RECOVERY_BOUND_SAFETY_FACTOR;
					if (!Number.isFinite(retryWaveSpeedBound)) {
						stopRecovery('non-finite wave-speed bound');
					}
					cflBounds = {
						...cflBounds,
						waveSpeedBound: Math.max(
							cflBounds.waveSpeedBound,
							retryWaveSpeedBound
						)
					};
				}

				const retryMicrostepBudget = maxMicrosteps - consumedMicrostepBudget;
				if (retryCount >= MAX_CFL_STEP_RETRIES) {
					stopRecovery('retry limit reached');
				}
				if (retryMicrostepBudget < 2) {
					session.device.queue.writeBuffer(
						runtime.buffers.stateError,
						0,
						new Uint32Array(NUMERICAL_STATUS_WORDS)
					);
					cflStateCflUsable = true;
					return makeBatchMeasurement(
						{
							maximumSpeed: observedMaximumSpeed,
							sequence: latestCflBatchSequence
						},
						retryVelocityBound
					);
				}

				const failedIntervalDuration = Math.min(
					remainingDuration,
					attemptPlan.microDt
				);
				const minimumStepCount = 2;
				try {
					attemptPlan = calculateMpmCflMicrosteps({
						macroDt: failedIntervalDuration,
						cellSize: state.config.grid.cellSize,
						waveSpeedBound: cflBounds.waveSpeedBound,
						velocityBound: retryVelocityBound,
						courantLimit: cflBounds.courantLimit,
						maxMicrosteps: retryMicrostepBudget,
						minimumStepCount
					});
				} catch (cause) {
					if (!isCflMicrostepCapacityError(cause)) {
						throw new Error(
							`Adaptive CFL recovery could not schedule a finer step (${describeError(cause)}; ` +
								`retry=${retryCount + 1}/${MAX_CFL_STEP_RETRIES}, failedIntervalDuration=${failedIntervalDuration}, ` +
								`previousMicroDt=${attemptPlan.microDt}, minimumStepCount=${minimumStepCount}, ` +
								`microstepBudget=${retryMicrostepBudget}): ${error.message}`,
							{ cause }
						);
					}
					session.device.queue.writeBuffer(
						runtime.buffers.stateError,
						0,
						new Uint32Array(NUMERICAL_STATUS_WORDS)
					);
					cflStateCflUsable = true;
					return makeBatchMeasurement(
						{
							maximumSpeed: observedMaximumSpeed,
							sequence: latestCflBatchSequence
						},
						retryVelocityBound
					);
				}

				session.device.queue.writeBuffer(
					runtime.buffers.stateError,
					0,
					new Uint32Array(NUMERICAL_STATUS_WORDS)
				);
				cflStateCflUsable = true;
				attemptMacroStepCount = 1;
				attemptMacroDt = failedIntervalDuration;
				attemptVelocityBound = retryVelocityBound;
				retryCount += 1;
				continue;
			}

			readbackMilliseconds += performance.now() - readbackStartedAt;
			consumedMicrostepBudget += attemptMicroStepCount;
			remainingDuration = Math.max(0, remainingDuration - attemptDuration);
			if (remainingDuration <= durationTolerance) {
				remainingDuration = 0;
				return makeBatchMeasurement(
					measurement,
					retryCount > 0
						? Math.max(attemptVelocityBound, measurement.maximumSpeed)
						: measurement.maximumSpeed
				);
			}

			const continuationBudget = maxMicrosteps - consumedMicrostepBudget;
			if (continuationBudget < 2) {
				return makeBatchMeasurement(
					measurement,
					Math.max(attemptVelocityBound, measurement.maximumSpeed)
				);
			}
			attemptVelocityBound = Math.max(
				attemptVelocityBound,
				measurement.maximumSpeed
			);
			const safeDt = calculateMpmCflSafeDt({
				cellSize: state.config.grid.cellSize,
				waveSpeedBound: cflBounds.waveSpeedBound,
				velocityBound: attemptVelocityBound,
				courantLimit: cflBounds.courantLimit
			});
			const maximumStepDuration = Math.min(attemptPlan.microDt, safeDt);
			if (!Number.isFinite(maximumStepDuration) || maximumStepDuration <= 0) {
				throw new RangeError(
					`Adaptive CFL recovery produced an invalid continuation timestep (${maximumStepDuration}).`
				);
			}
			const requiredStepCount = Math.ceil(remainingDuration / maximumStepDuration);
			let continuationDuration = remainingDuration;
			let minimumContinuationStepCount = Math.max(1, requiredStepCount);
			if (requiredStepCount >= continuationBudget) {
				const reservedStepCount = continuationBudget - 1;
				continuationDuration = Math.min(
					remainingDuration,
					maximumStepDuration * reservedStepCount * (1 - Number.EPSILON * 8)
				);
				minimumContinuationStepCount = Math.max(
					1,
					Math.ceil(continuationDuration / maximumStepDuration)
				);
			}
			if (!Number.isFinite(continuationDuration) || continuationDuration <= 0) {
				return makeBatchMeasurement(
					measurement,
					Math.max(attemptVelocityBound, measurement.maximumSpeed)
				);
			}
			try {
				attemptPlan = calculateMpmCflMicrosteps({
					macroDt: continuationDuration,
					cellSize: state.config.grid.cellSize,
					waveSpeedBound: cflBounds.waveSpeedBound,
					velocityBound: attemptVelocityBound,
					courantLimit: cflBounds.courantLimit,
					maxMicrosteps: continuationBudget,
					minimumStepCount: minimumContinuationStepCount
				});
			} catch (cause) {
				throw new Error(
					`Adaptive CFL recovery could not schedule a safe continuation of ${continuationDuration} seconds: ${describeError(cause)}`,
					{ cause }
				);
			}
			attemptMacroStepCount = 1;
			attemptMacroDt = continuationDuration;
		}
	}

	function render(mode: MpmMlsGpuRenderMode = 'particles'): void {
		assertActive();
		if (mode !== 'surface' && mode !== 'particles') {
			throw new RangeError('render mode must be "surface" or "particles".');
		}
		try {
			if (mode !== currentRenderMode) {
				session.device.queue.writeBuffer(
					runtime.buffers.renderParams,
					0,
					createRenderParameters(state, mode)
				);
				currentRenderMode = mode;
			}
			const encoder = session.device.createCommandEncoder({
				label:
					mode === 'surface'
						? 'MLS-MPM water/oil surface render'
						: 'MLS-MPM particle-marker diagnostic render'
			});
			if (mode === 'surface') {
				const computePass = encoder.beginComputePass({
					label: 'MLS-MPM final-particle occupancy gather'
				});
				const groups = runtime.bindings[currentBuffer];
				dispatchCompute(
					computePass,
					runtime.pipelines.clearSurfaceOccupancy,
					runtime.clearSurfaceOccupancy,
					layout.nodeCount
				);
				dispatchCompute(
					computePass,
					runtime.pipelines.buildBuckets,
					groups.buildBuckets,
					particleSlotCount
				);
				dispatchCompute(
					computePass,
					runtime.pipelines.gatherSurfaceOccupancy,
					groups.gatherSurfaceOccupancy,
					layout.nodeCount
				);
				computePass.end();
			}
			const pass = encoder.beginRenderPass({
				colorAttachments: [
					{
						view: session.context.getCurrentTexture().createView(),
						clearValue: { r: 0.035, g: 0.047, b: 0.063, a: 1 },
						loadOp: 'clear',
						storeOp: 'store'
					}
				]
			});
			if (mode === 'surface') {
				pass.setPipeline(runtime.pipelines.surfaceRender);
				pass.setBindGroup(0, runtime.surfaceRender);
				pass.draw(3, 1);
			}
			pass.setPipeline(runtime.pipelines.render);
			pass.setBindGroup(0, runtime.bindings[currentBuffer].render);
			pass.draw(6, particleSlotCount);
			pass.end();
			session.device.queue.submit([encoder.finish()]);
		} catch (cause) {
			throw new Error(`Failed to render MLS-MPM ${mode} view from GPU state.`, {
				cause
			});
		}
	}

	async function measureQueueCompletionMs(): Promise<number> {
		assertActive();
		const startedAt = performance.now();
		await session.device.queue.onSubmittedWorkDone();
		assertActive();
		return performance.now() - startedAt;
	}

	function appendParticles(particles: readonly MpmParticle[]): MpmMlsGpuAppendResult {
		assertActive();
		if (!Array.isArray(particles)) {
			throw new TypeError('particles must be an array of emitter particles.');
		}
		const requested = particles.length;
		const emptyResult = (
			accepted: number,
			refused: number,
			dropped: number,
			velocityBound = 0,
			reason?: string
		): MpmMlsGpuAppendResult => ({
			requested,
			accepted,
			refused,
			dropped,
			activeParticleCount,
			allocatedParticleCount: particleSlotCount,
			capacity: layout.particleCapacity,
			velocityBound,
			...(reason === undefined ? {} : { reason })
		});
		if (requested === 0) return emptyResult(0, 0, 0);
		if (cflSpeedReadbacksPending > 0) {
			return emptyResult(
				0,
				requested,
				0,
				0,
				'wait for the submitted CFL speed readback before emitting particles'
			);
		}
		if (removalPending) {
			return emptyResult(0, requested, 0, 0, 'outlet removal is still in progress');
		}
		const available = layout.particleCapacity - particleSlotCount;
		if (available <= 0) {
			return emptyResult(
				0,
				0,
				requested,
				0,
				`particle slot capacity ${layout.particleCapacity} is full`
			);
		}
		if (nextObjectId > MAX_U32) {
			return emptyResult(0, requested, 0, 0, 'no object ids remain for a new emitter group');
		}

		const objectId = nextObjectId;
		const sourceParticles = particles.map((particle) => ({
			...particle,
			objectId
		}));
		if (sourceParticles.some((particle) => particle.bondedTriangleVertex !== undefined)) {
			return emptyResult(
				0,
				requested,
				0,
				0,
				'triangle-bonded particles cannot be appended without a complete object topology'
			);
		}

		let candidateState: ValidatedMpmMlsGpuInput;
		try {
			candidateState = validateMpmMlsGpuInput(
				sourceParticles,
				{
					...state.config,
					capacity: requested,
					fractureBonds: []
				}
			);
		} catch (error) {
			const reason =
				error instanceof Error && error.message.trim()
					? error.message
					: 'emitter particles failed GPU input validation';
			return emptyResult(0, requested, 0, 0, reason);
		}
		if (candidateState.bondedBodyStarts.length > 0) {
			return emptyResult(
				0,
				requested,
				0,
				0,
				'triangle-bonded particles cannot be appended without a complete object topology'
			);
		}

		const accepted = Math.min(requested, available);
		const acceptedParticles = sourceParticles.slice(0, accepted);
		const additionBounds = deriveMpmMlsGpuCflBounds(
			acceptedParticles,
			candidateState.config
		);
		if (cflBounds !== null && additionBounds === null) {
			return emptyResult(
				0,
				requested,
				0,
				0,
				'appended material has no safely derived CFL wave-speed bound'
			);
		}

		const destinationStart = particleSlotCount;
		const destinationOffset = destinationStart * PARTICLE_STRIDE_BYTES;
		const appendedWords = accepted * PARTICLE_FLOATS;
		const nextSlotCount = particleSlotCount + accepted;
		try {
			for (const particleBuffer of runtime.buffers.particles) {
				session.device.queue.writeBuffer(
					particleBuffer,
					destinationOffset,
					candidateState.packedParticles.subarray(0, appendedWords)
				);
			}
			session.device.queue.writeBuffer(
				runtime.buffers.particleObjectIds,
				destinationStart * OBJECT_ID_STRIDE_BYTES,
				candidateState.packedObjectIds.subarray(0, accepted)
			);
			session.device.queue.writeBuffer(
				runtime.buffers.simulationParams,
				16,
				new Uint32Array([
					nextSlotCount,
					layout.nodeCount,
					nextSlotCount * 9,
					bondedBodyCount
				])
			);
		} catch (cause) {
			cflStateCflUsable = false;
			throw new Error('Failed to append a validated GPU MLS-MPM emitter batch.', {
				cause
			});
		}

		objectRanges.set(objectId, {
			objectId,
			start: destinationStart,
			count: accepted
		});
		particleSlotCount = nextSlotCount;
		activeParticleCount += accepted;
		nextObjectId += 1;
		if (additionBounds !== null && cflBounds !== null) {
			cflBounds = {
				...cflBounds,
				waveSpeedBound: Math.max(cflBounds.waveSpeedBound, additionBounds.waveSpeedBound),
				velocityBound: Math.max(cflBounds.velocityBound, additionBounds.velocityBound)
			};
			lastScheduledVelocityBound = Math.max(
				lastScheduledVelocityBound ?? 0,
				additionBounds.velocityBound
			);
		}
		return emptyResult(
			accepted,
			0,
			requested - accepted,
			additionBounds?.velocityBound ?? 0,
			requested === accepted
				? undefined
				: `accepted only ${accepted} of ${requested}; GPU slot capacity is exhausted`
		);
	}

	async function removeParticlesInOutlet(
		bounds: MpmMlsGpuOutletBounds
	): Promise<number> {
		assertActive();
		if (typeof bounds !== 'object' || bounds === null) {
			throw new TypeError('outlet bounds must be an object.');
		}
		const minimum = validateMpmWorldPoint(
			{ x: bounds.minX, y: bounds.minY },
			state,
			'outlet minimum'
		);
		const maximum = validateMpmWorldPoint(
			{ x: bounds.maxX, y: bounds.maxY },
			state,
			'outlet maximum'
		);
		if (minimum.x > maximum.x || minimum.y > maximum.y) {
			throw new RangeError('outlet minimum coordinates must not exceed the maximum.');
		}
		if (removalPending) {
			throw new Error('An MLS-MPM outlet removal is already in progress.');
		}
		if (pickPending) {
			throw new Error('Wait for the active particle pick before removing outlet particles.');
		}
		if (particleSlotCount === 0 || activeParticleCount === 0) return 0;
		removalPending = true;

		let staging: GPUBuffer | null = null;
		let mapped = false;
		try {
			staging = session.device.createBuffer({
				label: 'MLS-MPM outlet removal count readback',
				size: OUTLET_COUNTER_BYTES,
				usage: BUFFER_USAGE.MAP_READ | BUFFER_USAGE.COPY_DST
			});
			ownedBuffers.add(staging);
			temporaryReadbacks.add(staging);
			session.device.queue.writeBuffer(
				runtime.buffers.outletParams,
				0,
				new Float32Array([minimum.x, minimum.y, maximum.x, maximum.y])
			);
			session.device.queue.writeBuffer(
				runtime.buffers.outletRemovalCounter,
				0,
				new Uint32Array([0])
			);

			const encoder = session.device.createCommandEncoder({
				label: 'GPU MLS-MPM liquid outlet removal'
			});
			const pass = encoder.beginComputePass({
				label: 'mark liquid particles inside the outlet'
			});
			dispatchCompute(
				pass,
				runtime.pipelines.removeParticlesInOutlet,
				runtime.bindings[currentBuffer].removeParticlesInOutlet,
				particleSlotCount
			);
			pass.end();
			encoder.copyBufferToBuffer(
				runtime.buffers.outletRemovalCounter,
				0,
				staging,
				0,
				OUTLET_COUNTER_BYTES
			);
			session.device.queue.submit([encoder.finish()]);
			currentBuffer = 1 - currentBuffer;

			await staging.mapAsync(MAP_MODE_READ);
			mapped = true;
			assertActive();
			const removed = new Uint32Array(staging.getMappedRange().slice(0))[0];
			if (removed > activeParticleCount) {
				throw new Error('GPU reported an outlet removal count above the active particle count.');
			}
			activeParticleCount -= removed;
			hasRemovedParticles ||= removed > 0;
			return removed;
		} catch (cause) {
			cflStateCflUsable = false;
			throw new Error('Failed to remove liquid particles from the GPU outlet region.', {
				cause
			});
		} finally {
			if (mapped && staging) {
				try {
					staging.unmap();
				} catch {
					// Device teardown may already have unmapped the staging buffer.
				}
			}
			if (staging && temporaryReadbacks.delete(staging)) {
				ownedBuffers.delete(staging);
				try {
					staging.destroy();
				} catch {
					// The GPU state transition already settled; device teardown owns recovery.
				}
			}
			removalPending = false;
		}
	}

	async function measureMaximumParticleSpeed(): Promise<MpmMlsGpuSpeedMeasurement> {
		assertActive();
		if (!cflSpeedMeasurementSubmitted) {
			throw new Error('No completed CFL batch has a maximum-speed measurement to read.');
		}
		cflSpeedMeasurementSubmitted = false;
		const measurementSequence = latestCflBatchSequence;

		const progressOffset = Float32Array.BYTES_PER_ELEMENT;
		const numericalStatusOffset = progressOffset + Uint32Array.BYTES_PER_ELEMENT;
		const readbackBytes = numericalStatusOffset + NUMERICAL_STATUS_BYTES;
		let readback: GPUBuffer;
		try {
			readback = session.device.createBuffer({
				label: 'MLS-MPM compact CFL speed/status readback',
				size: readbackBytes,
				usage: BUFFER_USAGE.MAP_READ | BUFFER_USAGE.COPY_DST
			});
		} catch (cause) {
			cflStateCflUsable = false;
			throw new Error('Failed to allocate the compact CFL speed readback buffer.', {
				cause
			});
		}
		ownedBuffers.add(readback);
		temporaryReadbacks.add(readback);
		cflSpeedReadbacksPending += 1;

		let mapped = false;
		try {
			const encoder = session.device.createCommandEncoder({
				label: 'MLS-MPM compact CFL speed/status readback'
			});
			encoder.copyBufferToBuffer(
				runtime.buffers.cflMaximumSpeed,
				0,
				readback,
				0,
				Float32Array.BYTES_PER_ELEMENT
			);
			encoder.copyBufferToBuffer(
				runtime.buffers.cflStepProgress,
				0,
				readback,
				progressOffset,
				Uint32Array.BYTES_PER_ELEMENT
			);
			encoder.copyBufferToBuffer(
				runtime.buffers.stateError,
				0,
				readback,
				numericalStatusOffset,
				NUMERICAL_STATUS_BYTES
			);
			session.device.queue.submit([encoder.finish()]);
			await readback.mapAsync(MAP_MODE_READ);
			mapped = true;
			assertActive();
			const snapshot = readback.getMappedRange().slice(0);
			readback.unmap();
			mapped = false;

			const completedMicrosteps = new Uint32Array(
				snapshot,
				progressOffset,
				1
			)[0];
			if (completedMicrosteps > latestCflBatchMicrostepCount) {
				throw new RangeError(
					`GPU reported ${completedMicrosteps} completed microsteps for a batch of ${latestCflBatchMicrostepCount}.`
				);
			}
			const measuredMaximumSpeed = new Float32Array(snapshot, 0, 1)[0];
			const numericalError = decodeNumericalError(
				snapshot,
				numericalStatusOffset,
				undefined,
				{ completedMicrosteps, measuredMaximumSpeed }
			);
			if (numericalError) throw numericalError;
			const surfaceError = decodeSurfaceError(
				snapshot,
				numericalStatusOffset
			);
			if (surfaceError) throw surfaceError;
			const maximumSpeed = measuredMaximumSpeed;
			if (!Number.isFinite(maximumSpeed) || maximumSpeed < 0) {
				throw new RangeError('GPU returned an invalid maximum-particle-speed scalar.');
			}
			if (measurementSequence >= latestCompletedCflMeasurementSequence) {
				latestCompletedCflMeasurementSequence = measurementSequence;
				if (measurementSequence === latestCflBatchSequence) {
					lastScheduledVelocityBound = maximumSpeed;
				}
			}
			return { maximumSpeed, sequence: measurementSequence };
		} catch (cause) {
			cflStateCflUsable = false;
			if (
				cause instanceof MpmMlsGpuNumericalError ||
				cause instanceof MpmMlsGpuSurfaceError ||
				cause instanceof RangeError
			) {
				throw cause;
			}
			throw new Error('Failed to read the compact GPU CFL speed measurement.', { cause });
		} finally {
			if (mapped) {
				try {
					readback.unmap();
				} catch {
					// Device teardown may already have unmapped the staging buffer.
				}
			}
			if (temporaryReadbacks.delete(readback)) {
				ownedBuffers.delete(readback);
				try {
					readback.destroy();
				} catch {
					// The speed measurement is already settled; device teardown owns recovery.
				}
			}
			cflSpeedReadbacksPending = Math.max(0, cflSpeedReadbacksPending - 1);
		}
	}

	async function checkNumericalStatus(): Promise<void> {
		assertActive();
		let readback: GPUBuffer;
		try {
			readback = session.device.createBuffer({
				label: 'MLS-MPM numerical status readback',
				size: NUMERICAL_STATUS_BYTES,
				usage: BUFFER_USAGE.MAP_READ | BUFFER_USAGE.COPY_DST
			});
		} catch (cause) {
			throw new Error('Failed to allocate an MLS-MPM numerical status readback buffer.', {
				cause
			});
		}
		ownedBuffers.add(readback);
		temporaryReadbacks.add(readback);

		let mapped = false;
		try {
			const encoder = session.device.createCommandEncoder({
				label: 'MLS-MPM numerical status readback'
			});
			encoder.copyBufferToBuffer(
				runtime.buffers.stateError,
				0,
				readback,
				0,
				NUMERICAL_STATUS_BYTES
			);
			session.device.queue.submit([encoder.finish()]);
			await readback.mapAsync(MAP_MODE_READ);
			mapped = true;
			assertActive();
			const snapshot = readback.getMappedRange().slice(0);
			readback.unmap();
			mapped = false;

			const numericalError = decodeNumericalError(snapshot, 0);
			if (numericalError) throw numericalError;
			const surfaceError = decodeSurfaceError(snapshot, 0);
			if (surfaceError) throw surfaceError;
		} catch (cause) {
			if (
				cause instanceof MpmMlsGpuNumericalError ||
				cause instanceof MpmMlsGpuSurfaceError
			) {
				throw cause;
			}
			throw new Error('Failed to read GPU MLS-MPM numerical status.', { cause });
		} finally {
			if (mapped) {
				try {
					readback.unmap();
				} catch {
					// A destroyed or lost device may already have unmapped the staging buffer.
				}
			}
			if (temporaryReadbacks.delete(readback)) {
				ownedBuffers.delete(readback);
				try {
					readback.destroy();
				} catch {
					// The caller's status check is already settled; device teardown owns recovery.
				}
			}
		}
	}

	function validateReadbackIndices(indices: readonly number[]): readonly number[] {
		if (!Array.isArray(indices)) {
			throw new TypeError('particleIndices must be an array of active particle indices.');
		}
		for (const index of indices) {
			if (!Number.isSafeInteger(index) || index < 0 || index >= particleSlotCount) {
				throw new RangeError('particleIndices must identify active GPU particles.');
			}
		}
		return indices;
	}

	async function readParticles(
		particleIndices?: readonly number[]
	): Promise<readonly MpmParticle[]> {
		assertActive();
		const selectedIndices =
			particleIndices === undefined ? null : validateReadbackIndices(particleIndices);
		const readParticleCount = selectedIndices?.length ?? particleSlotCount;
		const particleStateBytes = readParticleCount * PARTICLE_STRIDE_BYTES;
		const objectIdBytes = readParticleCount * OBJECT_ID_STRIDE_BYTES;
		const errorOffset = particleStateBytes + objectIdBytes;
		const readbackBytes =
			selectedIndices === null
				? layout.readbackBytes
				: Math.max(NUMERICAL_STATUS_BYTES, errorOffset + NUMERICAL_STATUS_BYTES);
		let readback: GPUBuffer;
		try {
			readback = session.device.createBuffer({
				label: 'MLS-MPM explicit particle readback',
				size: readbackBytes,
				usage: BUFFER_USAGE.MAP_READ | BUFFER_USAGE.COPY_DST
			});
		} catch (cause) {
			throw new Error('Failed to allocate an MLS-MPM particle readback buffer.', {
				cause
			});
		}
		ownedBuffers.add(readback);
		temporaryReadbacks.add(readback);

		let mapped = false;
		try {
			const encoder = session.device.createCommandEncoder({
				label: 'MLS-MPM explicit particle readback'
			});
			if (selectedIndices === null && particleStateBytes > 0) {
				encoder.copyBufferToBuffer(
					runtime.buffers.particles[currentBuffer],
					0,
					readback,
					0,
					particleStateBytes
				);
				encoder.copyBufferToBuffer(
					runtime.buffers.particleObjectIds,
					0,
					readback,
					particleStateBytes,
					objectIdBytes
				);
			} else if (selectedIndices !== null) {
				for (let outputIndex = 0; outputIndex < selectedIndices.length; outputIndex += 1) {
					const sourceIndex = selectedIndices[outputIndex];
					encoder.copyBufferToBuffer(
						runtime.buffers.particles[currentBuffer],
						sourceIndex * PARTICLE_STRIDE_BYTES,
						readback,
						outputIndex * PARTICLE_STRIDE_BYTES,
						PARTICLE_STRIDE_BYTES
					);
					encoder.copyBufferToBuffer(
						runtime.buffers.particleObjectIds,
						sourceIndex * OBJECT_ID_STRIDE_BYTES,
						readback,
						particleStateBytes + outputIndex * OBJECT_ID_STRIDE_BYTES,
						OBJECT_ID_STRIDE_BYTES
					);
				}
			}
			encoder.copyBufferToBuffer(
				runtime.buffers.stateError,
				0,
				readback,
				errorOffset,
				NUMERICAL_STATUS_BYTES
			);
			session.device.queue.submit([encoder.finish()]);
			await readback.mapAsync(MAP_MODE_READ);
			mapped = true;
			assertActive();
			const snapshot = readback.getMappedRange().slice(0);
			readback.unmap();
			mapped = false;

			const particles = unpackParticles(
				new Float32Array(snapshot, 0, particleStateBytes / Float32Array.BYTES_PER_ELEMENT),
				new Uint32Array(snapshot, 0, particleStateBytes / Uint32Array.BYTES_PER_ELEMENT),
				readParticleCount,
				new Uint32Array(snapshot, particleStateBytes, readParticleCount)
			);
			const numericalError = decodeNumericalError(
				snapshot,
				errorOffset,
				hasRemovedParticles
					? undefined
					: (particleIndex) => {
							const stateIndex =
								particleIndex === undefined
									? -1
									: selectedIndices === null
										? particleIndex
										: selectedIndices.indexOf(particleIndex);
							return stateIndex >= 0 ? particles[stateIndex] : undefined;
						}
			);
			if (numericalError) {
				throw numericalError;
			}
			const surfaceError = decodeSurfaceError(snapshot, errorOffset);
			if (surfaceError) {
				throw surfaceError;
			}
			return particles;
		} catch (cause) {
			if (
				cause instanceof MpmMlsGpuNumericalError ||
				cause instanceof MpmMlsGpuSurfaceError
			) {
				throw cause;
			}
			throw new Error('Failed to read GPU MLS-MPM particle state.', { cause });
		} finally {
			if (mapped) {
				try {
					readback.unmap();
				} catch {
					// A destroyed or lost device may already have unmapped the staging buffer.
				}
			}
			if (temporaryReadbacks.delete(readback)) {
				ownedBuffers.delete(readback);
				try {
					readback.destroy();
				} catch {
					// The caller's readback result is already settled; device teardown owns recovery.
				}
			}
		}
	}

	async function pick(
		point: MpmVector2,
		radius: number
	): Promise<MpmMlsGpuSelection | null> {
		assertActive();
		const query = validateMpmWorldPoint(point, state, 'pick point');
		const pickRadius = validateMpmPickRadius(radius, state);
		if (activeParticleCount === 0) return null;
		if (pickPending) {
			throw new Error('An MLS-MPM particle pick is already in progress.');
		}
		pickPending = true;

		let staging: GPUBuffer | null = null;
		let mapped = false;
		const queryParticleCount = particleSlotCount;
		try {
			staging = session.device.createBuffer({
				label: 'one-particle MLS-MPM pick readback',
				size: PICK_RESULT_BYTES,
				usage: BUFFER_USAGE.MAP_READ | BUFFER_USAGE.COPY_DST
			});
			ownedBuffers.add(staging);
			temporaryReadbacks.add(staging);

			const parameters = new ArrayBuffer(PICK_PARAMS_BYTES);
			const floats = new Float32Array(parameters, 0, 4);
			floats.set([query.x, query.y, pickRadius, 0]);
			new Uint32Array(parameters, 16, 4).set([queryParticleCount, 0, 0, 0]);
			session.device.queue.writeBuffer(runtime.buffers.pickParams, 0, parameters);

			const encoder = session.device.createCommandEncoder({
				label: 'on-demand MLS-MPM particle pick'
			});
			const pass = encoder.beginComputePass({ label: 'nearest MLS-MPM particle pick' });
			pass.setPipeline(runtime.pipelines.pick);
			pass.setBindGroup(0, runtime.bindings[currentBuffer].pick);
			pass.dispatchWorkgroups(1);
			pass.end();
			encoder.copyBufferToBuffer(runtime.buffers.pickResult, 0, staging, 0, PICK_RESULT_BYTES);
			session.device.queue.submit([encoder.finish()]);

			await staging.mapAsync(MAP_MODE_READ);
			mapped = true;
			assertActive();
			const result = new DataView(staging.getMappedRange());
			const particleIndex = result.getUint32(0, true);
			if (particleIndex === MAX_U32) return null;
			if (particleIndex >= queryParticleCount) {
				throw new Error('GPU returned a particle index outside the active MLS-MPM range.');
			}
			const anchor = {
				x: result.getFloat32(4, true),
				y: result.getFloat32(8, true)
			};
			const objectId = result.getUint32(12, true);
			const range = objectRanges.get(objectId);
			if (
				!range ||
				particleIndex < range.start ||
				particleIndex >= range.start + range.count
			) {
				throw new Error('GPU returned an object id inconsistent with the selected particle.');
			}
			return {
				particleIndex,
				objectId,
				anchor: validateMpmWorldPoint(anchor, state, 'GPU selection anchor')
			};
		} catch (cause) {
			if (disposed) return null;
			throw new Error('Failed to pick an MLS-MPM particle from GPU state.', { cause });
		} finally {
			if (mapped && staging) {
				try {
					staging.unmap();
				} catch {
					// A lost device may already have unmapped the staging buffer.
				}
			}
			if (staging && temporaryReadbacks.delete(staging)) {
				ownedBuffers.delete(staging);
				try {
					staging.destroy();
				} catch {
					// Device teardown owns recovery after a failed or lost pick.
				}
			}
			pickPending = false;
		}
	}

	function beginDrag(
		selection: MpmMlsGpuSelection,
		target: MpmVector2,
		radius: number
	): void {
		assertActive();
		resolveSelection(selection);
		const safeTarget = validateMpmWorldPoint(target, state, 'drag target');
		const safeRadius = validateMpmDragRadius(radius, state);
		writeInteractionParameters(safeTarget, safeRadius);
		activeDrag = { radius: safeRadius };
	}

	function updateDrag(target: MpmVector2): void {
		assertActive();
		if (!activeDrag) return;
		const safeTarget = validateMpmWorldPoint(target, state, 'drag target');
		writeInteractionParameters(safeTarget, activeDrag.radius);
	}

	function endDrag(): void {
		if (!activeDrag) return;
		activeDrag = null;
		if (disposed || lostInfo) return;
		writeInteractionParameters({ x: 0, y: 0 }, 0);
	}

	function clone(selection: MpmMlsGpuSelection, offset: MpmVector2): boolean {
		assertActive();
		const source = resolveSelection(selection);
		const translation = validateMpmCloneOffset(offset);
		if (
			hasRemovedParticles ||
			removalPending ||
			(fractureBondCountsByObjectId.get(source.objectId) ?? 0) > 0
		) {
			return false;
		}
		if (source.count > layout.particleCapacity - particleSlotCount) return false;
		const sourceBondedBodyCount = bondedBodyCountsByObjectId.get(source.objectId) ?? 0;
		if (sourceBondedBodyCount > layout.bondedBodyCapacity - bondedBodyCount) {
			return false;
		}
		if (nextObjectId > MAX_U32) {
			throw new RangeError('No unsigned 32-bit object ids remain for MLS-MPM cloning.');
		}
		const nextBondedBodyCount = bondedBodyCount + sourceBondedBodyCount;
		if (nextBondedBodyCount <= layout.triangleFieldCandidateBodyLimit) {
			ensureTriangleFieldCandidateCapacity(nextBondedBodyCount);
		}

		const destinationStart = particleSlotCount;
		const clonedObjectId = nextObjectId;
		const parameters = new ArrayBuffer(CLONE_PARAMS_BYTES);
		new Uint32Array(parameters, 0, 4).set([
			source.start,
			source.count,
			destinationStart,
			clonedObjectId
		]);
		new Float32Array(parameters, 16, 4).set([
			translation.x,
			translation.y,
			state.domainWidthF32,
			state.domainHeightF32
		]);
		new Uint32Array(parameters, 32, 4).set([
			sourceBondedBodyCount,
			bondedBodyCount,
			layout.bondedBodyCapacity,
			0
		]);

		try {
			endDrag();
			session.device.queue.writeBuffer(runtime.buffers.cloneParams, 0, parameters);
			const encoder = session.device.createCommandEncoder({
				label: 'GPU-to-GPU MLS-MPM object clone'
			});
			const pass = encoder.beginComputePass({ label: 'MLS-MPM object clone' });
			dispatchCompute(
				pass,
				runtime.pipelines.cloneObject,
				runtime.cloneBindings[currentBuffer],
				source.count
			);
			pass.end();
			session.device.queue.submit([encoder.finish()]);
		} catch (cause) {
			throw new Error('Failed to clone the selected MLS-MPM object on the GPU.', {
				cause
			});
		}

		objectRanges.set(clonedObjectId, {
			objectId: clonedObjectId,
			start: destinationStart,
			count: source.count
		});
		bondedBodyCountsByObjectId.set(clonedObjectId, sourceBondedBodyCount);
		fractureBondCountsByObjectId.set(clonedObjectId, 0);
		particleSlotCount += source.count;
		activeParticleCount += source.count;
		bondedBodyCount = nextBondedBodyCount;
		nextObjectId++;
		try {
			// Refresh only live gather counts; preserve step and material parameters.
			session.device.queue.writeBuffer(
				runtime.buffers.simulationParams,
				16,
				new Uint32Array([
					particleSlotCount,
					layout.nodeCount,
					particleSlotCount * 9,
					bondedBodyCount
				])
			);
		} catch (cause) {
			throw new Error('Failed to update the active particle count for surface rendering.', {
				cause
			});
		}
		return true;
	}

	function dispose(): void {
		if (disposed) return;
		disposed = true;
		activeDrag = null;
		const errors = destroyBuffers(ownedBuffers);
		try {
			session.dispose();
		} catch (error) {
			errors.push(error);
		}
		if (errors.length === 1) throw errors[0];
		if (errors.length > 1) {
			throw new AggregateError(errors, 'Failed to dispose the GPU MLS-MPM solver.');
		}
	}

	return {
		deviceLost,
		get particleCount() {
			return activeParticleCount;
		},
		get allocatedParticleCount() {
			return particleSlotCount;
		},
		get particleCapacity() {
			return layout.particleCapacity;
		},
		get cflBounds() {
			return cflBounds;
		},
		step,
		stepBatch,
		stepCflBatch,
		stepCflBatchAdaptive,
		render,
		readParticles,
		measureMaximumParticleSpeed,
		checkNumericalStatus,
		measureQueueCompletionMs,
		appendParticles,
		removeParticlesInOutlet,
		pick,
		beginDrag,
		updateDrag,
		endDrag,
		clone,
		dispose
	};
}

function dispatchCompute(
	pass: GPUComputePassEncoder,
	pipeline: GPUComputePipeline,
	bindGroup: GPUBindGroup,
	count: number
): void {
	const workgroups = Math.ceil(count / COMPUTE_WORKGROUP_SIZE);
	if (workgroups === 0) return;
	pass.setPipeline(pipeline);
	pass.setBindGroup(0, bindGroup);
	pass.dispatchWorkgroups(workgroups);
}

async function createComputePipeline(
	device: GPUDevice,
	module: GPUShaderModule,
	entryPoint: string
): Promise<GPUComputePipeline> {
	try {
		return await device.createComputePipelineAsync({
			label: `MLS-MPM ${entryPoint} pipeline`,
			layout: 'auto',
			compute: { module, entryPoint }
		});
	} catch (cause) {
		throw new Error(
			`Failed to create the MLS-MPM ${entryPoint} compute pipeline: ${describeError(cause)}`,
			{ cause }
		);
	}
}

async function createRenderPipeline(
	device: GPUDevice,
	module: GPUShaderModule,
	format: GPUTextureFormat,
	options: {
		readonly label?: string;
		readonly vertexEntryPoint?: string;
		readonly fragmentEntryPoint?: string;
	} = {}
): Promise<GPURenderPipeline> {
	const label = options.label ?? 'MLS-MPM particle render pipeline';
	try {
		return await device.createRenderPipelineAsync({
			label,
			layout: 'auto',
			vertex: { module, entryPoint: options.vertexEntryPoint ?? 'vertexMain' },
			fragment: {
				module,
				entryPoint: options.fragmentEntryPoint ?? 'fragmentMain',
				targets: [{ format }]
			},
			primitive: { topology: 'triangle-list' }
		});
	} catch (cause) {
		throw new Error(
			`Failed to create the ${label}: ${describeError(cause)}`,
			{ cause }
		);
	}
}

async function assertShaderModuleCompiles(
	module: GPUShaderModule,
	label: string
): Promise<void> {
	const info = await module.getCompilationInfo();
	const errors = info.messages.filter((message) => message.type === 'error');
	if (errors.length === 0) return;
	const descriptions = errors
		.map(
			(message) =>
				`${message.lineNum}:${message.linePos}: ${message.message}`
		)
		.join('\n');
	throw new Error(`${label} WGSL compilation failed:\n${descriptions}`);
}

function createBindGroup(
	device: GPUDevice,
	pipeline: GPUComputePipeline | GPURenderPipeline,
	entries: readonly GPUBindGroupEntry[]
): GPUBindGroup {
	return device.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [...entries]
	});
}

function bufferEntry(binding: number, buffer: GPUBuffer): GPUBindGroupEntry {
	return { binding, resource: { buffer } };
}

function createRenderParameters(
	state: ValidatedMpmMlsGpuInput,
	mode: MpmMlsGpuRenderMode
): Float32Array {
	const radius = Math.fround(state.cellSizeF32 * 0.22);
	if (!Number.isFinite(radius) || radius <= 0) {
		throw new RangeError('grid.cellSize cannot produce a visible float32 particle radius');
	}
	return new Float32Array([
		state.domainWidthF32,
		state.domainHeightF32,
		radius,
		mode === 'surface' ? 1 : 0
	]);
}

function createSurfaceRenderParameters(
	state: ValidatedMpmMlsGpuInput,
	layout: MpmMlsGpuBufferLayout
): ArrayBuffer {
	const parameters = new ArrayBuffer(SURFACE_RENDER_PARAMS_BYTES);
	new Uint32Array(parameters, 0, 4).set([
		layout.nodeWidth,
		layout.nodeHeight,
		state.config.grid.cellsX,
		state.config.grid.cellsY
	]);
	new Float32Array(parameters, 16, 4).set([
		state.domainWidthF32,
		state.domainHeightF32,
		state.cellSizeF32,
		0
	]);
	return parameters;
}

function unpackParticles(
	values: Float32Array,
	words: Uint32Array,
	particleCount: number,
	objectIds: Uint32Array
): readonly MpmParticle[] {
	const particles: MpmParticle[] = [];
	for (let index = 0; index < particleCount; index++) {
		const offset = index * PARTICLE_FLOATS;
		if (words[offset + 13] === MAX_U32) continue;
		const model = decodeMpmConstitutiveModel(words[offset + 13]);
		const bondedVertexMarker = words[offset + 19];
		if (bondedVertexMarker > 3) {
			throw new RangeError(`GPU particle ${index} has an invalid bonded triangle marker.`);
		}
		const bondedTriangleVertex =
			bondedVertexMarker === 1
				? 0
				: bondedVertexMarker === 2
					? 1
					: bondedVertexMarker === 3
						? 2
						: undefined;
		const lambda = values[offset + 14];
		const mu = values[offset + 15];
		let material: MpmMaterialParameters;
		if (model === 'water' || model === 'oil') {
			material = {
				model,
				lambda,
				mu: 0,
				density: values[offset + 16]
			};
		} else if (model === 'powder') {
			material = {
				model,
				lambda,
				mu,
				frictionCoefficient: values[offset + 18]
			};
		} else {
			material = { model, lambda, mu };
		}
		particles.push({
			position: { x: values[offset], y: values[offset + 1] },
			velocity: { x: values[offset + 2], y: values[offset + 3] },
			C: {
				m00: values[offset + 4],
				m01: values[offset + 5],
				m10: values[offset + 6],
				m11: values[offset + 7]
			},
			F: {
				m00: values[offset + 8],
				m01: values[offset + 9],
				m10: values[offset + 10],
				m11: values[offset + 11]
			},
			plasticVolume: values[offset + 12],
			volume: values[offset + 17],
			material,
			...(model === 'neo-hookean' && values[offset + 18] > 0
				? { damage: values[offset + 18] }
				: {}),
			...(bondedTriangleVertex === undefined ? {} : { bondedTriangleVertex }),
			objectId: objectIds[index]
		});
	}
	return particles;
}

function validateMpmWorldPoint(
	point: MpmVector2,
	state: ValidatedMpmMlsGpuInput,
	name: string
): MpmVector2 {
	if (typeof point !== 'object' || point === null) {
		throw new TypeError(`${name} must be a vector.`);
	}
	const x = finiteInteractionFloat32(point.x, `${name}.x`);
	const y = finiteInteractionFloat32(point.y, `${name}.y`);
	if (x < 0 || x > state.domainWidthF32 || y < 0 || y > state.domainHeightF32) {
		throw new RangeError(`${name} must be inside the MLS-MPM domain.`);
	}
	return { x, y };
}

function validateMpmPickRadius(radius: number, state: ValidatedMpmMlsGpuInput): number {
	const value = finiteInteractionFloat32(radius, 'pick radius');
	if (value < 0 || value > Math.max(state.domainWidthF32, state.domainHeightF32)) {
		throw new RangeError('pick radius must be non-negative and within the MLS-MPM domain extent.');
	}
	return value;
}

function validateMpmDragRadius(radius: number, state: ValidatedMpmMlsGpuInput): number {
	const value = validateMpmPickRadius(radius, state);
	if (value <= 0) throw new RangeError('drag radius must be greater than zero.');
	return value;
}

function validateMpmCloneOffset(offset: MpmVector2): MpmVector2 {
	if (typeof offset !== 'object' || offset === null) {
		throw new TypeError('clone offset must be a vector.');
	}
	return {
		x: finiteInteractionFloat32(offset.x, 'clone offset.x'),
		y: finiteInteractionFloat32(offset.y, 'clone offset.y')
	};
}

function finiteInteractionFloat32(value: number, name: string): number {
	if (typeof value !== 'number' || !Number.isFinite(value)) {
		throw new TypeError(`${name} must be a finite number.`);
	}
	const result = Math.fround(value);
	if (!Number.isFinite(result)) {
		throw new RangeError(`${name} is outside the supported float32 range.`);
	}
	return result;
}

async function withDeviceSetupScopes<T>(
	device: GPUDevice,
	label: string,
	operation: () => Promise<T>
): Promise<T> {
	const scopes: Array<'out-of-memory' | 'internal' | 'validation'> = [];
	try {
		device.pushErrorScope('out-of-memory');
		scopes.push('out-of-memory');
		device.pushErrorScope('internal');
		scopes.push('internal');
		device.pushErrorScope('validation');
		scopes.push('validation');
	} catch (cause) {
		const errors = await popDeviceScopes(device, scopes);
		if (errors.length > 0) {
			throw new AggregateError(
				[cause, ...errors],
				`Failed to create error scopes for ${label}.`
			);
		}
		throw new Error(`Failed to create error scopes for ${label}.`, { cause });
	}

	let result: T | undefined;
	let operationFailed = false;
	let operationError: unknown;
	try {
		result = await operation();
	} catch (cause) {
		operationFailed = true;
		operationError = cause;
	}

	const errors = await popDeviceScopes(device, scopes);
	if (operationFailed || errors.length > 0) {
		const causes = operationFailed ? [operationError, ...errors] : errors;
		throw new Error(`WebGPU ${label} failed: ${causes.map(describeError).join('; ')}`, {
			cause: causes.length === 1 ? causes[0] : new AggregateError(causes)
		});
	}
	return result as T;
}

async function popDeviceScopes(
	device: GPUDevice,
	scopes: Array<'out-of-memory' | 'internal' | 'validation'>
): Promise<unknown[]> {
	const errors: unknown[] = [];
	while (scopes.length > 0) {
		const scope = scopes.pop();
		try {
			const error = await device.popErrorScope();
			if (error) {
				errors.push(new Error(`WebGPU ${scope} error: ${error.message}`));
			}
		} catch (error) {
			errors.push(error);
		}
	}
	return errors;
}

function destroyBuffers(buffers: Set<GPUBuffer>): unknown[] {
	const errors: unknown[] = [];
	for (const buffer of buffers) {
		try {
			buffer.destroy();
		} catch (error) {
			errors.push(error);
		}
	}
	buffers.clear();
	return errors;
}

function describeError(error: unknown): string {
	if (error instanceof Error) return error.message;
	return String(error);
}
