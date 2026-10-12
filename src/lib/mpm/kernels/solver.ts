import type {
	MpmMatrix2,
	MpmMaterialParameters,
	MpmParticle,
	MpmVector2
} from '../reference/solver.js';
import { decodeMpmConstitutiveModel } from '../materials.js';
import { createMpmWebGpuSession } from '../platform/webgpu.js';
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

export interface MpmMlsGpuSolver {
	readonly deviceLost: Promise<GPUDeviceLostInfo>;
	readonly particleCount: number;
	readonly particleCapacity: number;
	step(dt: number, gravity: MpmVector2): void;
	render(): void;
	readParticles(particleIndices?: readonly number[]): Promise<readonly MpmParticle[]>;
	measureQueueCompletionMs(): Promise<number>;
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
		readonly projectedDeformation?: MpmMatrix2
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
			[2048, 'non-finite P2G node output']
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
		super(
			`GPU MLS-MPM rejected a step (${message || `status 0x${flags.toString(16)}`}${particleContext}${diagnosticContext}${stateContext}${projectionContext}); ` +
				'all particles were rolled back to the last valid state.'
		);
		this.name = 'MpmMlsGpuNumericalError';
	}
}

interface MpmMlsGpuPipelines {
	readonly clearGrid: GPUComputePipeline;
	readonly buildBuckets: GPUComputePipeline;
	readonly p2g: GPUComputePipeline;
	readonly updateGrid: GPUComputePipeline;
	readonly g2p: GPUComputePipeline;
	readonly commitParticles: GPUComputePipeline;
	readonly pick: GPUComputePipeline;
	readonly cloneObject: GPUComputePipeline;
	readonly render: GPURenderPipeline;
}

interface MpmMlsGpuStepBindings {
	readonly buildBuckets: GPUBindGroup;
	readonly p2g: GPUBindGroup;
	readonly g2p: GPUBindGroup;
	readonly commitParticles: GPUBindGroup;
	readonly pick: GPUBindGroup;
	readonly render: GPUBindGroup;
}

const INTERACTION_PARAMS_BYTES = 32;
const PICK_PARAMS_BYTES = 32;
const PICK_RESULT_BYTES = 16;
const CLONE_PARAMS_BYTES = 32;
const MAX_U32 = 0xffff_ffff;
const DRAG_STIFFNESS = 20;
const DRAG_DAMPING = 8;
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
	config: MpmMlsGpuConfig
): Promise<MpmMlsGpuSolver> {
	if (canvas === null || typeof canvas?.getContext !== 'function') {
		throw new TypeError('canvas must be an HTML canvas with getContext()');
	}
	const initialState = validateMpmMlsGpuInput(particles, config);
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
		return createSolver(session, initialState, layout, runtime, ownedBuffers);
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

interface MpmMlsGpuRuntime {
	readonly buffers: {
		readonly simulationParams: GPUBuffer;
		readonly interactionParams: GPUBuffer;
		readonly renderParams: GPUBuffer;
		readonly stateError: GPUBuffer;
		readonly bucketHeads: GPUBuffer;
		readonly bucketNext: GPUBuffer;
		readonly gridNodes: GPUBuffer;
		readonly particleObjectIds: GPUBuffer;
		readonly pickParams: GPUBuffer;
		readonly pickResult: GPUBuffer;
		readonly cloneParams: GPUBuffer;
		readonly particles: readonly [GPUBuffer, GPUBuffer];
	};
	readonly pipelines: MpmMlsGpuPipelines;
	readonly clearGrid: GPUBindGroup;
	readonly updateGrid: GPUBindGroup;
	readonly bindings: readonly [MpmMlsGpuStepBindings, MpmMlsGpuStepBindings];
	readonly cloneBindings: readonly [GPUBindGroup, GPUBindGroup];
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
		buildBuckets: await createComputePipeline(device, computeModule, 'buildBuckets'),
		p2g: await createComputePipeline(device, computeModule, 'p2g'),
		updateGrid: await createComputePipeline(device, computeModule, 'updateGrid'),
		g2p: await createComputePipeline(device, computeModule, 'g2p'),
		commitParticles: await createComputePipeline(device, computeModule, 'commitParticles'),
		pick: await createComputePipeline(device, computeModule, 'pickParticle'),
		cloneObject: await createComputePipeline(device, computeModule, 'cloneObject'),
		render: await createRenderPipeline(device, renderModule, format)
	};

	const createBuffer = (label: string, size: number, usage: GPUBufferUsageFlags) => {
		const buffer = device.createBuffer({ label, size, usage });
		ownedBuffers.add(buffer);
		return buffer;
	};
	const particleBufferUsage =
		BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_SRC | BUFFER_USAGE.COPY_DST;

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
		particleObjectIds: createBuffer(
			'MLS-MPM particle object ids',
			layout.objectIdBufferBytes,
			particleBufferUsage
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
		new Uint32Array([0, MAX_U32, 0, 0, 0, 0, 0])
	);
	device.queue.writeBuffer(buffers.interactionParams, 0, new Float32Array(8));
	device.queue.writeBuffer(buffers.pickResult, 0, new Uint32Array([MAX_U32, 0, 0, 0]));
	device.queue.writeBuffer(
		buffers.renderParams,
		0,
		createRenderParameters(state)
	);
	if (layout.particleStateBytes > 0) {
		device.queue.writeBuffer(buffers.particles[0], 0, state.packedParticles);
		device.queue.writeBuffer(buffers.particleObjectIds, 0, state.packedObjectIds);
	}

	const clearGrid = createBindGroup(device, pipelines.clearGrid, [
		bufferEntry(0, buffers.simulationParams),
		bufferEntry(3, buffers.bucketHeads),
		bufferEntry(5, buffers.gridNodes)
	]);
	const updateGrid = createBindGroup(device, pipelines.updateGrid, [
		bufferEntry(0, buffers.simulationParams),
		bufferEntry(5, buffers.gridNodes),
		bufferEntry(6, buffers.stateError),
		bufferEntry(7, buffers.interactionParams)
	]);

	const createStepBindings = (inputIndex: 0 | 1): MpmMlsGpuStepBindings => {
		const inputBuffers = buffers.particles;
		const input = inputBuffers[inputIndex];
		const output = inputBuffers[1 - inputIndex];
		const buildBuckets = createBindGroup(device, pipelines.buildBuckets, [
			bufferEntry(0, buffers.simulationParams),
			bufferEntry(1, input),
			bufferEntry(3, buffers.bucketHeads),
			bufferEntry(4, buffers.bucketNext),
			bufferEntry(6, buffers.stateError)
		]);
		const p2g = createBindGroup(device, pipelines.p2g, [
			bufferEntry(0, buffers.simulationParams),
			bufferEntry(1, input),
			bufferEntry(3, buffers.bucketHeads),
			bufferEntry(4, buffers.bucketNext),
			bufferEntry(5, buffers.gridNodes),
			bufferEntry(6, buffers.stateError)
		]);
		const g2p = createBindGroup(device, pipelines.g2p, [
			bufferEntry(0, buffers.simulationParams),
			bufferEntry(1, input),
			bufferEntry(2, output),
			bufferEntry(5, buffers.gridNodes),
			bufferEntry(6, buffers.stateError)
		]);
		const commitParticles = createBindGroup(device, pipelines.commitParticles, [
			bufferEntry(0, buffers.simulationParams),
			bufferEntry(1, input),
			bufferEntry(2, output),
			bufferEntry(6, buffers.stateError)
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
		return { buildBuckets, p2g, g2p, commitParticles, pick, render };
	};
	const createCloneBinding = (inputIndex: 0 | 1): GPUBindGroup =>
		createBindGroup(device, pipelines.cloneObject, [
			bufferEntry(11, buffers.particles[inputIndex]),
			bufferEntry(12, buffers.particleObjectIds),
			bufferEntry(13, buffers.cloneParams)
		]);
	const bindings: readonly [
		MpmMlsGpuStepBindings,
		MpmMlsGpuStepBindings
	] = [createStepBindings(0), createStepBindings(1)];
	const cloneBindings: readonly [GPUBindGroup, GPUBindGroup] = [
		createCloneBinding(0),
		createCloneBinding(1)
	];

	return { buffers, pipelines, clearGrid, updateGrid, bindings, cloneBindings };
}

function createSolver(
	session: Awaited<ReturnType<typeof createMpmWebGpuSession>>,
	state: ValidatedMpmMlsGpuInput,
	layout: MpmMlsGpuBufferLayout,
	runtime: MpmMlsGpuRuntime,
	ownedBuffers: Set<GPUBuffer>
): MpmMlsGpuSolver {
	let currentBuffer = 0;
	let particleCount = state.particleCount;
	let disposed = false;
	let lostInfo: GPUDeviceLostInfo | null = null;
	let pickPending = false;
	let activeDrag: { readonly radius: number } | null = null;
	const temporaryReadbacks = new Set<GPUBuffer>();
	const objectRanges = new Map<number, MpmMlsGpuObjectRange>(
		state.objectRanges.map((range) => [range.objectId, range])
	);
	let nextObjectId = 0;
	for (const range of state.objectRanges) {
		nextObjectId = Math.max(nextObjectId, range.objectId + 1);
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
			selection.particleIndex >= particleCount ||
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
			DRAG_STIFFNESS,
			DRAG_DAMPING,
			0,
			0,
			0
		]);
		session.device.queue.writeBuffer(runtime.buffers.interactionParams, 0, parameters);
	}

	function step(dt: number, gravity: MpmVector2): void {
		assertActive();
		const parameters = packMpmMlsGpuStepParameters(state, layout, dt, gravity, particleCount);
		const inputIndex = currentBuffer;
		const groups = runtime.bindings[inputIndex];
		try {
			session.device.queue.writeBuffer(runtime.buffers.simulationParams, 0, parameters);
			const encoder = session.device.createCommandEncoder({
				label: 'MLS-MPM simulation step'
			});
			dispatchCompute(
				encoder,
				runtime.pipelines.clearGrid,
				runtime.clearGrid,
				layout.nodeCount
			);
			dispatchCompute(
				encoder,
				runtime.pipelines.buildBuckets,
				groups.buildBuckets,
				particleCount
			);
			dispatchCompute(encoder, runtime.pipelines.p2g, groups.p2g, layout.nodeCount);
			dispatchCompute(
				encoder,
				runtime.pipelines.updateGrid,
				runtime.updateGrid,
				layout.nodeCount
			);
			dispatchCompute(
				encoder,
				runtime.pipelines.g2p,
				groups.g2p,
				particleCount
			);
			dispatchCompute(
				encoder,
				runtime.pipelines.commitParticles,
				groups.commitParticles,
				particleCount
			);
			session.device.queue.submit([encoder.finish()]);
			currentBuffer = 1 - inputIndex;
		} catch (cause) {
			throw new Error('Failed to encode or submit an MLS-MPM step.', { cause });
		}
	}

	function render(): void {
		assertActive();
		try {
			const encoder = session.device.createCommandEncoder({
				label: 'MLS-MPM particle render'
			});
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
			pass.setPipeline(runtime.pipelines.render);
			pass.setBindGroup(0, runtime.bindings[currentBuffer].render);
			pass.draw(6, particleCount);
			pass.end();
			session.device.queue.submit([encoder.finish()]);
		} catch (cause) {
			throw new Error('Failed to render MLS-MPM particles from GPU state.', { cause });
		}
	}

	async function measureQueueCompletionMs(): Promise<number> {
		assertActive();
		const startedAt = performance.now();
		await session.device.queue.onSubmittedWorkDone();
		assertActive();
		return performance.now() - startedAt;
	}

	function validateReadbackIndices(indices: readonly number[]): readonly number[] {
		if (!Array.isArray(indices)) {
			throw new TypeError('particleIndices must be an array of active particle indices.');
		}
		for (const index of indices) {
			if (!Number.isSafeInteger(index) || index < 0 || index >= particleCount) {
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
		const readParticleCount = selectedIndices?.length ?? particleCount;
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

			const numericalStatus = new Uint32Array(
				snapshot,
				errorOffset,
				NUMERICAL_STATUS_WORDS
			);
			const particles = unpackParticles(
				new Float32Array(snapshot, 0, particleStateBytes / Float32Array.BYTES_PER_ELEMENT),
				new Uint32Array(snapshot, 0, particleStateBytes / Uint32Array.BYTES_PER_ELEMENT),
				readParticleCount,
				new Uint32Array(snapshot, particleStateBytes, readParticleCount)
			);
			if (numericalStatus[0] !== 0) {
				const particleIndex =
					numericalStatus[1] === MAX_U32 ? undefined : numericalStatus[1];
				const stateIndex =
					particleIndex === undefined
						? -1
						: selectedIndices === null
							? particleIndex
							: selectedIndices.indexOf(particleIndex);
				const particleState =
					stateIndex >= 0 ? particles[stateIndex] : undefined;
				const values = new Float32Array(
					snapshot,
					errorOffset + 2 * Uint32Array.BYTES_PER_ELEMENT,
					4
				);
				const diagnostics =
					numericalStatus[6] !== 1
						? undefined
						: {
								determinant: values[0],
								deformationIncrement: values[1],
								precedingDeterminant: values[2]
							};
				const projectedDeformation =
					numericalStatus[6] !== 2
						? undefined
						: {
								m00: values[0],
								m01: values[1],
								m10: values[2],
								m11: values[3]
							};
				throw new MpmMlsGpuNumericalError(
					numericalStatus[0],
					particleIndex,
					diagnostics,
					particleState,
					projectedDeformation
				);
			}
			return particles;
		} catch (cause) {
			if (cause instanceof MpmMlsGpuNumericalError) throw cause;
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
		if (particleCount === 0) return null;
		if (pickPending) {
			throw new Error('An MLS-MPM particle pick is already in progress.');
		}
		pickPending = true;

		let staging: GPUBuffer | null = null;
		let mapped = false;
		const queryParticleCount = particleCount;
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
		if (source.count > layout.particleCapacity - particleCount) return false;
		if (nextObjectId > MAX_U32) {
			throw new RangeError('No unsigned 32-bit object ids remain for MLS-MPM cloning.');
		}

		const destinationStart = particleCount;
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

		try {
			endDrag();
			session.device.queue.writeBuffer(runtime.buffers.cloneParams, 0, parameters);
			const encoder = session.device.createCommandEncoder({
				label: 'GPU-to-GPU MLS-MPM object clone'
			});
			dispatchCompute(
				encoder,
				runtime.pipelines.cloneObject,
				runtime.cloneBindings[currentBuffer],
				source.count
			);
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
		particleCount += source.count;
		nextObjectId++;
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
			return particleCount;
		},
		get particleCapacity() {
			return layout.particleCapacity;
		},
		step,
		render,
		readParticles,
		measureQueueCompletionMs,
		pick,
		beginDrag,
		updateDrag,
		endDrag,
		clone,
		dispose
	};
}

function dispatchCompute(
	encoder: GPUCommandEncoder,
	pipeline: GPUComputePipeline,
	bindGroup: GPUBindGroup,
	count: number
): void {
	const workgroups = Math.ceil(count / COMPUTE_WORKGROUP_SIZE);
	if (workgroups === 0) return;
	const pass = encoder.beginComputePass({ label: pipeline.label });
	pass.setPipeline(pipeline);
	pass.setBindGroup(0, bindGroup);
	pass.dispatchWorkgroups(workgroups);
	pass.end();
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
	format: GPUTextureFormat
): Promise<GPURenderPipeline> {
	try {
		return await device.createRenderPipelineAsync({
			label: 'MLS-MPM particle render pipeline',
			layout: 'auto',
			vertex: { module, entryPoint: 'vertexMain' },
			fragment: {
				module,
				entryPoint: 'fragmentMain',
				targets: [{ format }]
			},
			primitive: { topology: 'triangle-list' }
		});
	} catch (cause) {
		throw new Error(
			`Failed to create the MLS-MPM particle render pipeline: ${describeError(cause)}`,
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

function createRenderParameters(state: ValidatedMpmMlsGpuInput): Float32Array {
	const radius = Math.fround(state.cellSizeF32 * 0.22);
	if (!Number.isFinite(radius) || radius <= 0) {
		throw new RangeError('grid.cellSize cannot produce a visible float32 particle radius');
	}
	return new Float32Array([
		state.domainWidthF32,
		state.domainHeightF32,
		radius,
		0
	]);
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
		const model = decodeMpmConstitutiveModel(words[offset + 13]);
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
