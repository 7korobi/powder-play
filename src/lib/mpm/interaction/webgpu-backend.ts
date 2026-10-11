import { createMpmWebGpuSession } from '#lib/mpm/platform/webgpu.js';
import type {
	MpmBackend,
	MpmDragMode,
	MpmLabConfig,
	MpmSelection,
	MpmVector,
	MpmWebGpuSession
} from '#lib/mpm/contracts.js';
import {
	buildMpmInteractionLayout,
	MAX_INTERACTION_PARTICLES,
	type MpmInteractionLayout,
	type MpmObjectRange
} from './config.js';

const WORKGROUP_SIZE = 64;
const MAX_FRAME_DELTA_SECONDS = 1 / 30;
const MAX_GRAVITY_COMPONENT = 10_000;
const MAX_INTERACTION_COORDINATE = 2_000_000;
const MAX_PICK_RADIUS = MAX_INTERACTION_COORDINATE;
const MAX_U32 = 0xffff_ffff;

const PARTICLE_STRIDE_BYTES = 16;
const MATERIAL_STRIDE_BYTES = 4;
const STEP_PARAMS_BYTES = 32;
const DRAG_PARAMS_BYTES = 64;
const PICK_PARAMS_BYTES = 32;
const PICK_RESULT_BYTES = 16;
const CLONE_PARAMS_BYTES = 48;
const RENDER_PARAMS_BYTES = 32;

// These named values mirror the WebGPU constants for browser TypeScript libs that expose the
// WebGPU interfaces but omit the corresponding global constant objects.
const BUFFER_USAGE = {
	MAP_READ: 0x0001,
	COPY_SRC: 0x0004,
	COPY_DST: 0x0008,
	UNIFORM: 0x0040,
	STORAGE: 0x0080
} as const satisfies Record<string, GPUBufferUsageFlags>;
const MAP_MODE_READ = 0x0001;
const COLOR_WRITE_ALL = 0x000f;

const SIMULATION_SHADER: string = /* wgsl */ `
struct StepParams {
	worldSize: vec2<f32>,
	deltaSeconds: f32,
	activeCount: u32,
	gravity: vec2<f32>,
	particleRadius: f32,
	restitution: f32,
}

@group(0) @binding(0) var<storage, read_write> particles: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> params: StepParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
	let index = invocation.x;
	if (index >= params.activeCount) {
		return;
	}

	var state = particles[index];
	var velocity = (state.zw + params.gravity * params.deltaSeconds) * 0.9995;
	let speed = length(velocity);
	if (speed > 1200.0) {
		velocity = velocity * (1200.0 / speed);
	}

	var position = state.xy + velocity * params.deltaSeconds;
	let lower = vec2<f32>(params.particleRadius);
	let upper = max(params.worldSize - lower, lower);

	if (position.x < lower.x) {
		position.x = lower.x;
		velocity.x = abs(velocity.x) * params.restitution;
		velocity.y = velocity.y * 0.92;
	} else if (position.x > upper.x) {
		position.x = upper.x;
		velocity.x = -abs(velocity.x) * params.restitution;
		velocity.y = velocity.y * 0.92;
	}

	if (position.y < lower.y) {
		position.y = lower.y;
		velocity.y = abs(velocity.y) * params.restitution;
		velocity.x = velocity.x * 0.92;
	} else if (position.y > upper.y) {
		position.y = upper.y;
		velocity.y = -abs(velocity.y) * params.restitution;
		velocity.x = velocity.x * 0.92;
	}

	state = vec4<f32>(position, velocity);
	particles[index] = state;
}
`;

const DRAG_SHADER: string = /* wgsl */ `
struct DragParams {
	rangeAndMode: vec4<u32>,
	anchorTarget: vec4<f32>,
	constants: vec4<f32>,
	worldAndRadius: vec4<f32>,
}

@group(0) @binding(0) var<storage, read_write> particles: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> dragOrigins: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: DragParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
	let index = invocation.x;
	if (index >= params.rangeAndMode.x) {
		return;
	}

	let start = params.rangeAndMode.y;
	let end = start + params.rangeAndMode.z;
	if (index < start || index >= end) {
		return;
	}

	var state = particles[index];
	if (params.rangeAndMode.w == 1u) {
		let translation = params.anchorTarget.zw - params.anchorTarget.xy;
		let lower = vec2<f32>(params.worldAndRadius.z);
		let upper = max(params.worldAndRadius.xy - lower, lower);
		state.xy = clamp(dragOrigins[index].xy + translation, lower, upper);
		state.zw = vec2<f32>(0.0);
	} else {
		let distanceFromAnchor = length(state.xy - params.anchorTarget.xy);
		let radius = params.constants.y;
		if (distanceFromAnchor <= radius) {
			let falloff = 1.0 - clamp(distanceFromAnchor / radius, 0.0, 1.0);
			let spring = (params.anchorTarget.zw - state.xy) * params.constants.z;
			let damping = state.zw * params.constants.w;
			state.zw = state.zw + (spring - damping) * params.constants.x * falloff;
			let speed = length(state.zw);
			if (speed > 1200.0) {
				state.zw = state.zw * (1200.0 / speed);
			}
		}
	}

	particles[index] = state;
}
`;

const PICK_SHADER: string = /* wgsl */ `
struct PickParams {
	pointAndRadius: vec4<f32>,
	countAndPadding: vec4<u32>,
}

@group(0) @binding(0) var<storage, read> particles: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> params: PickParams;
@group(0) @binding(2) var<storage, read_write> result: array<u32>;

@compute @workgroup_size(1)
fn main() {
	let radiusSquared = params.pointAndRadius.z * params.pointAndRadius.z;
	var closestDistance = radiusSquared;
	var closestIndex = 0xffffffffu;
	var closestPosition = vec2<f32>(0.0);

	for (var index = 0u; index < params.countAndPadding.x; index = index + 1u) {
		let position = particles[index].xy;
		let offset = position - params.pointAndRadius.xy;
		let distanceSquared = dot(offset, offset);
		if (
			distanceSquared <= radiusSquared &&
			(closestIndex == 0xffffffffu || distanceSquared < closestDistance)
		) {
			closestDistance = distanceSquared;
			closestIndex = index;
			closestPosition = position;
		}
	}

	result[0] = closestIndex;
	result[1] = bitcast<u32>(closestPosition.x);
	result[2] = bitcast<u32>(closestPosition.y);
	result[3] = 0u;
}
`;

const CLONE_SHADER: string = /* wgsl */ `
struct CloneParams {
	rangeAndPadding: vec4<u32>,
	offsetAndWorld: vec4<f32>,
	radiusAndPadding: vec4<f32>,
}

@group(0) @binding(0) var<storage, read_write> particles: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> params: CloneParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
	let localIndex = invocation.x;
	if (localIndex >= params.rangeAndPadding.y) {
		return;
	}

	let index = params.rangeAndPadding.x + localIndex;
	var state = particles[index];
	let lower = vec2<f32>(params.radiusAndPadding.x);
	let upper = max(params.offsetAndWorld.zw - lower, lower);
	state.xy = clamp(state.xy + params.offsetAndWorld.xy, lower, upper);
	particles[index] = state;
}
`;

const RENDER_SHADER: string = /* wgsl */ `
struct RenderParams {
	worldAndCanvas: vec4<f32>,
	radiusAndPadding: vec4<f32>,
}

struct VertexOutput {
	@builtin(position) position: vec4<f32>,
	@location(0) localPosition: vec2<f32>,
	@location(1) @interpolate(flat) materialId: u32,
}

const QUAD_CORNERS = array<vec2<f32>, 6>(
	vec2<f32>(-1.0, -1.0),
	vec2<f32>(1.0, -1.0),
	vec2<f32>(-1.0, 1.0),
	vec2<f32>(-1.0, 1.0),
	vec2<f32>(1.0, -1.0),
	vec2<f32>(1.0, 1.0),
);

@group(0) @binding(0) var<storage, read> particles: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> materialIds: array<u32>;
@group(0) @binding(2) var<uniform> params: RenderParams;

@vertex
fn vertexMain(
	@builtin(vertex_index) vertexIndex: u32,
	@builtin(instance_index) particleIndex: u32
) -> VertexOutput {
	let corner = QUAD_CORNERS[vertexIndex];
	let particlePosition = particles[particleIndex].xy;
	let worldWidth = params.worldAndCanvas.x;
	let worldHeight = params.worldAndCanvas.y;
	let canvasWidth = params.worldAndCanvas.z;
	let canvasHeight = params.worldAndCanvas.w;
	let worldPosition = vec2<f32>(
		particlePosition.x / worldWidth * 2.0 - 1.0,
		1.0 - particlePosition.y / worldHeight * 2.0,
	);
	let pixelToClip = vec2<f32>(2.0 / canvasWidth, 2.0 / canvasHeight);

	var output: VertexOutput;
	output.position = vec4<f32>(
		worldPosition + corner * params.radiusAndPadding.x * pixelToClip,
		0.0,
		1.0,
	);
	output.localPosition = corner;
	output.materialId = materialIds[particleIndex];
	return output;
}

fn materialColor(materialId: u32) -> vec3<f32> {
	switch materialId % 5u {
		case 0u: {
			return vec3<f32>(0.25, 0.62, 0.96);
		}
		case 1u: {
			return vec3<f32>(0.73, 0.56, 0.92);
		}
		case 2u: {
			return vec3<f32>(0.92, 0.72, 0.33);
		}
		case 3u: {
			return vec3<f32>(0.46, 0.82, 0.71);
		}
		default: {
			return vec3<f32>(0.96, 0.52, 0.28);
		}
	}
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
	let distanceFromCenter = length(input.localPosition);
	if (distanceFromCenter > 1.0) {
		discard;
	}
	let alpha = 1.0 - smoothstep(0.86, 1.0, distanceFromCenter);
	return vec4<f32>(materialColor(input.materialId), alpha);
}
`;

interface GpuBuffers {
	particles: GPUBuffer;
	materialIds: GPUBuffer;
	dragOrigins: GPUBuffer;
	stepParams: GPUBuffer;
	dragParams: GPUBuffer;
	pickParams: GPUBuffer;
	pickResult: GPUBuffer;
	cloneParams: GPUBuffer;
	renderParams: GPUBuffer;
}

interface GpuResources {
	buffers: GpuBuffers;
	simulationPipeline: GPUComputePipeline;
	simulationBindGroup: GPUBindGroup;
	dragPipeline: GPUComputePipeline;
	dragBindGroup: GPUBindGroup;
	pickPipeline: GPUComputePipeline;
	pickBindGroup: GPUBindGroup;
	clonePipeline: GPUComputePipeline;
	cloneBindGroup: GPUBindGroup;
	renderPipeline: GPURenderPipeline;
	renderBindGroup: GPUBindGroup;
}

interface DragState {
	start: number;
	count: number;
	mode: MpmDragMode;
	anchorX: number;
	anchorY: number;
	offsetX: number;
	offsetY: number;
	targetX: number;
	targetY: number;
}

/**
 * Gate-0 interaction test backend: bounded 2D gravity, picking, GPU-state dragging and cloning.
 * This deliberately is not MLS-MPM physics and must not be treated as a production solver.
 */
export class MpmInteractionBackend implements MpmBackend {
	private session: MpmWebGpuSession | null = null;
	private resources: GpuResources | null = null;
	private canvas: HTMLCanvasElement | null = null;
	private worldWidth = 0;
	private worldHeight = 0;
	private capacity = 0;
	private particleCount = 0;
	private objectRanges: MpmObjectRange[] = [];
	private drag: DragState | null = null;
	private readonly pendingPickBuffers = new Set<GPUBuffer>();
	private lostPromise: Promise<GPUDeviceLostInfo> | null = null;
	private lifecycle = 0;
	private disposed = true;

	get deviceLost(): Promise<GPUDeviceLostInfo> | null {
		return this.lostPromise;
	}

	async initialize(canvas: HTMLCanvasElement, config: MpmLabConfig): Promise<void> {
		const layout = buildMpmInteractionLayout(config);
		this.dispose();
		const lifecycle = this.lifecycle;
		this.disposed = false;

		let session: MpmWebGpuSession | null = null;
		try {
			const createdSession = await createMpmWebGpuSession(canvas);
			if (this.disposed || lifecycle !== this.lifecycle) {
				safelyDisposeSession(createdSession);
				return;
			}

			session = createdSession;
			this.session = createdSession;
			this.lostPromise = createdSession.lost;
			void createdSession.lost.then(
				() => this.handleDeviceLoss(createdSession, lifecycle),
				() => this.handleDeviceLoss(createdSession, lifecycle)
			);

			assertDeviceCapacity(createdSession.device, layout.capacity);
			const resources = createGpuResources(createdSession.device, createdSession.format, layout);
			if (
				this.disposed ||
				lifecycle !== this.lifecycle ||
				this.session !== createdSession
			) {
				disposeGpuResourceSet(resources);
				return;
			}

			this.resources = resources;
			this.canvas = canvas;
			this.worldWidth = layout.worldWidth;
			this.worldHeight = layout.worldHeight;
			this.capacity = layout.capacity;
			this.particleCount = layout.particleCount;
			this.objectRanges = layout.objects;
		} catch (error) {
			if (lifecycle === this.lifecycle) {
				this.cleanupCurrentSession(session);
			} else if (session && this.session !== session) {
				safelyDisposeSession(session);
			}
			throw error;
		}
	}

	step(deltaSeconds: number, gravity: MpmVector): void {
		const active = this.getActiveResources();
		if (!active || this.particleCount === 0) {
			return;
		}
		if (!Number.isFinite(deltaSeconds) || deltaSeconds < 0) {
			throw new RangeError('deltaSeconds must be a finite non-negative number.');
		}
		const safeGravity = validateInteractionVector(gravity, 'gravity', MAX_GRAVITY_COMPONENT);
		const dt = Math.min(deltaSeconds, MAX_FRAME_DELTA_SECONDS);
		if (dt === 0 && this.drag === null) {
			return;
		}

		const stepParams = new ArrayBuffer(STEP_PARAMS_BYTES);
		const stepView = new DataView(stepParams);
		stepView.setFloat32(0, this.worldWidth, true);
		stepView.setFloat32(4, this.worldHeight, true);
		stepView.setFloat32(8, dt, true);
		stepView.setUint32(12, this.particleCount, true);
		stepView.setFloat32(16, safeGravity.x, true);
		stepView.setFloat32(20, safeGravity.y, true);
		stepView.setFloat32(24, this.particleRadius(), true);
		stepView.setFloat32(28, 0.38, true);
		active.session.device.queue.writeBuffer(active.resources.buffers.stepParams, 0, stepParams);

		const encoder = active.session.device.createCommandEncoder({
			label: 'Gate-0 particle interaction step'
		});
		if (dt > 0) {
			const simulationPass = encoder.beginComputePass({ label: 'bounded gravity and bounds' });
			simulationPass.setPipeline(active.resources.simulationPipeline);
			simulationPass.setBindGroup(0, active.resources.simulationBindGroup);
			simulationPass.dispatchWorkgroups(Math.ceil(this.particleCount / WORKGROUP_SIZE));
			simulationPass.end();
		}

		if (this.drag) {
			this.writeDragParams(active.session.device, active.resources.buffers.dragParams, dt);
			const dragPass = encoder.beginComputePass({ label: 'GPU particle drag' });
			dragPass.setPipeline(active.resources.dragPipeline);
			dragPass.setBindGroup(0, active.resources.dragBindGroup);
			dragPass.dispatchWorkgroups(Math.ceil(this.particleCount / WORKGROUP_SIZE));
			dragPass.end();
		}

		active.session.device.queue.submit([encoder.finish()]);
	}

	render(): void {
		const active = this.getActiveResources();
		if (!active || !this.canvas || this.canvas.width < 1 || this.canvas.height < 1) {
			return;
		}

		const params = new ArrayBuffer(RENDER_PARAMS_BYTES);
		const view = new DataView(params);
		view.setFloat32(0, this.worldWidth, true);
		view.setFloat32(4, this.worldHeight, true);
		view.setFloat32(8, this.canvas.width, true);
		view.setFloat32(12, this.canvas.height, true);
		view.setFloat32(
			16,
			Math.min(7, Math.max(1.5, Math.min(this.canvas.width, this.canvas.height) * 0.006)),
			true
		);
		active.session.device.queue.writeBuffer(active.resources.buffers.renderParams, 0, params);

		const encoder = active.session.device.createCommandEncoder({
			label: 'Gate-0 particle render'
		});
		const pass = encoder.beginRenderPass({
			label: 'colored particle circles',
			colorAttachments: [
				{
					view: active.session.context.getCurrentTexture().createView(),
					clearValue: { r: 0.035, g: 0.047, b: 0.065, a: 1 },
					loadOp: 'clear',
					storeOp: 'store'
				}
			]
		});
		pass.setPipeline(active.resources.renderPipeline);
		pass.setBindGroup(0, active.resources.renderBindGroup);
		if (this.particleCount > 0) {
			pass.draw(6, this.particleCount);
		}
		pass.end();
		active.session.device.queue.submit([encoder.finish()]);
	}

	async pick(point: MpmVector, radius: number): Promise<MpmSelection | null> {
		const active = this.getActiveResources();
		if (!active || this.particleCount === 0) {
			return null;
		}
		const query = validateInteractionVector(point, 'pick point', MAX_INTERACTION_COORDINATE);
		if (!Number.isFinite(radius) || radius < 0 || radius > MAX_PICK_RADIUS) {
			throw new RangeError(`pick radius must be finite and between 0 and ${MAX_PICK_RADIUS}.`);
		}

		let staging: GPUBuffer;
		try {
			staging = active.session.device.createBuffer({
				label: 'one-particle pick readback',
				size: PICK_RESULT_BYTES,
				usage: BUFFER_USAGE.MAP_READ | BUFFER_USAGE.COPY_DST
			});
		} catch (error) {
			if (!this.isCurrentSession(active.session, active.lifecycle)) {
				return null;
			}
			throw error;
		}
		this.pendingPickBuffers.add(staging);

		try {
			const params = new ArrayBuffer(PICK_PARAMS_BYTES);
			const view = new DataView(params);
			view.setFloat32(0, query.x, true);
			view.setFloat32(4, query.y, true);
			view.setFloat32(8, radius, true);
			view.setUint32(16, this.particleCount, true);
			active.session.device.queue.writeBuffer(active.resources.buffers.pickParams, 0, params);

			const encoder = active.session.device.createCommandEncoder({
				label: 'on-demand particle pick'
			});
			const pass = encoder.beginComputePass({ label: 'nearest particle pick' });
			pass.setPipeline(active.resources.pickPipeline);
			pass.setBindGroup(0, active.resources.pickBindGroup);
			pass.dispatchWorkgroups(1);
			pass.end();
			encoder.copyBufferToBuffer(
				active.resources.buffers.pickResult,
				0,
				staging,
				0,
				PICK_RESULT_BYTES
			);
			active.session.device.queue.submit([encoder.finish()]);

			await staging.mapAsync(MAP_MODE_READ);
			if (!this.isCurrentSession(active.session, active.lifecycle)) {
				return null;
			}

			const result = new DataView(staging.getMappedRange());
			const particleIndex = result.getUint32(0, true);
			if (particleIndex === MAX_U32 || particleIndex >= this.particleCount) {
				return null;
			}
			const object = this.objectRanges.find(
				(range) =>
					particleIndex >= range.start && particleIndex < range.start + range.count
			);
			if (!object) {
				return null;
			}

			const anchor = {
				x: result.getFloat32(4, true),
				y: result.getFloat32(8, true)
			};
			if (!Number.isFinite(anchor.x) || !Number.isFinite(anchor.y)) {
				return null;
			}
			return { particleIndex, objectId: object.objectId, anchor };
		} catch (error) {
			if (!this.isCurrentSession(active.session, active.lifecycle)) {
				return null;
			}
			throw error;
		} finally {
			this.pendingPickBuffers.delete(staging);
			safelyDestroyBuffer(staging);
		}
	}

	beginDrag(selection: MpmSelection, target: MpmVector, mode: MpmDragMode): void {
		const active = this.getActiveResources();
		if (!active) {
			return;
		}
		if (mode !== 'force' && mode !== 'rigid') {
			throw new TypeError('drag mode must be "force" or "rigid".');
		}

		const range = this.resolveSelection(selection);
		if (!range) {
			throw new RangeError('selection must identify a particle in the selected object.');
		}
		const pointer = validateInteractionVector(
			target,
			'drag target',
			MAX_INTERACTION_COORDINATE
		);
		const anchor = this.validateWorldPoint(selection.anchor, 'selection anchor');

		const encoder = active.session.device.createCommandEncoder({
			label: 'snapshot selected object for drag'
		});
		const byteOffset = range.start * PARTICLE_STRIDE_BYTES;
		const byteSize = range.count * PARTICLE_STRIDE_BYTES;
		encoder.copyBufferToBuffer(
			active.resources.buffers.particles,
			byteOffset,
			active.resources.buffers.dragOrigins,
			byteOffset,
			byteSize
		);
		active.session.device.queue.submit([encoder.finish()]);

		this.drag = {
			start: range.start,
			count: range.count,
			mode,
			anchorX: anchor.x,
			anchorY: anchor.y,
			offsetX: anchor.x - pointer.x,
			offsetY: anchor.y - pointer.y,
			targetX: pointer.x,
			targetY: pointer.y
		};
	}

	updateDrag(target: MpmVector): void {
		if (!this.drag || !this.getActiveResources()) {
			return;
		}
		const pointer = validateInteractionVector(
			target,
			'drag target',
			MAX_INTERACTION_COORDINATE
		);
		this.drag.targetX = pointer.x;
		this.drag.targetY = pointer.y;
	}

	endDrag(): void {
		this.drag = null;
	}

	async clone(selection: MpmSelection, offset: MpmVector): Promise<boolean> {
		const active = this.getActiveResources();
		if (!active) {
			return false;
		}
		const translation = validateInteractionVector(
			offset,
			'clone offset',
			MAX_INTERACTION_COORDINATE
		);
		const source = this.resolveSelection(selection);
		if (!source || source.count > this.capacity - this.particleCount) {
			return false;
		}
		const objectId = this.nextObjectId();
		if (objectId === null) {
			return false;
		}

		const start = this.particleCount;
		const particleOffset = start * PARTICLE_STRIDE_BYTES;
		const materialOffset = start * MATERIAL_STRIDE_BYTES;
		const particleBytes = source.count * PARTICLE_STRIDE_BYTES;
		const materialBytes = source.count * MATERIAL_STRIDE_BYTES;
		const encoder = active.session.device.createCommandEncoder({
			label: 'GPU-to-GPU object clone'
		});
		encoder.copyBufferToBuffer(
			active.resources.buffers.particles,
			source.start * PARTICLE_STRIDE_BYTES,
			active.resources.buffers.particles,
			particleOffset,
			particleBytes
		);
		encoder.copyBufferToBuffer(
			active.resources.buffers.materialIds,
			source.start * MATERIAL_STRIDE_BYTES,
			active.resources.buffers.materialIds,
			materialOffset,
			materialBytes
		);

		const params = new ArrayBuffer(CLONE_PARAMS_BYTES);
		const view = new DataView(params);
		view.setUint32(0, start, true);
		view.setUint32(4, source.count, true);
		view.setFloat32(16, translation.x, true);
		view.setFloat32(20, translation.y, true);
		view.setFloat32(24, this.worldWidth, true);
		view.setFloat32(28, this.worldHeight, true);
		view.setFloat32(32, this.particleRadius(), true);
		active.session.device.queue.writeBuffer(active.resources.buffers.cloneParams, 0, params);

		const pass = encoder.beginComputePass({ label: 'offset cloned particle range' });
		pass.setPipeline(active.resources.clonePipeline);
		pass.setBindGroup(0, active.resources.cloneBindGroup);
		pass.dispatchWorkgroups(Math.ceil(source.count / WORKGROUP_SIZE));
		pass.end();
		active.session.device.queue.submit([encoder.finish()]);

		this.objectRanges.push({ objectId, start, count: source.count });
		this.particleCount += source.count;
		return true;
	}

	dispose(): void {
		this.lifecycle++;
		this.disposed = true;
		const session = this.session;
		this.session = null;
		this.lostPromise = null;
		this.destroyGpuResources();
		this.destroyPendingPickBuffers();
		this.clearMetadata();
		safelyDisposeSession(session);
	}

	private getActiveResources(): {
		session: MpmWebGpuSession;
		resources: GpuResources;
		lifecycle: number;
	} | null {
		if (this.disposed || !this.session || !this.resources) {
			return null;
		}
		return {
			session: this.session,
			resources: this.resources,
			lifecycle: this.lifecycle
		};
	}

	private isCurrentSession(session: MpmWebGpuSession, lifecycle: number): boolean {
		return (
			!this.disposed &&
			this.lifecycle === lifecycle &&
			this.session === session &&
			this.resources !== null
		);
	}

	private resolveSelection(selection: MpmSelection): MpmObjectRange | null {
		if (
			typeof selection !== 'object' ||
			selection === null ||
			!Number.isSafeInteger(selection.particleIndex) ||
			selection.particleIndex < 0 ||
			!Number.isSafeInteger(selection.objectId) ||
			selection.objectId < 0
		) {
			return null;
		}
		const range = this.objectRanges.find((candidate) => candidate.objectId === selection.objectId);
		if (
			!range ||
			selection.particleIndex < range.start ||
			selection.particleIndex >= range.start + range.count
		) {
			return null;
		}
		return range;
	}

	private nextObjectId(): number | null {
		const used = new Set(this.objectRanges.map((range) => range.objectId));
		let highest = -1;
		for (const range of this.objectRanges) {
			highest = Math.max(highest, range.objectId);
		}
		if (highest < MAX_U32 && !used.has(highest + 1)) {
			return highest + 1;
		}
		for (let candidate = 0; candidate <= MAX_U32; candidate++) {
			if (!used.has(candidate)) {
				return candidate;
			}
		}
		return null;
	}

	private writeDragParams(device: GPUDevice, buffer: GPUBuffer, dt: number): void {
		const drag = this.drag;
		if (!drag) {
			return;
		}
		const params = new ArrayBuffer(DRAG_PARAMS_BYTES);
		const view = new DataView(params);
		view.setUint32(0, this.particleCount, true);
		view.setUint32(4, drag.start, true);
		view.setUint32(8, drag.count, true);
		view.setUint32(12, drag.mode === 'rigid' ? 1 : 0, true);
		view.setFloat32(16, drag.anchorX, true);
		view.setFloat32(20, drag.anchorY, true);
		view.setFloat32(24, drag.targetX + drag.offsetX, true);
		view.setFloat32(28, drag.targetY + drag.offsetY, true);
		view.setFloat32(32, dt, true);
		view.setFloat32(36, this.dragRadius(), true);
		view.setFloat32(40, 18, true);
		view.setFloat32(44, 4.5, true);
		view.setFloat32(48, this.worldWidth, true);
		view.setFloat32(52, this.worldHeight, true);
		view.setFloat32(56, this.particleRadius(), true);
		device.queue.writeBuffer(buffer, 0, params);
	}

	private validateWorldPoint(point: MpmVector, name: string): MpmVector {
		const vector = validateInteractionVector(point, name, MAX_INTERACTION_COORDINATE);
		if (
			vector.x < 0 ||
			vector.x > this.worldWidth ||
			vector.y < 0 ||
			vector.y > this.worldHeight
		) {
			throw new RangeError(`${name} must be inside the world bounds.`);
		}
		return vector;
	}

	private particleRadius(): number {
		return Math.min(0.75, Math.min(this.worldWidth, this.worldHeight) * 0.005);
	}

	private dragRadius(): number {
		return Math.max(
			this.particleRadius() * 2,
			Math.min(24, Math.min(this.worldWidth, this.worldHeight) * 0.08)
		);
	}

	private handleDeviceLoss(session: MpmWebGpuSession, lifecycle: number): void {
		if (this.lifecycle !== lifecycle || this.session !== session) {
			return;
		}
		this.session = null;
		this.disposed = true;
		this.destroyGpuResources();
		this.destroyPendingPickBuffers();
		this.clearMetadata();
		safelyDisposeSession(session);
	}

	private cleanupCurrentSession(session: MpmWebGpuSession | null): void {
		const activeSession = this.session;
		this.session = null;
		this.lostPromise = null;
		this.disposed = true;
		this.destroyGpuResources();
		this.destroyPendingPickBuffers();
		this.clearMetadata();
		safelyDisposeSession(session ?? activeSession);
	}

	private destroyGpuResources(): void {
		const resources = this.resources;
		this.resources = null;
		if (!resources) {
			return;
		}
		disposeGpuResourceSet(resources);
	}

	private destroyPendingPickBuffers(): void {
		for (const buffer of this.pendingPickBuffers) {
			this.pendingPickBuffers.delete(buffer);
			safelyDestroyBuffer(buffer);
		}
	}

	private clearMetadata(): void {
		this.canvas = null;
		this.worldWidth = 0;
		this.worldHeight = 0;
		this.capacity = 0;
		this.particleCount = 0;
		this.objectRanges = [];
		this.drag = null;
	}
}

/**
 * Creates the isolated Gate-0 interaction test backend. Its gravity/bounds integration is a
 * modest 2D demo, not MLS-MPM physics.
 */
export function createMpmInteractionBackend(): MpmBackend {
	return new MpmInteractionBackend();
}

function disposeGpuResourceSet(resources: GpuResources): void {
	for (const buffer of Object.values(resources.buffers)) {
		safelyDestroyBuffer(buffer);
	}
}

function createGpuResources(
	device: GPUDevice,
	format: GPUTextureFormat,
	layout: MpmInteractionLayout
): GpuResources {
	const allocated: GPUBuffer[] = [];
	const createBuffer = (label: string, size: number, usage: GPUBufferUsageFlags): GPUBuffer => {
		const buffer = device.createBuffer({ label, size, usage });
		allocated.push(buffer);
		return buffer;
	};

	try {
		const particleBytes = layout.capacity * PARTICLE_STRIDE_BYTES;
		const materialBytes = layout.capacity * MATERIAL_STRIDE_BYTES;
		const buffers: GpuBuffers = {
			particles: createBuffer(
				'GPU-authoritative particle position and velocity',
				particleBytes,
				BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_DST | BUFFER_USAGE.COPY_SRC
			),
			materialIds: createBuffer(
				'GPU particle material ids',
				materialBytes,
				BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_DST | BUFFER_USAGE.COPY_SRC
			),
			dragOrigins: createBuffer(
				'GPU drag origin snapshot',
				particleBytes,
				BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_DST
			),
			stepParams: createBuffer(
				'particle integration uniforms',
				STEP_PARAMS_BYTES,
				BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST
			),
			dragParams: createBuffer(
				'particle drag uniforms',
				DRAG_PARAMS_BYTES,
				BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST
			),
			pickParams: createBuffer(
				'on-demand pick uniforms',
				PICK_PARAMS_BYTES,
				BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST
			),
			pickResult: createBuffer(
				'on-demand one-particle pick result',
				PICK_RESULT_BYTES,
				BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_SRC | BUFFER_USAGE.COPY_DST
			),
			cloneParams: createBuffer(
				'GPU clone uniforms',
				CLONE_PARAMS_BYTES,
				BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST
			),
			renderParams: createBuffer(
				'particle render uniforms',
				RENDER_PARAMS_BYTES,
				BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST
			)
		};

		if (layout.particleState.byteLength > 0) {
			device.queue.writeBuffer(buffers.particles, 0, layout.particleState);
			device.queue.writeBuffer(buffers.materialIds, 0, layout.materialIds);
		}

		const simulationPipeline = device.createComputePipeline({
			label: 'bounded gravity and world bounds',
			layout: 'auto',
			compute: {
				module: device.createShaderModule({
					label: 'Gate-0 simulation WGSL',
					code: SIMULATION_SHADER
				}),
				entryPoint: 'main'
			}
		});
		const dragPipeline = device.createComputePipeline({
			label: 'GPU force or rigid drag',
			layout: 'auto',
			compute: {
				module: device.createShaderModule({
					label: 'Gate-0 drag WGSL',
					code: DRAG_SHADER
				}),
				entryPoint: 'main'
			}
		});
		const pickPipeline = device.createComputePipeline({
			label: 'on-demand nearest particle pick',
			layout: 'auto',
			compute: {
				module: device.createShaderModule({
					label: 'Gate-0 pick WGSL',
					code: PICK_SHADER
				}),
				entryPoint: 'main'
			}
		});
		const clonePipeline = device.createComputePipeline({
			label: 'GPU object clone offset',
			layout: 'auto',
			compute: {
				module: device.createShaderModule({
					label: 'Gate-0 clone WGSL',
					code: CLONE_SHADER
				}),
				entryPoint: 'main'
			}
		});
		const renderPipeline = device.createRenderPipeline({
			label: 'colored particle circles',
			layout: 'auto',
			vertex: {
				module: device.createShaderModule({
					label: 'Gate-0 particle render WGSL',
					code: RENDER_SHADER
				}),
				entryPoint: 'vertexMain'
			},
			fragment: {
				module: device.createShaderModule({
					label: 'Gate-0 particle render WGSL',
					code: RENDER_SHADER
				}),
				entryPoint: 'fragmentMain',
				targets: [
					{
						format,
						blend: {
							color: {
								srcFactor: 'src-alpha',
								dstFactor: 'one-minus-src-alpha',
								operation: 'add'
							},
							alpha: {
								srcFactor: 'one',
								dstFactor: 'one-minus-src-alpha',
								operation: 'add'
							}
						},
						writeMask: COLOR_WRITE_ALL
					}
				]
			},
			primitive: { topology: 'triangle-list' }
		});

		const simulationBindGroup = device.createBindGroup({
			label: 'particle simulation bindings',
			layout: simulationPipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: buffers.particles } },
				{ binding: 1, resource: { buffer: buffers.stepParams } }
			]
		});
		const dragBindGroup = device.createBindGroup({
			label: 'particle drag bindings',
			layout: dragPipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: buffers.particles } },
				{ binding: 1, resource: { buffer: buffers.dragOrigins } },
				{ binding: 2, resource: { buffer: buffers.dragParams } }
			]
		});
		const pickBindGroup = device.createBindGroup({
			label: 'one-particle pick bindings',
			layout: pickPipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: buffers.particles } },
				{ binding: 1, resource: { buffer: buffers.pickParams } },
				{ binding: 2, resource: { buffer: buffers.pickResult } }
			]
		});
		const cloneBindGroup = device.createBindGroup({
			label: 'GPU clone bindings',
			layout: clonePipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: buffers.particles } },
				{ binding: 1, resource: { buffer: buffers.cloneParams } }
			]
		});
		const renderBindGroup = device.createBindGroup({
			label: 'particle render bindings',
			layout: renderPipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: buffers.particles } },
				{ binding: 1, resource: { buffer: buffers.materialIds } },
				{ binding: 2, resource: { buffer: buffers.renderParams } }
			]
		});

		return {
			buffers,
			simulationPipeline,
			simulationBindGroup,
			dragPipeline,
			dragBindGroup,
			pickPipeline,
			pickBindGroup,
			clonePipeline,
			cloneBindGroup,
			renderPipeline,
			renderBindGroup
		};
	} catch (error) {
		for (const buffer of allocated) {
			safelyDestroyBuffer(buffer);
		}
		throw error;
	}
}

function assertDeviceCapacity(device: GPUDevice, capacity: number): void {
	const stateBytes = capacity * PARTICLE_STRIDE_BYTES;
	const materialBytes = capacity * MATERIAL_STRIDE_BYTES;
	if (
		stateBytes > device.limits.maxBufferSize ||
		stateBytes > device.limits.maxStorageBufferBindingSize ||
		materialBytes > device.limits.maxStorageBufferBindingSize
	) {
		throw new RangeError('capacity exceeds this WebGPU device’s storage-buffer limits.');
	}
	if (
		Math.ceil(capacity / WORKGROUP_SIZE) >
		device.limits.maxComputeWorkgroupsPerDimension
	) {
		throw new RangeError('capacity exceeds this WebGPU device’s compute-dispatch limits.');
	}
}

function validateInteractionVector(
	vector: MpmVector,
	name: string,
	maximumMagnitude: number
): MpmVector {
	if (typeof vector !== 'object' || vector === null) {
		throw new TypeError(`${name} must be a vector.`);
	}
	if (
		!Number.isFinite(vector.x) ||
		!Number.isFinite(vector.y) ||
		Math.abs(vector.x) > maximumMagnitude ||
		Math.abs(vector.y) > maximumMagnitude
	) {
		throw new RangeError(`${name} components must be finite and within ±${maximumMagnitude}.`);
	}
	return { x: Math.fround(vector.x), y: Math.fround(vector.y) };
}

function safelyDestroyBuffer(buffer: GPUBuffer): void {
	try {
		if (buffer.mapState === 'mapped') {
			buffer.unmap();
		}
	} catch (error) {
		console.warn('Could not unmap a WebGPU buffer during cleanup.', error);
	}
	try {
		buffer.destroy();
	} catch (error) {
		console.warn('Could not destroy a WebGPU buffer during cleanup.', error);
	}
}

function safelyDisposeSession(session: MpmWebGpuSession | null): void {
	if (!session) {
		return;
	}
	try {
		session.dispose();
	} catch (error) {
		console.warn('Could not dispose the WebGPU session cleanly.', error);
	}
}
