import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { MpmParticle } from '../reference/solver.js';
import { createMpmWebGpuSession } from '../platform/webgpu.js';
import { createMpmMlsGpuSolver } from './solver.js';

vi.mock('../platform/webgpu.js', () => ({
	createMpmWebGpuSession: vi.fn()
}));

beforeEach(() => {
	vi.clearAllMocks();
});

function setupDevice() {
	const dispatches: { pipeline: string; input?: string; workgroups: number }[] = [];
	const buffers: { label: string; size: number; destroy: ReturnType<typeof vi.fn> }[] = [];
	const pass = {
		setPipeline: vi.fn(),
		setBindGroup: vi.fn(),
		dispatchWorkgroups: vi.fn((workgroups: number) => {
			const pipeline = pass.setPipeline.mock.lastCall?.[0];
			const bindGroup = pass.setBindGroup.mock.lastCall?.[1];
			const input = bindGroup.entries.find((entry: GPUBindGroupEntry) => entry.binding === 1);
			dispatches.push({
				pipeline: pipeline.label,
				input: input?.resource.buffer.label,
				workgroups
			});
		}),
		end: vi.fn()
	};
	const encoder = { beginComputePass: vi.fn(() => pass), finish: vi.fn(() => ({})) };
	const queue = { writeBuffer: vi.fn(), submit: vi.fn() };
	const device = {
		limits: {
			maxBufferSize: 256 * 1024 * 1024,
			maxStorageBufferBindingSize: 128 * 1024 * 1024,
			maxUniformBufferBindingSize: 64 * 1024,
			maxComputeWorkgroupsPerDimension: 65_535,
			maxStorageBuffersPerShaderStage: 8
		},
		lost: new Promise<GPUDeviceLostInfo>(() => {}),
		queue,
		pushErrorScope: vi.fn(),
		popErrorScope: vi.fn(async () => null),
		createShaderModule: vi.fn(() => ({ getCompilationInfo: async () => ({ messages: [] }) })),
		createComputePipelineAsync: vi.fn(async ({ label }) => ({
			label, getBindGroupLayout: () => ({})
		})),
		createRenderPipelineAsync: vi.fn(async () => ({ getBindGroupLayout: () => ({}) })),
		createBuffer: vi.fn(({ label, size }) => {
			const buffer = { label, size, destroy: vi.fn() };
			buffers.push(buffer);
			return buffer;
		}),
		createBindGroup: vi.fn((descriptor) => descriptor),
		createCommandEncoder: vi.fn(() => encoder)
	};
	const dispose = vi.fn();
	vi.mocked(createMpmWebGpuSession).mockResolvedValue({
		adapter: {} as GPUAdapter,
		device: device as unknown as GPUDevice,
		context: {} as GPUCanvasContext,
		format: 'bgra8unorm',
		lost: device.lost,
		dispose
	});
	return { device, encoder, pass, queue, dispatches, buffers, dispose };
}

const CONFIG = {
	grid: { cellsX: 16, cellsY: 8, cellSize: 1 },
	particleMass: 0.1,
	particleVolume: 0.01,
	material: { lambda: 20, mu: 10 },
	capacity: 4
};
const PARTICLE: MpmParticle = {
	position: { x: 8, y: 4 },
	velocity: { x: 0, y: 0 },
	C: { m00: 0, m01: 0, m10: 0, m11: 0 },
	F: { m00: 1, m01: 0, m10: 0, m11: 1 }
};
const CANVAS = { getContext() {} } as unknown as HTMLCanvasElement;

describe('GPU MLS-MPM batch encoding', () => {
	it('uses one pass with ordered prepare and physics dispatches and correct ping-pong inputs', async () => {
		const gpu = setupDevice();
		const solver = await createMpmMlsGpuSolver(CANVAS, [PARTICLE], CONFIG);
		solver.stepBatch(3, 0.001, { x: 0, y: 0 });

		const names = ['clearGrid', 'prepareParticles', 'buildBuckets', 'p2g', 'updateGrid', 'g2p', 'commitParticles'];
		expect(gpu.dispatches.map(({ pipeline }) => pipeline)).toEqual(
			Array.from({ length: 3 }, () => names.map((name) => `MLS-MPM ${name} pipeline`)).flat()
		);
		expect(gpu.dispatches.filter(({ pipeline }) => pipeline.includes('prepareParticles')).map(({ input }) => input))
			.toEqual(['MLS-MPM particle state A', 'MLS-MPM particle state B', 'MLS-MPM particle state A']);
		expect(gpu.encoder.beginComputePass).toHaveBeenCalledOnce();
		expect(gpu.pass.end).toHaveBeenCalledOnce();
		expect(gpu.queue.submit).toHaveBeenCalledOnce();

		solver.step(0.001, { x: 0, y: 0 });
		expect(gpu.dispatches.at(22)?.input).toBe('MLS-MPM particle state B');
		solver.dispose();
		expect(gpu.buffers.find(({ label }) => label.includes('prepared affine'))?.size).toBe(64);
		for (const buffer of gpu.buffers) expect(buffer.destroy).toHaveBeenCalledOnce();
		expect(gpu.dispose).toHaveBeenCalledOnce();
	});

	it('skips zero-workgroup particle dispatches while updating the empty grid', async () => {
		const gpu = setupDevice();
		const solver = await createMpmMlsGpuSolver(CANVAS, [], { ...CONFIG, capacity: undefined });
		solver.stepBatch(2, 0.001, { x: 0, y: 0 });
		expect(gpu.dispatches.map(({ pipeline, workgroups }) => [pipeline, workgroups])).toEqual(
			Array.from({ length: 2 }, () => [
				['MLS-MPM clearGrid pipeline', 4],
				['MLS-MPM p2g pipeline', 4],
				['MLS-MPM updateGrid pipeline', 4]
			]).flat()
		);
		expect(gpu.buffers.find(({ label }) => label.includes('prepared affine'))?.size).toBe(16);
		solver.dispose();
	});

	it('rejects invalid batches before encoding and does not advance buffers after a failed submit', async () => {
		const gpu = setupDevice();
		const solver = await createMpmMlsGpuSolver(CANVAS, [PARTICLE], CONFIG);
		expect(() => solver.stepBatch(0, 0.001, { x: 0, y: 0 })).toThrow(/stepCount/);
		expect(() => solver.stepBatch(1, 0, { x: 0, y: 0 })).toThrow(/dt/);
		expect(gpu.device.createCommandEncoder).not.toHaveBeenCalled();

		gpu.queue.submit.mockImplementationOnce(() => { throw new Error('submit failure'); });
		expect(() => solver.stepBatch(3, 0.001, { x: 0, y: 0 })).toThrow(/encode or submit/);
		gpu.dispatches.length = 0;
		solver.step(0.001, { x: 0, y: 0 });
		expect(gpu.dispatches[1].input).toBe('MLS-MPM particle state A');
		solver.dispose();
	});

	it('keeps cloning in its own compute pass after a simulation batch', async () => {
		const gpu = setupDevice();
		const solver = await createMpmMlsGpuSolver(CANVAS, [PARTICLE], CONFIG);
		solver.stepBatch(2, 0.001, { x: 0, y: 0 });
		expect(solver.clone(
			{ particleIndex: 0, objectId: 0, anchor: PARTICLE.position },
			{ x: 0.1, y: 0 }
		)).toBe(true);
		expect(solver.particleCount).toBe(2);
		expect(gpu.dispatches.at(-1)?.pipeline).toBe('MLS-MPM cloneObject pipeline');
		expect(gpu.encoder.beginComputePass).toHaveBeenCalledTimes(2);
		expect(gpu.pass.end).toHaveBeenCalledTimes(2);
		solver.dispose();
	});
});
