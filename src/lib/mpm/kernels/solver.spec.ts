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
	const renderPass = {
		setPipeline: vi.fn(),
		setBindGroup: vi.fn(),
		draw: vi.fn(),
		end: vi.fn()
	};
	const encoder = {
		beginComputePass: vi.fn(() => pass),
		beginRenderPass: vi.fn(() => renderPass),
		finish: vi.fn(() => ({}))
	};
	const queue = { writeBuffer: vi.fn(), submit: vi.fn() };
	const getCompilationInfo = vi.fn(async () => ({ messages: [] }));
	const context = {
		getCurrentTexture: vi.fn(() => ({ createView: vi.fn(() => ({})) }))
	};
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
		createShaderModule: vi.fn(() => ({ getCompilationInfo })),
		createComputePipelineAsync: vi.fn(async ({ label }) => ({
			label, getBindGroupLayout: () => ({})
		})),
		createRenderPipelineAsync: vi.fn(async ({ label }) => ({
			label, getBindGroupLayout: () => ({})
		})),
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
		context: context as unknown as GPUCanvasContext,
		format: 'bgra8unorm',
		lost: device.lost,
		dispose
	});
	return { device, encoder, pass, renderPass, queue, dispatches, buffers, dispose, getCompilationInfo };
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
const WATER_PARTICLE: MpmParticle = {
	...PARTICLE,
	material: { lambda: 8, mu: 0, model: 'water', density: 0.02 },
	volume: 25
};
const OIL_PARTICLE: MpmParticle = {
	...PARTICLE,
	position: { x: 9, y: 4 },
	material: { lambda: 6, mu: 0, model: 'oil', density: 0.016 },
	volume: 25
};
const BONDED_TRIANGLE_PARTICLES: readonly MpmParticle[] = [
	{
		...PARTICLE,
		position: { x: 8, y: 4 },
		objectId: 9,
		bondedTriangleVertex: 0
	},
	{
		...PARTICLE,
		position: { x: 8.6, y: 4 },
		objectId: 9,
		bondedTriangleVertex: 1
	},
	{
		...PARTICLE,
		position: { x: 8.3, y: 4.519615242270663 },
		objectId: 9,
		bondedTriangleVertex: 2
	}
];
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

	it('dispatches bonded triangle field, reaction, and constraint passes in dependency order', async () => {
		const gpu = setupDevice();
		const solver = await createMpmMlsGpuSolver(CANVAS, BONDED_TRIANGLE_PARTICLES, {
			...CONFIG,
			capacity: 6,
			bondRestLength: 0.6
		});
		solver.step(0.001, { x: 0, y: 0 });

		expect(gpu.dispatches.map(({ pipeline }) => pipeline)).toEqual([
			'MLS-MPM clearGrid pipeline',
			'MLS-MPM prepareParticles pipeline',
			'MLS-MPM buildBuckets pipeline',
			'MLS-MPM p2g pipeline',
			'MLS-MPM buildTriangleField pipeline',
			'MLS-MPM updateGrid pipeline',
			'MLS-MPM g2p pipeline',
			'MLS-MPM applyTriangleContactImpulses pipeline',
			'MLS-MPM solveBondedTriangles pipeline',
			'MLS-MPM commitParticles pipeline'
		]);
		expect(gpu.buffers.find(({ label }) => label.includes('triangle contact field'))?.size).toBe(
			209 * 32
		);
		expect(gpu.buffers.find(({ label }) => label.includes('bonded triangle body indices'))?.size).toBe(
			2 * Uint32Array.BYTES_PER_ELEMENT
		);
		solver.dispose();
	});

	describe('GPU MLS-MPM free-surface rendering', () => {
		it('updates live bucket counts after GPU-side clones', async () => {
			const gpu = setupDevice();
			const solver = await createMpmMlsGpuSolver(CANVAS, [PARTICLE], CONFIG);
			expect(
				solver.clone(
					{ particleIndex: 0, objectId: 0, anchor: PARTICLE.position },
					{ x: 0.1, y: 0 }
				)
			).toBe(true);
			expect(gpu.queue.writeBuffer).toHaveBeenLastCalledWith(
				expect.objectContaining({ label: 'MLS-MPM simulation parameters' }),
				16,
				new Uint32Array([2, 209, 18, 0])
			);

			gpu.dispatches.length = 0;
			solver.render('surface');
			expect(gpu.dispatches.find(({ pipeline }) => pipeline === 'MLS-MPM buildBuckets pipeline'))
				.toMatchObject({ input: 'MLS-MPM particle state A', workgroups: 1 });
			solver.dispose();
		});

		it('gathers water and oil from the final batch buffer before drawing surfaces without liquid markers', async () => {
			const gpu = setupDevice();
			const solver = await createMpmMlsGpuSolver(
				CANVAS,
				[WATER_PARTICLE, OIL_PARTICLE],
				CONFIG
			);
			expect(gpu.getCompilationInfo).toHaveBeenCalledTimes(2);
			solver.stepBatch(1, 0.001, { x: 0, y: 0 });

			const physicsBucket = gpu.dispatches.find(
				({ pipeline }) => pipeline === 'MLS-MPM buildBuckets pipeline'
			);
			expect(physicsBucket?.input).toBe('MLS-MPM particle state A');

			gpu.dispatches.length = 0;
			solver.render('surface');

			expect(gpu.dispatches.map(({ pipeline, input }) => [pipeline, input])).toEqual([
				['MLS-MPM clearSurfaceOccupancy pipeline', undefined],
				['MLS-MPM buildBuckets pipeline', 'MLS-MPM particle state B'],
				['MLS-MPM gatherSurfaceOccupancy pipeline', 'MLS-MPM particle state B']
			]);
			expect(gpu.renderPass.setPipeline.mock.calls.map(([pipeline]) => pipeline.label)).toEqual([
				'MLS-MPM water/oil surface render pipeline',
				'MLS-MPM particle render pipeline'
			]);
			expect(gpu.renderPass.draw.mock.calls).toEqual([
				[3, 1],
				[6, 2]
			]);

			gpu.dispatches.length = 0;
			gpu.renderPass.draw.mockClear();
			solver.render('particles');
			expect(gpu.dispatches).toEqual([]);
			expect(gpu.renderPass.draw.mock.calls).toEqual([[6, 2]]);
			expect(() => solver.render('unsupported' as 'surface')).toThrow(/render mode/);
			solver.dispose();
		});
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
