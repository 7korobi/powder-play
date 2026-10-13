import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { MpmParticle } from '../reference/solver.js';
import { createMpmWebGpuSession } from '../platform/webgpu.js';
import { createMpmMlsGpuSolver } from './solver.js';
import { calculateMpmCflMicrosteps } from './cfl.js';
import { MLS_MPM_COMPUTE_WGSL } from './shaders.js';

vi.mock('../platform/webgpu.js', () => ({
	createMpmWebGpuSession: vi.fn()
}));

beforeEach(() => {
	vi.clearAllMocks();
});

function setupDevice() {
	const dispatches: { pipeline: string; input?: string; workgroups: number }[] = [];
	const buffers: { label: string; size: number; destroy: ReturnType<typeof vi.fn> }[] = [];
	let mappedBytes: ArrayBuffer | null = null;
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
		copyBufferToBuffer: vi.fn(),
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
			const buffer = {
				label,
				size,
				destroy: vi.fn(),
				mapAsync: vi.fn(async () => {}),
				getMappedRange: vi.fn(() => mappedBytes?.slice(0, size) ?? new ArrayBuffer(size)),
				unmap: vi.fn()
			};
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
	return {
		device,
		encoder,
		pass,
		renderPass,
		queue,
		dispatches,
		buffers,
		dispose,
		getCompilationInfo,
		setMappedBytes(bytes: ArrayBuffer) {
			mappedBytes = bytes;
		}
	};
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
const FRACTURE_PARTICLES: readonly MpmParticle[] = [
	{
		...PARTICLE,
		position: { x: 7, y: 4 },
		objectId: 11
	},
	{
		...PARTICLE,
		position: { x: 8, y: 4 },
		objectId: 11
	}
];
const CANVAS = { getContext() {} } as unknown as HTMLCanvasElement;

describe('float32 deformation integration', () => {
	it('preserves the orientation of a near-singular trial at the reported increment scale', () => {
		const f32 = Math.fround;
		const initial = [
			f32(0.06047571077942848),
			f32(0.17931117117404938),
			f32(0.17301689088344574),
			f32(0.5129971504211426)
		];
		const velocityGradient = [
			f32(0.1195390224456787),
			f32(0.15536771714687347),
			f32(0.040612027049064636),
			f32(0.3162637948989868)
		];
		const rowSum = (first: number, second: number) =>
			f32(f32(Math.abs(first)) + f32(Math.abs(second)));
		const deformationIncrement = Math.max(
			rowSum(velocityGradient[0], velocityGradient[1]),
			rowSum(velocityGradient[2], velocityGradient[3])
		);
		const substeps = Math.max(1, Math.ceil(deformationIncrement / 0.25));
		const substepDt = f32(1 / substeps);
		const update = [
			f32(1 + f32(substepDt * velocityGradient[0])),
			f32(substepDt * velocityGradient[1]),
			f32(substepDt * velocityGradient[2]),
			f32(1 + f32(substepDt * velocityGradient[3]))
		];
		const multiply = (left: number[], right: number[], fused: boolean) => {
			const dot = (a: number, b: number, c: number, d: number) => {
				const secondProduct = f32(c * d);
				return fused ? f32(a * b + secondProduct) : f32(f32(a * b) + secondProduct);
			};
			return [
				dot(left[0], right[0], left[1], right[2]),
				dot(left[0], right[1], left[1], right[3]),
				dot(left[2], right[0], left[3], right[2]),
				dot(left[2], right[1], left[3], right[3])
			];
		};
		const integrate = (fused: boolean) => {
			let result = initial;
			for (let step = 0; step < substeps; step += 1) {
				result = multiply(update, result, fused);
			}
			return result;
		};
		const determinant = (matrix: number[]) =>
			matrix[0] * matrix[3] - matrix[1] * matrix[2];
		const unfusedTrial = integrate(false);
		const fusedTrial = integrate(true);

		expect(deformationIncrement).toBe(0.35687583684921265);
		expect(substeps).toBe(2);
		expect(determinant(initial)).toBeGreaterThan(0);
		expect(determinant(initial)).toBeLessThan(1e-8);
		expect(determinant(unfusedTrial)).toBeLessThanOrEqual(0);
		expect(determinant(fusedTrial)).toBeGreaterThan(0);
		expect(MLS_MPM_COMPUTE_WGSL).toContain(
			'fma(update.x, result.x, update.y * result.z)'
		);
		expect(MLS_MPM_COMPUTE_WGSL).toContain(
			'let secondProductError = fma(matrix.y, matrix.z, -secondProduct);'
		);
		expect(MLS_MPM_COMPUTE_WGSL).toContain(
			'substeps = min(substeps * 2u, maximumSubsteps);'
		);
		expect(MLS_MPM_COMPUTE_WGSL).toContain('substeps < maximumSubsteps');
		expect(MLS_MPM_COMPUTE_WGSL).toContain('atomicOr(&stateError[0], ERROR_DEFORMATION);');
	});
});

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

	it('runs every complete solver stage for each planned CFL microstep', async () => {
		const gpu = setupDevice();
		const cflBounds = { waveSpeedBound: 25, velocityBound: 1 };
		const solver = await createMpmMlsGpuSolver(CANVAS, [PARTICLE], CONFIG, cflBounds);
		const plan = calculateMpmCflMicrosteps({
			macroDt: 0.025,
			cellSize: CONFIG.grid.cellSize,
			...cflBounds,
			maxMicrosteps: 6
		});
		expect(plan.stepCount).toBe(3);

		solver.stepCflBatch(2, 0.025, plan, 1, 6, { x: 0, y: 0 });

		const stageNames = [
			'clearGrid',
			'prepareParticles',
			'buildBuckets',
			'p2g',
			'updateGrid',
			'g2p',
			'validateCflParticles',
			'commitParticles'
		];
		expect(gpu.dispatches.map(({ pipeline }) => pipeline)).toEqual(
			[
				...Array.from({ length: 6 }, () =>
					stageNames.map((name) => `MLS-MPM ${name} pipeline`)
				).flat(),
				'MLS-MPM measureParticleSpeedPartials pipeline',
				'MLS-MPM reduceParticleSpeedPartials pipeline'
			]
		);
		expect(gpu.encoder.beginComputePass).toHaveBeenCalledOnce();
		expect(gpu.queue.submit).toHaveBeenCalledOnce();
		expect(solver.cflBounds).toEqual(cflBounds);
		const simulationParameters = gpu.queue.writeBuffer.mock.calls.findLast(
			([buffer, offset, data]) =>
				buffer.label === 'MLS-MPM simulation parameters' &&
				offset === 0 &&
				data instanceof ArrayBuffer
		)?.[2] as ArrayBuffer | undefined;
		expect(simulationParameters).toBeInstanceOf(ArrayBuffer);
		expect(new DataView(simulationParameters!).getFloat32(44, true)).toBeCloseTo(
			plan.microDt
		);
		expect(new DataView(simulationParameters!).getFloat32(96, true)).toBe(
			cflBounds.waveSpeedBound
		);
		expect(new DataView(simulationParameters!).getFloat32(100, true)).toBeGreaterThan(1);
		expect(new DataView(simulationParameters!).getFloat32(104, true)).toBe(1);
		solver.dispose();
	});

	it('validates CFL bounds after triangle contact and bond passes before commit', async () => {
		const gpu = setupDevice();
		const solver = await createMpmMlsGpuSolver(CANVAS, BONDED_TRIANGLE_PARTICLES, {
			...CONFIG,
			bondRestLength: 0.6
		});
		const bounds = solver.cflBounds!;
		const velocityBound = 0;
		const plan = calculateMpmCflMicrosteps({
			macroDt: 0.001,
			cellSize: CONFIG.grid.cellSize,
			waveSpeedBound: bounds.waveSpeedBound,
			velocityBound,
			maxMicrosteps: 1
		});

		solver.stepCflBatch(1, 0.001, plan, velocityBound, 1, { x: 0, y: 0 });

		expect(gpu.dispatches.map(({ pipeline }) => pipeline)).toEqual(
			[
				'clearGrid',
				'prepareParticles',
				'buildBuckets',
				'p2g',
				'buildTriangleField',
				'updateGrid',
				'g2p',
				'applyTriangleContactImpulses',
				'solveBondedTriangles',
				'validateCflParticles',
				'commitParticles',
				'measureParticleSpeedPartials',
				'reduceParticleSpeedPartials'
			].map((name) => `MLS-MPM ${name} pipeline`)
		);
		solver.dispose();
	});

	it('requires derivable material bounds and rejects mismatched plans and multiplied cap overflow', async () => {
		const gpu = setupDevice();
		const powderConfig = {
			...CONFIG,
			material: {
				lambda: 20,
				mu: 10,
				model: 'powder' as const,
				frictionCoefficient: 0.5
			}
		};
		const solverWithoutBounds = await createMpmMlsGpuSolver(
			CANVAS,
			[PARTICLE],
			powderConfig
		);
		expect(solverWithoutBounds.cflBounds).toBeNull();
		const bounds = { waveSpeedBound: 25, velocityBound: 1 };
		const plan = calculateMpmCflMicrosteps({
			macroDt: 0.025,
			cellSize: CONFIG.grid.cellSize,
			...bounds,
			maxMicrosteps: 6
		});

		expect(() =>
			solverWithoutBounds.stepCflBatch(1, 0.025, plan, 1, 6, { x: 0, y: 0 })
		).toThrow(/no safely derived CFL bounds/);
		solverWithoutBounds.dispose();

		const solver = await createMpmMlsGpuSolver(CANVAS, [PARTICLE], CONFIG, bounds);
		expect(() =>
			solver.stepCflBatch(
				1,
				0.025,
				{ ...plan, microDt: plan.microDt / 2 },
				1,
				6,
				{ x: 0, y: 0 }
			)
		).toThrow(/does not match/);
		expect(() =>
			solver.stepCflBatch(2, 0.025, plan, 1, 5, { x: 0, y: 0 })
		).toThrow(/requested microstep count .* exceeds maxMicrosteps/);
		expect(gpu.device.createCommandEncoder).not.toHaveBeenCalled();
		solver.dispose();
	});

	it('rejects a caller velocity bound below the configured initial state', async () => {
		const movingParticle: MpmParticle = {
			...PARTICLE,
			velocity: { x: 2, y: 0 }
		};
		await expect(
			createMpmMlsGpuSolver(CANVAS, [movingParticle], CONFIG, {
				waveSpeedBound: 3,
				velocityBound: 1
			})
		).rejects.toThrow(
			/cflBounds.velocityBound must not be lower than the configured initial-state bound/
		);
		expect(createMpmWebGpuSession).not.toHaveBeenCalled();
	});

	it('rejects a per-batch velocity bound below the last measured maximum', async () => {
		const gpu = setupDevice();
		const movingParticle = { ...PARTICLE, velocity: { x: 2, y: 0 } };
		const solver = await createMpmMlsGpuSolver(CANVAS, [movingParticle], CONFIG);
		const bounds = solver.cflBounds!;
		const velocityBound = 1;
		const plan = calculateMpmCflMicrosteps({
			macroDt: 0.001,
			cellSize: CONFIG.grid.cellSize,
			waveSpeedBound: bounds.waveSpeedBound,
			velocityBound,
			maxMicrosteps: 1
		});

		expect(() =>
			solver.stepCflBatch(1, 0.001, plan, velocityBound, 1, { x: 0, y: 0 })
		).toThrow(/must cover the last submitted GPU maximum-particle envelope/);
		expect(gpu.device.createCommandEncoder).not.toHaveBeenCalled();
		solver.dispose();
	});

	it('requires the explicit drag damping factor to remain dissipative at GPU dt', async () => {
		const gpu = setupDevice();
		const solver = await createMpmMlsGpuSolver(CANVAS, [PARTICLE], CONFIG);
		const selection = {
			particleIndex: 0,
			objectId: 0,
			anchor: PARTICLE.position
		};
		solver.beginDrag(selection, PARTICLE.position, 1);
		const bounds = solver.cflBounds!;
		const plan = calculateMpmCflMicrosteps({
			macroDt: 1,
			cellSize: CONFIG.grid.cellSize,
			waveSpeedBound: bounds.waveSpeedBound,
			velocityBound: bounds.velocityBound,
			maxMicrosteps: 16
		});

		expect(plan.stepCount).toBe(7);
		expect(() =>
			solver.stepCflBatch(
				1,
				1,
				plan,
				bounds.velocityBound,
				16,
				{ x: 0, y: 0 }
			)
		).toThrow(/explicit drag damping update to remain dissipative/);
		expect(gpu.device.createCommandEncoder).not.toHaveBeenCalled();
		solver.dispose();
	});

	it('invalidates the CFL velocity state after a direct non-CFL step', async () => {
		const gpu = setupDevice();
		const solver = await createMpmMlsGpuSolver(CANVAS, [PARTICLE], CONFIG);
		const bounds = solver.cflBounds!;
		const plan = calculateMpmCflMicrosteps({
			macroDt: 0.001,
			cellSize: CONFIG.grid.cellSize,
			waveSpeedBound: bounds.waveSpeedBound,
			velocityBound: bounds.velocityBound,
			maxMicrosteps: 1
		});
		solver.step(0.001, { x: 0, y: 0 });

		expect(() =>
			solver.stepCflBatch(1, 0.001, plan, bounds.velocityBound, 1, { x: 0, y: 0 })
		).toThrow(/invalidated the tracked velocity bound/);
		solver.dispose();
	});

	it('reads a compact speed scalar and numerical status without reading particle buffers', async () => {
		const gpu = setupDevice();
		const solver = await createMpmMlsGpuSolver(CANVAS, [PARTICLE], CONFIG);
		const bounds = solver.cflBounds!;
		const plan = calculateMpmCflMicrosteps({
			macroDt: 0.001,
			cellSize: CONFIG.grid.cellSize,
			waveSpeedBound: bounds.waveSpeedBound,
			velocityBound: 0,
			maxMicrosteps: 1
		});
		solver.stepCflBatch(1, 0.001, plan, 0, 1, { x: 0, y: 0 });
		expect(() =>
			solver.stepCflBatch(1, 0.001, plan, 0, 1, { x: 0, y: 0 })
		).toThrow(/queue measureMaximumParticleSpeed\(\).*before submitting another/);

		await expect(solver.measureMaximumParticleSpeed()).resolves.toEqual({
			maximumSpeed: 0,
			sequence: 1
		});

		expect(gpu.encoder.copyBufferToBuffer).toHaveBeenCalledTimes(2);
		expect(gpu.encoder.copyBufferToBuffer).toHaveBeenNthCalledWith(
			1,
			expect.objectContaining({ label: 'MLS-MPM CFL maximum-speed scalar' }),
			0,
			expect.objectContaining({ label: 'MLS-MPM compact CFL speed/status readback' }),
			0,
			4
		);
		expect(gpu.encoder.copyBufferToBuffer).toHaveBeenNthCalledWith(
			2,
			expect.objectContaining({ label: 'MLS-MPM numerical status' }),
			0,
			expect.objectContaining({ label: 'MLS-MPM compact CFL speed/status readback' }),
			4,
			32
		);
		expect(
			gpu.encoder.copyBufferToBuffer.mock.calls.some(([source]) =>
				String(source.label).includes('particle state')
			)
		).toBe(false);
		solver.dispose();
	});

	it('surfaces a GPU CFL-bound violation from the compact status readback', async () => {
		const gpu = setupDevice();
		const solver = await createMpmMlsGpuSolver(CANVAS, [PARTICLE], CONFIG);
		const bounds = solver.cflBounds!;
		const velocityBound = 1;
		const plan = calculateMpmCflMicrosteps({
			macroDt: 0.001,
			cellSize: CONFIG.grid.cellSize,
			waveSpeedBound: bounds.waveSpeedBound,
			velocityBound,
			maxMicrosteps: 1
		});
		solver.stepCflBatch(1, 0.001, plan, velocityBound, 1, { x: 0, y: 0 });

		const status = new ArrayBuffer(4 + 32);
		new DataView(status).setUint32(4, 16384, true);
		new DataView(status).setUint32(8, 0, true);
		gpu.setMappedBytes(status);
		await expect(solver.measureMaximumParticleSpeed()).rejects.toThrow(
			/particle exceeded the enforced CFL wave-speed or velocity bound at particle 0/
		);
		expect(() =>
			solver.stepCflBatch(1, 0.001, plan, velocityBound, 1, { x: 0, y: 0 })
		).toThrow(/failed CFL measurement invalidated the tracked velocity bound/);
		solver.dispose();
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

	it('appends only validated particles within reserved bounds and reports capacity outcomes', async () => {
		const gpu = setupDevice();
		const solver = await createMpmMlsGpuSolver(
			CANVAS,
			[WATER_PARTICLE, OIL_PARTICLE],
			{ ...CONFIG, capacity: 4 }
		);
		const writesBeforeInvalidBatch = gpu.queue.writeBuffer.mock.calls.length;
		const invalid = solver.appendParticles([
			{ ...PARTICLE, position: { x: CONFIG.grid.cellsX + 1, y: 4 } }
		]);
		expect(invalid).toMatchObject({
			requested: 1,
			accepted: 0,
			refused: 1,
			dropped: 0,
			activeParticleCount: 2
		});
		expect(gpu.queue.writeBuffer.mock.calls).toHaveLength(writesBeforeInvalidBatch);

		const appended = solver.appendParticles([
			{ ...PARTICLE, position: { x: 10, y: 4 }, velocity: { x: 3, y: 4 } },
			{ ...PARTICLE, position: { x: 11, y: 4 } },
			{ ...PARTICLE, position: { x: 12, y: 4 } }
		]);

		expect(appended).toMatchObject({
			requested: 3,
			accepted: 2,
			refused: 0,
			dropped: 1,
			activeParticleCount: 4,
			allocatedParticleCount: 4,
			capacity: 4
		});
		expect(appended.velocityBound).toBeCloseTo(5, 4);
		const stateWrites = gpu.queue.writeBuffer.mock.calls.filter(([buffer, offset]) =>
			String(buffer.label).includes('particle state') && offset === 2 * 80
		);
		expect(stateWrites).toHaveLength(2);
		for (const [buffer, offset, data] of stateWrites) {
			expect(offset).toBe(2 * 80);
			expect(offset + data.byteLength).toBeLessThanOrEqual(buffer.size);
			expect(data.byteLength).toBe(2 * 80);
		}

		const writesBeforeFullCapacityAttempt = gpu.queue.writeBuffer.mock.calls.length;
		const full = solver.appendParticles([{ ...PARTICLE, position: { x: 13, y: 4 } }]);
		expect(full).toMatchObject({
			requested: 1,
			accepted: 0,
			refused: 0,
			dropped: 1,
			activeParticleCount: 4,
			allocatedParticleCount: 4,
			capacity: 4
		});
		expect(gpu.queue.writeBuffer.mock.calls).toHaveLength(writesBeforeFullCapacityAttempt);
		solver.dispose();
	});

	it('removes outlet liquids on the GPU without freeing or reusing their slots', async () => {
		const gpu = setupDevice();
		const solver = await createMpmMlsGpuSolver(
			CANVAS,
			[WATER_PARTICLE, OIL_PARTICLE, PARTICLE],
			{ ...CONFIG, capacity: 4 }
		);
		gpu.setMappedBytes(new Uint32Array([1]).buffer);

		await expect(
			solver.removeParticlesInOutlet({
				minX: 0,
				minY: 0,
				maxX: CONFIG.grid.cellsX,
				maxY: CONFIG.grid.cellsY
			})
		).resolves.toBe(1);

		expect(gpu.dispatches.map(({ pipeline }) => pipeline)).toContain(
			'MLS-MPM removeParticlesInOutlet pipeline'
		);
		expect(solver.particleCount).toBe(2);
		expect(solver.allocatedParticleCount).toBe(3);
		const appended = solver.appendParticles([{ ...PARTICLE, position: { x: 10, y: 4 } }]);
		expect(appended).toMatchObject({
			accepted: 1,
			activeParticleCount: 3,
			allocatedParticleCount: 4,
			capacity: 4
		});
		const appendedStateWrites = gpu.queue.writeBuffer.mock.calls.filter(
			([buffer, offset]) =>
				String(buffer.label).includes('particle state') && offset === 3 * 80
		);
		expect(appendedStateWrites).toHaveLength(2);
		solver.dispose();
	});

	it('dispatches GPU fracture damage and refuses to clone a bonded object', async () => {
		const gpu = setupDevice();
		const solver = await createMpmMlsGpuSolver(
			CANVAS,
			FRACTURE_PARTICLES,
			{
				...CONFIG,
				capacity: 4,
				fractureBonds: [
					{
						firstParticle: 0,
						secondParticle: 1,
						restLength: 1,
						failureStrain: 0.1,
						damage: 1,
						broken: true
					}
				]
			}
		);

		expect(
			solver.appendParticles([{ ...PARTICLE, position: { x: 9, y: 4 } }])
		).toMatchObject({
			accepted: 1,
			activeParticleCount: 3,
			allocatedParticleCount: 3
		});
		solver.stepBatch(1, 0.001, { x: 0, y: 0 });
		const pipelines = gpu.dispatches.map(({ pipeline }) => pipeline);
		expect(pipelines).toContain('MLS-MPM updateFractureBonds pipeline');
		expect(pipelines).toContain('MLS-MPM applyFractureParticleDamage pipeline');
		expect(pipelines.indexOf('MLS-MPM updateFractureBonds pipeline')).toBeLessThan(
			pipelines.indexOf('MLS-MPM applyFractureParticleDamage pipeline')
		);
		expect(pipelines.indexOf('MLS-MPM applyFractureParticleDamage pipeline')).toBeLessThan(
			pipelines.indexOf('MLS-MPM commitParticles pipeline')
		);
		expect(
			solver.clone(
				{ particleIndex: 0, objectId: 11, anchor: FRACTURE_PARTICLES[0].position },
				{ x: 1, y: 0 }
			)
		).toBe(false);
		expect(solver.particleCount).toBe(3);
		expect(
			gpu.dispatches.some(({ pipeline }) => pipeline === 'MLS-MPM cloneObject pipeline')
		).toBe(false);
		expect(MLS_MPM_COMPUTE_WGSL).toContain('if (bond.broken != 0u)');
		const applyDamageStart = MLS_MPM_COMPUTE_WGSL.indexOf(
			'fn applyFractureParticleDamage'
		);
		const initialTopologyBound = MLS_MPM_COMPUTE_WGSL.indexOf(
			'particleIndex >= params.fracture.x',
			applyDamageStart
		);
		const firstTopologyRead = MLS_MPM_COMPUTE_WGSL.indexOf(
			'let firstOffset = fractureBondTopology[particleIndex]',
			applyDamageStart
		);
		expect(initialTopologyBound).toBeGreaterThan(applyDamageStart);
		expect(initialTopologyBound).toBeLessThan(firstTopologyRead);
		solver.dispose();
	});
});
