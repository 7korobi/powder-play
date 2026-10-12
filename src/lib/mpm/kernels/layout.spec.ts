import { describe, expect, it } from 'vitest';
import type { MpmParticle } from '../reference/solver.js';
import {
	calculateMpmMlsGpuBufferLayout,
	packMpmMlsGpuStepParameters,
	validateMpmMlsGpuInput,
	type MpmMlsGpuConfig,
	type MpmMlsGpuDeviceLimits
} from './layout.js';

const LIMITS: MpmMlsGpuDeviceLimits = {
	maxBufferSize: 256 * 1024 * 1024,
	maxStorageBufferBindingSize: 128 * 1024 * 1024,
	maxUniformBufferBindingSize: 64 * 1024,
	maxComputeWorkgroupsPerDimension: 65_535,
	maxStorageBuffersPerShaderStage: 8
};

const ZERO_MATRIX = { m00: 0, m01: 0, m10: 0, m11: 0 };
const IDENTITY = { m00: 1, m01: 0, m10: 0, m11: 1 };

function makeConfig(overrides: Partial<MpmMlsGpuConfig> = {}): MpmMlsGpuConfig {
	return {
		grid: { cellsX: 16, cellsY: 8, cellSize: 1 },
		particleMass: 0.1,
		particleVolume: 0.01,
		material: { lambda: 20, mu: 10 },
		...overrides
	};
}

function makeParticle(overrides: Partial<MpmParticle> = {}): MpmParticle {
	return {
		position: { x: 8, y: 4 },
		velocity: { x: 0, y: 0 },
		C: ZERO_MATRIX,
		F: IDENTITY,
		...overrides
	};
}

describe('GPU MLS-MPM input and buffer layout', () => {
	it('validates and packs the complete particle state, including both domain edges', () => {
		const state = validateMpmMlsGpuInput(
			[
				makeParticle({ position: { x: 0, y: 0 } }),
				makeParticle({ position: { x: 16, y: 8 } })
			],
			makeConfig()
		);

		expect(Array.from(state.packedParticles.slice(0, 12))).toEqual([
			0, 0, 0, 0,
			0, 0, 0, 0,
			1, 0, 0, 1
		]);
		expect(state.packedParticles[12]).toBe(1);
		expect(new Uint32Array(state.packedParticles.buffer)[13]).toBe(0);
		expect(Array.from(state.packedParticles.slice(14, 16))).toEqual([20, 10]);
		expect(state.particleCount).toBe(2);
		expect(state.particleCapacity).toBe(2);
		expect(Array.from(state.packedObjectIds)).toEqual([0, 0]);
		expect(state.domainWidth).toBe(16);
		expect(state.domainHeight).toBe(8);
		expect(state.domainWidthF32).toBe(16);
		expect(state.domainHeightF32).toBe(8);
	});

	it('reserves a one-node halo around the physical grid and sizes all buffers', () => {
		const state = validateMpmMlsGpuInput([makeParticle()], makeConfig());
		const layout = calculateMpmMlsGpuBufferLayout(state, LIMITS);

		expect(layout).toEqual({
			nodeWidth: 19,
			nodeHeight: 11,
			nodeCount: 209,
			pairCount: 9,
			particleCapacity: 1,
			particleStateBytes: 80,
			particleBufferBytes: 80,
			objectIdBufferBytes: 4,
			bucketHeadBytes: 836,
			bucketNextBytes: 36,
			gridStateBytes: 5016,
			readbackBytes: 88
		});
	});

	it('keeps empty-particle allocations valid without dispatching particle buffers of size zero', () => {
		const state = validateMpmMlsGpuInput([], makeConfig());
		const layout = calculateMpmMlsGpuBufferLayout(state, LIMITS);

		expect(layout.particleCapacity).toBe(0);
		expect(layout.particleStateBytes).toBe(0);
		expect(layout.particleBufferBytes).toBe(80);
		expect(layout.bucketNextBytes).toBe(4);
		expect(layout.objectIdBufferBytes).toBe(4);
		expect(layout.readbackBytes).toBe(8);
	});

	it('reserves capacity up front while packing only initial particle and object state', () => {
		const particles = [
			makeParticle({ objectId: 7 }),
			makeParticle({ position: { x: 9, y: 4 }, objectId: 7 })
		];
		const state = validateMpmMlsGpuInput(particles, makeConfig({ capacity: 4 }));
		const layout = calculateMpmMlsGpuBufferLayout(state, LIMITS);

		expect(state.particleCount).toBe(2);
		expect(state.particleCapacity).toBe(4);
		expect(Array.from(state.packedObjectIds)).toEqual([7, 7]);
		expect(state.objectRanges).toEqual([{ objectId: 7, start: 0, count: 2 }]);
		expect(layout.pairCount).toBe(36);
		expect(layout.particleStateBytes).toBe(160);
		expect(layout.particleBufferBytes).toBe(320);
		expect(layout.objectIdBufferBytes).toBe(16);
		expect(layout.bucketNextBytes).toBe(144);
		expect(layout.readbackBytes).toBe(340);
	});

	it('packs each particle material and snow plastic history with the GPU layout', () => {
		const particles = [
			makeParticle({
				objectId: 3,
				plasticVolume: 0.82,
				material: { lambda: 36, mu: 18, model: 'snow' }
			}),
			makeParticle({
				position: { x: 9, y: 4 },
				objectId: 3,
				material: { lambda: 24, mu: 12, model: 'neo-hookean' }
			})
		];
		const state = validateMpmMlsGpuInput(particles, makeConfig());
		const words = new Uint32Array(state.packedParticles.buffer);

		expect(state.packedParticles[12]).toBe(Math.fround(0.82));
		expect(words[13]).toBe(1);
		expect(Array.from(state.packedParticles.slice(14, 16))).toEqual([36, 18]);
		expect(state.packedParticles[32]).toBe(1);
		expect(words[33]).toBe(0);
		expect(Array.from(state.packedParticles.slice(34, 36))).toEqual([24, 12]);
		expect(Array.from(state.packedParticles.slice(16, 20))).toEqual([
			0,
			Math.fround(0.01),
			0,
			0
		]);
	});

	it('packs liquid density, per-particle volume, and powder friction', () => {
		const particles = [
			makeParticle({
				material: { lambda: 8, mu: 0, model: 'water', density: 1 },
				volume: 0.02
			}),
			makeParticle({
				position: { x: 9, y: 4 },
				material: { lambda: 6, mu: 0, model: 'oil', density: 0.8 }
			}),
			makeParticle({
				position: { x: 10, y: 4 },
				material: {
					lambda: 20,
					mu: 10,
					model: 'powder',
					frictionCoefficient: 0.55
				}
			})
		];
		const state = validateMpmMlsGpuInput(particles, makeConfig());
		const words = new Uint32Array(state.packedParticles.buffer);

		expect(words[13]).toBe(2);
		expect(Array.from(state.packedParticles.slice(14, 19))).toEqual([
			8, 0, 1, Math.fround(0.02), 0
		]);
		expect(words[33]).toBe(3);
		expect(state.packedParticles[36]).toBe(Math.fround(0.8));
		expect(state.packedParticles[37]).toBe(Math.fround(0.01));
		expect(words[53]).toBe(4);
		expect(state.packedParticles[58]).toBe(Math.fround(0.55));
	});

	it('rejects invalid physical parameters and particle states rather than repairing them', () => {
		expect(() =>
			validateMpmMlsGpuInput([], makeConfig({ grid: { cellsX: 0, cellsY: 8, cellSize: 1 } }))
		).toThrow(/cellsX/);
		expect(() =>
			validateMpmMlsGpuInput([], makeConfig({ particleMass: 0 }))
		).toThrow(/particleMass/);
		expect(() =>
			validateMpmMlsGpuInput(
				[],
				makeConfig({ grid: { cellsX: 16, cellsY: 8, cellSize: 1e-50 } })
			)
		).toThrow(/float32 range/);
		expect(() =>
			validateMpmMlsGpuInput(
				[makeParticle({ position: { x: 16.01, y: 4 } })],
				makeConfig()
			)
		).toThrow(/grid domain/);
		expect(() =>
			validateMpmMlsGpuInput(
				[makeParticle({ F: { m00: -1, m01: 0, m10: 0, m11: 1 } })],
				makeConfig()
			)
		).toThrow(/determinant/);
		expect(() =>
			validateMpmMlsGpuInput(
				[makeParticle({ velocity: { x: Number.MAX_VALUE, y: 0 } })],
				makeConfig()
			)
		).toThrow(/float32 range/);
		expect(() =>
			validateMpmMlsGpuInput(
				[makeParticle({ C: { ...ZERO_MATRIX, m00: 1e-50 } })],
				makeConfig()
			)
		).toThrow(/underflows/);
		expect(() =>
			validateMpmMlsGpuInput(
				[makeParticle({ plasticVolume: 0 })],
				makeConfig()
			)
		).toThrow(/plasticVolume/);
		expect(() =>
			validateMpmMlsGpuInput(
				[makeParticle({ material: { lambda: 20, mu: 10, model: 'liquid' as 'neo-hookean' } })],
				makeConfig()
			)
		).toThrow(/model/);
		expect(() =>
			validateMpmMlsGpuInput(
				[
					makeParticle({
						material: {
							lambda: 8,
							mu: 10,
							model: 'water',
							density: 1
						} as MpmParticle['material']
					})
				],
				makeConfig()
			)
		).toThrow(/mu must be zero/);
		expect(() =>
			validateMpmMlsGpuInput(
				[
					makeParticle({
						material: { lambda: 20, mu: 10, model: 'powder' } as MpmParticle['material']
					})
				],
				makeConfig()
			)
		).toThrow(/frictionCoefficient/);
		expect(() =>
			validateMpmMlsGpuInput(
				[makeParticle({ volume: 0 })],
				makeConfig()
			)
		).toThrow(/volume/);
		expect(() => validateMpmMlsGpuInput([], makeConfig({ capacity: 0 }))).toThrow(
			/capacity/
		);
		expect(() =>
			validateMpmMlsGpuInput([makeParticle()], makeConfig({ capacity: 0 }))
		).toThrow(/capacity/);
		expect(() =>
			validateMpmMlsGpuInput([makeParticle({ objectId: -1 })], makeConfig())
		).toThrow(/objectId/);
		expect(() =>
			validateMpmMlsGpuInput(
				[
					makeParticle({ objectId: 1 }),
					makeParticle({ objectId: 2 }),
					makeParticle({ objectId: 1 })
				],
				makeConfig()
			)
		).toThrow(/must be contiguous/);
	});

	it('rejects device-limit violations and signed linked-list index overflow', () => {
		const state = validateMpmMlsGpuInput([makeParticle()], makeConfig());
		expect(() =>
			calculateMpmMlsGpuBufferLayout(state, {
				...LIMITS,
				maxComputeWorkgroupsPerDimension: 1
			})
		).toThrow(/maxComputeWorkgroupsPerDimension/);
		expect(() =>
			calculateMpmMlsGpuBufferLayout(state, {
				...LIMITS,
				maxStorageBufferBindingSize: 100
			})
		).toThrow(/maxStorageBufferBindingSize/);

		expect(() =>
			calculateMpmMlsGpuBufferLayout(
				{
					...state,
					particleCount: Math.floor(0x7fff_ffff / 9) + 1,
					particleCapacity: Math.floor(0x7fff_ffff / 9) + 1
				},
				LIMITS
			)
		).toThrow(/linked-list index range/);
	});

	it('packs the reference parameters and rejects invalid step inputs', () => {
		const config = makeConfig({
			grid: { cellsX: 2, cellsY: 3, cellSize: 0.5 },
			particleMass: 0.2,
			particleVolume: 0.01,
			material: { lambda: 20, mu: 10 }
		});
		const state = validateMpmMlsGpuInput(
			[makeParticle({ position: { x: 0.5, y: 0.75 } })],
			config
		);
		const layout = calculateMpmMlsGpuBufferLayout(state, LIMITS);
		const parameters = packMpmMlsGpuStepParameters(state, layout, 0.001, {
			x: 1,
			y: -2
		});

		expect(Array.from(new Uint32Array(parameters, 0, 8))).toEqual([
			2, 3, 5, 6,
			1, 30, 9, 0
		]);
		const floats = new Float32Array(parameters, 32, 12);
		expect(Array.from(floats.slice(0, 4))).toEqual([
			0.5,
			Math.fround(0.2),
			Math.fround(0.01),
			Math.fround(0.001)
		]);
		expect(Array.from(floats.slice(4, 8))).toEqual([20, 10, Math.fround(-0.00016), 16]);
		expect(Array.from(floats.slice(8))).toEqual([1, -2, 1, 1.5]);

		expect(() => packMpmMlsGpuStepParameters(state, layout, 0, { x: 0, y: 0 }))
			.toThrow(/dt/);
		expect(() =>
			packMpmMlsGpuStepParameters(state, layout, 0.001, { x: 0, y: Number.NaN })
		).toThrow(/gravity.y/);
	});

	it('packs the live particle count and stencil count without using reserved capacity', () => {
		const state = validateMpmMlsGpuInput(
			[makeParticle(), makeParticle({ position: { x: 9, y: 4 } })],
			makeConfig({ capacity: 8 })
		);
		const layout = calculateMpmMlsGpuBufferLayout(state, LIMITS);
		const parameters = packMpmMlsGpuStepParameters(
			state,
			layout,
			0.001,
			{ x: 0, y: 0 },
			3
		);

		expect(Array.from(new Uint32Array(parameters, 16, 4))).toEqual([3, 209, 27, 0]);
		expect(() =>
			packMpmMlsGpuStepParameters(state, layout, 0.001, { x: 0, y: 0 }, 9)
		).toThrow(/capacity/);
	});
});
