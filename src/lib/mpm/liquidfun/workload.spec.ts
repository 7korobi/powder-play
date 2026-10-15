import { describe, expect, it } from 'vitest';
import {
	createLiquidFunGate4Workload,
	LIQUIDFUN_BOX2D_DOMAIN_METERS,
	LIQUIDFUN_BOX2D_GRAVITY,
	LIQUIDFUN_GATE4_DOMAIN,
	LIQUIDFUN_GATE4_GRAVITY,
	LIQUIDFUN_GATE4_UNITS_PER_METER,
	LIQUIDFUN_PEBBLE_SIZE_SCALE,
	LIQUIDFUN_SCENE_SEED,
	mapBox2DPositionToGate4,
	mapBox2DVelocityToGate4,
	mapGate4PositionToBox2D,
	mapGate4VelocityToBox2D,
	scaleLiquidFunPebbleLocalVertices,
	summarizeSamples
} from './workload.js';
import {
	createMpmGate4Scene,
	GATE4_FRACTURE_BOND_COUNT,
	GATE4_FRACTURE_FAILURE_STRAIN,
	GATE4_FRACTURE_OBJECT_ID,
	GATE4_FRACTURE_PARTICLE_COUNT,
	GATE4_PARTICLE_COUNT,
	GATE4_PEBBLE_COUNT,
	GATE4_PEBBLE_PARTICLE_COUNT,
	GATE4_OIL_PARTICLE_COUNT,
	GATE4_WATER_PARTICLE_COUNT
} from '../scenes/gate4.js';

describe('LiquidFun Gate 4 workload mapping', () => {
	it('converts Gate 4 screen/world coordinates and velocity to Box2D y-up units', () => {
		expect(LIQUIDFUN_GATE4_UNITS_PER_METER).toBe(100);
		expect(LIQUIDFUN_GATE4_DOMAIN).toEqual({ width: 2_200, height: 1_500 });
		expect(LIQUIDFUN_BOX2D_DOMAIN_METERS).toEqual({ width: 22, height: 15 });
		expect(LIQUIDFUN_GATE4_GRAVITY).toEqual({ x: 0, y: 90 });
		expect(LIQUIDFUN_BOX2D_GRAVITY).toEqual({ x: 0, y: -0.9 });

		const screenPosition = { x: 100, y: 640 };
		const screenVelocity = { x: 32, y: -64 };
		const box2dPosition = mapGate4PositionToBox2D(screenPosition);
		const box2dVelocity = mapGate4VelocityToBox2D(screenVelocity);
		expect(box2dPosition).toEqual({ x: 1, y: 8.6 });
		expect(box2dVelocity).toEqual({ x: 0.32, y: 0.64 });
		expect(mapBox2DPositionToGate4(box2dPosition)).toEqual(screenPosition);
		expect(mapBox2DVelocityToGate4(box2dVelocity)).toEqual(screenVelocity);
	});

	it('maps every seeded Gate 4 node and preserves the source topology', () => {
		const workload = createLiquidFunGate4Workload();
		const scene = createMpmGate4Scene(LIQUIDFUN_SCENE_SEED);
		const waterParticles = scene.particles.filter(
			(particle) => particle.material?.model === 'water'
		);
		const oilParticles = scene.particles.filter(
			(particle) => particle.material?.model === 'oil'
		);
		const pebbleNodes = scene.particles.filter(
			(particle) => particle.bondedTriangleVertex !== undefined
		);
		const fractureNodes = scene.particles.filter(
			(particle) => particle.objectId === GATE4_FRACTURE_OBJECT_ID
		);

		expect(workload.seed).toBe(0x4d504d34);
		expect(workload.seed).toBe(LIQUIDFUN_SCENE_SEED);
		expect(workload.sourceMpmNodeCount).toBe(GATE4_PARTICLE_COUNT);
		expect(workload.sourceMpmNodeCount).toBe(3_674);
		expect(workload.mappedMpmNodeCount).toBe(GATE4_PARTICLE_COUNT);
		expect(workload.waterParticles).toHaveLength(GATE4_WATER_PARTICLE_COUNT);
		expect(workload.waterParticles).toHaveLength(2_000);
		expect(workload.waterParticles).toEqual(waterParticles);
		expect(workload.oilParticles).toHaveLength(GATE4_OIL_PARTICLE_COUNT);
		expect(workload.oilParticles).toHaveLength(1_500);
		expect(workload.oilParticles).toEqual(oilParticles);
		expect(workload.pebbleNodes).toHaveLength(GATE4_PEBBLE_PARTICLE_COUNT);
		expect(workload.pebbleNodes).toEqual(pebbleNodes);
		expect(workload.pebbleGroups).toHaveLength(GATE4_PEBBLE_COUNT);
		expect(workload.pebbleGroups).toHaveLength(50);
		for (const group of workload.pebbleGroups) {
			const nodes = group.nodeIndices.map(
				(nodeIndex) => workload.pebbleNodes[nodeIndex]
			);
			expect(nodes.every((node) => node?.objectId === group.objectId)).toBe(true);
			expect(nodes.map((node) => node?.bondedTriangleVertex)).toEqual([0, 1, 2]);
		}
		expect(workload.fractureNodes).toHaveLength(GATE4_FRACTURE_PARTICLE_COUNT);
		expect(workload.fractureNodes).toHaveLength(24);
		expect(workload.fractureNodes).toEqual(fractureNodes);
		expect(workload.fractureBonds).toHaveLength(GATE4_FRACTURE_BOND_COUNT);
		expect(workload.fractureBonds).toHaveLength(53);
		expect(
			workload.fractureBonds.every(
				(bond) =>
					bond.failureStrain === GATE4_FRACTURE_FAILURE_STRAIN &&
					bond.firstNodeIndex >= 0 &&
					bond.firstNodeIndex < workload.fractureNodes.length &&
					bond.secondNodeIndex >= 0 &&
					bond.secondNodeIndex < workload.fractureNodes.length &&
					bond.restLengthMeters > 0
			)
		).toBe(true);
	});

	it('halves both dimensions of each LiquidFun pebble triangle', () => {
		const originalVertices = [
			{ x: -0.66, y: -0.66 / Math.sqrt(3) },
			{ x: 0.66, y: -0.66 / Math.sqrt(3) },
			{ x: 0, y: (2 * 0.66) / Math.sqrt(3) }
		] as const;
		const scaledVertices = scaleLiquidFunPebbleLocalVertices(originalVertices);
		const width = (vertices: readonly { x: number; y: number }[]) =>
			Math.max(...vertices.map((vertex) => vertex.x)) -
			Math.min(...vertices.map((vertex) => vertex.x));
		const height = (vertices: readonly { x: number; y: number }[]) =>
			Math.max(...vertices.map((vertex) => vertex.y)) -
			Math.min(...vertices.map((vertex) => vertex.y));

		expect(LIQUIDFUN_PEBBLE_SIZE_SCALE).toBe(0.5);
		expect(width(scaledVertices)).toBeCloseTo(width(originalVertices) / 2);
		expect(height(scaledVertices)).toBeCloseTo(height(originalVertices) / 2);
		expect(() => scaleLiquidFunPebbleLocalVertices(originalVertices.slice(0, 2))).toThrow(
			RangeError
		);
	});

	it('summarizes repeatable batches with the MPM benchmark quantile convention', () => {
		expect(summarizeSamples(Array.from({ length: 12 }, (_, index) => index + 1))).toEqual({
			median: 6,
			p95: 11,
			min: 1,
			max: 12,
			samples: 12
		});
		expect(() => summarizeSamples([])).toThrow(RangeError);
		expect(() => summarizeSamples([1, Number.NaN])).toThrow(RangeError);
	});
});
