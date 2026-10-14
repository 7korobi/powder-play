import { describe, expect, it } from 'vitest';
import {
	createLiquidFunGate4Workload,
	LIQUIDFUN_BOX2D_DOMAIN_METERS,
	LIQUIDFUN_BOX2D_GRAVITY,
	LIQUIDFUN_GATE4_DOMAIN,
	LIQUIDFUN_GATE4_GRAVITY,
	LIQUIDFUN_GATE4_UNITS_PER_METER,
	LIQUIDFUN_SCENE_SEED,
	mapBox2DPositionToGate4,
	mapBox2DVelocityToGate4,
	mapGate4PositionToBox2D,
	mapGate4VelocityToBox2D,
	mapGate4PebbleTriangle,
	summarizeSamples
} from './workload.js';
import {
	createMpmGate4Scene,
	GATE4_FRACTURE_PARTICLE_COUNT,
	GATE4_PARTICLE_COUNT,
	GATE4_PEBBLE_COUNT,
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

	it('uses the fixed Gate 4 seed and keeps source counts while reporting the fracture omission', () => {
		const workload = createLiquidFunGate4Workload();
		expect(workload.seed).toBe(0x4d504d34);
		expect(workload.seed).toBe(LIQUIDFUN_SCENE_SEED);
		expect(workload.sourceMpmNodeCount).toBe(GATE4_PARTICLE_COUNT);
		expect(workload.sourceMpmNodeCount).toBe(3_674);
		expect(workload.waterParticles).toHaveLength(GATE4_WATER_PARTICLE_COUNT);
		expect(workload.waterParticles).toHaveLength(2_000);
		expect(workload.oilParticles).toHaveLength(GATE4_OIL_PARTICLE_COUNT);
		expect(workload.oilParticles).toHaveLength(1_500);
		expect(workload.triangleBodies).toHaveLength(GATE4_PEBBLE_COUNT);
		expect(workload.triangleBodies).toHaveLength(50);
		expect(workload.omittedFractureNodeCount).toBe(GATE4_FRACTURE_PARTICLE_COUNT);
		expect(workload.omittedFractureNodeCount).toBe(24);
	});

	it('fits triangle linear/angular velocity from the original three pebble nodes', () => {
		const scene = createMpmGate4Scene(LIQUIDFUN_SCENE_SEED);
		const firstPebbleNodes = scene.particles.filter(
			(particle) => particle.objectId === 3
		);
		const triangle = mapGate4PebbleTriangle(firstPebbleNodes);
		const expectedVelocity = firstPebbleNodes.reduce(
			(velocity, node) => ({
				x: velocity.x + node.velocity.x / firstPebbleNodes.length /
					LIQUIDFUN_GATE4_UNITS_PER_METER,
				y: velocity.y - node.velocity.y / firstPebbleNodes.length /
					LIQUIDFUN_GATE4_UNITS_PER_METER
			}),
			{ x: 0, y: 0 }
		);

		expect(triangle.objectId).toBe(3);
		expect(triangle.localVertices).toHaveLength(3);
		expect(triangle.position.x).toBeGreaterThan(0);
		expect(triangle.position.y).toBeGreaterThan(0);
		expect(triangle.linearVelocity.x).toBeCloseTo(expectedVelocity.x);
		expect(triangle.linearVelocity.y).toBeCloseTo(expectedVelocity.y);
		expect(Number.isFinite(triangle.angularVelocity)).toBe(true);
		expect(triangle.areaSquareMeters).toBeGreaterThan(0);
		expect(triangle.density).toBeGreaterThan(0);
	});

	it('preserves the rigid-body angular velocity sign under the y-down to y-up mapping', () => {
		const screenPositions = [
			{ x: 100, y: 100 },
			{ x: 200, y: 100 },
			{ x: 150, y: 200 }
		] as const;
		const center = screenPositions
			.map(mapGate4PositionToBox2D)
			.reduce(
				(sum, position) => ({
					x: sum.x + position.x / screenPositions.length,
					y: sum.y + position.y / screenPositions.length
				}),
				{ x: 0, y: 0 }
			);
		const omega = 2;
		const translation = { x: 0.25, y: -0.3 };
		const vertexIds = [0, 1, 2] as const;
		const nodes = screenPositions.map((position, index) => {
			const box2dPosition = mapGate4PositionToBox2D(position);
			const relativePosition = {
				x: box2dPosition.x - center.x,
				y: box2dPosition.y - center.y
			};
			const box2dVelocity = {
				x: translation.x - omega * relativePosition.y,
				y: translation.y + omega * relativePosition.x
			};
			return {
				position,
				velocity: mapBox2DVelocityToGate4(box2dVelocity),
				C: { m00: 0, m01: 0, m10: 0, m11: 0 },
				F: { m00: 1, m01: 0, m10: 0, m11: 1 },
				objectId: 99,
				bondedTriangleVertex: vertexIds[index],
				material: { model: 'neo-hookean' as const, lambda: 20, mu: 10 }
			};
		});

		const triangle = mapGate4PebbleTriangle(nodes);
		expect(triangle.linearVelocity.x).toBeCloseTo(translation.x);
		expect(triangle.linearVelocity.y).toBeCloseTo(translation.y);
		expect(triangle.angularVelocity).toBeCloseTo(omega);
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
