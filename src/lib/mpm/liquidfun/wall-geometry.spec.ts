import { describe, expect, it } from 'vitest';
import {
	LIQUIDFUN_BOX2D_DOMAIN_UNITS,
	LIQUIDFUN_PARTICLE_RADIUS_BOX2D_UNITS
} from './workload.js';
import {
	createLiquidFunWallDefinitions,
	LIQUIDFUN_WALL_THICKNESS_BOX2D_UNITS,
	LIQUIDFUN_WALL_THICKNESS_METERS
} from './wall-geometry.js';

describe('LiquidFun wall geometry', () => {
	it('keeps inner faces at the domain edges and overlaps corners by two particle radii', () => {
		const { width, height } = LIQUIDFUN_BOX2D_DOMAIN_UNITS;
		const walls = createLiquidFunWallDefinitions(width, height);
		const [left, right, bottom, top] = walls;
		const cornerOverlap = LIQUIDFUN_PARTICLE_RADIUS_BOX2D_UNITS * 2;

		expect(LIQUIDFUN_WALL_THICKNESS_METERS).toBe(0.05);
		expect(LIQUIDFUN_WALL_THICKNESS_BOX2D_UNITS).toBe(50);
		expect(cornerOverlap).toBe(2);
		expect([
			{
				minX: left.position.x - left.halfWidth,
				maxX: left.position.x + left.halfWidth,
				minY: left.position.y - left.halfHeight,
				maxY: left.position.y + left.halfHeight
			},
			{
				minX: right.position.x - right.halfWidth,
				maxX: right.position.x + right.halfWidth,
				minY: right.position.y - right.halfHeight,
				maxY: right.position.y + right.halfHeight
			},
			{
				minX: bottom.position.x - bottom.halfWidth,
				maxX: bottom.position.x + bottom.halfWidth,
				minY: bottom.position.y - bottom.halfHeight,
				maxY: bottom.position.y + bottom.halfHeight
			},
			{
				minX: top.position.x - top.halfWidth,
				maxX: top.position.x + top.halfWidth,
				minY: top.position.y - top.halfHeight,
				maxY: top.position.y + top.halfHeight
			}
		]).toEqual([
			{
				minX: -LIQUIDFUN_WALL_THICKNESS_BOX2D_UNITS,
				maxX: 0,
				minY: -cornerOverlap,
				maxY: height + cornerOverlap
			},
			{
				minX: width,
				maxX: width + LIQUIDFUN_WALL_THICKNESS_BOX2D_UNITS,
				minY: -cornerOverlap,
				maxY: height + cornerOverlap
			},
			{
				minX: -cornerOverlap,
				maxX: width + cornerOverlap,
				minY: -LIQUIDFUN_WALL_THICKNESS_BOX2D_UNITS,
				maxY: 0
			},
			{
				minX: -cornerOverlap,
				maxX: width + cornerOverlap,
				minY: height,
				maxY: height + LIQUIDFUN_WALL_THICKNESS_BOX2D_UNITS
			}
		]);
	});

	it('rejects invalid domain dimensions', () => {
		expect(() => createLiquidFunWallDefinitions(0, 15)).toThrow(RangeError);
		expect(() => createLiquidFunWallDefinitions(28, Number.POSITIVE_INFINITY)).toThrow(
			RangeError
		);
	});
});
