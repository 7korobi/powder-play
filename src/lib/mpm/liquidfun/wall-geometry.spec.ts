import { describe, expect, it } from 'vitest';
import { LIQUIDFUN_BOX2D_DOMAIN_UNITS } from './workload.js';
import {
	createLiquidFunWallDefinitions,
	LIQUIDFUN_WALL_THICKNESS_BOX2D_UNITS,
	LIQUIDFUN_WALL_THICKNESS_METERS
} from './wall-geometry.js';

describe('LiquidFun wall geometry', () => {
	it('keeps inner faces at the domain edges and extends five meters outward', () => {
		const { width, height } = LIQUIDFUN_BOX2D_DOMAIN_UNITS;
		const walls = createLiquidFunWallDefinitions(width, height);
		const [left, right, bottom, top] = walls;

		expect(LIQUIDFUN_WALL_THICKNESS_METERS).toBe(5);
		expect(LIQUIDFUN_WALL_THICKNESS_BOX2D_UNITS).toBe(5_000);
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
				minY: -LIQUIDFUN_WALL_THICKNESS_BOX2D_UNITS,
				maxY: height + LIQUIDFUN_WALL_THICKNESS_BOX2D_UNITS
			},
			{
				minX: width,
				maxX: width + LIQUIDFUN_WALL_THICKNESS_BOX2D_UNITS,
				minY: -LIQUIDFUN_WALL_THICKNESS_BOX2D_UNITS,
				maxY: height + LIQUIDFUN_WALL_THICKNESS_BOX2D_UNITS
			},
			{
				minX: -LIQUIDFUN_WALL_THICKNESS_BOX2D_UNITS,
				maxX: width + LIQUIDFUN_WALL_THICKNESS_BOX2D_UNITS,
				minY: -LIQUIDFUN_WALL_THICKNESS_BOX2D_UNITS,
				maxY: 0
			},
			{
				minX: -LIQUIDFUN_WALL_THICKNESS_BOX2D_UNITS,
				maxX: width + LIQUIDFUN_WALL_THICKNESS_BOX2D_UNITS,
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
