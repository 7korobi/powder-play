import {
	LIQUIDFUN_BOX2D_UNITS_PER_METER,
	type LiquidFunVector
} from './workload.js';

export const LIQUIDFUN_WALL_THICKNESS_METERS = 5;
export const LIQUIDFUN_WALL_THICKNESS_BOX2D_UNITS =
	LIQUIDFUN_WALL_THICKNESS_METERS * LIQUIDFUN_BOX2D_UNITS_PER_METER;

export function createLiquidFunWallDefinitions(width: number, height: number) {
	if (!Number.isFinite(width) || width <= 0 || !Number.isFinite(height) || height <= 0) {
		throw new RangeError('LiquidFun wall dimensions must be finite and positive.');
	}

	const halfThickness = LIQUIDFUN_WALL_THICKNESS_BOX2D_UNITS / 2;
	const sideHalfHeight = height / 2 + LIQUIDFUN_WALL_THICKNESS_BOX2D_UNITS;
	const horizontalHalfWidth = width / 2 + LIQUIDFUN_WALL_THICKNESS_BOX2D_UNITS;

	return [
		{
			position: { x: -halfThickness, y: height / 2 } satisfies LiquidFunVector,
			halfWidth: halfThickness,
			halfHeight: sideHalfHeight
		},
		{
			position: { x: width + halfThickness, y: height / 2 } satisfies LiquidFunVector,
			halfWidth: halfThickness,
			halfHeight: sideHalfHeight
		},
		{
			position: { x: width / 2, y: -halfThickness } satisfies LiquidFunVector,
			halfWidth: horizontalHalfWidth,
			halfHeight: halfThickness
		},
		{
			position: { x: width / 2, y: height + halfThickness } satisfies LiquidFunVector,
			halfWidth: horizontalHalfWidth,
			halfHeight: halfThickness
		}
	] as const;
}
