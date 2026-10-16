import type { LiquidFunVector } from './workload.js';

export const LIQUIDFUN_WALL_THICKNESS_METERS = 1_000;

export function createLiquidFunWallDefinitions(width: number, height: number) {
	if (!Number.isFinite(width) || width <= 0 || !Number.isFinite(height) || height <= 0) {
		throw new RangeError('LiquidFun wall dimensions must be finite and positive.');
	}

	const halfThickness = LIQUIDFUN_WALL_THICKNESS_METERS / 2;
	const sideHalfHeight = height / 2 + LIQUIDFUN_WALL_THICKNESS_METERS;
	const horizontalHalfWidth = width / 2 + LIQUIDFUN_WALL_THICKNESS_METERS;

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
