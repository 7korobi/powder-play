import type { Point, Vector } from './world';

export type CameraMode = 'fit' | 'bleed';

export interface Camera {
	centerX: number;
	centerY: number;
	zoom: number;
}

export interface ViewportSize {
	width: number;
	height: number;
}

export const MIN_ZOOM = 1;
export const MAX_ZOOM = 4;
export const BLEED_ZOOM = 1.2;

export function createCamera(worldWidth: number, worldHeight: number, mode: CameraMode): Camera {
	return {
		centerX: worldWidth / 2,
		centerY: worldHeight / 2,
		zoom: mode === 'bleed' ? BLEED_ZOOM : MIN_ZOOM
	};
}

export function cameraScale(
	camera: Camera,
	viewport: ViewportSize,
	worldWidth: number,
	worldHeight: number
): number {
	return Math.min(viewport.width / worldWidth, viewport.height / worldHeight) * camera.zoom;
}

export function screenToWorld(
	point: Point,
	camera: Camera,
	viewport: ViewportSize,
	worldWidth: number,
	worldHeight: number
): Point {
	const scale = cameraScale(camera, viewport, worldWidth, worldHeight);
	return {
		x: camera.centerX + (point.x - viewport.width / 2) / scale,
		y: camera.centerY + (point.y - viewport.height / 2) / scale
	};
}

export function screenToCell(
	point: Point,
	camera: Camera,
	viewport: ViewportSize,
	worldWidth: number,
	worldHeight: number
): Point | null {
	const worldPoint = screenToWorld(point, camera, viewport, worldWidth, worldHeight);
	const cell = { x: Math.floor(worldPoint.x), y: Math.floor(worldPoint.y) };
	return cell.x >= 0 && cell.y >= 0 && cell.x < worldWidth && cell.y < worldHeight ? cell : null;
}

export function worldToScreen(
	point: Point,
	camera: Camera,
	viewport: ViewportSize,
	worldWidth: number,
	worldHeight: number
): Point {
	const scale = cameraScale(camera, viewport, worldWidth, worldHeight);
	return {
		x: viewport.width / 2 + (point.x - camera.centerX) * scale,
		y: viewport.height / 2 + (point.y - camera.centerY) * scale
	};
}

export function clampCamera(
	camera: Camera,
	viewport: ViewportSize,
	worldWidth: number,
	worldHeight: number
): Camera {
	const scale = cameraScale(camera, viewport, worldWidth, worldHeight);
	const visibleWidth = Math.min(worldWidth, viewport.width / scale);
	const visibleHeight = Math.min(worldHeight, viewport.height / scale);
	return {
		...camera,
		centerX: clamp(camera.centerX, visibleWidth / 2, worldWidth - visibleWidth / 2),
		centerY: clamp(camera.centerY, visibleHeight / 2, worldHeight - visibleHeight / 2)
	};
}

export function updateCameraForPinch(
	baseCamera: Camera,
	baseDistance: number,
	currentDistance: number,
	baseCentroid: Point,
	currentCentroid: Point,
	viewport: ViewportSize,
	worldWidth: number,
	worldHeight: number
): Camera {
	const baseScale = cameraScale(baseCamera, viewport, worldWidth, worldHeight);
	const anchoredWorldPoint = screenToWorld(
		baseCentroid,
		baseCamera,
		viewport,
		worldWidth,
		worldHeight
	);
	const zoom = clamp(
		baseCamera.zoom * (currentDistance / Math.max(1, baseDistance)),
		MIN_ZOOM,
		MAX_ZOOM
	);
	const nextScale =
		Math.min(viewport.width / worldWidth, viewport.height / worldHeight) * zoom;
	const next: Camera = {
		centerX: anchoredWorldPoint.x - (currentCentroid.x - viewport.width / 2) / nextScale,
		centerY: anchoredWorldPoint.y - (currentCentroid.y - viewport.height / 2) / nextScale,
		zoom
	};
	return clampCamera(next, viewport, worldWidth, worldHeight);
}

export function panCamera(
	camera: Camera,
	delta: Vector,
	viewport: ViewportSize,
	worldWidth: number,
	worldHeight: number
): Camera {
	const scale = cameraScale(camera, viewport, worldWidth, worldHeight);
	return clampCamera(
		{
			...camera,
			centerX: camera.centerX - delta.x / scale,
			centerY: camera.centerY - delta.y / scale
		},
		viewport,
		worldWidth,
		worldHeight
	);
}

export function setCameraMode(
	worldWidth: number,
	worldHeight: number,
	mode: CameraMode
): Camera {
	return createCamera(worldWidth, worldHeight, mode);
}

function clamp(value: number, minimum: number, maximum: number): number {
	return Math.min(maximum, Math.max(minimum, value));
}
