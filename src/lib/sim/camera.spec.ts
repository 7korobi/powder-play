import { describe, expect, it } from 'vitest';
import {
	BLEED_ZOOM,
	cameraScale,
	createCamera,
	screenToCell,
	screenToWorld,
	updateCameraForPinch
} from './camera';

describe('camera', () => {
	it('shows the full world in fit mode and reveals a 20% bleed in bleed mode', () => {
		const fit = createCamera(100, 200, 'fit');
		const bleed = createCamera(100, 200, 'bleed');
		const viewport = { width: 100, height: 200 };

		expect(cameraScale(fit, viewport, 100, 200)).toBe(1);
		expect(bleed.zoom).toBe(BLEED_ZOOM);
		expect(cameraScale(bleed, viewport, 100, 200)).toBeCloseTo(1.2);
	});

	it('maps screen positions to bounded world cells', () => {
		const camera = createCamera(100, 200, 'fit');
		const viewport = { width: 100, height: 200 };

		expect(screenToCell({ x: 30, y: 50 }, camera, viewport, 100, 200)).toEqual({
			x: 30,
			y: 50
		});
		expect(screenToCell({ x: -1, y: 50 }, camera, viewport, 100, 200)).toBeNull();
	});

	it('keeps the world point beneath the moving pinch centroid anchored', () => {
		const viewport = { width: 100, height: 200 };
		const camera = createCamera(100, 200, 'fit');
		const baseCentroid = { x: 40, y: 80 };
		const currentCentroid = { x: 50, y: 90 };
		const anchor = screenToWorld(baseCentroid, camera, viewport, 100, 200);

		const next = updateCameraForPinch(
			camera,
			20,
			40,
			baseCentroid,
			currentCentroid,
			viewport,
			100,
			200
		);
		const after = screenToWorld(currentCentroid, next, viewport, 100, 200);

		expect(next.zoom).toBe(2);
		expect(after.x).toBeCloseTo(anchor.x);
		expect(after.y).toBeCloseTo(anchor.y);
	});
});
