import { describe, expect, it } from 'vitest';
import {
	AdaptiveResolutionController,
	chooseInitialGridSize,
	type GridSize
} from './resolution';

describe('adaptive resolution', () => {
	it('chooses an initial grid from viewport shape and available cores', () => {
		const phone = chooseInitialGridSize(390, 844, 8);
		const desktop = chooseInitialGridSize(1440, 900, 16);

		expect(phone.width).toBeLessThan(phone.height);
		expect(phone.width * phone.height).toBeGreaterThanOrEqual(12_000);
		expect(desktop.width).toBeGreaterThan(desktop.height);
		expect(desktop.width * desktop.height).toBeGreaterThan(phone.width * phone.height);
	});

	it('matches the measured viewport aspect when choosing a grid', () => {
		const viewport = { width: 844, height: 390 };
		const grid = chooseInitialGridSize(viewport.width, viewport.height, 4);

		expect(
			Math.abs(grid.width / grid.height - viewport.width / viewport.height) /
				(viewport.width / viewport.height)
		).toBeLessThan(0.01);
	});

	it('reduces the grid when measured frame work exceeds the 30 fps budget', () => {
		const controller = new AdaptiveResolutionController(30);
		const current = { width: 160, height: 200 };
		let next: GridSize | null = null;
		for (let sample = 0; sample < 30; sample++) {
			next = controller.observe(30, current);
		}

		expect(next).not.toBeNull();
		expect((next?.width ?? 0) * (next?.height ?? 0)).toBeLessThan(current.width * current.height);
	});

	it('increases resolution only after sustained headroom', () => {
		const controller = new AdaptiveResolutionController(30);
		const current = { width: 100, height: 100 };
		let next: GridSize | null = null;
		for (let sample = 0; sample < 179; sample++) {
			next = controller.observe(2, current);
		}
		expect(next).toBeNull();
		next = controller.observe(2, current);

		expect(next).not.toBeNull();
		expect((next?.width ?? 0) * (next?.height ?? 0)).toBeGreaterThan(current.width * current.height);
	});
});
