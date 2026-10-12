import { describe, expect, it } from 'vitest';
import { MPM_RENDER_FPS_OPTIONS, MpmFramePacer, type MpmRenderFps } from './frame-pacer.js';

function countRenderedFrames(displayFps: number, targetFps: MpmRenderFps): number {
	const pacer = new MpmFramePacer();
	const refreshCount = displayFps * 10;
	let renderedFrames = 0;

	for (let frame = 0; frame < refreshCount; frame++) {
		if (pacer.shouldRender((frame * 1_000) / displayFps, targetFps)) {
			renderedFrames += 1;
		}
	}

	return renderedFrames;
}

describe('MPM render frame pacing', () => {
	it('offers the requested render-rate choices', () => {
		expect(MPM_RENDER_FPS_OPTIONS).toEqual([15, 30, 60, 90, 120]);
	});

	it.each([
		[60, 15, 150],
		[60, 30, 300],
		[60, 60, 600],
		[90, 60, 600],
		[90, 90, 900],
		[120, 90, 900],
		[120, 120, 1_200],
		[60, 120, 600]
	] as const)(
		'limits a %i Hz display to %i fps over ten seconds',
		(displayFps, targetFps, expectedFrames) => {
			expect(countRenderedFrames(displayFps, targetFps)).toBe(expectedFrames);
		}
	);

	it('applies a changed target on the next animation frame', () => {
		const pacer = new MpmFramePacer();

		expect(pacer.shouldRender(0, 15)).toBe(true);
		expect(pacer.shouldRender(16.67, 120)).toBe(true);
		expect(pacer.shouldRender(20, 120)).toBe(false);
	});

	it('resets its deadline when rendering resumes', () => {
		const pacer = new MpmFramePacer();

		expect(pacer.shouldRender(0, 60)).toBe(true);
		pacer.reset();
		expect(pacer.shouldRender(10_000, 60)).toBe(true);
	});

	it('rejects invalid animation timestamps', () => {
		const pacer = new MpmFramePacer();

		expect(() => pacer.shouldRender(Number.NaN, 60)).toThrow(RangeError);
		expect(() => pacer.shouldRender(-1, 60)).toThrow(RangeError);
	});
});
