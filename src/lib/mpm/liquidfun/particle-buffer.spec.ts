import { describe, expect, it } from 'vitest';
import { readLiquidFunParticlePosition } from './particle-buffer.js';

describe('LiquidFun particle position buffer', () => {
	it('reads each packed b2Vec2 from its byte-addressed WASM buffer', () => {
		const heap = new Float32Array(32);
		const bufferPointer = 8 * Float32Array.BYTES_PER_ELEMENT;
		heap.set([1.25, 4.5, 5.5, 8.75], bufferPointer / Float32Array.BYTES_PER_ELEMENT);

		expect(readLiquidFunParticlePosition(heap, bufferPointer, 0)).toEqual({
			x: 1.25,
			y: 4.5
		});
		expect(readLiquidFunParticlePosition(heap, bufferPointer, 1)).toEqual({
			x: 5.5,
			y: 8.75
		});
	});

	it('rejects invalid addresses and unavailable positions', () => {
		const heap = new Float32Array(4);

		expect(() => readLiquidFunParticlePosition(heap, 0, 0)).toThrow(RangeError);
		expect(() => readLiquidFunParticlePosition(heap, 2, 0)).toThrow(RangeError);
		expect(() => readLiquidFunParticlePosition(heap, 4, -1)).toThrow(RangeError);
		expect(() => readLiquidFunParticlePosition(heap, 12, 0)).toThrow(RangeError);
	});

	it('rejects non-finite coordinates rather than silently drawing them', () => {
		const heap = new Float32Array([1, Number.NaN]);

		expect(() => readLiquidFunParticlePosition(heap, 4, 0)).toThrow(RangeError);
	});
});
