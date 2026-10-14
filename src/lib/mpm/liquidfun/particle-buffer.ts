import type { LiquidFunVector } from './workload.js';

export function readLiquidFunParticlePosition(
	heap: Float32Array,
	bufferPointer: number,
	index: number
): LiquidFunVector {
	if (
		!Number.isSafeInteger(bufferPointer) ||
		bufferPointer <= 0 ||
		bufferPointer % Float32Array.BYTES_PER_ELEMENT !== 0 ||
		!Number.isSafeInteger(index) ||
		index < 0
	) {
		throw new RangeError('LiquidFun returned an invalid particle buffer address.');
	}

	const offset =
		bufferPointer / Float32Array.BYTES_PER_ELEMENT + index * 2;
	if (
		!Number.isSafeInteger(offset) ||
		offset + 1 >= heap.length ||
		!Number.isFinite(heap[offset]) ||
		!Number.isFinite(heap[offset + 1])
	) {
		throw new RangeError(`LiquidFun particle position ${index} is unavailable.`);
	}

	return { x: heap[offset], y: heap[offset + 1] };
}
