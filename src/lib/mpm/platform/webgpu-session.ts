import type { MpmWebGpuSession } from '../contracts.js';

const WEBGPU_UNAVAILABLE_REASON =
	'WebGPU is unavailable because navigator.gpu is not present in this environment.';

function getGpu(): GPU | undefined {
	if (typeof navigator === 'undefined') return undefined;
	return navigator.gpu ?? undefined;
}

function disposeResources(context: GPUCanvasContext, device: GPUDevice): void {
	const errors: unknown[] = [];

	try {
		context.unconfigure();
	} catch (error) {
		errors.push(error);
	}

	try {
		device.destroy();
	} catch (error) {
		errors.push(error);
	}

	if (errors.length === 1) throw errors[0];
	if (errors.length > 1) {
		throw new AggregateError(errors, 'Failed to release the WebGPU canvas context and device.');
	}
}

function throwAfterCleanup(error: Error, cleanup: () => void): never {
	try {
		cleanup();
	} catch (cleanupError) {
		throw new AggregateError(
			[error, cleanupError],
			'WebGPU session setup failed and cleanup also failed.'
		);
	}

	throw error;
}

export function getMpmWebGpuAvailability(): { supported: boolean; reason: string } {
	if (getGpu()) return { supported: true, reason: '' };
	return { supported: false, reason: WEBGPU_UNAVAILABLE_REASON };
}

export async function createMpmWebGpuSession(
	canvas: HTMLCanvasElement
): Promise<MpmWebGpuSession> {
	const gpu = getGpu();
	if (!gpu) throw new Error(WEBGPU_UNAVAILABLE_REASON);

	let format: GPUTextureFormat;
	try {
		format = gpu.getPreferredCanvasFormat();
	} catch (cause) {
		throw new Error('Failed to determine a preferred WebGPU canvas format.', { cause });
	}

	let adapter: GPUAdapter | null;
	try {
		adapter = await gpu.requestAdapter();
	} catch (cause) {
		throw new Error('Failed to request a WebGPU adapter.', { cause });
	}
	if (!adapter) throw new Error('No WebGPU adapter is available.');

	let device: GPUDevice;
	try {
		device = await adapter.requestDevice();
	} catch (cause) {
		throw new Error('Failed to request a WebGPU device.', { cause });
	}

	let context: GPUCanvasContext | null;
	try {
		// The DOM lib's generic getContext overload does not include the WebGPU context ID.
		context = canvas.getContext('webgpu') as unknown as GPUCanvasContext | null;
	} catch (cause) {
		throwAfterCleanup(
			new Error('Failed to acquire a WebGPU canvas context.', { cause }),
			() => device.destroy()
		);
	}
	if (!context) {
		throwAfterCleanup(
			new Error('The canvas does not provide a WebGPU context.'),
			() => device.destroy()
		);
	}

	try {
		context.configure({ device, format, alphaMode: 'opaque' });
	} catch (cause) {
		throwAfterCleanup(
			new Error('Failed to configure the WebGPU canvas context.', { cause }),
			() => disposeResources(context, device)
		);
	}

	let disposed = false;
	return {
		adapter,
		device,
		context,
		format,
		lost: device.lost,
		dispose() {
			if (disposed) return;
			disposed = true;
			disposeResources(context, device);
		}
	};
}
