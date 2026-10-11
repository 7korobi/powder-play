import { afterEach, describe, expect, it, vi } from 'vitest';

afterEach(() => {
	vi.unstubAllGlobals();
});

function createCanvas(context: GPUCanvasContext | null) {
	const getContext = vi.fn(() => context);
	return {
		canvas: { getContext } as unknown as HTMLCanvasElement,
		getContext
	};
}

function createGpu(adapter: GPUAdapter | null) {
	const requestAdapter = vi.fn(async () => adapter);
	const getPreferredCanvasFormat = vi.fn(() => 'bgra8unorm' as GPUTextureFormat);
	return {
		gpu: { requestAdapter, getPreferredCanvasFormat } as unknown as GPU,
		requestAdapter,
		getPreferredCanvasFormat
	};
}

function createDevice() {
	const lost = Promise.resolve({} as GPUDeviceLostInfo);
	const destroy = vi.fn();
	return {
		device: { lost, destroy } as unknown as GPUDevice,
		lost,
		destroy
	};
}

function createContext() {
	const configure = vi.fn();
	const unconfigure = vi.fn();
	return {
		context: { configure, unconfigure } as unknown as GPUCanvasContext,
		configure,
		unconfigure
	};
}

describe('WebGPU session', () => {
	it('can be imported and reports unavailable without browser globals', async () => {
		vi.stubGlobal('navigator', undefined);
		vi.stubGlobal('window', undefined);
		vi.resetModules();

		const platform = await import('./webgpu-session.js');

		expect(platform.getMpmWebGpuAvailability()).toEqual({
			supported: false,
			reason: expect.stringContaining('navigator.gpu')
		});
	});

	it('reports WebGPU support when navigator.gpu is present', async () => {
		const { gpu, requestAdapter } = createGpu(null);
		vi.stubGlobal('navigator', { gpu });
		const { getMpmWebGpuAvailability } = await import('./webgpu-session.js');

		expect(getMpmWebGpuAvailability()).toEqual({ supported: true, reason: '' });
		expect(requestAdapter).not.toHaveBeenCalled();
	});

	it('fails explicitly when navigator.gpu is missing', async () => {
		vi.stubGlobal('navigator', {});
		const { createMpmWebGpuSession } = await import('./webgpu-session.js');
		const { canvas, getContext } = createCanvas(null);

		await expect(createMpmWebGpuSession(canvas)).rejects.toThrow('navigator.gpu');
		expect(getContext).not.toHaveBeenCalled();
	});

	it('fails explicitly when no adapter is available', async () => {
		const { gpu, requestAdapter } = createGpu(null);
		vi.stubGlobal('navigator', { gpu });
		const { createMpmWebGpuSession } = await import('./webgpu-session.js');
		const { canvas, getContext } = createCanvas(null);

		await expect(createMpmWebGpuSession(canvas)).rejects.toThrow(
			'No WebGPU adapter is available.'
		);
		expect(requestAdapter).toHaveBeenCalledOnce();
		expect(getContext).not.toHaveBeenCalled();
	});

	it('wraps device request failures with an explicit message and cause', async () => {
		const deviceFailure = new Error('request denied');
		const requestDevice = vi.fn(() => Promise.reject(deviceFailure));
		const adapter = { requestDevice } as unknown as GPUAdapter;
		const { gpu } = createGpu(adapter);
		vi.stubGlobal('navigator', { gpu });
		const { createMpmWebGpuSession } = await import('./webgpu-session.js');
		const { canvas, getContext } = createCanvas(null);

		await expect(createMpmWebGpuSession(canvas)).rejects.toMatchObject({
			message: 'Failed to request a WebGPU device.',
			cause: deviceFailure
		});
		expect(requestDevice).toHaveBeenCalledOnce();
		expect(getContext).not.toHaveBeenCalled();
	});

	it('configures the preferred format, exposes device.lost, and disposes once', async () => {
		const { device, lost, destroy } = createDevice();
		const { context, configure, unconfigure } = createContext();
		const { gpu, getPreferredCanvasFormat } = createGpu({
			requestDevice: vi.fn(async () => device)
		} as unknown as GPUAdapter);
		vi.stubGlobal('navigator', { gpu });
		const { createMpmWebGpuSession } = await import('./webgpu-session.js');
		const { canvas, getContext } = createCanvas(context);

		const session = await createMpmWebGpuSession(canvas);

		expect(session.device).toBe(device);
		expect(session.lost).toBe(lost);
		expect(session.format).toBe('bgra8unorm');
		expect(getPreferredCanvasFormat).toHaveBeenCalledOnce();
		expect(getContext).toHaveBeenCalledWith('webgpu');
		expect(configure).toHaveBeenCalledWith({
			device,
			format: 'bgra8unorm',
			alphaMode: 'opaque'
		});

		session.dispose();
		session.dispose();

		expect(unconfigure).toHaveBeenCalledOnce();
		expect(destroy).toHaveBeenCalledOnce();
	});

	it('destroys the device when the canvas has no WebGPU context', async () => {
		const { device, destroy } = createDevice();
		const { gpu } = createGpu({
			requestDevice: vi.fn(async () => device)
		} as unknown as GPUAdapter);
		vi.stubGlobal('navigator', { gpu });
		const { createMpmWebGpuSession } = await import('./webgpu-session.js');
		const { canvas } = createCanvas(null);

		await expect(createMpmWebGpuSession(canvas)).rejects.toThrow(
			'The canvas does not provide a WebGPU context.'
		);
		expect(destroy).toHaveBeenCalledOnce();
	});

	it('releases resources after configuration fails', async () => {
		const configureFailure = new Error('bad canvas configuration');
		const { device, destroy } = createDevice();
		const { context, configure, unconfigure } = createContext();
		configure.mockImplementation(() => {
			throw configureFailure;
		});
		const { gpu } = createGpu({
			requestDevice: vi.fn(async () => device)
		} as unknown as GPUAdapter);
		vi.stubGlobal('navigator', { gpu });
		const { createMpmWebGpuSession } = await import('./webgpu-session.js');
		const { canvas } = createCanvas(context);

		await expect(createMpmWebGpuSession(canvas)).rejects.toMatchObject({
			message: 'Failed to configure the WebGPU canvas context.',
			cause: configureFailure
		});
		expect(unconfigure).toHaveBeenCalledOnce();
		expect(destroy).toHaveBeenCalledOnce();
	});

	it('still destroys the device and propagates unconfigure errors on dispose', async () => {
		const unconfigureFailure = new Error('unconfigure failed');
		const { device, destroy } = createDevice();
		const { context, unconfigure } = createContext();
		unconfigure.mockImplementation(() => {
			throw unconfigureFailure;
		});
		const { gpu } = createGpu({
			requestDevice: vi.fn(async () => device)
		} as unknown as GPUAdapter);
		vi.stubGlobal('navigator', { gpu });
		const { createMpmWebGpuSession } = await import('./webgpu-session.js');
		const { canvas } = createCanvas(context);
		const session = await createMpmWebGpuSession(canvas);

		expect(session.dispose).toThrow(unconfigureFailure);
		session.dispose();

		expect(unconfigure).toHaveBeenCalledOnce();
		expect(destroy).toHaveBeenCalledOnce();
	});
});
