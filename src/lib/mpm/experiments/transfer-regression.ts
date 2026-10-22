import {
	createExperimentalMpmMlsAtomicGpuSolver,
	createMpmMlsGpuSolver,
	MpmMlsGpuNumericalError
} from '../kernels/solver.js';
import { stepMpm, type MpmMaterialParameters, type MpmParticle } from '../reference/solver.js';
import { createSiFluidScene } from '../scenes/si-fluid.js';

function compare(expected: readonly MpmParticle[], actual: readonly MpmParticle[]) {
	if (expected.length !== actual.length) throw new Error('Particle count changed during transfer.');
	let maximumError = 0;
	let maximumContext = '';
	for (let index = 0; index < expected.length; index += 1) {
		const values = (particle: MpmParticle) => [
			...Object.values(particle.position), ...Object.values(particle.velocity),
			...Object.values(particle.C), ...Object.values(particle.F), particle.plasticVolume ?? 1
		];
		const reference = values(expected[index]);
		const candidate = values(actual[index]);
		for (let field = 0; field < reference.length; field += 1) {
			const difference = Math.abs(reference[field] - candidate[field]);
			if (!Number.isFinite(difference)) throw new Error('Non-finite transfer regression state.');
			if (difference > maximumError) {
				maximumError = difference;
				maximumContext = `particle ${index}, field ${field}: ${reference[field]} vs ${candidate[field]}`;
			}
		}
	}
	if (maximumError > 1e-3) throw new Error(`Transfer parity exceeded 1e-3: ${maximumError}; ${maximumContext}.`);
	return maximumError;
}

export async function runMpmTransferRegression(baselineFactory = createMpmMlsGpuSolver) {
	const materials: MpmMaterialParameters[] = [
		{ model: 'neo-hookean', lambda: 20, mu: 10 },
		{ model: 'snow', lambda: 20, mu: 10 },
		{ model: 'water', lambda: 8, mu: 0, density: 1 },
		{ model: 'oil', lambda: 6, mu: 0, density: 0.8 },
		{ model: 'powder', lambda: 20, mu: 10, frictionCoefficient: 0.55 }
	];
	const particles: MpmParticle[] = materials.flatMap((material, group) =>
		Array.from({ length: 9 }, (_, local) => ({
			position: { x: 4 + group * 8 + local % 3, y: 4 + Math.floor(local / 3) },
			velocity: { x: 0.03 * group, y: -0.02 },
			C: { m00: 0, m01: 0.001, m10: 0, m11: 0 },
			F: { m00: 0.98, m01: 0.01, m10: 0, m11: 0.99 },
			material, objectId: group + 1, plasticVolume: 1
		}))
	);
	const config = {
		grid: { cellsX: 48, cellsY: 16, cellSize: 1 },
		particleMass: 0.1, particleVolume: 0.05, material: materials[0], capacity: 54
	};
	const canvas = document.createElement('canvas');
	canvas.width = canvas.height = 256;
	document.body.replaceChildren(canvas);
	const dt = 0.001;
	const gravity = { x: 0, y: 0 };
	let reference: readonly MpmParticle[] = particles;
	for (let step = 0; step < 8; step += 1) {
		reference = stepMpm(reference, { ...config, dt, gravity }).particles;
	}
	const results = [];
	try {
		for (const [variant, factory] of [
			['gather', baselineFactory],
			['atomic', createExperimentalMpmMlsAtomicGpuSolver]
		] as const) {
			let batchState: readonly MpmParticle[];
			const batched = await factory(canvas, particles, config);
			try {
				batched.stepBatch(8, dt, gravity);
				batchState = await batched.readParticles();
			} finally {
				batched.dispose();
			}
			const singles = await factory(canvas, particles, config);
			try {
				for (let step = 0; step < 8; step += 1) singles.step(dt, gravity);
				results.push({
					variant,
					particles: particles.length,
					models: materials.map((material) => material.model),
					cpuMaximumAbsoluteError: compare(reference, batchState),
					batchVsSingleMaximumAbsoluteError: compare(batchState, await singles.readParticles())
				});
			} finally {
				singles.dispose();
			}
		}
		return results;
	} finally {
		canvas.remove();
	}
}

export async function runMpmAtomicOverflowRegression() {
	const scene = createSiFluidScene();
	const results = [];
	for (const contributionOverflow of [false, true]) {
		const count = contributionOverflow ? 1 : 512;
		const particles = Array.from({ length: count }, () => ({
			...scene.particles[0],
			position: { x: 0.1, y: 0.1 },
			velocity: { x: contributionOverflow ? 1e7 : 0, y: 0 }
		}));
		const canvas = document.createElement('canvas');
		canvas.width = canvas.height = 64;
		const nativeRequestDevice = GPUAdapter.prototype.requestDevice;
		let device: GPUDevice | undefined;
		let output: GPUBuffer | undefined;
		let input: GPUBuffer | undefined;
		let restoreCreateBuffer: (() => void) | undefined;
		GPUAdapter.prototype.requestDevice = async function (descriptor) {
			const created = await nativeRequestDevice.call(this, descriptor);
			device = created;
			const createBuffer = created.createBuffer.bind(created);
			created.createBuffer = (descriptor) => {
				const buffer = createBuffer(descriptor);
				if (descriptor.label === 'MLS-MPM particle state A') input = buffer;
				if (descriptor.label === 'MLS-MPM particle state B') output = buffer;
				return buffer;
			};
			restoreCreateBuffer = () => { created.createBuffer = createBuffer; };
			return created;
		};
		let solver: Awaited<ReturnType<typeof createMpmMlsGpuSolver>> | undefined;
		let staging: GPUBuffer | undefined;
		try {
			solver = await createExperimentalMpmMlsAtomicGpuSolver(canvas, particles, scene.config);
			solver.step(1e-5, { x: 0, y: 0 });
			for (let attempt = 0; attempt < 2; attempt += 1) {
				try {
					await solver.checkNumericalStatus();
					throw new Error('An overflowing fixed-point transfer was accepted.');
				} catch (error) {
					if (!(error instanceof MpmMlsGpuNumericalError) || error.flags !== 65536) throw error;
				}
			}
			if (!device || !input || !output) throw new Error('Missing regression GPU buffers.');
			const bytes = count * 80;
			staging = device.createBuffer({ size: 2 * bytes, usage: 1 | 8 });
			const encoder = device.createCommandEncoder();
			encoder.copyBufferToBuffer(input, 0, staging, 0, bytes);
			encoder.copyBufferToBuffer(output, 0, staging, bytes, bytes);
			device.queue.submit([encoder.finish()]);
			await staging.mapAsync(1);
			const values = new Uint32Array(staging.getMappedRange().slice(0));
			staging.unmap();
			const words = bytes / 4;
			if (values.subarray(0, words).some((value, index) => value !== values[index + words])) {
				throw new Error('Fixed-point overflow did not roll back every particle bit-for-bit.');
			}
			results.push({
				kind: contributionOverflow ? 'individual contribution' : 'accumulator sum',
				particles: count, flags: 65536, persistentStatus: true, bitwiseRollback: true
			});
		} finally {
			GPUAdapter.prototype.requestDevice = nativeRequestDevice;
			restoreCreateBuffer?.();
			try {
				solver?.dispose();
			} finally {
				staging?.destroy();
				canvas.remove();
			}
		}
	}
	return results;
}
