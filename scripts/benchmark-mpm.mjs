import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';

const args = new Map(process.argv.slice(2).map((argument) => {
	const [name, ...value] = argument.replace(/^--/, '').split('=');
	return [name, value.join('=')];
}));
for (const name of args.keys()) {
	if (!['origin', 'cdp', 'profile', 'output'].includes(name)) throw new Error(`Unknown option: --${name}`);
}
const origin = args.get('origin') ?? 'http://127.0.0.1:4181';
const cdpOrigin = args.get('cdp') ?? 'http://127.0.0.1:9222';
for (const endpoint of [origin, cdpOrigin]) {
	const url = new URL(endpoint);
	if (url.protocol !== 'http:' || !['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname)) {
		throw new Error('The benchmark only connects to local HTTP endpoints.');
	}
}
const profile = args.get('profile') ?? 'stages';
if (!['stages', 'batch'].includes(profile)) throw new Error('--profile must be stages or batch.');

async function benchmark(profile) {
	const { createMpmMlsGpuSolver, MpmMlsGpuNumericalError } = await import('/src/lib/mpm/kernels/solver.ts');
	const {
		createMpmGate4Scene,
		GATE4_GRID,
		GATE4_PEBBLE_EDGE_LENGTH
	} = await import('/src/lib/mpm/scenes/gate4.ts');
	const { stepMpm } = await import('/src/lib/mpm/reference/solver.ts');
	const gate4BenchmarkSeed = 0x4d504d34;
	if (!navigator.gpu) throw new Error('WebGPU is unavailable.');
	const adapter = await navigator.gpu.requestAdapter();
	if (!adapter?.features.has('timestamp-query')) throw new Error('timestamp-query is required; queue wait is not a GPU timing substitute.');
	if (adapter.info.isFallbackAdapter) throw new Error('A hardware adapter is required for this benchmark.');
	const canvas = document.createElement('canvas');
	canvas.width = 440;
	canvas.height = 300;
	document.body.replaceChildren(canvas);
	const requestDevice = GPUAdapter.prototype.requestDevice;
	let device;
	let buffers;
	let initialAllocatedBytes = 0;
	let captureFractureBondReadback = false;
	GPUAdapter.prototype.requestDevice = async function (descriptor = {}) {
		device = await requestDevice.call(this, {
			...descriptor,
			requiredFeatures: [...new Set([...(descriptor.requiredFeatures ?? []), 'timestamp-query'])]
		});
		buffers = new Map();
		initialAllocatedBytes = 0;
		const createBuffer = device.createBuffer.bind(device);
		device.createBuffer = (descriptor) => {
			// This local harness reads back bond state to verify fracture transitions.
			const instrumentedDescriptor =
				captureFractureBondReadback &&
				descriptor.label === 'MLS-MPM persistent fracture bond state'
					? { ...descriptor, usage: descriptor.usage | GPUBufferUsage.COPY_SRC }
					: descriptor;
			const buffer = createBuffer(instrumentedDescriptor);
			buffers.set(descriptor.label, buffer);
			initialAllocatedBytes += descriptor.size;
			return buffer;
		};
		return device;
	};
	let solver;
	let querySet;
	let resolveBuffer;
	let staging;
	const quantile = (values, fraction) => {
		const sorted = [...values].sort((a, b) => a - b);
		return sorted[Math.floor((sorted.length - 1) * fraction)];
	};
	const distribution = (values) => ({
		median: quantile(values, 0.5), p95: quantile(values, 0.95),
		min: Math.min(...values), max: Math.max(...values), samples: values.length
	});
	try {
		const scene = createMpmGate4Scene(gate4BenchmarkSeed);
		const config = {
			grid: GATE4_GRID, particleMass: 1, particleVolume: 25,
			bondRestLength: GATE4_PEBBLE_EDGE_LENGTH,
			material: { lambda: 20, mu: 10, model: 'neo-hookean' }, capacity: 1_000_000
		};
		solver = await createMpmMlsGpuSolver(canvas, scene.particles, config);
		const allocatedBytes = initialAllocatedBytes;
		querySet = device.createQuerySet({ type: 'timestamp', count: 512 });
		resolveBuffer = device.createBuffer({ size: 4096, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
		staging = device.createBuffer({ size: 4096, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
		let capture = null;
		const createEncoder = device.createCommandEncoder.bind(device);
		device.createCommandEncoder = (descriptor) => {
			const encoder = createEncoder(descriptor);
			if (capture === null) return encoder;
			const recording = capture;
			const begin = encoder.beginComputePass.bind(encoder);
			const finish = encoder.finish.bind(encoder);
			function timedPass(label) {
				const index = recording.labels.length * 2;
				if (index + 2 > 512) throw new Error('Timestamp capacity exceeded.');
				recording.labels.push(label);
				return begin({
					label,
					timestampWrites: { querySet, beginningOfPassWriteIndex: index, endOfPassWriteIndex: index + 1 }
				});
			}
			encoder.beginComputePass = (descriptor = {}) => {
				if (profile === 'batch') return timedPass(descriptor.label ?? 'unlabelled');
				let pipeline;
				const bindGroups = new Map();
				// Attribution deliberately splits dispatches; batch timing uses the real pass layout.
				return {
					setPipeline(value) { pipeline = value; },
					setBindGroup(index, ...parameters) { bindGroups.set(index, parameters); },
					dispatchWorkgroups(...workgroups) {
						if (!pipeline) throw new Error('Dispatch has no pipeline.');
						const pass = timedPass(pipeline.label);
						pass.setPipeline(pipeline);
						for (const [index, parameters] of bindGroups) pass.setBindGroup(index, ...parameters);
						pass.dispatchWorkgroups(...workgroups);
						pass.end();
					},
					end() {}
				};
			};
			encoder.finish = (...parameters) => {
				const count = recording.labels.length * 2;
				encoder.resolveQuerySet(querySet, 0, count, resolveBuffer, 0);
				encoder.copyBufferToBuffer(resolveBuffer, 0, staging, 0, count * 8);
				return finish(...parameters);
			};
			return encoder;
		};
		const dt = 1 / 240;
		const gravity = { x: 0, y: 90 };
		const stationaryWarmupMs = 600;
		const warmUntil = performance.now() + stationaryWarmupMs;
		let stationaryWarmupSteps = 0;
		while (performance.now() < warmUntil) {
			solver.stepBatch(8, dt, { x: 0, y: 0 });
			await solver.measureQueueCompletionMs();
			stationaryWarmupSteps += 8;
		}
		for (let batch = 0; batch < 8; batch += 1) {
			solver.stepBatch(8, dt, gravity);
			await solver.measureQueueCompletionMs();
		}
		const samples = [];
		for (let sample = 0; sample < 12; sample += 1) {
			capture = { labels: [] };
			const started = performance.now();
			solver.stepBatch(8, dt, gravity);
			const cpuProfileEncodeMs = performance.now() - started;
			const queueWaitMs = await solver.measureQueueCompletionMs();
			const recorded = capture;
			capture = null;
			await staging.mapAsync(GPUMapMode.READ, 0, recorded.labels.length * 16);
			const values = new BigUint64Array(staging.getMappedRange(0, recorded.labels.length * 16).slice(0));
			staging.unmap();
			const stages = {};
			for (let index = 0; index < recorded.labels.length; index += 1) {
				const delta = values[index * 2 + 1] - values[index * 2];
				if (delta < 0n) throw new Error('Timestamp order is invalid.');
				const label = recorded.labels[index];
				stages[label] = (stages[label] ?? 0) + Number(delta) / 1e6 / 8;
			}
			samples.push({
				cpuProfileEncodeMs, queueWaitMs,
				gpuStepMs: Number(values.at(-1) - values[0]) / 1e6 / 8,
				measuredPassCount: recorded.labels.length, stages
			});
		}
		const normalCpuSamples = [];
		for (let sample = 0; sample < 12; sample += 1) {
			const started = performance.now();
			solver.stepBatch(8, dt, gravity);
			normalCpuSamples.push(performance.now() - started);
			await solver.measureQueueCompletionMs();
		}
		await solver.checkNumericalStatus();
		const performanceResult = {
			profile,
			config: {
				particles: scene.particles.length, capacity: config.capacity, grid: config.grid,
				dt, gravity, warmupSteps: 64, stationaryWarmupMs, stationaryWarmupSteps,
				sampleBatches: 12, stepsPerBatch: 8, rendered: false
			},
			adapter: {
				vendor: adapter.info.vendor, architecture: adapter.info.architecture,
				isFallbackAdapter: adapter.info.isFallbackAdapter
			},
			browser: navigator.userAgent,
			deviceFeatures: [...device.features],
			deviceLimits: {
				maxBufferSize: device.limits.maxBufferSize,
				maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
				maxStorageBuffersPerShaderStage: device.limits.maxStorageBuffersPerShaderStage,
				maxComputeWorkgroupsPerDimension: device.limits.maxComputeWorkgroupsPerDimension
			},
			allocatedBufferBytes: allocatedBytes,
			gpuStepMs: distribution(samples.map((sample) => sample.gpuStepMs)),
			cpuNormalEncodeBatchMs: distribution(normalCpuSamples),
			cpuProfileEncodeBatchMs: distribution(samples.map((sample) => sample.cpuProfileEncodeMs)),
			queueWaitBatchMs: distribution(samples.map((sample) => sample.queueWaitMs)),
			measuredPassCountPerBatch: samples[0].measuredPassCount,
			stages: Object.fromEntries(Object.keys(samples[0].stages).map((name) => [
				name, distribution(samples.map((sample) => sample.stages[name]))
			]))
		};
		querySet.destroy(); querySet = null;
		resolveBuffer.destroy(); resolveBuffer = null;
		staging.destroy(); staging = null;
		solver.dispose(); solver = null;

		const materials = [
			{ model: 'neo-hookean', lambda: 20, mu: 10 },
			{ model: 'snow', lambda: 20, mu: 10 },
			{ model: 'water', lambda: 8, mu: 0, density: 1 },
			{ model: 'oil', lambda: 6, mu: 0, density: 0.8 },
			{ model: 'powder', lambda: 20, mu: 10, frictionCoefficient: 0.55 }
		];
		const particles = materials.flatMap((material, group) => Array.from({ length: 9 }, (_, local) => ({
			position: { x: 4 + group * 8 + local % 3, y: 4 + Math.floor(local / 3) },
			velocity: { x: 0.03 * group, y: -0.02 },
			C: { m00: 0, m01: 0.001, m10: 0, m11: 0 },
			F: { m00: 0.98, m01: 0.01, m10: 0, m11: 0.99 },
			material, objectId: group + 1, plasticVolume: 1
		})));
		const parityConfig = {
			grid: { cellsX: 48, cellsY: 16, cellSize: 1 }, particleMass: 0.1, particleVolume: 0.05,
			material: materials[0], dt: 0.001, gravity: { x: 0, y: 0 }, capacity: 54
		};
		const values = (p) => [
			p.position.x, p.position.y, p.velocity.x, p.velocity.y,
			p.C.m00, p.C.m01, p.C.m10, p.C.m11, p.F.m00, p.F.m01, p.F.m10, p.F.m11,
			p.plasticVolume ?? 1
		];
		function maxDifference(actual, expected) {
			let maximum = 0;
			if (actual.length !== expected.length) throw new Error('Particle count changed.');
			for (let index = 0; index < expected.length; index += 1) {
				const a = values(actual[index]), b = values(expected[index]);
				for (let item = 0; item < a.length; item += 1) {
					if (!Number.isFinite(a[item])) throw new Error('Non-finite particle state.');
					maximum = Math.max(maximum, Math.abs(a[item] - b[item]));
				}
				if (actual[index].objectId !== expected[index].objectId ||
					actual[index].material.model !== expected[index].material.model) {
					throw new Error('Particle identity changed.');
				}
			}
			return maximum;
		}
		solver = await createMpmMlsGpuSolver(canvas, particles, parityConfig);
		let expected = particles;
		for (let step = 0; step < 8; step += 1) expected = stepMpm(expected, parityConfig).particles;
		solver.stepBatch(8, parityConfig.dt, parityConfig.gravity);
		const batched = await solver.readParticles();
		const maxAbsoluteError = maxDifference(batched, expected);
		if (maxAbsoluteError > 1e-3) throw new Error(`CPU/GPU parity exceeds tolerance: ${maxAbsoluteError}`);
		const selection = await solver.pick(batched[0].position, 0.1);
		if (selection?.particleIndex !== 0 || selection.objectId !== 1) throw new Error('GPU picking regression.');
		solver.beginDrag(selection, { x: selection.anchor.x + 0.1, y: selection.anchor.y }, 2);
		solver.step(parityConfig.dt, parityConfig.gravity);
		solver.endDrag();
		const dragged = await solver.readParticles();
		if (!solver.clone(selection, { x: 0.1, y: 0 }) || solver.particleCount !== 54) throw new Error('GPU cloning regression.');
		if (solver.clone(selection, { x: 0.1, y: 0 })) throw new Error('Particle capacity was not enforced.');
		const cloned = await solver.readParticles(Array.from({ length: 9 }, (_, index) => 45 + index));
		for (let index = 0; index < cloned.length; index += 1) {
			const source = dragged[index], copy = cloned[index];
			if (copy.objectId !== 6 || copy.material.model !== source.material.model ||
				Math.abs(copy.position.x - source.position.x - 0.1) > 1e-5 ||
				JSON.stringify(values(copy).slice(2)) !== JSON.stringify(values(source).slice(2))) {
				throw new Error('Clone failed to preserve particle state.');
			}
		}
		solver.render();
		await solver.measureQueueCompletionMs();
		await solver.checkNumericalStatus();
		solver.dispose(); solver = null;
		solver = await createMpmMlsGpuSolver(canvas, particles, parityConfig);
		for (let step = 0; step < 8; step += 1) solver.step(parityConfig.dt, parityConfig.gravity);
		const singles = await solver.readParticles();
		const batchVsSingleError = maxDifference(singles, batched);
		if (batchVsSingleError > 1e-3) throw new Error(`Batch/single regression: ${batchVsSingleError}`);
		solver.dispose(); solver = null;

		const rollbackParticle = {
			position: { x: 32, y: 32 }, velocity: { x: 0, y: 0 },
			C: { m00: 0, m01: 1000, m10: 0, m11: 0 },
			F: { m00: 1, m01: 0, m10: 0, m11: 1 }, material: materials[0], objectId: 1
		};
		const rollbackParticles = Array.from({ length: 129 }, (_, index) => ({
			...rollbackParticle,
			position: { x: 12 + index % 10 * 2, y: 12 + Math.floor(index / 10) * 2 },
			velocity: { x: 0.03, y: 0.02 },
			C: { m00: 0, m01: 0, m10: 0, m11: 0 }
		}));
		rollbackParticles[128] = rollbackParticle;
		solver = await createMpmMlsGpuSolver(canvas, rollbackParticles, {
			...parityConfig, grid: { cellsX: 64, cellsY: 64, cellSize: 1 }, capacity: 129
		});
		async function rawState(buffer) {
			const readback = device.createBuffer({ size: buffer.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
			try {
				const encoder = device.createCommandEncoder();
				encoder.copyBufferToBuffer(buffer, 0, readback, 0, buffer.size);
				device.queue.submit([encoder.finish()]);
				await readback.mapAsync(GPUMapMode.READ);
				const words = Array.from(new Uint32Array(readback.getMappedRange().slice(0)));
				readback.unmap();
				return words;
			} finally { readback.destroy(); }
		}
		const before = await rawState(buffers.get('MLS-MPM particle state A'));
		solver.step(0.1, { x: 0, y: 0 });
		const rollbackFlags = await solver.checkNumericalStatus().then(
			() => { throw new Error('Expected a numerical substep-limit failure.'); },
			(error) => {
				if (!(error instanceof MpmMlsGpuNumericalError) || !(error.flags & 32)) throw error;
				return error.flags;
			}
		);
		for (const label of ['MLS-MPM particle state A', 'MLS-MPM particle state B']) {
			if (JSON.stringify(await rawState(buffers.get(label))) !== JSON.stringify(before)) {
				throw new Error('Failed GPU step did not roll back all particle state.');
			}
		}
		solver.step(0.001, { x: 0, y: 0 });
		await solver.checkNumericalStatus().then(
			() => { throw new Error('Numerical status was unexpectedly cleared.'); },
			(error) => {
				if (!(error instanceof MpmMlsGpuNumericalError) || error.flags !== rollbackFlags) throw error;
			}
		);
		solver.dispose(); solver = null;

		const physicsParityTolerance = 1e-3;
		const zeroMatrix = { m00: 0, m01: 0, m10: 0, m11: 0 };
		const identityMatrix = { m00: 1, m01: 0, m10: 0, m11: 1 };
		const totalMomentum = (state, massOf) => state.reduce(
			(momentum, particle) => {
				const mass = massOf(particle);
				momentum.x += mass * particle.velocity.x;
				momentum.y += mass * particle.velocity.y;
				return momentum;
			},
			{ x: 0, y: 0 }
		);
		const phaseMass = (state, model) => state
			.filter((particle) => particle.material?.model === model)
			.reduce((mass, particle) => mass + particle.volume * particle.material.density, 0);

		const waterMaterial = { model: 'water', lambda: 8, mu: 0, density: 1 };
		const oilMaterial = { model: 'oil', lambda: 8, mu: 0, density: 1 };
		const contactParticles = [
			{
				position: { x: 7.5, y: 8 }, velocity: { x: 5, y: 0 },
				C: zeroMatrix, F: identityMatrix, volume: 0.25,
				objectId: 1, material: waterMaterial
			},
			{
				position: { x: 8.5, y: 8 }, velocity: { x: -5, y: 0 },
				C: zeroMatrix, F: identityMatrix, volume: 0.25,
				objectId: 2, material: oilMaterial
			}
		];
		const contactConfig = {
			grid: { cellsX: 16, cellsY: 16, cellSize: 1 },
			particleMass: 0.1, particleVolume: 0.25,
			material: { model: 'neo-hookean', lambda: 20, mu: 10 },
			dt: 0.001, gravity: { x: 0, y: 0 }, capacity: 2
		};
		const contactSteps = 8;
		let expectedContactParticles = contactParticles;
		let actualContact = contactParticles;
		let contactMaxAbsoluteError = 0;
		let contactMaxMomentumResidual = 0;
		let contactMaxPhaseMassError = 0;
		const inputWaterMass = phaseMass(contactParticles, 'water');
		const inputOilMass = phaseMass(contactParticles, 'oil');
		const contactMassOf = (particle) => particle.volume * particle.material.density;
		const inputContactMomentum = totalMomentum(contactParticles, contactMassOf);
		solver = await createMpmMlsGpuSolver(canvas, contactParticles, contactConfig);
		for (let step = 0; step < contactSteps; step += 1) {
			const expectedContact = stepMpm(expectedContactParticles, contactConfig);
			solver.step(contactConfig.dt, contactConfig.gravity);
			actualContact = await solver.readParticles();
			await solver.checkNumericalStatus();
			const contactError = maxDifference(actualContact, expectedContact.particles);
			if (contactError > physicsParityTolerance) {
				throw new Error(`Water/oil CPU/GPU parity exceeds tolerance at step ${step + 1}: ${contactError}`);
			}
			contactMaxAbsoluteError = Math.max(contactMaxAbsoluteError, contactError);
			if (
				actualContact.length !== 2 ||
				actualContact[0].material.model !== 'water' ||
				actualContact[1].material.model !== 'oil' ||
				actualContact[0].objectId !== 1 ||
				actualContact[1].objectId !== 2
			) {
				throw new Error(`Water/oil phase identity changed at step ${step + 1}.`);
			}
			const outputWaterMass = phaseMass(actualContact, 'water');
			const outputOilMass = phaseMass(actualContact, 'oil');
			const phaseMassError = Math.max(
				Math.abs(outputWaterMass - inputWaterMass),
				Math.abs(outputOilMass - inputOilMass)
			);
			if (phaseMassError > 1e-6) {
				throw new Error(`Water/oil contact changed a phase mass at step ${step + 1}.`);
			}
			contactMaxPhaseMassError = Math.max(contactMaxPhaseMassError, phaseMassError);
			const outputContactMomentum = totalMomentum(actualContact, contactMassOf);
			const momentumResidual = Math.hypot(
				outputContactMomentum.x - inputContactMomentum.x,
				outputContactMomentum.y - inputContactMomentum.y
			);
			if (momentumResidual > physicsParityTolerance) {
				throw new Error(`Water/oil contact momentum residual exceeds tolerance at step ${step + 1}: ${momentumResidual}`);
			}
			contactMaxMomentumResidual = Math.max(contactMaxMomentumResidual, momentumResidual);
			if (actualContact[0].position.x > actualContact[1].position.x) {
				throw new Error(`Water/oil particles crossed at step ${step + 1}.`);
			}
			if (
				step === 0 &&
				!(
					actualContact[0].velocity.x < contactParticles[0].velocity.x &&
					actualContact[1].velocity.x > contactParticles[1].velocity.x
				)
			) {
				throw new Error('Water/oil interface contact did not reduce closing motion.');
			}
			expectedContactParticles = expectedContact.particles;
		}
		const liquidContactAcceptance = {
			particles: actualContact.length,
			steps: contactSteps,
			maxAbsoluteError: contactMaxAbsoluteError,
			tolerance: physicsParityTolerance,
			waterMass: { input: inputWaterMass, output: phaseMass(actualContact, 'water') },
			oilMass: { input: inputOilMass, output: phaseMass(actualContact, 'oil') },
			maxPhaseMassError: contactMaxPhaseMassError,
			maxMomentumResidual: contactMaxMomentumResidual,
			phaseIdentity: true,
			closingMotionReducedOnFirstStep: true,
			noCrossingAcrossAllSteps: true
		};
		solver.dispose(); solver = null;

		async function readFractureBondState(bondCount) {
			const buffer = buffers.get('MLS-MPM persistent fracture bond state');
			if (!buffer) throw new Error('GPU fracture-bond state buffer was not captured.');
			const words = await rawState(buffer);
			const floats = new DataView(Uint32Array.from(words).buffer);
			return Array.from({ length: bondCount }, (_, index) => {
				const wordOffset = index * 6;
				return {
					firstParticle: words[wordOffset],
					secondParticle: words[wordOffset + 1],
					damage: floats.getFloat32((wordOffset + 4) * 4, true),
					broken: words[wordOffset + 5] !== 0
				};
			});
		}
		function compareFractureState(actualParticles, actualBonds, expected, stepLabel) {
			if (!expected.fractureBonds) throw new Error(`${stepLabel}: CPU reference omitted fracture bonds.`);
			const particleError = maxDifference(actualParticles, expected.particles);
			if (particleError > physicsParityTolerance) {
				throw new Error(`${stepLabel}: fracture particle parity exceeds tolerance: ${particleError}`);
			}
			if (actualBonds.length !== expected.fractureBonds.length) {
				throw new Error(`${stepLabel}: GPU fracture-bond count changed.`);
			}
			let particleDamageError = 0;
			for (let index = 0; index < expected.particles.length; index += 1) {
				const actualDamage = actualParticles[index].damage ?? 0;
				const expectedDamage = expected.particles[index].damage ?? 0;
				if (!Number.isFinite(actualDamage)) {
					throw new Error(`${stepLabel}: GPU particle damage is non-finite.`);
				}
				particleDamageError = Math.max(
					particleDamageError,
					Math.abs(actualDamage - expectedDamage)
				);
			}
			let bondDamageError = 0;
			for (let index = 0; index < expected.fractureBonds.length; index += 1) {
				const actual = actualBonds[index];
				const reference = expected.fractureBonds[index];
				if (
					actual.firstParticle !== reference.firstParticle ||
					actual.secondParticle !== reference.secondParticle ||
					actual.broken !== (reference.broken ?? false) ||
					!Number.isFinite(actual.damage)
				) {
					throw new Error(`${stepLabel}: GPU fracture-bond state differs at bond ${index}.`);
				}
				bondDamageError = Math.max(
					bondDamageError,
					Math.abs(actual.damage - (reference.damage ?? 0))
				);
			}
			if (
				particleDamageError > physicsParityTolerance ||
				bondDamageError > physicsParityTolerance
			) {
				throw new Error(`${stepLabel}: fracture damage parity exceeds tolerance.`);
			}
			return {
				particleMaxAbsoluteError: particleError,
				particleDamageMaxAbsoluteError: particleDamageError,
				bondDamageMaxAbsoluteError: bondDamageError,
				brokenBondIndices: actualBonds
					.map((bond, index) => bond.broken ? index : null)
					.filter((index) => index !== null)
			};
		}

		const fractureMaterial = { model: 'neo-hookean', lambda: 20, mu: 10 };
		const fractureParticles = [16, 17.05, 18.16, 19.16].map((x) => ({
			position: { x, y: 32 },
			velocity: { x: 0, y: 0 },
			C: zeroMatrix,
			F: identityMatrix,
			objectId: 41,
			material: fractureMaterial
		}));
		const fractureBonds = [
			{ firstParticle: 0, secondParticle: 1, restLength: 1, failureStrain: 0.1 },
			{ firstParticle: 1, secondParticle: 2, restLength: 1, failureStrain: 0.1 },
			{ firstParticle: 2, secondParticle: 3, restLength: 1, failureStrain: 0.1 }
		];
		const fractureConfig = {
			grid: { cellsX: 64, cellsY: 64, cellSize: 1 },
			particleMass: 1, particleVolume: 1, material: fractureMaterial,
			capacity: fractureParticles.length, fractureBonds
		};
		const fractureStepConfig = {
			...fractureConfig, dt: 0.001, gravity: { x: 0, y: 0 }
		};
		const expectedBrokenPattern = (result) => {
			if (!result.fractureBonds) throw new Error('CPU reference omitted fracture bonds.');
			return result.fractureBonds
				.map((bond, index) => bond.broken ? index : null)
				.filter((index) => index !== null);
		};
		let expectedFracture = stepMpm(fractureParticles, fractureStepConfig);
		if (JSON.stringify(expectedBrokenPattern(expectedFracture)) !== '[1]') {
			throw new Error('Fracture fixture does not isolate below-, above-, and unstrained bonds.');
		}
		captureFractureBondReadback = true;
		solver = await createMpmMlsGpuSolver(canvas, fractureParticles, fractureConfig);
		solver.step(fractureStepConfig.dt, fractureStepConfig.gravity);
		let actualFractureParticles = await solver.readParticles();
		await solver.checkNumericalStatus();
		let actualFractureBonds = await readFractureBondState(fractureBonds.length);
		const firstFractureComparison = compareFractureState(
			actualFractureParticles,
			actualFractureBonds,
			expectedFracture,
			'Fracture step 1'
		);
		expectedFracture = stepMpm(expectedFracture.particles, {
			...fractureStepConfig, fractureBonds: expectedFracture.fractureBonds
		});
		solver.step(fractureStepConfig.dt, fractureStepConfig.gravity);
		actualFractureParticles = await solver.readParticles();
		await solver.checkNumericalStatus();
		actualFractureBonds = await readFractureBondState(fractureBonds.length);
		const secondFractureComparison = compareFractureState(
			actualFractureParticles,
			actualFractureBonds,
			expectedFracture,
			'Fracture step 2'
		);
		const fractureMass = actualFractureParticles.length * fractureConfig.particleMass;
		const fractureMomentum = totalMomentum(
			actualFractureParticles,
			() => fractureConfig.particleMass
		);
		if (
			actualFractureParticles.length !== fractureParticles.length ||
			Math.abs(fractureMass - fractureParticles.length * fractureConfig.particleMass) > 1e-6 ||
			Math.hypot(fractureMomentum.x, fractureMomentum.y) > physicsParityTolerance ||
			actualFractureParticles.some(
				({ position }) =>
					position.x < 0 || position.x > fractureConfig.grid.cellsX ||
					position.y < 0 || position.y > fractureConfig.grid.cellsY
			)
		) {
			throw new Error('Fracture changed particle mass/momentum or moved particles outside the grid.');
		}
		solver.dispose(); solver = null;

		const persistentParticles = [20, 21].map((x) => ({
			position: { x, y: 32 },
			velocity: { x: 0, y: 0 },
			C: zeroMatrix,
			F: identityMatrix,
			objectId: 42,
			material: fractureMaterial
		}));
		const persistentBond = [{
			firstParticle: 0, secondParticle: 1, restLength: 1,
			failureStrain: 0.1, damage: 1, broken: true
		}];
		const persistenceConfig = {
			...fractureConfig,
			capacity: persistentParticles.length,
			fractureBonds: persistentBond
		};
		const persistenceStepConfig = {
			...persistenceConfig, dt: 0.001, gravity: { x: 0, y: 0 }
		};
		let expectedPersistence = stepMpm(persistentParticles, persistenceStepConfig);
		expectedPersistence = stepMpm(expectedPersistence.particles, {
			...persistenceStepConfig, fractureBonds: expectedPersistence.fractureBonds
		});
		solver = await createMpmMlsGpuSolver(canvas, persistentParticles, persistenceConfig);
		solver.stepBatch(2, persistenceStepConfig.dt, persistenceStepConfig.gravity);
		const actualPersistenceParticles = await solver.readParticles();
		await solver.checkNumericalStatus();
		const actualPersistenceBonds = await readFractureBondState(persistentBond.length);
		const persistenceComparison = compareFractureState(
			actualPersistenceParticles,
			actualPersistenceBonds,
			expectedPersistence,
			'Broken-bond persistence'
		);
		if (!actualPersistenceBonds[0].broken || actualPersistenceBonds[0].damage !== 1) {
			throw new Error('An initialized broken bond revived below its failure strain.');
		}
		const fractureAcceptance = {
			particles: fractureParticles.length,
			bonds: fractureBonds.length,
			steps: 2,
			failureStrain: 0.1,
			testBondStrains: [0.05, 0.11, 0],
			brokenBondIndices: secondFractureComparison.brokenBondIndices,
			belowThresholdDamage: actualFractureBonds[0].damage,
			maxParticleError: Math.max(
				firstFractureComparison.particleMaxAbsoluteError,
				secondFractureComparison.particleMaxAbsoluteError
			),
			maxParticleDamageError: Math.max(
				firstFractureComparison.particleDamageMaxAbsoluteError,
				secondFractureComparison.particleDamageMaxAbsoluteError
			),
			maxBondDamageError: Math.max(
				firstFractureComparison.bondDamageMaxAbsoluteError,
				secondFractureComparison.bondDamageMaxAbsoluteError
			),
			persistence: {
				steps: 2,
				brokenAtRest: actualPersistenceBonds[0].broken,
				damage: actualPersistenceBonds[0].damage,
				maxParticleError: persistenceComparison.particleMaxAbsoluteError
			},
			mass: fractureMass,
			momentumResidual: Math.hypot(fractureMomentum.x, fractureMomentum.y)
		};
		solver.dispose(); solver = null;
		return {
			performance: performanceResult,
			regression: {
				particles: 45, steps: 8, models: materials.map((material) => material.model),
				maxAbsoluteError, batchVsSingleError,
				pick: true, drag: true, cloneStateAndCapacity: true, render: true,
				rollback: true, rollbackParticles: rollbackParticles.length,
				persistentNumericalStatus: true, rollbackFlags
			},
			physicsAcceptance: {
				tolerance: physicsParityTolerance,
				waterOilContact: liquidContactAcceptance,
				fracture: fractureAcceptance
			}
		};
	} finally {
		staging?.destroy(); resolveBuffer?.destroy(); querySet?.destroy(); solver?.dispose();
		GPUAdapter.prototype.requestDevice = requestDevice;
	}
}

const response = await fetch(`${cdpOrigin}/json/version`);
if (!response.ok) throw new Error(`CDP connection failed: ${response.status}`);
const version = await response.json();
const socket = new WebSocket(version.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
	socket.addEventListener('open', resolve, { once: true });
	socket.addEventListener('error', reject, { once: true });
});
const pending = new Map();
let nextId = 0;
async function readSourceHashes() {
	const hashes = {};
	for (const name of ['layout', 'shaders', 'solver']) {
		const file = new URL(`../src/lib/mpm/kernels/${name}.ts`, import.meta.url);
		hashes[name] = createHash('sha256').update(await readFile(file)).digest('hex');
	}
	return hashes;
}
const initialSourceHashes = await readSourceHashes();
socket.addEventListener('close', () => {
	for (const callback of pending.values()) callback.reject(new Error('The CDP connection closed during the benchmark.'));
	pending.clear();
});
socket.addEventListener('message', ({ data }) => {
	const message = JSON.parse(data);
	if (message.id === undefined) return;
	const callback = pending.get(message.id);
	if (!callback) return;
	pending.delete(message.id);
	if (message.error) callback.reject(new Error(JSON.stringify(message.error)));
	else callback.resolve(message.result);
});
function send(method, params = {}, sessionId) {
	const id = ++nextId;
	const result = new Promise((resolve, reject) => pending.set(id, { resolve, reject }));
	socket.send(JSON.stringify({ id, method, params, sessionId }));
	return result;
}
let targetId;
try {
	({ targetId } = await send('Target.createTarget', { url: `${origin}/@vite/client` }));
	const { sessionId } = await send('Target.attachToTarget', { targetId, flatten: true });
	await send('Runtime.enable', {}, sessionId);
	let ready = false;
	for (let attempt = 0; attempt < 100; attempt += 1) {
		const result = await send('Runtime.evaluate', {
			expression: `document.readyState === 'complete' && location.origin === ${JSON.stringify(origin)}`,
			returnByValue: true
		}, sessionId);
		if (result.result.value === true) { ready = true; break; }
		await new Promise((resolve) => setTimeout(resolve, 100));
	}
	if (!ready) throw new Error('The local source server did not load within 10 seconds.');
	const result = await send('Runtime.evaluate', {
		expression: `(${benchmark.toString()})(${JSON.stringify(profile)})`,
		awaitPromise: true, returnByValue: true, timeout: 180_000
	}, sessionId);
	if (result.exceptionDetails) {
		throw new Error(result.exceptionDetails.exception?.description ?? JSON.stringify(result.exceptionDetails));
	}
	const sourceHashes = await readSourceHashes();
	if (JSON.stringify(sourceHashes) !== JSON.stringify(initialSourceHashes)) {
		throw new Error('Simulation source changed while the benchmark was running; repeat the measurement.');
	}
	const output = JSON.stringify({ sourceHashes, ...result.result.value }, null, 2);
	if (args.has('output')) await writeFile(args.get('output'), `${output}\n`);
	console.log(output);
} finally {
	if (targetId && socket.readyState === WebSocket.OPEN) await send('Target.closeTarget', { targetId });
	socket.close();
}
