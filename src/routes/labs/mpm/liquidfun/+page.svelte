<script lang="ts">
	import { onMount } from 'svelte';
	import { projectDeviceOrientationGravity } from '#lib/input/sensors.js';
	import Gate4ParticleSizeControl from '#lib/mpm/Gate4ParticleSizeControl.svelte';
	import { DEFAULT_MPM_COURANT_LIMIT } from '#lib/mpm/kernels/cfl.js';
	import {
		MPM_RENDER_FPS_OPTIONS,
		MpmFramePacer,
		type MpmRenderFps
	} from '#lib/mpm/kernels/frame-pacer.js';
	import {
		DEFAULT_HORIZONTAL_OSCILLATION_AMPLITUDE,
		HORIZONTAL_OSCILLATION_FREQUENCY_HZ,
		INITIAL_HORIZONTAL_OSCILLATION_AMPLITUDE,
		MAX_HORIZONTAL_OSCILLATION_AMPLITUDE
	} from '#lib/mpm/kernels/oscillation.js';
	import {
		MpmCpuReferenceSimulation,
		MPM_CPU_MAX_CFL_MICROSTEPS_PER_MANUAL_STEP,
		type MpmCpuRenderMode,
		type MpmCpuWorkerRequest,
		type MpmCpuWorkerResponse
	} from '#lib/mpm/reference/cpu-simulation.js';
	import type { MpmVector2 } from '#lib/mpm/reference/solver.js';
	import {
		createMpmGate4Scene,
		GATE4_DEFAULT_GRAVITY,
		GATE4_FIXED_STEP_SECONDS,
		GATE4_FRACTURE_BOND_COUNT,
		GATE4_FRACTURE_PARTICLE_COUNT,
		GATE4_GRID,
		GATE4_LIQUID_PARTICLE_DIAMETER,
		GATE4_LIQUID_PARTICLE_DIAMETER_QUERY_KEY,
		GATE4_OIL_PARTICLE_COUNT,
		GATE4_PARTICLE_COUNT,
		GATE4_PEBBLE_COUNT,
		GATE4_PEBBLE_PARTICLE_COUNT,
		GATE4_SCENE_SEED,
		GATE4_WATER_PARTICLE_COUNT
	} from '#lib/mpm/scenes/gate4.js';

	type Phase = 'loading' | 'ready' | 'error';
	type OrientationStatus =
		| 'checking'
		| 'idle'
		| 'requesting'
		| 'enabled'
		| 'denied'
		| 'unsupported';

	const MAX_CPU_MICROSTEPS_PER_BATCH = 8;
	const METRICS_INTERVAL_MS = 500;
	const MAX_SIMULATION_SPEED = 32;
	const SIMULATION_SPEED_OPTIONS = [0.5, 1, 2, 4, 8, 16, MAX_SIMULATION_SPEED] as const;
	const DEFAULT_SIMULATION_SPEED = 2;
	const DEFAULT_RENDER_FPS: MpmRenderFps = 60;
	const STEP_SECONDS = GATE4_FIXED_STEP_SECONDS;
	const framePacer = new MpmFramePacer();

	let canvas: HTMLCanvasElement | undefined;
	let context: CanvasRenderingContext2D | null = null;
	let simulation = $state<MpmCpuReferenceSimulation | null>(null);
	let cpuWorker: Worker | null = null;
	let workerBusy = $state(false);
	let phase = $state<Phase>('loading');
	let paused = $state(false);
	let isVisible = $state(true);
	let errorMessage = $state('');
	let fps = $state(0);
	let physicsMsPerMicrostep = $state(0);
	let canvasDrawMsPerFrame = $state(0);
	let stepsPerSecond = $state(0);
	let microstepsPerSecond = $state(0);
	let pendingSimulationMs = $state(0);
	let gravityMagnitude = $state(GATE4_DEFAULT_GRAVITY.y);
	let manualGravityX = $state(GATE4_DEFAULT_GRAVITY.x);
	let manualGravityY = $state(GATE4_DEFAULT_GRAVITY.y);
	let orientationStatus = $state<OrientationStatus>('checking');
	let orientationMessage = $state('DeviceOrientation センサーを確認しています…');
	let sensorGravityUnit = $state<MpmVector2>({ x: 0, y: 0 });
	let hasSensorGravity = $state(false);
	let simulationSpeed = $state(DEFAULT_SIMULATION_SPEED);
	let renderTargetFps: MpmRenderFps = $state(DEFAULT_RENDER_FPS);
	let renderMode = $state<MpmCpuRenderMode>('surface');
	let liquidParticleDiameter = $state(GATE4_LIQUID_PARTICLE_DIAMETER);
	let appliedLiquidParticleDiameter = $state(GATE4_LIQUID_PARTICLE_DIAMETER);
	let horizontalOscillationAmplitude = $state(
		INITIAL_HORIZONTAL_OSCILLATION_AMPLITUDE
	);

	let mounted = false;
	let generation = 0;
	let animationFrame = 0;
	let lastFrameTime = 0;
	let lastMetricsTime = 0;
	let accumulatorSeconds = 0;
	let frameCount = 0;
	let physicsStepsForMetrics = 0;
	let physicsMicrostepsForMetrics = 0;
	let macroStepRemainderSeconds = 0;
	let physicsMsForMetrics = 0;
	let drawMsForMetrics = 0;

	function effectiveGravity(): MpmVector2 {
		if (orientationStatus === 'enabled' && hasSensorGravity) {
			return {
				x: sensorGravityUnit.x * gravityMagnitude,
				y: sensorGravityUnit.y * gravityMagnitude
			};
		}
		return { x: manualGravityX, y: manualGravityY };
	}

	let appliedGravity = $derived(effectiveGravity());

	const statusText = $derived(
		phase === 'loading'
			? 'CPU MLS-MPM reference を初期化中…'
			: phase === 'error'
				? 'シミュレーションを初期化できません'
				: !isVisible
					? 'ページ非表示中 · 物理ステップを停止中'
					: paused
						? '一時停止中 · 固定刻み 1/240 秒'
						: `実行中 · CPU · ${simulationSpeed}× · 固定刻み 1/240 秒`
	);

	function errorText(error: unknown): string {
		return error instanceof Error && error.message.trim()
			? error.message
			: 'CPU MLS-MPM の処理中に詳細不明のエラーが発生しました。';
	}

	function resetMetrics(): void {
		fps = 0;
		physicsMsPerMicrostep = 0;
		canvasDrawMsPerFrame = 0;
		stepsPerSecond = 0;
		microstepsPerSecond = 0;
		pendingSimulationMs = 0;
		frameCount = 0;
		physicsStepsForMetrics = 0;
		physicsMicrostepsForMetrics = 0;
		macroStepRemainderSeconds = 0;
		physicsMsForMetrics = 0;
		drawMsForMetrics = 0;
		accumulatorSeconds = 0;
		lastFrameTime = 0;
		lastMetricsTime = 0;
		framePacer.reset();
	}

	function postWorkerRequest(request: MpmCpuWorkerRequest): void {
		if (!cpuWorker) {
			throw new Error('CPU MLS-MPM worker is not initialized.');
		}
		if (workerBusy) {
			throw new Error('A CPU MLS-MPM worker batch is already in flight.');
		}
		workerBusy = true;
		try {
			cpuWorker.postMessage(request);
		} catch (error) {
			workerBusy = false;
			throw error;
		}
	}

	function handleWorkerResponse(response: MpmCpuWorkerResponse): void {
		if (!mounted || response.generation !== generation) return;
		if (response.type === 'error') {
			workerBusy = false;
			failSimulation(new Error(response.message));
			return;
		}
		if (!simulation) {
			workerBusy = false;
			failSimulation(new Error('CPU MLS-MPM worker returned state before initialization.'));
			return;
		}
		try {
			simulation.applyRenderSnapshot(response.snapshot);
		} catch (error) {
			workerBusy = false;
			failSimulation(error);
			return;
		}

		workerBusy = false;
		accumulatorSeconds += response.remainderSeconds;
		if (response.type === 'ready') {
			appliedLiquidParticleDiameter = liquidParticleDiameter;
			pendingSimulationMs = accumulatorSeconds * 1_000;
			phase = 'ready';
			return;
		}

		const macroStepProgress =
			macroStepRemainderSeconds + response.simulatedSeconds;
		const macroStepTolerance = STEP_SECONDS * Number.EPSILON * 8;
		const completedMacroSteps = Math.floor(
			(macroStepProgress + macroStepTolerance) / STEP_SECONDS
		);
		physicsStepsForMetrics += completedMacroSteps;
		macroStepRemainderSeconds = Math.max(
			0,
			macroStepProgress - completedMacroSteps * STEP_SECONDS
		);
		physicsMicrostepsForMetrics += response.microStepCount;
		physicsMsForMetrics += response.workerMilliseconds;
		pendingSimulationMs = accumulatorSeconds * 1_000;
	}

	function initializeSimulation(): void {
		const currentGeneration = ++generation;
		paused = false;
		phase = 'loading';
		errorMessage = '';
		resetMetrics();
		cpuWorker?.terminate();
		cpuWorker = null;
		workerBusy = false;
		simulation?.dispose();
		simulation = null;

		try {
			const particleDiameterParameter = new URLSearchParams(
				window.location.search
			).get(GATE4_LIQUID_PARTICLE_DIAMETER_QUERY_KEY);
			if (particleDiameterParameter !== null) {
				liquidParticleDiameter = Number(particleDiameterParameter);
			}
			const scene = createMpmGate4Scene(GATE4_SCENE_SEED, {
				liquidParticleDiameter
			});
			const created = new MpmCpuReferenceSimulation(scene);
			if (!mounted || currentGeneration !== generation) {
				created.dispose();
				return;
			}
			simulation = created;
			const worker = new Worker(
				new URL('../../../../lib/mpm/reference/cpu-simulation.worker.ts', import.meta.url),
				{ type: 'module' }
			);
			worker.onmessage = (event: MessageEvent<MpmCpuWorkerResponse>) =>
				handleWorkerResponse(event.data);
			worker.onerror = (event: ErrorEvent) => {
				if (!mounted || currentGeneration !== generation) return;
				failSimulation(
					new Error(event.message || 'CPU MLS-MPM worker terminated unexpectedly.')
				);
			};
			cpuWorker = worker;
			postWorkerRequest({
				type: 'initialize',
				generation: currentGeneration,
				liquidParticleDiameter,
				gravity: effectiveGravity()
			});
		} catch (error) {
			if (!mounted || currentGeneration !== generation) return;
			failSimulation(error);
		}
	}

	function resizeCanvas(): void {
		if (!canvas || !context) return;
		const bounds = canvas.getBoundingClientRect();
		const devicePixelRatio = Math.min(window.devicePixelRatio || 1, 2);
		const pixelWidth = Math.max(1, Math.round(bounds.width * devicePixelRatio));
		const pixelHeight = Math.max(1, Math.round(bounds.height * devicePixelRatio));
		if (canvas.width !== pixelWidth) canvas.width = pixelWidth;
		if (canvas.height !== pixelHeight) canvas.height = pixelHeight;
		context.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
	}

	function failSimulation(error: unknown): void {
		paused = true;
		phase = 'error';
		errorMessage = errorText(error);
		cpuWorker?.terminate();
		cpuWorker = null;
		workerBusy = false;
		simulation?.dispose();
		simulation = null;
	}

	function advanceOneCflBatch(): void {
		if (!simulation || !cpuWorker) {
			throw new Error('CPU MLS-MPM simulation is not initialized.');
		}
		postWorkerRequest({
			type: 'step',
			generation,
			maxMicrosteps: MPM_CPU_MAX_CFL_MICROSTEPS_PER_MANUAL_STEP,
			gravity: appliedGravity,
			oscillationAmplitude: horizontalOscillationAmplitude
		});
	}

	function applyCurrentGravity(): void {
		lastFrameTime = 0;
	}

	function applyLiquidParticleDiameter(): void {
		const url = new URL(window.location.href);
		url.searchParams.set(
			GATE4_LIQUID_PARTICLE_DIAMETER_QUERY_KEY,
			String(liquidParticleDiameter)
		);
		window.location.assign(url.href);
	}

	function renderFrame(now: number): void {
		if (!mounted) return;
		animationFrame = window.requestAnimationFrame(renderFrame);

		const previousFrameTime = lastFrameTime;
		lastFrameTime = now;
		const elapsedSeconds =
			previousFrameTime === 0 ? 0 : Math.max(0, now - previousFrameTime) / 1_000;

		if (simulation && phase === 'ready' && !paused && isVisible) {
			try {
				accumulatorSeconds += elapsedSeconds * simulationSpeed;
				if (!Number.isFinite(accumulatorSeconds)) {
					throw new RangeError('Accumulated CPU simulation time is outside the supported range.');
				}
				if (!workerBusy && accumulatorSeconds >= STEP_SECONDS) {
					const requestedSeconds = accumulatorSeconds;
					accumulatorSeconds = 0;
					postWorkerRequest({
						type: 'advance',
						generation,
						accumulatorSeconds: requestedSeconds,
						maxMicrosteps: MAX_CPU_MICROSTEPS_PER_BATCH,
						gravity: appliedGravity,
						oscillationAmplitude: horizontalOscillationAmplitude
					});
				}
			} catch (error) {
				failSimulation(error);
			}
		}

		if (isVisible && framePacer.shouldRender(now, renderTargetFps)) {
			const drawStartedAt = performance.now();
			if (canvas && context && simulation && phase === 'ready') {
				const bounds = canvas.getBoundingClientRect();
				try {
					simulation.draw(context, bounds.width, bounds.height, renderMode);
				} catch (error) {
					failSimulation(error);
				}
			} else if (canvas && context) {
				const bounds = canvas.getBoundingClientRect();
				context.clearRect(0, 0, bounds.width, bounds.height);
				context.fillStyle = '#101a28';
				context.fillRect(0, 0, bounds.width, bounds.height);
			}
			drawMsForMetrics += performance.now() - drawStartedAt;
			frameCount += 1;
		}

		if (lastMetricsTime === 0) lastMetricsTime = now;
		const metricsElapsedMs = now - lastMetricsTime;
		if (metricsElapsedMs >= METRICS_INTERVAL_MS) {
			fps = (frameCount * 1_000) / metricsElapsedMs;
			stepsPerSecond = (physicsStepsForMetrics * 1_000) / metricsElapsedMs;
			microstepsPerSecond =
				(physicsMicrostepsForMetrics * 1_000) / metricsElapsedMs;
			physicsMsPerMicrostep =
				physicsMicrostepsForMetrics > 0
					? physicsMsForMetrics / physicsMicrostepsForMetrics
					: 0;
			canvasDrawMsPerFrame =
				frameCount > 0 ? drawMsForMetrics / frameCount : 0;
			pendingSimulationMs = accumulatorSeconds * 1_000;
			frameCount = 0;
			physicsStepsForMetrics = 0;
			physicsMicrostepsForMetrics = 0;
			physicsMsForMetrics = 0;
			drawMsForMetrics = 0;
			lastMetricsTime = now;
		}
	}

	function togglePaused(): void {
		if (phase !== 'ready') return;
		paused = !paused;
		if (paused) pendingSimulationMs = accumulatorSeconds * 1_000;
		lastFrameTime = 0;
	}

	function stepOnce(): void {
		if (phase !== 'ready' || !paused || !simulation || workerBusy) return;
		try {
			advanceOneCflBatch();
		} catch (error) {
			failSimulation(error);
		}
	}

	function discardDeferredSimulationTime(): void {
		if (!paused || accumulatorSeconds <= 0) return;
		accumulatorSeconds = 0;
		pendingSimulationMs = 0;
	}

	function toggleHorizontalOscillation(): void {
		horizontalOscillationAmplitude =
			horizontalOscillationAmplitude > 0
				? 0
				: DEFAULT_HORIZONTAL_OSCILLATION_AMPLITUDE;
	}

	function refreshDeviceOrientationSupport(): void {
		const supported =
			typeof window.DeviceOrientationEvent !== 'undefined' && window.isSecureContext;
		if (!supported) {
			orientationStatus = 'unsupported';
			orientationMessage =
				'DeviceOrientation は非対応または安全でない接続です。手動の重力ベクトルを使用できます。';
			return;
		}
		orientationStatus = 'idle';
		orientationMessage =
			'手動ベクトルを使用中です。センサーはボタンをタップした場合のみ開始します。';
	}

	function handleDeviceOrientation(event: DeviceOrientationEvent): void {
		if (
			event.beta === null ||
			event.gamma === null ||
			!Number.isFinite(event.beta) ||
			!Number.isFinite(event.gamma)
		) {
			hasSensorGravity = false;
			orientationMessage = '傾きデータがありません。手動の重力ベクトルを使用します。';
			applyCurrentGravity();
			return;
		}
		sensorGravityUnit = projectDeviceOrientationGravity(
			event.beta,
			event.gamma,
			window.screen.orientation?.angle ?? 0
		);
		hasSensorGravity = true;
		orientationMessage =
			'DeviceOrientation を使用中です。重力ベクトルだけが変化し、画面は回転しません。';
		applyCurrentGravity();
	}

	async function toggleDeviceOrientation(): Promise<void> {
		if (orientationStatus === 'enabled') {
			window.removeEventListener('deviceorientation', handleDeviceOrientation);
			orientationStatus = 'idle';
			hasSensorGravity = false;
			orientationMessage = 'センサーを停止しました。手動の重力ベクトルを使用します。';
			applyCurrentGravity();
			return;
		}
		const constructor = window.DeviceOrientationEvent as
			| (typeof DeviceOrientationEvent & {
					requestPermission?: () => Promise<'granted' | 'denied'>;
			  })
			| undefined;
		if (!constructor || !window.isSecureContext) {
			orientationStatus = 'unsupported';
			orientationMessage =
				'DeviceOrientation は非対応または安全でない接続です。手動の重力ベクトルを使用できます。';
			return;
		}

		orientationStatus = 'requesting';
		orientationMessage = 'DeviceOrientation の許可を確認しています…';
		try {
			const permissionRequest = constructor.requestPermission
				? constructor.requestPermission.call(constructor)
				: Promise.resolve('granted' as const);
			const permission = await permissionRequest;
			if (permission !== 'granted') {
				orientationStatus = 'denied';
				hasSensorGravity = false;
				orientationMessage =
					'DeviceOrientation の許可が拒否されました。手動の重力ベクトルを使用できます。';
				return;
			}
			window.addEventListener('deviceorientation', handleDeviceOrientation);
			orientationStatus = 'enabled';
			hasSensorGravity = false;
			orientationMessage =
				'DeviceOrientation は許可済みでデータを待っています。データ到着までは手動ベクトルを使用します。';
		} catch {
			orientationStatus = 'denied';
			hasSensorGravity = false;
			orientationMessage =
				'DeviceOrientation の許可に失敗または拒否されました。手動の重力ベクトルを使用できます。';
		}
	}

	function handleVisibilityChange(): void {
		isVisible = document.visibilityState === 'visible';
		lastFrameTime = 0;
		framePacer.reset();
	}

	onMount(() => {
		mounted = true;
		isVisible = document.visibilityState === 'visible';
		refreshDeviceOrientationSupport();
		context = canvas?.getContext('2d') ?? null;
		if (!context) {
			phase = 'error';
			errorMessage = 'このブラウザーでは Canvas2D を利用できません。';
			return () => {
				mounted = false;
			};
		}

		const resizeObserver = new ResizeObserver(resizeCanvas);
		if (canvas) resizeObserver.observe(canvas);
		window.addEventListener('resize', resizeCanvas);
		document.addEventListener('visibilitychange', handleVisibilityChange);
		resizeCanvas();
		void initializeSimulation();
		animationFrame = window.requestAnimationFrame(renderFrame);

		return () => {
			mounted = false;
			generation += 1;
			window.cancelAnimationFrame(animationFrame);
			window.removeEventListener('resize', resizeCanvas);
			document.removeEventListener('visibilitychange', handleVisibilityChange);
			resizeObserver.disconnect();
			window.removeEventListener('deviceorientation', handleDeviceOrientation);
			cpuWorker?.terminate();
			cpuWorker = null;
			workerBusy = false;
			simulation?.dispose();
			simulation = null;
		};
	});
</script>

<svelte:head>
	<title>CPU MLS-MPM reference · Gate 4 comparison</title>
	<meta
		name="description"
		content="GPU MLS-MPM と同一の Gate 4 シーン・物理モデルを CPU の参照ソルバーで実行し、パフォーマンスと動作を比較するページ"
	/>
</svelte:head>

<main class="page">
	<header class="page-header">
		<p class="eyebrow">MLS-MPM · CPU/GPU solver comparison</p>
		<h1>CPU MLS-MPM reference</h1>
		<p class="intro">
			GPU ページと同じ seed・粒子配置・材料・固定刻み・CFL 条件を使い、同一の MLS-MPM 参照式を CPU で実行します。
			CPU solver は Web Worker 上で実行し、1 batch を最大 8 CFL microsteps に制限します。追いつけない時間は捨てずに backlog として保持します。
		</p>
		<nav class="solver-links" aria-label="Solver pages">
			<a
				href={`/labs/mpm/kernels?${GATE4_LIQUID_PARTICLE_DIAMETER_QUERY_KEY}=${appliedLiquidParticleDiameter}`}
			>
				GPU MLS-MPM page →
			</a>
			<a href="/labs/mpm">MPM lab</a>
		</nav>
	</header>

	<section class="simulation-layout" aria-label="CPU MLS-MPM reference simulation">
		<div class="stage">
			<div class="stage-heading">
				<div>
					<p class="eyebrow">CPU · shared MPM reference solver</p>
					<h2>Gate 4 fixed scene</h2>
				</div>
				<span class="engine-tag">JavaScript · CPU · f64</span>
			</div>

			<div class="canvas-frame">
				<canvas bind:this={canvas} aria-label="CPU MLS-MPM particle simulation">
					Canvas2D に対応したブラウザーでご利用ください。
				</canvas>
			</div>
			<p class="status" role="status" aria-live="polite">{statusText}</p>
			{#if errorMessage}
				<p class="error" role="alert">{errorMessage}</p>
			{/if}
		</div>

		<aside class="control-panel" aria-label="Simulation controls and metrics">
			<h2>Simulation controls</h2>
			<div class="controls">
				<button
					class="primary"
					type="button"
					disabled={phase !== 'ready'}
					aria-pressed={!paused}
					onclick={togglePaused}
				>
					{paused ? 'Resume' : 'Pause'}
				</button>
				<button
					type="button"
					disabled={phase !== 'ready' || !paused || workerBusy}
					onclick={stepOnce}
				>
					1 step
				</button>
				<button
					type="button"
					disabled={phase === 'loading'}
					onclick={initializeSimulation}
				>
					Reset
				</button>
			</div>

			<Gate4ParticleSizeControl
				bind:value={liquidParticleDiameter}
				appliedValue={appliedLiquidParticleDiameter}
				disabled={phase !== 'ready'}
				onApply={applyLiquidParticleDiameter}
			/>

			<fieldset class="gravity-controls">
				<legend>Gravity · Gate 4 units/s²</legend>
				<label class="range-control">
					<span>X</span>
					<output>{manualGravityX}</output>
					<input
						type="range"
						min="-180"
						max="180"
						step="5"
						bind:value={manualGravityX}
						oninput={applyCurrentGravity}
						disabled={orientationStatus === 'enabled' && hasSensorGravity}
					/>
				</label>
				<label class="range-control">
					<span>Y</span>
					<output>{manualGravityY}</output>
					<input
						type="range"
						min="-180"
						max="180"
						step="5"
						bind:value={manualGravityY}
						oninput={applyCurrentGravity}
						disabled={orientationStatus === 'enabled' && hasSensorGravity}
					/>
				</label>
				<label class="range-control">
					<span>Sensor strength</span>
					<output>{gravityMagnitude}</output>
					<input
						type="range"
						min="0"
						max="180"
						step="5"
						bind:value={gravityMagnitude}
						oninput={applyCurrentGravity}
						disabled={orientationStatus !== 'enabled' || !hasSensorGravity}
					/>
				</label>
				<button
					type="button"
					class="wide-button"
					disabled={orientationStatus === 'unsupported' || orientationStatus === 'requesting'}
					onclick={() => void toggleDeviceOrientation()}
				>
					{orientationStatus === 'enabled' ? 'Disable tilt sensor' : 'Enable tilt sensor'}
				</button>
				<p class="sensor-status" role="status">{orientationMessage}</p>
			</fieldset>

			<div class="setting-controls">
				<label>
					<span>Simulation speed</span>
					<select bind:value={simulationSpeed}>
						{#each SIMULATION_SPEED_OPTIONS as speed}
							<option value={speed}>{speed}×</option>
						{/each}
					</select>
				</label>
				<label>
					<span>Fluid display</span>
					<select bind:value={renderMode}>
						<option value="surface">Water/oil surface</option>
						<option value="particles">Particle markers</option>
					</select>
				</label>
				<label>
					<span>Render target</span>
					<select bind:value={renderTargetFps}>
						{#each MPM_RENDER_FPS_OPTIONS as targetFps}
							<option value={targetFps}>{targetFps} FPS</option>
						{/each}
					</select>
				</label>
			</div>

			<fieldset class="oscillation-controls">
				<legend>Horizontal oscillation</legend>
				<label class="range-control">
					<span>Acceleration</span>
					<output>{horizontalOscillationAmplitude}</output>
					<input
						type="range"
						min="0"
						max={MAX_HORIZONTAL_OSCILLATION_AMPLITUDE}
						step="10"
						bind:value={horizontalOscillationAmplitude}
					/>
				</label>
				<button type="button" class="wide-button" onclick={toggleHorizontalOscillation}>
					{horizontalOscillationAmplitude > 0
						? 'Stop oscillation'
						: `Start ±${DEFAULT_HORIZONTAL_OSCILLATION_AMPLITUDE} oscillation`}
				</button>
				<p class="control-note">Sinusoidal acceleration · {HORIZONTAL_OSCILLATION_FREQUENCY_HZ} Hz</p>
			</fieldset>

			<div class="backlog-controls">
				<p>Pending simulation time: {pendingSimulationMs.toFixed(1)} ms</p>
				<button
					type="button"
					class="wide-button"
					disabled={!paused || workerBusy || pendingSimulationMs <= 0}
					onclick={discardDeferredSimulationTime}
				>
					Discard pending time
				</button>
			</div>

			<h2>Live metrics</h2>
			<dl class="metrics">
				<div>
					<dt>描画 FPS</dt>
					<dd>{fps.toFixed(1)}</dd>
				</div>
				<div>
					<dt>Simulation CPU ms / CFL microstep</dt>
					<dd>{physicsMsPerMicrostep.toFixed(3)}</dd>
				</div>
				<div>
					<dt>Completed macrosteps / s</dt>
					<dd>{stepsPerSecond.toFixed(1)}</dd>
				</div>
				<div>
					<dt>CFL microsteps / s</dt>
					<dd>{microstepsPerSecond.toFixed(1)}</dd>
				</div>
				<div>
					<dt>Canvas2D draw ms / frame</dt>
					<dd>{canvasDrawMsPerFrame.toFixed(3)}</dd>
				</div>
				<div>
					<dt>未処理の固定刻み</dt>
					<dd>{pendingSimulationMs.toFixed(1)} ms</dd>
				</div>
				<div>
					<dt>MPM particles</dt>
					<dd>{simulation?.particleCount ?? '—'}</dd>
				</div>
				<div>
					<dt>Water / oil particles</dt>
					<dd>
						{simulation?.waterParticleCount ?? '—'} / {simulation?.oilParticleCount ?? '—'}
					</dd>
				</div>
				<div>
					<dt>Pebble vertices / deformable triangles</dt>
					<dd>{simulation?.pebbleNodeCount ?? '—'} / {simulation?.triangleBodyCount ?? '—'}</dd>
				</div>
				<div>
					<dt>Fracture nodes</dt>
					<dd>{simulation?.fractureNodeCount ?? '—'}</dd>
				</div>
				<div>
					<dt>Active fracture bonds</dt>
					<dd>
						{simulation?.activeFractureBondCount ?? '—'} / {simulation?.fractureBondCount ?? '—'}
					</dd>
				</div>
			</dl>

			<div class="configuration">
				<h2>Fixed workload</h2>
				<ul>
					<li>
						Seed <code>0x{GATE4_SCENE_SEED.toString(16)}</code>
					</li>
					<li>
						{GATE4_GRID.cellsX * GATE4_GRID.cellSize} × {GATE4_GRID.cellsY * GATE4_GRID.cellSize}
						world units · {GATE4_GRID.cellsX} × {GATE4_GRID.cellsY} grid · cell {GATE4_GRID.cellSize}
					</li>
					<li>
						dt = 1/{Math.round(1 / STEP_SECONDS)} s · default gravity = ({GATE4_DEFAULT_GRAVITY.x},
						{GATE4_DEFAULT_GRAVITY.y}) units/s²
					</li>
					<li>
						{GATE4_WATER_PARTICLE_COUNT.toLocaleString()} water + {GATE4_OIL_PARTICLE_COUNT.toLocaleString()}
						oil + {GATE4_PEBBLE_PARTICLE_COUNT} pebble vertices ({GATE4_PEBBLE_COUNT} deformable triangles)
						+ {GATE4_FRACTURE_PARTICLE_COUNT} fracture nodes / {GATE4_FRACTURE_BOND_COUNT} bonds
					</li>
					<li>
						Liquid reference diameter {appliedLiquidParticleDiameter} units · area
						{appliedLiquidParticleDiameter ** 2} units²
					</li>
					<li>
						Courant limit {DEFAULT_MPM_COURANT_LIMIT} · max {MAX_CPU_MICROSTEPS_PER_BATCH}
						CFL microsteps per worker batch; excess simulation time is retained
					</li>
				</ul>
			</div>
		</aside>
	</section>

	<section class="limitations" aria-labelledby="limits-heading">
		<h2 id="limits-heading">Comparison notes</h2>
		<ul>
			<li>
				CPU and GPU pages construct the same seeded Gate 4 particles, materials, fracture bonds, grid, gravity,
				fixed timestep, CFL limit, and oscillation. The CPU page runs the shared <code>stepMpm</code> reference
				equations instead of LiquidFun.
			</li>
			<li>
				The reference solver uses JavaScript number precision; WebGPU kernels use float32 and parallel
				accumulation. Expect small numerical drift rather than bit-for-bit identical trajectories.
			</li>
			<li>
				The CPU solver runs off the main thread and caps each worker batch at
				{MAX_CPU_MICROSTEPS_PER_BATCH} CFL microsteps. When it cannot meet the selected simulation speed, it
				retains the excess as pending simulation time instead of dropping it or taking an unbounded catch-up loop.
			</li>
			<li>
				CPU timing covers synchronous MLS-MPM stepping; Canvas2D drawing is reported separately. The GPU page
				reports GPU queue and kernel measurements, so compare completed macrosteps/s or CFL microsteps/s alongside
				per-step timing.
			</li>
			<li>
				Water/oil surface mode and particle-marker mode change only the drawing. The older standalone LiquidFun
				benchmark remains separate and is not used by either live comparison page.
			</li>
		</ul>
		<p class="docs-link">
			Comparison setup and interpretation are documented in
			<code>docs/MPM-CPU-GPU-COMPARISON.md</code>. The standalone LiquidFun benchmark remains at
			<code>docs/LIQUIDFUN-BENCHMARK.md</code>.
		</p>
	</section>
</main>

<style>
	:global(body) {
		margin: 0;
		background: #0b111b;
		color: #e7eef8;
		font-family:
			Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
	}

	.page {
		box-sizing: border-box;
		width: min(1012px, 100%);
		margin: 0 auto;
		padding: max(1rem, env(safe-area-inset-top)) max(1rem, env(safe-area-inset-right))
			max(2rem, env(safe-area-inset-bottom)) max(1rem, env(safe-area-inset-left));
	}

	.page-header {
		margin-bottom: 1.5rem;
	}

	.eyebrow {
		margin: 0 0 0.35rem;
		color: #7bd7f5;
		font-size: 0.75rem;
		font-weight: 800;
		letter-spacing: 0.12em;
		text-transform: uppercase;
	}

	h1,
	h2,
	p {
		margin-top: 0;
	}

	h1 {
		margin-bottom: 0.55rem;
		font-size: clamp(1.8rem, 4vw, 3rem);
		line-height: 1.1;
	}

	h2 {
		margin-bottom: 0.75rem;
		font-size: 1.05rem;
	}

	.intro {
		max-width: 74ch;
		margin-bottom: 0.8rem;
		color: #bdcada;
		line-height: 1.6;
	}

	.solver-links {
		display: flex;
		flex-wrap: wrap;
		gap: 0.8rem 1.2rem;
	}

	a {
		color: #8be0fa;
		text-underline-offset: 0.18em;
	}

	a:hover {
		color: #c4f2ff;
	}

	.simulation-layout {
		box-sizing: border-box;
		display: grid;
		grid-template-areas: 'canvas controls';
		grid-template-columns: minmax(0, 1fr) minmax(17rem, 20rem);
		gap: 1rem;
		align-items: start;
		padding: 1rem;
	}

	.simulation-layout,
	.limitations {
		border: 1px solid #26374d;
		border-radius: 14px;
		background: #121c2a;
	}

	.stage {
		grid-area: canvas;
		min-width: 0;
	}

	.stage-heading {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: space-between;
		gap: 0.6rem;
		margin-bottom: 0.8rem;
	}

	.stage-heading h2 {
		margin-bottom: 0;
	}

	.engine-tag {
		border: 1px solid #35536b;
		border-radius: 999px;
		padding: 0.3rem 0.65rem;
		color: #d0e7f2;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		font-size: 0.78rem;
	}

	.canvas-frame {
		overflow: hidden;
		width: 100%;
		aspect-ratio: 3 / 2;
		border: 1px solid #3b4d62;
		border-radius: 8px;
		background: #101a28;
	}

	canvas {
		display: block;
		width: 100%;
		height: 100%;
		touch-action: none;
	}

	.status {
		margin: 0.75rem 0 0;
		color: #b7c7d9;
		font-weight: 650;
	}

	.error {
		margin: 0.5rem 0 0;
		border-left: 3px solid #f17878;
		padding-left: 0.65rem;
		color: #ffb1b1;
		overflow-wrap: anywhere;
	}

	.control-panel {
		grid-area: controls;
		min-width: 0;
	}

	.controls {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 0.6rem;
		margin-bottom: 1rem;
	}

	button {
		box-sizing: border-box;
		min-height: 2.7rem;
		border: 1px solid #3b4d62;
		border-radius: 8px;
		padding: 0.55rem 0.8rem;
		background: #1a2839;
		color: #e7eef8;
		font: inherit;
		font-weight: 700;
		cursor: pointer;
	}

	select {
		box-sizing: border-box;
		width: 100%;
		min-height: 2.55rem;
		border: 1px solid #3b4d62;
		border-radius: 8px;
		padding: 0.45rem 0.65rem;
		background: #1a2839;
		color: #e7eef8;
		font: inherit;
	}

	button:hover:not(:disabled) {
		background: #263a50;
	}

	button:focus-visible,
	select:focus-visible,
	a:focus-visible {
		outline: 3px solid #7bd7f5;
		outline-offset: 3px;
	}

	button.primary {
		border-color: #3295b4;
		background: #14536a;
	}

	button:disabled {
		cursor: not-allowed;
		opacity: 0.55;
	}

	.gravity-controls,
	.oscillation-controls {
		display: grid;
		gap: 0.65rem;
		min-width: 0;
		margin: 0 0 0.9rem;
		border: 1px solid #31445a;
		border-radius: 0.55rem;
		padding: 0.75rem;
		background: #0d1823;
	}

	.gravity-controls legend,
	.oscillation-controls legend {
		padding: 0 0.35rem;
		font-weight: 650;
	}

	.range-control {
		display: grid;
		grid-template-columns: minmax(0, 1fr) auto;
		align-items: center;
		gap: 0.35rem 0.65rem;
		min-width: 0;
	}

	.range-control span {
		grid-column: 1;
		grid-row: 1;
	}

	.range-control output {
		grid-column: 2;
		grid-row: 1;
		font-variant-numeric: tabular-nums;
	}

	.range-control input {
		grid-column: 1 / -1;
		grid-row: 2;
		width: 100%;
		min-width: 0;
		margin: 0;
		accent-color: #55bddc;
	}

	.setting-controls {
		display: grid;
		gap: 0.7rem;
		margin: 0 0 0.9rem;
	}

	.setting-controls label {
		display: grid;
		gap: 0.35rem;
		color: #d9e5f2;
		font-size: 0.88rem;
	}

	.wide-button {
		width: 100%;
	}

	.sensor-status,
	.control-note,
	.backlog-controls p {
		margin: 0;
		color: #b8c7d8;
		font-size: 0.84rem;
		line-height: 1.5;
		overflow-wrap: anywhere;
	}

	.backlog-controls {
		display: grid;
		gap: 0.55rem;
		margin: 0 0 1rem;
		border: 1px solid #31445a;
		border-radius: 0.55rem;
		padding: 0.75rem;
		background: #0d1823;
	}

	.metrics {
		display: grid;
		gap: 0.15rem;
		margin: 0 0 1.25rem;
	}

	.metrics div {
		display: flex;
		justify-content: space-between;
		gap: 0.8rem;
		border-bottom: 1px solid #27384c;
		padding: 0.48rem 0;
	}

	dt {
		color: #b6c6d8;
		font-size: 0.82rem;
	}

	dd {
		margin: 0;
		color: #f1f6fc;
		font-variant-numeric: tabular-nums;
		font-weight: 700;
		text-align: right;
	}

	.configuration {
		border-top: 1px solid #304258;
		padding-top: 1rem;
	}

	.configuration ul,
	.limitations ul {
		margin: 0;
		padding-left: 1.2rem;
		color: #c4d0de;
		line-height: 1.55;
	}

	.configuration li + li,
	.limitations li + li {
		margin-top: 0.4rem;
	}

	code {
		color: #c5e6f1;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		font-size: 0.9em;
	}

	.limitations {
		margin-top: 1rem;
		padding: clamp(1rem, 2vw, 1.4rem);
	}

	.docs-link {
		margin: 1rem 0 0;
		color: #b9c8d8;
		line-height: 1.55;
		overflow-wrap: anywhere;
	}

	.docs-link {
		font-size: 0.9rem;
	}

	@media (max-width: 900px) {
		.simulation-layout {
			grid-template-areas:
				'controls'
				'canvas';
			grid-template-columns: minmax(0, 1fr);
		}
	}

	@media (max-width: 520px) {
		.page {
			padding: 1rem 0.75rem 1.5rem;
		}

		.canvas-frame {
			aspect-ratio: 1 / 1;
		}

		.simulation-layout {
			padding: 0.85rem;
		}
	}
</style>
