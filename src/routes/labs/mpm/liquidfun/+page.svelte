<script lang="ts">
	import { onMount } from 'svelte';
	import { projectDeviceOrientationGravity } from '#lib/input/sensors.js';
	import {
		MPM_RENDER_FPS_OPTIONS,
		MpmFramePacer,
		type MpmRenderFps
	} from '#lib/mpm/kernels/frame-pacer.js';
	import {
		DEFAULT_HORIZONTAL_OSCILLATION_AMPLITUDE,
		HORIZONTAL_OSCILLATION_FREQUENCY_HZ,
		INITIAL_HORIZONTAL_OSCILLATION_AMPLITUDE,
		MAX_HORIZONTAL_OSCILLATION_AMPLITUDE,
		meanSinusoidalAcceleration
	} from '#lib/mpm/kernels/oscillation.js';
	import type { MpmVector2 } from '#lib/mpm/reference/solver.js';
	import {
		createLiquidFunSimulation,
		type LiquidFunRenderMode,
		type LiquidFunSimulation
	} from '#lib/mpm/liquidfun/simulation.js';
	import {
		LIQUIDFUN_BOX2D_DOMAIN_METERS,
		LIQUIDFUN_DT_SECONDS,
		LIQUIDFUN_GATE4_DOMAIN,
		LIQUIDFUN_GATE4_UNITS_PER_METER,
		LIQUIDFUN_GRAVITY_WARMUP_STEPS,
		LIQUIDFUN_PACKAGE_VERSION,
		LIQUIDFUN_SAMPLE_BATCHES,
		LIQUIDFUN_SCENE_SEED,
		LIQUIDFUN_STATIONARY_WARMUP_MS,
		LIQUIDFUN_STEPS_PER_BATCH
	} from '#lib/mpm/liquidfun/workload.js';

	type Phase = 'loading' | 'ready' | 'error';
	type OrientationStatus =
		| 'checking'
		| 'idle'
		| 'requesting'
		| 'enabled'
		| 'denied'
		| 'unsupported';

	const MAX_STEPS_PER_FRAME = 32;
	const METRICS_INTERVAL_MS = 500;
	const MAX_SIMULATION_SPEED = 32;
	const SIMULATION_SPEED_OPTIONS = [0.5, 1, 2, 4, 8, 16, MAX_SIMULATION_SPEED] as const;
	const DEFAULT_SIMULATION_SPEED = 2;
	const DEFAULT_RENDER_FPS: MpmRenderFps = 60;
	const STEP_SECONDS = LIQUIDFUN_DT_SECONDS;
	const framePacer = new MpmFramePacer();

	let canvas: HTMLCanvasElement | undefined;
	let context: CanvasRenderingContext2D | null = null;
	let simulation = $state<LiquidFunSimulation | null>(null);
	let phase = $state<Phase>('loading');
	let paused = $state(false);
	let isVisible = $state(true);
	let errorMessage = $state('');
	let fps = $state(0);
	let physicsMsPerStep = $state(0);
	let canvasDrawMsPerFrame = $state(0);
	let stepsPerSecond = $state(0);
	let pendingSimulationMs = $state(0);
	let gravityMagnitude = $state(90);
	let manualGravityX = $state(0);
	let manualGravityY = $state(90);
	let orientationStatus = $state<OrientationStatus>('checking');
	let orientationMessage = $state('DeviceOrientation センサーを確認しています…');
	let sensorGravityUnit = $state<MpmVector2>({ x: 0, y: 0 });
	let hasSensorGravity = $state(false);
	let simulationSpeed = $state(DEFAULT_SIMULATION_SPEED);
	let renderTargetFps: MpmRenderFps = $state(DEFAULT_RENDER_FPS);
	let renderMode = $state<LiquidFunRenderMode>('surface');
	let horizontalOscillationAmplitude = $state(
		INITIAL_HORIZONTAL_OSCILLATION_AMPLITUDE
	);

	let mounted = false;
	let generation = 0;
	let animationFrame = 0;
	let lastFrameTime = 0;
	let lastMetricsTime = 0;
	let accumulatorSeconds = 0;
	let simulatedTimeSeconds = 0;
	let frameCount = 0;
	let physicsStepsForMetrics = 0;
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
			? 'LiquidFun WebAssembly を読み込み中…'
			: phase === 'error'
				? 'シミュレーションを初期化できません'
				: !isVisible
					? 'ページ非表示中 · 物理ステップを停止中'
					: paused
						? '一時停止中 · 固定刻み 1/240 秒'
						: `実行中 · ${simulationSpeed}× · 固定刻み 1/240 秒`
	);

	function errorText(error: unknown): string {
		return error instanceof Error && error.message.trim()
			? error.message
			: 'LiquidFun の処理中に詳細不明のエラーが発生しました。';
	}

	function resetMetrics(): void {
		fps = 0;
		physicsMsPerStep = 0;
		canvasDrawMsPerFrame = 0;
		stepsPerSecond = 0;
		pendingSimulationMs = 0;
		frameCount = 0;
		physicsStepsForMetrics = 0;
		physicsMsForMetrics = 0;
		drawMsForMetrics = 0;
		accumulatorSeconds = 0;
		simulatedTimeSeconds = 0;
		lastFrameTime = 0;
		lastMetricsTime = 0;
		framePacer.reset();
	}

	async function initializeSimulation(): Promise<void> {
		const currentGeneration = ++generation;
		paused = false;
		phase = 'loading';
		errorMessage = '';
		resetMetrics();
		simulation?.dispose();
		simulation = null;

		try {
			const created = await createLiquidFunSimulation();
			if (!mounted || currentGeneration !== generation) {
				created.dispose();
				return;
			}
			simulation = created;
			simulation.setGravityFromGate4(effectiveGravity());
			phase = 'ready';
		} catch (error) {
			if (!mounted || currentGeneration !== generation) return;
			simulation?.dispose();
			simulation = null;
			phase = 'error';
			errorMessage = errorText(error);
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
		simulation?.dispose();
		simulation = null;
	}

	function advanceOneStep(): void {
		if (!simulation) {
			throw new Error('LiquidFun simulation is not initialized.');
		}
		const oscillation = meanSinusoidalAcceleration(
			simulatedTimeSeconds,
			STEP_SECONDS,
			horizontalOscillationAmplitude
		);
		simulation.setGravityFromGate4({
			x: appliedGravity.x + oscillation,
			y: appliedGravity.y
		});
		simulation.step();
		simulatedTimeSeconds += STEP_SECONDS;
	}

	function applyCurrentGravity(): void {
		if (!simulation || phase !== 'ready') return;
		try {
			simulation.setGravityFromGate4(effectiveGravity());
		} catch (error) {
			failSimulation(error);
		}
	}

	function renderFrame(now: number): void {
		if (!mounted) return;
		animationFrame = window.requestAnimationFrame(renderFrame);

		const previousFrameTime = lastFrameTime;
		lastFrameTime = now;
		const elapsedSeconds =
			previousFrameTime === 0 ? 0 : Math.max(0, now - previousFrameTime) / 1_000;
		let stepsThisFrame = 0;
		let physicsMillisecondsThisFrame = 0;

		if (simulation && phase === 'ready' && !paused && isVisible) {
			accumulatorSeconds += elapsedSeconds * simulationSpeed;
			try {
				while (
					accumulatorSeconds >= STEP_SECONDS &&
					stepsThisFrame < MAX_STEPS_PER_FRAME
				) {
					const startedAt = performance.now();
					advanceOneStep();
					physicsMillisecondsThisFrame += performance.now() - startedAt;
					accumulatorSeconds -= STEP_SECONDS;
					stepsThisFrame += 1;
				}
			} catch (error) {
				failSimulation(error);
			}
		}

		physicsStepsForMetrics += stepsThisFrame;
		physicsMsForMetrics += physicsMillisecondsThisFrame;

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
			physicsMsPerStep =
				physicsStepsForMetrics > 0
					? physicsMsForMetrics / physicsStepsForMetrics
					: 0;
			canvasDrawMsPerFrame =
				frameCount > 0 ? drawMsForMetrics / frameCount : 0;
			pendingSimulationMs = accumulatorSeconds * 1_000;
			frameCount = 0;
			physicsStepsForMetrics = 0;
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
		if (phase !== 'ready' || !paused || !simulation) return;
		const startedAt = performance.now();
		try {
			advanceOneStep();
			physicsStepsForMetrics += 1;
			physicsMsForMetrics += performance.now() - startedAt;
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
			simulation?.dispose();
			simulation = null;
		};
	});
</script>

<svelte:head>
	<title>LiquidFun CPU · Gate 4 MPM comparison</title>
	<meta
		name="description"
		content="Gate 4 初期配置の流体粒子と剛体三角形を LiquidFun CPU で動かし、MLS-MPM GPU 版と条件を比較するページ"
	/>
</svelte:head>

<main class="page">
	<header class="page-header">
		<p class="eyebrow">MLS-MPM · solver comparison</p>
		<h1>LiquidFun CPU comparison</h1>
		<p class="intro">
			GPU MLS-MPM ページと同じ Gate 4 の初期配置を、LiquidFun の CPU solver で実行します。
			主要な重力・速度・描画・一時停止の操作を揃えました。LiquidFun の三角形は剛体近似であり、変形する MLS-MPM と物理的に同一ではありません。
		</p>
		<nav class="solver-links" aria-label="Solver pages">
			<a href="/labs/mpm/kernels">GPU MLS-MPM page →</a>
			<a href="/labs/mpm">MPM lab</a>
		</nav>
	</header>

	<section class="simulation-layout" aria-label="LiquidFun CPU simulation">
		<div class="stage">
			<div class="stage-heading">
				<div>
					<p class="eyebrow">CPU · LiquidFun particle system</p>
					<h2>Gate 4 fixed scene</h2>
				</div>
				<span class="engine-tag">liquidfun-wasm@{LIQUIDFUN_PACKAGE_VERSION}</span>
			</div>

			<div class="canvas-frame">
				<canvas bind:this={canvas} aria-label="LiquidFun CPU particle simulation">
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
					disabled={phase !== 'ready' || !paused}
					onclick={stepOnce}
				>
					1 step
				</button>
				<button
					type="button"
					disabled={phase === 'loading'}
					onclick={() => void initializeSimulation()}
				>
					Reset
				</button>
			</div>

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
					disabled={!paused || pendingSimulationMs <= 0}
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
					<dt>Simulation CPU ms / step</dt>
					<dd>{physicsMsPerStep.toFixed(3)}</dd>
				</div>
				<div>
					<dt>Simulation steps / s</dt>
					<dd>{stepsPerSecond.toFixed(1)}</dd>
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
					<dt>LiquidFun particles</dt>
					<dd>{simulation?.particleCount ?? '—'}</dd>
				</div>
				<div>
					<dt>Mapped / source MPM nodes</dt>
					<dd>
						{simulation?.mappedMpmNodeCount ?? '—'} / {simulation?.sourceMpmNodeCount ?? '—'}
					</dd>
				</div>
				<div>
					<dt>Pebble vertices / rigid triangles</dt>
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
						Seed <code>0x{LIQUIDFUN_SCENE_SEED.toString(16)}</code>
					</li>
					<li>
						{LIQUIDFUN_GATE4_DOMAIN.width} × {LIQUIDFUN_GATE4_DOMAIN.height} Gate 4 units
						({LIQUIDFUN_BOX2D_DOMAIN_METERS.width} × {LIQUIDFUN_BOX2D_DOMAIN_METERS.height}
						m)
					</li>
					<li>dt = 1/{Math.round(1 / LIQUIDFUN_DT_SECONDS)} s · default gravity = 90 units/s² down</li>
					<li>3,500 fluid particles + 50 polygon triangles + 24 fracture-node bodies · 53 breakable bonds</li>
					<li>
						Benchmark: {LIQUIDFUN_STATIONARY_WARMUP_MS} ms stationary warmup + {LIQUIDFUN_GRAVITY_WARMUP_STEPS}
						gravity steps + {LIQUIDFUN_SAMPLE_BATCHES} × {LIQUIDFUN_STEPS_PER_BATCH} steps
					</li>
				</ul>
			</div>
		</aside>
	</section>

	<section class="limitations" aria-labelledby="limits-heading">
		<h2 id="limits-heading">Physics and measurement limits</h2>
		<ul>
			<li>
				Water uses the default <code>b2_waterParticle</code> flag and oil uses
				<code>b2_viscousParticle</code> in one LiquidFun particle system. That system shares particle density;
				the MPM water/oil densities (0.02 / 0.016) cannot be represented independently here.
			</li>
			<li>
				All 3,674 seeded MPM nodes are mapped: 3,500 LiquidFun fluid particles, 150 pebble vertices grouped
				into 50 dynamic polygon bodies, and 24 fracture-node bodies. The polygon fixtures now cover the full
				rendered triangle, so triangle-to-triangle collision uses the visible shape. The fracture network keeps
				its 53 breakable distance joints and Gate 4 12% tensile-strain threshold.
			</li>
			<li>
				Each triangle keeps its source vertices and total source-node mass. Its center and velocity are projected
				to one rigid body, so the shape collides as a polygon but cannot deform like the MLS-MPM Neo-Hookean
				triangles. The fracture nodes remain circle bodies linked by Box2D distance joints.
			</li>
			<li>
				Surface mode enlarges the drawn liquid particles to make water and oil read as continuous regions;
				particle-marker mode shows smaller diagnostic markers. This is a rendering choice only. Both phases
				still share one LiquidFun particle-system density, unlike the MPM phase densities 0.02 and 0.016.
			</li>
			<li>
				Live timing wraps synchronous CPU simulation steps, including <code>world.Step()</code> and fracture-bond
				checks; Canvas2D drawing has a separate metric. The benchmark runs without drawing and reports neither GPU
				timestamps nor queue waits.
			</li>
		</ul>
		<p class="attribution">
			<code>liquidfun-wasm@{LIQUIDFUN_PACKAGE_VERSION}</code> is the LiquidFun-derived,
			Box2D 2.4.1-based WebAssembly fork from Birch-san/box2d-wasm. Package license:
			<a href="/licenses/liquidfun-wasm/LICENSE.zlib.txt">Zlib License</a>;
			WASM feature-detection license:
			<a href="/licenses/liquidfun-wasm/LICENSE.wasm-feature-detect.Apache.txt">Apache License 2.0</a>.
			Copied license texts are served with this app.
		</p>
		<p class="docs-link">
			Benchmark setup and interpretation are documented in
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

	.attribution,
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
