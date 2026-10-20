<script lang="ts">
	import { onMount } from 'svelte';
	import { projectDeviceOrientationGravity, STANDARD_GRAVITY } from '#lib/input/sensors.js';
	import {
		createMpmMlsGpuSolver,
		type MpmMlsGpuSolver
	} from '#lib/mpm/kernels/solver.js';
	import { calculateMpmCflStepBatch } from '#lib/mpm/kernels/step-scheduler.js';
	import type { MpmVector2 } from '#lib/mpm/reference/solver.js';
	import {
		createSiFluidScene,
		SI_FLUID_COURANT_LIMIT,
		SI_FLUID_EFFECTIVE_SOUND_SPEED_M_PER_S,
		SI_FLUID_FIXED_STEP_SECONDS,
		SI_FLUID_GRID,
		SI_FLUID_MAX_MICROSTEPS_PER_BATCH,
		SI_TAIT_WATER_PARAMETERS
	} from '#lib/mpm/scenes/si-fluid.js';
	import { MPM_SI_BOX_DOMAIN } from '#lib/mpm/si-domain.js';

	let canvas: HTMLCanvasElement;
	let solver: MpmMlsGpuSolver | null = null;
	let solverReady = $state(false);
	let paused = $state(false);
	let statusMessage = $state('WebGPU シミュレーションを準備しています。');
	let errorMessage = $state('');
	let particleCount = $state(0);
	let measuredFps = $state(0);
	let simulatedSeconds = $state(0);
	let lastBatchMicrosteps = $state(0);
	let lastBatchMilliseconds = $state(0);
	let orientationStatus = $state<'idle' | 'unsupported' | 'requesting' | 'enabled' | 'denied'>(
		'idle'
	);
	let orientationMessage = $state('手動の重力ベクトルを使用しています。');
	let hasSensorGravity = $state(false);
	let sensorGravityUnit = $state<MpmVector2>({ x: 0, y: 1 });
	let manualGravityX = $state(0);
	let manualGravityY = $state(1);

	function effectiveGravity(): MpmVector2 {
		const unit =
			orientationStatus === 'enabled' && hasSensorGravity
				? sensorGravityUnit
				: { x: manualGravityX, y: manualGravityY };
		const magnitude = Math.hypot(unit.x, unit.y);
		const scale = magnitude > 1 ? 1 / magnitude : 1;
		return {
			x: unit.x * scale * STANDARD_GRAVITY,
			y: unit.y * scale * STANDARD_GRAVITY
		};
	}

	let appliedGravity = $derived(effectiveGravity());

	function describeError(cause: unknown): string {
		return cause instanceof Error ? cause.message : String(cause);
	}

	function handleDeviceOrientation(event: DeviceOrientationEvent): void {
		if (
			event.beta === null ||
			event.gamma === null ||
			!Number.isFinite(event.beta) ||
			!Number.isFinite(event.gamma)
		) {
			hasSensorGravity = false;
			orientationMessage =
				'傾きデータがありません。手動の重力ベクトルに切り替えています。';
			return;
		}
		sensorGravityUnit = projectDeviceOrientationGravity(
			event.beta,
			event.gamma,
			window.screen.orientation?.angle ?? 0
		);
		hasSensorGravity = true;
		orientationMessage =
			'端末の傾きから重力を入力しています。シミュレーション画面は回転しません。';
	}

	async function toggleDeviceOrientation(): Promise<void> {
		if (orientationStatus === 'enabled') {
			window.removeEventListener('deviceorientation', handleDeviceOrientation);
			orientationStatus = 'idle';
			hasSensorGravity = false;
			orientationMessage = 'センサーを停止しました。手動の重力ベクトルを使用しています。';
			return;
		}

		const constructor = window.DeviceOrientationEvent as
			| (typeof DeviceOrientationEvent & {
					requestPermission?: () => Promise<'granted' | 'denied'>;
			  })
			| undefined;
		if (!constructor || !window.isSecureContext) {
			orientationStatus = 'unsupported';
			hasSensorGravity = false;
			orientationMessage =
				'端末センサーは非対応、または安全な接続ではありません。手動入力を使用できます。';
			return;
		}

		orientationStatus = 'requesting';
		try {
			const permission = constructor.requestPermission
				? await constructor.requestPermission.call(constructor)
				: 'granted';
			if (permission !== 'granted') {
				orientationStatus = 'denied';
				hasSensorGravity = false;
				orientationMessage =
					'端末センサーの許可がありません。手動の重力ベクトルを使用できます。';
				return;
			}
			window.addEventListener('deviceorientation', handleDeviceOrientation);
			orientationStatus = 'enabled';
			hasSensorGravity = false;
			orientationMessage =
				'端末センサーのデータを待っています。到着までは手動入力を使用します。';
		} catch (cause) {
			orientationStatus = 'denied';
			hasSensorGravity = false;
			orientationMessage = `端末センサーを有効化できませんでした: ${describeError(cause)}`;
		}
	}

	onMount(() => {
		let disposed = false;
		let frameHandle = 0;
		let lastFrameTimestamp: number | null = null;
		let renderWindowStartedAt = 0;
		let submittedRenderCount = 0;
		let simulationAccumulatorSeconds = 0;
		let latestVelocityBound = 0;
		let batchPending = false;

		const resizeCanvas = (): void => {
			const bounds = canvas.getBoundingClientRect();
			if (bounds.width <= 0 || bounds.height <= 0) return;
			const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
			const width = Math.max(1, Math.round(bounds.width * pixelRatio));
			const height = Math.max(1, Math.round(bounds.height * pixelRatio));
			if (canvas.width !== width) canvas.width = width;
			if (canvas.height !== height) canvas.height = height;
		};
		const resizeObserver = new ResizeObserver(resizeCanvas);
		resizeObserver.observe(canvas);
		window.addEventListener('resize', resizeCanvas);
		resizeCanvas();

		const submitRender = (
			activeSolver: MpmMlsGpuSolver,
			timestamp: number
		): void => {
			activeSolver.render('surface');
			if (renderWindowStartedAt === 0) renderWindowStartedAt = timestamp;
			submittedRenderCount += 1;
		};

		const fail = (cause: unknown): void => {
			paused = true;
			errorMessage = describeError(cause);
			statusMessage = 'シミュレーションを停止しました。';
		};

		const start = async (): Promise<void> => {
			statusMessage = 'WebGPU パイプラインを作成しています。';
			const scene = createSiFluidScene();
			particleCount = scene.particles.length;
			const createdSolver = await createMpmMlsGpuSolver(
				canvas,
				scene.particles,
				scene.config
			);
			if (disposed) {
				createdSolver.dispose();
				return;
			}
			if (createdSolver.cflBounds === null) {
				createdSolver.dispose();
				throw new Error('この材料から安全な CFL 上限を導出できませんでした。');
			}

			solver = createdSolver;
			solverReady = true;
			statusMessage = 'シミュレーションを実行中です。';
			submitRender(solver, performance.now());
			void solver.deviceLost.then((info) => {
				if (disposed || solver !== createdSolver || info.reason === 'destroyed') return;
				fail(new Error(`WebGPU device lost: ${info.message}`));
			});
		};

		const refreshDeviceOrientationSupport = (): void => {
			if (
				typeof window.DeviceOrientationEvent === 'undefined' ||
				!window.isSecureContext
			) {
				orientationStatus = 'unsupported';
				orientationMessage =
					'端末センサーは非対応、または安全な接続ではありません。手動入力を使用できます。';
			}
		};

		const renderFrame = (timestamp: number): void => {
			if (disposed) return;
			if (lastFrameTimestamp !== null && !paused && solver) {
				simulationAccumulatorSeconds += Math.max(
					0,
					(timestamp - lastFrameTimestamp) / 1000
				);
			}
			lastFrameTimestamp = timestamp;

			if (renderWindowStartedAt === 0) renderWindowStartedAt = timestamp;
			if (timestamp - renderWindowStartedAt >= 500) {
				measuredFps =
					(submittedRenderCount * 1000) / (timestamp - renderWindowStartedAt);
				submittedRenderCount = 0;
				renderWindowStartedAt = timestamp;
			}

			const activeSolver = solver;
			if (activeSolver && !errorMessage && !batchPending) {
				if (paused) {
					try {
						submitRender(activeSolver, timestamp);
					} catch (cause) {
						fail(cause);
					}
				} else {
					try {
						const bounds = activeSolver.cflBounds;
						if (bounds === null) {
							throw new Error('この材料から安全な CFL 上限を導出できませんでした。');
						}
						const gravity = effectiveGravity();
						const scheduled = calculateMpmCflStepBatch({
							accumulatorSeconds: simulationAccumulatorSeconds,
							elapsedSeconds: 0,
							simulationSpeed: 1,
							fixedStepSeconds: SI_FLUID_FIXED_STEP_SECONDS,
							cellSize: SI_FLUID_GRID.cellSize,
							waveSpeedBound: bounds.waveSpeedBound,
							currentVelocityBound: Math.max(
								bounds.velocityBound,
								latestVelocityBound
							),
							courantLimit: bounds.courantLimit ?? SI_FLUID_COURANT_LIMIT,
							maxMicrosteps: SI_FLUID_MAX_MICROSTEPS_PER_BATCH,
							allowPartialBatch: true,
							gravityMagnitude: Math.hypot(gravity.x, gravity.y),
							oscillationAmplitude: 0,
							dragRadius: 0,
							dragStiffness: 0,
							dragDamping: 0
						});
						const { batch, velocityBound } = scheduled;
						if (batch.macroStepCount > 0) {
							simulationAccumulatorSeconds = batch.remainderSeconds;
							batchPending = true;
							const startedAt = performance.now();
							void activeSolver
								.stepCflBatchAdaptive(
									batch.macroStepCount,
									batch.macroDt,
									batch.microstepPlan,
									velocityBound,
									SI_FLUID_MAX_MICROSTEPS_PER_BATCH,
									gravity
								)
								.then((measurement) => {
									if (disposed) return;
									simulatedSeconds += measurement.simulatedDuration;
									simulationAccumulatorSeconds += measurement.remainingDuration;
									if (
										measurement.remainingDuration > 0 &&
										measurement.simulatedDuration <= 0
									) {
										throw new Error(
											`CFL 回復が進行できず、${measurement.remainingDuration} 秒が未処理です。`
										);
									}
									latestVelocityBound = measurement.recommendedVelocityBound;
									lastBatchMicrosteps = measurement.attemptedMicroStepCount;
									lastBatchMilliseconds = performance.now() - startedAt;
								})
								.catch((cause: unknown) => {
									if (!disposed) fail(cause);
								})
								.finally(() => {
									batchPending = false;
									if (!disposed && solver === activeSolver && !errorMessage) {
										try {
											submitRender(activeSolver, performance.now());
										} catch (cause) {
											fail(cause);
										}
									}
								});
						} else {
							submitRender(activeSolver, timestamp);
						}
					} catch (cause) {
						fail(cause);
					}
				}
			}

			frameHandle = requestAnimationFrame(renderFrame);
		};

		refreshDeviceOrientationSupport();
		void start().catch(fail);
		frameHandle = requestAnimationFrame(renderFrame);

		return () => {
			disposed = true;
			cancelAnimationFrame(frameHandle);
			resizeObserver.disconnect();
			window.removeEventListener('resize', resizeCanvas);
			window.removeEventListener('deviceorientation', handleDeviceOrientation);
			solver?.dispose();
			solver = null;
			solverReady = false;
		};
	});
</script>

<svelte:head>
	<title>SI Tait 水槽 | powder-play</title>
	<meta
		name="description"
		content="20 cm のSI単位系水槽で、Tait状態方程式を使ったWebGPU液体シミュレーションを試す実験室。"
	/>
</svelte:head>

<main class="page">
	<header class="page-header">
		<div>
			<a class="back-link" href="/labs/mpm">← MPM labs</a>
			<h1>SI 液体シミュレーション</h1>
			<p>MLS-MPM / WebGPU · Tait 型状態方程式 · closed box</p>
		</div>
		<span class="experiment-badge">試作</span>
	</header>

	<div class="workspace">
		<section class="simulation-section" aria-labelledby="scene-title">
			<div class="section-heading">
				<div>
					<h2 id="scene-title">200 mm × 200 mm の内寸</h2>
					<p>画面全体が計算領域です。壁厚 50 mm は領域の外側に置く設計です。</p>
				</div>
				<span class="closed-badge">四辺閉鎖</span>
			</div>

			<div class="simulation-frame">
				<canvas
					bind:this={canvas}
					aria-label="200 mm角の閉じた水槽で動く液体のシミュレーション"
				>
					WebGPU 対応ブラウザーが必要です。
				</canvas>
			</div>

			<div class="metrics" aria-label="シミュレーション計測値">
				<div class="metric">
					<span>描画提出 / 秒</span>
					<strong>{measuredFps.toFixed(0)} <small>/ 120 目標</small></strong>
				</div>
				<div class="metric">
					<span>液体粒子</span>
					<strong>{particleCount.toLocaleString()}</strong>
				</div>
				<div class="metric">
					<span>最終バッチ</span>
					<strong>{lastBatchMicrosteps} <small>microsteps</small></strong>
				</div>
				<div class="metric">
					<span>シミュレーション時刻</span>
					<strong>{simulatedSeconds.toFixed(2)} <small>s</small></strong>
				</div>
			</div>
		</section>

		<aside class="control-panel" aria-label="シミュレーション設定">
			<div class="status-card" role="status" aria-live="polite">
				<span class:running={!paused && !errorMessage} class="status-dot"></span>
				<div>
					<strong>
						{paused && solverReady && !errorMessage
							? 'シミュレーションを一時停止しています。'
							: statusMessage}
					</strong>
					{#if lastBatchMilliseconds > 0}
						<small>直近バッチ {lastBatchMilliseconds.toFixed(1)} ms</small>
					{/if}
				</div>
			</div>

			{#if errorMessage}
				<p class="error-message" role="alert">{errorMessage}</p>
			{/if}

			<div class="button-row">
				<button
					class="primary-button"
					type="button"
					onclick={() => {
						paused = !paused;
					}}
					aria-pressed={paused}
					disabled={!solverReady || Boolean(errorMessage)}
				>
					{paused ? '再開' : '一時停止'}
				</button>
				<button class="secondary-button" type="button" onclick={() => window.location.reload()}>
					最初から
				</button>
			</div>

			<fieldset class="control-group gravity-controls">
				<legend>重力・端末の傾き</legend>
				<p class="control-description">
					手動ベクトルは合成して最大 1 g に制限します。現在値:
					x {appliedGravity.x.toFixed(2)}、y {appliedGravity.y.toFixed(2)} m/s²
					（y 正方向は画面下）
				</p>
				<label for="gravity-x">
					左右方向 <span>{manualGravityX.toFixed(2)} g</span>
				</label>
				<input
					id="gravity-x"
					type="range"
					min="-1"
					max="1"
					step="0.01"
					bind:value={manualGravityX}
				/>
				<label for="gravity-y">
					上下方向 <span>{manualGravityY.toFixed(2)} g</span>
				</label>
				<input
					id="gravity-y"
					type="range"
					min="-1"
					max="1"
					step="0.01"
					bind:value={manualGravityY}
				/>
				<button
					class="secondary-button full-width"
					type="button"
					onclick={toggleDeviceOrientation}
					disabled={orientationStatus === 'unsupported' || orientationStatus === 'requesting'}
				>
					{orientationStatus === 'enabled' ? '端末センサーを停止' : '端末の傾きを使う'}
				</button>
				<p class="sensor-message">{orientationMessage}</p>
			</fieldset>

			<fieldset class="control-group">
				<legend>初期物性と数値近似</legend>
				<dl class="property-list">
					<div>
						<dt>面外厚さ d</dt>
						<dd>{(MPM_SI_BOX_DOMAIN.extrusionDepthM * 1000).toFixed(0)} mm</dd>
					</div>
					<div>
						<dt>基準密度 ρ₀</dt>
						<dd>{SI_TAIT_WATER_PARAMETERS.referenceDensityKgPerM3} kg/m³</dd>
					</div>
					<div>
						<dt>Tait 指数 n</dt>
						<dd>{SI_TAIT_WATER_PARAMETERS.exponent}</dd>
					</div>
					<div>
						<dt>計算用有効音速</dt>
						<dd>{SI_FLUID_EFFECTIVE_SOUND_SPEED_M_PER_S} m/s</dd>
					</div>
					<div>
						<dt>対応する係数 B</dt>
						<dd>{(SI_TAIT_WATER_PARAMETERS.pressureCoefficientPa / 1000).toFixed(1)} kPa</dd>
					</div>
					<div>
						<dt>壁の接線摩擦</dt>
						<dd>no-slip 近似</dd>
					</div>
				</dl>
				<p class="caution">
					20 m/s は計算量を抑えるための暫定的な数値近似で、水の実音速ではありません。
					圧縮率と圧力波の伝播を変えるため、物理値と区別して評価します。
				</p>
			</fieldset>

			<p class="scope-note">
				この段階は水 1 種の重力応答と Tait 圧力が対象です。上部空気の圧力連成、気泡、
				油・剛体、粘度、熱・相変化、開放端、ポインター撹拌はまだ接続していません。
			</p>
		</aside>
	</div>
</main>

<style>
	.page,
	.page :global(*) {
		box-sizing: border-box;
	}

	.page {
		background: #0b1220;
		color: #e5edf7;
		font-family:
			Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
			sans-serif;
	}

	.page {
		min-height: 100svh;
		padding: clamp(1rem, 3vw, 2.5rem);
	}

	.page-header,
	.section-heading {
		display: flex;
		align-items: flex-start;
		justify-content: space-between;
		gap: 1rem;
	}

	.page-header {
		max-width: 1440px;
		margin: 0 auto 1.5rem;
	}

	.page-header h1 {
		margin: 0.35rem 0 0.25rem;
		font-size: clamp(1.5rem, 3vw, 2.15rem);
		letter-spacing: -0.035em;
	}

	.page-header p,
	.section-heading p {
		margin: 0;
		color: #98a9bd;
	}

	.back-link {
		color: #78baff;
		font-size: 0.9rem;
		text-decoration: none;
	}

	.back-link:hover {
		text-decoration: underline;
	}

	.experiment-badge,
	.closed-badge {
		display: inline-flex;
		align-items: center;
		border: 1px solid #36506c;
		border-radius: 999px;
		color: #a9d4ff;
		font-size: 0.78rem;
		font-weight: 700;
		padding: 0.35rem 0.7rem;
		white-space: nowrap;
	}

	.workspace {
		display: grid;
		grid-template-columns: minmax(0, 1fr) minmax(300px, 360px);
		gap: clamp(1rem, 2.2vw, 2rem);
		max-width: 1440px;
		margin: 0 auto;
		align-items: start;
	}

	.simulation-section,
	.control-panel {
		min-width: 0;
	}

	.section-heading {
		align-items: center;
		margin-bottom: 0.8rem;
	}

	.section-heading h2 {
		margin: 0 0 0.25rem;
		font-size: 1.05rem;
	}

	.section-heading p {
		font-size: 0.82rem;
		line-height: 1.5;
	}

	.simulation-frame {
		width: min(100%, 76vh);
		aspect-ratio: 1;
		margin: 0 auto;
		overflow: hidden;
		border: 1px solid #52657b;
		border-radius: 0.75rem;
		background: #0d1a27;
		box-shadow: 0 18px 50px #0005;
	}

	canvas {
		display: block;
		width: 100%;
		height: 100%;
		touch-action: none;
	}

	.metrics {
		display: grid;
		grid-template-columns: repeat(4, minmax(0, 1fr));
		gap: 0.65rem;
		margin-top: 0.8rem;
	}

	.metric {
		display: grid;
		gap: 0.25rem;
		min-width: 0;
		padding: 0.7rem 0.8rem;
		border: 1px solid #233449;
		border-radius: 0.65rem;
		background: #101b2b;
	}

	.metric span,
	.metric small {
		color: #93a5ba;
		font-size: 0.72rem;
	}

	.metric strong {
		font-variant-numeric: tabular-nums;
		font-size: 1rem;
		overflow-wrap: anywhere;
	}

	.control-panel {
		display: grid;
		gap: 0.9rem;
	}

	.status-card {
		display: flex;
		align-items: center;
		gap: 0.7rem;
		min-height: 3.5rem;
		padding: 0.8rem 0.9rem;
		border: 1px solid #283b50;
		border-radius: 0.75rem;
		background: #101b2b;
	}

	.status-card strong,
	.status-card small {
		display: block;
	}

	.status-card strong {
		font-size: 0.88rem;
	}

	.status-card small {
		margin-top: 0.2rem;
		color: #93a5ba;
		font-size: 0.75rem;
	}

	.status-dot {
		width: 0.65rem;
		height: 0.65rem;
		flex: 0 0 auto;
		border-radius: 50%;
		background: #718096;
	}

	.status-dot.running {
		background: #43d39e;
		box-shadow: 0 0 0.65rem #43d39e88;
	}

	.button-row {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 0.65rem;
	}

	button {
		min-height: 2.7rem;
		border-radius: 0.55rem;
		font: inherit;
		font-size: 0.88rem;
		font-weight: 700;
		cursor: pointer;
	}

	button:focus-visible,
	input:focus-visible,
	a:focus-visible {
		outline: 3px solid #81c5ff;
		outline-offset: 2px;
	}

	button:disabled {
		cursor: not-allowed;
		opacity: 0.55;
	}

	.primary-button {
		border: 1px solid #3f9ef4;
		background: #1675c8;
		color: white;
	}

	.primary-button:hover:not(:disabled) {
		background: #2388dc;
	}

	.secondary-button {
		border: 1px solid #3a4f66;
		background: #162438;
		color: #d9e6f5;
	}

	.secondary-button:hover:not(:disabled) {
		background: #20344c;
	}

	.control-group {
		display: grid;
		gap: 0.6rem;
		margin: 0;
		padding: 0.9rem;
		border: 1px solid #283b50;
		border-radius: 0.75rem;
		background: #101b2b;
	}

	.control-group legend {
		padding: 0 0.3rem;
		font-size: 0.9rem;
		font-weight: 750;
	}

	.control-description,
	.sensor-message {
		margin: 0;
		color: #9aadc2;
		font-size: 0.78rem;
		line-height: 1.5;
	}

	label {
		display: flex;
		justify-content: space-between;
		gap: 0.75rem;
		font-size: 0.82rem;
	}

	label span {
		color: #a9d4ff;
		font-variant-numeric: tabular-nums;
	}

	input[type="range"] {
		width: 100%;
		margin: 0 0 0.35rem;
		accent-color: #55adf5;
	}

	.full-width {
		width: 100%;
	}

	.property-list {
		display: grid;
		gap: 0.4rem;
		margin: 0;
	}

	.property-list div {
		display: flex;
		justify-content: space-between;
		gap: 0.8rem;
		font-size: 0.8rem;
	}

	.property-list dt {
		color: #9aadc2;
	}

	.property-list dd {
		margin: 0;
		font-variant-numeric: tabular-nums;
		text-align: right;
	}

	.caution,
	.scope-note,
	.error-message {
		margin: 0;
		font-size: 0.78rem;
		line-height: 1.55;
	}

	.caution {
		padding: 0.65rem;
		border-left: 3px solid #e6a746;
		background: #2a2418;
		color: #f1d6a6;
	}

	.scope-note {
		color: #9aadc2;
	}

	.error-message {
		padding: 0.75rem;
		border: 1px solid #8d3d49;
		border-radius: 0.6rem;
		background: #351c27;
		color: #ffd5da;
		overflow-wrap: anywhere;
	}

	@media (max-width: 900px) {
		.workspace {
			grid-template-columns: minmax(0, 1fr);
		}

		.simulation-frame {
			width: min(100%, 70vh, 720px);
		}

		.control-panel {
			grid-template-columns: repeat(2, minmax(0, 1fr));
		}

		.status-card,
		.button-row,
		.gravity-controls,
		.scope-note {
			grid-column: 1 / -1;
		}
	}

	@media (max-width: 560px) {
		.page {
			padding: 1rem 0.75rem 1.5rem;
		}

		.page-header {
			margin-bottom: 1rem;
		}

		.page-header p {
			max-width: 18rem;
			font-size: 0.8rem;
		}

		.section-heading {
			align-items: flex-start;
		}

		.section-heading p {
			max-width: 18rem;
		}

		.metrics {
			grid-template-columns: repeat(2, minmax(0, 1fr));
		}

		.control-panel {
			grid-template-columns: minmax(0, 1fr);
		}

		.status-card,
		.button-row,
		.gravity-controls,
		.scope-note {
			grid-column: auto;
		}
	}
</style>
