<script lang="ts">
	import { onMount } from 'svelte';
	import {
		createLiquidFunSimulation,
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

	const MAX_STEPS_PER_FRAME = 32;
	const METRICS_INTERVAL_MS = 500;
	const STEP_SECONDS = LIQUIDFUN_DT_SECONDS;

	let canvas: HTMLCanvasElement | undefined;
	let context: CanvasRenderingContext2D | null = null;
	let simulation = $state<LiquidFunSimulation | null>(null);
	let phase = $state<Phase>('loading');
	let isRunning = $state(false);
	let isVisible = $state(true);
	let errorMessage = $state('');
	let fps = $state(0);
	let physicsMsPerStep = $state(0);
	let canvasDrawMsPerFrame = $state(0);
	let stepsPerSecond = $state(0);
	let pendingSimulationMs = $state(0);

	let mounted = false;
	let generation = 0;
	let animationFrame = 0;
	let lastFrameTime = 0;
	let lastMetricsTime = 0;
	let accumulatorSeconds = 0;
	let frameCount = 0;
	let physicsStepsForMetrics = 0;
	let physicsMsForMetrics = 0;
	let drawMsForMetrics = 0;

	const statusText = $derived(
		phase === 'loading'
			? 'LiquidFun WebAssembly を読み込み中…'
			: phase === 'error'
				? 'シミュレーションを初期化できません'
				: !isVisible
					? 'ページ非表示中 · 物理ステップを停止中'
					: isRunning
						? '実行中 · 固定刻み 1/240 秒'
						: '一時停止中'
	);

	function errorText(error: unknown): string {
		return error instanceof Error && error.message.trim()
			? error.message
			: 'LiquidFun の初期化中に詳細不明のエラーが発生しました。';
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
		lastFrameTime = 0;
		lastMetricsTime = 0;
	}

	async function initializeSimulation(): Promise<void> {
		const currentGeneration = ++generation;
		isRunning = false;
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
			phase = 'ready';
		} catch (error) {
			if (!mounted || currentGeneration !== generation) return;
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
		isRunning = false;
		phase = 'error';
		errorMessage = errorText(error);
		simulation?.dispose();
		simulation = null;
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

		if (simulation && phase === 'ready' && isRunning && isVisible) {
			accumulatorSeconds += elapsedSeconds;
			try {
				while (
					accumulatorSeconds >= STEP_SECONDS &&
					stepsThisFrame < MAX_STEPS_PER_FRAME
				) {
					const startedAt = performance.now();
					simulation.step();
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

		const drawStartedAt = performance.now();
		if (canvas && context && simulation && phase === 'ready') {
			const bounds = canvas.getBoundingClientRect();
			try {
				simulation.draw(context, bounds.width, bounds.height);
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

	function toggleRunning(): void {
		if (phase !== 'ready') return;
		isRunning = !isRunning;
		lastFrameTime = 0;
	}

	function handleVisibilityChange(): void {
		isVisible = document.visibilityState === 'visible';
		lastFrameTime = 0;
	}

	onMount(() => {
		mounted = true;
		isVisible = document.visibilityState === 'visible';
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
			simulation?.dispose();
			simulation = null;
		};
	});
</script>

<svelte:head>
	<title>LiquidFun CPU · Gate 4 MPM comparison</title>
	<meta
		name="description"
		content="LiquidFun particle API を使った CPU-only の Gate 4 類似シミュレーションと再現可能な計測"
	/>
</svelte:head>

<main class="page">
	<header class="page-header">
		<p class="eyebrow">MLS-MPM · solver comparison</p>
		<h1>LiquidFun CPU comparison</h1>
		<p class="intro">
			LiquidFun particle API による CPU-only の類似シミュレーションです。GPU
			MLS-MPMと物理的に同等な実装ではなく、固定した Gate 4 入力で実行コストを比べるための別 solver です。
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
			<div class="controls">
				<button
					class="primary"
					type="button"
					disabled={phase !== 'ready'}
					aria-pressed={isRunning}
					onclick={toggleRunning}
				>
					{isRunning ? 'Pause' : 'Start'}
				</button>
				<button
					type="button"
					disabled={phase === 'loading'}
					onclick={() => void initializeSimulation()}
				>
					Reset
				</button>
			</div>

			<h2>Live metrics</h2>
			<dl class="metrics">
				<div>
					<dt>描画 FPS</dt>
					<dd>{fps.toFixed(1)}</dd>
				</div>
				<div>
					<dt>world.Step CPU ms / step</dt>
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
					<dt>Dynamic triangle bodies</dt>
					<dd>{simulation?.triangleBodyCount ?? '—'}</dd>
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
					<li>dt = 1/{Math.round(1 / LIQUIDFUN_DT_SECONDS)} s · gravity = 90 units/s² down</li>
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
				The source scene has 3,674 MPM nodes. This mapping creates 3,500 LiquidFun fluid particles and 50 dynamic
				triangle fixtures. The 24 fracture nodes are omitted; LiquidFun elastic behavior is not treated as equivalent
				to MPM damage/fracture.
			</li>
			<li>
				Each triangle body fits its initial linear and angular velocity from its three seeded MPM node velocities.
				A rigid Box2D fixture is not a deformable Neo-Hookean MPM pebble.
			</li>
			<li>
				Live physics timing wraps synchronous CPU <code>world.Step()</code> calls; Canvas2D drawing has a separate
				metric. The benchmark runs without drawing and reports neither GPU timestamps nor queue waits.
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
		width: min(1440px, 100%);
		margin: 0 auto;
		padding: clamp(1rem, 3vw, 2.5rem);
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
		display: grid;
		grid-template-columns: minmax(0, 1fr) minmax(260px, 330px);
		gap: 1rem;
		align-items: start;
	}

	.stage,
	.control-panel,
	.limitations {
		border: 1px solid #26374d;
		border-radius: 14px;
		background: #121c2a;
	}

	.stage {
		min-width: 0;
		padding: clamp(0.75rem, 2vw, 1.15rem);
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
		aspect-ratio: 22 / 15;
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
		padding: 1rem;
	}

	.controls {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 0.6rem;
		margin-bottom: 1.25rem;
	}

	button {
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

	button:hover:not(:disabled) {
		background: #263a50;
	}

	button:focus-visible,
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

		.control-panel {
			padding: 0.85rem;
		}
	}
</style>
