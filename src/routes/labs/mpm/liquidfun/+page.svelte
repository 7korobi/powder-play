<script lang="ts">
	import { onMount } from 'svelte';
	import type { MpmVector2 } from '#lib/mpm/reference/solver.js';
	import {
		MPM_RENDER_FPS_OPTIONS,
		MpmFramePacer,
		type MpmRenderFps
	} from '#lib/mpm/kernels/frame-pacer.js';
	import {
		HORIZONTAL_OSCILLATION_FREQUENCY_HZ
	} from '#lib/mpm/kernels/oscillation.js';
	import { calculateLiquidFunStepGravity } from '#lib/mpm/liquidfun/oscillation.js';
	import {
		createLiquidFunSimulation,
		type LiquidFunRenderMode,
		type LiquidFunSimulation
	} from '#lib/mpm/liquidfun/simulation.js';
	import { LIQUIDFUN_WALL_THICKNESS_METERS } from '#lib/mpm/liquidfun/wall-geometry.js';
	import {
		LIQUIDFUN_BOX2D_DOMAIN_UNITS,
		LIQUIDFUN_BOX2D_UNITS_PER_METER,
		LIQUIDFUN_DT_SECONDS,
		LIQUIDFUN_FRACTURE_BOND_FAILURE_STRAIN,
		LIQUIDFUN_GATE4_DOMAIN,
		LIQUIDFUN_GATE4_ACCELERATION_UNITS_PER_METER_PER_SECOND_SQUARED,
		LIQUIDFUN_GATE4_UNITS_PER_METER,
		LIQUIDFUN_OIL_PARTICLE_COUNT,
		LIQUIDFUN_OIL_PARTICLE_COUNT_MULTIPLIER,
		LIQUIDFUN_PACKAGE_VERSION,
		LIQUIDFUN_PARTICLE_RADIUS_METERS,
		LIQUIDFUN_PARTICLE_SPACING_METERS,
		LIQUIDFUN_PEBBLE_EDGE_LENGTH_METERS,
		LIQUIDFUN_PEBBLE_MASS,
		LIQUIDFUN_PHYSICAL_DOMAIN_METERS,
		LIQUIDFUN_SCENE_SEED,
		LIQUIDFUN_SHARED_PARTICLE_MASS,
		LIQUIDFUN_VELOCITY_ITERATIONS,
		LIQUIDFUN_POSITION_ITERATIONS,
		LIQUIDFUN_WATER_PARTICLE_COUNT_MULTIPLIER,
		LIQUIDFUN_WATER_PARTICLE_COUNT
	} from '#lib/mpm/liquidfun/workload.js';

	const MAX_STEPS_PER_FRAME = 8;
	const MAX_PHYSICS_FRAME_BUDGET_MS = 8;
	const SIMULATION_SPEED_OPTIONS = [0.25, 0.5, 1, 2, 4, 8, 16, 32] as const;
	const MANUAL_GRAVITY_COMPONENT_LIMIT = 999;
	const DEFAULT_LIQUIDFUN_OSCILLATION_AMPLITUDE = 999;
	const MAX_LIQUIDFUN_OSCILLATION_AMPLITUDE = 999;
	const DEFAULT_RENDER_FPS: MpmRenderFps = 90;
	const METRICS_INTERVAL_MS = 500;
	const framePacer = new MpmFramePacer();

	type Phase = 'loading' | 'ready' | 'error';

	let canvas: HTMLCanvasElement | undefined;
	let context: CanvasRenderingContext2D | null = null;
	let simulation = $state<LiquidFunSimulation | null>(null);
	let phase = $state<Phase>('loading');
	let paused = $state(false);
	let pageVisible = $state(true);
	let errorMessage = $state('');
	let gravityX = $state(0);
	let gravityY = $state(0);
	let horizontalOscillationAmplitude = $state(DEFAULT_LIQUIDFUN_OSCILLATION_AMPLITUDE);
	let simulationSpeed = $state(0.5);
	let renderTargetFps: MpmRenderFps = $state(DEFAULT_RENDER_FPS);
	let renderMode = $state<LiquidFunRenderMode>('surface');
	let fps = $state(0);
	let stepsPerSecond = $state(0);
	let physicsMsPerStep = $state(0);
	let pendingSimulationMs = $state(0);

	let mounted = false;
	let generation = 0;
	let animationFrame = 0;
	let lastFrameTime = 0;
	let lastMetricsTime = 0;
	let simulatedTimeSeconds = 0;
	let accumulatorSeconds = 0;
	let renderedFramesForMetrics = 0;
	let stepsForMetrics = 0;
	let physicsMsForMetrics = 0;

	const effectiveSimulationSpeed = $derived(
		stepsPerSecond * LIQUIDFUN_DT_SECONDS
	);

	const statusText = $derived(
		phase === 'loading'
			? 'LiquidFun WASM を初期化中…'
			: phase === 'error'
				? 'シミュレーションを停止しました'
				: !pageVisible
					? 'ページ非表示中 · 物理計算を停止中'
					: paused
						? '一時停止中 · 未処理時間は保持されています'
						: `実行中 · LiquidFun WASM · 要求 ${simulationSpeed}×`
	);

	function gravityForNextStep(): MpmVector2 {
		return calculateLiquidFunStepGravity(
			{ x: gravityX, y: gravityY },
			simulatedTimeSeconds,
			horizontalOscillationAmplitude
		);
	}

	function errorText(error: unknown): string {
		return error instanceof Error && error.message.trim()
			? error.message
			: 'LiquidFun WASM の処理中に詳細不明のエラーが発生しました。';
	}

	function resetMetrics(): void {
		fps = 0;
		stepsPerSecond = 0;
		physicsMsPerStep = 0;
		lastMetricsTime = 0;
		renderedFramesForMetrics = 0;
		stepsForMetrics = 0;
		physicsMsForMetrics = 0;
	}

	function failSimulation(error: unknown): void {
		paused = true;
		phase = 'error';
		errorMessage = errorText(error);
		console.error('LiquidFun WASM simulation failed.', error);
		const failedSimulation = simulation;
		simulation = null;
		try {
			failedSimulation?.dispose();
		} catch (disposeError) {
			console.error('LiquidFun WASM cleanup failed after a simulation error.', disposeError);
			errorMessage += ` WASM リソースの解放にも失敗しました: ${errorText(disposeError)}`;
		}
	}

	async function initializeSimulation(): Promise<void> {
		const currentGeneration = ++generation;
		paused = false;
		phase = 'loading';
		errorMessage = '';
		simulatedTimeSeconds = 0;
		accumulatorSeconds = 0;
		pendingSimulationMs = 0;
		lastFrameTime = 0;
		resetMetrics();

		try {
			const previousSimulation = simulation;
			simulation = null;
			previousSimulation?.dispose();

			const createdSimulation = await createLiquidFunSimulation();
			if (!mounted || currentGeneration !== generation) {
				try {
					createdSimulation.dispose();
				} catch (disposeError) {
					console.error('Stale LiquidFun WASM simulation cleanup failed.', disposeError);
				}
				return;
			}

			try {
				createdSimulation.setGravityFromGate4(gravityForNextStep());
			} catch (error) {
				try {
					createdSimulation.dispose();
				} catch (disposeError) {
					console.error('LiquidFun WASM cleanup failed after initialization.', disposeError);
				}
				throw error;
			}

			simulation = createdSimulation;
			phase = 'ready';
		} catch (error) {
			if (currentGeneration !== generation) {
				console.error('Superseded LiquidFun WASM initialization failed.', error);
				return;
			}
			failSimulation(error);
		}
	}

	function resizeCanvas(): void {
		if (!canvas) return;
		context = canvas.getContext('2d');
		if (!context) {
			failSimulation(new Error('このブラウザーでは Canvas2D を利用できません。'));
			return;
		}

		const bounds = canvas.getBoundingClientRect();
		const devicePixelRatio = Math.min(window.devicePixelRatio || 1, 2);
		canvas.width = Math.max(1, Math.round(bounds.width * devicePixelRatio));
		canvas.height = Math.max(1, Math.round(bounds.height * devicePixelRatio));
	}

	function setGravity(): void {
		try {
			simulation?.setGravityFromGate4(gravityForNextStep());
		} catch (error) {
			failSimulation(error);
		}
	}

	function toggleHorizontalOscillation(): void {
		horizontalOscillationAmplitude =
			horizontalOscillationAmplitude > 0
				? 0
				: DEFAULT_LIQUIDFUN_OSCILLATION_AMPLITUDE;
		setGravity();
	}

	function stepOnce(): void {
		const currentSimulation = simulation;
		if (!currentSimulation || phase !== 'ready' || !paused) return;
		try {
			currentSimulation.setGravityFromGate4(gravityForNextStep());
			const startedAt = performance.now();
			currentSimulation.step();
			simulatedTimeSeconds += LIQUIDFUN_DT_SECONDS;
			physicsMsForMetrics += performance.now() - startedAt;
			stepsForMetrics += 1;
		} catch (error) {
			failSimulation(error);
		}
	}

	function togglePaused(): void {
		paused = !paused;
		lastFrameTime = 0;
	}

	function discardPendingTime(): void {
		accumulatorSeconds = 0;
		pendingSimulationMs = 0;
	}

	function handleVisibilityChange(): void {
		pageVisible = document.visibilityState === 'visible';
		lastFrameTime = 0;
		framePacer.reset();
		resetMetrics();
	}

	function renderFrame(now: number): void {
		if (!mounted) return;
		animationFrame = window.requestAnimationFrame(renderFrame);

		const elapsedSeconds =
			lastFrameTime === 0 ? 0 : Math.max(0, (now - lastFrameTime) / 1_000);
		lastFrameTime = now;
		const currentSimulation = simulation;
		if (currentSimulation && phase === 'ready' && !paused && pageVisible) {
			try {
				accumulatorSeconds += elapsedSeconds * simulationSpeed;
				if (!Number.isFinite(accumulatorSeconds)) {
					throw new RangeError('LiquidFun simulation backlog exceeded the finite time range.');
				}

				const physicsStartedAt = performance.now();
				let stepsThisFrame = 0;
				while (
					accumulatorSeconds >= LIQUIDFUN_DT_SECONDS &&
					stepsThisFrame < MAX_STEPS_PER_FRAME &&
					performance.now() - physicsStartedAt < MAX_PHYSICS_FRAME_BUDGET_MS
				) {
					currentSimulation.setGravityFromGate4(gravityForNextStep());
					const stepStartedAt = performance.now();
					currentSimulation.step();
					simulatedTimeSeconds += LIQUIDFUN_DT_SECONDS;
					physicsMsForMetrics += performance.now() - stepStartedAt;
					accumulatorSeconds = Math.max(
						0,
						accumulatorSeconds - LIQUIDFUN_DT_SECONDS
					);
					stepsForMetrics += 1;
					stepsThisFrame += 1;
				}
			} catch (error) {
				failSimulation(error);
			}
		}

		pendingSimulationMs = accumulatorSeconds * 1_000;

		if (pageVisible && framePacer.shouldRender(now, renderTargetFps)) {
			if (context && canvas && simulation && phase === 'ready') {
				try {
					simulation.draw(context, canvas.width, canvas.height, renderMode);
					renderedFramesForMetrics += 1;
				} catch (error) {
					failSimulation(error);
				}
			}
		}

		if (lastMetricsTime === 0) {
			lastMetricsTime = now;
		} else if (now - lastMetricsTime >= METRICS_INTERVAL_MS) {
			const elapsedMs = now - lastMetricsTime;
			fps = (renderedFramesForMetrics * 1_000) / elapsedMs;
			stepsPerSecond = (stepsForMetrics * 1_000) / elapsedMs;
			physicsMsPerStep =
				stepsForMetrics > 0 ? physicsMsForMetrics / stepsForMetrics : 0;
			lastMetricsTime = now;
			renderedFramesForMetrics = 0;
			stepsForMetrics = 0;
			physicsMsForMetrics = 0;
		}
	}

	onMount(() => {
		mounted = true;
		pageVisible = document.visibilityState === 'visible';
		context = canvas?.getContext('2d') ?? null;
		if (!context) {
			failSimulation(new Error('このブラウザーでは Canvas2D を利用できません。'));
		} else {
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
		}

		return () => {
			mounted = false;
			generation += 1;
		};
	});
</script>

<svelte:head>
	<title>LiquidFun WASM · Gate 4 behavioral reference</title>
	<meta
		name="description"
		content="LiquidFun WASM を実際に動かして、MLS-MPM GPU 版と接触・波・安定性の挙動を比較するページ"
	/>
</svelte:head>

<main class="page">
	<header class="page-header">
		<p class="eyebrow">LIQUIDFUN WASM · BEHAVIORAL REFERENCE</p>
		<h1>LiquidFun WASM · Gate 4 behavior lab</h1>
		<p class="intro">
			既存の Gate 4 シーンを <code>liquidfun-wasm</code> の Box2D/LiquidFun world で実行します。
			このページは挙動・接触・安定性を観察する独立した参照先です。GPU MLS-MPM の同じ式を解く数値 oracle ではありません。
			要求速度が処理能力を超えたときは、1フレーム当たりの物理計算量を制限し、未処理時間を表示して catch-up の急増を防ぎます。
		</p>
		<nav class="solver-links" aria-label="Simulation pages">
			<a href="/labs/mpm/kernels">GPU MLS-MPM page →</a>
			<a href="/labs/mpm/reference">CPU MLS-MPM numerical reference</a>
			<a href="/labs/mpm">MPM lab</a>
		</nav>
	</header>

	<section class="simulation-layout" aria-label="LiquidFun WASM simulation">
		<div class="stage">
			<div class="stage-heading">
				<div>
					<p class="eyebrow">LiquidFun · fixed Gate 4 source scene</p>
					<h2>Gate 4 LiquidFun scene</h2>
				</div>
				<span class="engine-tag">WASM · liquidfun-wasm {LIQUIDFUN_PACKAGE_VERSION}</span>
			</div>

			<div class="canvas-frame">
				<canvas bind:this={canvas} aria-label="LiquidFun WASM water, oil and solid particle simulation">
					Canvas2D に対応したブラウザーでご利用ください。
				</canvas>
			</div>
			<p class="status" role="status" aria-live="polite">{statusText}</p>
			{#if errorMessage}
				<p class="error" role="alert">{errorMessage}</p>
				<button class="retry-button" type="button" onclick={() => void initializeSimulation()}>
					Reset and retry
				</button>
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
				<legend>
					Manual 2D gravity · Gate 4 units/s² (±{MANUAL_GRAVITY_COMPONENT_LIMIT})
				</legend>
				<label class="range-control">
					<span>X</span>
					<output>{gravityX}</output>
					<input
						type="range"
						min={-MANUAL_GRAVITY_COMPONENT_LIMIT}
						max={MANUAL_GRAVITY_COMPONENT_LIMIT}
						step="1"
						bind:value={gravityX}
						oninput={setGravity}
						disabled={phase !== 'ready'}
					/>
				</label>
				<label class="range-control">
					<span>Y</span>
					<output>{gravityY}</output>
					<input
						type="range"
						min={-MANUAL_GRAVITY_COMPONENT_LIMIT}
						max={MANUAL_GRAVITY_COMPONENT_LIMIT}
						step="1"
						bind:value={gravityY}
						oninput={setGravity}
						disabled={phase !== 'ready'}
					/>
				</label>
				<p class="control-note">
					Manual gravity defaults to (0, 0). {LIQUIDFUN_GATE4_ACCELERATION_UNITS_PER_METER_PER_SECOND_SQUARED}
					Gate 4 units/s² = 1 m/s², so 980 is approximately 1G. The configured range is experimental,
					not a safe-limit guarantee.
				</p>
			</fieldset>

			<fieldset class="oscillation-controls">
				<legend>Horizontal oscillation · {HORIZONTAL_OSCILLATION_FREQUENCY_HZ} Hz</legend>
				<label class="range-control">
					<span>Acceleration amplitude</span>
					<output>±{horizontalOscillationAmplitude}</output>
					<input
						type="range"
						min="0"
						max={MAX_LIQUIDFUN_OSCILLATION_AMPLITUDE}
						step="1"
						bind:value={horizontalOscillationAmplitude}
						oninput={setGravity}
						disabled={phase !== 'ready'}
						aria-label="Horizontal oscillation acceleration"
						aria-valuetext={`±${horizontalOscillationAmplitude} Gate 4 units/s²`}
					/>
				</label>
				<button
					class="wide-button"
					type="button"
					onclick={toggleHorizontalOscillation}
					disabled={phase !== 'ready'}
					aria-pressed={horizontalOscillationAmplitude > 0}
				>
					{horizontalOscillationAmplitude > 0
						? 'Stop horizontal oscillation'
						: '						Start horizontal oscillation (±{DEFAULT_LIQUIDFUN_OSCILLATION_AMPLITUDE})'}
				</button>
				<p class="control-note">
					Added to manual X gravity; amplitude can reach ±{MAX_LIQUIDFUN_OSCILLATION_AMPLITUDE}
					Gate 4 units/s² (about ±{(MAX_LIQUIDFUN_OSCILLATION_AMPLITUDE /
						LIQUIDFUN_GATE4_ACCELERATION_UNITS_PER_METER_PER_SECOND_SQUARED).toFixed(2)} m/s²)
					and is synchronized to simulation time. Set it to 0 to disable.
				</p>
			</fieldset>

			<div class="setting-controls">
				<label>
					<span>Requested simulation speed</span>
					<select bind:value={simulationSpeed}>
						{#each SIMULATION_SPEED_OPTIONS as speed}
							<option value={speed}>{speed}×</option>
						{/each}
					</select>
				</label>
				<label>
					<span>Fluid display</span>
					<select bind:value={renderMode}>
						<option value="surface">Surface</option>
						<option value="particles">Particle markers</option>
					</select>
				</label>
				<label>
					<span>Render target FPS</span>
					<select bind:value={renderTargetFps}>
						{#each MPM_RENDER_FPS_OPTIONS as targetFps}
							<option value={targetFps}>{targetFps} FPS</option>
						{/each}
					</select>
				</label>
			</div>

			<div class="backlog-controls">
				<p>Pending simulation time: {pendingSimulationMs.toFixed(1)} ms</p>
				<button
					class="wide-button"
					type="button"
					disabled={!paused || pendingSimulationMs <= 0}
					onclick={discardPendingTime}
				>
					Discard pending time
				</button>
				<p class="control-note">
					Per-frame physics budget: up to {MAX_STEPS_PER_FRAME} steps and
					{MAX_PHYSICS_FRAME_BUDGET_MS} ms. If the requested speed is too high, the backlog grows instead
					of triggering an unbounded catch-up loop. Render target FPS caps Canvas drawing only; physics
					continues to use the same fixed timestep.
				</p>
			</div>

			<h2>Live metrics · rolling 0.5 s</h2>
			<dl class="metrics">
				<div>
					<dt>Render FPS</dt>
					<dd>{fps.toFixed(1)}</dd>
				</div>
				<div>
					<dt>Render target (cap)</dt>
					<dd>{renderTargetFps} FPS</dd>
				</div>
				<div>
					<dt>Completed steps / s</dt>
					<dd>{stepsPerSecond.toFixed(1)}</dd>
				</div>
				<div>
					<dt>Effective simulation speed</dt>
					<dd>{effectiveSimulationSpeed.toFixed(2)}×</dd>
				</div>
				<div>
					<dt>JS + WASM CPU ms / step</dt>
					<dd>{physicsMsPerStep.toFixed(3)}</dd>
				</div>
				<div>
					<dt>LiquidFun particles</dt>
					<dd>{simulation?.particleCount.toLocaleString() ?? '—'}</dd>
				</div>
				<div>
					<dt>Water / oil flags</dt>
					<dd>
						{simulation?.waterParticleCount.toLocaleString() ?? '—'} /
						{simulation?.oilParticleCount.toLocaleString() ?? '—'}
					</dd>
				</div>
				<div>
					<dt>Rigid pebble triangle bodies</dt>
					<dd>{simulation?.triangleBodyCount ?? '—'}</dd>
				</div>
				<div>
					<dt>Fracture nodes / active bonds</dt>
					<dd>
						{simulation?.fractureNodeCount ?? '—'} /
						{simulation?.activeFractureBondCount ?? '—'}
					</dd>
				</div>
			</dl>

			<div class="configuration">
				<h2>Fixed workload</h2>
				<ul>
					<li>
						Seed <code>0x{LIQUIDFUN_SCENE_SEED.toString(16)}</code> ·
						{simulation?.sourceMpmNodeCount.toLocaleString() ?? '3,674'} source nodes ·
						{simulation?.mappedMpmNodeCount.toLocaleString() ?? '3,674'} mapped
					</li>
					<li>
						{LIQUIDFUN_WATER_PARTICLE_COUNT.toLocaleString()} water /
						{' '}{LIQUIDFUN_OIL_PARTICLE_COUNT.toLocaleString()} oil particles ·
						water {LIQUIDFUN_WATER_PARTICLE_COUNT_MULTIPLIER}× / oil
						{LIQUIDFUN_OIL_PARTICLE_COUNT_MULTIPLIER}× source count ·
						{(LIQUIDFUN_PARTICLE_SPACING_METERS * 1_000).toFixed(1)} mm spacing
					</li>
					<li>
						{LIQUIDFUN_GATE4_DOMAIN.width} × {LIQUIDFUN_GATE4_DOMAIN.height} Gate 4 units ·
						{(LIQUIDFUN_PHYSICAL_DOMAIN_METERS.width * 1_000).toFixed(0)} ×
						{(LIQUIDFUN_PHYSICAL_DOMAIN_METERS.height * 1_000).toFixed(0)} mm visible domain
					</li>
					<li>
						{(LIQUIDFUN_WALL_THICKNESS_METERS * 1_000).toLocaleString()} mm of static wall depth beyond each visible
						domain edge · {LIQUIDFUN_BOX2D_DOMAIN_UNITS.width} ×
						{LIQUIDFUN_BOX2D_DOMAIN_UNITS.height} Box2D engine units
					</li>
					<li>
						dt = 1/{Math.round(1 / LIQUIDFUN_DT_SECONDS)} s · position scale {LIQUIDFUN_GATE4_UNITS_PER_METER}
						Gate 4 units / m · Box2D scale {LIQUIDFUN_BOX2D_UNITS_PER_METER} units / m
					</li>
					<li>
						{LIQUIDFUN_VELOCITY_ITERATIONS} velocity + {LIQUIDFUN_POSITION_ITERATIONS} position
						iterations · particle iterations {simulation?.particleIterations ?? '…'}
					</li>
					<li>
						Fluid radius {(LIQUIDFUN_PARTICLE_RADIUS_METERS * 1_000).toFixed(1)} mm · shared water/oil
						particle mass {LIQUIDFUN_SHARED_PARTICLE_MASS} simulation units · pebble edge
						{(LIQUIDFUN_PEBBLE_EDGE_LENGTH_METERS * 1_000).toFixed(1)} mm / mass
						{LIQUIDFUN_PEBBLE_MASS} simulation units · fracture threshold
						{(LIQUIDFUN_FRACTURE_BOND_FAILURE_STRAIN * 100).toFixed(0)}%
					</li>
				</ul>
			</div>
		</aside>
	</section>

	<section class="limitations" aria-labelledby="comparison-heading">
		<h2 id="comparison-heading">How to use this reference</h2>
		<ul>
			<li>
				This page steps the installed <code>liquidfun-wasm</code> engine. Liquid particles are solved by
				LiquidFun; pebble triangles and the fracture network are represented by Box2D rigid bodies and
				distance joints. Canvas2D is only the renderer.
			</li>
			<li>
				Water and oil use separate LiquidFun particle flags in one particle system, so both use the same
				1.0 mm radius and 0.05 simulation mass units and retain mutual particle collisions. The triangular pebbles are
				rigid approximations, not deformable MLS-MPM triangles. Do not interpret their paths as a
				matched-physics comparison.
			</li>
			<li>
				Use the CPU MLS-MPM page at <a href="/labs/mpm/reference">/labs/mpm/reference</a> as the
				same-equation numerical reference for <a href="/labs/mpm/kernels">GPU MLS-MPM</a>. Use this
				LiquidFun page to judge whether its contact, wave, settling and stability behavior is a useful
				target. Tune the GPU implementation toward selected behaviors only after recording measurable
				criteria; do not require particle-by-particle trajectory parity across different solvers.
			</li>
			<li>
				The live timing includes synchronous JavaScript-to-WASM simulation calls and excludes rendering.
				Compare it only under the same browser/device and workload; this is not a GPU timing or a
				cross-device performance guarantee.
			</li>
		</ul>
		<p class="docs-link">
			See <code>docs/LIQUIDFUN-BENCHMARK.md</code> for mapping and benchmark methodology and
			<code>docs/MPM-CPU-GPU-COMPARISON.md</code> for the separate numerical-reference role.
		</p>
		<p class="license-links">
			Package notices:
			<a href="/licenses/liquidfun-wasm/LICENSE.zlib.txt">Zlib</a> ·
			<a href="/licenses/liquidfun-wasm/LICENSE.wasm-feature-detect.Apache.txt">Apache 2.0</a>
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
		width: min(100%, 1480px);
		margin: 0 auto;
		padding: clamp(1rem, 3vw, 2.5rem);
	}

	.page-header {
		margin-bottom: 1.25rem;
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
		max-width: 84ch;
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

	.simulation-layout,
	.limitations {
		border: 1px solid #26374d;
		border-radius: 14px;
		background: #121c2a;
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

	.backlog-controls {
		display: grid;
		gap: 0.55rem;
		margin: 0 0 1rem;
		border: 1px solid #31445a;
		border-radius: 0.55rem;
		padding: 0.75rem;
		background: #0d1823;
	}

	.backlog-controls p {
		margin: 0;
		color: #c4d0de;
		font-size: 0.84rem;
		line-height: 1.5;
		overflow-wrap: anywhere;
	}

	.control-note {
		color: #b8c7d8 !important;
	}

	.wide-button,
	.retry-button {
		width: 100%;
	}

	.retry-button {
		margin-top: 0.75rem;
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

	.limitations ul {
		margin-bottom: 1rem;
	}

	.docs-link,
	.license-links {
		margin: 0.75rem 0 0;
		color: #b9c8d8;
		font-size: 0.9rem;
		line-height: 1.55;
		overflow-wrap: anywhere;
	}

	@media (max-width: 900px) {
		.simulation-layout {
			grid-template-areas:
				'canvas'
				'controls';
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
