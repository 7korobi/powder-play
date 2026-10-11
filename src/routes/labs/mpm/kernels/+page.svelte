<script lang="ts">
	import { onMount } from 'svelte';
	import { createMpmMlsGpuSolver } from '#lib/mpm/kernels/solver.js';
	import {
		stepMpm,
		type MpmMatrix2,
		type MpmParticle,
		type MpmStepConfig
	} from '#lib/mpm/reference/solver.js';

	const GRID = { cellsX: 48, cellsY: 32, cellSize: 10 } as const;
	const FIXED_STEP_SECONDS = 1 / 120;
	const FRAME_INTERVAL_MS = 1000 / 30;
	const MAX_FRAME_SECONDS = 0.1;
	const MAX_STEPS_PER_FRAME = 4;
	const PARTICLE_MASS = 1;
	const PARTICLE_VOLUME = 25;
	const MATERIAL = { lambda: 20, mu: 10 } as const;
	const PARITY_TOLERANCE = 0.02;
	const ZERO_MATRIX: MpmMatrix2 = { m00: 0, m01: 0, m10: 0, m11: 0 };
	const INITIAL_DEFORMATION: MpmMatrix2 = {
		m00: 0.99,
		m01: 0,
		m10: 0,
		m11: 0.99
	};

	type Phase = 'initializing' | 'ready' | 'error' | 'lost';
	type GpuSolver = Awaited<ReturnType<typeof createMpmMlsGpuSolver>>;

	function makeParticles(): MpmParticle[] {
		const particles: MpmParticle[] = [];

		for (let row = 0; row < 7; row += 1) {
			for (let column = 0; column < 8; column += 1) {
				particles.push({
					position: { x: 160 + column * 8, y: 60 + row * 8 },
					velocity: { x: 0, y: 0 },
					C: ZERO_MATRIX,
					F: INITIAL_DEFORMATION
				});
			}
		}

		return particles;
	}

	function makeStepConfig(gravityMagnitude: number, dt = FIXED_STEP_SECONDS): MpmStepConfig {
		return {
			dt,
			grid: GRID,
			particleMass: PARTICLE_MASS,
			particleVolume: PARTICLE_VOLUME,
			material: MATERIAL,
			gravity: { x: 0, y: gravityMagnitude }
		};
	}

	function errorText(error: unknown): string {
		return error instanceof Error && error.message.trim()
			? error.message
			: 'GPU MLS-MPM の処理に失敗しました。';
	}

	function particleValues(particle: MpmParticle): number[] {
		return [
			particle.position.x,
			particle.position.y,
			particle.velocity.x,
			particle.velocity.y,
			particle.C.m00,
			particle.C.m01,
			particle.C.m10,
			particle.C.m11,
			particle.F.m00,
			particle.F.m01,
			particle.F.m10,
			particle.F.m11
		];
	}

	function maxParticleDifference(
		actual: readonly MpmParticle[],
		expected: readonly MpmParticle[]
	): number {
		if (actual.length !== expected.length) return Number.POSITIVE_INFINITY;

		let maximum = 0;
		for (let index = 0; index < actual.length; index += 1) {
			const actualValues = particleValues(actual[index]);
			const expectedValues = particleValues(expected[index]);

			for (let valueIndex = 0; valueIndex < actualValues.length; valueIndex += 1) {
				const actualValue = actualValues[valueIndex];
				const expectedValue = expectedValues[valueIndex];
				if (!Number.isFinite(actualValue) || !Number.isFinite(expectedValue)) {
					return Number.POSITIVE_INFINITY;
				}
				maximum = Math.max(maximum, Math.abs(actualValue - expectedValue));
			}
		}

		return maximum;
	}

	let canvas: HTMLCanvasElement | undefined;
	let solver: GpuSolver | null = null;
	let phase = $state<Phase>('initializing');
	let phaseMessage = $state('WebGPU MLS-MPM を初期化しています…');
	let gravityMagnitude = $state(90);
	let paused = $state(false);
	let parityMessage = $state('');
	let parityPassed = $state<boolean | null>(null);
	let stateCheckPending = $state(false);
	let stateCheckMessage = $state('');
	let stateCheckIsError = $state(false);
	let particleCount = 56;

	function togglePaused(): void {
		paused = !paused;
	}

	function stepOnce(): void {
		if (phase !== 'ready' || !paused || !solver) return;

		try {
			solver.step(FIXED_STEP_SECONDS, { x: 0, y: gravityMagnitude });
			solver.render();
		} catch (error) {
			phase = 'error';
			phaseMessage = `1 ステップを実行できませんでした: ${errorText(error)}`;
		}
	}

	async function checkGpuState(): Promise<void> {
		const currentSolver = solver;
		if (!currentSolver || phase !== 'ready' || !paused || stateCheckPending) return;

		stateCheckPending = true;
		stateCheckIsError = false;
		stateCheckMessage = 'GPU 数値状態を読み取り中…';
		try {
			const particles = await currentSolver.readParticles();
			if (solver !== currentSolver) return;
			stateCheckMessage = `${particles.length} 粒子の GPU 数値状態は正常です。`;
		} catch (error) {
			if (solver !== currentSolver) return;
			stateCheckIsError = true;
			stateCheckMessage = `GPU 数値状態を確認できませんでした: ${errorText(error)}`;
			phase = 'error';
			phaseMessage = 'GPU 数値状態の異常を検出したため停止しました。';
		} finally {
			if (solver === currentSolver) stateCheckPending = false;
		}
	}

	onMount(() => {
		let mounted = true;
		let activeSolver: GpuSolver | null = null;
		let frameRequest: number | null = null;
		let lastFrameTimestamp = 0;
		let accumulator = 0;
		let lastRenderTimestamp = 0;
		const initialParticles = makeParticles();
		const canvasElement = canvas;

		const resizeCanvas = () => {
			const bounds = canvasElement?.getBoundingClientRect();
			if (!canvasElement || !bounds) return;
			if (bounds.width <= 0 || bounds.height <= 0) return;

			const pixelRatio = window.devicePixelRatio || 1;
			const width = Math.max(1, Math.round(bounds.width * pixelRatio));
			const height = Math.max(1, Math.round(bounds.height * pixelRatio));
			if (canvasElement.width !== width) canvasElement.width = width;
			if (canvasElement.height !== height) canvasElement.height = height;
		};

		function fail(error: unknown): void {
			if (!mounted) return;
			phase = 'error';
			phaseMessage = errorText(error);
			console.error('GPU MLS-MPM initialization failed.', error);
		}

		function renderFrame(timestamp: number): void {
			if (!mounted || !activeSolver || phase !== 'ready') return;
			frameRequest = requestAnimationFrame(renderFrame);
			if (timestamp - lastRenderTimestamp < FRAME_INTERVAL_MS - 1) return;

			const elapsed =
				lastFrameTimestamp === 0
					? 0
					: Math.min((timestamp - lastFrameTimestamp) / 1000, MAX_FRAME_SECONDS);
			lastFrameTimestamp = timestamp;
			lastRenderTimestamp = timestamp;

			if (!paused && document.visibilityState === 'visible') {
				accumulator += elapsed;
				let steps = 0;
				while (accumulator >= FIXED_STEP_SECONDS && steps < MAX_STEPS_PER_FRAME) {
					try {
						activeSolver.step(FIXED_STEP_SECONDS, {
							x: 0,
							y: gravityMagnitude
						});
					} catch (error) {
						phase = 'error';
						phaseMessage = `シミュレーションを停止しました: ${errorText(error)}`;
						return;
					}
					accumulator -= FIXED_STEP_SECONDS;
					steps += 1;
				}
				if (steps === MAX_STEPS_PER_FRAME && accumulator >= FIXED_STEP_SECONDS) {
					accumulator = 0;
				}
			} else {
				accumulator = 0;
			}

			try {
				activeSolver.render();
			} catch (error) {
				phase = 'error';
				phaseMessage = `描画を停止しました: ${errorText(error)}`;
			}
		}

		function handleVisibilityChange(): void {
			lastFrameTimestamp = 0;
			accumulator = 0;
		}

		async function initialize(): Promise<void> {
			if (!canvasElement) {
				fail(new Error('GPU canvas が初期化されていません。'));
				return;
			}

			try {
				resizeCanvas();
				const candidate = await createMpmMlsGpuSolver(
					canvasElement,
					initialParticles,
					makeStepConfig(gravityMagnitude)
				);
				if (!mounted) {
					candidate.dispose();
					return;
				}

				activeSolver = candidate;
				solver = candidate;
				candidate.deviceLost.then((info) => {
					if (!mounted || activeSolver !== candidate) return;
					if (frameRequest !== null) cancelAnimationFrame(frameRequest);
					phase = 'lost';
					phaseMessage = info.message || 'WebGPU デバイスが失われました。';
				});

				const config = makeStepConfig(gravityMagnitude);
				const expected = stepMpm(initialParticles, config).particles;
				candidate.step(FIXED_STEP_SECONDS, config.gravity);
				const actual = await candidate.readParticles();
				if (!mounted || activeSolver !== candidate) return;

				const maxError = maxParticleDifference(actual, expected);
				parityPassed = maxError <= PARITY_TOLERANCE;
				parityMessage = Number.isFinite(maxError)
					? `初回1ステップの CPU reference 比較: 最大絶対誤差 ${maxError.toExponential(2)} ${
							parityPassed ? '· 許容範囲内' : '· 許容範囲外'
						}`
					: '初回1ステップの CPU reference 比較で粒子数または数値が一致しません。';
				phase = 'ready';
				phaseMessage = 'GPU MLS-MPM を実行中です · 目標 30 fps';
				lastFrameTimestamp = performance.now();
				frameRequest = requestAnimationFrame(renderFrame);
			} catch (error) {
				fail(error);
			}
		}

		if (!canvasElement) {
			fail(new Error('GPU canvas が初期化されていません。'));
			return;
		}
		const resizeObserver = new ResizeObserver(resizeCanvas);
		resizeObserver.observe(canvasElement);
		window.addEventListener('resize', resizeCanvas);
		resizeCanvas();
		document.addEventListener('visibilitychange', handleVisibilityChange);
		void initialize();

		return () => {
			mounted = false;
			resizeObserver.disconnect();
			window.removeEventListener('resize', resizeCanvas);
			document.removeEventListener('visibilitychange', handleVisibilityChange);
			if (frameRequest !== null) cancelAnimationFrame(frameRequest);
			const disposedSolver = activeSolver;
			activeSolver = null;
			solver = null;
			if (disposedSolver) {
				try {
					disposedSolver.dispose();
				} catch (error) {
					console.error('GPU MLS-MPM resources could not be released cleanly.', error);
				}
			}
		};
	});
</script>

<svelte:head>
	<title>GPU MLS-MPM kernels · powder-play</title>
	<meta
		name="description"
		content="Single-material 2D MLS-MPM GPU kernel and CPU reference comparison."
	/>
</svelte:head>

<main class="lab">
	<header class="header">
		<div>
			<p class="eyebrow">GATE 1 · NUMERICAL CORE</p>
			<h1>GPU 2D MLS-MPM</h1>
			<p class="intro">
				単一材料の弾性 MLS-MPM を GPU compute pass で実行し、初回の1ステップを CPU
				reference と比較します。
			</p>
		</div>
		<nav aria-label="実験ページ">
			<a href="/labs/mpm">粒子操作ラボ</a>
			<a href="/">通常画面</a>
		</nav>
	</header>

	<section class="panel" aria-labelledby="simulation-title">
		<div class="panel-heading">
			<div>
				<h2 id="simulation-title">弾性粒子クラスター</h2>
				<p>{particleCount} 粒子 · {GRID.cellsX} × {GRID.cellsY} grid · Neo-Hookean</p>
			</div>
			<p class="status" class:error={phase === 'error' || phase === 'lost'} role="status">
				{phaseMessage}
			</p>
		</div>

		<div class="canvas-frame">
			<canvas bind:this={canvas} aria-label="GPU 上で計算する MLS-MPM 粒子シミュレーション">
				WebGPU canvas に対応したブラウザーが必要です。
			</canvas>
		</div>

		{#if parityMessage}
			<p class="parity" class:passed={parityPassed} class:failed={parityPassed === false} role="status">
				{parityMessage}
			</p>
		{/if}
		{#if stateCheckMessage}
			<p class="numeric-status" class:error={stateCheckIsError} role="status">
				{stateCheckMessage}
			</p>
		{/if}

		<div class="controls">
			<label class="gravity-control">
				<span>重力</span>
				<input
					type="range"
					min="0"
					max="180"
					step="5"
					bind:value={gravityMagnitude}
					disabled={phase !== 'ready'}
					aria-label="重力の大きさ"
				/>
				<output>{gravityMagnitude}</output>
			</label>
			<div class="buttons">
				<button type="button" onclick={togglePaused} disabled={phase !== 'ready'}>
					{paused ? '再開' : '一時停止'}
				</button>
				<button
					type="button"
					class="secondary"
					onclick={stepOnce}
					disabled={phase !== 'ready' || !paused || stateCheckPending}
				>
					1 ステップ
				</button>
				<button
					type="button"
					class="secondary"
					onclick={checkGpuState}
					disabled={phase !== 'ready' || !paused || stateCheckPending}
				>
					{stateCheckPending ? '確認中…' : '状態を確認'}
				</button>
			</div>
		</div>
	</section>

	<aside class="note">
		<strong>検証範囲:</strong> P2G・grid update・G2P の単一材料 Neo-Hookean 弾性モデルです。
		液体の構成則、水・油の分離、複数材料、選択・ドラッグ統合は未実装です。粒子状態の GPU
		readback は初回比較時と一時停止中の明示的な状態確認時だけ行い、通常の各フレームでは行いません。
	</aside>
</main>

<style>
	.lab {
		box-sizing: border-box;
		width: min(100%, 980px);
		min-height: 100vh;
		margin: 0 auto;
		padding: max(1rem, env(safe-area-inset-top)) max(1rem, env(safe-area-inset-right))
			max(2rem, env(safe-area-inset-bottom)) max(1rem, env(safe-area-inset-left));
		color: #e8f1fa;
	}

	.header {
		display: flex;
		align-items: flex-start;
		justify-content: space-between;
		gap: 1.5rem;
		margin-bottom: 1.25rem;
	}

	.eyebrow {
		margin: 0 0 0.35rem;
		color: #8ec9ff;
		font-size: 0.75rem;
		font-weight: 700;
		letter-spacing: 0.12em;
	}

	h1,
	h2,
	p {
		margin-top: 0;
	}

	h1 {
		margin-bottom: 0.45rem;
		font-size: clamp(1.5rem, 4vw, 2.2rem);
	}

	h2 {
		margin-bottom: 0.25rem;
		font-size: 1.1rem;
	}

	.intro {
		max-width: 38rem;
		margin-bottom: 0;
		color: #b8c7d8;
		line-height: 1.55;
	}

	nav {
		display: flex;
		flex-wrap: wrap;
		gap: 0.6rem;
		justify-content: flex-end;
	}

	a {
		color: #b7dcff;
		text-underline-offset: 0.2em;
	}

	.panel,
	.note {
		border: 1px solid #31445a;
		border-radius: 0.85rem;
		background: #121e2b;
	}

	.panel {
		padding: 1rem;
	}

	.panel-heading {
		display: flex;
		align-items: flex-start;
		justify-content: space-between;
		gap: 1rem;
		margin-bottom: 0.75rem;
	}

	.panel-heading p {
		margin-bottom: 0;
		color: #b8c7d8;
		font-size: 0.9rem;
	}

	.status {
		max-width: 48%;
		color: #9de2bc !important;
		text-align: right;
	}

	.status.error {
		color: #ffb4a9 !important;
	}

	.canvas-frame {
		overflow: hidden;
		width: 100%;
		border: 1px solid #41576d;
		border-radius: 0.55rem;
		background: #07111b;
	}

	canvas {
		display: block;
		width: 100%;
		aspect-ratio: 3 / 2;
		touch-action: none;
	}

	.parity {
		margin: 0.8rem 0 0;
		color: #ffe09a;
		font-variant-numeric: tabular-nums;
	}

	.parity.passed {
		color: #9de2bc;
	}

	.parity.failed {
		color: #ffb4a9;
	}

	.numeric-status {
		margin: 0.8rem 0 0;
		color: #9de2bc;
	}

	.numeric-status.error {
		color: #ffb4a9;
	}

	.controls {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 1rem;
		margin-top: 1rem;
	}

	.gravity-control {
		display: grid;
		grid-template-columns: auto minmax(8rem, 1fr) 3ch;
		align-items: center;
		gap: 0.65rem;
		width: min(100%, 24rem);
	}

	output {
		font-variant-numeric: tabular-nums;
		text-align: right;
	}

	.buttons {
		display: flex;
		flex-wrap: wrap;
		gap: 0.6rem;
	}

	button {
		min-height: 2.75rem;
		padding: 0.55rem 0.9rem;
		border: 1px solid #8ec9ff;
		border-radius: 0.5rem;
		background: #24608b;
		color: white;
		font: inherit;
		font-weight: 650;
	}

	button.secondary {
		background: transparent;
		color: #d5e9fa;
	}

	button:disabled {
		cursor: not-allowed;
		opacity: 0.5;
	}

	.note {
		margin-top: 1rem;
		padding: 0.9rem 1rem;
		color: #b8c7d8;
		line-height: 1.6;
	}

	.note strong {
		color: #e8f1fa;
	}

	@media (max-width: 620px) {
		.header,
		.panel-heading,
		.controls {
			flex-direction: column;
			align-items: stretch;
		}

		nav {
			justify-content: flex-start;
		}

		.status {
			max-width: none;
			text-align: left;
		}

		.gravity-control {
			width: 100%;
		}

		.buttons button {
			flex: 1;
		}
	}
</style>
