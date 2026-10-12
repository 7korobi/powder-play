<script lang="ts">
	import { onMount } from 'svelte';
	import { encodeMpmConstitutiveModel } from '#lib/mpm/materials.js';
	import {
		createMpmMlsGpuSolver,
		type MpmMlsGpuSelection
	} from '#lib/mpm/kernels/solver.js';
	import {
		stepMpm,
		type MpmParticle,
		type MpmStepConfig,
		type MpmVector2
	} from '#lib/mpm/reference/solver.js';
	import {
		createMpmCpuParitySample,
		createMpmGate4Scene,
		GATE4_GRID,
		GATE4_PARTICLE_CAPACITY,
		GATE4_PARTICLE_COUNT
	} from '#lib/mpm/scenes/gate4.js';

	const GRID = GATE4_GRID;
	const WORLD_WIDTH = GRID.cellsX * GRID.cellSize;
	const WORLD_HEIGHT = GRID.cellsY * GRID.cellSize;
	const FIXED_STEP_SECONDS = 1 / 240;
	const FRAME_INTERVAL_MS = 1000 / 30;
	const MAX_FRAME_SECONDS = 0.1;
	const MAX_STEPS_PER_FRAME = 8;
	const GPU_PARTICLE_CAPACITY = GATE4_PARTICLE_CAPACITY;
	const PICK_RADIUS = 14;
	const BRUSH_RADIUS = 24;
	const CLONE_OFFSET: MpmVector2 = { x: 72, y: 0 };
	const PARTICLE_MASS = 1;
	const PARTICLE_VOLUME = 25;
	const MATERIAL = { lambda: 20, mu: 10, model: 'neo-hookean' } as const;
	const PARITY_TOLERANCE = 0.02;

	type Phase = 'initializing' | 'ready' | 'error' | 'lost';
	type GpuSolver = Awaited<ReturnType<typeof createMpmMlsGpuSolver>>;

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

	let submittedStepCount = 0;
	let submittedParticleSteps = 0;
	let submittedStepCpuMilliseconds = 0;

	function submitStep(currentSolver: GpuSolver, gravity: MpmVector2): void {
		const startedAt = performance.now();
		currentSolver.step(FIXED_STEP_SECONDS, gravity);
		submittedStepCount += 1;
		submittedParticleSteps += currentSolver.particleCount;
		submittedStepCpuMilliseconds += performance.now() - startedAt;
	}

	function errorText(error: unknown): string {
		return error instanceof Error && error.message.trim()
			? error.message
			: 'GPU MLS-MPM の処理に失敗しました。';
	}

	function particleValues(particle: MpmParticle): number[] {
		const material = particle.material ?? MATERIAL;
		const density =
			material.model === 'water' || material.model === 'oil'
				? material.density
				: 0;
		const frictionCoefficient =
			material.model === 'powder' ? material.frictionCoefficient : 0;
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
			particle.F.m11,
			particle.plasticVolume ?? 1,
			encodeMpmConstitutiveModel(material.model ?? 'neo-hookean'),
			material.lambda,
			material.mu,
			density,
			particle.volume ?? PARTICLE_VOLUME,
			frictionCoefficient
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
	let particleCount = $state(GATE4_PARTICLE_COUNT);
	let particleCapacity = $state(GPU_PARTICLE_CAPACITY);
	let measuredFps = $state<number | null>(null);
	let measuredStepsPerSecond = $state<number | null>(null);
	let measuredParticleStepsPerSecond = $state<number | null>(null);
	let averageStepSubmitMs = $state<number | null>(null);
	let averageQueueCompletionMs = $state<number | null>(null);
	let selection = $state<MpmMlsGpuSelection | null>(null);
	let interactionMessage = $state('初回の CPU/GPU 1 ステップ比較が完了すると操作できます。');
	let interactionIsError = $state(false);

	let componentMounted = false;
	let activePointerId = $state<number | null>(null);
	let pointerGeneration = 0;
	let pointerPosition: MpmVector2 | null = null;
	let dragActive = $state(false);

	function worldPointFromEvent(event: PointerEvent): MpmVector2 | null {
		if (!canvas) return null;
		const bounds = canvas.getBoundingClientRect();
		if (bounds.width <= 0 || bounds.height <= 0) return null;

		return {
			x: Math.min(
				WORLD_WIDTH,
				Math.max(0, ((event.clientX - bounds.left) / bounds.width) * WORLD_WIDTH)
			),
			y: Math.min(
				WORLD_HEIGHT,
				Math.max(0, ((event.clientY - bounds.top) / bounds.height) * WORLD_HEIGHT)
			)
		};
	}

	function finishPointerInteraction(): boolean {
		pointerGeneration += 1;
		const pointerId = activePointerId;
		activePointerId = null;
		pointerPosition = null;

		let succeeded = true;
		if (dragActive) {
			dragActive = false;
			try {
				solver?.endDrag();
			} catch (error) {
				succeeded = false;
				const message = `ドラッグを終了できませんでした: ${errorText(error)}`;
				if (componentMounted) {
					interactionIsError = true;
					interactionMessage = message;
				} else {
					console.error(message, error);
				}
			}
		}

		if (pointerId !== null) {
			try {
				if (canvas?.hasPointerCapture?.(pointerId)) {
					canvas.releasePointerCapture(pointerId);
				}
			} catch {
				// The browser may already have released capture during cancellation.
			}
		}

		return succeeded;
	}

	function handlePointerMove(event: PointerEvent): void {
		if (activePointerId !== event.pointerId) return;
		const point = worldPointFromEvent(event);
		if (!point) return;
		pointerPosition = point;

		if (!dragActive || !solver) return;
		try {
			solver.updateDrag(point);
		} catch (error) {
			finishPointerInteraction();
			interactionIsError = true;
			interactionMessage = `ドラッグを更新できませんでした: ${errorText(error)}`;
		}
	}

	async function handlePointerDown(event: PointerEvent): Promise<void> {
		if (!event.isPrimary || event.button !== 0 || phase !== 'ready' || !solver) return;
		const point = worldPointFromEvent(event);
		if (!point) return;

		event.preventDefault();
		if (activePointerId !== null) finishPointerInteraction();
		selection = null;
		interactionIsError = false;
		interactionMessage = '近くの粒子を GPU 上で検索しています…';
		activePointerId = event.pointerId;
		pointerPosition = point;
		const requestGeneration = ++pointerGeneration;
		const currentSolver = solver;

		try {
			canvas?.setPointerCapture?.(event.pointerId);
		} catch {
			// Pointer capture is an enhancement; the pick still works without it.
		}

		try {
			const picked = await currentSolver.pick(point, PICK_RADIUS);
			if (
				!componentMounted ||
				solver !== currentSolver ||
				phase !== 'ready' ||
				pointerGeneration !== requestGeneration ||
				activePointerId !== event.pointerId
			) {
				return;
			}

			if (!picked) {
				finishPointerInteraction();
				interactionMessage = '粒子が見つかりません。クラスター上をタップしてください。';
				return;
			}

			const target = pointerPosition ?? point;
			dragActive = true;
			currentSolver.beginDrag(picked, target, BRUSH_RADIUS);
			selection = picked;
			interactionMessage = `オブジェクト ${picked.objectId} を選択しました。有限半径 ${BRUSH_RADIUS} の力でドラッグできます。`;
		} catch (error) {
			if (
				!componentMounted ||
				solver !== currentSolver ||
				pointerGeneration !== requestGeneration ||
				activePointerId !== event.pointerId
			) {
				return;
			}
			finishPointerInteraction();
			interactionIsError = true;
			interactionMessage = `粒子を選択できませんでした: ${errorText(error)}`;
		}
	}

	function handlePointerUp(event: PointerEvent): void {
		if (activePointerId !== event.pointerId) return;
		const wasDragging = dragActive;
		const finished = finishPointerInteraction();
		if (!finished) return;

		interactionIsError = false;
		if (wasDragging && selection) {
			interactionMessage = `オブジェクト ${selection.objectId} を選択中です。複製するか、再度ドラッグできます。`;
		} else if (!selection) {
			interactionMessage = '粒子の検索中にポインターが離れたため、操作をキャンセルしました。';
		}
	}

	function handlePointerCancel(event: PointerEvent): void {
		if (activePointerId !== event.pointerId) return;
		const finished = finishPointerInteraction();
		if (finished) {
			interactionIsError = false;
			interactionMessage = 'ポインター操作がキャンセルされました。ドラッグを終了しました。';
		}
	}

	function clearSelection(): void {
		const finished = finishPointerInteraction();
		selection = null;
		if (finished) {
			interactionIsError = false;
			interactionMessage = 'Escape キーで操作をキャンセルしました。粒子をタップすると再選択できます。';
		}
	}

	function handleWindowKeydown(event: KeyboardEvent): void {
		if (
			event.key !== 'Escape' ||
			(activePointerId === null && !dragActive && !selection)
		) {
			return;
		}

		event.preventDefault();
		clearSelection();
	}

	function cloneSelection(): void {
		const currentSolver = solver;
		const currentSelection = selection;
		if (
			!currentSolver ||
			!currentSelection ||
			phase !== 'ready' ||
			activePointerId !== null ||
			dragActive
		) {
			return;
		}

		interactionIsError = false;
		try {
			const cloned = currentSolver.clone(currentSelection, CLONE_OFFSET);
			particleCount = currentSolver.particleCount;
			particleCapacity = currentSolver.particleCapacity;
			if (cloned) {
				interactionMessage = `オブジェクト ${currentSelection.objectId} を複製しました。粒子数 ${particleCount} / 容量 ${particleCapacity}。`;
			} else {
				interactionIsError = true;
				interactionMessage = `容量上限のため複製できませんでした。粒子数 ${particleCount} / 容量 ${particleCapacity}。`;
			}
		} catch (error) {
			interactionIsError = true;
			interactionMessage = `複製に失敗しました: ${errorText(error)}`;
		}
	}

	function togglePaused(): void {
		paused = !paused;
	}

	function stepOnce(): void {
		if (phase !== 'ready' || !paused || !solver) return;

		try {
			submitStep(solver, { x: 0, y: gravityMagnitude });
			solver.render();
		} catch (error) {
			finishPointerInteraction();
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
			if (solver !== currentSolver || phase !== 'ready') return;
			stateCheckMessage = `${particles.length} 粒子の GPU 数値状態は正常です。`;
		} catch (error) {
			if (solver !== currentSolver || phase !== 'ready') return;
			stateCheckIsError = true;
			stateCheckMessage = `GPU 数値状態を確認できませんでした: ${errorText(error)}`;
			finishPointerInteraction();
			phase = 'error';
			phaseMessage = 'GPU 数値状態の異常を検出したため停止しました。';
		} finally {
			if (solver === currentSolver) stateCheckPending = false;
		}
	}

	onMount(() => {
		let mounted = true;
		componentMounted = true;
		let activeSolver: GpuSolver | null = null;
		let frameRequest: number | null = null;
		let lastFrameTimestamp = 0;
		let accumulator = 0;
		let lastRenderTimestamp = 0;
		let metricsStartTimestamp: number | null = null;
		let performanceFrameCount = 0;
		let queueWorkPending = false;
		let queueSampleCount = 0;
		let queueSampleTotalMs = 0;
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
			if (!mounted || phase === 'lost') return;
			finishPointerInteraction();
			phase = 'error';
			phaseMessage = errorText(error);
			console.error('GPU MLS-MPM initialization failed.', error);
		}

		function renderFrame(timestamp: number): void {
			if (!mounted || !activeSolver || phase !== 'ready') return;
			frameRequest = requestAnimationFrame(renderFrame);
			if (timestamp - lastRenderTimestamp < FRAME_INTERVAL_MS - 1) return;
			if (queueWorkPending) return;

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
						submitStep(activeSolver, { x: 0, y: gravityMagnitude });
					} catch (error) {
						finishPointerInteraction();
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
				finishPointerInteraction();
				phase = 'error';
				phaseMessage = `描画を停止しました: ${errorText(error)}`;
				return;
			}

			performanceFrameCount += 1;
			if (metricsStartTimestamp === null) metricsStartTimestamp = timestamp;
			const metricsElapsed = timestamp - metricsStartTimestamp;
			if (metricsElapsed >= 1_000) {
				const samplesPerSecond = 1_000 / metricsElapsed;
				measuredFps = performanceFrameCount * samplesPerSecond;
				measuredStepsPerSecond = submittedStepCount * samplesPerSecond;
				measuredParticleStepsPerSecond = submittedParticleSteps * samplesPerSecond;
				averageStepSubmitMs =
					submittedStepCount > 0
						? submittedStepCpuMilliseconds / submittedStepCount
						: null;
				averageQueueCompletionMs =
					queueSampleCount > 0 ? queueSampleTotalMs / queueSampleCount : null;

				metricsStartTimestamp = timestamp;
				performanceFrameCount = 0;
				submittedStepCount = 0;
				submittedParticleSteps = 0;
				submittedStepCpuMilliseconds = 0;
				queueSampleCount = 0;
				queueSampleTotalMs = 0;
			}

			const submittedSolver = activeSolver;
			queueWorkPending = true;
			void submittedSolver
				.measureQueueCompletionMs()
				.then((completionMs) => {
					if (!mounted || activeSolver !== submittedSolver) return;
					queueSampleCount += 1;
					queueSampleTotalMs += completionMs;
				})
				.catch((error) => {
					if (!mounted || activeSolver !== submittedSolver) return;
					fail(error);
				})
				.finally(() => {
					if (mounted && activeSolver === submittedSolver) queueWorkPending = false;
				});
		}

		function handleVisibilityChange(): void {
			lastFrameTimestamp = 0;
			accumulator = 0;
			if (
				document.visibilityState !== 'visible' &&
				(activePointerId !== null || dragActive)
			) {
				const finished = finishPointerInteraction();
				if (finished) {
					interactionIsError = false;
					interactionMessage = 'ページが非表示になったため、粒子操作をキャンセルしました。';
				}
			}
		}

		async function initialize(): Promise<void> {
			if (!canvasElement) {
				fail(new Error('GPU canvas が初期化されていません。'));
				return;
			}

			try {
				resizeCanvas();
				const initialScene = createMpmGate4Scene();
				const paritySample = createMpmCpuParitySample(
					initialScene.particles,
					initialScene.parityProbeIndices,
					GRID.cellSize
				);
				const initialParticles = initialScene.particles;
				const config = makeStepConfig(gravityMagnitude);
				const candidate = await createMpmMlsGpuSolver(
					canvasElement,
					initialParticles,
					{
						...config,
						capacity: GPU_PARTICLE_CAPACITY
					}
				);
				if (!mounted) {
					candidate.dispose();
					return;
				}

				activeSolver = candidate;
				solver = candidate;
				particleCount = candidate.particleCount;
				particleCapacity = candidate.particleCapacity;
				candidate.deviceLost.then((info) => {
					if (!mounted || activeSolver !== candidate) return;
					if (frameRequest !== null) cancelAnimationFrame(frameRequest);
					finishPointerInteraction();
					paused = true;
					phase = 'lost';
					phaseMessage = info.message || 'WebGPU デバイスが失われました。';
				});

				const expectedSample = stepMpm(paritySample.particles, config).particles;
				candidate.step(FIXED_STEP_SECONDS, config.gravity);
				const actual = await candidate.readParticles(initialScene.parityProbeIndices);
				if (!mounted || activeSolver !== candidate || phase === 'lost') return;

				const expected = paritySample.probeSampleIndices.map(
					(index) => expectedSample[index]
				);
				const maxError = maxParticleDifference(actual, expected);
				parityPassed = maxError <= PARITY_TOLERANCE;
				parityMessage = Number.isFinite(maxError)
					? `初回1ステップ・各材料中央3粒子の CPU reference 比較: 最大絶対誤差 ${maxError.toExponential(2)} ${
							parityPassed ? '· 許容範囲内' : '· 許容範囲外'
						}`
					: '初回1ステップの CPU reference 比較で粒子数または数値が一致しません。';
				submittedStepCount = 0;
				submittedParticleSteps = 0;
				submittedStepCpuMilliseconds = 0;
				queueSampleCount = 0;
				queueSampleTotalMs = 0;
				metricsStartTimestamp = null;
				performanceFrameCount = 0;
				averageQueueCompletionMs = null;
				phase = 'ready';
				phaseMessage = 'GPU MLS-MPM を実行中です · 20,000 粒子 · 目標 30 fps';
				interactionMessage = '準備完了。粒子をタップ／クリックして、押したままドラッグできます。';
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
			componentMounted = false;
			finishPointerInteraction();
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

<svelte:window
	onkeydown={handleWindowKeydown}
	onpointermove={handlePointerMove}
	onpointerup={handlePointerUp}
	onpointercancel={handlePointerCancel}
/>

<svelte:head>
	<title>GPU MLS-MPM 水・油・粉体ラボ · powder-play</title>
	<meta
		name="description"
		content="水・油・粉体の試作構成則を CPU reference と比較し、GPU 粒子操作を検証します。"
	/>
</svelte:head>

<main class="lab">
	<header class="header">
		<div>
			<p class="eyebrow">GATE 4 · WATER / OIL / POWDER</p>
			<h1>GPU 2D MLS-MPM 混合材料ラボ</h1>
			<p class="intro">
				水・油・粉体の20,000粒子を WebGPU で計算します。初回ステップは材料ごとの局所粒子で CPU
				reference と比較し、描画・粒子更新量・queue 完了待ちを実測できます。
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
				<h2 id="simulation-title">水・油・粉体の20,000粒子シーン</h2>
				<p>
					{particleCount} / {particleCapacity} 粒子 · {GRID.cellsX} × {GRID.cellsY} grid
				</p>
			</div>
			<p class="status" class:error={phase === 'error' || phase === 'lost'} role="status">
				{phaseMessage}
			</p>
		</div>

		<div class="canvas-frame">
			<canvas
				bind:this={canvas}
				tabindex="0"
				aria-label="GPU 上で計算する水・油の弱圧縮性流体と粉体の MLS-MPM シミュレーション"
				aria-describedby="interaction-instructions interaction-status"
				onpointerdown={handlePointerDown}
				onlostpointercapture={handlePointerCancel}
			>
				WebGPU canvas に対応したブラウザーが必要です。
			</canvas>
		</div>

		<p class="material-legend">
			<span><i class="water-swatch"></i> 水（基準密度 0.04）</span>
			<span><i class="oil-swatch"></i> 油（基準密度 0.032）</span>
			<span><i class="powder-swatch"></i> 粉体（摩擦係数 0.55）</span>
			水・油はせん断応力を持たず体積圧力だけを持つ試作モデルです。水・油の界面維持や浮沈は未実装です。
		</p>

		<section class="performance" aria-labelledby="performance-title">
			<h2 id="performance-title">GPU 実測性能（直近1秒）</h2>
			<dl class="performance-metrics">
				<div>
					<dt>描画</dt>
					<dd>
						{measuredFps === null ? '計測中' : `${measuredFps.toFixed(1)} FPS`}
					</dd>
				</div>
				<div>
					<dt>シミュレーション</dt>
					<dd>
						{measuredStepsPerSecond === null
							? '計測中'
							: `${measuredStepsPerSecond.toFixed(1)} steps/s`}
					</dd>
				</div>
				<div>
					<dt>粒子更新量（submit 換算）</dt>
					<dd>
						{measuredParticleStepsPerSecond === null
							? '計測中'
							: `${Math.round(measuredParticleStepsPerSecond)} particle-steps/s`}
					</dd>
				</div>
				<div>
					<dt>CPU step submit 平均</dt>
					<dd>
						{averageStepSubmitMs === null ? '—' : `${averageStepSubmitMs.toFixed(2)} ms`}
					</dd>
				</div>
				<div>
					<dt>GPU queue 完了待ち平均</dt>
					<dd>
						{averageQueueCompletionMs === null
							? '計測中'
							: `${averageQueueCompletionMs.toFixed(2)} ms`}
					</dd>
				</div>
			</dl>
			<p>
				queue 完了待ちは WebGPU の submit 後から処理完了までの時間で、GPU 単体の実行時間ではありません。
				完了を待ってから次の frame を送信し、遅い端末で GPU queue が無制限に膨らまないようにします。
			</p>
		</section>

		<div class="interaction-controls" aria-labelledby="interaction-title">
			<h2 id="interaction-title">粒子操作</h2>
			<p id="interaction-instructions">
				キャンバス内をタップ／クリックして選択し、押したままドラッグすると半径 {BRUSH_RADIUS}
				の範囲に力を加えます。Escape キー、ポインターキャンセル、またはページ非表示で操作を終了します。
			</p>
			<div class="interaction-actions">
				<button
					type="button"
					class="secondary"
					onclick={cloneSelection}
					disabled={phase !== 'ready' || !selection || activePointerId !== null || dragActive}
				>
					選択オブジェクトを複製（+{CLONE_OFFSET.x}, {CLONE_OFFSET.y}）
				</button>
				<p class="particle-count" aria-label="現在の粒子数と容量">
					粒子数 {particleCount} / {particleCapacity}（使用数 / GPU 容量）
				</p>
			</div>
			<p
				id="interaction-status"
				class="interaction-status"
				class:error={interactionIsError}
				role="status"
				aria-live="polite"
				aria-atomic="true"
			>
				{interactionMessage}
			</p>
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
		<strong>検証範囲:</strong> P2G・grid update・G2Pで水・油の弱圧縮性EOSと粉体のDrucker–Prager風
		せん断降伏を粒子ごとに計算します。objectIdによる選択・力ベースドラッグ・複製は密度・材料状態も保持します。
		材料間の明示的接触、界面張力・非混和性、浮沈、実粉体の校正、長時間安定性は未実装または未検証です。
		初回比較は材料ごとの局所サンプルだけを GPU から読み戻します。粒子状態の全量 GPU readback は一時停止中の明示的な状態確認時だけです。
	</aside>
</main>

<style>
	.lab {
		box-sizing: border-box;
		min-height: 100vh;
		min-height: 100svh;
		padding: max(1rem, env(safe-area-inset-top)) max(1rem, env(safe-area-inset-right))
			max(2rem, env(safe-area-inset-bottom)) max(1rem, env(safe-area-inset-left));
		color: #e8f1fa;
		background:
			radial-gradient(ellipse at 12% 0%, rgb(38 101 151 / 22%), transparent 38%),
			linear-gradient(145deg, #071321 0%, #0b1725 54%, #10182a 100%);
	}

	.header,
	.panel,
	.note {
		box-sizing: border-box;
		width: min(100%, 980px);
		margin-right: auto;
		margin-left: auto;
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

	.material-legend {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 0.75rem 1rem;
		margin: 0.75rem 0 0;
		color: #b8c7d8;
		font-size: 0.9rem;
		line-height: 1.5;
	}

	.material-legend span {
		display: inline-flex;
		align-items: center;
		gap: 0.4rem;
		color: #e8f1fa;
	}

	.material-legend i {
		width: 0.75rem;
		height: 0.75rem;
		border-radius: 50%;
	}

	.performance {
		margin-top: 1rem;
		padding: 0.9rem 1rem;
		border: 1px solid #31445a;
		border-radius: 0.65rem;
		background: #0d1823;
	}

	.performance h2 {
		margin-bottom: 0;
	}

	.performance-metrics {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 10rem), 1fr));
		gap: 0.55rem;
		margin: 0.65rem 0 0;
	}

	.performance-metrics div {
		min-width: 0;
		padding: 0.6rem 0.7rem;
		border: 1px solid #26394b;
		border-radius: 0.45rem;
	}

	.performance-metrics dt {
		color: #b8c7d8;
		font-size: 0.82rem;
	}

	.performance-metrics dd {
		margin: 0.25rem 0 0;
		font-variant-numeric: tabular-nums;
		overflow-wrap: anywhere;
	}

	.performance > p {
		margin: 0.65rem 0 0;
		color: #b8c7d8;
		font-size: 0.85rem;
		line-height: 1.5;
	}

	.water-swatch {
		background: #2e8cf2;
	}

	.oil-swatch {
		background: #f2bf38;
	}

	.powder-swatch {
		background: #b87a4d;
	}

	canvas {
		display: block;
		width: 100%;
		aspect-ratio: 3 / 2;
		touch-action: none;
	}

	canvas:focus-visible {
		outline: 3px solid #8ec9ff;
		outline-offset: -4px;
	}

	.interaction-controls {
		margin-top: 1rem;
		padding-top: 1rem;
		border-top: 1px solid #31445a;
	}

	.interaction-controls h2 {
		margin-bottom: 0.35rem;
	}

	.interaction-controls p {
		margin-bottom: 0;
		color: #b8c7d8;
		line-height: 1.5;
	}

	.interaction-actions {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 0.75rem;
		margin-top: 0.75rem;
	}

	.interaction-actions .particle-count {
		font-variant-numeric: tabular-nums;
	}

	.interaction-status {
		margin-top: 0.75rem;
		color: #9de2bc !important;
	}

	.interaction-status.error {
		color: #ffb4a9 !important;
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

		.interaction-actions {
			align-items: stretch;
			flex-direction: column;
		}

		.gravity-control {
			width: 100%;
		}

		.buttons button {
			flex: 1;
		}
	}
</style>
