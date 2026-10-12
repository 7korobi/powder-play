<script lang="ts">
	import { onMount } from 'svelte';
	import { encodeMpmConstitutiveModel } from '#lib/mpm/materials.js';
	import {
		createMpmMlsGpuSolver,
		type MpmMlsGpuRenderMode,
		type MpmMlsGpuSelection
	} from '#lib/mpm/kernels/solver.js';
	import {
		DEFAULT_HORIZONTAL_OSCILLATION_AMPLITUDE,
		HORIZONTAL_OSCILLATION_FREQUENCY_HZ,
		MAX_HORIZONTAL_OSCILLATION_AMPLITUDE,
		meanSinusoidalAcceleration
	} from '#lib/mpm/kernels/oscillation.js';
	import {
		MPM_RENDER_FPS_OPTIONS,
		MpmFramePacer,
		type MpmRenderFps
	} from '#lib/mpm/kernels/frame-pacer.js';
	import { calculateMpmStepBatch } from '#lib/mpm/kernels/step-scheduler.js';
	import {
		stepMpm,
		type MpmParticle,
		type MpmStepConfig,
		type MpmVector2
	} from '#lib/mpm/reference/solver.js';
	import {
		createMpmCpuParitySample,
		createMpmGate4Scene,
		GATE4_CLONE_OFFSET,
		GATE4_GRID,
		GATE4_PARTICLE_CAPACITY,
		GATE4_PARTICLE_COUNT,
		GATE4_PEBBLE_EDGE_LENGTH
	} from '#lib/mpm/scenes/gate4.js';

	const GRID = GATE4_GRID;
	const WORLD_WIDTH = GRID.cellsX * GRID.cellSize;
	const WORLD_HEIGHT = GRID.cellsY * GRID.cellSize;
	const FIXED_STEP_SECONDS = 1 / 240;
	const MAX_FRAME_SECONDS = 0.1;
	const MAX_SIMULATION_SPEED = 32;
	const MAX_STEPS_PER_FRAME = Math.ceil(
		MAX_SIMULATION_SPEED / (MPM_RENDER_FPS_OPTIONS[0] * FIXED_STEP_SECONDS)
	);
	const DEFAULT_RENDER_FPS: MpmRenderFps = 60;
	const DEFAULT_SIMULATION_SPEED = 2;
	const SIMULATION_SPEED_OPTIONS = [0.5, 1, 2, 4, 8, 16, MAX_SIMULATION_SPEED] as const;
	const GPU_PARTICLE_CAPACITY = GATE4_PARTICLE_CAPACITY;
	const PICK_RADIUS = 14;
	const BRUSH_RADIUS = 24;
	const PARTICLE_MASS = 1;
	const PARTICLE_VOLUME = 25;
	const MATERIAL = { lambda: 20, mu: 10, model: 'neo-hookean' } as const;
	const PARITY_ABSOLUTE_TOLERANCE = 0.02;
	const PARITY_VELOCITY_ABSOLUTE_TOLERANCE = 0.025;
	const PARITY_VELOCITY_RELATIVE_TOLERANCE = 0.0005;

	type Phase = 'initializing' | 'ready' | 'error' | 'lost';
	type GpuSolver = Awaited<ReturnType<typeof createMpmMlsGpuSolver>>;
	let simulatedTimeSeconds = 0;

	function makeStepConfig(gravityMagnitude: number, dt = FIXED_STEP_SECONDS): MpmStepConfig {
		return {
			dt,
			grid: GRID,
			particleMass: PARTICLE_MASS,
			particleVolume: PARTICLE_VOLUME,
			bondRestLength: GATE4_PEBBLE_EDGE_LENGTH,
			material: MATERIAL,
			gravity: { x: 0, y: gravityMagnitude }
		};
	}

	let submittedStepCount = 0;
	let submittedParticleSteps = 0;
	let submittedBatchCount = 0;
	let submittedBatchCpuMilliseconds = 0;

	function submitBatch(
		currentSolver: GpuSolver,
		stepCount: number,
		verticalGravity: number,
		oscillationAmplitude: number
	): void {
		const durationSeconds = stepCount * FIXED_STEP_SECONDS;
		const startedAt = performance.now();
		currentSolver.stepBatch(stepCount, FIXED_STEP_SECONDS, {
			x: meanSinusoidalAcceleration(
				simulatedTimeSeconds,
				durationSeconds,
				oscillationAmplitude
			),
			y: verticalGravity
		});
		simulatedTimeSeconds += durationSeconds;
		submittedBatchCount += 1;
		submittedStepCount += stepCount;
		submittedParticleSteps += currentSolver.particleCount * stepCount;
		submittedBatchCpuMilliseconds += performance.now() - startedAt;
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

	function particleDifferenceWithinTolerance(
		actual: readonly MpmParticle[],
		expected: readonly MpmParticle[]
	): boolean {
		if (actual.length !== expected.length) return false;

		for (let index = 0; index < actual.length; index += 1) {
			const actualValues = particleValues(actual[index]);
			const expectedValues = particleValues(expected[index]);

			for (let valueIndex = 0; valueIndex < actualValues.length; valueIndex += 1) {
				const actualValue = actualValues[valueIndex];
				const expectedValue = expectedValues[valueIndex];
				if (!Number.isFinite(actualValue) || !Number.isFinite(expectedValue)) {
					return false;
				}
				const isVelocity = valueIndex === 2 || valueIndex === 3;
				const tolerance = isVelocity
					? PARITY_VELOCITY_ABSOLUTE_TOLERANCE +
						PARITY_VELOCITY_RELATIVE_TOLERANCE * Math.abs(expectedValue)
					: PARITY_ABSOLUTE_TOLERANCE;
				if (Math.abs(actualValue - expectedValue) > tolerance) return false;
			}
		}

		return true;
	}

	let canvas: HTMLCanvasElement | undefined;
	let solver: GpuSolver | null = null;
	let phase = $state<Phase>('initializing');
	let phaseMessage = $state('WebGPU MLS-MPM を初期化しています…');
	let gravityMagnitude = $state(90);
	let simulationSpeed = $state(DEFAULT_SIMULATION_SPEED);
	let renderTargetFps: MpmRenderFps = $state(DEFAULT_RENDER_FPS);
	let horizontalOscillationAmplitude = $state(DEFAULT_HORIZONTAL_OSCILLATION_AMPLITUDE);
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
	let measuredDiscardedSimSecondsPerSecond = $state<number | null>(null);
	let averageBatchSubmitMs = $state<number | null>(null);
	let averageQueueCompletionMs = $state<number | null>(null);
	let renderMode = $state<MpmMlsGpuRenderMode>('surface');
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
			const cloned = currentSolver.clone(currentSelection, GATE4_CLONE_OFFSET);
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
			submitBatch(
				solver,
				1,
				gravityMagnitude,
				horizontalOscillationAmplitude
			);
			solver.render(renderMode);
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
			await currentSolver.checkNumericalStatus();
			if (solver !== currentSolver || phase !== 'ready') return;
			stateCheckMessage = 'GPU 数値状態は正常です。';
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
		const framePacer = new MpmFramePacer();
		let lastFrameTimestamp: number | null = null;
		let accumulator = 0;
		let discardedSimulationSeconds = 0;
		let metricsStartTimestamp: number | null = null;
		let performanceFrameCount = 0;
		let queueWorkPending = false;
		let queueSampleCount = 0;
		let queueSampleTotalMs = 0;
		let lastNumericalStatusCheckTimestamp = 0;
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
			console.error('GPU MLS-MPM runtime failed.', error);
		}

		function renderFrame(timestamp: number): void {
			if (!mounted || !activeSolver || phase !== 'ready') return;
			frameRequest = requestAnimationFrame(renderFrame);
			if (!framePacer.shouldRender(timestamp, renderTargetFps)) return;
			if (queueWorkPending) return;

			const elapsed =
				lastFrameTimestamp === null
					? 0
					: Math.min(
							Math.max(0, (timestamp - lastFrameTimestamp) / 1000),
							MAX_FRAME_SECONDS
						);
			lastFrameTimestamp = timestamp;

			if (!paused && document.visibilityState === 'visible') {
				try {
					const batch = calculateMpmStepBatch({
						accumulatorSeconds: accumulator,
						elapsedSeconds: elapsed,
						simulationSpeed,
						fixedStepSeconds: FIXED_STEP_SECONDS,
						maxSteps: Math.min(
							MAX_STEPS_PER_FRAME,
							Math.max(
								1,
								Math.ceil(
									(accumulator + simulationSpeed * elapsed) / FIXED_STEP_SECONDS
								)
							)
						)
					});
					accumulator = batch.remainderSeconds;
					discardedSimulationSeconds += batch.discardedSeconds;
					if (batch.stepCount > 0) {
						submitBatch(
							activeSolver,
							batch.stepCount,
							gravityMagnitude,
							horizontalOscillationAmplitude
						);
					}
				} catch (error) {
					finishPointerInteraction();
					phase = 'error';
					phaseMessage = `シミュレーションを停止しました: ${errorText(error)}`;
					return;
				}
			} else {
				accumulator = 0;
			}

			try {
				activeSolver.render(renderMode);
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
				measuredDiscardedSimSecondsPerSecond =
					discardedSimulationSeconds * samplesPerSecond;
				averageBatchSubmitMs =
					submittedBatchCount > 0
						? submittedBatchCpuMilliseconds / submittedBatchCount
						: null;
				averageQueueCompletionMs =
					queueSampleCount > 0 ? queueSampleTotalMs / queueSampleCount : null;

				metricsStartTimestamp = timestamp;
				performanceFrameCount = 0;
				submittedStepCount = 0;
				submittedParticleSteps = 0;
				submittedBatchCount = 0;
				submittedBatchCpuMilliseconds = 0;
				discardedSimulationSeconds = 0;
				queueSampleCount = 0;
				queueSampleTotalMs = 0;
			}

			const submittedSolver = activeSolver;
			queueWorkPending = true;
			void submittedSolver
				.measureQueueCompletionMs()
				.then(async (completionMs) => {
					if (!mounted || activeSolver !== submittedSolver) return;
					queueSampleCount += 1;
					queueSampleTotalMs += completionMs;
					if (timestamp - lastNumericalStatusCheckTimestamp >= 1_000) {
						lastNumericalStatusCheckTimestamp = timestamp;
						await submittedSolver.checkNumericalStatus();
					}
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
			lastFrameTimestamp = null;
			accumulator = 0;
			framePacer.reset();
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
				parityPassed = particleDifferenceWithinTolerance(actual, expected);
				parityMessage = Number.isFinite(maxError)
					? `初回1ステップ・各材料中央3粒子の CPU reference 比較: 最大絶対誤差 ${maxError.toExponential(2)} ${
							parityPassed ? '· 誤差基準内' : '· 誤差基準超過'
						}`
					: '初回1ステップの CPU reference 比較で粒子数または数値が一致しません。';
				submittedStepCount = 0;
				submittedParticleSteps = 0;
				submittedBatchCount = 0;
				submittedBatchCpuMilliseconds = 0;
				discardedSimulationSeconds = 0;
				queueSampleCount = 0;
				queueSampleTotalMs = 0;
				simulatedTimeSeconds = 0;
				metricsStartTimestamp = null;
				performanceFrameCount = 0;
				measuredDiscardedSimSecondsPerSecond = null;
				averageBatchSubmitMs = null;
				averageQueueCompletionMs = null;
				phase = 'ready';
				phaseMessage = `GPU MLS-MPM を実行中です · ${particleCount.toLocaleString()} 粒子`;
				interactionMessage = '準備完了。粒子をタップ／クリックして、押したままドラッグできます。';
				lastFrameTimestamp = null;
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
	<title>GPU MLS-MPM 水・油・弾性小石ラボ · powder-play</title>
	<meta
		name="description"
		content="水・油と、ランダム初速を持つ200個の Neo-Hookean 弾性小石を CPU reference と比較し、GPU 粒子操作を検証します。"
	/>
</svelte:head>

<main class="lab">
	<header class="header">
		<div>
			<p class="eyebrow">GATE 4 · WATER / OIL / ELASTIC PEBBLE</p>
			<h1>GPU 2D MLS-MPM 混合材料ラボ</h1>
			<p class="intro">
				水8,000粒子・油6,000粒子と、枠幅の3%（66 world units）の三粒子弾性小石200個
				（計600粒子）を WebGPU で計算します。初期速度は水・油の各成分±60、小石の各頂点±120
				world units/s の範囲で粒子ごとにランダムに与えます。小石頂点の辺長制約、水・油の相別格子接触、
				液体と小石の双方向接触、小石同士の三角形接触を GPU で解きます。初回ステップは各材料の局所粒子で
				CPU reference と比較し、描画・粒子更新量・queue 完了待ちを実測できます。
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
				<h2 id="simulation-title">水・油・200個の弾性小石・14,600粒子シーン</h2>
				<p>
					{particleCount} / {particleCapacity} 粒子 · {GRID.cellsX} × {GRID.cellsY} grid
				</p>
			</div>
			<p class="status" class:error={phase === 'error' || phase === 'lost'} role="status">
				{phaseMessage}
			</p>
		</div>

		<div class="simulation-layout">
			<section class="simulation-controls" aria-label="シミュレーション操作">
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
					<label class="simulation-speed-control">
						<span>速度</span>
						<select
							bind:value={simulationSpeed}
							disabled={phase !== 'ready'}
							aria-label="シミュレーション速度倍率"
						>
							{#each SIMULATION_SPEED_OPTIONS as speed}
								<option value={speed}>{speed}×</option>
							{/each}
						</select>
						<output>{simulationSpeed.toFixed(2)}×</output>
					</label>
					<label class="display-mode-control">
						<span>表示モード</span>
						<select
							bind:value={renderMode}
							disabled={phase !== 'ready'}
							aria-label="表示モード"
							aria-describedby="render-mode-description"
						>
							<option value="surface">水・油の自由表面</option>
							<option value="particles">粒子マーカー（診断）</option>
						</select>
					</label>
					<label class="render-fps-control">
						<span>描画目標FPS</span>
						<select
							bind:value={renderTargetFps}
							disabled={phase !== 'ready'}
							aria-label="描画目標フレームレート"
						>
							{#each MPM_RENDER_FPS_OPTIONS as fps}
								<option value={fps}>{fps} fps</option>
							{/each}
						</select>
						<output>{renderTargetFps} fps</output>
					</label>
					<label class="rocking-control">
						<span>左右揺れ · {HORIZONTAL_OSCILLATION_FREQUENCY_HZ} Hz</span>
						<input
							type="range"
							min="0"
							max={MAX_HORIZONTAL_OSCILLATION_AMPLITUDE}
							step="10"
							bind:value={horizontalOscillationAmplitude}
							disabled={phase !== 'ready'}
							aria-label="左右揺れの加速度"
							aria-valuetext={`±${horizontalOscillationAmplitude} world units/s²`}
						/>
						<output>±{horizontalOscillationAmplitude}</output>
					</label>
					<div class="buttons">
						<button
							type="button"
							onclick={togglePaused}
							disabled={phase !== 'ready'}
							aria-pressed={paused}
						>
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
						<button
							type="button"
							class="secondary"
							onclick={cloneSelection}
							disabled={phase !== 'ready' || !selection || activePointerId !== null || dragActive}
							aria-label={`選択オブジェクトを複製（+${GATE4_CLONE_OFFSET.x}, ${GATE4_CLONE_OFFSET.y}）`}
							title={`選択オブジェクトを複製（+${GATE4_CLONE_OFFSET.x}, ${GATE4_CLONE_OFFSET.y}）`}
						>
							選択を複製
						</button>
					</div>
				</div>
				<p id="render-mode-description" class="render-mode-note">
					自由表面は粒子の現在面積から GPU 上で作る表示専用フィールドです。青は水、金色は油を示し、表面表示では液体の粒子マーカーを重ねません。粒子マーカー（診断）では全材料の粒子位置を表示します。表示色・境界は物理的な表面張力や界面曲率を表しません。材料間接触は表示場とは別の物理 pass で解きます。
				</p>
			</section>

			<div class="canvas-frame">
				<canvas
					bind:this={canvas}
					tabindex="0"
					aria-label={`GPU MLS-MPM シミュレーション。現在の表示: ${
						renderMode === 'surface' ? '水・油の自由表面' : '粒子マーカー診断'
					}`}
					aria-describedby="interaction-instructions interaction-status render-mode-description"
					onpointerdown={handlePointerDown}
					onlostpointercapture={handlePointerCancel}
				>
					WebGPU canvas に対応したブラウザーが必要です。
				</canvas>
			</div>
		</div>

		<div class="interaction-controls" aria-labelledby="interaction-title">
			<h2 id="interaction-title">粒子操作</h2>
			<p id="interaction-instructions">
				キャンバス内をタップ／クリックして選択し、押したままドラッグすると半径 {BRUSH_RADIUS}
				の範囲に力を加えます。Escape キー、ポインターキャンセル、またはページ非表示で操作を終了します。
			</p>
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

		<p class="material-legend">
			<span><i class="water-swatch"></i> 水（密度 0.02・圧縮係数 40,000）</span>
			<span><i class="oil-swatch"></i> 油（密度 0.016・圧縮係数 30,000）</span>
			<span><i class="pebble-swatch"></i> 弾性小石200個（各3粒子・Neo-Hookean・λ 20・μ 10）</span>
			初期速度は水・油の各成分±60、小石の各頂点±120 world units/s の独立した擬似乱数です。小石頂点ごとの速度差により回転運動も初期化されます。小石は一辺・水平幅66 world units（枠幅の3%）の正三角形で、14列×15行の並び（最終行は4個）、中心間隔68 world unitsで配置します。各三角形は別々の弾性 object です。Neo-Hookean 粒子に3辺の位置制約を加え、液体粒子は三角形の符号付き距離場で投影し、法線 impulse を小石頂点へ反力として返します。小石同士は三角形接触で貫入を補正します。水と油は別々の格子速度場を持ち、相別体積勾配に沿う法線接触を解きます。接線摩擦・表面張力・濡れ・完全な界面分離・連続衝突検出は未実装で、長時間の安定性も未検証です。
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
					<dt>描画目標（上限）</dt>
					<dd>{renderTargetFps} FPS</dd>
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
					<dt>実効時間倍率</dt>
					<dd>
						{measuredStepsPerSecond === null
							? '計測中'
							: `${(measuredStepsPerSecond * FIXED_STEP_SECONDS).toFixed(2)}×`}
					</dd>
				</div>
				<div>
					<dt>実効時間倍率</dt>
					<dd>
						{measuredDiscardedSimSecondsPerSecond === null
							? '計測中'
							: `${measuredDiscardedSimSecondsPerSecond.toFixed(2)} sim s/s`}
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
					<dt>CPU batch submit 平均</dt>
					<dd>
						{averageBatchSubmitMs === null
							? '—'
							: `${averageBatchSubmitMs.toFixed(2)} ms`}
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
				描画目標は15・30・60・90・120 FPSから選択できます。実際の描画FPSはディスプレイの更新頻度と GPU 処理時間に制約されます。
				固定刻みは 1/240 秒のまま、複数ステップを1つの GPU batch にまとめ、フレーム間隔に応じて最大32×のシミュレーション速度を保てるようにします。
				上限を超えたシミュレーション時間は破棄し、実効時間倍率と破棄量に反映します。
				左右揺れはシミュレーション時間に同期し、batch区間の平均加速度を適用します（0で停止）。queue 完了待ちは GPU 単体の実行時間ではなく、完了を待ってから次の frame を送信します。数値異常フラグは1秒ごとに小さな状態だけを読み戻して監視します。
			</p>
		</section>

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

	</section>

	<aside class="note">
		<strong>検証範囲:</strong> P2G・grid update・G2Pで水・油の弱圧縮性EOSと200個の三粒子 Neo-Hookean 弾性小石を
		計算します。水–油の法線 impulse、小石–液体の符号付き距離接触と両側反力、小石–小石の三角形接触、各小石の辺長制約を含みます。objectIdによる選択・力ベースドラッグ・複製は密度・材料・結合情報も保持します。
		水・油の局所密度差による相対沈降・浮上は近似実装です。水と油が静止状態ですでに強く混合している場合の完全な分離は保証しません。
		水・油の圧縮係数は従来比10倍です。液体の変形面積比 det(F) は0.5〜2に制限します。
		小石の頂点間隔は約6.6格子セル幅です。各三角形は3点と辺長制約による離散近似であり、連続体メッシュではありません。接線摩擦・表面張力・濡れ・連続衝突検出、長時間安定性は未実装または未検証です。
		初回CPU/GPU比較は材料ごとの局所サンプルです。数値異常フラグは1秒ごとに状態のみを読み取り、粒子状態の全量 GPU readback は一時停止中の明示的な状態確認時だけです。
		CPU/GPU比較はCPUの倍精度とGPUの単精度による差を考慮し、位置・変形などは絶対誤差0.02、速度は絶対誤差0.025に参照速度の0.05%を加えた基準で判定します。表示される最大絶対誤差は判定基準とは別にそのまま表示します。
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

	.simulation-layout {
		display: grid;
		grid-template-areas: 'canvas controls';
		grid-template-columns: minmax(0, 1fr) minmax(17rem, 20rem);
		align-items: start;
		gap: 1rem;
	}

	.simulation-controls {
		grid-area: controls;
		min-width: 0;
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
		grid-area: canvas;
		overflow: hidden;
		min-width: 0;
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

	.pebble-swatch {
		background: #47c7b8;
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
		margin-top: 0.75rem;
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
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		align-items: center;
		gap: 0.75rem;
	}

	.gravity-control,
	.rocking-control,
	.simulation-speed-control,
	.display-mode-control,
	.render-fps-control {
		display: grid;
		grid-template-columns: minmax(0, 1fr) auto;
		align-items: center;
		gap: 0.35rem 0.65rem;
		min-width: 0;
	}

	.gravity-control > span,
	.rocking-control > span,
	.simulation-speed-control > span,
	.display-mode-control > span,
	.render-fps-control > span {
		grid-column: 1;
		grid-row: 1;
	}

	.gravity-control input,
	.rocking-control input,
	.simulation-speed-control select,
	.display-mode-control select,
	.render-fps-control select {
		grid-column: 1 / -1;
		grid-row: 2;
		width: 100%;
		min-width: 0;
	}

	.gravity-control input,
	.rocking-control input {
		margin: 0;
	}

	.gravity-control output,
	.rocking-control output,
	.simulation-speed-control output,
	.render-fps-control output {
		grid-column: 2;
		grid-row: 1;
	}

	.simulation-speed-control select,
	.display-mode-control select,
	.render-fps-control select {
		box-sizing: border-box;
		min-height: 2.75rem;
		padding: 0.45rem 0.6rem;
		border: 1px solid #31445a;
		border-radius: 0.45rem;
		color: inherit;
		background: #0b1725;
		font: inherit;
	}

	.simulation-speed-control select:focus-visible,
	.display-mode-control select:focus-visible,
	.render-fps-control select:focus-visible {
		outline: 3px solid #8ec9ff;
		outline-offset: 2px;
	}

	.render-mode-note {
		margin: 0.65rem 0 0;
		color: #b8c7d8;
		font-size: 0.84rem;
		line-height: 1.5;
	}

	output {
		font-variant-numeric: tabular-nums;
		text-align: right;
	}

	.buttons {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 0.55rem;
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

	.buttons button {
		width: 100%;
		min-width: 0;
		padding-right: 0.4rem;
		padding-left: 0.4rem;
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

	@media (max-width: 820px) {
		.simulation-layout {
			grid-template-areas:
				'controls'
				'canvas';
			grid-template-columns: minmax(0, 1fr);
		}

		.controls {
			grid-template-columns: repeat(2, minmax(0, 1fr));
		}

		.rocking-control {
			grid-column: 1 / -1;
		}
	}

	@media (max-width: 620px) {
		.header,
		.panel-heading {
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

	}
</style>
