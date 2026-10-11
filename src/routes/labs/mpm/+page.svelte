<script lang="ts">
	import { onMount } from 'svelte';
	import type {
		MpmBackend,
		MpmDragMode,
		MpmLabConfig,
		MpmParticleSeed,
		MpmSelection,
		MpmVector
	} from '#lib/mpm/contracts.js';

	const WORLD_WIDTH = 960;
	const WORLD_HEIGHT = 640;
	const FLUID_OBJECT_ID = 101;
	const GEL_OBJECT_ID = 202;
	const FRAME_SECONDS = 1 / 30;
	const FRAME_INTERVAL_MS = 1000 / 30;
	const MAX_FRAME_SECONDS = 0.1;
	const MAX_STEPS_PER_FRAME = 3;
	const PICK_RADIUS = 34;
	const PICKUP_OFFSET: MpmVector = { x: 42, y: -32 };
	const DEFAULT_UNAVAILABLE_MESSAGE =
		'このブラウザーまたは GPU デバイスでは WebGPU を利用できません。対応環境で再試行するか、メイン画面をご利用ください。';

	type LabPhase = 'checking' | 'initializing' | 'ready' | 'unsupported' | 'error' | 'lost';

	function makeCluster(
		start: MpmVector,
		columns: number,
		rows: number,
		spacing: number,
		materialId: number,
		objectId: number,
		staggerRows = false
	): MpmParticleSeed[] {
		const particles: MpmParticleSeed[] = [];

		for (let row = 0; row < rows; row += 1) {
			for (let column = 0; column < columns; column += 1) {
				particles.push({
					position: {
						x: start.x + column * spacing + (staggerRows && row % 2 === 1 ? spacing / 2 : 0),
						y: start.y + row * spacing
					},
					materialId,
					objectId
				});
			}
		}

		return particles;
	}

	const labConfig: MpmLabConfig = {
		worldWidth: WORLD_WIDTH,
		worldHeight: WORLD_HEIGHT,
		capacity: 160,
		particles: [
			...makeCluster({ x: 154, y: 174 }, 8, 6, 18, 0, FLUID_OBJECT_ID, true),
			...makeCluster({ x: 628, y: 286 }, 6, 5, 20, 1, GEL_OBJECT_ID)
		]
	};

	function errorText(error: unknown): string {
		return error instanceof Error && error.message.trim()
			? error.message
			: '詳細を取得できない GPU エラーが発生しました。';
	}

	function objectLabel(objectId: number): string {
		if (objectId === FLUID_OBJECT_ID) return '青い流体風クラスター';
		if (objectId === GEL_OBJECT_ID) return 'ゲル風クラスター';
		return `オブジェクト ${objectId}`;
	}

	function clamp(value: number, min: number, max: number): number {
		return Math.min(max, Math.max(min, value));
	}

	let canvas: HTMLCanvasElement | undefined;
	let backend: MpmBackend | null = null;
	let phase = $state<LabPhase>('checking');
	let phaseMessage = $state('');
	let paused = $state(false);
	let pageVisible = $state(true);
	let gravityMagnitude = $state(90);
	let dragMode = $state<MpmDragMode>('force');
	let selection = $state<MpmSelection | null>(null);
	let clonePending = $state(false);
	let actionMessage = $state('粒子をタップして選択し、そのままドラッグできます。');
	let actionIsError = $state(false);

	let componentMounted = false;
	let activePointerId: number | null = null;
	let pointerGeneration = 0;
	let pointerPosition: MpmVector | null = null;
	let dragActive = false;
	let keyboardTarget: MpmVector | null = null;

	const statusText = $derived(
		phase === 'checking'
			? 'WebGPU 対応を確認中…'
			: phase === 'initializing'
				? 'GPU を初期化中…'
				: phase === 'unsupported'
					? 'WebGPU を利用できません'
					: phase === 'error'
						? 'GPU 初期化エラー'
						: phase === 'lost'
							? 'GPU デバイスが失われました'
							: !pageVisible
								? 'ページ非表示中 · シミュレーションを停止中'
								: paused
									? '一時停止中'
									: '実行中 · 目標 30 fps'
	);

	function worldPointFromEvent(event: PointerEvent): MpmVector | null {
		if (!canvas) return null;
		const bounds = canvas.getBoundingClientRect();
		if (bounds.width <= 0 || bounds.height <= 0) return null;

		return {
			x: clamp(((event.clientX - bounds.left) / bounds.width) * WORLD_WIDTH, 0, WORLD_WIDTH),
			y: clamp(((event.clientY - bounds.top) / bounds.height) * WORLD_HEIGHT, 0, WORLD_HEIGHT)
		};
	}

	function finishPointerInteraction(): void {
		pointerGeneration += 1;
		const pointerId = activePointerId;
		activePointerId = null;
		keyboardTarget = pointerPosition ?? selection?.anchor ?? null;
		pointerPosition = null;

		if (dragActive) {
			dragActive = false;
			try {
				backend?.endDrag();
			} catch (error) {
				const message = `ドラッグを終了できませんでした: ${errorText(error)}`;
				if (componentMounted) {
					actionIsError = true;
					actionMessage = message;
				} else {
					console.error(message, error);
				}
			}
		}

		if (pointerId !== null && canvas?.hasPointerCapture(pointerId)) {
			canvas.releasePointerCapture(pointerId);
		}
	}

	function clearSelection(): void {
		finishPointerInteraction();
		selection = null;
		keyboardTarget = null;
		actionIsError = false;
		actionMessage = '選択を解除しました。粒子をタップすると再選択できます。';
	}

	function handlePointerMove(event: PointerEvent): void {
		if (activePointerId !== event.pointerId) return;
		const point = worldPointFromEvent(event);
		if (!point) return;
		pointerPosition = point;

		if (!dragActive || !backend) return;
		try {
			backend.updateDrag(point);
		} catch (error) {
			actionIsError = true;
			actionMessage = `ドラッグを更新できませんでした: ${errorText(error)}`;
			finishPointerInteraction();
		}
	}

	async function handlePointerDown(event: PointerEvent): Promise<void> {
		if (event.button !== 0 || phase !== 'ready' || !backend) return;
		const point = worldPointFromEvent(event);
		if (!point) return;

		event.preventDefault();
		if (activePointerId !== null) finishPointerInteraction();
		selection = null;
		keyboardTarget = null;
		actionIsError = false;
		actionMessage = '近くの粒子を GPU 上で検索しています…';
		activePointerId = event.pointerId;
		pointerPosition = point;
		const requestGeneration = ++pointerGeneration;
		const currentBackend = backend;

		try {
			canvas?.setPointerCapture(event.pointerId);
		} catch {
			// Pointer capture is an enhancement; the pick still works without it.
		}

		try {
			const picked = await currentBackend.pick(point, PICK_RADIUS);
			if (
				!componentMounted ||
				backend !== currentBackend ||
				pointerGeneration !== requestGeneration ||
				activePointerId !== event.pointerId
			) {
				return;
			}

			if (!picked) {
				finishPointerInteraction();
				actionMessage = '粒子が見つかりません。クラスター上をタップしてください。';
				return;
			}

			selection = picked;
			keyboardTarget = pointerPosition ?? picked.anchor;
			currentBackend.beginDrag(picked, keyboardTarget, dragMode);
			dragActive = true;
			actionMessage = `${objectLabel(picked.objectId)}を選択しました。ドラッグ中です。`;
		} catch (error) {
			if (
				!componentMounted ||
				backend !== currentBackend ||
				pointerGeneration !== requestGeneration
			) {
				return;
			}
			finishPointerInteraction();
			actionIsError = true;
			actionMessage = `粒子を選択できませんでした: ${errorText(error)}`;
		}
	}

	function handlePointerEnd(event: PointerEvent): void {
		if (activePointerId !== event.pointerId) return;
		const wasDragging = dragActive;
		finishPointerInteraction();
		if (wasDragging && selection) {
			actionMessage = `${objectLabel(selection.objectId)}を選択中です。複製または再度ドラッグできます。`;
		} else if (!selection) {
			actionMessage = '粒子をタップして選択し、そのままドラッグできます。';
		}
	}

	function handleCanvasKeydown(event: KeyboardEvent): void {
		if (event.key === 'Escape') {
			if (selection || activePointerId !== null) {
				event.preventDefault();
				clearSelection();
			}
			return;
		}

		const movement: Record<string, MpmVector> = {
			ArrowLeft: { x: -18, y: 0 },
			ArrowRight: { x: 18, y: 0 },
			ArrowUp: { x: 0, y: -18 },
			ArrowDown: { x: 0, y: 18 }
		};
		const delta = movement[event.key];
		if (!delta || !selection || phase !== 'ready' || !backend) return;

		event.preventDefault();
		const currentBackend = backend;
		const from = keyboardTarget ?? selection.anchor;
		const target = {
			x: clamp(from.x + delta.x, 0, WORLD_WIDTH),
			y: clamp(from.y + delta.y, 0, WORLD_HEIGHT)
		};

		try {
			currentBackend.beginDrag(selection, from, 'rigid');
			try {
				currentBackend.updateDrag(target);
				currentBackend.step(0, { x: 0, y: 0 });
				currentBackend.render();
			} finally {
				currentBackend.endDrag();
			}
			keyboardTarget = target;
			actionIsError = false;
			actionMessage = `${objectLabel(selection.objectId)}をキーボードでまとめて移動しました。`;
		} catch (error) {
			try {
				currentBackend.endDrag();
			} catch {
				// Keep the original operation error for the status message.
			}
			actionIsError = true;
			actionMessage = `選択対象を移動できませんでした: ${errorText(error)}`;
		}
	}

	function togglePaused(): void {
		if (phase !== 'ready') return;
		paused = !paused;
		actionIsError = false;
		actionMessage = paused ? 'シミュレーションを一時停止しました。' : 'シミュレーションを再開しました。';
	}

	function chooseDragMode(mode: MpmDragMode): void {
		dragMode = mode;
		actionIsError = false;
		actionMessage =
			mode === 'force'
				? 'ドラッグ時は局所的な力として操作します。'
				: 'ドラッグ時は選択オブジェクトをまとめて移動します。';
	}

	async function cloneSelection(): Promise<void> {
		if (!backend || !selection || phase !== 'ready' || clonePending) return;

		const currentBackend = backend;
		const currentSelection = selection;
		clonePending = true;
		actionIsError = false;
		actionMessage = `${objectLabel(currentSelection.objectId)}を複製しています…`;

		try {
			const cloned = await currentBackend.clone(currentSelection, PICKUP_OFFSET);
			if (!componentMounted || backend !== currentBackend) return;
			actionIsError = !cloned;
			actionMessage = cloned
				? '複製しました。容量上限に達すると複製できません。'
				: '複製できませんでした。粒子容量の上限に達している可能性があります。';
		} catch (error) {
			if (!componentMounted || backend !== currentBackend) return;
			actionIsError = true;
			actionMessage = `複製に失敗しました: ${errorText(error)}`;
		} finally {
			if (componentMounted) clonePending = false;
		}
	}

	onMount(() => {
		componentMounted = true;
		pageVisible = document.visibilityState !== 'hidden';

		let frameRequest: number | null = null;
		let previousFrameTime: number | null = null;
		let accumulatedSeconds = 0;
		let lastRenderTime = 0;

		function cancelFrame(): void {
			if (frameRequest === null) return;
			cancelAnimationFrame(frameRequest);
			frameRequest = null;
		}

		function failRuntime(message: string): void {
			phase = 'error';
			phaseMessage = message;
			paused = true;
			cancelFrame();
			finishPointerInteraction();
		}

		function scheduleFrame(): void {
			if (
				componentMounted &&
				pageVisible &&
				phase === 'ready' &&
				frameRequest === null
			) {
				frameRequest = requestAnimationFrame(renderFrame);
			}
		}

		function renderFrame(timestamp: number): void {
			frameRequest = null;
			if (!componentMounted || !pageVisible || phase !== 'ready' || !backend) return;

			const elapsedSeconds =
				previousFrameTime === null
					? 0
					: Math.min((timestamp - previousFrameTime) / 1000, MAX_FRAME_SECONDS);
			previousFrameTime = timestamp;

			try {
				if (paused) {
					accumulatedSeconds = 0;
				} else {
					accumulatedSeconds += elapsedSeconds;
					let steps = 0;
					while (accumulatedSeconds >= FRAME_SECONDS && steps < MAX_STEPS_PER_FRAME) {
						backend.step(FRAME_SECONDS, { x: 0, y: gravityMagnitude });
						accumulatedSeconds -= FRAME_SECONDS;
						steps += 1;
					}
					if (steps === MAX_STEPS_PER_FRAME && accumulatedSeconds >= FRAME_SECONDS) {
						accumulatedSeconds = 0;
					}
				}

				if (timestamp - lastRenderTime >= FRAME_INTERVAL_MS) {
					backend.render();
					lastRenderTime = timestamp;
				}
			} catch (error) {
				failRuntime(`GPU の更新または描画に失敗しました: ${errorText(error)}`);
				return;
			}

			scheduleFrame();
		}

		function reportDeviceLoss(message: string): void {
			if (!componentMounted) return;
			phase = 'lost';
			phaseMessage = message;
			paused = true;
			cancelFrame();
			finishPointerInteraction();
		}

		function handleVisibilityChange(): void {
			pageVisible = document.visibilityState !== 'hidden';
			previousFrameTime = null;
			accumulatedSeconds = 0;

			if (!pageVisible) {
				cancelFrame();
				finishPointerInteraction();
			} else {
				scheduleFrame();
			}
		}

		async function initializeBackend(): Promise<void> {
			let availability: { supported: boolean; reason: string };
			try {
				const platform = await import('#lib/mpm/platform/webgpu.js');
				availability = platform.getMpmWebGpuAvailability();
			} catch (error) {
				if (!componentMounted) return;
				phase = 'unsupported';
				phaseMessage = `WebGPU 対応状況を確認できませんでした: ${errorText(error)}`;
				return;
			}

			if (!componentMounted) return;
			if (!availability.supported) {
				phase = 'unsupported';
				phaseMessage = availability.reason || DEFAULT_UNAVAILABLE_MESSAGE;
				return;
			}
			if (!canvas) {
				phase = 'error';
				phaseMessage = '描画用キャンバスを初期化できませんでした。';
				return;
			}

			phase = 'initializing';
			let candidate: MpmBackend | null = null;
			try {
				const { GpuParticleBackend } = await import(
					'#lib/mpm/interaction/gpu-particle-backend.js'
				);
				if (!componentMounted) return;
				candidate = new GpuParticleBackend();
				backend = candidate;
				await candidate.initialize(canvas, labConfig);
				if (!componentMounted) {
					if (backend === candidate) {
						backend = null;
						try {
							candidate.dispose();
						} catch (error) {
							console.error('Failed to dispose the GPU backend after route teardown.', error);
						}
					}
					return;
				}

				phase = 'ready';
				phaseMessage = '';
				candidate.render();
				scheduleFrame();

				const deviceLost = candidate.deviceLost;
				if (deviceLost) {
					void deviceLost
						.then((info: GPUDeviceLostInfo) => {
							const detail = info.message?.trim();
							reportDeviceLoss(
								detail
									? `WebGPU デバイスが失われました: ${detail}`
									: `WebGPU デバイスが失われました (${info.reason})。メイン画面は引き続き利用できます。`
							);
						})
						.catch((error: unknown) => {
							reportDeviceLoss(`WebGPU デバイスの状態を確認できませんでした: ${errorText(error)}`);
						});
				}
			} catch (error) {
				let message = errorText(error);
				if (candidate && backend === candidate) {
					backend = null;
					try {
						candidate.dispose();
					} catch (disposeError) {
						message += ` Cleanup also failed: ${errorText(disposeError)}`;
					}
				}
				if (!componentMounted) return;
				phase = 'error';
				phaseMessage = `WebGPU の初期化に失敗しました: ${message}`;
			}
		}

		document.addEventListener('visibilitychange', handleVisibilityChange);
		void initializeBackend();

		return () => {
			componentMounted = false;
			document.removeEventListener('visibilitychange', handleVisibilityChange);
			cancelFrame();
			finishPointerInteraction();
			const currentBackend = backend;
			backend = null;
			try {
				currentBackend?.dispose();
			} catch (error) {
				console.error('Failed to dispose the GPU backend during route teardown.', error);
			}
		};
	});
</script>

<svelte:head>
	<title>GPU 粒子操作ラボ | powder-play</title>
	<meta
		name="description"
		content="WebGPU 粒子の選択・ドラッグ・複製を試す Gate-0 操作プロトタイプ"
	/>
</svelte:head>

<main class="lab-page">
	<div class="page-content">
		<header class="topbar">
			<a class="back-link" href="/">← メイン画面へ戻る</a>
			<span class="lab-tag">GATE 0 · GPU INTERACTION</span>
		</header>

		<section class="intro" aria-labelledby="page-title">
			<div class="intro-copy">
				<p class="eyebrow">独立した操作プロトタイプ</p>
				<h1 id="page-title">GPU 粒子操作ラボ</h1>
				<p class="lead">
					粒子の選択、ドラッグ、複製を GPU 上の状態で試すための小さな実験画面です。
				</p>
				<div class="truth-note">
					<strong>重要: 物理演算はまだ MLS-MPM ではありません。</strong>
					<span>
						この画面は操作経路の PoC です。数値カーネルは
						<a href="/labs/mpm/kernels">Gate 1 GPU MLS-MPM 実験</a>
						で別途検証します。
					</span>
				</div>
			</div>

			<div
				class="status-pill"
				class:status-error={phase === 'unsupported' || phase === 'error' || phase === 'lost'}
				role="status"
				aria-live="polite"
			>
				<span class="status-dot" aria-hidden="true"></span>
				{statusText}
			</div>
		</section>

		{#if phase === 'unsupported' || phase === 'error' || phase === 'lost'}
			<aside class="error-notice" role="alert">
				<div>
					<h2>
						{phase === 'unsupported'
							? 'このブラウザー / デバイスでは利用できません'
							: phase === 'lost'
								? 'GPU デバイスとの接続が失われました'
								: 'GPU プロトタイプを開始できませんでした'}
					</h2>
					<p>
						{phaseMessage || DEFAULT_UNAVAILABLE_MESSAGE}
						メイン画面のシミュレーションは引き続き利用できます。
					</p>
				</div>
				<a class="button button-secondary" href="/">メイン画面へ戻る</a>
			</aside>
		{/if}

		<div class="lab-layout">
			<section class="stage-card" aria-labelledby="scene-title">
				<div class="stage-heading">
					<div>
						<p class="eyebrow">合成シード · 2 clusters</p>
						<h2 id="scene-title">操作シーン</h2>
					</div>
					<span class="capacity">容量 {labConfig.capacity} 粒子</span>
				</div>

				<div class="canvas-frame">
					<canvas
						bind:this={canvas}
						width={WORLD_WIDTH}
						height={WORLD_HEIGHT}
						aria-label="GPU 粒子シーン。粒子をタップして選択し、ドラッグで移動できます。"
						aria-describedby="canvas-help"
						aria-busy={phase === 'checking' || phase === 'initializing'}
						tabindex="0"
						onpointerdown={handlePointerDown}
						onpointermove={handlePointerMove}
						onpointerup={handlePointerEnd}
						onpointercancel={handlePointerEnd}
						onlostpointercapture={handlePointerEnd}
						onkeydown={handleCanvasKeydown}
					>
						このシーンは WebGPU 対応ブラウザーで表示されます。MLS-MPM の物理演算は未実装です。
					</canvas>
				</div>

				<p class="canvas-help" id="canvas-help">
					タップ / クリックで選択してドラッグ。選択後は矢印キーで少し移動、Escape
					で選択解除できます。
				</p>

				<ul class="legend" aria-label="初期粒子クラスター">
					<li>
						<span class="legend-swatch fluid" aria-hidden="true"></span>
						<span>
							<strong>青 · 流体風</strong>
							<small>material 0 / object {FLUID_OBJECT_ID}</small>
						</span>
					</li>
					<li>
						<span class="legend-swatch gel" aria-hidden="true"></span>
						<span>
							<strong>紫 · ゲル風</strong>
							<small>material 1 / object {GEL_OBJECT_ID}</small>
						</span>
					</li>
				</ul>

				<p class="selection-line" aria-live="polite">
					{#if selection}
						選択中: {objectLabel(selection.objectId)} · particle {selection.particleIndex}
					{:else}
						粒子を選択すると、ここに選択状態が表示されます。
					{/if}
				</p>
			</section>

			<aside class="controls-card" aria-labelledby="controls-title">
				<div>
					<p class="eyebrow">INPUT &amp; STATE</p>
					<h2 id="controls-title">操作コントロール</h2>
				</div>

				<div class="button-row">
					<button
						class="button button-primary"
						type="button"
						disabled={phase !== 'ready'}
						onclick={togglePaused}
					>
						{paused ? '再開' : '一時停止'}
					</button>
					<button
						class="button button-secondary"
						type="button"
						disabled={!selection || phase !== 'ready' || clonePending}
						onclick={cloneSelection}
					>
						{clonePending ? '複製中…' : '選択オブジェクトを複製'}
					</button>
				</div>

				<fieldset class="control-group">
					<legend>ドラッグ方式</legend>
					<div class="segmented-control">
						<button
							type="button"
							aria-pressed={dragMode === 'force'}
							disabled={phase !== 'ready'}
							onclick={() => chooseDragMode('force')}
						>
							局所的な力
						</button>
						<button
							type="button"
							aria-pressed={dragMode === 'rigid'}
							disabled={phase !== 'ready'}
							onclick={() => chooseDragMode('rigid')}
						>
							まとめて移動
						</button>
					</div>
					<p class="control-hint">
						どちらの方式も選択中のクラスターに適用できます。流体風では局所的な力、ゲル風ではまとめて移動を試せます。
					</p>
				</fieldset>

				<div class="control-group gravity-control">
					<label for="gravity">
						<span>やさしい重力</span>
						<output for="gravity">{gravityMagnitude} units/s²</output>
					</label>
					<input
						id="gravity"
						type="range"
						min="0"
						max="240"
						step="10"
						bind:value={gravityMagnitude}
						disabled={phase !== 'ready'}
					/>
					<p class="control-hint">下向き重力の入力値を調整します。物理精度を示すものではありません。</p>
				</div>

				<div class="control-group selection-control">
					<div class="selection-heading">
						<span>選択状態</span>
						<button
							class="text-button"
							type="button"
							disabled={!selection}
							onclick={clearSelection}
						>
							解除
						</button>
					</div>
					<p class="selection-name">
						{selection ? objectLabel(selection.objectId) : '未選択'}
					</p>
					<p class="control-hint">
						複製は容量上限で失敗する場合があります。失敗時はこの欄に結果を表示します。
					</p>
				</div>

				<p class:error-feedback={actionIsError} class="action-feedback" aria-live="polite">
					{actionMessage}
				</p>

				<p class="physics-footnote">
					重力は操作 PoC の入力です。MLS-MPM の P2G / G2P、圧力、ゲル弾性は含まれません。
				</p>
			</aside>
		</div>

		<footer class="page-footer">
			<span>Gate-0 · GPU 状態の操作検証</span>
			<a href="/">通常のシミュレーションへ戻る</a>
		</footer>
	</div>
</main>

<style>
	.lab-page {
		box-sizing: border-box;
		min-height: 100vh;
		min-height: 100svh;
		padding: 28px clamp(14px, 3vw, 40px) 34px;
		background:
			radial-gradient(ellipse at 12% 0%, rgb(38 101 151 / 22%), transparent 38%),
			linear-gradient(145deg, #071321 0%, #0b1725 54%, #10182a 100%);
		color: #e8f0fb;
		font-family:
			Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
	}

	.page-content {
		width: min(1180px, 100%);
		margin: 0 auto;
	}

	.topbar,
	.stage-heading,
	.selection-heading,
	.page-footer {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 16px;
	}

	.topbar {
		margin-bottom: clamp(28px, 5vw, 52px);
	}

	a {
		color: #a9d8ff;
	}

	.back-link,
	.page-footer a {
		font-size: 0.9rem;
		font-weight: 650;
		text-decoration-thickness: 1px;
		text-underline-offset: 4px;
	}

	.back-link:focus-visible,
	.page-footer a:focus-visible,
	.button:focus-visible,
	.segmented-control button:focus-visible,
	.text-button:focus-visible,
	canvas:focus-visible,
	input:focus-visible {
		outline: 3px solid #75c8ff;
		outline-offset: 3px;
	}

	.lab-tag,
	.capacity,
	.eyebrow {
		color: #93a8c0;
		font-size: 0.72rem;
		font-weight: 750;
		letter-spacing: 0.12em;
		text-transform: uppercase;
	}

	.lab-tag,
	.capacity {
		letter-spacing: 0.08em;
	}

	.intro {
		display: flex;
		align-items: flex-start;
		justify-content: space-between;
		gap: 28px;
		margin-bottom: 26px;
	}

	.intro-copy {
		max-width: 750px;
	}

	.eyebrow {
		margin: 0 0 9px;
	}

	h1,
	h2,
	p {
		margin-top: 0;
	}

	h1 {
		margin-bottom: 12px;
		font-size: clamp(2rem, 5vw, 3.5rem);
		letter-spacing: -0.045em;
		line-height: 1.06;
	}

	h2 {
		margin-bottom: 0;
		font-size: 1.12rem;
		letter-spacing: -0.02em;
	}

	.lead {
		max-width: 660px;
		margin-bottom: 18px;
		color: #b7c6d8;
		font-size: clamp(0.98rem, 1.7vw, 1.12rem);
		line-height: 1.65;
	}

	.truth-note {
		display: grid;
		gap: 4px;
		max-width: 760px;
		padding: 12px 15px;
		border: 1px solid rgb(255 190 92 / 40%);
		border-radius: 12px;
		background: rgb(105 67 21 / 21%);
		color: #ffd28c;
		font-size: 0.9rem;
		line-height: 1.5;
	}

	.truth-note span {
		color: #f2dcb8;
	}

	.status-pill {
		display: inline-flex;
		flex: 0 0 auto;
		align-items: center;
		gap: 9px;
		max-width: 310px;
		padding: 9px 12px;
		border: 1px solid rgb(94 207 156 / 30%);
		border-radius: 999px;
		background: rgb(33 116 82 / 19%);
		color: #a9edc9;
		font-size: 0.82rem;
		font-weight: 700;
		line-height: 1.35;
	}

	.status-dot {
		width: 8px;
		height: 8px;
		flex: 0 0 auto;
		border-radius: 50%;
		background: currentColor;
		box-shadow: 0 0 12px currentColor;
	}

	.status-error {
		border-color: rgb(255 136 136 / 40%);
		background: rgb(119 36 45 / 25%);
		color: #ffb1b1;
	}

	.error-notice {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 20px;
		margin-bottom: 20px;
		padding: 16px 18px;
		border: 1px solid rgb(255 136 136 / 38%);
		border-radius: 15px;
		background: rgb(100 32 42 / 24%);
	}

	.error-notice h2 {
		margin-bottom: 5px;
		color: #ffd0d0;
	}

	.error-notice p {
		margin-bottom: 0;
		color: #edc1c1;
		line-height: 1.55;
		overflow-wrap: anywhere;
	}

	.lab-layout {
		display: grid;
		grid-template-columns: minmax(0, 1.72fr) minmax(270px, 0.78fr);
		align-items: start;
		gap: 18px;
	}

	.stage-card,
	.controls-card {
		border: 1px solid rgb(150 180 216 / 15%);
		border-radius: 20px;
		background: linear-gradient(155deg, rgb(20 35 53 / 94%), rgb(12 25 41 / 96%));
		box-shadow: 0 20px 55px rgb(0 0 0 / 20%);
	}

	.stage-card {
		min-width: 0;
		padding: clamp(14px, 2.4vw, 24px);
	}

	.stage-heading {
		margin-bottom: 15px;
	}

	.capacity {
		padding: 7px 10px;
		border: 1px solid rgb(155 184 218 / 18%);
		border-radius: 999px;
		color: #a8b9cd;
		white-space: nowrap;
	}

	.canvas-frame {
		overflow: hidden;
		border: 1px solid rgb(129 183 226 / 28%);
		border-radius: 15px;
		background: #07101d;
		box-shadow: inset 0 0 45px rgb(22 55 91 / 26%);
	}

	canvas {
		display: block;
		width: 100%;
		height: auto;
		aspect-ratio: 3 / 2;
		touch-action: none;
		user-select: none;
	}

	.canvas-help {
		margin: 11px 2px 16px;
		color: #9badc1;
		font-size: 0.83rem;
		line-height: 1.55;
	}

	.legend {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 11px;
		margin: 0;
		padding: 0;
		list-style: none;
	}

	.legend li {
		display: flex;
		align-items: center;
		gap: 10px;
		min-width: 0;
		padding: 11px 12px;
		border: 1px solid rgb(150 180 216 / 13%);
		border-radius: 12px;
		background: rgb(5 15 28 / 40%);
	}

	.legend-swatch {
		width: 13px;
		height: 13px;
		flex: 0 0 auto;
		border-radius: 50%;
		box-shadow: 0 0 13px currentColor;
	}

	.legend-swatch.fluid {
		background: #49a9ff;
		color: #49a9ff;
	}

	.legend-swatch.gel {
		background: #d28cff;
		color: #d28cff;
	}

	.legend strong,
	.legend small {
		display: block;
	}

	.legend strong {
		font-size: 0.88rem;
	}

	.legend small {
		margin-top: 3px;
		color: #9badc1;
		font-size: 0.72rem;
	}

	.selection-line {
		margin: 14px 0 0;
		padding-top: 13px;
		border-top: 1px solid rgb(150 180 216 / 13%);
		color: #b9c9db;
		font-size: 0.84rem;
		line-height: 1.5;
	}

	.controls-card {
		display: grid;
		gap: 20px;
		padding: 21px;
	}

	.button-row {
		display: grid;
		gap: 9px;
	}

	.button {
		display: inline-flex;
		min-height: 46px;
		align-items: center;
		justify-content: center;
		padding: 10px 14px;
		border: 1px solid transparent;
		border-radius: 11px;
		font: inherit;
		font-size: 0.9rem;
		font-weight: 750;
		line-height: 1.25;
		text-align: center;
		text-decoration: none;
		cursor: pointer;
		transition:
			background-color 140ms ease,
			border-color 140ms ease,
			transform 140ms ease;
	}

	.button:not(:disabled):active {
		transform: translateY(1px);
	}

	.button:disabled,
	.segmented-control button:disabled,
	.text-button:disabled {
		cursor: not-allowed;
		opacity: 0.48;
	}

	.button-primary {
		background: linear-gradient(135deg, #58aaff, #477eea);
		color: #071321;
	}

	.button-primary:not(:disabled):hover {
		background: linear-gradient(135deg, #78bcff, #6192f2);
	}

	.button-secondary {
		border-color: rgb(146 181 221 / 28%);
		background: rgb(20 40 62 / 80%);
		color: #d7e9fb;
	}

	.button-secondary:not(:disabled):hover {
		border-color: rgb(146 198 248 / 55%);
		background: rgb(30 57 84 / 90%);
	}

	.control-group {
		min-width: 0;
		margin: 0;
		padding: 0;
		border: 0;
	}

	.control-group legend,
	.gravity-control label,
	.selection-heading {
		display: flex;
		width: 100%;
		align-items: center;
		justify-content: space-between;
		gap: 10px;
		margin-bottom: 9px;
		color: #e0eaf6;
		font-size: 0.9rem;
		font-weight: 700;
	}

	.segmented-control {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 5px;
		padding: 4px;
		border: 1px solid rgb(150 180 216 / 16%);
		border-radius: 12px;
		background: rgb(5 14 26 / 52%);
	}

	.segmented-control button {
		min-height: 42px;
		padding: 8px 7px;
		border: 1px solid transparent;
		border-radius: 8px;
		background: transparent;
		color: #aebed0;
		font: inherit;
		font-size: 0.79rem;
		font-weight: 700;
		cursor: pointer;
	}

	.segmented-control button[aria-pressed="true"] {
		border-color: rgb(103 176 249 / 30%);
		background: rgb(53 105 160 / 34%);
		color: #d8edff;
	}

	.control-hint {
		margin: 8px 0 0;
		color: #93a7bd;
		font-size: 0.76rem;
		line-height: 1.5;
	}

	.gravity-control label {
		margin-bottom: 10px;
	}

	.gravity-control output {
		color: #9bcfff;
		font-variant-numeric: tabular-nums;
		font-size: 0.8rem;
		white-space: nowrap;
	}

	input[type="range"] {
		display: block;
		width: 100%;
		min-height: 30px;
		margin: 0;
		accent-color: #61adf6;
		cursor: pointer;
	}

	input[type="range"]:disabled {
		cursor: not-allowed;
		opacity: 0.5;
	}

	.selection-control {
		padding-top: 15px;
		border-top: 1px solid rgb(150 180 216 / 13%);
	}

	.selection-heading {
		margin-bottom: 6px;
	}

	.text-button {
		min-height: 36px;
		padding: 5px 8px;
		border: 0;
		border-radius: 7px;
		background: transparent;
		color: #9bcfff;
		font: inherit;
		font-size: 0.8rem;
		font-weight: 700;
		text-decoration: underline;
		text-underline-offset: 3px;
		cursor: pointer;
	}

	.selection-name {
		margin: 0;
		color: #f1f6fc;
		font-size: 0.95rem;
		font-weight: 750;
	}

	.action-feedback {
		min-height: 2.8em;
		margin: 0;
		padding: 11px 12px;
		border: 1px solid rgb(91 159 220 / 22%);
		border-radius: 10px;
		background: rgb(27 66 101 / 22%);
		color: #bcdcff;
		font-size: 0.79rem;
		line-height: 1.5;
		overflow-wrap: anywhere;
	}

	.action-feedback.error-feedback {
		border-color: rgb(255 136 136 / 30%);
		background: rgb(100 32 42 / 20%);
		color: #ffc0c0;
	}

	.physics-footnote {
		margin: -6px 0 0;
		color: #8298b0;
		font-size: 0.73rem;
		line-height: 1.5;
	}

	.page-footer {
		margin-top: 20px;
		padding: 0 4px;
		color: #8297ae;
		font-size: 0.78rem;
	}

	.page-footer a {
		font-size: inherit;
	}

	@media (max-width: 800px) {
		.intro {
			flex-direction: column;
			gap: 16px;
		}

		.status-pill {
			max-width: none;
		}

		.lab-layout {
			grid-template-columns: minmax(0, 1fr);
		}

		.controls-card {
			grid-template-columns: repeat(2, minmax(0, 1fr));
			align-items: start;
		}

		.controls-card > :first-child,
		.button-row,
		.selection-control,
		.action-feedback,
		.physics-footnote {
			grid-column: 1 / -1;
		}
	}

	@media (max-width: 520px) {
		.lab-page {
			padding: 17px 12px 24px;
		}

		.topbar {
			align-items: flex-start;
			margin-bottom: 29px;
		}

		.lab-tag {
			max-width: 125px;
			font-size: 0.62rem;
			line-height: 1.4;
			text-align: right;
		}

		.truth-note {
			font-size: 0.82rem;
		}

		.error-notice {
			align-items: stretch;
			flex-direction: column;
		}

		.controls-card {
			grid-template-columns: minmax(0, 1fr);
			padding: 17px;
		}

		.controls-card > * {
			grid-column: auto;
		}

		.legend {
			grid-template-columns: minmax(0, 1fr);
		}

		.stage-heading {
			align-items: flex-start;
		}

		.capacity {
			max-width: 112px;
			white-space: normal;
			text-align: center;
		}

		.page-footer {
			align-items: flex-start;
			flex-direction: column;
			gap: 8px;
		}
	}

	@media (prefers-reduced-motion: reduce) {
		.button {
			transition: none;
		}
	}
</style>
