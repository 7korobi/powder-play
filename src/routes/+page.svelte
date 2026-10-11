<script lang="ts">
	import { onMount } from 'svelte';
	import {
		BLEED_ZOOM,
		cameraScale,
		createCamera,
		screenToCell,
		screenToWorld,
		setCameraMode,
		updateCameraForPinch,
		worldToScreen,
		type Camera,
		type CameraMode,
		type ViewportSize
	} from '#lib/sim/camera.js';
	import {
		classifyTwoFingerGesture,
		modeForDialRotation,
		MODE_ORDER,
		shortestAngleDelta,
		type GestureKind,
		type InteractionMode
	} from '#lib/input/gestures.js';
	import {
		blendVector,
		screenAccelerationForce,
		screenGravityFromAcceleration
	} from '#lib/input/sensors.js';
	import {
		chooseInitialGridSize,
		AdaptiveResolutionController,
		isRenderFrameDue,
		type GridSize
	} from '#lib/perf/resolution.js';
	import { MATERIAL, materialRegistry } from '#lib/sim/materials.js';
	import {
		createFlowStroke,
		createInitialWorld,
		createSelectionPattern,
		moveSelection,
		placePattern,
		resampleWorld,
		selectConnected,
		stepWorld,
		type FlowStroke,
		type Point,
		type SelectionParticle,
		type SimulationForces,
		type SimulationWorld,
		type Vector
	} from '#lib/sim/world.js';

	const TARGET_FPS = 30;
	const FIXED_STEP_MS = 1000 / TARGET_FPS;
	const PLACE_INTERVAL_MS = 120;
	const MAX_FLOW_STROKES = 6;
	const VIEWPORT_ASPECT_TOLERANCE = 0.02;

	const modes: {
		id: InteractionMode;
		label: string;
		icon: string;
		shortcut: string;
		description: string;
	}[] = [
		{
			id: 'select',
			label: '選択',
			icon: '◎',
			shortcut: '1',
			description: '粒子を選び、ドラッグして移動'
		},
		{
			id: 'place',
			label: '配置',
			icon: '＋',
			shortcut: '2',
			description: '選択物を複製。未選択なら粉を配置'
		},
		{
			id: 'flow',
			label: '流通',
			icon: '↗',
			shortcut: '3',
			description: '線を引いて、粒子に短い流れを加える'
		}
	];

	interface PointerPoint extends Point {}

	interface TwoFingerGesture {
		ids: [number, number];
		kind: 'pending' | GestureKind;
		baseDistance: number;
		lastAngle: number;
		rotation: number;
		baseCentroid: Point;
		baseCamera: Camera;
		startingMode: InteractionMode;
	}

	interface WorldDrawCache {
		world: SimulationWorld | null;
		cellRevision: number;
		particlePaths: Map<number, Path2D>;
	}

	type OrientationLock =
		| 'any'
		| 'natural'
		| 'landscape'
		| 'portrait'
		| 'portrait-primary'
		| 'portrait-secondary'
		| 'landscape-primary'
		| 'landscape-secondary';

	let canvasElement: HTMLCanvasElement;
	let activeMode = $state<InteractionMode>('select');
	let viewMode = $state<CameraMode>('fit');
	let isPaused = $state(false);
	let orientationMismatch = $state(false);
	let sensorEnabled = $state(false);
	let sensorStatus = $state('センサー未接続。矢印キーでも重力を操作できます。');
	let statusMessage = $state('同じ物質の粒子を選択してドラッグできます。');
	let errorMessage = $state('');
	let dialPreview = $state<InteractionMode | null>(null);
	let selectionCount = $state(0);
	let selectedMaterialName = $state<string | null>(null);
	let fps = $state(0);
	let gridSize = $state<GridSize>({ width: 0, height: 0 });

	let world: SimulationWorld | null = null;
	let camera: Camera = { centerX: 0, centerY: 0, zoom: 1 };
	let flows: FlowStroke[] = [];
	let flowDraft: Point[] = [];
	let selection: number[] = [];
	let protectedSelection: ReadonlySet<number> = new Set();
	let selectionPattern: SelectionParticle[] = [];
	let activePointerId: number | null = null;
	let lastDragCell: Point | null = null;
	let repeatTimer: number | null = null;
	let lastPlaceCell: Point | null = null;
	let gesture: TwoFingerGesture | null = null;
	let initialOrientation = '';
	let sensorGravity: Vector = { x: 0, y: 1 };
	let hasSensorGravity = false;
	let gravityKeys = new Set<string>();
	let pointerPositions = new Map<number, PointerPoint>();
	let ignoredPointers = new Set<number>();
	let forces: SimulationForces = {
		gravity: { x: 0, y: 1 },
		acceleration: { x: 0, y: 0 },
		rotationRate: 0
	};

	function describeMode(mode: InteractionMode): string {
		return modes.find((entry) => entry.id === mode)?.description ?? '';
	}

	function setMode(mode: InteractionMode): void {
		activeMode = mode;
		dialPreview = null;
		statusMessage = describeMode(mode);
	}

	function viewportSize(): ViewportSize {
		const rect = canvasElement.getBoundingClientRect();
		return { width: rect.width, height: rect.height };
	}

	function localPointerPoint(event: PointerEvent): Point {
		const rect = canvasElement.getBoundingClientRect();
		return { x: event.clientX - rect.left, y: event.clientY - rect.top };
	}

	function cellAt(point: Point): Point | null {
		if (!world) return null;
		return screenToCell(point, camera, viewportSize(), world.width, world.height);
	}

	function worldPointAt(point: Point): Point | null {
		if (!world) return null;
		const result = screenToWorld(point, camera, viewportSize(), world.width, world.height);
		return result.x >= 0 && result.y >= 0 && result.x < world.width && result.y < world.height
			? result
			: null;
	}

	function updateSelection(nextSelection: number[]): void {
		selection = nextSelection;
		selectionCount = nextSelection.length;
		if (world && nextSelection.length > 0) {
			protectedSelection = new Set(nextSelection);
			selectedMaterialName = materialRegistry.get(world.cells[nextSelection[0]]).name;
			selectionPattern = createSelectionPattern(world, nextSelection);
		} else {
			protectedSelection = new Set();
			selectedMaterialName = null;
			selectionPattern = [];
		}
	}

	function selectionPatternOrPowder(): SelectionParticle[] {
		return selectionPattern.length > 0
			? selectionPattern
			: [{ dx: 0, dy: 0, material: MATERIAL.powder }];
	}

	function placeAt(point: Point): void {
		if (!world) return;
		const cell = cellAt(point);
		if (!cell) return;
		const didPlace = placePattern(world, selectionPatternOrPowder(), cell.x, cell.y);
		statusMessage = didPlace
			? selectedMaterialName
				? `${selectedMaterialName}を複製しました。`
				: '粉を配置しました。'
			: '配置できません。空いている場所を選んでください。';
	}

	function addDraftPoint(point: Point): void {
		const worldPoint = worldPointAt(point);
		if (!worldPoint) return;
		const previous = flowDraft[flowDraft.length - 1];
		if (!previous || Math.hypot(worldPoint.x - previous.x, worldPoint.y - previous.y) >= 0.7) {
			flowDraft = [...flowDraft, worldPoint];
		}
	}

	function clearRepeatTimer(): void {
		if (repeatTimer !== null) {
			window.clearInterval(repeatTimer);
			repeatTimer = null;
		}
	}

	function beginSinglePointer(event: PointerEvent, point: Point): void {
		if (!world) return;
		activePointerId = event.pointerId;
		if (activeMode === 'select') {
			const cell = cellAt(point);
			if (!cell) {
				updateSelection([]);
				lastDragCell = null;
				statusMessage = '選択を解除しました。';
				return;
			}
			const nextSelection = selectConnected(world, cell.x, cell.y);
			updateSelection(nextSelection);
			lastDragCell = nextSelection.length > 0 ? cell : null;
			if (nextSelection.length > 0) {
				statusMessage = `${selectedMaterialName}を選択中。ドラッグで移動し、選択中は固定します。`;
			} else {
				statusMessage =
					world.get(cell.x, cell.y) === MATERIAL.air
						? '空気は透明のため選択できません。選択を解除しました。'
						: '選択を解除しました。';
			}
		} else if (activeMode === 'place') {
			lastPlaceCell = cellAt(point);
			placeAt(point);
			repeatTimer = window.setInterval(() => {
				const latest = pointerPositions.get(event.pointerId);
				if (latest) placeAt(latest);
			}, PLACE_INTERVAL_MS);
		} else {
			flowDraft = [];
			addDraftPoint(point);
		}
	}

	function beginTwoFingerGesture(): void {
		if (pointerPositions.size !== 2) return;
		clearRepeatTimer();
		flowDraft = [];
		activePointerId = null;
		lastDragCell = null;
		const entries = [...pointerPositions.entries()];
		const first = entries[0][1];
		const second = entries[1][1];
		const centroid = midpoint(first, second);
		const angle = Math.atan2(second.y - first.y, second.x - first.x);
		gesture = {
			ids: [entries[0][0], entries[1][0]],
			kind: 'pending',
			baseDistance: Math.max(1, distance(first, second)),
			lastAngle: angle,
			rotation: 0,
			baseCentroid: centroid,
			baseCamera: { ...camera },
			startingMode: activeMode
		};
	}

	function updateTwoFingerGesture(): void {
		if (!world || !gesture) return;
		const first = pointerPositions.get(gesture.ids[0]);
		const second = pointerPositions.get(gesture.ids[1]);
		if (!first || !second) return;
		const centroid = midpoint(first, second);
		const nextAngle = Math.atan2(second.y - first.y, second.x - first.x);
		gesture.rotation += shortestAngleDelta(gesture.lastAngle, nextAngle);
		gesture.lastAngle = nextAngle;

		const currentDistance = Math.max(1, distance(first, second));
		if (gesture.kind === 'pending') {
			const kind = classifyTwoFingerGesture(
				gesture.rotation,
				currentDistance / gesture.baseDistance - 1,
				distance(gesture.baseCentroid, centroid)
			);
			if (!kind) return;
			gesture.kind = kind;
		}

		if (gesture.kind === 'dial') {
			dialPreview = modeForDialRotation(gesture.startingMode, gesture.rotation);
			return;
		}
		camera = updateCameraForPinch(
			gesture.baseCamera,
			gesture.baseDistance,
			currentDistance,
			gesture.baseCentroid,
			centroid,
			viewportSize(),
			world.width,
			world.height
		);
	}

	function pointerDown(event: PointerEvent): void {
		if (event.pointerType === 'mouse' && event.button !== 0) return;
		event.preventDefault();
		canvasElement.setPointerCapture(event.pointerId);
		if (ignoredPointers.size > 0 || pointerPositions.size >= 2) {
			ignoredPointers.add(event.pointerId);
			return;
		}

		const point = localPointerPoint(event);
		pointerPositions.set(event.pointerId, point);
		if (pointerPositions.size === 1) {
			beginSinglePointer(event, point);
		} else {
			beginTwoFingerGesture();
		}
	}

	function pointerMove(event: PointerEvent): void {
		if (ignoredPointers.has(event.pointerId) || !pointerPositions.has(event.pointerId)) return;
		const point = localPointerPoint(event);
		pointerPositions.set(event.pointerId, point);
		if (gesture) {
			updateTwoFingerGesture();
			return;
		}
		if (activePointerId !== event.pointerId) return;

		if (activeMode === 'select' && world && lastDragCell) {
			const nextCell = cellAt(point);
			if (!nextCell) return;
			const deltaX = nextCell.x - lastDragCell.x;
			const deltaY = nextCell.y - lastDragCell.y;
			if (deltaX === 0 && deltaY === 0) return;
			const moved = moveSelection(world, selection, deltaX, deltaY);
			if (moved) {
				updateSelection(moved);
				lastDragCell = nextCell;
			}
		} else if (activeMode === 'place') {
			const nextCell = cellAt(point);
			if (
				nextCell &&
				(!lastPlaceCell || nextCell.x !== lastPlaceCell.x || nextCell.y !== lastPlaceCell.y)
			) {
				placeAt(point);
				lastPlaceCell = nextCell;
			}
		} else if (activeMode === 'flow') {
			addDraftPoint(point);
		}
	}

	function finishTwoFingerGesture(pointerId: number): void {
		if (!gesture || !gesture.ids.includes(pointerId)) return;
		if (gesture.kind === 'dial' && dialPreview) setMode(dialPreview);
		else if (gesture.kind === 'camera') statusMessage = '表示位置と倍率を変更しました。';
		gesture = null;
		dialPreview = null;
		for (const remainingId of pointerPositions.keys()) {
			if (remainingId !== pointerId) ignoredPointers.add(remainingId);
		}
		pointerPositions.clear();
		activePointerId = null;
		lastDragCell = null;
	}

	function finishSinglePointer(event: PointerEvent): void {
		clearRepeatTimer();
		if (activeMode === 'flow') {
			addDraftPoint(localPointerPoint(event));
			const flow = createFlowStroke(flowDraft);
			if (flow) {
				flows = [...flows, flow].slice(-MAX_FLOW_STROKES);
				statusMessage = '始点から終点へ向かう流れを作りました。';
			}
			flowDraft = [];
		}
		activePointerId = null;
		lastDragCell = null;
		lastPlaceCell = null;
	}

	function pointerUp(event: PointerEvent): void {
		if (ignoredPointers.delete(event.pointerId)) return;
		if (gesture) {
			finishTwoFingerGesture(event.pointerId);
			return;
		}
		if (!pointerPositions.has(event.pointerId)) return;
		if (activePointerId === event.pointerId) finishSinglePointer(event);
		pointerPositions.delete(event.pointerId);
	}

	function cancelPointerSequence(pointerId: number): void {
		if (ignoredPointers.delete(pointerId)) return;
		if (!pointerPositions.has(pointerId) && !gesture?.ids.includes(pointerId)) return;
		clearRepeatTimer();
		flowDraft = [];
		dialPreview = null;
		gesture = null;
		for (const remainingId of pointerPositions.keys()) {
			if (remainingId !== pointerId) ignoredPointers.add(remainingId);
		}
		pointerPositions.clear();
		activePointerId = null;
		lastDragCell = null;
		lastPlaceCell = null;
	}

	function pointerCancel(event: PointerEvent): void {
		cancelPointerSequence(event.pointerId);
	}

	function lostPointerCapture(event: PointerEvent): void {
		cancelPointerSequence(event.pointerId);
	}

	function toggleViewMode(mode: CameraMode): void {
		if (!world) return;
		viewMode = mode;
		camera = setCameraMode(world.width, world.height, mode);
		statusMessage =
			mode === 'bleed'
				? `幕裏表示です（ズーム ${BLEED_ZOOM.toFixed(1)}×）。二本指で移動できます。`
				: 'ワールド全体を表示しています。';
	}

	function zoomBy(factor: number): void {
		if (!world) return;
		const viewport = viewportSize();
		const nextZoom = Math.min(4, Math.max(1, camera.zoom * factor));
		const center = { x: viewport.width / 2, y: viewport.height / 2 };
		camera = updateCameraForPinch(
			camera,
			1,
			factor,
			center,
			center,
			viewport,
			world.width,
			world.height
		);
		camera = { ...camera, zoom: nextZoom };
	}

	function orientationType(): string {
		return window.screen.orientation?.type ?? (window.innerWidth >= window.innerHeight ? 'landscape' : 'portrait');
	}

	function checkOrientation(): void {
		if (!initialOrientation) return;
		const changed = orientationType() !== initialOrientation;
		if (changed) {
			orientationMismatch = true;
			isPaused = true;
			statusMessage = '画面の向きが変わったため一時停止しました。元の向きに戻してください。';
		} else if (orientationMismatch) {
			orientationMismatch = false;
			statusMessage = '元の向きに戻りました。内容を確認して再開してください。';
		}
	}

	function orientationLockTarget(type: string): OrientationLock {
		switch (type) {
			case 'portrait-primary':
			case 'portrait-secondary':
			case 'landscape-primary':
			case 'landscape-secondary':
				return type;
			case 'natural':
			case 'any':
				return window.innerWidth >= window.innerHeight ? 'landscape' : 'portrait';
			case 'landscape':
			case 'portrait':
				return type;
			default:
				return window.innerWidth >= window.innerHeight ? 'landscape' : 'portrait';
		}
	}

	function requestOrientationLock(): Promise<'locked' | 'unsupported' | 'failed'> {
		const orientation = window.screen.orientation as ScreenOrientation & {
			lock?: (value: OrientationLock) => Promise<void>;
		};
		if (typeof orientation?.lock !== 'function') return Promise.resolve('unsupported');
		try {
			return orientation.lock(orientationLockTarget(initialOrientation)).then(
				() => 'locked',
				(error: unknown) => {
					console.warn('The browser could not lock the current screen orientation.', error);
					return 'failed';
				}
			);
		} catch (error) {
			console.warn('The browser could not lock the current screen orientation.', error);
			return Promise.resolve('failed');
		}
	}

	async function enableSensors(userGesture: boolean): Promise<void> {
		if (!window.isSecureContext) {
			sensorStatus = 'センサーには HTTPS または localhost が必要です。矢印キーで操作できます。';
			return;
		}
		const MotionEventConstructor = window.DeviceMotionEvent as
			| (typeof DeviceMotionEvent & {
					requestPermission?: () => Promise<'granted' | 'denied'>;
			  })
			| undefined;
		if (!MotionEventConstructor) {
			sensorStatus = 'このブラウザーはモーションセンサーに対応していません。';
			return;
		}
		const requestPermission = MotionEventConstructor.requestPermission;
		if (requestPermission && !userGesture) {
			sensorStatus = 'センサーを使うには、ボタンをタップして許可してください。矢印キーでも操作できます。';
			return;
		}

		try {
			const permissionRequest =
				requestPermission
					? requestPermission.call(MotionEventConstructor)
					: Promise.resolve('granted' as const);
			const orientationRequest = requestOrientationLock();
			const [permissionResult, orientationResult] = await Promise.allSettled([
				permissionRequest,
				orientationRequest
			]);
			if (permissionResult.status === 'rejected') throw permissionResult.reason;
			if (permissionResult.value !== 'granted') {
				if (orientationResult.status === 'fulfilled' && orientationResult.value === 'locked') {
					const orientation = window.screen.orientation as ScreenOrientation & {
						unlock?: () => void;
					};
					orientation?.unlock?.();
				}
				sensorStatus = 'センサーが許可されませんでした。矢印キーで操作できます。';
				return;
			}

			if (orientationResult.status === 'rejected') {
				console.warn('The browser could not lock the current screen orientation.', orientationResult.reason);
			}
			const lockResult =
				orientationResult.status === 'fulfilled' ? orientationResult.value : 'failed';
			window.addEventListener('devicemotion', handleDeviceMotion);
			sensorGravity = { x: 0, y: 0 };
			hasSensorGravity = false;
			if (gravityKeys.size === 0) forces = { ...forces, gravity: sensorGravity };
			sensorEnabled = true;
			const startMessage = userGesture ? 'センサーを使用中。' : 'センサーを自動開始しました。';
			sensorStatus =
				lockResult === 'locked'
					? `${startMessage}画面向きを固定しました。`
					: lockResult === 'failed'
						? `${startMessage}向きを固定できないため、変更時は一時停止します。`
						: `${startMessage}向き固定は非対応のため、変更時は一時停止します。`;
		} catch (error) {
			console.error('Could not enable device-motion sensors.', error);
			sensorStatus = 'センサーの許可に失敗しました。ブラウザー設定を確認してください。';
		}
	}

	async function toggleSensors(): Promise<void> {
		if (sensorEnabled) {
			window.removeEventListener('devicemotion', handleDeviceMotion);
			const orientation = window.screen.orientation as ScreenOrientation & { unlock?: () => void };
			orientation?.unlock?.();
			sensorEnabled = false;
			hasSensorGravity = false;
			forces = { ...forces, acceleration: { x: 0, y: 0 }, rotationRate: 0 };
			updateGravityFromKeyboard();
			sensorStatus = 'センサーを停止しました。矢印キーで操作できます。';
			return;
		}
		await enableSensors(true);
	}

	function handleDeviceMotion(event: DeviceMotionEvent): void {
		const angle = window.screen.orientation?.angle ?? 0;
		const acceleration = event.accelerationIncludingGravity;
		if (acceleration && acceleration.x !== null && acceleration.y !== null) {
			const nextGravity = screenGravityFromAcceleration(acceleration.x, acceleration.y, angle);
			sensorGravity = hasSensorGravity
				? blendVector(sensorGravity, nextGravity, 0.18)
				: nextGravity;
			hasSensorGravity = true;
			if (gravityKeys.size === 0) forces = { ...forces, gravity: sensorGravity };
		}
		const linear = event.acceleration;
		const rotationRate = event.rotationRate?.gamma ?? 0;
		forces = {
			...forces,
			acceleration: screenAccelerationForce(linear?.x ?? 0, linear?.y ?? 0, angle),
			rotationRate
		};
	}

	function updateGravityFromKeyboard(): void {
		const x =
			(gravityKeys.has('ArrowRight') ? 1 : 0) - (gravityKeys.has('ArrowLeft') ? 1 : 0);
		const y = Number(gravityKeys.has('ArrowDown')) - Number(gravityKeys.has('ArrowUp'));
		const magnitude = Math.hypot(x, y);
		const gravity =
			magnitude > 0
				? { x: x / magnitude, y: y / magnitude }
				: sensorEnabled
					? sensorGravity
					: { x: 0, y: 1 };
		forces = { ...forces, gravity };
	}

	function keyboardDown(event: KeyboardEvent): void {
		const target = event.target;
		if (target instanceof HTMLElement && (target.isContentEditable || target.tagName === 'BUTTON')) {
			return;
		}
		if (event.code.startsWith('Arrow')) {
			event.preventDefault();
			gravityKeys.add(event.code);
			updateGravityFromKeyboard();
			return;
		}
		if (event.repeat) return;
		if (event.key === '1' || event.key === '2' || event.key === '3') {
			setMode(MODE_ORDER[Number(event.key) - 1]);
		} else if (event.key === '+' || event.key === '=') {
			zoomBy(1.15);
		} else if (event.key === '-' || event.key === '_') {
			zoomBy(1 / 1.15);
		} else if (event.key === 'Escape' && selection.length > 0) {
			updateSelection([]);
			statusMessage = '選択を解除しました。';
		} else if (event.code === 'Space' && !orientationMismatch) {
			event.preventDefault();
			isPaused = !isPaused;
			statusMessage = isPaused ? 'シミュレーションを一時停止しました。' : describeMode(activeMode);
		}
	}

	function keyboardUp(event: KeyboardEvent): void {
		if (!event.code.startsWith('Arrow')) return;
		gravityKeys.delete(event.code);
		updateGravityFromKeyboard();
	}

	function midpoint(first: Point, second: Point): Point {
		return { x: (first.x + second.x) / 2, y: (first.y + second.y) / 2 };
	}

	function distance(first: Point, second: Point): number {
		return Math.hypot(first.x - second.x, first.y - second.y);
	}

	function drawScene(
		context: CanvasRenderingContext2D,
		buffer: HTMLCanvasElement,
		bufferContext: CanvasRenderingContext2D,
		image: ImageData,
		canvas: HTMLCanvasElement,
		currentWorld: SimulationWorld,
		currentCamera: Camera,
		currentSelection: readonly number[],
		currentFlows: readonly FlowStroke[],
		currentDraft: readonly Point[],
		palette: Uint8ClampedArray,
		paletteReady: Uint8Array,
		particleRadii: Float32Array,
		worldDrawCache: WorldDrawCache
	): void {
		const rect = canvas.getBoundingClientRect();
		if (rect.width <= 0 || rect.height <= 0) return;
		const pixelRatio = Math.min(2, window.devicePixelRatio || 1);
		const pixelWidth = Math.max(1, Math.round(rect.width * pixelRatio));
		const pixelHeight = Math.max(1, Math.round(rect.height * pixelRatio));
		if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
			canvas.width = pixelWidth;
			canvas.height = pixelHeight;
		}
		context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
		context.clearRect(0, 0, rect.width, rect.height);
		const worldChanged =
			worldDrawCache.world !== currentWorld ||
			worldDrawCache.cellRevision !== currentWorld.cellRevision;
		if (worldChanged) {
			worldDrawCache.particlePaths.clear();
			for (let index = 0; index < currentWorld.cells.length; index++) {
				const material = currentWorld.cells[index];
				if (paletteReady[material] === 0) {
					const plugin = materialRegistry.get(material);
					palette.set(plugin.color, material * 4);
					particleRadii[material] = plugin.particleRadius ?? -1;
					paletteReady[material] = 1;
				}
				const colorOffset = material * 4;
				const pixelOffset = index * 4;
				const radius = particleRadii[material];
				if (radius >= 0) {
					image.data[pixelOffset] = 0;
					image.data[pixelOffset + 1] = 0;
					image.data[pixelOffset + 2] = 0;
					image.data[pixelOffset + 3] = 0;
					let path = worldDrawCache.particlePaths.get(material);
					if (!path) {
						path = new Path2D();
						worldDrawCache.particlePaths.set(material, path);
					}
					const x = index % currentWorld.width;
					const y = Math.floor(index / currentWorld.width);
					path.moveTo(x + 0.5 + radius, y + 0.5);
					path.arc(x + 0.5, y + 0.5, radius, 0, Math.PI * 2);
					continue;
				}
				image.data[pixelOffset] = palette[colorOffset];
				image.data[pixelOffset + 1] = palette[colorOffset + 1];
				image.data[pixelOffset + 2] = palette[colorOffset + 2];
				image.data[pixelOffset + 3] = palette[colorOffset + 3];
			}
			bufferContext.putImageData(image, 0, 0);
			worldDrawCache.world = currentWorld;
			worldDrawCache.cellRevision = currentWorld.cellRevision;
		}
		const viewport = { width: rect.width, height: rect.height };
		const scale = cameraScale(currentCamera, viewport, currentWorld.width, currentWorld.height);
		const left = rect.width / 2 - currentCamera.centerX * scale;
		const top = rect.height / 2 - currentCamera.centerY * scale;
		context.imageSmoothingEnabled = false;
		context.drawImage(buffer, left, top, currentWorld.width * scale, currentWorld.height * scale);
		if (worldDrawCache.particlePaths.size > 0) {
			context.save();
			context.translate(left, top);
			context.scale(scale, scale);
			for (const [material, path] of worldDrawCache.particlePaths) {
				const colorOffset = material * 4;
				context.fillStyle = `rgba(${palette[colorOffset]}, ${palette[colorOffset + 1]}, ${
					palette[colorOffset + 2]
				}, ${palette[colorOffset + 3] / 255})`;
				context.fill(path);
			}
			context.restore();
		}

		context.save();
		context.fillStyle = 'rgba(255, 204, 91, 0.30)';
		context.strokeStyle = 'rgba(255, 245, 211, 0.95)';
		context.lineWidth = Math.min(2, Math.max(0.65, 1 / scale));
		for (const index of currentSelection) {
			const point = worldToScreen(
				{ x: index % currentWorld.width, y: Math.floor(index / currentWorld.width) },
				currentCamera,
				viewport,
				currentWorld.width,
				currentWorld.height
			);
			context.fillRect(point.x, point.y, scale, scale);
			context.strokeRect(point.x + 0.5, point.y + 0.5, Math.max(0, scale - 1), Math.max(0, scale - 1));
		}

		for (const flow of currentFlows) drawFlow(context, flow, currentCamera, viewport, currentWorld);
		if (currentDraft.length > 1) {
			drawFlow(
				context,
				{
					points: [...currentDraft],
					direction: { x: 1, y: 0 },
					width: 2,
					strength: 0.7,
					remainingSteps: 1
				},
				currentCamera,
				viewport,
				currentWorld
			);
		}
		context.restore();
	}

	function drawFlow(
		context: CanvasRenderingContext2D,
		flow: FlowStroke,
		currentCamera: Camera,
		viewport: ViewportSize,
		currentWorld: SimulationWorld
	): void {
		if (flow.points.length < 2) return;
		const scale = cameraScale(currentCamera, viewport, currentWorld.width, currentWorld.height);
		const fade = flow.remainingSteps > 0 ? Math.min(1, flow.remainingSteps / 12) : 1;
		context.beginPath();
		for (let index = 0; index < flow.points.length; index++) {
			const point = worldToScreen(
				flow.points[index],
				currentCamera,
				viewport,
				currentWorld.width,
				currentWorld.height
			);
			if (index === 0) context.moveTo(point.x, point.y);
			else context.lineTo(point.x, point.y);
		}
		context.strokeStyle = `rgba(27, 111, 123, ${0.45 * fade})`;
		context.lineWidth = Math.max(2, flow.width * scale);
		context.lineCap = 'round';
		context.lineJoin = 'round';
		context.stroke();
	}

	onMount(() => {
		initialOrientation = orientationType();

		const context = canvasElement.getContext('2d', { alpha: true });
		const buffer = document.createElement('canvas');
		const bufferContext = buffer.getContext('2d', { alpha: true });
		if (!context || !bufferContext) {
			errorMessage = 'このブラウザーでは 2D Canvas を初期化できません。';
			return;
		}

		const resolution = new AdaptiveResolutionController(TARGET_FPS);
		const palette = new Uint8ClampedArray(256 * 4);
		const paletteReady = new Uint8Array(256);
		const particleRadii = new Float32Array(256).fill(-1);
		const worldDrawCache: WorldDrawCache = {
			world: null,
			cellRevision: -1,
			particlePaths: new Map()
		};
		let bufferImage: ImageData | null = null;
		let animationFrame = 0;
		let measurementFrame = 0;
		let previousFrameTime: number | null = null;
		let lastFpsUpdate = 0;
		let frameCount = 0;
		let accumulator = 0;
		const clearGravityKeys = (): void => {
			gravityKeys.clear();
			updateGravityFromKeyboard();
		};

		const resizeCanvasBuffer = (): void => {
			if (!world) return;
			const bufferChanged =
				!bufferImage ||
				buffer.width !== world.width ||
				buffer.height !== world.height;
			if (bufferChanged) {
				buffer.width = world.width;
				buffer.height = world.height;
				bufferImage = bufferContext.createImageData(world.width, world.height);
			}
		};

		const changeResolution = (nextSize: GridSize): void => {
			if (!world || (nextSize.width === world.width && nextSize.height === world.height)) return;
			const scaleX = nextSize.width / world.width;
			const scaleY = nextSize.height / world.height;
			world = resampleWorld(world, nextSize.width, nextSize.height);
			camera = {
				...camera,
				centerX: camera.centerX * scaleX,
				centerY: camera.centerY * scaleY
			};
			flows = flows.map((flow) => ({
				...flow,
				width: flow.width * ((scaleX + scaleY) / 2),
				points: flow.points.map((point) => ({ x: point.x * scaleX, y: point.y * scaleY }))
			}));
			flowDraft = flowDraft.map((point) => ({ x: point.x * scaleX, y: point.y * scaleY }));
			updateSelection([]);
			gridSize = nextSize;
			resizeCanvasBuffer();
		};

		const syncWorldToCanvas = (): void => {
			const bounds = canvasElement.getBoundingClientRect();
			if (bounds.width <= 0 || bounds.height <= 0) return;
			const nextSize = chooseInitialGridSize(
				bounds.width,
				bounds.height,
				navigator.hardwareConcurrency || 4
			);
			if (!world) {
				gridSize = nextSize;
				world = createInitialWorld(nextSize.width, nextSize.height);
				camera = createCamera(world.width, world.height, viewMode);
				resizeCanvasBuffer();
				return;
			}
			if (orientationMismatch) return;

			const viewportAspect = bounds.width / bounds.height;
			const worldAspect = world.width / world.height;
			if (Math.abs(viewportAspect / worldAspect - 1) <= VIEWPORT_ASPECT_TOLERANCE) return;
			changeResolution(nextSize);
		};

		const render = (time: number): void => {
			if (!world || !bufferImage) {
				syncWorldToCanvas();
				if (!world || !bufferImage) {
					animationFrame = window.requestAnimationFrame(render);
					return;
				}
			}
			if (!isRenderFrameDue(time, previousFrameTime, TARGET_FPS)) {
				animationFrame = window.requestAnimationFrame(render);
				return;
			}
			const frameStarted = performance.now();
			const elapsed =
				previousFrameTime === null ? 0 : Math.min(100, time - previousFrameTime);
			previousFrameTime = time;
			frameCount++;
			if (!isPaused) {
				accumulator += elapsed;
				let iterations = 0;
				while (accumulator >= FIXED_STEP_MS && iterations < 2) {
					stepWorld(world, forces, flows, Math.random, materialRegistry, protectedSelection);
					flows = flows.filter((flow) => flow.remainingSteps > 0);
					accumulator -= FIXED_STEP_MS;
					iterations++;
				}
			} else {
				accumulator = 0;
			}

			if (time - lastFpsUpdate >= 1000) {
				fps = Math.round((frameCount * 1000) / Math.max(1, time - lastFpsUpdate));
				frameCount = 0;
				lastFpsUpdate = time;
			}
			drawScene(
				context,
				buffer,
				bufferContext,
				bufferImage,
				canvasElement,
				world,
				camera,
				selection,
				flows,
				flowDraft,
				palette,
				paletteReady,
				particleRadii,
				worldDrawCache
			);
			if (!isPaused) {
				const nextSize = resolution.observe(performance.now() - frameStarted, {
					width: world.width,
					height: world.height
				});
				if (nextSize && pointerPositions.size === 0) changeResolution(nextSize);
			}
			animationFrame = window.requestAnimationFrame(render);
		};

		window.addEventListener('resize', checkOrientation);
		window.screen.orientation?.addEventListener('change', checkOrientation);
		window.addEventListener('resize', syncWorldToCanvas);
		window.visualViewport?.addEventListener('resize', syncWorldToCanvas);
		const resizeObserver =
			typeof ResizeObserver === 'undefined'
				? null
				: new ResizeObserver(syncWorldToCanvas);
		resizeObserver?.observe(canvasElement);
		syncWorldToCanvas();
		measurementFrame = window.requestAnimationFrame(() => {
			measurementFrame = 0;
			syncWorldToCanvas();
		});
		window.addEventListener('keydown', keyboardDown);
		window.addEventListener('keyup', keyboardUp);
		window.addEventListener('blur', clearGravityKeys);
		animationFrame = window.requestAnimationFrame(render);
		void enableSensors(false);

		return () => {
			window.cancelAnimationFrame(animationFrame);
			if (measurementFrame !== 0) window.cancelAnimationFrame(measurementFrame);
			resizeObserver?.disconnect();
			window.removeEventListener('resize', checkOrientation);
			window.screen.orientation?.removeEventListener('change', checkOrientation);
			window.removeEventListener('resize', syncWorldToCanvas);
			window.visualViewport?.removeEventListener('resize', syncWorldToCanvas);
			window.removeEventListener('keydown', keyboardDown);
			window.removeEventListener('keyup', keyboardUp);
			window.removeEventListener('blur', clearGravityKeys);
			window.removeEventListener('devicemotion', handleDeviceMotion);
			const orientation = window.screen.orientation as ScreenOrientation & { unlock?: () => void };
			orientation?.unlock?.();
			clearRepeatTimer();
		};
	});
</script>

<svelte:head>
	<title>powder-play — 粉・水・油・空気で遊ぶ</title>
	<meta
		name="description"
		content="スマートフォンの傾きやタッチで、粉・水・油・空気を動かして遊ぶブラウザーゲーム。"
	/>
	<meta name="theme-color" content="#b9e8ed" />
</svelte:head>

<main class="play-area">
	<canvas
		bind:this={canvasElement}
		class="world-canvas"
		aria-label="粉、水、油、空気を動かして遊ぶキャンバス"
		onpointerdown={pointerDown}
		onpointermove={pointerMove}
		onpointerup={pointerUp}
		onpointercancel={pointerCancel}
		onlostpointercapture={lostPointerCapture}
	></canvas>

	<div class="interface">
		<header class="topbar">
			<div class="brand">
				<span class="brand-mark" aria-hidden="true">p.</span>
				<div>
					<h1>powder-play</h1>
					<p>material playground <span>β 0.2</span></p>
				</div>
			</div>

			<div class="top-controls">
				<div class="segmented view-switch" role="group" aria-label="表示範囲">
					<button
						class:active={viewMode === 'fit'}
						aria-pressed={viewMode === 'fit'}
						onclick={() => toggleViewMode('fit')}
					>
						全体
					</button>
					<button
						class:active={viewMode === 'bleed'}
						aria-pressed={viewMode === 'bleed'}
						onclick={() => toggleViewMode('bleed')}
					>
						幕裏
					</button>
				</div>
				<button
					class="icon-button sensor-button"
					aria-pressed={sensorEnabled}
					onclick={toggleSensors}
					title={sensorEnabled ? 'センサーを停止' : '端末センサーを有効化'}
				>
					<span aria-hidden="true">{sensorEnabled ? '◉' : '⌁'}</span>
					<span class="button-label">{sensorEnabled ? 'センサー中' : 'センサー'}</span>
				</button>
				<button
					class="icon-button pause-button"
					disabled={orientationMismatch}
					onclick={() => {
						isPaused = !isPaused;
						statusMessage = isPaused
							? 'シミュレーションを一時停止しました。'
							: describeMode(activeMode);
					}}
				>
					<span aria-hidden="true">{isPaused ? '▶' : 'Ⅱ'}</span>
					<span class="button-label">{isPaused ? '再開' : '停止'}</span>
				</button>
			</div>
		</header>

		{#if dialPreview}
			<div class="mode-dial" aria-label="動作ダイアル">
				<div class="dial-caption">二本指で回して選択 · 離して決定</div>
				<div class="dial-options">
					{#each modes as mode}
						<div class:preview={dialPreview === mode.id} class="dial-option">
							<span>{mode.icon}</span>
							<strong>{mode.label}</strong>
						</div>
					{/each}
				</div>
			</div>
		{/if}

		{#if errorMessage}
			<div class="error-card" role="alert">{errorMessage}</div>
		{/if}

		<footer class="bottom-panel">
			<nav class="mode-toolbar" aria-label="操作モード">
				{#each modes as mode}
					<button
						class:active={activeMode === mode.id}
						aria-pressed={activeMode === mode.id}
						title={`${mode.description} (${mode.shortcut})`}
						onclick={() => setMode(mode.id)}
					>
						<span class="mode-icon" aria-hidden="true">{mode.icon}</span>
						<span class="mode-name">{mode.label}</span>
						<span class="mode-shortcut">{mode.shortcut}</span>
					</button>
				{/each}
				<div class="zoom-controls" aria-label="ズーム">
					<button aria-label="拡大" onclick={() => zoomBy(1.15)}>＋</button>
					<button aria-label="縮小" onclick={() => zoomBy(1 / 1.15)}>−</button>
				</div>
			</nav>

			<div class="status-row">
				<div class="status-copy" aria-live="polite" aria-atomic="true">
					<span class:paused={isPaused} class="status-dot"></span>
					<div>
						<p class="status-message">{statusMessage}</p>
						<p class="sensor-message">{sensorStatus}</p>
					</div>
				</div>
				<div class="readout" aria-label="シミュレーション情報">
					<span>{fps} <small>FPS</small></span>
					<span>{gridSize.width}×{gridSize.height}</span>
					{#if selectionCount > 0}
						<span>{selectedMaterialName} · {selectionCount}</span>
					{/if}
				</div>
			</div>
		</footer>
	</div>
</main>

<style>
	:global(*) {
		box-sizing: border-box;
	}

	:global(html),
	:global(body) {
		width: 100%;
		height: 100%;
		margin: 0;
		overflow: hidden;
	}

	:global(body) {
		font-family:
			Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
		color: #193546;
		background: #b9e8ed;
		overscroll-behavior: none;
	}

	:global(button) {
		font: inherit;
	}

	.play-area {
		position: fixed;
		inset: 0;
		overflow: hidden;
		background:
			radial-gradient(ellipse at 48% 32%, rgba(240, 255, 246, 0.72), transparent 47%),
			linear-gradient(180deg, #b4eaf0 0%, #d9f0e8 48%, #7aabc1 100%);
		isolation: isolate;
	}

	.world-canvas {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		touch-action: none;
		cursor: crosshair;
	}

	.interface {
		position: absolute;
		inset: 0;
		display: flex;
		flex-direction: column;
		justify-content: space-between;
		padding:
			max(18px, env(safe-area-inset-top))
			max(18px, env(safe-area-inset-right))
			max(18px, env(safe-area-inset-bottom))
			max(18px, env(safe-area-inset-left));
		pointer-events: none;
	}

	.topbar,
	.brand,
	.top-controls,
	.bottom-panel,
	.mode-toolbar,
	.status-row,
	.status-copy,
	.readout,
	.dial-options {
		display: flex;
		align-items: center;
	}

	.topbar {
		justify-content: space-between;
		gap: 14px;
	}

	.brand {
		gap: 10px;
	}

	.brand-mark {
		display: grid;
		width: 42px;
		height: 42px;
		place-items: center;
		border: 1px solid rgba(24, 61, 79, 0.12);
		border-radius: 14px;
		background: rgba(255, 255, 255, 0.55);
		box-shadow: 0 8px 24px rgba(36, 91, 111, 0.12);
		color: #26627b;
		font-family: Georgia, serif;
		font-size: 25px;
		font-weight: 700;
	}

	h1 {
		margin: 0;
		font-size: 15px;
		letter-spacing: -0.035em;
	}

	.brand p {
		margin: 3px 0 0;
		color: rgba(24, 54, 70, 0.64);
		font-size: 10px;
		letter-spacing: 0.09em;
		text-transform: uppercase;
	}

	.brand p span {
		margin-left: 4px;
		padding: 2px 5px;
		border-radius: 5px;
		background: rgba(255, 255, 255, 0.42);
		font-size: 9px;
		letter-spacing: 0.02em;
	}

	.top-controls {
		gap: 7px;
		pointer-events: auto;
	}

	.segmented,
	.icon-button,
	.mode-toolbar,
	.zoom-controls {
		border: 1px solid rgba(30, 73, 92, 0.12);
		background: rgba(249, 255, 251, 0.68);
		box-shadow: 0 8px 26px rgba(36, 91, 111, 0.1);
		backdrop-filter: blur(16px);
		-webkit-backdrop-filter: blur(16px);
	}

	.segmented {
		display: flex;
		padding: 3px;
		border-radius: 12px;
	}

	.segmented button,
	.icon-button,
	.mode-toolbar button,
	.zoom-controls button {
		border: 0;
		color: #305569;
		background: transparent;
		cursor: pointer;
		transition:
			background 140ms ease,
			color 140ms ease,
			transform 140ms ease;
	}

	.segmented button {
		min-height: 31px;
		padding: 0 10px;
		border-radius: 9px;
		font-size: 11px;
	}

	.segmented button.active {
		color: #143d53;
		background: rgba(255, 255, 255, 0.9);
		box-shadow: 0 2px 8px rgba(20, 66, 85, 0.12);
	}

	.icon-button {
		display: flex;
		min-height: 39px;
		align-items: center;
		gap: 7px;
		padding: 0 11px;
		border-radius: 12px;
		font-size: 11px;
		font-weight: 650;
	}

	.icon-button > span:first-child {
		font-size: 15px;
	}

	.pause-button {
		min-width: 76px;
		justify-content: center;
	}

	button:hover:not(:disabled) {
		color: #104c65;
		background-color: rgba(255, 255, 255, 0.85);
	}

	button:active:not(:disabled) {
		transform: translateY(1px);
	}

	button:focus-visible {
		outline: 3px solid rgba(30, 123, 148, 0.65);
		outline-offset: 2px;
	}

	button:disabled {
		cursor: not-allowed;
		opacity: 0.56;
	}

	.bottom-panel {
		flex-direction: column;
		align-items: stretch;
		gap: 10px;
	}

	.mode-toolbar {
		width: fit-content;
		gap: 4px;
		padding: 5px;
		border-radius: 16px;
		pointer-events: auto;
	}

	.mode-toolbar > button {
		display: grid;
		min-width: 79px;
		min-height: 53px;
		grid-template-columns: auto auto;
		grid-template-rows: 1fr 1fr;
		align-items: center;
		column-gap: 7px;
		padding: 6px 11px;
		border-radius: 11px;
		text-align: left;
	}

	.mode-toolbar > button.active {
		color: #fff;
		background: #245d73;
		box-shadow: 0 4px 12px rgba(28, 76, 96, 0.24);
	}

	.mode-icon {
		grid-row: 1 / 3;
		font-size: 19px;
		font-weight: 500;
	}

	.mode-name {
		align-self: end;
		font-size: 12px;
		font-weight: 700;
	}

	.mode-shortcut {
		align-self: start;
		color: rgba(43, 79, 94, 0.55);
		font-size: 9px;
	}

	.mode-toolbar > button.active .mode-shortcut {
		color: rgba(255, 255, 255, 0.72);
	}

	.zoom-controls {
		display: flex;
		align-items: stretch;
		gap: 2px;
		margin-left: 3px;
		padding: 3px;
		border-radius: 12px;
	}

	.zoom-controls button {
		width: 35px;
		border-radius: 9px;
		font-size: 19px;
	}

	.status-row {
		min-height: 46px;
		justify-content: space-between;
		gap: 12px;
		padding: 8px 12px;
		border: 1px solid rgba(30, 73, 92, 0.1);
		border-radius: 13px;
		background: rgba(249, 255, 251, 0.56);
		box-shadow: 0 7px 22px rgba(36, 91, 111, 0.08);
		backdrop-filter: blur(12px);
		-webkit-backdrop-filter: blur(12px);
	}

	.status-copy {
		min-width: 0;
		gap: 9px;
	}

	.status-dot {
		width: 7px;
		height: 7px;
		flex: 0 0 auto;
		border-radius: 100%;
		background: #2d9b73;
		box-shadow: 0 0 0 3px rgba(45, 155, 115, 0.12);
	}

	.status-dot.paused {
		background: #c78338;
		box-shadow: 0 0 0 3px rgba(199, 131, 56, 0.13);
	}

	.status-message,
	.sensor-message {
		overflow: hidden;
		margin: 0;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.status-message {
		color: #24485b;
		font-size: 11px;
		font-weight: 650;
	}

	.sensor-message {
		margin-top: 2px;
		color: rgba(36, 72, 91, 0.7);
		font-size: 9px;
	}

	.readout {
		flex: 0 0 auto;
		gap: 12px;
		color: rgba(34, 69, 85, 0.76);
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		font-size: 10px;
	}

	.readout small {
		color: rgba(34, 69, 85, 0.6);
		font-size: 8px;
	}

	.mode-dial {
		position: absolute;
		top: 50%;
		left: 50%;
		width: min(350px, calc(100vw - 42px));
		padding: 16px;
		transform: translate(-50%, -50%);
		border: 1px solid rgba(255, 255, 255, 0.68);
		border-radius: 22px;
		background: rgba(246, 255, 251, 0.76);
		box-shadow: 0 20px 70px rgba(23, 63, 79, 0.2);
		backdrop-filter: blur(18px);
		-webkit-backdrop-filter: blur(18px);
		text-align: center;
	}

	.dial-caption {
		margin-bottom: 12px;
		color: #3c6476;
		font-size: 11px;
		font-weight: 650;
	}

	.dial-options {
		justify-content: center;
		gap: 9px;
	}

	.dial-option {
		display: flex;
		min-width: 81px;
		flex-direction: column;
		gap: 5px;
		padding: 10px 8px;
		border: 1px solid transparent;
		border-radius: 14px;
		color: #4c6d7c;
		font-size: 11px;
		transition: all 120ms ease;
	}

	.dial-option span {
		font-size: 19px;
	}

	.dial-option.preview {
		border-color: rgba(44, 115, 138, 0.24);
		background: rgba(255, 255, 255, 0.78);
		color: #174f67;
		transform: scale(1.07);
	}

	.error-card {
		position: absolute;
		top: 88px;
		left: 50%;
		max-width: min(440px, calc(100vw - 32px));
		padding: 12px 16px;
		transform: translateX(-50%);
		border: 1px solid rgba(163, 75, 47, 0.22);
		border-radius: 12px;
		background: rgba(255, 239, 229, 0.92);
		color: #793d2c;
		font-size: 12px;
		box-shadow: 0 8px 24px rgba(60, 54, 48, 0.1);
	}

	@media (max-width: 620px) {
		.interface {
			padding:
				max(12px, env(safe-area-inset-top))
				max(12px, env(safe-area-inset-right))
				max(12px, env(safe-area-inset-bottom))
				max(12px, env(safe-area-inset-left));
		}

		.topbar {
			align-items: flex-start;
		}

		.brand-mark {
			width: 36px;
			height: 36px;
			border-radius: 12px;
			font-size: 22px;
		}

		.brand p {
			font-size: 8px;
		}

		.top-controls {
			gap: 5px;
		}

		.segmented button {
			min-height: 30px;
			padding: 0 7px;
			font-size: 10px;
		}

		.icon-button {
			min-height: 36px;
			gap: 4px;
			padding: 0 8px;
			font-size: 10px;
		}

		.pause-button {
			min-width: 58px;
		}

		.mode-toolbar {
			width: 100%;
			justify-content: center;
			gap: 3px;
			padding: 4px;
			border-radius: 14px;
		}

		.mode-toolbar > button {
			min-width: 0;
			min-height: 48px;
			flex: 1;
			column-gap: 5px;
			padding: 5px 8px;
		}

		.mode-icon {
			font-size: 17px;
		}

		.mode-name {
			font-size: 11px;
		}

		.zoom-controls button {
			width: 30px;
			font-size: 17px;
		}

		.status-row {
			min-height: 42px;
			padding: 7px 9px;
		}

		.status-message {
			font-size: 10px;
		}

		.sensor-message {
			max-width: 54vw;
			font-size: 8px;
		}

		.readout {
			gap: 7px;
			font-size: 9px;
		}
	}

	@media (max-width: 390px) {
		.button-label {
			display: none;
		}

		.pause-button {
			min-width: 36px;
			padding: 0 8px;
		}

		.status-message {
			max-width: 54vw;
		}

		.readout > span:nth-child(2) {
			display: none;
		}
	}

	@media (prefers-reduced-motion: reduce) {
		* {
			scroll-behavior: auto !important;
			transition-duration: 0.01ms !important;
		}
	}
</style>
