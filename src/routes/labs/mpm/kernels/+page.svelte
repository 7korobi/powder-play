<script lang="ts">
	import { onMount } from 'svelte';
	import { projectDeviceOrientationGravity } from '#lib/input/sensors.js';
	import { encodeMpmConstitutiveModel } from '#lib/mpm/materials.js';
	import {
		createMpmMlsGpuSolver,
		MPM_MLS_GPU_DRAG_DAMPING,
		MPM_MLS_GPU_DRAG_STIFFNESS,
		type MpmMlsGpuAppendResult,
		type MpmMlsGpuCflBatchMeasurement,
		type MpmMlsGpuOutletBounds,
		type MpmMlsGpuRenderMode,
		type MpmMlsGpuSelection,
		type MpmMlsGpuSpeedMeasurement
	} from '#lib/mpm/kernels/solver.js';
	import { calculateMpmCflVelocityBound } from '#lib/mpm/kernels/cfl-bounds.js';
	import {
		DEFAULT_HORIZONTAL_OSCILLATION_AMPLITUDE,
		HORIZONTAL_OSCILLATION_FREQUENCY_HZ,
		INITIAL_HORIZONTAL_OSCILLATION_AMPLITUDE,
		MAX_HORIZONTAL_OSCILLATION_AMPLITUDE,
		meanSinusoidalAcceleration
	} from '#lib/mpm/kernels/oscillation.js';
	import {
		MPM_RENDER_FPS_OPTIONS,
		MpmFramePacer,
		type MpmRenderFps
	} from '#lib/mpm/kernels/frame-pacer.js';
	import {
		calculateMpmCflStepBatch,
		calculateMpmStepBatch,
		type MpmStepBatch,
		type MpmStepBatchInput
	} from '#lib/mpm/kernels/step-scheduler.js';
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
		GATE4_FRACTURE_BOND_COUNT,
		GATE4_FRACTURE_COLUMNS,
		GATE4_FRACTURE_FAILURE_STRAIN,
		GATE4_FRACTURE_OBJECT_ID,
		GATE4_FRACTURE_PARTICLE_COUNT,
		GATE4_FRACTURE_ROWS,
		GATE4_GRID,
		GATE4_LIQUID_BULK_MODULUS_SCALE,
		GATE4_LIQUID_INITIAL_VELOCITY_COMPONENT_MAX,
		GATE4_LIQUID_PARTICLE_AREA,
		GATE4_LIQUID_PARTICLE_AREA_SCALE,
		GATE4_LIQUID_PARTICLE_SPACING,
		GATE4_PARTICLE_CAPACITY,
		GATE4_PARTICLE_COUNT,
		GATE4_PARTICLE_AREA,
		GATE4_PEBBLE_COUNT,
		GATE4_PEBBLE_EDGE_LENGTH,
		GATE4_PEBBLE_INITIAL_VELOCITY_COMPONENT_MAX,
		GATE4_SOLID_PARTICLE_MASS,
		GATE4_WATER_PARTICLE_COUNT,
		GATE4_OIL_PARTICLE_COUNT
	} from '#lib/mpm/scenes/gate4.js';
	import {
		createMpmEmitterParticles,
		type MpmEmitterMaterial,
		type MpmEmitterShape
	} from '#lib/mpm/scenes/emitter.js';

	const GRID = GATE4_GRID;
	const WORLD_WIDTH = GRID.cellsX * GRID.cellSize;
	const WORLD_HEIGHT = GRID.cellsY * GRID.cellSize;
	const OUTLET_BOUNDS: MpmMlsGpuOutletBounds = {
		minX: 0,
		minY: WORLD_HEIGHT - 80,
		maxX: WORLD_WIDTH,
		maxY: WORLD_HEIGHT
	};
	const FIXED_STEP_SECONDS = 1 / 240;
	const MAX_SIMULATION_SPEED = 32;
	const SENSOR_GRAVITY_MAXIMUM = 180;
	const CONTINUOUS_EMISSION_INTERVAL_MS = 250;
	const MAX_IN_FLIGHT_CFL_BATCHES = 1;
	const MAX_MICROSTEPS_PER_FRAME = Math.ceil(
		MAX_SIMULATION_SPEED / (MPM_RENDER_FPS_OPTIONS[0] * FIXED_STEP_SECONDS)
	);
	const MAX_ADAPTIVE_CFL_MICROSTEPS_PER_BATCH = MAX_MICROSTEPS_PER_FRAME * 2;
	const DEFAULT_RENDER_FPS: MpmRenderFps = 60;
	const DEFAULT_SIMULATION_SPEED = 2;
	const SIMULATION_SPEED_OPTIONS = [0.5, 1, 2, 4, 8, 16, MAX_SIMULATION_SPEED] as const;
	const GPU_PARTICLE_CAPACITY = GATE4_PARTICLE_CAPACITY;
	const PICK_RADIUS = 14;
	const BRUSH_RADIUS = 24;
	const PARTICLE_MASS = GATE4_SOLID_PARTICLE_MASS;
	const PARTICLE_VOLUME = GATE4_PARTICLE_AREA;
	const MATERIAL = { lambda: 20, mu: 10, model: 'neo-hookean' } as const;
	const PARITY_ABSOLUTE_TOLERANCE = 0.02;
	const PARITY_VELOCITY_ABSOLUTE_TOLERANCE = 0.025;
	const PARITY_VELOCITY_RELATIVE_TOLERANCE = 0.0005;
	const PARITY_PROBE_LABELS = ['水', '油', '弾性小石'] as const;
	const PARTICLE_VALUE_LABELS = [
		'position.x',
		'position.y',
		'velocity.x',
		'velocity.y',
		'C.m00',
		'C.m01',
		'C.m10',
		'C.m11',
		'F.m00',
		'F.m01',
		'F.m10',
		'F.m11',
		'plasticVolume',
		'material.model',
		'material.lambda',
		'material.mu',
		'material.density',
		'volume',
		'material.frictionCoefficient',
		'fracture.damage',
		'bondedTriangleVertex'
	] as const;

	type Phase = 'initializing' | 'ready' | 'error' | 'lost';
	type OrientationStatus =
		| 'checking'
		| 'idle'
		| 'requesting'
		| 'enabled'
		| 'denied'
		| 'unsupported';
	type GpuSolver = Awaited<ReturnType<typeof createMpmMlsGpuSolver>>;
	let simulatedTimeSeconds = 0;
	let simulationAccumulatorSeconds = $state(0);
	let latestSubmittedCflSequence = 0;
	let latestPlannedParticleSpeedBound = 0;

	function makeStepConfig(
		gravity: MpmVector2,
		dt = FIXED_STEP_SECONDS,
		fractureBonds?: MpmStepConfig['fractureBonds']
	): MpmStepConfig {
		return {
			dt,
			bondReferenceDt: FIXED_STEP_SECONDS,
			grid: GRID,
			particleMass: PARTICLE_MASS,
			particleVolume: PARTICLE_VOLUME,
			bondRestLength: GATE4_PEBBLE_EDGE_LENGTH,
			material: MATERIAL,
			gravity,
			...(fractureBonds === undefined ? {} : { fractureBonds })
		};
	}

	function toggleHorizontalOscillation(): void {
		horizontalOscillationAmplitude =
			horizontalOscillationAmplitude > 0
				? 0
				: DEFAULT_HORIZONTAL_OSCILLATION_AMPLITUDE;
	}

	let submittedMacroStepCount = 0;
	let submittedMacroStepRemainderSeconds = 0;
	let submittedMicroStepCount = 0;
	let submittedParticleMicrosteps = 0;
	let submittedBatchCount = 0;
	let submittedBatchCpuMilliseconds = 0;

	interface CflSpeedReadbackSample extends MpmMlsGpuCflBatchMeasurement {
		readonly elapsedMilliseconds: number;
	}

	function calculateBoundedBatch(
		currentSolver: GpuSolver,
		input: Omit<MpmStepBatchInput, 'waveSpeedBound' | 'velocityBound'>,
		gravity: MpmVector2,
		oscillationAmplitude: number,
		currentParticleSpeedBound = latestPlannedParticleSpeedBound
	): { readonly batch: MpmStepBatch; readonly velocityBound: number } {
		const cflBounds = currentSolver.cflBounds;
		if (cflBounds === null) {
			throw new RangeError(
				'このシーンには材料・初期状態から導出できる CFL 上限がありません。'
			);
		}
		return calculateMpmCflStepBatch({
			...input,
			allowPartialBatch: true,
			waveSpeedBound: cflBounds.waveSpeedBound,
			currentVelocityBound: Math.max(
				cflBounds.velocityBound,
				latestPlannedParticleSpeedBound,
				currentParticleSpeedBound
			),
			gravityMagnitude: Math.hypot(gravity.x, gravity.y),
			oscillationAmplitude,
			dragRadius: dragActive ? BRUSH_RADIUS : 0,
			dragStiffness: MPM_MLS_GPU_DRAG_STIFFNESS,
			dragDamping: dragActive ? MPM_MLS_GPU_DRAG_DAMPING : 0,
			courantLimit: cflBounds.courantLimit
		});
	}

	function submitBatch(
		currentSolver: GpuSolver,
		batch: MpmStepBatch,
		gravity: MpmVector2,
		oscillationAmplitude: number,
		velocityBound: number
	): Promise<CflSpeedReadbackSample> {
		const startTimeSeconds = simulatedTimeSeconds;
		const startedAt = performance.now();
		const measurement = currentSolver.stepCflBatchAdaptive(
			batch.macroStepCount,
			batch.macroDt,
			batch.microstepPlan,
			velocityBound,
			MAX_ADAPTIVE_CFL_MICROSTEPS_PER_BATCH,
			gravity,
			(offsetSeconds, durationSeconds) => ({
				x:
					gravity.x +
					meanSinusoidalAcceleration(
						startTimeSeconds + offsetSeconds,
						durationSeconds,
						oscillationAmplitude
					),
				y: gravity.y
			})
		);
		const submitMilliseconds = performance.now() - startedAt;
		const submittedSequence = latestSubmittedCflSequence + 1;
		latestSubmittedCflSequence = submittedSequence;
		latestPlannedParticleSpeedBound = velocityBound;
		cflSpeedReadbackCount += 1;
		return measurement
			.then((measurement) => {
				simulatedTimeSeconds += measurement.simulatedDuration;
				if (measurement.remainingDuration > 0) {
					if (measurement.simulatedDuration <= 0) {
						throw new Error(
							`Adaptive CFL recovery made no valid progress; ${measurement.remainingDuration} seconds remain.`
						);
					}
					simulationAccumulatorSeconds += measurement.remainingDuration;
					if (!Number.isFinite(simulationAccumulatorSeconds)) {
						throw new RangeError(
							'deferred simulation time is outside the supported range.'
						);
					}
				}
				submittedBatchCount += 1;
				const macroStepProgress =
					submittedMacroStepRemainderSeconds + measurement.simulatedDuration;
				const macroStepTolerance = FIXED_STEP_SECONDS * Number.EPSILON * 8;
				const completedMacroSteps = Math.floor(
					(macroStepProgress + macroStepTolerance) / FIXED_STEP_SECONDS
				);
				submittedMacroStepCount += completedMacroSteps;
				submittedMacroStepRemainderSeconds = Math.max(
					0,
					macroStepProgress - completedMacroSteps * FIXED_STEP_SECONDS
				);
				submittedMicroStepCount += measurement.attemptedMicroStepCount;
				submittedParticleMicrosteps +=
					currentSolver.particleCount * measurement.attemptedMicroStepCount;
				submittedBatchCpuMilliseconds +=
					submitMilliseconds + measurement.retrySubmissionMilliseconds;
				return {
					...measurement,
					sequence: submittedSequence,
					elapsedMilliseconds: measurement.readbackMilliseconds
				};
			})
			.finally(() => {
				cflSpeedReadbackCount = Math.max(0, cflSpeedReadbackCount - 1);
			});
	}

	function applyCflSpeedSample(
		sample: MpmMlsGpuSpeedMeasurement & {
			readonly recommendedVelocityBound?: number;
		}
	): void {
		if (sample.sequence === latestSubmittedCflSequence) {
			latestPlannedParticleSpeedBound =
				sample.recommendedVelocityBound ?? sample.maximumSpeed;
		}
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
			frictionCoefficient,
			particle.damage ?? 0,
			particle.bondedTriangleVertex ?? 0
		];
	}

	function maxParticleDifference(
		actual: readonly MpmParticle[],
		expected: readonly MpmParticle[]
	): {
		readonly absoluteDifference: number;
		readonly probeIndex: number;
		readonly field: string;
		readonly actual: number;
		readonly expected: number;
	} {
		if (actual.length !== expected.length) {
			return {
				absoluteDifference: Number.POSITIVE_INFINITY,
				probeIndex: -1,
				field: 'particle count',
				actual: actual.length,
				expected: expected.length
			};
		}

		let maximum = {
			absoluteDifference: 0,
			probeIndex: -1,
			field: '',
			actual: 0,
			expected: 0
		};
		for (let index = 0; index < actual.length; index += 1) {
			const actualValues = particleValues(actual[index]);
			const expectedValues = particleValues(expected[index]);

			for (let valueIndex = 0; valueIndex < actualValues.length; valueIndex += 1) {
				const actualValue = actualValues[valueIndex];
				const expectedValue = expectedValues[valueIndex];
				if (!Number.isFinite(actualValue) || !Number.isFinite(expectedValue)) {
					return {
						absoluteDifference: Number.POSITIVE_INFINITY,
						probeIndex: index,
						field: PARTICLE_VALUE_LABELS[valueIndex] ?? `value ${valueIndex}`,
						actual: actualValue,
						expected: expectedValue
					};
				}
				const absoluteDifference = Math.abs(actualValue - expectedValue);
				if (absoluteDifference > maximum.absoluteDifference) {
					maximum = {
						absoluteDifference,
						probeIndex: index,
						field: PARTICLE_VALUE_LABELS[valueIndex] ?? `value ${valueIndex}`,
						actual: actualValue,
						expected: expectedValue
					};
				}
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
	let manualGravityX = $state(0);
	let manualGravityY = $state(90);
	let orientationStatus = $state<OrientationStatus>('checking');
	let orientationMessage = $state('DeviceOrientation センサーを確認しています…');
	let sensorGravityUnit = $state<MpmVector2>({ x: 0, y: 0 });
	let hasSensorGravity = $state(false);
	let simulationSpeed = $state(DEFAULT_SIMULATION_SPEED);
	let renderTargetFps: MpmRenderFps = $state(DEFAULT_RENDER_FPS);
	let horizontalOscillationAmplitude = $state(INITIAL_HORIZONTAL_OSCILLATION_AMPLITUDE);
	let paused = $state(false);
	let parityMessage = $state('');
	let parityPassed = $state<boolean | null>(null);
	let stateCheckPending = $state(false);
	let stateCheckMessage = $state('');
	let stateCheckIsError = $state(false);
	let particleCount = $state(GATE4_PARTICLE_COUNT);
	let allocatedParticleCount = $state(GATE4_PARTICLE_COUNT);
	let particleCapacity = $state(GPU_PARTICLE_CAPACITY);
	let measuredFps = $state<number | null>(null);
	let measuredMacroStepsPerSecond = $state<number | null>(null);
	let measuredMicroStepsPerSecond = $state<number | null>(null);
	let measuredParticleMicrostepsPerSecond = $state<number | null>(null);
	let averageBatchSubmitMs = $state<number | null>(null);
	let averageQueueCompletionMs = $state<number | null>(null);
	let averageCflReadbackMs = $state<number | null>(null);
	let cflSpeedReadbackCount = $state(0);
	let renderMode = $state<MpmMlsGpuRenderMode>('surface');
	let selection = $state<MpmMlsGpuSelection | null>(null);
	let interactionMessage = $state('初回の CPU/GPU 1 ステップ比較が完了すると操作できます。');
	let interactionIsError = $state(false);
	let emitterMaterial: MpmEmitterMaterial = $state('water');
	let emitterShape: MpmEmitterShape = $state('circle');
	let emitterCount = $state(64);
	let emitterSeed = $state(0x5eed_1234);
	let emitterSequence = $state(0);
	let emitterX = $state(700);
	let emitterY = $state(200);
	let emitterSpacing = $state(GATE4_LIQUID_PARTICLE_SPACING);
	let emitterVelocityX = $state(0);
	let emitterVelocityY = $state(30);
	let continuousEmission = $state(false);
	let lastEmission = $state<MpmMlsGpuAppendResult | null>(null);
	let emissionMessage = $state('まだ粒子を追加していません。');
	let emissionIsError = $state(false);
	let outletPending = $state(false);
	let outletMessage = $state('出口は画面下端の80 world unitsです。除去粒子のスロットは再利用しません。');
	let outletIsError = $state(false);

	let componentMounted = false;
	let activePointerId = $state<number | null>(null);
	let pointerGeneration = 0;
	let pointerPosition: MpmVector2 | null = null;
	let dragActive = $state(false);

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
			orientationMessage = '傾きデータがありません。手動の重力ベクトルにフォールバックします。';
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
	}

	async function toggleDeviceOrientation(): Promise<void> {
		if (orientationStatus === 'enabled') {
			window.removeEventListener('deviceorientation', handleDeviceOrientation);
			orientationStatus = 'idle';
			hasSensorGravity = false;
			orientationMessage = 'センサーを停止しました。手動の重力ベクトルを使用します。';
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
		if (
			!event.isPrimary ||
			event.button !== 0 ||
			phase !== 'ready' ||
			outletPending ||
			cflSpeedReadbackCount > 0 ||
			!solver
		) {
			return;
		}
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
			interactionMessage =
				picked.objectId === GATE4_FRACTURE_OBJECT_ID
					? `GPU 破断 body を選択しました。損傷は表示色に反映されます。有限半径 ${BRUSH_RADIUS} の力でドラッグできます。結合状態を復活させないため複製は無効です。`
					: `オブジェクト ${picked.objectId} を選択しました。有限半径 ${BRUSH_RADIUS} の力でドラッグできます。`;
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
			currentSelection.objectId === GATE4_FRACTURE_OBJECT_ID ||
			phase !== 'ready' ||
			outletPending ||
			cflSpeedReadbackCount > 0 ||
			activePointerId !== null ||
			dragActive
		) {
			return;
		}

		interactionIsError = false;
		try {
			const cloned = currentSolver.clone(currentSelection, GATE4_CLONE_OFFSET);
			particleCount = currentSolver.particleCount;
			allocatedParticleCount = currentSolver.allocatedParticleCount;
			particleCapacity = currentSolver.particleCapacity;
			if (cloned) {
				interactionMessage = `オブジェクト ${currentSelection.objectId} を複製しました。活動 ${particleCount} / 使用スロット ${allocatedParticleCount} / 容量 ${particleCapacity}。`;
			} else {
				interactionIsError = true;
				interactionMessage = `容量不足または outlet 後の安全制限により複製できませんでした。活動 ${particleCount} / 使用スロット ${allocatedParticleCount} / 容量 ${particleCapacity}。`;
			}
		} catch (error) {
			interactionIsError = true;
			interactionMessage = `複製に失敗しました: ${errorText(error)}`;
		}
	}

	function togglePaused(): void {
		paused = !paused;
	}

	function discardDeferredSimulationTime(): void {
		if (
			phase !== 'ready' ||
			!paused ||
			outletPending ||
			cflSpeedReadbackCount > 0 ||
			simulationAccumulatorSeconds <= 0
		) {
			return;
		}

		const discardedSeconds = simulationAccumulatorSeconds;
		simulationAccumulatorSeconds = 0;
		phaseMessage =
			`${discardedSeconds.toFixed(3)} 秒分の未処理シミュレーション時間を破棄しました。` +
			'粒子状態を保持して一時停止中です。';
	}

	function emitSourceBurst(): MpmMlsGpuAppendResult | null {
		const currentSolver = solver;
		if (!currentSolver || phase !== 'ready') return null;
		const seed = (emitterSeed + emitterSequence) >>> 0;
		emitterSequence += 1;
		let particles: readonly MpmParticle[];
		try {
			particles = createMpmEmitterParticles({
				count: emitterCount,
				seed,
				center: { x: emitterX, y: emitterY },
				shape: emitterShape,
				spacing: emitterSpacing,
				velocity: { x: emitterVelocityX, y: emitterVelocityY },
				material: emitterMaterial,
				particleVolume:
					emitterMaterial === 'water' || emitterMaterial === 'oil'
						? GATE4_LIQUID_PARTICLE_AREA
						: PARTICLE_VOLUME,
				grid: GRID
			});
		} catch (error) {
			const requestedCount =
				Number.isSafeInteger(emitterCount) && emitterCount > 0 ? emitterCount : 0;
			lastEmission = {
				requested: requestedCount,
				accepted: 0,
				refused: requestedCount,
				dropped: 0,
				activeParticleCount: currentSolver.particleCount,
				allocatedParticleCount: currentSolver.allocatedParticleCount,
				capacity: currentSolver.particleCapacity,
				velocityBound: 0,
				reason: errorText(error)
			};
			emissionMessage =
				requestedCount > 0
					? `source batch を拒否しました。accepted 0 / refused ${requestedCount} / dropped 0 · ${errorText(error)}`
					: `個数が正の整数ではないため batch を生成しませんでした。accepted 0 / refused 0 / dropped 0 · ${errorText(error)}`;
			emissionIsError = true;
			continuousEmission = false;
			return lastEmission;
		}

		const emitterSpeedBound = Math.hypot(emitterVelocityX, emitterVelocityY);
		let cflPreview: ReturnType<typeof calculateBoundedBatch>;
		try {
			cflPreview = calculateBoundedBatch(
				currentSolver,
				{
					accumulatorSeconds: FIXED_STEP_SECONDS,
					elapsedSeconds: 0,
					simulationSpeed: 1,
					fixedStepSeconds: FIXED_STEP_SECONDS,
					cellSize: GRID.cellSize,
					maxMicrosteps: MAX_MICROSTEPS_PER_FRAME
				},
				effectiveGravity(),
				horizontalOscillationAmplitude,
				emitterSpeedBound
			);
		} catch (error) {
			if (!(error instanceof RangeError)) {
				console.error('Emitter CFL preflight failed.', error);
			}
			const reason = `CFL事前確認: ${errorText(error)}`;
			lastEmission = {
				requested: particles.length,
				accepted: 0,
				refused: particles.length,
				dropped: 0,
				activeParticleCount: currentSolver.particleCount,
				allocatedParticleCount: currentSolver.allocatedParticleCount,
				capacity: currentSolver.particleCapacity,
				velocityBound: emitterSpeedBound,
				reason
			};
			emissionMessage =
				`CFL事前確認で追加を拒否しました。accepted 0 / refused ${particles.length} / dropped 0 · ${reason}`;
			emissionIsError = true;
			continuousEmission = false;
			return lastEmission;
		}

		try {
			const result = currentSolver.appendParticles(particles);
			lastEmission = result;
			particleCount = result.activeParticleCount;
			allocatedParticleCount = result.allocatedParticleCount;
			particleCapacity = result.capacity;
			latestPlannedParticleSpeedBound = Math.max(
				latestPlannedParticleSpeedBound,
				result.velocityBound
			);
			emissionIsError = result.refused > 0 || result.dropped > 0;
			emissionMessage =
				`要求 ${result.requested} · 受入 ${result.accepted} · 拒否 ${result.refused} · 容量超過で未追加 ${result.dropped}。` +
				`活動粒子 ${result.activeParticleCount} / 予約スロット ${result.allocatedParticleCount} / 容量 ${result.capacity}.` +
				(result.reason ? ` ${result.reason}` : '') +
				(cflPreview.batch.macroDt < FIXED_STEP_SECONDS
					? ` CFL安全のため時間刻みを ${cflPreview.batch.macroDt.toExponential(2)} 秒に短縮。要求倍率より遅く進む場合があります。`
					: ` CFL事前確認: ${cflPreview.batch.microstepPlan.stepCount} microsteps/macrostep。`);
			if (result.accepted === 0 && (result.refused > 0 || result.dropped > 0)) {
				continuousEmission = false;
			}
			return result;
		} catch (error) {
			emissionMessage = `GPU 追加処理に失敗し状態を安全に確認できないため停止しました: ${errorText(error)}`;
			emissionIsError = true;
			continuousEmission = false;
			paused = true;
			phase = 'error';
			phaseMessage = '粒子追加処理の GPU state が不確かなため停止しました。';
			return null;
		}
	}

	function toggleContinuousEmission(): void {
		continuousEmission = !continuousEmission;
		if (!continuousEmission) {
			emissionIsError = false;
		}
	}

	function resetEmitterSequence(): void {
		continuousEmission = false;
		emitterSequence = 0;
		lastEmission = null;
		emissionIsError = false;
		emissionMessage = 'seed sequence をリセットしました。同じ設定で同じ particle batch を再現できます。';
	}

	async function drainOutlet(): Promise<void> {
		const currentSolver = solver;
		if (
			!currentSolver ||
			phase !== 'ready' ||
			outletPending ||
			cflSpeedReadbackCount > 0 ||
			activePointerId !== null ||
			dragActive
		) {
			return;
		}
		outletPending = true;
		outletIsError = false;
		outletMessage = '出口内の水・油粒子を GPU 上で確認しています…';
		try {
			const removed = await currentSolver.removeParticlesInOutlet(OUTLET_BOUNDS);
			if (solver !== currentSolver || phase !== 'ready') return;
			particleCount = currentSolver.particleCount;
			allocatedParticleCount = currentSolver.allocatedParticleCount;
			particleCapacity = currentSolver.particleCapacity;
			if (removed > 0) {
				selection = null;
				interactionMessage = 'outlet で particle slot を除去したため、選択を解除しました。';
			}
			outletMessage =
				`${removed} 粒子を除去しました。活動粒子 ${particleCount} / 予約スロット ${currentSolver.allocatedParticleCount}。` +
				'除去済みスロットは安全のため再利用しません。';
		} catch (error) {
			if (solver !== currentSolver || phase !== 'ready') return;
			outletIsError = true;
			outletMessage = `出口の除去に失敗しました: ${errorText(error)}`;
			paused = true;
			phase = 'error';
			phaseMessage = '出口処理の GPU readback に失敗したため停止しました。';
		} finally {
			outletPending = false;
		}
	}

	async function stepOnce(): Promise<void> {
		if (
			phase !== 'ready' ||
			!paused ||
			outletPending ||
			!solver ||
			cflSpeedReadbackCount > 0
		) {
			return;
		}
		const currentSolver = solver;
		const gravity = effectiveGravity();

		try {
			const planned = calculateBoundedBatch(
				currentSolver,
				{
					accumulatorSeconds: 0,
					elapsedSeconds: FIXED_STEP_SECONDS,
					simulationSpeed: 1,
					fixedStepSeconds: FIXED_STEP_SECONDS,
					cellSize: GRID.cellSize,
					maxMicrosteps: MAX_MICROSTEPS_PER_FRAME
				},
				gravity,
				horizontalOscillationAmplitude
			);
			const sample = await submitBatch(
				currentSolver,
				planned.batch,
				gravity,
				horizontalOscillationAmplitude,
				planned.velocityBound
			);
			if (solver !== currentSolver || phase !== 'ready') return;
			applyCflSpeedSample(sample);
			averageCflReadbackMs = sample.elapsedMilliseconds;
			currentSolver.render(renderMode);
		} catch (error) {
			finishPointerInteraction();
			phase = 'error';
			phaseMessage = `1 ステップを実行できませんでした: ${errorText(error)}`;
		}
	}

	async function checkGpuState(): Promise<void> {
		const currentSolver = solver;
		if (
			!currentSolver ||
			phase !== 'ready' ||
			!paused ||
			outletPending ||
			cflSpeedReadbackCount > 0 ||
			stateCheckPending
		) {
			return;
		}

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
		refreshDeviceOrientationSupport();
		let activeSolver: GpuSolver | null = null;
		let frameRequest: number | null = null;
		const framePacer = new MpmFramePacer();
		let lastFrameTimestamp: number | null = null;
		let metricsStartTimestamp: number | null = null;
		let performanceFrameCount = 0;
		let queueWorkPending = false;
		let queueSampleCount = 0;
		let queueSampleTotalMs = 0;
		let cflReadbackSampleCount = 0;
		let cflReadbackTotalMs = 0;
		let lastNumericalStatusCheckTimestamp = 0;
		let lastContinuousEmissionTimestamp: number | null = null;
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
			if (!mounted || phase === 'lost' || phase === 'error') return;
			finishPointerInteraction();
			phase = 'error';
			phaseMessage = errorText(error);
			console.error('GPU MLS-MPM runtime failed.', error);
		}

		function renderFrame(timestamp: number): void {
			if (!mounted || !activeSolver || phase !== 'ready') return;
			frameRequest = requestAnimationFrame(renderFrame);
			if (!framePacer.shouldRender(timestamp, renderTargetFps)) return;
			if (outletPending) {
				lastFrameTimestamp = null;
				return;
			}
			if (queueWorkPending) return;
			if (continuousEmission) {
				if (paused || document.visibilityState !== 'visible') {
					lastContinuousEmissionTimestamp = timestamp;
				} else if (lastContinuousEmissionTimestamp === null) {
					lastContinuousEmissionTimestamp = timestamp;
				} else if (
					cflSpeedReadbackCount === 0 &&
					timestamp - lastContinuousEmissionTimestamp >=
					CONTINUOUS_EMISSION_INTERVAL_MS
				) {
					lastContinuousEmissionTimestamp = timestamp;
					emitSourceBurst();
				}
			} else {
				lastContinuousEmissionTimestamp = null;
			}

			const elapsed =
				lastFrameTimestamp === null
					? 0
					: Math.max(0, (timestamp - lastFrameTimestamp) / 1000);
			lastFrameTimestamp = timestamp;
			const gravity = effectiveGravity();

			let cflSpeedMeasurement: Promise<CflSpeedReadbackSample> | null = null;
			if (!paused && document.visibilityState === 'visible') {
				try {
					if (cflSpeedReadbackCount >= MAX_IN_FLIGHT_CFL_BATCHES) {
						simulationAccumulatorSeconds += elapsed * simulationSpeed;
						if (!Number.isFinite(simulationAccumulatorSeconds)) {
							throw new RangeError(
								'accumulated simulation time is outside the supported range.'
							);
						}
					} else {
						const planned = calculateBoundedBatch(
							activeSolver,
							{
								accumulatorSeconds: simulationAccumulatorSeconds,
								elapsedSeconds: elapsed,
								simulationSpeed,
								fixedStepSeconds: FIXED_STEP_SECONDS,
								cellSize: GRID.cellSize,
								maxMicrosteps: MAX_MICROSTEPS_PER_FRAME
							},
							gravity,
							horizontalOscillationAmplitude
						);
						if (planned.batch.macroStepCount > 0) {
							cflSpeedMeasurement = submitBatch(
								activeSolver,
								planned.batch,
								gravity,
								horizontalOscillationAmplitude,
								planned.velocityBound
							);
						}
						simulationAccumulatorSeconds = planned.batch.remainderSeconds;
					}
				} catch (error) {
					finishPointerInteraction();
					phase = 'error';
					phaseMessage = `シミュレーションを停止しました: ${errorText(error)}`;
					return;
				}
			}

			try {
				activeSolver.render(renderMode);
			} catch (error) {
				if (cflSpeedMeasurement !== null) {
					void cflSpeedMeasurement.catch((readbackError) => {
						if (mounted && activeSolver) fail(readbackError);
					});
				}
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
				measuredMacroStepsPerSecond =
					submittedMacroStepCount * samplesPerSecond;
				measuredMicroStepsPerSecond =
					submittedMicroStepCount * samplesPerSecond;
				measuredParticleMicrostepsPerSecond =
					submittedParticleMicrosteps * samplesPerSecond;
				averageBatchSubmitMs =
					submittedBatchCount > 0
						? submittedBatchCpuMilliseconds / submittedBatchCount
						: null;
				averageQueueCompletionMs =
					queueSampleCount > 0 ? queueSampleTotalMs / queueSampleCount : null;
				averageCflReadbackMs =
					cflReadbackSampleCount > 0
						? cflReadbackTotalMs / cflReadbackSampleCount
						: null;

				metricsStartTimestamp = timestamp;
				performanceFrameCount = 0;
				submittedMacroStepCount = 0;
				submittedMacroStepRemainderSeconds = 0;
				submittedMicroStepCount = 0;
				submittedParticleMicrosteps = 0;
				submittedBatchCount = 0;
				submittedBatchCpuMilliseconds = 0;
				queueSampleCount = 0;
				queueSampleTotalMs = 0;
				cflReadbackSampleCount = 0;
				cflReadbackTotalMs = 0;
			}

			const submittedSolver = activeSolver;
			if (cflSpeedMeasurement !== null) {
				void cflSpeedMeasurement
					.then((sample) => {
						if (!mounted || activeSolver !== submittedSolver) return;
						applyCflSpeedSample(sample);
						cflReadbackSampleCount += 1;
						cflReadbackTotalMs += sample.elapsedMilliseconds;
					})
					.catch((error) => {
						if (!mounted || activeSolver !== submittedSolver) return;
						fail(error);
					});
			}
			queueWorkPending = true;
			void submittedSolver
				.measureQueueCompletionMs()
				.then(async (completionMs) => {
					if (!mounted || activeSolver !== submittedSolver) return;
					queueSampleCount += 1;
					queueSampleTotalMs += completionMs;
					if (
						cflSpeedReadbackCount === 0 &&
						timestamp - lastNumericalStatusCheckTimestamp >= 1_000
					) {
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
			lastContinuousEmissionTimestamp = null;
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
				const initialScene = createMpmGate4Scene(0x4d504d34);
				const paritySample = createMpmCpuParitySample(
					initialScene.particles,
					initialScene.parityProbeIndices,
					GRID.cellSize
				);
				const initialParticles = initialScene.particles;
				const config = makeStepConfig(
					effectiveGravity(),
					FIXED_STEP_SECONDS,
					initialScene.fractureBonds
				);
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
				if (candidate.cflBounds === null) {
					candidate.dispose();
					throw new RangeError(
						'Gate 4 の構成材料に対する安全な波速上限を導出できないため、CFL microstep は実行できません。'
					);
				}

				activeSolver = candidate;
				solver = candidate;
				particleCount = candidate.particleCount;
				allocatedParticleCount = candidate.allocatedParticleCount;
				particleCapacity = candidate.particleCapacity;
				candidate.deviceLost.then((info) => {
					if (!mounted || activeSolver !== candidate) return;
					if (frameRequest !== null) cancelAnimationFrame(frameRequest);
					finishPointerInteraction();
					paused = true;
					phase = 'lost';
					phaseMessage = info.message || 'WebGPU デバイスが失われました。';
				});

				const parityVelocityBound = calculateMpmCflVelocityBound({
					currentVelocityBound: candidate.cflBounds.velocityBound,
					durationSeconds: FIXED_STEP_SECONDS,
					gravityMagnitude: Math.hypot(config.gravity.x, config.gravity.y),
					oscillationAmplitude: 0,
					dragRadius: 0,
					dragStiffness: MPM_MLS_GPU_DRAG_STIFFNESS,
					dragDamping: 0
				});
				const parityBatch = calculateMpmStepBatch({
					accumulatorSeconds: 0,
					elapsedSeconds: FIXED_STEP_SECONDS,
					simulationSpeed: 1,
					fixedStepSeconds: FIXED_STEP_SECONDS,
					cellSize: GRID.cellSize,
					...candidate.cflBounds,
					velocityBound: parityVelocityBound,
					maxMicrosteps: MAX_MICROSTEPS_PER_FRAME
				});
				const parityMeasurement = await candidate.stepCflBatchAdaptive(
					parityBatch.macroStepCount,
					FIXED_STEP_SECONDS,
					parityBatch.microstepPlan,
					parityVelocityBound,
					MAX_MICROSTEPS_PER_FRAME,
					config.gravity
				);
				if (!mounted || activeSolver !== candidate || phase === 'lost') return;
				latestSubmittedCflSequence = parityMeasurement.sequence;
				applyCflSpeedSample(parityMeasurement);
				if (
					parityMeasurement.retryCount === 0 &&
					parityMeasurement.remainingDuration === 0
				) {
					const parityMicrostepConfig = makeStepConfig(
						config.gravity,
						parityBatch.microstepPlan.microDt
					);
					let expectedSample = paritySample.particles;
					for (
						let microstep = 0;
						microstep < parityBatch.microstepPlan.stepCount;
						microstep += 1
					) {
						expectedSample = stepMpm(
							expectedSample,
							parityMicrostepConfig
						).particles;
					}
					const actual = await candidate.readParticles(
						initialScene.parityProbeIndices
					);
					if (
						!mounted ||
						activeSolver !== candidate ||
						phase !== 'initializing'
					) {
						return;
					}
					const expected = paritySample.probeSampleIndices.map(
						(index) => expectedSample[index]
					);
					const maxError = maxParticleDifference(actual, expected);
					parityPassed = particleDifferenceWithinTolerance(actual, expected);
					const probeLabel =
						PARITY_PROBE_LABELS[maxError.probeIndex] ?? '粒子';
					parityMessage = Number.isFinite(maxError.absoluteDifference)
						? `初回1 macrostep・${parityBatch.microstepPlan.stepCount} CFL microsteps、各材料中央3粒子の CPU reference 比較: 最大絶対誤差 ${maxError.absoluteDifference.toExponential(2)}（${probeLabel} ${maxError.field}: GPU ${maxError.actual.toExponential(2)} / CPU ${maxError.expected.toExponential(2)}）${
								parityPassed ? '· 誤差基準内' : '· 誤差基準超過'
							}`
						: '初回1 macrostepの CPU reference 比較で粒子数または数値が一致しません。';
				} else {
					parityPassed = null;
					parityMessage =
						`初回macrostepを適応CFLで安全に進めました（${parityMeasurement.retryCount} retry、` +
						`${parityMeasurement.attemptedMicroStepCount} microsteps）。時間刻みを再試行時に変更したため、` +
						'この起動時CPU/GPU比較は省略しました。';
				}
				submittedMacroStepCount = 0;
				submittedMicroStepCount = 0;
				submittedParticleMicrosteps = 0;
				submittedBatchCount = 0;
				submittedBatchCpuMilliseconds = 0;
				queueSampleCount = 0;
				queueSampleTotalMs = 0;
				simulatedTimeSeconds = parityMeasurement.simulatedDuration;
				simulationAccumulatorSeconds = parityMeasurement.remainingDuration;
				metricsStartTimestamp = null;
				performanceFrameCount = 0;
				measuredMacroStepsPerSecond = null;
				measuredMicroStepsPerSecond = null;
				measuredParticleMicrostepsPerSecond = null;
				averageBatchSubmitMs = null;
				averageQueueCompletionMs = null;
				averageCflReadbackMs = null;
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
			window.removeEventListener('deviceorientation', handleDeviceOrientation);
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
	<title>GPU 2D MLS-MPM インタラクティブサンドボックス · powder-play</title>
	<meta
		name="description"
		content="2D GPU MLS-MPM の水・油相、材料 emitter、傾き重力と局所破断する弾性体を操作する実験サンドボックスです。"
	/>
</svelte:head>

<main class="lab">
	<header class="header">
		<div>
			<p class="eyebrow">STAGES 1–4 · INTERACTIVE 2D GPU MLS-MPM</p>
			<h1>GPU 2D MLS-MPM サンドボックス</h1>
			<p class="intro">
				水・油・Neo-Hookean 固体を repeatable / continuous emitter から追加し、傾きまたは手動の2D重力で操作します。
				水と油は相別速度場・node-owned normal contact で接触し、各相の mass と接触 impulse を小規模 CPU reference で検証します。
				初期シーンは{GATE4_PARTICLE_COUNT.toLocaleString()}粒子（旧14,696粒子の1/4）です。水・油の粒子参照面積と総質量は従来設定からさらに{GATE4_LIQUID_PARTICLE_AREA_SCALE}倍（{GATE4_PARTICLE_AREA}→{GATE4_LIQUID_PARTICLE_AREA}）にし、密度・粒子数・初期配置は維持します。粒子マーカーの半径は2倍です。圧縮抵抗はEOS係数を{GATE4_LIQUID_BULK_MODULUS_SCALE}倍に強化し、水・油の相境界には独立した重なり緩和を加えました。初期速度は水・油±{GATE4_LIQUID_INITIAL_VELOCITY_COMPONENT_MAX}、小石±{GATE4_PEBBLE_INITIAL_VELOCITY_COMPONENT_MAX}へ上げ、揺れは初期停止にしています。
				50個の三粒子小石に加え、{GATE4_FRACTURE_COLUMNS}×{GATE4_FRACTURE_ROWS}ノード・{GATE4_FRACTURE_BOND_COUNT} bond の固体を GPU 上で損傷・局所破断させます。
				これは制御可能な研究用 vertical slice であり、一般用途の流体・破壊エンジンや物理精度保証ではありません。
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
				<h2 id="simulation-title">水・油・弾性小石・GPU破断固体 · {GATE4_PARTICLE_COUNT.toLocaleString()}粒子</h2>
				<p>
					活動 {particleCount} · 使用スロット {allocatedParticleCount} / 容量 {particleCapacity} · 空き {Math.max(0, particleCapacity - allocatedParticleCount)} · {GRID.cellsX} × {GRID.cellsY} grid
				</p>
			</div>
			<p class="status" class:error={phase === 'error' || phase === 'lost'} role="status">
				{phaseMessage}
			</p>
		</div>
		{#if phase === 'error' || phase === 'lost'}
			<div class="recovery-controls">
				<p>
					現在のGPU状態から安全に続行できません。初期シーンで再起動すると、追加した粒子と現在の状態は失われます。
				</p>
				<button
					type="button"
					class="secondary"
					onclick={() => window.location.reload()}
				>
					初期シーンから再起動
				</button>
			</div>
		{/if}

		<div class="simulation-layout">
			<section class="simulation-controls" aria-label="シミュレーション操作">
				<div class="controls">
					<fieldset class="gravity-controls">
						<legend>2D 重力 · {orientationStatus === 'enabled' && hasSensorGravity ? 'センサー' : '手動'}</legend>
						<label>
							<span>手動 X</span>
							<input
								type="range"
								min="-180"
								max="180"
								step="5"
								bind:value={manualGravityX}
								disabled={phase !== 'ready' || (orientationStatus === 'enabled' && hasSensorGravity)}
								aria-label="手動重力 X 成分"
							/>
							<output>{manualGravityX}</output>
						</label>
						<label>
							<span>手動 Y</span>
							<input
								type="range"
								min="-180"
								max="180"
								step="5"
								bind:value={manualGravityY}
								disabled={phase !== 'ready' || (orientationStatus === 'enabled' && hasSensorGravity)}
								aria-label="手動重力 Y 成分"
							/>
							<output>{manualGravityY}</output>
						</label>
						<label>
							<span>センサー強度</span>
							<input
								type="range"
								min="0"
								max={SENSOR_GRAVITY_MAXIMUM}
								step="5"
								bind:value={gravityMagnitude}
								disabled={phase !== 'ready'}
								aria-label="センサー重力の大きさ"
							/>
							<output>{gravityMagnitude}</output>
						</label>
						<p class="gravity-readout">
							適用中: X {appliedGravity.x.toFixed(1)} / Y {appliedGravity.y.toFixed(1)} world units/s²
						</p>
						<button
							type="button"
							class="secondary"
							onclick={toggleDeviceOrientation}
							disabled={
								phase !== 'ready' ||
								orientationStatus === 'checking' ||
								orientationStatus === 'requesting' ||
								orientationStatus === 'unsupported'
							}
							aria-pressed={orientationStatus === 'enabled'}
						>
							{orientationStatus === 'requesting'
								? '許可を確認中…'
								: orientationStatus === 'enabled'
									? '傾きセンサーを停止'
									: '傾きセンサーを有効化'}
						</button>
						<p
							class="sensor-status"
							class:error={orientationStatus === 'denied' || orientationStatus === 'unsupported'}
							role="status"
						>
							{orientationStatus.toUpperCase()}: {orientationMessage}
						</p>
					</fieldset>
					<label class="simulation-speed-control">
						<span>速度</span>
						<select
							bind:value={simulationSpeed}
							disabled={phase !== 'ready' || outletPending}
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
					<div class="rocking-group">
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
						<button
							type="button"
							class="secondary rocking-toggle"
							onclick={toggleHorizontalOscillation}
							disabled={phase !== 'ready'}
							aria-pressed={horizontalOscillationAmplitude > 0}
						>
							{horizontalOscillationAmplitude > 0 ? '左右揺れを停止' : '既定の左右揺れを開始（±240）'}
						</button>
					</div>
					<div class="buttons">
						<button
							type="button"
							onclick={togglePaused}
							disabled={phase !== 'ready' || outletPending}
							aria-pressed={paused}
						>
							{paused ? '再開' : '一時停止'}
						</button>
						<button
							type="button"
							class="secondary"
							onclick={stepOnce}
							disabled={
								phase !== 'ready' ||
								!paused ||
								outletPending ||
								stateCheckPending ||
								cflSpeedReadbackCount > 0
							}
						>
							1 ステップ
						</button>
						<button
							type="button"
							class="secondary"
							onclick={checkGpuState}
							disabled={
								phase !== 'ready' ||
								!paused ||
								outletPending ||
								cflSpeedReadbackCount > 0 ||
								stateCheckPending
							}
						>
							{stateCheckPending ? '確認中…' : '状態を確認'}
						</button>
						<button
							type="button"
							class="secondary"
							onclick={cloneSelection}
							disabled={
								phase !== 'ready' ||
								!selection ||
								selection.objectId === GATE4_FRACTURE_OBJECT_ID ||
								outletPending ||
								cflSpeedReadbackCount > 0 ||
								activePointerId !== null ||
								dragActive
							}
							aria-label={`選択オブジェクトを複製（+${GATE4_CLONE_OFFSET.x}, ${GATE4_CLONE_OFFSET.y}）`}
							title={`選択オブジェクトを複製（+${GATE4_CLONE_OFFSET.x}, ${GATE4_CLONE_OFFSET.y}）`}
							aria-describedby="clone-note"
						>
							選択を複製
						</button>
					</div>
					<div class="simulation-backlog">
						<p>
							未処理シミュレーション時間:
							<output>{simulationAccumulatorSeconds.toFixed(3)} 秒</output>
						</p>
						<p id="simulation-backlog-note" class="control-note">
							処理上限超過分は保持され、要求倍率より遅れて進みます。一時停止中は粒子状態を保ったまま、未処理時間だけを明示的に破棄できます。
						</p>
						<button
							type="button"
							class="secondary"
							onclick={discardDeferredSimulationTime}
							disabled={
								phase !== 'ready' ||
								!paused ||
								outletPending ||
								cflSpeedReadbackCount > 0 ||
								simulationAccumulatorSeconds <= 0
							}
							aria-describedby="simulation-backlog-note"
						>
							未処理時間を破棄して続行
						</button>
					</div>
				</div>
				<p id="clone-note" class="control-note">
					通常の選択オブジェクトは clone できます。GPU 破断 body は選択・ドラッグできますが、壊れた bond を再生成しないよう clone は無効です。
				</p>
				<section class="source-controls" aria-labelledby="source-title">
					<h3 id="source-title">材料 emitter · 容量 · outlet</h3>
					<p class="control-note">
						各追加は seed + burst 番号から決定論的に配置します。水・油は参照面積{GATE4_LIQUID_PARTICLE_AREA.toLocaleString()}、固体は{PARTICLE_VOLUME}を使います。GPU 粒子 state が正本です。一時停止中は連続追加も停止します。容量超過分は追加せず件数を報告し、削除スロットは再利用しません。
					</p>
					<div class="source-grid">
						<label>
							<span>材料</span>
							<select bind:value={emitterMaterial} disabled={phase !== 'ready' || outletPending}>
								<option value="water">水</option>
								<option value="oil">油</option>
								<option value="neo-hookean">弾性固体（bondなし）</option>
							</select>
						</label>
						<label>
							<span>形状</span>
							<select bind:value={emitterShape} disabled={phase !== 'ready' || outletPending}>
								<option value="circle">円</option>
								<option value="rectangle">矩形</option>
							</select>
						</label>
						<label>
							<span>個数 / burst</span>
							<input type="number" min="1" max="100000" step="1" bind:value={emitterCount} disabled={phase !== 'ready' || outletPending} />
						</label>
						<label>
							<span>seed (uint32)</span>
							<input
								type="number"
								min="0"
								max="4294967295"
								step="1"
								bind:value={emitterSeed}
								onchange={resetEmitterSequence}
								disabled={phase !== 'ready' || outletPending}
							/>
						</label>
						<label>
							<span>中心 X · 0–{WORLD_WIDTH}</span>
							<input type="number" min="0" max={WORLD_WIDTH} step="5" bind:value={emitterX} disabled={phase !== 'ready' || outletPending} />
						</label>
						<label>
							<span>中心 Y · 0–{WORLD_HEIGHT}</span>
							<input type="number" min="0" max={WORLD_HEIGHT} step="5" bind:value={emitterY} disabled={phase !== 'ready' || outletPending} />
						</label>
						<label>
							<span>粒子間隔</span>
							<input type="number" min="0.1" max="100" step="0.5" bind:value={emitterSpacing} disabled={phase !== 'ready' || outletPending} />
						</label>
						<label>
							<span>初速 X</span>
							<input
								type="number"
								step="any"
								bind:value={emitterVelocityX}
								disabled={phase !== 'ready' || outletPending}
								aria-describedby="emitter-velocity-note"
							/>
						</label>
						<label>
							<span>初速 Y</span>
							<input
								type="number"
								step="any"
								bind:value={emitterVelocityY}
								disabled={phase !== 'ready' || outletPending}
								aria-describedby="emitter-velocity-note"
							/>
						</label>
					</div>
					<p id="emitter-velocity-note" class="control-note">
						初速に固定上限はありません。追加前にCFL安全性を確認し、必要なら時間刻みを短縮します。速度が大きいほど要求倍率より遅く進みます。float32で安全な時間刻みを作れない値は追加せず、既存のシミュレーションを継続します。
					</p>
					<p class="seed-note">次の burst seed: {(emitterSeed + emitterSequence) >>> 0}</p>
					<div class="source-buttons">
						<button
							type="button"
							class="secondary"
							onclick={emitSourceBurst}
							disabled={phase !== 'ready' || outletPending || cflSpeedReadbackCount > 0 || activePointerId !== null || dragActive}
						>
							1回追加
						</button>
						<button
							type="button"
							class="secondary"
							onclick={toggleContinuousEmission}
							disabled={phase !== 'ready' || outletPending}
							aria-pressed={continuousEmission}
						>
							{continuousEmission ? '連続追加を停止' : '連続追加を開始'}
						</button>
						<button
							type="button"
							class="secondary"
							onclick={resetEmitterSequence}
							disabled={phase !== 'ready' || outletPending}
						>
							seed 列をリセット
						</button>
					</div>
					<p class="emission-status" class:error={emissionIsError} role="status" aria-live="polite" aria-atomic="true">
						{emissionMessage}
					</p>
					<div class="outlet-controls">
						<p>
							出口領域: X 0–{WORLD_WIDTH}, Y {OUTLET_BOUNDS.minY}–{WORLD_HEIGHT} · 水・油のみ GPU 上で除去
						</p>
						<button
							type="button"
							class="secondary"
							onclick={drainOutlet}
							disabled={
								phase !== 'ready' ||
								outletPending ||
								cflSpeedReadbackCount > 0 ||
								activePointerId !== null ||
								dragActive
							}
						>
							{outletPending ? '出口処理中…' : '出口粒子を除去'}
						</button>
						<p class="outlet-status" class:error={outletIsError} role="status" aria-live="polite">
							{outletMessage}
						</p>
					</div>
					<p class="fracture-note">
						GPU 破断 body: {GATE4_FRACTURE_PARTICLE_COUNT} node · {GATE4_FRACTURE_BOND_COUNT} 初期 bond · 引張 strain 閾値 {GATE4_FRACTURE_FAILURE_STRAIN}。
						損傷に応じた赤色化は粒子診断表示でも確認できます。bond の破断は永続し、複製は無効です。bond は初期 body のみを接続し emitter 粒子には追加されません。
						表示は格子解像度と node spacing に依存する離散近似で、連続体の破壊強度を表しません。
					</p>
				</section>
				<p id="render-mode-description" class="render-mode-note">
					自由表面は粒子の現在面積から GPU 上で作る表示専用フィールドです。各画素は優勢な相の色で表示し、水と油の占有が同程度の境界だけを明るく示します。これは描画上の色分けであり、物理的な完全分離を意味しません。小石はGPU上で塗り三角形と明るい輪郭線を重ね、物体形状を表示します。粒子マーカー（診断）では全材料の粒子位置も表示します。材料間接触は表示場とは別の物理 pass で解きます。
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
			<span><i class="water-swatch"></i> 水 {GATE4_WATER_PARTICLE_COUNT.toLocaleString()}粒子（密度 0.02・圧縮係数 {(40_000 * GATE4_LIQUID_BULK_MODULUS_SCALE).toLocaleString()}）</span>
			<span><i class="oil-swatch"></i> 油 {GATE4_OIL_PARTICLE_COUNT.toLocaleString()}粒子（密度 0.016・圧縮係数 {(30_000 * GATE4_LIQUID_BULK_MODULUS_SCALE).toLocaleString()}）</span>
			<span><i class="pebble-swatch"></i> 弾性小石{GATE4_PEBBLE_COUNT}個（各3粒子・Neo-Hookean・λ 20・μ 10）</span>
			<span><i class="fracture-swatch"></i> GPU fracture body（{GATE4_FRACTURE_PARTICLE_COUNT} node・{GATE4_FRACTURE_BOND_COUNT} bond・赤は damage）</span>
			固体粒子の参照面積は旧25から100へ4倍、水・油は参照面積100・密度を保ち、総質量・参照体積も従来sceneから変えていません。水・油の弱圧縮EOS係数を4倍（160,000・120,000）にして、同じ圧縮に対する復元圧力を強めました。相別grid fieldで従来の接近速度除去に加え、両相が同じcellを占める割合と接近速度に応じた分離biasを与え、等量・反対向きのimpulseで運動量を保ちます。node-levelの接近速度に対するtargetはその2倍ですが、これは接触・圧縮抵抗の近似であり、任意状態の完全な非混和性や粒子貫入ゼロを保証しません。粒子数・初期座標・間隔18 world unitsは変更していません。小石は一辺132 world units、中心間隔150の正三角形を塗りと輪郭線で描画します。液体の初速は各成分±{GATE4_LIQUID_INITIAL_VELOCITY_COMPONENT_MAX}、小石頂点は±{GATE4_PEBBLE_INITIAL_VELOCITY_COMPONENT_MAX} world units/s、左右揺れの初期値は0です。界面表示は物理場とは独立した描画近似です。controlled reference testは圧縮応答・接触impulse・質量・運動量を検査しますが、幅広い速度・解像度・長時間挙動を保証しません。表面張力、濡れ、接線摩擦、連続衝突検出、界面の微視的物理は未実装です。
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
					<dt>未処理シミュレーション時間</dt>
					<dd>{simulationAccumulatorSeconds.toFixed(3)} 秒</dd>
				</div>
				<div>
					<dt>描画目標（上限）</dt>
					<dd>{renderTargetFps} FPS</dd>
				</div>
				<div>
					<dt>要求時間倍率</dt>
					<dd>{simulationSpeed.toFixed(2)}×</dd>
				</div>
				<div>
					<dt>物理 macrosteps</dt>
					<dd>
						{measuredMacroStepsPerSecond === null
							? '計測中'
							: `${measuredMacroStepsPerSecond.toFixed(1)} macrosteps/s`}
					</dd>
				</div>
				<div>
					<dt>CFL 数値 microsteps</dt>
					<dd>
						{measuredMicroStepsPerSecond === null
							? '計測中'
							: `${measuredMicroStepsPerSecond.toFixed(1)} microsteps/s`}
					</dd>
				</div>
				<div>
					<dt>実効時間倍率</dt>
					<dd>
						{measuredMacroStepsPerSecond === null
							? '計測中'
							: `${(measuredMacroStepsPerSecond * FIXED_STEP_SECONDS).toFixed(2)}×`}
					</dd>
				</div>
				<div>
					<dt>粒子 microstep 更新量</dt>
					<dd>
						{measuredParticleMicrostepsPerSecond === null
							? '計測中'
							: `${Math.round(measuredParticleMicrostepsPerSecond)} particle-microsteps/s`}
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
				<div>
					<dt>GPU 速度 scalar readback 平均</dt>
					<dd>
						{averageCflReadbackMs === null
							? cflSpeedReadbackCount > 0
								? '計測中'
								: '—'
							: `${averageCflReadbackMs.toFixed(2)} ms`}
					</dd>
				</div>
			</dl>
			<p>
				描画目標は15・30・60・90・120 FPSから選択できます。実際の描画FPSはディスプレイの更新頻度と GPU 処理時間に制約されます。
				標準物理 macrostep は 1/240 秒で、CFL 条件に応じて数値 microstep に分割します。1 batch の初回処理上限は合計512 microstepです。処理量が上限を超えた場合は一部を実行して未処理時間を累積し、後続batchへ繰り越します。固定macrostepが上限に収まらない高速度では、macrostep自体を正のfloat32安全刻みまで短縮するため、クラッシュではなく時間の進みが遅くなります。未処理時間は自動破棄せず、要求・実効倍率とともに表示します。一時停止中は粒子状態を保ったまま、未処理時間だけを明示的に破棄できます。
				Gate 4 の波速上限は液体 EOS と Neo-Hookean acoustic tensor から導出し、速度包絡は GPU 上の粒子最大速度と重力・左右揺れ・有限半径ドラッグの入力包絡から batch ごとに計算します。GPU guard は各 microstep の実速度を、選択した float32 microDt で波速と合わせて CFL 条件を満たす速度許容域内か検査します。Neo-Hookean 変形行列の主伸長が安全域[0.05, 20]を外れる場合は範囲内に射影し、float32での特異化・極端な伸長を防ぎます。CFL 違反や診断可能な変形失敗では、違反 microstep を rollback して最後の有効状態から残り時間を少なくとも倍の刻みで最大4回まで再試行します。初回batchは512 microstepまで、適応再試行は失敗済みworkを含め合計1024 microstepまでです。emitterの初速は追加前にCFL事前確認し、float32で正の安全刻みを作れないbatchは既存状態を止めずに拒否します。回復不能な数値異常や有効な進捗がない場合は理由を表示して停止し、初期シーンからの再起動を案内します。
				左右揺れはシミュレーション時間に同期し、batch区間の平均加速度を適用します（0で停止）。queue 完了待ちは GPU 単体の実行時間ではなく、完了を待ってから次の frame を送信します。毎 physics batch の速度計測は GPU 最大値 reduction と非同期の小さな scalar/status readback で、全 particle state の readback は行いません。表示する readback 平均値にはこのコストを含みます。
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
		<strong>検証範囲と限界:</strong> 2D weakly-compressible MLS-MPM の P2G・grid update・G2P、水・油の別々の材料/速度場、node-owned normal contact、粒子 emitter、GPU outlet、GPU solid-bond damage を含みます。Controlled reference tests は相 identity、相ごとの mass と接触 momentum、fracture damage の閾値・保存・局所化・fragment motion・bounds を確認します。初回 CPU/GPU 比較は fluid と既存小石の局所 sample で、GPU fracture の定量精度比較ではありません。
		水・油は field 分離と相別の法線接触によって制御ケースで非混合 interface を維持しますが、任意初期配置での完全分離は保証しません。relative buoyancy は近似、surface tension・wetting・接線摩擦・gas pressure・cavitation・bubble transport は未実装です。出口は water/oil のみを除去し、removed slot は圧縮・再利用しません。粒子数は容量を一度予約すると減りません。
		連結 fracture body は96個の node と局所 bond topology を GPU 上で更新し、damage を赤色化します。閾値はこの配置用の探索値で、破壊エネルギーや解像度非依存の material strength ではありません。mesh spacing 9 world units と grid cell 10 の解像度に依存し、fragment は bond failure 後も粒子/弾性 constitutive response で近似します。bonded body は clone を禁止し、broken topology を再生成しません。continuous solid fracture の一般性・負荷範囲・性能は未検証です。
		一時停止中の状態確認以外に全 particle state を CPU に readback しません。GPU 数値 guard・CFL 上限と初回 CPU parity は異常検出/局所比較であり、実機での網羅性や数値 fidelity の証明ではありません。液体の det(F) は既存の0.5〜2制限です。
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

	.gravity-controls {
		display: grid;
		gap: 0.65rem;
		min-width: 0;
		margin: 0;
		padding: 0.75rem;
		border: 1px solid #31445a;
		border-radius: 0.55rem;
		background: #0d1823;
	}

	.gravity-controls legend {
		padding: 0 0.35rem;
		font-weight: 650;
	}

	.gravity-controls label {
		display: grid;
		grid-template-columns: minmax(0, 1fr) auto;
		align-items: center;
		gap: 0.35rem 0.65rem;
		min-width: 0;
	}

	.gravity-controls label span {
		grid-column: 1;
		grid-row: 1;
	}

	.gravity-controls label input {
		grid-column: 1 / -1;
		grid-row: 2;
		width: 100%;
		min-width: 0;
		margin: 0;
	}

	.gravity-controls label output {
		grid-column: 2;
		grid-row: 1;
	}

	.gravity-readout,
	.sensor-status,
	.control-note,
	.seed-note,
	.emission-status,
	.outlet-controls p,
	.fracture-note,
	.simulation-backlog p,
	.recovery-controls p {
		margin: 0;
		color: #b8c7d8;
		font-size: 0.84rem;
		line-height: 1.5;
		overflow-wrap: anywhere;
	}

	.sensor-status.error,
	.emission-status.error,
	.outlet-status.error {
		color: #ffb4a9;
	}

	.gravity-controls button {
		width: 100%;
	}

	.source-controls {
		display: grid;
		gap: 0.7rem;
		margin-top: 0.9rem;
		padding: 0.8rem;
		border: 1px solid #31445a;
		border-radius: 0.55rem;
		background: #0d1823;
	}

	.source-controls h3 {
		margin: 0;
		font-size: 1rem;
	}

	.simulation-backlog,
	.recovery-controls {
		display: grid;
		gap: 0.55rem;
		padding: 0.75rem;
		border: 1px solid #31445a;
		border-radius: 0.55rem;
		background: #0d1823;
	}

	.simulation-backlog output {
		font-weight: 700;
	}

	.source-grid {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 0.6rem;
	}

	.source-grid label {
		display: grid;
		gap: 0.3rem;
		min-width: 0;
		color: #d5e9fa;
		font-size: 0.83rem;
	}

	.source-grid input,
	.source-grid select {
		box-sizing: border-box;
		width: 100%;
		min-width: 0;
		min-height: 2.5rem;
		padding: 0.35rem 0.45rem;
		border: 1px solid #31445a;
		border-radius: 0.4rem;
		color: inherit;
		background: #0b1725;
		font: inherit;
		font-variant-numeric: tabular-nums;
	}

	.source-grid input:focus-visible,
	.source-grid select:focus-visible,
	.gravity-controls input:focus-visible {
		outline: 3px solid #8ec9ff;
		outline-offset: 2px;
	}

	.source-buttons {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 0.55rem;
	}

	.outlet-controls {
		display: grid;
		gap: 0.5rem;
		padding-top: 0.7rem;
		border-top: 1px solid #31445a;
	}

	.outlet-status {
		margin: 0;
		color: #9de2bc;
		font-size: 0.84rem;
		line-height: 1.5;
		overflow-wrap: anywhere;
	}

	.fracture-note {
		padding-top: 0.65rem;
		border-top: 1px solid #31445a;
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

	.fracture-swatch {
		background: #ef2921;
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

	.rocking-group {
		display: grid;
		grid-column: 1 / -1;
		gap: 0.5rem;
		min-width: 0;
	}

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

	.rocking-control > span,
	.simulation-speed-control > span,
	.display-mode-control > span,
	.render-fps-control > span {
		grid-column: 1;
		grid-row: 1;
	}

	.rocking-control input,
	.simulation-speed-control select,
	.display-mode-control select,
	.render-fps-control select {
		grid-column: 1 / -1;
		grid-row: 2;
		width: 100%;
		min-width: 0;
	}

	.rocking-control input {
		margin: 0;
	}

	.rocking-toggle {
		width: 100%;
	}

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

		.gravity-controls {
			grid-column: 1 / -1;
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
