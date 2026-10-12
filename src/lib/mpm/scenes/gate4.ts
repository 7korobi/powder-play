import type { MpmMaterialParameters, MpmParticle, MpmVector2 } from '../reference/solver.js';

export const GATE4_GRID = { cellsX: 220, cellsY: 150, cellSize: 10 } as const;
export const GATE4_PARTICLE_CAPACITY = 1_000_000;
export const GATE4_CLONE_OFFSET: MpmVector2 = { x: 72, y: 0 };
export const GATE4_PEBBLE_COUNT = 200;
export const GATE4_PEBBLE_PARTICLES_PER_OBJECT = 3;
export const GATE4_PEBBLE_PARTICLE_COUNT =
	GATE4_PEBBLE_COUNT * GATE4_PEBBLE_PARTICLES_PER_OBJECT;
export const GATE4_PEBBLE_COLUMNS = 20;
export const GATE4_PEBBLE_ROWS = 10;
export const GATE4_PEBBLE_SPACING = 20;
export const GATE4_PEBBLE_EDGE_LENGTH =
	GATE4_GRID.cellsX * GATE4_GRID.cellSize * 0.005;
export const GATE4_PARTICLE_COUNT = 14_000 + GATE4_PEBBLE_PARTICLE_COUNT;

const PARTICLE_SPACING = 9;
const PEBBLE_CENTER: MpmVector2 = { x: 1_520, y: 300 };
const PEBBLE_FIRST_OBJECT_ID = 3;
const PEBBLE_MATERIAL = {
	lambda: 20,
	mu: 10,
	model: 'neo-hookean'
} as const satisfies MpmMaterialParameters;
const ZERO_MATRIX = { m00: 0, m01: 0, m10: 0, m11: 0 } as const;
const INITIAL_DEFORMATION = { m00: 1, m01: 0, m10: 0, m11: 1 } as const;
const WATER_MATERIAL = {
	lambda: 40_000,
	mu: 0,
	model: 'water',
	density: 0.02
} as const satisfies MpmMaterialParameters;
const OIL_MATERIAL = {
	lambda: 30_000,
	mu: 0,
	model: 'oil',
	density: 0.016
} as const satisfies MpmMaterialParameters;
const OBJECTS = [
	{
		objectId: 1,
		startX: 100,
		startY: 640,
		columns: 100,
		rows: 80,
		probeRow: 0,
		probeColumn: 50,
		material: WATER_MATERIAL
	},
	{
		objectId: 2,
		startX: 100,
		startY: 100,
		columns: 100,
		rows: 60,
		probeRow: 59,
		probeColumn: 50,
		material: OIL_MATERIAL
	},
] as const;

function getPebbleVertices(center: MpmVector2): readonly MpmVector2[] {
	return [
		{
			x: center.x - GATE4_PEBBLE_EDGE_LENGTH * 0.5,
			y: center.y - GATE4_PEBBLE_EDGE_LENGTH / (2 * Math.sqrt(3))
		},
		{
			x: center.x + GATE4_PEBBLE_EDGE_LENGTH * 0.5,
			y: center.y - GATE4_PEBBLE_EDGE_LENGTH / (2 * Math.sqrt(3))
		},
		{
			x: center.x,
			y: center.y + GATE4_PEBBLE_EDGE_LENGTH / Math.sqrt(3)
		}
	];
}

export interface MpmGate4Scene {
	readonly particles: readonly MpmParticle[];
	readonly parityProbeIndices: readonly number[];
}

export interface MpmCpuParitySample {
	readonly particles: readonly MpmParticle[];
	readonly sourceIndices: readonly number[];
	readonly probeSampleIndices: readonly number[];
}

export function createMpmGate4Scene(): MpmGate4Scene {
	const particles: MpmParticle[] = [];
	const parityProbeIndices: number[] = [];

	for (const object of OBJECTS) {
		const objectStart = particles.length;
		for (let row = 0; row < object.rows; row += 1) {
			for (let column = 0; column < object.columns; column += 1) {
				particles.push({
					position: {
						x: object.startX + column * PARTICLE_SPACING,
						y: object.startY + row * PARTICLE_SPACING
					},
					velocity: { x: 0, y: 0 },
					C: ZERO_MATRIX,
					F: INITIAL_DEFORMATION,
					objectId: object.objectId,
					plasticVolume: 1,
					material: object.material
				});
			}
		}
		parityProbeIndices.push(
			objectStart + object.probeRow * object.columns + object.probeColumn
		);
	}

	const parityPebbleIndex =
		Math.floor(GATE4_PEBBLE_ROWS / 2) * GATE4_PEBBLE_COLUMNS +
		Math.floor(GATE4_PEBBLE_COLUMNS / 2);
	for (let pebbleIndex = 0; pebbleIndex < GATE4_PEBBLE_COUNT; pebbleIndex += 1) {
		const center = {
			x:
				PEBBLE_CENTER.x +
				(pebbleIndex % GATE4_PEBBLE_COLUMNS - (GATE4_PEBBLE_COLUMNS - 1) / 2) *
					GATE4_PEBBLE_SPACING,
			y:
				PEBBLE_CENTER.y +
				(Math.floor(pebbleIndex / GATE4_PEBBLE_COLUMNS) -
					(GATE4_PEBBLE_ROWS - 1) / 2) *
					GATE4_PEBBLE_SPACING
		};
		const pebbleStart = particles.length;
		const objectId = PEBBLE_FIRST_OBJECT_ID + pebbleIndex;
		for (const position of getPebbleVertices(center)) {
			particles.push({
				position,
				velocity: { x: 0, y: 0 },
				C: ZERO_MATRIX,
				F: INITIAL_DEFORMATION,
				objectId,
				plasticVolume: 1,
				material: PEBBLE_MATERIAL
			});
		}
		if (pebbleIndex === parityPebbleIndex) {
			parityProbeIndices.push(pebbleStart + GATE4_PEBBLE_PARTICLES_PER_OBJECT - 1);
		}
	}

	if (particles.length !== GATE4_PARTICLE_COUNT) {
		throw new Error(`Gate 4 scene must contain ${GATE4_PARTICLE_COUNT} particles.`);
	}

	return { particles, parityProbeIndices };
}

export function createMpmCpuParitySample(
	particles: readonly MpmParticle[],
	probeIndices: readonly number[],
	gridCellSize: number
): MpmCpuParitySample {
	if (probeIndices.length === 0) {
		throw new RangeError('At least one CPU/GPU parity probe is required.');
	}
	if (!Number.isFinite(gridCellSize) || gridCellSize <= 0) {
		throw new RangeError('gridCellSize must be a positive finite number.');
	}

	const probeParticles = probeIndices.map((index) => {
		if (!Number.isSafeInteger(index) || index < 0 || index >= particles.length) {
			throw new RangeError('CPU/GPU parity probe indices must identify input particles.');
		}
		return particles[index];
	});
	const interactionRadius = gridCellSize * 3;
	const sourceIndices: number[] = [];
	for (let index = 0; index < particles.length; index += 1) {
		const particle = particles[index];
		if (
			probeParticles.some(
				(probe) =>
					Math.abs(particle.position.x - probe.position.x) <= interactionRadius &&
					Math.abs(particle.position.y - probe.position.y) <= interactionRadius
			)
		) {
			sourceIndices.push(index);
		}
	}

	const sampleIndexBySource = new Map(
		sourceIndices.map((sourceIndex, sampleIndex) => [sourceIndex, sampleIndex])
	);
	const probeSampleIndices = probeIndices.map((sourceIndex) => {
		const sampleIndex = sampleIndexBySource.get(sourceIndex);
		if (sampleIndex === undefined) {
			throw new Error(`CPU/GPU parity probe ${sourceIndex} is missing from its local sample.`);
		}
		return sampleIndex;
	});

	return {
		particles: sourceIndices.map((index) => particles[index]),
		sourceIndices,
		probeSampleIndices
	};
}
