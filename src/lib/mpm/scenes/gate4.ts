import type {
	MpmMaterialParameters,
	MpmParticle,
	MpmSolidBond,
	MpmVector2
} from '../reference/solver.js';

export const GATE4_GRID = { cellsX: 220, cellsY: 150, cellSize: 10 } as const;
export const GATE4_PARTICLE_CAPACITY = 1_000_000;
export const GATE4_CLONE_OFFSET: MpmVector2 = { x: 72, y: 0 };
export const GATE4_WATER_PARTICLE_COLUMNS = 50;
export const GATE4_WATER_PARTICLE_ROWS = 40;
export const GATE4_WATER_PARTICLE_COUNT =
	GATE4_WATER_PARTICLE_COLUMNS * GATE4_WATER_PARTICLE_ROWS;
export const GATE4_OIL_PARTICLE_COLUMNS = 50;
export const GATE4_OIL_PARTICLE_ROWS = 30;
export const GATE4_OIL_PARTICLE_COUNT =
	GATE4_OIL_PARTICLE_COLUMNS * GATE4_OIL_PARTICLE_ROWS;
export const GATE4_LIQUID_PARTICLE_SPACING = 18;
export const GATE4_LIQUID_START_X = 100;
export const GATE4_WATER_START_Y = 640;
export const GATE4_OIL_START_Y = 100;
export const GATE4_PARTICLE_AREA = 100;
export const GATE4_SOLID_PARTICLE_MASS = 4;
export const GATE4_PEBBLE_COUNT = 50;
export const GATE4_PEBBLE_PARTICLES_PER_OBJECT = 3;
export const GATE4_PEBBLE_PARTICLE_COUNT =
	GATE4_PEBBLE_COUNT * GATE4_PEBBLE_PARTICLES_PER_OBJECT;
export const GATE4_FRACTURE_COLUMNS = 6;
export const GATE4_FRACTURE_ROWS = 4;
export const GATE4_FRACTURE_SPACING = 20;
export const GATE4_FRACTURE_PARTICLE_COUNT =
	GATE4_FRACTURE_COLUMNS * GATE4_FRACTURE_ROWS;
export const GATE4_FRACTURE_BOND_COUNT =
	(GATE4_FRACTURE_COLUMNS - 1) * GATE4_FRACTURE_ROWS +
	(GATE4_FRACTURE_ROWS - 1) * GATE4_FRACTURE_COLUMNS +
	(GATE4_FRACTURE_COLUMNS - 1) * (GATE4_FRACTURE_ROWS - 1);
export const GATE4_FRACTURE_OBJECT_ID = 3 + GATE4_PEBBLE_COUNT;
export const GATE4_FRACTURE_FAILURE_STRAIN = 0.12;
export const GATE4_PEBBLE_COLUMNS = 8;
export const GATE4_PEBBLE_ROWS = Math.ceil(GATE4_PEBBLE_COUNT / GATE4_PEBBLE_COLUMNS);
export const GATE4_PEBBLE_SPACING = 150;
export const GATE4_LIQUID_INITIAL_VELOCITY_COMPONENT_MAX = 8;
export const GATE4_PEBBLE_INITIAL_VELOCITY_COMPONENT_MAX = 30;
const PEBBLE_BASE_WIDTH_RATIO = 0.005;
const PEBBLE_SIZE_MULTIPLIER = 12;
const UINT32_RANGE = 0x1_0000_0000;
export const GATE4_PEBBLE_EDGE_LENGTH =
	GATE4_GRID.cellsX *
	GATE4_GRID.cellSize *
	PEBBLE_BASE_WIDTH_RATIO *
	PEBBLE_SIZE_MULTIPLIER;
export const GATE4_PARTICLE_COUNT =
	GATE4_WATER_PARTICLE_COUNT +
	GATE4_OIL_PARTICLE_COUNT +
	GATE4_PEBBLE_PARTICLE_COUNT +
	GATE4_FRACTURE_PARTICLE_COUNT;

const PEBBLE_CENTER: MpmVector2 = { x: 700, y: 569 };
const PEBBLE_FIRST_OBJECT_ID = 3;
const FRACTURE_CENTER: MpmVector2 = { x: 1_520, y: 1_380 };
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
		startX: GATE4_LIQUID_START_X,
		startY: GATE4_WATER_START_Y,
		columns: GATE4_WATER_PARTICLE_COLUMNS,
		rows: GATE4_WATER_PARTICLE_ROWS,
		probeRow: 0,
		probeColumn: Math.floor(GATE4_WATER_PARTICLE_COLUMNS / 2),
		material: WATER_MATERIAL
	},
	{
		objectId: 2,
		startX: GATE4_LIQUID_START_X,
		startY: GATE4_OIL_START_Y,
		columns: GATE4_OIL_PARTICLE_COLUMNS,
		rows: GATE4_OIL_PARTICLE_ROWS,
		probeRow: GATE4_OIL_PARTICLE_ROWS - 1,
		probeColumn: Math.floor(GATE4_OIL_PARTICLE_COLUMNS / 2),
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

function createSeededRandom(seed: number): () => number {
	if (!Number.isSafeInteger(seed) || seed < 0 || seed >= UINT32_RANGE) {
		throw new RangeError('The scene seed must be an unsigned 32-bit integer.');
	}

	let state = seed;
	return () => {
		state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
		return state / UINT32_RANGE;
	};
}

function createRandomVelocity(random: () => number, componentMax: number): MpmVector2 {
	return {
		x: (random() * 2 - 1) * componentMax,
		y: (random() * 2 - 1) * componentMax
	};
}

export interface MpmGate4Scene {
	readonly particles: readonly MpmParticle[];
	readonly parityProbeIndices: readonly number[];
	readonly fractureBonds: readonly MpmSolidBond[];
}

export interface MpmCpuParitySample {
	readonly particles: readonly MpmParticle[];
	readonly sourceIndices: readonly number[];
	readonly probeSampleIndices: readonly number[];
}

export function createMpmGate4Scene(
	seed = Math.floor(Math.random() * UINT32_RANGE)
): MpmGate4Scene {
	const random = createSeededRandom(seed);
	const particles: MpmParticle[] = [];
	const parityProbeIndices: number[] = [];

	for (const object of OBJECTS) {
		const objectStart = particles.length;
		for (let row = 0; row < object.rows; row += 1) {
			for (let column = 0; column < object.columns; column += 1) {
				particles.push({
					position: {
						x: object.startX + column * GATE4_LIQUID_PARTICLE_SPACING,
						y: object.startY + row * GATE4_LIQUID_PARTICLE_SPACING
					},
					velocity: createRandomVelocity(
						random,
						GATE4_LIQUID_INITIAL_VELOCITY_COMPONENT_MAX
					),
					C: ZERO_MATRIX,
					F: INITIAL_DEFORMATION,
					objectId: object.objectId,
					plasticVolume: 1,
					volume: GATE4_PARTICLE_AREA,
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
		const row = Math.floor(pebbleIndex / GATE4_PEBBLE_COLUMNS);
		const rowStartIndex = row * GATE4_PEBBLE_COLUMNS;
		const pebblesInRow = Math.min(
			GATE4_PEBBLE_COLUMNS,
			GATE4_PEBBLE_COUNT - rowStartIndex
		);
		const column =
			pebbleIndex - rowStartIndex + (GATE4_PEBBLE_COLUMNS - pebblesInRow) * 0.5;
		const center = {
			x:
				PEBBLE_CENTER.x +
				(column - (GATE4_PEBBLE_COLUMNS - 1) / 2) *
					GATE4_PEBBLE_SPACING,
			y:
				PEBBLE_CENTER.y +
				(row - (GATE4_PEBBLE_ROWS - 1) / 2) * GATE4_PEBBLE_SPACING
		};
		const pebbleStart = particles.length;
		const objectId = PEBBLE_FIRST_OBJECT_ID + pebbleIndex;
		const vertices = getPebbleVertices(center);
		for (let vertex = 0; vertex < vertices.length; vertex += 1) {
			particles.push({
				position: vertices[vertex],
				velocity: createRandomVelocity(
					random,
					GATE4_PEBBLE_INITIAL_VELOCITY_COMPONENT_MAX
				),
				C: ZERO_MATRIX,
				F: INITIAL_DEFORMATION,
				objectId,
				bondedTriangleVertex: vertex as 0 | 1 | 2,
				plasticVolume: 1,
				volume: GATE4_PARTICLE_AREA,
				material: PEBBLE_MATERIAL
			});
		}
		if (pebbleIndex === parityPebbleIndex) {
			parityProbeIndices.push(pebbleStart + GATE4_PEBBLE_PARTICLES_PER_OBJECT - 1);
		}
	}

	const fractureStart = particles.length;
	const fractureVelocity = createRandomVelocity(random, 10);
	for (let row = 0; row < GATE4_FRACTURE_ROWS; row += 1) {
		for (let column = 0; column < GATE4_FRACTURE_COLUMNS; column += 1) {
			const centeredColumn = column - (GATE4_FRACTURE_COLUMNS - 1) / 2;
			const centeredRow = row - (GATE4_FRACTURE_ROWS - 1) / 2;
			particles.push({
				position: {
					x: FRACTURE_CENTER.x + centeredColumn * GATE4_FRACTURE_SPACING,
					y: FRACTURE_CENTER.y + centeredRow * GATE4_FRACTURE_SPACING
				},
				velocity: fractureVelocity,
				C: ZERO_MATRIX,
				F: INITIAL_DEFORMATION,
				objectId: GATE4_FRACTURE_OBJECT_ID,
				plasticVolume: 1,
				volume: GATE4_PARTICLE_AREA,
				material: PEBBLE_MATERIAL
			});
		}
	}
	const fractureBonds: MpmSolidBond[] = [];
	const fractureIndex = (row: number, column: number) =>
		fractureStart + row * GATE4_FRACTURE_COLUMNS + column;
	for (let row = 0; row < GATE4_FRACTURE_ROWS; row += 1) {
		for (let column = 0; column < GATE4_FRACTURE_COLUMNS; column += 1) {
			const firstParticle = fractureIndex(row, column);
			if (column + 1 < GATE4_FRACTURE_COLUMNS) {
				fractureBonds.push({
					firstParticle,
					secondParticle: firstParticle + 1,
					restLength: GATE4_FRACTURE_SPACING,
					failureStrain: GATE4_FRACTURE_FAILURE_STRAIN
				});
			}
			if (row + 1 < GATE4_FRACTURE_ROWS) {
				fractureBonds.push({
					firstParticle,
					secondParticle: firstParticle + GATE4_FRACTURE_COLUMNS,
					restLength: GATE4_FRACTURE_SPACING,
					failureStrain: GATE4_FRACTURE_FAILURE_STRAIN
				});
			}
			if (
				row + 1 < GATE4_FRACTURE_ROWS &&
				column + 1 < GATE4_FRACTURE_COLUMNS
			) {
				fractureBonds.push({
					firstParticle,
					secondParticle: firstParticle + GATE4_FRACTURE_COLUMNS + 1,
					restLength: GATE4_FRACTURE_SPACING * Math.SQRT2,
					failureStrain: GATE4_FRACTURE_FAILURE_STRAIN
				});
			}
		}
	}

	if (fractureBonds.length !== GATE4_FRACTURE_BOND_COUNT) {
		throw new Error(`Gate 4 fracture body must contain ${GATE4_FRACTURE_BOND_COUNT} bonds.`);
	}
	if (particles.length !== GATE4_PARTICLE_COUNT) {
		throw new Error(`Gate 4 scene must contain ${GATE4_PARTICLE_COUNT} particles.`);
	}

	return { particles, parityProbeIndices, fractureBonds };
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
	// Keep the parity neighborhood closed across enlarged pebble/liquid contacts over multiple microsteps.
	const interactionRadius = gridCellSize * 20;
	const sourceIndices = new Set<number>();
	for (let index = 0; index < particles.length; index += 1) {
		const particle = particles[index];
		if (
			probeParticles.some(
				(probe) =>
					Math.abs(particle.position.x - probe.position.x) <= interactionRadius &&
					Math.abs(particle.position.y - probe.position.y) <= interactionRadius
			)
		) {
			sourceIndices.add(index);
		}
	}
	const contactRadius = interactionRadius + 2 * gridCellSize;
	for (let bodyStart = 0; bodyStart < particles.length; bodyStart += 1) {
		if (particles[bodyStart].bondedTriangleVertex !== 0) continue;
		const body = particles.slice(bodyStart, bodyStart + 3);
		if (
			body.length !== 3 ||
			body.some(
				(particle, vertex) => particle.bondedTriangleVertex !== vertex
			)
		) {
			throw new RangeError(`Bonded triangle at particle ${bodyStart} is incomplete.`);
		}
		const minX = Math.min(...body.map(({ position }) => position.x));
		const maxX = Math.max(...body.map(({ position }) => position.x));
		const minY = Math.min(...body.map(({ position }) => position.y));
		const maxY = Math.max(...body.map(({ position }) => position.y));
		const relevant = probeParticles.some(
			({ position }) =>
				position.x >= minX - contactRadius &&
				position.x <= maxX + contactRadius &&
				position.y >= minY - contactRadius &&
				position.y <= maxY + contactRadius
		);
		if (!relevant) continue;
		for (let vertex = 0; vertex < 3; vertex += 1) {
			sourceIndices.add(bodyStart + vertex);
		}
		for (let index = 0; index < particles.length; index += 1) {
			const { position } = particles[index];
			if (
				position.x >= minX - interactionRadius &&
				position.x <= maxX + interactionRadius &&
				position.y >= minY - interactionRadius &&
				position.y <= maxY + interactionRadius
			) {
				sourceIndices.add(index);
			}
		}
	}
	for (const sourceIndex of [...sourceIndices]) {
		const vertex = particles[sourceIndex].bondedTriangleVertex;
		if (vertex === undefined) continue;
		const bodyStart = sourceIndex - vertex;
		const body = particles.slice(bodyStart, bodyStart + 3);
		if (
			body.length !== 3 ||
			body.some((particle, expectedVertex) => particle.bondedTriangleVertex !== expectedVertex)
		) {
			throw new RangeError(`Bonded triangle at particle ${bodyStart} is incomplete.`);
		}
		for (let expectedVertex = 0; expectedVertex < 3; expectedVertex += 1) {
			sourceIndices.add(bodyStart + expectedVertex);
		}
	}
	const orderedSourceIndices = [...sourceIndices].sort((first, second) => first - second);

	const sampleIndexBySource = new Map(
		orderedSourceIndices.map((sourceIndex, sampleIndex) => [
			sourceIndex,
			sampleIndex
		])
	);
	const probeSampleIndices = probeIndices.map((sourceIndex) => {
		const sampleIndex = sampleIndexBySource.get(sourceIndex);
		if (sampleIndex === undefined) {
			throw new Error(`CPU/GPU parity probe ${sourceIndex} is missing from its local sample.`);
		}
		return sampleIndex;
	});

	return {
		particles: orderedSourceIndices.map((index) => particles[index]),
		sourceIndices: orderedSourceIndices,
		probeSampleIndices
	};
}
