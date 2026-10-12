import type { MpmMaterialParameters, MpmParticle } from '../reference/solver.js';

export const GATE4_GRID = { cellsX: 270, cellsY: 180, cellSize: 10 } as const;
export const GATE4_PARTICLE_COUNT = 20_000;
export const GATE4_PARTICLE_CAPACITY = 40_000;

const PARTICLE_SPACING = 8;
const ZERO_MATRIX = { m00: 0, m01: 0, m10: 0, m11: 0 } as const;
const INITIAL_DEFORMATION = { m00: 1, m01: 0, m10: 0, m11: 1 } as const;
const WATER_MATERIAL = {
	lambda: 4_000,
	mu: 0,
	model: 'water',
	density: 0.04
} as const satisfies MpmMaterialParameters;
const OIL_MATERIAL = {
	lambda: 3_000,
	mu: 0,
	model: 'oil',
	density: 0.032
} as const satisfies MpmMaterialParameters;
const POWDER_MATERIAL = {
	lambda: 20,
	mu: 10,
	model: 'powder',
	frictionCoefficient: 0.55
} as const satisfies MpmMaterialParameters;

const OBJECTS = [
	{ objectId: 1, startX: 100, startY: 1_100, columns: 100, rows: 80, material: WATER_MATERIAL },
	{ objectId: 2, startX: 100, startY: 600, columns: 100, rows: 60, material: OIL_MATERIAL },
	{ objectId: 3, startX: 1_600, startY: 100, columns: 100, rows: 60, material: POWDER_MATERIAL }
] as const;

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
			objectStart + Math.floor(object.rows / 2) * object.columns + Math.floor(object.columns / 2)
		);
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
