import type { MpmParticle, MpmSolidBond, MpmVector2 } from '../reference/solver.js';
import {
	createMpmGate4Scene,
	GATE4_FRACTURE_BOND_COUNT,
	GATE4_FRACTURE_FAILURE_STRAIN,
	GATE4_FRACTURE_PARTICLE_COUNT,
	GATE4_FRACTURE_OBJECT_ID,
	GATE4_GRID,
	GATE4_LIQUID_PARTICLE_SPACING,
	GATE4_LIQUID_START_X,
	GATE4_OIL_PARTICLE_COLUMNS,
	GATE4_OIL_PARTICLE_COUNT,
	GATE4_OIL_PARTICLE_ROWS,
	GATE4_PARTICLE_AREA,
	GATE4_PARTICLE_COUNT,
	GATE4_PEBBLE_COUNT,
	GATE4_PEBBLE_PARTICLE_COUNT,
	GATE4_SOLID_PARTICLE_MASS,
	GATE4_WATER_PARTICLE_COLUMNS,
	GATE4_WATER_PARTICLE_COUNT,
	GATE4_WATER_PARTICLE_ROWS
} from '../scenes/gate4.js';

export const LIQUIDFUN_PACKAGE_VERSION = '7.0.0';
export const LIQUIDFUN_SCENE_SEED = 0x4d504d34;
export const LIQUIDFUN_GATE4_UNITS_PER_METER = 100;
export const LIQUIDFUN_DT_SECONDS = 1 / 240;
export const LIQUIDFUN_GATE4_GRAVITY: MpmVector2 = { x: 0, y: 90 };
export const LIQUIDFUN_BOX2D_GRAVITY: MpmVector2 = {
	x: LIQUIDFUN_GATE4_GRAVITY.x / LIQUIDFUN_GATE4_UNITS_PER_METER,
	y: -LIQUIDFUN_GATE4_GRAVITY.y / LIQUIDFUN_GATE4_UNITS_PER_METER
};
export const LIQUIDFUN_SHARED_PARTICLE_DENSITY = 1;
export const LIQUIDFUN_FLUID_PARTICLE_COUNT_MULTIPLIER = 3;
export const LIQUIDFUN_WATER_PARTICLE_COUNT =
	GATE4_WATER_PARTICLE_COUNT * LIQUIDFUN_FLUID_PARTICLE_COUNT_MULTIPLIER;
export const LIQUIDFUN_OIL_PARTICLE_COUNT =
	GATE4_OIL_PARTICLE_COUNT * LIQUIDFUN_FLUID_PARTICLE_COUNT_MULTIPLIER;
export const LIQUIDFUN_PARTICLE_RADIUS_METERS =
	GATE4_LIQUID_PARTICLE_SPACING / LIQUIDFUN_GATE4_UNITS_PER_METER / 2;
export const LIQUIDFUN_SOLID_NODE_AREA_SQUARE_METERS =
	GATE4_PARTICLE_AREA / LIQUIDFUN_GATE4_UNITS_PER_METER ** 2;
export const LIQUIDFUN_SOLID_NODE_RADIUS_METERS = Math.sqrt(
	LIQUIDFUN_SOLID_NODE_AREA_SQUARE_METERS / Math.PI
);
export const LIQUIDFUN_SOLID_NODE_DENSITY =
	GATE4_SOLID_PARTICLE_MASS / LIQUIDFUN_SOLID_NODE_AREA_SQUARE_METERS;
export const LIQUIDFUN_FRACTURE_BOND_FREQUENCY_HZ = 8;
export const LIQUIDFUN_BOND_DAMPING_RATIO = 1;
const LIQUIDFUN_FRACTURE_BOND_ANGULAR_FREQUENCY =
	2 * Math.PI * LIQUIDFUN_FRACTURE_BOND_FREQUENCY_HZ;
const LIQUIDFUN_REDUCED_SOLID_NODE_MASS = GATE4_SOLID_PARTICLE_MASS / 2;
export const LIQUIDFUN_FRACTURE_BOND_STIFFNESS =
	LIQUIDFUN_REDUCED_SOLID_NODE_MASS *
	LIQUIDFUN_FRACTURE_BOND_ANGULAR_FREQUENCY ** 2;
export const LIQUIDFUN_FRACTURE_BOND_DAMPING =
	2 *
	LIQUIDFUN_BOND_DAMPING_RATIO *
	LIQUIDFUN_REDUCED_SOLID_NODE_MASS *
	LIQUIDFUN_FRACTURE_BOND_ANGULAR_FREQUENCY;
export const LIQUIDFUN_FRACTURE_BOND_FAILURE_STRAIN =
	GATE4_FRACTURE_FAILURE_STRAIN;
export const LIQUIDFUN_VELOCITY_ITERATIONS = 8;
export const LIQUIDFUN_POSITION_ITERATIONS = 3;
export const LIQUIDFUN_PEBBLE_SIZE_SCALE = 0.5;
export const LIQUIDFUN_STATIONARY_WARMUP_MS = 600;
export const LIQUIDFUN_GRAVITY_WARMUP_STEPS = 64;
export const LIQUIDFUN_SAMPLE_BATCHES = 12;
export const LIQUIDFUN_STEPS_PER_BATCH = 8;

// Keep the widened water footprint one grid row clear of the fracture nodes.
// Shift both bands by half a spacing to keep water clear of the fracture network.
const LIQUIDFUN_FLUID_VERTICAL_OFFSET = -GATE4_LIQUID_PARTICLE_SPACING / 2;
const LIQUIDFUN_WATER_GRID = {
	columns: Math.ceil(LIQUIDFUN_WATER_PARTICLE_COUNT / GATE4_WATER_PARTICLE_ROWS),
	rows: GATE4_WATER_PARTICLE_ROWS
} as const;
const LIQUIDFUN_OIL_GRID = {
	rows: GATE4_OIL_PARTICLE_ROWS,
	columns: Math.ceil(LIQUIDFUN_OIL_PARTICLE_COUNT / GATE4_OIL_PARTICLE_ROWS)
} as const;
const LIQUIDFUN_FLUID_RIGHT_EDGE =
	GATE4_LIQUID_START_X +
	(Math.max(LIQUIDFUN_WATER_GRID.columns, LIQUIDFUN_OIL_GRID.columns) - 1) *
		GATE4_LIQUID_PARTICLE_SPACING;
const LIQUIDFUN_DOMAIN_RIGHT_MARGIN = GATE4_LIQUID_PARTICLE_SPACING;
const LIQUIDFUN_BASE_DOMAIN_WIDTH = GATE4_GRID.cellsX * GATE4_GRID.cellSize;

export const LIQUIDFUN_GATE4_DOMAIN = {
	width:
		Math.ceil(
			Math.max(
				LIQUIDFUN_BASE_DOMAIN_WIDTH,
				LIQUIDFUN_FLUID_RIGHT_EDGE + LIQUIDFUN_DOMAIN_RIGHT_MARGIN
			) / GATE4_GRID.cellSize
		) * GATE4_GRID.cellSize,
	height: GATE4_GRID.cellsY * GATE4_GRID.cellSize
} as const;

export const LIQUIDFUN_BOX2D_DOMAIN_METERS = {
	width: LIQUIDFUN_GATE4_DOMAIN.width / LIQUIDFUN_GATE4_UNITS_PER_METER,
	height: LIQUIDFUN_GATE4_DOMAIN.height / LIQUIDFUN_GATE4_UNITS_PER_METER
} as const;

export interface LiquidFunVector {
	readonly x: number;
	readonly y: number;
}

export interface LiquidFunTriangleGroupSeed {
	readonly objectId: number;
	readonly nodeIndices: readonly [number, number, number];
}

export interface LiquidFunBondSeed {
	readonly firstNodeIndex: number;
	readonly secondNodeIndex: number;
	readonly restLengthMeters: number;
}

export interface LiquidFunFractureBondSeed extends LiquidFunBondSeed {
	readonly failureStrain: number;
}

export interface LiquidFunGate4Workload {
	readonly seed: number;
	readonly sourceMpmNodeCount: number;
	readonly mappedMpmNodeCount: number;
	readonly waterParticles: readonly MpmParticle[];
	readonly oilParticles: readonly MpmParticle[];
	readonly pebbleNodes: readonly MpmParticle[];
	readonly pebbleGroups: readonly LiquidFunTriangleGroupSeed[];
	readonly fractureNodes: readonly MpmParticle[];
	readonly fractureBonds: readonly LiquidFunFractureBondSeed[];
}

export function scaleLiquidFunPebbleLocalVertices(
	vertices: readonly LiquidFunVector[]
): readonly LiquidFunVector[] {
	if (vertices.length !== 3) {
		throw new RangeError('A LiquidFun pebble must have exactly three triangle vertices.');
	}
	return vertices.map((vertex) => ({
		x: vertex.x * LIQUIDFUN_PEBBLE_SIZE_SCALE,
		y: vertex.y * LIQUIDFUN_PEBBLE_SIZE_SCALE
	}));
}

function expandLiquidParticles(
	sourceParticles: readonly MpmParticle[],
	sourceColumns: number,
	sourceRows: number,
	targetParticleCount: number,
	targetRows: number,
	verticalOffset: number
): readonly MpmParticle[] {
	if (sourceParticles.length !== sourceColumns * sourceRows) {
		throw new Error('LiquidFun source particle count does not match its grid dimensions.');
	}
	if (
		!Number.isSafeInteger(targetParticleCount) ||
		targetParticleCount < targetRows ||
		!Number.isSafeInteger(targetRows) ||
		targetRows < 2 ||
		targetRows !== sourceRows ||
		!Number.isFinite(verticalOffset)
	) {
		throw new RangeError('LiquidFun fluid grids must expand horizontally without changing rows.');
	}
	const firstSourceParticle = sourceParticles[0];
	if (firstSourceParticle === undefined) {
		throw new Error('LiquidFun source particle grid is incomplete.');
	}
	const targetColumns = Math.ceil(targetParticleCount / targetRows);
	const expandedParticles: MpmParticle[] = [];
	for (let row = 0; row < targetRows; row += 1) {
		const rowStartIndex = Math.floor((row * targetParticleCount) / targetRows);
		const rowEndIndex = Math.floor(((row + 1) * targetParticleCount) / targetRows);
		const particlesInRow = rowEndIndex - rowStartIndex;
		const columnOffset =
			((targetColumns - particlesInRow) * GATE4_LIQUID_PARTICLE_SPACING) / 2;
		for (let column = 0; column < particlesInRow; column += 1) {
			const sourceParticle =
				sourceParticles[row * sourceColumns + (column % sourceColumns)];
			if (sourceParticle === undefined) {
				throw new Error('LiquidFun expansion could not find a source particle.');
			}
			expandedParticles.push({
				...sourceParticle,
				position: {
					x:
						firstSourceParticle.position.x +
						columnOffset +
						column * GATE4_LIQUID_PARTICLE_SPACING,
					y:
						firstSourceParticle.position.y +
						verticalOffset +
						row * GATE4_LIQUID_PARTICLE_SPACING
				}
			});
		}
	}
	return expandedParticles;
}

export function mapGate4PositionToBox2D(position: MpmVector2): LiquidFunVector {
	return {
		x: position.x / LIQUIDFUN_GATE4_UNITS_PER_METER,
		y:
			(LIQUIDFUN_GATE4_DOMAIN.height - position.y) /
			LIQUIDFUN_GATE4_UNITS_PER_METER
	};
}

export function mapGate4VelocityToBox2D(velocity: MpmVector2): LiquidFunVector {
	return {
		x: velocity.x / LIQUIDFUN_GATE4_UNITS_PER_METER,
		y: -velocity.y / LIQUIDFUN_GATE4_UNITS_PER_METER
	};
}

export function mapBox2DPositionToGate4(position: LiquidFunVector): MpmVector2 {
	return {
		x: position.x * LIQUIDFUN_GATE4_UNITS_PER_METER,
		y:
			LIQUIDFUN_GATE4_DOMAIN.height -
			position.y * LIQUIDFUN_GATE4_UNITS_PER_METER
	};
}

export function mapBox2DVelocityToGate4(velocity: LiquidFunVector): MpmVector2 {
	return {
		x: velocity.x * LIQUIDFUN_GATE4_UNITS_PER_METER,
		y: -velocity.y * LIQUIDFUN_GATE4_UNITS_PER_METER
	};
}

export function createLiquidFunGate4Workload(): LiquidFunGate4Workload {
	const scene = createMpmGate4Scene(LIQUIDFUN_SCENE_SEED);
	const sourceWaterParticles: MpmParticle[] = [];
	const sourceOilParticles: MpmParticle[] = [];
	const pebbleNodesByObject = new Map<number, MpmParticle[]>();
	const fractureNodes: MpmParticle[] = [];
	const fractureNodeIndices = new Map<number, number>();

	for (let sourceIndex = 0; sourceIndex < scene.particles.length; sourceIndex += 1) {
		const particle = scene.particles[sourceIndex];
		const material = particle.material?.model;
		if (material === 'water') {
			sourceWaterParticles.push(particle);
		} else if (material === 'oil') {
			sourceOilParticles.push(particle);
		} else if (particle.bondedTriangleVertex !== undefined) {
			if (particle.objectId === undefined) {
				throw new Error('Every Gate 4 pebble node must have an object id.');
			}
			const nodes = pebbleNodesByObject.get(particle.objectId) ?? [];
			nodes.push(particle);
			pebbleNodesByObject.set(particle.objectId, nodes);
		} else if (particle.objectId === GATE4_FRACTURE_OBJECT_ID) {
			fractureNodeIndices.set(sourceIndex, fractureNodes.length);
			fractureNodes.push(particle);
		}
	}

	const pebbleNodes: MpmParticle[] = [];
	const pebbleGroups: LiquidFunTriangleGroupSeed[] = [];
	for (const [objectId, nodes] of [...pebbleNodesByObject.entries()].sort(
		([first], [second]) => first - second
	)) {
		const orderedNodes = [...nodes].sort(
			(first, second) =>
				(first.bondedTriangleVertex ?? -1) - (second.bondedTriangleVertex ?? -1)
		);
		if (
			orderedNodes.length !== 3 ||
			orderedNodes.some(
				(node, vertex) =>
					node.objectId !== objectId || node.bondedTriangleVertex !== vertex
			)
		) {
			throw new Error(`Gate 4 pebble ${objectId} must contain vertices 0, 1, and 2.`);
		}

		const firstNodeIndex = pebbleNodes.length;
		pebbleNodes.push(...orderedNodes);
		pebbleGroups.push({
			objectId,
			nodeIndices: [
				firstNodeIndex,
				firstNodeIndex + 1,
				firstNodeIndex + 2
			]
		});
	}

	const fractureBonds: LiquidFunFractureBondSeed[] = scene.fractureBonds.map(
		(bond: MpmSolidBond) => {
			const firstNodeIndex = fractureNodeIndices.get(bond.firstParticle);
			const secondNodeIndex = fractureNodeIndices.get(bond.secondParticle);
			if (firstNodeIndex === undefined || secondNodeIndex === undefined) {
				throw new Error('Every Gate 4 fracture bond must map to fracture nodes.');
			}
			return {
				firstNodeIndex,
				secondNodeIndex,
				restLengthMeters:
					bond.restLength / LIQUIDFUN_GATE4_UNITS_PER_METER,
				failureStrain: bond.failureStrain
			};
		}
	);
	const mappedMpmNodeCount =
		sourceWaterParticles.length +
		sourceOilParticles.length +
		pebbleNodes.length +
		fractureNodes.length;

	if (
		scene.particles.length !== GATE4_PARTICLE_COUNT ||
		sourceWaterParticles.length !== GATE4_WATER_PARTICLE_COUNT ||
		sourceOilParticles.length !== GATE4_OIL_PARTICLE_COUNT ||
		pebbleGroups.length !== GATE4_PEBBLE_COUNT ||
		pebbleNodes.length !== GATE4_PEBBLE_PARTICLE_COUNT ||
		fractureNodes.length !== GATE4_FRACTURE_PARTICLE_COUNT ||
		fractureBonds.length !== GATE4_FRACTURE_BOND_COUNT ||
		mappedMpmNodeCount !== scene.particles.length ||
		fractureBonds.some(
			(bond) =>
				bond.failureStrain !== GATE4_FRACTURE_FAILURE_STRAIN ||
				bond.firstNodeIndex < 0 ||
				bond.firstNodeIndex >= fractureNodes.length ||
				bond.secondNodeIndex < 0 ||
				bond.secondNodeIndex >= fractureNodes.length
		)
	) {
		throw new Error('The LiquidFun mapping no longer matches the Gate 4 scene nodes and bonds.');
	}

	return {
		seed: LIQUIDFUN_SCENE_SEED,
		sourceMpmNodeCount: scene.particles.length,
		mappedMpmNodeCount,
		waterParticles: expandLiquidParticles(
			sourceWaterParticles,
			GATE4_WATER_PARTICLE_COLUMNS,
			GATE4_WATER_PARTICLE_ROWS,
			LIQUIDFUN_WATER_PARTICLE_COUNT,
			LIQUIDFUN_WATER_GRID.rows,
			LIQUIDFUN_FLUID_VERTICAL_OFFSET
		),
		oilParticles: expandLiquidParticles(
			sourceOilParticles,
			GATE4_OIL_PARTICLE_COLUMNS,
			GATE4_OIL_PARTICLE_ROWS,
			LIQUIDFUN_OIL_PARTICLE_COUNT,
			LIQUIDFUN_OIL_GRID.rows,
			LIQUIDFUN_FLUID_VERTICAL_OFFSET
		),
		pebbleNodes,
		pebbleGroups,
		fractureNodes,
		fractureBonds
	};
}

export interface NumericDistribution {
	readonly median: number;
	readonly p95: number;
	readonly min: number;
	readonly max: number;
	readonly samples: number;
}

export function summarizeSamples(values: readonly number[]): NumericDistribution {
	if (
		values.length === 0 ||
		values.some((value) => !Number.isFinite(value) || value < 0)
	) {
		throw new RangeError('At least one finite, non-negative sample is required.');
	}
	const sorted = [...values].sort((first, second) => first - second);
	const quantile = (fraction: number) =>
		sorted[Math.floor((sorted.length - 1) * fraction)];
	return {
		median: quantile(0.5),
		p95: quantile(0.95),
		min: sorted[0],
		max: sorted[sorted.length - 1],
		samples: sorted.length
	};
}
