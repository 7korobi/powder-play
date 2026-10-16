import type { MpmParticle, MpmSolidBond, MpmVector2 } from '../reference/solver.js';
import {
	createMpmGate4Scene,
	GATE4_FRACTURE_BOND_COUNT,
	GATE4_FRACTURE_FAILURE_STRAIN,
	GATE4_FRACTURE_PARTICLE_COUNT,
	GATE4_FRACTURE_OBJECT_ID,
	GATE4_GRID,
	GATE4_LIQUID_START_X,
	GATE4_OIL_PARTICLE_COLUMNS,
	GATE4_OIL_PARTICLE_COUNT,
	GATE4_OIL_PARTICLE_ROWS,
	GATE4_PARTICLE_AREA,
	GATE4_PARTICLE_COUNT,
	GATE4_PEBBLE_COUNT,
	GATE4_PEBBLE_EDGE_LENGTH,
	GATE4_PEBBLE_PARTICLE_COUNT,
	GATE4_SOLID_PARTICLE_MASS,
	GATE4_WATER_PARTICLE_COLUMNS,
	GATE4_WATER_PARTICLE_COUNT,
	GATE4_WATER_PARTICLE_ROWS
} from '../scenes/gate4.js';

export const LIQUIDFUN_PACKAGE_VERSION = '7.0.0';
export const LIQUIDFUN_SCENE_SEED = 0x4d504d34;
export const LIQUIDFUN_GATE4_UNITS_PER_METER = 20_000;
export const LIQUIDFUN_BOX2D_UNITS_PER_METER = 1_000;
export const LIQUIDFUN_BOX2D_UNITS_PER_GATE4_UNIT =
	LIQUIDFUN_BOX2D_UNITS_PER_METER / LIQUIDFUN_GATE4_UNITS_PER_METER;
export const LIQUIDFUN_GATE4_ACCELERATION_UNITS_PER_METER_PER_SECOND_SQUARED = 100;
export const LIQUIDFUN_DT_SECONDS = 1 / 240;
export const LIQUIDFUN_GATE4_GRAVITY: MpmVector2 = { x: 0, y: 90 };
export const LIQUIDFUN_BOX2D_GRAVITY_UNITS_PER_SECOND_SQUARED: MpmVector2 = {
	x:
		(LIQUIDFUN_GATE4_GRAVITY.x * LIQUIDFUN_BOX2D_UNITS_PER_METER) /
		LIQUIDFUN_GATE4_ACCELERATION_UNITS_PER_METER_PER_SECOND_SQUARED,
	y:
		(-LIQUIDFUN_GATE4_GRAVITY.y * LIQUIDFUN_BOX2D_UNITS_PER_METER) /
		LIQUIDFUN_GATE4_ACCELERATION_UNITS_PER_METER_PER_SECOND_SQUARED
};
export const LIQUIDFUN_SHARED_PARTICLE_MASS = 0.01;
export const LIQUIDFUN_PRESSURE_STRENGTH = 1.0;
export const LIQUIDFUN_PARTICLE_RADIUS_METERS = 1e-3;
export const LIQUIDFUN_PARTICLE_RADIUS_BOX2D_UNITS =
	LIQUIDFUN_PARTICLE_RADIUS_METERS * LIQUIDFUN_BOX2D_UNITS_PER_METER;
const LIQUIDFUN_PARTICLE_STRIDE_BOX2D_UNITS =
	LIQUIDFUN_PARTICLE_RADIUS_BOX2D_UNITS * 2 * 0.75;
export const LIQUIDFUN_PARTICLE_SPACING_METERS =
	LIQUIDFUN_PARTICLE_STRIDE_BOX2D_UNITS / LIQUIDFUN_BOX2D_UNITS_PER_METER;
export const LIQUIDFUN_PARTICLE_SPACING_GATE4_UNITS =
	LIQUIDFUN_PARTICLE_SPACING_METERS * LIQUIDFUN_GATE4_UNITS_PER_METER;
// LiquidFun estimates particle mass as density times the squared 0.75-diameter stride.
export const LIQUIDFUN_SHARED_PARTICLE_DENSITY =
	LIQUIDFUN_SHARED_PARTICLE_MASS / LIQUIDFUN_PARTICLE_STRIDE_BOX2D_UNITS ** 2;
export const LIQUIDFUN_WATER_PARTICLE_COUNT_MULTIPLIER = 1;
export const LIQUIDFUN_OIL_PARTICLE_COUNT_MULTIPLIER = 1;
export const LIQUIDFUN_WATER_PARTICLE_COUNT =
	GATE4_WATER_PARTICLE_COUNT * LIQUIDFUN_WATER_PARTICLE_COUNT_MULTIPLIER;
export const LIQUIDFUN_OIL_PARTICLE_COUNT =
	GATE4_OIL_PARTICLE_COUNT * LIQUIDFUN_OIL_PARTICLE_COUNT_MULTIPLIER;
export const LIQUIDFUN_WATER_PARTICLE_ROWS = 23;
export const LIQUIDFUN_OIL_PARTICLE_ROWS = 20;
export const LIQUIDFUN_SOLID_NODE_AREA_SQUARE_METERS =
	GATE4_PARTICLE_AREA / LIQUIDFUN_GATE4_UNITS_PER_METER ** 2;
export const LIQUIDFUN_SOLID_NODE_AREA_BOX2D_UNITS_SQUARED =
	LIQUIDFUN_SOLID_NODE_AREA_SQUARE_METERS * LIQUIDFUN_BOX2D_UNITS_PER_METER ** 2;
export const LIQUIDFUN_SOLID_NODE_RADIUS_METERS = Math.sqrt(
	LIQUIDFUN_SOLID_NODE_AREA_SQUARE_METERS / Math.PI
);
export const LIQUIDFUN_SOLID_NODE_RADIUS_BOX2D_UNITS =
	LIQUIDFUN_SOLID_NODE_RADIUS_METERS * LIQUIDFUN_BOX2D_UNITS_PER_METER;
export const LIQUIDFUN_SOLID_NODE_DENSITY =
	GATE4_SOLID_PARTICLE_MASS / LIQUIDFUN_SOLID_NODE_AREA_BOX2D_UNITS_SQUARED;
export const LIQUIDFUN_PEBBLE_EDGE_LENGTH_METERS = 3e-3;
export const LIQUIDFUN_PEBBLE_EDGE_LENGTH_BOX2D_UNITS =
	LIQUIDFUN_PEBBLE_EDGE_LENGTH_METERS * LIQUIDFUN_BOX2D_UNITS_PER_METER;
export const LIQUIDFUN_PEBBLE_MASS = 0.008;
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
export const LIQUIDFUN_PEBBLE_SIZE_SCALE =
	(LIQUIDFUN_PEBBLE_EDGE_LENGTH_METERS * LIQUIDFUN_GATE4_UNITS_PER_METER) /
	GATE4_PEBBLE_EDGE_LENGTH;
export const LIQUIDFUN_STATIONARY_WARMUP_MS = 600;
export const LIQUIDFUN_GRAVITY_WARMUP_STEPS = 64;
export const LIQUIDFUN_SAMPLE_BATCHES = 12;
export const LIQUIDFUN_STEPS_PER_BATCH = 8;

const LIQUIDFUN_WATER_GRID = {
	columns: Math.ceil(LIQUIDFUN_WATER_PARTICLE_COUNT / LIQUIDFUN_WATER_PARTICLE_ROWS),
	rows: LIQUIDFUN_WATER_PARTICLE_ROWS
} as const;
const LIQUIDFUN_OIL_GRID = {
	rows: LIQUIDFUN_OIL_PARTICLE_ROWS,
	columns: Math.ceil(LIQUIDFUN_OIL_PARTICLE_COUNT / LIQUIDFUN_OIL_PARTICLE_ROWS)
} as const;
const LIQUIDFUN_OIL_START_Y =
	2 * LIQUIDFUN_PARTICLE_RADIUS_METERS * LIQUIDFUN_GATE4_UNITS_PER_METER;
const LIQUIDFUN_WATER_START_Y =
	LIQUIDFUN_OIL_START_Y +
	LIQUIDFUN_OIL_GRID.rows * LIQUIDFUN_PARTICLE_SPACING_GATE4_UNITS;
const LIQUIDFUN_FLUID_RIGHT_EDGE =
	GATE4_LIQUID_START_X +
	(Math.max(LIQUIDFUN_WATER_GRID.columns, LIQUIDFUN_OIL_GRID.columns) - 1) *
		LIQUIDFUN_PARTICLE_SPACING_GATE4_UNITS;
const LIQUIDFUN_DOMAIN_RIGHT_MARGIN = LIQUIDFUN_PARTICLE_SPACING_GATE4_UNITS;
const LIQUIDFUN_MINIMUM_DOMAIN_WIDTH_MILLIMETERS = 140;
const LIQUIDFUN_BASE_DOMAIN_WIDTH = Math.max(
	GATE4_GRID.cellsX * GATE4_GRID.cellSize,
	(LIQUIDFUN_MINIMUM_DOMAIN_WIDTH_MILLIMETERS * LIQUIDFUN_GATE4_UNITS_PER_METER) /
		1_000
);

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

export const LIQUIDFUN_PHYSICAL_DOMAIN_METERS = {
	width: LIQUIDFUN_GATE4_DOMAIN.width / LIQUIDFUN_GATE4_UNITS_PER_METER,
	height: LIQUIDFUN_GATE4_DOMAIN.height / LIQUIDFUN_GATE4_UNITS_PER_METER
} as const;
export const LIQUIDFUN_BOX2D_DOMAIN_UNITS = {
	width: LIQUIDFUN_GATE4_DOMAIN.width * LIQUIDFUN_BOX2D_UNITS_PER_GATE4_UNIT,
	height: LIQUIDFUN_GATE4_DOMAIN.height * LIQUIDFUN_BOX2D_UNITS_PER_GATE4_UNIT
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
	readonly restLengthBox2DUnits: number;
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
	targetStartY: number
): readonly MpmParticle[] {
	if (sourceParticles.length !== sourceColumns * sourceRows) {
		throw new Error('LiquidFun source particle count does not match its grid dimensions.');
	}
	if (
		!Number.isSafeInteger(targetParticleCount) ||
		targetParticleCount < targetRows ||
		!Number.isSafeInteger(targetRows) ||
		targetRows < 2 ||
		!Number.isFinite(targetStartY)
	) {
		throw new RangeError('LiquidFun fluid grids must have finite origins and at least two rows.');
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
			((targetColumns - particlesInRow) * LIQUIDFUN_PARTICLE_SPACING_GATE4_UNITS) / 2;
		const sourceRow = Math.floor((row * sourceRows) / targetRows);
		for (let column = 0; column < particlesInRow; column += 1) {
			const sourceParticle =
				sourceParticles[sourceRow * sourceColumns + (column % sourceColumns)];
			if (sourceParticle === undefined) {
				throw new Error('LiquidFun expansion could not find a source particle.');
			}
			expandedParticles.push({
				...sourceParticle,
				position: {
					x:
						firstSourceParticle.position.x +
						columnOffset +
						column * LIQUIDFUN_PARTICLE_SPACING_GATE4_UNITS,
					y: targetStartY + row * LIQUIDFUN_PARTICLE_SPACING_GATE4_UNITS
				}
			});
		}
	}
	return expandedParticles;
}

export function mapGate4PositionToBox2D(position: MpmVector2): LiquidFunVector {
	return {
		x: position.x * LIQUIDFUN_BOX2D_UNITS_PER_GATE4_UNIT,
		y:
			(LIQUIDFUN_GATE4_DOMAIN.height - position.y) *
			LIQUIDFUN_BOX2D_UNITS_PER_GATE4_UNIT
	};
}

export function mapGate4VelocityToBox2D(velocity: MpmVector2): LiquidFunVector {
	return {
		x: velocity.x * LIQUIDFUN_BOX2D_UNITS_PER_GATE4_UNIT,
		y: -velocity.y * LIQUIDFUN_BOX2D_UNITS_PER_GATE4_UNIT
	};
}

export function mapGate4AccelerationToBox2D(
	acceleration: MpmVector2
): LiquidFunVector {
	const box2dUnitsPerGate4Unit =
		LIQUIDFUN_BOX2D_UNITS_PER_METER /
		LIQUIDFUN_GATE4_ACCELERATION_UNITS_PER_METER_PER_SECOND_SQUARED;
	return {
		x: acceleration.x * box2dUnitsPerGate4Unit,
		y: -acceleration.y * box2dUnitsPerGate4Unit
	};
}

export function mapBox2DPositionToGate4(position: LiquidFunVector): MpmVector2 {
	return {
		x: position.x / LIQUIDFUN_BOX2D_UNITS_PER_GATE4_UNIT,
		y:
			LIQUIDFUN_GATE4_DOMAIN.height -
			position.y / LIQUIDFUN_BOX2D_UNITS_PER_GATE4_UNIT
	};
}

export function mapBox2DVelocityToGate4(velocity: LiquidFunVector): MpmVector2 {
	return {
		x: velocity.x / LIQUIDFUN_BOX2D_UNITS_PER_GATE4_UNIT,
		y: -velocity.y / LIQUIDFUN_BOX2D_UNITS_PER_GATE4_UNIT
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
				restLengthBox2DUnits:
					bond.restLength * LIQUIDFUN_BOX2D_UNITS_PER_GATE4_UNIT,
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
			LIQUIDFUN_WATER_START_Y
		),
		oilParticles: expandLiquidParticles(
			sourceOilParticles,
			GATE4_OIL_PARTICLE_COLUMNS,
			GATE4_OIL_PARTICLE_ROWS,
			LIQUIDFUN_OIL_PARTICLE_COUNT,
			LIQUIDFUN_OIL_GRID.rows,
			LIQUIDFUN_OIL_START_Y
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
