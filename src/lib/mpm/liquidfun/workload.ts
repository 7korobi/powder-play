import type { MpmParticle, MpmSolidBond, MpmVector2 } from '../reference/solver.js';
import {
	createMpmGate4Scene,
	GATE4_FRACTURE_BOND_COUNT,
	GATE4_FRACTURE_FAILURE_STRAIN,
	GATE4_FRACTURE_OBJECT_ID,
	GATE4_FRACTURE_PARTICLE_COUNT,
	GATE4_GRID,
	GATE4_OIL_PARTICLE_COLUMNS,
	GATE4_OIL_PARTICLE_COUNT,
	GATE4_OIL_PARTICLE_ROWS,
	GATE4_PARTICLE_AREA,
	GATE4_PARTICLE_COUNT,
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
export const LIQUIDFUN_VISIBLE_DOMAIN_WIDTH_METERS = 0.14;
export const LIQUIDFUN_GATE4_DOMAIN = {
	width:
		LIQUIDFUN_VISIBLE_DOMAIN_WIDTH_METERS * LIQUIDFUN_GATE4_UNITS_PER_METER,
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

export const LIQUIDFUN_SHARED_PARTICLE_MASS = 0.01;
export const LIQUIDFUN_PRESSURE_STRENGTH = 1.0;
export const LIQUIDFUN_ELASTIC_STRENGTH = 0.25;
export const LIQUIDFUN_SPRING_STRENGTH = 0.25;
export const LIQUIDFUN_PARTICLE_RADIUS_METERS = 1e-3;
export const LIQUIDFUN_PARTICLE_RADIUS_BOX2D_UNITS =
	LIQUIDFUN_PARTICLE_RADIUS_METERS * LIQUIDFUN_BOX2D_UNITS_PER_METER;
const LIQUIDFUN_PARTICLE_STRIDE_BOX2D_UNITS =
	LIQUIDFUN_PARTICLE_RADIUS_BOX2D_UNITS * 2 * 0.75;
export const LIQUIDFUN_PARTICLE_SPACING_METERS =
	LIQUIDFUN_PARTICLE_STRIDE_BOX2D_UNITS / LIQUIDFUN_BOX2D_UNITS_PER_METER;
export const LIQUIDFUN_PARTICLE_SPACING_GATE4_UNITS =
	LIQUIDFUN_PARTICLE_SPACING_METERS * LIQUIDFUN_GATE4_UNITS_PER_METER;
export const LIQUIDFUN_SHARED_PARTICLE_DENSITY =
	LIQUIDFUN_SHARED_PARTICLE_MASS / LIQUIDFUN_PARTICLE_STRIDE_BOX2D_UNITS ** 2;

export const LIQUIDFUN_FLUID_TYPE_PARTICLE_COUNT = 800;
export const LIQUIDFUN_WATER_PARTICLE_COUNT = LIQUIDFUN_FLUID_TYPE_PARTICLE_COUNT;
export const LIQUIDFUN_OIL_PARTICLE_COUNT = LIQUIDFUN_FLUID_TYPE_PARTICLE_COUNT;
export const LIQUIDFUN_INK_PARTICLE_COUNT = LIQUIDFUN_FLUID_TYPE_PARTICLE_COUNT;
export const LIQUIDFUN_SOIL_PARTICLE_COUNT = LIQUIDFUN_FLUID_TYPE_PARTICLE_COUNT;
export const LIQUIDFUN_FLUID_PARTICLE_COUNT =
	LIQUIDFUN_WATER_PARTICLE_COUNT +
	LIQUIDFUN_OIL_PARTICLE_COUNT +
	LIQUIDFUN_INK_PARTICLE_COUNT +
	LIQUIDFUN_SOIL_PARTICLE_COUNT;
export const LIQUIDFUN_FLUID_PARTICLE_ROWS = 43;
export const LIQUIDFUN_FLUID_COLUMN_START_X_METERS = 0.0015;
export const LIQUIDFUN_FLUID_COLUMN_SPACING_METERS = 0.029;
export const LIQUIDFUN_FLUID_START_Y_METERS = 0.01;

export const LIQUIDFUN_ELASTIC_BALL_COUNT = 10;
export const LIQUIDFUN_ELASTIC_BALL_PARTICLES_PER_GROUP = 12;
export const LIQUIDFUN_ELASTIC_BALL_PARTICLE_COUNT =
	LIQUIDFUN_ELASTIC_BALL_COUNT * LIQUIDFUN_ELASTIC_BALL_PARTICLES_PER_GROUP;
// The wider middle gap leaves room for the retained fracture lattice.
export const LIQUIDFUN_ELASTIC_BALL_CENTER_X_METERS = [
	0.005, 0.017, 0.029, 0.041, 0.053, 0.066, 0.078, 0.09, 0.102, 0.114
] as const;
export const LIQUIDFUN_ELASTIC_BALL_CENTER_Y_METERS = 0.00425;
const LIQUIDFUN_ELASTIC_BALL_GRID_COORDINATES = [
	{ column: 1, row: 0 },
	{ column: 2, row: 0 },
	{ column: 0, row: 1 },
	{ column: 1, row: 1 },
	{ column: 2, row: 1 },
	{ column: 3, row: 1 },
	{ column: 0, row: 2 },
	{ column: 1, row: 2 },
	{ column: 2, row: 2 },
	{ column: 3, row: 2 },
	{ column: 1, row: 3 },
	{ column: 2, row: 3 }
] as const;

export const LIQUIDFUN_SPRING_VINE_COUNT = 10;
export const LIQUIDFUN_SPRING_VINE_LENGTH_METERS = 0.07;
export const LIQUIDFUN_SPRING_VINE_PARTICLES_PER_GROUP =
	Math.ceil(
		LIQUIDFUN_SPRING_VINE_LENGTH_METERS / LIQUIDFUN_PARTICLE_SPACING_METERS
	) + 1;
export const LIQUIDFUN_SPRING_VINE_PARTICLE_COUNT =
	LIQUIDFUN_SPRING_VINE_COUNT *
	LIQUIDFUN_SPRING_VINE_PARTICLES_PER_GROUP;
export const LIQUIDFUN_SPRING_VINE_LINK_COUNT =
	LIQUIDFUN_SPRING_VINE_COUNT *
	(LIQUIDFUN_SPRING_VINE_PARTICLES_PER_GROUP - 1);
export const LIQUIDFUN_SPRING_VINE_SPAN_METERS =
	(LIQUIDFUN_SPRING_VINE_PARTICLES_PER_GROUP - 1) *
	LIQUIDFUN_PARTICLE_SPACING_METERS;
export const LIQUIDFUN_SPRING_VINE_CENTER_X_START_METERS = 0.1185;
export const LIQUIDFUN_SPRING_VINE_CENTER_X_SPACING_METERS = 0.0022;
export const LIQUIDFUN_SPRING_VINE_CENTER_Y_METERS =
	LIQUIDFUN_PHYSICAL_DOMAIN_METERS.height / 2;
export const LIQUIDFUN_TOTAL_PARTICLE_COUNT =
	LIQUIDFUN_FLUID_PARTICLE_COUNT +
	LIQUIDFUN_ELASTIC_BALL_PARTICLE_COUNT +
	LIQUIDFUN_SPRING_VINE_PARTICLE_COUNT;

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
export const LIQUIDFUN_FRACTURE_CENTER_METERS = {
	x: 0.0595,
	y: LIQUIDFUN_ELASTIC_BALL_CENTER_Y_METERS
} as const;
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
export const LIQUIDFUN_STATIONARY_WARMUP_MS = 600;
export const LIQUIDFUN_GRAVITY_WARMUP_STEPS = 64;
export const LIQUIDFUN_SAMPLE_BATCHES = 12;
export const LIQUIDFUN_STEPS_PER_BATCH = 8;

export interface LiquidFunVector {
	readonly x: number;
	readonly y: number;
}

export interface LiquidFunFluidParticleSeed {
	readonly position: MpmVector2;
	readonly velocity: MpmVector2;
}

export interface LiquidFunParticleGroupSeed {
	readonly groupIndex: number;
	readonly positions: readonly MpmVector2[];
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
	readonly waterParticles: readonly LiquidFunFluidParticleSeed[];
	readonly oilParticles: readonly LiquidFunFluidParticleSeed[];
	readonly inkParticles: readonly LiquidFunFluidParticleSeed[];
	readonly soilParticles: readonly LiquidFunFluidParticleSeed[];
	readonly elasticBallGroups: readonly LiquidFunParticleGroupSeed[];
	readonly springVines: readonly LiquidFunParticleGroupSeed[];
	readonly fractureNodes: readonly MpmParticle[];
	readonly fractureBonds: readonly LiquidFunFractureBondSeed[];
}

function metersToGate4Units(meters: number): number {
	return meters * LIQUIDFUN_GATE4_UNITS_PER_METER;
}

function expandLiquidParticles(
	sourceParticles: readonly MpmParticle[],
	sourceColumns: number,
	sourceRows: number,
	targetParticleCount: number,
	targetRows: number,
	targetStartX: number,
	targetStartY: number
): readonly LiquidFunFluidParticleSeed[] {
	if (sourceParticles.length !== sourceColumns * sourceRows) {
		throw new Error('LiquidFun source particle count does not match its grid dimensions.');
	}
	if (
		!Number.isSafeInteger(targetParticleCount) ||
		targetParticleCount < targetRows ||
		!Number.isSafeInteger(targetRows) ||
		targetRows < 2 ||
		!Number.isFinite(targetStartX) ||
		!Number.isFinite(targetStartY)
	) {
		throw new RangeError('LiquidFun fluid grids must have finite origins and at least two rows.');
	}
	if (sourceParticles[0] === undefined) {
		throw new Error('LiquidFun source particle grid is incomplete.');
	}

	const targetColumns = Math.ceil(targetParticleCount / targetRows);
	const expandedParticles: LiquidFunFluidParticleSeed[] = [];
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
				position: {
					x:
						targetStartX +
						columnOffset +
						column * LIQUIDFUN_PARTICLE_SPACING_GATE4_UNITS,
					y: targetStartY + row * LIQUIDFUN_PARTICLE_SPACING_GATE4_UNITS
				},
				velocity: sourceParticle.velocity
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

function createElasticBallGroups(): readonly LiquidFunParticleGroupSeed[] {
	if (
		LIQUIDFUN_ELASTIC_BALL_CENTER_X_METERS.length !==
		LIQUIDFUN_ELASTIC_BALL_COUNT
	) {
		throw new Error('LiquidFun elastic ball positions do not match the configured count.');
	}

	return LIQUIDFUN_ELASTIC_BALL_CENTER_X_METERS.map((centerX, groupIndex) => {
		const centerXGate4 = metersToGate4Units(centerX);
		const centerYGate4 = metersToGate4Units(
			LIQUIDFUN_ELASTIC_BALL_CENTER_Y_METERS
		);
		return {
			groupIndex,
			positions: LIQUIDFUN_ELASTIC_BALL_GRID_COORDINATES.map(
				({ column, row }) => ({
					x:
						centerXGate4 +
						(column - 1.5) * LIQUIDFUN_PARTICLE_SPACING_GATE4_UNITS,
					y:
						centerYGate4 +
						(row - 1.5) * LIQUIDFUN_PARTICLE_SPACING_GATE4_UNITS
				})
			)
		};
	});
}

function createSpringVines(): readonly LiquidFunParticleGroupSeed[] {
	return Array.from({ length: LIQUIDFUN_SPRING_VINE_COUNT }, (_, groupIndex) => {
		const x = metersToGate4Units(
			LIQUIDFUN_SPRING_VINE_CENTER_X_START_METERS +
				groupIndex * LIQUIDFUN_SPRING_VINE_CENTER_X_SPACING_METERS
		);
		const centerY = metersToGate4Units(
			LIQUIDFUN_SPRING_VINE_CENTER_Y_METERS
		);
		return {
			groupIndex,
			positions: Array.from(
				{ length: LIQUIDFUN_SPRING_VINE_PARTICLES_PER_GROUP },
				(_, particleIndex) => ({
					x,
					y:
						centerY +
						(particleIndex -
							(LIQUIDFUN_SPRING_VINE_PARTICLES_PER_GROUP - 1) / 2) *
							LIQUIDFUN_PARTICLE_SPACING_GATE4_UNITS
				})
			)
		};
	});
}

function moveFractureNodesToLayout(
	nodes: readonly MpmParticle[]
): readonly MpmParticle[] {
	if (nodes.length === 0) {
		throw new Error('The Gate 4 fracture node source is empty.');
	}
	const sourceCenter = nodes.reduce(
		(center, node) => ({
			x: center.x + node.position.x / nodes.length,
			y: center.y + node.position.y / nodes.length
		}),
		{ x: 0, y: 0 }
	);
	const targetCenter = {
		x: metersToGate4Units(LIQUIDFUN_FRACTURE_CENTER_METERS.x),
		y: metersToGate4Units(LIQUIDFUN_FRACTURE_CENTER_METERS.y)
	};
	const offset = {
		x: targetCenter.x - sourceCenter.x,
		y: targetCenter.y - sourceCenter.y
	};
	return nodes.map((node) => ({
		...node,
		position: {
			x: node.position.x + offset.x,
			y: node.position.y + offset.y
		}
	}));
}

export function createLiquidFunGate4Workload(): LiquidFunGate4Workload {
	const scene = createMpmGate4Scene(LIQUIDFUN_SCENE_SEED);
	const sourceWaterParticles: MpmParticle[] = [];
	const sourceOilParticles: MpmParticle[] = [];
	const fractureNodesBySourceIndex = new Map<number, MpmParticle>();

	for (let sourceIndex = 0; sourceIndex < scene.particles.length; sourceIndex += 1) {
		const particle = scene.particles[sourceIndex];
		const material = particle.material?.model;
		if (material === 'water') {
			sourceWaterParticles.push(particle);
		} else if (material === 'oil') {
			sourceOilParticles.push(particle);
		} else if (particle.objectId === GATE4_FRACTURE_OBJECT_ID) {
			fractureNodesBySourceIndex.set(sourceIndex, particle);
		}
	}

	const sourceFractureNodes = [...fractureNodesBySourceIndex.values()];
	const fractureNodes = moveFractureNodesToLayout(sourceFractureNodes);
	const fractureNodeIndexBySourceIndex = new Map(
		[...fractureNodesBySourceIndex.keys()].map((sourceIndex, nodeIndex) => [
			sourceIndex,
			nodeIndex
		])
	);
	const fractureBonds: LiquidFunFractureBondSeed[] = scene.fractureBonds.map(
		(bond: MpmSolidBond) => {
			const firstNodeIndex = fractureNodeIndexBySourceIndex.get(
				bond.firstParticle
			);
			const secondNodeIndex = fractureNodeIndexBySourceIndex.get(
				bond.secondParticle
			);
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

	const targetStartY = metersToGate4Units(LIQUIDFUN_FLUID_START_Y_METERS);
	const fluidStartX = (column: number) =>
		metersToGate4Units(
			LIQUIDFUN_FLUID_COLUMN_START_X_METERS +
				column * LIQUIDFUN_FLUID_COLUMN_SPACING_METERS
		);
	const fluidParticles = (
		sourceParticles: readonly MpmParticle[],
		sourceColumns: number,
		sourceRows: number,
		targetStartX: number
	) =>
		expandLiquidParticles(
			sourceParticles,
			sourceColumns,
			sourceRows,
			LIQUIDFUN_FLUID_TYPE_PARTICLE_COUNT,
			LIQUIDFUN_FLUID_PARTICLE_ROWS,
			targetStartX,
			targetStartY
		);
	const [waterParticles, oilParticles, inkParticles, soilParticles] = [
		fluidParticles(
			sourceWaterParticles,
			GATE4_WATER_PARTICLE_COLUMNS,
			GATE4_WATER_PARTICLE_ROWS,
			fluidStartX(0)
		),
		fluidParticles(
			sourceOilParticles,
			GATE4_OIL_PARTICLE_COLUMNS,
			GATE4_OIL_PARTICLE_ROWS,
			fluidStartX(1)
		),
		fluidParticles(
			sourceWaterParticles,
			GATE4_WATER_PARTICLE_COLUMNS,
			GATE4_WATER_PARTICLE_ROWS,
			fluidStartX(2)
		),
		fluidParticles(
			sourceOilParticles,
			GATE4_OIL_PARTICLE_COLUMNS,
			GATE4_OIL_PARTICLE_ROWS,
			fluidStartX(3)
		)
	];
	if (!waterParticles || !oilParticles || !inkParticles || !soilParticles) {
		throw new Error('LiquidFun fluid source grids could not be created.');
	}

	const elasticBallGroups = createElasticBallGroups();
	const springVines = createSpringVines();
	const finitePosition = (position: MpmVector2) =>
		Number.isFinite(position.x) && Number.isFinite(position.y);
	if (
		scene.particles.length !== GATE4_PARTICLE_COUNT ||
		sourceWaterParticles.length !== GATE4_WATER_PARTICLE_COUNT ||
		sourceOilParticles.length !== GATE4_OIL_PARTICLE_COUNT ||
		fractureNodes.length !== GATE4_FRACTURE_PARTICLE_COUNT ||
		fractureBonds.length !== GATE4_FRACTURE_BOND_COUNT ||
		[
			waterParticles,
			oilParticles,
			inkParticles,
			soilParticles
		].some(
			(particles) =>
				particles.length !== LIQUIDFUN_FLUID_TYPE_PARTICLE_COUNT ||
				particles.some(
					(particle) =>
						!finitePosition(particle.position) ||
						!finitePosition(particle.velocity)
				)
		) ||
		elasticBallGroups.length !== LIQUIDFUN_ELASTIC_BALL_COUNT ||
		elasticBallGroups.some(
			(group) =>
				group.positions.length !== LIQUIDFUN_ELASTIC_BALL_PARTICLES_PER_GROUP ||
				group.positions.some((position) => !finitePosition(position))
		) ||
		springVines.length !== LIQUIDFUN_SPRING_VINE_COUNT ||
		springVines.some(
			(group) =>
				group.positions.length !== LIQUIDFUN_SPRING_VINE_PARTICLES_PER_GROUP ||
				group.positions.some((position) => !finitePosition(position))
		) ||
		fractureNodes.some(
			(node) =>
				!finitePosition(node.position) || !finitePosition(node.velocity)
		) ||
		fractureBonds.some(
			(bond) =>
				bond.failureStrain !== GATE4_FRACTURE_FAILURE_STRAIN ||
				bond.firstNodeIndex < 0 ||
				bond.firstNodeIndex >= fractureNodes.length ||
				bond.secondNodeIndex < 0 ||
				bond.secondNodeIndex >= fractureNodes.length ||
				!Number.isFinite(bond.restLengthBox2DUnits) ||
				bond.restLengthBox2DUnits <= 0
		)
	) {
		throw new Error('The LiquidFun workload does not match its configured particle layout.');
	}

	return {
		seed: LIQUIDFUN_SCENE_SEED,
		sourceMpmNodeCount: scene.particles.length,
		waterParticles,
		oilParticles,
		inkParticles,
		soilParticles,
		elasticBallGroups,
		springVines,
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
