import type { MpmParticle, MpmVector2 } from '../reference/solver.js';
import {
	createMpmGate4Scene,
	GATE4_FRACTURE_PARTICLE_COUNT,
	GATE4_FRACTURE_OBJECT_ID,
	GATE4_GRID,
	GATE4_LIQUID_PARTICLE_SPACING,
	GATE4_OIL_PARTICLE_COUNT,
	GATE4_PARTICLE_COUNT,
	GATE4_PEBBLE_COUNT,
	GATE4_PEBBLE_PARTICLE_COUNT,
	GATE4_SOLID_PARTICLE_MASS,
	GATE4_WATER_PARTICLE_COUNT
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
export const LIQUIDFUN_PARTICLE_RADIUS_METERS =
	GATE4_LIQUID_PARTICLE_SPACING / LIQUIDFUN_GATE4_UNITS_PER_METER / 2;
export const LIQUIDFUN_VELOCITY_ITERATIONS = 8;
export const LIQUIDFUN_POSITION_ITERATIONS = 3;
export const LIQUIDFUN_STATIONARY_WARMUP_MS = 600;
export const LIQUIDFUN_GRAVITY_WARMUP_STEPS = 64;
export const LIQUIDFUN_SAMPLE_BATCHES = 12;
export const LIQUIDFUN_STEPS_PER_BATCH = 8;

export const LIQUIDFUN_GATE4_DOMAIN = {
	width: GATE4_GRID.cellsX * GATE4_GRID.cellSize,
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

export interface LiquidFunTriangleBodySeed {
	readonly objectId: number;
	readonly position: LiquidFunVector;
	readonly linearVelocity: LiquidFunVector;
	readonly angularVelocity: number;
	readonly localVertices: readonly LiquidFunVector[];
	readonly areaSquareMeters: number;
	readonly density: number;
}

export interface LiquidFunGate4Workload {
	readonly seed: number;
	readonly sourceMpmNodeCount: number;
	readonly waterParticles: readonly MpmParticle[];
	readonly oilParticles: readonly MpmParticle[];
	readonly triangleBodies: readonly LiquidFunTriangleBodySeed[];
	readonly omittedFractureNodeCount: number;
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

export function mapGate4PebbleTriangle(
	nodes: readonly MpmParticle[]
): LiquidFunTriangleBodySeed {
	if (nodes.length !== 3) {
		throw new RangeError('A Gate 4 pebble must contain exactly three MPM nodes.');
	}

	const orderedNodes = [...nodes].sort(
		(first, second) =>
			(first.bondedTriangleVertex ?? -1) - (second.bondedTriangleVertex ?? -1)
	);
	const firstObjectId = orderedNodes[0].objectId;
	if (
		firstObjectId === undefined ||
		orderedNodes.some(
			(node, vertex) =>
				node.objectId !== firstObjectId || node.bondedTriangleVertex !== vertex
		)
	) {
		throw new RangeError('Gate 4 pebble nodes must identify vertices 0, 1, and 2.');
	}

	const positions = orderedNodes.map(({ position }) =>
		mapGate4PositionToBox2D(position)
	);
	const velocities = orderedNodes.map(({ velocity }) =>
		mapGate4VelocityToBox2D(velocity)
	);
	const position = positions.reduce(
		(center, point) => ({
			x: center.x + point.x / positions.length,
			y: center.y + point.y / positions.length
		}),
		{ x: 0, y: 0 }
	);
	const linearVelocity = velocities.reduce(
		(center, velocity) => ({
			x: center.x + velocity.x / velocities.length,
			y: center.y + velocity.y / velocities.length
		}),
		{ x: 0, y: 0 }
	);
	const localVertices = positions.map((point) => ({
		x: point.x - position.x,
		y: point.y - position.y
	}));
	let angularMomentumNumerator = 0;
	let momentOfInertiaPerMass = 0;
	for (let index = 0; index < localVertices.length; index += 1) {
		const relativePosition = localVertices[index];
		const relativeVelocity = {
			x: velocities[index].x - linearVelocity.x,
			y: velocities[index].y - linearVelocity.y
		};
		angularMomentumNumerator +=
			relativePosition.x * relativeVelocity.y -
			relativePosition.y * relativeVelocity.x;
		momentOfInertiaPerMass +=
			relativePosition.x ** 2 + relativePosition.y ** 2;
	}
	if (momentOfInertiaPerMass <= 0) {
		throw new RangeError('Gate 4 pebble nodes must have nonzero area.');
	}

	let doubledArea = 0;
	for (let index = 0; index < localVertices.length; index += 1) {
		const first = localVertices[index];
		const second = localVertices[(index + 1) % localVertices.length];
		doubledArea += first.x * second.y - first.y * second.x;
	}
	const areaSquareMeters = Math.abs(doubledArea) / 2;
	if (!Number.isFinite(areaSquareMeters) || areaSquareMeters <= 0) {
		throw new RangeError('Gate 4 pebble triangle area must be positive and finite.');
	}

	const mappedNodeMass = GATE4_SOLID_PARTICLE_MASS * orderedNodes.length;
	return {
		objectId: firstObjectId,
		position,
		linearVelocity,
		angularVelocity: angularMomentumNumerator / momentOfInertiaPerMass,
		localVertices,
		areaSquareMeters,
		density: mappedNodeMass / areaSquareMeters
	};
}

export function createLiquidFunGate4Workload(): LiquidFunGate4Workload {
	const scene = createMpmGate4Scene(LIQUIDFUN_SCENE_SEED);
	const waterParticles: MpmParticle[] = [];
	const oilParticles: MpmParticle[] = [];
	const fractureNodes: MpmParticle[] = [];
	const pebbleNodesByObject = new Map<number, MpmParticle[]>();

	for (const particle of scene.particles) {
		const material = particle.material?.model;
		if (material === 'water') {
			waterParticles.push(particle);
		} else if (material === 'oil') {
			oilParticles.push(particle);
		} else if (particle.bondedTriangleVertex !== undefined) {
			if (particle.objectId === undefined) {
				throw new Error('Every Gate 4 pebble node must have an object id.');
			}
			const nodes = pebbleNodesByObject.get(particle.objectId) ?? [];
			nodes.push(particle);
			pebbleNodesByObject.set(particle.objectId, nodes);
		} else if (particle.objectId === GATE4_FRACTURE_OBJECT_ID) {
			fractureNodes.push(particle);
		}
	}

	const triangleBodies = [...pebbleNodesByObject.entries()]
		.sort(([first], [second]) => first - second)
		.map(([, nodes]) => mapGate4PebbleTriangle(nodes));

	if (
		scene.particles.length !== GATE4_PARTICLE_COUNT ||
		waterParticles.length !== GATE4_WATER_PARTICLE_COUNT ||
		oilParticles.length !== GATE4_OIL_PARTICLE_COUNT ||
		triangleBodies.length !== GATE4_PEBBLE_COUNT ||
		triangleBodies.reduce((count, body) => count + body.localVertices.length, 0) !==
			GATE4_PEBBLE_PARTICLE_COUNT ||
		fractureNodes.length !== GATE4_FRACTURE_PARTICLE_COUNT
	) {
		throw new Error('The LiquidFun mapping no longer matches the Gate 4 scene counts.');
	}

	return {
		seed: LIQUIDFUN_SCENE_SEED,
		sourceMpmNodeCount: scene.particles.length,
		waterParticles,
		oilParticles,
		triangleBodies,
		omittedFractureNodeCount: fractureNodes.length
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
