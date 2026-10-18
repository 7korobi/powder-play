import type {
	B2Body,
	B2BodyDef,
	B2CircleShape,
	B2DistanceJointDef,
	B2Joint,
	B2ParticleDef,
	B2ParticleGroup,
	B2ParticleGroupDef,
	B2ParticleSystem,
	B2ParticleSystemDef,
	B2PolygonShape,
	B2Vec2,
	B2World,
	LiquidFunModule
} from 'liquidfun-wasm';
import type { MpmParticle, MpmVector2 } from '../reference/solver.js';
import { GATE4_PEBBLE_PARTICLES_PER_OBJECT } from '../scenes/gate4.js';
import { readLiquidFunParticlePosition } from './particle-buffer.js';
import {
	createLiquidFunGate4Workload,
	LIQUIDFUN_BOX2D_DOMAIN_UNITS,
	LIQUIDFUN_BOX2D_GRAVITY_UNITS_PER_SECOND_SQUARED,
	LIQUIDFUN_BOX2D_UNITS_PER_METER,
	LIQUIDFUN_DT_SECONDS,
	LIQUIDFUN_GATE4_DOMAIN,
	LIQUIDFUN_GATE4_GRAVITY,
	LIQUIDFUN_GATE4_ACCELERATION_UNITS_PER_METER_PER_SECOND_SQUARED,
	LIQUIDFUN_GATE4_UNITS_PER_METER,
	LIQUIDFUN_ELASTIC_PARTICLE_COUNT,
	LIQUIDFUN_ELASTIC_STRENGTH,
	LIQUIDFUN_ELASTIC_TRIANGLE_COUNT,
	LIQUIDFUN_FRACTURE_BOND_DAMPING,
	LIQUIDFUN_FRACTURE_BOND_STIFFNESS,
	LIQUIDFUN_PARTICLE_RADIUS_METERS,
	LIQUIDFUN_PARTICLE_RADIUS_BOX2D_UNITS,
	LIQUIDFUN_PEBBLE_MASS,
	LIQUIDFUN_PEBBLE_TRIANGLE_COUNT,
	LIQUIDFUN_RIGID_TRIANGLE_COUNT,
	scaleLiquidFunPebbleLocalVertices,
	LIQUIDFUN_POSITION_ITERATIONS,
	LIQUIDFUN_SHARED_PARTICLE_DENSITY,
	LIQUIDFUN_SOFT_SQUARE_COUNT,
	LIQUIDFUN_SOFT_SQUARE_EDGE_LENGTH_METERS,
	LIQUIDFUN_SOFT_SQUARE_MASS,
	LIQUIDFUN_SOLID_NODE_DENSITY,
	LIQUIDFUN_SOLID_NODE_RADIUS_BOX2D_UNITS,
	LIQUIDFUN_SOFT_SQUARE_NODE_DENSITY,
	LIQUIDFUN_SOFT_SQUARE_NODE_RADIUS_BOX2D_UNITS,
	LIQUIDFUN_SOFT_SQUARE_SPRING_DAMPING,
	LIQUIDFUN_SOFT_SQUARE_SPRING_DAMPING_RATIO,
	LIQUIDFUN_SOFT_SQUARE_SPRING_FREQUENCY_HZ,
	LIQUIDFUN_SOFT_SQUARE_SPRING_STIFFNESS,
	LIQUIDFUN_SHARED_PARTICLE_MASS,
	LIQUIDFUN_PHYSICAL_DOMAIN_METERS,
	LIQUIDFUN_PRESSURE_STRENGTH,
	LIQUIDFUN_PEBBLE_EDGE_LENGTH_METERS,
	LIQUIDFUN_VELOCITY_ITERATIONS,
	mapGate4AccelerationToBox2D,
	mapGate4PositionToBox2D,
	mapGate4VelocityToBox2D,
	type LiquidFunBondSeed,
	type LiquidFunFractureBondSeed,
	type LiquidFunSoftSquareNodeSeed,
	type LiquidFunVector
} from './workload.js';
import { createLiquidFunWallDefinitions } from './wall-geometry.js';

const B2_STATIC_BODY = 0;
const B2_DYNAMIC_BODY = 2;

export type LiquidFunRenderMode = 'surface' | 'particles';

export interface LiquidFunSimulation {
	readonly particleCount: number;
	readonly waterParticleCount: number;
	readonly oilParticleCount: number;
	readonly mappedMpmNodeCount: number;
	readonly pebbleNodeCount: number;
	readonly triangleBodyCount: number;
	readonly elasticTriangleCount: number;
	readonly elasticParticleCount: number;
	readonly elasticTriadCount: number;
	readonly softSquareCount: number;
	readonly softSquareNodeCount: number;
	readonly softSquareJointCount: number;
	readonly pebbleVertexCount: number;
	readonly fractureNodeCount: number;
	readonly fractureBondCount: number;
	readonly activeFractureBondCount: number;
	readonly staticWallBodyCount: number;
	readonly sourceMpmNodeCount: number;
	readonly particleIterations: number;
	step(): void;
	setGravityFromGate4(gravity: MpmVector2): void;
	draw(
		context: CanvasRenderingContext2D,
		width: number,
		height: number,
		renderMode: LiquidFunRenderMode
	): void;
	dispose(): void;
}

interface PebbleTriangleBody {
	readonly body: B2Body;
	readonly localVertices: readonly LiquidFunVector[];
}

interface PebbleTriangleMotion {
	readonly center: LiquidFunVector;
	readonly linearVelocity: LiquidFunVector;
	readonly angularVelocity: number;
	readonly localVertices: readonly LiquidFunVector[];
	readonly areaBox2DUnitsSquared: number;
}

type SoftSquareBody = readonly [B2Body, B2Body, B2Body, B2Body];

interface FractureBondJoint extends LiquidFunFractureBondSeed {
	joint: B2Joint | null;
}

interface VectorWithAxisGetters {
	get_x(): number;
	get_y(): number;
}

function release(
	value:
		| { delete?: () => void; __destroy__?: () => void }
		| null
		| undefined
): void {
	if (typeof value?.delete === 'function') {
		value.delete();
	} else {
		value?.__destroy__?.();
	}
}

function releaseUnknown(value: unknown): void {
	if (
		typeof value === 'object' &&
		value !== null &&
		'delete' in value &&
		typeof value.delete === 'function'
	) {
		value.delete();
	} else if (
		typeof value === 'object' &&
		value !== null &&
		'__destroy__' in value &&
		typeof value.__destroy__ === 'function'
	) {
		value.__destroy__();
	}
}

function hasNumericCoordinates(
	value: unknown
): value is { readonly x: number; readonly y: number } {
	return (
		typeof value === 'object' &&
		value !== null &&
		'x' in value &&
		typeof value.x === 'number' &&
		'y' in value &&
		typeof value.y === 'number'
	);
}

function hasVectorAxisGetters(value: unknown): value is VectorWithAxisGetters {
	return (
		typeof value === 'object' &&
		value !== null &&
		'get_x' in value &&
		typeof value.get_x === 'function' &&
		'get_y' in value &&
		typeof value.get_y === 'function'
	);
}

function readVector(value: unknown): LiquidFunVector {
	if (
		Array.isArray(value) &&
		typeof value[0] === 'number' &&
		typeof value[1] === 'number'
	) {
		return { x: value[0], y: value[1] };
	}
	if (hasNumericCoordinates(value)) {
		return { x: value.x, y: value.y };
	}
	if (hasVectorAxisGetters(value)) {
		return { x: value.get_x(), y: value.get_y() };
	}
	throw new TypeError('LiquidFun returned an unsupported b2Vec2 representation.');
}

function createVector(module: LiquidFunModule, value: LiquidFunVector): B2Vec2 {
	return new module.b2Vec2(value.x, value.y);
}

function createBody(
	module: LiquidFunModule,
	world: B2World,
	type: number,
	position: LiquidFunVector,
	linearVelocity: LiquidFunVector,
	angularVelocity = 0
): B2Body {
	const definition: B2BodyDef = new module.b2BodyDef();
	const positionVector = createVector(module, position);
	const velocityVector = createVector(module, linearVelocity);
	try {
		definition.type = type;
		definition.position = positionVector;
		definition.angle = 0;
		definition.linearVelocity = velocityVector;
		definition.angularVelocity = angularVelocity;
		return world.CreateBody(definition);
	} finally {
		release(positionVector);
		release(velocityVector);
		release(definition);
	}
}

function addBoxFixture(
	module: LiquidFunModule,
	body: B2Body,
	halfWidth: number,
	halfHeight: number
): void {
	const shape = new module.b2PolygonShape();
	try {
		shape.SetAsBox(halfWidth, halfHeight);
		if (!body.CreateFixture(shape, 0)) {
			throw new Error('LiquidFun did not create a static wall fixture.');
		}
	} finally {
		release(shape);
	}
}

function addSolidNodeFixture(module: LiquidFunModule, body: B2Body): void {
	const shape: B2CircleShape = new module.b2CircleShape();
	try {
		shape.m_radius = LIQUIDFUN_SOLID_NODE_RADIUS_BOX2D_UNITS;
		if (!body.CreateFixture(shape, LIQUIDFUN_SOLID_NODE_DENSITY)) {
			throw new Error('LiquidFun did not create a Gate 4 solid-node fixture.');
		}
	} finally {
		release(shape);
	}
}

function createSolidNodeBody(
	module: LiquidFunModule,
	world: B2World,
	particle: MpmParticle
): B2Body {
	const body = createBody(
		module,
		world,
		B2_DYNAMIC_BODY,
		mapGate4PositionToBox2D(particle.position),
		mapGate4VelocityToBox2D(particle.velocity)
	);
	addSolidNodeFixture(module, body);
	return body;
}

function createSoftSquareNodeBody(
	module: LiquidFunModule,
	world: B2World,
	node: LiquidFunSoftSquareNodeSeed
): B2Body {
	const body = createBody(
		module,
		world,
		B2_DYNAMIC_BODY,
		mapGate4PositionToBox2D(node.position),
		mapGate4VelocityToBox2D(node.velocity)
	);
	const shape = new module.b2CircleShape();
	try {
		shape.m_radius = LIQUIDFUN_SOFT_SQUARE_NODE_RADIUS_BOX2D_UNITS;
		if (!body.CreateFixture(shape, LIQUIDFUN_SOFT_SQUARE_NODE_DENSITY)) {
			throw new Error('LiquidFun did not create a soft-square node fixture.');
		}
		return body;
	} finally {
		release(shape);
	}
}

function calculatePebbleTriangleMotion(
	nodes: readonly [MpmParticle, MpmParticle, MpmParticle]
): PebbleTriangleMotion {
	const positions = nodes.map((node) => mapGate4PositionToBox2D(node.position));
	const velocities = nodes.map((node) => mapGate4VelocityToBox2D(node.velocity));
	const positionSum = positions.reduce(average, { x: 0, y: 0 });
	const velocitySum = velocities.reduce(average, { x: 0, y: 0 });
	const center = {
		x: positionSum.x / positions.length,
		y: positionSum.y / positions.length
	};
	const linearVelocity = {
		x: velocitySum.x / velocities.length,
		y: velocitySum.y / velocities.length
	};

	let angularVelocityNumerator = 0;
	let angularVelocityDenominator = 0;
	const localVertices = scaleLiquidFunPebbleLocalVertices(
		positions.map((position) => ({
			x: position.x - center.x,
			y: position.y - center.y
		}))
	);
	for (let index = 0; index < localVertices.length; index += 1) {
		const relativePosition = localVertices[index];
		const velocity = velocities[index];
		if (!relativePosition || !velocity) {
			throw new RangeError('A Gate 4 pebble must identify three node velocities.');
		}
		const relativeVelocity = {
			x: velocity.x - linearVelocity.x,
			y: velocity.y - linearVelocity.y
		};
		angularVelocityNumerator +=
			relativePosition.x * relativeVelocity.y -
			relativePosition.y * relativeVelocity.x;
		angularVelocityDenominator +=
			relativePosition.x ** 2 + relativePosition.y ** 2;
	}
	if (angularVelocityDenominator <= Number.EPSILON) {
		throw new RangeError('A Gate 4 pebble triangle must have nonzero area.');
	}
	const angularVelocity = angularVelocityNumerator / angularVelocityDenominator;
	if (!Number.isFinite(angularVelocity)) {
		throw new RangeError('A Gate 4 pebble triangle must have finite angular velocity.');
	}

	let twiceArea = 0;
	for (let index = 0; index < localVertices.length; index += 1) {
		const current = localVertices[index];
		const next = localVertices[(index + 1) % localVertices.length];
		if (!current || !next) {
			throw new RangeError('A Gate 4 pebble must identify three polygon vertices.');
		}
		twiceArea += current.x * next.y - next.x * current.y;
	}
	const areaBox2DUnitsSquared = Math.abs(twiceArea) / 2;
	if (
		!Number.isFinite(areaBox2DUnitsSquared) ||
		areaBox2DUnitsSquared <= Number.EPSILON
	) {
		throw new RangeError('A Gate 4 pebble triangle must have finite, nonzero area.');
	}

	return {
		center,
		linearVelocity,
		angularVelocity,
		localVertices,
		areaBox2DUnitsSquared
	};
}

function createPebbleTriangleBody(
	module: LiquidFunModule,
	world: B2World,
	nodes: readonly [MpmParticle, MpmParticle, MpmParticle]
): PebbleTriangleBody {
	const {
		center,
		linearVelocity,
		angularVelocity,
		localVertices,
		areaBox2DUnitsSquared
	} = calculatePebbleTriangleMotion(nodes);
	const body = createBody(
		module,
		world,
		B2_DYNAMIC_BODY,
		center,
		linearVelocity,
		angularVelocity
	);
	let shape: B2PolygonShape | undefined;
	let destroyVertices: (() => void) | undefined;
	try {
		shape = new module.b2PolygonShape();
		const [vertices, destroy] = module.pointsToVec2Array(localVertices);
		destroyVertices = destroy;
		shape.Set(vertices, localVertices.length);
		const triangleMass = LIQUIDFUN_PEBBLE_MASS;
		if (!body.CreateFixture(shape, triangleMass / areaBox2DUnitsSquared)) {
			throw new Error('LiquidFun did not create a Gate 4 pebble polygon fixture.');
		}
		return { body, localVertices };
	} finally {
		release(shape);
		destroyVertices?.();
	}
}

function createElasticParticleTriangleGroup(
	module: LiquidFunModule,
	particleSystem: B2ParticleSystem,
	nodes: readonly [MpmParticle, MpmParticle, MpmParticle]
): B2ParticleGroup {
	const motion = calculatePebbleTriangleMotion(nodes);
	let particlePositions: B2Vec2 | undefined;
	let destroyParticlePositions: (() => void) | undefined;
	let position: B2Vec2 | undefined;
	let linearVelocity: B2Vec2 | undefined;
	let definition: B2ParticleGroupDef | undefined;
	let group: B2ParticleGroup | undefined;
	try {
		[particlePositions, destroyParticlePositions] = module.pointsToVec2Array(
			motion.localVertices
		);
		position = createVector(module, motion.center);
		linearVelocity = createVector(module, motion.linearVelocity);
		definition = new module.b2ParticleGroupDef();
		definition.flags = module.b2_elasticParticle;
		definition.position = position;
		definition.angle = 0;
		definition.linearVelocity = linearVelocity;
		definition.angularVelocity = motion.angularVelocity;
		definition.particleCount = motion.localVertices.length;
		definition.positionData = particlePositions;
		group = particleSystem.CreateParticleGroup(definition);
		if (
			group.GetParticleCount() !== GATE4_PEBBLE_PARTICLES_PER_OBJECT ||
			(group.GetAllParticleFlags() & module.b2_elasticParticle) === 0
		) {
			throw new Error('LiquidFun did not create the expected elastic pebble particles.');
		}
		return group;
	} finally {
		release(definition);
		release(linearVelocity);
		release(position);
		destroyParticlePositions?.();
	}
}

function average(
	total: LiquidFunVector,
	value: LiquidFunVector
): LiquidFunVector {
	return { x: total.x + value.x, y: total.y + value.y };
}

function createBondJoint(
	module: LiquidFunModule,
	world: B2World,
	nodeBodies: readonly B2Body[],
	bond: LiquidFunBondSeed,
	stiffness: number,
	damping: number
): B2Joint {
	const firstBody = nodeBodies[bond.firstNodeIndex];
	const secondBody = nodeBodies[bond.secondNodeIndex];
	if (!firstBody || !secondBody) {
		throw new RangeError('A Gate 4 bond must identify existing solid nodes.');
	}

	// GetPosition returns references owned by the bodies, not disposable vectors.
	const firstAnchor = firstBody.GetPosition();
	const secondAnchor = secondBody.GetPosition();
	const definition: B2DistanceJointDef = new module.b2DistanceJointDef();
	try {
		definition.Initialize(firstBody, secondBody, firstAnchor, secondAnchor);
		definition.length = bond.restLengthBox2DUnits;
		definition.stiffness = stiffness;
		definition.damping = damping;
		definition.collideConnected = false;
		const joint = world.CreateJoint(definition);
		if (!joint) {
			throw new Error('LiquidFun did not create a Gate 4 distance joint.');
		}
		return joint;
	} finally {
		release(definition);
	}
}

function bodyDistance(firstBody: B2Body, secondBody: B2Body): number {
	const first = readVector(firstBody.GetPosition());
	const second = readVector(secondBody.GetPosition());
	const distance = Math.hypot(second.x - first.x, second.y - first.y);
	if (!Number.isFinite(distance)) {
		throw new RangeError('A Gate 4 fracture bond length became non-finite.');
	}
	return distance;
}

function breakStretchedFractureBonds(
	world: B2World,
	nodeBodies: readonly B2Body[],
	bonds: readonly FractureBondJoint[]
): number {
	let activeBondCount = 0;
	for (const bond of bonds) {
		const joint = bond.joint;
		if (joint === null) continue;
		const firstBody = nodeBodies[bond.firstNodeIndex];
		const secondBody = nodeBodies[bond.secondNodeIndex];
		if (!firstBody || !secondBody) {
			throw new RangeError('A Gate 4 fracture bond must identify existing nodes.');
		}
		const tensileStrain =
			Math.max(
				0,
				bodyDistance(firstBody, secondBody) - bond.restLengthBox2DUnits
			) / bond.restLengthBox2DUnits;
		if (tensileStrain >= bond.failureStrain) {
			world.DestroyJoint(joint);
			bond.joint = null;
			continue;
		}
		activeBondCount += 1;
	}
	return activeBondCount;
}

function createStaticWalls(module: LiquidFunModule, world: B2World): number {
	const width = LIQUIDFUN_BOX2D_DOMAIN_UNITS.width;
	const height = LIQUIDFUN_BOX2D_DOMAIN_UNITS.height;
	const walls = createLiquidFunWallDefinitions(width, height);

	for (const wall of walls) {
		const body = createBody(
			module,
			world,
			B2_STATIC_BODY,
			wall.position,
			{ x: 0, y: 0 }
		);
		addBoxFixture(module, body, wall.halfWidth, wall.halfHeight);
	}
	return walls.length;
}

function addParticle(
	module: LiquidFunModule,
	particleSystem: B2ParticleSystem,
	position: MpmVector2,
	velocity: MpmVector2,
	flags: number
): void {
	const definition: B2ParticleDef = new module.b2ParticleDef();
	const box2dPosition = createVector(module, mapGate4PositionToBox2D(position));
	const box2dVelocity = createVector(module, mapGate4VelocityToBox2D(velocity));
	try {
		definition.flags = flags;
		definition.position = box2dPosition;
		definition.velocity = box2dVelocity;
		particleSystem.CreateParticle(definition);
	} finally {
		release(box2dPosition);
		release(box2dVelocity);
		release(definition);
	}
}

function pointOnCanvas(
	position: LiquidFunVector,
	width: number,
	height: number,
	pixelsPerBox2DUnit: number,
	offsetX: number,
	offsetY: number
): LiquidFunVector {
	return {
		x: offsetX + position.x * pixelsPerBox2DUnit,
		y: height - offsetY - position.y * pixelsPerBox2DUnit
	};
}

function drawParticles(
	module: LiquidFunModule,
	context: CanvasRenderingContext2D,
	particleSystem: B2ParticleSystem,
	particleCount: number,
	renderMode: LiquidFunRenderMode,
	width: number,
	height: number,
	pixelsPerBox2DUnit: number,
	offsetX: number,
	offsetY: number
): void {
	// This is a borrowed view into LiquidFun-owned memory; don't destroy the wrapper.
	const positionBuffer = particleSystem.GetPositionBuffer();
	const positionBufferPointer = module.getPointer(positionBuffer);
	const positionBufferHeap = module.HEAPF32;
	const count = Math.min(particleCount, particleSystem.GetParticleCount());
	const radius =
		renderMode === 'surface'
			? Math.max(
					2.2,
					LIQUIDFUN_PARTICLE_RADIUS_BOX2D_UNITS * pixelsPerBox2DUnit * 1.35
				)
			: Math.max(
					1.8,
					LIQUIDFUN_PARTICLE_RADIUS_BOX2D_UNITS * pixelsPerBox2DUnit * 0.48
				);
	for (let index = 0; index < count; index += 1) {
		const position = readLiquidFunParticlePosition(
			positionBufferHeap,
			positionBufferPointer,
			index
		);
		const point = pointOnCanvas(
			position,
			width,
			height,
			pixelsPerBox2DUnit,
			offsetX,
			offsetY
		);
		context.beginPath();
		context.arc(point.x, point.y, radius, 0, Math.PI * 2);
		const flags = particleSystem.GetParticleFlags(index);
		context.fillStyle =
			(flags & module.b2_viscousParticle) !== 0
				? '#f2bf38'
				: (flags & module.b2_elasticParticle) !== 0
					? '#c28cff'
					: '#2e8cf2';
		context.fill();
	}
}

function drawElasticParticleTriangles(
	module: LiquidFunModule,
	context: CanvasRenderingContext2D,
	particleSystem: B2ParticleSystem,
	groups: readonly B2ParticleGroup[],
	width: number,
	height: number,
	pixelsPerBox2DUnit: number,
	offsetX: number,
	offsetY: number
): void {
	if (groups.length === 0) return;
	const positionBuffer = particleSystem.GetPositionBuffer();
	const positionBufferPointer = module.getPointer(positionBuffer);
	const positionBufferHeap = module.HEAPF32;
	context.fillStyle = 'rgba(194, 140, 255, 0.24)';
	context.strokeStyle = '#e2c7ff';
	context.lineWidth = 1.25;
	for (const group of groups) {
		const particleCount = group.GetParticleCount();
		if (particleCount !== 3) {
			throw new RangeError('An elastic pebble triangle must contain three particles.');
		}
		const bufferIndex = group.GetBufferIndex();
		const positions = Array.from({ length: particleCount }, (_, index) =>
			readLiquidFunParticlePosition(
				positionBufferHeap,
				positionBufferPointer,
				bufferIndex + index
			)
		);
		const center = positions.reduce(
			(total, point) => ({
				x: total.x + point.x / positions.length,
				y: total.y + point.y / positions.length
			}),
			{ x: 0, y: 0 }
		);
		positions.sort(
			(first, second) =>
				Math.atan2(first.y - center.y, first.x - center.x) -
				Math.atan2(second.y - center.y, second.x - center.x)
		);
		const points = positions.map((position) =>
			pointOnCanvas(
				position,
				width,
				height,
				pixelsPerBox2DUnit,
				offsetX,
				offsetY
			)
		);
		const first = points[0];
		if (!first) {
			throw new RangeError('An elastic pebble triangle has no drawable particles.');
		}
		context.beginPath();
		context.moveTo(first.x, first.y);
		for (const point of points.slice(1)) {
			context.lineTo(point.x, point.y);
		}
		context.closePath();
		context.fill();
		context.stroke();
	}
}

function readBodyPosition(body: B2Body): LiquidFunVector {
	// GetPosition returns a borrowed view owned by Box2D.
	return readVector(body.GetPosition());
}

function drawPebbleTriangles(
	context: CanvasRenderingContext2D,
	triangleBodies: readonly PebbleTriangleBody[],
	width: number,
	height: number,
	pixelsPerBox2DUnit: number,
	offsetX: number,
	offsetY: number
): void {
	context.fillStyle = '#f5d77a';
	context.strokeStyle = '#fff0b8';
	context.lineWidth = 1.25;
	for (const triangle of triangleBodies) {
		const center = readBodyPosition(triangle.body);
		const angle = triangle.body.GetAngle();
		const cosine = Math.cos(angle);
		const sine = Math.sin(angle);
		const points = triangle.localVertices.map((vertex) =>
			pointOnCanvas(
				{
					x: center.x + cosine * vertex.x - sine * vertex.y,
					y: center.y + sine * vertex.x + cosine * vertex.y
				},
				width,
				height,
				pixelsPerBox2DUnit,
				offsetX,
				offsetY
			)
		);
		const first = points[0];
		const second = points[1];
		const third = points[2];
		if (!first || !second || !third) {
			throw new RangeError('A Gate 4 pebble must identify three drawable vertices.');
		}
		context.beginPath();
		context.moveTo(first.x, first.y);
		context.lineTo(second.x, second.y);
		context.lineTo(third.x, third.y);
		context.closePath();
		context.fill();
		context.stroke();
	}
}

function drawSoftSquares(
	context: CanvasRenderingContext2D,
	squares: readonly SoftSquareBody[],
	width: number,
	height: number,
	pixelsPerBox2DUnit: number,
	offsetX: number,
	offsetY: number
): void {
	context.fillStyle = '#70cfc0';
	context.strokeStyle = '#d7fff8';
	context.lineWidth = 1.25;
	for (const square of squares) {
		const points = square.map((body) =>
			pointOnCanvas(
				readBodyPosition(body),
				width,
				height,
				pixelsPerBox2DUnit,
				offsetX,
				offsetY
			)
		);
		const first = points[0];
		const second = points[1];
		const third = points[2];
		const fourth = points[3];
		if (!first || !second || !third || !fourth) {
			throw new RangeError('A soft square must identify four drawable vertices.');
		}
		context.beginPath();
		context.moveTo(first.x, first.y);
		context.lineTo(second.x, second.y);
		context.lineTo(third.x, third.y);
		context.lineTo(fourth.x, fourth.y);
		context.closePath();
		context.fill();
		context.stroke();
	}
}

function drawFractureNetwork(
	context: CanvasRenderingContext2D,
	fractureNodeBodies: readonly B2Body[],
	fractureBonds: readonly FractureBondJoint[],
	width: number,
	height: number,
	pixelsPerBox2DUnit: number,
	offsetX: number,
	offsetY: number
): void {
	const points = fractureNodeBodies.map((body) =>
		pointOnCanvas(
			readBodyPosition(body),
			width,
			height,
			pixelsPerBox2DUnit,
			offsetX,
			offsetY
		)
	);
	context.strokeStyle = 'rgba(245, 215, 122, 0.7)';
	context.lineWidth = 1;
	context.beginPath();
	for (const bond of fractureBonds) {
		if (bond.joint === null) continue;
		const first = points[bond.firstNodeIndex];
		const second = points[bond.secondNodeIndex];
		if (!first || !second) {
			throw new RangeError('A Gate 4 fracture bond must identify drawable nodes.');
		}
		context.moveTo(first.x, first.y);
		context.lineTo(second.x, second.y);
	}
	context.stroke();
	context.fillStyle = '#f5d77a';
	const radius = Math.max(
		1.5,
		LIQUIDFUN_SOLID_NODE_RADIUS_BOX2D_UNITS * pixelsPerBox2DUnit * 0.65
	);
	for (const point of points) {
		context.beginPath();
		context.arc(point.x, point.y, radius, 0, Math.PI * 2);
		context.fill();
	}
}

export async function createLiquidFunSimulation(): Promise<LiquidFunSimulation> {
	const workload = createLiquidFunGate4Workload();
	const moduleFactory = (await import('liquidfun-wasm')).default;
	const module = await moduleFactory();
	const gravity = createVector(
		module,
		LIQUIDFUN_BOX2D_GRAVITY_UNITS_PER_SECOND_SQUARED
	);
	let world: B2World;
	try {
		world = new module.b2World(gravity);
	} finally {
		release(gravity);
	}

	let particleSystem: B2ParticleSystem | undefined;
	const pebbleTriangleBodies: PebbleTriangleBody[] = [];
	const elasticParticleGroups: B2ParticleGroup[] = [];
	const softSquareBodies: SoftSquareBody[] = [];
	const softSquareNodeBodies: B2Body[] = [];
	const softSquareSpringJoints: B2Joint[] = [];
	const fractureNodeBodies: B2Body[] = [];
	const fractureBondJoints: FractureBondJoint[] = [];
	let disposed = false;
	try {
		const particleSystemDefinition: B2ParticleSystemDef =
			new module.b2ParticleSystemDef();
		particleSystemDefinition.density = LIQUIDFUN_SHARED_PARTICLE_DENSITY;
		particleSystemDefinition.pressureStrength = LIQUIDFUN_PRESSURE_STRENGTH;
		particleSystemDefinition.elasticStrength = LIQUIDFUN_ELASTIC_STRENGTH;
		particleSystemDefinition.gravityScale = 1;
		particleSystemDefinition.radius = LIQUIDFUN_PARTICLE_RADIUS_BOX2D_UNITS;
		try {
			particleSystem = world.CreateParticleSystem(particleSystemDefinition);
		} finally {
			release(particleSystemDefinition);
		}

		for (const particle of workload.waterParticles) {
			addParticle(
				module,
				particleSystem,
				particle.position,
				particle.velocity,
				module.b2_waterParticle
			);
		}
		for (const particle of workload.oilParticles) {
			addParticle(
				module,
				particleSystem,
				particle.position,
				particle.velocity,
				module.b2_viscousParticle
			);
		}
		const fluidParticleCount = particleSystem.GetParticleCount();
		const expectedFluidParticleCount =
			workload.waterParticles.length + workload.oilParticles.length;
		if (fluidParticleCount !== expectedFluidParticleCount) {
			throw new Error(
				`LiquidFun created ${fluidParticleCount} of ${expectedFluidParticleCount} Gate 4 fluid particles.`
			);
		}

		const staticWallBodyCount = createStaticWalls(module, world);
		for (const group of workload.pebbleGroups) {
			const first = workload.pebbleNodes[group.nodeIndices[0]];
			const second = workload.pebbleNodes[group.nodeIndices[1]];
			const third = workload.pebbleNodes[group.nodeIndices[2]];
			if (!first || !second || !third) {
				throw new RangeError('A Gate 4 pebble must identify three existing nodes.');
			}
			const triangleNodes: [MpmParticle, MpmParticle, MpmParticle] = [
				first,
				second,
				third
			];
			if (group.usesElasticParticleFlag) {
				const previousTriadCount = particleSystem.GetTriadCount();
				const elasticGroup = createElasticParticleTriangleGroup(
					module,
					particleSystem,
					triangleNodes
				);
				elasticParticleGroups.push(elasticGroup);
				if (particleSystem.GetTriadCount() !== previousTriadCount + 1) {
					throw new Error(
						'LiquidFun did not create one elastic triad for the pebble triangle.'
					);
				}
			} else {
				pebbleTriangleBodies.push(
					createPebbleTriangleBody(module, world, triangleNodes)
				);
			}
		}
		for (const square of workload.softSquareGroups) {
			const nodeBodies: B2Body[] = [];
			for (const node of square.nodes) {
				const body = createSoftSquareNodeBody(module, world, node);
				nodeBodies.push(body);
				softSquareNodeBodies.push(body);
			}
			const [first, second, third, fourth] = nodeBodies;
			if (!first || !second || !third || !fourth) {
				throw new RangeError('A soft square must identify four existing nodes.');
			}
			const squareBodies: SoftSquareBody = [first, second, third, fourth];
			softSquareBodies.push(squareBodies);
			for (const bond of square.bonds) {
				softSquareSpringJoints.push(
					createBondJoint(
						module,
						world,
						squareBodies,
						bond,
						LIQUIDFUN_SOFT_SQUARE_SPRING_STIFFNESS,
						LIQUIDFUN_SOFT_SQUARE_SPRING_DAMPING
					)
				);
			}
		}
		for (const particle of workload.fractureNodes) {
			fractureNodeBodies.push(createSolidNodeBody(module, world, particle));
		}
		for (const bond of workload.fractureBonds) {
			fractureBondJoints.push({
				...bond,
				joint: createBondJoint(
					module,
					world,
					fractureNodeBodies,
					bond,
					LIQUIDFUN_FRACTURE_BOND_STIFFNESS,
					LIQUIDFUN_FRACTURE_BOND_DAMPING
				)
			});
		}
		let activeFractureBondCount = fractureBondJoints.length;

		const elasticParticleCount = elasticParticleGroups.reduce(
			(count, group) => count + group.GetParticleCount(),
			0
		);
		const elasticTriadCount = particleSystem.GetTriadCount();
		const particleCount = particleSystem.GetParticleCount();
		if (
			pebbleTriangleBodies.length !== LIQUIDFUN_RIGID_TRIANGLE_COUNT ||
			elasticParticleGroups.length !== LIQUIDFUN_ELASTIC_TRIANGLE_COUNT ||
			elasticParticleCount !== LIQUIDFUN_ELASTIC_PARTICLE_COUNT ||
			elasticTriadCount !== LIQUIDFUN_ELASTIC_TRIANGLE_COUNT ||
			particleCount !== fluidParticleCount + elasticParticleCount
		) {
			throw new Error('LiquidFun pebble triangle counts do not match the configured workload.');
		}

		const particleIterations = world.CalculateReasonableParticleIterations(
			LIQUIDFUN_DT_SECONDS
		);
		if (!Number.isSafeInteger(particleIterations) || particleIterations < 1) {
			throw new RangeError('LiquidFun returned an invalid particle iteration count.');
		}
		const activeParticleSystem = particleSystem;

		return {
			particleCount,
			waterParticleCount: workload.waterParticles.length,
			oilParticleCount: workload.oilParticles.length,
			mappedMpmNodeCount: workload.mappedMpmNodeCount,
			pebbleNodeCount: workload.pebbleNodes.length,
			triangleBodyCount: pebbleTriangleBodies.length,
			elasticTriangleCount: elasticParticleGroups.length,
			elasticParticleCount,
			elasticTriadCount,
			softSquareCount: softSquareBodies.length,
			softSquareNodeCount: softSquareNodeBodies.length,
			softSquareJointCount: softSquareSpringJoints.length,
			pebbleVertexCount:
				pebbleTriangleBodies.reduce(
					(vertexCount, triangle) =>
						vertexCount + triangle.localVertices.length,
					0
				) +
				elasticParticleCount +
				softSquareNodeBodies.length,
			fractureNodeCount: fractureNodeBodies.length,
			fractureBondCount: fractureBondJoints.length,
			get activeFractureBondCount(): number {
				return activeFractureBondCount;
			},
			staticWallBodyCount,
			sourceMpmNodeCount: workload.sourceMpmNodeCount,
			particleIterations,
			step(): void {
				if (disposed) throw new Error('The LiquidFun simulation has been disposed.');
				world.Step(
					LIQUIDFUN_DT_SECONDS,
					LIQUIDFUN_VELOCITY_ITERATIONS,
					LIQUIDFUN_POSITION_ITERATIONS,
					particleIterations
				);
				activeFractureBondCount = breakStretchedFractureBonds(
					world,
					fractureNodeBodies,
					fractureBondJoints
				);
			},
			setGravityFromGate4(gravityInGate4Units: MpmVector2): void {
				if (disposed) throw new Error('The LiquidFun simulation has been disposed.');
				const vector = createVector(
					module,
					mapGate4AccelerationToBox2D(gravityInGate4Units)
				);
				try {
					world.SetGravity(vector);
				} finally {
					release(vector);
				}
			},
			draw(
				context: CanvasRenderingContext2D,
				width: number,
				height: number,
				renderMode: LiquidFunRenderMode
			): void {
				if (disposed || width <= 0 || height <= 0) return;
				context.clearRect(0, 0, width, height);
				context.fillStyle = '#101a28';
				context.fillRect(0, 0, width, height);

				const pixelsPerBox2DUnit = Math.min(
					width / LIQUIDFUN_BOX2D_DOMAIN_UNITS.width,
					height / LIQUIDFUN_BOX2D_DOMAIN_UNITS.height
				);
				const offsetX =
					(width - LIQUIDFUN_BOX2D_DOMAIN_UNITS.width * pixelsPerBox2DUnit) / 2;
				const offsetY =
					(height - LIQUIDFUN_BOX2D_DOMAIN_UNITS.height * pixelsPerBox2DUnit) / 2;
				const domainWidth =
					LIQUIDFUN_BOX2D_DOMAIN_UNITS.width * pixelsPerBox2DUnit;
				const domainHeight =
					LIQUIDFUN_BOX2D_DOMAIN_UNITS.height * pixelsPerBox2DUnit;

				context.save();
				try {
					context.beginPath();
					context.rect(offsetX, offsetY, domainWidth, domainHeight);
					context.clip();

					drawElasticParticleTriangles(
						module,
						context,
						activeParticleSystem,
						elasticParticleGroups,
						width,
						height,
						pixelsPerBox2DUnit,
						offsetX,
						offsetY
					);
					drawParticles(
						module,
						context,
						activeParticleSystem,
						particleCount,
						renderMode,
						width,
						height,
						pixelsPerBox2DUnit,
						offsetX,
						offsetY
					);
					drawPebbleTriangles(
						context,
						pebbleTriangleBodies,
						width,
						height,
						pixelsPerBox2DUnit,
						offsetX,
						offsetY
					);
					drawSoftSquares(
						context,
						softSquareBodies,
						width,
						height,
						pixelsPerBox2DUnit,
						offsetX,
						offsetY
					);
					drawFractureNetwork(
						context,
						fractureNodeBodies,
						fractureBondJoints,
						width,
						height,
						pixelsPerBox2DUnit,
						offsetX,
						offsetY
					);
				} finally {
					context.restore();
				}

				context.strokeStyle = '#d6e3f1';
				context.lineWidth = 1;
				context.strokeRect(offsetX, offsetY, domainWidth, domainHeight);
			},
			dispose(): void {
				if (disposed) return;
				disposed = true;
				for (const bond of fractureBondJoints) {
					bond.joint = null;
				}
				fractureBondJoints.length = 0;
				softSquareSpringJoints.length = 0;
				pebbleTriangleBodies.length = 0;
				elasticParticleGroups.length = 0;
				softSquareNodeBodies.length = 0;
				softSquareBodies.length = 0;
				fractureNodeBodies.length = 0;
				release(world);
			}
		};
	} catch (error) {
		release(world);
		throw error;
	}
}

export const LIQUIDFUN_REALTIME_CONFIG = {
	dtSeconds: LIQUIDFUN_DT_SECONDS,
	gate4Gravity: LIQUIDFUN_GATE4_GRAVITY,
	gate4UnitsPerMeter: LIQUIDFUN_GATE4_UNITS_PER_METER,
	gate4AccelerationUnitsPerMeterPerSecondSquared:
		LIQUIDFUN_GATE4_ACCELERATION_UNITS_PER_METER_PER_SECOND_SQUARED,
	domainGate4Units: LIQUIDFUN_GATE4_DOMAIN,
	domainMeters: LIQUIDFUN_PHYSICAL_DOMAIN_METERS,
	box2dUnitsPerMeter: LIQUIDFUN_BOX2D_UNITS_PER_METER,
	domainBox2dUnits: LIQUIDFUN_BOX2D_DOMAIN_UNITS,
	particleRadiusMeters: LIQUIDFUN_PARTICLE_RADIUS_METERS,
	particleRadiusBox2dUnits: LIQUIDFUN_PARTICLE_RADIUS_BOX2D_UNITS,
	sharedParticleMass: LIQUIDFUN_SHARED_PARTICLE_MASS,
	pebbleTriangleCount: LIQUIDFUN_PEBBLE_TRIANGLE_COUNT,
	rigidTriangleCount: LIQUIDFUN_RIGID_TRIANGLE_COUNT,
	elasticTriangleCount: LIQUIDFUN_ELASTIC_TRIANGLE_COUNT,
	elasticParticleCount: LIQUIDFUN_ELASTIC_PARTICLE_COUNT,
	elasticStrength: LIQUIDFUN_ELASTIC_STRENGTH,
	pebbleEdgeLengthMeters: LIQUIDFUN_PEBBLE_EDGE_LENGTH_METERS,
	pebbleMass: LIQUIDFUN_PEBBLE_MASS,
	softSquareCount: LIQUIDFUN_SOFT_SQUARE_COUNT,
	softSquareEdgeLengthMeters: LIQUIDFUN_SOFT_SQUARE_EDGE_LENGTH_METERS,
	softSquareMass: LIQUIDFUN_SOFT_SQUARE_MASS,
	softSquareSpringFrequencyHz: LIQUIDFUN_SOFT_SQUARE_SPRING_FREQUENCY_HZ,
	softSquareSpringDampingRatio: LIQUIDFUN_SOFT_SQUARE_SPRING_DAMPING_RATIO,
	velocityIterations: LIQUIDFUN_VELOCITY_ITERATIONS,
	positionIterations: LIQUIDFUN_POSITION_ITERATIONS
} as const;
