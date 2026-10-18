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
import { readLiquidFunParticlePosition } from './particle-buffer.js';
import {
	createLiquidFunGate4Workload,
	LIQUIDFUN_BOX2D_DOMAIN_UNITS,
	LIQUIDFUN_BOX2D_GRAVITY_UNITS_PER_SECOND_SQUARED,
	LIQUIDFUN_BOX2D_UNITS_PER_METER,
	LIQUIDFUN_DT_SECONDS,
	LIQUIDFUN_ELASTIC_BALL_COUNT,
	LIQUIDFUN_ELASTIC_BALL_PARTICLE_COUNT,
	LIQUIDFUN_ELASTIC_BALL_PARTICLES_PER_GROUP,
	LIQUIDFUN_ELASTIC_STRENGTH,
	LIQUIDFUN_FLUID_PARTICLE_COUNT,
	LIQUIDFUN_FRACTURE_BOND_DAMPING,
	LIQUIDFUN_FRACTURE_BOND_STIFFNESS,
	LIQUIDFUN_GATE4_ACCELERATION_UNITS_PER_METER_PER_SECOND_SQUARED,
	LIQUIDFUN_GATE4_DOMAIN,
	LIQUIDFUN_GATE4_GRAVITY,
	LIQUIDFUN_GATE4_UNITS_PER_METER,
	LIQUIDFUN_INK_PARTICLE_COUNT,
	LIQUIDFUN_OIL_PARTICLE_COUNT,
	LIQUIDFUN_PARTICLE_RADIUS_METERS,
	LIQUIDFUN_PARTICLE_RADIUS_BOX2D_UNITS,
	LIQUIDFUN_PHYSICAL_DOMAIN_METERS,
	LIQUIDFUN_POSITION_ITERATIONS,
	LIQUIDFUN_PRESSURE_STRENGTH,
	LIQUIDFUN_SHARED_PARTICLE_DENSITY,
	LIQUIDFUN_SHARED_PARTICLE_MASS,
	LIQUIDFUN_SOIL_PARTICLE_COUNT,
	LIQUIDFUN_SOLID_NODE_DENSITY,
	LIQUIDFUN_SOLID_NODE_RADIUS_BOX2D_UNITS,
	LIQUIDFUN_SPRING_STRENGTH,
	LIQUIDFUN_SPRING_VINE_COUNT,
	LIQUIDFUN_SPRING_VINE_LINK_COUNT,
	LIQUIDFUN_SPRING_VINE_PARTICLE_COUNT,
	LIQUIDFUN_SPRING_VINE_PARTICLES_PER_GROUP,
	LIQUIDFUN_SPRING_VINE_SPAN_METERS,
	LIQUIDFUN_TOTAL_PARTICLE_COUNT,
	LIQUIDFUN_VELOCITY_ITERATIONS,
	LIQUIDFUN_WATER_PARTICLE_COUNT,
	mapGate4AccelerationToBox2D,
	mapGate4PositionToBox2D,
	mapGate4VelocityToBox2D,
	type LiquidFunFractureBondSeed,
	type LiquidFunParticleGroupSeed,
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
	readonly inkParticleCount: number;
	readonly soilParticleCount: number;
	readonly elasticBallCount: number;
	readonly elasticParticleCount: number;
	readonly elasticTriadCount: number;
	readonly springVineCount: number;
	readonly springVineParticleCount: number;
	readonly springPairCount: number;
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

function createFlaggedParticleGroup(
	module: LiquidFunModule,
	particleSystem: B2ParticleSystem,
	seed: LiquidFunParticleGroupSeed,
	flags: number,
	label: string
): B2ParticleGroup {
	if (seed.positions.length === 0) {
		throw new RangeError(`${label} ${seed.groupIndex} must contain particles.`);
	}
	const positions = seed.positions.map(mapGate4PositionToBox2D);
	const center = positions.reduce(
		(total, position) => ({
			x: total.x + position.x / positions.length,
			y: total.y + position.y / positions.length
		}),
		{ x: 0, y: 0 }
	);
	const localPositions = positions.map((position) => ({
		x: position.x - center.x,
		y: position.y - center.y
	}));
	let particlePositions: B2Vec2 | undefined;
	let destroyParticlePositions: (() => void) | undefined;
	let position: B2Vec2 | undefined;
	let linearVelocity: B2Vec2 | undefined;
	let definition: B2ParticleGroupDef | undefined;
	let group: B2ParticleGroup | undefined;
	try {
		[particlePositions, destroyParticlePositions] = module.pointsToVec2Array(
			localPositions
		);
		position = createVector(module, center);
		linearVelocity = createVector(module, { x: 0, y: 0 });
		definition = new module.b2ParticleGroupDef();
		definition.flags = flags;
		definition.position = position;
		definition.angle = 0;
		definition.linearVelocity = linearVelocity;
		definition.angularVelocity = 0;
		definition.particleCount = localPositions.length;
		definition.positionData = particlePositions;
		group = particleSystem.CreateParticleGroup(definition);
		if (
			group.GetParticleCount() !== seed.positions.length ||
			(group.GetAllParticleFlags() & flags) !== flags
		) {
			throw new Error(`LiquidFun did not create the expected ${label} ${seed.groupIndex}.`);
		}
		return group;
	} finally {
		release(definition);
		release(linearVelocity);
		release(position);
		destroyParticlePositions?.();
	}
}

function createBondJoint(
	module: LiquidFunModule,
	world: B2World,
	nodeBodies: readonly B2Body[],
	bond: LiquidFunFractureBondSeed,
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
): number {
	const definition: B2ParticleDef = new module.b2ParticleDef();
	const box2dPosition = createVector(module, mapGate4PositionToBox2D(position));
	const box2dVelocity = createVector(module, mapGate4VelocityToBox2D(velocity));
	try {
		definition.flags = flags;
		definition.position = box2dPosition;
		definition.velocity = box2dVelocity;
		const particleIndex = particleSystem.CreateParticle(definition);
		if (!Number.isSafeInteger(particleIndex) || particleIndex < 0) {
			throw new Error('LiquidFun could not create a configured particle.');
		}
		return particleIndex;
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
			(flags & module.b2_elasticParticle) !== 0
				? '#c28cff'
				: (flags & module.b2_springParticle) !== 0
					? '#72d58c'
					: (flags & module.b2_powderParticle) !== 0
						? (flags & module.b2_viscousParticle) !== 0
							? '#98653e'
							: '#20242d'
						: (flags & module.b2_viscousParticle) !== 0
							? '#f2bf38'
							: '#2e8cf2';
		context.fill();
	}
}

function drawSpringVines(
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
	context.strokeStyle = '#a9f0b8';
	context.lineWidth = Math.max(1.25, pixelsPerBox2DUnit * 0.45);
	for (const group of groups) {
		const particleCount = group.GetParticleCount();
		if (particleCount < 2) {
			throw new RangeError('A spring vine must contain at least two particles.');
		}
		const bufferIndex = group.GetBufferIndex();
		context.beginPath();
		for (let index = 0; index < particleCount; index += 1) {
			const point = pointOnCanvas(
				readLiquidFunParticlePosition(
					positionBufferHeap,
					positionBufferPointer,
					bufferIndex + index
				),
				width,
				height,
				pixelsPerBox2DUnit,
				offsetX,
				offsetY
			)
			if (index === 0) {
				context.moveTo(point.x, point.y);
			} else {
				context.lineTo(point.x, point.y);
			}
		}
		context.stroke();
	}
}

function readBodyPosition(body: B2Body): LiquidFunVector {
	// GetPosition returns a borrowed view owned by Box2D.
	return readVector(body.GetPosition());
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
	const elasticBallGroups: B2ParticleGroup[] = [];
	const springVineGroups: B2ParticleGroup[] = [];
	const fractureNodeBodies: B2Body[] = [];
	const fractureBondJoints: FractureBondJoint[] = [];
	let disposed = false;
	try {
		const particleSystemDefinition: B2ParticleSystemDef =
			new module.b2ParticleSystemDef();
		particleSystemDefinition.density = LIQUIDFUN_SHARED_PARTICLE_DENSITY;
		particleSystemDefinition.pressureStrength = LIQUIDFUN_PRESSURE_STRENGTH;
		particleSystemDefinition.elasticStrength = LIQUIDFUN_ELASTIC_STRENGTH;
		particleSystemDefinition.springStrength = LIQUIDFUN_SPRING_STRENGTH;
		particleSystemDefinition.gravityScale = 1;
		particleSystemDefinition.radius = LIQUIDFUN_PARTICLE_RADIUS_BOX2D_UNITS;
		try {
			particleSystem = world.CreateParticleSystem(particleSystemDefinition);
		} finally {
			release(particleSystemDefinition);
		}

		const fluidParticleGroups = [
			{ particles: workload.waterParticles, flags: module.b2_waterParticle },
			{ particles: workload.oilParticles, flags: module.b2_viscousParticle },
			{ particles: workload.inkParticles, flags: module.b2_powderParticle },
			{
				particles: workload.soilParticles,
				flags: module.b2_powderParticle | module.b2_viscousParticle
			}
		] as const;
		for (const group of fluidParticleGroups) {
			for (const particle of group.particles) {
				const index = addParticle(
					module,
					particleSystem,
					particle.position,
					particle.velocity,
					group.flags
				);
				if (
					(particleSystem.GetParticleFlags(index) & group.flags) !== group.flags
				) {
					throw new Error('LiquidFun did not apply the configured fluid particle flags.');
				}
			}
		}
		const fluidParticleCount = particleSystem.GetParticleCount();
		if (fluidParticleCount !== LIQUIDFUN_FLUID_PARTICLE_COUNT) {
			throw new Error(
				`LiquidFun created ${fluidParticleCount} of ${LIQUIDFUN_FLUID_PARTICLE_COUNT} configured fluid particles.`
			);
		}

		const staticWallBodyCount = createStaticWalls(module, world);
		for (const group of workload.elasticBallGroups) {
			elasticBallGroups.push(
				createFlaggedParticleGroup(
					module,
					particleSystem,
					group,
					module.b2_elasticParticle,
					'elastic ball'
				)
			);
		}
		for (const group of workload.springVines) {
			springVineGroups.push(
				createFlaggedParticleGroup(
					module,
					particleSystem,
					group,
					module.b2_springParticle,
					'spring vine'
				)
			);
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

		const elasticParticleCount = elasticBallGroups.reduce(
			(count, group) => count + group.GetParticleCount(),
			0
		);
		const springVineParticleCount = springVineGroups.reduce(
			(count, group) => count + group.GetParticleCount(),
			0
		);
		const springPairCount = particleSystem.GetPairCount();
		const elasticTriadCount = particleSystem.GetTriadCount();
		const particleCount = particleSystem.GetParticleCount();
		if (springPairCount !== LIQUIDFUN_SPRING_VINE_LINK_COUNT) {
			throw new Error(
				`LiquidFun created ${springPairCount} of ${LIQUIDFUN_SPRING_VINE_LINK_COUNT} expected spring-vine links.`
			);
		}
		if (
			workload.waterParticles.length !== LIQUIDFUN_WATER_PARTICLE_COUNT ||
			workload.oilParticles.length !== LIQUIDFUN_OIL_PARTICLE_COUNT ||
			workload.inkParticles.length !== LIQUIDFUN_INK_PARTICLE_COUNT ||
			workload.soilParticles.length !== LIQUIDFUN_SOIL_PARTICLE_COUNT ||
			elasticBallGroups.length !== LIQUIDFUN_ELASTIC_BALL_COUNT ||
			elasticParticleCount !== LIQUIDFUN_ELASTIC_BALL_PARTICLE_COUNT ||
			elasticTriadCount === 0 ||
			springVineGroups.length !== LIQUIDFUN_SPRING_VINE_COUNT ||
			springVineParticleCount !== LIQUIDFUN_SPRING_VINE_PARTICLE_COUNT ||
			particleCount !== LIQUIDFUN_TOTAL_PARTICLE_COUNT
		) {
			throw new Error('LiquidFun particle counts do not match the configured workload.');
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
			inkParticleCount: workload.inkParticles.length,
			soilParticleCount: workload.soilParticles.length,
			elasticBallCount: elasticBallGroups.length,
			elasticParticleCount,
			elasticTriadCount,
			springVineCount: springVineGroups.length,
			springVineParticleCount,
			get springPairCount(): number {
				return activeParticleSystem.GetPairCount();
			},
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
					drawSpringVines(
						module,
						context,
						activeParticleSystem,
						springVineGroups,
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
				elasticBallGroups.length = 0;
				springVineGroups.length = 0;
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
	pressureStrength: LIQUIDFUN_PRESSURE_STRENGTH,
	waterParticleCount: LIQUIDFUN_WATER_PARTICLE_COUNT,
	oilParticleCount: LIQUIDFUN_OIL_PARTICLE_COUNT,
	inkParticleCount: LIQUIDFUN_INK_PARTICLE_COUNT,
	soilParticleCount: LIQUIDFUN_SOIL_PARTICLE_COUNT,
	elasticBallCount: LIQUIDFUN_ELASTIC_BALL_COUNT,
	elasticBallParticleCount: LIQUIDFUN_ELASTIC_BALL_PARTICLE_COUNT,
	elasticBallParticlesPerGroup: LIQUIDFUN_ELASTIC_BALL_PARTICLES_PER_GROUP,
	elasticStrength: LIQUIDFUN_ELASTIC_STRENGTH,
	springStrength: LIQUIDFUN_SPRING_STRENGTH,
	springVineCount: LIQUIDFUN_SPRING_VINE_COUNT,
	springVineLinkCount: LIQUIDFUN_SPRING_VINE_LINK_COUNT,
	springVineParticleCount: LIQUIDFUN_SPRING_VINE_PARTICLE_COUNT,
	springVineParticlesPerGroup: LIQUIDFUN_SPRING_VINE_PARTICLES_PER_GROUP,
	springVineSpanMeters: LIQUIDFUN_SPRING_VINE_SPAN_METERS,
	totalParticleCount: LIQUIDFUN_TOTAL_PARTICLE_COUNT,
	velocityIterations: LIQUIDFUN_VELOCITY_ITERATIONS,
	positionIterations: LIQUIDFUN_POSITION_ITERATIONS
} as const;
