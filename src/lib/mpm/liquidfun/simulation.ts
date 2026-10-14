import type {
	B2Body,
	B2BodyDef,
	B2CircleShape,
	B2DistanceJointDef,
	B2Fixture,
	B2Joint,
	B2ParticleDef,
	B2ParticleSystem,
	B2ParticleSystemDef,
	B2PolygonShape,
	B2Vec2,
	B2World,
	LiquidFunModule
} from 'liquidfun-wasm';
import type { MpmParticle, MpmVector2 } from '../reference/solver.js';
import {
	createLiquidFunGate4Workload,
	LIQUIDFUN_BOX2D_DOMAIN_METERS,
	LIQUIDFUN_BOX2D_GRAVITY,
	LIQUIDFUN_DT_SECONDS,
	LIQUIDFUN_GATE4_DOMAIN,
	LIQUIDFUN_GATE4_GRAVITY,
	LIQUIDFUN_GATE4_UNITS_PER_METER,
	LIQUIDFUN_FRACTURE_BOND_DAMPING,
	LIQUIDFUN_FRACTURE_BOND_STIFFNESS,
	LIQUIDFUN_PARTICLE_RADIUS_METERS,
	LIQUIDFUN_PEBBLE_BOND_DAMPING,
	LIQUIDFUN_PEBBLE_BOND_STIFFNESS,
	LIQUIDFUN_POSITION_ITERATIONS,
	LIQUIDFUN_SHARED_PARTICLE_DENSITY,
	LIQUIDFUN_SOLID_NODE_DENSITY,
	LIQUIDFUN_SOLID_NODE_RADIUS_METERS,
	LIQUIDFUN_VELOCITY_ITERATIONS,
	mapGate4PositionToBox2D,
	mapGate4VelocityToBox2D,
	type LiquidFunBondSeed,
	type LiquidFunFractureBondSeed,
	type LiquidFunTriangleGroupSeed,
	type LiquidFunVector
} from './workload.js';

const B2_STATIC_BODY = 0;
const B2_DYNAMIC_BODY = 2;
const WALL_THICKNESS_METERS = 0.1;

export interface LiquidFunSimulation {
	readonly particleCount: number;
	readonly waterParticleCount: number;
	readonly oilParticleCount: number;
	readonly mappedMpmNodeCount: number;
	readonly pebbleNodeCount: number;
	readonly triangleGroupCount: number;
	readonly pebbleBondCount: number;
	readonly fractureNodeCount: number;
	readonly fractureBondCount: number;
	readonly activeFractureBondCount: number;
	readonly staticWallBodyCount: number;
	readonly sourceMpmNodeCount: number;
	readonly particleIterations: number;
	step(): void;
	setGravityFromGate4(gravity: MpmVector2): void;
	draw(context: CanvasRenderingContext2D, width: number, height: number): void;
	dispose(): void;
}

interface PebbleBondJoint extends LiquidFunBondSeed {
	readonly joint: B2Joint;
}

interface FractureBondJoint extends LiquidFunFractureBondSeed {
	joint: B2Joint | null;
}

interface PositionBufferWithGet {
	get(index: number): unknown;
}

interface PositionBufferWithAt {
	at(index: number): unknown;
}

interface PositionBufferWithAxisGetters {
	get_x(index: number): number;
	get_y(index: number): number;
}

interface VectorWithAxisGetters {
	get_x(): number;
	get_y(): number;
}

function release(value: { delete?: () => void } | null | undefined): void {
	value?.delete?.();
}

function releaseUnknown(value: unknown): void {
	if (
		typeof value === 'object' &&
		value !== null &&
		'delete' in value &&
		typeof value.delete === 'function'
	) {
		value.delete();
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

function hasPositionBufferGet(value: unknown): value is PositionBufferWithGet {
	return (
		typeof value === 'object' &&
		value !== null &&
		'get' in value &&
		typeof value.get === 'function'
	);
}

function hasPositionBufferAt(value: unknown): value is PositionBufferWithAt {
	return (
		typeof value === 'object' &&
		value !== null &&
		'at' in value &&
		typeof value.at === 'function'
	);
}

function hasPositionBufferAxisGetters(
	value: unknown
): value is PositionBufferWithAxisGetters {
	return (
		typeof value === 'object' &&
		value !== null &&
		'get_x' in value &&
		typeof value.get_x === 'function' &&
		'get_y' in value &&
		typeof value.get_y === 'function'
	);
}

function readPositionAt(buffer: unknown, index: number): LiquidFunVector {
	if (buffer instanceof Float32Array || buffer instanceof Float64Array) {
		return { x: buffer[index * 2], y: buffer[index * 2 + 1] };
	}
	if (Array.isArray(buffer)) {
		if (
			typeof buffer[index * 2] === 'number' &&
			typeof buffer[index * 2 + 1] === 'number'
		) {
			return {
				x: buffer[index * 2],
				y: buffer[index * 2 + 1]
			};
		}
		return readVector(buffer[index]);
	}
	if (hasPositionBufferAxisGetters(buffer)) {
		return { x: buffer.get_x(index), y: buffer.get_y(index) };
	}
	if (hasPositionBufferGet(buffer)) {
		const point = buffer.get(index);
		try {
			return readVector(point);
		} finally {
			releaseUnknown(point);
		}
	}
	if (hasPositionBufferAt(buffer)) {
		const point = buffer.at(index);
		try {
			return readVector(point);
		} finally {
			releaseUnknown(point);
		}
	}
	if (index === 0) return readVector(buffer);
	throw new TypeError(
		'LiquidFun GetPositionBuffer() did not expose an indexed b2Vec2 buffer.'
	);
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
	const shape: B2PolygonShape = new module.b2PolygonShape();
	let fixture: B2Fixture | undefined;
	try {
		shape.SetAsBox(halfWidth, halfHeight);
		fixture = body.CreateFixture(shape, 0);
	} finally {
		release(fixture);
		release(shape);
	}
}

function addSolidNodeFixture(module: LiquidFunModule, body: B2Body): void {
	const shape: B2CircleShape = new module.b2CircleShape();
	let fixture: B2Fixture | undefined;
	try {
		shape.m_radius = LIQUIDFUN_SOLID_NODE_RADIUS_METERS;
		fixture = body.CreateFixture(shape, LIQUIDFUN_SOLID_NODE_DENSITY);
		if (!fixture) {
			throw new Error('LiquidFun did not create a Gate 4 solid-node fixture.');
		}
	} finally {
		release(fixture);
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
	try {
		addSolidNodeFixture(module, body);
		return body;
	} catch (error) {
		release(body);
		throw error;
	}
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

	const firstAnchor = firstBody.GetPosition();
	const secondAnchor = secondBody.GetPosition();
	const definition: B2DistanceJointDef = new module.b2DistanceJointDef();
	try {
		definition.Initialize(firstBody, secondBody, firstAnchor, secondAnchor);
		definition.length = bond.restLengthMeters;
		definition.stiffness = stiffness;
		definition.damping = damping;
		definition.collideConnected = false;
		const joint = world.CreateJoint(definition);
		if (!joint) {
			throw new Error('LiquidFun did not create a Gate 4 distance joint.');
		}
		return joint;
	} finally {
		release(firstAnchor);
		release(secondAnchor);
		release(definition);
	}
}

function bodyDistance(firstBody: B2Body, secondBody: B2Body): number {
	const firstPosition = firstBody.GetPosition();
	const secondPosition = secondBody.GetPosition();
	try {
		const first = readVector(firstPosition);
		const second = readVector(secondPosition);
		const distance = Math.hypot(second.x - first.x, second.y - first.y);
		if (!Number.isFinite(distance)) {
			throw new RangeError('A Gate 4 fracture bond length became non-finite.');
		}
		return distance;
	} finally {
		release(firstPosition);
		release(secondPosition);
	}
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
			Math.max(0, bodyDistance(firstBody, secondBody) - bond.restLengthMeters) /
			bond.restLengthMeters;
		if (tensileStrain >= bond.failureStrain) {
			world.DestroyJoint(joint);
			bond.joint = null;
			release(joint);
			continue;
		}
		activeBondCount += 1;
	}
	return activeBondCount;
}

function createStaticWalls(module: LiquidFunModule, world: B2World): number {
	const width = LIQUIDFUN_BOX2D_DOMAIN_METERS.width;
	const height = LIQUIDFUN_BOX2D_DOMAIN_METERS.height;
	const walls = [
		{
			position: { x: -WALL_THICKNESS_METERS, y: height / 2 },
			halfWidth: WALL_THICKNESS_METERS,
			halfHeight: height / 2 + WALL_THICKNESS_METERS
		},
		{
			position: { x: width + WALL_THICKNESS_METERS, y: height / 2 },
			halfWidth: WALL_THICKNESS_METERS,
			halfHeight: height / 2 + WALL_THICKNESS_METERS
		},
		{
			position: { x: width / 2, y: -WALL_THICKNESS_METERS },
			halfWidth: width / 2 + WALL_THICKNESS_METERS,
			halfHeight: WALL_THICKNESS_METERS
		},
		{
			position: { x: width / 2, y: height + WALL_THICKNESS_METERS },
			halfWidth: width / 2 + WALL_THICKNESS_METERS,
			halfHeight: WALL_THICKNESS_METERS
		}
	] as const;

	for (const wall of walls) {
		const body = createBody(
			module,
			world,
			B2_STATIC_BODY,
			wall.position,
			{ x: 0, y: 0 }
		);
		try {
			addBoxFixture(module, body, wall.halfWidth, wall.halfHeight);
		} finally {
			release(body);
		}
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
	pixelsPerMeter: number,
	offsetX: number,
	offsetY: number
): LiquidFunVector {
	return {
		x: offsetX + position.x * pixelsPerMeter,
		y: height - offsetY - position.y * pixelsPerMeter
	};
}

function drawParticles(
	context: CanvasRenderingContext2D,
	particleSystem: B2ParticleSystem,
	particleCount: number,
	waterParticleCount: number,
	width: number,
	height: number,
	pixelsPerMeter: number,
	offsetX: number,
	offsetY: number
): void {
	const positionBuffer = particleSystem.GetPositionBuffer();
	const count = Math.min(particleCount, particleSystem.GetParticleCount());
	const radius = Math.max(1.8, LIQUIDFUN_PARTICLE_RADIUS_METERS * pixelsPerMeter * 0.48);
	try {
		for (let index = 0; index < count; index += 1) {
			const position = readPositionAt(positionBuffer, index);
			const point = pointOnCanvas(
				position,
				width,
				height,
				pixelsPerMeter,
				offsetX,
				offsetY
			);
			context.beginPath();
			context.arc(point.x, point.y, radius, 0, Math.PI * 2);
			context.fillStyle = index < waterParticleCount ? '#64d8ff' : '#ff9e62';
			context.fill();
		}
	} finally {
		releaseUnknown(positionBuffer);
	}
}

function readBodyPosition(body: B2Body): LiquidFunVector {
	const position = body.GetPosition();
	try {
		return readVector(position);
	} finally {
		release(position);
	}
}

function drawPebbleTriangles(
	context: CanvasRenderingContext2D,
	pebbleGroups: readonly LiquidFunTriangleGroupSeed[],
	pebbleNodeBodies: readonly B2Body[],
	width: number,
	height: number,
	pixelsPerMeter: number,
	offsetX: number,
	offsetY: number
): void {
	const points = pebbleNodeBodies.map((body) =>
		pointOnCanvas(
			readBodyPosition(body),
			width,
			height,
			pixelsPerMeter,
			offsetX,
			offsetY
		)
	);
	context.fillStyle = '#f5d77a';
	context.strokeStyle = '#fff0b8';
	context.lineWidth = 1.25;
	for (const { nodeIndices } of pebbleGroups) {
		const first = points[nodeIndices[0]];
		const second = points[nodeIndices[1]];
		const third = points[nodeIndices[2]];
		if (!first || !second || !third) {
			throw new RangeError('A Gate 4 pebble must identify three drawable nodes.');
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

function drawFractureNetwork(
	context: CanvasRenderingContext2D,
	fractureNodeBodies: readonly B2Body[],
	fractureBonds: readonly FractureBondJoint[],
	width: number,
	height: number,
	pixelsPerMeter: number,
	offsetX: number,
	offsetY: number
): void {
	const points = fractureNodeBodies.map((body) =>
		pointOnCanvas(
			readBodyPosition(body),
			width,
			height,
			pixelsPerMeter,
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
	const radius = Math.max(1.5, LIQUIDFUN_SOLID_NODE_RADIUS_METERS * pixelsPerMeter * 0.65);
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
	const gravity = createVector(module, LIQUIDFUN_BOX2D_GRAVITY);
	let world: B2World;
	try {
		world = new module.b2World(gravity);
	} finally {
		release(gravity);
	}

	let particleSystem: B2ParticleSystem | undefined;
	const pebbleNodeBodies: B2Body[] = [];
	const pebbleBondJoints: PebbleBondJoint[] = [];
	const fractureNodeBodies: B2Body[] = [];
	const fractureBondJoints: FractureBondJoint[] = [];
	let disposed = false;
	try {
		const particleSystemDefinition: B2ParticleSystemDef =
			new module.b2ParticleSystemDef();
		particleSystemDefinition.density = LIQUIDFUN_SHARED_PARTICLE_DENSITY;
		particleSystemDefinition.gravityScale = 1;
		particleSystemDefinition.radius = LIQUIDFUN_PARTICLE_RADIUS_METERS;
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
		for (const particle of workload.pebbleNodes) {
			pebbleNodeBodies.push(createSolidNodeBody(module, world, particle));
		}
		for (const bond of workload.pebbleBonds) {
			pebbleBondJoints.push({
				...bond,
				joint: createBondJoint(
					module,
					world,
					pebbleNodeBodies,
					bond,
					LIQUIDFUN_PEBBLE_BOND_STIFFNESS,
					LIQUIDFUN_PEBBLE_BOND_DAMPING
				)
			});
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

		const particleIterations = world.CalculateReasonableParticleIterations(
			LIQUIDFUN_DT_SECONDS
		);
		if (!Number.isSafeInteger(particleIterations) || particleIterations < 1) {
			throw new RangeError('LiquidFun returned an invalid particle iteration count.');
		}
		const activeParticleSystem = particleSystem;

		return {
			particleCount: fluidParticleCount,
			waterParticleCount: workload.waterParticles.length,
			oilParticleCount: workload.oilParticles.length,
			mappedMpmNodeCount: workload.mappedMpmNodeCount,
			pebbleNodeCount: pebbleNodeBodies.length,
			triangleGroupCount: workload.pebbleGroups.length,
			pebbleBondCount: pebbleBondJoints.length,
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
					mapGate4VelocityToBox2D(gravityInGate4Units)
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
				height: number
			): void {
				if (disposed || width <= 0 || height <= 0) return;
				context.clearRect(0, 0, width, height);
				context.fillStyle = '#101a28';
				context.fillRect(0, 0, width, height);

				const pixelsPerMeter = Math.min(
					width / LIQUIDFUN_BOX2D_DOMAIN_METERS.width,
					height / LIQUIDFUN_BOX2D_DOMAIN_METERS.height
				);
				const offsetX =
					(width - LIQUIDFUN_BOX2D_DOMAIN_METERS.width * pixelsPerMeter) / 2;
				const offsetY =
					(height - LIQUIDFUN_BOX2D_DOMAIN_METERS.height * pixelsPerMeter) / 2;

				drawParticles(
					context,
					activeParticleSystem,
					workload.waterParticles.length + workload.oilParticles.length,
					workload.waterParticles.length,
					width,
					height,
					pixelsPerMeter,
					offsetX,
					offsetY
				);
				drawPebbleTriangles(
					context,
					workload.pebbleGroups,
					pebbleNodeBodies,
					width,
					height,
					pixelsPerMeter,
					offsetX,
					offsetY
				);
				drawFractureNetwork(
					context,
					fractureNodeBodies,
					fractureBondJoints,
					width,
					height,
					pixelsPerMeter,
					offsetX,
					offsetY
				);

				context.strokeStyle = '#d6e3f1';
				context.lineWidth = 1;
				context.strokeRect(
					offsetX,
					offsetY,
					LIQUIDFUN_BOX2D_DOMAIN_METERS.width * pixelsPerMeter,
					LIQUIDFUN_BOX2D_DOMAIN_METERS.height * pixelsPerMeter
				);
			},
			dispose(): void {
				if (disposed) return;
				disposed = true;
				for (const { joint } of pebbleBondJoints) release(joint);
				pebbleBondJoints.length = 0;
				for (const bond of fractureBondJoints) {
					release(bond.joint);
					bond.joint = null;
				}
				fractureBondJoints.length = 0;
				for (const body of pebbleNodeBodies) release(body);
				pebbleNodeBodies.length = 0;
				for (const body of fractureNodeBodies) release(body);
				fractureNodeBodies.length = 0;
				release(activeParticleSystem);
				release(world);
			}
		};
	} catch (error) {
		for (const { joint } of pebbleBondJoints) release(joint);
		for (const bond of fractureBondJoints) release(bond.joint);
		for (const body of pebbleNodeBodies) release(body);
		for (const body of fractureNodeBodies) release(body);
		release(particleSystem);
		release(world);
		throw error;
	}
}

export const LIQUIDFUN_REALTIME_CONFIG = {
	dtSeconds: LIQUIDFUN_DT_SECONDS,
	gate4Gravity: LIQUIDFUN_GATE4_GRAVITY,
	gate4UnitsPerMeter: LIQUIDFUN_GATE4_UNITS_PER_METER,
	domainGate4Units: LIQUIDFUN_GATE4_DOMAIN,
	particleRadiusMeters: LIQUIDFUN_PARTICLE_RADIUS_METERS,
	velocityIterations: LIQUIDFUN_VELOCITY_ITERATIONS,
	positionIterations: LIQUIDFUN_POSITION_ITERATIONS
} as const;
