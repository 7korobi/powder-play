import type {
	B2Body,
	B2BodyDef,
	B2Fixture,
	B2ParticleDef,
	B2ParticleSystem,
	B2ParticleSystemDef,
	B2PolygonShape,
	B2Vec2,
	B2World,
	LiquidFunModule
} from 'liquidfun-wasm';
import type { MpmVector2 } from '../reference/solver.js';
import {
	createLiquidFunGate4Workload,
	LIQUIDFUN_BOX2D_DOMAIN_METERS,
	LIQUIDFUN_BOX2D_GRAVITY,
	LIQUIDFUN_DT_SECONDS,
	LIQUIDFUN_GATE4_DOMAIN,
	LIQUIDFUN_GATE4_GRAVITY,
	LIQUIDFUN_GATE4_UNITS_PER_METER,
	LIQUIDFUN_PARTICLE_RADIUS_METERS,
	LIQUIDFUN_POSITION_ITERATIONS,
	LIQUIDFUN_SHARED_PARTICLE_DENSITY,
	LIQUIDFUN_VELOCITY_ITERATIONS,
	mapGate4PositionToBox2D,
	mapGate4VelocityToBox2D,
	type LiquidFunTriangleBodySeed,
	type LiquidFunVector
} from './workload.js';

const B2_STATIC_BODY = 0;
const B2_DYNAMIC_BODY = 2;
const WALL_THICKNESS_METERS = 0.1;

export interface LiquidFunSimulation {
	readonly particleCount: number;
	readonly waterParticleCount: number;
	readonly oilParticleCount: number;
	readonly triangleBodyCount: number;
	readonly staticWallBodyCount: number;
	readonly sourceMpmNodeCount: number;
	readonly omittedFractureNodeCount: number;
	readonly particleIterations: number;
	step(): void;
	setGravityFromGate4(gravity: MpmVector2): void;
	draw(context: CanvasRenderingContext2D, width: number, height: number): void;
	dispose(): void;
}

interface TriangleBody {
	readonly body: B2Body;
	readonly localVertices: readonly LiquidFunVector[];
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

function addTriangleFixture(
	module: LiquidFunModule,
	body: B2Body,
	triangle: LiquidFunTriangleBodySeed
): void {
	const shape: B2PolygonShape = new module.b2PolygonShape();
	const vertices = triangle.localVertices.map((vertex) => createVector(module, vertex));
	let fixture: B2Fixture | undefined;
	try {
		const setVertices = shape.Set ?? shape.SetAsArray;
		if (!setVertices) {
			throw new Error('LiquidFun b2PolygonShape does not expose a polygon setter.');
		}
		setVertices.call(shape, vertices, vertices.length);
		fixture = body.CreateFixture(shape, triangle.density);
	} finally {
		release(fixture);
		for (const vertex of vertices) release(vertex);
		release(shape);
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

function drawTriangles(
	context: CanvasRenderingContext2D,
	triangleBodies: readonly TriangleBody[],
	width: number,
	height: number,
	pixelsPerMeter: number,
	offsetX: number,
	offsetY: number
): void {
	context.fillStyle = '#f5d77a';
	context.strokeStyle = '#fff0b8';
	context.lineWidth = 1.25;
	for (const { body, localVertices } of triangleBodies) {
		const bodyPosition = body.GetPosition();
		try {
			const angle = body.GetAngle();
			const cosine = Math.cos(angle);
			const sine = Math.sin(angle);
			const points = localVertices.map((vertex) => {
				const position = {
					x:
						bodyPosition.x +
						cosine * vertex.x -
						sine * vertex.y,
					y:
						bodyPosition.y +
						sine * vertex.x +
						cosine * vertex.y
				};
				return pointOnCanvas(
					position,
					width,
					height,
					pixelsPerMeter,
					offsetX,
					offsetY
				);
			});
			context.beginPath();
			context.moveTo(points[0].x, points[0].y);
			for (let index = 1; index < points.length; index += 1) {
				context.lineTo(points[index].x, points[index].y);
			}
			context.closePath();
			context.fill();
			context.stroke();
		} finally {
			release(bodyPosition);
		}
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
	const triangleBodies: TriangleBody[] = [];
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

		const staticWallBodyCount = createStaticWalls(module, world);
		for (const triangle of workload.triangleBodies) {
			const body = createBody(
				module,
				world,
				B2_DYNAMIC_BODY,
				triangle.position,
				triangle.linearVelocity,
				triangle.angularVelocity
			);
			try {
				addTriangleFixture(module, body, triangle);
			} catch (error) {
				release(body);
				throw error;
			}
			triangleBodies.push({ body, localVertices: triangle.localVertices });
		}

		const particleIterations = world.CalculateReasonableParticleIterations(
			LIQUIDFUN_DT_SECONDS
		);
		if (!Number.isSafeInteger(particleIterations) || particleIterations < 1) {
			throw new RangeError('LiquidFun returned an invalid particle iteration count.');
		}
		const activeParticleSystem = particleSystem;

		return {
			particleCount:
				workload.waterParticles.length + workload.oilParticles.length,
			waterParticleCount: workload.waterParticles.length,
			oilParticleCount: workload.oilParticles.length,
			triangleBodyCount: triangleBodies.length,
			staticWallBodyCount,
			sourceMpmNodeCount: workload.sourceMpmNodeCount,
			omittedFractureNodeCount: workload.omittedFractureNodeCount,
			particleIterations,
			step(): void {
				if (disposed) throw new Error('The LiquidFun simulation has been disposed.');
				world.Step(
					LIQUIDFUN_DT_SECONDS,
					LIQUIDFUN_VELOCITY_ITERATIONS,
					LIQUIDFUN_POSITION_ITERATIONS,
					particleIterations
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
				drawTriangles(
					context,
					triangleBodies,
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
				for (const { body } of triangleBodies) release(body);
				triangleBodies.length = 0;
				release(activeParticleSystem);
				release(world);
			}
		};
	} catch (error) {
		for (const { body } of triangleBodies) release(body);
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
