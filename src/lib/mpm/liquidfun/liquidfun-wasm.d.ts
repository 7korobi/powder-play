declare module 'liquidfun-wasm' {
	export interface B2Vec2 {
		x: number;
		y: number;
		delete?(): void;
	}

	export interface B2BodyDef {
		type: number;
		position: B2Vec2;
		angle: number;
		linearVelocity: B2Vec2;
		angularVelocity: number;
		delete?(): void;
	}

	export interface B2Fixture {
		delete?(): void;
	}

	export interface B2Shape {
		m_radius: number;
		delete?(): void;
	}

	export interface B2Body {
		CreateFixture(shape: B2Shape, density: number): B2Fixture;
		GetPosition(): B2Vec2;
		GetAngle(): number;
		delete?(): void;
	}

	export interface B2PolygonShape extends B2Shape {
		SetAsBox(halfWidth: number, halfHeight: number): void;
		Set?(vertices: readonly B2Vec2[], count: number): void;
		SetAsArray?(vertices: readonly B2Vec2[], count: number): void;
	}

	export interface B2CircleShape extends B2Shape {}

	export interface B2Joint {
		delete?(): void;
	}

	export interface B2DistanceJointDef {
		Initialize(
			bodyA: B2Body,
			bodyB: B2Body,
			anchorA: B2Vec2,
			anchorB: B2Vec2
		): void;
		length: number;
		stiffness: number;
		damping: number;
		collideConnected: boolean;
		delete?(): void;
	}

	export interface B2ParticleDef {
		flags: number;
		position: B2Vec2;
		velocity: B2Vec2;
		delete?(): void;
	}

	export interface B2ParticleSystemDef {
		density: number;
		gravityScale: number;
		radius: number;
		delete?(): void;
	}

	export interface B2ParticleSystem {
		CreateParticle(definition: B2ParticleDef): number;
		GetParticleCount(): number;
		GetParticleFlags(index: number): number;
		GetPositionBuffer(): B2Vec2;
		delete?(): void;
	}

	export interface B2World {
		CreateBody(definition: B2BodyDef): B2Body;
		CreateParticleSystem(definition: B2ParticleSystemDef): B2ParticleSystem;
		CreateJoint(definition: B2DistanceJointDef): B2Joint;
		DestroyJoint(joint: B2Joint): void;
		CalculateReasonableParticleIterations(timeStep: number): number;
		SetGravity(gravity: B2Vec2): void;
		Step(
			timeStep: number,
			velocityIterations: number,
			positionIterations: number,
			particleIterations: number
		): void;
		delete?(): void;
	}

	export interface LiquidFunModule {
		b2Vec2: new (x?: number, y?: number) => B2Vec2;
		b2World: new (gravity: B2Vec2) => B2World;
		b2BodyDef: new () => B2BodyDef;
		b2PolygonShape: new () => B2PolygonShape;
		b2CircleShape: new () => B2CircleShape;
		b2DistanceJointDef: new () => B2DistanceJointDef;
		b2ParticleDef: new () => B2ParticleDef;
		b2ParticleSystemDef: new () => B2ParticleSystemDef;
		HEAPF32: Float32Array;
		getPointer(value: B2Vec2): number;
		b2_waterParticle: number;
		b2_viscousParticle: number;
	}

	export type LiquidFunFactory = () => Promise<LiquidFunModule>;
	const createLiquidFunModule: LiquidFunFactory;
	export default createLiquidFunModule;
}
