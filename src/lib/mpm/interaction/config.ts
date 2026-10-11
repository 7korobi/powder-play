import type { MpmLabConfig, MpmParticleSeed, MpmVector } from '#lib/mpm/contracts.js';

/** Keep the interaction PoC small and within WebGPU's guaranteed buffer limits. */
export const MAX_INTERACTION_PARTICLES = 65_536;

const MAX_WORLD_EXTENT = 1_000_000;
const MAX_INITIAL_VELOCITY = 1_000;
const MAX_ID = 0xffff_ffff;

export interface MpmObjectRange {
	objectId: number;
	start: number;
	count: number;
}

/**
 * Temporary initialization data. The backend uploads the particle arrays once and does not retain
 * their CPU-side positions or velocities.
 */
export interface MpmInteractionLayout {
	worldWidth: number;
	worldHeight: number;
	capacity: number;
	particleCount: number;
	particleState: Float32Array;
	materialIds: Uint32Array;
	objects: MpmObjectRange[];
}

interface ValidatedParticle {
	objectId: number;
	materialId: number;
	position: MpmVector;
	velocity: MpmVector;
}

/**
 * Validates the public config and packs each object into a contiguous GPU range.
 * The returned position/velocity arrays are only intended for the one-time initial upload.
 */
export function buildMpmInteractionLayout(config: MpmLabConfig): MpmInteractionLayout {
	if (typeof config !== 'object' || config === null) {
		throw new TypeError('MPM config must be an object.');
	}

	const worldWidth = validateWorldExtent(config.worldWidth, 'worldWidth');
	const worldHeight = validateWorldExtent(config.worldHeight, 'worldHeight');
	if (!Number.isSafeInteger(config.capacity) || config.capacity < 1) {
		throw new RangeError('capacity must be a positive safe integer.');
	}
	if (config.capacity > MAX_INTERACTION_PARTICLES) {
		throw new RangeError(`capacity must not exceed ${MAX_INTERACTION_PARTICLES}.`);
	}
	if (!Array.isArray(config.particles)) {
		throw new TypeError('particles must be an array.');
	}
	if (config.particles.length > config.capacity) {
		throw new RangeError('particle count must not exceed capacity.');
	}

	const grouped = new Map<number, ValidatedParticle[]>();
	for (const [index, seed] of config.particles.entries()) {
		const validated = validateSeed(seed, index, worldWidth, worldHeight);
		const group = grouped.get(validated.objectId);
		if (group) {
			group.push(validated);
		} else {
			grouped.set(validated.objectId, [validated]);
		}
	}

	const particleCount = config.particles.length;
	const particleState = new Float32Array(particleCount * 4);
	const materialIds = new Uint32Array(particleCount);
	const objects: MpmObjectRange[] = [];
	let particleIndex = 0;

	for (const [objectId, particles] of grouped) {
		const start = particleIndex;
		for (const particle of particles) {
			const stateOffset = particleIndex * 4;
			particleState[stateOffset] = particle.position.x;
			particleState[stateOffset + 1] = particle.position.y;
			particleState[stateOffset + 2] = particle.velocity.x;
			particleState[stateOffset + 3] = particle.velocity.y;
			materialIds[particleIndex] = particle.materialId;
			particleIndex++;
		}
		objects.push({ objectId, start, count: particleIndex - start });
	}

	return {
		worldWidth,
		worldHeight,
		capacity: config.capacity,
		particleCount,
		particleState,
		materialIds,
		objects
	};
}

function validateWorldExtent(value: number, name: string): number {
	const extent = toFiniteFloat32(value, name);
	if (extent <= 0 || extent > MAX_WORLD_EXTENT) {
		throw new RangeError(`${name} must be greater than 0 and at most ${MAX_WORLD_EXTENT}.`);
	}
	return extent;
}

function validateSeed(
	seed: MpmParticleSeed,
	index: number,
	worldWidth: number,
	worldHeight: number
): ValidatedParticle {
	if (typeof seed !== 'object' || seed === null) {
		throw new TypeError(`particles[${index}] must be an object.`);
	}

	const position = validateVector(seed.position, `particles[${index}].position`);
	if (
		position.x < 0 ||
		position.x > worldWidth ||
		position.y < 0 ||
		position.y > worldHeight
	) {
		throw new RangeError(`particles[${index}].position must be inside the world bounds.`);
	}

	const velocity =
		seed.velocity === undefined
			? { x: 0, y: 0 }
			: validateVector(seed.velocity, `particles[${index}].velocity`);
	if (
		Math.abs(velocity.x) > MAX_INITIAL_VELOCITY ||
		Math.abs(velocity.y) > MAX_INITIAL_VELOCITY
	) {
		throw new RangeError(
			`particles[${index}].velocity components must be within ±${MAX_INITIAL_VELOCITY}.`
		);
	}

	return {
		objectId: validateId(seed.objectId, `particles[${index}].objectId`),
		materialId: validateId(seed.materialId, `particles[${index}].materialId`),
		position,
		velocity
	};
}

function validateVector(vector: MpmVector, name: string): MpmVector {
	if (typeof vector !== 'object' || vector === null) {
		throw new TypeError(`${name} must be a vector.`);
	}
	return {
		x: toFiniteFloat32(vector.x, `${name}.x`),
		y: toFiniteFloat32(vector.y, `${name}.y`)
	};
}

function toFiniteFloat32(value: number, name: string): number {
	if (!Number.isFinite(value)) {
		throw new TypeError(`${name} must be finite.`);
	}
	const rounded = Math.fround(value);
	if (!Number.isFinite(rounded)) {
		throw new RangeError(`${name} is outside the finite float32 range.`);
	}
	return rounded;
}

function validateId(value: number, name: string): number {
	if (!Number.isSafeInteger(value) || value < 0 || value > MAX_ID) {
		throw new RangeError(`${name} must be an unsigned 32-bit integer.`);
	}
	return value;
}
