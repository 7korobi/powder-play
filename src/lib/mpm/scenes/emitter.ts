import type {
	MpmGridParameters,
	MpmMaterialParameters,
	MpmParticle,
	MpmVector2
} from '../reference/solver.js';

export const MAX_MPM_EMITTER_PARTICLES = 100_000;

export type MpmEmitterMaterial = 'water' | 'oil' | 'neo-hookean';
export type MpmEmitterShape = 'rectangle' | 'circle';

export interface MpmEmitterConfig {
	readonly count: number;
	readonly seed: number;
	readonly center: MpmVector2;
	readonly shape: MpmEmitterShape;
	readonly spacing: number;
	readonly velocity: MpmVector2;
	readonly material: MpmEmitterMaterial;
	readonly particleVolume?: number;
	readonly grid: MpmGridParameters;
}

const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));
const ZERO_MATRIX = { m00: 0, m01: 0, m10: 0, m11: 0 } as const;
const IDENTITY = { m00: 1, m01: 0, m10: 0, m11: 1 } as const;
const MATERIALS = {
	water: {
		lambda: 40_000,
		mu: 0,
		model: 'water',
		density: 0.02
	},
	oil: {
		lambda: 30_000,
		mu: 0,
		model: 'oil',
		density: 0.016
	},
	'neo-hookean': {
		lambda: 20,
		mu: 10,
		model: 'neo-hookean'
	}
} as const satisfies Record<MpmEmitterMaterial, MpmMaterialParameters>;

export function createMpmEmitterParticles(config: MpmEmitterConfig): readonly MpmParticle[] {
	const count = positiveInteger(config.count, 'count');
	if (count > MAX_MPM_EMITTER_PARTICLES) {
		throw new RangeError(`count must not exceed ${MAX_MPM_EMITTER_PARTICLES}`);
	}
	if (!Number.isSafeInteger(config.seed) || config.seed < 0 || config.seed > 0xffff_ffff) {
		throw new RangeError('seed must be an unsigned 32-bit integer');
	}
	if (config.shape !== 'rectangle' && config.shape !== 'circle') {
		throw new RangeError('shape must be rectangle or circle');
	}
	if (
		config.material !== 'water' &&
		config.material !== 'oil' &&
		config.material !== 'neo-hookean'
	) {
		throw new RangeError('material must be water, oil, or neo-hookean');
	}
	const center = readVector(config.center, 'center');
	const velocity = readVector(config.velocity, 'velocity');
	const grid = validateGrid(config.grid);
	const spacing = finitePositive(config.spacing, 'spacing');
	const particleVolume = finitePositive(config.particleVolume ?? 25, 'particleVolume');
	const width = grid.cellsX * grid.cellSize;
	const height = grid.cellsY * grid.cellSize;
	if (center.x < 0 || center.x > width || center.y < 0 || center.y > height) {
		throw new RangeError('center must be inside the simulation domain');
	}

	const random = createSeededRandom(config.seed);
	const columns = Math.ceil(Math.sqrt(count));
	const rows = Math.ceil(count / columns);
	const radius = spacing * Math.sqrt(count / Math.PI);
	const angularOffset = random() * Math.PI * 2;
	const particles: MpmParticle[] = [];
	for (let index = 0; index < count; index += 1) {
		let position: MpmVector2;
		if (config.shape === 'circle') {
			const radialFraction = Math.sqrt((index + 0.5) / count);
			const angle = angularOffset + index * GOLDEN_ANGLE;
			const jitter = (random() - 0.5) * spacing * 0.12;
			const distance = Math.max(0, radialFraction * radius + jitter);
			position = {
				x: center.x + Math.cos(angle) * distance,
				y: center.y + Math.sin(angle) * distance
			};
		} else {
			const column = index % columns;
			const row = Math.floor(index / columns);
			position = {
				x:
					center.x +
					(column - (columns - 1) * 0.5) * spacing +
					(random() - 0.5) * spacing * 0.12,
				y:
					center.y +
					(row - (rows - 1) * 0.5) * spacing +
					(random() - 0.5) * spacing * 0.12
			};
		}
		if (position.x < 0 || position.x > width || position.y < 0 || position.y > height) {
			throw new RangeError(`emitted particle ${index} would be outside the simulation domain`);
		}
		particles.push({
			position,
			velocity,
			C: ZERO_MATRIX,
			F: IDENTITY,
			plasticVolume: 1,
			volume: particleVolume,
			material: MATERIALS[config.material]
		});
	}
	return particles;
}

function validateGrid(value: MpmGridParameters): MpmGridParameters {
	if (typeof value !== 'object' || value === null) {
		throw new TypeError('grid must be an object');
	}
	if (
		!Number.isSafeInteger(value.cellsX) ||
		value.cellsX < 1 ||
		!Number.isSafeInteger(value.cellsY) ||
		value.cellsY < 1
	) {
		throw new RangeError('grid cell counts must be positive safe integers');
	}
	const cellSize = finitePositive(value.cellSize, 'grid.cellSize');
	return { cellsX: value.cellsX, cellsY: value.cellsY, cellSize };
}

function readVector(value: MpmVector2, name: string): MpmVector2 {
	if (typeof value !== 'object' || value === null) {
		throw new TypeError(`${name} must be a vector`);
	}
	if (!Number.isFinite(value.x) || !Number.isFinite(value.y)) {
		throw new RangeError(`${name} components must be finite`);
	}
	return { x: value.x, y: value.y };
}

function positiveInteger(value: number, name: string): number {
	if (!Number.isSafeInteger(value) || value < 1) {
		throw new RangeError(`${name} must be a positive safe integer`);
	}
	return value;
}

function finitePositive(value: number, name: string): number {
	if (!Number.isFinite(value) || value <= 0) {
		throw new RangeError(`${name} must be a positive finite number`);
	}
	return value;
}

function createSeededRandom(seed: number): () => number {
	let state = seed >>> 0;
	return () => {
		state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
		return state / 0x1_0000_0000;
	};
}
