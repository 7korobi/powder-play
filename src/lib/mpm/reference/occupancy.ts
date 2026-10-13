import {
	normalizeMpmConstitutiveModel,
	type MpmConstitutiveModel
} from '../materials.js';
import type {
	MpmGridParameters,
	MpmMatrix2,
	MpmParticle,
	MpmStepConfig,
	MpmVector2
} from './solver.js';

export type MpmCurrentAreaConfig = Pick<
	MpmStepConfig,
	'grid' | 'particleVolume' | 'material'
>;

export interface MpmCurrentAreaOccupancyNode {
	readonly xIndex: number;
	readonly yIndex: number;
	readonly isHalo: boolean;
	/** Water current area transferred to this node, divided by cell area. */
	readonly water: number;
	/** Oil current area transferred to this node, divided by cell area. */
	readonly oil: number;
}

export interface MpmCurrentAreaOccupancy {
	readonly grid: MpmGridParameters;
	/**
	 * Sorted by y then x. Contains the complete 3x3 stencil for each liquid
	 * particle, including halo nodes; untouched nodes are omitted.
	 */
	readonly nodes: readonly MpmCurrentAreaOccupancyNode[];
}

interface ValidatedConfig {
	readonly grid: MpmGridParameters;
	readonly particleVolume: number;
	readonly defaultMaterial: MpmConstitutiveModel;
	readonly domainWidth: number;
	readonly domainHeight: number;
	readonly cellArea: number;
}

interface LiquidParticle {
	readonly position: MpmVector2;
	readonly currentArea: number;
	readonly phase: 'water' | 'oil';
}

interface MutableOccupancyNode {
	readonly xIndex: number;
	readonly yIndex: number;
	water: number;
	oil: number;
}

interface QuadraticStencil {
	readonly baseX: number;
	readonly baseY: number;
	readonly weightsX: readonly [number, number, number];
	readonly weightsY: readonly [number, number, number];
}

/**
 * Deterministic CPU reference for material-separated current-area occupancy.
 * This helper is intentionally independent of the live MPM solver.
 */
export function gatherCurrentArea(
	particles: readonly MpmParticle[],
	config: MpmCurrentAreaConfig
): MpmCurrentAreaOccupancy {
	const validated = validateConfig(config);
	const liquidParticles = validateParticles(particles, validated);
	const occupancy = new Map<string, MutableOccupancyNode>();

	for (const particle of liquidParticles) {
		const stencil = particleStencil(particle.position, validated.grid.cellSize);
		for (let oy = 0; oy < 3; oy += 1) {
			for (let ox = 0; ox < 3; ox += 1) {
				const weight = stencil.weightsX[ox] * stencil.weightsY[oy];
				assertWeight(weight);

				const xIndex = stencil.baseX + ox;
				const yIndex = stencil.baseY + oy;
				if (!Number.isSafeInteger(xIndex) || !Number.isSafeInteger(yIndex)) {
					throw new RangeError('particle stencil node indices must be safe integers');
				}

				const key = gridNodeKey(xIndex, yIndex);
				let node = occupancy.get(key);
				if (!node) {
					node = { xIndex, yIndex, water: 0, oil: 0 };
					occupancy.set(key, node);
				}

				const weightedArea = weight * particle.currentArea;
				assertComputedFinite(weightedArea, `${particle.phase} weighted current area`);
				const contribution = weightedArea / validated.cellArea;
				assertComputedFinite(contribution, `${particle.phase} occupancy`);
				node[particle.phase] += contribution;
				assertComputedFinite(node[particle.phase], `${particle.phase} node occupancy`);
			}
		}
	}

	const nodes = [...occupancy.values()]
		.map((node): MpmCurrentAreaOccupancyNode => ({
			...node,
			isHalo:
				node.xIndex < 0 ||
				node.xIndex > validated.grid.cellsX ||
				node.yIndex < 0 ||
				node.yIndex > validated.grid.cellsY
		}))
		.sort(
			(left, right) =>
				compareNumbers(left.yIndex, right.yIndex) ||
				compareNumbers(left.xIndex, right.xIndex)
		);

	return { grid: validated.grid, nodes };
}

function validateConfig(config: unknown): ValidatedConfig {
	if (!isRecord(config)) throw new TypeError('config must be an object');
	if (!isRecord(config.grid)) throw new TypeError('grid must be an object');

	const cellsX = config.grid.cellsX;
	const cellsY = config.grid.cellsY;
	const cellSize = config.grid.cellSize;
	const particleVolume = config.particleVolume;
	const defaultMaterial = validateMaterial(config.material, 'material');

	assertFiniteNumber(cellsX, 'grid.cellsX');
	assertFiniteNumber(cellsY, 'grid.cellsY');
	assertFiniteNumber(cellSize, 'grid.cellSize');
	assertFiniteNumber(particleVolume, 'particleVolume');

	if (!Number.isSafeInteger(cellsX) || cellsX < 1 || cellsX >= Number.MAX_SAFE_INTEGER) {
		throw new RangeError('grid.cellsX must be a positive safe integer');
	}
	if (!Number.isSafeInteger(cellsY) || cellsY < 1 || cellsY >= Number.MAX_SAFE_INTEGER) {
		throw new RangeError('grid.cellsY must be a positive safe integer');
	}
	if (cellSize <= 0) throw new RangeError('grid.cellSize must be greater than zero');
	if (particleVolume <= 0) {
		throw new RangeError('particleVolume must be greater than zero');
	}

	const domainWidth = cellsX * cellSize;
	const domainHeight = cellsY * cellSize;
	const cellArea = cellSize * cellSize;
	if (!Number.isFinite(domainWidth) || !Number.isFinite(domainHeight)) {
		throw new RangeError('grid dimensions produce a non-finite domain');
	}
	if (
		!Number.isFinite((cellsX + 1) * cellSize) ||
		!Number.isFinite((cellsY + 1) * cellSize)
	) {
		throw new RangeError('grid dimensions produce non-finite halo coordinates');
	}
	if (!Number.isFinite(cellArea) || cellArea <= 0 || !Number.isFinite(1 / cellArea)) {
		throw new RangeError('grid.cellSize is outside the supported numeric range');
	}

	return {
		grid: { cellsX, cellsY, cellSize },
		particleVolume,
		defaultMaterial,
		domainWidth,
		domainHeight,
		cellArea
	};
}

function validateParticles(
	particles: unknown,
	config: ValidatedConfig
): readonly LiquidParticle[] {
	if (!Array.isArray(particles)) throw new TypeError('particles must be an array');

	const liquidParticles: LiquidParticle[] = [];
	for (let index = 0; index < particles.length; index += 1) {
		const value: unknown = particles[index];
		if (!isRecord(value)) throw new TypeError(`particles[${index}] must be an object`);
		assertVector(value.position, `particles[${index}].position`);
		assertVector(value.velocity, `particles[${index}].velocity`);
		assertMatrix(value.C, `particles[${index}].C`);
		assertMatrix(value.F, `particles[${index}].F`);

		const phase =
			value.material === undefined
				? config.defaultMaterial
				: validateMaterial(value.material, `particles[${index}].material`);
		let volume = config.particleVolume;
		if (value.volume !== undefined) {
			assertFiniteNumber(value.volume, `particles[${index}].volume`);
			if (value.volume <= 0) {
				throw new RangeError(`particles[${index}].volume must be greater than zero`);
			}
			volume = value.volume;
		}

		const position = value.position as MpmVector2;
		const deformation = value.F as MpmMatrix2;
		if (
			position.x < 0 ||
			position.x > config.domainWidth ||
			position.y < 0 ||
			position.y > config.domainHeight
		) {
			throw new RangeError(`particles[${index}].position must be inside the grid domain`);
		}

		const jacobian = determinant(deformation);
		if (!Number.isFinite(jacobian) || jacobian <= 0) {
			throw new RangeError(
				`particles[${index}].F must have a finite positive determinant`
			);
		}

		if (phase !== 'water' && phase !== 'oil') continue;
		const currentArea = volume * jacobian;
		assertComputedFinite(currentArea, `particles[${index}] current area`);
		if (currentArea <= 0) {
			throw new RangeError(`particles[${index}] current area must be greater than zero`);
		}
		liquidParticles.push({
			position: { x: position.x, y: position.y },
			currentArea,
			phase
		});
	}

	return liquidParticles;
}

function validateMaterial(value: unknown, name: string): MpmConstitutiveModel {
	if (!isRecord(value)) throw new TypeError(`${name} must be an object`);
	const lambda = value.lambda;
	const mu = value.mu;
	assertFiniteNumber(lambda, `${name}.lambda`);
	assertFiniteNumber(mu, `${name}.mu`);
	const model = normalizeMpmConstitutiveModel(value.model, `${name}.model`);
	if (!Number.isFinite(lambda + 2 * mu)) {
		throw new RangeError(`${name} parameters are outside the supported numeric range`);
	}

	if (model === 'water' || model === 'oil') {
		const density = value.density;
		assertFiniteNumber(density, `${name}.density`);
		if (lambda <= 0) throw new RangeError(`${name}.lambda must be greater than zero`);
		if (mu !== 0) throw new RangeError(`${name}.mu must be zero for a liquid material`);
		if (value.frictionCoefficient !== undefined) {
			throw new RangeError(`${name}.frictionCoefficient is only valid for powder`);
		}
		if (density <= 0) throw new RangeError(`${name}.density must be greater than zero`);
		return model;
	}

	if (model === 'powder') {
		const frictionCoefficient = value.frictionCoefficient;
		assertFiniteNumber(frictionCoefficient, `${name}.frictionCoefficient`);
		if (lambda < 0) throw new RangeError(`${name}.lambda must be non-negative`);
		if (mu <= 0) throw new RangeError(`${name}.mu must be greater than zero`);
		if (frictionCoefficient < 0) {
			throw new RangeError(`${name}.frictionCoefficient must be non-negative`);
		}
		if (value.density !== undefined) {
			throw new RangeError(`${name}.density is only valid for water or oil`);
		}
		return model;
	}

	if (value.density !== undefined || value.frictionCoefficient !== undefined) {
		throw new RangeError(`${name} has parameters that do not match its constitutive model`);
	}
	if (lambda < 0) throw new RangeError(`${name}.lambda must be non-negative`);
	if (mu <= 0) throw new RangeError(`${name}.mu must be greater than zero`);
	return model;
}

function particleStencil(position: MpmVector2, cellSize: number): QuadraticStencil {
	const gridX = position.x / cellSize;
	const gridY = position.y / cellSize;
	const baseX = Math.floor(gridX - 0.5);
	const baseY = Math.floor(gridY - 0.5);
	return {
		baseX,
		baseY,
		weightsX: quadraticWeights(gridX - baseX),
		weightsY: quadraticWeights(gridY - baseY)
	};
}

function quadraticWeights(fraction: number): readonly [number, number, number] {
	return [
		0.5 * (1.5 - fraction) ** 2,
		0.75 - (fraction - 1) ** 2,
		0.5 * (fraction - 0.5) ** 2
	];
}

function determinant(matrix: MpmMatrix2): number {
	return matrix.m00 * matrix.m11 - matrix.m01 * matrix.m10;
}

function gridNodeKey(xIndex: number, yIndex: number): string {
	return `${xIndex},${yIndex}`;
}

function compareNumbers(left: number, right: number): number {
	return left === right ? 0 : left < right ? -1 : 1;
}

function assertWeight(weight: number): void {
	if (!Number.isFinite(weight) || weight < 0) {
		throw new RangeError('quadratic B-spline weights must be finite and non-negative');
	}
}

function assertFiniteNumber(value: unknown, name: string): asserts value is number {
	if (typeof value !== 'number' || !Number.isFinite(value)) {
		throw new TypeError(`${name} must be a finite number`);
	}
}

function assertVector(value: unknown, name: string): asserts value is MpmVector2 {
	if (!isRecord(value)) throw new TypeError(`${name} must be a vector`);
	assertFiniteNumber(value.x, `${name}.x`);
	assertFiniteNumber(value.y, `${name}.y`);
}

function assertMatrix(value: unknown, name: string): asserts value is MpmMatrix2 {
	if (!isRecord(value)) throw new TypeError(`${name} must be a matrix`);
	assertFiniteNumber(value.m00, `${name}.m00`);
	assertFiniteNumber(value.m01, `${name}.m01`);
	assertFiniteNumber(value.m10, `${name}.m10`);
	assertFiniteNumber(value.m11, `${name}.m11`);
}

function assertComputedFinite(value: number, name: string): void {
	if (!Number.isFinite(value)) throw new RangeError(`${name} became non-finite`);
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === 'object' && value !== null && !Array.isArray(value);
}
