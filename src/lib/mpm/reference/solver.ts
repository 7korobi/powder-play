import {
	normalizeMpmConstitutiveModel,
	SNOW_PLASTICITY
} from '../materials.js';

export interface MpmVector2 {
	readonly x: number;
	readonly y: number;
}

export interface MpmMatrix2 {
	readonly m00: number;
	readonly m01: number;
	readonly m10: number;
	readonly m11: number;
}

export interface MpmParticle {
	readonly position: MpmVector2;
	readonly velocity: MpmVector2;
	readonly C: MpmMatrix2;
	readonly F: MpmMatrix2;
	readonly objectId?: number;
	readonly bondedTriangleVertex?: 0 | 1 | 2;
	readonly plasticVolume?: number;
	readonly volume?: number;
	readonly material?: MpmMaterialParameters;
}

export interface MpmGridParameters {
	readonly cellsX: number;
	readonly cellsY: number;
	readonly cellSize: number;
}

export type MpmMaterialParameters =
	| {
			readonly lambda: number;
			readonly mu: number;
			readonly model?: 'neo-hookean';
	  }
	| {
			readonly lambda: number;
			readonly mu: number;
			readonly model: 'snow';
	  }
	| {
			readonly lambda: number;
			readonly mu: 0;
			readonly model: 'water' | 'oil';
			readonly density: number;
	  }
	| {
			readonly lambda: number;
			readonly mu: number;
			readonly model: 'powder';
			readonly frictionCoefficient: number;
	  };

export interface MpmStepConfig {
	readonly dt: number;
	readonly grid: MpmGridParameters;
	readonly particleMass: number;
	readonly particleVolume: number;
	readonly bondRestLength?: number;
	readonly bondStiffness?: number;
	readonly material: MpmMaterialParameters;
	readonly gravity: MpmVector2;
}

export interface MpmStepDiagnostics {
	readonly inputMass: number;
	readonly gridMass: number;
	readonly outputMass: number;
	readonly inputMomentum: MpmVector2;
	readonly gridMomentum: MpmVector2;
	readonly outputMomentum: MpmVector2;
}

export interface MpmStepResult {
	readonly particles: readonly MpmParticle[];
	readonly diagnostics: MpmStepDiagnostics;
}

interface ValidatedConfig {
	readonly dt: number;
	readonly grid: MpmGridParameters;
	readonly particleMass: number;
	readonly particleVolume: number;
	readonly bondRestLength: number | undefined;
	readonly bondStiffness: number;
	readonly material: MpmMaterialParameters;
	readonly gravity: MpmVector2;
	readonly domainWidth: number;
	readonly domainHeight: number;
	readonly gradientScale: number;
	readonly stressScale: number;
}

interface GridNode {
	readonly xIndex: number;
	readonly yIndex: number;
	mass: number;
	momentumX: number;
	momentumY: number;
	velocityX: number;
	velocityY: number;
	waterMass: number;
	waterMomentumX: number;
	waterMomentumY: number;
	waterVelocityX: number;
	waterVelocityY: number;
	waterVolume: number;
	oilMass: number;
	oilMomentumX: number;
	oilMomentumY: number;
	oilVelocityX: number;
	oilVelocityY: number;
	oilVolume: number;
}

interface TriangleContactField {
	readonly bodyStart: number;
	readonly signedDistance: number;
	readonly normal: MpmVector2;
	readonly barycentric: readonly [number, number, number];
	impulseX: number;
	impulseY: number;
}

interface TriangleField {
	readonly nodes: ReadonlyMap<string, TriangleContactField>;
	readonly nodesByBody: ReadonlyMap<number, readonly TriangleContactField[]>;
}

interface ParticlePhysicalState {
	readonly mass: number;
	readonly volume: number;
	readonly material: MpmMaterialParameters;
}

interface ParticleStencil {
	readonly baseX: number;
	readonly baseY: number;
	readonly fx: number;
	readonly fy: number;
	readonly weightsX: readonly [number, number, number];
	readonly weightsY: readonly [number, number, number];
}

interface Decomposition2 {
	readonly sigma1: number;
	readonly sigma2: number;
	readonly u1: MpmVector2;
	readonly u2: MpmVector2;
	readonly v1: MpmVector2;
	readonly v2: MpmVector2;
	readonly rotation: MpmMatrix2;
}

const IDENTITY: MpmMatrix2 = { m00: 1, m01: 0, m10: 0, m11: 1 };
const WALL_CUSHION_CELLS = 8;
const MIN_LIQUID_DETERMINANT = 0.5;
const MAX_LIQUID_DETERMINANT = 2;
const MIN_POWDER_DETERMINANT = 0.5;
const MAX_DEFORMATION_INCREMENT = 0.25;
const MAX_DEFORMATION_SUBSTEPS = 64;
export const DEFAULT_BONDED_TRIANGLE_STIFFNESS = 0.35;
const BONDED_TRIANGLE_SOLVER_ITERATIONS = 2;
const TRIANGLE_CONTACT_SKIN_CELLS = 0.1;
const TRIANGLE_FIELD_BAND_CELLS = 2;
const GRID_CONTACT_REACH_CELLS = 0.75;

export function stepMpm(
	particles: readonly MpmParticle[],
	config: MpmStepConfig
): MpmStepResult {
	const validated = validateConfig(config);
	validateParticles(particles, validated);
	const bondedBodyStarts = collectBondedBodyStarts(particles, validated);
	const particleStates = particles.map((particle) => resolveParticleState(particle, validated));

	const inputMass = totalMass(particleStates, 'input mass');
	assertComputedFinite(inputMass, 'input mass');
	const inputMomentum = totalMomentum(particles, particleStates, 'input momentum');
	const grid = new Map<string, GridNode>();
	const dx = validated.grid.cellSize;

	for (let index = 0; index < particles.length; index++) {
		const particle = particles[index];
		const state = particleStates[index];
		const stencil = particleStencil(particle.position, dx);
		const plasticVolume = particle.plasticVolume ?? 1;
		const stressTimesFt = constitutiveStressTimesFt(
			particle.F,
			plasticVolume,
			state.material
		);
		const stressScale = -validated.dt * state.volume * validated.gradientScale;
		assertComputedFinite(stressScale, 'particle stress scale');
		const isLiquid = state.material.model === 'water' || state.material.model === 'oil';
		const liquidVolume = isLiquid ? state.volume * determinant(particle.F) : 0;
		if (isLiquid) assertComputedFinite(liquidVolume, 'liquid particle volume');
		const affine = addMatrices(
			scaleMatrix(particle.C, state.mass),
			scaleMatrix(stressTimesFt, stressScale)
		);
		assertComputedMatrixFinite(affine, 'particle affine momentum');

		for (let oy = 0; oy < 3; oy++) {
			for (let ox = 0; ox < 3; ox++) {
				const weight = stencil.weightsX[ox] * stencil.weightsY[oy];
				assertWeight(weight);
				if (weight === 0) continue;

				const xIndex = stencil.baseX + ox;
				const yIndex = stencil.baseY + oy;
				const key = gridNodeKey(xIndex, yIndex);
				let node = grid.get(key);
				if (!node) {
					node = {
						xIndex,
						yIndex,
						mass: 0,
						momentumX: 0,
						momentumY: 0,
						velocityX: 0,
						velocityY: 0,
						waterMass: 0,
						waterMomentumX: 0,
						waterMomentumY: 0,
						waterVelocityX: 0,
						waterVelocityY: 0,
						waterVolume: 0,
						oilMass: 0,
						oilMomentumX: 0,
						oilMomentumY: 0,
						oilVelocityX: 0,
						oilVelocityY: 0,
						oilVolume: 0
					};
					grid.set(key, node);
				}

				const dposX = (ox - stencil.fx) * dx;
				const dposY = (oy - stencil.fy) * dx;
				const particleMomentumX =
					state.mass * particle.velocity.x +
					affine.m00 * dposX +
					affine.m01 * dposY;
				const particleMomentumY =
					state.mass * particle.velocity.y +
					affine.m10 * dposX +
					affine.m11 * dposY;
				if (state.material.model === 'water') {
					node.waterMass += weight * state.mass;
					node.waterMomentumX += weight * particleMomentumX;
					node.waterMomentumY += weight * particleMomentumY;
					node.waterVolume += weight * liquidVolume;
					assertComputedFinite(node.waterMass, 'water grid mass');
					assertComputedFinite(node.waterVolume, 'water grid volume');
				} else if (state.material.model === 'oil') {
					node.oilMass += weight * state.mass;
					node.oilMomentumX += weight * particleMomentumX;
					node.oilMomentumY += weight * particleMomentumY;
					node.oilVolume += weight * liquidVolume;
					assertComputedFinite(node.oilMass, 'oil grid mass');
					assertComputedFinite(node.oilVolume, 'oil grid volume');
				} else {
					node.mass += weight * state.mass;
					node.momentumX += weight * particleMomentumX;
					node.momentumY += weight * particleMomentumY;
				}
				assertGridNodeFinite(node);
			}
		}
	}

	let gridMass = 0;
	let gridMomentumX = 0;
	let gridMomentumY = 0;
	for (const node of grid.values()) {
		gridMass += node.mass + node.waterMass + node.oilMass;
		gridMomentumX += node.momentumX + node.waterMomentumX + node.oilMomentumX;
		gridMomentumY += node.momentumY + node.waterMomentumY + node.oilMomentumY;
	}
	assertComputedFinite(gridMass, 'grid mass');
	assertComputedFinite(gridMomentumX, 'grid momentum');
	assertComputedFinite(gridMomentumY, 'grid momentum');

	const triangleField = buildTriangleField(
		bondedBodyStarts,
		particles,
		validated
	);
	for (const node of grid.values()) {
		const solidVelocity = updateGridVelocity(
			node.mass,
			node.momentumX,
			node.momentumY,
			validated
		);
		const waterVelocity = updateGridVelocity(
			node.waterMass,
			node.waterMomentumX,
			node.waterMomentumY,
			validated
		);
		const oilVelocity = updateGridVelocity(
			node.oilMass,
			node.oilMomentumX,
			node.oilMomentumY,
			validated
		);
		node.velocityX = solidVelocity.x;
		node.velocityY = solidVelocity.y;
		node.waterVelocityX = waterVelocity.x;
		node.waterVelocityY = waterVelocity.y;
		node.oilVelocityX = oilVelocity.x;
		node.oilVelocityY = oilVelocity.y;
		solveLiquidGridContact(node, grid, validated);
		const triangleContact = triangleField.nodes.get(
			gridNodeKey(node.xIndex, node.yIndex)
		);
		if (triangleContact) {
			solveFluidTriangleGridContact(
				node,
				triangleContact,
				particles,
				validated
			);
		}
		node.velocityX = applyWallCushion(
			node.velocityX,
			node.xIndex,
			validated.grid.cellsX
		);
		node.velocityY = applyWallCushion(
			node.velocityY,
			node.yIndex,
			validated.grid.cellsY
		);
		node.waterVelocityX = applyWallCushion(
			node.waterVelocityX,
			node.xIndex,
			validated.grid.cellsX
		);
		node.waterVelocityY = applyWallCushion(
			node.waterVelocityY,
			node.yIndex,
			validated.grid.cellsY
		);
		node.oilVelocityX = applyWallCushion(
			node.oilVelocityX,
			node.xIndex,
			validated.grid.cellsX
		);
		node.oilVelocityY = applyWallCushion(
			node.oilVelocityY,
			node.yIndex,
			validated.grid.cellsY
		);
		assertGridNodeFinite(node);
	}

	const outputParticles = particles.map((particle, index) => {
		const state = particleStates[index];
		const stencil = particleStencil(particle.position, dx);
		const plasticVolume = particle.plasticVolume ?? 1;
		let velocityX = 0;
		let velocityY = 0;
		let c00 = 0;
		let c01 = 0;
		let c10 = 0;
		let c11 = 0;
		let localLiquidMass = 0;
		let localLiquidVolume = 0;
		const isWater = state.material.model === 'water';
		const isOil = state.material.model === 'oil';
		const isLiquid = isWater || isOil;

		for (let oy = 0; oy < 3; oy++) {
			for (let ox = 0; ox < 3; ox++) {
				const weight = stencil.weightsX[ox] * stencil.weightsY[oy];
				assertWeight(weight);
				if (weight === 0) continue;

				const node = grid.get(gridNodeKey(stencil.baseX + ox, stencil.baseY + oy));
				if (!node) {
					throw new Error('particle stencil is missing a deposited grid node');
				}
				const dposX = (ox - stencil.fx) * dx;
				const dposY = (oy - stencil.fy) * dx;
				const nodeVelocityX = isWater
					? node.waterVelocityX
					: isOil
						? node.oilVelocityX
						: node.velocityX;
				const nodeVelocityY = isWater
					? node.waterVelocityY
					: isOil
						? node.oilVelocityY
						: node.velocityY;
				velocityX += weight * nodeVelocityX;
				velocityY += weight * nodeVelocityY;
				if (isWater) {
					localLiquidMass += weight * (node.waterMass + node.oilMass);
					localLiquidVolume += weight * (node.waterVolume + node.oilVolume);
				} else if (isOil) {
					localLiquidMass += weight * (node.waterMass + node.oilMass);
					localLiquidVolume += weight * (node.waterVolume + node.oilVolume);
				}
				c00 += weight * nodeVelocityX * dposX;
				c01 += weight * nodeVelocityX * dposY;
				c10 += weight * nodeVelocityY * dposX;
				c11 += weight * nodeVelocityY * dposY;
			}
		}

		if (isLiquid) {
			assertComputedFinite(localLiquidMass, 'local liquid mass');
			assertComputedFinite(localLiquidVolume, 'local liquid volume');
			if (localLiquidMass <= 0 || localLiquidVolume <= 0) {
				throw new RangeError('liquid particle stencil has no local liquid density');
			}
			const particleDensity =
				state.mass / (state.volume * determinant(particle.F));
			const localDensity = localLiquidMass / localLiquidVolume;
			assertComputedFinite(particleDensity, 'particle density');
			assertComputedFinite(localDensity, 'local liquid density');
			if (particleDensity <= 0 || localDensity <= 0) {
				throw new RangeError('liquid densities must be positive');
			}
			const relativeDensity = (particleDensity - localDensity) / particleDensity;
			velocityX += validated.dt * validated.gravity.x * relativeDensity;
			velocityY += validated.dt * validated.gravity.y * relativeDensity;
		}

		const C: MpmMatrix2 = {
			m00: c00 * validated.gradientScale,
			m01: c01 * validated.gradientScale,
			m10: c10 * validated.gradientScale,
			m11: c11 * validated.gradientScale
		};
		assertComputedFinite(velocityX, 'particle velocity');
		assertComputedFinite(velocityY, 'particle velocity');
		assertComputedMatrixFinite(C, 'particle affine velocity');
		const trialF = updateDeformationGradient(particle.F, C, validated.dt);
		const jacobian = determinant(trialF);
		if (!Number.isFinite(jacobian) || jacobian <= 0) {
			throw new RangeError('step would produce a non-positive deformation determinant');
		}
		let F = trialF;
		let nextPlasticVolume = plasticVolume;
		if (state.material.model === 'snow') {
			const projected = projectSnowDeformation(trialF, plasticVolume);
			F = projected.deformation;
			nextPlasticVolume = projected.plasticVolume;
		} else if (state.material.model === 'powder') {
			F = projectGranularDeformation(trialF, state.material);
		} else if (state.material.model === 'water' || state.material.model === 'oil') {
			F = projectLiquidDeformation(trialF);
		}

		const nextX = particle.position.x + validated.dt * velocityX;
		const nextY = particle.position.y + validated.dt * velocityY;
		assertComputedFinite(nextX, 'particle position');
		assertComputedFinite(nextY, 'particle position');
		const boundedX = confineAxis(nextX, velocityX, validated.domainWidth);
		const boundedY = confineAxis(nextY, velocityY, validated.domainHeight);
		const projectedPosition = isLiquid
			? projectParticleOutsideTriangles(
					{ x: boundedX.position, y: boundedY.position },
					Math.sqrt((state.volume * determinant(F)) / Math.PI),
					triangleField,
					validated
				)
			: { x: boundedX.position, y: boundedY.position };
		const finalX = confineAxis(
			projectedPosition.x,
			boundedX.velocity,
			validated.domainWidth
		);
		const finalY = confineAxis(
			projectedPosition.y,
			boundedY.velocity,
			validated.domainHeight
		);

		return {
			position: { x: finalX.position, y: finalY.position },
			velocity: { x: finalX.velocity, y: finalY.velocity },
			C,
			F,
			objectId: particle.objectId,
			...(particle.bondedTriangleVertex === undefined
				? {}
				: { bondedTriangleVertex: particle.bondedTriangleVertex }),
			plasticVolume: nextPlasticVolume,
			volume: state.volume,
			material: state.material
		};
	});

	applyTriangleContactImpulses(
		outputParticles,
		triangleField,
		bondedBodyStarts,
		particleStates
	);
	enforceBondedTriangleConstraints(
		outputParticles,
		bondedBodyStarts,
		validated
	);
	resolveBondedTriangleContacts(
		particles,
		outputParticles,
		bondedBodyStarts,
		validated
	);

	const outputMass = totalMass(particleStates, 'output mass');
	assertComputedFinite(outputMass, 'output mass');
	const outputMomentum = totalMomentum(
		outputParticles,
		particleStates,
		'output momentum'
	);

	return {
		particles: outputParticles,
		diagnostics: {
			inputMass,
			gridMass,
			outputMass,
			inputMomentum,
			gridMomentum: { x: gridMomentumX, y: gridMomentumY },
			outputMomentum
		}
	};
}

function validateConfig(config: MpmStepConfig): ValidatedConfig {
	if (!isRecord(config)) {
		throw new TypeError('config must be an object');
	}
	const raw = config as unknown as Record<string, unknown>;
	const grid = raw.grid;
	if (!isRecord(grid)) throw new TypeError('grid must be an object');
	const material = validateMaterial(raw.material, 'material');

	const dt = raw.dt;
	const cellsX = grid.cellsX;
	const cellsY = grid.cellsY;
	const cellSize = grid.cellSize;
	const particleMass = raw.particleMass;
	const particleVolume = raw.particleVolume;
	const bondRestLength = raw.bondRestLength;
	const bondStiffness =
		raw.bondStiffness === undefined
			? DEFAULT_BONDED_TRIANGLE_STIFFNESS
			: raw.bondStiffness;
	const lambda = material.lambda;
	const mu = material.mu;
	const gravity = raw.gravity;

	assertFiniteNumber(dt, 'dt');
	assertFiniteNumber(cellsX, 'grid.cellsX');
	assertFiniteNumber(cellsY, 'grid.cellsY');
	assertFiniteNumber(cellSize, 'grid.cellSize');
	assertFiniteNumber(particleMass, 'particleMass');
	assertFiniteNumber(particleVolume, 'particleVolume');
	assertFiniteNumber(bondStiffness, 'bondStiffness');
	assertFiniteNumber(lambda, 'material.lambda');
	assertFiniteNumber(mu, 'material.mu');
	assertVector(gravity, 'gravity');

	if (dt <= 0) throw new RangeError('dt must be greater than zero');
	if (!Number.isSafeInteger(cellsX) || cellsX < 1 || cellsX >= Number.MAX_SAFE_INTEGER) {
		throw new RangeError('grid.cellsX must be a positive safe integer');
	}
	if (!Number.isSafeInteger(cellsY) || cellsY < 1 || cellsY >= Number.MAX_SAFE_INTEGER) {
		throw new RangeError('grid.cellsY must be a positive safe integer');
	}
	if (cellSize <= 0) throw new RangeError('grid.cellSize must be greater than zero');
	if (particleMass <= 0) throw new RangeError('particleMass must be greater than zero');
	if (particleVolume <= 0) throw new RangeError('particleVolume must be greater than zero');
	if (bondRestLength !== undefined) {
		assertFiniteNumber(bondRestLength, 'bondRestLength');
		if (bondRestLength <= 0) {
			throw new RangeError('bondRestLength must be greater than zero');
		}
	}
	if (bondStiffness <= 0 || bondStiffness > 1) {
		throw new RangeError('bondStiffness must be greater than zero and at most one');
	}
	const domainWidth = cellsX * cellSize;
	const domainHeight = cellsY * cellSize;
	const cellSizeSquared = cellSize * cellSize;
	const gradientScale = 4 / cellSizeSquared;
	if (!Number.isFinite(domainWidth) || !Number.isFinite(domainHeight)) {
		throw new RangeError('grid dimensions produce a non-finite domain');
	}
	if (!Number.isFinite((cellsX + 1) * cellSize) ||
		!Number.isFinite((cellsY + 1) * cellSize)) {
		throw new RangeError('grid dimensions produce non-finite halo coordinates');
	}
	if (!Number.isFinite(cellSizeSquared) || cellSizeSquared <= 0 ||
		!Number.isFinite(gradientScale)) {
		throw new RangeError('grid.cellSize is outside the supported numeric range');
	}
	if (!Number.isFinite(particleMass / particleVolume) ||
		particleMass / particleVolume <= 0 ||
		!Number.isFinite(lambda + 2 * mu)) {
		throw new RangeError('mass and material parameters are outside the supported numeric range');
	}

	const stressScale = -dt * particleVolume * gradientScale;
	if (!Number.isFinite(stressScale)) {
		throw new RangeError('dt, particleVolume, and grid.cellSize produce a non-finite stress scale');
	}

	return {
		dt,
		grid: { cellsX, cellsY, cellSize },
		particleMass,
		particleVolume,
		bondRestLength,
		bondStiffness,
		material,
		gravity: { x: gravity.x, y: gravity.y },
		domainWidth,
		domainHeight,
		gradientScale,
		stressScale
	};
}

function validateParticles(
	particles: readonly MpmParticle[],
	config: ValidatedConfig
): void {
	if (!Array.isArray(particles)) {
		throw new TypeError('particles must be an array');
	}
	for (let index = 0; index < particles.length; index++) {
		const value: unknown = particles[index];
		if (!isRecord(value)) throw new TypeError(`particles[${index}] must be an object`);
		assertVector(value.position, `particles[${index}].position`);
		assertVector(value.velocity, `particles[${index}].velocity`);
		assertMatrix(value.C, `particles[${index}].C`);
		assertMatrix(value.F, `particles[${index}].F`);

		const particle = value as unknown as MpmParticle;
		if (value.material !== undefined) {
			validateMaterial(value.material, `particles[${index}].material`);
		}
		if (value.volume !== undefined) {
			assertFiniteNumber(value.volume, `particles[${index}].volume`);
			if (value.volume <= 0) {
				throw new RangeError(`particles[${index}].volume must be greater than zero`);
			}
		}
		if (value.plasticVolume !== undefined) {
			assertFiniteNumber(value.plasticVolume, `particles[${index}].plasticVolume`);
			if (
				value.plasticVolume < SNOW_PLASTICITY.minPlasticVolume ||
				value.plasticVolume > SNOW_PLASTICITY.maxPlasticVolume
			) {
				throw new RangeError(
					`particles[${index}].plasticVolume must be between ${SNOW_PLASTICITY.minPlasticVolume} and ${SNOW_PLASTICITY.maxPlasticVolume}`
				);
			}
		}
		if (
			value.bondedTriangleVertex !== undefined &&
			(typeof value.bondedTriangleVertex !== 'number' ||
				!Number.isSafeInteger(value.bondedTriangleVertex) ||
				value.bondedTriangleVertex < 0 ||
				value.bondedTriangleVertex > 2)
		) {
			throw new RangeError(
				`particles[${index}].bondedTriangleVertex must be 0, 1, or 2`
			);
		}
		if (
			particle.position.x < 0 ||
			particle.position.x > config.domainWidth ||
			particle.position.y < 0 ||
			particle.position.y > config.domainHeight
		) {
			throw new RangeError(`particles[${index}].position must be inside the grid domain`);
		}
		const jacobian = determinant(particle.F);
		if (!Number.isFinite(jacobian) || jacobian <= 0) {
			throw new RangeError(`particles[${index}].F must have a finite positive determinant`);
		}
	}
}

function collectBondedBodyStarts(
	particles: readonly MpmParticle[],
	config: ValidatedConfig
): number[] {
	const starts: number[] = [];
	for (let index = 0; index < particles.length; index += 1) {
		const particle = particles[index];
		if (particle.bondedTriangleVertex === undefined) continue;
		if (particle.bondedTriangleVertex !== 0) {
			throw new RangeError(
				`particles[${index}].bondedTriangleVertex must start each triangle at zero`
			);
		}
		if (config.bondRestLength === undefined) {
			throw new RangeError('bondRestLength is required for bonded triangle particles');
		}
		if (index + 2 >= particles.length) {
			throw new RangeError(`bonded triangle at particle ${index} is incomplete`);
		}
		const objectId = particle.objectId ?? 0;
		for (let vertex = 0; vertex < 3; vertex += 1) {
			const member = particles[index + vertex];
			if (
				member.bondedTriangleVertex !== vertex ||
				(member.objectId ?? 0) !== objectId
			) {
				throw new RangeError(
					`bonded triangle at particle ${index} must contain contiguous vertices 0, 1, and 2 of one object`
				);
			}
			const material = particleMaterial(member, config.material);
			if (material.model === 'water' || material.model === 'oil') {
				throw new RangeError('bonded triangle vertices must use a solid material');
			}
		}
		starts.push(index);
		index += 2;
	}
	return starts;
}

function buildTriangleField(
	bodyStarts: readonly number[],
	particles: readonly MpmParticle[],
	config: ValidatedConfig
): TriangleField {
	const nodes = new Map<string, TriangleContactField>();
	const nodesByBody = new Map<number, TriangleContactField[]>();
	if (bodyStarts.length === 0) return { nodes, nodesByBody };

	const { cellsX, cellsY, cellSize } = config.grid;
	const maxDistance = cellSize * TRIANGLE_FIELD_BAND_CELLS;
	for (let yIndex = -1; yIndex <= cellsY + 1; yIndex += 1) {
		for (let xIndex = -1; xIndex <= cellsX + 1; xIndex += 1) {
			const point = { x: xIndex * cellSize, y: yIndex * cellSize };
			let nearest: TriangleContactField | undefined;
			for (const bodyStart of bodyStarts) {
				const triangle = getTriangleVertices(particles, bodyStart);
				if (
					point.x < Math.min(triangle[0].x, triangle[1].x, triangle[2].x) - maxDistance ||
					point.x > Math.max(triangle[0].x, triangle[1].x, triangle[2].x) + maxDistance ||
					point.y < Math.min(triangle[0].y, triangle[1].y, triangle[2].y) - maxDistance ||
					point.y > Math.max(triangle[0].y, triangle[1].y, triangle[2].y) + maxDistance
				) {
					continue;
				}
				const sample = triangleDistanceSample(point, triangle);
				if (
					sample.signedDistance <= maxDistance &&
					(nearest === undefined ||
						sample.signedDistance < nearest.signedDistance)
				) {
					nearest = {
						bodyStart,
						signedDistance: sample.signedDistance,
						normal: sample.normal,
						barycentric: sample.barycentric,
						impulseX: 0,
						impulseY: 0
					};
				}
			}
			if (nearest) {
				nodes.set(gridNodeKey(xIndex, yIndex), nearest);
				const bodyNodes = nodesByBody.get(nearest.bodyStart) ?? [];
				bodyNodes.push(nearest);
				nodesByBody.set(nearest.bodyStart, bodyNodes);
			}
		}
	}
	return { nodes, nodesByBody };
}

function getTriangleVertices(
	particles: readonly MpmParticle[],
	start: number
): readonly [MpmVector2, MpmVector2, MpmVector2] {
	return [
		particles[start].position,
		particles[start + 1].position,
		particles[start + 2].position
	];
}

function triangleDistanceSample(
	point: MpmVector2,
	vertices: readonly [MpmVector2, MpmVector2, MpmVector2]
): {
	readonly signedDistance: number;
	readonly normal: MpmVector2;
	readonly barycentric: readonly [number, number, number];
} {
	const [a, b, c] = vertices;
	const signedArea = cross2(subtractVector(b, a), subtractVector(c, a));
	if (!Number.isFinite(signedArea) || Math.abs(signedArea) <= 1e-10) {
		throw new RangeError('bonded triangle became degenerate');
	}
	const crosses = [
		cross2(subtractVector(b, a), subtractVector(point, a)),
		cross2(subtractVector(c, b), subtractVector(point, b)),
		cross2(subtractVector(a, c), subtractVector(point, c))
	];
	const inside =
		signedArea > 0
			? crosses.every((value) => value >= -1e-10)
			: crosses.every((value) => value <= 1e-10);
	const edges = [
		{ start: a, end: b, barycentric: (t: number) => [1 - t, t, 0] as const },
		{ start: b, end: c, barycentric: (t: number) => [0, 1 - t, t] as const },
		{ start: c, end: a, barycentric: (t: number) => [t, 0, 1 - t] as const }
	] as const;

	let closestDistanceSquared = Number.POSITIVE_INFINITY;
	let closestPoint: MpmVector2 = a;
	let closestNormal: MpmVector2 = { x: 0, y: 0 };
	let barycentric: readonly [number, number, number] = [1, 0, 0];
	for (const edge of edges) {
		const direction = subtractVector(edge.end, edge.start);
		const lengthSquared = dot2(direction, direction);
		const t = Math.max(
			0,
			Math.min(
				1,
				dot2(subtractVector(point, edge.start), direction) / lengthSquared
			)
		);
		const candidate = addVector(edge.start, scaleVector(direction, t));
		const difference = subtractVector(point, candidate);
		const distanceSquared = dot2(difference, difference);
		if (distanceSquared >= closestDistanceSquared) continue;
		closestDistanceSquared = distanceSquared;
		closestPoint = candidate;
		barycentric = edge.barycentric(t);
		if (distanceSquared > 1e-20 && !inside) {
			const distance = Math.sqrt(distanceSquared);
			closestNormal = scaleVector(difference, 1 / distance);
		} else {
			const orientation = signedArea > 0 ? 1 : -1;
			const outward = {
				x: orientation * direction.y,
				y: -orientation * direction.x
			};
			const edgeLength = Math.sqrt(lengthSquared);
			closestNormal = scaleVector(outward, 1 / edgeLength);
		}
	}
	const distance = Math.sqrt(closestDistanceSquared);
	if (!Number.isFinite(distance) || !isFiniteVector(closestNormal)) {
		throw new RangeError('bonded triangle contact field became non-finite');
	}
	return {
		signedDistance: inside ? -distance : distance,
		normal: closestNormal,
		barycentric
	};
}

function solveLiquidGridContact(
	node: GridNode,
	grid: ReadonlyMap<string, GridNode>,
	config: ValidatedConfig
): void {
	if (
		node.waterMass <= 0 ||
		node.oilMass <= 0 ||
		node.waterVolume <= 0 ||
		node.oilVolume <= 0 ||
		node.xIndex < 0 ||
		node.xIndex > config.grid.cellsX ||
		node.yIndex < 0 ||
		node.yIndex > config.grid.cellsY
	) {
		return;
	}

	const waterGradient = volumeGradient(grid, node.xIndex, node.yIndex, 'water', config.grid.cellSize);
	const oilGradient = volumeGradient(grid, node.xIndex, node.yIndex, 'oil', config.grid.cellSize);
	const gradient = {
		x: oilGradient.x - waterGradient.x,
		y: oilGradient.y - waterGradient.y
	};
	const gradientLength = Math.hypot(gradient.x, gradient.y);
	if (!Number.isFinite(gradientLength) || gradientLength <= 1e-12) return;
	const normal = scaleVector(gradient, 1 / gradientLength);
	const relativeNormalVelocity =
		(node.oilVelocityX - node.waterVelocityX) * normal.x +
		(node.oilVelocityY - node.waterVelocityY) * normal.y;
	if (relativeNormalVelocity >= 0) return;
	const impulse =
		-relativeNormalVelocity /
		(1 / node.waterMass + 1 / node.oilMass);
	node.waterVelocityX -= (impulse / node.waterMass) * normal.x;
	node.waterVelocityY -= (impulse / node.waterMass) * normal.y;
	node.oilVelocityX += (impulse / node.oilMass) * normal.x;
	node.oilVelocityY += (impulse / node.oilMass) * normal.y;
}

function volumeGradient(
	grid: ReadonlyMap<string, GridNode>,
	xIndex: number,
	yIndex: number,
	phase: 'water' | 'oil',
	cellSize: number
): MpmVector2 {
	const volumeAt = (x: number, y: number): number => {
		const node = grid.get(gridNodeKey(x, y));
		return node === undefined
			? 0
			: phase === 'water'
				? node.waterVolume
				: node.oilVolume;
	};
	const scale = 1 / (2 * cellSize * cellSize);
	return {
		x: (volumeAt(xIndex + 1, yIndex) - volumeAt(xIndex - 1, yIndex)) * scale,
		y: (volumeAt(xIndex, yIndex + 1) - volumeAt(xIndex, yIndex - 1)) * scale
	};
}

function solveFluidTriangleGridContact(
	node: GridNode,
	field: TriangleContactField,
	particles: readonly MpmParticle[],
	config: ValidatedConfig
): void {
	const radius = Math.sqrt(config.particleVolume / Math.PI);
	const contactReach = radius + GRID_CONTACT_REACH_CELLS * config.grid.cellSize;
	if (field.signedDistance > contactReach) return;
	const bodyStart = field.bodyStart;
	const barycentric = field.barycentric;
	const surfaceVelocity = { x: 0, y: 0 };
	let inverseBodyMass = 0;
	for (let vertex = 0; vertex < 3; vertex += 1) {
		const weight = barycentric[vertex];
		surfaceVelocity.x += weight * particles[bodyStart + vertex].velocity.x;
		surfaceVelocity.y += weight * particles[bodyStart + vertex].velocity.y;
		inverseBodyMass += (weight * weight) / config.particleMass;
	}
	if (inverseBodyMass <= 0) return;

	if (node.waterMass > 0) {
		const relative =
			(node.waterVelocityX - surfaceVelocity.x) * field.normal.x +
			(node.waterVelocityY - surfaceVelocity.y) * field.normal.y;
		if (relative < 0) {
			const impulse = -relative / (1 / node.waterMass + inverseBodyMass);
			node.waterVelocityX += (impulse / node.waterMass) * field.normal.x;
			node.waterVelocityY += (impulse / node.waterMass) * field.normal.y;
			field.impulseX -= impulse * field.normal.x;
			field.impulseY -= impulse * field.normal.y;
		}
	}
	if (node.oilMass > 0) {
		const relative =
			(node.oilVelocityX - surfaceVelocity.x) * field.normal.x +
			(node.oilVelocityY - surfaceVelocity.y) * field.normal.y;
		if (relative < 0) {
			const impulse = -relative / (1 / node.oilMass + inverseBodyMass);
			node.oilVelocityX += (impulse / node.oilMass) * field.normal.x;
			node.oilVelocityY += (impulse / node.oilMass) * field.normal.y;
			field.impulseX -= impulse * field.normal.x;
			field.impulseY -= impulse * field.normal.y;
		}
	}
}

function updateGridVelocity(
	mass: number,
	momentumX: number,
	momentumY: number,
	config: ValidatedConfig
): MpmVector2 {
	if (mass <= 0) return { x: 0, y: 0 };
	const velocity = {
		x: momentumX / mass + config.dt * config.gravity.x,
		y: momentumY / mass + config.dt * config.gravity.y
	};
	assertComputedFinite(velocity.x, 'grid velocity');
	assertComputedFinite(velocity.y, 'grid velocity');
	return {
		x: velocity.x,
		y: velocity.y
	};
}

function projectParticleOutsideTriangles(
	position: MpmVector2,
	radius: number,
	triangleField: TriangleField,
	config: ValidatedConfig
): MpmVector2 {
	if (triangleField.nodes.size === 0) return position;
	const stencil = particleStencil(position, config.grid.cellSize);
	const band = config.grid.cellSize * TRIANGLE_FIELD_BAND_CELLS;
	let signedDistance = 0;
	let closestBodyStart: number | undefined;
	let closestDistance = Number.POSITIVE_INFINITY;
	let selectedNormal: MpmVector2 | undefined;
	for (let oy = 0; oy < 3; oy += 1) {
		for (let ox = 0; ox < 3; ox += 1) {
			const weight = stencil.weightsX[ox] * stencil.weightsY[oy];
			const field = triangleField.nodes.get(
				gridNodeKey(stencil.baseX + ox, stencil.baseY + oy)
			);
			signedDistance += weight * (field?.signedDistance ?? band);
			if (field && field.signedDistance < closestDistance) {
				closestDistance = field.signedDistance;
				closestBodyStart = field.bodyStart;
				selectedNormal = field.normal;
			}
		}
	}
	if (closestBodyStart === undefined || signedDistance >= radius) return position;
	let normalX = 0;
	let normalY = 0;
	for (let oy = 0; oy < 3; oy += 1) {
		for (let ox = 0; ox < 3; ox += 1) {
			const weight = stencil.weightsX[ox] * stencil.weightsY[oy];
			const field = triangleField.nodes.get(
				gridNodeKey(stencil.baseX + ox, stencil.baseY + oy)
			);
			if (!field || field.bodyStart !== closestBodyStart) continue;
			normalX += weight * field.normal.x;
			normalY += weight * field.normal.y;
		}
	}
	const normalLength = Math.hypot(normalX, normalY);
	const normal =
		normalLength > 1e-12 && Number.isFinite(normalLength)
			? { x: normalX / normalLength, y: normalY / normalLength }
			: selectedNormal;
	if (!normal) return position;
	const correction = radius - signedDistance;
	const projected = {
		x: position.x + correction * normal.x,
		y: position.y + correction * normal.y
	};
	assertComputedFinite(projected.x, 'triangle contact projection x');
	assertComputedFinite(projected.y, 'triangle contact projection y');
	return projected;
}

function applyTriangleContactImpulses(
	outputParticles: MpmParticle[],
	triangleField: TriangleField,
	bodyStarts: readonly number[],
	particleStates: readonly ParticlePhysicalState[]
): void {
	for (const bodyStart of bodyStarts) {
		const impulses = [0, 0, 0].map(() => ({ x: 0, y: 0 }));
		for (const field of triangleField.nodesByBody.get(bodyStart) ?? []) {
			for (let vertex = 0; vertex < 3; vertex += 1) {
				const weight = field.barycentric[vertex];
				impulses[vertex].x += weight * field.impulseX;
				impulses[vertex].y += weight * field.impulseY;
			}
		}
		for (let vertex = 0; vertex < 3; vertex += 1) {
			const index = bodyStart + vertex;
			const particle = outputParticles[index];
			const mass = particleStates[index].mass;
			outputParticles[index] = {
				...particle,
				velocity: {
					x: particle.velocity.x + impulses[vertex].x / mass,
					y: particle.velocity.y + impulses[vertex].y / mass
				}
			};
			assertComputedFinite(outputParticles[index].velocity.x, 'triangle reaction velocity x');
			assertComputedFinite(outputParticles[index].velocity.y, 'triangle reaction velocity y');
		}
	}
}

function enforceBondedTriangleConstraints(
	outputParticles: MpmParticle[],
	bodyStarts: readonly number[],
	config: ValidatedConfig
): void {
	if (bodyStarts.length === 0) return;
	const restLength = config.bondRestLength;
	if (restLength === undefined) {
		throw new RangeError('bondRestLength is required for bonded triangle particles');
	}
	const stiffness = config.bondStiffness;
	for (const bodyStart of bodyStarts) {
		const original = getTriangleVertices(outputParticles, bodyStart);
		const positions: MpmVector2[] = original.map(({ x, y }) => ({ x, y }));
		const edges = [
			[0, 1],
			[1, 2],
			[2, 0]
		] as const;
		for (let iteration = 0; iteration < BONDED_TRIANGLE_SOLVER_ITERATIONS; iteration += 1) {
			for (const [first, second] of edges) {
				const difference = subtractVector(positions[second], positions[first]);
				const distance = Math.hypot(difference.x, difference.y);
				if (!Number.isFinite(distance) || distance <= 1e-10) {
					throw new RangeError(`bonded triangle at particle ${bodyStart} collapsed`);
				}
				const correction = scaleVector(
					difference,
					((distance - restLength) * stiffness * 0.5) / distance
				);
				positions[first] = addVector(positions[first], correction);
				positions[second] = subtractVector(positions[second], correction);
			}
		}
		for (let vertex = 0; vertex < 3; vertex += 1) {
			const index = bodyStart + vertex;
			const particle = outputParticles[index];
			const displacement = subtractVector(positions[vertex], original[vertex]);
			const velocity = {
				x: particle.velocity.x + displacement.x / config.dt,
				y: particle.velocity.y + displacement.y / config.dt
			};
			const boundedX = confineAxis(
				positions[vertex].x,
				velocity.x,
				config.domainWidth
			);
			const boundedY = confineAxis(
				positions[vertex].y,
				velocity.y,
				config.domainHeight
			);
			outputParticles[index] = {
				...particle,
				position: { x: boundedX.position, y: boundedY.position },
				velocity: { x: boundedX.velocity, y: boundedY.velocity }
			};
		}
	}
}

function resolveBondedTriangleContacts(
	inputParticles: readonly MpmParticle[],
	outputParticles: MpmParticle[],
	bodyStarts: readonly number[],
	config: ValidatedConfig
): void {
	if (bodyStarts.length < 2) return;
	const skin = config.grid.cellSize * TRIANGLE_CONTACT_SKIN_CELLS;
	for (const bodyStart of bodyStarts) {
		const bodyId = inputParticles[bodyStart].objectId;
		const currentTriangle = getTriangleVertices(inputParticles, bodyStart);
		const currentVelocity = triangleMeanVelocity(inputParticles, bodyStart);
		let correctionX = 0;
		let correctionY = 0;
		let impulseX = 0;
		let impulseY = 0;
		const bodyMass = 3 * config.particleMass;

		for (const otherStart of bodyStarts) {
			if (
				otherStart === bodyStart ||
				inputParticles[otherStart].objectId === bodyId
			) {
				continue;
			}
			const contact = trianglePairContact(
				currentTriangle,
				getTriangleVertices(inputParticles, otherStart),
				skin
			);
			if (!contact) continue;
			const otherVelocity = triangleMeanVelocity(inputParticles, otherStart);
			const relativeNormalVelocity =
				(currentVelocity.x - otherVelocity.x) * contact.normal.x +
				(currentVelocity.y - otherVelocity.y) * contact.normal.y;
			if (relativeNormalVelocity < 0) {
				const impulse =
					-relativeNormalVelocity / (2 / bodyMass);
				impulseX += (impulse / bodyMass) * contact.normal.x;
				impulseY += (impulse / bodyMass) * contact.normal.y;
			}
			correctionX += contact.penetration * 0.5 * contact.normal.x;
			correctionY += contact.penetration * 0.5 * contact.normal.y;
		}

		for (let vertex = 0; vertex < 3; vertex += 1) {
			const index = bodyStart + vertex;
			const particle = outputParticles[index];
			const boundedX = confineAxis(
				particle.position.x + correctionX,
				particle.velocity.x + impulseX,
				config.domainWidth
			);
			const boundedY = confineAxis(
				particle.position.y + correctionY,
				particle.velocity.y + impulseY,
				config.domainHeight
			);
			outputParticles[index] = {
				...particle,
				position: { x: boundedX.position, y: boundedY.position },
				velocity: { x: boundedX.velocity, y: boundedY.velocity }
			};
		}
	}
}

function trianglePairContact(
	triangleA: readonly [MpmVector2, MpmVector2, MpmVector2],
	triangleB: readonly [MpmVector2, MpmVector2, MpmVector2],
	skin: number
): { readonly normal: MpmVector2; readonly penetration: number } | undefined {
	let separated = false;
	let largestGap = Number.NEGATIVE_INFINITY;
	let smallestOverlap = Number.POSITIVE_INFINITY;
	let contactAxis: MpmVector2 | undefined;
	for (const triangle of [triangleA, triangleB]) {
		for (let edgeIndex = 0; edgeIndex < 3; edgeIndex += 1) {
			const start = triangle[edgeIndex];
			const end = triangle[(edgeIndex + 1) % 3];
			const edge = subtractVector(end, start);
			const edgeLength = Math.hypot(edge.x, edge.y);
			if (edgeLength <= 1e-10) return undefined;
			const axis = { x: edge.y / edgeLength, y: -edge.x / edgeLength };
			const project = (vertices: readonly MpmVector2[]) => {
				const values = vertices.map((vertex) => dot2(vertex, axis));
				return {
					min: Math.min(...values),
					max: Math.max(...values)
				};
			};
			const projectionA = project(triangleA);
			const projectionB = project(triangleB);
			const gap = Math.max(
				projectionB.min - projectionA.max,
				projectionA.min - projectionB.max
			);
			if (gap > 0) {
				separated = true;
				if (gap > largestGap) {
					largestGap = gap;
					contactAxis = axis;
				}
			} else if (!separated && -gap < smallestOverlap) {
				smallestOverlap = -gap;
				contactAxis = axis;
			}
		}
	}
	if (separated && largestGap > skin) return undefined;
	if (!contactAxis) return undefined;
	const centerA = triangleCentroid(triangleA);
	const centerB = triangleCentroid(triangleB);
	const direction = subtractVector(centerA, centerB);
	const normal =
		dot2(direction, contactAxis) >= 0
			? contactAxis
			: scaleVector(contactAxis, -1);
	const penetration = separated
		? skin - largestGap
		: skin + smallestOverlap;
	return penetration > 0 ? { normal, penetration } : undefined;
}

function triangleCentroid(
	triangle: readonly [MpmVector2, MpmVector2, MpmVector2]
): MpmVector2 {
	return {
		x: (triangle[0].x + triangle[1].x + triangle[2].x) / 3,
		y: (triangle[0].y + triangle[1].y + triangle[2].y) / 3
	};
}

function triangleMeanVelocity(particles: readonly MpmParticle[], start: number): MpmVector2 {
	return {
		x:
			(particles[start].velocity.x +
				particles[start + 1].velocity.x +
				particles[start + 2].velocity.x) /
			3,
		y:
			(particles[start].velocity.y +
				particles[start + 1].velocity.y +
				particles[start + 2].velocity.y) /
			3
	};
}

function assertGridNodeFinite(node: GridNode): void {
	for (const [name, value] of Object.entries(node)) {
		if (typeof value === 'number') assertComputedFinite(value, `grid node ${name}`);
	}
	if (
		node.mass < 0 ||
		node.waterMass < 0 ||
		node.waterVolume < 0 ||
		node.oilMass < 0 ||
		node.oilVolume < 0
	) {
		throw new RangeError('grid node mass and phase volume must be non-negative');
	}
}

function subtractVector(first: MpmVector2, second: MpmVector2): MpmVector2 {
	return { x: first.x - second.x, y: first.y - second.y };
}

function addVector(first: MpmVector2, second: MpmVector2): MpmVector2 {
	return { x: first.x + second.x, y: first.y + second.y };
}

function scaleVector(vector: MpmVector2, scale: number): MpmVector2 {
	return { x: vector.x * scale, y: vector.y * scale };
}

function dot2(first: MpmVector2, second: MpmVector2): number {
	return first.x * second.x + first.y * second.y;
}

function cross2(first: MpmVector2, second: MpmVector2): number {
	return first.x * second.y - first.y * second.x;
}

function isFiniteVector(vector: MpmVector2): boolean {
	return Number.isFinite(vector.x) && Number.isFinite(vector.y);
}

function particleStencil(position: MpmVector2, cellSize: number): ParticleStencil {
	const gridX = position.x / cellSize;
	const gridY = position.y / cellSize;
	const baseX = Math.floor(gridX - 0.5);
	const baseY = Math.floor(gridY - 0.5);
	const fx = gridX - baseX;
	const fy = gridY - baseY;
	return {
		baseX,
		baseY,
		fx,
		fy,
		weightsX: quadraticWeights(fx),
		weightsY: quadraticWeights(fy)
	};
}

function quadraticWeights(fraction: number): readonly [number, number, number] {
	return [
		0.5 * (1.5 - fraction) ** 2,
		0.75 - (fraction - 1) ** 2,
		0.5 * (fraction - 0.5) ** 2
	];
}

function neoHookeanFirstPiola(
	F: MpmMatrix2,
	material: MpmMaterialParameters
): MpmMatrix2 {
	const jacobian = determinant(F);
	const inverseTranspose: MpmMatrix2 = {
		m00: F.m11 / jacobian,
		m01: -F.m10 / jacobian,
		m10: -F.m01 / jacobian,
		m11: F.m00 / jacobian
	};
	const logarithmicVolumeChange = Math.log(jacobian);
	const pressure = material.lambda * logarithmicVolumeChange;
	const P: MpmMatrix2 = {
		m00: material.mu * (F.m00 - inverseTranspose.m00) +
			pressure * inverseTranspose.m00,
		m01: material.mu * (F.m01 - inverseTranspose.m01) +
			pressure * inverseTranspose.m01,
		m10: material.mu * (F.m10 - inverseTranspose.m10) +
			pressure * inverseTranspose.m10,
		m11: material.mu * (F.m11 - inverseTranspose.m11) +
			pressure * inverseTranspose.m11
	};
	assertComputedMatrixFinite(P, 'first Piola stress');
	return P;
}

function validateMaterial(value: unknown, name: string): MpmMaterialParameters {
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
		return { lambda, mu: 0, model, density };
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
		return { lambda, mu, model, frictionCoefficient };
	}
	if (value.density !== undefined || value.frictionCoefficient !== undefined) {
		throw new RangeError(`${name} has parameters that do not match its constitutive model`);
	}
	if (lambda < 0) throw new RangeError(`${name}.lambda must be non-negative`);
	if (mu <= 0) throw new RangeError(`${name}.mu must be greater than zero`);
	return { lambda, mu, model };
}

function particleMaterial(
	particle: MpmParticle,
	defaultMaterial: MpmMaterialParameters
): MpmMaterialParameters {
	return particle.material === undefined
		? defaultMaterial
		: validateMaterial(particle.material, 'particle material');
}

function resolveParticleState(
	particle: MpmParticle,
	config: ValidatedConfig
): ParticlePhysicalState {
	const material = particleMaterial(particle, config.material);
	const volume = particle.volume ?? config.particleVolume;
	const mass =
		material.model === 'water' || material.model === 'oil'
			? material.density * volume
			: config.particleMass;
	assertComputedFinite(volume, 'particle volume');
	assertComputedFinite(mass, 'particle mass');
	if (volume <= 0 || mass <= 0) {
		throw new RangeError('particle mass and volume must be greater than zero');
	}
	return { mass, volume, material };
}

function constitutiveStressTimesFt(
	F: MpmMatrix2,
	plasticVolume: number,
	material: MpmMaterialParameters
): MpmMatrix2 {
	if (material.model === 'water' || material.model === 'oil') {
		const pressure = material.lambda * Math.min(determinant(F) - 1, 0);
		const stressTimesFt = { m00: pressure, m01: 0, m10: 0, m11: pressure };
		assertComputedMatrixFinite(stressTimesFt, `${material.model} stress`);
		return stressTimesFt;
	}
	if (material.model === 'snow') {
		const decomposition = decompose2x2(F);
		const hardeningExponent = Math.max(
			-SNOW_PLASTICITY.maxHardeningExponent,
			Math.min(
				SNOW_PLASTICITY.maxHardeningExponent,
				SNOW_PLASTICITY.hardening * (1 - plasticVolume)
			)
		);
		const hardening = Math.exp(hardeningExponent);
		const lambda = material.lambda * hardening;
		const mu = material.mu * hardening;
		const jacobian = determinant(F);
		const shear = 2 * mu;
		const volumetric = lambda * jacobian * (jacobian - 1);
		const difference = {
			m00: F.m00 - decomposition.rotation.m00,
			m01: F.m01 - decomposition.rotation.m01,
			m10: F.m10 - decomposition.rotation.m10,
			m11: F.m11 - decomposition.rotation.m11
		};
		const stressTimesFt: MpmMatrix2 = {
			m00:
				shear * (difference.m00 * F.m00 + difference.m01 * F.m01) +
				volumetric,
			m01: shear * (difference.m00 * F.m10 + difference.m01 * F.m11),
			m10: shear * (difference.m10 * F.m00 + difference.m11 * F.m01),
			m11:
				shear * (difference.m10 * F.m10 + difference.m11 * F.m11) +
				volumetric
		};
		assertComputedMatrixFinite(stressTimesFt, 'snow stress');
		return stressTimesFt;
	}
	if (material.model === 'powder') {
		return granularStressTimesFt(F, material);
	}
	const firstPiola = neoHookeanFirstPiola(F, material);
	return multiplyMatrices(firstPiola, transposeMatrix(F));
}

function granularStressTimesFt(
	F: MpmMatrix2,
	material: Extract<MpmMaterialParameters, { readonly model: 'powder' }>
): MpmMatrix2 {
	const decomposition = decompose2x2(F);
	const logStretch1 = Math.log(decomposition.sigma1);
	const logStretch2 = Math.log(decomposition.sigma2);
	const logVolume = logStretch1 + logStretch2;
	const boundedLogVolume = Math.max(
		Math.log(MIN_POWDER_DETERMINANT),
		logVolume
	);
	const pressure = Math.max(0, -(material.lambda + material.mu) * boundedLogVolume);
	const deviatoricLogStretch = 0.5 * (logStretch1 - logStretch2);
	const maximumDeviatoricLogStretch =
		(material.frictionCoefficient * pressure) / (2 * material.mu);
	const projectedDeviatoricLogStretch = Math.min(
		deviatoricLogStretch,
		maximumDeviatoricLogStretch
	);
	const meanLogStretch = 0.5 * logVolume;
	const principalStress1 =
		2 * material.mu * (meanLogStretch + projectedDeviatoricLogStretch) +
		material.lambda * logVolume;
	const principalStress2 =
		2 * material.mu * (meanLogStretch - projectedDeviatoricLogStretch) +
		material.lambda * logVolume;
	const stressTimesFt: MpmMatrix2 = {
		m00:
			principalStress1 * decomposition.u1.x * decomposition.u1.x +
			principalStress2 * decomposition.u2.x * decomposition.u2.x,
		m01:
			principalStress1 * decomposition.u1.x * decomposition.u1.y +
			principalStress2 * decomposition.u2.x * decomposition.u2.y,
		m10:
			principalStress1 * decomposition.u1.y * decomposition.u1.x +
			principalStress2 * decomposition.u2.y * decomposition.u2.x,
		m11:
			principalStress1 * decomposition.u1.y * decomposition.u1.y +
			principalStress2 * decomposition.u2.y * decomposition.u2.y
	};
	assertComputedMatrixFinite(stressTimesFt, 'powder stress');
	return stressTimesFt;
}

function projectSnowDeformation(
	deformation: MpmMatrix2,
	plasticVolume: number
): { readonly deformation: MpmMatrix2; readonly plasticVolume: number } {
	const decomposition = decompose2x2(deformation);
	const minimumStretch = 1 - SNOW_PLASTICITY.compressionYield;
	const maximumStretch = 1 + SNOW_PLASTICITY.stretchYield;
	const sigma1 = Math.max(minimumStretch, Math.min(maximumStretch, decomposition.sigma1));
	const sigma2 = Math.max(minimumStretch, Math.min(maximumStretch, decomposition.sigma2));
	const updatedPlasticVolume = Math.max(
		SNOW_PLASTICITY.minPlasticVolume,
		Math.min(
			SNOW_PLASTICITY.maxPlasticVolume,
			(plasticVolume * determinant(deformation)) / (sigma1 * sigma2)
		)
	);
	const projected: MpmMatrix2 = {
		m00:
			sigma1 * decomposition.u1.x * decomposition.v1.x +
			sigma2 * decomposition.u2.x * decomposition.v2.x,
		m01:
			sigma1 * decomposition.u1.x * decomposition.v1.y +
			sigma2 * decomposition.u2.x * decomposition.v2.y,
		m10:
			sigma1 * decomposition.u1.y * decomposition.v1.x +
			sigma2 * decomposition.u2.y * decomposition.v2.x,
		m11:
			sigma1 * decomposition.u1.y * decomposition.v1.y +
			sigma2 * decomposition.u2.y * decomposition.v2.y
	};
	assertComputedMatrixFinite(projected, 'projected snow deformation');
	assertComputedFinite(updatedPlasticVolume, 'snow plastic volume');
	if (determinant(projected) <= 0) {
		throw new RangeError('snow projection produced a non-positive deformation determinant');
	}
	return { deformation: projected, plasticVolume: updatedPlasticVolume };
}

function projectGranularDeformation(
	deformation: MpmMatrix2,
	material: Extract<MpmMaterialParameters, { readonly model: 'powder' }>
): MpmMatrix2 {
	const decomposition = decompose2x2(deformation);
	const logStretch1 = Math.log(decomposition.sigma1);
	const logStretch2 = Math.log(decomposition.sigma2);
	const logVolume = logStretch1 + logStretch2;
	const minimumLogVolume = Math.log(MIN_POWDER_DETERMINANT);
	const boundedLogVolume = Math.max(minimumLogVolume, logVolume);
	const pressure = Math.max(0, -(material.lambda + material.mu) * boundedLogVolume);
	const deviatoricLogStretch = 0.5 * (logStretch1 - logStretch2);
	const maximumDeviatoricLogStretch =
		(material.frictionCoefficient * pressure) / (2 * material.mu);
	if (
		deviatoricLogStretch <= maximumDeviatoricLogStretch &&
		logVolume >= minimumLogVolume
	) {
		return deformation;
	}
	const projectedDeviatoricLogStretch = Math.min(
		deviatoricLogStretch,
		maximumDeviatoricLogStretch
	);
	const meanLogStretch = 0.5 * boundedLogVolume;
	const projected: MpmMatrix2 = {
		m00:
			Math.exp(meanLogStretch + projectedDeviatoricLogStretch) *
				decomposition.u1.x * decomposition.v1.x +
			Math.exp(meanLogStretch - projectedDeviatoricLogStretch) *
				decomposition.u2.x * decomposition.v2.x,
		m01:
			Math.exp(meanLogStretch + projectedDeviatoricLogStretch) *
				decomposition.u1.x * decomposition.v1.y +
			Math.exp(meanLogStretch - projectedDeviatoricLogStretch) *
				decomposition.u2.x * decomposition.v2.y,
		m10:
			Math.exp(meanLogStretch + projectedDeviatoricLogStretch) *
				decomposition.u1.y * decomposition.v1.x +
			Math.exp(meanLogStretch - projectedDeviatoricLogStretch) *
				decomposition.u2.y * decomposition.v2.x,
		m11:
			Math.exp(meanLogStretch + projectedDeviatoricLogStretch) *
				decomposition.u1.y * decomposition.v1.y +
			Math.exp(meanLogStretch - projectedDeviatoricLogStretch) *
				decomposition.u2.y * decomposition.v2.y
	};
	assertComputedMatrixFinite(projected, 'projected powder deformation');
	if (determinant(projected) <= 0) {
		throw new RangeError('powder projection produced a non-positive deformation determinant');
	}
	return projected;
}

function projectLiquidDeformation(deformation: MpmMatrix2): MpmMatrix2 {
	const determinantF = determinant(deformation);
	const rotationCosineNumerator = deformation.m00 + deformation.m11;
	const rotationSineNumerator = deformation.m10 - deformation.m01;
	const rotationMagnitude = Math.hypot(
		rotationCosineNumerator,
		rotationSineNumerator
	);
	if (
		!Number.isFinite(rotationMagnitude) ||
		rotationMagnitude <= 0 ||
		!Number.isFinite(determinantF) ||
		determinantF <= 0
	) {
		throw new RangeError('liquid deformation cannot be projected to a finite rotation');
	}

	// The tension-free EOS needs a finite area-ratio bound to prevent runaway expansion.
	const boundedDeterminant = Math.min(
		MAX_LIQUID_DETERMINANT,
		Math.max(MIN_LIQUID_DETERMINANT, determinantF)
	);
	const scale = Math.sqrt(boundedDeterminant);
	const cosine = rotationCosineNumerator / rotationMagnitude;
	const sine = rotationSineNumerator / rotationMagnitude;
	const projected: MpmMatrix2 = {
		m00: scale * cosine,
		m01: -scale * sine,
		m10: scale * sine,
		m11: scale * cosine
	};
	assertComputedMatrixFinite(projected, 'projected liquid deformation');
	return projected;
}

function decompose2x2(F: MpmMatrix2): Decomposition2 {
	const c00 = F.m00 * F.m00 + F.m10 * F.m10;
	const c01 = F.m00 * F.m01 + F.m10 * F.m11;
	const c11 = F.m01 * F.m01 + F.m11 * F.m11;
	const radius = Math.hypot(c00 - c11, 2 * c01);
	const sigma1 = Math.sqrt(Math.max(0, (c00 + c11 + radius) * 0.5));
	const determinantF = determinant(F);
	const sigma2 = determinantF / sigma1;
	if (
		!Number.isFinite(sigma1) ||
		!Number.isFinite(sigma2) ||
		sigma1 <= 0 ||
		sigma2 <= 0
	) {
		throw new RangeError('snow deformation cannot be decomposed into finite positive stretches');
	}

	const angle = 0.5 * Math.atan2(2 * c01, c00 - c11);
	const cosine = Math.cos(angle);
	const sine = Math.sin(angle);
	const v1 = { x: cosine, y: sine };
	const v2 = { x: -sine, y: cosine };
	const u1 = {
		x: (F.m00 * v1.x + F.m01 * v1.y) / sigma1,
		y: (F.m10 * v1.x + F.m11 * v1.y) / sigma1
	};
	const u2 = {
		x: (F.m00 * v2.x + F.m01 * v2.y) / sigma2,
		y: (F.m10 * v2.x + F.m11 * v2.y) / sigma2
	};
	const rotation: MpmMatrix2 = {
		m00: u1.x * v1.x + u2.x * v2.x,
		m01: u1.x * v1.y + u2.x * v2.y,
		m10: u1.y * v1.x + u2.y * v2.x,
		m11: u1.y * v1.y + u2.y * v2.y
	};
	assertComputedMatrixFinite(rotation, 'snow polar rotation');
	return { sigma1, sigma2, u1, u2, v1, v2, rotation };
}

function transposeMatrix(matrix: MpmMatrix2): MpmMatrix2 {
	return {
		m00: matrix.m00,
		m01: matrix.m10,
		m10: matrix.m01,
		m11: matrix.m11
	};
}

function multiplyMatrices(left: MpmMatrix2, right: MpmMatrix2): MpmMatrix2 {
	return {
		m00: left.m00 * right.m00 + left.m01 * right.m10,
		m01: left.m00 * right.m01 + left.m01 * right.m11,
		m10: left.m10 * right.m00 + left.m11 * right.m10,
		m11: left.m10 * right.m01 + left.m11 * right.m11
	};
}

function addMatrices(left: MpmMatrix2, right: MpmMatrix2): MpmMatrix2 {
	return {
		m00: left.m00 + right.m00,
		m01: left.m01 + right.m01,
		m10: left.m10 + right.m10,
		m11: left.m11 + right.m11
	};
}

function scaleMatrix(matrix: MpmMatrix2, scale: number): MpmMatrix2 {
	return {
		m00: matrix.m00 * scale,
		m01: matrix.m01 * scale,
		m10: matrix.m10 * scale,
		m11: matrix.m11 * scale
	};
}

function determinant(matrix: MpmMatrix2): number {
	return matrix.m00 * matrix.m11 - matrix.m01 * matrix.m10;
}

function totalMomentum(
	particles: readonly MpmParticle[],
	states: readonly ParticlePhysicalState[],
	name: string
): MpmVector2 {
	let x = 0;
	let y = 0;
	for (let index = 0; index < particles.length; index++) {
		x += states[index].mass * particles[index].velocity.x;
		y += states[index].mass * particles[index].velocity.y;
	}
	assertComputedFinite(x, name);
	assertComputedFinite(y, name);
	return { x, y };
}

function totalMass(states: readonly ParticlePhysicalState[], name: string): number {
	let mass = 0;
	for (const state of states) mass += state.mass;
	assertComputedFinite(mass, name);
	return mass;
}

function confineAxis(
	nextPosition: number,
	velocity: number,
	maximum: number
): { readonly position: number; readonly velocity: number } {
	if (nextPosition < 0 || (nextPosition === 0 && velocity < 0)) {
		return { position: 0, velocity: 0 };
	}
	if (nextPosition > maximum || (nextPosition === maximum && velocity > 0)) {
		return { position: maximum, velocity: 0 };
	}
	return { position: nextPosition, velocity };
}

function applyWallCushion(velocity: number, coordinate: number, extent: number): number {
	const distance = velocity < 0 ? coordinate : extent - coordinate;
	const cushionCells = Math.min(WALL_CUSHION_CELLS, extent / 4);
	return velocity * Math.max(0, Math.min(1, distance / cushionCells));
}

function updateDeformationGradient(
	deformation: MpmMatrix2,
	velocityGradient: MpmMatrix2,
	dt: number
): MpmMatrix2 {
	const rowSum = Math.max(
		Math.abs(velocityGradient.m00) + Math.abs(velocityGradient.m01),
		Math.abs(velocityGradient.m10) + Math.abs(velocityGradient.m11)
	);
	const increment = dt * rowSum;
	assertComputedFinite(increment, 'deformation increment');
	const substeps = Math.max(1, Math.ceil(increment / MAX_DEFORMATION_INCREMENT));
	if (substeps > MAX_DEFORMATION_SUBSTEPS) {
		throw new RangeError('deformation update exceeds the adaptive substep limit');
	}

	const substepDt = dt / substeps;
	let updated = deformation;
	for (let step = 0; step < substeps; step += 1) {
		updated = multiplyMatrices(
			{
				m00: 1 + substepDt * velocityGradient.m00,
				m01: substepDt * velocityGradient.m01,
				m10: substepDt * velocityGradient.m10,
				m11: 1 + substepDt * velocityGradient.m11
			},
			updated
		);
	}
	assertComputedMatrixFinite(updated, 'deformation gradient');
	return updated;
}

function gridNodeKey(xIndex: number, yIndex: number): string {
	return `${xIndex},${yIndex}`;
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

function assertComputedMatrixFinite(matrix: MpmMatrix2, name: string): void {
	if (
		!Number.isFinite(matrix.m00) ||
		!Number.isFinite(matrix.m01) ||
		!Number.isFinite(matrix.m10) ||
		!Number.isFinite(matrix.m11)
	) {
		throw new RangeError(`${name} became non-finite`);
	}
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === 'object' && value !== null;
}
