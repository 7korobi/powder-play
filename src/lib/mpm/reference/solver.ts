import {
	normalizeMpmConstitutiveModel,
	MPM_LIQUID_CONTACT_APPROACH_SPEED_MULTIPLIER,
	MPM_LIQUID_CONTACT_RELAXATION_SECONDS,
	MPM_LIQUID_CONTACT_SEPARATION_FRACTION,
	MPM_LIQUID_STABLE_ORDER_EPSILON,
	MPM_TRIANGLE_FLUID_TANGENTIAL_DRAG_RATE,
	SNOW_PLASTICITY
} from '../materials.js';
import {
	updateMpmSolidBondDamage,
	validateMpmSolidBondGraph
} from './solid-bonds.js';

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
	readonly damage?: number;
	readonly plasticVolume?: number;
	readonly volume?: number;
	readonly material?: MpmMaterialParameters;
}

export interface MpmSolidBond {
	readonly firstParticle: number;
	readonly secondParticle: number;
	readonly restLength: number;
	readonly failureStrain: number;
	readonly damage?: number;
	readonly broken?: boolean;
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
	readonly bondReferenceDt?: number;
	readonly fractureBonds?: readonly MpmSolidBond[];
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
	readonly fractureBonds?: readonly MpmSolidBond[];
}

interface ValidatedConfig {
	readonly dt: number;
	readonly grid: MpmGridParameters;
	readonly particleMass: number;
	readonly particleVolume: number;
	readonly bondRestLength: number | undefined;
	readonly bondStiffness: number;
	readonly bondReferenceDt: number;
	readonly fractureBonds: readonly MpmSolidBond[] | undefined;
	readonly material: MpmMaterialParameters;
	readonly gravity: MpmVector2;
	readonly domainWidth: number;
	readonly domainHeight: number;
	readonly gradientScale: number;
	readonly stressScale: number;
	readonly gridNodeWidth: number;
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
	readonly nodeKey: number;
	readonly signedDistance: number;
	readonly normal: MpmVector2;
	readonly barycentric: readonly [number, number, number];
	impulseX: number;
	impulseY: number;
}

interface TriangleDistanceSample {
	signedDistance: number;
	normalX: number;
	normalY: number;
	barycentric: [number, number, number];
}

interface TriangleField {
	readonly nodes: ReadonlyMap<number, TriangleContactField>;
	readonly nodesByBody: ReadonlyMap<number, readonly TriangleContactField[]>;
}

interface ParticlePhysicalState {
	readonly mass: number;
	readonly volume: number;
	readonly material: MpmMaterialParameters;
}

interface ParticleStencil {
	baseX: number;
	baseY: number;
	fx: number;
	fy: number;
	weightsX: [number, number, number];
	weightsY: [number, number, number];
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
const MIN_NEO_HOOKEAN_STRETCH = 0.05;
const MAX_NEO_HOOKEAN_STRETCH = 20;
const MAX_NEO_HOOKEAN_STRETCH_SQUARED =
	MAX_NEO_HOOKEAN_STRETCH * MAX_NEO_HOOKEAN_STRETCH;
const MAX_DEFORMATION_INCREMENT = 0.25;
const MAX_DEFORMATION_SUBSTEPS = 64;
export const DEFAULT_BONDED_TRIANGLE_STIFFNESS = 0.35;
const BONDED_TRIANGLE_SOLVER_ITERATIONS = 2;
const TRIANGLE_CONTACT_SKIN_CELLS = 0.1;
const TRIANGLE_CONTACT_IMPULSE_RELAXATION = 0.25;
const TRIANGLE_FIELD_BAND_CELLS = 2;
const GRID_CONTACT_REACH_CELLS = 0.75;

export function stepMpm(
	particles: readonly MpmParticle[],
	config: MpmStepConfig
): MpmStepResult {
	const validated = validateConfig(config);
	validateParticles(particles, validated);
	if (validated.fractureBonds !== undefined) {
		validateMpmSolidBondGraph(particles, validated.fractureBonds, validated.material);
	}
	const bondedBodyStarts = collectBondedBodyStarts(particles, validated);
	const particleStates = particles.map((particle) => resolveParticleState(particle, validated));

	const inputMass = totalMass(particleStates, 'input mass');
	assertComputedFinite(inputMass, 'input mass');
	const inputMomentum = totalMomentum(particles, particleStates, 'input momentum');
	const grid = new Map<number, GridNode>();
	const dx = validated.grid.cellSize;
	const inputStencil = createParticleStencil();

	for (let index = 0; index < particles.length; index++) {
		const particle = particles[index];
		const state = particleStates[index];
		updateParticleStencil(particle.position, dx, inputStencil);
		const stencil = inputStencil;
		const plasticVolume = particle.plasticVolume ?? 1;
		const stressTimesFt = constitutiveStressTimesFt(
			particle.F,
			plasticVolume,
			state.material,
			particle.damage ?? 0
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
				const key = gridNodeKey(xIndex, yIndex, validated.gridNodeWidth);
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
				} else if (state.material.model === 'oil') {
					node.oilMass += weight * state.mass;
					node.oilMomentumX += weight * particleMomentumX;
					node.oilMomentumY += weight * particleMomentumY;
					node.oilVolume += weight * liquidVolume;
				} else {
					node.mass += weight * state.mass;
					node.momentumX += weight * particleMomentumX;
					node.momentumY += weight * particleMomentumY;
				}
			}
		}
	}

	let gridMass = 0;
	let gridMomentumX = 0;
	let gridMomentumY = 0;
	for (const node of grid.values()) {
		assertGridNodeFinite(node);
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
			gridNodeKey(
				node.xIndex,
				node.yIndex,
				validated.gridNodeWidth
			)
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

	const outputStencil = createParticleStencil();
	const projectionStencil = createParticleStencil();
	const outputParticles = particles.map((particle, index) => {
		const state = particleStates[index];
		updateParticleStencil(particle.position, dx, outputStencil);
		const stencil = outputStencil;
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

				const node = grid.get(
					gridNodeKey(
						stencil.baseX + ox,
						stencil.baseY + oy,
						validated.gridNodeWidth
					)
				);
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
		} else if (state.material.model === 'neo-hookean') {
			const stretchTrace =
				trialF.m00 * trialF.m00 +
				trialF.m01 * trialF.m01 +
				trialF.m10 * trialF.m10 +
				trialF.m11 * trialF.m11;
			if (!Number.isFinite(stretchTrace)) {
				throw new RangeError('Neo-Hookean deformation stretch is outside the supported range');
			}
			if (
				jacobian < MIN_NEO_HOOKEAN_STRETCH * Math.sqrt(stretchTrace) ||
				stretchTrace > MAX_NEO_HOOKEAN_STRETCH_SQUARED
			) {
				F = projectNeoHookeanDeformation(trialF);
			}
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
					validated,
					projectionStencil
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
			...(particle.damage === undefined ? {} : { damage: particle.damage }),
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
	const fractureResult =
		validated.fractureBonds === undefined
			? undefined
			: updateMpmSolidBondDamage(
					outputParticles,
					validated.fractureBonds,
					validated.material
				);
	const finalParticles = fractureResult?.particles ?? outputParticles;

	const outputMass = totalMass(particleStates, 'output mass');
	assertComputedFinite(outputMass, 'output mass');
	const outputMomentum = totalMomentum(
		finalParticles,
		particleStates,
		'output momentum'
	);

	return {
		particles: finalParticles,
		diagnostics: {
			inputMass,
			gridMass,
			outputMass,
			inputMomentum,
			gridMomentum: { x: gridMomentumX, y: gridMomentumY },
			outputMomentum
		},
		...(fractureResult === undefined
			? {}
			: { fractureBonds: fractureResult.bonds })
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
	const bondReferenceDt =
		raw.bondReferenceDt === undefined ? dt : raw.bondReferenceDt;
	const fractureBonds =
		raw.fractureBonds === undefined
			? undefined
			: validateSolidBondArray(raw.fractureBonds);
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
	assertFiniteNumber(bondReferenceDt, 'bondReferenceDt');
	assertFiniteNumber(lambda, 'material.lambda');
	assertFiniteNumber(mu, 'material.mu');
	assertVector(gravity, 'gravity');

	if (dt <= 0) throw new RangeError('dt must be greater than zero');
	if (bondReferenceDt <= 0) {
		throw new RangeError('bondReferenceDt must be greater than zero');
	}
	if (!Number.isSafeInteger(cellsX) || cellsX < 1 || cellsX >= Number.MAX_SAFE_INTEGER) {
		throw new RangeError('grid.cellsX must be a positive safe integer');
	}
	if (!Number.isSafeInteger(cellsY) || cellsY < 1 || cellsY >= Number.MAX_SAFE_INTEGER) {
		throw new RangeError('grid.cellsY must be a positive safe integer');
	}
	const gridNodeWidth = cellsX + 3;
	if (
		!Number.isSafeInteger(gridNodeWidth) ||
		!Number.isSafeInteger(gridNodeWidth * (cellsY + 3))
	) {
		throw new RangeError('grid dimensions exceed the safe node-index range');
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
		bondReferenceDt,
		fractureBonds,
		material,
		gravity: { x: gravity.x, y: gravity.y },
		domainWidth,
		domainHeight,
		gradientScale,
		stressScale,
		gridNodeWidth
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
		if (value.damage !== undefined) {
			assertFiniteNumber(value.damage, `particles[${index}].damage`);
			if (value.damage < 0 || value.damage > 1) {
				throw new RangeError(`particles[${index}].damage must be between zero and one`);
			}
			if (particleMaterial(particle, config.material).model !== 'neo-hookean') {
				throw new RangeError(
					`particles[${index}].damage is only valid for neo-hookean solids`
				);
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
	const nodes = new Map<number, TriangleContactField>();
	const nodesByBody = new Map<number, TriangleContactField[]>();
	const candidatesByBody = new Map<number, TriangleContactField[]>();
	if (bodyStarts.length === 0) return { nodes, nodesByBody };

	const { cellsX, cellsY, cellSize } = config.grid;
	const maxDistance = cellSize * TRIANGLE_FIELD_BAND_CELLS;
	const sample: TriangleDistanceSample = {
		signedDistance: 0,
		normalX: 0,
		normalY: 0,
		barycentric: [1, 0, 0]
	};
	for (const bodyStart of bodyStarts) {
		const triangle = getTriangleVertices(particles, bodyStart);
		const minimumX = Math.min(triangle[0].x, triangle[1].x, triangle[2].x);
		const maximumX = Math.max(triangle[0].x, triangle[1].x, triangle[2].x);
		const minimumY = Math.min(triangle[0].y, triangle[1].y, triangle[2].y);
		const maximumY = Math.max(triangle[0].y, triangle[1].y, triangle[2].y);
		const minimumXIndex = Math.max(
			-1,
			Math.floor((minimumX - maxDistance) / cellSize) - 1
		);
		const maximumXIndex = Math.min(
			cellsX + 1,
			Math.ceil((maximumX + maxDistance) / cellSize) + 1
		);
		const minimumYIndex = Math.max(
			-1,
			Math.floor((minimumY - maxDistance) / cellSize) - 1
		);
		const maximumYIndex = Math.min(
			cellsY + 1,
			Math.ceil((maximumY + maxDistance) / cellSize) + 1
		);
		const minimumFieldX = minimumX - maxDistance;
		const maximumFieldX = maximumX + maxDistance;
		const minimumFieldY = minimumY - maxDistance;
		const maximumFieldY = maximumY + maxDistance;

		// The extra grid cell makes the integer range conservative; the exact bounds remain authoritative.
		for (let yIndex = minimumYIndex; yIndex <= maximumYIndex; yIndex += 1) {
			const pointY = yIndex * cellSize;
			if (pointY < minimumFieldY || pointY > maximumFieldY) continue;
			for (let xIndex = minimumXIndex; xIndex <= maximumXIndex; xIndex += 1) {
				const pointX = xIndex * cellSize;
				if (pointX < minimumFieldX || pointX > maximumFieldX) continue;
				triangleDistanceSample(pointX, pointY, triangle, sample);
				if (sample.signedDistance > maxDistance) continue;
				const key = gridNodeKey(xIndex, yIndex, config.gridNodeWidth);
				const nearest = nodes.get(key);
				if (nearest && sample.signedDistance >= nearest.signedDistance) {
					continue;
				}
				const contact: TriangleContactField = {
					bodyStart,
					nodeKey: key,
					signedDistance: sample.signedDistance,
					normal: { x: sample.normalX, y: sample.normalY },
					barycentric: [
						sample.barycentric[0],
						sample.barycentric[1],
						sample.barycentric[2]
					],
					impulseX: 0,
					impulseY: 0
				};
				nodes.set(key, contact);
				let bodyCandidates = candidatesByBody.get(bodyStart);
				if (!bodyCandidates) {
					bodyCandidates = [];
					candidatesByBody.set(bodyStart, bodyCandidates);
				}
				bodyCandidates.push(contact);
			}
		}
	}

	for (const bodyStart of bodyStarts) {
		const bodyCandidates = candidatesByBody.get(bodyStart);
		if (!bodyCandidates) continue;
		const bodyNodes: TriangleContactField[] = [];
		for (const contact of bodyCandidates) {
			if (nodes.get(contact.nodeKey) === contact) bodyNodes.push(contact);
		}
		if (bodyNodes.length > 0) nodesByBody.set(bodyStart, bodyNodes);
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
	pointX: number,
	pointY: number,
	vertices: readonly [MpmVector2, MpmVector2, MpmVector2],
	sample: TriangleDistanceSample
): void {
	const [a, b, c] = vertices;
	const abX = b.x - a.x;
	const abY = b.y - a.y;
	const acX = c.x - a.x;
	const acY = c.y - a.y;
	const signedArea = abX * acY - abY * acX;
	if (!Number.isFinite(signedArea) || Math.abs(signedArea) <= 1e-10) {
		throw new RangeError('bonded triangle became degenerate');
	}
	const bcX = c.x - b.x;
	const bcY = c.y - b.y;
	const caX = a.x - c.x;
	const caY = a.y - c.y;
	const apX = pointX - a.x;
	const apY = pointY - a.y;
	const bpX = pointX - b.x;
	const bpY = pointY - b.y;
	const cpX = pointX - c.x;
	const cpY = pointY - c.y;
	const crossAB = abX * apY - abY * apX;
	const crossBC = bcX * bpY - bcY * bpX;
	const crossCA = caX * cpY - caY * cpX;
	const inside =
		signedArea > 0
			? crossAB >= -1e-10 && crossBC >= -1e-10 && crossCA >= -1e-10
			: crossAB <= 1e-10 && crossBC <= 1e-10 && crossCA <= 1e-10;

	let closestDistanceSquared = Number.POSITIVE_INFINITY;
	let closestNormalX = 0;
	let closestNormalY = 0;
	for (let edgeIndex = 0; edgeIndex < 3; edgeIndex += 1) {
		const startX = edgeIndex === 0 ? a.x : edgeIndex === 1 ? b.x : c.x;
		const startY = edgeIndex === 0 ? a.y : edgeIndex === 1 ? b.y : c.y;
		const endX = edgeIndex === 0 ? b.x : edgeIndex === 1 ? c.x : a.x;
		const endY = edgeIndex === 0 ? b.y : edgeIndex === 1 ? c.y : a.y;
		const directionX = endX - startX;
		const directionY = endY - startY;
		const lengthSquared =
			directionX * directionX + directionY * directionY;
		const offsetX = pointX - startX;
		const offsetY = pointY - startY;
		const t = Math.max(
			0,
			Math.min(
				1,
				(offsetX * directionX + offsetY * directionY) / lengthSquared
			)
		);
		const candidateX = startX + directionX * t;
		const candidateY = startY + directionY * t;
		const differenceX = pointX - candidateX;
		const differenceY = pointY - candidateY;
		const distanceSquared =
			differenceX * differenceX + differenceY * differenceY;
		if (distanceSquared >= closestDistanceSquared) continue;
		closestDistanceSquared = distanceSquared;
		if (edgeIndex === 0) {
			sample.barycentric[0] = 1 - t;
			sample.barycentric[1] = t;
			sample.barycentric[2] = 0;
		} else if (edgeIndex === 1) {
			sample.barycentric[0] = 0;
			sample.barycentric[1] = 1 - t;
			sample.barycentric[2] = t;
		} else {
			sample.barycentric[0] = t;
			sample.barycentric[1] = 0;
			sample.barycentric[2] = 1 - t;
		}
		if (distanceSquared > 1e-20 && !inside) {
			const distance = Math.sqrt(distanceSquared);
			const inverseDistance = 1 / distance;
			closestNormalX = differenceX * inverseDistance;
			closestNormalY = differenceY * inverseDistance;
		} else {
			const orientation = signedArea > 0 ? 1 : -1;
			const outwardX = orientation * directionY;
			const outwardY = -orientation * directionX;
			const edgeLength = Math.sqrt(lengthSquared);
			const inverseEdgeLength = 1 / edgeLength;
			closestNormalX = outwardX * inverseEdgeLength;
			closestNormalY = outwardY * inverseEdgeLength;
		}
	}
	const distance = Math.sqrt(closestDistanceSquared);
	if (
		!Number.isFinite(distance) ||
		!Number.isFinite(closestNormalX) ||
		!Number.isFinite(closestNormalY)
	) {
		throw new RangeError('bonded triangle contact field became non-finite');
	}
	sample.signedDistance = inside ? -distance : distance;
	sample.normalX = closestNormalX;
	sample.normalY = closestNormalY;
}

function solveLiquidGridContact(
	node: GridNode,
	grid: ReadonlyMap<number, GridNode>,
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

	const waterGradient = volumeGradient(
		grid,
		node.xIndex,
		node.yIndex,
		'water',
		config
	);
	const oilGradient = volumeGradient(
		grid,
		node.xIndex,
		node.yIndex,
		'oil',
		config
	);
	const gradient = {
		x: oilGradient.x - waterGradient.x,
		y: oilGradient.y - waterGradient.y
	};
	const gradientLength = Math.hypot(gradient.x, gradient.y);
	if (!Number.isFinite(gradientLength) || gradientLength <= 1e-12) return;
	let normal = scaleVector(gradient, 1 / gradientLength);
	const waterDensity = node.waterMass / node.waterVolume;
	const oilDensity = node.oilMass / node.oilVolume;
	const densityScale = Math.max(waterDensity, oilDensity);
	const densityContrast = (oilDensity - waterDensity) / densityScale;
	const gravityMagnitude = Math.hypot(config.gravity.x, config.gravity.y);
	const gravityAlignment =
		gravityMagnitude > 0
			? (normal.x * config.gravity.x + normal.y * config.gravity.y) /
				gravityMagnitude
			: 0;
	assertComputedFinite(waterDensity, 'water contact density');
	assertComputedFinite(oilDensity, 'oil contact density');
	assertComputedFinite(densityContrast, 'liquid contact density contrast');
	assertComputedFinite(gravityMagnitude, 'gravity magnitude');
	assertComputedFinite(gravityAlignment, 'liquid contact gravity alignment');
	if (densityScale <= 0 || waterDensity <= 0 || oilDensity <= 0) {
		throw new RangeError('liquid contact densities must be positive');
	}
	// Reverse an unstable interface normal so denser liquid is driven with gravity.
	if (densityContrast * gravityAlignment < -MPM_LIQUID_STABLE_ORDER_EPSILON) {
		normal = scaleVector(normal, -1);
	}
	const relativeNormalVelocity =
		(node.oilVelocityX - node.waterVelocityX) * normal.x +
		(node.oilVelocityY - node.waterVelocityY) * normal.y;
	const overlapFraction = Math.min(
		1,
		(2 * Math.min(node.waterVolume, node.oilVolume)) /
			(node.waterVolume + node.oilVolume)
	);
	const minimumSeparationVelocity =
		Math.max(
			(MPM_LIQUID_CONTACT_SEPARATION_FRACTION *
				overlapFraction *
				config.grid.cellSize) /
				MPM_LIQUID_CONTACT_RELAXATION_SECONDS,
			Math.max(-relativeNormalVelocity, 0) *
				MPM_LIQUID_CONTACT_APPROACH_SPEED_MULTIPLIER
		);
	if (relativeNormalVelocity >= minimumSeparationVelocity) return;
	const impulse =
		(minimumSeparationVelocity - relativeNormalVelocity) /
		(1 / node.waterMass + 1 / node.oilMass);
	node.waterVelocityX -= (impulse / node.waterMass) * normal.x;
	node.waterVelocityY -= (impulse / node.waterMass) * normal.y;
	node.oilVelocityX += (impulse / node.oilMass) * normal.x;
	node.oilVelocityY += (impulse / node.oilMass) * normal.y;
}

function volumeGradient(
	grid: ReadonlyMap<number, GridNode>,
	xIndex: number,
	yIndex: number,
	phase: 'water' | 'oil',
	config: ValidatedConfig
): MpmVector2 {
	const volumeAt = (x: number, y: number): number => {
		if (
			x < -1 ||
			x > config.grid.cellsX + 1 ||
			y < -1 ||
			y > config.grid.cellsY + 1
		) {
			return 0;
		}
		const node = grid.get(gridNodeKey(x, y, config.gridNodeWidth));
		return node === undefined
			? 0
			: phase === 'water'
				? node.waterVolume
				: node.oilVolume;
	};
	const scale = 1 / (2 * config.grid.cellSize * config.grid.cellSize);
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
		const response = resolveFluidTriangleContact(
			node.waterMass,
			{ x: node.waterVelocityX, y: node.waterVelocityY },
			surfaceVelocity,
			field.normal,
			inverseBodyMass,
			config.dt
		);
		node.waterVelocityX = response.fluidVelocity.x;
		node.waterVelocityY = response.fluidVelocity.y;
		field.impulseX += response.bodyImpulse.x;
		field.impulseY += response.bodyImpulse.y;
	}
	if (node.oilMass > 0) {
		const response = resolveFluidTriangleContact(
			node.oilMass,
			{ x: node.oilVelocityX, y: node.oilVelocityY },
			surfaceVelocity,
			field.normal,
			inverseBodyMass,
			config.dt
		);
		node.oilVelocityX = response.fluidVelocity.x;
		node.oilVelocityY = response.fluidVelocity.y;
		field.impulseX += response.bodyImpulse.x;
		field.impulseY += response.bodyImpulse.y;
	}
}

function resolveFluidTriangleContact(
	fluidMass: number,
	fluidVelocity: MpmVector2,
	surfaceVelocity: MpmVector2,
	normal: MpmVector2,
	inverseBodyMass: number,
	dt: number
): { readonly fluidVelocity: MpmVector2; readonly bodyImpulse: MpmVector2 } {
	const inverseEffectiveMass = 1 / fluidMass + inverseBodyMass;
	assertComputedFinite(inverseEffectiveMass, 'triangle contact inverse effective mass');
	if (inverseEffectiveMass <= 0) {
		throw new RangeError('triangle contact inverse effective mass must be positive');
	}

	const relativeVelocity = subtractVector(fluidVelocity, surfaceVelocity);
	const relativeNormalVelocity = dot2(relativeVelocity, normal);
	const normalImpulseMagnitude =
		relativeNormalVelocity < 0
			? -relativeNormalVelocity / inverseEffectiveMass
			: 0;
	const normalImpulse = scaleVector(normal, normalImpulseMagnitude);
	const tangentialVelocity = subtractVector(
		relativeVelocity,
		scaleVector(normal, relativeNormalVelocity)
	);
	const tangentialDamping =
		1 - Math.exp(-MPM_TRIANGLE_FLUID_TANGENTIAL_DRAG_RATE * dt);
	const tangentialImpulse = scaleVector(
		tangentialVelocity,
		-tangentialDamping / inverseEffectiveMass
	);
	const impulseOnFluid = addVector(normalImpulse, tangentialImpulse);
	const nextFluidVelocity = addVector(
		fluidVelocity,
		scaleVector(impulseOnFluid, 1 / fluidMass)
	);
	const bodyImpulse = scaleVector(impulseOnFluid, -1);
	assertComputedFinite(nextFluidVelocity.x, 'triangle contact fluid velocity x');
	assertComputedFinite(nextFluidVelocity.y, 'triangle contact fluid velocity y');
	assertComputedFinite(bodyImpulse.x, 'triangle contact body impulse x');
	assertComputedFinite(bodyImpulse.y, 'triangle contact body impulse y');

	return { fluidVelocity: nextFluidVelocity, bodyImpulse };
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
	config: ValidatedConfig,
	stencil: ParticleStencil
): MpmVector2 {
	if (triangleField.nodes.size === 0) return position;
	updateParticleStencil(position, config.grid.cellSize, stencil);
	const band = config.grid.cellSize * TRIANGLE_FIELD_BAND_CELLS;
	let signedDistance = 0;
	let closestBodyStart: number | undefined;
	let closestDistance = Number.POSITIVE_INFINITY;
	let selectedNormal: MpmVector2 | undefined;
	for (let oy = 0; oy < 3; oy += 1) {
		for (let ox = 0; ox < 3; ox += 1) {
			const weight = stencil.weightsX[ox] * stencil.weightsY[oy];
			const field = triangleField.nodes.get(
				gridNodeKey(
					stencil.baseX + ox,
					stencil.baseY + oy,
					config.gridNodeWidth
				)
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
				gridNodeKey(
					stencil.baseX + ox,
					stencil.baseY + oy,
					config.gridNodeWidth
				)
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
	const stiffness = calculateSubstepBondStiffness(
		config.bondStiffness,
		config.dt,
		config.bondReferenceDt
	);
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

function calculateSubstepBondStiffness(
	stiffness: number,
	dt: number,
	referenceDt: number
): number {
	const stepRatio = dt / referenceDt;
	const substepStiffness = Math.min(stiffness * stepRatio * stepRatio, 1);
	if (!Number.isFinite(substepStiffness)) {
		throw new RangeError('bonded triangle substep stiffness is outside the supported range');
	}
	return Math.max(0, Math.min(1, substepStiffness));
}

function resolveBondedTriangleContacts(
	inputParticles: readonly MpmParticle[],
	outputParticles: MpmParticle[],
	bodyStarts: readonly number[],
	config: ValidatedConfig
): void {
	if (bodyStarts.length < 2) return;
	const skin = config.grid.cellSize * TRIANGLE_CONTACT_SKIN_CELLS;
	const contactsByBody: {
		otherIndex: number;
		normal: MpmVector2;
		penetration: number;
	}[][] = bodyStarts.map(() => []);
	const triangles = bodyStarts.map((bodyStart) =>
		getTriangleVertices(inputParticles, bodyStart)
	);
	const velocities = bodyStarts.map((bodyStart) =>
		triangleMeanVelocity(outputParticles, bodyStart)
	);
	for (let firstIndex = 0; firstIndex < bodyStarts.length; firstIndex += 1) {
		const firstStart = bodyStarts[firstIndex];
		for (
			let secondIndex = firstIndex + 1;
			secondIndex < bodyStarts.length;
			secondIndex += 1
		) {
			const secondStart = bodyStarts[secondIndex];
			if (
				inputParticles[firstStart].objectId ===
				inputParticles[secondStart].objectId
			) {
				continue;
			}
			const contact = trianglePairContact(
				triangles[firstIndex],
				triangles[secondIndex],
				skin
			);
			if (!contact) continue;
			contactsByBody[firstIndex].push({
				otherIndex: secondIndex,
				normal: contact.normal,
				penetration: contact.penetration
			});
			contactsByBody[secondIndex].push({
				otherIndex: firstIndex,
				normal: scaleVector(contact.normal, -1),
				penetration: contact.penetration
			});
		}
	}

	const bodyMass = 3 * config.particleMass;
	for (let bodyIndex = 0; bodyIndex < bodyStarts.length; bodyIndex += 1) {
		const bodyStart = bodyStarts[bodyIndex];
		let impulseX = 0;
		let impulseY = 0;
		for (const contact of contactsByBody[bodyIndex]) {
			const contactWeight =
				1 /
				Math.max(
					contactsByBody[bodyIndex].length,
					contactsByBody[contact.otherIndex].length
				);
			const otherVelocity = velocities[contact.otherIndex];
			const relativeNormalVelocity =
				(velocities[bodyIndex].x - otherVelocity.x) * contact.normal.x +
				(velocities[bodyIndex].y - otherVelocity.y) * contact.normal.y;
			if (relativeNormalVelocity < 0) {
				const impulse =
					-relativeNormalVelocity / (2 / bodyMass);
				const impulseWeight =
					contactWeight * TRIANGLE_CONTACT_IMPULSE_RELAXATION;
				impulseX +=
					(impulseWeight * impulse * contact.normal.x) / bodyMass;
				impulseY +=
					(impulseWeight * impulse * contact.normal.y) / bodyMass;
			}
		}

		for (let vertex = 0; vertex < 3; vertex += 1) {
			const index = bodyStart + vertex;
			const particle = outputParticles[index];
			const boundedX = confineAxis(
				particle.position.x,
				particle.velocity.x + impulseX,
				config.domainWidth
			);
			const boundedY = confineAxis(
				particle.position.y,
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
	let contactAxisX = 0;
	let contactAxisY = 0;
	let hasContactAxis = false;
	for (let triangleIndex = 0; triangleIndex < 2; triangleIndex += 1) {
		const triangle = triangleIndex === 0 ? triangleA : triangleB;
		for (let edgeIndex = 0; edgeIndex < 3; edgeIndex += 1) {
			const start = triangle[edgeIndex];
			const end = triangle[(edgeIndex + 1) % 3];
			const edgeX = end.x - start.x;
			const edgeY = end.y - start.y;
			const edgeLength = Math.hypot(edgeX, edgeY);
			if (edgeLength <= 1e-10) return undefined;
			const axisX = edgeY / edgeLength;
			const axisY = -edgeX / edgeLength;
			const a0 = triangleA[0].x * axisX + triangleA[0].y * axisY;
			const a1 = triangleA[1].x * axisX + triangleA[1].y * axisY;
			const a2 = triangleA[2].x * axisX + triangleA[2].y * axisY;
			const b0 = triangleB[0].x * axisX + triangleB[0].y * axisY;
			const b1 = triangleB[1].x * axisX + triangleB[1].y * axisY;
			const b2 = triangleB[2].x * axisX + triangleB[2].y * axisY;
			const minA = Math.min(a0, a1, a2);
			const maxA = Math.max(a0, a1, a2);
			const minB = Math.min(b0, b1, b2);
			const maxB = Math.max(b0, b1, b2);
			const gap = Math.max(
				minB - maxA,
				minA - maxB
			);
			if (gap > 0) {
				separated = true;
				if (gap > largestGap) {
					largestGap = gap;
					contactAxisX = axisX;
					contactAxisY = axisY;
					hasContactAxis = true;
				}
			} else if (!separated && -gap < smallestOverlap) {
				smallestOverlap = -gap;
				contactAxisX = axisX;
				contactAxisY = axisY;
				hasContactAxis = true;
			}
		}
	}
	if (separated && largestGap > skin) return undefined;
	if (!hasContactAxis) return undefined;
	const directionX =
		(triangleA[0].x + triangleA[1].x + triangleA[2].x) / 3 -
		(triangleB[0].x + triangleB[1].x + triangleB[2].x) / 3;
	const directionY =
		(triangleA[0].y + triangleA[1].y + triangleA[2].y) / 3 -
		(triangleB[0].y + triangleB[1].y + triangleB[2].y) / 3;
	const normalSign =
		directionX * contactAxisX + directionY * contactAxisY >= 0 ? 1 : -1;
	const normal = {
		x: normalSign * contactAxisX,
		y: normalSign * contactAxisY
	};
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
	assertComputedFinite(node.xIndex, 'grid node xIndex');
	assertComputedFinite(node.yIndex, 'grid node yIndex');
	assertComputedFinite(node.mass, 'grid node mass');
	assertComputedFinite(node.momentumX, 'grid node momentumX');
	assertComputedFinite(node.momentumY, 'grid node momentumY');
	assertComputedFinite(node.velocityX, 'grid node velocityX');
	assertComputedFinite(node.velocityY, 'grid node velocityY');
	assertComputedFinite(node.waterMass, 'grid node waterMass');
	assertComputedFinite(node.waterMomentumX, 'grid node waterMomentumX');
	assertComputedFinite(node.waterMomentumY, 'grid node waterMomentumY');
	assertComputedFinite(node.waterVelocityX, 'grid node waterVelocityX');
	assertComputedFinite(node.waterVelocityY, 'grid node waterVelocityY');
	assertComputedFinite(node.waterVolume, 'grid node waterVolume');
	assertComputedFinite(node.oilMass, 'grid node oilMass');
	assertComputedFinite(node.oilMomentumX, 'grid node oilMomentumX');
	assertComputedFinite(node.oilMomentumY, 'grid node oilMomentumY');
	assertComputedFinite(node.oilVelocityX, 'grid node oilVelocityX');
	assertComputedFinite(node.oilVelocityY, 'grid node oilVelocityY');
	assertComputedFinite(node.oilVolume, 'grid node oilVolume');
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

function createParticleStencil(): ParticleStencil {
	return {
		baseX: 0,
		baseY: 0,
		fx: 0,
		fy: 0,
		weightsX: [0, 0, 0],
		weightsY: [0, 0, 0]
	};
}

function updateParticleStencil(
	position: MpmVector2,
	cellSize: number,
	stencil: ParticleStencil
): void {
	const gridX = position.x / cellSize;
	const gridY = position.y / cellSize;
	stencil.baseX = Math.floor(gridX - 0.5);
	stencil.baseY = Math.floor(gridY - 0.5);
	stencil.fx = gridX - stencil.baseX;
	stencil.fy = gridY - stencil.baseY;
	stencil.weightsX[0] = 0.5 * (1.5 - stencil.fx) ** 2;
	stencil.weightsX[1] = 0.75 - (stencil.fx - 1) ** 2;
	stencil.weightsX[2] = 0.5 * (stencil.fx - 0.5) ** 2;
	stencil.weightsY[0] = 0.5 * (1.5 - stencil.fy) ** 2;
	stencil.weightsY[1] = 0.75 - (stencil.fy - 1) ** 2;
	stencil.weightsY[2] = 0.5 * (stencil.fy - 0.5) ** 2;
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
	material: MpmMaterialParameters,
	damage = 0
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
	return scaleMatrix(
		multiplyMatrices(firstPiola, transposeMatrix(F)),
		Math.max(0, 1 - damage)
	);
}

function validateSolidBondArray(value: unknown): readonly MpmSolidBond[] {
	if (!Array.isArray(value)) {
		throw new TypeError('fractureBonds must be an array');
	}
	return value as readonly MpmSolidBond[];
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

function projectNeoHookeanDeformation(deformation: MpmMatrix2): MpmMatrix2 {
	const decomposition = decompose2x2(deformation);
	const sigma1 = Math.max(
		MIN_NEO_HOOKEAN_STRETCH,
		Math.min(MAX_NEO_HOOKEAN_STRETCH, decomposition.sigma1)
	);
	const sigma2 = Math.max(
		MIN_NEO_HOOKEAN_STRETCH,
		Math.min(MAX_NEO_HOOKEAN_STRETCH, decomposition.sigma2)
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
	assertComputedMatrixFinite(projected, 'projected Neo-Hookean deformation');
	if (determinant(projected) <= 0) {
		throw new RangeError('Neo-Hookean projection produced a non-positive deformation determinant');
	}
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
		x: -u1.y,
		y: u1.x
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

function gridNodeKey(
	xIndex: number,
	yIndex: number,
	gridNodeWidth: number
): number {
	return (yIndex + 1) * gridNodeWidth + (xIndex + 1);
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
