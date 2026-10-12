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

export function stepMpm(
	particles: readonly MpmParticle[],
	config: MpmStepConfig
): MpmStepResult {
	const validated = validateConfig(config);
	validateParticles(particles, validated);
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
						velocityY: 0
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
				node.mass += weight * state.mass;
				node.momentumX += weight * particleMomentumX;
				node.momentumY += weight * particleMomentumY;
				assertComputedFinite(node.mass, 'grid node mass');
				assertComputedFinite(node.momentumX, 'grid node momentum');
				assertComputedFinite(node.momentumY, 'grid node momentum');
			}
		}
	}

	let gridMass = 0;
	let gridMomentumX = 0;
	let gridMomentumY = 0;
	for (const node of grid.values()) {
		gridMass += node.mass;
		gridMomentumX += node.momentumX;
		gridMomentumY += node.momentumY;
	}
	assertComputedFinite(gridMass, 'grid mass');
	assertComputedFinite(gridMomentumX, 'grid momentum');
	assertComputedFinite(gridMomentumY, 'grid momentum');

	for (const node of grid.values()) {
		if (node.mass <= 0) {
			throw new RangeError('grid nodes must have positive mass');
		}
		let velocityX = node.momentumX / node.mass + validated.dt * validated.gravity.x;
		let velocityY = node.momentumY / node.mass + validated.dt * validated.gravity.y;
		assertComputedFinite(velocityX, 'grid velocity');
		assertComputedFinite(velocityY, 'grid velocity');

		if ((node.xIndex <= 0 && velocityX < 0) ||
			(node.xIndex >= validated.grid.cellsX && velocityX > 0)) {
			velocityX = 0;
		}
		if ((node.yIndex <= 0 && velocityY < 0) ||
			(node.yIndex >= validated.grid.cellsY && velocityY > 0)) {
			velocityY = 0;
		}
		node.velocityX = velocityX;
		node.velocityY = velocityY;
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
				velocityX += weight * node.velocityX;
				velocityY += weight * node.velocityY;
				c00 += weight * node.velocityX * dposX;
				c01 += weight * node.velocityX * dposY;
				c10 += weight * node.velocityY * dposX;
				c11 += weight * node.velocityY * dposY;
			}
		}

		const C: MpmMatrix2 = {
			m00: c00 * validated.gradientScale,
			m01: c01 * validated.gradientScale,
			m10: c10 * validated.gradientScale,
			m11: c11 * validated.gradientScale
		};
		const trialF = multiplyMatrices(
			{
				m00: 1 + validated.dt * C.m00,
				m01: validated.dt * C.m01,
				m10: validated.dt * C.m10,
				m11: 1 + validated.dt * C.m11
			},
			particle.F
		);
		assertComputedFinite(velocityX, 'particle velocity');
		assertComputedFinite(velocityY, 'particle velocity');
		assertComputedMatrixFinite(C, 'particle affine velocity');
		assertComputedMatrixFinite(trialF, 'deformation gradient');
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
		}

		const nextX = particle.position.x + validated.dt * velocityX;
		const nextY = particle.position.y + validated.dt * velocityY;
		assertComputedFinite(nextX, 'particle position');
		assertComputedFinite(nextY, 'particle position');
		const boundedX = confineAxis(nextX, velocityX, validated.domainWidth);
		const boundedY = confineAxis(nextY, velocityY, validated.domainHeight);

		return {
			position: { x: boundedX.position, y: boundedY.position },
			velocity: { x: boundedX.velocity, y: boundedY.velocity },
			C,
			F,
			objectId: particle.objectId,
			plasticVolume: nextPlasticVolume,
			volume: state.volume,
			material: state.material
		};
	});

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
	const lambda = material.lambda;
	const mu = material.mu;
	const gravity = raw.gravity;

	assertFiniteNumber(dt, 'dt');
	assertFiniteNumber(cellsX, 'grid.cellsX');
	assertFiniteNumber(cellsY, 'grid.cellsY');
	assertFiniteNumber(cellSize, 'grid.cellSize');
	assertFiniteNumber(particleMass, 'particleMass');
	assertFiniteNumber(particleVolume, 'particleVolume');
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
	const pressure = Math.max(0, -(material.lambda + material.mu) * logVolume);
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
	const pressure = Math.max(0, -(material.lambda + material.mu) * logVolume);
	const deviatoricLogStretch = 0.5 * (logStretch1 - logStretch2);
	const maximumDeviatoricLogStretch =
		(material.frictionCoefficient * pressure) / (2 * material.mu);
	const projectedDeviatoricLogStretch = Math.min(
		deviatoricLogStretch,
		maximumDeviatoricLogStretch
	);
	const meanLogStretch = 0.5 * logVolume;
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
