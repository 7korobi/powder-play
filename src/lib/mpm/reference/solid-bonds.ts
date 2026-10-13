import type {
	MpmMaterialParameters,
	MpmParticle,
	MpmSolidBond
} from './solver.js';

export interface MpmSolidBondDamageResult {
	readonly particles: readonly MpmParticle[];
	readonly bonds: readonly MpmSolidBond[];
}

export function validateMpmSolidBondGraph(
	particles: readonly MpmParticle[],
	bonds: readonly MpmSolidBond[],
	defaultMaterial?: MpmMaterialParameters
): void {
	if (!Array.isArray(bonds)) {
		throw new TypeError('fractureBonds must be an array');
	}
	const seenPairs = new Set<string>();
	for (let index = 0; index < bonds.length; index += 1) {
		const value: unknown = bonds[index];
		if (typeof value !== 'object' || value === null) {
			throw new TypeError(`fractureBonds[${index}] must be an object`);
		}
		const bond = value as MpmSolidBond;
		validateEndpoint(bond.firstParticle, index, 'firstParticle', particles.length);
		validateEndpoint(bond.secondParticle, index, 'secondParticle', particles.length);
		if (bond.firstParticle === bond.secondParticle) {
			throw new RangeError(`fractureBonds[${index}] must connect two distinct particles`);
		}
		if (!Number.isFinite(bond.restLength) || bond.restLength <= 0) {
			throw new RangeError(`fractureBonds[${index}].restLength must be positive and finite`);
		}
		if (!Number.isFinite(bond.failureStrain) || bond.failureStrain <= 0) {
			throw new RangeError(`fractureBonds[${index}].failureStrain must be positive and finite`);
		}
		const damage = bond.damage ?? 0;
		if (!Number.isFinite(damage) || damage < 0 || damage > 1) {
			throw new RangeError(`fractureBonds[${index}].damage must be between zero and one`);
		}
		if (bond.broken !== undefined && typeof bond.broken !== 'boolean') {
			throw new TypeError(`fractureBonds[${index}].broken must be a boolean`);
		}
		const first = particles[bond.firstParticle];
		const second = particles[bond.secondParticle];
		if ((first.objectId ?? 0) !== (second.objectId ?? 0)) {
			throw new RangeError(`fractureBonds[${index}] must connect particles of one object`);
		}
		if (
			first.bondedTriangleVertex !== undefined ||
			second.bondedTriangleVertex !== undefined
		) {
			throw new RangeError(`fractureBonds[${index}] cannot reuse bonded triangle vertices`);
		}
		if (
			materialModel(first.material, defaultMaterial) !== 'neo-hookean' ||
			materialModel(second.material, defaultMaterial) !== 'neo-hookean'
		) {
			throw new RangeError(`fractureBonds[${index}] endpoints must be neo-hookean solids`);
		}
		const low = Math.min(bond.firstParticle, bond.secondParticle);
		const high = Math.max(bond.firstParticle, bond.secondParticle);
		const pairKey = `${low}:${high}`;
		if (seenPairs.has(pairKey)) {
			throw new RangeError(`fractureBonds[${index}] duplicates an existing particle pair`);
		}
		seenPairs.add(pairKey);
	}
}

export function updateMpmSolidBondDamage(
	particles: readonly MpmParticle[],
	bonds: readonly MpmSolidBond[],
	defaultMaterial?: MpmMaterialParameters
): MpmSolidBondDamageResult {
	validateMpmSolidBondGraph(particles, bonds, defaultMaterial);
	if (bonds.length === 0) return { particles, bonds };

	const nextDamageByParticle = new Map<number, number>();
	const nextBonds = bonds.map((bond) => {
		const first = particles[bond.firstParticle].position;
		const second = particles[bond.secondParticle].position;
		const distance = Math.hypot(second.x - first.x, second.y - first.y);
		if (!Number.isFinite(distance)) {
			throw new RangeError('fracture bond length became non-finite');
		}
		const tensileStrain = Math.max(0, (distance - bond.restLength) / bond.restLength);
		const wasBroken = bond.broken ?? false;
		const broken = wasBroken || tensileStrain >= bond.failureStrain;
		const damage = broken
			? 1
			: Math.max(bond.damage ?? 0, Math.min(1, tensileStrain / bond.failureStrain));
		for (const particleIndex of [bond.firstParticle, bond.secondParticle]) {
			nextDamageByParticle.set(
				particleIndex,
				Math.max(nextDamageByParticle.get(particleIndex) ?? 0, damage)
			);
		}
		return { ...bond, damage, broken };
	});

	const nextParticles = particles.map((particle, index) => {
		const damage = nextDamageByParticle.get(index);
		if (damage === undefined || damage <= (particle.damage ?? 0)) return particle;
		return { ...particle, damage };
	});
	return { particles: nextParticles, bonds: nextBonds };
}

function validateEndpoint(
	value: number,
	bondIndex: number,
	name: string,
	particleCount: number
): void {
	if (!Number.isSafeInteger(value) || value < 0 || value >= particleCount) {
		throw new RangeError(
			`fractureBonds[${bondIndex}].${name} must identify an existing particle`
		);
	}
}

function materialModel(
	particleMaterial: MpmMaterialParameters | undefined,
	defaultMaterial: MpmMaterialParameters | undefined
): string {
	return particleMaterial?.model ?? defaultMaterial?.model ?? 'neo-hookean';
}
