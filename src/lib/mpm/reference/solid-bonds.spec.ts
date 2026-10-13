import { describe, expect, it } from 'vitest';
import type { MpmParticle, MpmSolidBond } from './solver.js';
import { updateMpmSolidBondDamage } from './solid-bonds.js';

const ZERO = { m00: 0, m01: 0, m10: 0, m11: 0 };
const IDENTITY = { m00: 1, m01: 0, m10: 0, m11: 1 };
const MATERIAL = { lambda: 20, mu: 10, model: 'neo-hookean' as const };

function particle(x: number, velocityX = 0): MpmParticle {
	return {
		position: { x, y: 20 },
		velocity: { x: velocityX, y: 0 },
		C: ZERO,
		F: IDENTITY,
		objectId: 4,
		material: MATERIAL
	};
}

const bond = (firstParticle: number, secondParticle: number): MpmSolidBond => ({
	firstParticle,
	secondParticle,
	restLength: 10,
	failureStrain: 0.2
});

describe('reference solid-bond damage', () => {
	it('accumulates tensile damage to the threshold and localizes it to incident particles', () => {
		const particles = [particle(10), particle(20), particle(30)];
		const result = updateMpmSolidBondDamage(particles, [
			bond(0, 1),
			bond(1, 2)
		]);

		expect(result.bonds[0]).toMatchObject({ damage: 0, broken: false });
		expect(result.particles.map(({ damage }) => damage ?? 0)).toEqual([0, 0, 0]);

		const strained = updateMpmSolidBondDamage(
			[particle(10), particle(22), particle(32)],
			[bond(0, 1), bond(1, 2)]
		);
		expect(strained.bonds[0]).toMatchObject({ damage: 1, broken: true });
		expect(strained.bonds[1]).toMatchObject({ damage: 0, broken: false });
		expect(strained.particles.map(({ damage }) => damage ?? 0)).toEqual([1, 1, 0]);
	});

	it('keeps broken edges broken after unloading and preserves particle mass and momentum', () => {
		const initiallyBroken = {
			...bond(0, 1),
			broken: true,
			damage: 1
		};
		const particles = [particle(10, 2), particle(20, -2), particle(30, 0)];
		const initialMomentum = particles.reduce((sum, value) => sum + value.velocity.x, 0);
		const result = updateMpmSolidBondDamage(
			[particle(10), particle(10.1), particle(30)],
			[initiallyBroken, bond(1, 2)]
		);

		expect(result.bonds[0]).toMatchObject({ broken: true, damage: 1 });
		expect(result.particles[0].damage).toBe(1);
		expect(result.particles).toHaveLength(particles.length);
		expect(result.particles.reduce((sum, value) => sum + value.velocity.x, 0)).toBe(
			initialMomentum
		);
	});

	it('rejects out-of-range, duplicate, mixed-object, and liquid endpoints', () => {
		expect(() =>
			updateMpmSolidBondDamage([particle(10)], [bond(0, 1)])
		).toThrow(/existing particle/);
		expect(() =>
			updateMpmSolidBondDamage([particle(10), particle(20)], [
				bond(0, 1),
				bond(1, 0)
			])
		).toThrow(/duplicates/);
		expect(() =>
			updateMpmSolidBondDamage(
				[particle(10), { ...particle(20), objectId: 5 }],
				[bond(0, 1)]
			)
		).toThrow(/one object/);
		expect(() =>
			updateMpmSolidBondDamage(
				[particle(10), { ...particle(20), material: { lambda: 10, mu: 0, model: 'water', density: 1 } }],
				[bond(0, 1)]
			)
		).toThrow(/neo-hookean solids/);
	});
});
