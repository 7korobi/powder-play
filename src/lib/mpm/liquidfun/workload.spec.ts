import { describe, expect, it } from 'vitest';
import {
	createLiquidFunGate4Workload,
	LIQUIDFUN_BOX2D_DOMAIN_UNITS,
	LIQUIDFUN_BOX2D_GRAVITY_UNITS_PER_SECOND_SQUARED,
	LIQUIDFUN_BOX2D_UNITS_PER_GATE4_UNIT,
	LIQUIDFUN_ELASTIC_BALL_CENTER_X_METERS,
	LIQUIDFUN_ELASTIC_BALL_CENTER_Y_METERS,
	LIQUIDFUN_ELASTIC_BALL_COUNT,
	LIQUIDFUN_ELASTIC_BALL_PARTICLE_COUNT,
	LIQUIDFUN_ELASTIC_BALL_PARTICLES_PER_GROUP,
	LIQUIDFUN_ELASTIC_STRENGTH,
	LIQUIDFUN_FLUID_PARTICLE_COUNT,
	LIQUIDFUN_FLUID_PARTICLE_ROWS,
	LIQUIDFUN_FLUID_TYPE_PARTICLE_COUNT,
	LIQUIDFUN_GATE4_ACCELERATION_UNITS_PER_METER_PER_SECOND_SQUARED,
	LIQUIDFUN_GATE4_DOMAIN,
	LIQUIDFUN_GATE4_GRAVITY,
	LIQUIDFUN_GATE4_UNITS_PER_METER,
	LIQUIDFUN_INK_PARTICLE_COUNT,
	LIQUIDFUN_OIL_PARTICLE_COUNT,
	LIQUIDFUN_PARTICLE_RADIUS_BOX2D_UNITS,
	LIQUIDFUN_PARTICLE_RADIUS_METERS,
	LIQUIDFUN_PARTICLE_SPACING_GATE4_UNITS,
	LIQUIDFUN_PARTICLE_SPACING_METERS,
	LIQUIDFUN_PHYSICAL_DOMAIN_METERS,
	LIQUIDFUN_PRESSURE_STRENGTH,
	LIQUIDFUN_SCENE_SEED,
	LIQUIDFUN_SHARED_PARTICLE_DENSITY,
	LIQUIDFUN_SHARED_PARTICLE_MASS,
	LIQUIDFUN_SOIL_PARTICLE_COUNT,
	LIQUIDFUN_SPRING_STRENGTH,
	LIQUIDFUN_SPRING_VINE_CENTER_X_SPACING_METERS,
	LIQUIDFUN_SPRING_VINE_CENTER_X_START_METERS,
	LIQUIDFUN_SPRING_VINE_CENTER_Y_METERS,
	LIQUIDFUN_SPRING_VINE_COUNT,
	LIQUIDFUN_SPRING_VINE_LENGTH_METERS,
	LIQUIDFUN_SPRING_VINE_LINK_COUNT,
	LIQUIDFUN_SPRING_VINE_PARTICLE_COUNT,
	LIQUIDFUN_SPRING_VINE_PARTICLES_PER_GROUP,
	LIQUIDFUN_SPRING_VINE_SPAN_METERS,
	LIQUIDFUN_TOTAL_PARTICLE_COUNT,
	LIQUIDFUN_WATER_PARTICLE_COUNT,
	mapBox2DPositionToGate4,
	mapBox2DVelocityToGate4,
	mapGate4AccelerationToBox2D,
	mapGate4PositionToBox2D,
	mapGate4VelocityToBox2D,
	summarizeSamples
} from './workload.js';
import {
	createMpmGate4Scene,
	GATE4_FRACTURE_BOND_COUNT,
	GATE4_FRACTURE_FAILURE_STRAIN,
	GATE4_FRACTURE_OBJECT_ID,
	GATE4_FRACTURE_PARTICLE_COUNT,
	GATE4_OIL_PARTICLE_COUNT,
	GATE4_PARTICLE_COUNT,
	GATE4_OIL_PARTICLE_COLUMNS,
	GATE4_OIL_PARTICLE_ROWS,
	GATE4_WATER_PARTICLE_COLUMNS,
	GATE4_WATER_PARTICLE_COUNT,
	GATE4_WATER_PARTICLE_ROWS
} from '../scenes/gate4.js';

describe('LiquidFun mixed-material workload', () => {
	it('maps Gate 4 coordinates, velocity, and acceleration to Box2D units', () => {
		expect(LIQUIDFUN_GATE4_UNITS_PER_METER).toBe(20_000);
		expect(LIQUIDFUN_GATE4_ACCELERATION_UNITS_PER_METER_PER_SECOND_SQUARED).toBe(
			100
		);
		expect(LIQUIDFUN_GATE4_DOMAIN.width).toBeCloseTo(2_800);
		expect(LIQUIDFUN_GATE4_DOMAIN.height).toBe(1_500);
		expect(LIQUIDFUN_PHYSICAL_DOMAIN_METERS.width).toBeCloseTo(0.14);
		expect(LIQUIDFUN_PHYSICAL_DOMAIN_METERS.height).toBeCloseTo(0.075);
		expect(LIQUIDFUN_BOX2D_DOMAIN_UNITS.width).toBeCloseTo(140);
		expect(LIQUIDFUN_BOX2D_DOMAIN_UNITS.height).toBeCloseTo(75);
		expect(LIQUIDFUN_GATE4_GRAVITY).toEqual({ x: 0, y: 90 });
		expect(LIQUIDFUN_BOX2D_GRAVITY_UNITS_PER_SECOND_SQUARED).toEqual({
			x: 0,
			y: -900
		});

		const gate4Position = { x: 100, y: 640 };
		const gate4Velocity = { x: 32, y: -64 };
		const box2dPosition = mapGate4PositionToBox2D(gate4Position);
		const box2dVelocity = mapGate4VelocityToBox2D(gate4Velocity);
		expect(box2dPosition).toEqual({ x: 5, y: 43 });
		expect(box2dVelocity).toEqual({ x: 1.6, y: 3.2 });
		expect(mapBox2DPositionToGate4(box2dPosition)).toEqual(gate4Position);
		expect(mapBox2DVelocityToGate4(box2dVelocity)).toEqual(gate4Velocity);
		expect(mapGate4AccelerationToBox2D({ x: 0, y: 980 })).toEqual({
			x: 0,
			y: -9_800
		});
	});

	it('uses one shared millimeter-scale particle system and solver strengths', () => {
		expect(LIQUIDFUN_PARTICLE_RADIUS_METERS).toBe(0.001);
		expect(LIQUIDFUN_PARTICLE_RADIUS_BOX2D_UNITS).toBe(1);
		expect(LIQUIDFUN_PARTICLE_SPACING_METERS).toBe(0.0015);
		expect(LIQUIDFUN_PARTICLE_SPACING_GATE4_UNITS).toBe(30);
		expect(LIQUIDFUN_SHARED_PARTICLE_MASS).toBe(0.01);
		expect(LIQUIDFUN_SHARED_PARTICLE_DENSITY * 1.5 ** 2).toBeCloseTo(0.01);
		expect(LIQUIDFUN_PRESSURE_STRENGTH).toBe(1);
		expect(LIQUIDFUN_ELASTIC_STRENGTH).toBe(0.25);
		expect(LIQUIDFUN_SPRING_STRENGTH).toBe(0.25);
	});

	it('builds the requested four fluid types from the seeded Gate 4 source', () => {
		const workload = createLiquidFunGate4Workload();
		const scene = createMpmGate4Scene(LIQUIDFUN_SCENE_SEED);
		const sourceWater = scene.particles.filter(
			(particle) => particle.material?.model === 'water'
		);
		const sourceOil = scene.particles.filter(
			(particle) => particle.material?.model === 'oil'
		);
		const sourceFractureNodes = scene.particles.filter(
			(particle) => particle.objectId === GATE4_FRACTURE_OBJECT_ID
		);
		const fluidGroups = [
			workload.waterParticles,
			workload.oilParticles,
			workload.inkParticles,
			workload.soilParticles
		];

		expect(workload.seed).toBe(0x4d504d34);
		expect(workload.sourceMpmNodeCount).toBe(GATE4_PARTICLE_COUNT);
		expect(workload.sourceMpmNodeCount).toBe(3_674);
		expect(sourceWater).toHaveLength(GATE4_WATER_PARTICLE_COUNT);
		expect(sourceOil).toHaveLength(GATE4_OIL_PARTICLE_COUNT);
		expect(workload.waterParticles).toHaveLength(LIQUIDFUN_WATER_PARTICLE_COUNT);
		expect(workload.oilParticles).toHaveLength(LIQUIDFUN_OIL_PARTICLE_COUNT);
		expect(workload.inkParticles).toHaveLength(LIQUIDFUN_INK_PARTICLE_COUNT);
		expect(workload.soilParticles).toHaveLength(LIQUIDFUN_SOIL_PARTICLE_COUNT);
		expect(LIQUIDFUN_FLUID_TYPE_PARTICLE_COUNT).toBe(800);
		expect(LIQUIDFUN_FLUID_PARTICLE_ROWS).toBe(43);
		expect(LIQUIDFUN_FLUID_PARTICLE_COUNT).toBe(3_200);
		expect(
			fluidGroups.flat().every(
				({ position, velocity }) =>
					Number.isFinite(position.x) &&
					Number.isFinite(position.y) &&
					Number.isFinite(velocity.x) &&
					Number.isFinite(velocity.y)
			)
		).toBe(true);
		const particleRadiusGate4Units =
			LIQUIDFUN_PARTICLE_RADIUS_METERS * LIQUIDFUN_GATE4_UNITS_PER_METER;
		for (const particles of fluidGroups) {
			expect(
				new Set(particles.map(({ position }) => `${position.x},${position.y}`))
					.size
			).toBe(particles.length);
			for (const { position } of particles) {
				expect(position.x).toBeGreaterThanOrEqual(particleRadiusGate4Units);
				expect(position.x).toBeLessThanOrEqual(
					LIQUIDFUN_GATE4_DOMAIN.width - particleRadiusGate4Units
				);
				expect(position.y).toBeGreaterThanOrEqual(particleRadiusGate4Units);
				expect(position.y).toBeLessThanOrEqual(
					LIQUIDFUN_GATE4_DOMAIN.height - particleRadiusGate4Units
				);
			}
		}

		expect(GATE4_WATER_PARTICLE_COLUMNS * GATE4_WATER_PARTICLE_ROWS).toBe(
			GATE4_WATER_PARTICLE_COUNT
		);
		expect(GATE4_OIL_PARTICLE_COLUMNS * GATE4_OIL_PARTICLE_ROWS).toBe(
			GATE4_OIL_PARTICLE_COUNT
		);
		expect(sourceFractureNodes).toHaveLength(GATE4_FRACTURE_PARTICLE_COUNT);
	});

	it('places ten independent elastic balls with twelve particles each inside the domain', () => {
		const workload = createLiquidFunGate4Workload();
		expect(workload.elasticBallGroups).toHaveLength(LIQUIDFUN_ELASTIC_BALL_COUNT);
		expect(LIQUIDFUN_ELASTIC_BALL_COUNT).toBe(10);
		expect(LIQUIDFUN_ELASTIC_BALL_PARTICLES_PER_GROUP).toBe(12);
		expect(LIQUIDFUN_ELASTIC_BALL_PARTICLE_COUNT).toBe(120);

		for (const group of workload.elasticBallGroups) {
			expect(group.positions).toHaveLength(
				LIQUIDFUN_ELASTIC_BALL_PARTICLES_PER_GROUP
			);
			const center = group.positions.reduce(
				(total, position) => ({
					x: total.x + position.x / group.positions.length,
					y: total.y + position.y / group.positions.length
				}),
				{ x: 0, y: 0 }
			);
			const expectedCenterX = LIQUIDFUN_ELASTIC_BALL_CENTER_X_METERS[group.groupIndex];
			expect(expectedCenterX).toBeDefined();
			expect(center.x / LIQUIDFUN_GATE4_UNITS_PER_METER).toBeCloseTo(
				expectedCenterX ?? Number.NaN
			);
			expect(center.y / LIQUIDFUN_GATE4_UNITS_PER_METER).toBeCloseTo(
				LIQUIDFUN_ELASTIC_BALL_CENTER_Y_METERS
			);
			for (const position of group.positions) {
				expect(position.x).toBeGreaterThanOrEqual(
					LIQUIDFUN_PARTICLE_RADIUS_METERS * LIQUIDFUN_GATE4_UNITS_PER_METER
				);
				expect(position.x).toBeLessThanOrEqual(
					LIQUIDFUN_GATE4_DOMAIN.width -
						LIQUIDFUN_PARTICLE_RADIUS_METERS * LIQUIDFUN_GATE4_UNITS_PER_METER
				);
				expect(position.y).toBeGreaterThanOrEqual(
					LIQUIDFUN_PARTICLE_RADIUS_METERS * LIQUIDFUN_GATE4_UNITS_PER_METER
				);
				expect(position.y).toBeLessThanOrEqual(
					LIQUIDFUN_GATE4_DOMAIN.height -
						LIQUIDFUN_PARTICLE_RADIUS_METERS * LIQUIDFUN_GATE4_UNITS_PER_METER
				);
			}
		}
	});

	it('creates ten 70 mm spring vines without overlap or wall penetration', () => {
		const workload = createLiquidFunGate4Workload();
		expect(workload.springVines).toHaveLength(LIQUIDFUN_SPRING_VINE_COUNT);
		expect(LIQUIDFUN_SPRING_VINE_COUNT).toBe(10);
		expect(LIQUIDFUN_SPRING_VINE_PARTICLES_PER_GROUP).toBe(48);
		expect(LIQUIDFUN_SPRING_VINE_PARTICLE_COUNT).toBe(480);
		expect(LIQUIDFUN_SPRING_VINE_LINK_COUNT).toBe(470);
		expect(LIQUIDFUN_SPRING_VINE_SPAN_METERS).toBeGreaterThanOrEqual(
			LIQUIDFUN_SPRING_VINE_LENGTH_METERS
		);
		expect(LIQUIDFUN_SPRING_VINE_CENTER_X_SPACING_METERS).toBeGreaterThan(
			LIQUIDFUN_PARTICLE_RADIUS_METERS * 2
		);

		for (const [groupIndex, group] of workload.springVines.entries()) {
			expect(group.positions).toHaveLength(
				LIQUIDFUN_SPRING_VINE_PARTICLES_PER_GROUP
			);
			const first = group.positions[0];
			const last = group.positions.at(-1);
			expect(first).toBeDefined();
			expect(last).toBeDefined();
			if (!first || !last) continue;
			expect(
				Math.abs(last.y - first.y) / LIQUIDFUN_GATE4_UNITS_PER_METER
			).toBeCloseTo(LIQUIDFUN_SPRING_VINE_SPAN_METERS);
			expect(
				first.x / LIQUIDFUN_GATE4_UNITS_PER_METER
			).toBeCloseTo(
				LIQUIDFUN_SPRING_VINE_CENTER_X_START_METERS +
					groupIndex * LIQUIDFUN_SPRING_VINE_CENTER_X_SPACING_METERS
			);
			expect(
				first.y / LIQUIDFUN_GATE4_UNITS_PER_METER +
					LIQUIDFUN_SPRING_VINE_SPAN_METERS / 2
			).toBeCloseTo(LIQUIDFUN_SPRING_VINE_CENTER_Y_METERS);
			for (let index = 1; index < group.positions.length; index += 1) {
				const previous = group.positions[index - 1];
				const current = group.positions[index];
				expect(previous).toBeDefined();
				expect(current).toBeDefined();
				if (!previous || !current) continue;
				expect(current.x).toBe(previous.x);
				expect(current.y - previous.y).toBe(LIQUIDFUN_PARTICLE_SPACING_GATE4_UNITS);
			}
			for (const position of group.positions) {
				expect(position.x).toBeGreaterThanOrEqual(
					LIQUIDFUN_PARTICLE_RADIUS_METERS * LIQUIDFUN_GATE4_UNITS_PER_METER
				);
				expect(position.x).toBeLessThanOrEqual(
					LIQUIDFUN_GATE4_DOMAIN.width -
						LIQUIDFUN_PARTICLE_RADIUS_METERS * LIQUIDFUN_GATE4_UNITS_PER_METER
				);
				expect(position.y).toBeGreaterThanOrEqual(
					LIQUIDFUN_PARTICLE_RADIUS_METERS * LIQUIDFUN_GATE4_UNITS_PER_METER
				);
				expect(position.y).toBeLessThanOrEqual(
					LIQUIDFUN_GATE4_DOMAIN.height -
						LIQUIDFUN_PARTICLE_RADIUS_METERS * LIQUIDFUN_GATE4_UNITS_PER_METER
				);
			}
		}
	});

	it('retains the seeded fracture network and its 12 percent break threshold', () => {
		const workload = createLiquidFunGate4Workload();
		expect(workload.fractureNodes).toHaveLength(GATE4_FRACTURE_PARTICLE_COUNT);
		expect(workload.fractureNodes).toHaveLength(24);
		expect(workload.fractureBonds).toHaveLength(GATE4_FRACTURE_BOND_COUNT);
		expect(workload.fractureBonds).toHaveLength(53);
		expect(
			workload.fractureBonds.every(
				(bond) =>
					bond.failureStrain === GATE4_FRACTURE_FAILURE_STRAIN &&
					bond.firstNodeIndex >= 0 &&
					bond.firstNodeIndex < workload.fractureNodes.length &&
					bond.secondNodeIndex >= 0 &&
					bond.secondNodeIndex < workload.fractureNodes.length &&
					Number.isFinite(bond.restLengthBox2DUnits) &&
					bond.restLengthBox2DUnits > 0
			)
		).toBe(true);
	});

	it('summarizes valid benchmark samples and rejects invalid input', () => {
		expect(summarizeSamples([4, 1, 3, 2])).toEqual({
			median: 2,
			p95: 3,
			min: 1,
			max: 4,
			samples: 4
		});
		expect(() => summarizeSamples([])).toThrow(RangeError);
		expect(() => summarizeSamples([1, -1])).toThrow(RangeError);
		expect(() => summarizeSamples([Number.POSITIVE_INFINITY])).toThrow(
			RangeError
		);
	});

	it('accounts for the complete 3,800-particle workload', () => {
		expect(LIQUIDFUN_TOTAL_PARTICLE_COUNT).toBe(
			LIQUIDFUN_FLUID_PARTICLE_COUNT +
				LIQUIDFUN_ELASTIC_BALL_PARTICLE_COUNT +
				LIQUIDFUN_SPRING_VINE_PARTICLE_COUNT
		);
		expect(LIQUIDFUN_TOTAL_PARTICLE_COUNT).toBe(3_800);
		expect(
			LIQUIDFUN_BOX2D_UNITS_PER_GATE4_UNIT *
				LIQUIDFUN_GATE4_UNITS_PER_METER
		).toBe(1_000);
	});
});
