import { describe, expect, it } from 'vitest';
import {
	createLiquidFunGate4Workload,
	LIQUIDFUN_BOX2D_DOMAIN_UNITS,
	LIQUIDFUN_BOX2D_GRAVITY_UNITS_PER_SECOND_SQUARED,
	LIQUIDFUN_GATE4_ACCELERATION_UNITS_PER_METER_PER_SECOND_SQUARED,
	LIQUIDFUN_GATE4_DOMAIN,
	LIQUIDFUN_GATE4_GRAVITY,
	LIQUIDFUN_GATE4_UNITS_PER_METER,
	LIQUIDFUN_FLUID_PARTICLE_COUNT_MULTIPLIER,
	LIQUIDFUN_OIL_PARTICLE_COUNT,
	LIQUIDFUN_PARTICLE_RADIUS_BOX2D_UNITS,
	LIQUIDFUN_PEBBLE_SIZE_SCALE,
	LIQUIDFUN_PEBBLE_EDGE_LENGTH_BOX2D_UNITS,
	LIQUIDFUN_PEBBLE_EDGE_LENGTH_METERS,
	LIQUIDFUN_PEBBLE_MASS,
	LIQUIDFUN_PARTICLE_RADIUS_METERS,
	LIQUIDFUN_SHARED_PARTICLE_DENSITY,
	LIQUIDFUN_SHARED_PARTICLE_MASS,
	LIQUIDFUN_PHYSICAL_DOMAIN_METERS,
	LIQUIDFUN_SCENE_SEED,
	LIQUIDFUN_SOLID_NODE_RADIUS_METERS,
	LIQUIDFUN_WATER_PARTICLE_COUNT,
	mapBox2DPositionToGate4,
	mapBox2DVelocityToGate4,
	mapGate4AccelerationToBox2D,
	mapGate4PositionToBox2D,
	mapGate4VelocityToBox2D,
	scaleLiquidFunPebbleLocalVertices,
	summarizeSamples
} from './workload.js';
import {
	createMpmGate4Scene,
	GATE4_FRACTURE_BOND_COUNT,
	GATE4_FRACTURE_FAILURE_STRAIN,
	GATE4_FRACTURE_OBJECT_ID,
	GATE4_FRACTURE_PARTICLE_COUNT,
	GATE4_LIQUID_PARTICLE_SPACING,
	GATE4_PARTICLE_COUNT,
	GATE4_OIL_PARTICLE_COLUMNS,
	GATE4_PEBBLE_COUNT,
	GATE4_PEBBLE_PARTICLE_COUNT,
	GATE4_OIL_PARTICLE_COUNT,
	GATE4_OIL_PARTICLE_ROWS,
	GATE4_WATER_PARTICLE_COLUMNS,
	GATE4_WATER_PARTICLE_COUNT,
	GATE4_WATER_PARTICLE_ROWS
} from '../scenes/gate4.js';

describe('LiquidFun Gate 4 workload mapping', () => {
	it('converts Gate 4 screen/world coordinates and velocity to Box2D y-up units', () => {
		expect(LIQUIDFUN_GATE4_UNITS_PER_METER).toBe(20_000);
		expect(LIQUIDFUN_GATE4_ACCELERATION_UNITS_PER_METER_PER_SECOND_SQUARED).toBe(100);
		expect(LIQUIDFUN_GATE4_DOMAIN).toEqual({ width: 2_800, height: 1_500 });
		expect(LIQUIDFUN_PHYSICAL_DOMAIN_METERS).toEqual({ width: 0.14, height: 0.075 });
		expect(LIQUIDFUN_BOX2D_DOMAIN_UNITS).toEqual({ width: 140, height: 75 });
		expect(LIQUIDFUN_GATE4_GRAVITY).toEqual({ x: 0, y: 90 });
		expect(LIQUIDFUN_BOX2D_GRAVITY_UNITS_PER_SECOND_SQUARED).toEqual({
			x: 0,
			y: -900
		});

		const screenPosition = { x: 100, y: 640 };
		const screenVelocity = { x: 32, y: -64 };
		const box2dPosition = mapGate4PositionToBox2D(screenPosition);
		const box2dVelocity = mapGate4VelocityToBox2D(screenVelocity);
		expect(box2dPosition).toEqual({ x: 5, y: 43 });
		expect(box2dVelocity).toEqual({ x: 1.6, y: 3.2 });
		expect(mapBox2DPositionToGate4(box2dPosition)).toEqual(screenPosition);
		expect(mapBox2DVelocityToGate4(box2dVelocity)).toEqual(screenVelocity);
		expect(mapGate4AccelerationToBox2D({ x: 0, y: 980 })).toEqual({
			x: 0,
			y: -9_800
		});
	});

	it('calibrates particle and pebble masses at the requested millimeter scale', () => {
		expect(LIQUIDFUN_PARTICLE_RADIUS_METERS).toBe(0.0005);
		expect(LIQUIDFUN_PARTICLE_RADIUS_BOX2D_UNITS).toBe(0.5);
		expect(LIQUIDFUN_SHARED_PARTICLE_MASS).toBe(0.05);
		expect(LIQUIDFUN_SHARED_PARTICLE_DENSITY * 0.75 ** 2).toBeCloseTo(0.05);
		expect(LIQUIDFUN_PEBBLE_EDGE_LENGTH_METERS).toBe(0.0015);
		expect(LIQUIDFUN_PEBBLE_EDGE_LENGTH_BOX2D_UNITS).toBe(1.5);
		expect(LIQUIDFUN_PEBBLE_MASS).toBe(0.045);
	});

	it('preserves source counts and solid topology while expanding liquid flags', () => {
		const workload = createLiquidFunGate4Workload();
		const scene = createMpmGate4Scene(LIQUIDFUN_SCENE_SEED);
		const waterParticles = scene.particles.filter(
			(particle) => particle.material?.model === 'water'
		);
		const oilParticles = scene.particles.filter(
			(particle) => particle.material?.model === 'oil'
		);
		const pebbleNodes = scene.particles.filter(
			(particle) => particle.bondedTriangleVertex !== undefined
		);
		const fractureNodes = scene.particles.filter(
			(particle) => particle.objectId === GATE4_FRACTURE_OBJECT_ID
		);

		expect(workload.seed).toBe(0x4d504d34);
		expect(workload.seed).toBe(LIQUIDFUN_SCENE_SEED);
		expect(workload.sourceMpmNodeCount).toBe(GATE4_PARTICLE_COUNT);
		expect(workload.sourceMpmNodeCount).toBe(3_674);
		expect(workload.mappedMpmNodeCount).toBe(GATE4_PARTICLE_COUNT);
		expect(workload.mappedMpmNodeCount).toBe(3_674);
		expect(LIQUIDFUN_FLUID_PARTICLE_COUNT_MULTIPLIER).toBe(3);
		expect(GATE4_WATER_PARTICLE_COUNT).toBe(2_000);
		expect(GATE4_OIL_PARTICLE_COUNT).toBe(1_500);
		expect(LIQUIDFUN_WATER_PARTICLE_COUNT).toBe(6_000);
		expect(LIQUIDFUN_OIL_PARTICLE_COUNT).toBe(4_500);
		expect(waterParticles).toHaveLength(GATE4_WATER_PARTICLE_COUNT);
		expect(oilParticles).toHaveLength(GATE4_OIL_PARTICLE_COUNT);
		expect(workload.waterParticles).toHaveLength(LIQUIDFUN_WATER_PARTICLE_COUNT);
		expect(
			workload.waterParticles.every((particle) => particle.material?.model === 'water')
		).toBe(true);
		expect(workload.oilParticles).toHaveLength(LIQUIDFUN_OIL_PARTICLE_COUNT);
		expect(
			workload.oilParticles.every((particle) => particle.material?.model === 'oil')
		).toBe(true);
		expect(workload.pebbleNodes).toHaveLength(GATE4_PEBBLE_PARTICLE_COUNT);
		expect(workload.pebbleNodes).toEqual(pebbleNodes);
		expect(workload.pebbleGroups).toHaveLength(GATE4_PEBBLE_COUNT);
		expect(workload.pebbleGroups).toHaveLength(50);
		for (const group of workload.pebbleGroups) {
			const nodes = group.nodeIndices.map(
				(nodeIndex) => workload.pebbleNodes[nodeIndex]
			);
			expect(nodes.every((node) => node?.objectId === group.objectId)).toBe(true);
			expect(nodes.map((node) => node?.bondedTriangleVertex)).toEqual([0, 1, 2]);
		}
		expect(workload.fractureNodes).toHaveLength(GATE4_FRACTURE_PARTICLE_COUNT);
		expect(workload.fractureNodes).toHaveLength(24);
		expect(workload.fractureNodes).toEqual(fractureNodes);
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
					bond.restLengthBox2DUnits > 0
			)
		).toBe(true);
	});

	it('expands each fluid footprint threefold while retaining source particle spacing', () => {
		const workload = createLiquidFunGate4Workload();
		const scene = createMpmGate4Scene(LIQUIDFUN_SCENE_SEED);
		const sourceWaterParticles = scene.particles.filter(
			(particle) => particle.material?.model === 'water'
		);
		const sourceOilParticles = scene.particles.filter(
			(particle) => particle.material?.model === 'oil'
		);
		const sourceFractureParticles = scene.particles.filter(
			(particle) => particle.objectId === GATE4_FRACTURE_OBJECT_ID
		);
		const expectExpandedGrid = (
			sourceParticles: readonly {
				position: { x: number; y: number };
				velocity: { x: number; y: number };
			}[],
			targetParticles: readonly {
				position: { x: number; y: number };
				velocity: { x: number; y: number };
			}[],
			sourceColumns: number,
			sourceRows: number,
			targetParticleCount: number,
			targetRows: number,
			verticalOffset: number
		) => {
			const first = sourceParticles[0];
			if (!first) {
				throw new Error('The Gate 4 fluid source grid is incomplete.');
			}
			const targetColumns = Math.ceil(targetParticleCount / targetRows);
			expect(targetParticles).toHaveLength(targetParticleCount);
			expect(targetParticleCount).toBe(
				sourceParticles.length * LIQUIDFUN_FLUID_PARTICLE_COUNT_MULTIPLIER
			);
			expect(Math.min(...targetParticles.map(({ position }) => position.x))).toBeCloseTo(
				first.position.x
			);
			expect(Math.max(...targetParticles.map(({ position }) => position.x))).toBeCloseTo(
				first.position.x + (targetColumns - 1) * GATE4_LIQUID_PARTICLE_SPACING
			);
			expect(
				first.position.x +
					targetColumns * GATE4_LIQUID_PARTICLE_SPACING
			).toBeLessThanOrEqual(LIQUIDFUN_GATE4_DOMAIN.width);
			expect(Math.min(...targetParticles.map(({ position }) => position.y))).toBeCloseTo(
				first.position.y + verticalOffset
			);
			expect(Math.max(...targetParticles.map(({ position }) => position.y))).toBeCloseTo(
				first.position.y +
					verticalOffset +
					(targetRows - 1) * GATE4_LIQUID_PARTICLE_SPACING
			);
			expect(
				new Set(
					targetParticles.map(({ position }) => `${position.x},${position.y}`)
				).size
			).toBe(targetParticles.length);
			for (let row = 0; row < targetRows; row += 1) {
				const rowStartIndex = Math.floor(
					(row * targetParticleCount) / targetRows
				);
				const rowEndIndex = Math.floor(
					((row + 1) * targetParticleCount) / targetRows
				);
				const particlesInRow = rowEndIndex - rowStartIndex;
				const columnOffset =
					((targetColumns - particlesInRow) * GATE4_LIQUID_PARTICLE_SPACING) /
					2;
				for (let column = 0; column < particlesInRow; column += 1) {
					const targetParticle =
						targetParticles[rowStartIndex + column];
					const sourceParticle =
						sourceParticles[row * sourceColumns + (column % sourceColumns)];
					if (!targetParticle || !sourceParticle) {
						throw new Error('The expanded Gate 4 fluid grid is incomplete.');
					}
					expect(targetParticle.position).toEqual({
						x:
							first.position.x +
							columnOffset +
							column * GATE4_LIQUID_PARTICLE_SPACING,
						y:
							first.position.y +
							verticalOffset +
							row * GATE4_LIQUID_PARTICLE_SPACING
					});
					expect(targetParticle.velocity).toEqual(sourceParticle.velocity);
				}
			}
		};
		const waterTargetRows = GATE4_WATER_PARTICLE_ROWS;
		const fluidVerticalOffset = -GATE4_LIQUID_PARTICLE_SPACING / 2;

		expectExpandedGrid(
			sourceWaterParticles,
			workload.waterParticles,
			GATE4_WATER_PARTICLE_COLUMNS,
			GATE4_WATER_PARTICLE_ROWS,
			LIQUIDFUN_WATER_PARTICLE_COUNT,
			waterTargetRows,
			fluidVerticalOffset
		);
		expectExpandedGrid(
			sourceOilParticles,
			workload.oilParticles,
			GATE4_OIL_PARTICLE_COLUMNS,
			GATE4_OIL_PARTICLE_ROWS,
			LIQUIDFUN_OIL_PARTICLE_COUNT,
			GATE4_OIL_PARTICLE_ROWS,
			fluidVerticalOffset
		);
		expect(LIQUIDFUN_PARTICLE_RADIUS_METERS).toBe(0.0005);
		const fractureTopY = Math.min(
			...sourceFractureParticles.map(({ position }) => position.y)
		);
		const waterBottomY = Math.max(
			...workload.waterParticles.map(({ position }) => position.y)
		);
		expect(fractureTopY - waterBottomY).toBeGreaterThan(
			(LIQUIDFUN_PARTICLE_RADIUS_METERS + LIQUIDFUN_SOLID_NODE_RADIUS_METERS) *
				LIQUIDFUN_GATE4_UNITS_PER_METER
		);
		const oilBottomY = Math.max(
			...workload.oilParticles.map(({ position }) => position.y)
		);
		const waterTopY = Math.min(
			...workload.waterParticles.map(({ position }) => position.y)
		);
		expect(waterTopY - oilBottomY).toBe(GATE4_LIQUID_PARTICLE_SPACING);
	});

	it('sets each LiquidFun pebble triangle to the requested physical edge length', () => {
		const originalVertices = [
			{ x: -3.3, y: -3.3 / Math.sqrt(3) },
			{ x: 3.3, y: -3.3 / Math.sqrt(3) },
			{ x: 0, y: (2 * 3.3) / Math.sqrt(3) }
		] as const;
		const scaledVertices = scaleLiquidFunPebbleLocalVertices(originalVertices);
		const width = (vertices: readonly { x: number; y: number }[]) =>
			Math.max(...vertices.map((vertex) => vertex.x)) -
			Math.min(...vertices.map((vertex) => vertex.x));
		const height = (vertices: readonly { x: number; y: number }[]) =>
			Math.max(...vertices.map((vertex) => vertex.y)) -
			Math.min(...vertices.map((vertex) => vertex.y));

		expect(width(scaledVertices)).toBeCloseTo(LIQUIDFUN_PEBBLE_EDGE_LENGTH_BOX2D_UNITS);
		expect(width(scaledVertices)).toBeCloseTo(
			width(originalVertices) * LIQUIDFUN_PEBBLE_SIZE_SCALE
		);
		expect(height(scaledVertices)).toBeCloseTo(
			height(originalVertices) * LIQUIDFUN_PEBBLE_SIZE_SCALE
		);
		expect(() => scaleLiquidFunPebbleLocalVertices(originalVertices.slice(0, 2))).toThrow(
			RangeError
		);
	});

	it('summarizes repeatable batches with the MPM benchmark quantile convention', () => {
		expect(summarizeSamples(Array.from({ length: 12 }, (_, index) => index + 1))).toEqual({
			median: 6,
			p95: 11,
			min: 1,
			max: 12,
			samples: 12
		});
		expect(() => summarizeSamples([])).toThrow(RangeError);
		expect(() => summarizeSamples([1, Number.NaN])).toThrow(RangeError);
	});
});
