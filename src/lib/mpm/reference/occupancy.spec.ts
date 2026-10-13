import { describe, expect, it } from 'vitest';
import type { MpmMaterialParameters, MpmParticle } from './solver.js';
import { gatherCurrentArea, type MpmCurrentAreaConfig } from './occupancy.js';

const IDENTITY = { m00: 1, m01: 0, m10: 0, m11: 1 } as const;
const ZERO_MATRIX = { m00: 0, m01: 0, m10: 0, m11: 0 } as const;
const WATER: MpmMaterialParameters = {
	lambda: 20,
	mu: 0,
	model: 'water',
	density: 1
};
const OIL: MpmMaterialParameters = {
	lambda: 18,
	mu: 0,
	model: 'oil',
	density: 0.8
};
const POWDER: MpmMaterialParameters = {
	lambda: 20,
	mu: 10,
	model: 'powder',
	frictionCoefficient: 0.5
};

function makeConfig(overrides: Partial<MpmCurrentAreaConfig> = {}): MpmCurrentAreaConfig {
	return {
		grid: { cellsX: 8, cellsY: 8, cellSize: 1 },
		particleVolume: 0.5,
		material: WATER,
		...overrides
	};
}

function makeParticle(overrides: Partial<MpmParticle> = {}): MpmParticle {
	return {
		position: { x: 2.25, y: 3.75 },
		velocity: { x: 0, y: 0 },
		C: ZERO_MATRIX,
		F: IDENTITY,
		material: WATER,
		...overrides
	};
}

function nodeAt(
	nodes: ReturnType<typeof gatherCurrentArea>['nodes'],
	xIndex: number,
	yIndex: number
) {
	const node = nodes.find((sample) => sample.xIndex === xIndex && sample.yIndex === yIndex);
	if (!node) throw new Error(`missing occupancy node (${xIndex}, ${yIndex})`);
	return node;
}

function sumPhase(nodes: ReturnType<typeof gatherCurrentArea>['nodes'], phase: 'water' | 'oil') {
	return nodes.reduce((sum, node) => sum + node[phase], 0);
}

function closeEnough(actual: number, expected: number, tolerance = 1e-12): void {
	expect(Math.abs(actual - expected)).toBeLessThanOrEqual(tolerance);
}

describe('CPU current-area occupancy reference', () => {
	it('keeps water and oil in separate fields and ignores non-liquid particles', () => {
		const config = makeConfig({ particleVolume: 0.75 });
		const water = makeParticle({ volume: 0.4 });
		const defaultWater = makeParticle({
			position: { x: 4.25, y: 2.75 },
			material: undefined,
			volume: undefined
		});
		const oil = makeParticle({
			material: OIL,
			volume: 0.2,
			F: { m00: 1.5, m01: 0, m10: 0, m11: 1 }
		});
		const powder = makeParticle({ material: POWDER, position: { x: 6.25, y: 5.75 } });

		const combined = gatherCurrentArea([water, defaultWater, oil, powder], config);
		const waterOnly = gatherCurrentArea([water, defaultWater], config);
		const oilOnly = gatherCurrentArea([oil], config);
		const powderOnly = gatherCurrentArea([powder], config);

		expect(combined.nodes.length).toBeGreaterThan(9);
		expect(powderOnly.nodes).toEqual([]);
		for (const node of combined.nodes) {
			const waterNode = waterOnly.nodes.find(
				(sample) => sample.xIndex === node.xIndex && sample.yIndex === node.yIndex
			);
			const oilNode = oilOnly.nodes.find(
				(sample) => sample.xIndex === node.xIndex && sample.yIndex === node.yIndex
			);
			expect(node.water).toBe(waterNode?.water ?? 0);
			expect(node.oil).toBe(oilNode?.oil ?? 0);
		}
		expect(sumPhase(combined.nodes, 'water')).toBeGreaterThan(0);
		expect(sumPhase(combined.nodes, 'oil')).toBeGreaterThan(0);
	});

	it('translates the quadratic stencil with a particle and retains boundary halo nodes', () => {
		const config = makeConfig();
		const particle = makeParticle();
		const translated = makeParticle({
			position: {
				x: particle.position.x + 2,
				y: particle.position.y + 1
			}
		});
		const originalField = gatherCurrentArea([particle], config);
		const translatedField = gatherCurrentArea([translated], config);

		expect(translatedField.nodes).toHaveLength(originalField.nodes.length);
		for (const node of originalField.nodes) {
			const shiftedNode = nodeAt(
				translatedField.nodes,
				node.xIndex + 2,
				node.yIndex + 1
			);
			expect(shiftedNode.water).toBe(node.water);
			expect(shiftedNode.oil).toBe(node.oil);
		}

		const lowerBoundary = gatherCurrentArea(
			[makeParticle({ position: { x: 0, y: 0 } })],
			config
		);
		expect(lowerBoundary.nodes.some((node) => node.xIndex < 0 || node.yIndex < 0)).toBe(true);
		expect(nodeAt(lowerBoundary.nodes, 0, 0).isHalo).toBe(false);
		expect(nodeAt(lowerBoundary.nodes, -1, -1).isHalo).toBe(true);

		const upperBoundary = gatherCurrentArea(
			[makeParticle({ position: { x: 8, y: 8 } })],
			config
		);
		expect(upperBoundary.nodes.some((node) => node.xIndex > 8 || node.yIndex > 8)).toBe(true);
		expect(nodeAt(upperBoundary.nodes, 8, 8).isHalo).toBe(false);
		expect(nodeAt(upperBoundary.nodes, 9, 9).isHalo).toBe(true);
	});

	it('scales occupancy with the deformation determinant and current particle area', () => {
		const config = makeConfig();
		const undeformed = makeParticle({ volume: 0.6 });
		const stretched = makeParticle({
			volume: 0.6,
			F: { m00: 2, m01: 0, m10: 0, m11: 1 }
		});
		const reference = gatherCurrentArea([undeformed], config);
		const deformed = gatherCurrentArea([stretched], config);

		expect(deformed.nodes).toHaveLength(reference.nodes.length);
		for (const node of reference.nodes) {
			closeEnough(
				nodeAt(deformed.nodes, node.xIndex, node.yIndex).water,
				node.water * 2
			);
		}
	});

	it('conserves water and oil current area independently across physical and halo nodes', () => {
		const config = makeConfig({
			grid: { cellsX: 6, cellsY: 6, cellSize: 2 },
			particleVolume: 1
		});
		const particles = [
			makeParticle({
				position: { x: 0, y: 0 },
				volume: 1.2,
				F: { m00: 1.5, m01: 0, m10: 0, m11: 1 }
			}),
			makeParticle({
				position: { x: 5, y: 7 },
				volume: 0.8,
				F: { m00: 1, m01: 0, m10: 0, m11: 1.25 }
			}),
			makeParticle({
				position: { x: 12, y: 12 },
				material: OIL,
				volume: 0.75,
				F: { m00: 1, m01: 0, m10: 0, m11: 0.5 }
			}),
			makeParticle({
				position: { x: 6, y: 3 },
				material: OIL,
				volume: 0.4,
				F: { m00: 2, m01: 0, m10: 0, m11: 1 }
			}),
			makeParticle({ material: POWDER })
		];
		const result = gatherCurrentArea(particles, config);

		closeEnough(sumPhase(result.nodes, 'water') * config.grid.cellSize ** 2, 2.8);
		closeEnough(sumPhase(result.nodes, 'oil') * config.grid.cellSize ** 2, 1.175);
		expect(result.nodes.some((node) => node.isHalo)).toBe(true);
	});

	it('rejects invalid grids and particle inputs', () => {
		const config = makeConfig();
		expect(() =>
			gatherCurrentArea([], {
				...config,
				grid: { cellsX: 0, cellsY: 8, cellSize: 1 }
			})
		).toThrow(/grid\.cellsX/);
		expect(() =>
			gatherCurrentArea([], {
				...config,
				grid: { cellsX: 8, cellsY: 8, cellSize: 0 }
			})
		).toThrow(/grid\.cellSize/);
		expect(() => gatherCurrentArea([], { ...config, particleVolume: 0 })).toThrow(
			/particleVolume/
		);
		expect(() =>
			gatherCurrentArea(
				[makeParticle({ position: { x: -0.01, y: 2 } })],
				config
			)
		).toThrow(/grid domain/);
		expect(() =>
			gatherCurrentArea(
				[makeParticle({ F: { m00: -1, m01: 0, m10: 0, m11: 1 } })],
				config
			)
		).toThrow(/determinant/);
		expect(() => gatherCurrentArea([makeParticle({ volume: 0 })], config)).toThrow(
			/volume/
		);
		expect(() =>
			gatherCurrentArea({} as unknown as readonly MpmParticle[], config)
		).toThrow(/particles must be an array/);
	});
});
