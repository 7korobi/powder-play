import { describe, expect, it } from 'vitest';
import { MATERIAL, MaterialRegistry, materialRegistry } from './materials';
import {
	createFlowStroke,
	createInitialWorld,
	createSelectionPattern,
	moveSelection,
	placePattern,
	selectConnected,
	SimulationWorld,
	stepWorld
} from './world';

const defaultForces = {
	gravity: { x: 0, y: 1 },
	acceleration: { x: 0, y: 0 },
	rotationRate: 0
};

describe('world initialization and movement', () => {
	it('starts with one quarter of each material layer populated by larger particles', () => {
		const world = createInitialWorld(100, 100);
		const counts = [0, 0, 0, 0];
		for (const material of world.cells) counts[material]++;

		expect(counts[MATERIAL.water]).toBe(1_000);
		expect(counts[MATERIAL.oil]).toBe(500);
		expect(counts[MATERIAL.powder]).toBe(125);
		expect(counts[MATERIAL.air]).toBe(8_375);
		expect(world.get(0, 34)).toBe(MATERIAL.air);
		expect(world.get(0, 35)).toBe(MATERIAL.powder);
		expect(world.get(0, 40)).toBe(MATERIAL.oil);
		expect(world.get(0, 60)).toBe(MATERIAL.water);
		expect(world.get(1, 60)).toBe(MATERIAL.air);
		expect(materialRegistry.get(MATERIAL.water).particleRadius).toBe(0.9);
		expect(materialRegistry.get(MATERIAL.oil).particleRadius).toBe(0.9);
		expect(materialRegistry.get(MATERIAL.powder).particleRadius).toBe(0.76);
		expect(materialRegistry.get(MATERIAL.water).packingDensity).toBe(0.18);
		expect(materialRegistry.get(MATERIAL.powder).packingDensity).toBe(0.11);
	});

	it('lets powder fall through air but keeps it above water', () => {
		const world = new SimulationWorld(3, 4);
		for (let x = 0; x < world.width; x++) world.set(x, 3, MATERIAL.water);
		world.set(1, 1, MATERIAL.powder);

		stepWorld(world, defaultForces, [], () => 0.5);
		expect(world.get(1, 2)).toBe(MATERIAL.powder);
		stepWorld(world, defaultForces, [], () => 0.5);

		expect(world.get(1, 2)).toBe(MATERIAL.powder);
		expect(world.get(1, 3)).toBe(MATERIAL.water);
	});

	it('scales gravity movement with the measured tilt magnitude', () => {
		const weakWorld = new SimulationWorld(3, 4);
		weakWorld.set(1, 1, MATERIAL.powder);
		stepWorld(weakWorld, { ...defaultForces, gravity: { x: 0, y: 0.25 } }, [], () => 0.5);
		expect(weakWorld.get(1, 1)).toBe(MATERIAL.powder);

		const strongWorld = new SimulationWorld(3, 4);
		strongWorld.set(1, 1, MATERIAL.powder);
		stepWorld(strongWorld, defaultForces, [], () => 0.5);
		expect(strongWorld.get(1, 2)).toBe(MATERIAL.powder);
	});

	it('interpolates gravity between adjacent grid directions', () => {
		const registry = new MaterialRegistry([
			{
				id: MATERIAL.air,
				name: 'air',
				phase: 'gas',
				density: 0,
				color: [0, 0, 0, 0]
			},
			{
				id: MATERIAL.powder,
				name: 'powder',
				phase: 'granular',
				density: 1,
				color: [210, 220, 230, 255]
			}
		]);
		const moveOneParticle = (gravity: { x: number; y: number }): SimulationWorld => {
			const world = new SimulationWorld(4, 4);
			world.set(1, 1, MATERIAL.powder);
			stepWorld(world, { ...defaultForces, gravity }, [], () => 0, registry);
			return world;
		};

		const verticalWorld = moveOneParticle({ x: 0, y: 1 });
		const tiltedWorld = moveOneParticle({ x: 0.05, y: 1 });
		expect(verticalWorld.get(1, 2)).toBe(MATERIAL.powder);
		expect(tiltedWorld.get(2, 2)).toBe(MATERIAL.powder);
	});

	it('tracks cell changes used to invalidate cached rendering', () => {
		const world = new SimulationWorld(4, 4);
		expect(world.cellRevision).toBe(0);

		world.set(1, 1, MATERIAL.powder);
		const placedRevision = world.cellRevision;
		stepWorld(world, defaultForces, [], () => 0.5);
		expect(world.cellRevision).toBeGreaterThan(placedRevision);

		const afterStepRevision = world.cellRevision;
		const moved = moveSelection(world, selectConnected(world, 1, 2), 1, 0);
		expect(moved).not.toBeNull();
		expect(world.cellRevision).toBeGreaterThan(afterStepRevision);

		const afterMoveRevision = world.cellRevision;
		const pattern = createSelectionPattern(world, moved ?? []);
		expect(placePattern(world, pattern, 0, 0)).toBe(true);
		expect(world.cellRevision).toBeGreaterThan(afterMoveRevision);
	});

	it('spreads over-packed particles into nearby air while conserving their count', () => {
		const world = new SimulationWorld(7, 7);
		for (let y = 2; y <= 4; y++) {
			for (let x = 2; x <= 4; x++) world.set(x, y, MATERIAL.powder);
		}

		stepWorld(
			world,
			{ ...defaultForces, gravity: { x: 0, y: 0 } },
			[],
			() => 0
		);

		const powderCells = [...world.cells]
			.map((material, index) => (material === MATERIAL.powder ? index : -1))
			.filter((index) => index >= 0);
		expect(powderCells).toHaveLength(9);
		expect(
			powderCells.some((index) => {
				const x = index % world.width;
				const y = Math.floor(index / world.width);
				return x < 2 || x > 4 || y < 2 || y > 4;
			})
		).toBe(true);
	});

	it('allows denser water to displace powder upward', () => {
		const registry = new MaterialRegistry([
			...[
				materialRegistry.get(MATERIAL.air),
				materialRegistry.get(MATERIAL.water),
				materialRegistry.get(MATERIAL.powder),
				materialRegistry.get(MATERIAL.oil)
			],
			{
				id: 4,
				name: 'wall',
				phase: 'solid',
				density: 3,
				color: [0, 0, 0, 255]
			}
		]);
		const world = new SimulationWorld(3, 4);
		for (let x = 0; x < world.width; x++) world.set(x, 3, 4);
		world.set(0, 2, 4);
		world.set(2, 2, 4);
		world.set(1, 1, MATERIAL.water);
		world.set(1, 2, MATERIAL.powder);

		stepWorld(world, defaultForces, [], () => 0.5, registry);

		expect(world.get(1, 1)).toBe(MATERIAL.powder);
		expect(world.get(1, 2)).toBe(MATERIAL.water);
	});

	it('keeps oil above water and lets denser water displace oil upward', () => {
		const registry = new MaterialRegistry([
			...[
				materialRegistry.get(MATERIAL.air),
				materialRegistry.get(MATERIAL.water),
				materialRegistry.get(MATERIAL.powder),
				materialRegistry.get(MATERIAL.oil)
			],
			{
				id: 4,
				name: 'wall',
				phase: 'solid',
				density: 3,
				color: [0, 0, 0, 255]
			}
		]);

		const oilAboveWater = new SimulationWorld(1, 4);
		oilAboveWater.set(0, 3, 4);
		oilAboveWater.set(0, 2, MATERIAL.water);
		oilAboveWater.set(0, 1, MATERIAL.oil);
		stepWorld(oilAboveWater, defaultForces, [], () => 0.5, registry);
		expect(oilAboveWater.get(0, 1)).toBe(MATERIAL.oil);
		expect(oilAboveWater.get(0, 2)).toBe(MATERIAL.water);

		const waterAboveOil = new SimulationWorld(1, 4);
		waterAboveOil.set(0, 3, 4);
		waterAboveOil.set(0, 2, MATERIAL.oil);
		waterAboveOil.set(0, 1, MATERIAL.water);
		stepWorld(waterAboveOil, defaultForces, [], () => 0.5, registry);
		expect(waterAboveOil.get(0, 1)).toBe(MATERIAL.oil);
		expect(waterAboveOil.get(0, 2)).toBe(MATERIAL.water);
	});

	it('applies gas repulsion only at a nearby movable material boundary', () => {
		const registry = new MaterialRegistry([
			{
				id: MATERIAL.air,
				name: 'air',
				phase: 'gas',
				density: 0,
				color: [0, 0, 0, 0],
				repulsion: 1
			},
			materialRegistry.get(MATERIAL.water),
			materialRegistry.get(MATERIAL.powder),
			materialRegistry.get(MATERIAL.oil),
			{
				id: 4,
				name: 'wall',
				phase: 'solid',
				density: 3,
				color: [0, 0, 0, 255]
			}
		]);
		const world = new SimulationWorld(5, 5);
		for (let x = 0; x < world.width; x++) world.set(x, 3, 4);
		world.set(1, 2, 4);
		world.set(2, 2, MATERIAL.water);
		world.set(3, 2, 4);

		stepWorld(world, defaultForces, [], () => 0, registry);

		expect(world.get(2, 1)).toBe(MATERIAL.water);
		expect(world.get(2, 2)).toBe(MATERIAL.air);
		expect([...world.cells].filter((material) => material === MATERIAL.water)).toHaveLength(1);
		expect([...world.cells].filter((material) => material === MATERIAL.air)).toHaveLength(17);
	});
});

describe('selection and placement', () => {
	it('selects particles within contact range and respects the cap', () => {
		const world = new SimulationWorld(5, 5);
		world.set(1, 1, MATERIAL.powder);
		world.set(2, 1, MATERIAL.powder);
		world.set(1, 2, MATERIAL.powder);
		world.set(4, 4, MATERIAL.powder);

		expect(selectConnected(world, 1, 1)).toHaveLength(3);
		expect(selectConnected(world, 1, 1, 2)).toHaveLength(2);
		expect(selectConnected(world, 4, 4)).toHaveLength(1);
		expect(selectConnected(world, 0, 0)).toEqual([]);
	});

	it('treats particles two cells apart as touching', () => {
		const world = new SimulationWorld(9, 9);
		world.set(1, 1, MATERIAL.powder);
		world.set(3, 1, MATERIAL.powder);
		world.set(3, 3, MATERIAL.powder);
		world.set(5, 3, MATERIAL.powder);

		expect(selectConnected(world, 1, 1)).toHaveLength(4);
	});

	it('moves a selection atomically into empty cells and rejects collisions', () => {
		const world = new SimulationWorld(5, 5);
		world.set(1, 1, MATERIAL.powder);
		world.set(2, 1, MATERIAL.powder);
		const selection = selectConnected(world, 1, 1);

		const moved = moveSelection(world, selection, 1, 0);
		expect(moved).not.toBeNull();
		expect(world.get(1, 1)).toBe(MATERIAL.air);
		expect(world.get(2, 1)).toBe(MATERIAL.powder);
		expect(world.get(3, 1)).toBe(MATERIAL.powder);

		world.set(4, 1, MATERIAL.water);
		const before = [...world.cells];
		expect(moveSelection(world, moved ?? [], 1, 0)).toBeNull();
		expect([...world.cells]).toEqual(before);
	});

	it('copies a selected pattern only when every destination is empty', () => {
		const world = new SimulationWorld(5, 5);
		world.set(0, 0, MATERIAL.powder);
		world.set(1, 0, MATERIAL.powder);
		world.set(0, 1, MATERIAL.powder);
		const pattern = createSelectionPattern(world, selectConnected(world, 0, 0));

		expect(placePattern(world, pattern, 2, 2)).toBe(true);
		expect(world.get(2, 2)).toBe(MATERIAL.powder);
		expect(world.get(3, 2)).toBe(MATERIAL.powder);
		expect(world.get(2, 3)).toBe(MATERIAL.powder);

		world.set(4, 2, MATERIAL.water);
		expect(placePattern(world, pattern, 3, 2)).toBe(false);
		expect(world.get(3, 3)).toBe(MATERIAL.air);
	});

	it('turns a drawn line into a temporary directional flow', () => {
		const world = new SimulationWorld(7, 5);
		world.set(1, 2, MATERIAL.powder);
		const flow = createFlowStroke([
			{ x: 0, y: 2 },
			{ x: 4, y: 2 }
		]);
		expect(flow?.direction.x).toBe(1);
		expect(flow?.remainingSteps).toBe(45);
		expect(flow).not.toBeNull();

		stepWorld(world, defaultForces, flow ? [flow] : [], () => 0);

		expect(world.get(2, 2)).toBe(MATERIAL.powder);
		expect(flow?.remainingSteps).toBe(44);
		expect(createFlowStroke([{ x: 1, y: 1 }, { x: 1.5, y: 1 }])).toBeNull();
	});

	it('keeps selected particles stable while the simulation continues around them', () => {
		const world = new SimulationWorld(3, 4);
		world.set(0, 1, MATERIAL.powder);
		world.set(1, 1, MATERIAL.powder);
		const selectedCell = 1 * world.width + 1;

		stepWorld(world, defaultForces, [], () => 0.5, materialRegistry, new Set([selectedCell]));

		expect(world.get(1, 1)).toBe(MATERIAL.powder);
		expect(world.get(0, 2)).toBe(MATERIAL.powder);
	});
});
