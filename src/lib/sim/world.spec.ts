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
	it('starts with about 47.5% water, 47.5% air, and 5% powder', () => {
		const world = createInitialWorld(100, 100);
		const counts = [0, 0, 0];
		for (const material of world.cells) counts[material]++;

		expect(counts[MATERIAL.water]).toBe(4_700);
		expect(counts[MATERIAL.air]).toBe(4_800);
		expect(counts[MATERIAL.powder]).toBe(500);
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

	it('allows denser water to displace powder upward', () => {
		const registry = new MaterialRegistry([
			...[
				materialRegistry.get(MATERIAL.air),
				materialRegistry.get(MATERIAL.water),
				materialRegistry.get(MATERIAL.powder)
			],
			{
				id: 3,
				name: 'wall',
				phase: 'solid',
				density: 3,
				color: [0, 0, 0, 255]
			}
		]);
		const world = new SimulationWorld(3, 4);
		for (let x = 0; x < world.width; x++) world.set(x, 3, 3);
		world.set(0, 2, 3);
		world.set(2, 2, 3);
		world.set(1, 1, MATERIAL.water);
		world.set(1, 2, MATERIAL.powder);

		stepWorld(world, defaultForces, [], () => 0.5, registry);

		expect(world.get(1, 1)).toBe(MATERIAL.powder);
		expect(world.get(1, 2)).toBe(MATERIAL.water);
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
			{
				id: 3,
				name: 'wall',
				phase: 'solid',
				density: 3,
				color: [0, 0, 0, 255]
			}
		]);
		const world = new SimulationWorld(5, 5);
		for (let x = 0; x < world.width; x++) world.set(x, 3, 3);
		world.set(1, 2, 3);
		world.set(2, 2, MATERIAL.water);
		world.set(3, 2, 3);

		stepWorld(world, defaultForces, [], () => 0, registry);

		expect(world.get(2, 1)).toBe(MATERIAL.water);
		expect(world.get(2, 2)).toBe(MATERIAL.air);
		expect([...world.cells].filter((material) => material === MATERIAL.water)).toHaveLength(1);
		expect([...world.cells].filter((material) => material === MATERIAL.air)).toHaveLength(17);
	});
});

describe('selection and placement', () => {
	it('selects only orthogonally connected particles and respects the cap', () => {
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
