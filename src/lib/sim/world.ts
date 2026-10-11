import { MATERIAL, materialRegistry, type MaterialPlugin, type MaterialRegistry } from './materials';

export interface Vector {
	x: number;
	y: number;
}

export interface Point extends Vector {}

export interface FlowStroke {
	points: Point[];
	direction: Vector;
	width: number;
	strength: number;
	remainingSteps: number;
}

export interface SelectionParticle {
	dx: number;
	dy: number;
	material: number;
}

const MAX_WORLD_CELLS = 1_000_000;
const NO_PROTECTED_CELLS: ReadonlySet<number> = new Set();
const NEIGHBORS: readonly Vector[] = [
	{ x: 0, y: -1 },
	{ x: 1, y: 0 },
	{ x: 0, y: 1 },
	{ x: -1, y: 0 }
];

export class SimulationWorld {
	readonly cells: Uint8Array;
	readonly lastMoved: Uint32Array;
	tick = 0;

	constructor(
		readonly width: number,
		readonly height: number,
		cells?: Uint8Array
	) {
		if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1) {
			throw new RangeError('World dimensions must be positive integers.');
		}
		const cellCount = width * height;
		if (!Number.isSafeInteger(cellCount) || cellCount > MAX_WORLD_CELLS) {
			throw new RangeError(`Worlds are limited to ${MAX_WORLD_CELLS} cells.`);
		}
		if (cells && cells.length !== cellCount) {
			throw new RangeError('The cell array length must match the world dimensions.');
		}

		this.cells = cells ? new Uint8Array(cells) : new Uint8Array(cellCount).fill(MATERIAL.air);
		this.lastMoved = new Uint32Array(cellCount);
	}

	get(x: number, y: number): number {
		if (!this.contains(x, y)) throw new RangeError('Cell is outside the world.');
		return this.cells[y * this.width + x];
	}

	set(x: number, y: number, material: number): void {
		if (!this.contains(x, y)) throw new RangeError('Cell is outside the world.');
		if (!Number.isInteger(material) || material < 0 || material > 255) {
			throw new RangeError('Material ids must be integers between 0 and 255.');
		}
		this.cells[y * this.width + x] = material;
	}

	contains(x: number, y: number): boolean {
		return (
			Number.isInteger(x) &&
			Number.isInteger(y) &&
			x >= 0 &&
			y >= 0 &&
			x < this.width &&
			y < this.height
		);
	}
}

export function createInitialWorld(width: number, height: number): SimulationWorld {
	const world = new SimulationWorld(width, height);
	const waterRows = Math.max(1, Math.floor(height * 0.475));
	const powderRows = Math.max(1, Math.floor(height * 0.05));
	const powderStart = Math.max(0, height - waterRows - powderRows);
	const waterStart = height - waterRows;

	for (let y = powderStart; y < waterStart; y++) {
		world.cells.fill(MATERIAL.powder, y * width, (y + 1) * width);
	}
	for (let y = waterStart; y < height; y++) {
		world.cells.fill(MATERIAL.water, y * width, (y + 1) * width);
	}

	return world;
}

export function resampleWorld(world: SimulationWorld, width: number, height: number): SimulationWorld {
	const next = new SimulationWorld(width, height);
	for (let y = 0; y < height; y++) {
		const sourceY = Math.min(world.height - 1, Math.floor(((y + 0.5) * world.height) / height));
		for (let x = 0; x < width; x++) {
			const sourceX = Math.min(world.width - 1, Math.floor(((x + 0.5) * world.width) / width));
			next.cells[y * width + x] = world.cells[sourceY * world.width + sourceX];
		}
	}
	next.tick = world.tick;
	return next;
}

export function selectConnected(
	world: SimulationWorld,
	startX: number,
	startY: number,
	maxCells = 256
): number[] {
	if (!world.contains(startX, startY) || maxCells < 1) return [];
	const material = world.get(startX, startY);
	if (material === MATERIAL.air) return [];

	const start = startY * world.width + startX;
	const visited = new Set<number>([start]);
	const queue = [start];

	for (let cursor = 0; cursor < queue.length && queue.length < maxCells; cursor++) {
		const index = queue[cursor];
		const x = index % world.width;
		const y = Math.floor(index / world.width);
		for (const neighbor of NEIGHBORS) {
			const nextX = x + neighbor.x;
			const nextY = y + neighbor.y;
			if (!world.contains(nextX, nextY) || world.get(nextX, nextY) !== material) continue;
			const nextIndex = nextY * world.width + nextX;
			if (visited.has(nextIndex)) continue;
			visited.add(nextIndex);
			queue.push(nextIndex);
			if (queue.length >= maxCells) break;
		}
	}

	return queue;
}

export function moveSelection(
	world: SimulationWorld,
	selection: readonly number[],
	deltaX: number,
	deltaY: number
): number[] | null {
	if (
		!Number.isInteger(deltaX) ||
		!Number.isInteger(deltaY) ||
		(selection.length === 0) ||
		(deltaX === 0 && deltaY === 0)
	) {
		return selection.length === 0 ? null : [...selection];
	}

	const selected = new Set(selection);
	const destinations: number[] = [];
	for (const source of selection) {
		const x = source % world.width;
		const y = Math.floor(source / world.width);
		const targetX = x + deltaX;
		const targetY = y + deltaY;
		if (!world.contains(targetX, targetY)) return null;
		const target = targetY * world.width + targetX;
		if (!selected.has(target) && world.cells[target] !== MATERIAL.air) return null;
		destinations.push(target);
	}

	const materials = selection.map((source) => world.cells[source]);
	for (const source of selection) world.cells[source] = MATERIAL.air;
	for (let index = 0; index < destinations.length; index++) {
		world.cells[destinations[index]] = materials[index];
	}
	return destinations;
}

export function createSelectionPattern(
	world: SimulationWorld,
	selection: readonly number[]
): SelectionParticle[] {
	if (selection.length === 0) return [];
	const anchorX = selection[0] % world.width;
	const anchorY = Math.floor(selection[0] / world.width);
	return selection.map((index) => ({
		dx: (index % world.width) - anchorX,
		dy: Math.floor(index / world.width) - anchorY,
		material: world.cells[index]
	}));
}

export function placePattern(
	world: SimulationWorld,
	pattern: readonly SelectionParticle[],
	anchorX: number,
	anchorY: number
): boolean {
	if (!Number.isInteger(anchorX) || !Number.isInteger(anchorY) || pattern.length === 0) return false;
	const destinations: number[] = [];
	for (const particle of pattern) {
		const x = anchorX + particle.dx;
		const y = anchorY + particle.dy;
		if (!world.contains(x, y)) return false;
		const index = y * world.width + x;
		if (world.cells[index] !== MATERIAL.air) return false;
		destinations.push(index);
	}
	for (let index = 0; index < destinations.length; index++) {
		world.cells[destinations[index]] = pattern[index].material;
	}
	return true;
}

export function createFlowStroke(points: readonly Point[]): FlowStroke | null {
	if (points.length < 2) return null;
	const first = points[0];
	const last = points[points.length - 1];
	const dx = last.x - first.x;
	const dy = last.y - first.y;
	const magnitude = Math.hypot(dx, dy);
	if (magnitude < 1) return null;
	return {
		points: points.map(({ x, y }) => ({ x, y })),
		direction: { x: dx / magnitude, y: dy / magnitude },
		width: 2,
		strength: 0.7,
		remainingSteps: 45
	};
}

export interface SimulationForces {
	gravity: Vector;
	acceleration: Vector;
	rotationRate: number;
}

export function stepWorld(
	world: SimulationWorld,
	forces: SimulationForces,
	flows: FlowStroke[] = [],
	random: () => number = Math.random,
	registry: MaterialRegistry = materialRegistry,
	protectedCells: ReadonlySet<number> = NO_PROTECTED_CELLS
): void {
	world.tick = (world.tick + 1) >>> 0;
	if (world.tick === 0) {
		world.lastMoved.fill(0);
		world.tick = 1;
	}

	for (const flow of flows) {
		if (flow.remainingSteps > 0) applyFlow(world, flow, random, registry, protectedCells);
		flow.remainingSteps = Math.max(0, flow.remainingSteps - 1);
	}
	applyLinearAcceleration(world, forces.acceleration, random, registry, protectedCells);
	applyRotation(world, forces.rotationRate, random, registry, protectedCells);

	const gravity = usableGravity(forces.gravity);
	const downhill = orderedDirections(gravity);
	const uphill = orderedDirections({ x: -gravity.x, y: -gravity.y });
	const xStart =
		gravity.x > 0.05 ? world.width - 1 : gravity.x < -0.05 ? 0 : world.tick % 2 ? 0 : world.width - 1;
	const xStep = xStart === 0 ? 1 : -1;
	const yStart = gravity.y > 0.05 ? world.height - 1 : gravity.y < -0.05 ? 0 : world.height - 1;
	const yStep = yStart === 0 ? 1 : -1;

	for (let y = yStart; y >= 0 && y < world.height; y += yStep) {
		for (let x = xStart; x >= 0 && x < world.width; x += xStep) {
			const index = y * world.width + x;
			if (world.lastMoved[index] === world.tick || protectedCells.has(index)) continue;
			const plugin = registry.get(world.cells[index]);
			if (plugin.phase === 'solid') continue;
			if (plugin.phase === 'gas') {
				if (tryGasRepulsion(world, x, y, plugin, random, registry, protectedCells)) continue;
				tryMoveInDirections(world, x, y, uphill, plugin, registry, protectedCells);
				continue;
			}

			if (tryMoveInDirections(world, x, y, downhill, plugin, registry, protectedCells)) continue;
			if (plugin.phase === 'liquid') {
				tryMoveSideways(world, x, y, gravity, plugin, random, registry, protectedCells);
			}
		}
	}
}

function usableGravity(gravity: Vector): Vector {
	const magnitude = Math.hypot(gravity.x, gravity.y);
	if (!Number.isFinite(magnitude) || magnitude < 0.05) return { x: 0, y: 1 };
	return { x: gravity.x / magnitude, y: gravity.y / magnitude };
}

function orderedDirections(vector: Vector): Vector[] {
	const directions: Vector[] = [
		{ x: 0, y: -1 },
		{ x: 1, y: -1 },
		{ x: 1, y: 0 },
		{ x: 1, y: 1 },
		{ x: 0, y: 1 },
		{ x: -1, y: 1 },
		{ x: -1, y: 0 },
		{ x: -1, y: -1 }
	];
	return directions
		.map((direction) => ({
			direction,
			score:
				(direction.x * vector.x + direction.y * vector.y) /
				Math.hypot(direction.x, direction.y)
		}))
		.filter(({ score }) => score > 0.05)
		.sort((left, right) => right.score - left.score)
		.map(({ direction }) => direction);
}

function tryMoveInDirections(
	world: SimulationWorld,
	x: number,
	y: number,
	directions: readonly Vector[],
	plugin: MaterialPlugin,
	registry: MaterialRegistry,
	protectedCells: ReadonlySet<number>
): boolean {
	const sourceIndex = y * world.width + x;
	if (world.lastMoved[sourceIndex] === world.tick || protectedCells.has(sourceIndex)) return false;
	for (const direction of directions) {
		const targetX = x + direction.x;
		const targetY = y + direction.y;
		if (!world.contains(targetX, targetY)) continue;
		const targetIndex = targetY * world.width + targetX;
		if (world.lastMoved[targetIndex] === world.tick || protectedCells.has(targetIndex)) continue;
		const target = registry.get(world.cells[targetIndex]);
		const canMove =
			plugin.phase === 'gas'
				? target.phase === 'liquid' && target.density > plugin.density
				: plugin.phase === 'liquid'
					? target.density < plugin.density
					: plugin.phase === 'granular'
						? target.phase === 'gas'
						: false;
		if (!canMove) continue;
		swapCells(world, sourceIndex, targetIndex);
		return true;
	}
	return false;
}

function tryGasRepulsion(
	world: SimulationWorld,
	x: number,
	y: number,
	plugin: MaterialPlugin,
	random: () => number,
	registry: MaterialRegistry,
	protectedCells: ReadonlySet<number>
): boolean {
	if (!plugin.repulsion) return false;
	const targets: number[] = [];
	for (const direction of NEIGHBORS) {
		const targetX = x + direction.x;
		const targetY = y + direction.y;
		if (!world.contains(targetX, targetY)) continue;
		const index = targetY * world.width + targetX;
		if (world.lastMoved[index] === world.tick || protectedCells.has(index)) continue;
		const target = registry.get(world.cells[index]);
		if (target.phase === 'liquid' || target.phase === 'granular') targets.push(index);
	}
	if (targets.length === 0) return false;
	if (random() >= plugin.repulsion) return false;
	const target = targets[Math.floor(random() * targets.length)] ?? targets[0];
	swapCells(world, y * world.width + x, target);
	return true;
}

function tryMoveSideways(
	world: SimulationWorld,
	x: number,
	y: number,
	gravity: Vector,
	plugin: MaterialPlugin,
	random: () => number,
	registry: MaterialRegistry,
	protectedCells: ReadonlySet<number>
): boolean {
	if (random() >= 0.24) return false;
	const perpendicular = quantizedPerpendicular(gravity);
	if (perpendicular.x === 0 && perpendicular.y === 0) return false;
	const first = random() < 0.5 ? perpendicular : { x: -perpendicular.x, y: -perpendicular.y };
	const second = { x: -first.x, y: -first.y };
	return tryMoveInDirections(world, x, y, [first, second], plugin, registry, protectedCells);
}

function quantizedPerpendicular(gravity: Vector): Vector {
	const x = -gravity.y;
	const y = gravity.x;
	return {
		x: Math.abs(x) < 0.25 ? 0 : Math.sign(x),
		y: Math.abs(y) < 0.25 ? 0 : Math.sign(y)
	};
}

function applyFlow(
	world: SimulationWorld,
	flow: FlowStroke,
	random: () => number,
	registry: MaterialRegistry,
	protectedCells: ReadonlySet<number>
): void {
	const step = quantizedDirection(flow.direction);
	if (step.x === 0 && step.y === 0) return;
	const radius = Math.max(0.5, flow.width / 2);
	const reach = Math.ceil(radius);
	for (let segment = 1; segment < flow.points.length; segment++) {
		const start = flow.points[segment - 1];
		const end = flow.points[segment];
		const length = Math.max(Math.abs(end.x - start.x), Math.abs(end.y - start.y));
		const samples = Math.max(1, Math.ceil(length));
		for (let sample = 0; sample <= samples; sample++) {
			const ratio = sample / samples;
			const centerX = Math.round(start.x + (end.x - start.x) * ratio);
			const centerY = Math.round(start.y + (end.y - start.y) * ratio);
			for (let offsetY = -reach; offsetY <= reach; offsetY++) {
				for (let offsetX = -reach; offsetX <= reach; offsetX++) {
					if (offsetX * offsetX + offsetY * offsetY > radius * radius) continue;
					const x = centerX + offsetX;
					const y = centerY + offsetY;
					if (!world.contains(x, y)) continue;
					const index = y * world.width + x;
					if (protectedCells.has(index) || world.lastMoved[index] === world.tick) continue;
					const plugin = registry.get(world.cells[y * world.width + x]);
					if (
						plugin.phase !== 'gas' &&
						plugin.phase !== 'solid' &&
						random() < flow.strength
					) {
						tryMoveInDirections(world, x, y, [step], plugin, registry, protectedCells);
					}
				}
			}
		}
	}
}

function applyLinearAcceleration(
	world: SimulationWorld,
	acceleration: Vector,
	random: () => number,
	registry: MaterialRegistry,
	protectedCells: ReadonlySet<number>
): void {
	const magnitude = Math.hypot(acceleration.x, acceleration.y);
	if (!Number.isFinite(magnitude) || magnitude < 0.12) return;
	const direction = quantizedDirection(acceleration);
	if (direction.x === 0 && direction.y === 0) return;
	const chance = Math.min(0.003, magnitude * 0.003);
	for (let index = 0; index < world.cells.length; index++) {
		if (world.lastMoved[index] === world.tick || protectedCells.has(index)) continue;
		const plugin = registry.get(world.cells[index]);
		if (plugin.phase === 'gas' || plugin.phase === 'solid') continue;
		if (random() >= chance) continue;
		const x = index % world.width;
		const y = Math.floor(index / world.width);
		tryMoveInDirections(world, x, y, [direction], plugin, registry, protectedCells);
	}
}

function applyRotation(
	world: SimulationWorld,
	rotationRate: number,
	random: () => number,
	registry: MaterialRegistry,
	protectedCells: ReadonlySet<number>
): void {
	if (!Number.isFinite(rotationRate) || Math.abs(rotationRate) < 4) return;
	const strength = Math.min(1, Math.abs(rotationRate) / 180);
	const chance = strength * 0.0015;
	const centerX = (world.width - 1) / 2;
	const centerY = (world.height - 1) / 2;
	for (let index = 0; index < world.cells.length; index++) {
		if (world.lastMoved[index] === world.tick || protectedCells.has(index)) continue;
		const plugin = registry.get(world.cells[index]);
		if (plugin.phase === 'gas' || plugin.phase === 'solid') continue;
		if (random() >= chance) continue;
		const x = index % world.width;
		const y = Math.floor(index / world.width);
		const radiusX = (x - centerX) / Math.max(1, centerX);
		const radiusY = (y - centerY) / Math.max(1, centerY);
		const sign = Math.sign(rotationRate);
		const tangent = quantizedDirection({ x: -radiusY * sign, y: radiusX * sign });
		if (tangent.x === 0 && tangent.y === 0) continue;
		tryMoveInDirections(world, x, y, [tangent], plugin, registry, protectedCells);
	}
}

function quantizedDirection(vector: Vector): Vector {
	return {
		x: Math.abs(vector.x) < 0.15 ? 0 : Math.sign(vector.x),
		y: Math.abs(vector.y) < 0.15 ? 0 : Math.sign(vector.y)
	};
}

function swapCells(world: SimulationWorld, firstIndex: number, secondIndex: number): void {
	const first = world.cells[firstIndex];
	world.cells[firstIndex] = world.cells[secondIndex];
	world.cells[secondIndex] = first;
	world.lastMoved[firstIndex] = world.tick;
	world.lastMoved[secondIndex] = world.tick;
}
