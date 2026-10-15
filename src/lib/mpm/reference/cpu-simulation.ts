import type {
	MpmParticle,
	MpmSolidBond,
	MpmStepConfig,
	MpmStepResult,
	MpmVector2
} from './solver.js';
import { MpmSolverWorkspace, stepMpm } from './solver.js';
import {
	GATE4_FRACTURE_OBJECT_ID,
	GATE4_GRID,
	GATE4_PEBBLE_PARTICLE_COUNT,
	GATE4_PEBBLE_PARTICLES_PER_OBJECT,
	GATE4_PARTICLE_AREA,
	type MpmGate4Scene
} from '../scenes/gate4.js';

export type MpmCpuRenderMode = 'surface' | 'particles';
export const MPM_CPU_MAX_CFL_MICROSTEPS_PER_MANUAL_STEP = 256;

export interface MpmCpuRenderSnapshot {
	readonly positions: Float64Array;
	readonly currentLiquidAreas: Float32Array;
	readonly brokenFractureBonds: Uint8Array;
}

export type MpmCpuWorkerRequest =
	| {
			readonly type: 'initialize';
			readonly generation: number;
			readonly liquidParticleDiameter: number;
			readonly gravity: MpmVector2;
	  }
	| {
			readonly type: 'advance';
			readonly generation: number;
			readonly accumulatorSeconds: number;
			readonly maxMicrosteps: number;
			readonly gravity: MpmVector2;
			readonly oscillationAmplitude: number;
	  }
	| {
			readonly type: 'step';
			readonly generation: number;
			readonly maxMicrosteps: number;
			readonly gravity: MpmVector2;
			readonly oscillationAmplitude: number;
	  };

export type MpmCpuWorkerResponse =
	| {
			readonly type: 'ready';
			readonly generation: number;
			readonly snapshot: MpmCpuRenderSnapshot;
			readonly remainderSeconds: number;
	  }
	| {
			readonly type: 'advanced' | 'stepped';
			readonly generation: number;
			readonly snapshot: MpmCpuRenderSnapshot;
			readonly simulatedSeconds: number;
			readonly remainderSeconds: number;
			readonly microStepCount: number;
			readonly workerMilliseconds: number;
	  }
	| {
			readonly type: 'error';
			readonly generation: number;
			readonly message: string;
	  };

export class MpmCpuReferenceSimulation {
	private currentParticles: readonly MpmParticle[];
	private currentLiquidAreas: Float32Array;
	private currentFractureBonds: readonly MpmSolidBond[];
	private readonly solverWorkspace = new MpmSolverWorkspace();
	private disposed = false;

	readonly particleCount: number;
	readonly waterParticleCount: number;
	readonly oilParticleCount: number;
	readonly mappedMpmNodeCount: number;
	readonly pebbleNodeCount: number;
	readonly triangleBodyCount: number;
	readonly fractureNodeCount: number;
	readonly fractureBondCount: number;
	readonly sourceMpmNodeCount: number;

	constructor(scene: MpmGate4Scene) {
		this.currentParticles = scene.particles;
		this.currentLiquidAreas = this.calculateCurrentLiquidAreas(scene.particles);
		this.currentFractureBonds = scene.fractureBonds;
		this.particleCount = scene.particles.length;
		this.waterParticleCount = scene.particles.filter(
			(particle) => particle.material?.model === 'water'
		).length;
		this.oilParticleCount = scene.particles.filter(
			(particle) => particle.material?.model === 'oil'
		).length;
		this.pebbleNodeCount = scene.particles.filter(
			(particle) => particle.bondedTriangleVertex !== undefined
		).length;
		this.triangleBodyCount = this.pebbleNodeCount / GATE4_PEBBLE_PARTICLES_PER_OBJECT;
		this.fractureNodeCount = scene.particles.filter(
			(particle) => particle.objectId === GATE4_FRACTURE_OBJECT_ID
		).length;
		this.fractureBondCount = scene.fractureBonds.length;
		this.sourceMpmNodeCount = scene.particles.length;
		this.mappedMpmNodeCount = scene.particles.length;

		if (
			this.pebbleNodeCount !== GATE4_PEBBLE_PARTICLE_COUNT ||
			!Number.isInteger(this.triangleBodyCount)
		) {
			throw new RangeError('The CPU reference scene has an invalid pebble topology.');
		}
	}

	get particles(): readonly MpmParticle[] {
		this.assertActive();
		return this.currentParticles;
	}

	get fractureBonds(): readonly MpmSolidBond[] {
		this.assertActive();
		return this.currentFractureBonds;
	}

	get activeFractureBondCount(): number {
		return this.currentFractureBonds.filter((bond) => !bond.broken).length;
	}

	step(config: MpmStepConfig): MpmStepResult {
		this.assertActive();
		const result = stepMpm(this.currentParticles, {
			...config,
			fractureBonds: this.currentFractureBonds
		}, this.solverWorkspace);
		this.currentParticles = result.particles;
		this.currentLiquidAreas = this.calculateCurrentLiquidAreas(result.particles);
		this.currentFractureBonds = result.fractureBonds ?? this.currentFractureBonds;
		return result;
	}

	createRenderSnapshot(): MpmCpuRenderSnapshot {
		this.assertActive();
		const positions = new Float64Array(this.currentParticles.length * 2);
		const currentLiquidAreas = new Float32Array(this.currentLiquidAreas);
		for (let index = 0; index < this.currentParticles.length; index += 1) {
			const particle = this.currentParticles[index];
			if (!particle) {
				throw new RangeError(`Particle ${index} is missing from the CPU simulation.`);
			}
			positions[index * 2] = particle.position.x;
			positions[index * 2 + 1] = particle.position.y;
		}
		const brokenFractureBonds = new Uint8Array(this.currentFractureBonds.length);
		for (let index = 0; index < this.currentFractureBonds.length; index += 1) {
			const bond = this.currentFractureBonds[index];
			if (!bond) {
				throw new RangeError(`Fracture bond ${index} is missing from the CPU simulation.`);
			}
			brokenFractureBonds[index] = bond.broken ? 1 : 0;
		}
		return { positions, currentLiquidAreas, brokenFractureBonds };
	}

	applyRenderSnapshot(snapshot: MpmCpuRenderSnapshot): void {
		this.assertActive();
		if (
			snapshot.positions.length !== this.particleCount * 2 ||
			snapshot.currentLiquidAreas.length !== this.particleCount ||
			snapshot.brokenFractureBonds.length !== this.fractureBondCount
		) {
			throw new RangeError('The CPU worker returned a render snapshot with invalid dimensions.');
		}
		for (let index = 0; index < this.particleCount; index += 1) {
			const particle = this.currentParticles[index];
			const area = snapshot.currentLiquidAreas[index];
			if (!particle) {
				throw new RangeError(`Particle ${index} is missing from the CPU simulation.`);
			}
			const isLiquid =
				particle.material?.model === 'water' ||
				particle.material?.model === 'oil';
			if (
				(isLiquid && (!Number.isFinite(area) || area <= 0)) ||
				(!isLiquid && area !== 0)
			) {
				throw new RangeError(`Particle ${index} has an invalid current liquid area.`);
			}
		}
		this.currentParticles = this.currentParticles.map((particle, index) => {
			const x = snapshot.positions[index * 2];
			const y = snapshot.positions[index * 2 + 1];
			if (!Number.isFinite(x) || !Number.isFinite(y)) {
				throw new RangeError(`Particle ${index} has a non-finite render position.`);
			}
			return { ...particle, position: { x, y } };
		});
		this.currentLiquidAreas = new Float32Array(snapshot.currentLiquidAreas);
		this.currentFractureBonds = this.currentFractureBonds.map((bond, index) => {
			const broken = snapshot.brokenFractureBonds[index];
			if (broken !== 0 && broken !== 1) {
				throw new RangeError(`Fracture bond ${index} has an invalid render flag.`);
			}
			return { ...bond, broken: broken === 1 };
		});
	}

	draw(
		context: CanvasRenderingContext2D,
		width: number,
		height: number,
		renderMode: MpmCpuRenderMode
	): void {
		this.assertActive();
		context.fillStyle = '#101a28';
		context.fillRect(0, 0, width, height);
		if (width <= 0 || height <= 0) return;

		const worldWidth = GATE4_GRID.cellsX * GATE4_GRID.cellSize;
		const worldHeight = GATE4_GRID.cellsY * GATE4_GRID.cellSize;
		const pixelsPerUnit = Math.min(width / worldWidth, height / worldHeight);
		const offsetX = (width - worldWidth * pixelsPerUnit) * 0.5;
		const offsetY = (height - worldHeight * pixelsPerUnit) * 0.5;

		context.save();
		context.beginPath();
		context.rect(
			offsetX,
			offsetY,
			worldWidth * pixelsPerUnit,
			worldHeight * pixelsPerUnit
		);
		context.clip();
		context.fillStyle = '#0b1521';
		context.fillRect(
			offsetX,
			offsetY,
			worldWidth * pixelsPerUnit,
			worldHeight * pixelsPerUnit
		);
		this.drawLiquidPhase(
			context,
			'water',
			'#2e8cf2',
			pixelsPerUnit,
			offsetX,
			offsetY,
			renderMode
		);
		this.drawLiquidPhase(
			context,
			'oil',
			'#f2bf38',
			pixelsPerUnit,
			offsetX,
			offsetY,
			renderMode
		);
		this.drawPebbles(context, pixelsPerUnit, offsetX, offsetY);
		this.drawFractureNetwork(context, pixelsPerUnit, offsetX, offsetY);
		context.strokeStyle = '#68809a';
		context.lineWidth = 1;
		context.strokeRect(
			offsetX,
			offsetY,
			worldWidth * pixelsPerUnit,
			worldHeight * pixelsPerUnit
		);
		context.restore();
	}

	dispose(): void {
		this.disposed = true;
		this.currentParticles = [];
		this.currentLiquidAreas = new Float32Array();
		this.currentFractureBonds = [];
		this.solverWorkspace.dispose();
	}

	private drawLiquidPhase(
		context: CanvasRenderingContext2D,
		model: 'water' | 'oil',
		color: string,
		pixelsPerUnit: number,
		offsetX: number,
		offsetY: number,
		renderMode: MpmCpuRenderMode
	): void {
		context.beginPath();
		for (let index = 0; index < this.currentParticles.length; index += 1) {
			const particle = this.currentParticles[index];
			if (!particle) {
				throw new RangeError(`Particle ${index} is missing from the CPU simulation.`);
			}
			if (particle.material?.model !== model) continue;
			const area = this.currentLiquidAreas[index];
			if (area === undefined || !Number.isFinite(area) || area <= 0) {
				throw new RangeError(`A ${model} particle has an invalid current area.`);
			}
			const radius =
				Math.sqrt(area) *
				pixelsPerUnit *
				(renderMode === 'surface' ? 0.675 : 0.24);
			const point = this.pointOnCanvas(
				particle.position.x,
				particle.position.y,
				pixelsPerUnit,
				offsetX,
				offsetY
			);
			context.moveTo(point.x + radius, point.y);
			context.arc(point.x, point.y, Math.max(1.8, radius), 0, Math.PI * 2);
		}
		context.fillStyle = color;
		context.fill();
	}

	private drawPebbles(
		context: CanvasRenderingContext2D,
		pixelsPerUnit: number,
		offsetX: number,
		offsetY: number
	): void {
		const verticesByObject = new Map<number, (MpmParticle | undefined)[]>();
		for (const particle of this.currentParticles) {
			const vertex = particle.bondedTriangleVertex;
			if (vertex === undefined) continue;
			if (particle.objectId === undefined) {
				throw new RangeError('A pebble vertex has no object id.');
			}
			const vertices = verticesByObject.get(particle.objectId) ?? [];
			vertices[vertex] = particle;
			verticesByObject.set(particle.objectId, vertices);
		}

		context.fillStyle = '#f5d77a';
		context.strokeStyle = '#fff0b8';
		context.lineWidth = 1.25;
		for (const vertices of verticesByObject.values()) {
			const first = vertices[0];
			const second = vertices[1];
			const third = vertices[2];
			if (!first || !second || !third) {
				throw new RangeError('A pebble must have exactly three drawable vertices.');
			}
			const points = [first, second, third].map(({ position }) =>
				this.pointOnCanvas(
					position.x,
					position.y,
					pixelsPerUnit,
					offsetX,
					offsetY
				)
			);
			const firstPoint = points[0];
			const secondPoint = points[1];
			const thirdPoint = points[2];
			if (!firstPoint || !secondPoint || !thirdPoint) {
				throw new RangeError('A pebble triangle is missing a drawable point.');
			}
			context.beginPath();
			context.moveTo(firstPoint.x, firstPoint.y);
			context.lineTo(secondPoint.x, secondPoint.y);
			context.lineTo(thirdPoint.x, thirdPoint.y);
			context.closePath();
			context.fill();
			context.stroke();
		}
	}

	private calculateCurrentLiquidAreas(
		particles: readonly MpmParticle[]
	): Float32Array {
		const areas = new Float32Array(particles.length);
		for (let index = 0; index < particles.length; index += 1) {
			const particle = particles[index];
			if (!particle) {
				throw new RangeError(`Particle ${index} is missing from the CPU simulation.`);
			}
			if (
				particle.material?.model !== 'water' &&
				particle.material?.model !== 'oil'
			) {
				continue;
			}
			const referenceArea = particle.volume;
			const deformation = particle.F;
			const determinant =
				deformation.m00 * deformation.m11 -
				deformation.m01 * deformation.m10;
			const area =
				referenceArea === undefined
					? Number.NaN
					: Math.fround(referenceArea * determinant);
			if (
				referenceArea === undefined ||
				!Number.isFinite(referenceArea) ||
				referenceArea <= 0 ||
				!Number.isFinite(determinant) ||
				determinant <= 0 ||
				!Number.isFinite(area) ||
				area <= 0
			) {
				throw new RangeError(`Liquid particle ${index} has an invalid current area.`);
			}
			areas[index] = area;
		}
		return areas;
	}

	private drawFractureNetwork(
		context: CanvasRenderingContext2D,
		pixelsPerUnit: number,
		offsetX: number,
		offsetY: number
	): void {
		context.beginPath();
		context.strokeStyle = 'rgba(245, 215, 122, 0.7)';
		context.lineWidth = 1;
		for (const bond of this.currentFractureBonds) {
			if (bond.broken) continue;
			const first = this.currentParticles[bond.firstParticle];
			const second = this.currentParticles[bond.secondParticle];
			if (!first || !second) {
				throw new RangeError('A fracture bond references a missing particle.');
			}
			const firstPoint = {
				x: offsetX + first.position.x * pixelsPerUnit,
				y: offsetY + first.position.y * pixelsPerUnit
			};
			const secondPoint = {
				x: offsetX + second.position.x * pixelsPerUnit,
				y: offsetY + second.position.y * pixelsPerUnit
			};
			context.moveTo(firstPoint.x, firstPoint.y);
			context.lineTo(secondPoint.x, secondPoint.y);
		}
		context.stroke();

		context.fillStyle = '#f5d77a';
		const radius = Math.sqrt(GATE4_PARTICLE_AREA / Math.PI) * pixelsPerUnit * 0.65;
		for (const particle of this.currentParticles) {
			if (particle.objectId !== GATE4_FRACTURE_OBJECT_ID) continue;
			const point = {
				x: offsetX + particle.position.x * pixelsPerUnit,
				y: offsetY + particle.position.y * pixelsPerUnit
			};
			context.beginPath();
			context.arc(point.x, point.y, Math.max(1.5, radius), 0, Math.PI * 2);
			context.fill();
		}
	}

	private pointOnCanvas(
		x: number,
		y: number,
		pixelsPerUnit: number,
		offsetX: number,
		offsetY: number
	): { readonly x: number; readonly y: number } {
		return {
			x: offsetX + x * pixelsPerUnit,
			y: offsetY + y * pixelsPerUnit
		};
	}

	private assertActive(): void {
		if (this.disposed) {
			throw new Error('The CPU reference simulation has been disposed.');
		}
	}
}
