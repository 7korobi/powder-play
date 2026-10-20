import type { MpmMlsGpuConfig } from '../kernels/layout.js';
import type { MpmParticle } from '../reference/solver.js';
import {
	createMpmTaitLiquidMaterial,
	deriveTaitPressureCoefficientPa
} from '../reference/tait-eos.js';
import { MPM_SI_BOX_DOMAIN } from '../si-domain.js';

export const SI_FLUID_GRID_CELLS = 40;
export const SI_FLUID_GRID = {
	cellsX: SI_FLUID_GRID_CELLS,
	cellsY: SI_FLUID_GRID_CELLS,
	cellSize: MPM_SI_BOX_DOMAIN.innerWidthM / SI_FLUID_GRID_CELLS
} as const;
export const SI_FLUID_PARTICLE_SPACING_M = 0.0025;
export const SI_FLUID_FILL_HEIGHT_M = 0.05;
export const SI_FLUID_EFFECTIVE_SOUND_SPEED_M_PER_S = 20;
export const SI_FLUID_COURANT_LIMIT = 0.3;
export const SI_FLUID_FIXED_STEP_SECONDS = 1 / 120;
export const SI_FLUID_MAX_MICROSTEPS_PER_BATCH = 512;
export const SI_FLUID_WALL_FRICTION = 1;

export const SI_TAIT_WATER_PARAMETERS = {
	referenceDensityKgPerM3: 998.2,
	exponent: 7.15,
	pressureCoefficientPa: deriveTaitPressureCoefficientPa(
		998.2,
		7.15,
		SI_FLUID_EFFECTIVE_SOUND_SPEED_M_PER_S
	),
	ambientPressurePa: 101_325
} as const;

export interface SiFluidScene {
	readonly particles: readonly MpmParticle[];
	readonly config: MpmMlsGpuConfig;
}

export function createSiFluidScene(): SiFluidScene {
	const material = createMpmTaitLiquidMaterial(
		SI_TAIT_WATER_PARAMETERS,
		MPM_SI_BOX_DOMAIN.extrusionDepthM
	);
	const particleArea = SI_FLUID_PARTICLE_SPACING_M ** 2;
	const columns = Math.floor(
		MPM_SI_BOX_DOMAIN.innerWidthM / SI_FLUID_PARTICLE_SPACING_M
	);
	const rows = Math.floor(SI_FLUID_FILL_HEIGHT_M / SI_FLUID_PARTICLE_SPACING_M);
	const particles: MpmParticle[] = [];

	for (let row = 0; row < rows; row += 1) {
		for (let column = 0; column < columns; column += 1) {
			particles.push({
				position: {
					x: (column + 0.5) * SI_FLUID_PARTICLE_SPACING_M,
					y:
						MPM_SI_BOX_DOMAIN.innerHeightM -
						SI_FLUID_FILL_HEIGHT_M +
						(row + 0.5) * SI_FLUID_PARTICLE_SPACING_M
				},
				velocity: { x: 0, y: 0 },
				C: { m00: 0, m01: 0, m10: 0, m11: 0 },
				F: { m00: 1, m01: 0, m10: 0, m11: 1 },
				volume: particleArea
			});
		}
	}

	return {
		particles,
		config: {
			grid: SI_FLUID_GRID,
			particleMass: material.density * particleArea,
			particleVolume: particleArea,
			material,
			wallFriction: SI_FLUID_WALL_FRICTION
		}
	};
}
