import { describe, expect, it } from 'vitest';
import { MPM_SI_BOX_DOMAIN } from '../si-domain.js';
import {
	createSiFluidScene,
	SI_FLUID_EFFECTIVE_SOUND_SPEED_M_PER_S,
	SI_FLUID_FILL_HEIGHT_M,
	SI_FLUID_GRID,
	SI_FLUID_PARTICLE_SPACING_M,
	SI_FLUID_WALL_FRICTION,
	SI_TAIT_WATER_PARAMETERS
} from './si-fluid.js';

describe('SI Tait water scene', () => {
	it('fills a 200 mm closed-box test region with evenly spaced particles', () => {
		const { particles, config } = createSiFluidScene();

		expect(config.grid).toEqual(SI_FLUID_GRID);
		expect(config.grid.cellsX * config.grid.cellSize).toBe(
			MPM_SI_BOX_DOMAIN.innerWidthM
		);
		expect(config.grid.cellsY * config.grid.cellSize).toBe(
			MPM_SI_BOX_DOMAIN.innerHeightM
		);
		expect(particles).toHaveLength(80 * 20);
		expect(particles[0].position).toEqual({
			x: SI_FLUID_PARTICLE_SPACING_M / 2,
			y:
				MPM_SI_BOX_DOMAIN.innerHeightM -
				SI_FLUID_FILL_HEIGHT_M +
				SI_FLUID_PARTICLE_SPACING_M / 2
		});
		expect(particles.at(-1)?.position.y).toBeLessThan(
			MPM_SI_BOX_DOMAIN.innerHeightM
		);
		expect(particles.at(-1)?.position.x).toBeLessThan(
			MPM_SI_BOX_DOMAIN.innerWidthM
		);
	});

	it('uses the 20 mm extrusion depth and labels the relaxed sound-speed profile', () => {
		const { config } = createSiFluidScene();
		const material = config.material;

		expect(MPM_SI_BOX_DOMAIN.extrusionDepthM).toBe(0.02);
		expect(material.model).toBe('water');
		if (material.model !== 'water') throw new Error('Expected a water material.');
		expect(material.equationOfState).toEqual({ model: 'tait', exponent: 7.15 });
		expect(material.density).toBeCloseTo(
			SI_TAIT_WATER_PARAMETERS.referenceDensityKgPerM3 *
				MPM_SI_BOX_DOMAIN.extrusionDepthM
		);
		expect(material.lambda).toBeCloseTo(
			SI_TAIT_WATER_PARAMETERS.pressureCoefficientPa *
				MPM_SI_BOX_DOMAIN.extrusionDepthM
		);
		expect(SI_FLUID_EFFECTIVE_SOUND_SPEED_M_PER_S).toBe(20);
		expect(config.wallFriction).toBe(SI_FLUID_WALL_FRICTION);
	});
});
