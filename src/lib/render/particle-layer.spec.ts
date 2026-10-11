import { describe, expect, it } from 'vitest';
import { MATERIAL, materialRegistry } from '#lib/sim/materials.js';
import { SimulationWorld } from '#lib/sim/world.js';
import {
	PARTICLE_LAYER_SCALE,
	renderParticleLayer,
	type ParticleLayerCanvas,
	type ParticleLayerContext
} from './particle-layer.js';

describe('particle layer rendering', () => {
	it('reuses a bounded supersampled canvas across particle updates', () => {
		let width = 0;
		let height = 0;
		let resizeCount = 0;
		const canvas: ParticleLayerCanvas = {
			get width() {
				return width;
			},
			set width(value: number) {
				width = value;
				resizeCount++;
			},
			get height() {
				return height;
			},
			set height(value: number) {
				height = value;
				resizeCount++;
			}
		};
		let transformCount = 0;
		let beginPathCount = 0;
		let arcCount = 0;
		let fillCount = 0;
		const context: ParticleLayerContext = {
			setTransform() {
				transformCount++;
			},
			clearRect() {},
			beginPath() {
				beginPathCount++;
			},
			moveTo() {},
			arc() {
				arcCount++;
			},
			fill() {
				fillCount++;
			},
			fillStyle: ''
		};
		const world = new SimulationWorld(4, 3);
		const waterIndex = 1 * world.width + 1;
		const powderIndex = 1 * world.width + 2;
		const particles = Array.from({ length: 256 }, () => [] as number[]);
		particles[MATERIAL.water].push(waterIndex);
		particles[MATERIAL.powder].push(powderIndex);
		const palette = new Uint8ClampedArray(256 * 4);
		const radii = new Float32Array(256).fill(-1);
		for (const material of [MATERIAL.water, MATERIAL.powder]) {
			const plugin = materialRegistry.get(material);
			palette.set(plugin.color, material * 4);
			radii[material] = plugin.particleRadius ?? -1;
		}

		renderParticleLayer(canvas, context, world, particles, palette, radii);
		renderParticleLayer(canvas, context, world, particles, palette, radii);

		expect(canvas.width).toBe(world.width * PARTICLE_LAYER_SCALE);
		expect(canvas.height).toBe(world.height * PARTICLE_LAYER_SCALE);
		expect(resizeCount).toBe(2);
		expect(transformCount).toBe(2);
		expect(beginPathCount).toBe(4);
		expect(arcCount).toBe(4);
		expect(fillCount).toBe(4);
	});
});
