import type { SimulationWorld } from '#lib/sim/world.js';

export const PARTICLE_LAYER_SCALE = 4;

export type ParticleLayerCanvas = Pick<HTMLCanvasElement, 'width' | 'height'>;
export type ParticleLayerContext = Pick<
	CanvasRenderingContext2D,
	'setTransform' | 'clearRect' | 'beginPath' | 'moveTo' | 'arc' | 'fillStyle' | 'fill'
>;

export function renderParticleLayer(
	canvas: ParticleLayerCanvas,
	context: ParticleLayerContext,
	world: SimulationWorld,
	particleIndicesByMaterial: readonly number[][],
	palette: Uint8ClampedArray,
	particleRadii: Float32Array
): void {
	const width = world.width * PARTICLE_LAYER_SCALE;
	const height = world.height * PARTICLE_LAYER_SCALE;
	if (canvas.width !== width) canvas.width = width;
	if (canvas.height !== height) canvas.height = height;

	context.setTransform(PARTICLE_LAYER_SCALE, 0, 0, PARTICLE_LAYER_SCALE, 0, 0);
	context.clearRect(0, 0, world.width, world.height);
	for (let material = 0; material < particleIndicesByMaterial.length; material++) {
		const indices = particleIndicesByMaterial[material];
		if (!indices || indices.length === 0) continue;

		const radius = particleRadii[material];
		context.beginPath();
		for (const index of indices) {
			const x = index % world.width;
			const y = Math.floor(index / world.width);
			context.moveTo(x + 0.5 + radius, y + 0.5);
			context.arc(x + 0.5, y + 0.5, radius, 0, Math.PI * 2);
		}
		const colorOffset = material * 4;
		context.fillStyle = `rgba(${palette[colorOffset]}, ${palette[colorOffset + 1]}, ${
			palette[colorOffset + 2]
		}, ${palette[colorOffset + 3] / 255})`;
		context.fill();
	}
}
