export type MaterialPhase = 'gas' | 'liquid' | 'granular' | 'solid';
export type Rgba = readonly [number, number, number, number];

export interface MaterialPlugin {
	id: number;
	name: string;
	phase: MaterialPhase;
	density: number;
	color: Rgba;
	repulsion?: number;
	packingDensity?: number;
	particleRadius?: number;
}

export class MaterialRegistry {
	private readonly definitions = new Map<number, MaterialPlugin>();

	constructor(plugins: readonly MaterialPlugin[] = []) {
		for (const plugin of plugins) this.register(plugin);
	}

	register(plugin: MaterialPlugin): void {
		if (!Number.isInteger(plugin.id) || plugin.id < 0 || plugin.id > 255) {
			throw new RangeError('Material ids must be integers between 0 and 255.');
		}
		if (this.definitions.has(plugin.id)) {
			throw new Error(`Material id ${plugin.id} is already registered.`);
		}
		if (!Number.isFinite(plugin.density) || plugin.density < 0) {
			throw new RangeError('Material density must be a non-negative number.');
		}
		if (
			plugin.color.length !== 4 ||
			plugin.color.some((component) => !Number.isInteger(component) || component < 0 || component > 255)
		) {
			throw new RangeError('Material colors must be RGBA byte values.');
		}
		if (
			plugin.repulsion !== undefined &&
			(!Number.isFinite(plugin.repulsion) || plugin.repulsion < 0 || plugin.repulsion > 1)
		) {
			throw new RangeError('Material repulsion must be between 0 and 1.');
		}
		if (
			plugin.packingDensity !== undefined &&
			(!Number.isFinite(plugin.packingDensity) ||
				plugin.packingDensity < 0 ||
				plugin.packingDensity > 1)
		) {
			throw new RangeError('Material packing density must be between 0 and 1.');
		}
		if (
			plugin.particleRadius !== undefined &&
			(!Number.isFinite(plugin.particleRadius) ||
				plugin.particleRadius <= 0 ||
				plugin.particleRadius > 1)
		) {
			throw new RangeError('Material particle radius must be greater than 0 and at most 1.');
		}

		this.definitions.set(plugin.id, Object.freeze({ ...plugin, color: [...plugin.color] as Rgba }));
	}

	get(id: number): MaterialPlugin {
		const plugin = this.definitions.get(id);
		if (!plugin) throw new RangeError(`Material id ${id} is not registered.`);
		return plugin;
	}

	has(id: number): boolean {
		return this.definitions.has(id);
	}
}

export const MATERIAL = {
	air: 0,
	water: 1,
	powder: 2,
	oil: 3
} as const;

export type MaterialId = (typeof MATERIAL)[keyof typeof MATERIAL];

export const materialRegistry = new MaterialRegistry([
	{
		id: MATERIAL.air,
		name: '空気',
		phase: 'gas',
		density: 0,
		color: [0, 0, 0, 0],
		repulsion: 0.0006
	},
	{
		id: MATERIAL.water,
		name: '水',
		phase: 'liquid',
		density: 2,
		color: [28, 111, 238, 26],
		packingDensity: 0.18,
		particleRadius: 0.9
	},
	{
		id: MATERIAL.powder,
		name: '粉',
		phase: 'granular',
		density: 1,
		color: [210, 220, 230, 255],
		packingDensity: 0.11,
		particleRadius: 0.76
	},
	{
		id: MATERIAL.oil,
		name: '油',
		phase: 'liquid',
		density: 1,
		color: [224, 154, 46, 180],
		packingDensity: 0.18,
		particleRadius: 0.9
	}
]);
