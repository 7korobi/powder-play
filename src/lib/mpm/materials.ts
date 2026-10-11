export type MpmConstitutiveModel = 'neo-hookean' | 'snow';

export const SNOW_PLASTICITY = {
	compressionYield: 0.025,
	stretchYield: 0.0075,
	hardening: 10,
	minPlasticVolume: 0.6,
	maxPlasticVolume: 20,
	maxHardeningExponent: 20
} as const;

export function normalizeMpmConstitutiveModel(
	value: unknown,
	name: string
): MpmConstitutiveModel {
	if (value === undefined || value === 'neo-hookean') return 'neo-hookean';
	if (value === 'snow') return 'snow';
	throw new RangeError(`${name} must be 'neo-hookean' or 'snow'`);
}
