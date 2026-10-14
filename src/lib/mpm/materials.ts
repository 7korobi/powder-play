export type MpmConstitutiveModel = 'neo-hookean' | 'snow' | 'water' | 'oil' | 'powder';

export const MPM_CONSTITUTIVE_MODEL_CODES: Readonly<
	Record<MpmConstitutiveModel, number>
> = {
	'neo-hookean': 0,
	snow: 1,
	water: 2,
	oil: 3,
	powder: 4
};

export const MPM_LIQUID_CONTACT_SEPARATION_FRACTION = 0.25;
export const MPM_LIQUID_CONTACT_APPROACH_SPEED_MULTIPLIER = 1;
export const MPM_LIQUID_CONTACT_RELAXATION_SECONDS = 0.05;
export const MPM_LIQUID_STABLE_ORDER_EPSILON = 1e-3;
export const MPM_TRIANGLE_FLUID_TANGENTIAL_DRAG_RATE = 4;

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
	if (value === 'snow' || value === 'water' || value === 'oil' || value === 'powder') {
		return value;
	}
	throw new RangeError(
		`${name} must be 'neo-hookean', 'snow', 'water', 'oil', or 'powder'`
	);
}

export function encodeMpmConstitutiveModel(model: MpmConstitutiveModel): number {
	return MPM_CONSTITUTIVE_MODEL_CODES[model];
}

export function decodeMpmConstitutiveModel(code: number): MpmConstitutiveModel {
	switch (code) {
		case 0:
			return 'neo-hookean';
		case 1:
			return 'snow';
		case 2:
			return 'water';
		case 3:
			return 'oil';
		case 4:
			return 'powder';
		default:
			throw new RangeError(`GPU returned an unsupported MPM material model ${code}.`);
	}
}
