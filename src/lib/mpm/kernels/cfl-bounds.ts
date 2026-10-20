import type { MpmMaterialParameters, MpmParticle } from '../reference/solver.js';
import type { MpmMlsGpuConfig } from './layout.js';

export interface MpmMlsGpuCflBounds {
	readonly waveSpeedBound: number;
	readonly velocityBound: number;
}

export interface MpmCflVelocityEnvelopeInput {
	readonly currentVelocityBound: number;
	readonly durationSeconds: number;
	readonly gravityMagnitude: number;
	readonly oscillationAmplitude: number;
	readonly dragRadius: number;
	readonly dragStiffness: number;
	readonly dragDamping: number;
}

const F32_ROUNDING_ULPS = 8;

/**
 * Derive reference/current-state wave-speed bounds and the actual initial
 * particle-speed bound from the configured particle/material data.
 *
 * Neo-Hookean speed uses an upper bound on the current acoustic tensor, with
 * the largest singular value bounded analytically in 2D. Gate 4's tension-free
 * liquids use pressure = lambda * min(J - 1, 0), giving c² = lambda / rho₀.
 * Snow and powder are deliberately unsupported until their tangent bounds are
 * derived; returning null keeps direct non-CFL solver use available while
 * making CFL use fail explicitly.
 */
export function deriveMpmMlsGpuCflBounds(
	particles: readonly MpmParticle[],
	config: Pick<MpmMlsGpuConfig, 'particleMass' | 'particleVolume' | 'material'>
): MpmMlsGpuCflBounds | null {
	let velocityBound = 0;
	let waveSpeedBound = 0;

	for (let particleIndex = 0; particleIndex < particles.length; particleIndex += 1) {
		const particle = particles[particleIndex];
		const vx = Math.fround(particle.velocity.x);
		const vy = Math.fround(particle.velocity.y);
		if (!Number.isFinite(vx) || !Number.isFinite(vy)) {
			throw new RangeError(`particle ${particleIndex} has a non-finite initial velocity.`);
		}
		velocityBound = Math.max(velocityBound, Math.hypot(vx, vy));

		const material = particle.material ?? config.material;
		const particleWaveSpeed = deriveParticleWaveSpeedBound(
			particle,
			material,
			config
		);
		if (particleWaveSpeed === null) return null;
		waveSpeedBound = Math.max(waveSpeedBound, particleWaveSpeed);
	}

	return Object.freeze({
		waveSpeedBound: roundUpFloat32(waveSpeedBound),
		velocityBound: roundUpFloat32(velocityBound)
	});
}

function deriveParticleWaveSpeedBound(
	particle: MpmParticle,
	material: MpmMaterialParameters,
	config: Pick<MpmMlsGpuConfig, 'particleMass' | 'particleVolume'>
): number | null {
	const lambda = Math.fround(material.lambda);
	const mu = Math.fround(material.mu);
	if (!Number.isFinite(lambda) || !Number.isFinite(mu)) {
		throw new RangeError('CFL material parameters must be finite float32 values.');
	}

	if (material.model === 'water' || material.model === 'oil') {
		const density = Math.fround(material.density);
		if (!Number.isFinite(density) || density <= 0 || lambda <= 0) {
			throw new RangeError('liquid CFL parameters require positive finite lambda and density.');
		}
		if (material.equationOfState?.model !== 'tait') {
			return Math.sqrt(lambda / density);
		}
		const exponent = Math.fround(material.equationOfState.exponent);
		const f00 = Math.fround(particle.F.m00);
		const f01 = Math.fround(particle.F.m01);
		const f10 = Math.fround(particle.F.m10);
		const f11 = Math.fround(particle.F.m11);
		const determinant = Math.fround(f00 * f11 - f01 * f10);
		if (
			!Number.isFinite(exponent) ||
			exponent <= 0 ||
			!Number.isFinite(determinant) ||
			determinant <= 0
		) {
			throw new RangeError('Tait CFL bounds require a positive finite exponent and determinant.');
		}
		const waveSpeedSquared =
			(exponent * lambda * Math.pow(1 / determinant, exponent - 1)) / density;
		if (!Number.isFinite(waveSpeedSquared) || waveSpeedSquared < 0) {
			throw new RangeError('Tait CFL wave-speed bound is outside the finite range.');
		}
		return Math.sqrt(waveSpeedSquared);
	}
	if (material.model !== undefined && material.model !== 'neo-hookean') return null;
	if (lambda < 0 || mu < 0) {
		throw new RangeError('Neo-Hookean CFL parameters must be non-negative.');
	}

	const volume = Math.fround(particle.volume ?? config.particleVolume);
	const mass = Math.fround(config.particleMass);
	const referenceDensity = mass / volume;
	const f00 = Math.fround(particle.F.m00);
	const f01 = Math.fround(particle.F.m01);
	const f10 = Math.fround(particle.F.m10);
	const f11 = Math.fround(particle.F.m11);
	const determinant = f00 * f11 - f01 * f10;
	if (
		!Number.isFinite(mass) ||
		mass <= 0 ||
		!Number.isFinite(volume) ||
		volume <= 0 ||
		!Number.isFinite(referenceDensity) ||
		referenceDensity <= 0 ||
		!Number.isFinite(determinant) ||
		determinant <= 0
	) {
		throw new RangeError('Neo-Hookean CFL bounds require positive density and determinant.');
	}

	const deformationTrace = f00 * f00 + f01 * f01 + f10 * f10 + f11 * f11;
	const singularValueDiscriminant = Math.sqrt(
		Math.max(0, deformationTrace * deformationTrace - 4 * determinant * determinant)
	);
	const maximumStretchSquared = 0.5 * (deformationTrace + singularValueDiscriminant);
	const volumetricTangent = Math.abs(lambda + mu - lambda * Math.log(determinant));
	const waveSpeedSquared =
		(mu * maximumStretchSquared + volumetricTangent) / referenceDensity;
	if (!Number.isFinite(waveSpeedSquared) || waveSpeedSquared < 0) {
		throw new RangeError('Neo-Hookean CFL wave-speed bound is outside the finite range.');
	}
	return Math.sqrt(waveSpeedSquared);
}

/**
 * Bound speed over one scheduled interval from the observed current maximum
 * and the configured external acceleration envelope. The interaction spring
 * magnitude is at most stiffness * radius. Drag damping cannot increase the
 * grid speed when its explicit Euler factor remains in [0, 1]; the CFL solver
 * enforces that condition for every submitted microstep. Internal
 * constitutive/contact changes are separately rejected by the GPU
 * per-microstep guard.
 */
export function calculateMpmCflVelocityBound({
	currentVelocityBound,
	durationSeconds,
	gravityMagnitude,
	oscillationAmplitude,
	dragRadius,
	dragStiffness,
	dragDamping
}: MpmCflVelocityEnvelopeInput): number {
	const nonNegativeInputs = [
		['currentVelocityBound', currentVelocityBound],
		['durationSeconds', durationSeconds],
		['gravityMagnitude', gravityMagnitude],
		['oscillationAmplitude', oscillationAmplitude],
		['dragRadius', dragRadius],
		['dragStiffness', dragStiffness],
		['dragDamping', dragDamping]
	] as const;
	for (const [name, value] of nonNegativeInputs) {
		if (!Number.isFinite(value) || value < 0) {
			throw new RangeError(`${name} must be finite and non-negative.`);
		}
	}

	const boundedAcceleration =
		Math.hypot(gravityMagnitude, oscillationAmplitude) + dragRadius * dragStiffness;
	const velocityBound = currentVelocityBound + boundedAcceleration * durationSeconds;
	if (!Number.isFinite(velocityBound)) {
		throw new RangeError('external-input velocity bound exceeds the finite numeric range.');
	}
	return roundUpFloat32(velocityBound);
}

function roundUpFloat32(value: number): number {
	if (!Number.isFinite(value) || value < 0) {
		throw new RangeError('CFL bounds must be finite and non-negative.');
	}
	if (value === 0) return 0;

	const data = new DataView(new ArrayBuffer(Float32Array.BYTES_PER_ELEMENT));
	data.setFloat32(0, Math.fround(value), true);
	let bits = data.getUint32(0, true);
	while (data.getFloat32(0, true) < value) {
		bits += 1;
		data.setUint32(0, bits, true);
	}
	for (let ulp = 0; ulp < F32_ROUNDING_ULPS; ulp += 1) {
		bits += 1;
		data.setUint32(0, bits, true);
	}
	const rounded = data.getFloat32(0, true);
	if (!Number.isFinite(rounded)) {
		throw new RangeError('CFL bound cannot be represented as a finite float32 value.');
	}
	return rounded;
}
