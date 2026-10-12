export const MLS_MPM_COMPUTE_WGSL = /* wgsl */ `
struct Particle {
	position: vec2<f32>,
	velocity: vec2<f32>,
	c: vec4<f32>,
	f: vec4<f32>,
	plasticVolume: f32,
	materialModel: u32,
	lambda: f32,
	mu: f32,
	density: f32,
	volume: f32,
	frictionCoefficient: f32,
}

struct GridNode {
	mass: f32,
	momentum: vec2<f32>,
	velocity: vec2<f32>,
}

struct SimulationParams {
	grid: vec4<u32>,
	counts: vec4<u32>,
	physical: vec4<f32>,
	material: vec4<f32>,
	gravityAndDomain: vec4<f32>,
}

struct InteractionParams {
	targetAndRadius: vec4<f32>,
	force: vec4<f32>,
}

struct PickParams {
	pointAndRadius: vec4<f32>,
	countAndPadding: vec4<u32>,
}

struct CloneParams {
	rangeAndObjectId: vec4<u32>,
	offsetAndDomain: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: SimulationParams;
@group(0) @binding(1) var<storage, read> particlesIn: array<Particle>;
@group(0) @binding(2) var<storage, read_write> particlesOut: array<Particle>;
@group(0) @binding(3) var<storage, read_write> bucketHeads: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> bucketNext: array<i32>;
@group(0) @binding(5) var<storage, read_write> gridNodes: array<GridNode>;
@group(0) @binding(6) var<storage, read_write> stateError: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> interactionParams: InteractionParams;
@group(0) @binding(8) var<storage, read> particleObjectIds: array<u32>;
@group(0) @binding(9) var<uniform> pickParams: PickParams;
@group(0) @binding(10) var<storage, read_write> pickResult: array<u32>;
@group(0) @binding(11) var<storage, read_write> cloneParticles: array<Particle>;
@group(0) @binding(12) var<storage, read_write> cloneObjectIds: array<u32>;
@group(0) @binding(13) var<uniform> cloneParams: CloneParams;

const ERROR_STENCIL: u32 = 1u;
const ERROR_DEFORMATION: u32 = 2u;
const ERROR_P2G: u32 = 4u;
const ERROR_GRID: u32 = 8u;
const ERROR_G2P: u32 = 16u;
const MODEL_NEO_HOOKEAN: u32 = 0u;
const MODEL_SNOW: u32 = 1u;
const MODEL_WATER: u32 = 2u;
const MODEL_OIL: u32 = 3u;
const MODEL_POWDER: u32 = 4u;
const SNOW_COMPRESSION_YIELD: f32 = 0.025;
const SNOW_STRETCH_YIELD: f32 = 0.0075;
const SNOW_HARDENING: f32 = 10.0;
const SNOW_MIN_PLASTIC_VOLUME: f32 = 0.6;
const SNOW_MAX_PLASTIC_VOLUME: f32 = 20.0;
const SNOW_MAX_HARDENING_EXPONENT: f32 = 20.0;

struct Decomposition2 {
	sigma: vec2<f32>,
	u0: vec2<f32>,
	u1: vec2<f32>,
	v0: vec2<f32>,
	v1: vec2<f32>,
}

fn finiteFloat(value: f32) -> bool {
	return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

fn finiteVec2(value: vec2<f32>) -> bool {
	return finiteFloat(value.x) && finiteFloat(value.y);
}

fn finiteVec4(value: vec4<f32>) -> bool {
	return finiteFloat(value.x) &&
		finiteFloat(value.y) &&
		finiteFloat(value.z) &&
		finiteFloat(value.w);
}

fn decompose2x2(matrix: vec4<f32>) -> Decomposition2 {
	let c00 = matrix.x * matrix.x + matrix.z * matrix.z;
	let c01 = matrix.x * matrix.y + matrix.z * matrix.w;
	let c11 = matrix.y * matrix.y + matrix.w * matrix.w;
	let radius = sqrt((c00 - c11) * (c00 - c11) + 4.0 * c01 * c01);
	let sigma0 = sqrt(max(0.0, 0.5 * (c00 + c11 + radius)));
	let determinant = matrix.x * matrix.w - matrix.y * matrix.z;
	let sigma1 = determinant / max(sigma0, 1.0e-30);
	let angle = 0.5 * atan2(2.0 * c01, c00 - c11);
	let cosine = cos(angle);
	let sine = sin(angle);
	let v0 = vec2<f32>(cosine, sine);
	let v1 = vec2<f32>(-sine, cosine);
	let u0 = vec2<f32>(
		(matrix.x * v0.x + matrix.y * v0.y) / max(sigma0, 1.0e-30),
		(matrix.z * v0.x + matrix.w * v0.y) / max(sigma0, 1.0e-30),
	);
	let u1 = vec2<f32>(
		(matrix.x * v1.x + matrix.y * v1.y) / max(sigma1, 1.0e-30),
		(matrix.z * v1.x + matrix.w * v1.y) / max(sigma1, 1.0e-30),
	);
	return Decomposition2(vec2<f32>(sigma0, sigma1), u0, u1, v0, v1);
}

fn polarRotation(decomposition: Decomposition2) -> vec4<f32> {
	return vec4<f32>(
		decomposition.u0.x * decomposition.v0.x +
			decomposition.u1.x * decomposition.v1.x,
		decomposition.u0.x * decomposition.v0.y +
			decomposition.u1.x * decomposition.v1.y,
		decomposition.u0.y * decomposition.v0.x +
			decomposition.u1.y * decomposition.v1.x,
		decomposition.u0.y * decomposition.v0.y +
			decomposition.u1.y * decomposition.v1.y,
	);
}

fn reconstructDeformation(
	decomposition: Decomposition2,
	sigma: vec2<f32>,
) -> vec4<f32> {
	return vec4<f32>(
		sigma.x * decomposition.u0.x * decomposition.v0.x +
			sigma.y * decomposition.u1.x * decomposition.v1.x,
		sigma.x * decomposition.u0.x * decomposition.v0.y +
			sigma.y * decomposition.u1.x * decomposition.v1.y,
		sigma.x * decomposition.u0.y * decomposition.v0.x +
			sigma.y * decomposition.u1.y * decomposition.v1.x,
		sigma.x * decomposition.u0.y * decomposition.v0.y +
			sigma.y * decomposition.u1.y * decomposition.v1.y,
	);
}

fn granularStressTimesFt(
	matrix: vec4<f32>,
	lambda: f32,
	mu: f32,
	frictionCoefficient: f32,
) -> vec4<f32> {
	let decomposition = decompose2x2(matrix);
	let logStretch = log(decomposition.sigma);
	let logVolume = logStretch.x + logStretch.y;
	let pressure = max(0.0, -(lambda + mu) * logVolume);
	let deviatoricLogStretch = 0.5 * (logStretch.x - logStretch.y);
	let maximumDeviatoricLogStretch =
		frictionCoefficient * pressure / (2.0 * mu);
	let projectedDeviatoricLogStretch =
		min(deviatoricLogStretch, maximumDeviatoricLogStretch);
	let meanLogStretch = 0.5 * logVolume;
	let principalStress = vec2<f32>(
		2.0 * mu * (meanLogStretch + projectedDeviatoricLogStretch) +
			lambda * logVolume,
		2.0 * mu * (meanLogStretch - projectedDeviatoricLogStretch) +
			lambda * logVolume,
	);
	return vec4<f32>(
		principalStress.x * decomposition.u0.x * decomposition.u0.x +
			principalStress.y * decomposition.u1.x * decomposition.u1.x,
		principalStress.x * decomposition.u0.x * decomposition.u0.y +
			principalStress.y * decomposition.u1.x * decomposition.u1.y,
		principalStress.x * decomposition.u0.y * decomposition.u0.x +
			principalStress.y * decomposition.u1.y * decomposition.u1.x,
		principalStress.x * decomposition.u0.y * decomposition.u0.y +
			principalStress.y * decomposition.u1.y * decomposition.u1.y,
	);
}

fn projectGranularDeformation(
	matrix: vec4<f32>,
	lambda: f32,
	mu: f32,
	frictionCoefficient: f32,
) -> vec4<f32> {
	let decomposition = decompose2x2(matrix);
	let logStretch = log(decomposition.sigma);
	let logVolume = logStretch.x + logStretch.y;
	let pressure = max(0.0, -(lambda + mu) * logVolume);
	let deviatoricLogStretch = 0.5 * (logStretch.x - logStretch.y);
	let maximumDeviatoricLogStretch =
		frictionCoefficient * pressure / (2.0 * mu);
	let projectedDeviatoricLogStretch =
		min(deviatoricLogStretch, maximumDeviatoricLogStretch);
	let meanLogStretch = 0.5 * logVolume;
	let projectedSigma = exp(vec2<f32>(
		meanLogStretch + projectedDeviatoricLogStretch,
		meanLogStretch - projectedDeviatoricLogStretch,
	));
	return reconstructDeformation(decomposition, projectedSigma);
}

fn quadraticWeights(fraction: f32) -> vec3<f32> {
	let left = 1.5 - fraction;
	let center = fraction - 1.0;
	let right = fraction - 0.5;
	return vec3<f32>(
		0.5 * left * left,
		0.75 - center * center,
		0.5 * right * right,
	);
}

fn stencilFitsHalo(base: vec2<i32>) -> bool {
	return base.x >= -1 &&
		base.y >= -1 &&
		base.x + 2 <= i32(params.grid.x + 1u) &&
		base.y + 2 <= i32(params.grid.y + 1u);
}

fn gridIndex(node: vec2<i32>) -> u32 {
	return u32(node.y + 1) * params.grid.z + u32(node.x + 1);
}

fn confineAxis(
	nextPosition: f32,
	velocity: f32,
	maximum: f32,
) -> vec2<f32> {
	if (nextPosition < 0.0 || (nextPosition == 0.0 && velocity < 0.0)) {
		return vec2<f32>(0.0, 0.0);
	}
	if (nextPosition > maximum || (nextPosition == maximum && velocity > 0.0)) {
		return vec2<f32>(maximum, 0.0);
	}
	return vec2<f32>(nextPosition, velocity);
}

@compute @workgroup_size(64)
fn clearGrid(@builtin(global_invocation_id) invocation: vec3<u32>) {
	let index = invocation.x;
	if (index >= params.counts.y) {
		return;
	}
	atomicStore(&bucketHeads[index], -1);
	gridNodes[index].mass = 0.0;
	gridNodes[index].momentum = vec2<f32>(0.0, 0.0);
	gridNodes[index].velocity = vec2<f32>(0.0, 0.0);
}

@compute @workgroup_size(64)
fn buildBuckets(@builtin(global_invocation_id) invocation: vec3<u32>) {
	let particleIndex = invocation.x;
	if (particleIndex >= params.counts.x) {
		return;
	}
	if (atomicLoad(&stateError[0]) != 0u) {
		return;
	}

	let particle = particlesIn[particleIndex];
	let gridPosition = particle.position / params.physical.x;
	if (!finiteVec2(gridPosition)) {
		atomicOr(&stateError[0], ERROR_STENCIL);
		return;
	}
	let base = vec2<i32>(floor(gridPosition - vec2<f32>(0.5, 0.5)));
	if (!stencilFitsHalo(base)) {
		atomicOr(&stateError[0], ERROR_STENCIL);
		return;
	}

	for (var oy = 0u; oy < 3u; oy = oy + 1u) {
		for (var ox = 0u; ox < 3u; ox = ox + 1u) {
			let node = base + vec2<i32>(i32(ox), i32(oy));
			let nodeIndex = gridIndex(node);
			if (nodeIndex >= params.counts.y) {
				atomicOr(&stateError[0], ERROR_STENCIL);
				return;
			}
			let pairIndex = particleIndex * 9u + oy * 3u + ox;
			let previousHead = atomicExchange(&bucketHeads[nodeIndex], i32(pairIndex));
			bucketNext[pairIndex] = previousHead;
		}
	}
}

@compute @workgroup_size(64)
fn p2g(@builtin(global_invocation_id) invocation: vec3<u32>) {
	let nodeIndex = invocation.x;
	if (nodeIndex >= params.counts.y) {
		return;
	}
	if (atomicLoad(&stateError[0]) != 0u) {
		return;
	}

	var mass = 0.0;
	var momentum = vec2<f32>(0.0, 0.0);
	var entry = atomicLoad(&bucketHeads[nodeIndex]);
	var visited = 0u;

	loop {
		if (entry == -1) {
			break;
		}
		if (entry < -1 || visited >= params.counts.x) {
			atomicOr(&stateError[0], ERROR_STENCIL);
			return;
		}
		let pairIndex = u32(entry);
		if (pairIndex >= params.counts.z) {
			atomicOr(&stateError[0], ERROR_STENCIL);
			return;
		}
		let particleIndex = pairIndex / 9u;
		if (particleIndex >= params.counts.x) {
			atomicOr(&stateError[0], ERROR_STENCIL);
			return;
		}

		let supportIndex = pairIndex % 9u;
		let ox = supportIndex % 3u;
		let oy = supportIndex / 3u;
		let particle = particlesIn[particleIndex];
		let gridPosition = particle.position / params.physical.x;
		if (!finiteVec2(gridPosition)) {
			atomicOr(&stateError[0], ERROR_STENCIL);
			return;
		}
		let base = vec2<i32>(floor(gridPosition - vec2<f32>(0.5, 0.5)));
		let fraction = gridPosition - vec2<f32>(base);
		let weightsX = quadraticWeights(fraction.x);
		let weightsY = quadraticWeights(fraction.y);
		let weight = weightsX[ox] * weightsY[oy];
		let supportNode = base + vec2<i32>(i32(ox), i32(oy));
		if (!stencilFitsHalo(base) || gridIndex(supportNode) != nodeIndex) {
			atomicOr(&stateError[0], ERROR_STENCIL);
			return;
		}

		let determinant = particle.f.x * particle.f.w - particle.f.y * particle.f.z;
		if (
			!finiteFloat(determinant) ||
			determinant <= 0.0 ||
			!finiteFloat(particle.volume) ||
			particle.volume <= 0.0
		) {
			atomicOr(&stateError[0], ERROR_DEFORMATION);
			return;
		}
		var stressTimesFt = vec4<f32>(0.0, 0.0, 0.0, 0.0);
		if (particle.materialModel == MODEL_NEO_HOOKEAN) {
			let inverseTranspose =
				vec4<f32>(particle.f.w, -particle.f.z, -particle.f.y, particle.f.x) /
				determinant;
			let pressure = particle.lambda * log(determinant);
			let firstPiola =
				particle.mu * (particle.f - inverseTranspose) +
				pressure * inverseTranspose;
			stressTimesFt = vec4<f32>(
				firstPiola.x * particle.f.x + firstPiola.y * particle.f.y,
				firstPiola.x * particle.f.z + firstPiola.y * particle.f.w,
				firstPiola.z * particle.f.x + firstPiola.w * particle.f.y,
				firstPiola.z * particle.f.z + firstPiola.w * particle.f.w,
			);
		} else if (particle.materialModel == MODEL_SNOW) {
			if (
				!finiteFloat(particle.plasticVolume) ||
				particle.plasticVolume < SNOW_MIN_PLASTIC_VOLUME ||
				particle.plasticVolume > SNOW_MAX_PLASTIC_VOLUME
			) {
				atomicOr(&stateError[0], ERROR_P2G);
				return;
			}
			let decomposition = decompose2x2(particle.f);
			if (
				!finiteVec2(decomposition.sigma) ||
				any(decomposition.sigma <= vec2<f32>(0.0))
			) {
				atomicOr(&stateError[0], ERROR_P2G);
				return;
			}
			let rotation = polarRotation(decomposition);
			let hardeningExponent = clamp(
				SNOW_HARDENING * (1.0 - particle.plasticVolume),
				-SNOW_MAX_HARDENING_EXPONENT,
				SNOW_MAX_HARDENING_EXPONENT,
			);
			let hardening = exp(hardeningExponent);
			let mu = particle.mu * hardening;
			let lambda = particle.lambda * hardening;
			let difference = particle.f - rotation;
			let shear = 2.0 * mu;
			let volumetric = lambda * determinant * (determinant - 1.0);
			stressTimesFt = vec4<f32>(
				shear * (difference.x * particle.f.x + difference.y * particle.f.y) +
					volumetric,
				shear * (difference.x * particle.f.z + difference.y * particle.f.w),
				shear * (difference.z * particle.f.x + difference.w * particle.f.y),
				shear * (difference.z * particle.f.z + difference.w * particle.f.w) +
					volumetric,
			);
		} else if (
			particle.materialModel == MODEL_WATER ||
			particle.materialModel == MODEL_OIL
		) {
			if (
				!finiteFloat(particle.density) ||
				particle.density <= 0.0 ||
				!finiteFloat(particle.lambda) ||
				particle.lambda <= 0.0
			) {
				atomicOr(&stateError[0], ERROR_P2G);
				return;
			}
			let pressure = particle.lambda * min(determinant - 1.0, 0.0);
			stressTimesFt = vec4<f32>(pressure, 0.0, 0.0, pressure);
		} else if (particle.materialModel == MODEL_POWDER) {
			if (
				!finiteFloat(particle.lambda) ||
				!finiteFloat(particle.mu) ||
				particle.mu <= 0.0 ||
				!finiteFloat(particle.frictionCoefficient) ||
				particle.frictionCoefficient < 0.0
			) {
				atomicOr(&stateError[0], ERROR_P2G);
				return;
			}
			stressTimesFt = granularStressTimesFt(
				particle.f,
				particle.lambda,
				particle.mu,
				particle.frictionCoefficient,
			);
		} else {
			atomicOr(&stateError[0], ERROR_P2G);
			return;
		}
		var particleMass = params.physical.y;
		if (
			particle.materialModel == MODEL_WATER ||
			particle.materialModel == MODEL_OIL
		) {
			particleMass = particle.density * particle.volume;
		}
		let particleStressScale =
			-params.physical.w * particle.volume * params.material.w;
		let affine =
			particleMass * particle.c +
			particleStressScale * stressTimesFt;
		let dpos =
			(vec2<f32>(f32(ox), f32(oy)) - fraction) * params.physical.x;
		let particleMomentum = vec2<f32>(
			particleMass * particle.velocity.x +
				affine.x * dpos.x +
				affine.y * dpos.y,
			particleMass * particle.velocity.y +
				affine.z * dpos.x +
				affine.w * dpos.y,
		);
		if (
			!finiteFloat(weight) ||
			weight < 0.0 ||
			!finiteVec4(stressTimesFt) ||
			!finiteVec4(affine) ||
			!finiteVec2(dpos) ||
			!finiteVec2(particleMomentum)
		) {
			atomicOr(&stateError[0], ERROR_P2G);
			return;
		}

		let weightedMass = weight * particleMass;
		let weightedMomentum = weight * particleMomentum;
		if (
			(weight > 0.0 && weightedMass <= 0.0) ||
			(weight > 0.0 &&
				((particleMomentum.x != 0.0 && weightedMomentum.x == 0.0) ||
					(particleMomentum.y != 0.0 && weightedMomentum.y == 0.0)))
		) {
			atomicOr(&stateError[0], ERROR_P2G);
			return;
		}
		mass = mass + weightedMass;
		momentum = momentum + weightedMomentum;
		if (!finiteFloat(mass) || !finiteVec2(momentum)) {
			atomicOr(&stateError[0], ERROR_P2G);
			return;
		}
		entry = bucketNext[pairIndex];
		visited = visited + 1u;
	}

	if (!finiteFloat(mass) || !finiteVec2(momentum)) {
		atomicOr(&stateError[0], ERROR_P2G);
		return;
	}
	gridNodes[nodeIndex].mass = mass;
	gridNodes[nodeIndex].momentum = momentum;
	gridNodes[nodeIndex].velocity = vec2<f32>(0.0, 0.0);
}

@compute @workgroup_size(64)
fn updateGrid(@builtin(global_invocation_id) invocation: vec3<u32>) {
	let index = invocation.x;
	if (index >= params.counts.y) {
		return;
	}
	if (atomicLoad(&stateError[0]) != 0u) {
		return;
	}

	let mass = gridNodes[index].mass;
	if (!finiteFloat(mass) || mass < 0.0) {
		atomicOr(&stateError[0], ERROR_GRID);
		return;
	}
	if (mass <= 0.0) {
		gridNodes[index].velocity = vec2<f32>(0.0, 0.0);
		return;
	}

	var velocity =
		gridNodes[index].momentum / mass +
		params.physical.w * params.gravityAndDomain.xy;
	if (!finiteVec2(velocity)) {
		atomicOr(&stateError[0], ERROR_GRID);
		return;
	}

	let nodeStorage = vec2<u32>(index % params.grid.z, index / params.grid.z);
	let nodeCoordinate = vec2<i32>(i32(nodeStorage.x) - 1, i32(nodeStorage.y) - 1);
	let dragRadius = interactionParams.targetAndRadius.z;
	if (
		dragRadius > 0.0 &&
		nodeCoordinate.x >= 0 &&
		nodeCoordinate.y >= 0 &&
		nodeCoordinate.x <= i32(params.grid.x) &&
		nodeCoordinate.y <= i32(params.grid.y)
	) {
		let nodePosition = vec2<f32>(nodeCoordinate) * params.physical.x;
		let displacement = interactionParams.targetAndRadius.xy - nodePosition;
		let distance = length(displacement);
		if (distance < dragRadius) {
			let falloff = 1.0 - distance / dragRadius;
			let acceleration =
				displacement * interactionParams.targetAndRadius.w -
				velocity * interactionParams.force.x;
			velocity = velocity + params.physical.w * falloff * acceleration;
			let speed = length(velocity);
			if (speed > 1200.0) {
				velocity = velocity * (1200.0 / speed);
			}
		}
	}
	if (!finiteVec2(velocity)) {
		atomicOr(&stateError[0], ERROR_GRID);
		return;
	}
	if (
		(nodeCoordinate.x <= 0 && velocity.x < 0.0) ||
		(nodeCoordinate.x >= i32(params.grid.x) && velocity.x > 0.0)
	) {
		velocity.x = 0.0;
	}
	if (
		(nodeCoordinate.y <= 0 && velocity.y < 0.0) ||
		(nodeCoordinate.y >= i32(params.grid.y) && velocity.y > 0.0)
	) {
		velocity.y = 0.0;
	}
	gridNodes[index].velocity = velocity;
}

@compute @workgroup_size(64)
fn g2p(@builtin(global_invocation_id) invocation: vec3<u32>) {
	let particleIndex = invocation.x;
	if (particleIndex >= params.counts.x) {
		return;
	}
	let particle = particlesIn[particleIndex];
	if (atomicLoad(&stateError[0]) != 0u) {
		particlesOut[particleIndex] = particle;
		return;
	}

	let gridPosition = particle.position / params.physical.x;
	if (!finiteVec2(gridPosition)) {
		atomicOr(&stateError[0], ERROR_STENCIL);
		return;
	}
	let base = vec2<i32>(floor(gridPosition - vec2<f32>(0.5, 0.5)));
	if (!stencilFitsHalo(base)) {
		atomicOr(&stateError[0], ERROR_STENCIL);
		return;
	}
	let fraction = gridPosition - vec2<f32>(base);
	let weightsX = quadraticWeights(fraction.x);
	let weightsY = quadraticWeights(fraction.y);
	var velocity = vec2<f32>(0.0, 0.0);
	var affine = vec4<f32>(0.0, 0.0, 0.0, 0.0);

	for (var oy = 0u; oy < 3u; oy = oy + 1u) {
		for (var ox = 0u; ox < 3u; ox = ox + 1u) {
			let node = base + vec2<i32>(i32(ox), i32(oy));
			let nodeIndex = gridIndex(node);
			if (nodeIndex >= params.counts.y) {
				atomicOr(&stateError[0], ERROR_STENCIL);
				return;
			}
			let weight = weightsX[ox] * weightsY[oy];
			let dpos =
				(vec2<f32>(f32(ox), f32(oy)) - fraction) * params.physical.x;
			let gridVelocity = gridNodes[nodeIndex].velocity;
			if (
				!finiteFloat(weight) ||
				weight < 0.0 ||
				!finiteVec2(dpos) ||
				!finiteVec2(gridVelocity)
			) {
				atomicOr(&stateError[0], ERROR_G2P);
				return;
			}
			velocity = velocity + weight * gridVelocity;
			affine = affine + vec4<f32>(
				weight * gridVelocity.x * dpos.x,
				weight * gridVelocity.x * dpos.y,
				weight * gridVelocity.y * dpos.x,
				weight * gridVelocity.y * dpos.y,
			);
		}
	}

	let updatedC = affine * params.material.w;
	let dt = params.physical.w;
	let deformationUpdate = vec4<f32>(
		1.0 + dt * updatedC.x,
		dt * updatedC.y,
		dt * updatedC.z,
		1.0 + dt * updatedC.w,
	);
	let trialF = vec4<f32>(
		deformationUpdate.x * particle.f.x +
			deformationUpdate.y * particle.f.z,
		deformationUpdate.x * particle.f.y +
			deformationUpdate.y * particle.f.w,
		deformationUpdate.z * particle.f.x +
			deformationUpdate.w * particle.f.z,
		deformationUpdate.z * particle.f.y +
			deformationUpdate.w * particle.f.w,
	);
	let updatedPosition = particle.position + dt * velocity;
	var updatedF = trialF;
	var updatedPlasticVolume = particle.plasticVolume;
	let trialDeterminant = trialF.x * trialF.w - trialF.y * trialF.z;
	if (
		!finiteVec2(velocity) ||
		!finiteVec4(updatedC) ||
		!finiteVec4(trialF) ||
		!finiteVec2(updatedPosition)
	) {
		atomicOr(&stateError[0], ERROR_G2P);
		return;
	}
	if (!finiteFloat(trialDeterminant) || trialDeterminant <= 0.0) {
		atomicOr(&stateError[0], ERROR_DEFORMATION);
		return;
	}
	if (particle.materialModel == MODEL_SNOW) {
		let decomposition = decompose2x2(trialF);
		if (
			!finiteVec2(decomposition.sigma) ||
			any(decomposition.sigma <= vec2<f32>(0.0))
		) {
			atomicOr(&stateError[0], ERROR_G2P);
			return;
		}
		let minimumStretch = 1.0 - SNOW_COMPRESSION_YIELD;
		let maximumStretch = 1.0 + SNOW_STRETCH_YIELD;
		let projectedSigma = clamp(
			decomposition.sigma,
			vec2<f32>(minimumStretch),
			vec2<f32>(maximumStretch),
		);
		let projectedDeterminant = projectedSigma.x * projectedSigma.y;
		updatedPlasticVolume = clamp(
			particle.plasticVolume * trialDeterminant / projectedDeterminant,
			SNOW_MIN_PLASTIC_VOLUME,
			SNOW_MAX_PLASTIC_VOLUME,
		);
		updatedF = reconstructDeformation(decomposition, projectedSigma);
	} else if (particle.materialModel == MODEL_POWDER) {
		if (
			!finiteFloat(particle.lambda) ||
			!finiteFloat(particle.mu) ||
			particle.mu <= 0.0 ||
			!finiteFloat(particle.frictionCoefficient) ||
			particle.frictionCoefficient < 0.0
		) {
			atomicOr(&stateError[0], ERROR_G2P);
			return;
		}
		updatedF = projectGranularDeformation(
			trialF,
			particle.lambda,
			particle.mu,
			particle.frictionCoefficient,
		);
	} else if (
		particle.materialModel != MODEL_NEO_HOOKEAN &&
		particle.materialModel != MODEL_WATER &&
		particle.materialModel != MODEL_OIL
	) {
		atomicOr(&stateError[0], ERROR_G2P);
		return;
	}
	let updatedDeterminant = updatedF.x * updatedF.w - updatedF.y * updatedF.z;
	if (
		!finiteVec4(updatedF) ||
		!finiteFloat(updatedPlasticVolume) ||
		updatedPlasticVolume < SNOW_MIN_PLASTIC_VOLUME ||
		updatedPlasticVolume > SNOW_MAX_PLASTIC_VOLUME
	) {
		atomicOr(&stateError[0], ERROR_G2P);
		return;
	}
	if (!finiteFloat(updatedDeterminant) || updatedDeterminant <= 0.0) {
		atomicOr(&stateError[0], ERROR_DEFORMATION);
		return;
	}

	let boundedX = confineAxis(
		updatedPosition.x,
		velocity.x,
		params.gravityAndDomain.z,
	);
	let boundedY = confineAxis(
		updatedPosition.y,
		velocity.y,
		params.gravityAndDomain.w,
	);
	var updatedParticle = particle;
	updatedParticle.position = vec2<f32>(boundedX.x, boundedY.x);
	updatedParticle.velocity = vec2<f32>(boundedX.y, boundedY.y);
	updatedParticle.c = updatedC;
	updatedParticle.f = updatedF;
	updatedParticle.plasticVolume = updatedPlasticVolume;
	particlesOut[particleIndex] = updatedParticle;
}

@compute @workgroup_size(64)
fn commitParticles(@builtin(global_invocation_id) invocation: vec3<u32>) {
	let particleIndex = invocation.x;
	if (particleIndex >= params.counts.x) {
		return;
	}
	if (atomicLoad(&stateError[0]) != 0u) {
		particlesOut[particleIndex] = particlesIn[particleIndex];
	}
}

@compute @workgroup_size(1)
fn pickParticle() {
	let point = pickParams.pointAndRadius.xy;
	let radiusSquared =
		pickParams.pointAndRadius.z * pickParams.pointAndRadius.z;
	var closestIndex = 0xffffffffu;
	var closestDistance = radiusSquared;

	for (var index = 0u; index < pickParams.countAndPadding.x; index = index + 1u) {
		let position = particlesIn[index].position;
		let offset = position - point;
		let distanceSquared = dot(offset, offset);
		if (
			distanceSquared <= radiusSquared &&
			(closestIndex == 0xffffffffu || distanceSquared < closestDistance)
		) {
			closestDistance = distanceSquared;
			closestIndex = index;
		}
	}

	pickResult[0] = closestIndex;
	if (closestIndex == 0xffffffffu) {
		pickResult[1] = 0u;
		pickResult[2] = 0u;
		pickResult[3] = 0u;
		return;
	}

	let position = particlesIn[closestIndex].position;
	pickResult[1] = bitcast<u32>(position.x);
	pickResult[2] = bitcast<u32>(position.y);
	pickResult[3] = particleObjectIds[closestIndex];
}

@compute @workgroup_size(64)
fn cloneObject(@builtin(global_invocation_id) invocation: vec3<u32>) {
	let localIndex = invocation.x;
	if (localIndex >= cloneParams.rangeAndObjectId.y) {
		return;
	}

	let sourceIndex = cloneParams.rangeAndObjectId.x + localIndex;
	let destinationIndex = cloneParams.rangeAndObjectId.z + localIndex;
	var particle = cloneParticles[sourceIndex];
	particle.position = clamp(
		particle.position + cloneParams.offsetAndDomain.xy,
		vec2<f32>(0.0, 0.0),
		cloneParams.offsetAndDomain.zw,
	);
	cloneParticles[destinationIndex] = particle;
	cloneObjectIds[destinationIndex] = cloneParams.rangeAndObjectId.w;
}
`;

export const MLS_MPM_RENDER_WGSL = /* wgsl */ `
struct Particle {
	position: vec2<f32>,
	velocity: vec2<f32>,
	c: vec4<f32>,
	f: vec4<f32>,
	plasticVolume: f32,
	materialModel: u32,
	lambda: f32,
	mu: f32,
	density: f32,
	volume: f32,
	frictionCoefficient: f32,
}

const MODEL_NEO_HOOKEAN: u32 = 0u;
const MODEL_SNOW: u32 = 1u;
const MODEL_WATER: u32 = 2u;
const MODEL_OIL: u32 = 3u;
const MODEL_POWDER: u32 = 4u;

struct RenderParams {
	domainAndRadius: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: RenderParams;

struct VertexOutput {
	@builtin(position) position: vec4<f32>,
	@location(0) localPosition: vec2<f32>,
	@location(1) materialColor: vec3<f32>,
}

@vertex
fn vertexMain(
	@builtin(vertex_index) vertexIndex: u32,
	@builtin(instance_index) instanceIndex: u32,
) -> VertexOutput {
	let corners = array<vec2<f32>, 6>(
		vec2<f32>(-1.0, -1.0),
		vec2<f32>(1.0, -1.0),
		vec2<f32>(-1.0, 1.0),
		vec2<f32>(-1.0, 1.0),
		vec2<f32>(1.0, -1.0),
		vec2<f32>(1.0, 1.0),
	);
	let local = corners[vertexIndex];
	let particle = particles[instanceIndex];
	let worldPosition =
		particle.position + local * params.domainAndRadius.z;
	let clipPosition = vec2<f32>(
		2.0 * worldPosition.x / params.domainAndRadius.x - 1.0,
		1.0 - 2.0 * worldPosition.y / params.domainAndRadius.y,
	);
	var output: VertexOutput;
	output.position = vec4<f32>(clipPosition, 0.0, 1.0);
	output.localPosition = local;
	var materialColor = vec3<f32>(0.28, 0.78, 0.72);
	if (particle.materialModel == MODEL_SNOW) {
		materialColor = vec3<f32>(0.98, 0.56, 0.28);
	} else if (particle.materialModel == MODEL_WATER) {
		materialColor = vec3<f32>(0.18, 0.55, 0.95);
	} else if (particle.materialModel == MODEL_OIL) {
		materialColor = vec3<f32>(0.95, 0.75, 0.22);
	} else if (particle.materialModel == MODEL_POWDER) {
		materialColor = vec3<f32>(0.72, 0.48, 0.3);
	}
	output.materialColor = materialColor;
	return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
	if (dot(input.localPosition, input.localPosition) > 1.0) {
		discard;
	}
	return vec4<f32>(input.materialColor, 1.0);
}
`;
