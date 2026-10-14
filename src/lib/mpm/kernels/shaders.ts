import {
	MPM_LIQUID_CONTACT_APPROACH_SPEED_MULTIPLIER,
	MPM_LIQUID_CONTACT_RELAXATION_SECONDS,
	MPM_LIQUID_CONTACT_SEPARATION_FRACTION,
	MPM_LIQUID_STABLE_ORDER_EPSILON
} from '../materials.js';

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
	bondedTriangleVertex: u32,
}

struct GridNode {
	mass: f32,
	momentum: vec2<f32>,
	velocity: vec2<f32>,
	waterMass: f32,
	waterMomentum: vec2<f32>,
	waterVelocity: vec2<f32>,
	waterVolume: f32,
	oilMass: f32,
	oilMomentum: vec2<f32>,
	oilVelocity: vec2<f32>,
	oilVolume: f32,
}

struct TriangleFieldNode {
	signedDistance: f32,
	bodyStartPlusOne: u32,
	normal: vec2<f32>,
	barycentric: vec2<f32>,
	impulse: vec2<f32>,
}

struct TriangleFieldBody {
	first: vec2<f32>,
	second: vec2<f32>,
	third: vec2<f32>,
	minimum: vec2<f32>,
	maximum: vec2<f32>,
	bodyStart: u32,
	valid: u32,
}

struct TriangleContactState {
	velocity: vec2<f32>,
	contactCount: atomic<u32>,
	padding: u32,
}

struct TriangleDistanceSample {
	signedDistance: f32,
	normal: vec2<f32>,
	barycentric: vec2<f32>,
	valid: u32,
}

struct SolidBond {
	endpoints: vec2<u32>,
	restLength: f32,
	failureStrain: f32,
	damage: f32,
	broken: u32,
}

struct SimulationParams {
	grid: vec4<u32>,
	counts: vec4<u32>,
	physical: vec4<f32>,
	material: vec4<f32>,
	gravityAndDomain: vec4<f32>,
	bonding: vec4<f32>,
	cflBounds: vec4<f32>,
	fracture: vec4<u32>,
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
	bondRangeAndCapacity: vec4<u32>,
}

struct OutletParams {
	bounds: vec4<f32>,
}

const WALL_CUSHION_CELLS: f32 = 8.0;

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
@group(0) @binding(14) var<storage, read_write> preparedAffines: array<vec4<f32>>;
@group(0) @binding(15) var<storage, read_write> surfaceFields: array<vec2<f32>>;
@group(0) @binding(16) var<storage, read_write> triangleFieldNodes: array<TriangleFieldNode>;
@group(0) @binding(17) var<storage, read_write> bondedBodyStarts: array<u32>;
@group(0) @binding(18) var<storage, read_write> cflSpeedPartials: array<f32>;
@group(0) @binding(19) var<storage, read_write> cflMaximumSpeed: array<f32>;
@group(0) @binding(20) var<storage, read_write> fractureBonds: array<SolidBond>;
@group(0) @binding(21) var<storage, read> fractureBondTopology: array<u32>;
@group(0) @binding(22) var<uniform> outletParams: OutletParams;
@group(0) @binding(23) var<storage, read_write> outletRemovalCounter: array<atomic<u32>>;
@group(0) @binding(24) var<storage, read_write> cflStepProgress: array<atomic<u32>>;
@group(0) @binding(25) var<storage, read_write> triangleContactStates: array<TriangleContactState>;
@group(0) @binding(26) var<storage, read_write> trianglePairContactCache: array<vec4<f32>>;
@group(0) @binding(27) var<storage, read_write> trianglePairContactCandidates: array<atomic<u32>>;
@group(0) @binding(28) var<storage, read_write> triangleFieldCandidates: array<u32>;

var<workgroup> cflSpeedReduction: array<f32, 64>;
var<workgroup> triangleFieldBodyCache: array<TriangleFieldBody, 64>;

const ERROR_STENCIL: u32 = 1u;
const ERROR_DEFORMATION: u32 = 2u;
const ERROR_P2G: u32 = 4u;
const ERROR_GRID: u32 = 8u;
const ERROR_G2P: u32 = 16u;
const ERROR_DEFORMATION_SUBSTEP_LIMIT: u32 = 32u;
const ERROR_PROJECTED_DEFORMATION: u32 = 64u;
const ERROR_P2G_INVALID_MATERIAL: u32 = 128u;
const ERROR_P2G_NONFINITE_PARTICLE: u32 = 256u;
const ERROR_P2G_NONFINITE_ACCUMULATION: u32 = 1024u;
const ERROR_P2G_NONFINITE_NODE: u32 = 2048u;
const ERROR_TRIANGLE_CONTACT: u32 = 4096u;
const ERROR_TRIANGLE_BOND: u32 = 8192u;
const ERROR_CFL_BOUND: u32 = 16384u;
const ERROR_FRACTURE_BOND: u32 = 32768u;
const BONDED_TRIANGLE_SOLVER_ITERATIONS: u32 = 2u;
const TRIANGLE_CONTACT_SKIN_CELLS: f32 = 0.1;
const TRIANGLE_CONTACT_IMPULSE_RELAXATION: f32 = 0.25;
const LIQUID_CONTACT_SEPARATION_FRACTION: f32 = ${MPM_LIQUID_CONTACT_SEPARATION_FRACTION};
const LIQUID_CONTACT_APPROACH_SPEED_MULTIPLIER: f32 = ${MPM_LIQUID_CONTACT_APPROACH_SPEED_MULTIPLIER};
const LIQUID_CONTACT_RELAXATION_SECONDS: f32 = ${MPM_LIQUID_CONTACT_RELAXATION_SECONDS};
const LIQUID_STABLE_ORDER_EPSILON: f32 = ${MPM_LIQUID_STABLE_ORDER_EPSILON};
const TRIANGLE_FIELD_CANDIDATE_BODY_LIMIT: u32 = 512u;
const TRIANGLE_FIELD_CANDIDATE_BODY_WORD_BITS: u32 = 32u;
const TRIANGLE_FIELD_CANDIDATE_TILE_SIZE: u32 = 16u;
const TRIANGLE_PAIR_CONTACT_CANDIDATE_WORD_BITS: u32 = 32u;
const SURFACE_ERROR_BUCKET: u32 = 1u;
const SURFACE_ERROR_PARTICLE_AREA: u32 = 2u;
const SURFACE_ERROR_ACCUMULATION: u32 = 4u;
const MODEL_NEO_HOOKEAN: u32 = 0u;
const MODEL_SNOW: u32 = 1u;
const MODEL_WATER: u32 = 2u;
const MODEL_OIL: u32 = 3u;
const MODEL_POWDER: u32 = 4u;
const MODEL_REMOVED: u32 = 0xffffffffu;
const SNOW_COMPRESSION_YIELD: f32 = 0.025;
const SNOW_STRETCH_YIELD: f32 = 0.0075;
const SNOW_HARDENING: f32 = 10.0;
const SNOW_MIN_PLASTIC_VOLUME: f32 = 0.6;
const SNOW_MAX_PLASTIC_VOLUME: f32 = 20.0;
const SNOW_MAX_HARDENING_EXPONENT: f32 = 20.0;
const MIN_LIQUID_DETERMINANT: f32 = 0.5;
const MAX_LIQUID_DETERMINANT: f32 = 2.0;
const MIN_POWDER_DETERMINANT: f32 = 0.5;
const MIN_NEO_HOOKEAN_STRETCH: f32 = 0.05;
const MAX_NEO_HOOKEAN_STRETCH: f32 = 20.0;
const MAX_NEO_HOOKEAN_STRETCH_SQUARED: f32 =
	MAX_NEO_HOOKEAN_STRETCH * MAX_NEO_HOOKEAN_STRETCH;
const MAX_DEFORMATION_INCREMENT: f32 = 0.25;
const MAX_DEFORMATION_SUBSTEPS: f32 = 64.0;

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

fn recordDeformationFailure(
	particleIndex: u32,
	failedDeterminant: f32,
	deformationIncrement: f32,
	precedingDeterminant: f32,
) {
	let previousClaim = atomicOr(&stateError[6], 1u);
	if (previousClaim == 0u) {
		atomicStore(&stateError[1], particleIndex);
		atomicStore(&stateError[2], bitcast<u32>(failedDeterminant));
		atomicStore(&stateError[3], bitcast<u32>(deformationIncrement));
		atomicStore(&stateError[4], bitcast<u32>(precedingDeterminant));
	}
}

fn recordProjectedDeformationFailure(particleIndex: u32, deformation: vec4<f32>) {
	let previousClaim = atomicOr(&stateError[6], 2u);
	if (previousClaim == 0u) {
		atomicStore(&stateError[1], particleIndex);
		atomicStore(&stateError[2], bitcast<u32>(deformation.x));
		atomicStore(&stateError[3], bitcast<u32>(deformation.y));
		atomicStore(&stateError[4], bitcast<u32>(deformation.z));
		atomicStore(&stateError[5], bitcast<u32>(deformation.w));
	}
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

fn determinant2x2(matrix: vec4<f32>) -> f32 {
	let secondProduct = matrix.y * matrix.z;
	// Keep the product residual when the determinant terms nearly cancel.
	let secondProductError = fma(matrix.y, matrix.z, -secondProduct);
	return fma(matrix.x, matrix.w, -secondProduct) - secondProductError;
}

fn particleWaveSpeedBound(particle: Particle) -> f32 {
	if (
		particle.materialModel == MODEL_WATER ||
		particle.materialModel == MODEL_OIL
	) {
		if (
			!finiteFloat(particle.lambda) ||
			!finiteFloat(particle.density) ||
			particle.lambda <= 0.0 ||
			particle.density <= 0.0
		) {
			return 3.402823466e38;
		}
		return sqrt(particle.lambda / particle.density);
	}
	if (particle.materialModel != MODEL_NEO_HOOKEAN) {
		return 3.402823466e38;
	}
	if (!finiteFloat(particle.frictionCoefficient)) {
		return 3.402823466e38;
	}
	let remainingStrength = 1.0 - clamp(particle.frictionCoefficient, 0.0, 1.0);
	if (remainingStrength <= 0.0) {
		return 0.0;
	}
	let lambda = particle.lambda * remainingStrength;
	let mu = particle.mu * remainingStrength;

	let determinant = determinant2x2(particle.f);
	let deformationTrace =
		particle.f.x * particle.f.x +
		particle.f.y * particle.f.y +
		particle.f.z * particle.f.z +
		particle.f.w * particle.f.w;
	if (
		!finiteFloat(determinant) ||
		determinant <= 0.0 ||
		!finiteFloat(deformationTrace) ||
		!finiteFloat(lambda) ||
		!finiteFloat(mu) ||
		mu < 0.0
	) {
		return 3.402823466e38;
	}
	let discriminant = sqrt(max(
		0.0,
		deformationTrace * deformationTrace -
			4.0 * determinant * determinant,
	));
	let maximumStretchSquared = 0.5 * (deformationTrace + discriminant);
	let volumetricTangent = abs(
		lambda + mu - lambda * log(determinant),
	);
	let referenceDensity = params.physical.y / particle.volume;
	let waveSpeedSquared =
		(mu * maximumStretchSquared + volumetricTangent) /
		referenceDensity;
	if (
		!finiteFloat(referenceDensity) ||
		referenceDensity <= 0.0 ||
		!finiteFloat(waveSpeedSquared) ||
		waveSpeedSquared < 0.0
	) {
		return 3.402823466e38;
	}
	return sqrt(waveSpeedSquared);
}

fn decompose2x2(matrix: vec4<f32>) -> Decomposition2 {
	let c00 = matrix.x * matrix.x + matrix.z * matrix.z;
	let c01 = matrix.x * matrix.y + matrix.z * matrix.w;
	let c11 = matrix.y * matrix.y + matrix.w * matrix.w;
	let radius = sqrt((c00 - c11) * (c00 - c11) + 4.0 * c01 * c01);
	let sigma0 = sqrt(max(0.0, 0.5 * (c00 + c11 + radius)));
	let determinant = determinant2x2(matrix);
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
		-u0.y,
		u0.x,
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
	let boundedLogVolume = max(log(MIN_POWDER_DETERMINANT), logVolume);
	let pressure = max(0.0, -(lambda + mu) * boundedLogVolume);
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
	let minimumLogVolume = log(MIN_POWDER_DETERMINANT);
	let boundedLogVolume = max(minimumLogVolume, logVolume);
	let pressure = max(0.0, -(lambda + mu) * boundedLogVolume);
	let deviatoricLogStretch = 0.5 * (logStretch.x - logStretch.y);
	let maximumDeviatoricLogStretch =
		frictionCoefficient * pressure / (2.0 * mu);
	if (
		deviatoricLogStretch <= maximumDeviatoricLogStretch &&
		logVolume >= minimumLogVolume
	) {
		return matrix;
	}
	let projectedDeviatoricLogStretch =
		min(deviatoricLogStretch, maximumDeviatoricLogStretch);
	let meanLogStretch = 0.5 * boundedLogVolume;
	let projectedSigma = exp(vec2<f32>(
		meanLogStretch + projectedDeviatoricLogStretch,
		meanLogStretch - projectedDeviatoricLogStretch,
	));
	return reconstructDeformation(decomposition, projectedSigma);
}

fn projectLiquidDeformation(matrix: vec4<f32>) -> vec4<f32> {
	let rotationNumerator = vec2<f32>(matrix.x + matrix.w, matrix.z - matrix.y);
	let rotationScale = max(abs(rotationNumerator.x), abs(rotationNumerator.y));
	let normalizedRotation = rotationNumerator / rotationScale;
	let rotationLength = length(normalizedRotation);
	let rotation = normalizedRotation / rotationLength;
	let volume = determinant2x2(matrix);
	let boundedVolume = clamp(
		volume,
		MIN_LIQUID_DETERMINANT,
		MAX_LIQUID_DETERMINANT,
	);
	let scale = sqrt(boundedVolume);
	return scale * vec4<f32>(rotation.x, -rotation.y, rotation.y, rotation.x);
}

fn projectNeoHookeanDeformation(matrix: vec4<f32>) -> vec4<f32> {
	let decomposition = decompose2x2(matrix);
	let projectedSigma = clamp(
		decomposition.sigma,
		vec2<f32>(MIN_NEO_HOOKEAN_STRETCH),
		vec2<f32>(MAX_NEO_HOOKEAN_STRETCH),
	);
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

fn cross2(first: vec2<f32>, second: vec2<f32>) -> f32 {
	return first.x * second.y - first.y * second.x;
}

fn closestPointOnSegment(
	point: vec2<f32>,
	start: vec2<f32>,
	end: vec2<f32>,
) -> vec3<f32> {
	let edge = end - start;
	let parameter = clamp(dot(point - start, edge) / dot(edge, edge), 0.0, 1.0);
	return vec3<f32>(start + parameter * edge, parameter);
}

fn triangleDistanceSample(
	point: vec2<f32>,
	first: vec2<f32>,
	second: vec2<f32>,
	third: vec2<f32>,
) -> TriangleDistanceSample {
	let signedArea = cross2(second - first, third - first);
	if (!finiteFloat(signedArea) || abs(signedArea) <= 1e-7) {
		return TriangleDistanceSample(0.0, vec2<f32>(0.0), vec2<f32>(0.0), 0u);
	}

	let closestFirstSecond = closestPointOnSegment(point, first, second);
	let closestSecondThird = closestPointOnSegment(point, second, third);
	let closestThirdFirst = closestPointOnSegment(point, third, first);
	var closestPoint = closestFirstSecond.xy;
	var closestEdge = second - first;
	var barycentric = vec2<f32>(closestFirstSecond.z, 0.0);
	var distanceSquared = dot(point - closestPoint, point - closestPoint);
	let secondThirdDistanceSquared =
		dot(point - closestSecondThird.xy, point - closestSecondThird.xy);
	if (secondThirdDistanceSquared < distanceSquared) {
		closestPoint = closestSecondThird.xy;
		closestEdge = third - second;
		barycentric = vec2<f32>(1.0 - closestSecondThird.z, closestSecondThird.z);
		distanceSquared = secondThirdDistanceSquared;
	}
	let thirdFirstDistanceSquared =
		dot(point - closestThirdFirst.xy, point - closestThirdFirst.xy);
	if (thirdFirstDistanceSquared < distanceSquared) {
		closestPoint = closestThirdFirst.xy;
		closestEdge = first - third;
		barycentric = vec2<f32>(0.0, 1.0 - closestThirdFirst.z);
		distanceSquared = thirdFirstDistanceSquared;
	}

	let sideFirst = cross2(second - first, point - first);
	let sideSecond = cross2(third - second, point - second);
	let sideThird = cross2(first - third, point - third);
	let inside =
		(signedArea > 0.0 &&
			sideFirst >= -1e-6 &&
			sideSecond >= -1e-6 &&
			sideThird >= -1e-6) ||
		(signedArea < 0.0 &&
			sideFirst <= 1e-6 &&
			sideSecond <= 1e-6 &&
			sideThird <= 1e-6);
	let distance = sqrt(max(0.0, distanceSquared));
	var normal = vec2<f32>(closestEdge.y, -closestEdge.x);
	if (signedArea < 0.0) {
		normal = -normal;
	}
	let normalLength = length(normal);
	if (normalLength <= 1e-8 || !finiteFloat(distance) || !finiteFloat(normalLength)) {
		return TriangleDistanceSample(0.0, vec2<f32>(0.0), vec2<f32>(0.0), 0u);
	}
	normal = normal / normalLength;
	if (!inside && distance > 1e-8) {
		normal = (point - closestPoint) / distance;
	}
	return TriangleDistanceSample(
		select(distance, -distance, inside),
		normal,
		barycentric,
		1u,
	);
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
	gridNodes[index].waterMass = 0.0;
	gridNodes[index].waterMomentum = vec2<f32>(0.0, 0.0);
	gridNodes[index].waterVelocity = vec2<f32>(0.0, 0.0);
	gridNodes[index].waterVolume = 0.0;
	gridNodes[index].oilMass = 0.0;
	gridNodes[index].oilMomentum = vec2<f32>(0.0, 0.0);
	gridNodes[index].oilVelocity = vec2<f32>(0.0, 0.0);
	gridNodes[index].oilVolume = 0.0;
	triangleFieldNodes[index].signedDistance = params.bonding.z;
	triangleFieldNodes[index].bodyStartPlusOne = 0u;
	triangleFieldNodes[index].normal = vec2<f32>(0.0, 0.0);
	triangleFieldNodes[index].barycentric = vec2<f32>(0.0, 0.0);
	triangleFieldNodes[index].impulse = vec2<f32>(0.0, 0.0);
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
	if (particle.materialModel == MODEL_REMOVED) {
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
fn clearSurfaceOccupancy(@builtin(global_invocation_id) invocation: vec3<u32>) {
	let index = invocation.x;
	if (index >= params.counts.y) {
		return;
	}
	atomicStore(&bucketHeads[index], -1);
	surfaceFields[index] = vec2<f32>(0.0, 0.0);
}

@compute @workgroup_size(64)
fn gatherSurfaceOccupancy(@builtin(global_invocation_id) invocation: vec3<u32>) {
	let nodeIndex = invocation.x;
	if (
		nodeIndex >= params.counts.y ||
		atomicLoad(&stateError[0]) != 0u ||
		atomicLoad(&stateError[7]) != 0u
	) {
		return;
	}

	var waterArea = 0.0;
	var oilArea = 0.0;
	var entry = atomicLoad(&bucketHeads[nodeIndex]);
	var visited = 0u;

	loop {
		if (entry == -1) {
			break;
		}
		if (entry < -1 || visited >= params.counts.x) {
			atomicOr(&stateError[7], SURFACE_ERROR_BUCKET);
			return;
		}
		let pairIndex = u32(entry);
		if (pairIndex >= params.counts.z) {
			atomicOr(&stateError[7], SURFACE_ERROR_BUCKET);
			return;
		}
		let particleIndex = pairIndex / 9u;
		if (particleIndex >= params.counts.x) {
			atomicOr(&stateError[7], SURFACE_ERROR_BUCKET);
			return;
		}

		let supportIndex = pairIndex % 9u;
		let ox = supportIndex % 3u;
		let oy = supportIndex / 3u;
		let particle = particlesIn[particleIndex];
		if (
			particle.materialModel == MODEL_WATER ||
			particle.materialModel == MODEL_OIL
		) {
			let gridPosition = particle.position / params.physical.x;
			if (!finiteVec2(gridPosition)) {
				atomicOr(&stateError[7], SURFACE_ERROR_PARTICLE_AREA);
				return;
			}
			let base = vec2<i32>(floor(gridPosition - vec2<f32>(0.5, 0.5)));
			let fraction = gridPosition - vec2<f32>(base);
			let weightsX = quadraticWeights(fraction.x);
			let weightsY = quadraticWeights(fraction.y);
			let weight = weightsX[ox] * weightsY[oy];
			let supportNode = base + vec2<i32>(i32(ox), i32(oy));
			if (!stencilFitsHalo(base) || gridIndex(supportNode) != nodeIndex) {
				atomicOr(&stateError[7], SURFACE_ERROR_BUCKET);
				return;
			}

			let determinant = determinant2x2(particle.f);
			let currentArea = particle.volume * determinant;
			let displayArea = currentArea * 2.0;
			let normalizedArea = displayArea * weight /
				(params.physical.x * params.physical.x);
			if (
				!finiteFloat(determinant) ||
				determinant <= 0.0 ||
				!finiteFloat(particle.volume) ||
				particle.volume <= 0.0 ||
				!finiteFloat(currentArea) ||
				currentArea <= 0.0 ||
				!finiteFloat(weight) ||
				weight < 0.0 ||
				!finiteFloat(normalizedArea) ||
				normalizedArea < 0.0
			) {
				atomicOr(&stateError[7], SURFACE_ERROR_PARTICLE_AREA);
				return;
			}

			if (particle.materialModel == MODEL_WATER) {
				waterArea = waterArea + normalizedArea;
			} else {
				oilArea = oilArea + normalizedArea;
			}
			if (!finiteFloat(waterArea) || !finiteFloat(oilArea)) {
				atomicOr(&stateError[7], SURFACE_ERROR_ACCUMULATION);
				return;
			}
		}

		entry = bucketNext[pairIndex];
		visited = visited + 1u;
	}

	surfaceFields[nodeIndex] = vec2<f32>(waterArea, oilArea);
}

@compute @workgroup_size(64)
fn prepareParticles(@builtin(global_invocation_id) invocation: vec3<u32>) {
	let particleIndex = invocation.x;
	if (particleIndex >= params.counts.x) {
		return;
	}
	if (atomicLoad(&stateError[0]) != 0u) {
		return;
	}
	let particle = particlesIn[particleIndex];
	if (particle.materialModel == MODEL_REMOVED) {
		preparedAffines[particleIndex] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
		return;
	}

		let determinant = determinant2x2(particle.f);
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
			if (
				!finiteFloat(particle.frictionCoefficient) ||
				particle.frictionCoefficient < 0.0 ||
				particle.frictionCoefficient > 1.0
			) {
				atomicMin(&stateError[1], particleIndex);
				atomicOr(&stateError[0], ERROR_P2G | ERROR_P2G_INVALID_MATERIAL);
				return;
			}
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
			stressTimesFt =
				stressTimesFt * (1.0 - clamp(particle.frictionCoefficient, 0.0, 1.0));
		} else if (particle.materialModel == MODEL_SNOW) {
			if (
				!finiteFloat(particle.plasticVolume) ||
				particle.plasticVolume < SNOW_MIN_PLASTIC_VOLUME ||
				particle.plasticVolume > SNOW_MAX_PLASTIC_VOLUME
			) {
				atomicMin(&stateError[1], particleIndex);
				atomicOr(&stateError[0], ERROR_P2G | ERROR_P2G_INVALID_MATERIAL);
				return;
			}
			let decomposition = decompose2x2(particle.f);
			if (
				!finiteVec2(decomposition.sigma) ||
				any(decomposition.sigma <= vec2<f32>(0.0))
			) {
				atomicMin(&stateError[1], particleIndex);
				atomicOr(&stateError[0], ERROR_P2G | ERROR_P2G_INVALID_MATERIAL);
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
				atomicMin(&stateError[1], particleIndex);
				atomicOr(&stateError[0], ERROR_P2G | ERROR_P2G_INVALID_MATERIAL);
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
				atomicMin(&stateError[1], particleIndex);
				atomicOr(&stateError[0], ERROR_P2G | ERROR_P2G_INVALID_MATERIAL);
				return;
			}
			stressTimesFt = granularStressTimesFt(
				particle.f,
				particle.lambda,
				particle.mu,
				particle.frictionCoefficient,
			);
		} else {
			atomicMin(&stateError[1], particleIndex);
			atomicOr(&stateError[0], ERROR_P2G | ERROR_P2G_INVALID_MATERIAL);
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
	if (!finiteVec4(stressTimesFt) || !finiteVec4(affine)) {
		atomicMin(&stateError[1], particleIndex);
		atomicOr(&stateError[0], ERROR_P2G | ERROR_P2G_NONFINITE_PARTICLE);
		return;
	}
	preparedAffines[particleIndex] = affine;
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

	var solidMass = 0.0;
	var solidMomentum = vec2<f32>(0.0, 0.0);
	var waterMass = 0.0;
	var waterMomentum = vec2<f32>(0.0, 0.0);
	var waterVolume = 0.0;
	var oilMass = 0.0;
	var oilMomentum = vec2<f32>(0.0, 0.0);
	var oilVolume = 0.0;
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
		var particleMass = params.physical.y;
		if (
			particle.materialModel == MODEL_WATER ||
			particle.materialModel == MODEL_OIL
		) {
			particleMass = particle.density * particle.volume;
		}
		let affine = preparedAffines[particleIndex];
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
			!finiteVec4(affine) ||
			!finiteVec2(dpos) ||
			!finiteVec2(particleMomentum)
		) {
			atomicMin(&stateError[1], particleIndex);
			atomicOr(&stateError[0], ERROR_P2G | ERROR_P2G_NONFINITE_PARTICLE);
			return;
		}

		let weightedMass = weight * particleMass;
		let weightedMomentum = weight * particleMomentum;
		if (particle.materialModel == MODEL_WATER) {
			let determinant = determinant2x2(particle.f);
			waterMass = waterMass + weightedMass;
			waterMomentum = waterMomentum + weightedMomentum;
			waterVolume = waterVolume + weight * particle.volume * determinant;
		} else if (particle.materialModel == MODEL_OIL) {
			let determinant = determinant2x2(particle.f);
			oilMass = oilMass + weightedMass;
			oilMomentum = oilMomentum + weightedMomentum;
			oilVolume = oilVolume + weight * particle.volume * determinant;
		} else {
			solidMass = solidMass + weightedMass;
			solidMomentum = solidMomentum + weightedMomentum;
		}
		if (
			!finiteFloat(solidMass) ||
			!finiteVec2(solidMomentum) ||
			!finiteFloat(waterMass) ||
			!finiteVec2(waterMomentum) ||
			!finiteFloat(waterVolume) ||
			!finiteFloat(oilMass) ||
			!finiteVec2(oilMomentum) ||
			!finiteFloat(oilVolume)
		) {
			atomicMin(&stateError[1], particleIndex);
			atomicOr(&stateError[0], ERROR_P2G | ERROR_P2G_NONFINITE_ACCUMULATION);
			return;
		}
		entry = bucketNext[pairIndex];
		visited = visited + 1u;
	}

	if (!finiteFloat(solidMass) || !finiteVec2(solidMomentum)) {
		atomicOr(&stateError[0], ERROR_P2G | ERROR_P2G_NONFINITE_NODE);
		return;
	}
	gridNodes[nodeIndex].mass = solidMass;
	gridNodes[nodeIndex].momentum = solidMomentum;
	gridNodes[nodeIndex].velocity = vec2<f32>(0.0, 0.0);
	gridNodes[nodeIndex].waterMass = waterMass;
	gridNodes[nodeIndex].waterMomentum = waterMomentum;
	gridNodes[nodeIndex].waterVolume = waterVolume;
	gridNodes[nodeIndex].oilMass = oilMass;
	gridNodes[nodeIndex].oilMomentum = oilMomentum;
	gridNodes[nodeIndex].oilVolume = oilVolume;
}

@compute @workgroup_size(64)
fn buildTriangleFieldCandidates(@builtin(global_invocation_id) invocation: vec3<u32>) {
	let outputIndex = invocation.x;
	let tileColumns =
		(params.grid.z + TRIANGLE_FIELD_CANDIDATE_TILE_SIZE - 1u) /
		TRIANGLE_FIELD_CANDIDATE_TILE_SIZE;
	let tileRows =
		(params.grid.w + TRIANGLE_FIELD_CANDIDATE_TILE_SIZE - 1u) /
		TRIANGLE_FIELD_CANDIDATE_TILE_SIZE;
	let candidateBodyCount = min(
		params.counts.w,
		TRIANGLE_FIELD_CANDIDATE_BODY_LIMIT,
	);
	let candidateWordsPerTile =
		(candidateBodyCount + TRIANGLE_FIELD_CANDIDATE_BODY_WORD_BITS - 1u) /
		TRIANGLE_FIELD_CANDIDATE_BODY_WORD_BITS;
	let candidateWordCount =
		tileColumns * tileRows * candidateWordsPerTile;
	if (outputIndex >= candidateWordCount) {
		return;
	}

	let tileIndex = outputIndex / candidateWordsPerTile;
	let bodyWord = outputIndex % candidateWordsPerTile;
	let tileColumn = tileIndex % tileColumns;
	let tileRow = tileIndex / tileColumns;
	let tileStorageMinimum = vec2<u32>(
		tileColumn * TRIANGLE_FIELD_CANDIDATE_TILE_SIZE,
		tileRow * TRIANGLE_FIELD_CANDIDATE_TILE_SIZE,
	);
	let tileStorageMaximum =
		min(
			tileStorageMinimum + vec2<u32>(TRIANGLE_FIELD_CANDIDATE_TILE_SIZE),
			params.grid.zw,
		) - vec2<u32>(1u);
	let tileNodeMinimum = vec2<i32>(tileStorageMinimum) - vec2<i32>(1);
	let tileNodeMaximum = vec2<i32>(tileStorageMaximum) - vec2<i32>(1);
	let tileMinimum =
		vec2<f32>(tileNodeMinimum) * params.physical.x;
	let tileMaximum =
		vec2<f32>(tileNodeMaximum) * params.physical.x;
	var candidateWord = 0u;

	if (params.counts.w <= TRIANGLE_FIELD_CANDIDATE_BODY_LIMIT) {
		for (
			var bodyLane = 0u;
			bodyLane < TRIANGLE_FIELD_CANDIDATE_BODY_WORD_BITS;
			bodyLane += 1u
		) {
			let bodyIndex =
				bodyWord * TRIANGLE_FIELD_CANDIDATE_BODY_WORD_BITS + bodyLane;
			if (bodyIndex >= candidateBodyCount) {
				break;
			}

			let bodyStart = bondedBodyStarts[bodyIndex];
			if (bodyStart + 2u >= params.counts.x) {
				candidateWord |= 1u << bodyLane;
				continue;
			}

			let first = particlesIn[bodyStart].position;
			let second = particlesIn[bodyStart + 1u].position;
			let third = particlesIn[bodyStart + 2u].position;
			if (!finiteVec2(first) || !finiteVec2(second) || !finiteVec2(third)) {
				candidateWord |= 1u << bodyLane;
				continue;
			}

			let minimum = min(first, min(second, third));
			let maximum = max(first, max(second, third));
			// The tile interval encloses every node point, so its AABB distance is conservative.
			let outside = max(
				max(minimum - tileMaximum, vec2<f32>(0.0)),
				tileMinimum - maximum,
			);
			if (dot(outside, outside) <= params.bonding.z * params.bonding.z) {
				candidateWord |= 1u << bodyLane;
			}
		}
	}

	// One invocation owns each tile/body-word entry, so every consumed word is fully overwritten.
	triangleFieldCandidates[outputIndex] = candidateWord;
}

@compute @workgroup_size(64)
fn buildTriangleField(
	@builtin(global_invocation_id) invocation: vec3<u32>,
	@builtin(local_invocation_index) localIndex: u32,
) {
	let index = invocation.x;
	let fieldActive =
		index < params.counts.y && atomicLoad(&stateError[0]) == 0u;
	let nodeStorage = vec2<u32>(index % params.grid.z, index / params.grid.z);
	let nodeCoordinate = vec2<i32>(i32(nodeStorage.x) - 1, i32(nodeStorage.y) - 1);
	let point = vec2<f32>(nodeCoordinate) * params.physical.x;
	var bestDistance = params.bonding.z;
	var bestBodyStartPlusOne = 0u;
	var bestNormal = vec2<f32>(0.0, 0.0);
	var bestBarycentric = vec2<f32>(0.0, 0.0);
	var failed = false;
	let candidateTileColumns =
		(params.grid.z + TRIANGLE_FIELD_CANDIDATE_TILE_SIZE - 1u) /
		TRIANGLE_FIELD_CANDIDATE_TILE_SIZE;
	let candidateTileIndex =
		(nodeStorage.y / TRIANGLE_FIELD_CANDIDATE_TILE_SIZE) * candidateTileColumns +
		nodeStorage.x / TRIANGLE_FIELD_CANDIDATE_TILE_SIZE;

	// Stage each chunk once per workgroup; candidate words retain ascending body order.
	for (var bodyChunkStart = 0u; bodyChunkStart < params.counts.w; bodyChunkStart += 64u) {
		let cachedBodyIndex = bodyChunkStart + localIndex;
		if (cachedBodyIndex < params.counts.w) {
			let bodyStart = bondedBodyStarts[cachedBodyIndex];
			if (bodyStart + 2u >= params.counts.x) {
				triangleFieldBodyCache[localIndex] = TriangleFieldBody(
					vec2<f32>(0.0),
					vec2<f32>(0.0),
					vec2<f32>(0.0),
					vec2<f32>(0.0),
					vec2<f32>(0.0),
					bodyStart,
					0u,
				);
			} else {
				let first = particlesIn[bodyStart].position;
				let second = particlesIn[bodyStart + 1u].position;
				let third = particlesIn[bodyStart + 2u].position;
				triangleFieldBodyCache[localIndex] = TriangleFieldBody(
					first,
					second,
					third,
					min(first, min(second, third)),
					max(first, max(second, third)),
					bodyStart,
					1u,
				);
			}
		}

		workgroupBarrier();
		if (fieldActive && !failed) {
			var candidateWord0 = 0xffffffffu;
			var candidateWord1 = 0xffffffffu;
			if (params.counts.w <= TRIANGLE_FIELD_CANDIDATE_BODY_LIMIT) {
				let candidateWordsPerTile =
					(params.counts.w + TRIANGLE_FIELD_CANDIDATE_BODY_WORD_BITS - 1u) /
					TRIANGLE_FIELD_CANDIDATE_BODY_WORD_BITS;
				let candidateWordIndex =
					candidateTileIndex * candidateWordsPerTile +
					bodyChunkStart / TRIANGLE_FIELD_CANDIDATE_BODY_WORD_BITS;
				candidateWord0 = triangleFieldCandidates[candidateWordIndex];
				candidateWord1 = 0u;
				if (bodyChunkStart + TRIANGLE_FIELD_CANDIDATE_BODY_WORD_BITS < params.counts.w) {
					candidateWord1 = triangleFieldCandidates[candidateWordIndex + 1u];
				}
			}

			for (
				var candidateWordOffset = 0u;
				candidateWordOffset < 2u && !failed;
				candidateWordOffset += 1u
			) {
				var candidateWord = select(
					candidateWord0,
					candidateWord1,
					candidateWordOffset == 1u,
				);
				while (candidateWord != 0u && !failed) {
					let bodyOffset =
						candidateWordOffset * TRIANGLE_FIELD_CANDIDATE_BODY_WORD_BITS +
						countTrailingZeros(candidateWord);
					candidateWord = candidateWord & (candidateWord - 1u);
					let bodyIndex = bodyChunkStart + bodyOffset;
					if (bodyIndex >= params.counts.w) {
						break;
					}

					let body = triangleFieldBodyCache[bodyOffset];
					if (body.valid == 0u) {
						atomicOr(&stateError[0], ERROR_TRIANGLE_BOND);
						failed = true;
						break;
					}
					let outside = max(
						max(body.minimum - point, vec2<f32>(0.0)),
						point - body.maximum,
					);
					if (dot(outside, outside) > params.bonding.z * params.bonding.z) {
						continue;
					}

					let sample = triangleDistanceSample(
						point,
						body.first,
						body.second,
						body.third,
					);
					if (sample.valid == 0u) {
						atomicMin(&stateError[1], body.bodyStart);
						atomicOr(&stateError[0], ERROR_TRIANGLE_CONTACT);
						failed = true;
						break;
					}
					if (sample.signedDistance < bestDistance) {
						bestDistance = sample.signedDistance;
						bestBodyStartPlusOne = body.bodyStart + 1u;
						bestNormal = sample.normal;
						bestBarycentric = sample.barycentric;
					}
				}
			}
		}
		workgroupBarrier();
	}

	if (fieldActive && !failed) {
		triangleFieldNodes[index].signedDistance = bestDistance;
		triangleFieldNodes[index].bodyStartPlusOne = bestBodyStartPlusOne;
		triangleFieldNodes[index].normal = bestNormal;
		triangleFieldNodes[index].barycentric = bestBarycentric;
		triangleFieldNodes[index].impulse = vec2<f32>(0.0, 0.0);
	}
}

fn applyWallCushion(velocity: f32, coordinate: i32, extent: i32) -> f32 {
	let cushionCells = min(WALL_CUSHION_CELLS, f32(extent) / 4.0);
	let wallDistance = select(f32(extent - coordinate), f32(coordinate), velocity < 0.0);
	return velocity * clamp(wallDistance / cushionCells, 0.0, 1.0);
}

fn integrateDeformation(
	deformation: vec4<f32>,
	velocityGradient: vec4<f32>,
	dt: f32,
	substeps: u32,
) -> vec4<f32> {
	let substepDt = dt / f32(substeps);
	let update = vec4<f32>(
		1.0 + substepDt * velocityGradient.x,
		substepDt * velocityGradient.y,
		substepDt * velocityGradient.z,
		1.0 + substepDt * velocityGradient.w,
	);
	var result = deformation;
	for (var step = 0u; step < substeps; step += 1u) {
		// Preserve low-order terms in each matrix dot product near singularity.
		result = vec4<f32>(
			fma(update.x, result.x, update.y * result.z),
			fma(update.x, result.y, update.y * result.w),
			fma(update.z, result.x, update.w * result.z),
			fma(update.z, result.y, update.w * result.w),
		);
	}
	return result;
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

	let nodeStorage = vec2<u32>(index % params.grid.z, index / params.grid.z);
	let nodeCoordinate = vec2<i32>(i32(nodeStorage.x) - 1, i32(nodeStorage.y) - 1);
	var solidMass = gridNodes[index].mass;
	var solidMomentum = gridNodes[index].momentum;
	var waterMass = gridNodes[index].waterMass;
	var waterMomentum = gridNodes[index].waterMomentum;
	var waterVolume = gridNodes[index].waterVolume;
	var oilMass = gridNodes[index].oilMass;
	var oilMomentum = gridNodes[index].oilMomentum;
	var oilVolume = gridNodes[index].oilVolume;
	if (
		!finiteFloat(solidMass) ||
		solidMass < 0.0 ||
		!finiteVec2(solidMomentum) ||
		!finiteFloat(waterMass) ||
		waterMass < 0.0 ||
		!finiteVec2(waterMomentum) ||
		!finiteFloat(waterVolume) ||
		waterVolume < 0.0 ||
		!finiteFloat(oilMass) ||
		oilMass < 0.0 ||
		!finiteVec2(oilMomentum) ||
		!finiteFloat(oilVolume) ||
		oilVolume < 0.0
	) {
		atomicOr(&stateError[0], ERROR_GRID);
		return;
	}

	var solidVelocity = finishGridVelocity(solidMass, solidMomentum, nodeCoordinate);
	var waterVelocity = finishGridVelocity(waterMass, waterMomentum, nodeCoordinate);
	var oilVelocity = finishGridVelocity(oilMass, oilMomentum, nodeCoordinate);
	if (
		nodeCoordinate.x >= 0 &&
		nodeCoordinate.y >= 0 &&
		nodeCoordinate.x <= i32(params.grid.x) &&
		nodeCoordinate.y <= i32(params.grid.y) &&
		waterMass > 0.0 &&
		oilMass > 0.0 &&
		waterVolume > 0.0 &&
		oilVolume > 0.0
	) {
		let waterGradient = phaseVolumeGradient(nodeCoordinate, 0u);
		let oilGradient = phaseVolumeGradient(nodeCoordinate, 1u);
		let separation = oilGradient - waterGradient;
		let separationLength = length(separation);
		if (finiteFloat(separationLength) && separationLength > 1e-8) {
			var normal = separation / separationLength;
			let waterDensity = waterMass / waterVolume;
			let oilDensity = oilMass / oilVolume;
			let densityScale = max(waterDensity, oilDensity);
			let densityContrast = (oilDensity - waterDensity) / densityScale;
			let gravity = params.gravityAndDomain.xy;
			let gravityMagnitude = length(gravity);
			var gravityAlignment = 0.0;
			if (
				!finiteFloat(waterDensity) ||
				waterDensity <= 0.0 ||
				!finiteFloat(oilDensity) ||
				oilDensity <= 0.0 ||
				!finiteFloat(densityContrast) ||
				!finiteFloat(gravityMagnitude)
			) {
				atomicOr(&stateError[0], ERROR_GRID);
				return;
			}
			if (gravityMagnitude > 0.0) {
				gravityAlignment = dot(normal, gravity) / gravityMagnitude;
			}
			if (!finiteFloat(gravityAlignment)) {
				atomicOr(&stateError[0], ERROR_GRID);
				return;
			}
			if (densityContrast * gravityAlignment < -LIQUID_STABLE_ORDER_EPSILON) {
				normal = -normal;
			}
			let relativeNormalVelocity = dot(oilVelocity - waterVelocity, normal);
			let overlapFraction = clamp(
				2.0 * min(waterVolume, oilVolume) / (waterVolume + oilVolume),
				0.0,
				1.0,
			);
			let minimumSeparationVelocity = max(
				LIQUID_CONTACT_SEPARATION_FRACTION *
					overlapFraction *
					params.physical.x /
					LIQUID_CONTACT_RELAXATION_SECONDS,
				max(-relativeNormalVelocity, 0.0) *
					LIQUID_CONTACT_APPROACH_SPEED_MULTIPLIER,
			);
			if (relativeNormalVelocity < minimumSeparationVelocity) {
				let impulseMagnitude =
					(minimumSeparationVelocity - relativeNormalVelocity) /
					(1.0 / waterMass + 1.0 / oilMass);
				let impulse = impulseMagnitude * normal;
				waterVelocity = waterVelocity - impulse / waterMass;
				oilVelocity = oilVelocity + impulse / oilMass;
			}
		}
	}

	var field = triangleFieldNodes[index];
	if (field.bodyStartPlusOne != 0u && field.signedDistance <= params.bonding.w) {
		let bodyStart = field.bodyStartPlusOne - 1u;
		if (bodyStart + 2u >= params.counts.x) {
			atomicOr(&stateError[0], ERROR_TRIANGLE_CONTACT);
			return;
		}
		let first = particlesIn[bodyStart];
		let second = particlesIn[bodyStart + 1u];
		let third = particlesIn[bodyStart + 2u];
		let barycentric = vec3<f32>(
			1.0 - field.barycentric.x - field.barycentric.y,
			field.barycentric.x,
			field.barycentric.y,
		);
		let surfaceVelocity =
			barycentric.x * first.velocity +
			barycentric.y * second.velocity +
			barycentric.z * third.velocity;
		let inverseBodyMass =
			dot(barycentric, barycentric) / params.physical.y;
		let normalLength = length(field.normal);
		if (
			!finiteVec2(surfaceVelocity) ||
			!finiteFloat(inverseBodyMass) ||
			inverseBodyMass <= 0.0 ||
			!finiteFloat(normalLength) ||
			normalLength <= 1e-8
		) {
			atomicOr(&stateError[0], ERROR_TRIANGLE_CONTACT);
			return;
		}
		let normal = field.normal / normalLength;
		if (waterMass > 0.0 && waterVolume > 0.0) {
			let relativeNormalVelocity = dot(waterVelocity - surfaceVelocity, normal);
			if (relativeNormalVelocity < 0.0) {
				let impulseMagnitude =
					-relativeNormalVelocity / (1.0 / waterMass + inverseBodyMass);
				let impulse = impulseMagnitude * normal;
				waterVelocity = waterVelocity + impulse / waterMass;
				field.impulse = field.impulse - impulse;
			}
		}
		if (oilMass > 0.0 && oilVolume > 0.0) {
			let relativeNormalVelocity = dot(oilVelocity - surfaceVelocity, normal);
			if (relativeNormalVelocity < 0.0) {
				let impulseMagnitude =
					-relativeNormalVelocity / (1.0 / oilMass + inverseBodyMass);
				let impulse = impulseMagnitude * normal;
				oilVelocity = oilVelocity + impulse / oilMass;
				field.impulse = field.impulse - impulse;
			}
		}
	}

	if (
		!finiteVec2(solidVelocity) ||
		!finiteVec2(waterVelocity) ||
		!finiteVec2(oilVelocity) ||
		!finiteVec2(field.impulse)
	) {
		atomicOr(&stateError[0], ERROR_GRID | ERROR_TRIANGLE_CONTACT);
		return;
	}
	solidVelocity.x = applyWallCushion(
		solidVelocity.x,
		nodeCoordinate.x,
		i32(params.grid.x),
	);
	solidVelocity.y = applyWallCushion(
		solidVelocity.y,
		nodeCoordinate.y,
		i32(params.grid.y),
	);
	waterVelocity.x = applyWallCushion(
		waterVelocity.x,
		nodeCoordinate.x,
		i32(params.grid.x),
	);
	waterVelocity.y = applyWallCushion(
		waterVelocity.y,
		nodeCoordinate.y,
		i32(params.grid.y),
	);
	oilVelocity.x = applyWallCushion(
		oilVelocity.x,
		nodeCoordinate.x,
		i32(params.grid.x),
	);
	oilVelocity.y = applyWallCushion(
		oilVelocity.y,
		nodeCoordinate.y,
		i32(params.grid.y),
	);
	gridNodes[index].velocity = solidVelocity;
	gridNodes[index].waterVelocity = waterVelocity;
	gridNodes[index].oilVelocity = oilVelocity;
	triangleFieldNodes[index].impulse = field.impulse;
}

fn finishGridVelocity(
	mass: f32,
	momentum: vec2<f32>,
	nodeCoordinate: vec2<i32>,
) -> vec2<f32> {
	if (mass <= 0.0) {
		return vec2<f32>(0.0, 0.0);
	}
	var velocity = momentum / mass + params.physical.w * params.gravityAndDomain.xy;
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
		}
	}
	return velocity;
}

fn volumeAtNode(nodeCoordinate: vec2<i32>, phase: u32) -> f32 {
	if (
		nodeCoordinate.x < -1 ||
		nodeCoordinate.y < -1 ||
		nodeCoordinate.x > i32(params.grid.x + 1u) ||
		nodeCoordinate.y > i32(params.grid.y + 1u)
	) {
		return 0.0;
	}
	let node = gridNodes[gridIndex(nodeCoordinate)];
	return select(node.waterVolume, node.oilVolume, phase == 1u);
}

fn phaseVolumeGradient(nodeCoordinate: vec2<i32>, phase: u32) -> vec2<f32> {
	return vec2<f32>(
		volumeAtNode(nodeCoordinate + vec2<i32>(1, 0), phase) -
			volumeAtNode(nodeCoordinate - vec2<i32>(1, 0), phase),
		volumeAtNode(nodeCoordinate + vec2<i32>(0, 1), phase) -
			volumeAtNode(nodeCoordinate - vec2<i32>(0, 1), phase),
	);
}

fn projectLiquidFromTriangles(position: vec2<f32>, radius: f32) -> vec2<f32> {
	let gridPosition = position / params.physical.x;
	let base = vec2<i32>(floor(gridPosition - vec2<f32>(0.5, 0.5)));
	let fraction = gridPosition - vec2<f32>(base);
	let weightsX = quadraticWeights(fraction.x);
	let weightsY = quadraticWeights(fraction.y);
	var signedDistance = 0.0;
	var closestDistance = 3.402823466e38;
	var bodyStartPlusOne = 0u;
	var selectedNormal = vec2<f32>(0.0, 0.0);
	for (var y = 0u; y < 3u; y += 1u) {
		for (var x = 0u; x < 3u; x += 1u) {
			let node = base + vec2<i32>(i32(x), i32(y));
			let field = triangleFieldNodes[gridIndex(node)];
			let weight = weightsX[x] * weightsY[y];
			signedDistance += weight * field.signedDistance;
			if (
				field.bodyStartPlusOne != 0u &&
				field.signedDistance < closestDistance
			) {
				closestDistance = field.signedDistance;
				bodyStartPlusOne = field.bodyStartPlusOne;
				selectedNormal = field.normal;
			}
		}
	}
	if (bodyStartPlusOne != 0u) {
		var normal = vec2<f32>(0.0, 0.0);
		for (var y = 0u; y < 3u; y += 1u) {
			for (var x = 0u; x < 3u; x += 1u) {
				let node = base + vec2<i32>(i32(x), i32(y));
				let field = triangleFieldNodes[gridIndex(node)];
				if (field.bodyStartPlusOne == bodyStartPlusOne) {
					normal += weightsX[x] * weightsY[y] * field.normal;
				}
			}
		}
		let normalLength = length(normal);
		if (!finiteFloat(normalLength) || normalLength <= 1e-8) {
			normal = selectedNormal;
		} else {
			normal = normal / normalLength;
		}
		if (
			!finiteFloat(signedDistance) ||
			signedDistance >= radius ||
			!finiteVec2(normal)
		) {
			return position;
		}
		return position + (radius - signedDistance) * normal;
	}
	return position;
}

fn bondEdgeCorrection(
	first: vec2<f32>,
	second: vec2<f32>,
	restLength: f32,
	stiffness: f32,
) -> vec3<f32> {
	let difference = second - first;
	let distance = length(difference);
	if (!finiteFloat(distance) || distance <= 1e-10) {
		return vec3<f32>(0.0, 0.0, 0.0);
	}
	return vec3<f32>(
		difference * ((distance - restLength) * stiffness * 0.5 / distance),
		1.0,
	);
}

fn trianglePairContact(
	firstA: vec2<f32>,
	secondA: vec2<f32>,
	thirdA: vec2<f32>,
	firstB: vec2<f32>,
	secondB: vec2<f32>,
	thirdB: vec2<f32>,
	skin: f32,
) -> vec3<f32> {
	let edges = array<vec2<f32>, 6>(
		secondA - firstA,
		thirdA - secondA,
		firstA - thirdA,
		secondB - firstB,
		thirdB - secondB,
		firstB - thirdB,
	);
	var axes = array<vec2<f32>, 6>(
		vec2<f32>(0.0),
		vec2<f32>(0.0),
		vec2<f32>(0.0),
		vec2<f32>(0.0),
		vec2<f32>(0.0),
		vec2<f32>(0.0),
	);
	for (var axisIndex = 0u; axisIndex < 6u; axisIndex += 1u) {
		let edgeLength = length(edges[axisIndex]);
		if (!finiteFloat(edgeLength) || edgeLength <= 1e-10) {
			return vec3<f32>(0.0, 0.0, 0.0);
		}
		axes[axisIndex] =
			vec2<f32>(edges[axisIndex].y, -edges[axisIndex].x) / edgeLength;
	}

	var separated = false;
	var largestGap = -3.402823466e38;
	var smallestOverlap = 3.402823466e38;
	var contactAxis = vec2<f32>(0.0, 0.0);
	for (var axisIndex = 0u; axisIndex < 6u; axisIndex += 1u) {
		let axis = axes[axisIndex];
		let minA = min(
			dot(firstA, axis),
			min(dot(secondA, axis), dot(thirdA, axis)),
		);
		let maxA = max(
			dot(firstA, axis),
			max(dot(secondA, axis), dot(thirdA, axis)),
		);
		let minB = min(
			dot(firstB, axis),
			min(dot(secondB, axis), dot(thirdB, axis)),
		);
		let maxB = max(
			dot(firstB, axis),
			max(dot(secondB, axis), dot(thirdB, axis)),
		);
		let gap = max(minB - maxA, minA - maxB);
		if (gap > 0.0) {
			separated = true;
			if (gap > largestGap) {
				largestGap = gap;
				contactAxis = axis;
			}
		} else if (!separated && -gap < smallestOverlap) {
			smallestOverlap = -gap;
			contactAxis = axis;
		}
	}
	if (separated && largestGap > skin) {
		return vec3<f32>(0.0, 0.0, 0.0);
	}
	let centerDelta = (firstA + secondA + thirdA - firstB - secondB - thirdB) / 3.0;
	if (dot(centerDelta, contactAxis) < 0.0) {
		contactAxis = -contactAxis;
	}
	let penetration = select(skin + smallestOverlap, skin - largestGap, separated);
	if (!finiteFloat(penetration) || penetration <= 0.0) {
		return vec3<f32>(0.0, 0.0, 0.0);
	}
	return vec3<f32>(contactAxis, penetration);
}

fn triangleBodyContact(bodyIndex: u32, otherIndex: u32) -> vec3<f32> {
	let bodyStart = bondedBodyStarts[bodyIndex];
	let otherStart = bondedBodyStarts[otherIndex];
	let isFirst = bodyIndex < otherIndex;
	let firstStart = select(otherStart, bodyStart, isFirst);
	let secondStart = select(bodyStart, otherStart, isFirst);
	let firstMinimum = min(
		particlesIn[firstStart].position,
		min(particlesIn[firstStart + 1u].position, particlesIn[firstStart + 2u].position),
	);
	let firstMaximum = max(
		particlesIn[firstStart].position,
		max(particlesIn[firstStart + 1u].position, particlesIn[firstStart + 2u].position),
	);
	let secondMinimum = min(
		particlesIn[secondStart].position,
		min(particlesIn[secondStart + 1u].position, particlesIn[secondStart + 2u].position),
	);
	let secondMaximum = max(
		particlesIn[secondStart].position,
		max(particlesIn[secondStart + 1u].position, particlesIn[secondStart + 2u].position),
	);
	let aabbGap = max(secondMinimum - firstMaximum, firstMinimum - secondMaximum);
	let skin = params.physical.x * TRIANGLE_CONTACT_SKIN_CELLS;
	if (max(aabbGap.x, aabbGap.y) > skin) {
		return vec3<f32>(0.0, 0.0, 0.0);
	}
	let contact = trianglePairContact(
		particlesIn[firstStart].position,
		particlesIn[firstStart + 1u].position,
		particlesIn[firstStart + 2u].position,
		particlesIn[secondStart].position,
		particlesIn[secondStart + 1u].position,
		particlesIn[secondStart + 2u].position,
		skin,
	);
	return vec3<f32>(select(-contact.xy, contact.xy, isFirst), contact.z);
}

@compute @workgroup_size(64)
fn gatherTrianglePairContacts(
	@builtin(global_invocation_id) invocation: vec3<u32>,
) {
	let bodyCount = params.counts.w;
	if (bodyCount == 0u || params.fracture.w == 0u) {
		return;
	}
	let pairIndex = invocation.x;
	if (pairIndex >= bodyCount * bodyCount) {
		return;
	}
	let bodyIndex = pairIndex / bodyCount;
	let otherIndex = pairIndex % bodyCount;
	if (bodyIndex >= otherIndex || atomicLoad(&stateError[0]) != 0u) {
		return;
	}
	// gatherTriangleContactStates validates every body before this pair pass.
	let bodyStart = bondedBodyStarts[bodyIndex];
	let otherStart = bondedBodyStarts[otherIndex];
	let objectId = particleObjectIds[bodyStart];
	if (particleObjectIds[otherStart] == objectId) {
		return;
	}
	let contact = triangleBodyContact(bodyIndex, otherIndex);
	let cacheIndex = bodyIndex * bodyCount + otherIndex;
	trianglePairContactCache[cacheIndex] = vec4<f32>(contact, 0.0);
	if (contact.z > 0.0) {
		atomicAdd(&triangleContactStates[bodyIndex].contactCount, 1u);
		atomicAdd(&triangleContactStates[otherIndex].contactCount, 1u);
		let candidateWordCount =
			(bodyCount + TRIANGLE_PAIR_CONTACT_CANDIDATE_WORD_BITS - 1u) /
			TRIANGLE_PAIR_CONTACT_CANDIDATE_WORD_BITS;
		let bodyCandidateWord =
			bodyIndex * candidateWordCount +
			otherIndex / TRIANGLE_PAIR_CONTACT_CANDIDATE_WORD_BITS;
		let otherCandidateWord =
			otherIndex * candidateWordCount +
			bodyIndex / TRIANGLE_PAIR_CONTACT_CANDIDATE_WORD_BITS;
		atomicOr(
			&trianglePairContactCandidates[bodyCandidateWord],
			1u << (otherIndex % TRIANGLE_PAIR_CONTACT_CANDIDATE_WORD_BITS),
		);
		atomicOr(
			&trianglePairContactCandidates[otherCandidateWord],
			1u << (bodyIndex % TRIANGLE_PAIR_CONTACT_CANDIDATE_WORD_BITS),
		);
	}
}

@compute @workgroup_size(64)
fn gatherTriangleContactStates(
	@builtin(global_invocation_id) invocation: vec3<u32>,
) {
	let bodyIndex = invocation.x;
	if (bodyIndex >= params.counts.w || atomicLoad(&stateError[0]) != 0u) {
		return;
	}
	let bodyStart = bondedBodyStarts[bodyIndex];
	if (
		bodyStart + 2u >= params.counts.x ||
		particlesIn[bodyStart].bondedTriangleVertex != 1u ||
		particlesIn[bodyStart + 1u].bondedTriangleVertex != 2u ||
		particlesIn[bodyStart + 2u].bondedTriangleVertex != 3u
	) {
		atomicMin(&stateError[1], bodyStart);
		atomicOr(&stateError[0], ERROR_TRIANGLE_BOND);
		return;
	}
	var contactCount = 0u;
	if (params.fracture.w == 0u) {
		let objectId = particleObjectIds[bodyStart];
		for (var otherIndex = 0u; otherIndex < params.counts.w; otherIndex += 1u) {
			if (otherIndex == bodyIndex) {
				continue;
			}
			let otherStart = bondedBodyStarts[otherIndex];
			if (
				otherStart + 2u >= params.counts.x ||
				particlesIn[otherStart].bondedTriangleVertex != 1u ||
				particlesIn[otherStart + 1u].bondedTriangleVertex != 2u ||
				particlesIn[otherStart + 2u].bondedTriangleVertex != 3u
			) {
				atomicMin(&stateError[1], otherStart);
				atomicOr(&stateError[0], ERROR_TRIANGLE_BOND);
				return;
			}
			if (particleObjectIds[otherStart] == objectId) {
				continue;
			}
			let contact = triangleBodyContact(bodyIndex, otherIndex);
			if (contact.z > 0.0) {
				contactCount += 1u;
			}
		}
	}
	if (params.fracture.w != 0u) {
		let candidateWordCount =
			(params.counts.w + TRIANGLE_PAIR_CONTACT_CANDIDATE_WORD_BITS - 1u) /
			TRIANGLE_PAIR_CONTACT_CANDIDATE_WORD_BITS;
		for (
			var wordIndex = 0u;
			wordIndex < candidateWordCount;
			wordIndex += 1u
		) {
			atomicStore(
				&trianglePairContactCandidates[bodyIndex * candidateWordCount + wordIndex],
				0u,
			);
		}
	}
	let velocity =
		(particlesOut[bodyStart].velocity +
			particlesOut[bodyStart + 1u].velocity +
			particlesOut[bodyStart + 2u].velocity) / 3.0;
	triangleContactStates[bodyIndex].velocity = velocity;
	atomicStore(&triangleContactStates[bodyIndex].contactCount, contactCount);
	triangleContactStates[bodyIndex].padding = 0u;
}

@compute @workgroup_size(64)
fn g2p(@builtin(global_invocation_id) invocation: vec3<u32>) {
	let particleIndex = invocation.x;
	if (particleIndex >= params.counts.x) {
		return;
	}
	let particle = particlesIn[particleIndex];
	if (particle.materialModel == MODEL_REMOVED) {
		particlesOut[particleIndex] = particle;
		return;
	}
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
	let isWaterParticle = particle.materialModel == MODEL_WATER;
	let isOilParticle = particle.materialModel == MODEL_OIL;
	let isLiquidParticle = isWaterParticle || isOilParticle;
	var velocity = vec2<f32>(0.0, 0.0);
	var affine = vec4<f32>(0.0, 0.0, 0.0, 0.0);
	var phaseMass = 0.0;
	var phaseVolume = 0.0;

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
			var gridVelocity = gridNodes[nodeIndex].velocity;
			if (isWaterParticle) {
				gridVelocity = gridNodes[nodeIndex].waterVelocity;
			} else if (isOilParticle) {
				gridVelocity = gridNodes[nodeIndex].oilVelocity;
			}
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
			if (isLiquidParticle) {
				let nodePhaseMass =
					gridNodes[nodeIndex].waterMass + gridNodes[nodeIndex].oilMass;
				let nodePhaseVolume =
					gridNodes[nodeIndex].waterVolume + gridNodes[nodeIndex].oilVolume;
				if (
					!finiteFloat(nodePhaseMass) ||
					nodePhaseMass < 0.0 ||
					!finiteFloat(nodePhaseVolume) ||
					nodePhaseVolume < 0.0
				) {
					atomicMin(&stateError[1], particleIndex);
					atomicOr(&stateError[0], ERROR_G2P);
					return;
				}
				phaseMass = phaseMass + weight * nodePhaseMass;
				phaseVolume = phaseVolume + weight * nodePhaseVolume;
			}
			affine = affine + vec4<f32>(
				weight * gridVelocity.x * dpos.x,
				weight * gridVelocity.x * dpos.y,
				weight * gridVelocity.y * dpos.x,
				weight * gridVelocity.y * dpos.y,
			);
		}
	}

	if (isLiquidParticle) {
		let particleDeterminant = determinant2x2(particle.f);
		if (
			!finiteFloat(phaseMass) ||
			phaseMass <= 0.0 ||
			!finiteFloat(phaseVolume) ||
			phaseVolume <= 0.0 ||
			!finiteFloat(particleDeterminant) ||
			particleDeterminant <= 0.0
		) {
			atomicMin(&stateError[1], particleIndex);
			atomicOr(&stateError[0], ERROR_G2P);
			return;
		}
		let particleDensity = particle.density / particleDeterminant;
		let localDensity = phaseMass / phaseVolume;
		let relativeDensity = (particleDensity - localDensity) / particleDensity;
		let separationAcceleration = relativeDensity * params.gravityAndDomain.xy;
		velocity = velocity + params.physical.w * separationAcceleration;
		if (
			!finiteFloat(particleDensity) ||
			particleDensity <= 0.0 ||
			!finiteFloat(localDensity) ||
			localDensity <= 0.0 ||
			!finiteFloat(relativeDensity) ||
			!finiteVec2(velocity)
		) {
			atomicMin(&stateError[1], particleIndex);
			atomicOr(&stateError[0], ERROR_G2P);
			return;
		}
	}

	let updatedC = affine * params.material.w;
	let dt = params.physical.w;
	let updatedPosition = particle.position + dt * velocity;
	if (
		!finiteVec2(velocity) ||
		!finiteVec4(updatedC) ||
		!finiteVec2(updatedPosition)
	) {
		atomicOr(&stateError[0], ERROR_G2P);
		return;
	}
	let deformationRowSum = max(
		abs(updatedC.x) + abs(updatedC.y),
		abs(updatedC.z) + abs(updatedC.w),
	);
	let deformationIncrement = dt * deformationRowSum;
	let requiredSubsteps = ceil(deformationIncrement / MAX_DEFORMATION_INCREMENT);
	if (
		!finiteFloat(requiredSubsteps) ||
		requiredSubsteps > MAX_DEFORMATION_SUBSTEPS
	) {
		atomicMin(&stateError[1], particleIndex);
		atomicOr(&stateError[0], ERROR_DEFORMATION_SUBSTEP_LIMIT);
		return;
	}
	var substeps = max(1u, u32(requiredSubsteps));
	let maximumSubsteps = u32(MAX_DEFORMATION_SUBSTEPS);
	var trialF = integrateDeformation(particle.f, updatedC, dt, substeps);
	if (!finiteVec4(trialF)) {
		atomicOr(&stateError[0], ERROR_G2P);
		return;
	}
	var trialDeterminant = determinant2x2(trialF);
	// A near-singular F can lose its orientation to float32 rounding even
	// though each bounded update has positive determinant. Retry only that
	// update with finer substeps, and never accept a non-positive trial.
	while (
		finiteFloat(trialDeterminant) &&
		trialDeterminant <= 0.0 &&
		substeps < maximumSubsteps
	) {
		substeps = min(substeps * 2u, maximumSubsteps);
		trialF = integrateDeformation(particle.f, updatedC, dt, substeps);
		if (!finiteVec4(trialF)) {
			atomicOr(&stateError[0], ERROR_G2P);
			return;
		}
		trialDeterminant = determinant2x2(trialF);
	}
	var updatedF = trialF;
	var updatedPlasticVolume = particle.plasticVolume;
	if (!finiteFloat(trialDeterminant) || trialDeterminant <= 0.0) {
		recordDeformationFailure(
			particleIndex,
			trialDeterminant,
			deformationIncrement,
			determinant2x2(particle.f),
		);
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
		particle.materialModel == MODEL_WATER ||
		particle.materialModel == MODEL_OIL
	) {
		updatedF = projectLiquidDeformation(trialF);
	} else if (particle.materialModel == MODEL_NEO_HOOKEAN) {
		let stretchTrace = dot(trialF, trialF);
		if (!finiteFloat(stretchTrace)) {
			atomicOr(&stateError[0], ERROR_G2P);
			return;
		}
		if (
			trialDeterminant <
				MIN_NEO_HOOKEAN_STRETCH * sqrt(stretchTrace) ||
			stretchTrace > MAX_NEO_HOOKEAN_STRETCH_SQUARED
		) {
			updatedF = projectNeoHookeanDeformation(trialF);
		}
	} else if (
		particle.materialModel != MODEL_NEO_HOOKEAN &&
		particle.materialModel != MODEL_WATER &&
		particle.materialModel != MODEL_OIL
	) {
		atomicOr(&stateError[0], ERROR_G2P);
		return;
	}
	let updatedDeterminant = determinant2x2(updatedF);
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
		recordProjectedDeformationFailure(particleIndex, updatedF);
		atomicOr(&stateError[0], ERROR_PROJECTED_DEFORMATION);
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
	var projectedPosition = vec2<f32>(boundedX.x, boundedY.x);
	if (isLiquidParticle) {
		let particleRadius = sqrt(particle.volume * updatedDeterminant / 3.14159265359);
		if (!finiteFloat(particleRadius)) {
			atomicMin(&stateError[1], particleIndex);
			atomicOr(&stateError[0], ERROR_G2P);
			return;
		}
		projectedPosition = projectLiquidFromTriangles(projectedPosition, particleRadius);
	}
	let finalX = confineAxis(
		projectedPosition.x,
		boundedX.y,
		params.gravityAndDomain.z,
	);
	let finalY = confineAxis(
		projectedPosition.y,
		boundedY.y,
		params.gravityAndDomain.w,
	);
	var updatedParticle = particle;
	updatedParticle.position = vec2<f32>(finalX.x, finalY.x);
	updatedParticle.velocity = vec2<f32>(finalX.y, finalY.y);
	updatedParticle.c = updatedC;
	updatedParticle.f = updatedF;
	updatedParticle.plasticVolume = updatedPlasticVolume;
	particlesOut[particleIndex] = updatedParticle;
}

@compute @workgroup_size(64)
fn validateCflParticles(@builtin(global_invocation_id) invocation: vec3<u32>) {
	let particleIndex = invocation.x;
	if (
		particleIndex >= params.counts.x ||
		params.cflBounds.z < 0.5 ||
		atomicLoad(&stateError[0]) != 0u
	) {
		return;
	}

	let particle = particlesIn[particleIndex];
	if (particle.materialModel == MODEL_REMOVED) {
		return;
	}
	let speed = length(particle.velocity);
	let waveSpeed = particleWaveSpeedBound(particle);
	if (
		!finiteVec2(particle.velocity) ||
		!finiteFloat(speed) ||
		!finiteFloat(waveSpeed) ||
		speed > params.cflBounds.y ||
		waveSpeed > params.cflBounds.x
	) {
		let previousClaim = atomicOr(&stateError[6], 3u);
		if (previousClaim == 0u) {
			atomicStore(&stateError[1], particleIndex);
			atomicStore(&stateError[2], bitcast<u32>(speed));
			atomicStore(&stateError[3], bitcast<u32>(waveSpeed));
			atomicStore(&stateError[4], bitcast<u32>(params.cflBounds.y));
			atomicStore(&stateError[5], bitcast<u32>(params.cflBounds.x));
		}
		atomicOr(&stateError[0], ERROR_CFL_BOUND);
	}
}

@compute @workgroup_size(64)
fn measureParticleSpeedPartials(
	@builtin(workgroup_id) workgroup: vec3<u32>,
	@builtin(local_invocation_id) local: vec3<u32>,
) {
	let particleIndex = workgroup.x * 64u + local.x;
	var maximumSpeed = 0.0;
	if (particleIndex < params.counts.x) {
		let particle = particlesIn[particleIndex];
		if (particle.materialModel != MODEL_REMOVED) {
			let speed = length(particle.velocity);
			maximumSpeed = select(3.402823466e38, speed, finiteFloat(speed));
		}
	}
	cflSpeedReduction[local.x] = maximumSpeed;
	workgroupBarrier();

	var stride = 32u;
	loop {
		if (stride == 0u) {
			break;
		}
		if (local.x < stride) {
			cflSpeedReduction[local.x] = max(
				cflSpeedReduction[local.x],
				cflSpeedReduction[local.x + stride],
			);
		}
		workgroupBarrier();
		stride = stride / 2u;
	}
	if (local.x == 0u) {
		cflSpeedPartials[workgroup.x] = cflSpeedReduction[0];
	}
}

@compute @workgroup_size(64)
fn reduceParticleSpeedPartials(
	@builtin(local_invocation_id) local: vec3<u32>,
) {
	let partialCount = (params.counts.x + 63u) / 64u;
	var maximumSpeed = 0.0;
	var partialIndex = local.x;
	loop {
		if (partialIndex >= partialCount) {
			break;
		}
		maximumSpeed = max(maximumSpeed, cflSpeedPartials[partialIndex]);
		partialIndex = partialIndex + 64u;
	}
	cflSpeedReduction[local.x] = maximumSpeed;
	workgroupBarrier();

	var stride = 32u;
	loop {
		if (stride == 0u) {
			break;
		}
		if (local.x < stride) {
			cflSpeedReduction[local.x] = max(
				cflSpeedReduction[local.x],
				cflSpeedReduction[local.x + stride],
			);
		}
		workgroupBarrier();
		stride = stride / 2u;
	}
	if (local.x == 0u) {
		cflMaximumSpeed[0] = cflSpeedReduction[0];
	}
}

@compute @workgroup_size(64)
fn applyTriangleContactImpulses(
	@builtin(global_invocation_id) invocation: vec3<u32>,
) {
	let bodyIndex = invocation.x;
	if (bodyIndex >= params.counts.w || atomicLoad(&stateError[0]) != 0u) {
		return;
	}
	let bodyStart = bondedBodyStarts[bodyIndex];
	if (
		bodyStart + 2u >= params.counts.x ||
		particlesIn[bodyStart].bondedTriangleVertex != 1u ||
		particlesIn[bodyStart + 1u].bondedTriangleVertex != 2u ||
		particlesIn[bodyStart + 2u].bondedTriangleVertex != 3u
	) {
		atomicMin(&stateError[1], bodyStart);
		atomicOr(&stateError[0], ERROR_TRIANGLE_BOND);
		return;
	}
	let first = particlesIn[bodyStart].position;
	let second = particlesIn[bodyStart + 1u].position;
	let third = particlesIn[bodyStart + 2u].position;
	let minimum = min(first, min(second, third)) - vec2<f32>(params.bonding.w);
	let maximum = max(first, max(second, third)) + vec2<f32>(params.bonding.w);
	let minimumNode = clamp(
		vec2<i32>(floor(minimum / params.physical.x)),
		vec2<i32>(-1),
		vec2<i32>(i32(params.grid.x + 1u), i32(params.grid.y + 1u)),
	);
	let maximumNode = clamp(
		vec2<i32>(ceil(maximum / params.physical.x)),
		vec2<i32>(-1),
		vec2<i32>(i32(params.grid.x + 1u), i32(params.grid.y + 1u)),
	);
	var impulseFirst = vec2<f32>(0.0, 0.0);
	var impulseSecond = vec2<f32>(0.0, 0.0);
	var impulseThird = vec2<f32>(0.0, 0.0);
	for (var y = minimumNode.y; y <= maximumNode.y; y += 1) {
		for (var x = minimumNode.x; x <= maximumNode.x; x += 1) {
			let field = triangleFieldNodes[gridIndex(vec2<i32>(x, y))];
			if (field.bodyStartPlusOne != bodyStart + 1u) {
				continue;
			}
			let firstWeight = 1.0 - field.barycentric.x - field.barycentric.y;
			impulseFirst += firstWeight * field.impulse;
			impulseSecond += field.barycentric.x * field.impulse;
			impulseThird += field.barycentric.y * field.impulse;
		}
	}
	let inverseMass = 1.0 / params.physical.y;
	let firstVelocity = particlesOut[bodyStart].velocity + inverseMass * impulseFirst;
	let secondVelocity = particlesOut[bodyStart + 1u].velocity + inverseMass * impulseSecond;
	let thirdVelocity = particlesOut[bodyStart + 2u].velocity + inverseMass * impulseThird;
	if (
		!finiteVec2(firstVelocity) ||
		!finiteVec2(secondVelocity) ||
		!finiteVec2(thirdVelocity)
	) {
		atomicMin(&stateError[1], bodyStart);
		atomicOr(&stateError[0], ERROR_TRIANGLE_CONTACT);
		return;
	}
	particlesOut[bodyStart].velocity = firstVelocity;
	particlesOut[bodyStart + 1u].velocity = secondVelocity;
	particlesOut[bodyStart + 2u].velocity = thirdVelocity;
}

@compute @workgroup_size(64)
fn solveBondedTriangles(@builtin(global_invocation_id) invocation: vec3<u32>) {
	let bodyIndex = invocation.x;
	if (bodyIndex >= params.counts.w || atomicLoad(&stateError[0]) != 0u) {
		return;
	}
	let bodyStart = bondedBodyStarts[bodyIndex];
	if (
		bodyStart + 2u >= params.counts.x ||
		particlesIn[bodyStart].bondedTriangleVertex != 1u ||
		particlesIn[bodyStart + 1u].bondedTriangleVertex != 2u ||
		particlesIn[bodyStart + 2u].bondedTriangleVertex != 3u
	) {
		atomicMin(&stateError[1], bodyStart);
		atomicOr(&stateError[0], ERROR_TRIANGLE_BOND);
		return;
	}
	var first = particlesOut[bodyStart];
	var second = particlesOut[bodyStart + 1u];
	var third = particlesOut[bodyStart + 2u];
	let originalFirst = first.position;
	let originalSecond = second.position;
	let originalThird = third.position;
	let originalOffsets = array<vec2<f32>, 3>(
		vec2<f32>(0.0, 0.0),
		originalSecond - originalFirst,
		originalThird - originalFirst,
	);
	var firstOffset = originalOffsets[0];
	var secondOffset = originalOffsets[1];
	var thirdOffset = originalOffsets[2];
	let dt = params.physical.w;
	let inverseDt = 1.0 / dt;
	let stepRatio = dt / params.cflBounds.w;
	let bondStiffness =
		min(params.bonding.y * stepRatio * stepRatio, 1.0);
	for (
		var iteration = 0u;
		iteration < BONDED_TRIANGLE_SOLVER_ITERATIONS;
		iteration += 1u
	) {
		let firstEdge = bondEdgeCorrection(
			firstOffset,
			secondOffset,
			params.bonding.x,
			bondStiffness,
		);
		if (firstEdge.z == 0.0) {
			atomicMin(&stateError[1], bodyStart);
			atomicOr(&stateError[0], ERROR_TRIANGLE_BOND);
			return;
		}
		firstOffset += firstEdge.xy;
		secondOffset -= firstEdge.xy;

		let secondEdge = bondEdgeCorrection(
			secondOffset,
			thirdOffset,
			params.bonding.x,
			bondStiffness,
		);
		if (secondEdge.z == 0.0) {
			atomicMin(&stateError[1], bodyStart);
			atomicOr(&stateError[0], ERROR_TRIANGLE_BOND);
			return;
		}
		secondOffset += secondEdge.xy;
		thirdOffset -= secondEdge.xy;

		let thirdEdge = bondEdgeCorrection(
			thirdOffset,
			firstOffset,
			params.bonding.x,
			bondStiffness,
		);
		if (thirdEdge.z == 0.0) {
			atomicMin(&stateError[1], bodyStart);
			atomicOr(&stateError[0], ERROR_TRIANGLE_BOND);
			return;
		}
		thirdOffset += thirdEdge.xy;
		firstOffset -= thirdEdge.xy;
	}

	let bondedOffsets = array<vec2<f32>, 3>(
		firstOffset,
		secondOffset,
		thirdOffset,
	);
	let originalPositions = array<vec2<f32>, 3>(
		originalFirst,
		originalSecond,
		originalThird,
	);
	var particles = array<Particle, 3>(first, second, third);
	for (var vertex = 0u; vertex < 3u; vertex += 1u) {
		let displacement = bondedOffsets[vertex] - originalOffsets[vertex];
		let position = originalPositions[vertex] + displacement;
		let velocity =
			particles[vertex].velocity +
			displacement * inverseDt;
		let boundedX = confineAxis(
			position.x,
			velocity.x,
			params.gravityAndDomain.z,
		);
		let boundedY = confineAxis(
			position.y,
			velocity.y,
			params.gravityAndDomain.w,
		);
		particles[vertex].position = vec2<f32>(boundedX.x, boundedY.x);
		particles[vertex].velocity = vec2<f32>(boundedX.y, boundedY.y);
		if (
			!finiteVec2(particles[vertex].position) ||
			!finiteVec2(particles[vertex].velocity)
		) {
			atomicMin(&stateError[1], bodyStart);
			atomicOr(&stateError[0], ERROR_TRIANGLE_CONTACT);
			return;
		}
	}
	particlesOut[bodyStart] = particles[0];
	particlesOut[bodyStart + 1u] = particles[1];
	particlesOut[bodyStart + 2u] = particles[2];
}

@compute @workgroup_size(64)
fn applyTrianglePairContacts(@builtin(global_invocation_id) invocation: vec3<u32>) {
	let bodyIndex = invocation.x;
	if (bodyIndex >= params.counts.w || atomicLoad(&stateError[0]) != 0u) {
		return;
	}
	let bodyStart = bondedBodyStarts[bodyIndex];
	if (
		bodyStart + 2u >= params.counts.x ||
		particlesOut[bodyStart].bondedTriangleVertex != 1u ||
		particlesOut[bodyStart + 1u].bondedTriangleVertex != 2u ||
		particlesOut[bodyStart + 2u].bondedTriangleVertex != 3u
	) {
		atomicMin(&stateError[1], bodyStart);
		atomicOr(&stateError[0], ERROR_TRIANGLE_BOND);
		return;
	}
	let bodyVelocity = triangleContactStates[bodyIndex].velocity;
	let bodyContactCount = atomicLoad(&triangleContactStates[bodyIndex].contactCount);
	let objectId = particleObjectIds[bodyStart];
	let bodyMass = 3.0 * params.physical.y;
	var collisionVelocityDelta = vec2<f32>(0.0, 0.0);
	var otherIndex = 0u;
	var fallbackIndex = 0u;
	var candidateWordIndex = 0u;
	var candidateWord = 0u;
	let candidateWordCount =
		(params.counts.w + TRIANGLE_PAIR_CONTACT_CANDIDATE_WORD_BITS - 1u) /
		TRIANGLE_PAIR_CONTACT_CANDIDATE_WORD_BITS;
	// Exact cached contacts are visited in ascending body order; larger body sets
	// retain the uncached full scan without imposing a candidate cap.
	loop {
		if (bodyContactCount == 0u) {
			break;
		}
		if (params.fracture.w != 0u) {
			if (candidateWord == 0u) {
				if (candidateWordIndex >= candidateWordCount) {
					break;
				}
				candidateWord = atomicLoad(
					&trianglePairContactCandidates[
						bodyIndex * candidateWordCount + candidateWordIndex
					],
				);
				if (candidateWord == 0u) {
					candidateWordIndex += 1u;
					continue;
				}
			}
			otherIndex =
				candidateWordIndex * TRIANGLE_PAIR_CONTACT_CANDIDATE_WORD_BITS +
				countTrailingZeros(candidateWord);
			candidateWord = candidateWord & (candidateWord - 1u);
			if (candidateWord == 0u) {
				candidateWordIndex += 1u;
			}
		} else {
			if (fallbackIndex >= params.counts.w) {
				break;
			}
			otherIndex = fallbackIndex;
			fallbackIndex += 1u;
			if (otherIndex == bodyIndex) {
				continue;
			}
		}
		var contact = vec3<f32>(0.0);
		if (params.fracture.w != 0u) {
			let firstIndex = min(bodyIndex, otherIndex);
			let secondIndex = max(bodyIndex, otherIndex);
			let cacheIndex = firstIndex * params.counts.w + secondIndex;
			let cachedContact = trianglePairContactCache[cacheIndex].xyz;
			// Avoid peer particle loads for the usual cached non-contact pair.
			if (cachedContact.z <= 0.0) {
				continue;
			}
			contact = vec3<f32>(
				select(-cachedContact.xy, cachedContact.xy, bodyIndex < otherIndex),
				cachedContact.z
			);
		}
		let otherStart = bondedBodyStarts[otherIndex];
		if (
			otherStart + 2u >= params.counts.x ||
			particlesOut[otherStart].bondedTriangleVertex != 1u ||
			particlesOut[otherStart + 1u].bondedTriangleVertex != 2u ||
			particlesOut[otherStart + 2u].bondedTriangleVertex != 3u
		) {
			atomicMin(&stateError[1], otherStart);
			atomicOr(&stateError[0], ERROR_TRIANGLE_BOND);
			return;
		}
		if (particleObjectIds[otherStart] == objectId) {
			continue;
		}
		if (params.fracture.w == 0u) {
			contact = triangleBodyContact(bodyIndex, otherIndex);
			if (contact.z <= 0.0) {
				continue;
			}
		}
		let otherVelocity = triangleContactStates[otherIndex].velocity;
		let otherContactCount =
			atomicLoad(&triangleContactStates[otherIndex].contactCount);
		let contactDegree = max(bodyContactCount, otherContactCount);
		if (contactDegree == 0u) {
			atomicMin(&stateError[1], bodyStart);
			atomicOr(&stateError[0], ERROR_TRIANGLE_CONTACT);
			return;
		}
		let contactWeight = 1.0 / f32(contactDegree);
		let relativeNormalVelocity =
			dot(bodyVelocity - otherVelocity, contact.xy);
		if (relativeNormalVelocity < 0.0) {
			let impulseMagnitude =
				-relativeNormalVelocity / (2.0 / bodyMass);
			let impulseWeight =
				TRIANGLE_CONTACT_IMPULSE_RELAXATION * contactWeight;
			collisionVelocityDelta +=
				impulseWeight * impulseMagnitude * contact.xy / bodyMass;
		}
	}

	var particles = array<Particle, 3>(
		particlesOut[bodyStart],
		particlesOut[bodyStart + 1u],
		particlesOut[bodyStart + 2u],
	);
	for (var vertex = 0u; vertex < 3u; vertex += 1u) {
		let position = particles[vertex].position;
		let velocity = particles[vertex].velocity + collisionVelocityDelta;
		let boundedX = confineAxis(
			position.x,
			velocity.x,
			params.gravityAndDomain.z,
		);
		let boundedY = confineAxis(
			position.y,
			velocity.y,
			params.gravityAndDomain.w,
		);
		particles[vertex].position = vec2<f32>(boundedX.x, boundedY.x);
		particles[vertex].velocity = vec2<f32>(boundedX.y, boundedY.y);
		if (
			!finiteVec2(particles[vertex].position) ||
			!finiteVec2(particles[vertex].velocity)
		) {
			atomicMin(&stateError[1], bodyStart);
			atomicOr(&stateError[0], ERROR_TRIANGLE_CONTACT);
			return;
		}
	}
	particlesOut[bodyStart] = particles[0];
	particlesOut[bodyStart + 1u] = particles[1];
	particlesOut[bodyStart + 2u] = particles[2];
}

@compute @workgroup_size(64)
fn updateFractureBonds(@builtin(global_invocation_id) invocation: vec3<u32>) {
	let bondIndex = invocation.x;
	if (
		bondIndex >= params.fracture.y ||
		atomicLoad(&stateError[0]) != 0u
	) {
		return;
	}
	var bond = fractureBonds[bondIndex];
	let firstParticle = bond.endpoints.x;
	let secondParticle = bond.endpoints.y;
	if (
		firstParticle >= params.fracture.x ||
		secondParticle >= params.fracture.x ||
		firstParticle >= params.counts.x ||
		secondParticle >= params.counts.x ||
		!finiteFloat(bond.restLength) ||
		bond.restLength <= 0.0 ||
		!finiteFloat(bond.failureStrain) ||
		bond.failureStrain <= 0.0
	) {
		atomicMin(&stateError[1], min(firstParticle, secondParticle));
		atomicOr(&stateError[0], ERROR_FRACTURE_BOND);
		return;
	}
	if (bond.broken != 0u) {
		bond.damage = 1.0;
		fractureBonds[bondIndex] = bond;
		return;
	}
	let separation =
		particlesOut[secondParticle].position - particlesOut[firstParticle].position;
	let distance = length(separation);
	if (!finiteFloat(distance)) {
		atomicMin(&stateError[1], firstParticle);
		atomicOr(&stateError[0], ERROR_FRACTURE_BOND);
		return;
	}
	let tensileStrain = max(0.0, (distance - bond.restLength) / bond.restLength);
	bond.damage = max(bond.damage, min(1.0, tensileStrain / bond.failureStrain));
	if (tensileStrain >= bond.failureStrain) {
		bond.damage = 1.0;
		bond.broken = 1u;
	}
	fractureBonds[bondIndex] = bond;
}

@compute @workgroup_size(64)
fn applyFractureParticleDamage(@builtin(global_invocation_id) invocation: vec3<u32>) {
	let particleIndex = invocation.x;
	if (
		particleIndex >= params.fracture.x ||
		particleIndex >= params.counts.x ||
		atomicLoad(&stateError[0]) != 0u
	) {
		return;
	}
	let firstOffset = fractureBondTopology[particleIndex];
	let nextOffset = fractureBondTopology[particleIndex + 1u];
	if (nextOffset < firstOffset || nextOffset > params.fracture.z) {
		atomicMin(&stateError[1], particleIndex);
		atomicOr(&stateError[0], ERROR_FRACTURE_BOND);
		return;
	}
	var particle = particlesOut[particleIndex];
	if (particle.materialModel != MODEL_NEO_HOOKEAN) {
		return;
	}
	var damage = clamp(particle.frictionCoefficient, 0.0, 1.0);
	let incidenceBase = params.fracture.x + 1u;
	for (var incidence = firstOffset; incidence < nextOffset; incidence = incidence + 1u) {
		let bondIndex = fractureBondTopology[incidenceBase + incidence];
		if (bondIndex >= params.fracture.y) {
			atomicMin(&stateError[1], particleIndex);
			atomicOr(&stateError[0], ERROR_FRACTURE_BOND);
			return;
		}
		damage = max(damage, fractureBonds[bondIndex].damage);
	}
	particle.frictionCoefficient = damage;
	particlesOut[particleIndex] = particle;
}

@compute @workgroup_size(64)
fn removeParticlesInOutlet(@builtin(global_invocation_id) invocation: vec3<u32>) {
	let particleIndex = invocation.x;
	if (particleIndex >= params.counts.x) {
		return;
	}
	var particle = particlesIn[particleIndex];
	if (particle.materialModel == MODEL_REMOVED) {
		particlesOut[particleIndex] = particle;
		return;
	}
	let position = particle.position;
	let bounds = outletParams.bounds;
	if (
		(particle.materialModel == MODEL_WATER || particle.materialModel == MODEL_OIL) &&
		position.x >= bounds.x &&
		position.y >= bounds.y &&
		position.x <= bounds.z &&
		position.y <= bounds.w
	) {
		particle.materialModel = MODEL_REMOVED;
		atomicAdd(&outletRemovalCounter[0], 1u);
	}
	particlesOut[particleIndex] = particle;
}

@compute @workgroup_size(64)
fn commitParticles(@builtin(global_invocation_id) invocation: vec3<u32>) {
	let particleIndex = invocation.x;
	if (particleIndex >= params.counts.x) {
		return;
	}
	if (atomicLoad(&stateError[0]) != 0u) {
		particlesOut[particleIndex] = particlesIn[particleIndex];
		return;
	}
	if (particleIndex == 0u && params.cflBounds.z >= 0.5) {
		atomicAdd(&cflStepProgress[0], 1u);
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
		if (particlesIn[index].materialModel == MODEL_REMOVED) {
			continue;
		}
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
	if (
		localIndex % 3u == 0u &&
		localIndex / 3u < cloneParams.bondRangeAndCapacity.x
	) {
		let bodyListIndex =
			cloneParams.bondRangeAndCapacity.y + localIndex / 3u;
		if (bodyListIndex < cloneParams.bondRangeAndCapacity.z) {
			bondedBodyStarts[bodyListIndex] = destinationIndex;
		}
	}
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
const MODEL_REMOVED: u32 = 0xffffffffu;

struct RenderParams {
	domainAndRadius: vec4<f32>,
}

struct SurfaceRenderParams {
	grid: vec4<u32>,
	domainAndCell: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: RenderParams;
@group(0) @binding(2) var<storage, read> surfaceFields: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> surfaceParams: SurfaceRenderParams;
@group(0) @binding(4) var<storage, read> bondedBodyStarts: array<u32>;

const PARTICLE_MARKER_REFERENCE_AREA: f32 = 25.0;
const PARTICLE_MARKER_AREA_SCALE: f32 = 2.0;

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
	if (particle.materialModel == MODEL_REMOVED) {
		var hidden: VertexOutput;
		hidden.position = vec4<f32>(2.0, 2.0, 0.0, 1.0);
		hidden.localPosition = local;
		hidden.materialColor = vec3<f32>(0.0, 0.0, 0.0);
		return hidden;
	}
	let isLiquid =
		particle.materialModel == MODEL_WATER ||
		particle.materialModel == MODEL_OIL;
	if (params.domainAndRadius.w > 0.5 && isLiquid) {
		var hidden: VertexOutput;
		hidden.position = vec4<f32>(2.0, 2.0, 0.0, 1.0);
		hidden.localPosition = local;
		hidden.materialColor = vec3<f32>(0.0, 0.0, 0.0);
		return hidden;
	}
	let worldPosition =
		particle.position +
		local *
		params.domainAndRadius.z *
		sqrt(
			particle.volume * PARTICLE_MARKER_AREA_SCALE /
				PARTICLE_MARKER_REFERENCE_AREA,
		);
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
	if (particle.materialModel == MODEL_NEO_HOOKEAN) {
		materialColor = mix(
			materialColor,
			vec3<f32>(0.94, 0.16, 0.13),
			clamp(particle.frictionCoefficient, 0.0, 1.0),
		);
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

struct PebbleVertexOutput {
	@builtin(position) position: vec4<f32>,
	@location(0) barycentric: vec3<f32>,
}

@vertex
fn pebbleVertexMain(
	@builtin(vertex_index) vertexIndex: u32,
	@builtin(instance_index) instanceIndex: u32,
) -> PebbleVertexOutput {
	let start = bondedBodyStarts[instanceIndex];
	let particle = particles[start + vertexIndex];
	let worldPosition = particle.position;
	let clipPosition = vec2<f32>(
		2.0 * worldPosition.x / params.domainAndRadius.x - 1.0,
		1.0 - 2.0 * worldPosition.y / params.domainAndRadius.y,
	);
	let barycentricCoordinates = array<vec3<f32>, 3>(
		vec3<f32>(1.0, 0.0, 0.0),
		vec3<f32>(0.0, 1.0, 0.0),
		vec3<f32>(0.0, 0.0, 1.0),
	);
	var output: PebbleVertexOutput;
	output.position = vec4<f32>(clipPosition, 0.0, 1.0);
	output.barycentric = barycentricCoordinates[vertexIndex];
	return output;
}

@fragment
fn pebbleFragmentMain(input: PebbleVertexOutput) -> @location(0) vec4<f32> {
	let edgeDistance = min(
		input.barycentric.x,
		min(input.barycentric.y, input.barycentric.z),
	);
	let fill = smoothstep(0.0, max(fwidth(edgeDistance) * 2.5, 1.0e-5), edgeDistance);
	let outlineColor = vec3<f32>(0.98, 0.78, 0.36);
	let pebbleColor = vec3<f32>(0.42, 0.34, 0.24);
	return vec4<f32>(mix(outlineColor, pebbleColor, fill), 1.0);
}

struct SurfaceVertexOutput {
	@builtin(position) position: vec4<f32>,
	@location(0) uv: vec2<f32>,
}

@vertex
fn surfaceVertexMain(@builtin(vertex_index) vertexIndex: u32) -> SurfaceVertexOutput {
	let positions = array<vec2<f32>, 3>(
		vec2<f32>(-1.0, -1.0),
		vec2<f32>(3.0, -1.0),
		vec2<f32>(-1.0, 3.0),
	);
	let clipPosition = positions[vertexIndex];
	var output: SurfaceVertexOutput;
	output.position = vec4<f32>(clipPosition, 0.0, 1.0);
	output.uv = vec2<f32>(
		0.5 * (clipPosition.x + 1.0),
		0.5 * (1.0 - clipPosition.y),
	);
	return output;
}

fn surfaceFieldAt(coordinate: vec2<u32>) -> vec2<f32> {
	return surfaceFields[coordinate.y * surfaceParams.grid.x + coordinate.x];
}

fn sampleSurfaceField(uv: vec2<f32>) -> vec2<f32> {
	let worldPosition = uv * surfaceParams.domainAndCell.xy;
	let gridPosition = worldPosition / surfaceParams.domainAndCell.z + vec2<f32>(1.0, 1.0);
	let maximumCoordinate = surfaceParams.grid.xy - vec2<u32>(1u, 1u);
	let clampedPosition = clamp(
		gridPosition,
		vec2<f32>(0.0, 0.0),
		vec2<f32>(maximumCoordinate),
	);
	let lower = vec2<u32>(floor(clampedPosition));
	let upper = min(lower + vec2<u32>(1u, 1u), maximumCoordinate);
	let fraction = clampedPosition - vec2<f32>(lower);
	let lowerRow = mix(surfaceFieldAt(lower), surfaceFieldAt(vec2<u32>(upper.x, lower.y)), fraction.x);
	let upperRow = mix(surfaceFieldAt(vec2<u32>(lower.x, upper.y)), surfaceFieldAt(upper), fraction.x);
	return mix(lowerRow, upperRow, fraction.y);
}

@fragment
fn surfaceFragmentMain(input: SurfaceVertexOutput) -> @location(0) vec4<f32> {
	let occupancy = sampleSurfaceField(input.uv);
	// Surface fields are display-scaled; the physical phase fields remain separate.
	let waterArea = max(occupancy.x, 0.0);
	let oilArea = max(occupancy.y, 0.0);
	let totalArea = waterArea + oilArea;
	let totalCoverage = smoothstep(0.115, 0.195, totalArea);
	if (totalCoverage <= 1.0e-5) {
		return vec4<f32>(0.035, 0.047, 0.063, 1.0);
	}

	let waterColor = vec3<f32>(0.12, 0.48, 0.88);
	let oilColor = vec3<f32>(0.91, 0.62, 0.18);
	let waterShare = waterArea / max(totalArea, 1.0e-5);
	let dominantPhaseColor = mix(oilColor, waterColor, smoothstep(0.35, 0.65, waterShare));
	let interfaceHighlight =
		(1.0 - smoothstep(0.06, 0.24, abs(waterShare - 0.5))) * 0.18;
	let materialColor = mix(
		dominantPhaseColor,
		vec3<f32>(0.96, 0.89, 0.68),
		interfaceHighlight,
	);
	let opacity = clamp(totalCoverage, 0.0, 1.0);
	let background = vec3<f32>(0.035, 0.047, 0.063);
	return vec4<f32>(mix(background, materialColor, opacity), 1.0);
}
`;
