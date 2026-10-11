export const MLS_MPM_COMPUTE_WGSL = /* wgsl */ `
struct Particle {
	position: vec2<f32>,
	velocity: vec2<f32>,
	c: vec4<f32>,
	f: vec4<f32>,
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
		if (!finiteFloat(determinant) || determinant <= 0.0) {
			atomicOr(&stateError[0], ERROR_DEFORMATION);
			return;
		}
		let inverseTranspose =
			vec4<f32>(particle.f.w, -particle.f.z, -particle.f.y, particle.f.x) /
			determinant;
		let logVolumeChange = log(determinant);
		let pressure = params.material.x * logVolumeChange;
		let firstPiola =
			params.material.y * (particle.f - inverseTranspose) +
			pressure * inverseTranspose;
		let stressTimesFt = vec4<f32>(
			firstPiola.x * particle.f.x + firstPiola.y * particle.f.y,
			firstPiola.x * particle.f.z + firstPiola.y * particle.f.w,
			firstPiola.z * particle.f.x + firstPiola.w * particle.f.y,
			firstPiola.z * particle.f.z + firstPiola.w * particle.f.w,
		);
		let affine =
			params.physical.y * particle.c +
			params.material.z * stressTimesFt;
		let dpos =
			(vec2<f32>(f32(ox), f32(oy)) - fraction) * params.physical.x;
		let particleMomentum = vec2<f32>(
			params.physical.y * particle.velocity.x +
				affine.x * dpos.x +
				affine.y * dpos.y,
			params.physical.y * particle.velocity.y +
				affine.z * dpos.x +
				affine.w * dpos.y,
		);
		if (
			!finiteFloat(weight) ||
			weight < 0.0 ||
			!finiteVec4(inverseTranspose) ||
			!finiteVec4(firstPiola) ||
			!finiteVec4(stressTimesFt) ||
			!finiteVec4(affine) ||
			!finiteVec2(dpos) ||
			!finiteVec2(particleMomentum)
		) {
			atomicOr(&stateError[0], ERROR_P2G);
			return;
		}

		let weightedMass = weight * params.physical.y;
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
	let updatedF = vec4<f32>(
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
	let updatedDeterminant =
		updatedF.x * updatedF.w - updatedF.y * updatedF.z;
	if (
		!finiteVec2(velocity) ||
		!finiteVec4(updatedC) ||
		!finiteVec4(updatedF) ||
		!finiteVec2(updatedPosition)
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
	var updatedParticle: Particle;
	updatedParticle.position = vec2<f32>(boundedX.x, boundedY.x);
	updatedParticle.velocity = vec2<f32>(boundedX.y, boundedY.y);
	updatedParticle.c = updatedC;
	updatedParticle.f = updatedF;
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
}

struct RenderParams {
	domainAndRadius: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: RenderParams;

struct VertexOutput {
	@builtin(position) position: vec4<f32>,
	@location(0) localPosition: vec2<f32>,
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
	return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
	if (dot(input.localPosition, input.localPosition) > 1.0) {
		discard;
	}
	return vec4<f32>(0.92, 0.72, 0.38, 1.0);
}
`;
