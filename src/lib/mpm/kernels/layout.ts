import type {
	MpmMaterialParameters,
	MpmParticle,
	MpmSolidBond,
	MpmStepConfig,
	MpmVector2
} from '../reference/solver.js';
import { DEFAULT_BONDED_TRIANGLE_STIFFNESS } from '../reference/solver.js';
import { validateMpmSolidBondGraph } from '../reference/solid-bonds.js';
import {
	encodeMpmConstitutiveModel,
	normalizeMpmConstitutiveModel,
	SNOW_PLASTICITY
} from '../materials.js';

export type MpmMlsGpuConfig = Omit<MpmStepConfig, 'dt' | 'gravity'> & {
	readonly capacity?: number;
};

export interface MpmMlsGpuDeviceLimits {
	readonly maxBufferSize: number;
	readonly maxStorageBufferBindingSize: number;
	readonly maxUniformBufferBindingSize: number;
	readonly maxComputeWorkgroupsPerDimension: number;
	readonly maxStorageBuffersPerShaderStage: number;
}

export interface MpmMlsGpuBufferLayout {
	readonly nodeWidth: number;
	readonly nodeHeight: number;
	readonly nodeCount: number;
	readonly pairCount: number;
	readonly particleCapacity: number;
	readonly particleStateBytes: number;
	readonly particleBufferBytes: number;
	readonly preparedAffineBytes: number;
	readonly objectIdBufferBytes: number;
	readonly bucketHeadBytes: number;
	readonly bucketNextBytes: number;
	readonly gridStateBytes: number;
	readonly triangleFieldBytes: number;
	readonly triangleFieldCandidateBodyLimit: number;
	readonly triangleFieldCandidateTileCount: number;
	readonly triangleFieldCandidateWordsPerTile: number;
	readonly triangleFieldCandidateWordCount: number;
	readonly triangleFieldCandidateBytes: number;
	readonly bondedBodyStartBytes: number;
	readonly triangleContactStateBytes: number;
	readonly trianglePairContactCacheCapacity: number;
	readonly trianglePairContactCacheBytes: number;
	readonly trianglePairContactCandidateBytes: number;
	readonly bondedBodyCapacity: number;
	readonly fractureBondBufferBytes: number;
	readonly fractureBondTopologyBufferBytes: number;
	readonly surfaceFieldBytes: number;
	readonly readbackBytes: number;
}

export interface MpmMlsGpuObjectRange {
	readonly objectId: number;
	readonly start: number;
	readonly count: number;
}

export interface ValidatedMpmMlsGpuInput {
	readonly config: MpmMlsGpuConfig;
	readonly particleCount: number;
	readonly particleCapacity: number;
	readonly packedParticles: Float32Array;
	readonly packedObjectIds: Uint32Array;
	readonly bondedBodyStarts: Uint32Array;
	readonly bondedBodyCountsByObjectId: ReadonlyMap<number, number>;
	readonly packedFractureBonds: ArrayBuffer;
	readonly packedFractureBondTopology: Uint32Array;
	readonly fractureBondCountsByObjectId: ReadonlyMap<number, number>;
	readonly fractureBondCount: number;
	readonly fractureBondIncidenceCount: number;
	readonly objectRanges: readonly MpmMlsGpuObjectRange[];
	readonly domainWidth: number;
	readonly domainHeight: number;
	readonly domainWidthF32: number;
	readonly domainHeightF32: number;
	readonly cellSizeF32: number;
	readonly particleMassF32: number;
	readonly particleVolumeF32: number;
	readonly bondRestLengthF32: number;
	readonly bondStiffnessF32: number;
	readonly bondReferenceDtF32: number | null;
	readonly lambdaF32: number;
	readonly muF32: number;
	readonly gradientScale: number;
	readonly gradientScaleF32: number;
}

export const PARTICLE_FLOATS = 20;
export const PARTICLE_STRIDE_BYTES = PARTICLE_FLOATS * Float32Array.BYTES_PER_ELEMENT;
export const PREPARED_AFFINE_STRIDE_BYTES = 4 * Float32Array.BYTES_PER_ELEMENT;
export const OBJECT_ID_STRIDE_BYTES = Uint32Array.BYTES_PER_ELEMENT;
export const SIMULATION_PARAMS_BYTES = 128;
export const RENDER_PARAMS_BYTES = 16;
export const SURFACE_RENDER_PARAMS_BYTES = 32;
export const COMPUTE_WORKGROUP_SIZE = 64;
export const NUMERICAL_STATUS_WORDS = 8;
export const NUMERICAL_STATUS_BYTES =
	NUMERICAL_STATUS_WORDS * Uint32Array.BYTES_PER_ELEMENT;

const UINT32_MAX = 0xffff_ffff;
const INT32_MAX = 0x7fff_ffff;
const MAX_FLOAT32 = 3.4028234663852886e38;
const GRID_NODE_STRIDE_BYTES = 80;
const TRIANGLE_FIELD_NODE_STRIDE_BYTES = 32;
const TRIANGLE_FIELD_CANDIDATE_BODY_LIMIT = 512;
const TRIANGLE_FIELD_CANDIDATE_TILE_SIZE = 16;
const TRIANGLE_FIELD_CANDIDATE_WORD_BITS = 32;
const TRIANGLE_FIELD_CANDIDATE_WORD_BYTES = Uint32Array.BYTES_PER_ELEMENT;
const MAX_TRIANGLE_PAIR_CONTACT_CACHE_BODIES = 512;
const TRIANGLE_PAIR_CONTACT_CACHE_ENTRY_BYTES = 4 * Float32Array.BYTES_PER_ELEMENT;
const TRIANGLE_PAIR_CONTACT_CANDIDATE_WORD_BITS = 32;

export function validateMpmMlsGpuInput(
	particles: readonly MpmParticle[],
	config: MpmMlsGpuConfig
): ValidatedMpmMlsGpuInput {
	const rawConfig = assertRecord(config, 'config');
	const rawGrid = assertRecord(rawConfig.grid, 'grid');
	const material = validateMaterial(rawConfig.material, 'material');

	const cellsX = finiteNumber(rawGrid.cellsX, 'grid.cellsX');
	const cellsY = finiteNumber(rawGrid.cellsY, 'grid.cellsY');
	const cellSize = finiteNumber(rawGrid.cellSize, 'grid.cellSize');
	const particleMass = finiteNumber(rawConfig.particleMass, 'particleMass');
	const particleVolume = finiteNumber(rawConfig.particleVolume, 'particleVolume');
	const bondRestLength =
		rawConfig.bondRestLength === undefined
			? undefined
			: finiteNumber(rawConfig.bondRestLength, 'bondRestLength');
	const bondStiffness =
		rawConfig.bondStiffness === undefined
			? DEFAULT_BONDED_TRIANGLE_STIFFNESS
			: finiteNumber(rawConfig.bondStiffness, 'bondStiffness');
	const bondReferenceDt =
		rawConfig.bondReferenceDt === undefined
			? null
			: finiteNumber(rawConfig.bondReferenceDt, 'bondReferenceDt');
	const lambda = material.lambda;
	const mu = material.mu;

	if (!Number.isSafeInteger(cellsX) || cellsX < 1 || cellsX + 3 > UINT32_MAX) {
		throw new RangeError('grid.cellsX must be a positive safe integer that fits the GPU layout');
	}
	if (!Number.isSafeInteger(cellsY) || cellsY < 1 || cellsY + 3 > UINT32_MAX) {
		throw new RangeError('grid.cellsY must be a positive safe integer that fits the GPU layout');
	}
	if (cellSize <= 0) throw new RangeError('grid.cellSize must be greater than zero');
	if (particleMass <= 0) throw new RangeError('particleMass must be greater than zero');
	if (particleVolume <= 0) throw new RangeError('particleVolume must be greater than zero');
	if (bondRestLength !== undefined && bondRestLength <= 0) {
		throw new RangeError('bondRestLength must be greater than zero');
	}
	if (bondStiffness <= 0 || bondStiffness > 1) {
		throw new RangeError('bondStiffness must be greater than zero and at most one');
	}
	if (bondReferenceDt !== null && bondReferenceDt <= 0) {
		throw new RangeError('bondReferenceDt must be greater than zero');
	}
	if (lambda < 0) throw new RangeError('material.lambda must be non-negative');
	if (mu <= 0) throw new RangeError('material.mu must be greater than zero');

	const domainWidth = cellsX * cellSize;
	const domainHeight = cellsY * cellSize;
	const cellSizeSquared = cellSize * cellSize;
	const gradientScale = 4 / cellSizeSquared;
	if (!Number.isFinite(domainWidth) || !Number.isFinite(domainHeight)) {
		throw new RangeError('grid dimensions produce a non-finite domain');
	}
	if (
		!Number.isFinite((cellsX + 1) * cellSize) ||
		!Number.isFinite((cellsY + 1) * cellSize)
	) {
		throw new RangeError('grid dimensions produce non-finite halo coordinates');
	}
	if (
		!Number.isFinite(cellSizeSquared) ||
		cellSizeSquared <= 0 ||
		!Number.isFinite(gradientScale)
	) {
		throw new RangeError('grid.cellSize is outside the supported numeric range');
	}
	if (
		!Number.isFinite(particleMass / particleVolume) ||
		particleMass / particleVolume <= 0 ||
		!Number.isFinite(lambda + 2 * mu)
	) {
		throw new RangeError('mass and material parameters are outside the supported numeric range');
	}

	const cellSizeF32 = finiteFloat32(cellSize, 'grid.cellSize', true);
	const particleMassF32 = finiteFloat32(particleMass, 'particleMass', true);
	const particleVolumeF32 = finiteFloat32(particleVolume, 'particleVolume', true);
	const bondRestLengthF32 =
		bondRestLength === undefined
			? 0
			: finiteFloat32(bondRestLength, 'bondRestLength', true);
	const bondStiffnessF32 = finiteFloat32(
		bondStiffness,
		'bondStiffness',
		true
	);
	const bondReferenceDtF32 =
		bondReferenceDt === null
			? null
			: finiteFloat32(bondReferenceDt, 'bondReferenceDt', true);
	const lambdaF32 = finiteFloat32(lambda, 'material.lambda');
	const muF32 = finiteFloat32(
		mu,
		'material.mu',
		material.model !== 'water' && material.model !== 'oil'
	);
	const gradientScaleF32 = finiteFloat32(gradientScale, 'grid gradient scale', true);
	const domainWidthF32 = finiteFloat32(
		cellsX * cellSizeF32,
		'GPU grid width',
		true
	);
	const domainHeightF32 = finiteFloat32(
		cellsY * cellSizeF32,
		'GPU grid height',
		true
	);

	if (!Array.isArray(particles)) {
		throw new TypeError('particles must be an array');
	}
	const particleCount = particles.length;
	const particleCapacity =
		rawConfig.capacity === undefined
			? particleCount
			: positiveSafeInteger(rawConfig.capacity, 'capacity');
	if (particleCapacity < particleCount || particleCapacity > UINT32_MAX) {
		throw new RangeError('capacity must be at least the initial particle count and fit the GPU layout');
	}
	const pairCount = safeProduct(particleCapacity, 9, 'particle stencil entry count');
	if (pairCount > INT32_MAX) {
		throw new RangeError('particle stencil entries exceed the linked-list index range');
	}
	safeProduct(
		particleCapacity,
		PARTICLE_STRIDE_BYTES,
		'particle state buffer size'
	);
	safeProduct(particleCapacity, PARTICLE_FLOATS, 'particle state element count');

	let packedParticles: Float32Array;
	let packedObjectIds: Uint32Array;
	let packedParticleWords: Uint32Array;
	try {
		packedParticles = new Float32Array(particleCount * PARTICLE_FLOATS);
		packedParticleWords = new Uint32Array(packedParticles.buffer);
		packedObjectIds = new Uint32Array(particleCount);
	} catch (cause) {
		throw new RangeError('initial particle state exceeds the supported typed-array size', {
			cause
		});
	}

	const objectRanges: MpmMlsGpuObjectRange[] = [];
	const closedObjectIds = new Set<number>();
	for (let index = 0; index < particleCount; index++) {
		const rawParticle = assertRecord(particles[index], `particles[${index}]`);
		const objectId = validateObjectId(rawParticle.objectId, `particles[${index}].objectId`);
		const position = readVector(rawParticle.position, `particles[${index}].position`);
		const velocity = readVector(rawParticle.velocity, `particles[${index}].velocity`);
		const C = readMatrix(rawParticle.C, `particles[${index}].C`);
		const F = readMatrix(rawParticle.F, `particles[${index}].F`);
		const particleMaterial =
			rawParticle.material === undefined
				? material
				: validateMaterial(rawParticle.material, `particles[${index}].material`);
		const bondedTriangleVertex = rawParticle.bondedTriangleVertex;
		if (
			bondedTriangleVertex !== undefined &&
			(typeof bondedTriangleVertex !== 'number' ||
				!Number.isSafeInteger(bondedTriangleVertex) ||
				bondedTriangleVertex < 0 ||
				bondedTriangleVertex > 2)
		) {
			throw new RangeError(
				`particles[${index}].bondedTriangleVertex must be 0, 1, or 2`
			);
		}
		const plasticVolume =
			rawParticle.plasticVolume === undefined
				? 1
				: finiteNumber(rawParticle.plasticVolume, `particles[${index}].plasticVolume`);
		const particleVolume =
			rawParticle.volume === undefined
				? particleVolumeF32
				: finiteNumber(rawParticle.volume, `particles[${index}].volume`);
		const damage =
			rawParticle.damage === undefined
				? 0
				: finiteNumber(rawParticle.damage, `particles[${index}].damage`);
		if (particleVolume <= 0) {
			throw new RangeError(`particles[${index}].volume must be greater than zero`);
		}
		if (damage < 0 || damage > 1) {
			throw new RangeError(`particles[${index}].damage must be between zero and one`);
		}
		if (
			rawParticle.damage !== undefined &&
			particleMaterial.model !== 'neo-hookean'
		) {
			throw new RangeError(
				`particles[${index}].damage is only valid for neo-hookean solids`
			);
		}
		if (
			plasticVolume < SNOW_PLASTICITY.minPlasticVolume ||
			plasticVolume > SNOW_PLASTICITY.maxPlasticVolume
		) {
			throw new RangeError(
				`particles[${index}].plasticVolume must be between ${SNOW_PLASTICITY.minPlasticVolume} and ${SNOW_PLASTICITY.maxPlasticVolume}`
			);
		}

		if (
			position.x < 0 ||
			position.x > domainWidth ||
			position.y < 0 ||
			position.y > domainHeight
		) {
			throw new RangeError(`particles[${index}].position must be inside the grid domain`);
		}
		const determinant = F.m00 * F.m11 - F.m01 * F.m10;
		if (!Number.isFinite(determinant) || determinant <= 0) {
			throw new RangeError(
				`particles[${index}].F must have a finite positive determinant`
			);
		}

		const isLiquid =
			particleMaterial.model === 'water' || particleMaterial.model === 'oil';
		const density = isLiquid ? particleMaterial.density : 0;
		const frictionCoefficient =
			particleMaterial.model === 'powder'
				? particleMaterial.frictionCoefficient
				: particleMaterial.model === 'neo-hookean'
					? damage
					: 0;
		const packedVolume = finiteFloat32(
			particleVolume,
			`particles[${index}].volume`,
			true
		);
		const packedDensity = finiteFloat32(
			density,
			`particles[${index}].material.density`,
			isLiquid
		);
		const packedMass = isLiquid
			? Math.fround(packedDensity * packedVolume)
			: particleMassF32;
		if (!Number.isFinite(packedMass) || packedMass <= 0) {
			throw new RangeError(`particles[${index}] mass is outside the supported float32 range`);
		}
		const offset = index * PARTICLE_FLOATS;
		const packed = [
			finiteFloat32(position.x, `particles[${index}].position.x`),
			finiteFloat32(position.y, `particles[${index}].position.y`),
			finiteFloat32(velocity.x, `particles[${index}].velocity.x`),
			finiteFloat32(velocity.y, `particles[${index}].velocity.y`),
			finiteFloat32(C.m00, `particles[${index}].C.m00`),
			finiteFloat32(C.m01, `particles[${index}].C.m01`),
			finiteFloat32(C.m10, `particles[${index}].C.m10`),
			finiteFloat32(C.m11, `particles[${index}].C.m11`),
			finiteFloat32(F.m00, `particles[${index}].F.m00`),
			finiteFloat32(F.m01, `particles[${index}].F.m01`),
			finiteFloat32(F.m10, `particles[${index}].F.m10`),
			finiteFloat32(F.m11, `particles[${index}].F.m11`),
			finiteFloat32(plasticVolume, `particles[${index}].plasticVolume`, true),
			0,
			finiteFloat32(particleMaterial.lambda, `particles[${index}].material.lambda`),
			finiteFloat32(
				particleMaterial.mu,
				`particles[${index}].material.mu`,
				!isLiquid
			),
			packedDensity,
			packedVolume,
			finiteFloat32(
				frictionCoefficient,
				`particles[${index}].material.frictionCoefficient`
			),
			0
		];

		const packedDeterminant = Math.fround(
			Math.fround(packed[8] * packed[11]) -
				Math.fround(packed[9] * packed[10])
		);
		if (!Number.isFinite(packedDeterminant) || packedDeterminant <= 0) {
			throw new RangeError(
				`particles[${index}].F must retain a finite positive determinant in float32`
			);
		}
		if (
			packed[0] < 0 ||
			packed[0] > domainWidthF32 ||
			packed[1] < 0 ||
			packed[1] > domainHeightF32
		) {
			throw new RangeError(
				`particles[${index}].position is outside the float32 GPU grid domain`
			);
		}
		packedParticles.set(packed, offset);
		packedParticleWords[offset + 13] = encodeMpmConstitutiveModel(
			particleMaterial.model ?? 'neo-hookean'
		);
		packedParticleWords[offset + 19] =
			bondedTriangleVertex === undefined ? 0 : bondedTriangleVertex + 1;
		packedObjectIds[index] = objectId;

		const currentRange = objectRanges.at(-1);
		if (currentRange?.objectId === objectId) {
			objectRanges[objectRanges.length - 1] = {
				...currentRange,
				count: currentRange.count + 1
			};
		} else {
			if (closedObjectIds.has(objectId)) {
				throw new RangeError(
					`particles with objectId ${objectId} must be contiguous for GPU cloning`
				);
			}
			if (currentRange) closedObjectIds.add(currentRange.objectId);
			objectRanges.push({ objectId, start: index, count: 1 });
		}
	}

	const bondedBodyStarts: number[] = [];
	const bondedBodyCountsByObjectId = new Map<number, number>();
	for (const range of objectRanges) {
		const objectParticles = particles.slice(range.start, range.start + range.count);
		if (
			!objectParticles.some(
				(particle) => particle.bondedTriangleVertex !== undefined
			)
		) {
			continue;
		}
		if (bondRestLength === undefined) {
			throw new RangeError(
				`object ${range.objectId} uses bonded triangles but bondRestLength is missing`
			);
		}
		if (range.count % 3 !== 0) {
			throw new RangeError(
				`bonded object ${range.objectId} must contain complete three-particle triangles`
			);
		}
		for (let localIndex = 0; localIndex < range.count; localIndex += 1) {
			const particle = objectParticles[localIndex];
			const expectedVertex = localIndex % 3;
			if (particle.bondedTriangleVertex !== expectedVertex) {
				throw new RangeError(
					`bonded object ${range.objectId} must store contiguous vertices 0, 1, and 2`
				);
			}
			const model = particle.material?.model ?? material.model;
			if (model === 'water' || model === 'oil') {
				throw new RangeError('bonded triangle vertices must use a solid material');
			}
			if (expectedVertex === 0) bondedBodyStarts.push(range.start + localIndex);
		}
		bondedBodyCountsByObjectId.set(range.objectId, range.count / 3);
	}
	if (bondedBodyStarts.length > 0 && bondRestLengthF32 <= 0) {
		throw new RangeError('bondRestLength must remain positive in float32');
	}

	const fractureBonds = config.fractureBonds ?? [];
	validateMpmSolidBondGraph(particles, fractureBonds, material);
	const fractureBondIncidenceCount = safeProduct(
		fractureBonds.length,
		2,
		'fracture bond incidence count'
	);
	if (fractureBonds.length > UINT32_MAX || fractureBondIncidenceCount > UINT32_MAX) {
		throw new RangeError('fracture bond topology exceeds the GPU index range');
	}
	const packedFractureBonds = new ArrayBuffer(
		safeProduct(fractureBonds.length, 24, 'fracture bond state buffer size')
	);
	const packedFractureBondWords = new Uint32Array(packedFractureBonds);
	const packedFractureBondFloats = new Float32Array(packedFractureBonds);
	const fractureBondCountsByObjectId = new Map<number, number>();
	const fractureBondOffsets =
		fractureBonds.length === 0 ? new Uint32Array(0) : new Uint32Array(particleCount + 1);
	for (let bondIndex = 0; bondIndex < fractureBonds.length; bondIndex += 1) {
		const bond = fractureBonds[bondIndex];
		const first = bond.firstParticle;
		const second = bond.secondParticle;
		const objectId = particles[first].objectId ?? 0;
		fractureBondCountsByObjectId.set(
			objectId,
			(fractureBondCountsByObjectId.get(objectId) ?? 0) + 1
		);
		fractureBondOffsets[first + 1] += 1;
		fractureBondOffsets[second + 1] += 1;
		const wordOffset = bondIndex * 6;
		packedFractureBondWords[wordOffset] = first;
		packedFractureBondWords[wordOffset + 1] = second;
		packedFractureBondFloats[wordOffset + 2] = finiteFloat32(
			bond.restLength,
			`fractureBonds[${bondIndex}].restLength`,
			true
		);
		packedFractureBondFloats[wordOffset + 3] = finiteFloat32(
			bond.failureStrain,
			`fractureBonds[${bondIndex}].failureStrain`,
			true
		);
		packedFractureBondFloats[wordOffset + 4] = finiteFloat32(
			bond.broken ? 1 : (bond.damage ?? 0),
			`fractureBonds[${bondIndex}].damage`
		);
		packedFractureBondWords[wordOffset + 5] = bond.broken ? 1 : 0;
	}
	const packedFractureBondTopology =
		fractureBonds.length === 0
			? new Uint32Array(0)
			: new Uint32Array(particleCount + 1 + fractureBondIncidenceCount);
	if (fractureBonds.length > 0) {
		for (let index = 1; index < fractureBondOffsets.length; index += 1) {
			fractureBondOffsets[index] += fractureBondOffsets[index - 1];
		}
		packedFractureBondTopology.set(fractureBondOffsets);
		const cursors = fractureBondOffsets.slice(0, particleCount);
		for (let bondIndex = 0; bondIndex < fractureBonds.length; bondIndex += 1) {
			const bond = fractureBonds[bondIndex];
			const incidenceStart = particleCount + 1;
			packedFractureBondTopology[incidenceStart + cursors[bond.firstParticle]++] =
				bondIndex;
			packedFractureBondTopology[incidenceStart + cursors[bond.secondParticle]++] =
				bondIndex;
		}
	}

	return {
		config: {
			grid: { cellsX, cellsY, cellSize },
			particleMass,
			particleVolume,
			...(bondRestLength === undefined ? {} : { bondRestLength }),
			bondStiffness,
			...(bondReferenceDt === null ? {} : { bondReferenceDt }),
			...(config.fractureBonds === undefined ? {} : { fractureBonds }),
			material,
			capacity: particleCapacity
		},
		particleCount,
		particleCapacity,
		packedParticles,
		packedObjectIds,
		bondedBodyStarts: new Uint32Array(bondedBodyStarts),
		bondedBodyCountsByObjectId,
		packedFractureBonds,
		packedFractureBondTopology,
		fractureBondCountsByObjectId,
		fractureBondCount: fractureBonds.length,
		fractureBondIncidenceCount,
		objectRanges,
		domainWidth,
		domainHeight,
		domainWidthF32,
		domainHeightF32,
		cellSizeF32,
		particleMassF32,
		particleVolumeF32,
		bondRestLengthF32,
		bondStiffnessF32,
		bondReferenceDtF32,
		lambdaF32,
		muF32,
		gradientScale,
		gradientScaleF32
	};
}

export function calculateMpmMlsGpuBufferLayout(
	state: Pick<
		ValidatedMpmMlsGpuInput,
		| 'config'
		| 'particleCount'
		| 'particleCapacity'
		| 'bondedBodyStarts'
		| 'fractureBondCount'
		| 'fractureBondIncidenceCount'
	>,
	limits: MpmMlsGpuDeviceLimits
): MpmMlsGpuBufferLayout {
	const { cellsX, cellsY } = state.config.grid;
	const nodeWidth = cellsX + 3;
	const nodeHeight = cellsY + 3;
	const nodeCount = safeProduct(nodeWidth, nodeHeight, 'halo grid node count');
	if (state.particleCount > state.particleCapacity) {
		throw new RangeError('initial particle count must not exceed particle capacity');
	}
	const pairCount = safeProduct(state.particleCapacity, 9, 'particle stencil entry count');

	if (nodeWidth > UINT32_MAX || nodeHeight > UINT32_MAX || nodeCount > UINT32_MAX) {
		throw new RangeError('halo grid dimensions exceed the GPU index range');
	}
	if (state.particleCapacity > UINT32_MAX || pairCount > INT32_MAX) {
		throw new RangeError('particle stencil entries exceed the linked-list index range');
	}

	const particleStateBytes = safeProduct(
		state.particleCount,
		PARTICLE_STRIDE_BYTES,
		'particle state buffer size'
	);
	const capacityStateBytes = safeProduct(
		state.particleCapacity,
		PARTICLE_STRIDE_BYTES,
		'particle state capacity size'
	);
	const particleBufferBytes = Math.max(PARTICLE_STRIDE_BYTES, capacityStateBytes);
	const preparedAffineBytes = Math.max(
		PREPARED_AFFINE_STRIDE_BYTES,
		safeProduct(state.particleCapacity, PREPARED_AFFINE_STRIDE_BYTES, 'prepared affine buffer size')
	);
	const objectIdBufferBytes = Math.max(
		OBJECT_ID_STRIDE_BYTES,
		safeProduct(state.particleCapacity, OBJECT_ID_STRIDE_BYTES, 'particle object id buffer size')
	);
	const bucketHeadBytes = safeProduct(nodeCount, Int32Array.BYTES_PER_ELEMENT, 'bucket head size');
	const bucketNextBytes = Math.max(
		Int32Array.BYTES_PER_ELEMENT,
		safeProduct(pairCount, Int32Array.BYTES_PER_ELEMENT, 'bucket link size')
	);
	const gridStateBytes = safeProduct(
		nodeCount,
		GRID_NODE_STRIDE_BYTES,
		'grid state buffer size'
	);
	const triangleFieldBytes = safeProduct(
		nodeCount,
		TRIANGLE_FIELD_NODE_STRIDE_BYTES,
		'triangle contact field size'
	);
	const triangleFieldCandidateTileColumns = Math.ceil(
		nodeWidth / TRIANGLE_FIELD_CANDIDATE_TILE_SIZE
	);
	const triangleFieldCandidateTileRows = Math.ceil(
		nodeHeight / TRIANGLE_FIELD_CANDIDATE_TILE_SIZE
	);
	const triangleFieldCandidateTileCount = safeProduct(
		triangleFieldCandidateTileColumns,
		triangleFieldCandidateTileRows,
		'triangle field candidate tile count'
	);
	const triangleFieldCandidateWordsPerTile = Math.ceil(
		Math.min(state.bondedBodyStarts.length, TRIANGLE_FIELD_CANDIDATE_BODY_LIMIT) /
			TRIANGLE_FIELD_CANDIDATE_WORD_BITS
	);
	const triangleFieldCandidateWordCount = safeProduct(
		triangleFieldCandidateTileCount,
		triangleFieldCandidateWordsPerTile,
		'triangle field candidate bitset word count'
	);
	const triangleFieldCandidateBytes = Math.max(
		TRIANGLE_FIELD_CANDIDATE_WORD_BYTES,
		safeProduct(
			triangleFieldCandidateWordCount,
			TRIANGLE_FIELD_CANDIDATE_WORD_BYTES,
			'triangle field candidate bitset buffer size'
		)
	);
	// With both halo dimensions >= 4, 16 words per 16x16 tile fit within nodeCount.
	if (triangleFieldCandidateWordCount > nodeCount) {
		throw new RangeError(
			'triangle field candidate dispatch exceeds the halo grid node count'
		);
	}
	const bondedBodyCapacity = Math.max(1, Math.floor(state.particleCapacity / 3));
	const bondedBodyStartBytes = safeProduct(
		bondedBodyCapacity,
		Uint32Array.BYTES_PER_ELEMENT,
		'bonded triangle body index buffer size'
	);
	const triangleContactStateBytes = safeProduct(
		bondedBodyCapacity,
		4 * Float32Array.BYTES_PER_ELEMENT,
		'triangle-pair contact state buffer size'
	);
	const trianglePairContactCacheCapacity =
		state.bondedBodyStarts.length === 0
			? 0
			: Math.min(bondedBodyCapacity, MAX_TRIANGLE_PAIR_CONTACT_CACHE_BODIES);
	const trianglePairContactCacheEntryCount = safeProduct(
		trianglePairContactCacheCapacity,
		trianglePairContactCacheCapacity,
		'triangle-pair contact cache entry count'
	);
	const trianglePairContactCacheBytes = Math.max(
		TRIANGLE_PAIR_CONTACT_CACHE_ENTRY_BYTES,
		safeProduct(
			trianglePairContactCacheEntryCount,
			TRIANGLE_PAIR_CONTACT_CACHE_ENTRY_BYTES,
			'triangle-pair contact cache buffer size'
		)
	);
	const trianglePairContactCandidateWordsPerBody = Math.ceil(
		trianglePairContactCacheCapacity / TRIANGLE_PAIR_CONTACT_CANDIDATE_WORD_BITS
	);
	const trianglePairContactCandidateWordCount = safeProduct(
		trianglePairContactCacheCapacity,
		trianglePairContactCandidateWordsPerBody,
		'triangle-pair contact candidate bitset entry count'
	);
	const trianglePairContactCandidateBytes = Math.max(
		Uint32Array.BYTES_PER_ELEMENT,
		safeProduct(
			trianglePairContactCandidateWordCount,
			Uint32Array.BYTES_PER_ELEMENT,
			'triangle-pair contact candidate bitset buffer size'
		)
	);
	const fractureBondBufferBytes = Math.max(
		24,
		safeProduct(state.fractureBondCount, 24, 'fracture bond state buffer size')
	);
	const fractureBondTopologyBufferBytes =
		state.fractureBondCount === 0
			? Uint32Array.BYTES_PER_ELEMENT
			: safeProduct(
					safeSum(
						safeSum(state.particleCount, 1, 'fracture bond offset count'),
						state.fractureBondIncidenceCount,
						'fracture bond topology element count'
					),
					Uint32Array.BYTES_PER_ELEMENT,
					'fracture bond topology buffer size'
				);
	const surfaceFieldBytes = safeProduct(
		nodeCount,
		2 * Float32Array.BYTES_PER_ELEMENT,
		'water/oil surface field buffer size'
	);
	const readbackBytes = Math.max(
		NUMERICAL_STATUS_BYTES,
		safeSum(
			safeSum(
				capacityStateBytes,
				objectIdBufferBytes,
				'particle and object id readback size'
			),
			NUMERICAL_STATUS_BYTES,
			'readback buffer size'
		)
	);

	assertLimit(limits.maxBufferSize, 'maxBufferSize');
	assertLimit(limits.maxStorageBufferBindingSize, 'maxStorageBufferBindingSize');
	assertLimit(limits.maxUniformBufferBindingSize, 'maxUniformBufferBindingSize');
	assertLimit(limits.maxComputeWorkgroupsPerDimension, 'maxComputeWorkgroupsPerDimension');
	assertLimit(limits.maxStorageBuffersPerShaderStage, 'maxStorageBuffersPerShaderStage');
	if (limits.maxUniformBufferBindingSize < SIMULATION_PARAMS_BYTES) {
		throw new RangeError('GPU maxUniformBufferBindingSize is too small for simulation parameters');
	}
	if (limits.maxUniformBufferBindingSize < SURFACE_RENDER_PARAMS_BYTES) {
		throw new RangeError('GPU maxUniformBufferBindingSize is too small for surface render parameters');
	}
	if (limits.maxStorageBuffersPerShaderStage < 8) {
		throw new RangeError('GPU supports fewer than eight storage buffers per shader stage');
	}

	const buffers = [
		['particle buffers', particleBufferBytes],
		['prepared affine buffer', preparedAffineBytes],
		['particle object ids', objectIdBufferBytes],
		['bucket heads', bucketHeadBytes],
		['bucket links', bucketNextBytes],
		['grid state', gridStateBytes],
		['triangle contact field', triangleFieldBytes],
		['triangle field candidate bitset', triangleFieldCandidateBytes],
		['bonded triangle body indices', bondedBodyStartBytes],
		['triangle-pair contact states', triangleContactStateBytes],
		['triangle-pair contact cache', trianglePairContactCacheBytes],
		['triangle-pair contact candidate bitset', trianglePairContactCandidateBytes],
		['fracture bond state', fractureBondBufferBytes],
		['fracture bond topology', fractureBondTopologyBufferBytes],
		['water/oil surface field', surfaceFieldBytes],
		['simulation parameters', SIMULATION_PARAMS_BYTES],
		['render parameters', RENDER_PARAMS_BYTES],
		['surface render parameters', SURFACE_RENDER_PARAMS_BYTES],
		['numerical status', NUMERICAL_STATUS_BYTES],
		['particle readback', readbackBytes]
	] as const;
	for (const [name, size] of buffers) {
		if (size > limits.maxBufferSize) {
			throw new RangeError(`${name} requires ${size} bytes, beyond GPU maxBufferSize`);
		}
	}

	const storageBuffers = [
		['particle buffers', particleBufferBytes],
		['prepared affine buffer', preparedAffineBytes],
		['particle object ids', objectIdBufferBytes],
		['bucket heads', bucketHeadBytes],
		['bucket links', bucketNextBytes],
		['grid state', gridStateBytes],
		['triangle contact field', triangleFieldBytes],
		['triangle field candidate bitset', triangleFieldCandidateBytes],
		['bonded triangle body indices', bondedBodyStartBytes],
		['triangle-pair contact states', triangleContactStateBytes],
		['triangle-pair contact cache', trianglePairContactCacheBytes],
		['triangle-pair contact candidate bitset', trianglePairContactCandidateBytes],
		['fracture bond state', fractureBondBufferBytes],
		['fracture bond topology', fractureBondTopologyBufferBytes],
		['water/oil surface field', surfaceFieldBytes],
		['numerical status', NUMERICAL_STATUS_BYTES]
	] as const;
	for (const [name, size] of storageBuffers) {
		if (size > limits.maxStorageBufferBindingSize) {
			throw new RangeError(
				`${name} requires ${size} bytes, beyond GPU maxStorageBufferBindingSize`
			);
		}
	}

	const maxWorkgroups = limits.maxComputeWorkgroupsPerDimension;
	for (const [name, count] of [
		['grid nodes', nodeCount],
		['triangle field candidate words', triangleFieldCandidateWordCount],
		['particles', state.particleCapacity]
	] as const) {
		const workgroups = Math.ceil(count / COMPUTE_WORKGROUP_SIZE);
		if (workgroups > maxWorkgroups) {
			throw new RangeError(`${name} exceed GPU maxComputeWorkgroupsPerDimension`);
		}
	}

	return {
		nodeWidth,
		nodeHeight,
		nodeCount,
		pairCount,
		particleCapacity: state.particleCapacity,
		particleStateBytes,
		particleBufferBytes,
		preparedAffineBytes,
		objectIdBufferBytes,
		bucketHeadBytes,
		bucketNextBytes,
		gridStateBytes,
		triangleFieldBytes,
		triangleFieldCandidateBodyLimit: TRIANGLE_FIELD_CANDIDATE_BODY_LIMIT,
		triangleFieldCandidateTileCount,
		triangleFieldCandidateWordsPerTile,
		triangleFieldCandidateWordCount,
		triangleFieldCandidateBytes,
		bondedBodyStartBytes,
		triangleContactStateBytes,
		trianglePairContactCacheCapacity,
		trianglePairContactCacheBytes,
		trianglePairContactCandidateBytes,
		bondedBodyCapacity,
		fractureBondBufferBytes,
		fractureBondTopologyBufferBytes,
		surfaceFieldBytes,
		readbackBytes
	};
}

export function packMpmMlsGpuStepParameters(
	state: ValidatedMpmMlsGpuInput,
	layout: MpmMlsGpuBufferLayout,
	dt: number,
	gravity: MpmVector2,
	particleCount = state.particleCount,
	bondedBodyCount = state.bondedBodyStarts.length,
	cflGuard:
		| {
				readonly waveSpeedBound: number;
				readonly velocityBound: number;
		  }
		| null = null,
	bondReferenceDt?: number
): ArrayBuffer {
	if (!Number.isSafeInteger(particleCount) || particleCount < 0 || particleCount > layout.particleCapacity) {
		throw new RangeError('particleCount must be an integer within the allocated GPU capacity');
	}
	if (
		!Number.isSafeInteger(bondedBodyCount) ||
		bondedBodyCount < 0 ||
		bondedBodyCount > layout.bondedBodyCapacity
	) {
		throw new RangeError('bondedBodyCount must fit the allocated triangle body capacity');
	}
	const pairCount = safeProduct(particleCount, 9, 'active particle stencil entry count');
	if (pairCount > INT32_MAX) {
		throw new RangeError('active particle stencil entries exceed the linked-list index range');
	}
	const stepDt = finiteNumber(dt, 'dt');
	if (stepDt <= 0) throw new RangeError('dt must be greater than zero');
	const stepDtF32 = finiteFloat32(stepDt, 'dt', true);
	const acceleration = readVector(gravity, 'gravity');
	const gravityX = finiteFloat32(acceleration.x, 'gravity.x');
	const gravityY = finiteFloat32(acceleration.y, 'gravity.y');
	const cflWaveSpeedBound =
		cflGuard === null
			? 0
			: finiteFloat32(cflGuard.waveSpeedBound, 'cflGuard.waveSpeedBound');
	const cflVelocityBound =
		cflGuard === null
			? 0
			: finiteFloat32(cflGuard.velocityBound, 'cflGuard.velocityBound');
	const bondReferenceDtF32 = finiteFloat32(
		bondReferenceDt ?? state.bondReferenceDtF32 ?? stepDtF32,
		'bondReferenceDt',
		true
	);
	const stressScale = -stepDt * state.config.particleVolume * state.gradientScale;
	if (!Number.isFinite(stressScale)) {
		throw new RangeError('dt, particleVolume, and grid.cellSize produce a non-finite stress scale');
	}
	const stressScaleF32 = finiteFloat32(stressScale, 'stress scale');

	const result = new ArrayBuffer(SIMULATION_PARAMS_BYTES);
	const integers = new Uint32Array(result, 0, 8);
	integers.set([
		state.config.grid.cellsX,
		state.config.grid.cellsY,
		layout.nodeWidth,
		layout.nodeHeight,
		particleCount,
		layout.nodeCount,
		pairCount,
		bondedBodyCount
	]);
	const floats = new Float32Array(result, 32, 20);
	floats.set([
		state.cellSizeF32,
		state.particleMassF32,
		state.particleVolumeF32,
		stepDtF32,
		state.lambdaF32,
		state.muF32,
		stressScaleF32,
		state.gradientScaleF32,
		gravityX,
		gravityY,
		state.domainWidthF32,
		state.domainHeightF32,
		state.bondRestLengthF32,
		state.bondStiffnessF32,
		state.cellSizeF32 * 2,
		Math.sqrt(state.particleVolumeF32 / Math.PI) +
			state.cellSizeF32 * 0.75,
		cflWaveSpeedBound,
		cflVelocityBound,
		cflGuard === null ? 0 : 1,
		bondReferenceDtF32
	]);
	new Uint32Array(result, 112, 4).set([
		state.fractureBondCount === 0 ? 0 : state.particleCount,
		state.fractureBondCount,
		state.fractureBondIncidenceCount,
		bondedBodyCount > 0 &&
		bondedBodyCount <= layout.trianglePairContactCacheCapacity
			? 1
			: 0
	]);
	return result;
}

function positiveSafeInteger(value: unknown, name: string): number {
	if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 1) {
		throw new RangeError(`${name} must be a positive safe integer`);
	}
	return value;
}

function validateObjectId(value: unknown, name: string): number {
	if (value === undefined) return 0;
	if (
		typeof value !== 'number' ||
		!Number.isSafeInteger(value) ||
		value < 0 ||
		value > UINT32_MAX
	) {
		throw new RangeError(`${name} must be an unsigned 32-bit integer`);
	}
	return value;
}

function validateMaterial(value: unknown, name: string): MpmMaterialParameters {
	const record = assertRecord(value, name);
	const lambda = finiteNumber(record.lambda, `${name}.lambda`);
	const mu = finiteNumber(record.mu, `${name}.mu`);
	const model = normalizeMpmConstitutiveModel(record.model, `${name}.model`);
	if (!Number.isFinite(lambda + 2 * mu)) {
		throw new RangeError(`${name} parameters are outside the supported numeric range`);
	}
	if (model === 'water' || model === 'oil') {
		const density = finiteNumber(record.density, `${name}.density`);
		if (lambda <= 0) throw new RangeError(`${name}.lambda must be greater than zero`);
		if (mu !== 0) throw new RangeError(`${name}.mu must be zero for a liquid material`);
		if (record.frictionCoefficient !== undefined) {
			throw new RangeError(`${name}.frictionCoefficient is only valid for powder`);
		}
		if (density <= 0) throw new RangeError(`${name}.density must be greater than zero`);
		finiteFloat32(lambda, `${name}.lambda`, true);
		finiteFloat32(mu, `${name}.mu`);
		finiteFloat32(density, `${name}.density`, true);
		return { lambda, mu: 0, model, density };
	}
	if (model === 'powder') {
		const frictionCoefficient = finiteNumber(
			record.frictionCoefficient,
			`${name}.frictionCoefficient`
		);
		if (lambda < 0) throw new RangeError(`${name}.lambda must be non-negative`);
		if (mu <= 0) throw new RangeError(`${name}.mu must be greater than zero`);
		if (frictionCoefficient < 0) {
			throw new RangeError(`${name}.frictionCoefficient must be non-negative`);
		}
		if (record.density !== undefined) {
			throw new RangeError(`${name}.density is only valid for water or oil`);
		}
		finiteFloat32(lambda, `${name}.lambda`);
		finiteFloat32(mu, `${name}.mu`, true);
		finiteFloat32(frictionCoefficient, `${name}.frictionCoefficient`);
		return { lambda, mu, model, frictionCoefficient };
	}
	if (record.density !== undefined || record.frictionCoefficient !== undefined) {
		throw new RangeError(`${name} has parameters that do not match its constitutive model`);
	}
	if (lambda < 0) throw new RangeError(`${name}.lambda must be non-negative`);
	if (mu <= 0) throw new RangeError(`${name}.mu must be greater than zero`);
	finiteFloat32(lambda, `${name}.lambda`);
	finiteFloat32(mu, `${name}.mu`, true);
	return { lambda, mu, model };
}

function readVector(value: unknown, name: string): MpmVector2 {
	const record = assertRecord(value, name);
	return {
		x: finiteNumber(record.x, `${name}.x`),
		y: finiteNumber(record.y, `${name}.y`)
	};
}

function readMatrix(
	value: unknown,
	name: string
): { readonly m00: number; readonly m01: number; readonly m10: number; readonly m11: number } {
	const record = assertRecord(value, name);
	return {
		m00: finiteNumber(record.m00, `${name}.m00`),
		m01: finiteNumber(record.m01, `${name}.m01`),
		m10: finiteNumber(record.m10, `${name}.m10`),
		m11: finiteNumber(record.m11, `${name}.m11`)
	};
}

function assertRecord(value: unknown, name: string): Record<string, unknown> {
	if (typeof value !== 'object' || value === null) {
		throw new TypeError(`${name} must be an object`);
	}
	return value as Record<string, unknown>;
}

function finiteNumber(value: unknown, name: string): number {
	if (typeof value !== 'number' || !Number.isFinite(value)) {
		throw new TypeError(`${name} must be a finite number`);
	}
	return value;
}

function finiteFloat32(value: number, name: string, positive = false): number {
	const result = Math.fround(value);
	if (!Number.isFinite(result) || Math.abs(value) > MAX_FLOAT32) {
		throw new RangeError(`${name} is outside the supported float32 range`);
	}
	if (value !== 0 && result === 0) {
		throw new RangeError(`${name} underflows the supported float32 range`);
	}
	if (positive && result <= 0) {
		throw new RangeError(`${name} must remain positive in the supported float32 range`);
	}
	return result;
}

function safeProduct(left: number, right: number, name: string): number {
	const result = left * right;
	if (!Number.isSafeInteger(result) || result < 0) {
		throw new RangeError(`${name} exceeds the safe GPU buffer size range`);
	}
	return result;
}

function safeSum(left: number, right: number, name: string): number {
	const result = left + right;
	if (!Number.isSafeInteger(result) || result < 0) {
		throw new RangeError(`${name} exceeds the safe GPU buffer size range`);
	}
	return result;
}

function assertLimit(value: number, name: string): void {
	if (!Number.isSafeInteger(value) || value < 1) {
		throw new RangeError(`GPU ${name} is invalid`);
	}
}
