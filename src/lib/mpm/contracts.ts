export interface MpmVector {
	x: number;
	y: number;
}

export interface MpmParticleSeed {
	position: MpmVector;
	velocity?: MpmVector;
	materialId: number;
	objectId: number;
}

export interface MpmLabConfig {
	worldWidth: number;
	worldHeight: number;
	capacity: number;
	particles: readonly MpmParticleSeed[];
}

export interface MpmSelection {
	particleIndex: number;
	objectId: number;
	anchor: MpmVector;
}

export type MpmDragMode = 'force' | 'rigid';

export interface MpmBackend {
	readonly deviceLost: Promise<GPUDeviceLostInfo> | null;
	initialize(canvas: HTMLCanvasElement, config: MpmLabConfig): Promise<void>;
	step(deltaSeconds: number, gravity: MpmVector): void;
	render(): void;
	pick(point: MpmVector, radius: number): Promise<MpmSelection | null>;
	beginDrag(selection: MpmSelection, target: MpmVector, mode: MpmDragMode): void;
	updateDrag(target: MpmVector): void;
	endDrag(): void;
	clone(selection: MpmSelection, offset: MpmVector): Promise<boolean>;
	dispose(): void;
}

export interface MpmWebGpuSession {
	adapter: GPUAdapter;
	device: GPUDevice;
	context: GPUCanvasContext;
	format: GPUTextureFormat;
	lost: Promise<GPUDeviceLostInfo>;
	dispose(): void;
}
