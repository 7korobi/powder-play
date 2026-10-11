export interface GridSize {
	width: number;
	height: number;
}

const MIN_CELLS = 6_000;
const MAX_CELLS = 60_000;
const FRAME_TIME_TOLERANCE_MS = 0.5;

export function isRenderFrameDue(
	timestamp: number,
	lastRenderedAt: number | null,
	targetFps: number
): boolean {
	if (!Number.isFinite(timestamp)) throw new RangeError('Frame timestamps must be finite.');
	if (lastRenderedAt !== null && !Number.isFinite(lastRenderedAt)) {
		throw new RangeError('The previous frame timestamp must be finite.');
	}
	if (!Number.isFinite(targetFps) || targetFps <= 0) {
		throw new RangeError('Target fps must be a positive number.');
	}
	return (
		lastRenderedAt === null ||
		timestamp - lastRenderedAt >= 1000 / targetFps - FRAME_TIME_TOLERANCE_MS
	);
}

export function chooseInitialGridSize(
	viewportWidth: number,
	viewportHeight: number,
	hardwareConcurrency = 4
): GridSize {
	const aspect = safeAspect(viewportWidth, viewportHeight);
	const cores = Number.isFinite(hardwareConcurrency) ? Math.max(1, hardwareConcurrency) : 4;
	const budget = Math.min(MAX_CELLS, Math.max(12_000, 12_000 + Math.floor(cores) * 1_700));
	return sizeForBudget(aspect, budget);
}

export class AdaptiveResolutionController {
	private sampleCount = 0;
	private totalDuration = 0;
	private fastWindows = 0;

	constructor(private readonly targetFps = 30) {
		if (!Number.isFinite(targetFps) || targetFps <= 0) {
			throw new RangeError('Target fps must be a positive number.');
		}
	}

	observe(stepDurationMs: number, current: GridSize): GridSize | null {
		if (!Number.isFinite(stepDurationMs) || stepDurationMs < 0) {
			throw new RangeError('Step duration must be a non-negative number.');
		}
		this.sampleCount++;
		this.totalDuration += stepDurationMs;
		if (this.sampleCount < this.targetFps) return null;

		const averageDuration = this.totalDuration / this.sampleCount;
		this.sampleCount = 0;
		this.totalDuration = 0;
		const currentCells = current.width * current.height;
		const frameBudget = 1000 / this.targetFps;

		if (averageDuration > frameBudget * 0.65 && currentCells > MIN_CELLS) {
			this.fastWindows = 0;
			return sizeForBudget(current.width / current.height, Math.max(MIN_CELLS, currentCells * 0.78));
		}
		if (averageDuration < frameBudget * 0.22 && currentCells < MAX_CELLS) {
			this.fastWindows++;
			if (this.fastWindows >= 6) {
				this.fastWindows = 0;
				return sizeForBudget(current.width / current.height, Math.min(MAX_CELLS, currentCells * 1.12));
			}
		} else {
			this.fastWindows = 0;
		}
		return null;
	}
}

function sizeForBudget(aspect: number, cellBudget: number): GridSize {
	const safe = safeAspect(aspect, 1);
	const cells = Math.min(MAX_CELLS, Math.max(MIN_CELLS, Math.round(cellBudget)));
	const width = Math.max(40, Math.round(Math.sqrt(cells * safe)));
	const height = Math.max(40, Math.round(cells / width));
	return { width, height };
}

function safeAspect(width: number, height: number): number {
	if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
		return 390 / 844;
	}
	return Math.min(3, Math.max(0.35, width / height));
}
