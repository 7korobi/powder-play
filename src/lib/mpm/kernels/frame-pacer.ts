export const MPM_RENDER_FPS_OPTIONS = [15, 30, 60, 90, 120] as const;
export type MpmRenderFps = (typeof MPM_RENDER_FPS_OPTIONS)[number];

const FRAME_PACING_TOLERANCE_MS = 0.5;

export class MpmFramePacer {
	private targetFps: MpmRenderFps | null = null;
	private nextFrameTimestampMs: number | null = null;

	shouldRender(timestampMs: number, targetFps: MpmRenderFps): boolean {
		if (!Number.isFinite(timestampMs) || timestampMs < 0) {
			throw new RangeError('timestampMs must be finite and non-negative.');
		}
		if (!MPM_RENDER_FPS_OPTIONS.includes(targetFps)) {
			throw new RangeError('targetFps must be a supported render rate.');
		}

		if (this.targetFps !== targetFps) {
			this.targetFps = targetFps;
			this.nextFrameTimestampMs = timestampMs;
		}

		const frameIntervalMs = 1_000 / targetFps;
		const nextFrameTimestampMs = this.nextFrameTimestampMs;
		if (
			nextFrameTimestampMs !== null &&
			timestampMs + FRAME_PACING_TOLERANCE_MS < nextFrameTimestampMs
		) {
			return false;
		}

		if (nextFrameTimestampMs === null) {
			this.nextFrameTimestampMs = timestampMs + frameIntervalMs;
			return true;
		}

		const elapsedBeyondDeadline = Math.max(0, timestampMs - nextFrameTimestampMs);
		const intervalsToAdvance = Math.floor(elapsedBeyondDeadline / frameIntervalMs) + 1;
		this.nextFrameTimestampMs =
			nextFrameTimestampMs + intervalsToAdvance * frameIntervalMs;
		return true;
	}

	reset(): void {
		this.targetFps = null;
		this.nextFrameTimestampMs = null;
	}
}
