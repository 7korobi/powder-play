import type { MpmVector2 } from '../reference/solver.js';
import { meanSinusoidalAcceleration } from '../kernels/oscillation.js';
import { LIQUIDFUN_DT_SECONDS } from './workload.js';

export function calculateLiquidFunStepGravity(
	manualGravity: MpmVector2,
	simulatedTimeSeconds: number,
	oscillationAmplitude: number
): MpmVector2 {
	return {
		x:
			manualGravity.x +
			meanSinusoidalAcceleration(
				simulatedTimeSeconds,
				LIQUIDFUN_DT_SECONDS,
				oscillationAmplitude
			),
		y: manualGravity.y
	};
}
