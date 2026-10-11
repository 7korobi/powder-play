export type GestureKind = 'camera' | 'dial';
export type InteractionMode = 'select' | 'place' | 'flow';

export const MODE_ORDER: readonly InteractionMode[] = ['select', 'place', 'flow'];

const ROTATION_THRESHOLD = (16 * Math.PI) / 180;
const PINCH_THRESHOLD = 0.08;
const PAN_THRESHOLD = 12;
const MODE_STEP = (2 * Math.PI) / MODE_ORDER.length;

export function classifyTwoFingerGesture(
	rotationRadians: number,
	pinchChange: number,
	panDistance: number
): GestureKind | null {
	const rotationScore = Math.abs(rotationRadians) / ROTATION_THRESHOLD;
	const pinchScore = Math.abs(pinchChange) / PINCH_THRESHOLD;
	const panScore = Math.abs(panDistance) / PAN_THRESHOLD;
	const strongest = Math.max(rotationScore, pinchScore, panScore);
	if (strongest < 1) return null;
	return rotationScore >= Math.max(pinchScore, panScore) ? 'dial' : 'camera';
}

export function modeForDialRotation(
	startingMode: InteractionMode,
	rotationRadians: number
): InteractionMode {
	const startIndex = MODE_ORDER.indexOf(startingMode);
	const offset = Math.round(rotationRadians / MODE_STEP);
	const nextIndex = (startIndex + offset + MODE_ORDER.length * 10) % MODE_ORDER.length;
	return MODE_ORDER[nextIndex];
}

export function shortestAngleDelta(previous: number, current: number): number {
	let difference = current - previous;
	while (difference > Math.PI) difference -= Math.PI * 2;
	while (difference < -Math.PI) difference += Math.PI * 2;
	return difference;
}
