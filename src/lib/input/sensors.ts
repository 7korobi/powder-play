import type { Vector } from '../sim/world';

export const STANDARD_GRAVITY = 9.81;
const GRAVITY_NOISE_THRESHOLD = 0.025;

export function screenGravityFromAcceleration(
	ax: number | null,
	ay: number | null,
	orientationAngle: number
): Vector {
	if (ax === null || ay === null || !Number.isFinite(ax) || !Number.isFinite(ay)) {
		return { x: 0, y: 0 };
	}
	const vector = rotateToScreen(-ax, -ay, orientationAngle);
	const gravity = { x: vector.x / STANDARD_GRAVITY, y: vector.y / STANDARD_GRAVITY };
	const magnitude = Math.hypot(gravity.x, gravity.y);
	if (magnitude < GRAVITY_NOISE_THRESHOLD) return { x: 0, y: 0 };

	const scale = Math.min(1, magnitude) / magnitude;
	return { x: gravity.x * scale, y: gravity.y * scale };
}

export function screenAccelerationForce(
	ax: number | null,
	ay: number | null,
	orientationAngle: number
): Vector {
	if (ax === null || ay === null || !Number.isFinite(ax) || !Number.isFinite(ay)) {
		return { x: 0, y: 0 };
	}
	const vector = rotateToScreen(-ax, -ay, orientationAngle);
	return {
		x: clamp(vector.x / STANDARD_GRAVITY, -1, 1),
		y: clamp(vector.y / STANDARD_GRAVITY, -1, 1)
	};
}

export function blendVector(previous: Vector, next: Vector, amount: number): Vector {
	if (!Number.isFinite(amount) || amount < 0 || amount > 1) {
		throw new RangeError('Blend amount must be between 0 and 1.');
	}
	return {
		x: previous.x + (next.x - previous.x) * amount,
		y: previous.y + (next.y - previous.y) * amount
	};
}

function rotateToScreen(x: number, y: number, orientationAngle: number): Vector {
	if (!Number.isFinite(orientationAngle)) return { x, y };
	const radians = (orientationAngle * Math.PI) / 180;
	const cosine = Math.cos(radians);
	const sine = Math.sin(radians);
	return {
		x: x * cosine - y * sine,
		y: -(x * sine + y * cosine)
	};
}

function clamp(value: number, minimum: number, maximum: number): number {
	const result = Math.min(maximum, Math.max(minimum, value));
	return result === 0 ? 0 : result;
}
