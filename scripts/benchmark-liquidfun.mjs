import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';

const args = new Map(
	process.argv.slice(2).map((argument) => {
		const [name, ...value] = argument.replace(/^--/, '').split('=');
		return [name, value.join('=')];
	})
);
for (const name of args.keys()) {
	if (!['origin', 'cdp', 'output'].includes(name)) {
		throw new Error(`Unknown option: --${name}`);
	}
}

function localHttpOrigin(value, name) {
	const url = new URL(value);
	if (
		url.protocol !== 'http:' ||
		!['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname)
	) {
		throw new Error(`${name} must use local HTTP (localhost, 127.0.0.1, or ::1).`);
	}
	return url.origin;
}

const origin = localHttpOrigin(
	args.get('origin') ?? 'http://127.0.0.1:4181',
	'--origin'
);
const cdpOrigin = localHttpOrigin(
	args.get('cdp') ?? 'http://127.0.0.1:9222',
	'--cdp'
);

const packageLock = JSON.parse(
	await readFile(new URL('../package-lock.json', import.meta.url), 'utf8')
);
const packageVersion =
	packageLock.packages?.['node_modules/liquidfun-wasm']?.version;
if (typeof packageVersion !== 'string') {
	throw new Error('liquidfun-wasm version was not found in package-lock.json.');
}

async function benchmarkLiquidFun(packageVersion) {
	const { runLiquidFunBenchmark } = await import(
		'/src/lib/mpm/liquidfun/benchmark.ts'
	);
	return runLiquidFunBenchmark({ packageVersion });
}

async function readSourceHashes() {
	const hashes = {};
	for (const path of [
		'../src/lib/mpm/liquidfun/workload.ts',
		'../src/lib/mpm/liquidfun/simulation.ts',
		'../src/lib/mpm/liquidfun/benchmark.ts',
		'../src/lib/mpm/scenes/gate4.ts',
		'./benchmark-liquidfun.mjs'
	]) {
		const file = new URL(path, import.meta.url);
		hashes[path] = createHash('sha256')
			.update(await readFile(file))
			.digest('hex');
	}
	return hashes;
}

const initialSourceHashes = await readSourceHashes();
const response = await fetch(`${cdpOrigin}/json/version`);
if (!response.ok) {
	throw new Error(`CDP connection failed: ${response.status}`);
}
const version = await response.json();
const socketUrl = new URL(version.webSocketDebuggerUrl);
if (
	!['ws:', 'wss:'].includes(socketUrl.protocol) ||
	!['127.0.0.1', 'localhost', '[::1]'].includes(socketUrl.hostname)
) {
	throw new Error('CDP returned a non-local WebSocket endpoint.');
}

const socket = new WebSocket(socketUrl);
await new Promise((resolve, reject) => {
	socket.addEventListener('open', resolve, { once: true });
	socket.addEventListener('error', reject, { once: true });
});

const pending = new Map();
let nextId = 0;
let targetId;
socket.addEventListener('close', () => {
	for (const callback of pending.values()) {
		callback.reject(new Error('The CDP connection closed during the benchmark.'));
	}
	pending.clear();
});
socket.addEventListener('message', ({ data }) => {
	const message = JSON.parse(data);
	if (message.id === undefined) return;
	const callback = pending.get(message.id);
	if (!callback) return;
	pending.delete(message.id);
	if (message.error) {
		callback.reject(
			new Error(`${callback.method}: ${JSON.stringify(message.error)}`)
		);
	}
	else callback.resolve(message.result);
});

function send(method, params = {}, sessionId) {
	const id = ++nextId;
	const result = new Promise((resolve, reject) => {
		pending.set(id, { method, resolve, reject });
	});
	socket.send(JSON.stringify({ id, method, params, sessionId }));
	return result;
}

try {
	({ targetId } = await send('Target.createTarget', { url: 'about:blank' }));
	const { sessionId } = await send('Target.attachToTarget', {
		targetId,
		flatten: true
	});
	await send('Page.enable', {}, sessionId);
	await send('Runtime.enable', {}, sessionId);
	await send(
		'Page.navigate',
		{ url: `${origin}/@vite/client` },
		sessionId
	);

	let ready = false;
	for (let attempt = 0; attempt < 100; attempt += 1) {
		try {
			const result = await send(
				'Runtime.evaluate',
				{
					expression: `document.readyState === 'complete' && location.origin === ${JSON.stringify(origin)}`,
					returnByValue: true
				},
				sessionId
			);
			if (result.result.value === true) {
				ready = true;
				break;
			}
		} catch (error) {
			if (
				!(error instanceof Error) ||
				!error.message.includes('Inspected target navigated or closed')
			) {
				throw error;
			}
		}
		await new Promise((resolve) => setTimeout(resolve, 100));
	}
	if (!ready) {
		throw new Error('The local source server did not load within 10 seconds.');
	}

	const result = await send(
		'Runtime.evaluate',
		{
			expression: `(${benchmarkLiquidFun.toString()})(${JSON.stringify(packageVersion)})`,
			awaitPromise: true,
			returnByValue: true,
			timeout: 180_000
		},
		sessionId
	);
	if (result.exceptionDetails) {
		throw new Error(
			result.exceptionDetails.exception?.description ??
				JSON.stringify(result.exceptionDetails)
		);
	}

	const sourceHashes = await readSourceHashes();
	if (JSON.stringify(sourceHashes) !== JSON.stringify(initialSourceHashes)) {
		throw new Error(
			'LiquidFun workload source changed while the benchmark was running; repeat the measurement.'
		);
	}
	const output = JSON.stringify(
		{ sourceHashes, ...result.result.value },
		null,
		2
	);
	if (args.has('output')) {
		await writeFile(args.get('output'), `${output}\n`);
	}
	console.log(output);
} finally {
	if (targetId && socket.readyState === WebSocket.OPEN) {
		try {
			await send('Target.closeTarget', { targetId });
		} catch {
			// A browser may already have closed this benchmark-created tab.
		}
	}
	socket.close();
}
