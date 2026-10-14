# LiquidFun CPU benchmark

This document describes the standalone `benchmark:liquidfun` harness and its
historical LiquidFun implementation. The live `/labs/mpm/liquidfun` route now
runs the CPU MLS-MPM reference solver so it can be compared with
`/labs/mpm/kernels` under the same physical model. See
[MPM CPU/GPU comparison](./MPM-CPU-GPU-COMPARISON.md) for the current setup.

The benchmark harness still uses `liquidfun-wasm@7.0.0`, the LiquidFun
particle API in Birch-san's `box2d-wasm` WebAssembly fork based on Box2D 2.4.1.
It is a standalone CPU benchmark, not a physically equivalent implementation
of MLS-MPM: the solvers, material models, and liquid representations differ.

## Attribution and license files

The application uses the package as installed; it does not substitute a
TypeScript port or a local physics implementation. The package's Zlib license
and its WASM feature-detection Apache 2.0 license are copied unchanged to:

- [Zlib License](../static/licenses/liquidfun-wasm/LICENSE.zlib.txt)
- [WASM feature-detection Apache License 2.0](../static/licenses/liquidfun-wasm/LICENSE.wasm-feature-detect.Apache.txt)

Those files are linked from the page so the notices are available with the
served application.

The package's declaration file names the `box2d-wasm` module while the runtime
package is imported as `liquidfun-wasm`; `liquidfun-wasm.d.ts` supplies a narrow
type contract for the runtime surface used here. It does not replace or emulate
the LiquidFun engine.

## Benchmark workload and coordinate mapping

The benchmark calls `createMpmGate4Scene(0x4d504d34)` and uses the current
`GATE4_*` dimensions and counts. The mapped scene has a 2,200 × 1,500 Gate 4
unit domain (220 × 150 cells, 10 units per cell). With the declared scale of
100 Gate 4 units per meter, the Box2D world is 22 × 15 m.

Gate 4 coordinates are y-down; Box2D coordinates are y-up:

- `box2d.x = gate4.x / 100`
- `box2d.y = (1500 - gate4.y) / 100`
- `box2d.vx = gate4.vx / 100`
- `box2d.vy = -gate4.vy / 100`
- Gate 4 gravity `(0, 90 units/s²)` becomes Box2D gravity `(0, -0.9 m/s²)`
- Every step uses `dt = 1/240 s`; the benchmark does not change the seed,
  timestep, or source scene counts.

The source has 3,674 MPM nodes: 2,000 water, 1,500 oil, 150 pebble vertices,
and 24 fracture nodes. Water uses the default `b2_waterParticle` flag; oil uses
`b2_viscousParticle`. Both are created in the **same** LiquidFun particle
system, with a shared radius and system density.

Each set of three pebble vertices initializes one dynamic Box2D polygon body,
for 50 full-triangle collision shapes. The initial polygon keeps the three
source positions; its center velocity is their mean, its angular velocity is
the least-squares rigid fit to their velocities, and its mass is the sum of the
three source-node masses. Unlike the previous three-circle representation,
the fixture used for collision now covers the same triangle that is drawn.
The 24 fracture nodes remain individual dynamic circle bodies connected by
the same 53 Gate 4 bonds. Those bonds break when tensile strain reaches the
Gate 4 12% threshold.

This is a rigid-body approximation of the MPM pebble material, not a
deformable match: it does not reproduce MPM deformation gradients or
Neo-Hookean constitutive behavior, and Box2D contact/constraint solving differs
from MLS-MPM. One LiquidFun particle system shares density across water and
oil, so it cannot preserve the MPM phase densities 0.02 and 0.016
independently. Expect different trajectories and breakage timing; the
comparison holds the seeded source scene and fixed timestep, not the exact
physical response.

## Current live comparison

The live route no longer loads this LiquidFun simulation. It now uses the CPU
MLS-MPM reference solver and shares the seeded scene, liquid particle-size
control, fixed timestep, CFL policy, gravity, and oscillation with the GPU
page. The GPU route retains optional GPU-only interaction and emitter
experiments; those are outside the default CPU/GPU baseline. The benchmark
below remains available to measure the independent LiquidFun implementation.

## Repeatable simulation-only benchmark

Use a local Vite server and a Chromium DevTools Protocol endpoint on loopback.
For example:

```sh
npm run dev -- --host 127.0.0.1 --port 4181
npm run benchmark:liquidfun -- --origin=http://127.0.0.1:4181 --cdp=http://127.0.0.1:9222
```

The script rejects non-local HTTP origins and non-local CDP WebSocket endpoints.
It creates a dedicated CDP tab for the run and closes only that tab; existing
browser tabs are not navigated or closed. It prints JSON to stdout and accepts
the optional `--output=<path>` argument to save the same JSON. This is local
measurement only; it does not deploy or publish.

Each run creates a fresh, non-rendered scene, then performs:

1. a 600 ms wall-clock stationary warmup using zero gravity (the number of
   completed steps is recorded because the warmup is time-based);
2. 64 fixed-timestep steps with Gate 4 gravity;
3. 12 measured batches, each containing 8 consecutive simulation steps.

`performance.now()` starts immediately before and stops immediately after the
eight synchronous calls. The JSON includes median, p95, min, max, and sample
count for batch CPU milliseconds, plus derived per-step time and steps/second.
The p95 uses the same lower-index quantile convention as
`scripts/benchmark-mpm.mjs`. Scope includes JavaScript-to-WASM call overhead,
LiquidFun/Box2D `world.Step()` work, and fracture-bond strain checks. Scene
construction, Canvas2D, UI work, GPU timestamps, and GPU queue completion/wait
are excluded. No GPU or `navigator.gpu` is used by this benchmark.

The result also records package/engine version, browser user agent,
`hardwareConcurrency`, source hashes, scene seed/counts, grid/domain, scale,
gravity, fixed `dt`, iteration counts, warmup parameters, and sample
configuration. The package loader automatically selects its SIMD or non-SIMD
WASM build when supported; results can therefore differ by browser and
hardware. Use the recorded environment and configuration when interpreting
timings; they are not bit-identical or cross-device performance guarantees.
