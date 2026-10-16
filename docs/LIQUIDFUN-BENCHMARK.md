# LiquidFun CPU benchmark

This document describes the live LiquidFun behavior lab at
`/labs/mpm/liquidfun` and the standalone `benchmark:liquidfun` harness. The
interactive page and benchmark use the same `createLiquidFunSimulation()`
implementation; the page adds Canvas2D rendering and live controls, while the
benchmark measures simulation-only batches. The CPU MLS-MPM numerical
reference lives at `/labs/mpm/reference`, and `/labs/mpm/kernels` is the GPU
MLS-MPM implementation. See
[MPM CPU/GPU comparison](./MPM-CPU-GPU-COMPARISON.md) for how to use these
three references together.

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
source center but scales both local axes and drawn dimensions to one half of
the source triangle (66 world units per side rather than 132). Its center
velocity is the vertex mean, and angular velocity is fitted to the scaled
vertices and source velocities. Its mass remains the sum of the three
source-node masses, so the smaller fixture has four times the density. The
fixture used for collision covers the same three vertices that are drawn;
the vertex array is passed through LiquidFun's contiguous `pointsToVec2Array`
adapter before Box2D constructs the polygon, avoiding a mismatched pointer
array or a box-shaped collision proxy.
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

## Live LiquidFun lab and MPM references

The live LiquidFun route executes the WASM engine described above and exposes
pause/resume, single-step, reset, manual 2D gravity (each component ranges to
±10000 Gate 4 units/s², initially (0, 0)), horizontal oscillation (default
amplitude ±1000 Gate 4 units/s² at 0.16 Hz, adjustable up to ±10000), requested
simulation speed (default 0.5×), render mode, render-target FPS (default 90),
and live step/backlog metrics. The oscillation is added to X gravity and
integrated as the average acceleration over each fixed step using simulation
time; setting its amplitude to zero disables it. The FPS target limits Canvas
drawing only; the accumulator and LiquidFun fixed simulation timestep are
unchanged. A per-frame step count and physics-time budget prevent a high
requested speed from producing an unbounded synchronous catch-up loop. If the
browser cannot keep up, pending simulation time is shown and can be discarded
while paused.

The CPU and GPU MLS-MPM pages remain a separate same-equation numerical
comparison. LiquidFun is a behavioral reference for contact, waves, settling,
and stability, not an exact trajectory or constitutive-model oracle. Water
and oil use different LiquidFun particle flags in one particle system, so they
share its density/radius parameters. The 50 pebble triangles are rigid Box2D
bodies, not deformable MLS-MPM triangles; fracture nodes use dynamic bodies
and breakable distance joints. These model differences should be retained
when interpreting the visual and timing results.

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
