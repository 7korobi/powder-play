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
`GATE4_*` source dimensions and counts. The source MPM scene has a 2,200 ×
1,500 Gate 4 unit domain (220 × 150 cells, 10 units per cell). The LiquidFun
domain is expanded horizontally to 2,800 × 1,500 Gate 4 units to fit its wider
fluid grids, with at least one source particle spacing between their right edge
and the wall. The visible LiquidFun domain is 140 × 75 mm.

Physical Gate 4 positions use 20,000 units/m. Box2D runs in internal units at
1,000 units/m (one internal unit represents 1 mm), so one Gate 4 unit maps to
0.05 Box2D units. This keeps the 1 mm particle radius and 1.5 mm spring-link
spacing large relative to Box2D's fixed length tolerance. Internal Box2D coordinates
and physical meters are deliberately distinct. The LiquidFun scene regrids its
fluid particles to fit the specified 140 × 75 mm visible domain; the CPU/GPU
MPM source scene is unchanged.

Four static polygon walls have inner faces on the domain edges and extend 50 mm
(50 Box2D units) outward. They overlap at the corners by 2 mm (two fluid
particle radii), rather than extending the full wall depth along both axes.
The previous multi-meter fixtures allowed fluid particles to penetrate the
walls. Their fixtures are not drawn, and particle/body rendering is clipped to
the visible domain. A 50 mm wall contains particles at the current forcing and
fixed timestep, but no finite wall guarantees containment for arbitrarily
large per-step displacements.

Gate 4 coordinates are y-down; Box2D engine coordinates are y-up:

- `box2d.x = gate4.x × 0.05`
- `box2d.y = (1500 - gate4.y) × 0.05`
- Box2D velocities use the same `0.05` position scale
- `physical gravity (m/s²) = Gate 4 gravity / 100`
- `box2d gravity (internal units/s²) = Gate 4 gravity × 10`
- Gate 4 gravity `(0, 90 units/s²)` is `0.9 m/s²` physically and `(0, -900)`
  in Box2D engine coordinates
- A control value of `980` is approximately `9.8 m/s²`, or `1G`
- Every step uses `dt = 1/240 s`; the benchmark does not change the seed,
  timestep, or source scene counts.

The source MPM scene has 3,674 nodes: 2,000 water, 1,500 oil, 150 pebble
vertices, and 24 fracture nodes. The LiquidFun workbench uses the seed for
fluid velocity samples and the fracture network, but its material workload is
not a one-to-one mapping of those source nodes. It creates 800 particles for
each of water, oil, ink, and soil (3,200 fluid particles), ten elastic balls
with 12 particles each (120), and ten spring vines with 48 particles each
(480), for 3,800 LiquidFun particles total. The 24 fracture nodes are separate
Box2D bodies and are not included in that particle count.

Water uses the default `b2_waterParticle` flag; oil uses
`b2_viscousParticle`; ink uses `b2_powderParticle`; and soil combines
`b2_powderParticle` with `b2_viscousParticle`. These are flags in the **same**
LiquidFun particle system, not four isolated fluids: every type shares a 1 mm
radius, 0.01 simulation mass units per particle, and particle contacts with
the other types. Color distinguishes the requested material in the renderer;
it does not isolate solver interactions. LiquidFun sets density at the
particle-system level. At the 1.5 mm stride, the shared density is 0.004444
simulation mass units per square Box2D unit. These are simulation mass units,
not calibrated kilograms; LiquidFun is two-dimensional, so its density is per
area rather than volumetric.

LiquidFun represents fluid with radius-bearing particles and neighbor pressure
and contact solves; triangular fluid elements are not required. The
application sets the shared particle-system `pressureStrength` to 1.00,
twenty times the installed `liquidfun-wasm@7.0.0` default of 0.05. This is a
solver coefficient, not a physical pressure unit or a per-material setting.
The increased value is a tuning experiment; check volume retention,
oscillation, and stability under the target forcing rather than treating it as
a calibrated material parameter.

The four fluid grids each contain 800 particles across 43 rows, using a 30
Gate 4 unit stride (1.5 mm, 75% of the 2 mm particle diameter). Their x
origins are 1.5, 30.5, 59.5, and 88.5 mm; all start at y = 10 mm. The maximum
row width is 19 particles (27 mm between first and last centers), and the last
row is at y = 73 mm, so particle centers plus their 1 mm radii remain within
the 140 × 75 mm visible domain. Water and ink reuse sampled water velocities;
oil and soil reuse sampled oil velocities. Source rows are sampled and
horizontal velocities repeated. The source Gate 4 scene and its 2,200 × 1,500
domain used by the CPU and GPU MLS-MPM pages remain unchanged.

The ten elastic balls are independent LiquidFun particle groups with
`b2_elasticParticle`; each contains 12 particles arranged on a 1.5 mm grid
with the four corners omitted. `elasticStrength` is explicitly set to
LiquidFun's default value of 0.25. The balls share the particle system's
1 mm radius and 0.01 particle mass, so each has a nominal particle mass of
0.12 simulation mass units. Their group boundaries are not rigid fixtures,
and their mass is not calibrated to a physical ball.

The ten spring vines are also independent particle groups, each using
`b2_springParticle` for 48 particles in a vertical chain. Adjacent particles
are 1.5 mm apart, giving 47 intervals and a 70.5 mm center-to-center span,
close to the requested 70 mm. Their x centers are 2.2 mm apart, just beyond
the 2 mm particle diameter. `springStrength` is set to LiquidFun's default
0.25. At initialization, each vine is expected to form 47 spring links
(470 across the ten groups); startup checks this count, and the live page
continues to report the engine's actual `GetPairCount()`.

The retained fracture network has 24 dynamic circle bodies connected by the
same 53 Gate 4 bonds. Those bonds break when tensile strain reaches the Gate 4
12% threshold. Elastic balls and spring vines use LiquidFun's particle
constraints, not Box2D rigid bodies or hand-authored distance joints.

These flag-based materials are a behavioral comparison, not a matched-physics
replacement for MPM. LiquidFun flags do not provide independent mass or radius
per material, and all four fluid types continue to contact one another. Its
elastic-triad and spring-pair solvers do not reproduce MPM deformation
gradients or Neo-Hookean constitutive behavior. Expect different trajectories,
mixing, and breakage timing; the comparison holds a fixed source seed and
timestep, not exact physical response.

## Live LiquidFun lab and MPM references

The live LiquidFun route executes the WASM engine described above and exposes
pause/resume, single-step, reset, manual 2D gravity (each component ranges to
±999 Gate 4 units/s², initially (0, 0)), horizontal oscillation (default
amplitude ±999 Gate 4 units/s² at 0.16 Hz, adjustable up to ±999), requested
simulation speed (default 1×; options 1/16×, 1/8×, 0.25×, 0.5×, 1×, 2×, 4×,
8×, 16×, and 32×), render mode, render-target FPS (default 90), and live
step/backlog metrics. The gravity range corresponds to approximately
±9.99 m/s² (about ±1.02G). The oscillation is added to X gravity and integrated
as the average acceleration over each fixed step using simulation time; setting
its amplitude to zero disables it. The FPS target limits Canvas drawing only;
the accumulator and LiquidFun fixed simulation timestep are unchanged. A
per-frame step count and physics-time budget prevent a high requested speed
from producing an unbounded synchronous catch-up loop. If the browser cannot
keep up, pending simulation time is shown and can be discarded while paused.

In an observation reported by the user, rigid-body collision calculations up
to a manual-gravity input of 250,000 showed no clearly severe visible failure.
With the documented conversion of 100 Gate 4 acceleration units per m/s², that
input is 2,500 m/s² (about 255G), not 25,000 m/s² (about 2,550G). This is a
limited visual observation, not a stability guarantee or safe upper bound.
Leakage was previously reported around an input of 30,000 with the former
0.2 m-thick walls. Horizontal oscillation is a separate input added to X
gravity; isolate it from manual-gravity tests by setting its amplitude to zero.

The CPU and GPU MLS-MPM pages remain a separate same-equation numerical
comparison. LiquidFun is a behavioral reference for contact, waves, settling,
and stability, not an exact trajectory or constitutive-model oracle. Water,
oil, ink, and soil use different flag combinations in one particle system,
so they share its density, radius, 0.01 particle mass, and cross-material
contacts. Elastic balls and spring vines use LiquidFun's particle constraints;
they do not reproduce the deformable MLS-MPM triangle constitutive model.
Fracture nodes use dynamic bodies and breakable distance joints. Retain these
model differences when interpreting visual and timing results.

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
LiquidFun `world.Step()` work, mixed-material particle contacts, elastic-triad
and spring-pair solving, and fracture-bond strain checks. Scene construction,
Canvas2D, UI work, GPU
timestamps, and GPU queue completion/wait are excluded. No GPU or
`navigator.gpu` is used by this benchmark.

The result also records package/engine version, browser user agent,
`hardwareConcurrency`, source hashes, seed and material counts, grid/domain,
scale, gravity, fixed `dt`, iteration counts, warmup parameters, and sample
configuration. The package loader automatically selects its SIMD or non-SIMD
WASM build when supported; results can therefore differ by browser and
hardware. Use the recorded environment and configuration when interpreting
timings; they are not bit-identical or cross-device performance guarantees.
