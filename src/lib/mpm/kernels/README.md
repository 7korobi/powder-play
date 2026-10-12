# GPU MLS-MPM kernel

`solver.ts` exports `createMpmMlsGpuSolver(canvas, particles, config)`. Creation
is asynchronous because it requests the WebGPU session and builds the compute
and render pipelines. `config` contains the CPU reference's `grid`,
`particleMass`, `particleVolume`, and default `material`; particles may override
their material and reference volume individually. Water and oil particle mass
is their reference density multiplied by reference volume. Time step and gravity are supplied to
`step(dt, gravity)`. The caller owns wall-clock stepping and
pause/resume. `render()` reads the current particle buffer directly on the GPU.
CPU particle data is returned only when `readParticles()` is explicitly called;
an optional list of active particle indices reads only those records and their
object IDs. `checkNumericalStatus()` reads only the small persistent numerical
status buffer and rejects if a GPU step was rolled back. The lab calls it about
once per second after queue completion; it does not read particle state.
`measureQueueCompletionMs()` measures how long the submitted GPU queue takes
to complete; it is not a GPU-only execution-time measurement.
Positions, grid spacing, time, gravity, mass, volume, and material parameters use
the same units and conventions as `reference/README.md`.
The solver owns the canvas context and device obtained from the shared WebGPU
session; call `dispose()` when the lab is finished. `step(dt, gravity)` queues
one step. `stepBatch(stepCount, dt, gravity)` encodes sequential fixed steps
into one command buffer and submits them together; each step keeps its normal
grid passes and particle-buffer ping-pong.

The Gate 4 lab keeps the solver step fixed at `1 / 240` seconds, targets 60 FPS,
and starts at a 2× simulation-time multiplier. The UI offers 0.5×–32× without
increasing the per-step `dt`. At 32× and 60 FPS this requires 128 solver steps
per rendered frame; they are encoded as one simulation batch before rendering.
The 128-step frame cap bounds work after long stalls; excess accumulated time
is discarded rather than queued indefinitely, and the lab reports both the
effective simulation speed and discarded time. Slower devices may fall short
of the selected multiplier.

The Gate 4 sample starts with 8,000 water, 6,000 oil, and 200 Neo-Hookean
elastic pebbles built from three particles each (14,600 active particles
total). The pebbles form a 20 × 10 array with 20-world-unit center spacing.
Each is an equilateral triangle with 11-world-unit edges and width, about
0.5% of the simulation-domain width. Each triangle spans roughly 1.1 grid
cells and has its own `objectId`; this does not provide explicit contact, so
mutual penetration and sustained elastic recovery are not validated. Each
pebble uses lambda 20 and mu 10.

Water and oil reference densities are 0.02 and 0.016, half the previous
values. Their reference volumes and EOS stiffnesses are unchanged. The liquid
particle spacing is 9 world units instead of 8, reducing initial liquid areal
density by about 21%. The grid is 220 × 150 cells at the same 10-unit cell
size; grid-wide passes dispatch 34,119 rather than 49,959 nodes, about 31.7%
fewer. A previous 20,000-particle scene used 6,000 powder particles; the
initial powder patch has now been replaced by 200 elastic pebble objects. The sample
still reserves capacity for 1,000,000 particles. The smaller domain keeps the
liquid objects and all pebbles inside the eight-cell wall cushion.

The lab applies a horizontal sinusoidal acceleration with default amplitude
±240 world units/s² and frequency 0.16 Hz. The UI can disable or adjust its
amplitude up to 360. To preserve batching, each batch applies the time-average
acceleration over its simulation-time interval; the simulation clock advances
only after a batch is submitted.

## Interaction and capacity

`config.capacity` reserves particle and object-id buffers during initialization.
If omitted, capacity equals the initial particle count. `particleCount` reports
the current active count and `particleCapacity` reports the fixed allocation.
Cloning never reallocates those buffers; it appends a GPU-side copy of the
selected contiguous `objectId` range and returns `false` when capacity is full.
`objectId` is optional on input particles and defaults to `0`; an object's
particles must be contiguous in the initial array. Each particle carries its
constitutive model and material parameters through GPU stepping, readback, and
cloning. The GPU state also preserves liquid density, particle volume, powder
friction, and the Snow model's plastic-volume history.

`pick()` runs a GPU nearest-particle search and reads back only a 16-byte
selection result. `beginDrag()`, `updateDrag()`, and `endDrag()` update a small
uniform; the next grid update applies a damped, local force around the pointer.
Drag and clone do not read back the particle array. Full particle state remains
available only through explicit `readParticles()` calls. Compute dispatches use
the active particle count rather than the reserved capacity.

The compute path has ordered clear, particle-prepare, linked-list insertion,
P2G gather, grid update, G2P, and particle-commit dispatches. A simulation batch
encodes them in one compute pass, with dispatch ordering and particle-buffer
ping-pong preserved. Constitutive stress and affine momentum are evaluated once
per active particle in `prepareParticles`, rather than up to nine times in P2G.
A 16-byte-per-capacity affine scratch buffer is recomputed each step; cached
values never survive into the next step without preparation. P2G requires six
storage bindings, checked against device limits. Every particle inserts its nine
quadratic B-spline stencil entries into per-node linked lists using integer
atomics; one invocation gathers each node's list, so no floating-point atomics
or global grid-by-particle scan is used. Grid storage adds one halo node beyond
each side of the physical `[0, cellsX] × [0, cellsY]` node range. Outward grid
velocity tapers across up to eight cells at each wall (scaled down on small
grids) before post-G2P particle clipping; this cushions impacts and matches the
CPU reference.

The solver supports per-particle compressible 2D Neo-Hookean elasticity,
Snow-style corotated plasticity, and separate water, oil, and powder models.
The Gate 4 scene uses water, oil, and 200 three-particle Neo-Hookean pebbles;
powder remains an optional experimental constitutive model, not part of this
initial sample.
Water and oil use a tension-free, weakly-compressible bulk equation of state;
their reference densities affect particle mass. Liquid deformation is
projected to its volume-preserving polar rotation, discarding shear that does
not contribute to fluid stress. Powder uses a
Drucker-Prager-like pressure-dependent shear cap in log-stretch space, and its
area ratio is floored at 0.5 after G2P. This prevents extreme powder collapse
but is not particle contact or plastic compaction. These are exploratory
constitutive models, not calibrated real-material models.
The Snow model clamps principal stretches to `[0.975, 1.0075]`, tracks plastic
volume in `[0.6, 20]`, and applies bounded exponential hardening.

All materials share one grid velocity field. Water/oil particles additionally
receive an approximate local density-contrast acceleration after G2P, based on
liquid mass and current area deposited to neighboring grid nodes. This makes
denser liquid move along effective gravity relative to lighter liquid, without
adding a second velocity grid. The liquid polar projection bounds `det(F)` to
`[0.5, 2]` to prevent unbounded volume drift in the tension-free equation of
state. There is still no explicit water/oil interface,
surface tension, immiscibility, cross-material contact, powder compaction, or
damage. The phases can still interpenetrate or mix; the deformation-gradient
update alone adaptively substeps large increments, up to 64 substeps; the P2G
and grid-velocity update remain explicit.

## Numerical and runtime limits

- GPU numerical particle and grid state is `f32`; the per-particle material
  selector is `u32`. The CPU oracle uses JavaScript double-precision numbers.
  P2G summation follows the atomic linked-list
  insertion order, which is not deterministic across GPU executions. Expect
  small CPU/GPU differences, especially in dense cells or after many steps.
- A GPU numerical failure sets a persistent device-side status and rolls the
  whole step back to the last valid particle state. `checkNumericalStatus()` or
  the next explicit `readParticles()` rejects with `MpmMlsGpuNumericalError`;
  deformation errors
  include the failing particle, determinant, and strain increment; projection
  errors include the projected matrix. When its requested readback includes
  that particle, the error also reports its last valid position, velocity,
  affine matrix, and deformation gradient. P2G errors distinguish invalid
  material input, non-finite particle values, and node accumulation. Tiny
  positive stencil contributions that round to zero in `f32` are treated as
  negligible. Status is not polled from `step()` or `render()`; the lab checks
  the small status buffer periodically after submitted work completes.
- The solver has no automatic CFL selection. Choose `dt` small enough for the
  material stiffness, grid spacing, and particle travel. Adaptive
  deformation-gradient integration prevents one large strain increment from
  flipping a particle, but does not make an arbitrarily large step stable.
- Buffer sizes and 1D dispatch counts are checked against the acquired device's
  WebGPU limits. Devices that cannot hold the requested state fail setup with
  an explicit error.
- The capacity is fixed for the solver lifetime. Reaching it is an explicit
  clone failure rather than a silent buffer growth or particle drop.
- Particle rendering color-codes the constitutive model and uses a radius
  proportional to `cellSize`. The lab route should size the canvas to match the
  configured world aspect ratio.
- The Gate 4 lab starts with 14,600 mixed-material particles and reserves
  capacity for 1,000,000. The fixed allocation uses about 217.2 MB of GPU
  buffers; an explicit full-state readback can allocate up to 84 MB more.
  The affine scratch adds 16 MB at this capacity; the active particle count
  still controls dispatches.
  This large capacity is intended for PC performance testing and may fail on
  devices without enough GPU memory. Its initial CPU/GPU parity check uses a
  small local neighborhood around one probe per material; the explicit
  paused-state check remains a full readback.

The pure config/layout and batch-encoding tests run without WebGPU. On an Apple
Metal hardware adapter in Chrome 154, the performance benchmark verifies shader
compilation, all five models in a 45-particle eight-step CPU/GPU comparison,
batch-versus-single stepping, picking, force-drag, cloning and capacity,
render submission, whole-step rollback, and persistent numerical status.
This does not validate sustained plastic loading, interfaces, granular
calibration, mobile performance, or long-term visual behavior.

See [performance measurement](../../../../docs/PERFORMANCE-MLS-MPM.md) and
[the tuning roadmap](../../../../docs/ROADMAP-MLS-MPM-PERFORMANCE.md).
With a local source server and a dedicated local Chrome CDP endpoint available,
`npm run benchmark:mpm -- --profile=batch` measures normal batch GPU time;
`--profile=stages` intentionally splits dispatches for kernel attribution.
Both require hardware WebGPU and `timestamp-query`, and fail explicitly when
unavailable. Profiling and small-scene readback are benchmark-only; the
interactive solver does not enable timestamps or add full particle readback.
