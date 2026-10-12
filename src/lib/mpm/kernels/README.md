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
object IDs. `measureQueueCompletionMs()` measures how long the submitted GPU
queue takes to complete; it is not a GPU-only execution-time measurement.
Positions, grid spacing, time, gravity, mass, volume, and material parameters use
the same units and conventions as `reference/README.md`.
The solver owns the canvas context and device obtained from the shared WebGPU
session; call `dispose()` when the lab is finished.

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

The compute path has separate clear, linked-list insertion, P2G gather, grid
update, G2P, and particle-commit passes. Every particle inserts its nine
quadratic B-spline stencil entries into per-node linked lists using integer
atomics; one invocation gathers each node's list, so no floating-point atomics
or global grid-by-particle scan is used. Grid storage adds one halo node beyond
each side of the physical `[0, cellsX] × [0, cellsY]` node range. Outward grid
velocity tapers across up to eight cells at each wall (scaled down on small
grids) before post-G2P particle clipping; this cushions impacts and matches the
CPU reference.

The solver supports per-particle compressible 2D Neo-Hookean elasticity,
Snow-style corotated plasticity, and separate water, oil, and powder models.
Water and oil use a tension-free, weakly-compressible bulk equation of state;
their reference densities affect particle mass. Liquid deformation is
projected to its volume-preserving polar rotation, discarding shear that does
not contribute to fluid stress. Powder uses a
Drucker-Prager-like pressure-dependent shear cap in log-stretch space. These
are exploratory constitutive models, not calibrated real-material models.
The Snow model clamps principal stretches to `[0.975, 1.0075]`, tracks plastic
volume in `[0.6, 20]`, and applies bounded exponential hardening.

All materials currently share one grid velocity field. There is no explicit
water/oil interface, surface tension, immiscibility, buoyancy separation,
cross-material contact, powder compaction, or damage. The deformation-gradient
update alone adaptively substeps large increments, up to 64 substeps; the P2G
and grid-velocity update remain explicit.

## Numerical and runtime limits

- GPU numerical particle and grid state is `f32`; the per-particle material
  selector is `u32`. The CPU oracle uses JavaScript double-precision numbers.
  P2G summation follows the atomic linked-list
  insertion order, which is not deterministic across GPU executions. Expect
  small CPU/GPU differences, especially in dense cells or after many steps.
- A GPU numerical failure sets a persistent device-side status and rolls the
  whole step back to the last valid particle state. The next explicit
  `readParticles()` rejects with `MpmMlsGpuNumericalError`; deformation errors
  include the failing particle, determinant, and strain increment; projection
  errors include the projected matrix. When its requested readback includes
  that particle, the error also reports its last valid position, velocity,
  affine matrix, and deformation gradient. P2G errors distinguish invalid
  material input, non-finite particle values, and node accumulation. Tiny
  positive stencil contributions that round to zero in `f32` are treated as
  negligible. Status is intentionally not polled from `step()` or `render()`.
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
- The Gate 4 lab starts with 20,000 mixed-material particles and reserves
  capacity for 40,000. Its initial CPU/GPU parity check uses a small local
  neighborhood around one probe per material; the explicit paused-state check
  remains a full readback.

The pure config/layout tests run without WebGPU. Shader compilation, mixed
material GPU-vs-CPU error, plastic deformation under sustained loading,
water/oil interface behavior, powder behavior under sustained loading, picking,
force-drag behavior, cloning, and visual behavior still need validation on a
WebGPU device.
