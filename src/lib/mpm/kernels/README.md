# GPU MLS-MPM kernel

`solver.ts` exports `createMpmMlsGpuSolver(canvas, particles, config, cflBounds?)`.
Creation is asynchronous because it requests the WebGPU session and builds the
compute and render pipelines. For Neo-Hookean, water, and oil particles the
solver derives initial wave-speed and velocity bounds from the configured
material, deformation, and actual initial particle velocities. Optional
`cflBounds` may widen those derived initial bounds, but may not lower them.
Snow and powder have no derived acoustic bound and are rejected for CFL
stepping rather than assigned a guessed value. Every CFL microstep checks the
GPU-resident velocities and current constitutive wave-speed bound before
commit; exceeding either records a numerical error and rolls back that
microstep. `config` contains the CPU reference's `grid`,
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
`measureMaximumParticleSpeed()` asynchronously reads one GPU-reduced float
and the small numerical status (36 bytes total), not the particle array.
Positions, grid spacing, time, gravity, mass, volume, and material parameters use
the same units and conventions as `reference/README.md`.
The solver owns the canvas context and device obtained from the shared WebGPU
session; call `dispose()` when the lab is finished. `step(dt, gravity)` queues
one direct step. `stepBatch(stepCount, dt, gravity)` encodes sequential fixed
steps into one command buffer and submits them together. `stepCflBatch()`
recomputes the plan with `calculateMpmCflMicrosteps()`, rejects a mismatched
plan or a total microstep count above the caller's cap, and submits
`macroStepCount * plan.stepCount` full steps at `plan.microDt`. It accepts the
velocity envelope for the whole batch to choose the microstep count. The
runtime guard checks each particle's actual speed against the maximum speed
allowed by that float32 microstep after reserving the derived wave-speed bound.
Every numerical microstep completes the entire solver stage sequence and
particle-buffer ping-pong before the next begins. With bonded triangles and
fracture bonds, that sequence includes clear grid, prepare particles, build
particle buckets, P2G, build the triangle contact field, update grid, G2P,
gather triangle reactions, solve bonded triangles, validate CFL bounds, update
fracture bond damage, apply incident damage to particles, and commit. A batch
still uses one compute pass. A final GPU reduction finds maximum particle
speed; only its scalar and the compact status buffer are asynchronously mapped
on the CPU.

The Gate 4 lab uses a `1 / 240`-second macrostep and starts at a 2×
simulation-time multiplier. The UI offers 0.5×–32×; each macrostep is split
into the complete CFL-planned numerical microsteps. Macrostep rate determines
simulated time and sinusoidal acceleration, while microstep and particle
microstep rates are reported separately. A 512-microstep per-frame cap bounds
work, including after long stalls; if the requested time exceeds that cap, the
lab reports the error and stops instead of clamping or discarding the time.

Gate 4 derives the liquid acoustic bounds from its tension-free EOS
(`c² = lambda / referenceDensity`) and the Neo-Hookean bound from the material
acoustic tensor and each configured initial deformation. The per-batch
velocity envelope starts from the last GPU-reduced maximum speed and includes
the selected gravity, sinusoidal acceleration amplitude, and the bounded
pointer spring over the requested simulated duration. Pointer damping is
non-increasing only while its float32 explicit-Euler factor stays in `[0, 1]`;
the scheduler and solver reject drag-enabled plans that violate this condition.
Constitutive and contact accelerations are not assumed away: the GPU guard
checks the resulting speed against the selected microstep's remaining CFL
velocity allowance and checks the updated acoustic bound after every
microstep. It rolls back a violating microstep and surfaces the numerical
error in the lab status.
This is a guarded, evolving bound, not a claim that the initial speeds are a
lifetime guarantee. The default 1× macro-time scheduling path is exercised by
tests; the lab's existing 2× default remains unchanged. At most two compact
speed/status readbacks may be pending; later frame time remains in the
accumulator until capacity is available, and any resulting work-cap overflow
stops explicitly instead of dropping simulated time.

The Gate 4 sample starts with 8,000 water, 6,000 oil, 200 Neo-Hookean
elastic pebbles built from three particles each, and a 96-particle connected
solid (14,696 active particles total). The pebbles form a 14-column × 15-row array with 68-world-unit center
spacing; the last row is centered and contains four pebbles. Each is an
equilateral triangle with 66-world-unit edges and width, about 3% of the
simulation-domain width. Each triangle spans roughly 6.6 grid cells and has
its own `objectId`. Each pebble uses lambda 20 and mu 10; its three edges also
use position constraints. The GPU solves these constraints in triangle-local
offsets and derives velocity corrections from those small displacements before
reconstructing world positions, avoiding precision loss from subtracting
near-equal large coordinates. Water and oil have separate grid velocity/volume
channels and receive normal contact impulses from their phase-volume gradients.
The triangle signed-distance field projects liquid particles out of the
pebbles and gathers the opposing normal impulse back to the three vertices.
Pebble pairs use triangle SAT contact with a normal impulse and positional
correction. These are discrete, approximate contacts, not a continuous
collider or continuum deformable mesh.
At scene initialization, water and oil velocity components are sampled
independently from ±60 world units/s and each pebble vertex from ±120 world
units/s. A fixed scene seed reproduces the same velocities for tests and
benchmarks; the lab chooses a fresh seed on initialization.
The triangle field scans all bonded bodies for each grid node; the current
200-pebble scene performs up to about 6.82 million node/body candidate checks
per step. This GPU pass and the pairwise SAT solver are profiled in
`docs/PERFORMANCE-MLS-MPM.md`; larger scenes and sustained stability remain
unverified. Contact friction, surface tension, wetting, complete water/oil
separation in arbitrary flows, and swept collision detection are not
implemented.

The additional solid is a 12 × 8 node Neo-Hookean object with 249 initial
horizontal, vertical, and diagonal bonds. GPU bond damage accumulates from
tensile strain against the configured 0.12 threshold; broken edges remain
broken and damage fades particle stress and changes diagnostic color. The
fixed CSR graph covers only this initial body. Newly emitted solids have no
fracture bonds, and cloning a bonded object is refused to avoid silently
recreating broken topology. This is a grid-dependent discrete demo, not a
calibrated continuum fracture model; GPU numerical accuracy and sustained
damage loading have not been measured across devices/resolutions.

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
only after the batch's compact CFL speed/status readback succeeds. The
performance panel reports this asynchronous scalar-readback latency separately
from queue completion time.

## Interaction and capacity

`config.capacity` reserves particle and object-id buffers during initialization.
If omitted, capacity equals the initial particle count. `particleCount` reports
the current active count, `allocatedParticleCount` reports used slots, and
`particleCapacity` reports the fixed allocation. `appendParticles()` writes
validated accepted particles to both ping-pong buffers, then returns requested,
accepted, refused, dropped, active, allocated, and capacity counts. Capacity
exhaustion never grows buffers or reports a dropped particle as accepted.
`removeParticlesInOutlet()` marks water/oil particles with a removed sentinel on
the GPU and reads back only the removed count; removed slots remain allocated
and are not reused. Cloning never reallocates buffers; it appends a GPU-side
copy of a selected contiguous `objectId` range and returns `false` when the
capacity is full, a removal has occurred, or the object has fracture bonds.
`objectId` is optional on input particles and defaults to `0`; an object's
particles must be contiguous in the initial array. Each particle carries its
constitutive model and material parameters through GPU stepping, readback, and
cloning. The GPU state also preserves liquid density, particle volume, powder
friction, Neo-Hookean fracture damage, and the Snow model's plastic-volume
history.

`pick()` runs a GPU nearest-particle search and reads back only a 16-byte
selection result. `beginDrag()`, `updateDrag()`, and `endDrag()` update a small
uniform; the next grid update applies a damped, local force around the pointer.
Drag and clone do not read back the particle array. Full particle state remains
available only through explicit `readParticles()` calls. Compute dispatches
use the allocated/used slot count (including removed tombstones) rather than
the full reserved capacity.

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

Solid, water, and oil use distinct node mass/momentum/velocity fields; water
and oil keep separate current-volume fields. Mixed water/oil nodes apply an
equal-and-opposite node-owned normal impulse from phase-volume gradients.
Water/oil particles also receive an approximate local density-contrast
acceleration after G2P, based on liquid mass and current area deposited to
neighboring grid nodes. These operations retain phase IDs and make the
controlled reference interface resist crossing, but do not prove complete
immiscibility or general interface stability. GPU surface occupancy is
render-only. The liquid polar projection bounds `det(F)` to `[0.5, 2]` to
prevent unbounded volume drift in the tension-free equation of state.
Surface tension, wetting, tangential interface friction, powder compaction,
gas, and thermal behavior remain unimplemented. The deformation-gradient
update uses fused 2×2 products and a compensated determinant. It adaptively
substeps large increments and retries non-positive trials with finer steps, up
to 64 substeps; P2G and grid-velocity update remain explicit.

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
- `stepCflBatch()` calculates the timestep plan from material wave-speed and
  velocity envelopes and checks actual GPU particle velocities/current wave
  speed before commit. Direct `step()` and `stepBatch()` do not select CFL
  steps automatically. A trial that remains non-positive after deformation
  substepping is rejected and triggers whole-step rollback; this does not make
  an arbitrarily large step stable.
- Buffer sizes and 1D dispatch counts are checked against the acquired device's
  WebGPU limits. Devices that cannot hold the requested state fail setup with
  an explicit error.
- Capacity is fixed for the solver lifetime. Emitters use explicit
  accepted/refused/dropped counts. The lab initializes a million slots, so a
  single source burst is bounded at 100,000 candidates; emitted batches are
  all-or-nothing for validation, while a valid prefix is accepted when only
  part of the remaining capacity is available.
- Particle rendering color-codes the constitutive model and uses a radius
  proportional to `cellSize`. The lab route should size the canvas to match the
  configured world aspect ratio.
- The Gate 4 lab starts with 14,696 mixed-material particles and reserves
  capacity for 1,000,000. Summing the current fixed buffer layout gives about
  221.7 MB (211.4 MiB) before an on-demand full-state readback, which can
  allocate about 84 MB more. This is a layout-derived estimate, not measured
  adapter/process peak memory. The affine scratch adds 16 MB at this capacity.
  Active particles and allocated slots are reported separately.
  This large capacity is intended for PC performance testing and may fail on
  devices without enough GPU memory. Its initial CPU/GPU parity check uses a
  small local neighborhood around one probe per fluid and legacy triangle
  pebble; it does not measure fracture GPU accuracy. The explicit paused-state
  check remains a full readback.

The pure config/layout and batch-encoding tests run without WebGPU. A historical
Apple Metal/Chrome 154 performance benchmark verified shader
compilation, all five models in a 45-particle eight-step CPU/GPU comparison,
batch-versus-single stepping, picking, force-drag, cloning and capacity,
render submission, whole-step rollback, and persistent numerical status.
That historical run predates the current emitter/outlet/fracture additions and
does not validate them, sustained plastic loading, interfaces, granular
calibration, mobile performance, or long-term visual behavior.

See [performance measurement](../../../../docs/PERFORMANCE-MLS-MPM.md) and
[the tuning roadmap](../../../../docs/ROADMAP-MLS-MPM-PERFORMANCE.md).
With a local source server and a dedicated local Chrome CDP endpoint available,
`npm run benchmark:mpm -- --profile=batch` measures normal batch GPU time;
`--profile=stages` intentionally splits dispatches for kernel attribution.
Both require hardware WebGPU and `timestamp-query`, and fail explicitly when
unavailable. Profiling and small-scene readback are benchmark-only; the
interactive solver does not enable timestamps or add full particle readback.
