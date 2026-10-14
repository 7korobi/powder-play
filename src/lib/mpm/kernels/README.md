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
microstep. When a Neo-Hookean deformation approaches a collapsed or extreme
stretch, its principal stretches are projected into `[0.05, 20]` before
commit, preventing valid-but-ill-conditioned states from destabilizing later
float32 updates; the CPU reference uses the same projection. `config` contains
the CPU reference's `grid`,
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
`measureMaximumParticleSpeed()` asynchronously reads one GPU-reduced float, the
completed-microstep counter, and the small numerical status (40 bytes total),
not the particle array.
Positions, grid spacing, time, gravity, mass, volume, and material parameters use
the same units and conventions as `reference/README.md`.
The solver owns the canvas context and device obtained from the shared WebGPU
session; call `dispose()` when the lab is finished. `step(dt, gravity)` queues
one direct step. `stepBatch(stepCount, dt, gravity)` encodes sequential fixed
steps into one command buffer and submits them together. `stepCflBatch()`
recomputes the plan with `calculateMpmCflMicrosteps()`, accepts only a matching
plan or a CFL-certified finer refinement, rejects a total microstep count
above the caller's cap, and submits
`macroStepCount * plan.stepCount` full steps at `plan.microDt`. It accepts the
velocity envelope for the whole batch to choose the microstep count. The
runtime guard checks each particle's actual speed against the maximum speed
allowed by that float32 microstep after reserving the derived wave-speed bound.
Every numerical microstep completes the entire solver stage sequence and
particle-buffer ping-pong before the next begins. With bonded triangles and
fracture bonds, that sequence includes clear grid, prepare particles, build
particle buckets, P2G, build conservative tiled triangle-field candidates when
the bounded path is active, build the triangle contact field, update grid, G2P,
apply field-contact impulses, solve bonded triangles, gather body contact state,
gather canonical pair contacts when the bounded cache is active, apply pair
impulses, validate CFL bounds, update fracture bond damage, apply incident
damage to particles, and commit. A batch still uses one compute pass. A final
GPU reduction finds maximum particle
speed; only its scalar, the completed-microstep counter, and the compact status
buffer are asynchronously mapped on the CPU.

`stepCflBatchAdaptive()` additionally reads how many microsteps committed
before a numerical failure. For CFL violations and recoverable deformation
failures with finite diagnostics, it retries only the failed interval using at
least twice as many microsteps and updates velocity and wave-speed envelopes
from the failed trial. Retries share the caller's total microstep budget,
including work from rejected attempts; there is no separate retry-count cap.
The result reports
`simulatedDuration`, `remainingDuration`, and a `recommendedVelocityBound`.
When a safe retry or continuation does not fit the remaining budget, the solver
preserves committed progress and returns the unprocessed duration for the
caller to schedule later; it never advances through an unvalidated step or
discards elapsed simulation time. The Gate 4 page restores that duration to its
time accumulator and resumes it in a later bounded batch. Unsupported
numerical errors, repeated instability, and a batch that cannot make valid
progress remain explicit failures. The page permits only one adaptive batch at
a time so periodic status checks cannot mistake a recoverable trial failure
for a final solver error.

The Gate 4 lab uses a `1 / 240`-second macrostep and starts at a 2×
simulation-time multiplier. The UI offers 0.5×–32×; each macrostep is split
into the complete CFL-planned numerical microsteps. Macrostep rate determines
simulated time and sinusoidal acceleration, while microstep and particle
microstep rates are reported separately. The scheduler caps an initial
per-frame batch at 512 microsteps and defers excess requested work to later
batches; adaptive recovery counts failed work and retries against a separate
1,024-microstep batch budget. Unprocessed simulation time remains in the
accumulator rather than being clamped or discarded. When a fixed macrostep
cannot fit the cap, the scheduler shortens the simulation-time slice to a
positive, representable CFL-safe duration and retains the rest. The UI shows
requested/effective time scale and pending simulation time; while paused, the
user may explicitly discard only that pending time without changing particle
state. An emitter burst is checked against the current CFL envelope before GPU
append. Values that cannot produce a positive float32 CFL slice are refused
without stopping the existing simulation. If no safe slice can be represented
or a batch makes no valid progress, the lab reports an explicit error and
offers a reload from the initial scene.

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
microstep. A violating microstep rolls back to the last valid state; the lab
then retries the remaining interval with the adaptive policy above.
This is a guarded, evolving bound, not a claim that the initial speeds are a
lifetime guarantee. The default 1× macro-time scheduling path is exercised by
tests; the lab's existing 2× default remains unchanged. At most one adaptive
CFL batch and its compact readback may be pending; later frame time remains in
the accumulator until it completes. Work-cap overflow is deferred rather than
treated as a fatal solver error or silently dropping simulated time.

The retuned Gate 4 MPM sample starts with 2,000 water, 1,500 oil, 50 Neo-Hookean
elastic pebbles built from three particles each, and a 24-particle connected
solid (3,674 active particles total, one quarter of the previous 14,696).
The MPM page uses a dedicated 220 × 600 grid at 10-unit cell size
(2,200 × 6,000 world units); the legacy 220 × 150 `GATE4_GRID` remains unchanged
for the LiquidFun comparison. The scene is shifted 4,500 world units upward in
Y so its liquid, pebble, and fracture-body floor clearances remain unchanged.
Relative to the preceding 3,674-particle MPM tuning, water and oil particle
reference area grows from 400 to 1,600 (4×); the resulting total reference
areas are 3.2M / 2.4M world-area units, and total masses are 64,000 / 38,400.
Each phase's reference volume and mass both grow 4× at unchanged density.
Liquid particle counts, spacing (18 world units), and horizontal positions remain
unchanged. The particle-marker radius therefore doubles, while the independent
display-area multiplier remains 2×. The compressive EOS stiffness remains four
times the earlier baseline (water 40,000 → 160,000; oil 30,000 → 120,000) to
resist volume collapse. Pebble and fracture-solid reference area remains 100,
and solid particle mass remains 4. The pebbles form an 8-column × 7-row array with
150-world-unit center spacing; the last row contains two centered pebbles.
Each is an equilateral triangle with 132-world-unit edges and width, about
6% of the simulation-domain width. The array overlaps the liquid region
instead of starting outside it. Each triangle has its own `objectId` and uses
lambda 20 and mu 10; its three edges also use position constraints. The GPU
solves these constraints in triangle-local
offsets and derives velocity corrections from those small displacements before
reconstructing world positions, avoiding precision loss from subtracting
near-equal large coordinates. Constraint stiffness scales with the square of
the microstep/reference-step ratio (1/240 second in Gate 4), because positional
corrections are converted into velocity changes. CFL subdivision therefore
does not accumulate an oversized correction impulse. Water and oil have
separate grid velocity/volume channels and receive normal contact impulses from
their phase-volume gradients. The phase contact also applies an equal-and-
opposite separation impulse scaled by local phase overlap and approach speed;
its node-level target is the larger of `0.25 × overlap × cellSize / 0.05 s`
and the closing normal speed. The approach-speed term does not amplify relative
kinetic energy. This is an approximate interface response,
not full immiscibility. When the phase-gradient normal conflicts with the
gravitationally stable density order, the solver reverses it so the denser
phase is driven with gravity and the lighter phase against it. This promotes
layer reordering but is not a continuum buoyancy or surface-tension solve.
The triangle signed-distance field projects liquid particles out of the
pebbles and gathers equal-and-opposite normal and tangential impulses back to
the three vertices. Tangential relative velocity is exponentially relaxed at
a rate of 4 s⁻¹, with each CFL microstep applying `1 - exp(-4 × dt)`. This
rate-based drag transfers some wave motion into pebble translation and rotation;
it is not Coulomb friction.
Pebble pairs use triangle SAT contact to reduce approaching normal velocity;
the pair impulse pass does not translate triangle vertices. Repeated post-G2P
position corrections destabilized the triangle contact field during CFL
microsteps. The GPU snapshots body velocities and derives contact degrees
before the pair impulse pass, then
symmetrically relaxes each impulse by 0.25 divided by the larger contact degree
of its endpoints. This prevents simultaneous contacts from over-accumulating
Jacobi impulses. For up to 512 active bonded bodies, `gatherTrianglePairContacts`
computes one contact result per unordered pair (AABB rejection followed by SAT
when needed), and the pair impulse pass reuses that canonical contact with the
opposite normal for the other body. Contact counts are accumulated for both
endpoints. The cache is bounded to 4 MiB; above 512
bodies the solver retains the direct all-pairs contact path. This removes
duplicate narrow-phase work but does not reduce the `O(B²)` pair enumeration.
These are discrete, approximate contacts, not a continuous collider or
continuum deformable mesh.
For up to 512 bonded bodies, the field builder also uses a conservative
16×16-node tile bitset to skip bodies whose AABBs cannot reach any node in the
tile. The original per-node AABB and triangle-distance tests still decide field
values, and candidates are visited in body order. Above 512 bodies it falls
back to the full field scan; dense overlap remains the worst case.
At scene initialization, water and oil velocity components are sampled
independently from ±64 world units/s, each pebble vertex from ±240 world
units/s, and each fracture-solid particle from ±80 world units/s. Pebbles are
rendered as filled triangles with a bright outline in both surface and
particle-marker modes. Horizontal oscillation starts disabled; a button starts/stops the
default ±240 world-units/s², 0.16 Hz motion, and the slider adjusts its
amplitude up to 360. A fixed scene seed reproduces particle velocities for
tests and benchmarks; the lab chooses a fresh seed on initialization.
The 16×16 tile candidate bitset reduces sparse field scans for up to 512 bodies.
With 50 active pebble bodies, the full-scan upper bound is about 1.71 million
node/body candidates per step and unordered SAT pairs fall from 19,900 to
1,225. A historical no-render `batch` profile on Apple Metal 3 / Headless Chrome
154 used 3,674 particles and 12 samples of eight steps: GPU step median was
0.217 ms and p95 was 0.256 ms before the liquid area increase from 100 to 400.
The fixed-step harness does not run the lab's adaptive CFL scheduler. This
revision's 400-area batch run rejected a step after the deformation update
exceeded its adaptive substep limit at particle 1170; this is not a profile of
the interactive adaptive route. The old timing is historical and current
performance is unmeasured. Both pages remained active during a 10-second
foreground browser smoke with zero console errors. The diagnostic
`stages` profile splits the batch into multiple passes and its times are not
directly comparable. Node-level field candidate counts remain unmeasured. The
GPU contact fixture also passed at
opposed ±64 world-units/s with no particle crossing over eight 0.001-second
steps, while preserving phase mass and momentum within the stated tolerance.
Coulomb friction, surface tension, wetting, complete water/oil separation in arbitrary
flows, and swept collision detection are not implemented.

The additional solid is a 6 × 4 node Neo-Hookean object with 53 initial
horizontal, vertical, and diagonal bonds. GPU bond damage accumulates from
tensile strain against the configured 0.12 threshold; broken edges remain
broken and damage fades particle stress and changes diagnostic color. The
fixed CSR graph covers only this initial body. Newly emitted solids have no
fracture bonds, and cloning a bonded object is refused to avoid silently
recreating broken topology. This is a grid-dependent discrete demo, not a
calibrated continuum fracture model; GPU numerical accuracy and sustained
damage loading have not been measured across devices/resolutions.

Water and oil reference densities remain 0.02 and 0.016, with EOS stiffnesses
160,000 and 120,000 (four times the preceding 40,000 and 30,000). Relative to
the original 14,696-particle scene, the liquid count is one quarter while
particle area is 64 times larger (25 → 1,600), so total liquid reference volume
and mass are 16× that baseline. Relative to the preceding 3,674-particle MPM
tuning at area 400, the area and total liquid volume/mass are each 4×. The
18-unit spacing is unchanged; reference area per lattice spacing is now
`1,600 / 18²`, 4× the preceding MPM value `400 / 18²`. The MPM grid is
220 × 600 cells at 10-unit cell size; the independent LiquidFun baseline remains
220 × 150. At J≈1.03 the liquid volume estimate is about 5.77 million units²,
or 44% of the 13.2-million-unit² MPM domain; even at the configured J=2 ceiling,
the reference volume would occupy about 85%. A previous 20,000-particle scene
used 6,000 powder particles; the later 200-pebble and 14,696-particle layouts
are historical. The current sample reserves capacity for 1,000,000 particles,
so reducing active count does not materially reduce the fixed GPU allocation.

The surface and particle-marker renderers apply an additional 2× display-area
scale independently of physical particle area. The 1,600-unit reference area
makes the marker radius 2× larger than in the preceding area-400 MPM tuning;
the display multiplier remains unchanged.
Surface pixels use the
dominant water/oil phase color with a narrow interface highlight; this is a
visual cue, not a guarantee of physical separation. The sinusoidal horizontal
acceleration averages over each batch's simulation-time interval; the
performance panel reports asynchronous scalar-readback latency separately from
queue completion time.

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
neighboring grid nodes. Together with the density-aware contact impulse,
these operations favor gravitationally stable layering while retaining phase
IDs; they do not prove complete immiscibility or general interface stability.
GPU surface occupancy is render-only. The liquid polar projection bounds
`det(F)` to `[0.5, 2]` to prevent unbounded volume drift in the tension-free
equation of state.
Surface tension, wetting, tangential water/oil interface friction, powder
compaction, gas, and thermal behavior remain unimplemented. The deformation-gradient
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
  substepping is rejected and triggers microstep rollback; the adaptive API may
  retry the remaining duration with finer CFL-certified steps, but does not
  make an arbitrarily large step stable.
- Buffer sizes and 1D dispatch counts are checked against the acquired device's
  WebGPU limits. Triangle-contact pipelines require at least eight storage
  buffers per compute stage. Devices that cannot hold the requested state fail
  setup with an explicit error.
- Capacity is fixed for the solver lifetime. Emitters use explicit
  accepted/refused/dropped counts. The lab initializes a million slots, so a
  single source burst is bounded at 100,000 candidates; emitted batches are
  all-or-nothing for validation, while a valid prefix is accepted when only
  part of the remaining capacity is available.
- Particle rendering color-codes the constitutive model and scales marker
  radius with the square root of reference area in addition to the base
  `cellSize` radius. The lab route should size the canvas to match the
  configured world aspect ratio.
- The Gate 4 lab starts with 3,674 mixed-material particles and reserves
  capacity for 1,000,000. Summing the current fixed buffer layout gives about
  221.7 MB (211.4 MiB) before an on-demand full-state readback, which can
  allocate about 84 MB more. This is a layout-derived estimate based on the
  prior fixed-capacity layout, not a measurement of the retuned scene's
  adapter/process peak memory. The affine scratch adds 16 MB at this capacity.
  A scene with bonded triangles also allocates up to 4 MiB for the bounded pair
  cache, a pair-candidate bitset, and a grid-sized field-candidate bitset. The
  The historical 14,696-particle benchmark measured 231,196,340 total buffer
  bytes with 1,000,000 capacity, 4,236,032 bytes above the pre-cache benchmark
  allocation; the field-candidate bitset used 8,960 bytes in that grid.
  Active particles and allocated slots are reported separately.
  This large capacity is intended for PC performance testing and may fail on
  devices without enough GPU memory. Its initial CPU/GPU parity check uses a
  small local neighborhood around one probe per fluid and the central triangle
  pebble, now positioned at the liquid interface; it does not measure fracture
  GPU accuracy or scene-scale mixing. The explicit paused-state
  check remains a full readback and reports water/oil `det(F)` ranges and means
  separately for all particles and the bottom eight grid cells, together with
  each phase's vertical extent.

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
