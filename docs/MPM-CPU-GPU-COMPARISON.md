# Gate 4 CPU/GPU MLS-MPM comparison

`/labs/mpm/reference` is the CPU numerical reference and
`/labs/mpm/kernels` is the WebGPU implementation. Both pages build the same
Gate 4 scene and use the same `stepMpm` constitutive/contact model, material
parameters, fracture network, fixed timestep, and CFL limit.
`/labs/mpm/liquidfun` is a separate LiquidFun WASM behavior reference; its
solver and solid representation differ and it is not a numerical oracle for
MLS-MPM. See [LIQUIDFUN-BENCHMARK.md](./LIQUIDFUN-BENCHMARK.md) for its
mapping, limitations, and measurement method.

Liquid contact with bonded triangles uses the same normal response and
rate-based tangential drag on both implementations. Tangential relative
velocity is exponentially relaxed at 4 s⁻¹, with equal-and-opposite impulses
so fluid flow can carry and rotate triangles without adding contact energy in
the isolated two-body response.

## Shared scene and adjustable liquid size

The shared scene is created by `createMpmGate4Scene` with seed `0x4d504d34`.
It occupies a 2,200 × 1,500 world-unit domain on a 220 × 150 grid with 10
world units per cell. Both pages use `dt = 1/240 s`, default gravity
`(0, 90) world units/s²`, the same initial velocities, and the same material
configuration.

The default workload is:

- 2,000 water particles and 1,500 oil particles;
- 150 vertices forming 50 deformable Neo-Hookean triangles;
- 24 fracture particles connected by 53 breakable bonds.

The common **liquid reference diameter** control accepts 10–21 world units
and defaults to 20. The shared query parameter is `liquidParticleDiameter`;
following the link between pages preserves the applied value. The reference
area is `diameter²` (400 units² by default). MPM liquid mass is derived from
the phase density and this area, and the renderer uses the particle volume to
scale its display. The control does not change particle count or placement
spacing.

The default horizontal acceleration is enabled at ±240 world units/s² and
oscillates at 0.16 Hz. Both pages expose matching controls to stop/start it
or change its amplitude. The startup parity step is unoscillated on both
pages; normal simulation begins with the same oscillation phase afterward.

## What differs

- The CPU route runs the reference TypeScript solver using JavaScript number
  precision. The GPU route runs the MLS-MPM kernels in WebGPU float32. The
  equations and input state are shared, but floating-point precision and
  accumulation order can cause trajectories to drift; bitwise identity is
  not expected.
- The CPU route runs its synchronous solver in a Web Worker and caps each
  worker batch at 8 CFL microsteps. It retains unsimulated time as backlog
  rather than dropping it or running an unbounded catch-up loop. At high
  requested speeds, the CPU scene can therefore lag behind wall-clock time.
- The GPU route reports queue/kernel measurements and has optional
  GPU-specific diagnostics and experiments (including particle interaction,
  emission, and outlet operations). Those optional tools are not part of the
  default CPU/GPU baseline; use Reset before comparing the seeded initial
  scene.
- Rendering is implementation-specific. The CPU route uses Canvas2D, while
  the GPU route uses WebGPU surface/particle render paths. Compare physical
  state and step throughput separately from the visual renderer. Both paths
  now scale liquid display area using the current deformation, although their
  surface extraction and rasterization remain different.

The CPU and GPU pages provide matching baseline controls for pause/resume, one step,
gravity and device tilt, simulation speed, liquid particle size, oscillation,
render mode, and render-rate target. For performance comparisons, record
browser/device, applied diameter, speed, render target, oscillation setting,
completed macrosteps/s, CFL microsteps/s, and the separate physics/render
timings. Reset both pages after changing their initial settings.

## LiquidFun as a behavioral target

Use LiquidFun to identify desirable qualitative and measurable outcomes (for
example maximum pebble penetration, settling time, wave height/decay, and
whether fluid motion carries a pebble) before tuning the GPU MLS-MPM page.
Compare those behavior metrics separately from same-equation CPU/GPU parity
and from per-step performance. Different solvers, shared LiquidFun
particle-system parameters, and rigid-body pebble approximations mean
particle-by-particle trajectory matching is neither expected nor a reliable
correctness test.
