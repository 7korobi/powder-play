# GPU MLS-MPM kernel

`solver.ts` exports `createMpmMlsGpuSolver(canvas, particles, config)`. Creation
is asynchronous because it requests the WebGPU session and builds the compute
and render pipelines. `config` contains the CPU reference's `grid`,
`particleMass`, `particleVolume`, and single `material`; time step and gravity
are supplied to `step(dt, gravity)`. The caller owns wall-clock stepping and
pause/resume. `render()` reads the current particle buffer directly on the GPU.
CPU particle data is returned only when `readParticles()` is explicitly called.
Positions, grid spacing, time, gravity, mass, volume, and material parameters use
the same units and conventions as `reference/README.md`.
The solver owns the canvas context and device obtained from the shared WebGPU
session; call `dispose()` when the lab is finished.

The compute path has separate clear, linked-list insertion, P2G gather, grid
update, G2P, and particle-commit passes. Every particle inserts its nine
quadratic B-spline stencil entries into per-node linked lists using integer
atomics; one invocation gathers each node's list, so no floating-point atomics
or global grid-by-particle scan is used. Grid storage adds one halo node beyond
each side of the physical `[0, cellsX] × [0, cellsY]` node range. The grid wall
rule and post-G2P particle clipping match the CPU reference.

The constitutive model is the reference's single-material, compressible 2D
Neo-Hookean elasticity. This kernel is not a liquid solver and has no contact,
plasticity, damage, substepping, or material mixing.

## Numerical and runtime limits

- GPU particle and grid state is `f32`; the CPU oracle uses JavaScript
  double-precision numbers. P2G summation follows the atomic linked-list
  insertion order, which is not deterministic across GPU executions. Expect
  small CPU/GPU differences, especially in dense cells or after many steps.
- A GPU numerical failure sets a persistent device-side status and rolls the
  whole step back to the last valid particle state. The next explicit
  `readParticles()` rejects with `MpmMlsGpuNumericalError`; it is intentionally
  not polled from `step()` or `render()`.
- The update is explicit and has no automatic CFL selection. Choose `dt` small
  enough for the material stiffness, grid spacing, and particle travel. A
  finite positive time step alone does not guarantee stability.
- Buffer sizes and 1D dispatch counts are checked against the acquired device's
  WebGPU limits. Devices that cannot hold the requested state fail setup with
  an explicit error.
- Particle rendering uses a simple fixed color and a radius proportional to
  `cellSize`. The lab route should size the canvas to match the configured
  world aspect ratio.

The pure config/layout tests run without WebGPU. Shader compilation, GPU-vs-CPU
numerical error, and visual behavior still need validation on a WebGPU device.
