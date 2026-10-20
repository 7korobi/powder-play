# CPU MLS-MPM reference

This directory contains a deterministic, double-precision CPU reference for a
single explicit 2D MLS-MPM step. The GPU MLS-MPM implementation is in
`../kernels/`. `tait-eos.ts` provides the analytical SI equation-of-state
reference and a conversion to the fixed-depth 2D liquid material used by the
GPU and CPU steps. The SI lab is an experimental profile, not an accepted
physical-validation gate.

## State and units

Each particle supplies `position`, `velocity`, its affine velocity matrix `C`,
and deformation gradient `F`. `particleMass` and `particleVolume` in the step
config are the defaults. A particle may override its reference volume and
material. Water and oil derive particle mass from their material density
multiplied by reference volume; other models use the configured particle mass.
`objectId` remains independent of the constitutive model.

`position`, `grid.cellSize`, and the domain extents use the same length unit;
velocity is length/time, `dt` is time, gravity is length/time², mass is mass,
and `particleVolume` is a reference area; a particle's `volume` overrides it.
Water/oil `density` is reference surface mass density, and `mu` is fixed at
zero. Without an `equationOfState` profile, `lambda` is the legacy linear
pressure coefficient. With `{ model: 'tait', exponent: n }`, `density` stores
`rho0 * d` and `lambda` stores `B * d`, where `d` is the extrusion depth;
the pressure resultant is `B * d * (J^-n - 1)`. For elastic and powder models,
the 2D Lamé parameters `lambda` and `mu` are force/length (2D stress-resultant)
units, so `particleMass / particleVolume` is surface mass density. A
consistently nondimensional system is also valid. `F` maps reference-space column vectors
to current space; the row-major matrix entries are `m00`, `m01`, `m10`, `m11`,
and matrix-vector products use column vectors.

The physical grid domain is the closed rectangle
`[0, cellsX * cellSize] × [0, cellsY * cellSize]`. The P2G stencil uses its
complete 3×3 quadratic B-spline support, including ghost grid nodes beyond the
domain edges. Grid walls at the domain edge remove only outward normal velocity;
after G2P, any particle crossing is clipped to the edge and its outward normal
velocity is set to zero. `wallFriction` defaults to zero; values from zero to
one damp tangential velocity near walls over the existing wall-cushion width.
One is a no-slip approximation, not a calibrated Coulomb-friction coefficient.

## Equations and model

For each axis, let `base = floor(x / cellSize - 1/2)` and
`f = x / cellSize - base`. The three quadratic weights are

```text
w0 = 1/2 (3/2 - f)^2
w1 = 3/4 - (f - 1)^2
w2 = 1/2 (f - 1/2)^2
```

For `dpos = gridNodePosition - particlePosition`, P2G accumulates

```text
m_i += w_ip * m_p
q_i += w_ip * (m_p * v_p + A_p * dpos)
A_p = m_p * C_p - 4 * dt * particleVolume / cellSize^2 * (P_p * F_p^T)
```

The grid velocity is `q_i / m_i + dt * gravity`, followed by the wall rule
above. G2P computes

```text
v_p' = sum_i w_ip * v_i
C_p' = 4 / cellSize^2 * sum_i w_ip * (v_i * dpos^T)
x_p' = x_p + dt * v_p'
F_p' = (I + dt * C_p') * F_p
```

For water/oil particles, P2G additionally accumulates local liquid mass and
current area at each node:

```text
M_i^liq += w_ip * m_p
V_i^liq += w_ip * particleVolume_p * det(F_p)
rho_local = sum_i(w_ip * M_i^liq) / sum_i(w_ip * V_i^liq)
rho_p = referenceDensity_p / det(F_p)
a_separation = gravity * (rho_p - rho_local) / rho_p
v_p' = sum_i(w_ip * v_i) + dt * a_separation
```

This correction changes the phases' relative acceleration in mixed
neighborhoods; it is not, by itself, an Archimedean buoyancy force and may not
accelerate a lighter phase upward against gravity. Water and oil have separate
grid velocities. At mixed nodes, a phase-volume-gradient contact impulse
provides an additional ordering bias: if the current interface normal points
against the density ordering implied by gravity, the normal is reversed so
the denser phase is driven with gravity and the lighter phase against it.
Near-equal density contrasts or near-tangential gravity do not trigger the
reversal. This is a discrete sorting heuristic, not a full interface or
buoyancy model.

Near bonded triangles, grid contact removes closing normal velocity and
applies equal-and-opposite tangential drag at a rate of 4 s⁻¹. The exponential
relaxation scales with each timestep, so CFL microsteps do not linearly
multiply the intended drag. This couples fluid motion into triangle translation
and rotation; it is viscous drag, not Coulomb friction.

The default constitutive law is compressible 2D Neo-Hookean elasticity:

```text
J = det(F) > 0
P = mu * (F - F^(-T)) + lambda * log(J) * F^(-T)
```

`lambda` must be non-negative and `mu` positive. Invalid or inverted input
`F`, and a step that would invert `F`, are rejected.

Particles may instead select the 2D Snow-style corotated elastoplastic model.
Its hardening is `exp(clamp(10 * (1 - Jp), -20, 20))`, applied to both Lamé
parameters, and its stress-times-transpose is

```text
2 * mu_h * (F - R) * F^T + lambda_h * J * (J - 1) * I
```

After each G2P update, the two principal stretches are clamped to
`[1 - 0.025, 1 + 0.0075]`. The plastic-volume history is updated by the ratio
of trial to projected deformation determinant and clamped to `[0.6, 20]`.

Water and oil remain separate fluid phases. The legacy profile is
tension-free and uses the linear stress-times-transpose

```text
P * F^T = lambda * min(J - 1, 0) * I
```

For a Tait profile, with areal `lambda = B * d`, the stress-times-transpose used
by P2G is

```text
P * F^T = -J * lambda * (J^-n - 1) * I
```

Neither profile has shear stress. Their reference areal density changes
particle mass; the local mixture-density correction and density-aware contact
bias above promote buoyant ordering, but do not provide surface tension or
guarantee a clean water/oil interface. After each G2P update, the liquid polar
projection clamps `det(F)` to `[0.5, 2]`. The Tait profile derives the reference
sound speed from `n * lambda / density` and increases its CFL bound with current
compression.

Powder uses a simplified Drucker-Prager-like shear cap in log-stretch space.
With principal log stretches `ell_i = log(sigma_i)`, the volumetric log strain
is `log(J)`, and pressure uses
`max(0, -(lambda + mu) * max(log(J), log(0.5)))`. The deviatoric log stretch is
limited to `frictionCoefficient * pressure / (2*mu)`. After G2P, its area
ratio is bounded below by `0.5`. This prevents near-zero powder volumes but is
a hard deformation bound, not particle contact or plastic compaction. This is
an exploratory model without cohesion or material calibration; it is not a
complete granular solver.

These models can coexist in one particle array and grid. Solids use the common
grid velocity, while water and oil use separate phase velocity and volume
fields. The density-aware grid contact is approximate: there is no
particle-level interphase contact, surface tension, wetting, or full
immiscibility, and phases can still interpenetrate or numerically mix. The
triangle drag is a grid-contact approximation, not a calibrated fluid-solid
viscosity model. There is also no automatic CFL-based physical substepping in
this CPU reference;
large deformation-gradient increments alone are adaptively subdivided, up to
64 increments.

## Numerical limits

This is an explicit, first-order update. Finite positive `dt` is required but
does not guarantee stability. As a practical limit, keep the elastic Courant
number well below one:

```text
dt / cellSize * sqrt((lambda + 2 * mu) / (particleMass / particleVolume)) << 1
```

Also keep particle travel per step small relative to `cellSize`, especially near
walls. There is no automatic substepping or CFL selection; large steps can
produce inaccurate motion or an inverted deformation gradient. The Snow model
only projects principal stretches after G2P; it does not silently repair an
invalid input deformation.

The deterministic tests use absolute tolerances of `1e-12` for mass and
`1e-10` for momentum in the selected interior, zero-gravity case. These are
double-precision test tolerances, not global error bounds. Momentum is expected
to be conserved only when gravity is zero and no wall acts on a particle's
stencil or motion; wall collisions intentionally remove normal momentum.
