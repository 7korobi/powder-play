# CPU MLS-MPM reference

This directory contains a deterministic, double-precision CPU reference for a
single explicit 2D MLS-MPM step. It is a correctness oracle for planned GPU
kernels; it is not a GPU MLS-MPM implementation or the Gate-0 interaction demo.

## State and units

Each particle supplies `position`, `velocity`, its affine velocity matrix `C`,
and deformation gradient `F`. All particles use the one `particleMass` and
`particleVolume` in the step config. `material` in the step config is the
default; a particle may override it with its own Lamé parameters and
constitutive model. Missing particle material uses the configured default.
`objectId` remains independent of the constitutive model.

`position`, `grid.cellSize`, and the domain extents use the same length unit;
velocity is length/time, `dt` is time, gravity is length/time², mass is mass,
and `particleVolume` is a reference area. The 2D Lamé parameters `lambda` and
`mu` are force/length (2D stress-resultant) units, so `particleMass /
particleVolume` is surface mass density. A consistently nondimensional system
is also valid. `F` maps reference-space column vectors to current space; the
row-major matrix entries are `m00`, `m01`, `m10`, `m11`, and matrix-vector
products use column vectors.

The physical grid domain is the closed rectangle
`[0, cellsX * cellSize] × [0, cellsY * cellSize]`. The P2G stencil uses its
complete 3×3 quadratic B-spline support, including ghost grid nodes beyond the
domain edges. Grid walls at the domain edge remove only outward normal velocity;
after G2P, any particle crossing is clipped to the edge and its outward normal
velocity is set to zero. Tangential velocity is unchanged.

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
This is a compact elastoplastic PoC, not a calibrated snow, granular, or
powder model. Neo-Hookean and Snow particles may coexist in one grid; they do
not yet have explicit particle contact, friction, damage, liquid behavior, or
an interface model.

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
