# 2D Linear Acoustics FDTD Method

This document describes the Finite-Difference Time-Domain (FDTD) method for simulating 2D linear acoustic wave propagation. The implementation is based on the Yee grid staggering approach, adapted from electromagnetic FDTD methods to solve the acoustic wave equations.

## Table of Contents

1. [Governing Equations](#governing-equations)
2. [Velocity-Pressure Formulation](#velocity-pressure-formulation)
3. [Yee Grid Staggering](#yee-grid-staggering)
4. [Discretization](#discretization)
5. [Update Equations](#update-equations)
6. [Update Coefficients](#update-coefficients)
7. [Stability (CFL Condition)](#stability-cfl-condition)
8. [Boundary Conditions](#boundary-conditions)
9. [Sources](#sources)
10. [Material Properties](#material-properties)
11. [Implementation Data Structures](#implementation-data-structures)
12. [Analogy to Electromagnetic FDTD](#analogy-to-electromagnetic-fdtd)

---

## Governing Equations

Linear acoustics describes small-amplitude pressure perturbations in a fluid. The governing equations relate pressure p and velocity v through the linearized Euler equations:

**Momentum equation (Newton's second law):**
```
ρ ∂v/∂t = -∇p
```

**Continuity equation (mass conservation):**
```
∂p/∂t = -κ ∇·v
```

where:
- **p** = acoustic pressure [Pa]
- **v** = (vx, vy) = particle velocity vector [m/s]
- **ρ** = material density [kg/m³]
- **κ** = bulk modulus [Pa]

The wave speed is related to these by:
```
c = √(κ/ρ)    or equivalently    κ = ρc²
```

---

## Velocity-Pressure Formulation

In 2D (x-y plane), the governing equations become:

**Velocity updates (from momentum equation):**
```
ρ ∂vx/∂t = -∂p/∂x
ρ ∂vy/∂t = -∂p/∂y
```

**Pressure update (from continuity equation):**
```
∂p/∂t = -κ (∂vx/∂x + ∂vy/∂y)
```

This first-order system is the acoustic analog of Maxwell's equations in electromagnetics:

| Electromagnetics (TM mode) | Acoustics |
|---------------------------|-----------|
| Electric field Ez | Pressure p |
| Magnetic field Hx, Hy | Velocity vx, vy |
| Permittivity ε | 1/κ (inverse bulk modulus) |
| Permeability μ | ρ (density) |

---

## Yee Grid Staggering

The Yee grid staggers field components in both space and time. This provides second-order accuracy and naturally satisfies continuity conditions at material interfaces.

### Spatial Staggering

For a 2D grid with cells indexed by (i, j):

```
         j+1  ─────┬─────┬─────
              │    │     │
              │ vy │ vy  │
              │    │     │
          j   ──vx─┼──p──┼──vx─
              │    │     │
              │ vy │ vy  │
              │    │     │
         j-1  ─────┴─────┴─────
             i-1   i    i+1
```

**Field locations:**
- **p[i,j]**: Pressure at cell centers (integer indices)
- **vx[i,j]**: x-velocity at cell x-faces (i+½, j)
- **vy[i,j]**: y-velocity at cell y-faces (i, j+½)

More precisely:
- p(i,j) is located at position (i·Δx, j·Δy)
- vx(i,j) is located at position ((i+½)·Δx, j·Δy)
- vy(i,j) is located at position (i·Δx, (j+½)·Δy)

### Temporal Staggering (Leapfrog)

Velocity and pressure are staggered by half a time step:

```
Time:    n-½      n      n+½      n+1
         │        │       │        │
  vx,vy ─●────────┼───────●────────┼─
         │        │       │        │
    p    ┼────────●───────┼────────●─
```

- Velocity is known at times n-½, n+½, n+3/2, ...
- Pressure is known at times n, n+1, n+2, ...

This leapfrog scheme is explicit, second-order accurate in time, and energy-conserving.

---

## Discretization

Using central differences with the staggered grid:

### Spatial Derivatives

For pressure gradient (used in velocity update):
```
∂p/∂x|_{i+½,j} ≈ (p[i+1,j] - p[i,j]) / Δx
∂p/∂y|_{i,j+½} ≈ (p[i,j+1] - p[i,j]) / Δy
```

For velocity divergence (used in pressure update):
```
∂vx/∂x|_{i,j} ≈ (vx[i,j] - vx[i-1,j]) / Δx
∂vy/∂y|_{i,j} ≈ (vy[i,j] - vy[i,j-1]) / Δy
```

### Temporal Derivatives

For velocity:
```
∂vx/∂t|_{n} ≈ (vx^{n+½} - vx^{n-½}) / Δt
```

For pressure:
```
∂p/∂t|_{n+½} ≈ (p^{n+1} - p^{n}) / Δt
```

---

## Update Equations

### Velocity Update (at time n → n+½)

Given pressure at time n, update velocity from n-½ to n+½:

```
vx^{n+½}[i,j] = Cvxvx[i,j] · vx^{n-½}[i,j] + Cvxp[i,j] · (p^n[i+1,j] - p^n[i,j])

vy^{n+½}[i,j] = Cvyvy[i,j] · vy^{n-½}[i,j] + Cvyp[i,j] · (p^n[i,j+1] - p^n[i,j])
```

### Pressure Update (at time n+½ → n+1)

Given velocity at time n+½, update pressure from n to n+1:

```
p^{n+1}[i,j] = Cpp[i,j] · p^n[i,j] +
               Cpvx[i,j] · (vx^{n+½}[i,j] - vx^{n+½}[i-1,j]) +
               Cpvy[i,j] · (vy^{n+½}[i,j] - vy^{n+½}[i,j-1])
```

---

## Update Coefficients

The update coefficients incorporate material properties and grid spacing. For a lossless medium (no damping):

### Velocity Coefficients

For vx at position (i+½, j):
```
ρ_x = ρ[i+½,j]  (density at vx location, averaged if needed)

Cvxvx[i,j] = 1                           (no damping)
Cvxp[i,j]  = -Δt / (ρ_x · Δx)
```

For vy at position (i, j+½):
```
ρ_y = ρ[i,j+½]  (density at vy location, averaged if needed)

Cvyvy[i,j] = 1                           (no damping)
Cvyp[i,j]  = -Δt / (ρ_y · Δy)
```

### Pressure Coefficients

For p at position (i, j):
```
κ = ρ[i,j] · c[i,j]²  (bulk modulus at pressure location)

Cpp[i,j]  = 1                            (no damping)
Cpvx[i,j] = -κ · Δt / Δx
Cpvy[i,j] = -κ · Δt / Δy
```

### With Damping (Optional)

For a medium with damping coefficient σ:

**Velocity (with resistive loss σ_v):**
```
Cvxvx = (2ρ - σ_v·Δt) / (2ρ + σ_v·Δt)
Cvxp  = -2Δt / ((2ρ + σ_v·Δt) · Δx)
```

**Pressure (with compressive loss σ_p):**
```
Cpp  = (2/κ - σ_p·Δt) / (2/κ + σ_p·Δt)
Cpvx = -2Δt / ((2/κ + σ_p·Δt) · Δx · κ)  ... simplifies for κ dependence
```

For simplicity, the initial implementation uses lossless coefficients.

---

## Stability (CFL Condition)

The FDTD scheme is conditionally stable. The Courant-Friedrichs-Lewy (CFL) condition for 2D requires:

```
Δt ≤ 1 / (c_max · √(1/Δx² + 1/Δy²))
```

where c_max is the maximum wave speed in the domain.

For a uniform grid (Δx = Δy = Δ):
```
Δt ≤ Δ / (c_max · √2)
```

In practice, we use a Courant factor S < 1 for stability margin:
```
Δt = S / (c_max · √(1/Δx² + 1/Δy²))

with S ≈ 0.9 (typical choice)
```

---

## Boundary Conditions

### Perfectly Reflecting (Rigid Wall)

For a rigid wall, the normal velocity component is zero. This is the default boundary condition and requires no explicit implementation if the domain boundaries are handled properly:

**At x-boundaries:**
- Left wall (i=0): vx[0,j] = 0 (don't update vx at i=-1)
- Right wall (i=nx): vx[nx,j] = 0 (don't update vx beyond domain)

**At y-boundaries:**
- Bottom wall (j=0): vy[i,0] = 0
- Top wall (j=ny): vy[i,ny] = 0

In the staggered grid, this means velocity components at the boundary faces are simply not updated (remain zero).

### Absorbing Boundary Conditions (Future Extension)

For non-reflecting boundaries, PML (Perfectly Matched Layer) or simple first-order Mur absorbing conditions can be added. The electromagnetic code has a full PML implementation that could be adapted.

---

## Sources

### Pressure Source (Soft Source)

A soft source adds a pressure contribution at each time step without overwriting the field:

```
p[i_s, j_s] = p[i_s, j_s] + S(t)
```

where S(t) is the source waveform.

### Pressure Source (Hard Source)

A hard source directly sets the pressure value:

```
p[i_s, j_s] = S(t)
```

### Common Waveforms

**Gaussian pulse:**
```
S(t) = A · exp(-(t - t₀)² / (2σ²))
```
where:
- A = amplitude
- t₀ = pulse delay (typically 1/(2f) for frequency f)
- σ = pulse width (typically t₀/4)

**Sinusoidal (continuous wave):**
```
S(t) = A · sin(2π·f·t)
```

**Ricker wavelet (Mexican hat):**
```
S(t) = A · (1 - 2(π·f·(t-t₀))²) · exp(-(π·f·(t-t₀))²)
```

---

## Material Properties

### Heterogeneous Media

The FDTD method naturally handles spatially varying material properties. At each grid point:

- **ρ[i,j]**: density
- **c[i,j]**: wave speed
- **κ[i,j] = ρ[i,j] · c[i,j]²**: bulk modulus (derived)

### Material Interface Averaging

When material properties change at cell boundaries, the effective property at staggered grid locations should be averaged:

For vx at (i+½, j):
```
ρ_effective = 0.5 · (ρ[i,j] + ρ[i+1,j])
```

For vy at (i, j+½):
```
ρ_effective = 0.5 · (ρ[i,j] + ρ[i,j+1])
```

This averaging ensures proper wave behavior at interfaces.

### Inclusions

A rectangular or other-shaped inclusion with different material properties is defined by setting ρ and c values differently in that region:

```python
# Example: inclusion with higher density
density[x1:x2, y1:y2] = inclusion_density
wave_speed[x1:x2, y1:y2] = inclusion_wave_speed
```

---

## Implementation Data Structures

### Grid Class

Manages the computational grid:

```python
class Grid:
    nx: int           # Number of cells in x
    ny: int           # Number of cells in y
    dx: float         # Cell size in x [m]
    dy: float         # Cell size in y [m]

    # Field arrays (stored at appropriate staggered locations)
    p:  ndarray[nx+1, ny+1]   # Pressure at cell centers
    vx: ndarray[nx+1, ny]     # x-velocity at x-faces
    vy: ndarray[nx, ny+1]     # y-velocity at y-faces
```

Note: Array dimensions account for the staggered grid layout.

### Material Class

Stores material properties:

```python
class Material:
    density: ndarray[nx+1, ny+1]     # Density at pressure points
    wave_speed: ndarray[nx+1, ny+1]  # Wave speed at pressure points

    # Derived quantities
    bulk_modulus: ndarray            # κ = ρc²

    # Averaged properties for velocity updates
    density_x: ndarray               # Density at vx locations
    density_y: ndarray               # Density at vy locations
```

### UpdateCoefficients Class

Precomputed coefficients for efficient updates:

```python
class UpdateCoefficients:
    # Velocity update
    Cvxvx: ndarray    # vx self-coefficient
    Cvxp: ndarray     # vx from pressure gradient
    Cvyvy: ndarray    # vy self-coefficient
    Cvyp: ndarray     # vy from pressure gradient

    # Pressure update
    Cpp: ndarray      # p self-coefficient
    Cpvx: ndarray     # p from vx divergence
    Cpvy: ndarray     # p from vy divergence
```

### Simulation Loop

The main simulation loop follows this structure:

```python
for timestep in range(num_timesteps):
    # 1. Update velocity fields (vx, vy) from pressure gradient
    update_velocity(vx, vy, p, coefficients)

    # 2. Apply velocity boundary conditions
    apply_velocity_bc(vx, vy)

    # 3. Update pressure field from velocity divergence
    update_pressure(p, vx, vy, coefficients)

    # 4. Add source contribution
    apply_source(p, source, time)

    # 5. Advance time
    time += dt
```

---

## Analogy to Electromagnetic FDTD

The acoustic FDTD is mathematically equivalent to the TM-mode electromagnetic FDTD. This table shows the correspondence:

| EM (TM mode) | Acoustics | Role |
|--------------|-----------|------|
| Ez | p | Scalar field (pressure/E-field) |
| Hx | vy | Vector component perpendicular to x |
| Hy | -vx | Vector component perpendicular to y |
| ε | 1/κ | Relates scalar to vector time derivative |
| μ | ρ | Relates vector to scalar spatial derivative |
| σ_e | σ_p | Scalar field damping |
| σ_m | σ_v | Vector field damping |
| J_z (current) | pressure source | Source term |

The sign conventions differ slightly because:
- In EM: ∂E/∂t = (1/ε)(∇×H - J)
- In acoustics: ∂p/∂t = -κ(∇·v)

The staggered grid and leapfrog time stepping work identically.

---

## References

1. Taflove, A. & Hagness, S.C. (2005). *Computational Electrodynamics: The Finite-Difference Time-Domain Method*. Artech House.
   - The foundational FDTD textbook; acoustic FDTD follows the same principles.

2. Botteldooren, D. (1995). "Finite-difference time-domain simulation of low-frequency room acoustic problems." *JASA* 98(6), 3302-3308.
   - Application of FDTD to room acoustics.

3. Liu, Q.H. & Tao, J. (1997). "The perfectly matched layer for acoustic waves in absorptive media." *JASA* 102(4), 2072-2082.
   - PML boundary conditions for acoustic FDTD.

4. Hesthaven, J.S. & Warburton, T. (2008). *Nodal Discontinuous Galerkin Methods*. Springer.
   - Higher-order alternative method (used in our 3D DG code for comparison).
