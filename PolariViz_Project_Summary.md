# PolariViz — Project Summary & Handoff Document

## Purpose of This Document
This document is a complete handoff summary for continuing development of
PolariViz in a new chat session. It covers what has been built, all key
design decisions, the current state of the code, and what remains to be done.

---

## 1. What Is PolariViz?

PolariViz is an **interactive web visualization app** for AMO (atomic,
molecular, and optical) physics. It visualizes how polarized light is
absorbed by atoms undergoing a **J=0 → J=1 electric dipole (E1) transition**,
as a function of:

- The **incident beam direction** (θ, φ, χ — ZYZ Euler angles)
- The **polarization state** of the light (pure basis states OR a waveplate chain)
- The **quantization axis** orientation (set by B-field direction θ_B, φ_B)

The app is intended for **research colleagues** in AMO physics. It is being
built by Benjamin, an AMO physicist with strong Python skills but limited
web development experience.

---

## 2. Technology Stack & Deployment Plan

### Development Stack
- **Python** — all physics and visualization logic
- **Plotly** — all figures (3D and 2D)
- **Dash** — web app framework for local testing

### Deployment Plan (two-phase)
1. **Phase 1 (current):** Build and test locally using Python/Dash
2. **Phase 2 (final):** Convert to a **static Plotly.js site** hosted on
   **GitHub Pages** — free forever, zero maintenance, no server required

The conversion from Dash → static site is planned as a single pass once
the physics and visuals are verified correct. The entire codebase has been
written with this conversion in mind (see Section 4).

---

## 3. Project File Structure

```
polarization_rotation_visualizer_dash/
├── app.py              # Dash wiring ONLY (layout + single callback)
├── math_helpers.py     # Pure math library (no NumPy, no Plotly)
├── physics.py          # Physics computations (pure functions)
├── traces.py           # Plotly figure builders (pure functions)
├── requirements.txt    # plotly, dash
└── assets/
    └── style.css       # Dark theme styling
```

**All four Python files have been written and syntax-checked.**
`math_helpers.py` and `physics.py` have passing self-tests.
`traces.py` passed its self-test on Benjamin's local machine.
`app.py` is running locally and Benjamin is currently debugging it.

---

## 4. Key Architectural Decisions

### 4a. Conversion-Ready Code
Every function in `math_helpers.py` and `physics.py` has its JavaScript
equivalent written in the docstring. The code follows strict rules to
make Python → JS conversion mechanical:

- **No NumPy** — only the built-in `math` module
- **No `@` operator** — use `mat_vec_multiply()` and `mat_mat_multiply()`
- **Plain Python lists** for all vectors and matrices (not NumPy arrays)
- **Complex numbers as `[re, im]` pairs** (not Python `complex` type)
- **Single master callback** — one `compute_all()` → one `update_all()` in JS

### 4b. Complex Arithmetic
Almost all vectors and matrices in the physics are complex (the polarization
is a rank-1 spherical tensor). A complete complex arithmetic library has been
built in `math_helpers.py` using `[re, im]` pairs:

- Scalar: `c_make`, `c_add`, `c_sub`, `c_mul`, `c_div`, `c_conj`, `c_abs`, etc.
- Vector: `c_vec_add`, `c_vec_scale`, `c_dot` (Hermitian), `c_cross`, `c_norm`
- Matrix: `c_mat_vec_multiply`, `c_mat_mat_multiply`, `c_conjugate_transpose`
- Constants: `B_CARTESIAN_TO_SPHERICAL`, `B_SPHERICAL_TO_CARTESIAN`

### 4c. No Matrix Inversion Needed
All matrix inversions in the physics are one of two special cases:
- **Real rotation matrices:** R⁻¹ = Rᵀ (transpose)
- **Unitary matrices (B matrix):** U⁻¹ = U† (conjugate transpose)

`np.linalg.inv()` is never needed and never used.

### 4d. Spherical Basis Convention
The B matrix converts Cartesian [x, y, z] → spherical [σ+, π, σ−]:

```
B = [ 1/√2   -i/√2   0 ]
    [   0      0      1 ]
    [-1/√2   -i/√2   0 ]
```

This is the standard AMO physics convention. B is unitary: B⁻¹ = B†.

---

## 5. Physics Pipeline

The complete pipeline, implemented in `physics.py`:

```
Slider inputs (angles, polarization params)
      ↓
① Geometry
   make_beam_frame(θ, φ, χ)  →  {ê₁, ê₂, k̂}
   make_quant_axis(θ_B, φ_B) →  q̂

      ↓
② Polarization state construction (two modes)
   Mode A (Tab 1 — Basis states):
     jones_from_basis_state('sigma_plus'|'pi'|'sigma_minus')
     → applies B† to get lab-frame complex 3-vector E_lab

   Mode B (Tab 2 — Waveplate chain):
     vertical input E₀ = [0, 1]
     → QWP(α₁) → HWP(α₂) → QWP(α₃)   (2x2 Jones matrices)
     → embed_jones_in_lab(jones_2d, ê₁, ê₂)
     → E_lab (complex 3-vector in lab frame)

      ↓
③ Spherical decomposition
   rotate_efield_to_quant_frame(E_lab, θ_B, φ_B)
     → E_quant  (rotate lab frame to quantization axis frame, R⁻¹ = Rᵀ)
   decompose_to_spherical(E_quant)
     → [σ+, π, σ−]  (apply B matrix)

      ↓
④ Absorption strengths (J=0→J=1)
   All CG coefficients equal (1/√3) for J=0→J=1
   absorption[mJ] = |CG|² × |spherical_component|²

      ↓
⑤ Density matrix
   ρᵢⱼ = Eᵢ × conj(Eⱼ)   for i,j ∈ {σ+, π, σ−}
   (outer product — valid for classical driving field)

      ↓
⑥ Poincaré / ellipse
   Stokes: S0,S1,S2,S3 from Jones 2-vector
   Ellipse: Re(jones × exp(iωt)) parametric curve, embedded in 3D

      ↓
⑦ compute_all() — single master function
   Returns a dict with all quantities needed for all plots
```

---

## 6. App Layout

```
┌──────────────────────────────────┬───────────────────────────────┐
│                                  │  CONTROLS  (~38% width)       │
│   3D SCENE  (~60% width)         │                               │
│   60% height                     │  Beam Direction               │
│                                  │    θ, φ, χ sliders            │
│   Objects (always):              │                               │
│   • Unit sphere (r=0.15, origin) │  Quantization Axis            │
│   • Lab x,y,z reference axes    │    θ_B, φ_B sliders           │
│   • k̂ arrow  (tail=-L×k̂,       │                               │
│               tip=r_sphere×k̂)   │  [Basis States|Waveplate]tabs │
│   • Quantization axis arrow      │    Basis: σ+/π/σ− radio       │
│     (tail=origin, tip=L×q̂)      │    Waveplate: α₁,α₂,α₃sliders│
│                                  │                               │
│   Objects (checkbox-gated):      │  Checkboxes:                  │
│   • ê₁, ê₂ axes                 │  ☑ Show ellipse in 3D         │
│     (tail=k̂ tail, subtle/teal)  │  ☑ Show ê₁,ê₂ axes           │
│   • Polarization ellipse         │                               │
│     (centered at k̂ tail, gold)  │                               │
├──────────────────┬───────────────┴───────────────────────────────┤
│  J=0→J=1 LEVEL   │  DENSITY MATRIX    │  [Ellipse|Poincaré] tabs │
│  DIAGRAM         │  3×3 bar chart     │                          │
│  (~30% width)    │  (~30% width)      │  (~38% width)            │
│  permanent       │  permanent         │  switchable              │
│                  │                    │  (ellipse default)       │
└──────────────────┴────────────────────┴──────────────────────────┘
```

### 3D Object Geometry (exact)
| Object | Tail | Tip | Style |
|---|---|---|---|
| k̂ arrow | `−L × k̂` | `r_sphere × k̂` | Bold red, cone head |
| Quantization axis | origin | `L × q̂` | Bold blue, cone head |
| ê₁ axis | `−L × k̂` | `−L×k̂ + 0.5×ê₁` | Teal dotted, subtle |
| ê₂ axis | `−L × k̂` | `−L×k̂ + 0.5×ê₂` | Teal dotted, subtle |
| Polarization ellipse | centered at `−L×k̂` | (parametric) | Gold, thin |
| Atom sphere | origin | (surface r=0.15) | Pale blue, 50% opacity |
| Lab x/y/z axes | origin | `0.9×SCENE_RANGE` | Grey, thin |

Constants: `L_ARROW=1.0`, `R_SPHERE=0.15`, `L_EAXES=0.5`, `SCENE_RANGE=1.6`

### Plot Details
- **J=0→J=1 diagram:** Arrow width ∝ absorption strength, color = σ+/π/σ−
  type, percentage label next to each arrowhead
- **Density matrix:** 3D bar chart — height = |ρᵢⱼ|, color = arg(ρᵢⱼ) via HSV
  colormap, rows/cols labeled [σ+, π, σ−]
- **Polarization ellipse (2D):** In {ê₁, ê₂} plane, normalized, handedness
  annotated (RHC/LHC/Linear from S3)
- **Poincaré sphere:** Unit sphere + S1/S2/S3 labeled axes + current state
  as bold point

### Slider Ranges
| Slider | Min | Max | Step |
|---|---|---|---|
| θ (beam polar) | 0° | 180° | 1° |
| φ (beam azimuthal) | 0° | 360° | 1° |
| χ (beam roll) | 0° | 360° | 1° |
| θ_B (B-field polar) | 0° | 180° | 1° |
| φ_B (B-field azimuthal) | 0° | 360° | 1° |
| α₁ (QWP₁ fast axis) | 0° | 180° | 1° |
| α₂ (HWP fast axis) | 0° | 180° | 1° |
| α₃ (QWP₂ fast axis) | 0° | 180° | 1° |

---

## 7. Color Scheme (Dark Theme)

```python
COLOR_K_ARROW     = '#E63946'   # red      — k̂ vector
COLOR_QUANT       = '#457B9D'   # blue     — quantization axis
COLOR_EAXES       = '#2A9D8F'   # teal     — ê₁ and ê₂
COLOR_ELLIPSE     = '#E9C46A'   # gold     — polarization ellipse
COLOR_SPHERE      = '#A8DADC'   # pale blue — atom cloud
COLOR_LAB_AXES    = '#999999'   # grey     — x, y, z reference lines
COLOR_SIGMA_PLUS  = '#E63946'   # red      — σ+ transitions
COLOR_PI          = '#2A9D8F'   # teal     — π transitions
COLOR_SIGMA_MINUS = '#457B9D'   # blue     — σ− transitions
COLOR_BG          = '#1A1A2E'   # dark navy — background
COLOR_PAPER       = '#16213E'   # slightly lighter — panel background
COLOR_TEXT        = '#E0E0E0'   # light grey — labels
```

---

## 8. What Remains To Be Done

### Immediate (current session — debugging app.py)
Benjamin has the app running locally and is working through visual bugs
one at a time. Known issues have not yet been written down — this document
was created before the debugging session was completed.

**Approach for debugging:** Go one bug at a time. Describe the issue,
confirm understanding, then fix. Do not batch multiple fixes.

### After Debugging
1. **Polish visual details** — camera angles, label positions, arrow sizing
2. **Verify physics** — check known cases:
   - σ+ input, quant axis = lab z, beam along z → 100% σ+, 0% π, 0% σ−
   - Linear polarization along z, beam along x → 100% π
   - Circular polarization, beam along quant axis → 100% σ+ or σ−
3. **Deploy locally to colleagues** for feedback (share localhost via ngrok
   or deploy to Hugging Face Spaces as interim step)
4. **Convert to static Plotly.js site** for GitHub Pages (final deployment)

### Static Site Conversion Notes
When ready to convert:
- `math_helpers.py` → `math_helpers.js` (mechanical, docstrings have JS code)
- `physics.py` → `physics.js` (mechanical, same structure)
- `traces.py` → `traces.js` (Plotly.js API is nearly identical to Python)
- `app.py` layout → `index.html`
- `app.py` callback → `app.js` `updateAll()` function
- Plotly loaded via CDN: `<script src="https://cdn.plot.ly/plotly-2.35.0.min.js">`
- No backend, no server, deploys to GitHub Pages as a single folder

---

## 9. How To Continue In A New Chat

Paste this document at the start of the new conversation, then say:

> "I'm debugging the app. Here is the first issue I see: [describe bug]"

The assistant should:
1. Read the full document before responding
2. Ask clarifying questions if the bug description is ambiguous
3. Identify which file needs to change (`traces.py` for visual bugs,
   `physics.py` for physics bugs, `app.py` for layout/wiring bugs)
4. Confirm understanding before proposing a fix
5. Make targeted edits — do not rewrite whole files

---

## 10. Important Physics Notes

- Light polarization is a **rank-1 spherical tensor** — no Wigner D-matrices
  needed for E1/M1 transitions. The B matrix handles the full transformation.
- The **roll angle χ** is essential for non-circular polarization states —
  it defines the orientation of the transverse frame around k̂.
- The **χ=0 convention:** ê₁ = normalize(ẑ × k̂), i.e. ê₁ is horizontal
  (perpendicular to both lab z and k̂). Singular points (beam along ±z)
  fall back to ê₁ = x̂.
- The **waveplate QWP+HWP+QWP chain** is a complete parametrization of all
  polarization states reachable from vertical linear polarization.
- For **J=0→J=1**, all Clebsch-Gordan coefficients are equal (1/√3), so
  absorption strength is purely proportional to |σ+|², |π|², |σ−|².
- The **density matrix** ρᵢⱼ = Eᵢ·Eⱼ* is the outer product of the spherical
  component vector with itself — valid for a classical driving field.
