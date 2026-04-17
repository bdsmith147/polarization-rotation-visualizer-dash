# physics.py
# ─────────────────────────────────────────────────────────────────────────────
# Physics computations for PolariViz.
#
# This module computes everything needed to visualize how polarized light
# is absorbed by a J=0 → J=1 atomic transition as a function of beam
# direction, polarization state, and quantization axis orientation.
#
# PIPELINE OVERVIEW:
#   Slider inputs (angles, polarization params)
#         ↓
#   ① Geometry: k̂, {ê₁, ê₂}, quantization axis
#         ↓
#   ② Polarization state: Jones vector in lab frame (complex 3-vector)
#         ↓
#   ③ Spherical decomposition: rotate to quant frame, apply B → (σ+, π, σ−)
#         ↓
#   ④ Absorption: |CG|² × |σ±,π|² per mJ sublevel
#         ↓
#   ⑤ Density matrix: ρᵢⱼ = Eᵢ Eⱼ*
#         ↓
#   ⑥ Poincaré / ellipse: Stokes parameters, polarization ellipse in 3D
#         ↓
#   ⑦ compute_all(): single master function → dict of all plot data
#
# CONVERSION RULES (same as math_helpers.py):
#   - No NumPy — only math module and math_helpers functions
#   - All vectors are plain lists; all complex numbers are [re, im] pairs
#   - Every function is a direct JS conversion target
#
# NOTE: For general F→F' transitions, the spherical decomposition step
#   should be extended with Clebsch-Gordan coefficients for arbitrary F.
#   For J=0→J=1 (rank-1 tensors), the current approach is exact.
# ─────────────────────────────────────────────────────────────────────────────

import math
from math_helpers import (
    # Scalar helpers
    degrees_to_radians, linspace,
    # Real vector ops
    dot, cross, norm, normalize, scale, add, subtract,
    # Real matrix ops
    mat_vec_multiply, mat_mat_multiply, transpose, invert_rotation,
    # Rotation matrices
    rotation_x, rotation_y, rotation_z,
    # Coordinate conversions
    spherical_to_cartesian,
    # Complex scalar ops
    c_make as c, c_add, c_sub, c_mul, c_conj, c_abs, c_abs_sq,
    c_real, c_imag, c_from_real as real_to_c,
    # Complex vector ops
    c_vec_add as c_add_vec, c_vec_sub as c_sub_vec,
    c_vec_scale as c_scale, c_dot, c_norm, c_normalize,
    real_to_c_vec as real_vec_to_c,
    # Complex matrix ops
    c_mat_vec_multiply, c_mat_mat_multiply, c_conjugate_transpose,
    real_to_c_mat as real_mat_to_c,
    # B matrix constants (Cartesian ↔ spherical tensor basis)
    B_CARTESIAN_TO_SPHERICAL, B_SPHERICAL_TO_CARTESIAN,
    # Unitary inversion
    invert_unitary,
    # Debug
    assert_rotation_matrix, assert_unitary_matrix,
)


# ── SECTION 1: GEOMETRY ───────────────────────────────────────────────────────

def make_k_hat(theta_rad, phi_rad):
    """Unit vector in the beam propagation direction k̂.

    Uses physics spherical coordinate convention:
      theta = polar angle from lab +z axis  (0 = beam along +z)
      phi   = azimuthal angle in lab x-y plane

    Returns a real 3-vector [x, y, z].

    JS: function makeKHat(theta, phi) {
            return sphericalToCartesian(1, theta, phi);
        }
    """
    return spherical_to_cartesian(1, theta_rad, phi_rad)


def make_quant_axis(theta_B_rad, phi_B_rad):
    """Unit vector along the quantization axis (B-field direction).

    Same parameterization as make_k_hat but for the B-field orientation.
    The quantization axis defines the z-axis of the atomic frame.

    Returns a real 3-vector [x, y, z].

    JS: function makeQuantAxis(thetaB, phiB) {
            return sphericalToCartesian(1, thetaB, phiB);
        }
    """
    return spherical_to_cartesian(1, theta_B_rad, phi_B_rad)


def rodrigues(v, axis, angle_rad):
    """Rotate vector v by angle_rad around unit vector axis (Rodrigues' formula).

    v_rot = v cosθ + (axis × v) sinθ + axis (axis·v)(1 − cosθ)

    This is more efficient than building a full rotation matrix when only
    one or two vectors need rotating. Converts directly to JS.

    v:         real 3-vector to rotate
    axis:      real unit 3-vector (rotation axis)
    angle_rad: rotation angle in radians (right-hand rule)
    Returns a real 3-vector.

    JS: function rodrigues(v, axis, angleRad) {
            const cosA = Math.cos(angleRad);
            const sinA = Math.sin(angleRad);
            const d    = dot(axis, v);
            const cr   = cross(axis, v);
            return [
                v[0]*cosA + cr[0]*sinA + axis[0]*d*(1-cosA),
                v[1]*cosA + cr[1]*sinA + axis[1]*d*(1-cosA),
                v[2]*cosA + cr[2]*sinA + axis[2]*d*(1-cosA),
            ];
        }
    """
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    d  = dot(axis, v)
    cr = cross(axis, v)
    return [
        v[0]*cos_a + cr[0]*sin_a + axis[0]*d*(1 - cos_a),
        v[1]*cos_a + cr[1]*sin_a + axis[1]*d*(1 - cos_a),
        v[2]*cos_a + cr[2]*sin_a + axis[2]*d*(1 - cos_a),
    ]


def make_beam_frame(theta_rad, phi_rad, chi_rad):
    """Build the orthonormal beam frame {ê₁, ê₂, k̂} using ZYZ Euler angles.

    Convention (χ=0 reference):
      k̂       = direction of beam propagation
      ê₁(χ=0) = normalize(ẑ × k̂)  — horizontal, perpendicular to lab z and k̂
      ê₂(χ=0) = k̂ × ê₁            — vertical, right-handed completion
      χ ≠ 0:  ê₁ and ê₂ rotated by χ around k̂ (roll)

    Singular points (θ ≈ 0 or θ ≈ π, i.e. beam along ±ẑ):
      ê₁ falls back to x̂ (no unique horizontal direction exists)

    The input linear polarization for the waveplate chain is always
    along ê₂ (vertical at χ=0).

    Returns (e1, e2, k_hat) — three real 3-vectors.

    JS: function makeBeamFrame(theta, phi, chi) {
            const kHat = makeKHat(theta, phi);
            const zHat = [0, 0, 1];
            let e1_0 = cross(zHat, kHat);
            if (norm(e1_0) < 1e-6) e1_0 = [1, 0, 0];
            else e1_0 = normalize(e1_0);
            const e2_0 = normalize(cross(kHat, e1_0));
            const e1   = rodrigues(e1_0, kHat, chi);
            const e2   = rodrigues(e2_0, kHat, chi);
            return [e1, e2, kHat];
        }
    """
    k_hat = make_k_hat(theta_rad, phi_rad)
    z_hat = [0, 0, 1]

    # Build ê₁ reference at χ=0
    e1_0 = cross(z_hat, k_hat)
    if norm(e1_0) < 1e-6:
        # Singular point: beam along ±ẑ, fallback to x̂
        e1_0 = [1, 0, 0]
    else:
        e1_0 = normalize(e1_0)

    # Complete right-handed frame
    e2_0 = normalize(cross(k_hat, e1_0))

    # Apply roll χ around k̂
    e1 = rodrigues(e1_0, k_hat, chi_rad)
    e2 = rodrigues(e2_0, k_hat, chi_rad)

    return e1, e2, k_hat


# ── SECTION 2: POLARIZATION STATE CONSTRUCTION ───────────────────────────────

def jones_from_basis_state(basis):
    """Return the lab-frame complex 3-vector E for a pure σ+, π, or σ− state.

    Pure basis states are already defined in the spherical basis, so we
    apply B† (spherical_tensor_to_cartesian) to convert to the Cartesian
    lab-frame representation.

    basis: one of 'sigma_plus', 'pi', 'sigma_minus'
    Returns a complex 3-vector [[re,im], [re,im], [re,im]].

    JS: function jonesFromBasisState(basis) {
            const states = {
                sigma_plus:  [[1,0],[0,0],[0,0]],
                pi:          [[0,0],[1,0],[0,0]],
                sigma_minus: [[0,0],[0,0],[1,0]],
            };
            return sphericalTensorToCartesian(states[basis]);
        }
    """
    states = {
        'sigma_plus':  [c(1,0), c(0,0), c(0,0)],
        'pi':          [c(0,0), c(1,0), c(0,0)],
        'sigma_minus': [c(0,0), c(0,0), c(1,0)],
    }
    if basis not in states:
        raise ValueError(f"basis must be 'sigma_plus', 'pi', or 'sigma_minus'. Got: {basis}")
    return c_mat_vec_multiply(B_SPHERICAL_TO_CARTESIAN, states[basis])


def jones_matrix_qwp(fast_axis_angle_rad):
    """2x2 Jones matrix for a quarter-wave plate (QWP).

    The QWP introduces a π/2 phase retardance between the fast and slow axes.
    fast_axis_angle_rad: angle of fast axis from ê₁ in the transverse plane.

    Matrix in the rotated frame (fast axis along ê₁'):
        J_QWP = exp(iπ/4) × [[1, 0], [0, i]]  (fast axis = ê₁')

    Rotated back to the ê₁/ê₂ frame:
        J = R(-α) J_QWP R(α)

    where R(α) = [[cosα, sinα], [-sinα, cosα]] rotates into fast-axis frame.

    Expanding explicitly (global phase dropped, physically irrelevant):
        J[0][0] = cos²α + i sin²α
        J[0][1] = (1-i) sinα cosα
        J[1][0] = (1-i) sinα cosα
        J[1][1] = sin²α + i cos²α

    Returns a 2x2 complex matrix (nested list of [re,im] pairs).

    JS: function jonesMatrixQwp(alpha) {
            const ca = Math.cos(alpha), sa = Math.sin(alpha);
            const ca2 = ca*ca, sa2 = sa*sa, scP = sa*ca;
            return [
                [[ca2, sa2],    [scP*(1), -scP]],
                [[scP,  -scP],  [sa2, ca2]],
            ];
        }
    """
    ca  = math.cos(fast_axis_angle_rad)
    sa  = math.sin(fast_axis_angle_rad)
    ca2 = ca * ca
    sa2 = sa * sa
    sc  = sa * ca   # sinα cosα

    return [
        [ c(ca2, sa2), c(sc, -sc)  ],
        [ c(sc, -sc),  c(sa2, ca2) ],
    ]


def jones_matrix_hwp(fast_axis_angle_rad):
    """2x2 Jones matrix for a half-wave plate (HWP).

    The HWP introduces a π phase retardance between fast and slow axes.
    fast_axis_angle_rad: angle of fast axis from ê₁ in the transverse plane.

    Expanding explicitly (global phase dropped):
        J[0][0] = cos2α   (real)
        J[0][1] = sin2α   (real)
        J[1][0] = sin2α   (real)
        J[1][1] = -cos2α  (real)

    Note: HWP matrix is purely real (no imaginary components).

    Returns a 2x2 complex matrix (nested list of [re,im] pairs).

    JS: function jonesMatrixHwp(alpha) {
            const c2 = Math.cos(2*alpha), s2 = Math.sin(2*alpha);
            return [
                [[c2, 0],  [s2, 0]],
                [[s2, 0],  [-c2, 0]],
            ];
        }
    """
    c2 = math.cos(2 * fast_axis_angle_rad)
    s2 = math.sin(2 * fast_axis_angle_rad)

    return [
        [ c(c2, 0),  c(s2, 0)  ],
        [ c(s2, 0),  c(-c2, 0) ],
    ]


def jones_mat2_vec2_multiply(M, v):
    """Multiply a 2x2 complex Jones matrix by a 2-vector Jones vector.

    M: 2x2 complex matrix (nested list of [re,im] pairs)
    v: complex 2-vector (list of 2 [re,im] pairs)
    Returns a complex 2-vector.

    JS: function jonesMat2Vec2Multiply(M, v) {
            return [
                cAdd(cMul(M[0][0], v[0]), cMul(M[0][1], v[1])),
                cAdd(cMul(M[1][0], v[0]), cMul(M[1][1], v[1])),
            ];
        }
    """
    return [
        c_add(c_mul(M[0][0], v[0]), c_mul(M[0][1], v[1])),
        c_add(c_mul(M[1][0], v[0]), c_mul(M[1][1], v[1])),
    ]


def jones_mat2_mat2_multiply(A, B):
    """Multiply two 2x2 complex Jones matrices.

    Returns a 2x2 complex matrix.

    JS: function jonesMat2Mat2Multiply(A, B) {
            return [
                [cAdd(cMul(A[0][0],B[0][0]), cMul(A[0][1],B[1][0])),
                 cAdd(cMul(A[0][0],B[0][1]), cMul(A[0][1],B[1][1]))],
                [cAdd(cMul(A[1][0],B[0][0]), cMul(A[1][1],B[1][0])),
                 cAdd(cMul(A[1][0],B[0][1]), cMul(A[1][1],B[1][1]))],
            ];
        }
    """
    return [
        [
            c_add(c_mul(A[0][0], B[0][0]), c_mul(A[0][1], B[1][0])),
            c_add(c_mul(A[0][0], B[0][1]), c_mul(A[0][1], B[1][1])),
        ],
        [
            c_add(c_mul(A[1][0], B[0][0]), c_mul(A[1][1], B[1][0])),
            c_add(c_mul(A[1][0], B[0][1]), c_mul(A[1][1], B[1][1])),
        ],
    ]


def apply_waveplate_chain(alpha1_rad, alpha2_rad, alpha3_rad):
    """Apply QWP(α₁) → HWP(α₂) → QWP(α₃) to vertical input polarization.

    Input state: E₀ = [0, 1] (vertical, along ê₂) as complex 2-vector.
    The chain is applied left to right: QWP₃ · HWP₂ · QWP₁ · E₀

    alpha1_rad: fast axis angle of first QWP
    alpha2_rad: fast axis angle of HWP
    alpha3_rad: fast axis angle of second QWP

    Returns a complex 2-vector [Ex, Ey] in the beam's transverse frame.

    JS: function applyWaveplateChain(a1, a2, a3) {
            const E0    = [[0,0],[1,0]];
            const QWP1  = jonesMatrixQwp(a1);
            const HWP   = jonesMatrixHwp(a2);
            const QWP2  = jonesMatrixQwp(a3);
            const chain = jonesMat2Mat2Multiply(QWP2,
                            jonesMat2Mat2Multiply(HWP, QWP1));
            return jonesMat2Vec2Multiply(chain, E0);
        }
    """
    E0   = [c(0,0), c(1,0)]    # vertical linear polarization
    QWP1 = jones_matrix_qwp(alpha1_rad)
    HWP  = jones_matrix_hwp(alpha2_rad)
    QWP2 = jones_matrix_qwp(alpha3_rad)

    # Build combined Jones matrix: QWP2 · HWP · QWP1
    chain = jones_mat2_mat2_multiply(
                QWP2,
                jones_mat2_mat2_multiply(HWP, QWP1)
            )

    return jones_mat2_vec2_multiply(chain, E0)


def embed_jones_in_lab(jones_2d, e1, e2):
    """Embed a 2D Jones vector into the 3D lab frame.

    E_3D = Ex * ê₁ + Ey * ê₂

    jones_2d: complex 2-vector [Ex, Ey] in beam transverse frame
    e1, e2:   real 3-vectors (from make_beam_frame)
    Returns a complex 3-vector in lab frame.

    JS: function embedJonesInLab(jones2d, e1, e2) {
            const Ex = jones2d[0], Ey = jones2d[1];
            return cAddVec(cScale(realVecToC(e1), Ex),
                           cScale(realVecToC(e2), Ey));
        }
    """
    Ex = jones_2d[0]
    Ey = jones_2d[1]
    # E_3D = Ex * ê₁ + Ey * ê₂
    term1 = c_scale(real_vec_to_c(e1), Ex)
    term2 = c_scale(real_vec_to_c(e2), Ey)
    return c_add_vec(term1, term2)


# ── SECTION 3: SPHERICAL DECOMPOSITION ───────────────────────────────────────

def rotate_efield_to_quant_frame(E_lab, theta_B_rad, phi_B_rad):  # FIXME - Change E_lab to E_beam_frame
    """Rotate lab-frame E-field into the quantization axis frame.

    The quantization axis frame has ẑ_q along the B-field direction,
    reached by R = Rz(φ_B) Ry(θ_B) applied to the lab frame.
    To go from lab → quant frame we apply R⁻¹ = Rᵀ.

    E_lab:      complex 3-vector in lab frame
    theta_B_rad: polar angle of B-field
    phi_B_rad:   azimuthal angle of B-field
    Returns a complex 3-vector in the quantization axis frame.

    JS: function rotateEfieldToQuantFrame(E_lab, thetaB, phiB) {
            const R     = matMatMultiply(rotationZ(phiB), rotationY(thetaB));
            const R_inv = invert_rotation(R);
            return cMatVecMultiply(realMatToC(R_inv), E_lab);
        }
    """  # TODO: Check if the Rz has the theta or the phi argument
    R     = mat_mat_multiply(rotation_z(phi_B_rad), rotation_y(theta_B_rad))
    R_inv = invert_rotation(R)
    return c_mat_vec_multiply(real_mat_to_c(R_inv), E_lab)


def rotate_efield_to_beam_frame(E_input, theta_rad, phi_rad, chi_rad):
    """Rotate the input E-field into the beam frame.
    
    The input E-field is assumed to lie along the z-axis. To go from input
    E-field to that in the beam frame, we apply a rotation matrix 
    R = Rz()
    
    JS: function rotateEfieldToQuantFrame(E_lab, thetaB, phiB) {
            const R     = matMatMultiply(rotationZ(phiB), rotationY(thetaB));
            const R_inv = invert_rotation(R);
            return cMatVecMultiply(realMatToC(R_inv), E_lab);
        }
    """
    Rz1 = rotation_z(theta_rad)
    Ry = rotation_y(phi_rad)
    R_z2 = rotation_z(chi_rad)
    R = mat_mat_multiply(Rz1, mat_mat_multiply(Ry, R_z2))
    return c_mat_vec_multiply(real_mat_to_c(R), E_input)

def decompose_to_spherical(E_quant):
    """Decompose E_quant into spherical (σ+, π, σ−) components.

    Applies the spherical basis transformation B directly:
        [σ+, π, σ−]ᵀ = B @ E_quant

    E_quant must be a complex 3-vector (list of [re,im] pairs).
    This uses B_CARTESIAN_TO_SPHERICAL directly so that complex input
    vectors are handled correctly.

    E_quant: complex 3-vector in quantization axis frame
    Returns a complex 3-vector [sigma_plus, pi_comp, sigma_minus].
    Each element is a [re, im] complex pair.

    JS: function decomposeToSpherical(E_quant) {
            return cMatVecMultiply(B_CARTESIAN_TO_SPHERICAL, E_quant);
        }
    """
    return c_mat_vec_multiply(B_CARTESIAN_TO_SPHERICAL, E_quant)


def compute_spherical_intensities(spherical_components):
    """Compute intensity |amplitude|² of each spherical component.

    spherical_components: complex 3-vector [σ+, π, σ−]
    Returns a dict with real float values:
    {
        'sigma_plus':  float,
        'pi':          float,
        'sigma_minus': float,
        'total':       float,
    }

    JS: function computeSphericalIntensities(sc) {
            const sp = cAbsSq(sc[0]), pi = cAbsSq(sc[1]), sm = cAbsSq(sc[2]);
            return { sigma_plus: sp, pi: pi, sigma_minus: sm,
                     total: sp + pi + sm };
        }
    """
    sp = c_abs_sq(spherical_components[0])
    pi = c_abs_sq(spherical_components[1])
    sm = c_abs_sq(spherical_components[2])
    return {
        'sigma_plus':  sp,
        'pi':          pi,
        'sigma_minus': sm,
        'total':       sp + pi + sm,
    }


def compute_spherical_fractions(spherical_components):
    """Normalize spherical intensities to fractions of total intensity.

    Returns same dict structure as compute_spherical_intensities()
    with values summing to 1.0. Returns zeros if total intensity is zero.

    JS: function computeSphericalFractions(sc) {
            const ints = computeSphericalIntensities(sc);
            const tot  = ints.total || 1.0;
            return { sigma_plus:  ints.sigma_plus  / tot,
                     pi:          ints.pi          / tot,
                     sigma_minus: ints.sigma_minus / tot,
                     total:       1.0 };
        }
    """
    ints  = compute_spherical_intensities(spherical_components)
    total = ints['total'] if ints['total'] > 0 else 1.0
    return {
        'sigma_plus':  ints['sigma_plus']  / total,
        'pi':          ints['pi']          / total,
        'sigma_minus': ints['sigma_minus'] / total,
        'total':       1.0,
    }


# ── SECTION 4: ABSORPTION (J=0 → J=1) ────────────────────────────────────────

def clebsch_gordan_j0_j1(delta_m):
    """Clebsch-Gordan coefficient for J=0,mJ=0 → J=1,mJ=delta_m transition.

    For J=0 → J=1, all three CG coefficients are equal: |CG|² = 1/3.
    This function is explicit for correctness and future extensibility
    to general F → F' transitions.

    delta_m: -1, 0, or +1
    Returns a real float (the CG coefficient, not its square).

    JS: function clebschGordanJ0J1(deltaM) {
            return 1.0 / Math.sqrt(3);
        }
    """
    if delta_m not in (-1, 0, 1):
        raise ValueError(f"delta_m must be -1, 0, or +1. Got: {delta_m}")
    return 1.0 / math.sqrt(3)


def compute_absorption_j0_j1(spherical_components):
    """Compute absorption strength into each mJ sublevel of J=1.

    Absorption strength = |CG(J=0→J=1, ΔmJ)|² × |spherical component|²

    mJ = +1: driven by σ+  (Δm = +1)
    mJ =  0: driven by π   (Δm =  0)
    mJ = -1: driven by σ−  (Δm = -1)

    spherical_components: complex 3-vector [σ+, π, σ−]
    Returns a dict of real floats:
    {
        'mJ_plus1':  float,
        'mJ_0':      float,
        'mJ_minus1': float,
    }

    JS: function computeAbsorptionJ0J1(sc) {
            const cg2 = 1.0 / 3.0;
            return {
                mJ_plus1:  cg2 * cAbsSq(sc[0]),
                mJ_0:      cg2 * cAbsSq(sc[1]),
                mJ_minus1: cg2 * cAbsSq(sc[2]),
            };
        }
    """
    cg_sq = clebsch_gordan_j0_j1(0) ** 2   # = 1/3 for all transitions
    return {
        'mJ_plus1':  cg_sq * c_abs_sq(spherical_components[0]),
        'mJ_0':      cg_sq * c_abs_sq(spherical_components[1]),
        'mJ_minus1': cg_sq * c_abs_sq(spherical_components[2]),
    }


# ── SECTION 5: DENSITY MATRIX ─────────────────────────────────────────────────

def compute_density_matrix(spherical_components):
    """Compute the 3x3 density matrix ρ of the driven J=1 excited state.

    ρᵢⱼ = Eᵢ × conj(Eⱼ)   where i,j ∈ {σ+, π, σ−}

    Diagonal elements (i=j) are real and non-negative (populations).
    Off-diagonal elements are complex (coherences between sublevels).

    spherical_components: complex 3-vector [σ+, π, σ−]
    Returns a 3x3 nested list of [re, im] complex pairs,
    rows and columns ordered [σ+, π, σ−].

    JS: function computeDensityMatrix(sc) {
            return sc.map(Ei =>
                sc.map(Ej => cMul(Ei, cConj(Ej)))
            );
        }
    """
    rho = [[None, None, None],
           [None, None, None],
           [None, None, None]]
    for i in range(3):
        for j in range(3):
            rho[i][j] = c_mul(spherical_components[i],
                               c_conj(spherical_components[j]))
    return rho


def density_matrix_amplitudes(rho):
    """Extract |ρᵢⱼ| (amplitude) for each density matrix element.

    Used for bar heights in the 3x3 visualization.
    Returns a 3x3 nested list of real floats.

    JS: function densityMatrixAmplitudes(rho) {
            return rho.map(row => row.map(el => cAbs(el)));
        }
    """
    return [[c_abs(rho[i][j]) for j in range(3)] for i in range(3)]


def density_matrix_phases(rho):
    """Extract arg(ρᵢⱼ) (phase angle in radians) for each element.

    Used for color/hue in the 3x3 visualization.
    Returns a 3x3 nested list of real floats in [-π, π].

    JS: function densityMatrixPhases(rho) {
            return rho.map(row =>
                row.map(el => Math.atan2(el[1], el[0]))
            );
        }
    """
    return [[math.atan2(rho[i][j][1], rho[i][j][0])
             for j in range(3)]
            for i in range(3)]


# ── SECTION 6: POINCARÉ SPHERE AND POLARIZATION ELLIPSE ──────────────────────

def compute_stokes(jones_2d):
    """Compute Stokes parameters from a 2D Jones vector.

    S0 = |Ex|² + |Ey|²        total intensity
    S1 = |Ex|² - |Ey|²        linear H vs V
    S2 = 2 Re(Ex* Ey)         linear ±45°
    S3 = 2 Im(Ex* Ey)         circular (RHC vs LHC)

    The Poincaré sphere point is (S1/S0, S2/S0, S3/S0) on the unit sphere.

    jones_2d: complex 2-vector [Ex, Ey]
    Returns a dict of real floats:
    {
        'S0': float,   'S1': float,   'S2': float,   'S3': float,
        'poincare': [float, float, float],   # normalized point on unit sphere
    }

    JS: function computeStokes(jones2d) {
            const Ex = jones2d[0], Ey = jones2d[1];
            const S0 = cAbsSq(Ex) + cAbsSq(Ey);
            const S1 = cAbsSq(Ex) - cAbsSq(Ey);
            const ExcEy = cMul(cConj(Ex), Ey);
            const S2 = 2 * ExcEy[0];
            const S3 = 2 * ExcEy[1];
            const norm = S0 > 0 ? S0 : 1.0;
            return { S0, S1, S2, S3,
                     poincare: [S1/norm, S2/norm, S3/norm] };
        }
    """
    Ex = jones_2d[0]
    Ey = jones_2d[1]
    S0 = c_abs_sq(Ex) + c_abs_sq(Ey)
    S1 = c_abs_sq(Ex) - c_abs_sq(Ey)
    ExcEy = c_mul(c_conj(Ex), Ey)
    S2 = 2 * ExcEy[0]   # 2 Re(Ex* Ey)
    S3 = 2 * ExcEy[1]   # 2 Im(Ex* Ey)
    norm_s = S0 if S0 > 0 else 1.0
    return {
        'S0': S0,
        'S1': S1,
        'S2': S2,
        'S3': S3,
        'poincare': [S1/norm_s, S2/norm_s, S3/norm_s],
    }


def compute_polarization_ellipse(jones_2d, n_points=100):
    """Compute the polarization ellipse in the beam's transverse plane.

    The E-field traces an ellipse as the wave propagates:
        E(t) = Re( jones_2d × exp(iωt) )   for t ∈ [0, 2π]

    jones_2d: complex 2-vector [Ex, Ey]
    n_points: number of points on the ellipse
    Returns (e1_vals, e2_vals) — two plain lists of n_points real floats
    representing the ellipse coordinates along ê₁ and ê₂.

    JS: function computePolarizationEllipse(jones2d, nPoints=100) {
            const ts = linspace(0, 2*Math.PI, nPoints);
            const e1_vals = [], e2_vals = [];
            for (const t of ts) {
                const phase = [Math.cos(t), Math.sin(t)];
                const E = [cMul(jones2d[0], phase), cMul(jones2d[1], phase)];
                e1_vals.push(E[0][0]);
                e2_vals.push(E[1][0]);
            }
            return [e1_vals, e2_vals];
        }
    """
    ts = linspace(0, 2 * math.pi, n_points)
    e1_vals = []
    e2_vals = []
    for t in ts:
        phase = c(math.cos(t), math.sin(t))
        E1 = c_mul(jones_2d[0], phase)
        E2 = c_mul(jones_2d[1], phase)
        e1_vals.append(E1[0])   # real part
        e2_vals.append(E2[0])   # real part
    return e1_vals, e2_vals


def embed_ellipse_in_lab(e1_vals, e2_vals, e1_vec, e2_vec):
    """Embed the 2D polarization ellipse into the 3D lab frame.

    Maps each (e1, e2) point to a 3D lab-frame coordinate:
        P_3D = e1_val * ê₁ + e2_val * ê₂

    e1_vals, e2_vals: plain lists of real floats (from compute_polarization_ellipse)
    e1_vec, e2_vec:   real 3-vectors (from make_beam_frame)
    Returns (xs, ys, zs) — three plain lists of floats for a Plotly trace.

    JS: function embedEllipseInLab(e1Vals, e2Vals, e1Vec, e2Vec) {
            return e1Vals.map((e1, i) => {
                const e2 = e2Vals[i];
                return [e1*e1Vec[0]+e2*e2Vec[0],
                        e1*e1Vec[1]+e2*e2Vec[1],
                        e1*e1Vec[2]+e2*e2Vec[2]];
            });
        }
    """
    xs, ys, zs = [], [], []
    for i in range(len(e1_vals)):
        e1 = e1_vals[i]
        e2 = e2_vals[i]
        xs.append(e1 * e1_vec[0] + e2 * e2_vec[0])
        ys.append(e1 * e1_vec[1] + e2 * e2_vec[1])
        zs.append(e1 * e1_vec[2] + e2 * e2_vec[2])
    return xs, ys, zs


# ── SECTION 7: MASTER PIPELINE FUNCTION ──────────────────────────────────────

def compute_all(
    # Beam geometry
    theta_rad, phi_rad, chi_rad,
    # Quantization axis
    theta_B_rad, phi_B_rad,
    # Polarization input mode
    input_mode,         # 'basis' or 'waveplate'
    # Tab 1 params (used if input_mode == 'basis')
    basis_state=None,   # 'sigma_plus', 'pi', 'sigma_minus'
    # Tab 2 params (used if input_mode == 'waveplate')
    alpha1_rad=0.0, alpha2_rad=0.0, alpha3_rad=0.0,
    # Ellipse resolution
    n_ellipse_points=50,
):
    """Single master function called by the Dash callback.

    Runs the complete physics pipeline and returns a dict containing
    everything needed to update all plots in the app.

    Returns:
    {
        # ── Geometry (for 3D plot) ────────────────────────────────────────
        'k_hat':        real 3-vector,       # beam direction
        'e1':           real 3-vector,       # horizontal transverse axis
        'e2':           real 3-vector,       # vertical transverse axis
        'quant_axis':   real 3-vector,       # quantization axis (B-field)
        'E_lab':        complex 3-vector,    # E-field in lab frame

        # ── Polarization state ────────────────────────────────────────────
        'jones_2d':     complex 2-vector,    # Jones vector in beam frame
                                             # (None for basis state input)
        'stokes':       dict,                # S0,S1,S2,S3 + poincare point

        # ── Polarization ellipse ──────────────────────────────────────────
        'ellipse_e1':   list of floats,      # ellipse along ê₁
        'ellipse_e2':   list of floats,      # ellipse along ê₂
        'ellipse_xs':   list of floats,      # ellipse x in lab frame
        'ellipse_ys':   list of floats,      # ellipse y in lab frame
        'ellipse_zs':   list of floats,      # ellipse z in lab frame

        # ── Spherical decomposition ───────────────────────────────────────
        'spherical':    complex 3-vector,    # [σ+, π, σ−] components
        'intensities':  dict,                # |σ+|², |π|², |σ−|², total
        'fractions':    dict,                # normalized to sum = 1

        # ── Absorption (J=0→J=1) ─────────────────────────────────────────
        'absorption':   dict,                # mJ_plus1, mJ_0, mJ_minus1

        # ── Density matrix ────────────────────────────────────────────────
        'rho':          3x3 complex matrix,  # full density matrix
        'rho_amps':     3x3 real matrix,     # |ρᵢⱼ| for bar heights
        'rho_phases':   3x3 real matrix,     # arg(ρᵢⱼ) for colors
    }
    """
    # ── Step 1: Geometry ─────────────────────────────────────────────────────
    e1, e2, k_hat = make_beam_frame(theta_rad, phi_rad, chi_rad)
    quant_axis    = make_quant_axis(theta_B_rad, phi_B_rad)

    # ── Step 2: Polarization state ────────────────────────────────────────────
    if input_mode == 'basis':
        if basis_state is None:
            raise ValueError("basis_state must be provided when input_mode='basis'")
        E_lab = jones_from_basis_state(basis_state)  # FIXME - change E_lab to E_beam_frame
        jones_2d = None
        # For Stokes/ellipse: project E_lab back to 2D beam frame
        # (approximate — basis states are defined in spherical basis, not beam frame)
        # Use the real parts of projections for visualization purposes
        Ex = c_dot(real_vec_to_c(e1), E_lab)
        Ey = c_dot(real_vec_to_c(e2), E_lab)
        # print(E_lab)
        jones_2d_for_ellipse = [Ex, Ey]
        
        # FIXME: Doesn't work for pi-pol because field is in z-component in the lab frame
        # jones_2d_for_ellipse = E_beam_frame  

    elif input_mode == 'waveplate':
        jones_2d = apply_waveplate_chain(alpha1_rad, alpha2_rad, alpha3_rad)
        E_lab    = embed_jones_in_lab(jones_2d, e1, e2)
        jones_2d_for_ellipse = jones_2d

    else:
        raise ValueError(f"input_mode must be 'basis' or 'waveplate'. Got: {input_mode}")

    # ── Step 3: Stokes parameters and polarization ellipse ───────────────────
    stokes               = compute_stokes(jones_2d_for_ellipse)
    e1_vals, e2_vals     = compute_polarization_ellipse(
                               jones_2d_for_ellipse, n_ellipse_points)
    ellipse_xs, ellipse_ys, ellipse_zs = embed_ellipse_in_lab(
                               e1_vals, e2_vals, e1, e2)

    # ── Step 4: Spherical decomposition ──────────────────────────────────────
    E_quant    = rotate_efield_to_quant_frame(E_lab, theta_B_rad, phi_B_rad)
    spherical  = decompose_to_spherical(E_quant)
    intensities = compute_spherical_intensities(spherical)
    fractions   = compute_spherical_fractions(spherical)

    # ── Step 5: Absorption ────────────────────────────────────────────────────
    absorption = compute_absorption_j0_j1(spherical)

    # ── Step 6: Density matrix ────────────────────────────────────────────────
    rho        = compute_density_matrix(spherical)
    rho_amps   = density_matrix_amplitudes(rho)
    rho_phases = density_matrix_phases(rho)

    return {
        # Geometry
        'k_hat':        k_hat,
        'e1':           e1,
        'e2':           e2,
        'quant_axis':   quant_axis,
        'E_lab':        E_lab,
        # Polarization
        'jones_2d':     jones_2d,
        'stokes':       stokes,
        # Ellipse
        'ellipse_e1':   e1_vals,
        'ellipse_e2':   e2_vals,
        'ellipse_xs':   ellipse_xs,
        'ellipse_ys':   ellipse_ys,
        'ellipse_zs':   ellipse_zs,
        # Spherical decomposition
        'spherical':    spherical,
        'intensities':  intensities,
        'fractions':    fractions,
        # Absorption
        'absorption':   absorption,
        # Density matrix
        'rho':          rho,
        'rho_amps':     rho_amps,
        'rho_phases':   rho_phases,
    }


# ── QUICK SELF-TEST ───────────────────────────────────────────────────────────
# Run directly to verify the physics pipeline:
#   python physics.py

if __name__ == "__main__":
    print("Running physics.py self-test...")

    # ── Geometry ──────────────────────────────────────────────────────────────
    # k̂ along +z when θ=0
    k = make_k_hat(0, 0)
    assert abs(k[2] - 1.0) < 1e-10, f"k_hat(0,0) should be z-hat: {k}"
    
    # k̂ along +x when θ=90°, φ=0
    k = make_k_hat(degrees_to_radians(90), 0)
    assert abs(k[0] - 1.0) < 1e-10, f"k_hat(90,0) should be x-hat: {k}"
    
    # Beam frame at θ=0 should give e1=x̂, e2=ŷ, k̂=ẑ (at χ=0)
    e1, e2, k_hat = make_beam_frame(0, 0, 0)
    assert abs(e1[0] - 1.0) < 1e-10, f"e1 at theta=0 should be x-hat: {e1}"
    assert abs(e2[1] - 1.0) < 1e-10, f"e2 at theta=0 should be y-hat: {e2}"
    assert abs(k_hat[2] - 1.0) < 1e-10, f"k_hat at theta=0 should be z-hat: {k_hat}"
    
    # Frame should be orthonormal
    assert abs(dot(e1, e2))    < 1e-10, "e1 · e2 ≠ 0"
    assert abs(dot(e1, k_hat)) < 1e-10, "e1 · k̂ ≠ 0"
    assert abs(dot(e2, k_hat)) < 1e-10, "e2 · k̂ ≠ 0"
    assert abs(norm(e1) - 1.0) < 1e-10, "e1 not unit"
    assert abs(norm(e2) - 1.0) < 1e-10, "e2 not unit"


    # Rodrigues: rotating x̂ by 90° around ẑ should give ŷ
    result = rodrigues([1,0,0], [0,0,1], degrees_to_radians(90))
    assert abs(result[1] - 1.0) < 1e-10, f"Rodrigues failed: {result}"
    
    # ── Waveplate matrices ────────────────────────────────────────────────────
    # QWP at 45°: converts vertical linear → circular (equal amplitudes Ex, Ey)
    qwp45 = jones_matrix_qwp(degrees_to_radians(45))
    E_vert = [c(0,0), c(1,0)]
    E_circ = jones_mat2_vec2_multiply(qwp45, E_vert)
    assert abs(c_abs(E_circ[0]) - c_abs(E_circ[1])) < 1e-10, \
        f"QWP(45°) on vertical should give circular: {E_circ}"
    # HWP at 45°: should rotate vertical → horizontal
    hwp45 = jones_matrix_hwp(degrees_to_radians(45))
    E_horiz = jones_mat2_vec2_multiply(hwp45, E_vert)
    assert abs(c_abs(E_horiz[0]) - 1.0) < 1e-10, \
        f"HWP(45°) on vertical should give horizontal: {E_horiz}"
    assert abs(c_abs(E_horiz[1])) < 1e-10, \
        f"HWP(45°) on vertical: Ey should be 0: {E_horiz}"

    # ── Spherical decomposition ───────────────────────────────────────────────
    # σ+ basis state should give σ+ fraction = 1
    E_sp = jones_from_basis_state('sigma_plus')
    E_q  = rotate_efield_to_quant_frame(E_sp, 0, 0)   # quant axis = lab z
    sph  = decompose_to_spherical(E_q)
    ints = compute_spherical_intensities(sph)
    assert abs(ints['sigma_plus'] - 1.0) < 1e-6, \
        f"σ+ state should have σ+ intensity=1: {ints}"
    assert abs(ints['pi'])          < 1e-6, f"σ+ state π should be 0: {ints}"
    assert abs(ints['sigma_minus']) < 1e-6, f"σ+ state σ− should be 0: {ints}"

    # π basis state should give π fraction = 1
    E_pi = jones_from_basis_state('pi')
    E_q  = rotate_efield_to_quant_frame(E_pi, 0, 0)
    sph  = decompose_to_spherical(E_q)
    ints = compute_spherical_intensities(sph)
    assert abs(ints['pi'] - 1.0) < 1e-6, \
        f"π state should have π intensity=1: {ints}"

    # ── Stokes parameters ─────────────────────────────────────────────────────
    # Horizontal linear: S1=1, S2=S3=0
    E_h = [c(1,0), c(0,0)]
    st  = compute_stokes(E_h)
    assert abs(st['S1'] - 1.0) < 1e-10, f"Horizontal: S1 should be 1: {st}"
    assert abs(st['S2'])        < 1e-10, f"Horizontal: S2 should be 0: {st}"
    assert abs(st['S3'])        < 1e-10, f"Horizontal: S3 should be 0: {st}"

    # RHC: S3=1, S1=S2=0
    s = 1/math.sqrt(2)
    E_rh = [c(s,0), c(0, s)]
    st   = compute_stokes(E_rh)
    assert abs(st['S3'] - 1.0) < 1e-10, f"RHC: S3 should be 1: {st}"
    assert abs(st['S1'])        < 1e-10, f"RHC: S1 should be 0: {st}"

    # ── Density matrix ────────────────────────────────────────────────────────
    # For a pure σ+ state, ρ₀₀=1 and all others=0
    E_sp  = jones_from_basis_state('sigma_plus')
    E_q   = rotate_efield_to_quant_frame(E_sp, 0, 0)
    sph   = decompose_to_spherical(E_q)
    rho   = compute_density_matrix(sph)
    amps  = density_matrix_amplitudes(rho)
    assert abs(amps[0][0] - 1.0) < 1e-6, \
        f"σ+ density matrix: ρ₀₀ should be 1: {amps[0][0]}"
    assert abs(amps[0][1]) < 1e-6, \
        f"σ+ density matrix: ρ₀₁ should be 0: {amps[0][1]}"

    # ── Full pipeline ─────────────────────────────────────────────────────────
    result = compute_all(
        theta_rad=degrees_to_radians(45),
        phi_rad=degrees_to_radians(30),
        chi_rad=degrees_to_radians(0),
        theta_B_rad=degrees_to_radians(0),
        phi_B_rad=degrees_to_radians(0),
        input_mode='waveplate',
        alpha1_rad=degrees_to_radians(0),
        alpha2_rad=degrees_to_radians(45),
        alpha3_rad=degrees_to_radians(0),
    )
    # Fractions should sum to 1
    fracs = result['fractions']
    total = fracs['sigma_plus'] + fracs['pi'] + fracs['sigma_minus']
    assert abs(total - 1.0) < 1e-6, f"Fractions should sum to 1: {total}"

    # Density matrix should be 3x3
    assert len(result['rho']) == 3 and len(result['rho'][0]) == 3, \
        "rho should be 3x3"

    # Ellipse should have correct number of points
    assert len(result['ellipse_xs']) == 50, \
        f"Ellipse should have 100 points: {len(result['ellipse_xs'])}"

    # Poincaré point should be on or inside unit sphere
    p = result['stokes']['poincare']
    r = math.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
    assert r <= 1.0 + 1e-6, f"Poincaré point should be on unit sphere: r={r}"

    print("All tests passed.")