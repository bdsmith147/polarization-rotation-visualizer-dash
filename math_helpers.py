# math_helpers.py
# ─────────────────────────────────────────────────────────────────────────────
# Pure math helper library for PolariViz.
#
# CONVERSION RULES (keep these in mind when editing this file):
#   - All matrices are plain nested Python lists (no NumPy arrays)
#   - All vectors are plain Python lists of length 3
#   - No NumPy imports — only the built-in `math` module
#   - No `@` operator — use mat_vec_multiply() and mat_mat_multiply()
#   - Complex numbers are represented as plain 2-lists [real, imag]
#   - No Python `complex` type or `j` notation — use c_make(), c_real(), etc.
#   - All functions take and return plain numbers or plain lists
#   - These rules make every function in this file directly portable to JS
#
# JAVASCRIPT EQUIVALENTS:
#   Each function here has a 1-to-1 JS counterpart (see app.js after conversion)
#   Naming convention: snake_case (Python) → camelCase (JS)
#   e.g. mat_vec_multiply  → matVecMultiply
#        c_mat_vec_multiply → cMatVecMultiply
#        rotation_x         → rotationX
#
# FILE STRUCTURE:
#   1. Scalar helpers
#   2. Complex number primitives
#   3. Real vector operations
#   4. Complex vector operations
#   5. Real matrix operations
#   6. Complex matrix operations
#   7. Real rotation matrices
#   8. Coordinate conversions
#   9. Spherical tensor basis matrices
#  10. Development / debug helpers
#  11. Self-test
# ─────────────────────────────────────────────────────────────────────────────

import math


# ══════════════════════════════════════════════════════════════════════════════
# 1. SCALAR HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def degrees_to_radians(deg):
    """Convert degrees to radians.
    JS: function degreesToRadians(deg) { return deg * Math.PI / 180; }
    """
    return deg * math.pi / 180


def radians_to_degrees(rad):
    """Convert radians to degrees.
    JS: function radiansToDegrees(rad) { return rad * 180 / Math.PI; }
    """
    return rad * 180 / math.pi


def linspace(start, stop, n):
    """Return a list of n evenly spaced values from start to stop (inclusive).
    Equivalent to numpy.linspace(start, stop, n) but returns a plain list.
    JS: function linspace(start, stop, n) {
            if (n < 2) return [start];
            const step = (stop - start) / (n - 1);
            return Array.from({length: n}, (_, i) => start + i * step);
        }
    """
    if n < 2:
        return [start]
    step = (stop - start) / (n - 1)
    return [start + i * step for i in range(n)]


# ══════════════════════════════════════════════════════════════════════════════
# 2. COMPLEX NUMBER PRIMITIVES
#
# Convention: a complex number is a plain 2-list [real, imag].
# Example: 1 + 2i  →  [1, 2]
#          i       →  [0, 1]
#          3       →  [3, 0]
#
# Never use Python's built-in complex type or `j` notation anywhere in this
# codebase. Always use these helpers instead so the JS conversion is direct.
# ══════════════════════════════════════════════════════════════════════════════

def c_make(re, im):
    """Construct a complex number [re, im] from real and imaginary parts.
    JS: function cMake(re, im) { return [re, im]; }
    """
    return [re, im]


def c_real(a):
    """Return the real part of a complex number.
    JS: function cReal(a) { return a[0]; }
    """
    return a[0]


def c_imag(a):
    """Return the imaginary part of a complex number.
    JS: function cImag(a) { return a[1]; }
    """
    return a[1]


def c_conj(a):
    """Complex conjugate: [re, im] → [re, -im].
    JS: function cConj(a) { return [a[0], -a[1]]; }
    """
    return [a[0], -a[1]]


def c_abs(a):
    """Absolute value (modulus) of a complex number: |a| = sqrt(re²+im²).
    JS: function cAbs(a) { return Math.sqrt(a[0]**2 + a[1]**2); }
    """
    return math.sqrt(a[0]**2 + a[1]**2)


def c_abs_sq(a):
    """Squared absolute value: |a|² = re²+im². Avoids a sqrt when not needed.
    JS: function cAbsSq(a) { return a[0]**2 + a[1]**2; }
    """
    return a[0]**2 + a[1]**2


def c_add(a, b):
    """Add two complex numbers.
    JS: function cAdd(a, b) { return [a[0]+b[0], a[1]+b[1]]; }
    """
    return [a[0]+b[0], a[1]+b[1]]


def c_sub(a, b):
    """Subtract complex number b from a.
    JS: function cSub(a, b) { return [a[0]-b[0], a[1]-b[1]]; }
    """
    return [a[0]-b[0], a[1]-b[1]]


def c_mul(a, b):
    """Multiply two complex numbers: (a_re + i*a_im)(b_re + i*b_im).
    JS: function cMul(a, b) {
            return [a[0]*b[0] - a[1]*b[1],
                    a[0]*b[1] + a[1]*b[0]];
        }
    """
    return [a[0]*b[0] - a[1]*b[1],
            a[0]*b[1] + a[1]*b[0]]


def c_div(a, b):
    """Divide complex number a by b.
    JS: function cDiv(a, b) {
            const d = b[0]**2 + b[1]**2;
            return [(a[0]*b[0] + a[1]*b[1])/d,
                    (a[1]*b[0] - a[0]*b[1])/d];
        }
    """
    d = b[0]**2 + b[1]**2
    if d == 0:
        raise ZeroDivisionError("Division by zero complex number.")
    return [(a[0]*b[0] + a[1]*b[1]) / d,
            (a[1]*b[0] - a[0]*b[1]) / d]


def c_scale(a, s):
    """Multiply a complex number a by a real scalar s.
    JS: function cScale(a, s) { return [a[0]*s, a[1]*s]; }
    """
    return [a[0]*s, a[1]*s]


def c_exp(a):
    """Complex exponential: e^(re+i*im) = e^re * (cos(im) + i*sin(im)).
    Useful for computing phase factors like e^{i*phi}.
    JS: function cExp(a) {
            const r = Math.exp(a[0]);
            return [r * Math.cos(a[1]), r * Math.sin(a[1])];
        }
    """
    r = math.exp(a[0])
    return [r * math.cos(a[1]), r * math.sin(a[1])]


def c_phase(phi_rad):
    """Pure phase factor e^{i*phi} = cos(phi) + i*sin(phi).
    Shortcut for c_exp([0, phi_rad]).
    JS: function cPhase(phi) { return [Math.cos(phi), Math.sin(phi)]; }
    """
    return [math.cos(phi_rad), math.sin(phi_rad)]


def c_from_real(x):
    """Promote a real number to a complex number [x, 0].
    JS: function cFromReal(x) { return [x, 0]; }
    """
    return [x, 0]


# ══════════════════════════════════════════════════════════════════════════════
# 3. REAL VECTOR OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════

def dot(a, b):
    """Dot product of two real 3-vectors.
    JS: function dot(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
    """
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def cross(a, b):
    """Cross product of two real 3-vectors. Returns a plain list of length 3.
    JS: function cross(a, b) {
            return [a[1]*b[2] - a[2]*b[1],
                    a[2]*b[0] - a[0]*b[2],
                    a[0]*b[1] - a[1]*b[0]];
        }
    """
    return [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ]


def norm(v):
    """Magnitude (Euclidean length) of a real 3-vector.
    JS: function norm(v) { return Math.sqrt(v[0]**2 + v[1]**2 + v[2]**2); }
    """
    return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)


def normalize(v):
    """Return the unit vector in the direction of real 3-vector v.
    JS: function normalize(v) {
            const mag = norm(v);
            return [v[0]/mag, v[1]/mag, v[2]/mag];
        }
    """
    mag = norm(v)
    if mag == 0:
        raise ValueError("Cannot normalize the zero vector.")
    return [v[0]/mag, v[1]/mag, v[2]/mag]


def scale(v, s):
    """Multiply a real 3-vector by a real scalar s.
    JS: function scale(v, s) { return [v[0]*s, v[1]*s, v[2]*s]; }
    """
    return [v[0]*s, v[1]*s, v[2]*s]


def add(a, b):
    """Add two real 3-vectors.
    JS: function add(a, b) { return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]; }
    """
    return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]


def subtract(a, b):
    """Subtract real 3-vector b from a.
    JS: function subtract(a, b) { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
    """
    return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]


# ══════════════════════════════════════════════════════════════════════════════
# 4. COMPLEX VECTOR OPERATIONS
#
# A complex 3-vector is a plain list of three complex numbers:
#   v = [ [re0,im0], [re1,im1], [re2,im2] ]
# ══════════════════════════════════════════════════════════════════════════════

def c_vec_add(a, b):
    """Add two complex 3-vectors.
    JS: function cVecAdd(a, b) {
            return [cAdd(a[0],b[0]), cAdd(a[1],b[1]), cAdd(a[2],b[2])];
        }
    """
    return [c_add(a[i], b[i]) for i in range(3)]


def c_vec_sub(a, b):
    """Subtract complex 3-vector b from a.
    JS: function cVecSub(a, b) {
            return [cSub(a[0],b[0]), cSub(a[1],b[1]), cSub(a[2],b[2])];
        }
    """
    return [c_sub(a[i], b[i]) for i in range(3)]


def c_vec_scale(v, s):
    """Multiply a complex 3-vector by a complex scalar s.
    JS: function cVecScale(v, s) {
            return [cMul(v[0],s), cMul(v[1],s), cMul(v[2],s)];
        }
    """
    return [c_mul(v[i], s) for i in range(3)]


def c_vec_scale_real(v, s):
    """Multiply a complex 3-vector by a real scalar s.
    JS: function cVecScaleReal(v, s) {
            return [cScale(v[0],s), cScale(v[1],s), cScale(v[2],s)];
        }
    """
    return [c_scale(v[i], s) for i in range(3)]


def c_dot(a, b):
    """Hermitian inner product <a|b> = sum_i conj(a_i) * b_i.
    Returns a complex number [re, im].
    JS: function cDot(a, b) {
            let result = [0, 0];
            for (let i = 0; i < 3; i++)
                result = cAdd(result, cMul(cConj(a[i]), b[i]));
            return result;
        }
    """
    result = [0, 0]
    for i in range(3):
        result = c_add(result, c_mul(c_conj(a[i]), b[i]))
    return result


def c_norm(v):
    """Euclidean norm of a complex 3-vector: sqrt(<v|v>).
    JS: function cNorm(v) { return Math.sqrt(cReal(cDot(v, v))); }
    """
    return math.sqrt(c_real(c_dot(v, v)))


def c_normalize(v):
    """Return the unit complex 3-vector in the direction of v.
    JS: function cNormalize(v) {
            const mag = cNorm(v);
            return cVecScaleReal(v, 1/mag);
        }
    """
    mag = c_norm(v)
    if mag == 0:
        raise ValueError("Cannot normalize the zero complex vector.")
    return c_vec_scale_real(v, 1.0 / mag)


def c_vec_conj(v):
    """Elementwise complex conjugate of a complex 3-vector.
    JS: function cVecConj(v) {
            return [cConj(v[0]), cConj(v[1]), cConj(v[2])];
        }
    """
    return [c_conj(v[i]) for i in range(3)]


def real_to_c_vec(v):
    """Promote a real 3-vector to a complex 3-vector with zero imaginary parts.
    JS: function realToCVec(v) { return [[v[0],0],[v[1],0],[v[2],0]]; }
    """
    return [[v[i], 0] for i in range(3)]


def c_vec_real_part(v):
    """Extract the real parts of a complex 3-vector as a plain real list.
    JS: function cVecRealPart(v) { return [v[0][0], v[1][0], v[2][0]]; }
    """
    return [v[i][0] for i in range(3)]


def c_vec_imag_part(v):
    """Extract the imaginary parts of a complex 3-vector as a plain real list.
    JS: function cVecImagPart(v) { return [v[0][1], v[1][1], v[2][1]]; }
    """
    return [v[i][1] for i in range(3)]


# ══════════════════════════════════════════════════════════════════════════════
# 5. REAL MATRIX OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════

def mat_vec_multiply(M, v):
    """Multiply a real 3x3 matrix M by a real 3-vector v.
    JS: function matVecMultiply(M, v) {
            return [M[0][0]*v[0]+M[0][1]*v[1]+M[0][2]*v[2],
                    M[1][0]*v[0]+M[1][1]*v[1]+M[1][2]*v[2],
                    M[2][0]*v[0]+M[2][1]*v[1]+M[2][2]*v[2]];
        }
    """
    return [
        M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2],
        M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2],
        M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2],
    ]


def mat_mat_multiply(A, B):
    """Multiply two real 3x3 matrices A and B.
    JS: function matMatMultiply(A, B) {
            let R = [[0,0,0],[0,0,0],[0,0,0]];
            for (let i=0;i<3;i++)
                for (let j=0;j<3;j++)
                    for (let k=0;k<3;k++)
                        R[i][j] += A[i][k]*B[k][j];
            return R;
        }
    """
    result = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                result[i][j] += A[i][k] * B[k][j]
    return result


def transpose(M):
    """Transpose a real 3x3 matrix. For rotation matrices equals the inverse.
    JS: function transpose(M) {
            return [[M[0][0],M[1][0],M[2][0]],
                    [M[0][1],M[1][1],M[2][1]],
                    [M[0][2],M[1][2],M[2][2]]];
        }
    """
    return [
        [M[0][0], M[1][0], M[2][0]],
        [M[0][1], M[1][1], M[2][1]],
        [M[0][2], M[1][2], M[2][2]],
    ]


def invert_rotation(R):
    """Invert a real rotation matrix via transposition (valid for orthogonal R only).
    JS: function invertRotation(R) { return transpose(R); }
    """
    return transpose(R)


# ══════════════════════════════════════════════════════════════════════════════
# 6. COMPLEX MATRIX OPERATIONS
#
# A complex 3x3 matrix is a nested list of complex numbers:
#   M = [ [M00, M01, M02],   where each Mij = [re, im]
#         [M10, M11, M12],
#         [M20, M21, M22] ]
# ══════════════════════════════════════════════════════════════════════════════

def c_mat_vec_multiply(M, v):
    """Multiply a complex 3x3 matrix M by a complex 3-vector v.
    JS: function cMatVecMultiply(M, v) {
            return M.map(row =>
                row.reduce((acc, Mij, j) => cAdd(acc, cMul(Mij, v[j])), [0,0])
            );
        }
    """
    result = []
    for row in M:
        val = [0, 0]
        for j in range(3):
            val = c_add(val, c_mul(row[j], v[j]))
        result.append(val)
    return result


def c_mat_mat_multiply(A, B):
    """Multiply two complex 3x3 matrices A and B.
    JS: function cMatMatMultiply(A, B) {
            let R = Array.from({length:3}, () => Array(3).fill(null).map(()=>[0,0]));
            for (let i=0;i<3;i++)
                for (let j=0;j<3;j++)
                    for (let k=0;k<3;k++)
                        R[i][j] = cAdd(R[i][j], cMul(A[i][k], B[k][j]));
            return R;
        }
    """
    result = [[[0, 0] for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                result[i][j] = c_add(result[i][j], c_mul(A[i][k], B[k][j]))
    return result


def c_conjugate_transpose(M):
    """Conjugate transpose (Hermitian adjoint) M† of a complex 3x3 matrix.
    For unitary matrices U, U† = U^{-1}.
    JS: function cConjugateTranspose(M) {
            return [[cConj(M[0][0]),cConj(M[1][0]),cConj(M[2][0])],
                    [cConj(M[0][1]),cConj(M[1][1]),cConj(M[2][1])],
                    [cConj(M[0][2]),cConj(M[1][2]),cConj(M[2][2])]];
        }
    """
    return [
        [c_conj(M[0][0]), c_conj(M[1][0]), c_conj(M[2][0])],
        [c_conj(M[0][1]), c_conj(M[1][1]), c_conj(M[2][1])],
        [c_conj(M[0][2]), c_conj(M[1][2]), c_conj(M[2][2])],
    ]


def invert_unitary(U):
    """Invert a unitary matrix via conjugate transposition (U^{-1} = U†).
    Valid ONLY for unitary matrices where U†U = I.
    Do NOT use this for general complex matrices.
    JS: function invertUnitary(U) { return cConjugateTranspose(U); }
    """
    return c_conjugate_transpose(U)


def real_to_c_mat(M):
    """Promote a real 3x3 matrix to a complex 3x3 matrix with zero imaginary parts.
    JS: function realToCMat(M) { return M.map(row => row.map(x => [x, 0])); }
    """
    return [[[M[i][j], 0] for j in range(3)] for i in range(3)]


# ══════════════════════════════════════════════════════════════════════════════
# 7. REAL ROTATION MATRICES
# ══════════════════════════════════════════════════════════════════════════════

def rotation_x(angle_rad):
    """3x3 rotation matrix for a right-handed rotation about the x-axis.
    JS: function rotationX(a) {
            const [c,s]=[Math.cos(a),Math.sin(a)];
            return [[1,0,0],[0,c,-s],[0,s,c]];
        }
    """
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return [[1,  0,  0],
            [0,  c, -s],
            [0,  s,  c]]


def rotation_y(angle_rad):
    """3x3 rotation matrix for a right-handed rotation about the y-axis.
    JS: function rotationY(a) {
            const [c,s]=[Math.cos(a),Math.sin(a)];
            return [[c,0,s],[0,1,0],[-s,0,c]];
        }
    """
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return [[ c,  0,  s],
            [ 0,  1,  0],
            [-s,  0,  c]]


def rotation_z(angle_rad):
    """3x3 rotation matrix for a right-handed rotation about the z-axis.
    JS: function rotationZ(a) {
            const [c,s]=[Math.cos(a),Math.sin(a)];
            return [[c,-s,0],[s,c,0],[0,0,1]];
        }
    """
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return [[c, -s,  0],
            [s,  c,  0],
            [0,  0,  1]]


# ══════════════════════════════════════════════════════════════════════════════
# 8. COORDINATE CONVERSIONS
# ══════════════════════════════════════════════════════════════════════════════

def spherical_to_cartesian(r, theta_rad, phi_rad):
    """Convert spherical (r, theta, phi) to Cartesian (x, y, z).
    Physics convention: theta = polar angle from +z, phi = azimuthal from +x.
    JS: function sphericalToCartesian(r, theta, phi) {
            return [r*Math.sin(theta)*Math.cos(phi),
                    r*Math.sin(theta)*Math.sin(phi),
                    r*Math.cos(theta)];
        }
    """
    return [
        r * math.sin(theta_rad) * math.cos(phi_rad),
        r * math.sin(theta_rad) * math.sin(phi_rad),
        r * math.cos(theta_rad),
    ]


def cartesian_to_spherical(v):
    """Convert Cartesian [x, y, z] to [r, theta_rad, phi_rad].
    Physics convention: theta = polar angle from +z, phi = azimuthal from +x.
    JS: function cartesianToSpherical(v) {
            const r = norm(v);
            if (r === 0) return [0, 0, 0];
            return [r,
                    Math.acos(Math.max(-1, Math.min(1, v[2]/r))),
                    Math.atan2(v[1], v[0])];
        }
    """
    r = norm(v)
    if r == 0:
        return [0, 0, 0]
    theta = math.acos(max(-1.0, min(1.0, v[2] / r)))
    phi = math.atan2(v[1], v[0])
    return [r, theta, phi]


# ══════════════════════════════════════════════════════════════════════════════
# 9. SPHERICAL TENSOR BASIS MATRICES
#
# These matrices convert between Cartesian (x, y, z) and spherical tensor
# (σ+, π, σ−) representations of vectors, as used in AMO physics.
#
# Convention (rows ordered as σ+, π, σ−):
#
#        ⎡  1/√2   -i/√2   0 ⎤
#   B  = ⎢  0       0      1 ⎥
#        ⎣ -1/√2   -i/√2   0 ⎦
#
# B transforms a Cartesian column vector [x, y, z] → spherical [σ+, π, σ−].
# B is unitary, so B^{-1} = B†  (conjugate transpose).
#
# Usage:
#   v_sph  = cartesian_to_spherical_tensor(v_cart)   → [σ+, π, σ−]
#   v_cart = spherical_tensor_to_cartesian(v_sph)    → [x, y, z]
# ══════════════════════════════════════════════════════════════════════════════

_s = 1.0 / math.sqrt(2)

# B: Cartesian → Spherical  (rows: σ+, π, σ−)
B_CARTESIAN_TO_SPHERICAL = [
    [ [_s,  0], [0, -_s], [0, 0] ],   # σ+ row
    [ [0,   0], [0,   0], [1, 0] ],   # π  row
    [ [-_s, 0], [0, -_s], [0, 0] ],   # σ− row
]

# B†: Spherical → Cartesian  (conjugate transpose of B, computed at import time)
B_SPHERICAL_TO_CARTESIAN = c_conjugate_transpose(B_CARTESIAN_TO_SPHERICAL)


def cartesian_to_spherical_tensor(v_cart):
    """Convert a real Cartesian 3-vector to a complex spherical tensor vector.
    Returns a complex 3-vector [σ+, π, σ−] where each component is [re, im].

    Example:
        v_sph       = cartesian_to_spherical_tensor([0, 0, 1])
        sigma_plus  = v_sph[0]   # [re, im]
        pi_comp     = v_sph[1]   # [re, im]  → [1, 0] for z-hat (pure π)
        sigma_minus = v_sph[2]   # [re, im]

    JS: function cartesianToSphericalTensor(v) {
            return cMatVecMultiply(B_CARTESIAN_TO_SPHERICAL, realToCVec(v));
        }
    """
    return c_mat_vec_multiply(B_CARTESIAN_TO_SPHERICAL, real_to_c_vec(v_cart))


def spherical_tensor_to_cartesian(v_sph):
    """Convert a complex spherical tensor vector [σ+, π, σ−] to Cartesian [x,y,z].
    Returns a complex 3-vector. For a physically valid polarization vector
    the imaginary parts will be negligible — use c_vec_real_part() to extract
    a plain real [x, y, z] if needed.

    JS: function sphericalTensorToCartesian(v) {
            return cMatVecMultiply(B_SPHERICAL_TO_CARTESIAN, v);
        }
    """
    return c_mat_vec_multiply(B_SPHERICAL_TO_CARTESIAN, v_sph)


# ══════════════════════════════════════════════════════════════════════════════
# 10. DEVELOPMENT / DEBUG HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def assert_rotation_matrix(R, tol=1e-6, name="R"):
    """Verify that a real matrix R is a valid rotation matrix (orthogonal, det=+1).
    Raises AssertionError with a descriptive message if not.
    REMOVE calls to this in production.
    """
    RtR = mat_mat_multiply(transpose(R), R)
    identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for i in range(3):
        for j in range(3):
            diff = abs(RtR[i][j] - identity[i][j])
            assert diff < tol, (
                f"assert_rotation_matrix: '{name}' failed orthogonality "
                f"at [{i}][{j}]: R^T R = {RtR[i][j]:.8f}, expected {identity[i][j]}"
            )
    det = (
        R[0][0] * (R[1][1]*R[2][2] - R[1][2]*R[2][1]) -
        R[0][1] * (R[1][0]*R[2][2] - R[1][2]*R[2][0]) +
        R[0][2] * (R[1][0]*R[2][1] - R[1][1]*R[2][0])
    )
    assert abs(det - 1.0) < tol, (
        f"assert_rotation_matrix: '{name}' has det = {det:.8f}, expected +1."
    )


def assert_unitary_matrix(U, tol=1e-6, name="U"):
    """Verify that a complex matrix U is unitary: U†U = I.
    Raises AssertionError with a descriptive message if not.
    REMOVE calls to this in production.
    """
    UdU = c_mat_mat_multiply(c_conjugate_transpose(U), U)
    for i in range(3):
        for j in range(3):
            expected_re = 1.0 if i == j else 0.0
            assert abs(UdU[i][j][0] - expected_re) < tol, (
                f"assert_unitary_matrix: '{name}' failed at [{i}][{j}]: "
                f"U†U = {UdU[i][j][0]:.6f}+{UdU[i][j][1]:.6f}i, "
                f"expected {expected_re}+0i"
            )
            assert abs(UdU[i][j][1]) < tol, (
                f"assert_unitary_matrix: '{name}' non-zero imaginary at [{i}][{j}]: "
                f"{UdU[i][j][1]:.6f}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# 11. SELF-TEST
# Run:  python math_helpers.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Running math_helpers self-test...")

    # ── Scalar helpers ──
    assert abs(degrees_to_radians(180) - math.pi) < 1e-10
    assert abs(radians_to_degrees(math.pi) - 180) < 1e-10
    ls = linspace(0, 1, 5)
    assert ls == [0.0, 0.25, 0.5, 0.75, 1.0], f"linspace: {ls}"
    print("  scalars OK")

    # ── Complex primitives ──
    a = c_make(1, 2)
    b = c_make(3, -1)
    assert c_add(a, b)  == [4, 1]
    assert c_sub(a, b)  == [-2, 3]
    assert c_mul(a, b)  == [5, 5]     # (1+2i)(3-i) = 3-i+6i+2 = 5+5i
    assert c_conj(a)    == [1, -2]
    assert abs(c_abs(c_make(3, 4)) - 5.0) < 1e-10
    assert c_abs_sq(c_make(3, 4)) == 25
    eip = c_exp(c_make(0, math.pi))   # e^{i*pi} = -1
    assert abs(eip[0] - (-1)) < 1e-10 and abs(eip[1]) < 1e-10
    print("  complex primitives OK")

    # ── Real vectors ──
    x = [1, 0, 0]; y = [0, 1, 0]; z = [0, 0, 1]
    assert dot(x, y)        == 0
    assert cross(x, y)      == z
    assert normalize([3, 0, 0]) == [1.0, 0.0, 0.0]
    print("  real vectors OK")

    # ── Complex vectors ──
    cv = [[1, 0], [0, 1], [0, 0]]     # [1, i, 0]
    inner = c_dot(cv, cv)              # <v|v> = 1*1 + (-i)(i) = 1+1 = 2
    assert abs(inner[0] - 2.0) < 1e-10 and abs(inner[1]) < 1e-10
    assert abs(c_norm(cv) - math.sqrt(2)) < 1e-10
    print("  complex vectors OK")

    # ── Real matrices ──
    Rz90 = rotation_z(degrees_to_radians(90))
    result = mat_vec_multiply(Rz90, x)
    assert all(abs(result[i] - y[i]) < 1e-10 for i in range(3))
    assert_rotation_matrix(Rz90, name="Rz(90)")
    for angle in [0, 30, 45, 90, 180]:
        a_rad = degrees_to_radians(angle)
        assert_rotation_matrix(rotation_x(a_rad), name=f"Rx({angle})")
        assert_rotation_matrix(rotation_y(a_rad), name=f"Ry({angle})")
        assert_rotation_matrix(rotation_z(a_rad), name=f"Rz({angle})")
    Rx37 = rotation_x(degrees_to_radians(37))
    RRt  = mat_mat_multiply(Rx37, transpose(Rx37))
    for i in range(3):
        for j in range(3):
            assert abs(RRt[i][j] - (1 if i == j else 0)) < 1e-10
    print("  real matrices OK")

    # ── Complex matrices ──
    assert_unitary_matrix(B_CARTESIAN_TO_SPHERICAL, name="B")
    assert_unitary_matrix(B_SPHERICAL_TO_CARTESIAN, name="B†")
    BdB = c_mat_mat_multiply(B_SPHERICAL_TO_CARTESIAN, B_CARTESIAN_TO_SPHERICAL)
    for i in range(3):
        for j in range(3):
            assert abs(BdB[i][j][0] - (1 if i == j else 0)) < 1e-10
            assert abs(BdB[i][j][1]) < 1e-10
    print("  complex matrices OK")

    # ── Spherical tensor conversions ──
    # z-hat → purely π
    v_z = cartesian_to_spherical_tensor([0, 0, 1])
    assert abs(v_z[0][0]) < 1e-10          # σ+ = 0
    assert abs(v_z[1][0] - 1.0) < 1e-10   # π  = 1
    assert abs(v_z[2][0]) < 1e-10          # σ− = 0

    # x-hat → σ+ = 1/√2, π = 0, σ− = -1/√2
    v_x = cartesian_to_spherical_tensor([1, 0, 0])
    assert abs(v_x[0][0] - _s)    < 1e-10
    assert abs(v_x[1][0])         < 1e-10
    assert abs(v_x[2][0] - (-_s)) < 1e-10

    # round-trip: Cartesian → spherical tensor → Cartesian
    v_orig  = [0.5, -0.3, 0.8]
    v_back  = c_vec_real_part(
                  spherical_tensor_to_cartesian(
                      cartesian_to_spherical_tensor(v_orig)))
    assert all(abs(v_back[i] - v_orig[i]) < 1e-10 for i in range(3)), \
        f"Round-trip failed: {v_back}"
    print("  spherical tensor conversions OK")

    # ── Spherical coordinate conversions ──
    sc = spherical_to_cartesian(1, degrees_to_radians(90), 0)
    assert all(abs(sc[i] - x[i]) < 1e-10 for i in range(3))
    v2  = [1, 2, 3]
    r, th, ph = cartesian_to_spherical(v2)
    v2b = spherical_to_cartesian(r, th, ph)
    assert all(abs(v2b[i] - v2[i]) < 1e-10 for i in range(3))
    print("  spherical coordinate conversions OK")

    print("\nAll tests passed.")