"""
verify_berry_phase.py
=====================
Self-contained verification of the Berry phase = 1/phi^4 result on the
600-cell polytope (binary icosahedral group 2I as unit quaternions on S^3).

What this script does:
  1. Builds all 120 vertices of the 600-cell.
  2. Builds adjacency (720 edges, edge length^2 = 2 - phi).
  3. Finds all 1200 triangular faces.
  4. For each triangle, computes the SO(3) holonomy matrix via parallel
     transport on S^3, using the quaternionic basis {T_i, T_j, T_k}
     where T_i is the Hopf fiber direction.
  5. Verifies face-transitivity: all triangles give the same holonomy angle.
  6. Verifies Omega = 3*arccos(1/sqrt(5)) - pi and cos(Omega) = 11*sqrt(5)/25.
  7. Computes the fiber fraction |n . T_i(p)| for each triangle, where n is the
     triangle normal in T_p S^3.
  8. Verifies the 5 distinct fiber fraction values with correct multiplicities:
     {0, 1/(2*phi), 1/2, phi/2, 1} x {160, 320, 320, 320, 80}.
  9. Decomposes the holonomy into fiber and base (Berry) components using the
     horizontal 2x2 block of H in the quaternionic basis.
  10. Verifies mean Berry phase = 1/phi^4 to better than 0.1%.

The key insight: The quaternionic basis {T_i, T_j, T_k} at each point p
naturally decomposes T_p S^3 = fiber (T_i) + horizontal (T_j, T_k).
The Berry phase is the rotation angle of the horizontal 2x2 block of
the holonomy matrix in this basis.

Dependencies: numpy only.
Encoding: ASCII only (safe for Windows cp1252).

Author: Razvan-Constantin Anghelina
"""

import numpy as np
from itertools import product as iterproduct

# ============================================================================
# CONSTANTS
# ============================================================================
a1 = 5
phi = (1.0 + np.sqrt(5.0)) / 2.0
b1 = a1 + 1  # = 6
N = 120       # order of 2I = number of 600-cell vertices

TOL = 1e-10   # tolerance for floating-point comparisons

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def pct_error(predicted, observed):
    """Percentage error: 100 * |predicted - observed| / |observed|."""
    return 100.0 * abs(predicted - observed) / abs(observed)


def pass_fail(condition):
    """Return PASS or FAIL string."""
    return "PASS" if condition else "FAIL"


# ============================================================================
# SECTION 1: BUILD THE 120 VERTICES OF THE 600-CELL
# ============================================================================
# The 600-cell vertices are the 120 elements of the binary icosahedral group
# 2I, represented as unit quaternions (w, x, y, z) on S^3.
#
# Three families:
#   8  axis vertices:   all permutations of (+/-1, 0, 0, 0)
#   16 half-integer:    all (+/-1/2, +/-1/2, +/-1/2, +/-1/2)
#   96 golden:          even permutations of (0, +/-1/2, +/-phi/2, +/-1/(2*phi))

def build_600cell_vertices():
    """Build the 120 vertices of the 600-cell as unit quaternions on S^3."""
    elements = set()

    def add(q):
        """Add q and -q (rounded for dedup)."""
        q_r = tuple(round(x, 10) for x in q)
        elements.add(q_r)
        elements.add(tuple(-x for x in q_r))

    # --- 8 axis vertices: permutations of (+/-1, 0, 0, 0) ---
    for i in range(4):
        v = [0.0, 0.0, 0.0, 0.0]
        v[i] = 1.0
        add(v)

    # --- 16 half-integer vertices: all (+/-1/2, +/-1/2, +/-1/2, +/-1/2) ---
    for s0 in [0.5, -0.5]:
        for s1 in [0.5, -0.5]:
            for s2 in [0.5, -0.5]:
                for s3 in [0.5, -0.5]:
                    add([s0, s1, s2, s3])

    # --- 96 golden vertices ---
    # Even permutations of the absolute values (0, 1/2, 1/(2*phi), phi/2),
    # with all sign choices on the nonzero entries.
    phi_p = (1.0 - np.sqrt(5.0)) / 2.0  # conjugate golden ratio
    abs_vals = [0.0, 0.5, abs(phi_p) / 2.0, phi / 2.0]

    even_perms = [
        (0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2),
        (1, 0, 3, 2), (1, 2, 0, 3), (1, 3, 2, 0),
        (2, 0, 1, 3), (2, 1, 3, 0), (2, 3, 0, 1),
        (3, 0, 2, 1), (3, 1, 0, 2), (3, 2, 1, 0),
    ]

    for perm in even_perms:
        base = [abs_vals[perm[k]] for k in range(4)]
        non_zero = [i for i in range(4) if abs(base[i]) > 1e-10]
        for signs in iterproduct([1.0, -1.0], repeat=len(non_zero)):
            v = list(base)
            for idx, s in zip(non_zero, signs):
                v[idx] *= s
            add(v)

    return np.array(sorted(elements))


# ============================================================================
# SECTION 2: BUILD ADJACENCY (EDGES)
# ============================================================================
# Two vertices are connected by an edge iff their inner product = phi/2,
# equivalently |q1 - q2|^2 = 2 - phi.

def build_edges(verts):
    """Find all edges using inner product criterion: p.q = phi/2."""
    n = len(verts)
    IP = verts @ verts.T  # inner product matrix
    edges = []
    adj = [set() for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if abs(IP[i, j] - phi / 2.0) < 0.01:
                edges.append((i, j))
                adj[i].add(j)
                adj[j].add(i)

    return edges, adj


# ============================================================================
# SECTION 3: FIND ALL 1200 TRIANGULAR FACES
# ============================================================================

def find_triangles(edges, adj):
    """Find all triangular faces (triples of mutually adjacent vertices)."""
    triangles = []
    for (i, j) in edges:
        common = adj[i] & adj[j]
        for k in common:
            if k > j:
                triangles.append((i, j, k))
    return triangles


# ============================================================================
# SECTION 4: PARALLEL TRANSPORT AND HOLONOMY ON S^3
# ============================================================================
# Parallel transport of tangent vector v from T_p(S^3) to T_q(S^3)
# along the geodesic (Levi-Civita connection on round S^3):
#
#   P_{p->q}(v) = v - [v.q / (1 + p.q)] * (p + q)

def parallel_transport(v, p, q):
    """Parallel transport v in T_p S^3 to T_q S^3 along geodesic."""
    c = np.dot(p, q)
    if abs(1.0 + c) < 1e-10:
        return -v  # antipodal case
    return v - (np.dot(v, q) / (1.0 + c)) * (p + q)


def quaternionic_basis(p):
    """
    Quaternionic orthonormal basis for T_p S^3.

    Given p = (w, x, y, z) on S^3, the three tangent vectors are:
      T_i(p) = p * i = (-x, w, -z, y)     [Hopf fiber direction]
      T_j(p) = p * j = (-y, z, w, -x)
      T_k(p) = p * k = (-z, -y, x, w)

    These are mutually orthonormal and all orthogonal to p.
    Returns a 3x4 matrix with rows [T_i, T_j, T_k].
    """
    w, x, y, z = p
    Ti = np.array([-x, w, -z, y])
    Tj = np.array([-y, z, w, -x])
    Tk = np.array([-z, -y, x, w])
    return np.array([Ti, Tj, Tk])


def holonomy_matrix_quat(p, q, r):
    """
    Compute the 3x3 holonomy matrix for parallel transport around the
    geodesic triangle p -> q -> r -> p on S^3, expressed in the
    quaternionic basis {T_i, T_j, T_k} at p.

    The first basis vector T_i is the Hopf fiber direction, so:
      H[0,0] component = fiber-fiber
      H[1:3, 1:3] block = horizontal (base) rotation = Berry phase
    """
    basis = quaternionic_basis(p)  # 3x4, rows = [T_i, T_j, T_k]

    H = np.zeros((3, 3))
    for col in range(3):
        v = basis[col].copy()

        # Transport around triangle: p -> q -> r -> p
        v = parallel_transport(v, p, q)
        v = parallel_transport(v, q, r)
        v = parallel_transport(v, r, p)

        # Express result in the basis at p
        for row in range(3):
            H[row, col] = np.dot(v, basis[row])

    return H


# ============================================================================
# SECTION 5: TRIANGLE NORMAL AND FIBER FRACTION
# ============================================================================
# The 4D cross product of (p, u, w) gives a vector perpendicular to all three.
# For a triangle at vertex p with tangent vectors u = proj_p(q), w = proj_p(r),
# the triangle normal in T_p S^3 is the cross product direction.
#
# The fiber fraction is |n_hat . T_i(p)|, where n_hat is the unit normal
# to the triangle plane in T_p S^3.

def triple_cross_4d(p, u, w):
    """4D cross product: n = *(p ^ u ^ w), perpendicular to all three."""
    M = np.array([p, u, w])  # 3x4
    n = np.zeros(4)
    for i in range(4):
        cols = [j for j in range(4) if j != i]
        n[i] = ((-1.0) ** i) * np.linalg.det(M[:, cols])
    return n


def hopf_fiber_tangent(p):
    """Hopf fiber tangent T_i(p) = p * i = (-x, w, -z, y)."""
    w, x, y, z = p
    return np.array([-x, w, -z, y])


# ============================================================================
# MAIN COMPUTATION
# ============================================================================

print("=" * 72)
print("  BERRY PHASE ON THE 600-CELL  --  VERIFICATION SCRIPT")
print("=" * 72)
print()

# ------------------------------------------------------------------
# Step 1: Build vertices
# ------------------------------------------------------------------
print("STEP 1: Building 600-cell vertices...")
verts = build_600cell_vertices()
n_verts = len(verts)

# Verify all vertices are on the unit sphere
norms = np.linalg.norm(verts, axis=1)
all_unit = np.all(np.abs(norms - 1.0) < TOL)

print("  Number of vertices: %d (expected 120)" % n_verts)
print("  All on unit S^3:    %s" % ("YES" if all_unit else "NO"))
verts_pass = (n_verts == 120) and all_unit
print("  Verdict: %s" % pass_fail(verts_pass))
print()

# ------------------------------------------------------------------
# Step 2: Build edges
# ------------------------------------------------------------------
edge_len_sq = 2.0 - phi
print("STEP 2: Building edges (edge length^2 = 2 - phi = %.10f)..." % edge_len_sq)
edges, adj = build_edges(verts)
n_edges = len(edges)

print("  Number of edges: %d (expected 720)" % n_edges)
edges_pass = (n_edges == 720)
print("  Verdict: %s" % pass_fail(edges_pass))
print()

# ------------------------------------------------------------------
# Step 3: Find triangular faces
# ------------------------------------------------------------------
print("STEP 3: Finding triangular faces...")
triangles = find_triangles(edges, adj)
n_triangles = len(triangles)

print("  Number of triangles: %d (expected 1200)" % n_triangles)
triangles_pass = (n_triangles == 1200)
print("  Verdict: %s" % pass_fail(triangles_pass))
print()

# ------------------------------------------------------------------
# Step 4: Compute holonomy for all triangles
# ------------------------------------------------------------------
print("STEP 4: Computing SO(3) holonomy in quaternionic basis for all %d triangles..." % n_triangles)
print("  (Quaternionic basis: T_i = fiber, T_j/T_k = horizontal)")
print("  (This may take a minute...)")

omega_values = np.zeros(n_triangles)
hol_matrices = []
fiber_components = np.zeros(n_triangles)  # |axis . e_fiber| in quat basis
fiber_fractions = np.zeros(n_triangles)   # |normal . T_i(p)| from 4D cross product

for t_idx, (i, j, k) in enumerate(triangles):
    p, q, r = verts[i], verts[j], verts[k]

    # --- Holonomy matrix in quaternionic basis ---
    H = holonomy_matrix_quat(p, q, r)
    hol_matrices.append(H)

    # --- Rotation angle from trace ---
    tr = np.trace(H)
    cos_th = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    omega_values[t_idx] = np.arccos(cos_th)

    # --- Rotation axis from antisymmetric part ---
    axis = np.array([
        H[2, 1] - H[1, 2],
        H[0, 2] - H[2, 0],
        H[1, 0] - H[0, 1]
    ])
    axis_norm = np.linalg.norm(axis)
    if axis_norm > 1e-10:
        axis /= axis_norm
    # axis[0] = component along T_i (fiber direction) in quat basis
    fiber_components[t_idx] = abs(axis[0])

    # --- Triangle normal in T_p S^3 ---
    # Project q, r into tangent space at p
    u = q - np.dot(q, p) * p
    w = r - np.dot(r, p) * p
    # 4D cross product of (p, u, w) gives the normal
    n4d = triple_cross_4d(p, u, w)
    n4d_norm = np.linalg.norm(n4d)
    if n4d_norm > 1e-10:
        n4d /= n4d_norm
    # Fiber fraction = |normal . T_i(p)|
    Ti = hopf_fiber_tangent(p)
    fiber_fractions[t_idx] = abs(np.dot(n4d, Ti))

print("  Done.")
print()

# ------------------------------------------------------------------
# Step 5: Verify holonomy angle properties
# ------------------------------------------------------------------
print("STEP 5: Verifying holonomy angle properties")
print("-" * 72)

omega_mean = np.mean(omega_values)
omega_std = np.std(omega_values)

# Expected: Omega = 3*arccos(1/sqrt(5)) - pi
omega_expected = 3.0 * np.arccos(1.0 / np.sqrt(5.0)) - np.pi

# Expected: cos(Omega) = 11*sqrt(5)/25
cos_omega_expected = 11.0 * np.sqrt(5.0) / 25.0

print("  Holonomy angle statistics:")
print("    Mean  = %.15f rad" % omega_mean)
print("    Std   = %.2e rad" % omega_std)
print("    Min   = %.15f rad" % np.min(omega_values))
print("    Max   = %.15f rad" % np.max(omega_values))
print()

# Face-transitivity check
face_transitive = (omega_std < 1e-8)
print("  Face-transitive (all angles identical): %s" % ("YES" if face_transitive else "NO"))
print("  Verdict: %s" % pass_fail(face_transitive))
print()

# Verify holonomy angle value
omega_err_pct = pct_error(omega_mean, omega_expected)
print("  Omega computed = %.15f rad" % omega_mean)
print("  Omega expected = 3*arccos(1/sqrt(5)) - pi = %.15f rad" % omega_expected)
print("  Error          = %.6e%%" % omega_err_pct)

omega_val_pass = (omega_err_pct < 0.001)
print("  Verdict: %s" % pass_fail(omega_val_pass))
print()

# Verify cos(Omega) = 11*sqrt(5)/25
cos_omega_computed = np.cos(omega_mean)
cos_omega_err_pct = pct_error(cos_omega_computed, cos_omega_expected)
print("  cos(Omega) computed = %.15f" % cos_omega_computed)
print("  cos(Omega) expected = 11*sqrt(5)/25 = %.15f" % cos_omega_expected)
print("  Error               = %.6e%%" % cos_omega_err_pct)

cos_omega_pass = (cos_omega_err_pct < 0.001)
print("  Verdict: %s" % pass_fail(cos_omega_pass))
print()

# Verify SO(3) properties of holonomy matrices (spot check first 100)
det_errs = []
orth_errs = []
for H in hol_matrices[:100]:
    det_errs.append(abs(np.linalg.det(H) - 1.0))
    orth_errs.append(np.max(np.abs(H @ H.T - np.eye(3))))
print("  SO(3) check (first 100 matrices):")
print("    max |det(H) - 1| = %.2e" % max(det_errs))
print("    max |H*H^T - I|  = %.2e" % max(orth_errs))
so3_pass = (max(det_errs) < 1e-8) and (max(orth_errs) < 1e-8)
print("  Verdict: %s" % pass_fail(so3_pass))
print()

# ------------------------------------------------------------------
# Step 6: Verify the 5 distinct fiber fraction values
# ------------------------------------------------------------------
print("STEP 6: Verifying 5 distinct fiber fraction values |n . T_i(p)|")
print("-" * 72)

# Expected values and multiplicities:
expected_ff = np.array([0.0, 1.0 / (2.0 * phi), 0.5, phi / 2.0, 1.0])
expected_mult = np.array([160, 320, 320, 320, 80])

# Find unique values
ff_rounded = np.round(fiber_fractions, 4)
unique_ff, counts = np.unique(ff_rounded, return_counts=True)

print("  Expected fiber fractions and multiplicities:")
for val, mult in zip(expected_ff, expected_mult):
    print("    |n.T_i| = %.10f  x %d" % (val, mult))
print()

print("  Computed fiber fractions and multiplicities:")
for val, cnt in zip(unique_ff, counts):
    print("    |n.T_i| = %.10f  x %d" % (val, cnt))
print()

# Check values match
n_distinct = len(unique_ff)
ff_correct = (n_distinct == 5)
if ff_correct:
    for exp_val in expected_ff:
        found = False
        for comp_val in unique_ff:
            if abs(comp_val - exp_val) < 1e-3:
                found = True
                break
        if not found:
            ff_correct = False
            break

# Check multiplicities match
mult_correct = False
if ff_correct and n_distinct == 5:
    exp_sorted = sorted(zip(expected_ff, expected_mult), key=lambda x: x[0])
    comp_sorted = sorted(zip(unique_ff, counts), key=lambda x: x[0])
    mult_correct = True
    for (ev, em), (cv, cm) in zip(exp_sorted, comp_sorted):
        if abs(ev - cv) > 1e-3 or em != cm:
            mult_correct = False
            break

print("  Number of distinct values: %d (expected 5)" % n_distinct)
print("  Values match:  %s" % ("YES" if ff_correct else "NO"))
print("  Multiplicities match: %s" % ("YES" if mult_correct else "NO"))

# Verify Galois conjugate pair: 1/(2*phi) and phi/2
galois_1 = 1.0 / (2.0 * phi)
galois_2 = phi / 2.0
print()
print("  Galois conjugate pair:")
print("    1/(2*phi) = %.10f" % galois_1)
print("    phi/2     = %.10f" % galois_2)
print("    Product   = %.10f (exact: 1/4 = 0.25)" % (galois_1 * galois_2))
print("    Sum       = %.10f (exact: sqrt(5)/2 = %.10f)" % (galois_1 + galois_2, np.sqrt(5.0) / 2.0))

ff_pass = ff_correct and mult_correct
print("  Verdict: %s" % pass_fail(ff_pass))
print()

# Cross-check: fiber_components (from rotation axis) vs fiber_fractions (from normal)
corr = np.corrcoef(fiber_components, fiber_fractions)[0, 1]
diff_fc = np.abs(fiber_components - fiber_fractions)
print("  Cross-check: rotation axis fiber comp vs normal fiber fraction:")
print("    Correlation:      %.6f" % corr)
print("    Max difference:   %.2e" % diff_fc.max())
print("    Mean difference:  %.2e" % diff_fc.mean())
print()

# ------------------------------------------------------------------
# Step 7: Compute Berry phase from horizontal block
# ------------------------------------------------------------------
print("STEP 7: Computing Berry phase from horizontal 2x2 block of H")
print("-" * 72)
print()
print("  Method: In the quaternionic basis {T_i, T_j, T_k}, the fiber")
print("  direction is T_i (first basis vector). The holonomy matrix H")
print("  decomposes as H[0,0] = fiber-fiber and H[1:3, 1:3] = horizontal")
print("  block. The Berry phase is the rotation angle of this 2x2 block:")
print("    cos(Berry) = (H[1,1] + H[2,2]) / 2 = Tr(H_horiz) / 2")
print()

berry_phases = np.zeros(n_triangles)

for fi in range(n_triangles):
    H = hol_matrices[fi]
    # Horizontal 2x2 block (base of Hopf bundle)
    H_horiz = H[1:3, 1:3]
    # Berry phase = rotation angle of the 2x2 block
    cos_berry = np.clip((H_horiz[0, 0] + H_horiz[1, 1]) / 2.0, -1.0, 1.0)
    berry_phases[fi] = np.arccos(cos_berry)

mean_berry = np.mean(berry_phases)
berry_expected = 1.0 / phi**4

berry_err_pct = pct_error(mean_berry, berry_expected)

print("  Berry phase statistics:")
print("    Mean   = %.15f" % mean_berry)
print("    Std    = %.6f" % np.std(berry_phases))
print("    Min    = %.15f" % np.min(berry_phases))
print("    Max    = %.15f" % np.max(berry_phases))
print()
print("  Comparison with 1/phi^4:")
print("    Mean Berry phase = %.15f" % mean_berry)
print("    1/phi^4          = %.15f" % berry_expected)
print("    Error            = %.4f%%" % berry_err_pct)
print()

berry_pass = (berry_err_pct < 0.1)
print("  Verdict: %s" % pass_fail(berry_pass))
print()

# Additional decomposition statistics
print("  Additional decomposition:")
print("    Holonomy angle Omega    = %.15f rad" % omega_mean)
print("    Mean fiber comp |a.e_i| = %.6f" % np.mean(fiber_components))
print("    Mean |n . T_i(p)|       = %.6f" % np.mean(fiber_fractions))
print()

# Distribution of Berry phases
bp_rounded = np.round(berry_phases, 6)
unique_bp, counts_bp = np.unique(bp_rounded, return_counts=True)
print("  Distinct Berry phase values:")
for val, cnt in zip(unique_bp, counts_bp):
    print("    Berry = %.10f  x %d" % (val, cnt))
print()

# ------------------------------------------------------------------
# Step 8: Verify mean holonomy matrix structure
# ------------------------------------------------------------------
print("STEP 8: Mean holonomy matrix in quaternionic basis")
print("-" * 72)

H_mean = np.mean(hol_matrices, axis=0)
print("  <H> averaged over all 1200 faces:")
for row in range(3):
    print("    [%9.6f %9.6f %9.6f]" % (H_mean[row, 0], H_mean[row, 1], H_mean[row, 2]))
print()
print("  <H[0,0]> (fiber-fiber) = %.6f" % H_mean[0, 0])
print("  <H[1,1]> (horiz j-j)  = %.6f" % H_mean[1, 1])
print("  <H[2,2]> (horiz k-k)  = %.6f" % H_mean[2, 2])
print("  Tr(<H_horiz>)/2        = %.6f" % ((H_mean[1, 1] + H_mean[2, 2]) / 2.0))
print("  cos(1/phi^4)           = %.6f" % np.cos(berry_expected))
print()

# ============================================================================
# SUMMARY TABLE
# ============================================================================
print("=" * 72)
print("  SUMMARY TABLE")
print("=" * 72)
print()
print("  %-4s | %-45s | %s" % ("#", "Test", "Verdict"))
print("  %s" % ("-" * 63))
print("  %-4s | %-45s | %s" % ("1", "120 vertices on unit S^3", pass_fail(verts_pass)))
print("  %-4s | %-45s | %s" % ("2", "720 edges (p.q = phi/2)", pass_fail(edges_pass)))
print("  %-4s | %-45s | %s" % ("3", "1200 triangular faces", pass_fail(triangles_pass)))
print("  %-4s | %-45s | %s" % ("4", "Face-transitive holonomy (std < 1e-8)", pass_fail(face_transitive)))
print("  %-4s | %-45s | %s" % ("5", "Omega = 3*arccos(1/sqrt(5)) - pi", pass_fail(omega_val_pass)))
print("  %-4s | %-45s | %s" % ("6", "cos(Omega) = 11*sqrt(5)/25", pass_fail(cos_omega_pass)))
print("  %-4s | %-45s | %s" % ("7", "SO(3) holonomy matrices", pass_fail(so3_pass)))
print("  %-4s | %-45s | %s" % ("8", "5 fiber fractions with multiplicities", pass_fail(ff_pass)))
print("  %-4s | %-45s | %s" % ("9", "Mean Berry phase = 1/phi^4 (<0.1%%)", pass_fail(berry_pass)))
print("  %s" % ("-" * 63))
print()

all_pass = all([
    verts_pass, edges_pass, triangles_pass,
    face_transitive, omega_val_pass, cos_omega_pass,
    so3_pass, ff_pass, berry_pass
])

print("  OVERALL: %s" % ("ALL TESTS PASSED" if all_pass
                          else "SOME TESTS FAILED -- SEE ABOVE"))
print()

# Key numerical results
print("  Key results:")
print("    phi               = %.15f" % phi)
print("    1/phi^4           = %.15f" % berry_expected)
print("    Holonomy angle    = %.15f rad" % omega_mean)
print("    Mean Berry phase  = %.15f" % mean_berry)
print("    Berry error       = %.4f%%" % berry_err_pct)
print()
print("  Note: The Berry phase is the mean rotation angle of the horizontal")
print("  (base) 2x2 block of the SO(3) holonomy matrix, computed in the")
print("  quaternionic basis {T_i, T_j, T_k} where T_i is the Hopf fiber.")
print("  This is NOT the same as the total holonomy angle Omega, which is")
print("  face-transitive. The Berry phase VARIES from face to face because")
print("  the holonomy axis has different alignment with the fiber direction.")
print("=" * 72)
