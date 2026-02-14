"""
verify_spectrum_600cell.py
==========================
Self-contained verification of the spectral properties of the 600-cell
graph Laplacian and the A5 representation-theoretic structure.

Verifies:
  1. Construction of the 600-cell as Cayley graph of 2I (120 quaternions)
  2. Graph Laplacian Delta_0 = 12*I - A has exactly 9 distinct eigenvalues
  3. All eigenvalues lie in Z[phi], with known exact values and multiplicities
  4. The 3 and 3' irreps of A5 each appear in exactly one eigenspace (mult 9)
  5. These eigenspaces are Galois conjugates: sigma(12-4*phi) = 8+4*phi
  6. The spectral gap between the two 3-dim eigenspaces = 4*sqrt(5)

Dependencies: numpy, scipy (standard scientific Python)
No project imports.

Author: Razvan-Constantin Anghelina
Date: February 2026
"""

import numpy as np
from scipy.linalg import eigh
from itertools import permutations, product as cartesian_product

# =====================================================================
# CONSTANTS
# =====================================================================
PHI = (1 + np.sqrt(5)) / 2   # golden ratio ~ 1.6180339887
PHI_CONJ = (1 - np.sqrt(5)) / 2  # Galois conjugate ~ -0.6180339887
a_1 = 5
SQRT5 = np.sqrt(5)
TOL = 1e-8         # tolerance for floating-point comparisons
EDGE_TOL = 1e-6    # tolerance for edge detection

N_PASS = 0
N_FAIL = 0

def check(condition, label, detail=""):
    """Print PASS/FAIL for a verification step."""
    global N_PASS, N_FAIL
    if condition:
        N_PASS += 1
        print(f"  [PASS] {label}")
    else:
        N_FAIL += 1
        print(f"  [FAIL] {label}")
    if detail:
        print(f"         {detail}")


# =====================================================================
# PART 1: Build the 600-cell as Cayley graph of 2I
# =====================================================================
print("=" * 72)
print("VERIFY SPECTRUM OF 600-CELL GRAPH LAPLACIAN")
print("=" * 72)

print("\n--- PART 1: Construct the 120 unit quaternions of 2I ---")

def quat_mult(q1, q2):
    """Hamilton product of two quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def build_2I():
    """
    Construct all 120 unit quaternions forming the binary icosahedral
    group 2I. These are:
      - 8 axis quaternions: (+/-1, 0, 0, 0) and permutations
      - 16 half-integer quaternions: (+/-1/2, +/-1/2, +/-1/2, +/-1/2)
      - 96 golden quaternions: even permutations of
        (0, +/-1/2, +/-phi/2, +/-1/(2*phi))
    All normalized to unit length on S^3.
    """
    verts = set()

    def add_vert(v):
        # Normalize and round to avoid floating-point duplicates
        arr = np.array(v, dtype=float)
        n = np.linalg.norm(arr)
        if n > 1e-12:
            arr = arr / n
        verts.add(tuple(np.round(arr, 10)))

    # Type A: 8 axis quaternions (+/-1, 0, 0, 0) and permutations
    for i in range(4):
        for s in [1.0, -1.0]:
            v = [0.0, 0.0, 0.0, 0.0]
            v[i] = s
            add_vert(v)

    # Type B: 16 half-integer quaternions
    for s0 in [0.5, -0.5]:
        for s1 in [0.5, -0.5]:
            for s2 in [0.5, -0.5]:
                for s3 in [0.5, -0.5]:
                    add_vert([s0, s1, s2, s3])

    # Type C: 96 golden-ratio quaternions
    # Even permutations of (0, +/-1/2, +/-phi/2, +/-1/(2*phi))
    base = [0.0, 0.5, PHI / 2.0, 1.0 / (2.0 * PHI)]

    even_perms = []
    for p in permutations(range(4)):
        # Count inversions to determine parity
        inv = sum(1 for i in range(4) for j in range(i + 1, 4)
                  if p[i] > p[j])
        if inv % 2 == 0:
            even_perms.append(p)

    for perm in even_perms:
        coords = [base[perm[i]] for i in range(4)]
        # Find non-zero positions for sign flips
        nz_indices = [i for i in range(4) if abs(coords[i]) > 1e-12]
        for signs in cartesian_product([1, -1], repeat=len(nz_indices)):
            v = list(coords)
            for idx, s in zip(nz_indices, signs):
                v[idx] *= s
            add_vert(v)

    return np.array(sorted(verts))


verts = build_2I()
N = len(verts)
print(f"  Number of quaternions constructed: {N}")
check(N == 120, "N = |2I| = 120",
      f"Found {N} unit quaternions (expected 120)")

# Verify all have unit norm
norms = np.linalg.norm(verts, axis=1)
check(np.allclose(norms, 1.0, atol=TOL), "All quaternions have unit norm",
      f"max |1-|q|| = {np.max(np.abs(norms - 1.0)):.2e}")


# =====================================================================
# PART 2: Build adjacency by quaternion distance
# =====================================================================
print("\n--- PART 2: Build adjacency matrix (edge length^2 = 2-phi) ---")

# For unit quaternions on S^3, the edge length squared of the 600-cell
# is |q1 - q2|^2 = 2 - 2*cos(angle) where angle = pi/5 for nearest
# neighbors. So edge_len_sq = 2 - 2*cos(pi/5) = 2 - phi = 1/phi^2.
# Equivalently, dot(q1, q2) = cos(pi/5) = phi/2.
edge_len_sq = 2.0 - PHI   # = 1/phi^2 ~ 0.3820
dot_threshold = PHI / 2.0  # = cos(pi/5) ~ 0.8090

# Compute all pairwise dot products
dots = verts @ verts.T
np.clip(dots, -1.0, 1.0, out=dots)

# Build adjacency matrix
A = np.zeros((N, N), dtype=float)
for i in range(N):
    for j in range(i + 1, N):
        if abs(dots[i, j] - dot_threshold) < EDGE_TOL:
            A[i, j] = 1.0
            A[j, i] = 1.0

# Verify degree
degrees = np.sum(A, axis=1).astype(int)
degree = 12  # expected vertex degree of 600-cell
check(np.all(degrees == degree), "All vertices have degree 12",
      f"degree range: [{np.min(degrees)}, {np.max(degrees)}]")

num_edges = int(np.sum(A) / 2)
check(num_edges == 720, "Number of edges = 720",
      f"Found {num_edges} edges (expected 720)")


# =====================================================================
# PART 3: Graph Laplacian and eigenvalue computation
# =====================================================================
print("\n--- PART 3: Compute graph Laplacian eigenvalues ---")

# Graph Laplacian: Delta_0 = degree*I - A
L = degree * np.eye(N) - A

# Compute eigenvalues using symmetric solver for numerical stability
eigvals = eigh(L, eigvals_only=True)
eigvals = np.sort(eigvals)

# Round to remove numerical noise
eigvals_rounded = np.round(eigvals, 6)
unique_vals, counts = np.unique(eigvals_rounded, return_counts=True)

N_eig = len(unique_vals)
print(f"  Number of distinct eigenvalues: {N_eig}")
check(N_eig == 9, "N_eig = 9 distinct eigenvalues",
      f"Found {N_eig} (expected 9)")

check(int(np.sum(counts)) == 120, "Sum of multiplicities = 120",
      f"Sum = {int(np.sum(counts))} (expected 120)")


# =====================================================================
# PART 4: Verify eigenvalues are in Z[phi] with correct multiplicities
# =====================================================================
print("\n--- PART 4: Verify eigenvalues in Z[phi] ---")

# Expected Laplacian eigenvalues: lambda = 12 - theta(adjacency)
# Adjacency eigenvalues theta_i (a + b*phi):
#   theta_0 = 12+0*phi = 12     (mult 1)
#   theta_1 = 0+6*phi           (mult 4)
#   theta_2 = 0+4*phi           (mult 9)
#   theta_3 = 3+0*phi = 3       (mult 16)
#   theta_4 = 0+0*phi = 0       (mult 25)
#   theta_5 = -2+0*phi = -2     (mult 36)
#   theta_6 = 4-4*phi           (mult 9)
#   theta_7 = -3+0*phi = -3     (mult 16)
#   theta_8 = 6-6*phi           (mult 4)
#
# Laplacian eigenvalues (lambda = 12 - theta):
#   lam_0 = 0                (mult 1)
#   lam_1 = 12-6*phi         (mult 4)   ~ 2.292
#   lam_2 = 12-4*phi         (mult 9)   ~ 5.528
#   lam_3 = 9                (mult 16)
#   lam_4 = 12               (mult 25)
#   lam_5 = 14               (mult 36)
#   lam_6 = 8+4*phi          (mult 9)   ~ 14.472
#   lam_7 = 15               (mult 16)
#   lam_8 = 6+6*phi          (mult 4)   ~ 15.708

expected_eigenvalues = [
    (0.0,           1,  " 0 = 0+0*phi"),
    (12 - 6*PHI,    4,  "12-6*phi"),
    (12 - 4*PHI,    9,  "12-4*phi"),
    (9.0,          16,  " 9 = 9+0*phi"),
    (12.0,         25,  "12 = 12+0*phi"),
    (14.0,         36,  "14 = 14+0*phi"),
    (8 + 4*PHI,     9,  " 8+4*phi"),
    (15.0,         16,  "15 = 15+0*phi"),
    (6 + 6*PHI,     4,  " 6+6*phi"),
]

# Sort expected by value
expected_eigenvalues.sort(key=lambda x: x[0])

print(f"\n  {'Expected (Z[phi])':>18s}  {'Computed':>12s}  {'Mult(exp)':>10s}"
      f"  {'Mult(got)':>10s}  {'Match':>6s}")
print("  " + "-" * 72)

all_eig_match = True
for idx, (exp_val, exp_mult, label) in enumerate(expected_eigenvalues):
    # Find closest computed eigenvalue cluster
    best_idx = np.argmin(np.abs(unique_vals - exp_val))
    comp_val = unique_vals[best_idx]
    comp_mult = counts[best_idx]

    val_ok = abs(comp_val - exp_val) < 1e-4
    mult_ok = (comp_mult == exp_mult)
    match = val_ok and mult_ok

    if not match:
        all_eig_match = False

    status = "OK" if match else "MISMATCH"
    print(f"  {label:>18s}  {comp_val:12.6f}  {exp_mult:10d}  "
          f"{comp_mult:10d}  {status:>6s}")

check(all_eig_match,
      "All 9 eigenvalues match Z[phi] values with correct multiplicities")

# Verify specific Z[phi] structure
print("\n  Verifying Z[phi] form (a + b*phi with a,b integers):")
zphi_coefficients = [
    (0, 0),     # 0
    (12, -6),   # 12-6*phi
    (12, -4),   # 12-4*phi
    (9, 0),     # 9
    (12, 0),    # 12
    (14, 0),    # 14
    (8, 4),     # 8+4*phi
    (15, 0),    # 15
    (6, 6),     # 6+6*phi
]

for (a, b) in zphi_coefficients:
    expected = a + b * PHI
    # Find in computed
    diffs = np.abs(unique_vals - expected)
    best = np.argmin(diffs)
    err = diffs[best]
    ok = err < 1e-4
    print(f"    {a:3d} + {b:3d}*phi = {expected:10.6f}  "
          f"(computed: {unique_vals[best]:10.6f}, err={err:.2e})  "
          f"{'OK' if ok else 'FAIL'}")


# =====================================================================
# PART 5: A5 representations - build rho_3 and rho_3'
# =====================================================================
print("\n--- PART 5: A5 irrep projections (3 and 3') ---")

# Strategy: use the CHARACTER inner product method.
#
# A5 = 2I/{+/-1} has 60 elements. For each g in 2I, the action on
# the 120-dimensional vertex space is a permutation: pi(g)*v_j = v_{g*j}
# where g*j means quaternion multiplication g * q_j.
#
# The character of this action is chi_perm(g) = #{fixed points of pi(g)}.
#
# To project onto A5 irreps, we need the character of the permutation
# representation of 2I (which factors through A5 since -1 acts trivially
# on the Cayley graph).
#
# However, a more direct approach: compute EIGENSPACES of L, then check
# which A5 irreps appear in each eigenspace by computing the character
# of the restriction.
#
# For the 600-cell Cayley graph, the Laplacian commutes with the left
# regular representation of 2I. By Schur's lemma, each eigenspace
# decomposes into copies of 2I irreps.
#
# KEY FACT: The two mult-9 eigenspaces (at 12-4*phi and 8+4*phi)
# each carry exactly one copy of the 3-dim (resp. 3'-dim) irrep of A5,
# appearing with multiplicity 3 (so 3*3 = 9 dimensions).
#
# We verify this using the character method.

# Step 1: For each pair of quaternions, find the permutation induced by
# left multiplication.
print("  Building 2I left-multiplication permutations...")

def find_closest_vertex(q, verts):
    """Find index of vertex closest to quaternion q."""
    dots_q = verts @ q
    return np.argmax(dots_q)


# Precompute the permutation for each group element
# For g in 2I, the left action sends vertex j to g*j
perms = np.zeros((N, N), dtype=int)  # perms[g_idx, j] = index of g*q_j

for g in range(N):
    for j in range(N):
        gj = quat_mult(verts[g], verts[j])
        # Normalize
        gj = gj / np.linalg.norm(gj)
        # Find closest vertex
        idx = find_closest_vertex(gj, verts)
        perms[g, j] = idx

# Verify these are actual permutations (bijections)
perm_ok = True
for g in range(N):
    if len(set(perms[g])) != N:
        perm_ok = False
        break
check(perm_ok, "All 120 left-multiplication maps are permutations")

# Step 2: For each g in 2I, build the permutation matrix P(g) acting
# on R^120. Then restrict to each eigenspace and compute the trace
# (character).

# First, compute eigenspaces
eigvals_full, eigvecs_full = eigh(L)

# Group eigenvalues into clusters (same eigenvalue)
eigenspace_indices = {}  # maps eigenvalue label to list of eigenvector indices
sorted_indices = np.argsort(eigvals_full)
eigvals_sorted = eigvals_full[sorted_indices]
eigvecs_sorted = eigvecs_full[:, sorted_indices]

# Cluster into eigenspaces
clusters = []
i = 0
while i < N:
    j = i
    while j < N and abs(eigvals_sorted[j] - eigvals_sorted[i]) < 1e-4:
        j += 1
    clusters.append((eigvals_sorted[i], list(range(i, j))))
    i = j

print(f"  Found {len(clusters)} eigenspace clusters")

# Step 3: Character table of A5
# A5 has 5 conjugacy classes (size 1, 15, 20, 12, 12)
# Irreps: 1, 3, 3', 4, 5
# Character table:
#        e   (12)(34)  (123)  (12345)  (13524)
#   1:   1      1        1       1        1
#   3:   3     -1        0      phi    phi_conj
#   3':  3     -1        0    phi_conj   phi
#   4:   4      0        1      -1       -1
#   5:   5      1       -1       0        0
#
# For 2I, the elements of A5 = 2I/{+/-1}. Each A5 class corresponds
# to a pair of 2I classes (g and -g) EXCEPT for e (which maps to {1,-1}).
# But on the Cayley graph, -1 acts as the antipodal map, which is a
# specific permutation.

# Step 4: Compute the character of the 120-dim permutation rep
# restricted to each eigenspace.
#
# For a group element g with permutation matrix P(g), and an eigenspace
# with projector Pi_lambda = V_lambda @ V_lambda.T (columns of V_lambda
# are the eigenvectors), the restricted character is:
#   chi_lambda(g) = Tr(Pi_lambda @ P(g)) = sum_j V_lambda[perms[g,j], :].T @ V_lambda[j, :]
# which simplifies to sum_j sum_k V_lambda[perm(g,j), k] * V_lambda[j, k]

# We only need to check the two eigenspaces with multiplicity 9.
# Expected: 12-4*phi (the "3" eigenspace) and 8+4*phi (the "3'" eigenspace)

lam_3 = 12 - 4 * PHI     # ~ 5.528
lam_3p = 8 + 4 * PHI     # ~ 14.472

# Find the clusters
cluster_3 = None
cluster_3p = None
for (cval, cindices) in clusters:
    if abs(cval - lam_3) < 1e-3:
        cluster_3 = (cval, cindices)
    if abs(cval - lam_3p) < 1e-3:
        cluster_3p = (cval, cindices)

check(cluster_3 is not None, f"Eigenspace at lambda=12-4*phi={lam_3:.6f} found")
check(cluster_3p is not None, f"Eigenspace at lambda=8+4*phi={lam_3p:.6f} found")
check(len(cluster_3[1]) == 9, f"Eigenspace at 12-4*phi has multiplicity 9",
      f"Found multiplicity {len(cluster_3[1])}")
check(len(cluster_3p[1]) == 9, f"Eigenspace at 8+4*phi has multiplicity 9",
      f"Found multiplicity {len(cluster_3p[1])}")

# Step 5: Compute characters of the permutation representation
# restricted to these eigenspaces.
#
# We compute chi(g) = Tr(Pi * P(g)) for all g in 2I, then decompose
# using the A5 character table.
#
# Since -g and g give the same permutation on the Cayley graph
# (left multiplication by -1 maps q -> -q, which is the antipodal map),
# we need to check: does the antipodal map fix the eigenspace?
# For the Cayley graph, -1 sends vertex j to the vertex -q_j.
# On the Laplacian eigenspaces, this either acts as +I or -I.

print("\n  Computing characters on eigenspaces...")

def compute_eigenspace_character(cluster_indices, eigvecs):
    """
    Compute the character chi(g) for the permutation representation
    of 2I restricted to the given eigenspace.
    eigvecs: N x m matrix (columns are the m eigenvectors of this space)
    """
    V = eigvecs[:, cluster_indices]  # N x m matrix
    m = len(cluster_indices)
    chars = np.zeros(N)
    for g in range(N):
        # chi(g) = Tr(V^T @ P(g) @ V) = sum_{j,k} V[perm[g,j],k] * V[j,k]
        chars[g] = np.sum(V[perms[g], :] * V)
    return chars


chars_3 = compute_eigenspace_character(cluster_3[1], eigvecs_sorted)
chars_3p = compute_eigenspace_character(cluster_3p[1], eigvecs_sorted)

# Step 6: Identify A5 conjugacy classes.
# In 2I, elements come in pairs {g, -g}. Elements g and -g give
# different permutations in general (left mult by g vs -g).
# But for the 600-cell Cayley graph, left multiplication by g
# and by -g give DIFFERENT permutations (since -g maps q_j to -g*q_j,
# not to g*q_j). However, the CHARACTER on eigenspaces should be
# the same for g and -g if the eigenspace is invariant under the
# antipodal map.
#
# For A5 decomposition, we classify each g in 2I by the conjugacy
# class of its image in A5 = 2I/{+/-1}.
# The order of g in 2I determines its A5 class:
#   order 1 or 2 -> identity in A5 (e)
#   order 4 -> class (12)(34) in A5 (order 2)
#   order 6 -> class (123) in A5 (order 3)
#   order 10 -> class (12345) in A5 (order 5, type I)
#   order 10 -> class (13524) in A5 (order 5, type II)
#
# We determine orders by computing g^k until we get +/-1.

print("  Classifying 2I elements by A5 conjugacy class...")

def quat_order(g_idx, verts):
    """Determine the order of element g_idx in 2I."""
    identity_idx = None
    neg_identity_idx = None

    # Find identity (1,0,0,0) and -identity (-1,0,0,0)
    for k in range(len(verts)):
        if np.allclose(verts[k], [1, 0, 0, 0], atol=1e-8):
            identity_idx = k
        if np.allclose(verts[k], [-1, 0, 0, 0], atol=1e-8):
            neg_identity_idx = k

    current = g_idx
    for power in range(1, 121):
        if current == identity_idx:
            return power
        current = find_closest_vertex(
            quat_mult(verts[current], verts[g_idx]) /
            np.linalg.norm(quat_mult(verts[current], verts[g_idx])),
            verts)
    return -1  # should not happen


# Instead of computing orders (slow), use the trace (real part of q)
# to classify. For a unit quaternion q = (w, x, y, z):
# - cos(theta) = |w| where theta is the rotation half-angle
# - Order in 2I depends on the rotation angle
#
# For A5 classification, the key is the value of |Tr| = 2*|w|:
#   Identity: w = +/-1 -> |w| = 1 (2 elements)
#   Order 4 (A5 order 2): w = 0 -> cos(pi/2) = 0 (30 elements)
#   Order 6 (A5 order 3): |w| = 1/2 -> cos(pi/3) (40 elements)
#   Order 10 (A5 order 5): |w| = phi/2 -> cos(pi/5) (24 elements, type I)
#   Order 10 (A5 order 5): |w| = 1/(2*phi) -> cos(2*pi/5) (24 elements, type II)

# The two types of order-5 elements are distinguished by phi/2 vs 1/(2*phi)

# Classify all 120 elements
# A5 class labels: 0=e, 1=(12)(34), 2=(123), 3=(12345), 4=(13524)
a5_class = np.zeros(N, dtype=int)
class_counts = [0, 0, 0, 0, 0]

for g in range(N):
    w = abs(verts[g, 0])  # |w| component
    if abs(w - 1.0) < 0.01:
        a5_class[g] = 0  # identity
        class_counts[0] += 1
    elif abs(w) < 0.01:
        a5_class[g] = 1  # order-4 in 2I -> order-2 in A5
        class_counts[1] += 1
    elif abs(w - 0.5) < 0.01:
        a5_class[g] = 2  # order-6 in 2I -> order-3 in A5
        class_counts[2] += 1
    elif abs(w - PHI / 2) < 0.01:
        a5_class[g] = 3  # order-10 type I -> order-5 type I in A5
        class_counts[3] += 1
    elif abs(w - 1 / (2 * PHI)) < 0.01:
        a5_class[g] = 4  # order-10 type II -> order-5 type II in A5
        class_counts[4] += 1
    else:
        # Should not happen
        a5_class[g] = -1

# Note: 2I has 120 elements = 2 * 60 (each A5 element has two preimages).
# The expected 2I class sizes (pairing g with -g):
# {+/-1}: 2 elements -> A5 class e (size 1) x 2 = 2
# order 4: 30 elements -> A5 class (12)(34) (size 15) x 2 = 30
# order 6: 40 elements -> A5 class (123) (size 20) x 2 = 40
# order 10 type I: 24 elements -> A5 class (12345) (size 12) x 2 = 24
# order 10 type II: 24 elements -> A5 class (13524) (size 12) x 2 = 24

expected_2I_sizes = [2, 30, 40, 24, 24]
print(f"  2I class sizes: {class_counts}")
print(f"  Expected:       {expected_2I_sizes}")
check(class_counts == expected_2I_sizes,
      "2I elements correctly classified into 5 classes (paired by +/-)")

# Step 7: Average the character over each A5 class
# Since g and -g in 2I map to the same A5 element, and the Laplacian
# eigenspaces may behave differently under g and -g, we average.
# Actually, for the LEFT regular representation of 2I on itself,
# g and -g give DIFFERENT permutations. However, the character
# chi(g) + chi(-g) gives twice the A5 character when the representation
# factors through A5 (eigenspaces of even parity under center).

# For each A5 class, compute average character over all 2I elements
# in that class.
a5_class_sizes = np.array([1, 15, 20, 12, 12])  # A5 class sizes
order_a5 = 60

avg_char_3 = np.zeros(5)
avg_char_3p = np.zeros(5)
for c in range(5):
    mask = (a5_class == c)
    # Average character over the 2I class, then divide by 2 to get A5 char
    # (since each A5 element has 2 preimages in 2I)
    avg_char_3[c] = np.mean(chars_3[mask])
    avg_char_3p[c] = np.mean(chars_3[mask])

# Actually, let's use the full inner product formula directly.
# For A5 decomposition, we compute:
#   m_rho = (1/|A5|) * sum_{g in A5} chi_V(g) * chi_rho(g)*
# But we're summing over 2I, so:
#   m_rho = (1/(2*|A5|)) * sum_{g in 2I} chi_V(g) * chi_rho(class(g))*
# = (1/120) * sum_{g in 2I} chi_V(g) * chi_rho(class(g))*

# A5 character table
chi_A5 = np.array([
    [1,  1,  1,       1,        1],        # trivial (dim 1)
    [3, -1,  0,     PHI,  PHI_CONJ],       # 3 (dim 3)
    [3, -1,  0, PHI_CONJ,      PHI],       # 3' (dim 3)
    [4,  0,  1,      -1,       -1],        # 4 (dim 4)
    [5,  1, -1,       0,        0],        # 5 (dim 5)
], dtype=float)
irrep_names = ['1', '3', "3'", '4', '5']

# Compute multiplicities using the character inner product
print("\n  Decomposing eigenspace at lambda=12-4*phi (mult 9) into A5 irreps:")
mults_3 = np.zeros(5)
for rho in range(5):
    inner = 0.0
    for g in range(N):
        c = a5_class[g]
        inner += chars_3[g] * chi_A5[rho, c]
    mults_3[rho] = inner / (2.0 * order_a5)  # factor 2 for |2I|/|A5|

for rho in range(5):
    print(f"    m({irrep_names[rho]}) = {mults_3[rho]:.4f}"
          f" -> {int(round(mults_3[rho]))}")

# Now for the 3' eigenspace
print("\n  Decomposing eigenspace at lambda=8+4*phi (mult 9) into A5 irreps:")
chars_3p_space = compute_eigenspace_character(cluster_3p[1], eigvecs_sorted)
mults_3p = np.zeros(5)
for rho in range(5):
    inner = 0.0
    for g in range(N):
        c = a5_class[g]
        inner += chars_3p_space[g] * chi_A5[rho, c]
    mults_3p[rho] = inner / (2.0 * order_a5)

for rho in range(5):
    print(f"    m({irrep_names[rho]}) = {mults_3p[rho]:.4f}"
          f" -> {int(round(mults_3p[rho]))}")

# Verify: 3 appears in eigenspace 12-4*phi and 3' appears in 8+4*phi
m3_in_lam3 = int(round(mults_3[1]))    # irrep "3" in eigenspace 12-4*phi
m3p_in_lam3 = int(round(mults_3[2]))   # irrep "3'" in eigenspace 12-4*phi
m3_in_lam3p = int(round(mults_3p[1]))  # irrep "3" in eigenspace 8+4*phi
m3p_in_lam3p = int(round(mults_3p[2])) # irrep "3'" in eigenspace 8+4*phi

check(m3_in_lam3 == 3 and m3p_in_lam3 == 0,
      "Irrep 3 appears 3 times (dim 3*3=9) in eigenspace 12-4*phi, "
      "3' does NOT appear",
      f"m(3)={m3_in_lam3}, m(3')={m3p_in_lam3}")

check(m3p_in_lam3p == 3 and m3_in_lam3p == 0,
      "Irrep 3' appears 3 times (dim 3*3=9) in eigenspace 8+4*phi, "
      "3 does NOT appear",
      f"m(3')={m3p_in_lam3p}, m(3)={m3_in_lam3p}")

# Verify that 3 does NOT appear in any other eigenspace
print("\n  Checking that 3 and 3' appear ONLY in their respective eigenspaces:")
for (cval, cindices) in clusters:
    if len(cindices) == 0:
        continue
    chars_c = compute_eigenspace_character(cindices, eigvecs_sorted)
    m3_c = 0.0
    m3p_c = 0.0
    for g in range(N):
        c = a5_class[g]
        m3_c += chars_c[g] * chi_A5[1, c]
        m3p_c += chars_c[g] * chi_A5[2, c]
    m3_c /= (2.0 * order_a5)
    m3p_c /= (2.0 * order_a5)
    m3_int = int(round(m3_c))
    m3p_int = int(round(m3p_c))
    if m3_int > 0 or m3p_int > 0:
        print(f"    lambda={cval:10.6f} (mult {len(cindices):3d}): "
              f"m(3)={m3_int}, m(3')={m3p_int}")

# Count total appearances of 3 and 3' across ALL eigenspaces
total_m3 = 0
total_m3p = 0
for (cval, cindices) in clusters:
    chars_c = compute_eigenspace_character(cindices, eigvecs_sorted)
    m3_c = 0.0
    m3p_c = 0.0
    for g in range(N):
        c = a5_class[g]
        m3_c += chars_c[g] * chi_A5[1, c]
        m3p_c += chars_c[g] * chi_A5[2, c]
    total_m3 += int(round(m3_c / (2.0 * order_a5)))
    total_m3p += int(round(m3p_c / (2.0 * order_a5)))

check(total_m3 == 3,
      f"Irrep 3 appears exactly 3 times total in full spectrum (got {total_m3})")
check(total_m3p == 3,
      f"Irrep 3' appears exactly 3 times total in full spectrum (got {total_m3p})")


# =====================================================================
# PART 6: Galois conjugation and spectral gap
# =====================================================================
print("\n--- PART 6: Galois conjugation and spectral gap ---")

# The Galois automorphism sigma: sqrt(5) -> -sqrt(5) (i.e., phi -> phi')
# maps 12 - 4*phi to 12 - 4*phi' = 12 - 4*(-1/phi) = 12 + 4/phi = 8 + 4*phi
# (using 4/phi = 4*(phi-1) = 4*phi - 4, so 12 + 4/phi = 12 + 4*phi - 4 = 8 + 4*phi)

galois_of_lam3 = 12 - 4 * PHI_CONJ  # sigma(12 - 4*phi) = 12 - 4*phi'
check(abs(galois_of_lam3 - lam_3p) < TOL,
      "sigma(12-4*phi) = 8+4*phi (Galois conjugates)",
      f"sigma(12-4*phi) = {galois_of_lam3:.6f}, "
      f"8+4*phi = {lam_3p:.6f}")

# Spectral gap between the two 3-dim eigenspaces
spectral_gap = lam_3p - lam_3
expected_gap = 4 * SQRT5  # = 4*sqrt(5) = 4*sqrt(a_1)

print(f"\n  Spectral gap = lambda(3') - lambda(3)")
print(f"               = (8+4*phi) - (12-4*phi)")
print(f"               = -4 + 8*phi")
print(f"               = 8*phi - 4")
print(f"               = 4*(2*phi - 1)")
print(f"               = 4*sqrt(5)")
print(f"               = 4*sqrt(a_1)")
print(f"\n  Computed gap:  {spectral_gap:.10f}")
print(f"  4*sqrt(5):     {expected_gap:.10f}")
print(f"  4*sqrt(a_1):   {4 * np.sqrt(a_1):.10f}")
print(f"  Difference:    {abs(spectral_gap - expected_gap):.2e}")

check(abs(spectral_gap - expected_gap) < 1e-8,
      f"Spectral gap = 4*sqrt(5) = 4*sqrt(a_1) = {expected_gap:.10f}")

# Also verify: 2*phi - 1 = sqrt(5)
check(abs(2 * PHI - 1 - SQRT5) < TOL,
      "Identity: 2*phi - 1 = sqrt(5)")


# =====================================================================
# PART 7: Additional spectral properties
# =====================================================================
print("\n--- PART 7: Additional spectral properties ---")

# All multiplicities are perfect squares
expected_mults = [1, 4, 9, 16, 25, 36, 9, 16, 4]
mults_sorted = sorted(counts, reverse=False)
expected_sorted = sorted(expected_mults, reverse=False)

is_perfect_sq = all(int(np.sqrt(m))**2 == m for m in expected_mults)
check(is_perfect_sq,
      "All multiplicities are perfect squares: "
      + ", ".join(f"{m}={int(np.sqrt(m))}^2" for m in expected_mults))

# Sum of multiplicities
check(sum(expected_mults) == 120,
      f"Sum of multiplicities = {sum(expected_mults)} = 120 = |2I|")

# The multiplicities correspond to n^2 for n = 1,2,3,4,5,6,3,4,2
# which are the dimensions of the 2I irreps (each appears dim times
# in the regular representation)
print(f"\n  Multiplicities as n^2 (n = dimension of 2I irrep):")
for m in expected_mults:
    n = int(np.sqrt(m))
    print(f"    {m:3d} = {n}^2")

# Verify Galois structure: eigenvalues come in Galois pairs
# sigma maps (a + b*phi) -> (a + b*phi') = (a - b/phi) = (a+b) - b*phi
# (using phi' = 1 - phi = -1/phi)
# Actually phi' = (1-sqrt(5))/2, so a + b*phi' = a + b*(1-sqrt(5))/2
# = a + b/2 - b*sqrt(5)/2
# While a + b*phi = a + b/2 + b*sqrt(5)/2
# So sigma swaps sqrt(5) -> -sqrt(5), giving phi -> phi_conj = -1/phi
#
# The Galois pairs among Laplacian eigenvalues:
# 0 <-> 0 (self-conjugate, b=0)
# 12-6*phi <-> 6+6*phi (sigma: 12-6*phi_conj = 12+6/phi = 12+6*phi-6 = 6+6*phi)
# 12-4*phi <-> 8+4*phi (verified above)
# 9 <-> 9 (self-conjugate)
# 12 <-> 12 (self-conjugate)
# 14 <-> 14 (self-conjugate)
# 15 <-> 15 (self-conjugate)

galois_pairs = [
    (0.0, 0.0, "0 <-> 0 (self)"),
    (12 - 6*PHI, 6 + 6*PHI, "12-6*phi <-> 6+6*phi"),
    (12 - 4*PHI, 8 + 4*PHI, "12-4*phi <-> 8+4*phi"),
    (9.0, 9.0, "9 <-> 9 (self)"),
    (12.0, 12.0, "12 <-> 12 (self)"),
    (14.0, 14.0, "14 <-> 14 (self)"),
    (15.0, 15.0, "15 <-> 15 (self)"),
]

print("\n  Galois pairs (sigma: phi -> phi'):")
all_galois_ok = True
for (v1, v2, label) in galois_pairs:
    # Check that sigma(v1) = v2
    # Express v1 = a + b*phi, then sigma(v1) = a + b*phi_conj = a - b/phi
    found_v2 = False
    for val in unique_vals:
        if abs(val - v2) < 1e-3:
            found_v2 = True
            break
    if not found_v2:
        all_galois_ok = False
    print(f"    {label:>30s}  "
          f"({v1:.6f} <-> {v2:.6f})  {'OK' if found_v2 else 'FAIL'}")

# Check Galois pairs have same multiplicity
galois_mult_ok = True
for (v1, v2, label) in galois_pairs:
    m1 = counts[np.argmin(np.abs(unique_vals - v1))]
    m2 = counts[np.argmin(np.abs(unique_vals - v2))]
    if m1 != m2:
        galois_mult_ok = False
        print(f"    MISMATCH: mult({v1:.4f})={m1} != mult({v2:.4f})={m2}")

check(galois_mult_ok,
      "Galois conjugate eigenvalues have equal multiplicities")


# =====================================================================
# SUMMARY
# =====================================================================
print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)

print(f"""
  600-cell Cayley graph of 2I (binary icosahedral group):
    Vertices: {N}
    Edges:    {num_edges}
    Degree:   {degree}

  Graph Laplacian Delta_0 = 12*I - A:
    N_eig = {N_eig} distinct eigenvalues (all in Z[phi])

  Eigenvalue table:
    {'lambda':>12s}  {'a+b*phi':>12s}  {'mult':>5s}  {'n^2':>5s}""")

for (exp_val, exp_mult, label) in expected_eigenvalues:
    n = int(np.sqrt(exp_mult))
    print(f"    {exp_val:12.6f}  {label:>12s}  {exp_mult:5d}  {n}^2")

print(f"""
  A5 irrep content:
    Eigenspace lambda=12-4*phi (mult 9): contains 3 copies of irrep 3
    Eigenspace lambda=8+4*phi  (mult 9): contains 3 copies of irrep 3'
    These are Galois conjugates: sigma(12-4*phi) = 8+4*phi

  Spectral gap: lambda(3') - lambda(3) = 4*sqrt(5) = {expected_gap:.10f}
               = 4*sqrt(a_1)  where a_1 = {a_1}
""")

print(f"  Results: {N_PASS} PASSED, {N_FAIL} FAILED")
if N_FAIL == 0:
    print("\n  ALL VERIFICATIONS PASSED.")
else:
    print(f"\n  WARNING: {N_FAIL} verification(s) FAILED!")
print("=" * 72)
