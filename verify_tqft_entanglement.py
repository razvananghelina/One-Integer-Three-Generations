"""
verify_tqft_entanglement.py
============================
Verification of the TQFT/entanglement section and related theorems:

  1. McKay graph power sums: p_2 = 16 = (a1-1)^2, p_4/p_2 = 3 = N_gen
  2. SU(2) Chern-Simons at level k = a1-2: Fibonacci anyons with d = phi
  3. Total quantum dimension D^2 = a1 + sqrt(a1)
  4. Galois norm/trace: D^2*D'^2 = a1*(a1-1), D^2+D'^2 = 2*a1
  5. Entropy difference: gamma - gamma' = ln(phi)
  6. Verlinde ring isomorphic to Z[phi]
  7. McKay chirality entanglement: S = ln((a1-1)^2)
  8. Moment-Weinberg theorem: a4/a2 * sin^2(tW) = N_gen/2

Dependencies: numpy (standard scientific Python).
No project imports.
Encoding: ASCII only (safe for Windows cp1252).

Author: Razvan-Constantin Anghelina
Version: 4.0 (February 2026)
"""

import math
import numpy as np

# ============================================================================
# CONSTANTS
# ============================================================================

a1 = 5
b1 = a1 + 1  # = 6
phi = (1.0 + math.sqrt(a1)) / 2.0
phi_prime = (1.0 - math.sqrt(a1)) / 2.0  # = -1/phi
N = 120
N_gen = 3
N_eig = 9
rank_E8 = 8
TOL = 1e-10

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


def print_section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)
    print()


# ============================================================================
# PART 1: McKAY GRAPH POWER SUMS
# ============================================================================

print_section("VERIFY TQFT, ENTANGLEMENT, AND MOMENT-WEINBERG THEOREM")

print("--- PART 1: McKay graph adjacency eigenvalues ---")

# Construct affine E8 (McKay graph of 2I)
# Correct edges: (0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(5,8)
E8_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (5, 8)]
n_mckay = 9

A_mckay = np.zeros((n_mckay, n_mckay))
for i, j in E8_edges:
    A_mckay[i][j] = 1
    A_mckay[j][i] = 1

# Verify degrees
degrees = np.sum(A_mckay, axis=1).astype(int)
check(degrees[5] == 3, "Node 5 (branch point) has degree 3",
      f"Found degree {degrees[5]}")
check(sum(1 for d in degrees if d == 1) == 3,
      "Exactly 3 leaves (degree 1)")

# Eigenvalues
eigs = sorted(np.linalg.eigvalsh(A_mckay), reverse=True)

# Expected: {2, phi, 1, 1/phi, 0, -1/phi, -1, -phi, -2}
expected_eigs = sorted([2, phi, 1, 1/phi, 0, -1/phi, -1, -phi, -2],
                       reverse=True)

eig_match = all(abs(eigs[i] - expected_eigs[i]) < TOL
                for i in range(n_mckay))
check(eig_match, "Eigenvalues = {2, phi, 1, 1/phi, 0, -1/phi, -1, -phi, -2}")

# Power sums
p2 = sum(e**2 for e in eigs)
p4 = sum(e**4 for e in eigs)
p2_int = round(p2)
p4_int = round(p4)

check(p2_int == 16, f"p_2 = Tr(A^2) = 16 = (a1-1)^2",
      f"Found {p2_int}")
check(p2_int == 2 * rank_E8, f"p_2 = 2*rank(E8) = {2*rank_E8}")
check(p4_int == 48, f"p_4 = Tr(A^4) = 48 = 16*N_gen",
      f"Found {p4_int}")
check(abs(p4 / p2 - N_gen) < TOL, f"p_4/p_2 = N_gen = {N_gen}",
      f"Found {p4/p2:.10f}")

# Uniqueness: (a1-1)^2 = 2*rank only for E8
for name, a_val, rank in [("E6", 3, 6), ("E7", 4, 7), ("E8", 5, 8)]:
    lhs = (a_val - 1)**2
    rhs = 2 * rank
    if name == "E8":
        check(lhs == rhs, f"(a1-1)^2 = 2*rank for E8: {lhs} = {rhs}")
    else:
        check(lhs != rhs, f"(a1-1)^2 != 2*rank for {name}: {lhs} != {rhs}")


# ============================================================================
# PART 2: FIBONACCI ANYONS FROM SU(2) CHERN-SIMONS
# ============================================================================

print_section("PART 2: Fibonacci anyons from SU(2) Chern-Simons")

k = a1 - 2  # CS level
check(k == 3, f"CS level k = a1-2 = {k}")

# Quantum dimensions: d_j = sin(pi*(2j+1)/(k+2)) / sin(pi/(k+2))
q_dims = []
for j2 in range(k + 1):  # j = 0, 1/2, 1, 3/2
    j = j2 / 2
    d_j = math.sin(math.pi * (j2 + 1) / a1) / math.sin(math.pi / a1)
    q_dims.append(d_j)

check(abs(q_dims[0] - 1.0) < TOL, "d(j=0) = 1")
check(abs(q_dims[1] - phi) < TOL, f"d(j=1/2) = phi = {phi:.6f}",
      f"Found {q_dims[1]:.10f}")
check(abs(q_dims[2] - phi) < TOL, f"d(j=1) = phi")
check(abs(q_dims[3] - 1.0) < TOL, "d(j=3/2) = 1")

# d_tau = 2*cos(pi/a1) = phi
d_tau = 2 * math.cos(math.pi / a1)
check(abs(d_tau - phi) < TOL, f"d_tau = 2*cos(pi/a1) = phi",
      f"2*cos(pi/{a1}) = {d_tau:.10f}")


# ============================================================================
# PART 3: TOTAL QUANTUM DIMENSIONS AND GALOIS
# ============================================================================

print_section("PART 3: Total quantum dimensions and Galois norm")

D_sq = sum(d**2 for d in q_dims)
D_sq_exact = a1 + math.sqrt(a1)

check(abs(D_sq - D_sq_exact) < TOL,
      f"D^2 = a1 + sqrt(a1) = {a1} + sqrt({a1}) = {D_sq_exact:.6f}",
      f"Found {D_sq:.10f}")

# Galois conjugate: sqrt(5) -> -sqrt(5)
D_prime_sq = a1 - math.sqrt(a1)

# Galois conjugate quantum dimensions
q_dims_conj = []
for j2 in range(k + 1):
    d_j_prime = math.sin(math.pi * (j2 + 1) * 2 / a1) / \
                math.sin(2 * math.pi / a1)
    q_dims_conj.append(d_j_prime)

D_prime_sq_computed = sum(d**2 for d in q_dims_conj)
check(abs(D_prime_sq_computed - D_prime_sq) < TOL,
      f"D'^2 = a1 - sqrt(a1) = {D_prime_sq:.6f}",
      f"Found {D_prime_sq_computed:.10f}")

# Non-unitary check: some conjugate dimensions are negative
check(q_dims_conj[2] < 0, "d'(j=1) < 0 (non-unitary conjugate TQFT)",
      f"d'(j=1) = {q_dims_conj[2]:.6f}")

# Galois norm
galois_norm = D_sq * D_prime_sq
check(abs(galois_norm - a1 * (a1 - 1)) < TOL,
      f"D^2 * D'^2 = a1*(a1-1) = {a1*(a1-1)}",
      f"Found {galois_norm:.10f}")

# Galois trace
galois_trace = D_sq + D_prime_sq
check(abs(galois_trace - 2 * a1) < TOL,
      f"D^2 + D'^2 = 2*a1 = {2*a1}",
      f"Found {galois_trace:.10f}")

# Minimal polynomial
# x^2 - 2*a1*x + a1*(a1-1) = 0
poly_check = D_sq**2 - 2 * a1 * D_sq + a1 * (a1 - 1)
check(abs(poly_check) < TOL,
      "D^2 satisfies x^2 - 2*a1*x + a1*(a1-1) = 0",
      f"Residual: {poly_check:.2e}")

# Ratio D^2/D'^2 = phi^2
ratio = D_sq / D_prime_sq
check(abs(ratio - phi**2) < TOL,
      f"D^2/D'^2 = phi^2 = phi+1 = {phi**2:.6f}",
      f"Found {ratio:.10f}")

# Galois norm of quantum dimension
d_tau_prime = abs(q_dims_conj[1])  # = 1/phi
check(abs(d_tau_prime - 1.0 / phi) < TOL,
      f"|d'_tau| = 1/phi = {1/phi:.6f}")
check(abs(phi * d_tau_prime - 1.0) < TOL,
      "d_tau * |d'_tau| = 1 (Galois norm)")


# ============================================================================
# PART 4: TOPOLOGICAL ENTANGLEMENT ENTROPIES
# ============================================================================

print_section("PART 4: Topological entanglement entropies")

gamma = 0.5 * math.log(D_sq)
gamma_prime = 0.5 * math.log(D_prime_sq)

print(f"  gamma = ln(D) = {gamma:.10f}")
print(f"  gamma' = ln(D') = {gamma_prime:.10f}")

# KEY: gamma - gamma' = ln(phi)
diff = gamma - gamma_prime
check(abs(diff - math.log(phi)) < TOL,
      f"gamma - gamma' = ln(phi) = {math.log(phi):.10f}",
      f"Found {diff:.10f}")

# gamma + gamma' = (1/2)*ln(a1*(a1-1))
summ = gamma + gamma_prime
expected_sum = 0.5 * math.log(a1 * (a1 - 1))
check(abs(summ - expected_sum) < TOL,
      f"gamma + gamma' = (1/2)*ln(a1*(a1-1)) = {expected_sum:.10f}",
      f"Found {summ:.10f}")

# Equivalently: gamma + gamma' = ln(2) + (1/2)*ln(a1)
alt_sum = math.log(2) + 0.5 * math.log(a1)
check(abs(summ - alt_sum) < TOL,
      "gamma + gamma' = ln(2) + (1/2)*ln(a1)")


# ============================================================================
# PART 5: VERLINDE RING = Z[phi]
# ============================================================================

print_section("PART 5: Verlinde fusion ring")

# Modular S-matrix for SU(2)_3
S = np.zeros((k + 1, k + 1))
for i in range(k + 1):
    for j in range(k + 1):
        S[i][j] = math.sqrt(2.0 / a1) * \
                   math.sin(math.pi * (i + 1) * (j + 1) / a1)

# S^2 = identity (for SU(2)_k)
S_sq = S @ S
check(np.allclose(S_sq, np.eye(k + 1), atol=1e-10),
      "S^2 = I (modular S-matrix is involutory)")

# Fusion rules from Verlinde formula: N_{ij}^k = sum_l S_il S_jl S*_kl / S_0l
# Check tau x tau = 1 + tau (Fibonacci fusion)
# tau = j=1/2, index 1
N_tau_tau = np.zeros(k + 1)
for kk in range(k + 1):
    N_tau_tau[kk] = sum(
        S[1][l] * S[1][l] * S[kk][l] / S[0][l]
        for l in range(k + 1)
    )

N_tau_tau_int = [round(x) for x in N_tau_tau]
check(N_tau_tau_int == [1, 0, 1, 0],
      "tau x tau = 1 + tau (Fibonacci fusion rule)",
      f"N_{{tau,tau}}^k = {N_tau_tau_int}")

# This corresponds to phi^2 = 1 + phi
check(abs(phi**2 - 1 - phi) < TOL,
      "phi^2 = 1 + phi (golden ratio = Fibonacci fusion)",
      f"phi^2 = {phi**2:.10f}, 1+phi = {1+phi:.10f}")


# ============================================================================
# PART 6: McKAY CHIRALITY ENTANGLEMENT
# ============================================================================

print_section("PART 6: McKay chirality entanglement")

# Tight-binding Hamiltonian H = -A at half-filling
evals_tb, evecs_tb = np.linalg.eigh(-A_mckay)

# Half-filling: 5 occupied states (majority filling for 9 sites)
n_occ = 5
C_full = evecs_tb[:, :n_occ] @ evecs_tb[:, :n_occ].T

# Bipartition: BLACK = {0,2,4,6,8}, WHITE = {1,3,5,7}
black = [0, 2, 4, 6, 8]
white = [1, 3, 5, 7]

C_white = C_full[np.ix_(white, white)]
evals_white = np.linalg.eigvalsh(C_white)

# Entanglement entropy
S_ent = 0
for ev in evals_white:
    if 1e-12 < ev < 1 - 1e-12:
        S_ent -= ev * math.log(ev) + (1 - ev) * math.log(1 - ev)

# All eigenvalues should be 1/2 (maximally mixed)
all_half = all(abs(ev - 0.5) < 1e-8 for ev in evals_white)
check(all_half, "All correlation eigenvalues = 1/2 (maximal entanglement)",
      f"Eigenvalues: {[round(e, 8) for e in evals_white]}")

expected_S = 4 * math.log(2)
check(abs(S_ent - expected_S) < 1e-8,
      f"S(chirality) = 4*ln(2) = ln(16) = ln((a1-1)^2) = {expected_S:.6f}",
      f"Found {S_ent:.10f}")

# Also check for n_occ = 4
n_occ_4 = 4
C_4 = evecs_tb[:, :n_occ_4] @ evecs_tb[:, :n_occ_4].T
C_white_4 = C_4[np.ix_(white, white)]
evals_white_4 = np.linalg.eigvalsh(C_white_4)
S_ent_4 = 0
for ev in evals_white_4:
    if 1e-12 < ev < 1 - 1e-12:
        S_ent_4 -= ev * math.log(ev) + (1 - ev) * math.log(1 - ev)

check(abs(S_ent_4 - expected_S) < 1e-8,
      "S(chirality) = 4*ln(2) also at n_occ=4 (robust)",
      f"Found {S_ent_4:.10f}")


# ============================================================================
# PART 7: BRAIDING PHASE
# ============================================================================

print_section("PART 7: Braiding phase")

theta_tau = 4 * math.pi / a1
check(abs(theta_tau - math.radians(144)) < TOL,
      f"Topological spin theta_tau = 4*pi/a1 = 144 deg",
      f"= {math.degrees(theta_tau):.6f} deg")


# ============================================================================
# PART 8: MOMENT-WEINBERG THEOREM
# ============================================================================

print_section("PART 8: Moment-Weinberg theorem")

# Heat kernel (Seeley-DeWitt) convention for graph Laplacian L = D - A:
#   a_2 = Tr(L)     = N*d          (first Seeley-DeWitt coefficient)
#   a_4 = Tr(L^2)/2 = N*d*(d+1)/2  (second coefficient, factor 1/2 from expansion)
# For d-regular graph: a_4/a_2 = (d+1)/2
# See exp401 for full derivation.

DEG = 2 * b1  # = 12 (Cayley graph degree)
n_600 = 120

# Heat kernel coefficients
hk_a2 = n_600 * DEG                    # = Tr(L) = 1440
hk_a4 = n_600 * DEG * (DEG + 1) // 2   # = Tr(L^2)/2 = 9360
hk_ratio = (DEG + 1) / 2.0             # = 13/2 = 6.5

check(abs(hk_ratio - hk_a4 / hk_a2) < TOL,
      f"a_4/a_2 = (DEG+1)/2 = {(DEG+1)/2} (heat kernel convention)",
      f"a_2={hk_a2}, a_4={hk_a4}, ratio={hk_a4/hk_a2:.6f}")

sin2_tW = b1 / (a1**2 + 1)  # = 6/26

# The algebraic identity (from exp401):
# (2*a1+3)*(a1+1) = 3*(a1^2+1) is equivalent to a1*(a1-5) = 0
lhs_alg = (2 * a1 + 3) * (a1 + 1)
rhs_alg = 3 * (a1**2 + 1)
check(lhs_alg == rhs_alg,
      f"(2*a1+3)*(a1+1) = 3*(a1^2+1): {lhs_alg} = {rhs_alg}")

# Factored form
factored = a1 * (a1 - 5)
check(factored == 0,
      f"Equivalent to a1*(a1-5) = 0: {a1}*({a1}-5) = {factored}")

# Numerical verification
mw_product = hk_ratio * sin2_tW
check(abs(mw_product - N_gen / 2.0) < 1e-10,
      f"a_4/a_2 * sin^2(tW) = N_gen/2 = {N_gen/2}",
      f"Found {mw_product:.10f}")

# Also verify using explicit Cayley graph eigenvalues
# Cayley graph of 2I: eigenvalues from representation theory (mult = dim^2)
cayley_eigs = [
    (12.0,          1),
    (6*phi,         4),
    (4*phi,         9),
    (3.0,          16),
    (0.0,          25),
    (-2.0,         36),
    (4*phi_prime,   9),
    (-3.0,         16),
    (6*phi_prime,   4),
]
total_mult = sum(m for _, m in cayley_eigs)
check(total_mult == n_600, f"Total multiplicity = {n_600}")

# Verify Tr(L) and Tr(L^2) from spectrum
TrL = sum(m * (DEG - lam) for lam, m in cayley_eigs)
TrL2 = sum(m * (DEG - lam)**2 for lam, m in cayley_eigs)
check(abs(TrL - n_600 * DEG) < TOL,
      f"Tr(L) = N*DEG = {n_600*DEG} (from spectrum)",
      f"Found {TrL:.1f}")
check(abs(TrL2 - n_600 * DEG * (DEG + 1)) < TOL,
      f"Tr(L^2) = N*DEG*(DEG+1) = {n_600*DEG*(DEG+1)} (from spectrum)",
      f"Found {TrL2:.1f}")


# ============================================================================
# SUMMARY
# ============================================================================

print_section("SUMMARY")

print(f"  Tests passed: {N_PASS}")
print(f"  Tests failed: {N_FAIL}")
print()

if N_FAIL == 0:
    print("  ALL TESTS PASSED")
    print()
    print("  Key results verified:")
    print(f"    p_2(McKay) = 16 = (a1-1)^2 = 2*rank(E8)")
    print(f"    p_4/p_2 = 3 = N_gen (generation count from McKay spectrum)")
    print(f"    d_tau = phi (Fibonacci anyon quantum dimension)")
    print(f"    D^2 = a1+sqrt(a1), D'^2 = a1-sqrt(a1)")
    print(f"    D^2*D'^2 = a1*(a1-1) = 20 = N/b1 (Galois norm)")
    print(f"    gamma - gamma' = ln(phi) (entropy asymmetry)")
    print(f"    tau x tau = 1 + tau (Verlinde = Z[phi])")
    print(f"    S(chirality) = ln(16) = ln(fermions/gen)")
    print(f"    a4/a2 * sin^2(tW) = N_gen/2 (Moment-Weinberg theorem)")
else:
    print(f"  WARNING: {N_FAIL} tests failed!")

import sys
sys.exit(0 if N_FAIL == 0 else 1)
