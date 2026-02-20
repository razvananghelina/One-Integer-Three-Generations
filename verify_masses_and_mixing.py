"""
verify_masses_and_mixing.py
============================
Self-contained verification of fermion masses and mixing angles derived from
the single integer a1 = 5 in the 600-cell / E8 framework.

What this script does:
  1. Defines constants: a1=5, b1=6, phi, N=120, N_gen=3, N_eig=9, degree=12.
  2. Verifies bare fermion masses  m_f = m_e * phi^(5a + 6b)  for all 9 fermions.
  3. Verifies norm-log corrected masses (C = 2/13, zero free parameters).
  4. Verifies CKM mixing angles from bare exponents + corrections.
  5. Verifies PMNS mixing angles from A5 representation theory.
  6. Verifies CKM and PMNS CP-violation phases.
  7. Prints a summary table.

Dependencies: numpy (for sqrt, pi, arctan, sin -- no exotic packages).
Encoding: ASCII only (safe for Windows cp1252).

Author: Razvan-Constantin Anghelina
Version: 3.8 (February 2026)
"""

import numpy as np

# ============================================================================
# SECTION 0: THE SINGLE FREE INTEGER AND DERIVED CONSTANTS
# ============================================================================

a1     = 5                               # THE one integer
b1     = a1 + 1                          # = 6
phi    = (1.0 + np.sqrt(a1)) / 2.0       # golden ratio = 1.6180339887...
N      = 120                             # = a1! = |icosahedral double group|
N_gen  = 3                               # number of generations (Nyquist on icosahedron)
N_eig  = 9                               # distinct eigenvalues of Laplacian on 600-cell
degree = 12                              # vertex degree = 2 * b1

# Mass scale (the one dimensional input)
m_e = 0.51099895  # MeV (electron mass, CODATA 2022)

# Derived coupling constants
alpha_s_fw = 1.0 / (2.0 * phi**3)       # framework strong coupling
sin2tW     = float(b1) / (a1**2 + 1)    # = 6/26 = 0.230769...

# Fine-structure constant from spectral equation: 2*pi*a^2 - 4*a1*phi^4*a + 1 = 0
A_coef = 2.0 * np.pi
B_coef = -4.0 * a1 * phi**4
C_coef = 1.0
disc   = B_coef**2 - 4.0 * A_coef * C_coef
alpha  = (-B_coef - np.sqrt(disc)) / (2.0 * A_coef)  # smaller root

# ============================================================================
# SECTION 1: PDG / EXPERIMENTAL REFERENCE VALUES
# ============================================================================

# --- Fermion masses (PDG 2024) ---
# Leptons: pole masses in MeV
# Quarks: MS-bar running masses at conventional scales in MeV
# (u,d,s at 2 GeV; c at m_c; b at m_b; t is pole mass)
pdg_masses = {
    'e':   0.51099895,
    'mu':  105.6584,
    'tau': 1776.86,
    'u':   2.16,
    'c':   1270.0,
    't':   172760.0,
    'd':   4.67,
    's':   93.4,
    'b':   4180.0,
}

# --- CKM mixing angles (PDG 2024, sine values) ---
sin_ckm_12_pdg = 0.22500   # sin(theta_12) = lambda (Wolfenstein)
sin_ckm_23_pdg = 0.04182   # sin(theta_23)
sin_ckm_13_pdg = 0.00369   # sin(theta_13)

# --- PMNS mixing angles (PDG 2024, sin^2 values) ---
sin2_pmns_12_pdg = 0.303    # sin^2(theta_12)
sin2_pmns_23_pdg = 0.546    # sin^2(theta_23)
sin2_pmns_13_pdg = 0.02203  # sin^2(theta_13)

# --- CP-violation phases ---
delta_ckm_pdg  = 65.4       # degrees (PDG central value, +/- 3.2 deg)
delta_pmns_pdg = 197.0       # degrees (PDG central value, +/- 25 deg)


# ============================================================================
# SECTION 2: HELPER FUNCTIONS
# ============================================================================

def pct_error(predicted, observed):
    """Signed percentage error: (predicted - observed) / observed * 100."""
    if observed == 0:
        return float('inf')
    return (predicted - observed) / observed * 100.0

def abs_pct_error(predicted, observed):
    """Absolute percentage error."""
    return abs(pct_error(predicted, observed))

def rms(values):
    """Root mean square of a list of values."""
    arr = np.array(values)
    return np.sqrt(np.mean(arr**2))

def print_divider(char='=', width=78):
    print(char * width)

def print_section(title):
    print()
    print_divider()
    print(title)
    print_divider()
    print()


# ============================================================================
# SECTION 3: BARE FERMION MASSES
# ============================================================================

# (a, b) quantum number assignments (from exp231, Fibonacci-Galois structure)
# Fermion: name, generation, a, b
fermion_data = [
    ('e',   0,  0,  0),
    ('mu',  1,  1,  1),
    ('tau', 2,  1,  2),
    ('u',   0,  3, -2),
    ('c',   1,  2,  1),
    ('t',   2,  4,  1),
    ('d',   0,  1,  0),
    ('s',   1,  1,  1),
    ('b',   2, -1,  4),
]

print_section("SECTION 3: BARE FERMION MASSES -- m_f = m_e * phi^(5a + 6b)")

print("Constants:")
print("  a1 = %d,  b1 = %d,  phi = %.10f" % (a1, b1, phi))
print("  N = %d,  N_gen = %d,  N_eig = %d,  degree = %d" % (N, N_gen, N_eig, degree))
print("  m_e = %.8f MeV" % m_e)
print("  alpha_s(fw) = 1/(2*phi^3) = %.6f" % alpha_s_fw)
print("  sin^2(tW) = %d/%d = %.6f" % (b1, a1**2 + 1, sin2tW))
print("  alpha = %.10f  (1/alpha = %.6f)" % (alpha, 1.0/alpha))
print()

print("%-6s %4s %6s %5s %12s %12s %10s" % (
    "Name", "n", "(a,b)", "gen", "m_pred/MeV", "m_exp/MeV", "Error"))
print("-" * 62)

bare_errors = []

for name, gen, a, b in fermion_data:
    n = a1 * a + b1 * b     # = 5a + 6b
    m_pred = m_e * phi**n
    m_exp  = pdg_masses[name]
    if name == 'e':
        err_str = "(input)"
    else:
        err = pct_error(m_pred, m_exp)
        bare_errors.append(abs(err))
        err_str = "%+.2f%%" % err
    print("%-6s %4d (%+2d,%+2d) %3d %12.2f %12.2f %10s" % (
        name, n, a, b, gen, m_pred, m_exp, err_str))

bare_rms = rms(bare_errors)
print()
print("Bare mass RMS error (8 fermions, excl. electron): %.2f%%" % bare_rms)


# ============================================================================
# SECTION 4: NORM-LOG CORRECTED FERMION MASSES (Paper Eq. 43-44)
# ============================================================================

print_section("SECTION 4: NORM-LOG CORRECTED MASSES (zero free parameters)")

# Universal coefficient: C = 4 / (a1^2 + 1) = 2/13
C_coeff = 4.0 / (a1**2 + 1)    # = 2/13 = 0.15385

# Lepton coefficient: c_ell = C * phi^3 / dim(ST)
# where dim(ST) = Tr(phi^3) = 4 = dimension of spacetime
d_ST = 4.0  # = Tr(phi^3) = phi^3 + phi'^3 = 4
c_ell = C_coeff * phi**3 / d_ST

print("Correction parameters (all derived from a1 = %d):" % a1)
print("  C = 4/(a1^2 + 1) = 4/%d = %.6f = 2*sin^2(tW)/N_gen" % (a1**2+1, C_coeff))
print("  c_ell = C*phi^3/d_ST = %.6f * %.4f / %d = %.6f" % (
    C_coeff, phi**3, int(d_ST), c_ell))
print("  Lepton exponent k = 3/4 = 1 - 1/d_ST")
print()

# Norm in Z[phi]: N(a + b*phi) = a^2 + a*b - b^2
def zphi_norm(a, b):
    return a**2 + a*b - b**2

# Galois conjugate: z' = a + b*phi', where phi' = (1-sqrt(5))/2
phi_conj = (1.0 - np.sqrt(a1)) / 2.0  # = -1/phi

def galois_conj(a, b):
    return a + b * phi_conj

# Compute norm-log correction for each fermion
def normlog_delta(name, a, b):
    """Compute delta_f for fermion with quantum numbers (a,b)."""
    z_prime = galois_conj(a, b)
    N_z = zphi_norm(a, b)

    # Electron: z' = 0, delta = 0
    if name == 'e':
        return 0.0, "electron (z'=0)"

    # Leptons: delta = c_ell * sign(z') * |z'|^(3/4)
    if name in ('mu', 'tau'):
        delta = c_ell * np.sign(z_prime) * abs(z_prime)**0.75
        return delta, "lepton: c_ell*sign(z')*|z'|^{3/4}"

    # Up quarks (T3 = +1/2)
    if name in ('u', 'c', 't'):
        if abs(N_z) <= 1:
            return 0.0, "up quark, |N|=1"
        delta = C_coeff * np.log(abs(N_z))
        return delta, "+C*ln|N|, N=%d" % N_z

    # Down quarks (T3 = -1/2)
    if name in ('d', 's', 'b'):
        if b == 0:
            # Rational sector
            return -2.0/a1, "-2/a1 (rational, b=0)"
        if abs(N_z) <= 1:
            # Unit sector
            delta = -N_gen * C_coeff / phi**2
            return delta, "-N_gen*C/phi^2 (unit, |N|=1)"
        # Prime sector
        delta = -C_coeff * np.log(abs(N_z)) / phi
        return delta, "-C*ln|N|/phi, N=%d" % N_z

    return 0.0, "unknown"

print("Norm-log correction formula (Eq. 43 in paper):")
print("  Unified quark formula (|N|>1):")
print("    delta_q = 2*T3 * C * ln|N(z)| * phi^(T3-1/2)")
print("  Leptons: delta_ell = c_ell * sign(z') * |z'|^(3/4)")
print("  Three Z[phi] sectors: rational (b=0), unit (|N|=1), prime (|N|>1)")
print()

print("%-6s (%+2s,%+2s) %6s %10s %8s %12s %12s %10s" % (
    "Name", "a", "b", "N(z)", "delta", "n_eff", "m_corr/MeV", "m_exp/MeV",
    "Error"))
print("-" * 88)

corrected_errors = []
corrected_quark_errors = []
corrected_lepton_errors = []

for name, gen, a, b in fermion_data:
    m_exp = pdg_masses[name]
    n_bare = a1 * a + b1 * b
    N_z = zphi_norm(a, b)

    if name == 'e':
        print("%-6s (%+2d,%+2d) %6d %10s %8s %12.4f %12.4f %10s" % (
            name, a, b, N_z, "0", "0.000", m_exp, m_exp, "(input)"))
        continue

    delta, desc = normlog_delta(name, a, b)
    n_eff = n_bare + delta
    m_corr = m_e * phi**n_eff

    err_corr = pct_error(m_corr, m_exp)
    corrected_errors.append(abs(err_corr))

    if name in ('u', 'c', 't', 'd', 's', 'b'):
        corrected_quark_errors.append(abs(err_corr))
    if name in ('mu', 'tau'):
        corrected_lepton_errors.append(abs(err_corr))

    print("%-6s (%+2d,%+2d) %6d %+10.4f %8.3f %12.4f %12.4f %+9.2f%%" % (
        name, a, b, N_z, delta, n_eff, m_corr, m_exp, err_corr))

corr_rms = rms(corrected_errors)
quark_rms = rms(corrected_quark_errors) if corrected_quark_errors else 0
lepton_rms = rms(corrected_lepton_errors) if corrected_lepton_errors else 0
print()
print("RMS errors (norm-log corrected, zero free parameters):")
print("  All 8 fermions (excl. electron): %.3f%%" % corr_rms)
print("  6 quarks:                        %.3f%%" % quark_rms)
print("  2 leptons (mu, tau):             %.3f%%" % lepton_rms)
print("  Improvement: bare %.1f%% -> corrected %.2f%%" % (bare_rms, corr_rms))
print()

# Verify unified quark formula for |N|>1 quarks
print("Unified quark formula verification (Eq. 44):")
print("  delta_q = 2*T3 * C * ln|N| * phi^(T3-1/2)")
for name, gen, a, b in fermion_data:
    N_z = zphi_norm(a, b)
    if name in ('c', 't'):  # up quarks with |N|>1
        T3 = 0.5
        delta_unified = 2*T3 * C_coeff * np.log(abs(N_z)) * phi**(T3-0.5)
        delta_direct, _ = normlog_delta(name, a, b)
        print("  %s: T3=%+.1f, |N|=%d, delta_unified=%.4f, delta_direct=%.4f, match=%s" % (
            name, T3, abs(N_z), delta_unified, delta_direct,
            "PASS" if abs(delta_unified - delta_direct) < 1e-10 else "FAIL"))
    if name == 'b':  # down quark with |N|>1
        T3 = -0.5
        delta_unified = 2*T3 * C_coeff * np.log(abs(N_z)) * phi**(T3-0.5)
        delta_direct, _ = normlog_delta(name, a, b)
        print("  %s: T3=%+.1f, |N|=%d, delta_unified=%.4f, delta_direct=%.4f, match=%s" % (
            name, T3, abs(N_z), delta_unified, delta_direct,
            "PASS" if abs(delta_unified - delta_direct) < 1e-10 else "FAIL"))

# Build lookup for summary section
fermion_sector = {}
for name, gen, a, b in fermion_data:
    if name in ('mu', 'tau', 'e'):
        fermion_sector[name] = 'leptons'
    elif name in ('u', 'c', 't'):
        fermion_sector[name] = 'up_quarks'
    else:
        fermion_sector[name] = 'down_quarks'


# ============================================================================
# SECTION 5: CKM MIXING ANGLES
# ============================================================================

print_section("SECTION 5: CKM MIXING ANGLES")

# Bare exponents (all derived from a1 = 5):
#   n_12 = N_gen = 3
#   n_23 = degree - a1 = 12 - 5 = 7
#   n_13 = degree = 12

n_12 = N_gen                  # = 3
n_23 = degree - a1            # = 7
n_13 = degree                 # = 12

# Correction coefficients (from E8 leg structure):
#   c_12 = 2/3         (color Casimir C_F(SU(3))/2 = Leg_B / N_eig)
#   c_23 = a1 = 5      (Leg_A - Leg_B on McKay graph)
#   c_13 = sin^2(tW)   (= 6/26, electroweak)

c_12 = 2.0 / 3.0             # = b1 / N_eig = 6/9 = 2/3
c_23 = float(a1)             # = 5
c_13 = sin2tW                # = 6/26

# CKM angles from the paper (Eq. in Section 5.1):
#   theta_12 = arctan(phi^(-n_12) * (1 - c_12 * alpha_s / pi))
#   theta_23 = arctan(phi^(-n_23) * (1 + c_23 * alpha_s / pi))
#   theta_13 = arctan(phi^(-n_13) * (1 + c_13))
#
# The corrections are multiplicative on the phi^(-n) suppression factor.
# For theta_12 and theta_23 the coupling is alpha_s/pi (QCD loop).
# For theta_13 the correction uses sin^2(tW) directly (electroweak).
# Then sin(theta) = sin(arctan(x)) = x / sqrt(1 + x^2).

print("Bare exponents (derived from a1 = %d):" % a1)
print("  n_12 = N_gen = %d" % n_12)
print("  n_23 = degree - a1 = %d - %d = %d" % (degree, a1, n_23))
print("  n_13 = degree = %d" % n_13)
print()

print("Identities check:")
print("  n_12 + n_23 = %d = 2*a1 = %d : %s" % (
    n_12 + n_23, 2*a1, "PASS" if n_12 + n_23 == 2*a1 else "FAIL"))
print("  n_12 * n_13 = %d = b1^2 = %d : %s" % (
    n_12 * n_13, b1**2, "PASS" if n_12 * n_13 == b1**2 else "FAIL"))
print("  n_23 + n_13 = %d : consistent" % (n_23 + n_13))
print()

print("Correction coefficients (from E8 leg ratios):")
print("  c_12 = b1/N_eig = %d/%d = 2/3    (color Casimir C_F/2)" % (b1, N_eig))
print("  c_23 = Leg_A - Leg_B = a1 = %d    (E8 leg difference)" % a1)
print("  c_13 = sin^2(tW) = %d/%d = %.6f   (electroweak)" % (b1, a1**2 + 1, c_13))
print()

print("Corrected CKM formula:")
print("  theta_12 = arctan(phi^(-3) * (1 - (2/3)*alpha_s/pi))")
print("  theta_23 = arctan(phi^(-7) * (1 + 5*alpha_s/pi))")
print("  theta_13 = arctan(phi^(-12) * (1 + sin^2(tW)))")
print()

# Compute multiplicative correction factors
corr_factor_12 = 1.0 - c_12 * alpha_s_fw / np.pi   # 1 - (2/3)*alpha_s/pi
corr_factor_23 = 1.0 + c_23 * alpha_s_fw / np.pi   # 1 + 5*alpha_s/pi
corr_factor_13 = 1.0 + c_13                         # 1 + sin^2(tW)

print("  Correction factors:")
print("    C_12 = 1 - (2/3)*alpha_s/pi = %.8f" % corr_factor_12)
print("    C_23 = 1 + 5*alpha_s/pi     = %.8f" % corr_factor_23)
print("    C_13 = 1 + sin^2(tW)        = %.8f" % corr_factor_13)
print()

# Compute CKM angles: theta = arctan(phi^(-n) * C)
x_12 = phi**(-n_12) * corr_factor_12
x_23 = phi**(-n_23) * corr_factor_23
x_13 = phi**(-n_13) * corr_factor_13

# theta in degrees
theta_ckm_12 = np.degrees(np.arctan(x_12))
theta_ckm_23 = np.degrees(np.arctan(x_23))
theta_ckm_13 = np.degrees(np.arctan(x_13))

# sin(arctan(x)) = x / sqrt(1 + x^2)
sin_ckm_12_pred = x_12 / np.sqrt(1.0 + x_12**2)
sin_ckm_23_pred = x_23 / np.sqrt(1.0 + x_23**2)
sin_ckm_13_pred = x_13 / np.sqrt(1.0 + x_13**2)

print("  Angles: theta_12 = %.3f deg, theta_23 = %.3f deg, theta_13 = %.4f deg" % (
    theta_ckm_12, theta_ckm_23, theta_ckm_13))
print()

print("%-12s %12s %12s %10s" % ("Angle", "Predicted", "PDG 2024", "Error"))
print("-" * 50)
print("%-12s %12.6f %12.6f %+9.2f%%" % (
    "sin(th_12)", sin_ckm_12_pred, sin_ckm_12_pdg,
    pct_error(sin_ckm_12_pred, sin_ckm_12_pdg)))
print("%-12s %12.6f %12.6f %+9.2f%%" % (
    "sin(th_23)", sin_ckm_23_pred, sin_ckm_23_pdg,
    pct_error(sin_ckm_23_pred, sin_ckm_23_pdg)))
print("%-12s %12.6f %12.6f %+9.2f%%" % (
    "sin(th_13)", sin_ckm_13_pred, sin_ckm_13_pdg,
    pct_error(sin_ckm_13_pred, sin_ckm_13_pdg)))


# ============================================================================
# SECTION 6: PMNS MIXING ANGLES
# ============================================================================

print_section("SECTION 6: PMNS MIXING ANGLES")

# All three PMNS angles derived from A5 representation theory:
#
#   sin^2(th_23) = dim(4) / (dim(3) + dim(4)) = 4/7
#       From A5 Clebsch-Gordan: 3 x 3' = 4 + 5
#       Atmospheric mixing = branching ratio into 4-dim channel
#
#   sin^2(th_12) = 2 / (phi + a1)
#       Numerator: |chi_3 - chi_3'|^2 = 2 (Galois distance between 3 and 3')
#       Denominator: phi + a1 (TBM + Galois renormalization)
#       Equivalent: (1/3) * b1 / (phi + a1) since 2 = (1/3)*b1
#
#   sin^2(th_13) = 1 / (a1 * N_eig) = 1/45
#       Spectral democratic suppression: 1/(dim(l=2) * total_modes)
#       l=0 -> l=2 transition on Hopf base S^2

sin2_pmns_23_pred = 4.0 / 7.0                    # = (a1-1)/(a1+2)
sin2_pmns_12_pred = 2.0 / (phi + a1)             # TBM + Galois renorm
sin2_pmns_13_pred = 1.0 / (a1 * N_eig)           # = 1/45

# Cross-check closed forms
assert abs(sin2_pmns_23_pred - (a1 - 1.0)/(a1 + 2.0)) < 1e-15, "th23 identity failed"
assert abs(sin2_pmns_12_pred - (1.0/3.0)*b1/(phi + a1)) < 1e-15, "th12 identity failed"
assert abs(sin2_pmns_13_pred - 1.0/45.0) < 1e-15, "th13 identity failed"

print("PMNS angles from A5 representation theory:")
print()
print("  sin^2(th_23) = dim(4)/(dim(3)+dim(4)) = 4/7")
print("    From: 3 x 3' = 4 + 5  (A5 Clebsch-Gordan)")
print()
print("  sin^2(th_12) = 2/(phi + a1) = 2/(phi + 5)")
print("    From: TBM + Galois renormalization")
print("    Numerator 2 = |chi_3 - chi_3'|^2 (character distance)")
print()
print("  sin^2(th_13) = 1/(a1 * N_eig) = 1/45")
print("    From: spectral democratic suppression, l=0 -> l=2")
print()

print("%-14s %12s %12s %10s" % ("Angle", "Predicted", "PDG 2024", "Error"))
print("-" * 52)
print("%-14s %12.6f %12.6f %+9.2f%%" % (
    "sin^2(th_23)", sin2_pmns_23_pred, sin2_pmns_23_pdg,
    pct_error(sin2_pmns_23_pred, sin2_pmns_23_pdg)))
print("%-14s %12.6f %12.6f %+9.2f%%" % (
    "sin^2(th_12)", sin2_pmns_12_pred, sin2_pmns_12_pdg,
    pct_error(sin2_pmns_12_pred, sin2_pmns_12_pdg)))
print("%-14s %12.6f %12.6f %+9.2f%%" % (
    "sin^2(th_13)", sin2_pmns_13_pred, sin2_pmns_13_pdg,
    pct_error(sin2_pmns_13_pred, sin2_pmns_13_pdg)))


# ============================================================================
# SECTION 7: CP-VIOLATION PHASES
# ============================================================================

print_section("SECTION 7: CP-VIOLATION PHASES")

# CKM phase: delta_CKM = arctan(sqrt(a1)) = arctan(sqrt(5))
#   Half-angle between 3 and 3' irreps on 5-cycle classes.
#
# PMNS phase: delta_PMNS = N_gen * delta_CKM = 3 * arctan(sqrt(5))
#   Berry phase from generation loop; multiplied by N_gen.

delta_ckm_pred  = np.degrees(np.arctan(np.sqrt(a1)))            # arctan(sqrt(5)) in degrees
delta_pmns_pred = N_gen * np.degrees(np.arctan(np.sqrt(a1)))    # 3 * arctan(sqrt(5)) in degrees

# Derived quantities for the PMNS phase
cos_delta_pmns = np.cos(np.radians(delta_pmns_pred))

print("CKM CP phase:")
print("  delta_CKM = arctan(sqrt(a1)) = arctan(sqrt(5))")
print("  Predicted:  %.4f deg" % delta_ckm_pred)
print("  PDG 2024:   %.1f +/- 3.2 deg" % delta_ckm_pdg)
print("  Error:      %+.2f%%" % pct_error(delta_ckm_pred, delta_ckm_pdg))
print("  Deviation:  %.2f sigma" % (abs(delta_ckm_pred - delta_ckm_pdg) / 3.2))
print()

print("PMNS CP phase:")
print("  delta_PMNS = N_gen * arctan(sqrt(a1)) = 3 * arctan(sqrt(5))")
print("  Predicted:  %.4f deg" % delta_pmns_pred)
print("  PDG 2024:   %.0f +/- 25 deg" % delta_pmns_pdg)
print("  Error:      %+.2f%%" % pct_error(delta_pmns_pred, delta_pmns_pdg))
print("  Deviation:  %.2f sigma" % (abs(delta_pmns_pred - delta_pmns_pdg) / 25.0))
print()

print("  cos(delta_PMNS) = %.6f" % cos_delta_pmns)
print("  Analytic form: cos(delta_PMNS) = -7/(3*sqrt(b1)) = %.6f" % (
    -7.0 / (3.0 * np.sqrt(b1))))


# ============================================================================
# SECTION 8: SUMMARY TABLE
# ============================================================================

print_section("SECTION 8: SUMMARY OF ALL RESULTS")

print("All predictions derive from the single integer a1 = %d" % a1)
print("plus the electron mass m_e = %.8f MeV (dimensional anchor)." % m_e)
print("No free parameters are adjusted.")
print()

# Collect all results
results = []

# Bare masses
for name, gen, a, b in fermion_data:
    if name == 'e':
        continue
    n = a1 * a + b1 * b
    m_pred = m_e * phi**n
    m_exp  = pdg_masses[name]
    err = pct_error(m_pred, m_exp)
    results.append(("m_%s (bare)" % name, "%.2f MeV" % m_pred,
                     "%.2f MeV" % m_exp, "%+.2f%%" % err))

# Norm-log corrected masses
for name, gen, a, b in fermion_data:
    if name == 'e':
        continue
    m_exp = pdg_masses[name]
    n_bare = a1 * a + b1 * b
    delta, _ = normlog_delta(name, a, b)
    n_eff = n_bare + delta
    m_corr = m_e * phi**n_eff
    err = pct_error(m_corr, m_exp)
    results.append(("m_%s (corr)" % name, "%.2f MeV" % m_corr,
                     "%.2f MeV" % m_exp, "%+.2f%%" % err))

# CKM
results.append(("sin(th12) CKM", "%.6f" % sin_ckm_12_pred,
                "%.6f" % sin_ckm_12_pdg,
                "%+.2f%%" % pct_error(sin_ckm_12_pred, sin_ckm_12_pdg)))
results.append(("sin(th23) CKM", "%.6f" % sin_ckm_23_pred,
                "%.6f" % sin_ckm_23_pdg,
                "%+.2f%%" % pct_error(sin_ckm_23_pred, sin_ckm_23_pdg)))
results.append(("sin(th13) CKM", "%.6f" % sin_ckm_13_pred,
                "%.6f" % sin_ckm_13_pdg,
                "%+.2f%%" % pct_error(sin_ckm_13_pred, sin_ckm_13_pdg)))

# PMNS
results.append(("sin2(th23) PMNS", "%.6f" % sin2_pmns_23_pred,
                "%.6f" % sin2_pmns_23_pdg,
                "%+.2f%%" % pct_error(sin2_pmns_23_pred, sin2_pmns_23_pdg)))
results.append(("sin2(th12) PMNS", "%.6f" % sin2_pmns_12_pred,
                "%.6f" % sin2_pmns_12_pdg,
                "%+.2f%%" % pct_error(sin2_pmns_12_pred, sin2_pmns_12_pdg)))
results.append(("sin2(th13) PMNS", "%.6f" % sin2_pmns_13_pred,
                "%.6f" % sin2_pmns_13_pdg,
                "%+.2f%%" % pct_error(sin2_pmns_13_pred, sin2_pmns_13_pdg)))

# CP phases
results.append(("delta_CKM", "%.2f deg" % delta_ckm_pred,
                "%.1f deg" % delta_ckm_pdg,
                "%+.2f%%" % pct_error(delta_ckm_pred, delta_ckm_pdg)))
results.append(("delta_PMNS", "%.2f deg" % delta_pmns_pred,
                "%.0f deg" % delta_pmns_pdg,
                "%+.2f%%" % pct_error(delta_pmns_pred, delta_pmns_pdg)))

print("%-18s %16s %16s %10s" % ("Observable", "Predicted", "Experiment", "Error"))
print("-" * 64)
for label, pred, obs, err in results:
    print("%-18s %16s %16s %10s" % (label, pred, obs, err))

print()
print_divider()
print("RMS ERRORS:")
print("  Bare masses (8 fermions):       %.2f%%" % bare_rms)
print("  Corrected masses (8 fermions):  %.2f%%" % corr_rms)

# CKM RMS
ckm_errors = [
    abs_pct_error(sin_ckm_12_pred, sin_ckm_12_pdg),
    abs_pct_error(sin_ckm_23_pred, sin_ckm_23_pdg),
    abs_pct_error(sin_ckm_13_pred, sin_ckm_13_pdg),
]
print("  CKM angles (3 angles):          %.2f%%" % rms(ckm_errors))

# PMNS RMS
pmns_errors = [
    abs_pct_error(sin2_pmns_23_pred, sin2_pmns_23_pdg),
    abs_pct_error(sin2_pmns_12_pred, sin2_pmns_12_pdg),
    abs_pct_error(sin2_pmns_13_pred, sin2_pmns_13_pdg),
]
print("  PMNS angles (3 angles):         %.2f%%" % rms(pmns_errors))

# CP phases
cp_errors = [
    abs_pct_error(delta_ckm_pred, delta_ckm_pdg),
    abs_pct_error(delta_pmns_pred, delta_pmns_pdg),
]
print("  CP phases (2 phases):           %.2f%%" % rms(cp_errors))

print_divider()
print()
print("Framework inputs:  a1 = 5  (one integer)  +  m_e  (dimensional anchor)")
print("Free parameters:   0  (zero adjustable constants)")
print()
print("End of verification.")
