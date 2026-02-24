# Reproducible Verification Code

**Paper:** "One Integer, Three Generations: Deriving Particle Physics from a_1 = 5"
**Author:** Razvan-Constantin Anghelina
**Version:** 3.9 (February 2026)

## Requirements

- Python 3.8+
- NumPy
- SciPy

```
pip install numpy scipy
```

## Scripts

Each script is self-contained (no cross-imports) and prints PASS/FAIL for every claim.

| Script | What it verifies | Runtime |
|--------|-----------------|---------|
| `verify_coupling_constants.py` | alpha, alpha_s, sin^2(theta_W) from a_1=5 | <1s |
| `verify_spectrum_600cell.py` | 600-cell Laplacian spectrum, 9 eigenvalues in Z[phi], localization of 3/3' irreps in Galois-conjugate eigenspaces, spectral gap 4*sqrt(5) | ~30s |
| `verify_masses_and_mixing.py` | All 9 fermion masses (bare + norm-log corrected to 0.11% RMS), CKM angles, PMNS angles, CP phases | <1s |
| `verify_berry_phase.py` | Berry phase = 1/phi^4 over all 1200 faces, face-transitivity, 5 fiber values | ~60s |
| `verify_spectral_action.py` | Simplicial complex counts, boundary operators, Hodge decomposition 119+601=720, Betti numbers, Seeley-DeWitt coefficients, gauge group from A_5 | ~60s |
| `verify_mckay_chirality.py` | McKay graph = affine E8 (from 2I character table), bipartite chirality gamma_F = (-1)^{2j}, Casimir sum rules (sum C_2 = 26), Wilson line masses (9/9), fermionic action dimensions (H+ = H- = 39600), mass quantum number sum rules, rep ring Z/2 grading (45/45) | <1s |
| `verify_galois_kernel.py` | Galois kernel theorem: ker(b1*A_fiber - A) = rho_0+rho_1+rho_8 (dim=9), b1=6 unique, stability across Hopf fibrations, alpha*alpha'=1/(2pi) | ~60s |
| `verify_neutrino_masses.py` | Neutrino masses m_3=2*m_e/phi^35, splitting ratio r=alpha*phi^3, seesaw exponent n=35 (8 identities), PMNS angles, cosmological bounds | <1s |
| `verify_polytope_uniqueness.py` | All 6 regular 4D polytopes tested against 7 SM criteria. 600-cell: 7/7, 120-cell (dual): 6/7, all independent alternatives: 0/7. Uniqueness of a1=5, N_gen from ring structure, Weinberg angle, alpha, mass hierarchy, mixing angles, anomaly cancellation | <1s |

## Running

```bash
python verify_coupling_constants.py
python verify_spectrum_600cell.py
python verify_masses_and_mixing.py
python verify_berry_phase.py
python verify_spectral_action.py
python verify_mckay_chirality.py
python verify_galois_kernel.py
python verify_neutrino_masses.py
python verify_polytope_uniqueness.py
```

Or run all at once:
```bash
python run_all.py
```

## Expected Output

Each script prints a summary table ending with:
```
TOTAL: X/Y tests PASSED
```

All tests should pass. Any failure indicates either a bug in the verification code or a discrepancy with the paper's claims.

## What These Scripts Do NOT Verify

- The cosmological constant formula (pattern, not derivation)
- The dark matter abundance ratio (speculative)
- The 3D->4D continuum limit of the spectral triple

## License

MIT License. Use freely for verification purposes.
