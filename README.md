# Reproducible Verification Code

**Paper:** "One Integer, Three Generations: Deriving Particle Physics from a_1 = 5"
**Author:** Razvan-Constantin Anghelina
**Version:** 3.2 (February 2026)

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
| `verify_masses_and_mixing.py` | All 9 fermion masses (bare + corrected), CKM angles, PMNS angles, CP phases | <1s |
| `verify_berry_phase.py` | Berry phase = 1/phi^4 over all 1200 faces, face-transitivity, 5 fiber values | ~60s |
| `verify_spectral_action.py` | Simplicial complex counts, boundary operators, Hodge decomposition 119+601=720, Betti numbers, Seeley-DeWitt coefficients, gauge group from A_5 | ~60s |

## Running

```bash
python verify_coupling_constants.py
python verify_spectrum_600cell.py
python verify_masses_and_mixing.py
python verify_berry_phase.py
python verify_spectral_action.py
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

- The Diophantine equation a_1! = 4*a_1*(a_1+1) having a unique solution (trivial to check by hand)
- The holonomy correction sector assignments (these are algebraically motivated, not derived; see Proposition 3 in the paper)
- The cosmological constant formula (pattern, not derivation)
- The dark matter abundance ratio (speculative)

## License

MIT License. Use freely for verification purposes.
