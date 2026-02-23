"""
Run all verification scripts for the 600-cell framework paper.
"""
import os
import subprocess
import sys
import time

# Always run from the directory containing this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

scripts = [
    "verify_coupling_constants.py",
    "verify_spectrum_600cell.py",
    "verify_masses_and_mixing.py",
    "verify_berry_phase.py",
    "verify_spectral_action.py",
    "verify_mckay_chirality.py",
    "verify_galois_kernel.py",
    "verify_neutrino_masses.py",
]

results = {}
total_time = 0

for script in scripts:
    print("\n" + "=" * 70)
    print(f"RUNNING: {script}")
    print("=" * 70)
    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, script],
            capture_output=False,
            timeout=300
        )
        elapsed = time.time() - t0
        total_time += elapsed
        results[script] = "PASS" if result.returncode == 0 else "FAIL"
        print(f"\n  Completed in {elapsed:.1f}s -- {results[script]}")
    except subprocess.TimeoutExpired:
        results[script] = "TIMEOUT"
        print(f"\n  TIMEOUT after 300s")
    except Exception as e:
        results[script] = f"ERROR: {e}"
        print(f"\n  ERROR: {e}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for script, status in results.items():
    print(f"  {script:40s} {status}")
print(f"\nTotal time: {total_time:.1f}s")

n_pass = sum(1 for s in results.values() if s == "PASS")
print(f"Result: {n_pass}/{len(scripts)} scripts completed successfully.")
