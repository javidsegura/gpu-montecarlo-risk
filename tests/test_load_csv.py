"""
Test program for load_csv.py

Tests the read_csv_parameters() function with various CSV files.

RUN:
  cd tests && python3 test_load_csv.py
"""

import sys
sys.path.insert(0, '../preprocessing')

from load_csv import read_csv_parameters


def run_test(mu_file, sigma_file, N, should_fail, error_type, test_name):
    """Run a single test case."""
    print(f"\nTest: {test_name}")
    print(f"  Files: {mu_file}, {sigma_file}")

    try:
        mu, Sigma = read_csv_parameters(mu_file, sigma_file, N)

        if should_fail:
            print(f"  Result: ✗ FAIL (expected {error_type.__name__})")
            return False
        else:
            print(f"  Result: ✓ PASS")
            print(f"  Data loaded:")
            print(f"    mu.shape = {mu.shape}")
            print(f"    Sigma.shape = {Sigma.shape}")
            return True

    except Exception as e:
        if should_fail and isinstance(e, error_type):
            print(f"  Result: ✓ PASS ({error_type.__name__} caught)")
            return True
        else:
            print(f"  Result: ✗ FAIL ({type(e).__name__}: {e})")
            return False


def main():
    print("=" * 40)
    print("CSV Loader Test Suite (Python)")
    print("=" * 40)

    N = 9
    passed = 0
    total = 0

    # Test 1: Valid files
    total += 1
    if run_test("mu.csv", "Sigma.csv", N, False, None,
                "Test 1: Load valid CSV files"):
        passed += 1

    # Test 2: Missing mu file
    total += 1
    if run_test("nonexistent_mu.csv", "Sigma.csv", N, True, FileNotFoundError,
                "Test 2: Missing mu file"):
        passed += 1

    # Test 3: Missing Sigma file
    total += 1
    if run_test("mu.csv", "nonexistent_Sigma.csv", N, True, FileNotFoundError,
                "Test 3: Missing Sigma file"):
        passed += 1

    # Test 4: Wrong dimension for mu
    total += 1
    if run_test("mu_wrong_dim.csv", "Sigma.csv", N, True, ValueError,
                "Test 4: Dimension mismatch in mu"):
        passed += 1

    # Test 5: Wrong dimension for Sigma
    total += 1
    if run_test("mu.csv", "Sigma_wrong_dim.csv", N, True, ValueError,
                "Test 5: Dimension mismatch in Sigma"):
        passed += 1

    # Print summary
    print("\n" + "=" * 40)
    print("Test Summary")
    print("=" * 40)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("Result: ✓ ALL TESTS PASSED")
        return 0
    else:
        print("Result: ✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
