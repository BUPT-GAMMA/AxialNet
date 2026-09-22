#!/usr/bin/env python3
"""Run the core tests without requiring pytest in a reproduction environment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests import test_smoke  # noqa: E402


def main():
    tests = [
        test_smoke.test_cpu_forward_shape,
        test_smoke.test_historical_mode_ignores_graph_updates,
        test_smoke.test_corrected_mode_uses_graph_updates,
        test_smoke.test_state_dict_strict_round_trip,
        test_smoke.test_canonical_split_counts_and_disjointness,
    ]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")


if __name__ == "__main__":
    main()
