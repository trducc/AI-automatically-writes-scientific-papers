"""
Smoke test suite for ConvNeur model, autograd backward pass, and CIFAR-10 training batch.
Compatible with pytest and direct python invocation.
"""

import os
import sys

# Ensure template root is in sys.path
convneur_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../templates/convneur"))
if convneur_dir not in sys.path:
    sys.path.insert(0, convneur_dir)

from smoke_test import run_smoke_test


def test_convneur_smoke():
    success = run_smoke_test()
    assert success is True, "Smoke test failed"


if __name__ == "__main__":
    test_convneur_smoke()
    print("PyTest/Unit test wrapper succeeded!")
