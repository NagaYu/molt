"""Every molt_mlx module carries a ``selftest()``; this runs them all under pytest.

Each self-test uses only the tiny random Qwen2 pair in ``tinymodels`` (no
downloads), except ``tiers`` which skips models that are not present locally.
"""

import importlib

import pytest

MODULES = ["footprint", "watchdog", "rope", "kvcache", "mappers", "fit", "calib_data",
           "generate", "migrate", "metrics", "tiers", "convert", "energy"]


@pytest.mark.parametrize("name", MODULES)
def test_module_selftest(name):
    mod = importlib.import_module(f"molt_mlx.{name}")
    mod.selftest()
