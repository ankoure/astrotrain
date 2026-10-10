import importlib.util

import astrotrain


def test_preload_does_nothing_without_the_gpu_extra(monkeypatch):
    # Regression: without the extra there's no `nvidia` package, and find_spec
    # raises for a missing parent package rather than returning None, which
    # broke importing astrotrain anywhere without the gpu extra (like CI)
    def no_nvidia(name, *args):
        raise ModuleNotFoundError(f"No module named '{name.split('.')[0]}'")

    monkeypatch.setattr(importlib.util, "find_spec", no_nvidia)

    astrotrain._preload_cusolver()
