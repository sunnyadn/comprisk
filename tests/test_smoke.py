"""Smoke tests — verify the package is importable and reports a version."""

import comprisk


def test_package_imports():
    assert comprisk is not None


def test_version_matches_expected():
    # Compare against installed metadata (pyproject) so __init__ and
    # pyproject cannot drift apart again (they did for 0.7.1).
    import importlib.metadata

    assert comprisk.__version__ == importlib.metadata.version("comprisk")
