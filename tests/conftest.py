"""Root pytest configuration.

Automatically skips tests marked with @pytest.mark.integration unless the
-m integration flag is explicitly passed on the command line.
"""
import pytest


def pytest_collection_modifyitems(config, items):
    if config.getoption("-m", default="") == "integration":
        return
    skip_integration = pytest.mark.skip(reason="Integration test: run with -m integration")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
