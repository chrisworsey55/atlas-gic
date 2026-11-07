import pytest


def pytest_addoption(parser):
    parser.addoption("--live", action="store_true", default=False, help="run tests marked live")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live"):
        return
    skip = pytest.mark.skip(reason="live test; pass --live to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
