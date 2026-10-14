from pathlib import Path

import pytest

from synthscript.shared.path_bundle import PathBundle


@pytest.fixture
def paths(tmp_path: Path) -> PathBundle:
    return PathBundle(tmp_path)
