import tempfile
from pathlib import Path

from solar_eclipse_analysis.paths import ProjectPaths


def test_defaults_resolve_from_project_root() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        paths = ProjectPaths.create(root)

        assert paths.raw == root / "source/raw"
        assert paths.crops == root / "analysis/crops"


def test_absolute_override_is_preserved() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        external = root.parent / "external-raw"

        paths = ProjectPaths.create(root, raw=external)

        assert paths.raw == external.resolve()
