from dataclasses import dataclass
from pathlib import Path


def resolve_path(root: Path, value: Path | None, default: str) -> Path:
    path = value if value is not None else Path(default)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    archives: Path
    raw: Path
    data: Path
    crops: Path
    renders: Path

    @classmethod
    def create(
        cls,
        root: Path,
        archives: Path | None = None,
        raw: Path | None = None,
        data: Path | None = None,
        crops: Path | None = None,
        renders: Path | None = None,
    ) -> ProjectPaths:
        project_root = root.resolve()
        return cls(
            root=project_root,
            archives=resolve_path(project_root, archives, "source/raw-archives"),
            raw=resolve_path(project_root, raw, "source/raw"),
            data=resolve_path(project_root, data, "analysis/data"),
            crops=resolve_path(project_root, crops, "analysis/crops"),
            renders=resolve_path(project_root, renders, "analysis/renders"),
        )

    def ensure_outputs(self) -> None:
        for directory in (self.raw, self.data, self.crops, self.renders):
            directory.mkdir(parents=True, exist_ok=True)


def raw_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.casefold() == ".cr3"
    )
