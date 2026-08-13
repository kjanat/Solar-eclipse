import shutil
import zipfile
from pathlib import Path


def extract_raw_archives(archives_dir: Path, raw_dir: Path) -> list[Path]:
    archives = sorted(archives_dir.glob("*.zip"))
    if not archives:
        raise FileNotFoundError(f"No ZIP archives found in {archives_dir}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    for archive_path in archives:
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                name = Path(member.filename).name
                if member.is_dir() or Path(name).suffix.casefold() != ".cr3":
                    continue
                destination = raw_dir / name
                if destination.exists():
                    if destination.stat().st_size != member.file_size:
                        raise FileExistsError(
                            f"Refusing to overwrite different file: {destination}"
                        )
                    extracted.append(destination)
                    continue
                temporary = destination.with_suffix(f"{destination.suffix}.part")
                try:
                    with archive.open(member) as source, temporary.open("xb") as target:
                        shutil.copyfileobj(source, target)
                    temporary.replace(destination)
                except BaseException:
                    temporary.unlink(missing_ok=True)
                    raise
                extracted.append(destination)
    if not extracted:
        raise RuntimeError(f"No CR3 files found in archives under {archives_dir}")
    return sorted(set(extracted))
