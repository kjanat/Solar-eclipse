import shutil
import zipfile
from pathlib import Path

MAX_RAW_BYTES = 256 * 1024 * 1024
MAX_TOTAL_BYTES = 16 * 1024 * 1024 * 1024


def same_archive_member(
    archive: zipfile.ZipFile, member: zipfile.ZipInfo, destination: Path
) -> bool:
    if destination.stat().st_size != member.file_size:
        return False
    with archive.open(member) as archived, destination.open("rb") as existing:
        while archived_chunk := archived.read(1024 * 1024):
            if archived_chunk != existing.read(len(archived_chunk)):
                return False
        return not existing.read(1)


def extract_raw_archives(archives_dir: Path, raw_dir: Path) -> list[Path]:
    archives = sorted(archives_dir.glob("*.zip"))
    if not archives:
        raise FileNotFoundError(f"No ZIP archives found in {archives_dir}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    total_bytes = 0
    for archive_path in archives:
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                name = Path(member.filename).name
                if member.is_dir() or Path(name).suffix.casefold() != ".cr3":
                    continue
                if member.file_size > MAX_RAW_BYTES:
                    raise RuntimeError(
                        f"Archive member exceeds {MAX_RAW_BYTES} bytes: {member.filename}"
                    )
                total_bytes += member.file_size
                if total_bytes > MAX_TOTAL_BYTES:
                    raise RuntimeError(
                        f"RAW archives exceed {MAX_TOTAL_BYTES} uncompressed bytes"
                    )
                destination = raw_dir / name
                if destination.exists():
                    if not same_archive_member(archive, member, destination):
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
