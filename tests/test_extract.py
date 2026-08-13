import tempfile
import zipfile
from pathlib import Path

import pytest

from solar_eclipse_analysis.extract import extract_raw_archives


def test_extracts_cr3_files_only() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        archives = root / "archives"
        raw = root / "raw"
        archives.mkdir()
        with zipfile.ZipFile(archives / "photos.zip", "w") as archive:
            archive.writestr("nested/photo.CR3", b"raw-data")
            archive.writestr("notes.txt", b"ignored")

        extracted = extract_raw_archives(archives, raw)

        assert extracted == [raw / "photo.CR3"]
        assert (raw / "photo.CR3").read_bytes() == b"raw-data"
        assert not (raw / "notes.txt").exists()


def test_refuses_to_overwrite_different_file() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        archives = root / "archives"
        raw = root / "raw"
        archives.mkdir()
        raw.mkdir()
        (raw / "photo.CR3").write_bytes(b"existing")
        with zipfile.ZipFile(archives / "photos.zip", "w") as archive:
            archive.writestr("photo.CR3", b"different-size")

        with pytest.raises(FileExistsError):
            extract_raw_archives(archives, raw)
