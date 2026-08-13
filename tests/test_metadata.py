import subprocess
from pathlib import Path

import pytest
from PIL import ExifTags, Image
from PIL.TiffImagePlugin import IFDRational

from solar_eclipse_analysis.metadata import (
    COMMON_SOURCE_TAGS,
    ImageMetadata,
    common_source_tags,
    save_jpeg,
    save_png,
    source_subjects,
)

RAW_DIR = Path(__file__).parents[1] / "source" / "raw"


def require_raw(name: str) -> Path:
    path = RAW_DIR / name
    if not path.is_file():
        pytest.skip("local CR3 test fixtures unavailable")
    return path


def rational_value(value: object) -> float:
    if not isinstance(value, int | float | IFDRational):
        raise TypeError(f"Expected numeric EXIF value, got {type(value).__name__}")
    return float(value)


@pytest.mark.parametrize("suffix", ["png", "jpg"])
def test_single_source_preserves_safe_standard_metadata(
    tmp_path: Path, suffix: str
) -> None:
    output = tmp_path / f"render.{suffix}"
    image = Image.new("RGB", (20, 10))
    metadata = ImageMetadata("Eclipse render", (require_raw("_J3A8490.CR3"),))
    if suffix == "png":
        save_png(image, output, metadata, compress_level=3)
    else:
        save_jpeg(image, output, metadata, quality=90)

    with Image.open(output) as rendered:
        exif = rendered.getexif()
        photographic = exif.get_ifd(ExifTags.IFD.Exif)
        assert exif[ExifTags.Base.ImageDescription] == "Eclipse render"
        assert exif[ExifTags.Base.Make] == "Canon"
        assert exif[ExifTags.Base.Model] == "Canon EOS R7"
        assert exif[ExifTags.Base.Artist] == "Kaj Kowalski"
        assert exif[ExifTags.Base.Copyright] == "Kaj Kowalski, All Rights Reserved"
        assert photographic[ExifTags.Base.DateTimeOriginal] == "2026:08:12 20:04:41"
        assert photographic[ExifTags.Base.OffsetTimeOriginal] == "+02:00"
        assert rational_value(photographic[ExifTags.Base.ExposureTime]) == 0.4
        assert rational_value(photographic[ExifTags.Base.FNumber]) == 40.0
        assert photographic[ExifTags.Base.ISOSpeedRatings] == 160
        assert photographic[ExifTags.Base.LensModel] == "RF-S18-150mm F3.5-6.3 IS STM"
        assert photographic[ExifTags.Base.ExifImageWidth] == 20
        assert photographic[ExifTags.Base.ExifImageHeight] == 10
        assert exif[ExifTags.Base.Orientation] == 1
        assert photographic[ExifTags.Base.ColorSpace] == 1
        assert ExifTags.Base.GPSInfo not in exif
        assert 0xA431 not in photographic
        assert ExifTags.Base.LensSerialNumber not in photographic

    xmp = subprocess.run(
        [
            "exiftool",
            "-T",
            "-XMP-dc:Creator",
            "-XMP-dc:Rights",
            "-XMP-dc:Source",
            str(output),
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    assert xmp.stdout.rstrip().split("\t") == [
        "Kaj Kowalski",
        "Kaj Kowalski, All Rights Reserved",
        "_J3A8490.CR3",
    ]
    inventory = subprocess.run(
        ["exiftool", "-G1", "-s", "-a", str(output)],
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    assert "[GPS]" not in inventory
    assert "[Canon]" not in inventory
    assert "SerialNumber" not in inventory


def test_composite_preserves_only_shared_settings(tmp_path: Path) -> None:
    output = tmp_path / "progression.jpg"
    sources = (
        require_raw("_J3A8490.CR3"),
        require_raw("_J3A8513.CR3"),
    )
    save_jpeg(
        Image.new("RGB", (20, 10)),
        output,
        ImageMetadata("Eclipse progression", sources),
        quality=90,
    )

    with Image.open(output) as rendered:
        photographic = rendered.getexif().get_ifd(ExifTags.IFD.Exif)
        assert ExifTags.Base.ExposureTime not in photographic
        assert ExifTags.Base.FNumber not in photographic
        assert photographic[ExifTags.Base.ISOSpeedRatings] == 160
        assert rational_value(photographic[ExifTags.Base.FocalLength]) == 150.0
        assert photographic[ExifTags.Base.DateTimeOriginal] == "2026:08:12 20:04:41"


def test_common_source_tags_keeps_fields_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_index = COMMON_SOURCE_TAGS.index("ExifIFD:ISO")
    mixed_index = COMMON_SOURCE_TAGS.index("ExifIFD:ExposureTime")
    first = ["-" for _ in COMMON_SOURCE_TAGS]
    second = list(first)
    first[shared_index] = second[shared_index] = "160"
    first[mixed_index] = "0.4"
    second[mixed_index] = "0.0166667"
    completed = subprocess.CompletedProcess(
        args=["exiftool"],
        returncode=0,
        stdout=f"{'\t'.join(first)}\n{'\t'.join(second)}\n",
        stderr="",
    )
    monkeypatch.setattr(
        "solar_eclipse_analysis.metadata.run_exiftool", lambda _arguments: completed
    )

    tags = common_source_tags((Path("one.CR3"), Path("two.CR3")))

    assert "ExifIFD:ISO" in tags
    assert "ExifIFD:ExposureTime" not in tags


def test_source_subjects_unions_list_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = subprocess.CompletedProcess(
        args=["exiftool"],
        returncode=0,
        stdout="correct exposure|night\nwrong exposure|sharp|night\n",
        stderr="",
    )

    def fake_exiftool(_arguments: list[str]) -> subprocess.CompletedProcess[str]:
        return completed

    monkeypatch.setattr("solar_eclipse_analysis.metadata.run_exiftool", fake_exiftool)

    subjects = source_subjects((Path("one.CR3"), Path("two.CR3")))

    assert subjects == ("correct exposure", "night", "wrong exposure", "sharp")
