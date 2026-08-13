import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

SOFTWARE = "solar-eclipse-analysis"

# Every populated, portable photographic EXIF field present in the source CR3s.
# Output geometry, orientation, color, and rendering fields are generated below.
COMMON_SOURCE_TAGS = (
    "IFD0:Make",
    "IFD0:Model",
    "IFD0:XResolution",
    "IFD0:YResolution",
    "IFD0:ResolutionUnit",
    "IFD0:Artist",
    "IFD0:Copyright",
    "ExifIFD:ExposureTime",
    "ExifIFD:FNumber",
    "ExifIFD:ExposureProgram",
    "ExifIFD:ISO",
    "ExifIFD:SensitivityType",
    "ExifIFD:RecommendedExposureIndex",
    "ExifIFD:ExifVersion",
    "ExifIFD:ComponentsConfiguration",
    "ExifIFD:ApertureValue",
    "ExifIFD:ShutterSpeedValue",
    "ExifIFD:ExposureCompensation",
    "ExifIFD:MeteringMode",
    "ExifIFD:Flash",
    "ExifIFD:FlashpixVersion",
    "ExifIFD:FocalLength",
    "ExifIFD:FocalPlaneXResolution",
    "ExifIFD:FocalPlaneYResolution",
    "ExifIFD:FocalPlaneResolutionUnit",
    "ExifIFD:ExposureMode",
    "ExifIFD:WhiteBalance",
    "ExifIFD:SceneCaptureType",
    "ExifIFD:OwnerName",
    "ExifIFD:LensInfo",
    "ExifIFD:LensModel",
    "XMP-xmp:Rating",
)

FIRST_SOURCE_TAGS = (
    "IFD0:ModifyDate",
    "ExifIFD:DateTimeOriginal",
    "ExifIFD:CreateDate",
    "ExifIFD:OffsetTime",
    "ExifIFD:OffsetTimeOriginal",
    "ExifIFD:OffsetTimeDigitized",
    "ExifIFD:SubSecTime",
    "ExifIFD:SubSecTimeOriginal",
    "ExifIFD:SubSecTimeDigitized",
)


@dataclass(frozen=True)
class ImageMetadata:
    description: str
    sources: tuple[Path, ...]


def exiftool_path() -> str:
    executable = shutil.which("exiftool")
    if executable is None:
        raise RuntimeError("ExifTool is required to write image metadata")
    return executable


def run_exiftool(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [exiftool_path(), *arguments],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("ExifTool failed while writing image metadata")
    return result


def common_source_tags(sources: tuple[Path, ...]) -> tuple[str, ...]:
    result = run_exiftool([
        "-n",
        "-T",
        *(f"-{tag}" for tag in COMMON_SOURCE_TAGS),
        *map(str, sources),
    ])
    rows = [line.split("\t") for line in result.stdout.splitlines()]
    if len(rows) != len(sources) or any(
        len(row) != len(COMMON_SOURCE_TAGS) for row in rows
    ):
        raise RuntimeError("ExifTool returned unexpected metadata inventory output")
    return tuple(
        tag
        for index, tag in enumerate(COMMON_SOURCE_TAGS)
        if rows[0][index] != "-" and all(row[index] == rows[0][index] for row in rows)
    )


def source_subjects(sources: tuple[Path, ...]) -> tuple[str, ...]:
    separator = "|"
    result = run_exiftool([
        "-n",
        "-sep",
        separator,
        "-T",
        "-XMP-dc:Subject",
        *map(str, sources),
    ])
    rows = [
        tuple(subject for subject in line.split(separator) if subject != "-")
        for line in result.stdout.splitlines()
    ]
    if len(rows) != len(sources):
        raise RuntimeError("ExifTool returned unexpected XMP subject inventory output")
    return tuple(dict.fromkeys(subject for row in rows for subject in row))


def apply_metadata(path: Path, metadata: ImageMetadata) -> None:
    if not metadata.sources:
        raise ValueError("Image metadata requires at least one source")
    missing = [source for source in metadata.sources if not source.is_file()]
    if missing:
        raise FileNotFoundError(f"Metadata source not found: {missing[0]}")
    copied_tags = (*FIRST_SOURCE_TAGS, *common_source_tags(metadata.sources))
    subjects = source_subjects(metadata.sources)
    media_type = (
        "image/jpeg" if path.suffix.casefold() in {".jpg", ".jpeg"} else "image/png"
    )
    with Image.open(path) as image:
        width, height = image.size
    arguments = [
        "-overwrite_original",
        "-TagsFromFile",
        str(metadata.sources[0]),
        *(f"-{tag}" for tag in copied_tags),
        "-Orientation#=1",
        "-ColorSpace#=1",
        "-CustomRendered#=1",
        f"-ExifImageWidth={width}",
        f"-ExifImageHeight={height}",
        f"-ImageDescription={metadata.description}",
        f"-Software={SOFTWARE}",
        f"-XMP-dc:Description={metadata.description}",
        "-XMP-dc:Creator<IFD0:Artist",
        "-XMP-dc:Rights<IFD0:Copyright",
        f"-XMP-dc:Source={', '.join(source.name for source in metadata.sources)}",
        f"-XMP-dc:Format={media_type}",
        f"-XMP-xmp:CreatorTool={SOFTWARE}",
        "-XMP-dc:Subject=",
        *(f"-XMP-dc:Subject+={subject}" for subject in subjects),
        str(path),
    ]
    _ = run_exiftool(arguments)


def save_png(
    image: Image.Image,
    path: Path,
    metadata: ImageMetadata,
    compress_level: int,
) -> None:
    image.save(path, compress_level=compress_level)
    apply_metadata(path, metadata)


def save_jpeg(
    image: Image.Image,
    path: Path,
    metadata: ImageMetadata,
    quality: int,
    subsampling: int = 0,
) -> None:
    image.save(path, quality=quality, subsampling=subsampling)
    apply_metadata(path, metadata)
