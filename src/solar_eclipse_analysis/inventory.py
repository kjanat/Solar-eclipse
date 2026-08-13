from pathlib import Path

from solar_eclipse_analysis.models import DataRow
from solar_eclipse_analysis.paths import raw_files
from solar_eclipse_analysis.raw import RawDecoder


def create_inventory(raw_dir: Path, decoder: RawDecoder) -> list[DataRow]:
    files = raw_files(raw_dir)
    if not files:
        raise FileNotFoundError(f"No CR3 files found in {raw_dir}; run extract first")
    return [decoder.inspect(path) for path in files]


def print_inventory(rows: list[DataRow]) -> None:
    for row in rows:
        shutter = float(row["shutter_s"])
        shutter_label = (
            f"1/{round(1 / shutter)}" if 0 < shutter < 1 else f"{shutter:.4g}s"
        )
        summary = (
            f"{row['file']:12} {float(row['bytes']) / 1024 / 1024:6.2f} MiB  "
            f"ISO {float(row['iso']):5.0f}  {shutter_label:8}  "
            f"f/{float(row['aperture']):g}  "
            f"{float(row['focal_mm']):g}mm  {row['timestamp']}"
        )
        print(summary)
