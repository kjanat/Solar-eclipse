import csv
from pathlib import Path

from solar_eclipse_analysis.libraw import LibRaw
from solar_eclipse_analysis.paths import raw_files


def create_inventory(
    raw_dir: Path, output: Path, decoder: LibRaw
) -> list[dict[str, str | int | float]]:
    files = raw_files(raw_dir)
    if not files:
        raise FileNotFoundError(f"No CR3 files found in {raw_dir}; run extract first")
    rows = [decoder.inspect(path) for path in files]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def print_inventory(rows: list[dict[str, str | int | float]]) -> None:
    for row in rows:
        shutter = float(row["shutter_s"])
        shutter_label = (
            f"1/{round(1 / shutter)}" if 0 < shutter < 1 else f"{shutter:.4g}s"
        )
        print(
            f"{row['file']:12} {float(row['bytes']) / 1024 / 1024:6.2f} MiB  "
            f"ISO {float(row['iso']):5.0f}  {shutter_label:8}  "
            f"f/{float(row['aperture']):g}  {float(row['focal_mm']):g}mm  {row['timestamp']}"
        )
