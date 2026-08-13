# Solar Eclipse

Photographs and a portable RAW analysis pipeline for the 12 August 2026 solar eclipse in Amsterdam.

## Run The Pipeline

Requirements:

- Python 3.14
- [uv](https://docs.astral.sh/uv/)
- [ExifTool](https://exiftool.org/) for metadata propagation

Install the locked Python environment:

```bash
uv sync
```

Run the complete pipeline from the repository root:

```bash
uv run solar-eclipse-analysis all
```

Run individual stages:

```bash
uv run solar-eclipse-analysis extract
uv run solar-eclipse-analysis inventory
uv run solar-eclipse-analysis analyze
uv run solar-eclipse-analysis edge-spread
uv run solar-eclipse-analysis render
```

`render` writes diagnostic images to `analysis/renders/` and the corrected 4K/1920 progression to `final/`. It does not regenerate the early boxed diagnostic montage retained in the archive.

Rendered PNG and JPEG files preserve every populated, portable photographic EXIF field available in the CR3 sources when its value is shared by all contributing frames. This includes authorship, copyright, camera/lens, capture time, exposure, metering, flash, white balance, sensor resolution, and XMP rating data. All distinct source XMP subjects are retained in source order. Output geometry, orientation, color space, rendering status, description, software, and XMP provenance describe the generated file itself. Mixed composite settings are omitted rather than misrepresented.

GPS and location metadata, serial numbers, unique identifiers, firmware, device telemetry, embedded previews, filesystem data, CR3 container/track fields, and Canon MakerNotes are never copied. MakerNotes cannot be safely transplanted individually and their original block also contains excluded identifiers and stale RAW geometry. The sources contain no IPTC metadata to preserve.

Run tests:

```bash
uv run pytest
```

All paths default to locations inside the current project root. Use `--root` to run against another checkout, or override individual locations with `--archives`, `--raw`, `--crops`, `--renders`, and `--final`. Relative overrides resolve against `--root`; absolute paths also work.

```bash
uv run solar-eclipse-analysis \
  --root /path/to/project \
  --raw /path/to/cr3-files \
  inventory
```

## Layout

- `final/` - finished 4K PNG and 1920px JPEG compositions
- `analysis/renders/` - generated diagnostic and processed images
- `src/solar_eclipse_analysis/` - maintained Python package and CLI pipeline
- `analysis/scripts/` - compatibility entrypoints for the original script names
- `docs/` - local analysis conversation and processing notes (not tracked)
- `source/` - local Canon CR3 sources and archives (not tracked)
- `archive/` - local preserved bundles and duplicate exports (not tracked)

## Notes

The two local RAW archives are complementary: `8487-8489` and `8490-8514`. The `extract` stage combines them under `source/raw/` without overwriting a conflicting file.

Archived duplicates are retained intentionally. Canonical working copies live in `final/` and `analysis/`.

Inventory, metrics, edge measurements, and alignment transforms pass directly between pipeline stages in memory. Historical files under `analysis/data/` may remain locally but are not read, regenerated, or tracked.

![Final 4K solar eclipse progression](final/eclipse_progression_amsterdam_urth_final_4k.png)
