# Solar Eclipse

Photographs and a portable RAW analysis pipeline for the 12 August 2026 solar eclipse in Amsterdam.

## Run The Pipeline

Requirements:

- Python 3.14
- [uv](https://docs.astral.sh/uv/)

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

Run tests:

```bash
uv run pytest
```

All paths default to locations inside the current project root. Use `--root` to run against another checkout, or override individual locations with `--archives`, `--raw`, `--data`, `--crops`, `--renders`, and `--final`. Relative overrides resolve against `--root`; absolute paths also work.

```bash
uv run solar-eclipse-analysis \
  --root /path/to/project \
  --raw /path/to/cr3-files \
  inventory
```

## Layout

- `final/` - finished 4K PNG and 1920px JPEG compositions
- `analysis/data/` - inventory, metrics, edge measurements, and alignment data
- `analysis/renders/` - generated diagnostic and processed images
- `src/solar_eclipse_analysis/` - maintained Python package and CLI pipeline
- `analysis/scripts/` - compatibility entrypoints for the original script names
- `docs/` - analysis conversation and processing notes
- `source/raw-archives/` - original Canon CR3 photographs, preserved as ZIP archives
- `archive/bundles/` - original downloadable analysis bundle and preserved extracted artifacts
- `archive/duplicates/` - intentionally preserved duplicate exports

## Notes

The two RAW archives are complementary: `8487-8489` and `8490-8514`. The `extract` stage combines them under `source/raw/` without overwriting a conflicting file.

Archived duplicates are retained intentionally. Canonical working copies live in `final/` and `analysis/`.
