import csv
import gc
import math
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from solar_eclipse_analysis.analyze import crop_around
from solar_eclipse_analysis.raw import RawDecoder, Rgb16

matplotlib.use("Agg")
from matplotlib import colormaps

FloatVector = np.ndarray[tuple[int], np.dtype[np.float32]]
FloatImage = np.ndarray[tuple[int, int], np.dtype[np.float32]]
FloatRgb = np.ndarray[tuple[int, int, int], np.dtype[np.float32]]
Rgb8 = np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]
Rgba64 = np.ndarray[tuple[int, int, int], np.dtype[np.float64]]


def load_metrics(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Metrics not found: {path}; run analyze first")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return {row["file"]: row for row in rows}


def require_raw(raw_dir: Path, name: str) -> Path:
    path = raw_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"Required RAW not found: {path}")
    return path


def metric_crop(
    rgb: Rgb16, metrics: dict[str, dict[str, str]], name: str, size: int
) -> Rgb16:
    if name not in metrics:
        raise RuntimeError(f"Metrics have no row for {name}")
    row = metrics[name]
    crop, _, _ = crop_around(
        rgb, int(float(row["sun_x"])), int(float(row["sun_y"])), size
    )
    return crop


def tonemap(
    rgb: Rgb16,
    white_percentile: float = 99.8,
    gamma: float = 2.2,
    strength: float = 6.0,
    black: float = 0.0,
    saturation: float = 1.0,
) -> Rgb8:
    image: FloatRgb = np.asarray(rgb.astype(np.float32) / 65535.0, dtype=np.float32)
    luminance: FloatImage = np.asarray(
        0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2],
        dtype=np.float32,
    )
    white = max(np.percentile(luminance, white_percentile).item(), 1e-6)
    logarithm_input: FloatImage = np.asarray(
        strength * np.maximum(luminance / white - black, 0) + 1,
        dtype=np.float32,
    )
    mapped: FloatImage = np.asarray(
        np.clip(
            cv2.log(logarithm_input) / math.log1p(strength),
            0,
            1,
        ),
        dtype=np.float32,
    )
    output: FloatImage = np.asarray(
        np.clip(image * (mapped / (luminance + 1e-6))[:, :, None], 0, 1),
        dtype=np.float32,
    )
    if saturation != 1:
        output_luminance = (
            0.2126 * output[:, :, 0]
            + 0.7152 * output[:, :, 1]
            + 0.0722 * output[:, :, 2]
        )
        output = np.asarray(
            np.clip(
                output_luminance[:, :, None]
                + (output - output_luminance[:, :, None]) * saturation,
                0,
                1,
            ),
            dtype=np.float32,
        )
    gamma_output: FloatRgb = np.asarray(output ** (1 / gamma), dtype=np.float32)
    return np.asarray(np.clip(gamma_output * 255 + 0.5, 0, 255), dtype=np.uint8)


def unsharp(image: Rgb8, sigma: float, amount: float) -> Rgb8:
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return np.asarray(
        np.clip(
            image.astype(np.float32) * (1 + amount)
            - blurred.astype(np.float32) * amount,
            0,
            255,
        ),
        dtype=np.uint8,
    )


def render_long_exposure(
    raw_dir: Path, renders_dir: Path, decoder: RawDecoder
) -> list[Path]:
    rgb = decoder.decode(require_raw(raw_dir, "_J3A8490.CR3"), highlight=5)
    rgb = np.rot90(rgb, k=3)
    _, width, _ = rgb.shape
    scale = 2200 / rgb.shape[0]
    rgb = np.asarray(
        cv2.resize(rgb, (round(width * scale), 2200), interpolation=cv2.INTER_AREA),
        dtype=np.uint16,
    )
    gold = tonemap(rgb, 99.75, 2.25, 12, 0.002, 1.15)
    display_luminance: FloatImage = np.asarray(
        cv2.cvtColor(gold, cv2.COLOR_RGB2GRAY).astype(np.float32) / np.float32(255),
        dtype=np.float32,
    )
    mask = np.clip((display_luminance - 0.015) / 0.12, 0, 1)[:, :, None]
    gold = np.clip(gold.astype(np.float32) * (0.82 + 0.18 * mask), 0, 255).astype(
        np.uint8
    )
    gold_path = renders_dir / "8490_long_exposure_gold.png"
    Image.fromarray(gold).save(gold_path, compress_level=3)
    linear: FloatRgb = np.asarray(rgb.astype(np.float32) / 65535, dtype=np.float32)
    linear_luminance: FloatImage = np.asarray(
        0.2126 * linear[:, :, 0] + 0.7152 * linear[:, :, 1] + 0.0722 * linear[:, :, 2],
        dtype=np.float32,
    )
    white = max(np.percentile(linear_luminance, 99.8).item(), 1e-6)
    logarithm_input: FloatImage = np.asarray(
        40 * linear_luminance / white + 1, dtype=np.float32
    )
    mapped: FloatImage = np.asarray(
        np.clip(cv2.log(logarithm_input) / math.log1p(40), 0, 1),
        dtype=np.float32,
    )
    colored: Rgba64 = np.asarray(colormaps["magma"](mapped), dtype=np.float64)
    false_color: Rgb8 = np.asarray(colored[:, :, :3] * 255, dtype=np.uint8)
    false_color_path = renders_dir / "8490_linear_falsecolor_magma.png"
    Image.fromarray(false_color).save(false_color_path, compress_level=3)
    return [gold_path, false_color_path]


def render_stack(
    raw_dir: Path,
    renders_dir: Path,
    alignment_path: Path,
    metrics: dict[str, dict[str, str]],
    decoder: RawDecoder,
) -> Path:
    names = ["_J3A8507.CR3", "_J3A8508.CR3", "_J3A8509.CR3"]
    crops: list[Rgb16] = []
    for name in names:
        rgb = decoder.decode(require_raw(raw_dir, name), highlight=3)
        crops.append(metric_crop(rgb, metrics, name, 1400))
        del rgb
        _ = gc.collect()
    reference = crops[-1].astype(np.float32) / 65535
    reference_luminance = np.log1p(
        50
        * (
            0.2126 * reference[:, :, 0]
            + 0.7152 * reference[:, :, 1]
            + 0.0722 * reference[:, :, 2]
        )
    ).astype(np.float32)
    aligned: list[np.ndarray] = []
    transforms: list[tuple[float, np.ndarray]] = []
    for crop in crops:
        image = crop.astype(np.float32) / 65535
        luminance = np.log1p(
            50
            * (
                0.2126 * image[:, :, 0]
                + 0.7152 * image[:, :, 1]
                + 0.0722 * image[:, :, 2]
            )
        ).astype(np.float32)
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            correlation, warp = cv2.findTransformECC(
                reference_luminance,
                luminance,
                warp,
                cv2.MOTION_TRANSLATION,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-7),
                np.ones(reference_luminance.shape, dtype=np.uint8),
                3,
            )
        except cv2.error:
            correlation = math.nan
        transforms.append((float(correlation), warp.copy()))
        aligned.append(
            cv2.warpAffine(
                crop,
                warp,
                (crop.shape[1], crop.shape[0]),
                flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT,
            ).astype(np.float32)
        )
    stack = np.median(np.stack(aligned), axis=0).astype(np.uint16)
    output = unsharp(tonemap(stack, 99.7, 2.2, 4, 0.0005, 0.9), 1.0, 0.45)
    output_path = renders_dir / "8507-8509_aligned_median_stack.png"
    Image.fromarray(output).save(output_path, compress_level=3)
    with alignment_path.open("w", encoding="utf-8") as stream:
        for name, (correlation, warp) in zip(names, transforms, strict=True):
            _ = stream.write(f"{name} ECC={correlation} warp={warp.tolist()}\n")
    return output_path


def render_details(
    raw_dir: Path,
    renders_dir: Path,
    metrics: dict[str, dict[str, str]],
    decoder: RawDecoder,
) -> list[Path]:
    created: list[Path] = []
    name = "_J3A8514.CR3"
    crop = metric_crop(decoder.decode(require_raw(raw_dir, name)), metrics, name, 1200)
    clean = unsharp(tonemap(crop, 99.85, 2.15, 3.2, 0.0002, 0.75), 0.9, 0.35)
    clean_path = renders_dir / "8514_clean_unclipped_crescent.png"
    Image.fromarray(clean).save(clean_path, compress_level=3)
    created.append(clean_path)
    monochrome = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY)
    monochrome = np.clip((monochrome.astype(np.float32) - 2) * 1.08, 0, 255).astype(
        np.uint8
    )
    monochrome_path = renders_dir / "8514_clean_monochrome.png"
    Image.fromarray(monochrome, "L").save(monochrome_path, compress_level=3)
    created.append(monochrome_path)

    name = "_J3A8513.CR3"
    crop = metric_crop(
        decoder.decode(require_raw(raw_dir, name), highlight=3), metrics, name, 1200
    )
    detail = unsharp(tonemap(crop, 99.65, 2.2, 3.5, 0.0002, 0.85), 0.8, 0.4)
    detail_path = renders_dir / "8513_sharp_f16_detail.png"
    Image.fromarray(detail).save(detail_path, compress_level=3)
    created.append(detail_path)
    gray = cv2.cvtColor(detail, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    gradient_x: FloatImage = np.asarray(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), dtype=np.float32
    )
    gradient_y: FloatImage = np.asarray(
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3), dtype=np.float32
    )
    gradient: FloatImage = np.asarray(
        cv2.magnitude(gradient_x, gradient_y), dtype=np.float32
    )
    scale = max(np.percentile(gradient, 99.9).item(), 1e-6)
    normalized_gradient: FloatImage = np.asarray(
        np.clip(gradient / scale, 0, 1), dtype=np.float32
    )
    colored_edges: Rgba64 = np.asarray(
        colormaps["inferno"](normalized_gradient), dtype=np.float64
    )
    edges: Rgb8 = np.asarray(colored_edges[:, :, :3] * 255, dtype=np.uint8)
    edges_path = renders_dir / "8513_edge_structure_falsecolor.png"
    Image.fromarray(edges).save(edges_path, compress_level=3)
    created.append(edges_path)
    return created


def progression_tile(rgb: Rgb16, metrics: dict[str, str]) -> Image.Image:
    crop, _, _ = crop_around(
        rgb,
        int(float(metrics["sun_x"])),
        int(float(metrics["sun_y"])),
        900,
    )
    tile = unsharp(tonemap(crop, 99.8, 2.2, 5.0, 0, 0.9), 0.8, 0.25)
    tile = cv2.resize(tile, (300, 300), interpolation=cv2.INTER_AREA)
    return Image.fromarray(tile, "RGB")


def feathered_rgba(image: Image.Image) -> Image.Image:
    rgb: FloatRgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    size = rgb.shape[0]
    coordinates: FloatVector = np.arange(size, dtype=np.float32)
    raw_x_grid, raw_y_grid = np.meshgrid(coordinates, coordinates)
    x_grid: FloatImage = np.asarray(raw_x_grid, dtype=np.float32)
    y_grid: FloatImage = np.asarray(raw_y_grid, dtype=np.float32)
    center = (size - 1) / 2
    squared_distance: FloatImage = np.asarray(
        (x_grid - center) ** 2 + (y_grid - center) ** 2, dtype=np.float32
    )
    radius: FloatImage = np.asarray(
        cv2.sqrt(squared_distance) / (size / 2), dtype=np.float32
    )
    radial: FloatImage = np.asarray(
        np.clip((1.02 - radius) / (1.02 - 0.70), 0, 1), dtype=np.float32
    )
    radial = radial * radial * (3 - 2 * radial)
    luminance = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    clipped_content: FloatImage = np.asarray(
        np.clip((luminance - 0.008) / 0.12, 0, 1), dtype=np.float32
    )
    content: FloatImage = np.asarray(cv2.sqrt(clipped_content), dtype=np.float32)
    alpha: FloatImage = np.asarray(radial * np.maximum(content, 0.16), dtype=np.float32)
    rgba = np.dstack([rgb, alpha])
    return Image.fromarray(np.clip(rgba * 255 + 0.5, 0, 255).astype(np.uint8), "RGBA")


def format_shutter(value: str) -> str:
    shutter = float(value)
    return f"{shutter:.1f} s" if shutter >= 0.1 else f"1/{round(1 / shutter)} s"


def dejavu_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / filename
    return ImageFont.truetype(path, size)


def render_progression(
    raw_dir: Path,
    final_dir: Path,
    metrics: dict[str, dict[str, str]],
    decoder: RawDecoder,
) -> list[Path]:
    names = sorted(metrics)
    if len(names) < 2:
        raise RuntimeError("At least two metric rows required for the progression")
    width, height = 3840, 1280
    canvas = Image.new("RGB", (width, height), (2, 2, 3))
    draw = ImageDraw.Draw(canvas)
    title_font = dejavu_font(52, bold=True)
    metadata_font = dejavu_font(25)
    filter_font = dejavu_font(22)
    copyright_font = dejavu_font(24)
    id_font = dejavu_font(20, bold=True)
    settings_font = dejavu_font(16)
    draw.text((86, 70), "PARTIAL SOLAR ECLIPSE", font=title_font, fill=(242, 242, 244))
    metadata_text = (
        "12 AUG 2026  \u00b7  Amsterdam, Netherlands  \u00b7  Canon EOS R7  \u00b7  "
        "RF-S 18\u2013150mm F3.5\u20136.3 IS STM @ 150 mm  \u00b7  ISO 160  \u00b7  "
        "21 CR3 frames"
    )
    draw.text(
        (88, 139),
        metadata_text,
        font=metadata_font,
        fill=(154, 154, 162),
    )
    draw.text(
        (88, 177),
        "Filters: Urth ND1000 \u00d7 ND8  \u00b7  stacked  \u00b7  \u224813 stops combined",
        font=filter_font,
        fill=(126, 126, 134),
    )
    copyright_text = "\u00a9 2026 Kaj Kowalski"
    copyright_box = draw.textbbox((0, 0), copyright_text, font=copyright_font)
    draw.text(
        (width - 86 - (copyright_box[2] - copyright_box[0]), 84),
        copyright_text,
        font=copyright_font,
        fill=(196, 196, 202),
    )
    display_size = 250
    for index, name in enumerate(names):
        row = metrics[name]
        progress = index / (len(names) - 1)
        center_x = 155 + progress * (width - 310)
        center_y = 690 - 300 * math.sin(math.pi * progress)
        rgb = decoder.decode(require_raw(raw_dir, name), highlight=3)
        tile = progression_tile(rgb, row).resize(
            (display_size, display_size), Image.Resampling.LANCZOS
        )
        del rgb
        tile_rgba = feathered_rgba(tile)
        x = round(center_x - display_size / 2)
        y = round(center_y - display_size / 2)
        canvas.paste(tile_rgba, (x, y), tile_rgba)
        short_id = Path(name).stem.replace("_J3A", "")
        settings = (
            f"{format_shutter(row['shutter_s'])}  \u00b7  f/{float(row['aperture']):g}"
        )
        id_box = draw.textbbox((0, 0), short_id, font=id_font)
        settings_box = draw.textbbox((0, 0), settings, font=settings_font)
        label_y = y + display_size + 8
        draw.text(
            (round(center_x - (id_box[2] - id_box[0]) / 2), label_y),
            short_id,
            font=id_font,
            fill=(198, 198, 203),
        )
        draw.text(
            (round(center_x - (settings_box[2] - settings_box[0]) / 2), label_y + 28),
            settings,
            font=settings_font,
            fill=(116, 116, 124),
        )
    final_dir.mkdir(parents=True, exist_ok=True)
    png_path = final_dir / "eclipse_progression_amsterdam_urth_final_4k.png"
    jpg_path = final_dir / "eclipse_progression_amsterdam_urth_final_1920.jpg"
    canvas.save(png_path, compress_level=4)
    preview = canvas.copy()
    preview.thumbnail((1920, 1920), Image.Resampling.LANCZOS)
    preview.save(jpg_path, quality=94, subsampling=0)
    return [png_path, jpg_path]


def render_all(
    raw_dir: Path,
    metrics_path: Path,
    renders_dir: Path,
    final_dir: Path,
    alignment_path: Path,
    decoder: RawDecoder,
) -> list[Path]:
    renders_dir.mkdir(parents=True, exist_ok=True)
    alignment_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(metrics_path)
    created = render_long_exposure(raw_dir, renders_dir, decoder)
    created.append(render_stack(raw_dir, renders_dir, alignment_path, metrics, decoder))
    created.extend(render_details(raw_dir, renders_dir, metrics, decoder))
    created.extend(render_progression(raw_dir, final_dir, metrics, decoder))
    return created
