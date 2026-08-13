import csv
import gc
import math
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from solar_eclipse_analysis.analyze import crop_around
from solar_eclipse_analysis.libraw import LibRaw, Rgb16

matplotlib.use("Agg")
from matplotlib import colormaps


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
) -> np.ndarray:
    image = rgb.astype(np.float32) / 65535.0
    luminance = (
        0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]
    )
    white = max(float(np.percentile(luminance, white_percentile)), 1e-6)
    mapped = np.clip(
        np.log1p(strength * np.maximum(luminance / white - black, 0))
        / np.log1p(strength),
        0,
        1,
    )
    output = np.clip(image * (mapped / (luminance + 1e-6))[:, :, None], 0, 1)
    if saturation != 1:
        output_luminance = (
            0.2126 * output[:, :, 0]
            + 0.7152 * output[:, :, 1]
            + 0.0722 * output[:, :, 2]
        )
        output = np.clip(
            output_luminance[:, :, None]
            + (output - output_luminance[:, :, None]) * saturation,
            0,
            1,
        )
    return np.clip(np.power(output, 1 / gamma) * 255 + 0.5, 0, 255).astype(np.uint8)


def unsharp(image: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return np.clip(
        image.astype(np.float32) * (1 + amount) - blurred.astype(np.float32) * amount,
        0,
        255,
    ).astype(np.uint8)


def render_long_exposure(
    raw_dir: Path, renders_dir: Path, decoder: LibRaw
) -> list[Path]:
    rgb = decoder.decode(require_raw(raw_dir, "_J3A8490.CR3"), highlight=5)
    rgb = np.rot90(rgb, k=3)
    scale = 2200 / rgb.shape[0]
    rgb = np.asarray(
        cv2.resize(
            rgb, (round(rgb.shape[1] * scale), 2200), interpolation=cv2.INTER_AREA
        ),
        dtype=np.uint16,
    )
    gold = tonemap(rgb, 99.75, 2.25, 12, 0.002, 1.15)
    luminance = cv2.cvtColor(gold, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    mask = np.clip((luminance - 0.015) / 0.12, 0, 1)[:, :, None]
    gold = np.clip(gold.astype(np.float32) * (0.82 + 0.18 * mask), 0, 255).astype(
        np.uint8
    )
    gold_path = renders_dir / "8490_long_exposure_gold.png"
    Image.fromarray(gold).save(gold_path, compress_level=3)
    linear = rgb.astype(np.float32) / 65535
    luminance = (
        0.2126 * linear[:, :, 0] + 0.7152 * linear[:, :, 1] + 0.0722 * linear[:, :, 2]
    )
    white = max(float(np.percentile(luminance, 99.8)), 1e-6)
    mapped = np.clip(np.log1p(40 * luminance / white) / np.log1p(40), 0, 1)
    false_color = (colormaps["magma"](mapped)[:, :, :3] * 255).astype(np.uint8)
    false_color_path = renders_dir / "8490_linear_falsecolor_magma.png"
    Image.fromarray(false_color).save(false_color_path, compress_level=3)
    return [gold_path, false_color_path]


def render_stack(
    raw_dir: Path,
    renders_dir: Path,
    alignment_path: Path,
    metrics: dict[str, dict[str, str]],
    decoder: LibRaw,
) -> Path:
    names = ["_J3A8507.CR3", "_J3A8508.CR3", "_J3A8509.CR3"]
    crops: list[Rgb16] = []
    for name in names:
        rgb = decoder.decode(require_raw(raw_dir, name), highlight=3)
        crops.append(metric_crop(rgb, metrics, name, 1400))
        del rgb
        gc.collect()
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
            stream.write(f"{name} ECC={correlation} warp={warp.tolist()}\n")
    return output_path


def render_details(
    raw_dir: Path,
    renders_dir: Path,
    metrics: dict[str, dict[str, str]],
    decoder: LibRaw,
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
    gradient = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    scale = max(float(np.percentile(gradient, 99.9)), 1e-6)
    edges = (
        colormaps["inferno"](np.clip(gradient / scale, 0, 1))[:, :, :3] * 255
    ).astype(np.uint8)
    edges_path = renders_dir / "8513_edge_structure_falsecolor.png"
    Image.fromarray(edges).save(edges_path, compress_level=3)
    created.append(edges_path)
    return created


def render_montage(crops_dir: Path, renders_dir: Path) -> Path:
    files = sorted(crops_dir.glob("*.jpg"))
    if len(files) < 2:
        raise RuntimeError(f"At least two analysis crops required in {crops_dir}")
    width, height = 2800, 1050
    canvas = Image.new("RGB", (width, height), (2, 2, 4))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=24)
    small = ImageFont.load_default(size=16)
    for index, path in enumerate(files):
        image = Image.open(path).convert("RGB")
        gray = np.asarray(image.convert("L"))
        y_values, x_values = np.where(gray > 22)
        if len(x_values):
            image = image.crop((
                max(int(x_values.min()) - 25, 0),
                max(int(y_values.min()) - 25, 0),
                min(int(x_values.max()) + 26, image.width),
                min(int(y_values.max()) + 26, image.height),
            ))
        image.thumbnail((155, 155), Image.Resampling.LANCZOS)
        x = 90 + index * (width - 180) / (len(files) - 1)
        y = 500 - 250 * math.sin(index / (len(files) - 1) * math.pi)
        canvas.paste(image, (round(x - image.width / 2), round(y - image.height / 2)))
        draw.text(
            (round(x - 44), round(y + 95)),
            path.stem.replace("_J3A", ""),
            fill=(160, 160, 170),
            font=small,
        )
    draw.text(
        (70, 60), "Eclipse progression - RAW sequence", fill=(235, 235, 240), font=font
    )
    draw.text(
        (70, 94),
        "Centered montage; positions are compositional, not the physical solar trajectory.",
        fill=(145, 145, 155),
        font=small,
    )
    output = renders_dir / "raw_eclipse_progression_montage.png"
    canvas.save(output, compress_level=3)
    return output


def render_all(
    raw_dir: Path,
    metrics_path: Path,
    crops_dir: Path,
    renders_dir: Path,
    alignment_path: Path,
    decoder: LibRaw,
) -> list[Path]:
    renders_dir.mkdir(parents=True, exist_ok=True)
    alignment_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(metrics_path)
    created = render_long_exposure(raw_dir, renders_dir, decoder)
    created.append(render_stack(raw_dir, renders_dir, alignment_path, metrics, decoder))
    created.extend(render_details(raw_dir, renders_dir, metrics, decoder))
    created.append(render_montage(crops_dir, renders_dir))
    return created
