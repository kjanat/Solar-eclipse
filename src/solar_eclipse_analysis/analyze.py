import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from solar_eclipse_analysis.json_value import parse_flat_json_object
from solar_eclipse_analysis.paths import raw_files
from solar_eclipse_analysis.raw import RawDecoder, Rgb16

CHECKPOINT_VERSION = 1
FloatImage = np.ndarray[tuple[int, int], np.dtype[np.float32]]
IntImage = np.ndarray[tuple[int, int], np.dtype[np.int32]]
Centroids = np.ndarray[tuple[int, int], np.dtype[np.float64]]


def locate_sun(rgb: Rgb16) -> tuple[int, int]:
    downsampled = rgb[::4, ::4, :]
    maximum: FloatImage = downsampled.max(axis=2).astype(np.float32)
    threshold = max(float(np.percentile(maximum, 99.9)), float(maximum.max()) * 0.55)
    mask = (maximum >= threshold).astype(np.uint8)
    count, raw_labels, raw_stats, raw_centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    labels: IntImage = np.asarray(raw_labels, dtype=np.int32)
    stats: IntImage = np.asarray(raw_stats, dtype=np.int32)
    centroids: Centroids = np.asarray(raw_centroids, dtype=np.float64)
    candidates: list[tuple[float, int]] = []
    for index in range(1, count):
        area = stats.item(index, cv2.CC_STAT_AREA)
        if area < 3:
            continue
        component_values = maximum[labels == index]
        score = (
            area
            * (float(component_values.mean()) / max(float(maximum.max()), 1.0)) ** 2
        )
        candidates.append((score, index))
    if not candidates:
        flat_index = int(np.argmax(maximum))
        row_width = maximum.shape[1]
        y, x = divmod(flat_index, row_width)
        return x * 4, y * 4
    _, best = max(candidates)
    center_x = centroids.item(best, 0)
    center_y = centroids.item(best, 1)
    return round(center_x * 4), round(center_y * 4)


def crop_around(
    rgb: Rgb16, center_x: int, center_y: int, size: int
) -> tuple[Rgb16, int, int]:
    height, width, _ = rgb.shape
    crop_size = min(size, height, width)
    half = crop_size // 2
    x0 = max(0, min(width - crop_size, center_x - half))
    y0 = max(0, min(height - crop_size, center_y - half))
    return rgb[y0 : y0 + crop_size, x0 : x0 + crop_size].copy(), x0, y0


def measure_crop(crop: Rgb16) -> dict[str, float]:
    normalized = crop.astype(np.float32) / 65535.0
    luminance = (
        0.2126 * normalized[:, :, 0]
        + 0.7152 * normalized[:, :, 1]
        + 0.0722 * normalized[:, :, 2]
    )
    any_clipped = (crop >= 65530).any(axis=2)
    all_clipped = (crop >= 65530).all(axis=2)
    percentile = max(np.percentile(luminance, 99.5).item(), 1e-6)
    scaled = np.clip(luminance / percentile, 0, 1)
    smooth = cv2.GaussianBlur(scaled, (0, 0), 0.8)
    gradient_x = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gradient_y = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    gradient = np.hypot(gradient_x, gradient_y)
    values: np.ndarray[tuple[int], np.dtype[np.float32]] = np.asarray(
        gradient[(smooth > 0.06) & (smooth < 0.94)], dtype=np.float32
    )
    if values.size:
        edge_p99 = float(np.quantile(values, 0.99))
        edge_p995 = float(np.quantile(values, 0.995))
        edge_topmean = values[values >= np.percentile(values, 99)].mean().item()
    else:
        edge_p99 = edge_p995 = edge_topmean = 0.0
    bright = (smooth > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(
        cv2.arcLength(contour, True)
        for contour in contours
        if cv2.contourArea(contour) > 10
    )
    transition = int(((smooth > 0.1) & (smooth < 0.9)).sum())
    width = float(transition / perimeter) if perimeter > 0 else math.nan
    return {
        "clip_any_pct": (any_clipped.mean() * 100).item(),
        "clip_all_pct": (all_clipped.mean() * 100).item(),
        "lum_p99": np.percentile(luminance, 99).item(),
        "lum_p995": percentile,
        "lum_max": luminance.max().item(),
        "edge_p99": edge_p99,
        "edge_p995": edge_p995,
        "edge_topmean": edge_topmean,
        "edge_width_proxy_px": width,
    }


def save_diagnostic(crop: Rgb16, output: Path) -> None:
    normalized = crop.astype(np.float32) / 65535.0
    scale = max(np.percentile(normalized, 99.8).item(), 1e-6)
    image = np.clip(
        np.power(np.clip(normalized / scale, 0, 1), 1 / 2.2) * 255 + 0.5, 0, 255
    )
    Image.fromarray(image.astype(np.uint8), "RGB").save(output, quality=95)


def analyze_raws(
    raw_dir: Path,
    inventory_path: Path,
    metrics_path: Path,
    checkpoint_path: Path,
    crops_dir: Path,
    decoder: RawDecoder,
    crop_size: int = 1400,
) -> list[dict[str, str | int | float]]:
    files = raw_files(raw_dir)
    if not files:
        raise FileNotFoundError(f"No CR3 files found in {raw_dir}; run extract first")
    if not inventory_path.is_file():
        raise FileNotFoundError(
            f"Inventory not found: {inventory_path}; run inventory first"
        )
    with inventory_path.open(newline="", encoding="utf-8") as stream:
        inventory = {row["file"]: row for row in csv.DictReader(stream)}
    crops_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    completed: dict[str, dict[str, str | int | float]] = {}
    if checkpoint_path.is_file():
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                decoded = parse_flat_json_object(line)
                parsed_row = {
                    key: value
                    for key, value in decoded.items()
                    if isinstance(value, (str, int, float))
                }
                filename = parsed_row.get("file")
                if (
                    isinstance(filename, str)
                    and parsed_row.get("checkpoint_version") == CHECKPOINT_VERSION
                    and parsed_row.get("crop_size") == crop_size
                ):
                    completed[filename] = parsed_row
    rows: list[dict[str, str | int | float]] = []
    for index, path in enumerate(files, 1):
        crop_path = crops_dir / f"{path.stem}.jpg"
        checkpoint_row = completed.get(path.name)
        if (
            checkpoint_row is not None
            and checkpoint_row.get("source_bytes") == path.stat().st_size
            and checkpoint_row.get("source_mtime_ns") == path.stat().st_mtime_ns
            and crop_path.is_file()
        ):
            print(
                f"[{index:02d}/{len(files):02d}] {path.name} (checkpoint)", flush=True
            )
            rows.append(checkpoint_row)
            continue
        if path.name not in inventory:
            raise RuntimeError(f"Inventory has no row for {path.name}")
        print(f"[{index:02d}/{len(files):02d}] {path.name}", flush=True)
        rgb = decoder.decode(path)
        center_x, center_y = locate_sun(rgb)
        crop, x0, y0 = crop_around(rgb, center_x, center_y, crop_size)
        measurements = measure_crop(crop)
        row: dict[str, str | int | float] = dict(inventory[path.name])
        row.update({"sun_x": center_x, "sun_y": center_y, "crop_x0": x0, "crop_y0": y0})
        row.update(measurements)
        row.update({
            "checkpoint_version": CHECKPOINT_VERSION,
            "crop_size": crop_size,
            "source_bytes": path.stat().st_size,
            "source_mtime_ns": path.stat().st_mtime_ns,
        })
        row["detail_score"] = (
            measurements["edge_p995"]
            / max(measurements["edge_width_proxy_px"], 0.1) ** 0.5
            * math.exp(-measurements["clip_all_pct"] / 6.0)
        )
        rows.append(row)
        save_diagnostic(crop, crop_path)
        with checkpoint_path.open("a", encoding="utf-8") as stream:
            _ = stream.write(json.dumps(row) + "\n")
    if not rows:
        raise RuntimeError("No RAW files analyzed")
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def print_rankings(rows: list[dict[str, str | int | float]]) -> None:
    print("\nRANK BY detail_score")
    for row in sorted(rows, key=lambda item: float(item["detail_score"]), reverse=True):
        summary = (
            f"{row['file']:12} score={float(row['detail_score']):.5f} "
            f"edge={float(row['edge_p995']):.4f} "
            f"width={float(row['edge_width_proxy_px']):.2f}px "
            f"clip_all={float(row['clip_all_pct']):.3f}%"
        )
        print(summary)
