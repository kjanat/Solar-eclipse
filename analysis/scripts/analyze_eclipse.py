import ctypes, os, glob, csv, math, time, json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/mnt/data/eclipse_raws")
OUT = Path("/mnt/data/eclipse_analysis")
OUT.mkdir(exist_ok=True)
CROPS = OUT / "sun_crops"
CROPS.mkdir(exist_ok=True)

lib = ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libraw.so.23")
for n, args, ret in [
    ("libraw_init", [ctypes.c_uint], ctypes.c_void_p),
    ("libraw_open_file", [ctypes.c_void_p, ctypes.c_char_p], ctypes.c_int),
    ("libraw_unpack", [ctypes.c_void_p], ctypes.c_int),
    ("libraw_dcraw_process", [ctypes.c_void_p], ctypes.c_int),
    (
        "libraw_dcraw_make_mem_image",
        [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)],
        ctypes.c_void_p,
    ),
    ("libraw_dcraw_clear_mem", [ctypes.c_void_p], None),
    ("libraw_close", [ctypes.c_void_p], None),
    ("libraw_set_output_bps", [ctypes.c_void_p, ctypes.c_int], None),
    ("libraw_set_no_auto_bright", [ctypes.c_void_p, ctypes.c_int], None),
    ("libraw_set_bright", [ctypes.c_void_p, ctypes.c_float], None),
    ("libraw_set_highlight", [ctypes.c_void_p, ctypes.c_int], None),
    ("libraw_set_gamma", [ctypes.c_void_p, ctypes.c_int, ctypes.c_float], None),
    ("libraw_set_output_color", [ctypes.c_void_p, ctypes.c_int], None),
]:
    f = getattr(lib, n)
    f.argtypes = args
    f.restype = ret


class Img(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("height", ctypes.c_ushort),
        ("width", ctypes.c_ushort),
        ("colors", ctypes.c_ushort),
        ("bits", ctypes.c_ushort),
        ("data_size", ctypes.c_uint),
    ]


def decode(path):
    raw = lib.libraw_init(0)
    try:
        rc = lib.libraw_open_file(raw, os.fsencode(path))
        if rc:
            raise RuntimeError(f"open {rc}")
        lib.libraw_set_output_bps(raw, 16)
        lib.libraw_set_no_auto_bright(raw, 1)
        lib.libraw_set_bright(raw, ctypes.c_float(1.0))
        lib.libraw_set_highlight(raw, 0)
        lib.libraw_set_gamma(raw, 0, ctypes.c_float(1.0))
        lib.libraw_set_gamma(raw, 1, ctypes.c_float(1.0))
        lib.libraw_set_output_color(raw, 1)
        rc = lib.libraw_unpack(raw)
        if rc:
            raise RuntimeError(f"unpack {rc}")
        rc = lib.libraw_dcraw_process(raw)
        if rc:
            raise RuntimeError(f"process {rc}")
        err = ctypes.c_int()
        p = lib.libraw_dcraw_make_mem_image(raw, ctypes.byref(err))
        if not p:
            raise RuntimeError(f"mem {err.value}")
        im = Img.from_address(p)
        buf = ctypes.string_at(p + ctypes.sizeof(Img), im.data_size)
        arr = (
            np
            .frombuffer(buf, dtype="<u2")
            .reshape(im.height, im.width, im.colors)
            .copy()
        )
        lib.libraw_dcraw_clear_mem(p)
        return arr
    finally:
        lib.libraw_close(raw)


def locate_sun(rgb):
    # Downsample; use max RGB because clipped/highlight cores are reliable anchors.
    ds = rgb[::4, ::4, :]
    mx = ds.max(axis=2).astype(np.float32)
    # Adaptive top-tail threshold; enough pixels to form the solar region without swallowing the bright sky.
    q = np.percentile(mx, 99.90)
    thr = max(q, float(mx.max()) * 0.55)
    mask = (mx >= thr).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        y, x = np.unravel_index(np.argmax(mx), mx.shape)
        return x * 4, y * 4
    # Prefer large bright component; component score includes mean brightness.
    best = None
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 3:
            continue
        yy, xx = np.where(labels == i)
        mean = float(mx[yy, xx].mean())
        score = area * (mean / max(float(mx.max()), 1)) ** 2
        if best is None or score > best[0]:
            best = (score, i)
    if best is None:
        y, x = np.unravel_index(np.argmax(mx), mx.shape)
        return x * 4, y * 4
    cx, cy = cents[best[1]]
    return int(round(cx * 4)), int(round(cy * 4))


def get_crop(rgb, cx, cy, size=1400):
    h, w = rgb.shape[:2]
    half = size // 2
    x0 = max(0, min(w - size, cx - half))
    y0 = max(0, min(h - size, cy - half))
    return rgb[y0 : y0 + size, x0 : x0 + size], x0, y0


def metrics(crop):
    f = crop.astype(np.float32) / 65535.0
    # linear luminance in output RGB space
    lum = 0.2126 * f[:, :, 0] + 0.7152 * f[:, :, 1] + 0.0722 * f[:, :, 2]
    anyclip = (crop >= 65530).any(axis=2)
    allclip = (crop >= 65530).all(axis=2)
    # Normalize to high percentile rather than max, then log-ish display transform for robust edge detection.
    p995 = max(float(np.percentile(lum, 99.5)), 1e-6)
    n = np.clip(lum / p995, 0, 1)
    # Suppress pixel noise before edge sharpness; sigma ~1px remains sensitive to optical edge spread.
    sm = cv2.GaussianBlur(n, (0, 0), 0.8)
    gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    grad = np.hypot(gx, gy)
    # Candidate solar boundary: pixels where normalized signal is neither deep sky nor flat clipped core.
    band = (sm > 0.06) & (sm < 0.94)
    vals = grad[band]
    if vals.size:
        edge_p99 = float(np.percentile(vals, 99))
        edge_p995 = float(np.percentile(vals, 99.5))
        edge_topmean = float(vals[vals >= np.percentile(vals, 99)].mean())
    else:
        edge_p99 = edge_p995 = edge_topmean = 0.0
    # Width proxy: threshold-transition area divided by perimeter of >50% bright region.
    bright = (sm > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours if cv2.contourArea(c) > 10)
    transition = ((sm > 0.1) & (sm < 0.9)).sum()
    width_proxy = float(transition / perimeter) if perimeter > 0 else float("nan")
    # local contrast around edge, robustly
    return {
        "clip_any_pct": float(anyclip.mean() * 100),
        "clip_all_pct": float(allclip.mean() * 100),
        "lum_p99": float(np.percentile(lum, 99)),
        "lum_p995": p995,
        "lum_max": float(lum.max()),
        "edge_p99": edge_p99,
        "edge_p995": edge_p995,
        "edge_topmean": edge_topmean,
        "edge_width_proxy_px": width_proxy,
    }


def render_diag(crop, out, gamma=2.2):
    # percentile-scaled, intentionally consistent analysis render
    f = crop.astype(np.float32) / 65535.0
    scale = max(float(np.percentile(f, 99.8)), 1e-6)
    f = np.clip(f / scale, 0, 1)
    f = np.power(f, 1 / gamma)
    u8 = np.clip(f * 255 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(u8, "RGB").save(out, quality=95)


inv = {r["file"]: r for r in csv.DictReader(open(ROOT / "inventory.csv"))}
rows = []
checkpoint = OUT / "metrics.jsonl"
done = {}
if checkpoint.exists():
    for line in checkpoint.read_text().splitlines():
        if line.strip():
            import json

            rr = json.loads(line)
            done[rr["file"]] = rr
            rows.append(rr)
for i, path in enumerate(sorted(ROOT.glob("*.CR3")), 1):
    if path.name in done:
        print(f"[{i:02d}/21] {path.name} (checkpoint)", flush=True)
        continue
    print(f"[{i:02d}/21] {path.name}", flush=True)
    rgb = decode(path)
    cx, cy = locate_sun(rgb)
    crop, x0, y0 = get_crop(rgb, cx, cy, 1400)
    m = metrics(crop)
    r = dict(inv[path.name])
    r.update({"sun_x": cx, "sun_y": cy, "crop_x0": x0, "crop_y0": y0})
    r.update(m)
    # crude combined detail score: high edge acutance, low transition width, penalize severe all-channel clipping
    r["detail_score"] = (
        m["edge_p995"]
        / (max(m["edge_width_proxy_px"], 0.1) ** 0.5)
        * math.exp(-m["clip_all_pct"] / 6.0)
    )
    rows.append(r)
    with checkpoint.open("a") as fh:
        fh.write(json.dumps(r) + "\n")
    render_diag(crop, CROPS / (path.stem + ".jpg"))

fields = list(rows[0].keys())
with open(OUT / "metrics.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

print("\nRANK BY detail_score")
for r in sorted(rows, key=lambda x: float(x["detail_score"]), reverse=True):
    print(
        f"{r['file']:12} score={float(r['detail_score']):.5f} edge={float(r['edge_p995']):.4f} width={float(r['edge_width_proxy_px']):.2f}px clip_all={float(r['clip_all_pct']):.3f}% clip_any={float(r['clip_any_pct']):.3f}%"
    )
print("\nRANK BY least all-channel clipping")
for r in sorted(rows, key=lambda x: float(x["clip_all_pct"])):
    print(
        f"{r['file']:12} clip_all={float(r['clip_all_pct']):.4f}% any={float(r['clip_any_pct']):.4f}% edge={float(r['edge_p995']):.4f}"
    )
