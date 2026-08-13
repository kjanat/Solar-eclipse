import csv, math
from pathlib import Path
import numpy as np, cv2
from PIL import Image

CROP_DIR = Path("/mnt/data/eclipse_analysis/sun_crops")
metrics = list(csv.DictReader(open("/mnt/data/eclipse_analysis/metrics.csv")))


# These are tone-mapped diagnostics, not linear RAW. For edge width, monotonic gamma does not move geometry much,
# but we measure relative 10-90 widths after local endpoint normalization. Use luminance and sample contour normals.
def edge_spread(path):
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255
    lum = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    sm = cv2.GaussianBlur(lum, (0, 0), 0.6)
    # adaptive threshold at 50% of the bright crescent, robust to background glow
    hi = np.percentile(sm, 99.7)
    bg = np.percentile(sm, 50)
    if hi - bg < 0.1:
        return float("nan"), 0
    n = np.clip((sm - bg) / (hi - bg), 0, 1)
    mask = (n > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.contourArea(c) > 30]
    if not contours:
        return float("nan"), 0
    c = max(contours, key=cv2.contourArea)[:, 0, :]
    gx = cv2.Sobel(n, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(n, cv2.CV_32F, 0, 1, ksize=3)
    widths = []
    # avoid correlated neighboring samples
    for x, y in c[::3]:
        vx = float(gx[y, x])
        vy = float(gy[y, x])
        norm = math.hypot(vx, vy)
        if norm < 0.03:
            continue
        vx /= norm
        vy /= norm
        t = np.linspace(-12, 12, 97, dtype=np.float32)
        xs = np.asarray(x + vx * t, dtype=np.float32)
        ys = np.asarray(y + vy * t, dtype=np.float32)
        prof = cv2.remap(
            n.astype(np.float32, copy=False),
            xs.reshape(1, -1),
            ys.reshape(1, -1),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        ).ravel()
        # Orient dark->bright.
        if prof[-8:].mean() < prof[:8].mean():
            prof = prof[::-1]
        lo = float(np.median(prof[:8]))
        hi2 = float(np.median(prof[-8:]))
        contrast = hi2 - lo
        if contrast < 0.35:
            continue
        p = (prof - lo) / contrast
        # Smooth profile lightly.
        p = np.convolve(p, np.ones(5) / 5, mode="same")

        # find first 0.1 and 0.9 crossings in central useful region
        def cross(level):
            inds = np.where((p[:-1] < level) & (p[1:] >= level))[0]
            if not len(inds):
                return None
            # pick crossing nearest center
            i = min(inds, key=lambda j: abs(j - (len(p) // 2)))
            den = p[i + 1] - p[i]
            return i + (level - p[i]) / den if abs(den) > 1e-6 else float(i)

        a = cross(0.1)
        b = cross(0.9)
        if a is None or b is None or b <= a:
            continue
        width = (b - a) * (24 / 96)
        if 0.3 <= width <= 15:
            widths.append(width)
    if not widths:
        return float("nan"), 0
    return float(np.median(widths)), len(widths)


for r in metrics:
    p = CROP_DIR / (Path(r["file"]).stem + ".jpg")
    w, n = edge_spread(p)
    r["edge_10_90_px"] = w
    r["edge_samples"] = n

out = "/mnt/data/eclipse_analysis/edge_spread.csv"
with open(out, "w", newline="") as f:
    fields = list(metrics[0].keys())
    wri = csv.DictWriter(f, fieldnames=fields)
    wri.writeheader()
    wri.writerows(metrics)
print("RANK 10-90 edge width (lower = sharper)")
for r in sorted(
    metrics,
    key=lambda x: (
        float(x["edge_10_90_px"]) if not math.isnan(float(x["edge_10_90_px"])) else 999
    ),
):
    print(
        f"{r['file']:12} {float(r['edge_10_90_px']):5.2f}px  n={int(r['edge_samples']):4d}  f/{float(r['aperture']):g} {float(r['shutter_s']):.6g}s  clip={float(r['clip_all_pct']):.3f}%"
    )
