import ctypes, os, csv, math, gc
from pathlib import Path
import numpy as np, cv2
from PIL import Image, ImageOps, ImageDraw, ImageFont
import matplotlib.cm as cm

RAW = Path("/mnt/data/eclipse_raws")
OUT = Path("/mnt/data/eclipse_renders")
OUT.mkdir(exist_ok=True)
metrics = {
    r["file"]: r for r in csv.DictReader(open("/mnt/data/eclipse_analysis/metrics.csv"))
}

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


def decode(name, highlight=0):
    path = RAW / name
    raw = lib.libraw_init(0)
    try:
        if lib.libraw_open_file(raw, os.fsencode(path)):
            raise RuntimeError("open")
        lib.libraw_set_output_bps(raw, 16)
        lib.libraw_set_no_auto_bright(raw, 1)
        lib.libraw_set_bright(raw, 1.0)
        lib.libraw_set_highlight(raw, highlight)
        lib.libraw_set_gamma(raw, 0, 1.0)
        lib.libraw_set_gamma(raw, 1, 1.0)
        lib.libraw_set_output_color(raw, 1)
        if lib.libraw_unpack(raw):
            raise RuntimeError("unpack")
        if lib.libraw_dcraw_process(raw):
            raise RuntimeError("process")
        err = ctypes.c_int()
        p = lib.libraw_dcraw_make_mem_image(raw, ctypes.byref(err))
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


def sun_crop(arr, name, size=1400):
    r = metrics[name]
    cx = int(float(r["sun_x"]))
    cy = int(float(r["sun_y"]))
    h, w = arr.shape[:2]
    hs = size // 2
    x0 = max(0, min(w - size, cx - hs))
    y0 = max(0, min(h - size, cy - hs))
    return arr[y0 : y0 + size, x0 : x0 + size].copy()


def tonemap(rgb, white_pct=99.8, gamma=2.2, strength=6.0, black=0.0, saturation=1.0):
    x = rgb.astype(np.float32) / 65535.0
    L = 0.2126 * x[:, :, 0] + 0.7152 * x[:, :, 1] + 0.0722 * x[:, :, 2]
    white = max(float(np.percentile(L, white_pct)), 1e-6)
    z = np.maximum(L / white - black, 0)
    y = np.log1p(strength * z) / np.log1p(strength)
    y = np.clip(y, 0, 1)
    ratio = y / (L + 1e-6)
    o = np.clip(x * ratio[:, :, None], 0, 1)
    # saturation around luminance
    if saturation != 1:
        ol = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
        o = np.clip(ol[:, :, None] + (o - ol[:, :, None]) * saturation, 0, 1)
    o = np.power(o, 1 / gamma)
    return np.clip(o * 255 + 0.5, 0, 255).astype(np.uint8)


def unsharp(u8, sigma=1.2, amount=0.5):
    blur = cv2.GaussianBlur(u8, (0, 0), sigma)
    return np.clip(
        u8.astype(np.float32) * (1 + amount) - blur.astype(np.float32) * amount, 0, 255
    ).astype(np.uint8)


# 1: Long-exposure rays, highlight reconstruction, portrait orientation.
print("render 8490 long exposure", flush=True)
a = decode("_J3A8490.CR3", highlight=5)
a = np.rot90(a, k=3)  # Canon portrait orientation, clockwise
scale = 2200 / a.shape[0]
a = cv2.resize(a, (int(a.shape[1] * scale), 2200), interpolation=cv2.INTER_AREA)
gold = tonemap(
    a, white_pct=99.75, gamma=2.25, strength=12, black=0.002, saturation=1.15
)
# crush darkest foreground/noise slightly while preserving haze
L = cv2.cvtColor(gold, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
mask = np.clip((L - 0.015) / 0.12, 0, 1)[:, :, None]
gold = np.clip(gold.astype(np.float32) * (0.82 + 0.18 * mask), 0, 255).astype(np.uint8)
Image.fromarray(gold).save(OUT / "8490_long_exposure_gold.png", compress_level=3)
# false-color luminance from exactly same RAW data
lin = a.astype(np.float32) / 65535
lum = 0.2126 * lin[:, :, 0] + 0.7152 * lin[:, :, 1] + 0.0722 * lin[:, :, 2]
wp = max(np.percentile(lum, 99.8), 1e-6)
q = np.clip(np.log1p(40 * lum / wp) / np.log1p(40), 0, 1)
fc = (cm.magma(q)[:, :, :3] * 255).astype(np.uint8)
Image.fromarray(fc).save(OUT / "8490_linear_falsecolor_magma.png", compress_level=3)
del a, lin, lum, gold, fc
gc.collect()

# 2: Three-frame stack 8507-8509 from linear RAW crops.
print("stack 8507-8509", flush=True)
names = ["_J3A8507.CR3", "_J3A8508.CR3", "_J3A8509.CR3"]
crops = []
for n in names:
    arr = decode(n, highlight=3)
    crops.append(sun_crop(arr, n))
    del arr
    gc.collect()
ref = crops[-1].astype(np.float32) / 65535
refL = np.log1p(
    50 * (0.2126 * ref[:, :, 0] + 0.7152 * ref[:, :, 1] + 0.0722 * ref[:, :, 2])
).astype(np.float32)
aligned = []
transforms = []
for c in crops:
    f = c.astype(np.float32) / 65535
    L = np.log1p(
        50 * (0.2126 * f[:, :, 0] + 0.7152 * f[:, :, 1] + 0.0722 * f[:, :, 2])
    ).astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        cc, warp = cv2.findTransformECC(
            refL,
            L,
            warp,
            cv2.MOTION_TRANSLATION,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-7),
            None,
            3,
        )
    except cv2.error:
        cc = float("nan")
    transforms.append((cc, warp.copy()))
    w = cv2.warpAffine(
        c,
        warp,
        (c.shape[1], c.shape[0]),
        flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT,
    )
    aligned.append(w.astype(np.float32))
stack = np.median(np.stack(aligned), axis=0).astype(np.uint16)
out = tonemap(
    stack, white_pct=99.7, gamma=2.2, strength=4, black=0.0005, saturation=0.9
)
out = unsharp(out, 1.0, 0.45)
Image.fromarray(out).save(OUT / "8507-8509_aligned_median_stack.png", compress_level=3)
with open(OUT / "8507-8509_alignment.txt", "w") as f:
    for n, (cc, w) in zip(names, transforms):
        f.write(f"{n} ECC={cc} warp={w.tolist()}\n")
del crops, aligned, stack, out, ref, refL
gc.collect()

# 3: clean un-clipped outline from 8514.
print("render 8514 clean", flush=True)
a = decode("_J3A8514.CR3", highlight=0)
c = sun_crop(a, "_J3A8514.CR3", 1200)
del a
o = tonemap(c, white_pct=99.85, gamma=2.15, strength=3.2, black=0.0002, saturation=0.75)
# preserve black background and crisp limb; conservative sharpening
o = unsharp(o, 0.9, 0.35)
Image.fromarray(o).save(OUT / "8514_clean_unclipped_crescent.png", compress_level=3)
# monochrome variant
mono = cv2.cvtColor(o, cv2.COLOR_RGB2GRAY)
mono = np.clip((mono.astype(np.float32) - 2) * 1.08, 0, 255).astype(np.uint8)
Image.fromarray(mono, "L").save(OUT / "8514_clean_monochrome.png", compress_level=3)
del c, o, mono
gc.collect()

# 4: sharp f/16 detail from 8513; also an edge-structure visualization.
print("render 8513 detail", flush=True)
a = decode("_J3A8513.CR3", highlight=3)
c = sun_crop(a, "_J3A8513.CR3", 1200)
del a
o = tonemap(c, white_pct=99.65, gamma=2.2, strength=3.5, black=0.0002, saturation=0.85)
o = unsharp(o, 0.8, 0.4)
Image.fromarray(o).save(OUT / "8513_sharp_f16_detail.png", compress_level=3)
gray = cv2.cvtColor(o, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
g = cv2.magnitude(
    cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
    cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
)
g = np.clip(g / np.percentile(g, 99.9), 0, 1)
edge = (cm.inferno(g)[:, :, :3] * 255).astype(np.uint8)
Image.fromarray(edge).save(OUT / "8513_edge_structure_falsecolor.png", compress_level=3)
del c, o, gray, g, edge
gc.collect()

# 5: chronological montage made only from RAW-derived analysis crops. Not a physical sky trajectory.
print("progression montage", flush=True)
crdir = Path("/mnt/data/eclipse_analysis/sun_crops")
files = sorted(crdir.glob("*.jpg"))
W, H = 2800, 1050
canvas = Image.new("RGB", (W, H), (2, 2, 4))
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
except:
    font = small = None
for i, p in enumerate(files):
    im = Image.open(p).convert("RGB")
    # tightly crop around non-black crescent / halo, then equal visual size.
    ar = np.asarray(im)
    gr = np.asarray(im.convert("L"))
    yy, xx = np.where(gr > 22)
    if len(xx):
        x0, x1 = max(xx.min() - 25, 0), min(xx.max() + 26, im.width)
        y0, y1 = max(yy.min() - 25, 0), min(yy.max() + 26, im.height)
        im = im.crop((x0, y0, x1, y1))
    im.thumbnail((155, 155), Image.Resampling.LANCZOS)
    x = 90 + i * (W - 180) / (len(files) - 1)
    y = 500 - 250 * math.sin(i / (len(files) - 1) * math.pi)
    canvas.paste(im, (int(x - im.width / 2), int(y - im.height / 2)))
    draw.text(
        (int(x - 44), int(y + 95)),
        p.stem.replace("_J3A", ""),
        fill=(160, 160, 170),
        font=small,
    )
draw.text(
    (70, 60), "Eclipse progression — RAW sequence", fill=(235, 235, 240), font=font
)
draw.text(
    (70, 94),
    "Centered montage; positions are compositional, not the physical solar trajectory.",
    fill=(145, 145, 155),
    font=small,
)
canvas.save(OUT / "raw_eclipse_progression_montage.png", compress_level=3)

print("\nCreated:")
for p in sorted(OUT.iterdir()):
    print(p)
