import ctypes, os, glob, datetime, csv
from pathlib import Path

LIB = "/usr/lib/x86_64-linux-gnu/libraw.so.23"
lib = ctypes.CDLL(LIB)
lib.libraw_init.argtypes = [ctypes.c_uint]
lib.libraw_init.restype = ctypes.c_void_p
lib.libraw_open_file.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.libraw_open_file.restype = ctypes.c_int
lib.libraw_close.argtypes = [ctypes.c_void_p]
lib.libraw_close.restype = None
for name in [
    "libraw_get_raw_width",
    "libraw_get_raw_height",
    "libraw_get_iwidth",
    "libraw_get_iheight",
]:
    f = getattr(lib, name)
    f.argtypes = [ctypes.c_void_p]
    f.restype = ctypes.c_int


class GPS(ctypes.Structure):
    _fields_ = [
        ("latitude", ctypes.c_float * 3),
        ("longitude", ctypes.c_float * 3),
        ("gpstimestamp", ctypes.c_float * 3),
        ("altitude", ctypes.c_float),
        ("altref", ctypes.c_char),
        ("latref", ctypes.c_char),
        ("longref", ctypes.c_char),
        ("gpsstatus", ctypes.c_char),
        ("gpsparsed", ctypes.c_char),
    ]


class ImgOther(ctypes.Structure):
    _fields_ = [
        ("iso_speed", ctypes.c_float),
        ("shutter", ctypes.c_float),
        ("aperture", ctypes.c_float),
        ("focal_len", ctypes.c_float),
        ("timestamp", ctypes.c_long),
        ("shot_order", ctypes.c_uint),
        ("gpsdata", ctypes.c_uint * 32),
        ("parsed_gps", GPS),
        ("desc", ctypes.c_char * 512),
        ("artist", ctypes.c_char * 64),
        ("analogbalance", ctypes.c_float * 4),
    ]


lib.libraw_get_imgother.argtypes = [ctypes.c_void_p]
lib.libraw_get_imgother.restype = ctypes.POINTER(ImgOther)

rows = []
for path in sorted(glob.glob("/mnt/data/eclipse_raws/*.CR3")):
    raw = lib.libraw_init(0)
    try:
        rc = lib.libraw_open_file(raw, os.fsencode(path))
        if rc:
            raise RuntimeError((path, rc))
        m = lib.libraw_get_imgother(raw).contents
        ts = (
            datetime.datetime.fromtimestamp(m.timestamp).astimezone().isoformat()
            if m.timestamp
            else ""
        )
        rows.append({
            "file": Path(path).name,
            "bytes": os.path.getsize(path),
            "iso": m.iso_speed,
            "shutter_s": m.shutter,
            "aperture": m.aperture,
            "focal_mm": m.focal_len,
            "timestamp": ts,
            "shot_order": m.shot_order,
            "raw_w": lib.libraw_get_raw_width(raw),
            "raw_h": lib.libraw_get_raw_height(raw),
            "image_w": lib.libraw_get_iwidth(raw),
            "image_h": lib.libraw_get_iheight(raw),
        })
    finally:
        lib.libraw_close(raw)

w = csv.DictWriter(
    open("/mnt/data/eclipse_raws/inventory.csv", "w", newline=""),
    fieldnames=rows[0].keys(),
)
w.writeheader()
w.writerows(rows)
for r in rows:
    sh = (
        f"1/{round(1 / r['shutter_s'])}"
        if 0 < r["shutter_s"] < 1
        else f"{r['shutter_s']:.4g}s"
    )
    print(
        f"{r['file']:12} {r['bytes'] / 1024 / 1024:6.2f} MiB  ISO {r['iso']:5.0f}  {sh:8}  f/{r['aperture']:<4g}  {r['focal_mm']:<6g}mm  {r['timestamp']}  raw {r['raw_w']}x{r['raw_h']} img {r['image_w']}x{r['image_h']}"
    )
