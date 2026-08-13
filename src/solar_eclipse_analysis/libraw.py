import ctypes
import ctypes.util
import datetime
import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

Rgb16 = NDArray[np.uint16]


class ProcessedImage(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("height", ctypes.c_ushort),
        ("width", ctypes.c_ushort),
        ("colors", ctypes.c_ushort),
        ("bits", ctypes.c_ushort),
        ("data_size", ctypes.c_uint),
    ]


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


class ImageMetadata(ctypes.Structure):
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


def discover_library(explicit: Path | None = None) -> str:
    if explicit is not None:
        library = explicit.expanduser().resolve()
        if not library.is_file():
            raise FileNotFoundError(f"LibRaw library not found: {library}")
        return os.fspath(library)
    for name in ("raw", "libraw", "LibRaw"):
        found = ctypes.util.find_library(name)
        if found:
            return found
    raise RuntimeError("LibRaw not found; install LibRaw or pass --libraw PATH")


class LibRaw:
    def __init__(self, library: Path | None = None) -> None:
        self._lib = ctypes.CDLL(discover_library(library))
        self._configure()

    def _configure(self) -> None:
        signatures = [
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
            ("libraw_get_imgother", [ctypes.c_void_p], ctypes.POINTER(ImageMetadata)),
        ]
        for name, arguments, result in signatures:
            function = getattr(self._lib, name)
            function.argtypes = arguments
            function.restype = result
        for name in (
            "libraw_get_raw_width",
            "libraw_get_raw_height",
            "libraw_get_iwidth",
            "libraw_get_iheight",
        ):
            function = getattr(self._lib, name)
            function.argtypes = [ctypes.c_void_p]
            function.restype = ctypes.c_int

    def _open(self, path: Path) -> int:
        context = self._lib.libraw_init(0)
        if not context:
            raise RuntimeError("LibRaw initialization failed")
        result = self._lib.libraw_open_file(context, os.fsencode(path))
        if result:
            self._lib.libraw_close(context)
            raise RuntimeError(f"LibRaw could not open {path}: error {result}")
        return context

    def inspect(self, path: Path) -> dict[str, str | int | float]:
        context = self._open(path)
        try:
            metadata_pointer = self._lib.libraw_get_imgother(context)
            if not metadata_pointer:
                raise RuntimeError(f"LibRaw returned no metadata for {path}")
            metadata = metadata_pointer.contents
            timestamp = (
                datetime.datetime
                .fromtimestamp(metadata.timestamp)
                .astimezone()
                .isoformat()
                if metadata.timestamp
                else ""
            )
            return {
                "file": path.name,
                "bytes": path.stat().st_size,
                "iso": metadata.iso_speed,
                "shutter_s": metadata.shutter,
                "aperture": metadata.aperture,
                "focal_mm": metadata.focal_len,
                "timestamp": timestamp,
                "shot_order": metadata.shot_order,
                "raw_w": self._lib.libraw_get_raw_width(context),
                "raw_h": self._lib.libraw_get_raw_height(context),
                "image_w": self._lib.libraw_get_iwidth(context),
                "image_h": self._lib.libraw_get_iheight(context),
            }
        finally:
            self._lib.libraw_close(context)

    def decode(self, path: Path, highlight: int = 0) -> Rgb16:
        context = self._open(path)
        image_pointer = None
        try:
            self._lib.libraw_set_output_bps(context, 16)
            self._lib.libraw_set_no_auto_bright(context, 1)
            self._lib.libraw_set_bright(context, ctypes.c_float(1.0))
            self._lib.libraw_set_highlight(context, highlight)
            self._lib.libraw_set_gamma(context, 0, ctypes.c_float(1.0))
            self._lib.libraw_set_gamma(context, 1, ctypes.c_float(1.0))
            self._lib.libraw_set_output_color(context, 1)
            unpack_result = self._lib.libraw_unpack(context)
            if unpack_result:
                raise RuntimeError(
                    f"LibRaw unpack failed for {path}: error {unpack_result}"
                )
            process_result = self._lib.libraw_dcraw_process(context)
            if process_result:
                raise RuntimeError(
                    f"LibRaw processing failed for {path}: error {process_result}"
                )
            error = ctypes.c_int()
            image_pointer = self._lib.libraw_dcraw_make_mem_image(
                context, ctypes.byref(error)
            )
            if not image_pointer:
                raise RuntimeError(
                    f"LibRaw image allocation failed for {path}: error {error.value}"
                )
            image = ProcessedImage.from_address(image_pointer)
            buffer = ctypes.string_at(
                image_pointer + ctypes.sizeof(ProcessedImage), image.data_size
            )
            return (
                np
                .frombuffer(buffer, dtype="<u2")
                .reshape(image.height, image.width, image.colors)
                .copy()
            )
        finally:
            if image_pointer:
                self._lib.libraw_dcraw_clear_mem(image_pointer)
            self._lib.libraw_close(context)
