import datetime
from pathlib import Path

import numpy as np
import rawpy
from numpy.typing import NDArray

Rgb16 = NDArray[np.uint16]


class RawDecoder:
    def inspect(self, path: Path) -> dict[str, str | int | float]:
        with rawpy.imread(str(path)) as raw:
            metadata = raw.other
            sizes = raw.sizes
            timestamp = metadata.timestamp
            if isinstance(timestamp, datetime.datetime):
                timestamp_text = timestamp.astimezone().isoformat()
            else:
                timestamp_text = ""
            return {
                "file": path.name,
                "bytes": path.stat().st_size,
                "iso": metadata.iso_speed,
                "shutter_s": metadata.shutter_speed,
                "aperture": metadata.aperture,
                "focal_mm": metadata.focal_length,
                "timestamp": timestamp_text,
                "shot_order": metadata.shot_order,
                "raw_w": sizes.raw_width,
                "raw_h": sizes.raw_height,
                "image_w": sizes.iwidth,
                "image_h": sizes.iheight,
            }

    def decode(self, path: Path, highlight: int = 0) -> Rgb16:
        highlight_mode = (
            rawpy.HighlightMode.Clip
            if highlight <= 0
            else rawpy.HighlightMode.ReconstructDefault
        )
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(
                output_bps=16,
                no_auto_bright=True,
                bright=1.0,
                highlight_mode=highlight_mode,
                gamma=(1.0, 1.0),
                output_color=rawpy.ColorSpace.sRGB,
            )
        rgb16 = np.asarray(rgb, dtype=np.uint16)
        if rgb16.ndim != 3 or rgb16.shape[2] != 3:
            raise RuntimeError(f"Unexpected decoded image format for {path}")
        return rgb16
