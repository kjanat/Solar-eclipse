from pathlib import Path

import numpy as np

from solar_eclipse_analysis.raw import RawDecoder


def test_decoder_rejects_unexpected_image_shape(monkeypatch, tmp_path: Path) -> None:
    class FakeRaw:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def postprocess(self, **_arguments):
            return np.zeros((2, 2), dtype=np.uint16)

    monkeypatch.setattr(
        "solar_eclipse_analysis.raw.rawpy.imread", lambda _path: FakeRaw()
    )

    try:
        RawDecoder().decode(tmp_path / "image.CR3")
    except RuntimeError as error:
        assert "Unexpected decoded image format" in str(error)
    else:
        raise AssertionError("Expected invalid decoded image shape to fail")
