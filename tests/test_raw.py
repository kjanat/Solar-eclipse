from pathlib import Path
from types import TracebackType
from typing import Self

import numpy as np
import pytest

from solar_eclipse_analysis.raw import RawDecoder


def test_decoder_rejects_unexpected_image_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeRaw:
        def __enter__(self) -> Self:
            return self

        def __exit__(
            self,
            exception_type: type[BaseException] | None,
            exception: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
            return None

        def postprocess(self, **_arguments: object) -> np.ndarray:
            return np.zeros((2, 2), dtype=np.uint16)

    def fake_imread(_path: str) -> FakeRaw:
        return FakeRaw()

    monkeypatch.setattr("solar_eclipse_analysis.raw.rawpy.imread", fake_imread)

    try:
        _ = RawDecoder().decode(tmp_path / "image.CR3")
    except RuntimeError as error:
        assert "Unexpected decoded image format" in str(error)
    else:
        raise AssertionError("Expected invalid decoded image shape to fail")
