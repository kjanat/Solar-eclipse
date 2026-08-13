import json
import math

from solar_eclipse_analysis.json_value import parse_flat_json_object


def test_parses_checkpoint_json_written_by_stdlib() -> None:
    row = {
        "file": 'eclipse "raw".CR3',
        "crop_size": 1400,
        "score": 0.125,
        "valid": True,
        "missing": None,
    }

    assert parse_flat_json_object(json.dumps(row)) == row


def test_parses_nonfinite_checkpoint_number() -> None:
    result = parse_flat_json_object('{"width": NaN}')

    width = result["width"]
    assert isinstance(width, float)
    assert math.isnan(width)
