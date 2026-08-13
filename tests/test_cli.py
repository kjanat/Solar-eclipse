import pytest

from solar_eclipse_analysis.cli import main


def test_expected_error_has_no_traceback(tmp_path, capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--root", str(tmp_path), "analyze"])

    captured = capsys.readouterr()
    assert exit_info.value.code == 1
    assert captured.out == ""
    assert "error: No CR3 files found" in captured.err
    assert "Traceback" not in captured.err


def test_edge_spread_does_not_require_raw_decoder(tmp_path, capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--root", str(tmp_path), "edge-spread"])

    captured = capsys.readouterr()
    assert exit_info.value.code == 1
    assert "Metrics not found" in captured.err
    assert "rawpy" not in captured.err.casefold()
