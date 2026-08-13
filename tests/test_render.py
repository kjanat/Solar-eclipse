from PIL import Image

from solar_eclipse_analysis.render import feathered_rgba, format_shutter


def test_feathered_tile_fades_edges() -> None:
    image = Image.new("RGB", (300, 300), "white")

    result = feathered_rgba(image)

    alpha = result.getchannel("A")
    assert alpha.getpixel((150, 150)) == 255
    assert alpha.getpixel((0, 0)) == 0


def test_format_shutter() -> None:
    assert format_shutter("0.4") == "0.4 s"
    assert format_shutter("0.0166667") == "1/60 s"
