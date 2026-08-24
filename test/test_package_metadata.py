from music_art_generator import __version__
from music_art_generator.music_video_app.cli import build_parser


def test_package_version_is_1_0_0():
    assert __version__ == "1.0.0"


def test_cli_uses_installed_command_name():
    assert build_parser().prog == "music-art-generator"
