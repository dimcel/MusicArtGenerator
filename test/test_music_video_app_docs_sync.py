from pathlib import Path
import re

from music_art_generator.music_video_app.cli import build_parser


def _cli_flags() -> set[str]:
    parser = build_parser()
    flags = set()
    for action in parser._actions:
        for option in action.option_strings:
            if option == "--help":
                continue
            if option.startswith("--"):
                flags.add(option)
    return flags


def _readme_flags(text: str) -> set[str]:
    return set(re.findall(r"`(--[a-z0-9-]+)`", text))


def test_every_cli_flag_is_documented_in_package_readme():
    readme = Path("src/music_art_generator/music_video_app/README.md")
    assert readme.exists(), "Expected src/music_art_generator/music_video_app/README.md to exist"

    text = readme.read_text(encoding="utf-8")
    cli_flags = _cli_flags()
    documented_flags = _readme_flags(text)

    missing = sorted(cli_flags - documented_flags)
    assert not missing, (
        "Missing CLI flags in src/music_art_generator/music_video_app/README.md: "
        + ", ".join(missing)
    )
