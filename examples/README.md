# Examples

These examples show common MusicArtGenerator configurations. Run them from the
repository directory and replace the input paths and prompts with your own.

## Generated Example

[![Twist and quiet-hold preview](assets/twist_preview.gif)](assets/twist_preview_with_audio.mp4)

This excerpt was produced by the current pipeline with cadence mode, ControlNet,
quiet hold, music-reactive zoom, and onset-reactive twist. The GIF is silent.
[Open the MP4 to watch the example with music](assets/twist_preview_with_audio.mp4).

## Basic Cadence Generation

```bash
python -m src.music_video_app \
  --audio "path/to/track.wav" \
  --prompt "an abstract landscape made of light and flowing color" \
  --max-seconds 8 \
  --mode cadence \
  --cadence 2 \
  --with-audio
```

Cadence mode runs diffusion at regular intervals and detected beats. Frames
between diffusion updates use the selected coherence method.

## Onset Jitter And Quiet Hold

```bash
python -m src.music_video_app \
  --audio "path/to/track.wav" \
  --init-image "path/to/starting_image.png" \
  --prompt "a surreal night landscape with glowing geometric forms" \
  --max-seconds 12 \
  --mode cadence \
  --cadence 2 \
  --controlnet \
  --onset-jitter \
  --quiet-hold \
  --with-audio
```

`--onset-jitter` adds small deterministic camera movement around strong onset
events. `--quiet-hold` reduces motion and prompt switching during quieter
sections.

## Twist And Quiet Hold

```bash
python -m src.music_video_app \
  --audio "path/to/track.wav" \
  --init-image "path/to/starting_image.png" \
  --prompt "a cosmic landscape with soft purple light and distant planets" \
  --max-seconds 12 \
  --mode cadence \
  --cadence 2 \
  --coherence none \
  --controlnet \
  --zoom-base 1.01 \
  --zoom-beat-boost 0.08 \
  --steady-twist \
  --steady-twist-max-deg 4.0 \
  --quiet-hold \
  --with-audio
```

This preset combines beat-responsive zoom with a short rotation pulse driven by
onsets. Quiet hold calms both effects in lower-activity sections.

## Prompt Changes And ControlNet

```bash
python -m src.music_video_app \
  --audio "path/to/track.wav" \
  --init-image "path/to/starting_image.png" \
  --prompt "glowing forest at night || crystalline city at dawn || clouds above a violet ocean" \
  --prompt-change-every-beats 4 \
  --max-seconds 12 \
  --mode cadence \
  --cadence 2 \
  --controlnet \
  --control-scale-base 0.8 \
  --quiet-hold \
  --with-audio
```

Separate prompts with `||`. The active prompt changes after the requested
number of detected beats. Canny ControlNet uses the previous transformed frame
to retain more visible structure.

## Resume A Run

```bash
python -m src.music_video_app \
  --audio "path/to/track.wav" \
  --resume-dir "output_cadence_20260823_153000_288f" \
  --max-seconds 8 \
  --mode cadence \
  --cadence 2 \
  --with-audio
```

In resume mode, `--max-seconds 8` adds eight seconds after the final saved
frame. Use the same audio, prompt, frame rate, and generation settings when you
want a direct continuation.

## Choosing A Configuration

| Goal | Useful settings |
|---|---|
| Faster first test | short `--max-seconds`, cadence mode, larger `--cadence` |
| More reactions to transients | `--onset-jitter` or `--steady-twist` |
| Calmer quiet sections | `--quiet-hold` |
| Stronger structural continuity | `--controlnet` or `--re-anchor` |
| Several visual concepts | a `||` prompt list and `--prompt-change-every-beats` |
| Continue a long generation | `--resume-dir` |

See the [complete CLI reference](../src/music_video_app/README.md) for every
available setting and its default value.
