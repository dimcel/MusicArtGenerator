# Legacy Manual Demos

These files are historical development demonstrations. They are not automated
tests and are not used by the current `music-art-generator` command.

Most demos load diffusion models, write images or videos, and may require a
CUDA GPU. Run them from the repository root with Python module syntax, for
example:

```bash
python -m scripts.legacy_demos.image_generator_demo
```

The maintained automated tests remain under `test/`. New behavior should be
tested there with assertions and without downloading models or writing large
output files.
