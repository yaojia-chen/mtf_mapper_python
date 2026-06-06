# MTF Mapper Python

Standalone Python port of the core `mtf_mapper` analyzer workflow.

This project focuses on practical core behavior:

- load standard images with OpenCV
- load ImageJ-style raw pixel streams with explicit metadata
- detect dark rectangular/slanted-edge targets
- estimate per-edge MTF/SFR values
- write `annotated.png`, `edge_mtf_values.csv`, `edge_sfr_values.csv`, and optional `mtf_heatmap.png`

It is not a full replacement for the C++ GUI or advanced scientific modes.

## Setup

```bash
uv sync --dev
```

## Run

```bash
uv run mtf-mapper-py input.png out --linear
```

The reported MTF value defaults to `mtf_ny4`, meaning the SFR/MTF contrast at
Nyquist frequency divided by 4. Use `--mtf-metric mtf_ny2` for Nyquist/2, or
`--mtf-metric mtf50` for the legacy MTF50 crossing frequency.

Add `--heatmap` to write `mtf_heatmap.png`, a field-of-view map interpolated
from the measured edge MTF values.

## GUI

```bash
uv run mtf-mapper-gui
```

The GUI keeps the original C++ app's basic workflow, organized into a compact
workspace:

- top toolbar for opening files, loading the sample chart, choosing output, and running analysis
- left sidebar for input, measurement, output, advanced, and raw import settings
- central annotated image preview
- lower tabs for results, curve inspection, and log output

When an annotated image is shown, click near an annotated edge/ROI to update
the docked curve inspector. Switch between SFR, ESF, and LSF, then hover over
the curve to show a dashed vertical readout line with the corresponding values.

## Sample Image

A synthetic slanted-edge chart is included at `samples/mtf_test_chart.png`.
The more realistic field sample `samples/mtf_realistic_field_chart.png` has
sharper targets near the image center, softer targets toward the corners, and
anisotropic blur so one measured direction remains higher than the other.
The soft field sample `samples/mtf_soft_field_chart.png` avoids sharpening and
keeps the measured MTF values below 1.

```bash
uv run mtf-mapper-py samples/mtf_test_chart.png sample-output --linear
uv run mtf-mapper-py samples/mtf_realistic_field_chart.png realistic-output --linear --heatmap --edges
uv run mtf-mapper-py samples/mtf_soft_field_chart.png soft-output --linear --heatmap --edges
```

For raw files:

```bash
uv run mtf-mapper-py image.raw out \
  --raw \
  --raw-width 2048 \
  --raw-height 1536 \
  --raw-dtype uint16 \
  --raw-byte-order little
```

## Test

```bash
uv run pytest -q
```
