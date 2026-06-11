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

Target detection defaults to `--threshold-mode hybrid`, which combines adaptive
and global thresholding. Use `--threshold-mode adaptive` for uneven illumination
or mid-gray targets, or `--threshold-mode global` for a fixed cutoff only.
Use `--roi-radius` to control the cross-edge sampling radius used for ESF, LSF,
and SFR calculations.
ESF construction defaults to `--esf-method pixel-binned`, which projects
original source-pixel values into 1/8-pixel distance bins. Use
`--esf-method interpolated` for the previous bilinearly sampled profile method,
or `--esf-method auto` to fall back to interpolated profiles when edge angle or
bin occupancy is unsuitable.
Add `--auto-tune` to search the available threshold modes, thresholds, and
adaptive-window sizes automatically. Every command-line run also writes
`analysis_diagnostics.json` with detection counts, suggestions, and per-edge
quality scores.
For charts with small rectangular alignment fiducials, use
`--exclude-small-fiducials`. Adjust `--fiducial-max-area-ratio` to exclude
candidates below that fraction of the largest detected target area.

## GUI

```bash
uv run mtf-mapper-gui
```

The GUI keeps the original C++ app's basic workflow, organized into a compact
workspace:

- top toolbar for opening files, loading the sample chart, choosing output, and running analysis
- full-width image preview as the primary workspace
- collapsible right dock for setup, advanced settings, results, logs, and diagnostics

Use **Settings** to open the Setup dock and **Hide dock** to give the image
nearly the full window. The divider can also be dragged to choose the preferred
preview-to-dock balance. This side-by-side layout preserves vertical space for
the common square and 4:3 source-image formats on wide monitors.
The preview supports smooth mouse-wheel and trackpad zoom, including macOS
pinch gestures when exposed by the installed Tk version. Zoom stays anchored
under the pointer.

Use the preview's View selector to switch between the original and annotated
image after analysis. The original image remains available when analysis fails.
The Measurement section exposes the edge ROI radius, while Advanced offers
Hybrid, Adaptive only, and Global only target-detection modes.
Advanced also selects Pixel binning, Automatic fallback, or Interpolated
profiles for ESF construction. Diagnostics report the method used and flag
sparse pixel bins or nearly axis-aligned edges.

When an annotated image is shown, click near an annotated edge/ROI to open the
resizable Edge Inspector window. Switch between SFR, ESF, and LSF, then hover
over the curve to show a dashed vertical readout line with the corresponding
values. Clicking another edge updates the same inspector window. The selected
edge remains highlighted in the main preview.

Use **Preview detection** before analysis to inspect the detected targets.
Click a target to include or exclude it, or Shift-drag on the image to add a
rectangular ROI. **Edit ROIs** lets you move, resize, or delete a target.
Advanced settings can automatically tune detection and filter out low-quality
edges. Enable **Exclude small fiducials** for a quick relative-area filter;
preview detection first when the chart intentionally mixes target sizes.
The Diagnostics tab explains rejected candidates and suggests
adjustments, and the View selector can show the threshold mask.

Enable **MTF heat map** to make a Spatial map view available beside Original
and Annotated. Opening multiple files produces a `batch_summary.csv` comparison
in the selected output folder. File menu commands save and load reusable
settings presets and complete `.mtfproject` sessions.

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

Raw samples default to robust automatic black/white levels, so 8-, 10-, or
12-bit values stored in `uint16` usefully fill the detection range. For known
formats, use `--raw-normalization bit-depth --raw-bit-depth 12` and add
`--raw-alignment left` when samples occupy the high bits. Manual black and
white levels and the legacy full-data-type range are also available. Color RAW
streams default to RGB; use `--raw-channel-order bgr` for BGR-interleaved data.

In the GUI, a failed `.raw`, `.bin`, or `.dat` import opens the Advanced tab
and prompts you to verify the Raw import dimensions, data type, byte order,
header bytes, channel count, and level mapping. Packed 10/12/14-bit streams
must be unpacked before import.

## Test

```bash
uv run pytest -q
```
