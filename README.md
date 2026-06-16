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

## Desktop App Stack

```bash
uv sync --dev
cd frontend && npm install
```

The GUI now uses a modern local desktop architecture:

- React + TypeScript + Vite for the interface
- FastAPI for the local analysis service
- native Python, NumPy, SciPy, and OpenCV for image processing
- a Tauri scaffold for packaging the frontend as a desktop app

During development, run the local API from the repository root in one terminal:

```bash
uv run mtf-mapper-api
```

Then run the frontend from the repository root in another terminal:

```bash
npm run frontend:dev
```

Open [http://127.0.0.1:1420](http://127.0.0.1:1420).

The React app keeps the image-first workflow:

- open a standard image and see the original preview immediately
- use **Preview detection** to inspect detected targets and the threshold mask
- use **Run analysis** to create annotated, SFR CSV, MTF CSV, diagnostics, and
  optional spatial-map outputs
- switch between Original, Detection, Threshold mask, Annotated, and Spatial map
- inspect SFR, ESF, and LSF curves for measured edges
- download generated images, CSV files, and diagnostics directly from the page

For `.raw`, `.bin`, and `.dat` files, enable **Read as raw pixel stream**, set
the Raw import metadata, then click **Reload with Raw settings**. If the current
metadata is wrong, the app keeps you in the Advanced panel so you can update the
settings and reload without reopening the file.

To build the frontend:

```bash
npm run frontend:build
```

To run the Tauri shell after installing Rust/Cargo:

```bash
npm run desktop:dev
```

The current Tauri scaffold opens the React app. The FastAPI service is still run
as a companion local process in development; bundling and auto-starting that
process is the next packaging step.

## GitHub Builds

GitHub Actions can compile and verify the app without needing Rust/Cargo on
your local machine. The workflow in `.github/workflows/ci.yml` runs on pushes
to `main` and on pull requests. It:

- installs and tests the Python analyzer/API
- audits and builds the React frontend
- installs Linux Tauri system dependencies
- builds the Tauri desktop shell on Ubuntu
- uploads the Linux desktop bundle as a workflow artifact

The first packaged desktop build is Linux-only. Add macOS and Windows runners
once the Linux bundle is passing and the Python API packaging strategy is final.

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

In the app, a failed `.raw`, `.bin`, or `.dat` import keeps you in the Advanced
settings area so you can verify the Raw import dimensions, data type, byte
order, header bytes, channel count, and level mapping. Packed 10/12/14-bit
streams must be unpacked before import.

## Test

```bash
uv run pytest -q
```
