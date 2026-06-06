# MTF Mapper Product Design Review

## Audit Scope

The main desktop workflow: opening an image, configuring analysis, running the
sample chart, and interpreting the first result.

## User Goal

Move from a test image to a trustworthy MTF result, then inspect individual
edges or exported files when more detail is needed.

## Captured Steps

1. `00-revised-empty-state.png` - Ready to begin. Healthy: the primary actions,
   default setup, empty preview, and expected result summary are visible.
2. `01-revised-sample-results.png` - Sample analysis complete. Healthy: the
   annotated image and key measurement summary are visible before file-level
   detail.
3. `02-compact-layout-image-outline.png` - Iterated workspace. Healthy: setup
   begins immediately, the toolbar is quieter, and the image boundary remains
   visible against the preview surround.
4. `03-toolbar-hierarchy.png` - Refined toolbar hierarchy. Healthy: file
   actions sit on the left while the primary analysis action is isolated on
   the right.
5. `04-magenta-annotations.png` - Refined annotation treatment. Healthy: vivid
   magenta labels remain distinct from both the black targets and white chart
   at fit-to-window scale.

## Strengths

- The annotated preview keeps the measured image at the center of the workflow.
- The sample chart gives new users a low-risk way to understand the product.
- Detailed SFR curves and exported tables remain available for expert review.
- Setup and Advanced tabs separate routine decisions from infrequent controls.

## UX Risks

- Metric names such as `mtf_ny4` are scientifically precise but not explained
  in the interface.
- The summary reports distribution facts, but does not yet say whether a result
  is acceptable for a user's lens, sensor, or test standard.
- Batch analysis summarizes only the most recently completed image.
- Result files still require users to know which artifact answers which
  question.

## Accessibility Risks

- Native widget appearance and contrast vary by operating system theme.
- Keyboard order, screen-reader names, and focus visibility need a dedicated
  assistive-technology test; screenshots cannot confirm them.
- The annotated image uses small blue labels that may be difficult to read at
  fit-to-window scale.
- Some controls use specialist abbreviations without persistent explanations.

## Implemented Improvements

- Added stronger product and primary-action hierarchy.
- Staged the setup into four numbered sections and moved raw/SFR controls into
  an Advanced tab.
- Added immediate input preview and visible analysis progress.
- Added a result summary with edge count, block count, median, and range.
- Added clearer empty, no-edge, and completion states.
- Added direct access to the output folder.
- Removed redundant introductory copy and added a clear image-frame outline.
- Replaced the annotation color with vivid magenta and strengthened label
  strokes for clearer fit-to-window viewing.

## Recommended Next Pass

Add short metric explanations and a result-quality framework that users can
configure for their own test standard. For batch workflows, add a comparison
table with one row per image and sortable summary metrics.

## Evidence Limits

This review covers the visible macOS desktop experience and the sample flow.
It does not establish WCAG compliance, validate Windows/Linux rendering, or
test the workflow with real lab users.
