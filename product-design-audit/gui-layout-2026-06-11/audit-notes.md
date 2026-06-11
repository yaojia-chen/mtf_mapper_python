# MTF Mapper GUI Layout Audit

## Audit Scope

- Evidence: `01-analysis-result.png`
- Surface: completed-analysis Result screen on macOS
- User goal: understand measurement quality, inspect individual edges, and open useful outputs
- Audit mode: combined UX and accessibility screenshot audit

## Step 1 - Review Completed Analysis

Health: **Usable, but visually busy and inefficient**

### Strengths

1. The image preview remains the largest part of the screen, matching the core inspection task.
2. The completed state is obvious from the strong result heading.
3. The four summary metrics are readable and provide a useful scan-level overview.
4. Original/annotated view switching and image dimensions are visible near the preview.
5. The primary `Run analysis` action is visually distinct from secondary toolbar actions.

### UX Risks

1. **The annotation layer overwhelms the source image.**
   Every edge shows a full numeric value, creating 36 equally prominent labels. The repeated magenta labels make it difficult to compare blocks, see edge geometry, or identify unusual values.

2. **The Result dock spends most of its area on a mostly empty file list.**
   The file tree occupies the largest region in the dock, but contains only four outputs. This displaces more useful analysis content such as edge ranking, warnings, selected-edge detail, or quick actions.

3. **The top toolbar has too many equal-weight actions.**
   `Open`, `Try sample`, `Output folder`, `Preview detection`, `Edit ROIs`, `Settings`, and `Hide dock` all receive similar visual treatment. The intended workflow is not immediately clear.

4. **The screen duplicates navigation and layout controls.**
   `Settings` duplicates the Setup tab, while `Hide dock` controls the panel containing those tabs. This adds toolbar density without clearly communicating state.

5. **The image preview wastes space and still shows scrollbars.**
   At `61%`, the chart is surrounded by a large gray gutter while both scrollbars remain visible. Fit mode should ideally maximize the image without showing inactive scrollbars.

6. **The preview has too many nested borders.**
   The outer panel, canvas boundary, white image edge, and dark image outline create a heavy boxed-in appearance that competes with the chart.

7. **The result summary uses technical wording before user meaning.**
   `Measured mtf_ny4` is precise but not immediately explanatory. A plain label such as `Contrast at Nyquist/4` with the technical name secondary would improve comprehension.

8. **The result file list is implementation-oriented.**
   Raw filenames such as `analysis_diagnostics.json` and `edge_sfr_values.csv` are useful to technical users, but they do not explain what each output is for.

9. **The dock splitter affordance is unclear.**
   The small vertical mark between preview and dock does not clearly look draggable.

10. **The annotation view lacks a density control.**
    The user can switch between Original and Annotated, but cannot show only selected edges, outliers, block averages, or markers without labels.

### Accessibility Risks

1. Muted gray labels and secondary copy may have insufficient contrast against the light gray panel. Contrast needs measurement in the rendered theme.
2. Short tab labels such as `Adv.` and `Diag.` reduce clarity and may be difficult for unfamiliar users.
3. The annotation meaning relies heavily on bright magenta styling. Numeric text helps, but selected, warning, and normal states should not rely on color alone.
4. Several toolbar and tab controls appear compact. Target dimensions and spacing should be verified against a 44-by-44-pixel touch-target guideline where applicable.
5. Keyboard order, visible focus, screen-reader labels, splitter keyboard control, and zoom reflow cannot be confirmed from the screenshot.

## Recommendations

### Priority 1

1. Add annotation display modes: `Markers only`, `Selected edge`, `Outliers`, `Block average`, and `All values`. Default to markers or block averages.
2. Replace the large output file tree with a compact Outputs section and use the remaining dock area for an edge-quality table or selected-edge details.
3. Make Fit mode hide inactive scrollbars and reduce the gray gutter around the image.

### Priority 2

4. Group the toolbar by workflow:
   - Input: Open, Try sample
   - Detection: Preview detection, Edit ROIs
   - Run: Run analysis
   - Move Output folder and dock visibility into menus or secondary controls
5. Remove `Settings` from the toolbar or rename it to `Show setup` and keep its state synchronized with the dock.
6. Simplify the image frame to one subtle outline.
7. Replace abbreviated tabs with full labels where space permits: `Advanced` and `Diagnostics`.

### Priority 3

8. Add plain-language metric descriptions and tooltips.
9. Replace raw output filenames with labeled actions such as `Open annotated image`, `Open edge values`, and `Open diagnostics`.
10. Increase the visibility and hit area of the draggable dock divider.

## Evidence Limits

- This is a single-screen screenshot audit. It does not verify keyboard behavior, focus states, resizing, loading states, error recovery, tab behavior, tooltips, screen-reader output, or actual color-contrast ratios.
- No interaction flow was captured beyond the completed Result state.
