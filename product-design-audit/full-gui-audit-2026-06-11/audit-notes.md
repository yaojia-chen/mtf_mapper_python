# MTF Mapper Python GUI Audit

Date: June 11, 2026

## Audit Scope

Combined product-design and accessibility audit of the desktop workflow from empty state through sample analysis, image inspection, detection tuning, ROI editing, diagnostics, and SFR/ESF/LSF inspection.

Primary user goal: confidently load an image, confirm that targets were detected correctly, run a trustworthy analysis, and inspect results without losing image context.

## Overall Assessment

The current left-right layout is a strong foundation. It keeps the image dominant, provides a useful collapsible dock, and makes original, annotated, detection, and threshold views available without leaving the analysis workspace.

The main weakness is workflow clarity. The top toolbar, image mode, dock tab, detection state, and analysis state can each change independently. This makes the screen feel like several technical tools sharing a window instead of one guided analysis workflow. The app is usable for an expert who already understands its model, but it asks new or occasional users to infer too much.

## Strengths

- Image-first wide-screen layout is appropriate for 4:3 and square test images.
- Original and annotated views are easy to switch between.
- Result summary exposes the most important measurements quickly.
- Detection preview uses a crisp cyan outline and target IDs that are easier to interpret than the dense MTF labels.
- Collapsing the dock gives the image substantially more room.
- Edge Inspector is a good use of a separate window; SFR, ESF, and LSF charts have clear axes and readable plots.
- The selected edge uses a distinct cyan highlight, linking the chart back to the image.

## Priority Findings

### P1: Create a clearer task flow

The toolbar gives nearly equal visual weight to opening files, changing output location, previewing detection, editing ROIs, opening settings, hiding the dock, running analysis, and clearing. Their availability also does not clearly communicate the expected order.

Recommendation: group the workflow into `Open`, `Tune detection`, and `Run analysis`. Keep `Run analysis` as the single strong primary action. Move output-folder selection and clearing into secondary menus. Disable or explain actions that are not yet available.

### P1: Make detection and ROI editing self-explanatory

Detection and threshold views have no visible legend or on-canvas instructions. The ROI editor is numeric-only and can open while the threshold mask is displayed, so users lose the visual connection between the values and the target they are editing.

Recommendation: add a compact overlay explaining cyan outlines, target numbers, excluded state, click-to-toggle, and Shift-drag. Open ROI editing on the Detection view, highlight the selected target, provide previous/next navigation, and update the rectangle live as values change.

### P1: Prevent contradictory analysis context

After detection preview, the Diagnostics tab reports `0 good, 0 review, 0 poor` while the Result tab still shows 36 measured edges. Both are individually explainable, but the UI does not say that diagnostics now describe the preview-detection pass rather than the completed analysis.

Recommendation: label diagnostics by source and time, such as `Detection preview diagnostics` or `Last completed analysis`. Do not silently replace completed-analysis quality statistics with preview-only statistics.

### P1: Fix Advanced-panel discoverability and clipping

The Advanced tab is a long form with a subtle scrollbar. Raw controls extend below the visible area, and the final fields are easy to miss. Disabled raw fields create a large, low-contrast block that dominates the panel even when raw import is off.

Recommendation: use collapsible sections for `Detection`, `SFR`, and `Raw import`. Collapse Raw import by default, reveal its fields only when enabled, and keep validation/help next to each setting.

### P2: Reduce annotation density

The magenta labels are readable, but 36 always-visible values create substantial visual noise and cover parts of the targets. At higher zoom the labels become especially dominant.

Recommendation: support `Markers only`, `Selected/hovered values`, and `All values` annotation modes. Scale labels within a bounded range and optionally color-code quality using shape or icon differences as well as color.

### P2: Simplify the visual hierarchy

Always-visible scrollbars, nested borders around the canvas, equal-weight toolbar buttons, and abbreviated dock tabs make the interface feel denser than necessary.

Recommendation: hide inactive scrollbars in Fit mode, reduce nested canvas borders, use full labels such as `Advanced` and `Diagnostics`, and preserve a small summary strip when the dock is collapsed.

### P2: Make charts more decision-oriented

The curve plots are clean, but they do not show key interpretation guides.

Recommendation: add threshold/Nyquist reference lines where relevant, show the measured crossing/value directly on the chart, explain hover interaction, and offer export/copy actions. Keep the selected edge visible behind the inspector.

## Accessibility Risks

- macOS Computer Use could see the Tk window but could not read or activate its controls. This is a strong screen-reader/accessibility-tree risk and should be tested with VoiceOver.
- Most buttons and tabs appear roughly 24-30 px high, below common 44 px touch-target guidance.
- No keyboard-focus indicator was visible in the captured states. Keyboard order and shortcuts need direct testing.
- Detection states rely heavily on cyan outlines and color changes. Add labels, patterns, or icons for excluded/selected states.
- Abbreviated tabs (`Adv.`, `Diag.`) reduce clarity for cognitive accessibility and assistive technology.
- Disabled Raw import text has very low contrast. Although disabled controls have different requirements, the section remains visually prominent and hard to parse.
- Image zoom changes the image but not the surrounding UI; test the app with larger system text and display scaling.

## Step-by-Step Evidence

1. `01-start.png` - Empty state - **Mixed**
   - Calm image-first layout and clear empty message.
   - The canvas message does not provide direct Open or Try sample actions, while the toolbar presents many equal-weight choices.

2. `02-advanced-settings.png` - Advanced settings - **Needs work**
   - Settings are logically grouped.
   - Raw import consumes most of the panel while disabled, and lower controls are clipped below the viewport.

3. `03-result-annotated.png` - Completed analysis and annotated result - **Mixed**
   - Strong summary metrics and clear completion title.
   - Dense labels compete with the image; the output tree uses a large area but provides little guidance on what to inspect next.

4. `04-original-view.png` - Original image view - **Healthy**
   - Switching to the unannotated source is straightforward and preserves results context.
   - The persistent scrollbars and double outline add unnecessary visual weight in Fit mode.

5. `05-detection-preview.png` - Detection preview - **Mixed**
   - Cyan target outlines and IDs are crisp and easy to distinguish.
   - No legend or visible instruction explains how to include, exclude, or add targets.

6. `06-threshold-mask.png` - Threshold mask - **Mixed**
   - Useful debugging view and clear black/white separation.
   - The relationship between this view, detection settings, and detected targets is not explained.

7. `07-edit-rois-dialog.png` - ROI editor - **Needs work**
   - Compact and exposes exact geometry.
   - Numeric-only editing lacks live visual feedback, units, undo/reset, and clear connection to the selected ROI.

8. `08-diagnostics.png` - Diagnostics - **Needs work**
   - Technical detection counts and a plain-language health sentence are useful.
   - Raw monospace text has weak hierarchy, and preview diagnostics conflict visually with the still-present completed-analysis context.

9. `09-log.png` - Analysis log - **Mixed**
   - Concise and useful for troubleshooting.
   - Full paths wrap awkwardly; no timestamps, copy action, severity styling, or quick link to the related setting/error.

10. `10-edge-inspector-sfr.png` - SFR curve - **Healthy**
    - Clear chart, axes, selected-edge metadata, and curve switcher.
    - Add interpretation guides and a stronger visual link to the selected image edge.

11. `11-edge-inspector-esf.png` - ESF curve - **Healthy**
    - Curve switching is immediate and the ESF shape is readable.
    - Export and comparison controls would improve expert workflows.

12. `12-edge-inspector-lsf.png` - LSF curve - **Healthy**
    - LSF is presented consistently with the other curves.
    - Reference markers and explanatory metadata would reduce interpretation effort.

13. `13-dock-collapsed.png` - Collapsed dock - **Healthy**
    - Excellent use of wide-screen space and a clear `Show dock` recovery action.
    - A compact result/selection summary would prevent total loss of context.

14. `14-zoomed-preview.png` - Zoomed annotated image - **Mixed**
    - Zoom provides useful detailed inspection.
    - Labels become oversized and dominate the targets; annotation density controls are needed.

## Evidence Limits

- Screenshots were captured from the live app in this audit run. The Computer Use accessibility bridge timed out on the Tk controls, so the flow was exercised through the app's existing callbacks and captured with macOS screenshots.
- A Codex desktop notification overlaps the lower-right corner of several screenshots; it is not part of the product UI.
- Screenshot review cannot confirm keyboard order, focus behavior, VoiceOver output, dynamic error recovery, color-contrast ratios, or touchpad gesture quality. These require direct accessibility and interaction testing.

## Recommended Design Sequence

1. Restructure the toolbar and dock around the user flow: open, tune, analyze, inspect.
2. Redesign Detection and ROI editing as one visually guided mode.
3. Collapse and progressively reveal Advanced and Raw import settings.
4. Add annotation-density controls and bounded label scaling.
5. Separate preview diagnostics from completed-analysis diagnostics.
6. Run a focused keyboard, VoiceOver, target-size, and contrast pass.
