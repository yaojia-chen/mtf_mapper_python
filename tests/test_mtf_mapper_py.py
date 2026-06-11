import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import mtf_mapper_gui
import mtf_mapper_py


class MtfMapperPyTests(unittest.TestCase):
    def test_srgb_to_linear_known_points(self):
        values = np.array([0.0, 0.04045, 1.0])
        linear = mtf_mapper_py.srgb_to_linear(values)
        self.assertAlmostEqual(linear[0], 0.0)
        self.assertAlmostEqual(linear[1], 0.04045 / 12.92)
        self.assertAlmostEqual(linear[2], 1.0)

    def test_parse_args_defaults_to_annotate_and_edges(self):
        args = mtf_mapper_py.parse_args(["input.png", "out"])
        self.assertTrue(args.annotate)
        self.assertTrue(args.edges)
        self.assertFalse(args.heatmap)
        self.assertEqual(args.mtf, 50.0)
        self.assertEqual(args.mtf_metric, "mtf_ny4")
        self.assertEqual(args.threshold_mode, "hybrid")
        self.assertEqual(args.roi_radius, 12.0)
        self.assertEqual(args.esf_method, "pixel-binned")
        self.assertFalse(args.auto_tune)
        self.assertFalse(args.exclude_small_fiducials)
        self.assertEqual(args.fiducial_max_area_ratio, 0.2)
        self.assertEqual(args.raw_normalization, "auto")

    def test_raw_requires_dimensions(self):
        with self.assertRaises(SystemExit):
            mtf_mapper_py.parse_args(["input.raw", "out", "--raw"])

    def test_gui_namespace_builder(self):
        args = mtf_mapper_gui.namespace_from_gui_values(
            Path("input.raw"),
            Path("out"),
            {
                "threshold": 0.4,
                "threshold_mode": "adaptive",
                "threshold_window": 0.25,
                "roi_radius": 18.0,
                "esf_method": "Auto fallback",
                "linear": True,
                "invert": True,
                "single_roi": True,
                "mtf_metric": "mtf_ny4",
                "mtf": 50.0,
                "annotate": True,
                "edges": False,
                "heatmap": True,
                "full_sfr": True,
                "nosmoothing": True,
                "pixelsize": "",
                "raw": True,
                "raw_width": "20",
                "raw_height": "10",
                "raw_dtype": "uint16",
                "raw_byte_order": "big",
                "raw_header": 4,
                "raw_channels": 1,
                "raw_channel_order": "bgr",
                "raw_normalization": "Bit depth",
                "raw_bit_depth": 12,
                "raw_alignment": "right",
                "raw_black_level": "",
                "raw_white_level": "",
                "exclude_small_fiducials": True,
                "fiducial_max_area_percent": 15.0,
                "annotation_labels": "Markers only",
            },
        )
        self.assertEqual(args.input_image, "input.raw")
        self.assertEqual(args.output_dir, "out")
        self.assertTrue(args.raw)
        self.assertEqual(args.raw_width, 20)
        self.assertEqual(args.raw_height, 10)
        self.assertEqual(args.raw_byte_order, "big")
        self.assertEqual(args.raw_normalization, "bit-depth")
        self.assertEqual(args.raw_bit_depth, 12)
        self.assertEqual(args.raw_channel_order, "bgr")
        self.assertTrue(args.exclude_small_fiducials)
        self.assertEqual(args.fiducial_max_area_ratio, 0.15)
        self.assertIsNone(args.pixelsize)
        self.assertEqual(args.mtf_metric, "mtf_ny4")
        self.assertTrue(args.heatmap)
        self.assertEqual(args.threshold_mode, "adaptive")
        self.assertEqual(args.roi_radius, 18.0)
        self.assertEqual(args.esf_method, "auto")
        self.assertEqual(args.annotation_labels, "Markers only")
        self.assertEqual(
            mtf_mapper_gui.normalize_threshold_mode("Hybrid (adaptive + global)"),
            "hybrid",
        )

    def test_gui_click_helpers(self):
        measurements = [
            mtf_mapper_py.EdgeMeasurement(
                1, 10.0, 10.0, 0.5, "mtf_ny2", "mtf_ny2", 0.0, 0.0, 4.0, 0.0, np.array([1.0, 0.5]), 0.8,
                edge_start_x=0.0, edge_start_y=10.0, edge_end_x=20.0, edge_end_y=10.0,
            ),
            mtf_mapper_py.EdgeMeasurement(
                2, 80.0, 40.0, 0.4, "mtf_ny2", "mtf_ny2", 0.0, 0.0, 6.0, 0.0, np.array([1.0, 0.4]), 0.8,
                edge_start_x=50.0, edge_start_y=40.0, edge_end_x=110.0, edge_end_y=40.0,
            ),
        ]
        nearest = mtf_mapper_gui.nearest_measurement(measurements, 106.0, 43.0)
        self.assertEqual(nearest.block_id, 2)
        self.assertAlmostEqual(mtf_mapper_gui.distance_to_measurement(measurements[1], 106.0, 43.0), 3.0)
        state = mtf_mapper_gui.PreviewState(
            Path("annotated.png"),
            measurements,
            display_scale=0.5,
            image_width=200,
            image_height=100,
            offset_x=5,
            offset_y=10,
        )
        self.assertEqual(mtf_mapper_gui.preview_to_image_coords(15, 20, state), (20, 20))

    def test_gui_trackpad_zoom_factors_are_smooth_and_bounded(self):
        self.assertAlmostEqual(mtf_mapper_gui.wheel_zoom_factor(120), 1.15)
        self.assertAlmostEqual(mtf_mapper_gui.wheel_zoom_factor(-120), 1 / 1.15)
        self.assertGreater(mtf_mapper_gui.wheel_zoom_factor(1), 1.0)
        self.assertLess(mtf_mapper_gui.wheel_zoom_factor(1), 1.1)
        self.assertGreater(mtf_mapper_gui.magnify_zoom_factor(0.1), 1.0)
        self.assertLess(mtf_mapper_gui.magnify_zoom_factor(-0.1), 1.0)
        self.assertLessEqual(mtf_mapper_gui.magnify_zoom_factor(10.0), math.exp(0.7))

    def test_gui_measurement_summary(self):
        measurements = [
            mtf_mapper_py.EdgeMeasurement(
                1, 10.0, 10.0, 0.2, "mtf_ny4", "mtf_ny4", 0.0, 0.0, 4.0, 0.0, np.array([1.0]), 0.8
            ),
            mtf_mapper_py.EdgeMeasurement(
                1, 20.0, 10.0, 0.4, "mtf_ny4", "mtf_ny4", 0.0, 0.0, 4.0, 0.0, np.array([1.0]), 0.8
            ),
            mtf_mapper_py.EdgeMeasurement(
                2, 30.0, 10.0, 0.9, "mtf_ny4", "mtf_ny4", 0.0, 0.0, 4.0, 0.0, np.array([1.0]), 0.8
            ),
        ]
        summary = mtf_mapper_gui.summarize_measurements(measurements)
        self.assertEqual(summary["edges"], 3)
        self.assertEqual(summary["blocks"], 2)
        self.assertAlmostEqual(summary["median"], 0.4)
        self.assertAlmostEqual(summary["minimum"], 0.2)
        self.assertAlmostEqual(summary["maximum"], 0.9)

    @unittest.skipIf(mtf_mapper_py.cv2 is None, "OpenCV is not installed")
    def test_gui_prepares_raw_original_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_path = tmp_path / "tiny.raw"
            raw_path.write_bytes(np.array([0, 1000, 2000, 3000], dtype="<u2").tobytes())
            preview_path = mtf_mapper_gui.prepare_original_preview(
                raw_path,
                tmp_path / "preview",
                {
                    "raw": True,
                    "raw_width": "2",
                    "raw_height": "2",
                    "raw_dtype": "uint16",
                    "raw_byte_order": "little",
                    "raw_header": 0,
                    "raw_channels": 1,
                    "invert": False,
                },
            )
            self.assertTrue(preview_path.exists())
            self.assertEqual(mtf_mapper_py.cv2.imread(str(preview_path)).shape[:2], (2, 2))

    @unittest.skipIf(mtf_mapper_py.cv2 is None, "OpenCV is not installed")
    def test_gui_prepares_standard_original_preview_without_conversion(self):
        path = Path("image.png")
        self.assertEqual(mtf_mapper_gui.prepare_original_preview(path, Path("preview"), {"raw": False}), path)

    @unittest.skipIf(mtf_mapper_py.cv2 is None, "OpenCV is not installed")
    def test_gui_decodes_preview_source_to_rgb_once_ready_for_cached_resizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "color.png"
            bgr = np.zeros((12, 16, 3), dtype=np.uint8)
            bgr[:, :, 2] = 255
            self.assertTrue(mtf_mapper_py.cv2.imwrite(str(path), bgr))
            rgb = mtf_mapper_gui.rgb_image_from_cv(path)
            self.assertEqual(rgb.shape, (12, 16, 3))
            self.assertEqual(tuple(rgb[0, 0]), (255, 0, 0))

    def test_gui_raw_import_error_prompt(self):
        path = Path("capture.RAW")
        message = mtf_mapper_gui.raw_import_error_message(path, ValueError("raw input ended early"))
        self.assertTrue(mtf_mapper_gui.is_raw_input_path(path))
        self.assertIn("current Raw import settings", message)
        self.assertIn("Width", message)
        self.assertIn("Byte order", message)
        self.assertIn("Reload with Raw settings", message)
        self.assertIn("raw input ended early", message)
        self.assertFalse(mtf_mapper_gui.is_raw_input_path(Path("capture.png")))

    def test_load_raw_image_with_header_and_byte_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_path = Path(tmp) / "tiny.raw"
            raw_path.write_bytes(b"HEAD" + np.array([1, 256, 513, 1024], dtype=">u2").tobytes())
            img = mtf_mapper_py.load_raw_image(
                raw_path,
                width=2,
                height=2,
                raw_dtype="uint16",
                byte_order="big",
                header=4,
                channels=1,
            )
            self.assertEqual(img.shape, (2, 2))
            self.assertEqual(int(img[0, 1]), 256)

    def test_raw_import_rejects_trailing_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_path = Path(tmp) / "oversized.raw"
            raw_path.write_bytes(np.arange(10, dtype=np.uint16).tobytes())
            with self.assertRaisesRegex(ValueError, "trailing data"):
                mtf_mapper_py.load_raw_image(raw_path, 2, 2, "uint16", "native", 0, 1)

    def test_raw_rgb_channel_order_uses_rgb_luminance(self):
        image = np.array([[[1.0, 0.0, 0.0]]])
        lum, _original = mtf_mapper_py.luminance_from_array(
            image, linear=True, invert=False, apply_srgb_for_uint8=False, normalized=True, channel_order="rgb"
        )
        self.assertAlmostEqual(float(lum[0, 0]), 0.2126)

    def test_raw_bit_depth_normalization_supports_right_and_left_alignment(self):
        right = np.array([0, 1023, 2048, 4095], dtype=np.uint16)
        normalized, report = mtf_mapper_py.normalize_raw_image(right, mode="bit-depth", bit_depth=12)
        self.assertAlmostEqual(float(normalized[-1]), 1.0)
        self.assertAlmostEqual(float(normalized[1]), 1023 / 4095)
        self.assertEqual(report.effective_bit_depth, 12)
        self.assertEqual(report.alignment, "right")

        left = np.array([0, 512 << 6, 1023 << 6], dtype=np.uint16)
        normalized, report = mtf_mapper_py.normalize_raw_image(
            left, mode="bit-depth", bit_depth=10, alignment="left"
        )
        self.assertAlmostEqual(float(normalized[-1]), 1.0)
        self.assertAlmostEqual(float(normalized[1]), 512 / 1023)
        self.assertEqual(report.effective_bit_depth, 10)
        self.assertEqual(report.alignment, "left")

    def test_raw_auto_levels_expand_effective_range(self):
        image = np.full((100, 100), 3500, dtype=np.uint16)
        image[20:80, 20:80] = 400
        normalized, report = mtf_mapper_py.normalize_raw_image(image, mode="auto")
        self.assertAlmostEqual(float(normalized[0, 0]), 1.0)
        self.assertAlmostEqual(float(normalized[30, 30]), 0.0)
        self.assertEqual(report.effective_bit_depth, 12)

    def test_raw_normalization_rejects_nonfinite_samples(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            mtf_mapper_py.normalize_raw_image(np.array([0.0, 1.0, np.nan]), mode="auto")

    def test_sfr_interpolation_on_smooth_edge(self):
        x = np.linspace(-8.0, 8.0, 257)
        esf = 0.5 + 0.5 * np.tanh(x)
        freqs, sfr = mtf_mapper_py.sfr_from_esf(esf, sample_spacing=0.125, smooth=False, full_sfr=False)
        mtf50 = mtf_mapper_py.interpolate_mtf(freqs, sfr, 0.5)
        self.assertTrue(math.isfinite(mtf50))
        self.assertGreater(mtf50, 0.0)

    def test_sfr_normalizes_windowed_dc_and_applies_derivative_correction(self):
        x = np.linspace(-8.0, 8.0, 257)
        esf = 0.5 + 0.5 * np.tanh(x)
        spacing = 0.125
        lsf = np.gradient(esf, spacing)
        response = np.abs(np.fft.rfft(lsf * np.hamming(lsf.size)))
        expected_freqs = np.fft.rfftfreq(lsf.size, d=spacing)
        argument = 2.0 * np.pi * expected_freqs * spacing
        correction = np.ones_like(argument)
        correction[1:] = argument[1:] / np.sin(argument[1:])
        expected = response / response[0] * correction
        freqs, sfr = mtf_mapper_py.sfr_from_esf(esf, spacing, smooth=False, full_sfr=True)
        self.assertTrue(np.allclose(sfr, expected[expected_freqs <= 2.0]))
        self.assertAlmostEqual(float(sfr[0]), 1.0)

    def test_sfr_preserves_lsf_dc_instead_of_subtracting_mean(self):
        esf = np.linspace(0.0, 1.0, 129)
        _freqs, sfr = mtf_mapper_py.sfr_from_esf(esf, sample_spacing=0.125, smooth=False, full_sfr=False)
        self.assertAlmostEqual(float(sfr[0]), 1.0)

    def test_sfr_allows_valid_sharpening_overshoot(self):
        x = np.linspace(-12.0, 12.0, 193)
        spacing = float(x[1] - x[0])
        lsf = np.exp(-(x**2) / (2 * 0.7**2)) - 0.1 * np.exp(-(x**2) / (2 * 2.0**2))
        esf = np.cumsum(lsf) * spacing
        esf = (esf - esf[0]) / (esf[-1] - esf[0])
        _freqs, sfr = mtf_mapper_py.sfr_from_esf(esf, spacing, smooth=False, full_sfr=False)
        self.assertGreater(float(np.max(sfr)), 1.0)

    def test_sfr_is_stable_when_edge_shifts_within_roi(self):
        x = np.linspace(-12.0, 12.0, 193)
        mtf50_values = []
        for shift in (0.0, 4.0):
            esf = 0.5 + 0.5 * np.tanh(x - shift)
            freqs, sfr = mtf_mapper_py.sfr_from_esf(esf, 0.125, smooth=False, full_sfr=False)
            mtf50_values.append(mtf_mapper_py.interpolate_mtf(freqs, sfr, 0.5))
        self.assertAlmostEqual(mtf50_values[0], mtf50_values[1], delta=0.005)

    def test_moving_average_preserves_esf_plateaus(self):
        values = np.concatenate((np.zeros(50), np.ones(50)))
        smoothed = mtf_mapper_py.moving_average(values, 21)
        self.assertAlmostEqual(float(smoothed[0]), 0.0)
        self.assertAlmostEqual(float(smoothed[-1]), 1.0)

    @unittest.skipIf(mtf_mapper_py.cv2 is None, "OpenCV is not installed")
    def test_adaptive_only_threshold_recovers_mid_gray_target(self):
        lum = np.full((180, 240), 0.82, dtype=np.float64)
        lum[55:125, 75:165] = 0.68
        hybrid = mtf_mapper_py.detect_boxes(lum, threshold=0.55, threshold_window=0.3, threshold_mode="hybrid")
        adaptive = mtf_mapper_py.detect_boxes(lum, threshold=0.55, threshold_window=0.3, threshold_mode="adaptive")
        self.assertEqual(hybrid, [])
        self.assertEqual(len(adaptive), 1)

    @unittest.skipIf(mtf_mapper_py.cv2 is None, "OpenCV is not installed")
    def test_edge_roi_radius_controls_esf_width(self):
        lum = np.zeros((80, 100), dtype=np.float64)
        lum[:, 50:] = 1.0
        p0 = np.array([50.0, 10.0])
        p1 = np.array([50.0, 70.0])
        small_esf, spacing, _contrast = mtf_mapper_py.esf_from_edge(lum, p0, p1, radius=6.0)
        large_esf, _, _ = mtf_mapper_py.esf_from_edge(lum, p0, p1, radius=18.0)
        self.assertEqual(spacing, 0.125)
        self.assertGreater(len(large_esf), len(small_esf))

    @unittest.skipIf(mtf_mapper_py.cv2 is None, "OpenCV is not installed")
    def test_pixel_binned_esf_uses_original_pixels_and_reports_occupancy(self):
        lum = np.zeros((100, 120), dtype=np.float64)
        yy, xx = np.indices(lum.shape)
        lum[xx > 50.0 + 0.15 * yy] = 1.0
        p0 = np.array([53.0, 20.0])
        p1 = np.array([62.0, 80.0])
        result = mtf_mapper_py.pixel_binned_esf_from_edge(lum, p0, p1, oversampling=8, radius=10.0)
        self.assertEqual(result.method, "pixel-binned")
        self.assertEqual(result.sample_spacing, 0.125)
        self.assertGreater(result.bin_occupancy, 0.75)
        self.assertGreater(result.contrast, 0.8)

    @unittest.skipIf(mtf_mapper_py.cv2 is None, "OpenCV is not installed")
    def test_auto_esf_falls_back_for_axis_aligned_edge(self):
        lum = np.zeros((80, 100), dtype=np.float64)
        lum[:, 50:] = 1.0
        p0 = np.array([50.0, 10.0])
        p1 = np.array([50.0, 70.0])
        result = mtf_mapper_py.create_esf_from_edge(lum, p0, p1, method="auto")
        self.assertEqual(result.method, "interpolated")

    @unittest.skipIf(mtf_mapper_py.cv2 is None, "OpenCV is not installed")
    def test_measure_edge_reports_sparse_pixel_bins(self):
        lum = np.zeros((80, 100), dtype=np.float64)
        lum[:, 50:] = 1.0
        measurement = mtf_mapper_py.measure_edge(
            lum,
            1,
            np.array([50.0, 10.0]),
            np.array([50.0, 70.0]),
            np.array([50.0, 10.0]),
            50.0,
            "mtf_ny4",
            False,
            True,
            None,
            esf_method="pixel-binned",
        )
        self.assertEqual(measurement.esf_method, "pixel-binned")
        self.assertLess(measurement.bin_occupancy, 0.5)
        self.assertTrue(any("sparse pixel bins" in note for note in measurement.quality_notes))

    @unittest.skipIf(mtf_mapper_py.cv2 is None, "OpenCV is not installed")
    def test_detection_report_and_auto_tune(self):
        lum = np.full((160, 220), 0.9, dtype=np.float64)
        lum[50:110, 70:150] = 0.25
        boxes, report = mtf_mapper_py.detect_boxes_with_diagnostics(lum, 0.55, 0.3, "hybrid")
        self.assertEqual(len(boxes), 1)
        self.assertEqual(report.accepted_count, 1)
        tuned_boxes, tuned_report = mtf_mapper_py.auto_tune_detection(lum)
        self.assertGreaterEqual(len(tuned_boxes), 1)
        self.assertIn(tuned_report.threshold_mode, {"hybrid", "adaptive", "global"})

    @unittest.skipIf(mtf_mapper_py.cv2 is None, "OpenCV is not installed")
    def test_relative_area_filter_excludes_small_fiducials(self):
        cv2 = mtf_mapper_py.cv2
        lum = np.full((300, 400), 0.9, dtype=np.float64)
        cv2.rectangle(lum, (100, 90), (220, 190), 0.2, -1)
        cv2.rectangle(lum, (300, 30), (320, 50), 0.2, -1)
        unfiltered, _report = mtf_mapper_py.detect_boxes_with_diagnostics(lum, 0.55, 0.3, "global")
        filtered, report = mtf_mapper_py.detect_boxes_with_diagnostics(lum, 0.55, 0.3, "global", 0.2)
        self.assertEqual(len(unfiltered), 2)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(report.rejected_fiducial, 1)
        self.assertEqual(report.fiducial_filter_ratio, 0.2)
        self.assertTrue(any("fiducial" in suggestion for suggestion in report.suggestions()))

    @unittest.skipIf(mtf_mapper_py.cv2 is None, "OpenCV is not installed")
    def test_auto_tune_rejects_tiny_rectangular_distractors(self):
        cv2 = mtf_mapper_py.cv2
        lum = np.full((500, 700), 0.9, dtype=np.float64)
        cv2.rectangle(lum, (250, 180), (450, 320), 0.2, -1)
        for x in range(20, 380, 20):
            cv2.rectangle(lum, (x, 20), (x + 10, 30), 0.1, -1)
        boxes, report = mtf_mapper_py.auto_tune_detection(lum)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(report.accepted_count, 1)

    @unittest.skipIf(mtf_mapper_py.cv2 is None, "OpenCV is not installed")
    def test_refine_edge_line_follows_actual_transition(self):
        yy, xx = np.indices((300, 300), dtype=np.float64)
        actual_angle = 5.0
        actual_slope = math.tan(math.radians(actual_angle))
        distance = (xx - 150.0 - actual_slope * (yy - 150.0)) / math.sqrt(1.0 + actual_slope**2)
        lum = 0.5 + 0.5 * np.tanh(distance)
        proposed_slope = math.tan(math.radians(5.5))
        tangent = np.array([proposed_slope, 1.0])
        tangent /= np.linalg.norm(tangent)
        p0 = np.array([150.0, 150.0]) - 100.0 * tangent
        p1 = np.array([150.0, 150.0]) + 100.0 * tangent
        refined0, refined1 = mtf_mapper_py.refine_edge_line(lum, p0, p1)
        self.assertAlmostEqual(mtf_mapper_py.folded_edge_angle(refined0, refined1), actual_angle, delta=0.15)

    def test_diagnostics_export_includes_quality(self):
        measurement = mtf_mapper_py.EdgeMeasurement(
            1, 10.0, 10.0, 0.5, "mtf_ny4", "mtf_ny4", 0.0, 0.0, 4.0, 0.0,
            np.array([1.0]), 0.8, quality_score=0.9, quality_label="Good",
        )
        report = mtf_mapper_py.DetectionReport("hybrid", 0.55, 0.33, contour_count=2, accepted_count=1)
        with tempfile.TemporaryDirectory() as tmp:
            mtf_mapper_py.write_diagnostics(Path(tmp), report, [measurement])
            data = json.loads((Path(tmp) / "analysis_diagnostics.json").read_text(encoding="utf-8"))
        self.assertEqual(data["detection"]["accepted_count"], 1)
        self.assertEqual(data["edge_quality"][0]["label"], "Good")

    def test_edge_profiles_and_gui_curve_data(self):
        x = np.linspace(-8.0, 8.0, 129)
        source_esf = 0.5 + 0.5 * np.tanh(x)
        esf, lsf = mtf_mapper_py.edge_profiles_from_esf(source_esf, sample_spacing=0.125, smooth=False)
        measurement = mtf_mapper_py.EdgeMeasurement(
            1,
            10.0,
            10.0,
            0.5,
            "mtf_ny4",
            "mtf_ny4",
            0.0,
            0.0,
            4.0,
            0.0,
            np.array([1.0, 0.5]),
            0.8,
            esf=esf,
            lsf=lsf,
            sample_spacing=0.125,
        )
        values, x_min, x_max, x_label, y_label, unit = mtf_mapper_gui.curve_data(measurement, "LSF")
        self.assertEqual(len(values), len(source_esf))
        self.assertLess(x_min, 0.0)
        self.assertGreater(x_max, 0.0)
        self.assertEqual(x_label, "Distance across edge (pixels)")
        self.assertEqual(y_label, "LSF")
        self.assertEqual(unit, "px")
        self.assertGreater(float(np.max(lsf)), 0.0)

    def test_reported_fixed_frequency_mtf(self):
        freqs = np.array([0.0, 0.125, 0.25, 0.5])
        sfr = np.array([1.0, 0.8, 0.6, 0.2])
        self.assertAlmostEqual(
            mtf_mapper_py.reported_mtf_value(freqs, sfr, "mtf_ny2", 50.0, None),
            0.6,
        )
        self.assertAlmostEqual(
            mtf_mapper_py.reported_mtf_value(freqs, sfr, "mtf_ny4", 50.0, None),
            0.8,
        )

    @unittest.skipIf(mtf_mapper_py.cv2 is None, "OpenCV is not installed")
    def test_integration_synthetic_rectangle_outputs(self):
        cv2 = mtf_mapper_py.cv2
        image = np.full((160, 220), 235, dtype=np.uint8)
        rect = ((110, 80), (80, 54), 7)
        points = cv2.boxPoints(rect).astype(np.int32)
        cv2.fillConvexPoly(image, points, 25)
        image = cv2.GaussianBlur(image, (0, 0), 1.2)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "synthetic.png"
            output_dir = tmp_path / "out"
            cv2.imwrite(str(input_path), image)
            rc = mtf_mapper_py.main([str(input_path), str(output_dir), "--linear"])
            self.assertEqual(rc, 0)
            self.assertTrue((output_dir / "annotated.png").exists())
            self.assertTrue((output_dir / "edge_mtf_values.csv").exists())
            self.assertTrue((output_dir / "edge_sfr_values.csv").exists())
            self.assertFalse((output_dir / "mtf_heatmap.png").exists())
            self.assertGreater((output_dir / "edge_mtf_values.csv").stat().st_size, 0)
            with (output_dir / "edge_mtf_values.csv").open(encoding="utf-8", newline="") as fin:
                self.assertEqual(
                    next(csv.reader(fin)),
                    ["block_id", "edge_x", "edge_y", "mtf_ny4", "corner_x", "corner_y"],
                )
            with (output_dir / "edge_sfr_values.csv").open(encoding="utf-8", newline="") as fin:
                header = next(csv.reader(fin))
                self.assertEqual(header[:5], ["block_id", "edge_x", "edge_y", "edge_angle", "radial_angle"])
                self.assertEqual(header[5], "sfr_000")

    @unittest.skipIf(mtf_mapper_py.cv2 is None, "OpenCV is not installed")
    def test_heatmap_optional_output(self):
        cv2 = mtf_mapper_py.cv2
        image = np.full((160, 220), 235, dtype=np.uint8)
        rect = ((110, 80), (80, 54), 7)
        points = cv2.boxPoints(rect).astype(np.int32)
        cv2.fillConvexPoly(image, points, 25)
        image = cv2.GaussianBlur(image, (0, 0), 1.2)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "synthetic.png"
            output_dir = tmp_path / "out"
            cv2.imwrite(str(input_path), image)
            rc = mtf_mapper_py.main([str(input_path), str(output_dir), "--linear", "--heatmap"])
            self.assertEqual(rc, 0)
            self.assertTrue((output_dir / "mtf_heatmap.png").exists())
            self.assertFalse((output_dir / "annotated.png").exists())
            self.assertFalse((output_dir / "edge_mtf_values.csv").exists())

    def test_heatmap_uses_block_average_values(self):
        measurements = [
            mtf_mapper_py.EdgeMeasurement(
                1, 0.0, 0.0, 0.2, "mtf_ny4", "mtf_ny4", 0.0, 0.0, 0.0, 0.0, np.array([1.0]), 1.0
            ),
            mtf_mapper_py.EdgeMeasurement(
                1, 10.0, 10.0, 0.4, "mtf_ny4", "mtf_ny4", 0.0, 0.0, 0.0, 0.0, np.array([1.0]), 1.0
            ),
            mtf_mapper_py.EdgeMeasurement(
                2, 40.0, 20.0, 0.8, "mtf_ny4", "mtf_ny4", 0.0, 0.0, 0.0, 0.0, np.array([1.0]), 1.0
            ),
        ]
        blocks = mtf_mapper_py.heatmap_blocks_from_measurements(measurements)
        self.assertEqual(len(blocks), 2)
        self.assertAlmostEqual(blocks[0].x, 5.0)
        self.assertAlmostEqual(blocks[0].y, 5.0)
        self.assertAlmostEqual(blocks[0].mtf_value, 0.3)
        self.assertAlmostEqual(blocks[1].mtf_value, 0.8)

    def test_annotation_style_scales_with_image_size(self):
        small = mtf_mapper_py.annotation_style((100, 100, 3))
        medium = mtf_mapper_py.annotation_style((800, 800, 3))
        large = mtf_mapper_py.annotation_style((2400, 2400, 3))
        self.assertLess(small[0], medium[0])
        self.assertLess(small[3], medium[3])
        self.assertLess(medium[0], large[0])
        self.assertLess(medium[3], large[3])

    @unittest.skipIf(mtf_mapper_py.cv2 is None, "OpenCV is not installed")
    def test_small_image_annotation_remains_compact(self):
        lum = np.full((80, 100), 0.5, dtype=np.float64)
        measurement = mtf_mapper_py.EdgeMeasurement(
            1, 50.0, 40.0, 0.5, "mtf_ny4", "mtf_ny4", 0.0, 0.0, 0.0, 0.0, np.array([1.0]), 1.0
        )
        annotated = mtf_mapper_py.make_annotation(lum, lum, [measurement])
        base = mtf_mapper_py.luminance_to_bgr(lum)
        changed = np.any(annotated != base, axis=2)
        self.assertLess(int(np.count_nonzero(changed)), lum.size // 5)

    @unittest.skipIf(mtf_mapper_py.cv2 is None, "OpenCV is not installed")
    def test_marker_only_annotation_reduces_visual_clutter(self):
        lum = np.full((300, 400), 0.5, dtype=np.float64)
        measurement = mtf_mapper_py.EdgeMeasurement(
            1, 200.0, 150.0, 0.586, "mtf_ny4", "mtf_ny4", 0.0, 0.0, 0.0, 0.0, np.array([1.0]), 1.0
        )
        base = mtf_mapper_py.luminance_to_bgr(lum)
        full = mtf_mapper_py.make_annotation(lum, lum, [measurement], "All values")
        markers = mtf_mapper_py.make_annotation(lum, lum, [measurement], "Markers only")
        full_changed = np.count_nonzero(np.any(full != base, axis=2))
        marker_changed = np.count_nonzero(np.any(markers != base, axis=2))
        self.assertLess(marker_changed, full_changed)

    @unittest.skipIf(mtf_mapper_py.cv2 is None, "OpenCV is not installed")
    def test_detection_preview_emphasizes_selected_target(self):
        lum = np.full((200, 300), 0.8, dtype=np.float64)
        box = np.array([[80, 60], [220, 60], [220, 140], [80, 140]], dtype=np.float64)
        normal = mtf_mapper_py.make_detection_preview(lum, lum, [box])
        selected = mtf_mapper_py.make_detection_preview(lum, lum, [box], selected_block=1)
        self.assertGreater(np.count_nonzero(normal != selected), 0)

    @unittest.skipIf(mtf_mapper_py.cv2 is None, "OpenCV is not installed")
    def test_tiny_heatmap_does_not_crash(self):
        measurement = mtf_mapper_py.EdgeMeasurement(
            1, 10.0, 10.0, 0.5, "mtf_ny4", "mtf_ny4", 0.0, 0.0, 0.0, 0.0, np.array([1.0]), 1.0
        )
        heatmap = mtf_mapper_py.make_mtf_heatmap(np.ones((20, 20)), [measurement])
        self.assertEqual(heatmap.shape, (20, 20, 3))

    def test_prepare_output_dir_removes_disabled_stale_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            for name in ("annotated.png", "edge_mtf_values.csv", "edge_sfr_values.csv", "mtf_heatmap.png"):
                (output_dir / name).write_text("stale", encoding="utf-8")
            mtf_mapper_py.prepare_output_dir(output_dir, annotate=False, edges=False, heatmap=True)
            self.assertFalse((output_dir / "annotated.png").exists())
            self.assertFalse((output_dir / "edge_mtf_values.csv").exists())
            self.assertTrue((output_dir / "mtf_heatmap.png").exists())

    def test_excluded_blocks_keep_original_block_ids(self):
        boxes = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64),
            np.array([[20, 0], [30, 0], [30, 10], [20, 10]], dtype=np.float64),
            np.array([[40, 0], [50, 0], [50, 10], [40, 10]], dtype=np.float64),
        ]
        args = mtf_mapper_py.parse_args(["input.png", "out"])
        args.manual_boxes = boxes
        args.excluded_blocks = [2]

        def fake_measure(_lum, block_id, p0, p1, corner, **_kwargs):
            center = (p0 + p1) * 0.5
            return mtf_mapper_py.EdgeMeasurement(
                block_id, center[0], center[1], 0.5, "mtf_ny4", "mtf_ny4",
                corner[0], corner[1], 0.0, 0.0, np.array([1.0]), 1.0,
            )

        with mock.patch.object(
            mtf_mapper_py, "load_input_luminance", return_value=(np.ones((60, 60)), np.ones((60, 60)))
        ), mock.patch.object(mtf_mapper_py, "measure_edge", side_effect=fake_measure):
            _lum, _annotation, measurements = mtf_mapper_py.analyze_image(args)
        self.assertEqual({measurement.block_id for measurement in measurements}, {1, 3})

    @unittest.skipIf(mtf_mapper_py.cv2 is None, "OpenCV is not installed")
    def test_single_roi_and_pixelsize(self):
        cv2 = mtf_mapper_py.cv2
        image = np.full((120, 120), 230, dtype=np.uint8)
        pts = np.array([[20, 22], [95, 30], [92, 92], [18, 84]], dtype=np.int32)
        cv2.fillConvexPoly(image, pts, 20)
        image = cv2.GaussianBlur(image, (0, 0), 1.0)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "roi.png"
            output_dir = tmp_path / "out"
            cv2.imwrite(str(input_path), image)
            rc = mtf_mapper_py.main(
                [
                    str(input_path),
                    str(output_dir),
                    "--single-roi",
                    "--pixelsize",
                    "4.0",
                    "--linear",
                    "--mtf-metric",
                    "mtf50",
                ]
            )
            self.assertEqual(rc, 0)
            with (output_dir / "edge_mtf_values.csv").open(encoding="utf-8", newline="") as fin:
                rows = list(csv.DictReader(fin))
            self.assertIn("mtf50_lpmm", rows[0])
            self.assertGreater(float(rows[0]["mtf50_lpmm"]), 1.0)

    @unittest.skipIf(mtf_mapper_py.cv2 is None, "OpenCV is not installed")
    def test_nyquist_quarter_metric_header(self):
        cv2 = mtf_mapper_py.cv2
        image = np.full((120, 120), 230, dtype=np.uint8)
        pts = np.array([[20, 22], [95, 30], [92, 92], [18, 84]], dtype=np.int32)
        cv2.fillConvexPoly(image, pts, 20)
        image = cv2.GaussianBlur(image, (0, 0), 1.0)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "roi.png"
            output_dir = tmp_path / "out"
            cv2.imwrite(str(input_path), image)
            rc = mtf_mapper_py.main(
                [str(input_path), str(output_dir), "--single-roi", "--linear", "--mtf-metric", "mtf_ny4"]
            )
            self.assertEqual(rc, 0)
            with (output_dir / "edge_mtf_values.csv").open(encoding="utf-8", newline="") as fin:
                rows = list(csv.DictReader(fin))
            self.assertIn("mtf_ny4", rows[0])
            self.assertGreater(float(rows[0]["mtf_ny4"]), 0.0)

    @unittest.skipIf(mtf_mapper_py.cv2 is None, "OpenCV is not installed")
    def test_raw_uint16_integration(self):
        cv2 = mtf_mapper_py.cv2
        image = np.full((140, 180), 60000, dtype=np.uint16)
        rect = ((90, 70), (70, 46), -6)
        points = cv2.boxPoints(rect).astype(np.int32)
        cv2.fillConvexPoly(image, points, 7000)
        image = cv2.GaussianBlur(image, (0, 0), 1.0)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "synthetic_u16.raw"
            output_dir = tmp_path / "out"
            input_path.write_bytes(image.astype("<u2").tobytes())
            rc = mtf_mapper_py.main(
                [
                    str(input_path),
                    str(output_dir),
                    "--raw",
                    "--raw-width",
                    "180",
                    "--raw-height",
                    "140",
                    "--raw-dtype",
                    "uint16",
                    "--raw-byte-order",
                    "little",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue((output_dir / "annotated.png").exists())
            with (output_dir / "edge_mtf_values.csv").open(encoding="utf-8", newline="") as fin:
                rows = list(csv.DictReader(fin))
            self.assertGreaterEqual(len(rows), 4)

    @unittest.skipIf(mtf_mapper_py.cv2 is None, "OpenCV is not installed")
    def test_raw_12_bit_in_uint16_detects_target_with_auto_levels(self):
        cv2 = mtf_mapper_py.cv2
        image = np.full((140, 180), 3600, dtype=np.uint16)
        points = cv2.boxPoints(((90, 70), (70, 46), -6)).astype(np.int32)
        cv2.fillConvexPoly(image, points, 450)
        image = cv2.GaussianBlur(image, (0, 0), 1.0)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "synthetic_12bit.raw"
            output_dir = tmp_path / "out"
            input_path.write_bytes(image.astype("<u2").tobytes())
            rc = mtf_mapper_py.main(
                [
                    str(input_path),
                    str(output_dir),
                    "--raw",
                    "--raw-width",
                    "180",
                    "--raw-height",
                    "140",
                    "--raw-dtype",
                    "uint16",
                ]
            )
            self.assertEqual(rc, 0)
            diagnostics = json.loads((output_dir / "analysis_diagnostics.json").read_text(encoding="utf-8"))
            self.assertEqual(diagnostics["raw_normalization"]["mode"], "auto")
            self.assertLessEqual(diagnostics["raw_normalization"]["observed_max"], 4095)


if __name__ == "__main__":
    unittest.main()
