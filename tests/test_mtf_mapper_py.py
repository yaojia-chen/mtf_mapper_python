import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

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
            },
        )
        self.assertEqual(args.input_image, "input.raw")
        self.assertEqual(args.output_dir, "out")
        self.assertTrue(args.raw)
        self.assertEqual(args.raw_width, 20)
        self.assertEqual(args.raw_height, 10)
        self.assertEqual(args.raw_byte_order, "big")
        self.assertIsNone(args.pixelsize)
        self.assertEqual(args.mtf_metric, "mtf_ny4")
        self.assertTrue(args.heatmap)
        self.assertEqual(args.threshold_mode, "adaptive")
        self.assertEqual(args.roi_radius, 18.0)
        self.assertEqual(args.esf_method, "auto")
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

    def test_gui_raw_import_error_prompt(self):
        path = Path("capture.RAW")
        message = mtf_mapper_gui.raw_import_error_message(path, ValueError("raw input ended early"))
        self.assertTrue(mtf_mapper_gui.is_raw_input_path(path))
        self.assertIn("current Raw import settings", message)
        self.assertIn("Width", message)
        self.assertIn("Byte order", message)
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

    def test_sfr_interpolation_on_smooth_edge(self):
        x = np.linspace(-8.0, 8.0, 257)
        esf = 0.5 + 0.5 * np.tanh(x)
        freqs, sfr = mtf_mapper_py.sfr_from_esf(esf, sample_spacing=0.125, smooth=False, full_sfr=False)
        mtf50 = mtf_mapper_py.interpolate_mtf(freqs, sfr, 0.5)
        self.assertTrue(math.isfinite(mtf50))
        self.assertGreater(mtf50, 0.0)

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


if __name__ == "__main__":
    unittest.main()
