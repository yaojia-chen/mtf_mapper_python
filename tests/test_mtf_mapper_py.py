import csv
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

    def test_raw_requires_dimensions(self):
        with self.assertRaises(SystemExit):
            mtf_mapper_py.parse_args(["input.raw", "out", "--raw"])

    def test_gui_namespace_builder(self):
        args = mtf_mapper_gui.namespace_from_gui_values(
            Path("input.raw"),
            Path("out"),
            {
                "threshold": 0.4,
                "threshold_window": 0.25,
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
