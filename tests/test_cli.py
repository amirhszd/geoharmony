import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from osgeo import gdal

from geoharmony.splash.cli import main
from geoharmony.splash.create_index_raster_from_waterfall import prepare_rows
from geoharmony.splash.demo import create_demo


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_prepare_native_rows_preserves_measurements_and_source(self):
        source, output = self.root / "native.tif", self.root / "with_rows.tif"
        ds = gdal.GetDriverByName("GTiff").Create(
            str(source), 20, 12, 3, gdal.GDT_Float32
        )
        data = np.arange(240, dtype=np.float32).reshape(12, 20) / 1000
        for i in range(1, 4):
            ds.GetRasterBand(i).WriteArray(data + i / 10)
        ds = None
        original = source.read_bytes()
        self.assertEqual(
            main(["prepare-rows", str(source), "--output", str(output)]), 0
        )
        result = gdal.Open(str(output))
        self.assertEqual(result.RasterCount, 4)
        np.testing.assert_array_equal(
            result.GetRasterBand(4).ReadAsArray(),
            np.repeat(np.arange(1, 13, dtype=np.float32)[:, None], 20, axis=1),
        )
        np.testing.assert_array_equal(result.GetRasterBand(3).ReadAsArray(), data + 0.3)
        self.assertEqual(source.read_bytes(), original)
        with self.assertRaisesRegex(ValueError, "already has"):
            prepare_rows(output, self.root / "twice.tif")
        with self.assertRaisesRegex(ValueError, "new output"):
            prepare_rows(source, source)

    def test_validation_does_not_require_manual_homographies_or_run_stages(self):
        config = create_demo(self.root / "demo")
        folder = Path(config.out_folder)
        config.use_available_homography = False
        config_path = self.root / "validation.json"
        config.save(config_path)
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = main(["--config", str(config_path), "--validate-only"])
        self.assertEqual(code, 0)
        self.assertIn("No registration was performed", stream.getvalue())
        self.assertFalse((folder / "run_metadata.json").exists())

    def test_bad_config_is_a_clear_command_error(self):
        path = self.root / "invalid.json"
        path.write_text('{"vnir_path": 123}')
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            self.assertEqual(main(["--config", str(path), "--validate-only"]), 1)
        self.assertIn("vnir_path must be a path string", stream.getvalue())

    def test_missing_headless_config_and_invalid_port_are_rejected(self):
        for arguments in (["--no-gui"], ["--validate-only"], ["--port", "0"]):
            with (
                self.subTest(arguments=arguments),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(main(arguments), 1)


if __name__ == "__main__":
    unittest.main()
