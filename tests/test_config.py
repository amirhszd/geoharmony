"""Public configuration and raster-validation contracts on tiny generated data."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from osgeo import gdal, osr

from geoharmony.splash.config import SplashConfig

gdal.UseExceptions()


class SplashConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="splash-config-tests-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self._raster(self.inputs / "vnir.tif", row_band=True)
        self._raster(self.inputs / "swir.tif", row_band=True)
        self._raster(self.inputs / "msi.tif", row_band=False)

    def _raster(self, path, *, row_band, bands=3, rows=16, columns=18,
                with_crs=True, row_values=None, transform=None):
        count = bands + int(row_band)
        raster = gdal.GetDriverByName("GTiff").Create(
            str(path), columns, rows, count, gdal.GDT_Float32,
        )
        raster.SetGeoTransform(transform or (100, 1, 0, 200, 0, -1))
        if with_crs:
            crs = osr.SpatialReference()
            crs.ImportFromEPSG(32618)
            raster.SetProjection(crs.ExportToWkt())
        yy, xx = np.indices((rows, columns))
        for index in range(bands):
            raster.GetRasterBand(index + 1).WriteArray(
                ((index + 1) * (xx + yy + 1)).astype(np.float32),
            )
        if row_band:
            values = row_values if row_values is not None else np.broadcast_to(
                np.arange(1, rows + 1, dtype=np.float32)[:, None], (rows, columns),
            )
            band = raster.GetRasterBand(count)
            band.WriteArray(np.ascontiguousarray(values))
            band.SetDescription("Acquisition row index")
            band.SetMetadataItem("SPLASH_ROLE", "acquisition_row")
        raster.FlushCache()
        raster = None
        return path

    def _data(self, **updates):
        data = {
            "vnir_path": "inputs/vnir.tif",
            "swir_path": "inputs/swir.tif",
            "mica_path": "inputs/msi.tif",
            "out_folder": "outputs",
        }
        data.update(updates)
        return data

    def _config(self, **updates):
        return SplashConfig.from_dict(self._data(**updates), base_dir=self.root)

    def test_relative_paths_are_resolved_from_json_directory(self):
        path = self.root / "config.json"
        path.write_text(json.dumps(self._data()), encoding="utf-8")
        config = SplashConfig.load(path)
        self.assertEqual(config.vnir_path, str(self.inputs / "vnir.tif"))
        self.assertEqual(config.swir_path, str(self.inputs / "swir.tif"))
        self.assertEqual(config.mica_path, str(self.inputs / "msi.tif"))
        self.assertEqual(config.out_folder, str(self.root / "outputs"))
        self.assertIsNone(config.mica_dsm_path)

    def test_valid_inputs_and_distinct_msi_selections_roundtrip(self):
        config = self._config(ss_vnir_mica_band_index=0, ss_swir_mica_band_index=2)
        info = config.validate_inputs()
        self.assertEqual(info["vnir"]["bands"], 4)
        self.assertEqual(info["swir"]["bands"], 4)
        self.assertEqual(info["mica"]["bands"], 3)
        path = self.root / "roundtrip.json"
        config.save(path)
        restored = SplashConfig.load(path)
        self.assertEqual(restored.ss_vnir_mica_band_index, 0)
        self.assertEqual(restored.ss_swir_mica_band_index, 2)
        self.assertEqual(restored.to_dict(), json.loads(json.dumps(config.to_dict())))

    def test_inclusive_ranges_accept_last_spectral_band(self):
        config = self._config(
            ss_vnir_band_range=[2, 2], ss_swir_band_range=[0, 2],
            ereg_vnir_band_range=[0, 2], ereg_swir_band_range=[2, 2],
        )
        config.validate_inputs()
        self.assertEqual(config.ss_vnir_band_range, [2, 2])

    def test_row_band_cannot_be_selected_as_spectral_input(self):
        for key in ("ss_vnir_band_range", "ss_swir_band_range",
                    "ereg_vnir_band_range", "ereg_swir_band_range"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, key):
                    self._config(**{key: [0, 3]}).validate_inputs()
        for key in ("coreg_vnir_band_indices", "coreg_swir_band_indices"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, key):
                    self._config(**{key: [0, 1, 3]}).validate_inputs()

    def test_invalid_row_ids_are_rejected(self):
        original = np.broadcast_to(np.arange(1, 17, dtype=np.float32)[:, None], (16, 18)).copy()
        for bad_value in (0.25, -1, float("nan"), float("inf")):
            with self.subTest(bad_value=bad_value):
                rows = original.copy()
                rows[4, 4] = bad_value
                self._raster(self.inputs / "bad_rows.tif", row_band=True, row_values=rows)
                with self.assertRaisesRegex(ValueError, "acquisition-row"):
                    self._config(vnir_path="inputs/bad_rows.tif").validate_inputs()

    def test_background_zero_ids_are_valid(self):
        rows = np.broadcast_to(np.arange(1, 17, dtype=np.float32)[:, None], (16, 18)).copy()
        rows[0] = 0
        self._raster(self.inputs / "background.tif", row_band=True, row_values=rows)
        self._config(vnir_path="inputs/background.tif").validate_inputs()

    def test_empty_or_short_row_support_is_rejected(self):
        cases = (
            ("empty.tif", 18, np.zeros((16, 18), dtype=np.float32)),
            ("short.tif", 10, None),
        )
        for name, columns, rows in cases:
            with self.subTest(name=name):
                self._raster(self.inputs / name, row_band=True, columns=columns, row_values=rows)
                with self.assertRaisesRegex(ValueError, "at least 11 pixels"):
                    self._config(vnir_path=f"inputs/{name}").validate_inputs()

    def test_missing_crs_and_rotated_grid_are_rejected(self):
        self._raster(self.inputs / "no_crs.tif", row_band=True, with_crs=False)
        with self.assertRaisesRegex(ValueError, "CRS"):
            self._config(vnir_path="inputs/no_crs.tif").validate_inputs()
        self._raster(self.inputs / "rotated.tif", row_band=True,
                     transform=(100, 1, 0.1, 200, 0, -1))
        with self.assertRaisesRegex(ValueError, "north-up"):
            self._config(swir_path="inputs/rotated.tif").validate_inputs()

    def test_reference_band_and_minimum_band_counts(self):
        with self.assertRaisesRegex(ValueError, "ss_swir_mica_band_index"):
            self._config(ss_swir_mica_band_index=3).validate_inputs()
        self._raster(self.inputs / "few_msi.tif", row_band=False, bands=2)
        with self.assertRaisesRegex(ValueError, "mica_path"):
            self._config(mica_path="inputs/few_msi.tif").validate_inputs()

    def test_optional_dsm_requires_one_georeferenced_band(self):
        self._raster(self.inputs / "dsm.tif", row_band=False, bands=1)
        self._config(mica_dsm_path="inputs/dsm.tif").validate_inputs()
        with self.assertRaisesRegex(ValueError, "mica_dsm_path"):
            self._config(mica_dsm_path="inputs/msi.tif").validate_inputs()

    def test_output_directory_is_separate_from_input_directory(self):
        with self.assertRaisesRegex(ValueError, "separate output directory"):
            self._config(out_folder="inputs").validate_inputs()

    def test_invalid_ranges_and_display_indices(self):
        for key in ("ss_vnir_band_range", "ss_swir_band_range",
                    "ereg_vnir_band_range", "ereg_swir_band_range"):
            for value in ([2, 1], [-1, 1], [0, 1.5], [0], [False, 1]):
                with self.subTest(key=key, value=value):
                    with self.assertRaisesRegex(ValueError, key):
                        self._config(**{key: value})
        for value in ([0, 1], [0, 1, -1], [0, 1, True]):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "coreg_vnir_band_indices"):
                    self._config(coreg_vnir_band_indices=value)

    def test_worker_counts_and_integer_controls_are_typed(self):
        for key in ("num_threads", "elastic_workers", "itk_threads"):
            for value in (0, -1, 1.5, True, "2"):
                with self.subTest(key=key, value=value):
                    with self.assertRaisesRegex(ValueError, key):
                        self._config(**{key: value})
        for key in ("pixel_shift_vnir", "pixel_shift_swir", "random_seed"):
            self._config(**{key: 0})
            for value in (-1, False, 3.0):
                with self.subTest(key=key, value=value):
                    with self.assertRaisesRegex(ValueError, key):
                        self._config(**{key: value})

    def test_filter_controls_reject_nonfinite_and_invalid_values(self):
        self._config(kernel_size=1, elastic_sigma_color=2.5)
        for value in (0, 2, 3.0):
            with self.subTest(kernel_size=value):
                with self.assertRaisesRegex(ValueError, "kernel_size"):
                    self._config(kernel_size=value)
        for key in ("elastic_sigma_color", "elastic_sigma_space", "elastic_flow_sigma"):
            for value in (0, -1, float("nan"), float("inf"), True, "3"):
                with self.subTest(key=key, value=value):
                    with self.assertRaisesRegex(ValueError, key):
                        self._config(**{key: value})

    def test_boolean_and_unknown_settings_are_rejected(self):
        for key in ("save_to_memory", "use_available_homography"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, key):
                    self._config(**{key: "false"})
        with self.assertRaisesRegex(ValueError, "Unknown configuration fields"):
            self._config(elastic_steps=8)

    def test_nonstring_paths_are_configuration_errors(self):
        for key, value in (("vnir_path", 123), ("swir_path", True),
                           ("mica_path", ["inputs/msi.tif"]), ("out_folder", 123)):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, key):
                    self._config(**{key: value})

    def test_json_must_be_an_object(self):
        path = self.root / "not_an_object.json"
        path.write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "JSON object"):
            SplashConfig.load(path)


if __name__ == "__main__":
    unittest.main()
