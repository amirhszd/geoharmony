"""Exercise local Dash interfaces without a browser, server, or registration."""
from dataclasses import fields
import json
from pathlib import Path
from queue import Empty, Queue
import tempfile
import unittest

import numpy as np
from osgeo import gdal, osr

from geoharmony.rigid_coregisteration.get_points_app import make_dash_app
from geoharmony.rigid_coregisteration.show_warped_app import make_blend_app
from geoharmony.rigid_coregisteration.splash_app import make_splash_app
from geoharmony.splash.config import SplashConfig


class BrowserInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gdal.UseExceptions()
        cls.temporary = tempfile.TemporaryDirectory(prefix="splash-ui-")
        cls.root = Path(cls.temporary.name)
        inputs = cls.root / "inputs"
        inputs.mkdir()
        cls.paths = {sensor: inputs / f"{sensor}.tif" for sensor in ("vnir", "swir", "mica", "dsm")}
        for sensor, bands, rows in (("vnir", 4, True), ("swir", 3, True), ("mica", 5, False), ("dsm", 1, False)):
            cls._write_raster(cls.paths[sensor], bands, rows)
        cls.initial = {
            "vnir_path": str(cls.paths["vnir"]),
            "swir_path": str(cls.paths["swir"]),
            "mica_path": str(cls.paths["mica"]),
            "mica_dsm_path": str(cls.paths["dsm"]),
            "out_folder": str(cls.root / "results"),
            "coreg_vnir_band_indices": [3, 1, 0],
            "coreg_swir_band_indices": [2, 0, 1],
            "coreg_mica_band_indices": [2, 1, 0],
            "ss_vnir_band_range": [1, 3],
            "ss_swir_band_range": [0, 2],
            "ss_vnir_mica_band_index": 3,
            "ss_swir_mica_band_index": 4,
            "pixel_shift_vnir": 2,
            "pixel_shift_swir": 5,
            "kernel_size": 1,
            "ereg_vnir_band_range": [1, 2],
            "ereg_swir_band_range": [0, 1],
            "elastic_kernel": 9,
            "elastic_sigma_color": 12.5,
            "elastic_sigma_space": 7.5,
            "elastic_flow_sigma": 2.5,
            "num_threads": 1,
            "elastic_workers": 1,
            "itk_threads": 1,
            "random_seed": 17,
            "save_to_memory": True,
            "use_available_homography": True,
        }

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @staticmethod
    def _write_raster(path, spectral_bands, acquisition_rows):
        width, height = 16, 12
        ds = gdal.GetDriverByName("GTiff").Create(str(path), width, height, spectral_bands + int(acquisition_rows), gdal.GDT_Float32)
        ds.SetGeoTransform((500000, 0.25, 0, 4300000, 0, -0.25))
        crs = osr.SpatialReference()
        if crs.ImportFromEPSG(32618) != 0:
            raise RuntimeError("Could not resolve the fixture CRS through the active PROJ installation")
        ds.SetProjection(crs.ExportToWkt())
        values = np.linspace(0.05, 0.95, width * height, dtype=np.float32).reshape(height, width)
        for index in range(spectral_bands):
            band = ds.GetRasterBand(index + 1)
            band.WriteArray(values + index * 0.01)
            band.SetDescription(f"Spectral channel {index}: {450 + 50 * index} nm")
        if acquisition_rows:
            band = ds.GetRasterBand(spectral_bands + 1)
            rows = np.repeat(np.arange(1, height + 1, dtype=np.float32)[:, None], width, axis=1)
            band.WriteArray(rows)
            if not np.array_equal(band.ReadAsArray(), rows):
                raise RuntimeError("Fixture acquisition-row IDs did not survive the raster write")
            band.SetDescription("Acquisition row ID")
        ds.FlushCache()
        ds = None

    def _settings(self):
        queue = Queue()
        app = make_splash_app(queue, self.initial)
        response = app.server.test_client().get("/_dash-layout")
        self.assertEqual(response.status_code, 200)
        controls = {}

        def visit(value):
            if isinstance(value, dict):
                props = value.get("props", {})
                if "id" in props:
                    controls[props["id"]] = props
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(response.get_json())
        states = {field.name: controls[field.name]["value"] for field in fields(SplashConfig)}
        return app, queue, states

    def _invoke(self, app, output_id, output_property, inputs, states=None, changed=None):
        """POST the actual Dash callback protocol, including callback context."""
        matches = []
        for key, metadata in app.callback_map.items():
            outputs = metadata["output"]
            outputs = outputs if isinstance(outputs, list) else [outputs]
            if any(output.component_id == output_id and output.component_property == output_property for output in outputs):
                matches.append((key, metadata, outputs))
        self.assertEqual(len(matches), 1)
        key, metadata, outputs = matches[0]
        descriptions = [{"id": output.component_id, "property": output.component_property} for output in outputs]
        payload = {
            "output": key,
            "outputs": descriptions if isinstance(metadata["output"], list) else descriptions[0],
            "inputs": [{**item, "value": inputs[item["id"]]} for item in metadata["inputs"]],
            "state": [{**item, "value": states[item["id"]]} for item in metadata["state"]],
            "changedPropIds": [changed or f"{metadata['inputs'][0]['id']}.{metadata['inputs'][0]['property']}"],
        }
        response = app.server.test_client().post("/_dash-update-component", json=payload)
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        return response.get_json()["response"]

    def test_three_apps_render_layout_dependencies_and_local_assets(self):
        image = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
        apps = {
            "settings": make_splash_app(Queue(), self.initial),
            "picker": make_dash_app(image, image, Queue()),
            "blend": make_blend_app(image, image, Queue()),
        }
        for name, app in apps.items():
            with self.subTest(interface=name):
                client = app.server.test_client()
                self.assertEqual(client.get("/").status_code, 200)
                self.assertEqual(client.get("/_dash-layout").status_code, 200)
                dependencies = client.get("/_dash-dependencies")
                self.assertEqual(dependencies.status_code, 200)
                self.assertGreater(len(dependencies.get_json()), 0)
                for asset in ("splash.css", "splashlogo.png"):
                    with client.get(f"/assets/{asset}") as response:
                        self.assertEqual(response.status_code, 200, asset)
                self.assertEqual(app.config.external_stylesheets, [])

    def test_metadata_options_exclude_acquisition_rows_and_keep_all_msi_bands(self):
        app, _, _ = self._settings()
        for sensor, count in (("vnir", 4), ("swir", 3), ("mica", 5)):
            with self.subTest(sensor=sensor):
                response = self._invoke(app, f"coreg_{sensor}_band_indices", "options", {f"{sensor}_path": str(self.paths[sensor])})
                targets = [f"coreg_{sensor}_band_indices"]
                targets += ["ss_vnir_mica_band_index", "ss_swir_mica_band_index"] if sensor == "mica" else [f"ss_{sensor}_band_range", f"ereg_{sensor}_band_range"]
                for target in targets:
                    choices = response[target]["options"]
                    self.assertEqual([choice["value"] for choice in choices], list(range(count)))
                    self.assertEqual(choices[0]["label"], "Band 1 (index 0): Spectral channel 0: 450 nm")
                    self.assertIn(f"Band {count} (index {count-1})", choices[-1]["label"])
                    self.assertFalse(any("Acquisition row" in choice["label"] for choice in choices))

    def test_settings_run_roundtrips_every_field_and_boolean_toggle(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                app, queue, states = self._settings()
                states["save_to_memory"] = ["on"] if enabled else []
                states["use_available_homography"] = ["on"] if enabled else []
                if not enabled:
                    states["mica_dsm_path"] = ""
                response = self._invoke(app, "run-status", "children", {"run": 1, "cancel": 0}, states)
                self.assertIn("Inputs validated", response["run-status"]["children"])
                expected = dict(self.initial, save_to_memory=enabled, use_available_homography=enabled)
                if not enabled:
                    expected["mica_dsm_path"] = None
                expected = json.loads(json.dumps(SplashConfig.from_dict(expected).to_dict()))
                self.assertEqual(queue.get_nowait(), expected)
                self.assertTrue(queue.empty())

    def test_invalid_settings_report_errors_without_queuing_work(self):
        cases = (
            ({"kernel_size": 2}, "kernel_size must be odd"),
            ({"vnir_path": ""}, "vnir_path: select an existing raster"),
            ({"ss_vnir_band_range": [0, 4]}, "excludes the last acquisition-row band"),
            ({"coreg_swir_band_indices": [0, 1, 3]}, "selected band exceeds"),
        )
        for changes, message in cases:
            with self.subTest(changes=changes):
                app, queue, states = self._settings()
                states.update(changes)
                response = self._invoke(app, "run-status", "children", {"run": 1, "cancel": 0}, states)
                self.assertIn(message, response["run-status"]["children"])
                with self.assertRaises(Empty):
                    queue.get_nowait()

    def test_cancel_returns_cancellation_without_validating_inputs(self):
        app, queue, states = self._settings()
        states["vnir_path"] = ""
        response = self._invoke(app, "run-status", "children", {"run": 0, "cancel": 1}, states, "cancel.n_clicks")
        self.assertEqual(response["run-status"]["children"], "Cancelled.")
        self.assertEqual(queue.get_nowait(), {"cancelled": True})

    def test_preview_callback_accepts_metadata_indices_and_reports_invalid_bands(self):
        app, _, _ = self._settings()
        valid = self._invoke(app, "preview-vnir", "figure", {"vnir_path": str(self.paths["vnir"]), "coreg_vnir_band_indices": [3, 1, 0]})
        self.assertEqual(valid["preview-vnir"]["figure"]["layout"]["title"]["text"], "VNIR")
        self.assertEqual(len(valid["preview-vnir"]["figure"]["data"]), 1)
        invalid = self._invoke(app, "preview-vnir", "figure", {"vnir_path": str(self.paths["vnir"]), "coreg_vnir_band_indices": [99, 1, 0]})
        self.assertIn("invalid", invalid["preview-vnir"]["figure"]["layout"]["title"]["text"])

    def test_picker_validation_and_blend_slider_callbacks(self):
        image = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
        picker = make_dash_app(image, image, Queue())
        picker.server.test_client().get("/_dash-layout")
        points = [{"x": 1, "y": 1}, {"x": 8, "y": 1}, {"x": 8, "y": 8}, {"x": 1, "y": 8}]
        valid = self._invoke(picker, "submit-button", "disabled", {"store-left": points, "store-right": points})
        self.assertFalse(valid["submit-button"]["disabled"])
        invalid = self._invoke(picker, "submit-button", "disabled", {"store-left": points[:3], "store-right": points})
        self.assertTrue(invalid["submit-button"]["disabled"])
        self.assertIn("at least 4", invalid["warning-div"]["children"])
        blend = make_blend_app(image, image[::-1], Queue())
        blend.server.test_client().get("/_dash-layout")
        result = self._invoke(blend, "img", "figure", {"alpha-slider": 0.25})
        self.assertEqual(len(result["img"]["figure"]["data"]), 1)


if __name__ == "__main__":
    unittest.main()
