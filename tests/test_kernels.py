"""Small numerical regressions; no user data or fitted scene registration."""
import os
os.environ.setdefault('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', '1')

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import ants
import joblib
import numpy as np
from osgeo import gdal, osr
gdal.UseExceptions()

from geoharmony.utils.gdalimage import GdalImage
from geoharmony.utils.gdalwriter import to_envi_scale, set_zeros_base_ref, warp_extent_res
from geoharmony.splash.scripts.shape_shifter import shape_shift_mpp_gimg, shift_rows_from_model
from geoharmony.splash.scripts.antspy_registration_onraster import (
    apply_registration_to_band, antspy_vnir_swir_gimg,
)


def image(path, values, transform=(500000, 1, 0, 4750000, 0, -1)):
    values = np.asarray(values, dtype=np.float32)
    ds = gdal.GetDriverByName('ENVI').Create(str(path), values.shape[2], values.shape[1], values.shape[0], gdal.GDT_Float32)
    crs = osr.SpatialReference()
    crs.ImportFromEPSG(32618)
    ds.SetProjection(crs.ExportToWkt())
    ds.SetGeoTransform(transform)
    for index, band in enumerate(values):
        ds.GetRasterBand(index + 1).WriteArray(band)
        ds.GetRasterBand(index + 1).SetDescription(f'Band {index + 1}')
    ds.SetMetadataItem('wavelength', '{500,600,700,800}', 'ENVI')
    ds.FlushCache()
    ds = None
    return GdalImage(str(path))


class KernelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_export_identity_and_scale_drop_exactly_one_band(self):
        values = np.arange(4 * 6 * 7, dtype=np.float32).reshape(4, 6, 7) / 257
        source = image(self.directory / 'source', values)
        original = (source.bands, list(source.band_description), list(source.band_nodata))
        exported = to_envi_scale(source, str(self.directory / 'identity'), None, dtype=np.float32)
        np.testing.assert_array_equal(exported.read(), values[:3])
        self.assertEqual(exported.bands, 3)
        self.assertEqual(exported.dtype_classes['string'], 'Float32')
        self.assertEqual(exported.ds.GetMetadata('ENVI')['wavelength'], '{500,600,700}')
        scaled = to_envi_scale(source, str(self.directory / 'scaled'), [2, 3, 4], dtype=np.float32)
        np.testing.assert_allclose(scaled.read(), values[:3] * np.array([2, 3, 4])[:, None, None], rtol=1e-7)
        self.assertEqual(original, (source.bands, source.band_description, source.band_nodata))
        self.assertEqual(scaled.geotransform, source.geotransform)

    def test_msi_mask_retains_float_reflectance_and_valid_zero(self):
        rows = np.ones((8, 8), np.float32) * 7
        rows[:, :2] = 0
        hs = image(self.directory / 'hs', np.stack([np.zeros_like(rows), rows]))
        values = np.full((2, 8, 8), .237, np.float32)
        values[0, 4, 4] = 0
        msi = image(self.directory / 'msi', values)
        output = set_zeros_base_ref(hs, msi, output_folder=str(self.directory))
        expected = values.copy()
        expected[:, :, :2] = 0
        np.testing.assert_array_equal(output.read(), expected)
        self.assertEqual(output.dtype_classes['string'], 'Float32')

    def test_spectral_wavelengths_survive_row_augmented_intermediate(self):
        from geoharmony.utils.gdalwriter import to_envi
        source = image(self.directory / 'wavelength_source', np.ones((4, 4, 4)))
        source.ds = None
        update = gdal.Open(source.path, gdal.GA_Update)
        update.SetMetadataItem('wavelength', '{500,600,700}', 'ENVI')
        update.FlushCache()
        update = None
        source = GdalImage(source.path)
        intermediate = to_envi(source.read(), source, str(self.directory / 'stage'))
        final = to_envi_scale(intermediate, str(self.directory / 'final'))
        self.assertEqual(final.ds.GetMetadata('ENVI')['wavelength'], '{500,600,700}')

    def test_reprojection_keeps_categorical_rows_and_normalizes_edge_support(self):
        rows = np.zeros((8, 8), np.float32)
        rows[1:7, 1:7] = np.arange(10, 16, dtype=np.float32)[:, None]
        spectra = np.where(rows > 0, .237, 0).astype(np.float32)
        source = image(self.directory / 'source', np.stack([spectra, rows]))
        target = image(self.directory / 'target', np.zeros((1, 8, 8)),
                       transform=(500000.5, 1, 0, 4750000, 0, -1))
        result = warp_extent_res(source, target, 'regrid', 'bilinear', str(self.directory), categorical_last_band=True)
        output = result.read()
        self.assertTrue(set(np.unique(output[-1])).issubset(set(np.unique(rows))))
        np.testing.assert_allclose(output[0][output[-1] > 0], .237, atol=1e-7)
        np.testing.assert_array_equal(output[0][output[-1] == 0], 0)
        self.assertEqual(result.geotransform, target.geotransform)

    def test_invalid_joint_destination_and_negative_indices_do_not_wrap(self):
        class Line:
            def predict(self, x):
                return -3 * np.asarray(x).ravel()
        data = np.arange(25).reshape(5, 5)
        old = data.copy()
        coordinates = np.array([0] * 6 + [4] * 6)
        result = shift_rows_from_model(data, Line(), coordinates, coordinates, 1)
        self.assertEqual(result, (None, None, None))
        np.testing.assert_array_equal(data, old)
        class Flat:
            def predict(self, x):
                return np.zeros(len(x))
        result = shift_rows_from_model(data, Flat(), np.zeros(12, int), np.zeros(12, int), -1)
        self.assertEqual(result, (None, None, None))
        np.testing.assert_array_equal(data, old)

    def test_seeded_shape_serial_parallel_disk_cache_with_ram_outputs(self):
        rng = np.random.default_rng(19)
        plane = rng.integers(0, 8, (32, 64)).astype(np.float32) / 7
        rows = np.broadcast_to(np.arange(1, 33, dtype=np.float32)[:, None], plane.shape)
        source = image(self.directory / 'tiny_hs', np.stack([plane, plane * .4, rows]))
        reference = np.roll(plane * .7, 1, axis=1)
        msi = image(self.directory / 'tiny_msi', reference[None])
        outputs = []
        for workers in (1, 2):
            cache = self.directory / f'models{workers}'
            result, qa = shape_shift_mpp_gimg(source, msi, [0, 1], 0, pixel_shift=2,
                num_threads=workers, random_seed=42, name='same', output_folder='/vsimem', model_folder=str(cache))
            outputs.append(result.read())
            payload = joblib.load(cache / 'results_same.pkl')
            self.assertTrue(payload['complete'])
            self.assertGreater(sum(model is not None for model, _, _ in payload['results']), 0)
            result.unlink()
            qa.unlink()
        np.testing.assert_array_equal(outputs[0], outputs[1])
        np.testing.assert_array_equal(source.read(), np.stack([plane, plane * .4, rows]))
        joblib.dump([], self.directory / 'results_same.pkl')
        with self.assertRaisesRegex(ValueError, 'legacy'):
            shape_shift_mpp_gimg(source, msi, [0, 1], 0, use_available_model=True,
                output_folder=str(self.directory), model_folder=str(self.directory), name='same')

    def affine_path(self, translation=(0, 0)):
        transform = ants.create_ants_transform(transform_type='AffineTransform', dimension=2, translation=translation)
        path = str(self.directory / 'known_affine.mat')
        ants.write_transform(transform, path)
        return path

    def test_float_ants_interpolation_preserves_zeros_and_sub_uint8_precision(self):
        plane = np.full((16, 16), .237123, np.float32)
        plane[7, 7] = 0
        valid = np.ones(plane.shape, bool)
        valid[:, :2] = False
        output = apply_registration_to_band(plane, np.zeros_like(plane), [self.affine_path()], valid)[..., 0]
        np.testing.assert_allclose(output[valid], plane[valid], atol=1e-7)
        np.testing.assert_array_equal(output[~valid], 0)
        translated = apply_registration_to_band(plane, np.zeros_like(plane), [self.affine_path((.5, 0))], valid)[..., 0]
        self.assertAlmostEqual(float(translated[6, 7]), .237123 / 2, places=6)

    def test_elastic_wrapper_keeps_row_ids_and_honors_syN_seed(self):
        yy, xx = np.indices((32, 32))
        plane = ((xx + 2 * yy) / 170).astype(np.float32)
        rows = np.broadcast_to((100 + 3 * yy[:, 0])[:, None], plane.shape).astype(np.float32)
        source = image(self.directory / 'elastic_hs', np.stack([plane, plane * .4, rows]))
        identity = self.affine_path()
        outputs = []
        with patch('geoharmony.splash.scripts.antspy_registration_onraster.ants.registration',
                   return_value={'fwdtransforms': [identity]}) as register:
            for workers in (1, 2):
                folder = self.directory / f'elastic{workers}'
                folder.mkdir()
                output, deformation, quality, mask = antspy_vnir_swir_gimg(source, source, [0], [1],
                    output_folder=str(folder), model_folder=str(self.directory / 'models'), num_threads=workers, random_seed=73)
                outputs.append(output.read())
        self.assertEqual(register.call_args.kwargs['type_of_transform'], 'SyN')
        self.assertEqual(register.call_args.kwargs['random_seed'], 73)
        np.testing.assert_allclose(output.read()[:2], source.read()[:2], atol=1e-7)
        np.testing.assert_array_equal(output.read()[-1], rows)
        np.testing.assert_array_equal(outputs[0], outputs[1])
        self.assertEqual(output.dtype_classes['string'], 'Float32')
        self.assertEqual(deformation.bands, 1)
        self.assertEqual(quality.bands, 1)
        self.assertFalse(mask.any())


if __name__ == '__main__':
    unittest.main()
