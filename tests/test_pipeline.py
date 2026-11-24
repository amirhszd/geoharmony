"""Generated scenes exercise all three stages with real, small ANTs SyN runs."""
import os
os.environ.setdefault('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', '1')

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from osgeo import gdal, osr

from geoharmony.splash.demo import create_demo
from geoharmony.splash.pipeline import run_splash
from geoharmony.splash.pipeline import manual_reference_identity
from geoharmony.splash.provenance import raster_identity
from geoharmony.rigid_coregisteration.launchers import homography_metadata
from geoharmony.utils.gdalimage import GdalImage
from geoharmony.splash.scripts.nonground_mask_creator import crop_image_to_extent_gimg

gdal.UseExceptions()


def write_dsm(path, reference_path, *, geographic=False, unit='m'):
    reference = gdal.Open(str(reference_path))
    transform = reference.GetGeoTransform()
    projection = reference.GetProjection()
    size = reference.RasterXSize
    if geographic:
        source = osr.SpatialReference(wkt=projection)
        target = osr.SpatialReference()
        target.ImportFromEPSG(4326)
        for crs in (source, target):
            crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        convert = osr.CoordinateTransformation(source, target)
        west, north, _ = convert.TransformPoint(transform[0], transform[3])
        east, south, _ = convert.TransformPoint(transform[0] + size * transform[1], transform[3] + size * transform[5])
        transform = (west, (east-west)/size, 0, north, 0, (south-north)/size)
        projection = target.ExportToWkt()
    values = np.full((size, size), 100, np.float32)
    values[20:32, 20:32] = 102
    values[:, :4] = -9999
    output = gdal.GetDriverByName('GTiff').Create(str(path), size, size, 1, gdal.GDT_Float32)
    output.SetGeoTransform(transform)
    output.SetProjection(projection)
    output.GetRasterBand(1).WriteArray(values)
    output.GetRasterBand(1).SetNoDataValue(-9999)
    output.GetRasterBand(1).SetUnitType(unit)
    output.FlushCache()
    output = None


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='splash-pipeline-test-')
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def check_outputs(self, config, outputs):
        manifest = json.loads((Path(config.out_folder) / 'run_metadata.json').read_text())
        self.assertEqual(manifest['status'], 'complete')
        self.assertEqual(set(manifest['stage_seconds']), {'manual', 'shapeshifter', 'elastic'})
        self.assertTrue(all(seconds >= 0 for seconds in manifest['stage_seconds'].values()))
        reference = gdal.Open(config.vnir_path)
        for sensor in ('vnir', 'swir'):
            image = gdal.Open(outputs[sensor])
            data = image.ReadAsArray()
            footprint = gdal.Open(outputs[sensor + '_valid_mask']).ReadAsArray()
            self.assertEqual(image.RasterCount, 5)
            self.assertEqual(image.GetRasterBand(1).DataType, gdal.GDT_Float32)
            self.assertEqual(image.GetGeoTransform(), reference.GetGeoTransform())
            self.assertEqual(image.GetProjection(), reference.GetProjection())
            self.assertTrue(np.isfinite(data).all())
            self.assertGreater(np.count_nonzero(footprint), 100)
            self.assertTrue(set(np.unique(footprint)).issubset({0, 1}))
            np.testing.assert_array_equal(data[:, footprint == 0], 0)
            # Float reflectance remains in its original scale, including the
            # last spectral channel, which the old exporter left unwritten.
            self.assertGreater(float(data[-1].max()), .1)
            self.assertLess(float(data.max()), 1.0)
            self.assertTrue((Path(config.out_folder) / 'models' / sensor / f'results_{sensor}.pkl').is_file())
        transforms = list((Path(config.out_folder) / 'models' / 'elastic').glob('*'))
        self.assertTrue(any(path.name.endswith('Warp.nii.gz') for path in transforms))
        self.assertFalse((Path(config.out_folder) / '.splash-running').exists())
        return manifest

    def test_three_stage_disk_run_and_saved_homography_reuse(self):
        config = create_demo(self.directory / 'disk')
        # A common exporter declares zero nodata globally even though a zero
        # measurement inside an acquisition row remains part of that footprint.
        dataset = gdal.Open(config.vnir_path, gdal.GA_Update)
        dataset.GetRasterBand(1).SetNoDataValue(0)
        dataset.FlushCache()
        dataset = None
        identities = {sensor: raster_identity(getattr(config, sensor + '_path')) for sensor in ('vnir', 'swir', 'mica')}
        for sensor in ('vnir', 'swir'):
            path = Path(config.out_folder) / 'transformations' / f'homography_{sensor}.json'
            metadata = json.loads(path.read_text())
            metadata.update(homography_metadata(identities[sensor], manual_reference_identity(identities, sensor)))
            path.write_text(json.dumps(metadata))
        outputs = run_splash(config, no_gui=True, open_browser=False)
        self.check_outputs(config, outputs)
        # A zero-valued HS measurement with a valid row ID survives input
        # cleaning, independently of the later intentional median filter.
        cleaned = gdal.Open(str(Path(config.out_folder) / 'intermediates' / 'vnir' / 'vnir_input'))
        self.assertEqual(float(cleaned.GetRasterBand(1).ReadAsArray()[20, 20]), 0)
        self.assertGreater(float(cleaned.GetRasterBand(6).ReadAsArray()[20, 20]), 0)
        matrix_path = Path(config.out_folder) / 'transformations' / 'homography_vnir.npy'
        matrix_bytes = matrix_path.read_bytes()
        second = run_splash(config, no_gui=True, open_browser=False)
        self.check_outputs(config, second)
        self.assertEqual(matrix_path.read_bytes(), matrix_bytes)
        for sensor in identities:
            self.assertEqual(raster_identity(getattr(config, sensor + '_path')), identities[sensor])

    def test_ram_run_with_geographic_dsm_and_persistent_outputs(self):
        config = create_demo(self.directory / 'ram')
        dsm = self.directory / 'ram' / 'data' / 'dsm.tif'
        write_dsm(dsm, config.mica_path, geographic=True)
        config.mica_dsm_path = str(dsm)
        config.save_to_memory = True
        memory_before = set(gdal.ReadDirRecursive('/vsimem') or [])
        outputs = run_splash(config, no_gui=True, open_browser=False)
        manifest = self.check_outputs(config, outputs)
        self.assertIn('dsm', manifest['inputs'])
        self.assertEqual(set(gdal.ReadDirRecursive('/vsimem') or []), memory_before)
        self.assertFalse((Path(config.out_folder) / 'intermediates').exists())
        self.assertTrue(all(Path(path).is_file() for path in outputs.values()))

    def test_homography_and_input_tampering_rejected_before_processing(self):
        for case in ('matrix', 'input'):
            with self.subTest(case=case):
                config = create_demo(self.directory / case)
                if case == 'matrix':
                    path = Path(config.out_folder) / 'transformations' / 'homography_vnir.npy'
                    matrix = np.load(path)
                    matrix[0, 2] += 1
                    np.save(path, matrix)
                else:
                    dataset = gdal.Open(config.vnir_path, gdal.GA_Update)
                    band = dataset.GetRasterBand(1)
                    values = band.ReadAsArray()
                    values[10, 10] += .01
                    band.WriteArray(values)
                    dataset.FlushCache()
                    dataset = None
                with self.assertRaisesRegex(ValueError, 'Saved transformation does not match'):
                    run_splash(config, no_gui=True, open_browser=False)
                manifest = json.loads((Path(config.out_folder) / 'run_metadata.json').read_text())
                self.assertEqual(manifest['status'], 'failed')
                self.assertEqual(manifest['stage_seconds'], {})
                self.assertFalse((Path(config.out_folder) / '.splash-running').exists())

    def test_dsm_reprojection_excludes_missing_and_rejects_feet(self):
        config = create_demo(self.directory / 'dsm')
        dsm = self.directory / 'dsm' / 'data' / 'dsm.tif'
        write_dsm(dsm, config.mica_path, geographic=True)
        result = crop_image_to_extent_gimg(GdalImage(config.mica_path), GdalImage(str(dsm)),
                    str(self.directory / 'cropped'))
        self.assertEqual(result.geotransform, GdalImage(config.mica_path).geotransform)
        self.assertTrue(np.isnan(result.read(1)[:, :3]).all())
        self.assertAlmostEqual(float(result.read(1)[10, 10]), 100, places=4)
        feet = self.directory / 'feet.tif'
        write_dsm(feet, config.mica_path, unit='ft')
        with self.assertRaisesRegex(ValueError, 'metres'):
            crop_image_to_extent_gimg(GdalImage(config.mica_path), GdalImage(str(feet)),
                                      str(self.directory / 'feet_output'))


if __name__ == '__main__':
    unittest.main()
