"""Optional DSM-derived ground masks on the exact MSI reference grid.

Heights must be in metres. Missing DSM cells remain excluded from the mask.
Filtering affects only the terrain/mask estimate, never sensor measurements.
"""
import copy
import numpy as np
from osgeo import gdal
from scipy.ndimage import minimum_filter, median_filter, binary_dilation, distance_transform_edt
from geoharmony.utils.gdalimage import GdalImage
from geoharmony.utils.gdalwriter import to_envi


def _single_metadata(image, description, nodata=None):
    result = copy.copy(image)
    result.bands = 1
    result.band_description = [description]
    result.band_nodata = [nodata]
    return result


def crop_image_to_extent_gimg(ref_gimg, target_gimg, out_path, out_type='GTiff', output_folder=None):
    """Reproject a DSM to the exact reference grid, including across CRSs.

    Output is ENVI regardless of the legacy ``out_type`` argument. Input height
    units, if declared, must be metres; absent units follow the documented
    metres requirement. NaN nodata prevents uncovered cells becoming ground.
    """
    if target_gimg.bands != 1:
        raise ValueError('DSM must contain exactly one height band.')
    band = target_gimg.ds.GetRasterBand(1)
    unit = (band.GetUnitType() or '').strip().lower()
    if unit and unit not in ('m', 'meter', 'meters', 'metre', 'metres'):
        raise ValueError(f'DSM heights must be in metres; declared unit is {unit!r}. Convert before running SPLASH.')
    values = band.ReadAsArray().astype(np.float32)
    valid = np.isfinite(values) & (band.GetMaskBand().ReadAsArray() > 0)
    nodata = band.GetNoDataValue()
    if nodata is not None:
        valid &= values != nodata
    source = gdal.GetDriverByName('MEM').Create('', target_gimg.cols, target_gimg.rows, 1, gdal.GDT_Float32)
    source.SetProjection(target_gimg.projection)
    source.SetGeoTransform(target_gimg.geotransform)
    source.GetRasterBand(1).WriteArray(np.where(valid, values, np.nan))
    source.GetRasterBand(1).SetNoDataValue(float('nan'))
    destination = gdal.GetDriverByName('MEM').Create('', ref_gimg.cols, ref_gimg.rows, 1, gdal.GDT_Float32)
    destination.SetProjection(ref_gimg.projection)
    destination.SetGeoTransform(ref_gimg.geotransform)
    destination.GetRasterBand(1).Fill(float('nan'))
    destination.GetRasterBand(1).SetNoDataValue(float('nan'))
    gdal.ReprojectImage(source, destination, target_gimg.projection, ref_gimg.projection, gdal.GRA_Bilinear)
    metadata = _single_metadata(target_gimg, 'DSM on MSI grid (metres)', float('nan'))
    metadata.rows, metadata.cols = ref_gimg.rows, ref_gimg.cols
    metadata.shape = (1, metadata.rows, metadata.cols)
    for field in ('projection', 'geotransform', 'crs', 'x_res', 'y_res', 'xmin', 'ymin', 'xmax', 'ymax'):
        setattr(metadata, field, getattr(ref_gimg, field))
    return to_envi(destination.ReadAsArray()[None], metadata, out_path,
                   dtype=np.float32, output_folder=output_folder)


def get_dsm_gimg(dsm_crop_gimg, kernel_size=30, output_folder=None):
    """Estimate above-ground height with spatial median/minimum filters."""
    values = dsm_crop_gimg.read(1).astype(np.float32)
    valid = np.isfinite(values)
    nodata = dsm_crop_gimg.band_nodata[0]
    if nodata is not None:
        valid &= values != nodata
    if not valid.any():
        raise ValueError('DSM has no valid overlap with the MSI reference.')
    # Extend only the auxiliary terrain estimate into missing cells to avoid
    # zero-padding border bias. Original missing cells are excluded again below.
    indices = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = values[tuple(indices)]
    terrain = median_filter(filled, size=29, mode='reflect')
    terrain = minimum_filter(terrain, size=int(kernel_size), mode='reflect')
    terrain = median_filter(terrain, size=5, mode='reflect')
    height = np.clip(values - terrain, 0, 5)
    height[height < .5] = 0
    height[~valid] = np.nan
    metadata = _single_metadata(dsm_crop_gimg, 'Estimated above-ground height (metres)', float('nan'))
    to_envi(height[None], metadata, 'dsm_height', dtype=np.float32, output_folder=output_folder)
    return height[None]


def get_mask_gimg(mica_gimg, dsm_gimg, kernel_size=30, output_folder=None):
    """Return 1 for valid ground and 0 for elevated/missing DSM cells."""
    cropped = crop_image_to_extent_gimg(mica_gimg, dsm_gimg, 'dsm_on_msi', output_folder=output_folder)
    height = get_dsm_gimg(cropped, kernel_size=kernel_size, output_folder=output_folder)[0]
    valid = np.isfinite(height)
    elevated = median_filter((np.nan_to_num(height, nan=0) > 0).astype(np.uint8), size=9, mode='reflect') > 0
    elevated = binary_dilation(elevated, iterations=5)
    mask = (valid & ~elevated).astype(np.uint8)
    metadata = _single_metadata(cropped, 'Ground mask (1 ground, 0 elevated or missing)', None)
    return to_envi(mask[None], metadata, 'dsm_ground_mask', dtype=np.uint8, output_folder=output_folder)
