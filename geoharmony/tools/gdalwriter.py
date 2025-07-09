import numpy as np
from osgeo import gdal
import cv2
from tqdm import tqdm
from osgeo import gdal, osr, gdal_array
from geoharmony.tools.gdalimage import GdalImage

def to_envi(array, gdalimage, path):
    """
    Write an array to an ENVI file using GDAL.
    """
    if array.shape[1] != gdalimage.rows or array.shape[2] != gdalimage.cols or array.shape[0] != gdalimage.bands:
        raise ValueError("Array shape does not match metadata: "
                         f"array shape {array.shape}, "
                         f"expected (bands={gdalimage.bands}, rows={gdalimage.rows}, cols={gdalimage.cols})")

    driver = gdal.GetDriverByName('ENVI')
    out_ds = driver.Create(path, gdalimage.cols, gdalimage.rows, gdalimage.bands, gdalimage.dtype_classes['gdal'])

    if out_ds is None:
        raise RuntimeError(f"Could not create dataset: {path}")

    out_ds.SetGeoTransform(gdalimage.geotransform)
    out_ds.SetProjection(gdalimage.projection)

    for band in range(1, gdalimage.bands + 1):
        out_ds.GetRasterBand(band).WriteArray(array[band - 1])
        out_ds.GetRasterBand(band).SetDescription(gdalimage.band_description[band - 1])
        if gdalimage.band_nodata[band - 1] is not None:
            out_ds.GetRasterBand(band).SetNoDataValue(gdalimage.band_nodata[band - 1])

    out_ds.FlushCache()
    out_ds = None
    print(f"ENVI file saved to: {path}")


def remove_bands_from_metadata(gdalimage, bands_to_remove):
    """
    Remove specified bands and their metadata from a gdalmetadata object.
    """
    bands_to_remove = sorted([b - 1 for b in bands_to_remove], reverse=True)
    for b in bands_to_remove:
        if 0 <= b < gdalimage.bands:
            del gdalimage.band_description[b]
            del gdalimage.band_nodata[b]
    gdalimage.bands -= len(bands_to_remove)


def _add_waterfall_band_to_image(array, gdalimage, row_band_array):
    """
    Add a waterfall row band to the array and update metadata.
    """
    if array.shape[1:] != row_band_array.shape[1:]:
        raise ValueError("Row band array must have the shape of the original array.")

    new_array = np.concatenate((array, row_band_array[None, ...]), axis=0)

    gdalimage.bands += 1
    gdalimage.band_description.append("Waterfall Row Band (Created By SPLASH)")
    gdalimage.band_nodata.append(None)
    gdalimage.band_wavelengths.append(None)

    return new_array


def save_image_envi_add_wr(array, gdalimage, row_band_array, output_path):
    updated_array = _add_waterfall_band_to_image(array, gdalimage, row_band_array)
    to_envi(updated_array, gdalimage, output_path)


def save_crop_warp(target_gdalImage, ref_gdalImage, homography_matrix):

    # a sample warped band to get the extents of the image
    sample_warped_band = cv2.warpPerspective(target_gdalImage.read(int(target_gdalImage.bands/2)),
                                             homography_matrix,
                                             (ref_gdalImage.cols, ref_gdalImage.rows))[..., None]
    xmin, xmax, ymin, ymax = 0, sample_warped_band.shape[1], 0, sample_warped_band.shape[0]

    # creating lat and lon rasters and then cropping them accordingly
    lon_raster, lat_raster = ref_gdalImage.create_latlon_raster()

    # croppping it to the extents of the warped band
    lon_raster_crop = lon_raster[ymin:ymax + 1, xmin:xmax + 1]
    lat_raster_crop = lat_raster[ymin:ymax + 1, xmin:xmax + 1]
    del lon_raster, lat_raster

    # warping and cropping the data to the extents
    target_warped_bands = []
    target_arr = target_gdalImage.read()
    for i in tqdm(range(target_gdalImage.bands)):
        target_warped_bands.append(
            cv2.warpPerspective(target_arr[i],
                                homography_matrix,
                                (ref_gdalImage.cols, ref_gdalImage.rows),
                                flags=cv2.INTER_NEAREST)
            [None, ymin:ymax + 1, xmin:xmax + 1]
        )
    target_warped_bands = np.concatenate(target_warped_bands, 0)

    # replicating vnir metadata except the bands and wavelength
    from copy import copy
    target_gdalImage_warped = copy(target_gdalImage)

    # update the metadata
    target_gdalImage_warped.cols = ref_gdalImage.cols
    target_gdalImage_warped.rows = ref_gdalImage.rows
    target_gdalImage_warped.geotransform = ref_gdalImage.geotransform
    target_gdalImage_warped.projection = ref_gdalImage.projection
    target_gdalImage_warped.x_res = ref_gdalImage.x_res
    target_gdalImage_warped.y_res = ref_gdalImage.y_res
    target_gdalImage_warped.xmin = ref_gdalImage.xmin
    target_gdalImage_warped.ymax = ref_gdalImage.ymax
    target_gdalImage_warped.xmax = ref_gdalImage.xmax
    target_gdalImage_warped.ymin = ref_gdalImage.ymin
    target_gdalImage_warped.path = target_gdalImage.path + "_warped"

    #write to envi
    to_envi(target_warped_bands, target_gdalImage_warped, target_gdalImage_warped.path)

    return GdalImage(target_gdalImage_warped.path)
