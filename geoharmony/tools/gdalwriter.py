import numpy as np
from osgeo import gdal
import cv2
from tqdm import tqdm
from osgeo import gdal, osr, gdal_array
from geoharmony.tools.gdalimage import GdalImage
import os

def to_envi(array, gdalimage, path, dtype = None):
    """
    Write an array to an ENVI file using GDAL.

    the default dtype is set to the gdalimage dtype. otherwise it is:
    envi_datatypes = ["numpy.uint8", "numpy.int16", "numpy.int32", "numpy.float32", "numpy.float64", "numpy.complex64",
    "numpy.complex128", "numpy.uint16", "numpy.uint32","numpy.int64", "numpy.uint64"]

    """

    assert dtype is None or dtype in [
        "numpy.uint8", "numpy.int16", "numpy.int32", "numpy.float32", "numpy.float64",
        "numpy.complex64", "numpy.complex128", "numpy.uint16", "numpy.uint32",
        "numpy.int64", "numpy.uint64"], f"dtype must be None or one of the supported ENVI datatypes, got {dtype}"

    if dtype is None:
        dtype = gdalimage.dtype_classes['gdal']

    # if file already exists, remove it
    if os.path.exists(path) and os.path.isfile(path):
        os.remove(path)

    if array.shape[1] != gdalimage.rows or array.shape[2] != gdalimage.cols or array.shape[0] != gdalimage.bands:
        raise ValueError("Array shape does not match metadata: "
                         f"array shape {array.shape}, "
                         f"expected (bands={gdalimage.bands}, rows={gdalimage.rows}, cols={gdalimage.cols})")

    driver = gdal.GetDriverByName('ENVI')
    out_ds = driver.Create(path, gdalimage.cols, gdalimage.rows, gdalimage.bands, dtype)

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

def set_zeros_base_ref(ref_gimg, target_gimg, dtype = "uint16"):

    # Check if the two arrays are the same size
    if ref_gimg.shape[1:] != target_gimg.shape[1:]:
        raise ValueError("arrays are not the same size!")

    # Load the first image
    ref_arr = ref_gimg.read()
    target_arr = target_gimg.read()

    # Find indices where values are zero and setting it to zero
    try:
        zero_indices = np.nanmean(ref_arr,0) == 0
        target_arr[:, zero_indices] = 0
        to_envi(target_arr, target_gimg, target_gimg.path)
    except:
        print("Did not find zeros in the reference image! Moving on.")

    return GdalImage(target_gimg.path)


def warp_extent_res(gdalimage_input: GdalImage,
                    gdalimage_target: GdalImage,
                    extension_string: str,
                    resampling_algorithm: str = "near") -> str:
    """
    Applies the extent and resolution of the target image to the input image.

    Args:
        gdalimage_input (GdalImage): The input image to be warped.
        gdalimage_target (GdalImage): The target image whose extent and resolution will be used.
        extension_string (str): Suffix or extension for the output file.
        resampling_algorithm (str): Resampling algorithm to use. Must be one of:
            "near", "bilinear", "cubic", "cubicspline", "lanczos", "average", "rms", "mode",
            "max", "min", "med", "q1", "q3", "sum". Default is "near".

    Returns:
        str: Path to the output file with updated extent and resolution.
    """

    if resampling_algorithm not in [
        "near", "bilinear", "cubic", "cubicspline", "lanczos", "average", "rms", "mode",
        "max", "min", "med", "q1", "q3", "sum"
    ]:
        raise ValueError(
            "resampling_algorithm must be one of: 'near', 'bilinear', 'cubic', 'cubicspline', 'lanczos', "
            "'average', 'rms', 'mode', 'max', 'min', 'med', 'q1', 'q3', 'sum'"
        )

    gdalimage_input_crs = gdalimage_input.crs
    gdalimage_target_crs = gdalimage_target.crs

    xmin, ymin, xmax, ymax = gdalimage_target.xmin, gdalimage_target.ymin, gdalimage_target.xmax, gdalimage_target.ymax
    x_res_target, y_res_target = gdalimage_target.x_res, gdalimage_target.y_res

    output_filename = gdalimage_input.path + f'_{extension_string}'

    warp_options = gdal.WarpOptions(
        format="ENVI",
        outputBounds=(xmin, ymin, xmax, ymax),
        outputBoundsSRS=gdalimage_target_crs,
        xRes=x_res_target,
        yRes=abs(y_res_target),
        dstSRS=gdalimage_target_crs,
        srcSRS=gdalimage_input_crs,
        resampleAlg=resampling_algorithm
    )
    gdal.Warp(
        destNameOrDestDS=output_filename,
        srcDSOrSrcDSTab=gdalimage_input.path,
        options=warp_options
    )

    print(f"{gdalimage_input.path} warped to {gdalimage_target_crs}: {output_filename}")

    return GdalImage(output_filename)
