import numpy as np
from osgeo import gdal
import cv2
from tqdm import tqdm
from osgeo import gdal, osr, gdal_array
from geoharmony.tools.gdalimage import GdalImage
import os
import subprocess
def remove_envi_file(path):
    """
    Remove an ENVI file and its associated header file.
    """
    header_path = path + ".hdr"
    xml = path + ".aux.xml"
    if os.path.exists(path):
        os.remove(path)
        if os.path.exists(header_path):
            os.remove(header_path)
        if os.path.exists(xml):
            os.remove(xml)


def to_envi_scale(gdalimage, path, coefficients, dtype = None, drop_last_band = True, output_folder = None):
    """
    Write an array to an ENVI file using GDAL.

    the default dtype is set to the gdalimage dtype. pass numpy dtype to change it.

    dtype in [
        np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64,
        np.float32, np.float64, np.complex64, np.complex128
    ]

    """

    if dtype in [
        np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64,
        np.float32, np.float64, np.complex64, np.complex128]:
        dtype_gdal = gdal_array.NumericTypeCodeToGDALTypeCode(dtype)
    elif dtype is None:
        dtype = gdalimage.dtype_classes['gdal']
    else:
        raise TypeError(f"dtype must be None or one of the numpy datatypes, got {dtype}")


    path = os.path.join(output_folder, os.path.basename(path))

    # if file already exists, remove it
    remove_envi_file(path)

    array = gdalimage.read()
    # drop the last band from description if it already exists
    if drop_last_band:
        if hasattr(gdalimage, "band_description") and gdalimage.band_description:
            gdalimage.band_description = gdalimage.band_description[:-1]
        if hasattr(gdalimage, "band_nodata") and gdalimage.band_nodata:
            gdalimage.band_nodata = gdalimage.band_nodata[:-1]
        if hasattr(gdalimage, "band_wavelengths") and gdalimage.band_wavelengths:
            gdalimage.band_wavelengths = gdalimage.band_wavelengths[:-1]
        gdalimage.bands -= 1

    scaled_data = np.zeros((gdalimage.bands - 1 if drop_last_band else gdalimage.bands,
                            gdalimage.rows,
                            gdalimage.cols), dtype = dtype)
    for coefficient, band_n in zip(coefficients, range(gdalimage.bands - 1 if drop_last_band else gdalimage.bands)):
        scaled_data[band_n] = (array[band_n]*coefficient).squeeze()

    with gdal.GetDriverByName('ENVI').Create(path, gdalimage.cols, gdalimage.rows, gdalimage.bands, dtype_gdal) as out_ds:
        out_ds.SetGeoTransform(gdalimage.geotransform)
        out_ds.SetProjection(gdalimage.projection)

        desc = f"Processing {path} [1/1] : "
        with tqdm(total=gdalimage.bands, desc=desc, bar_format="{desc}{percentage:3.0f}% - done.", ncols=120,
                  position=0, leave=True) as pbar:
            for band in range(1, gdalimage.bands + 1 if not drop_last_band else gdalimage.bands):
                out_ds.GetRasterBand(band).WriteArray(array[band - 1])
                out_ds.GetRasterBand(band).SetDescription(gdalimage.band_description[band - 1])
                if len(gdalimage.band_nodata) >= 1:
                    if gdalimage.band_nodata[band - 1] is not None:
                        out_ds.GetRasterBand(band).SetNoDataValue(gdalimage.band_nodata[band - 1])
                pbar.update(1)

        out_ds.FlushCache()

    print(f"ENVI file saved to: {path}")

def to_envi(array, gdalimage, path, dtype = None, output_folder = None):
    """
    Write an array to an ENVI file using GDAL.

    the default dtype is set to the gdalimage dtype. pass numpy dtype to change it.

    dtype in [
        np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64,
        np.float32, np.float64, np.complex64, np.complex128
    ]

    """

    if dtype in [
        np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64,
        np.float32, np.float64, np.complex64, np.complex128]:
        dtype = gdal_array.NumericTypeCodeToGDALTypeCode(dtype)
    elif dtype is None:
        dtype = gdalimage.dtype_classes['gdal']
    else:
        raise TypeError(f"dtype must be None or one of the numpy datatypes, got {dtype}")

    path = os.path.join(output_folder, os.path.basename(path))

    # if file already exists, remove it
    remove_envi_file(path)

    if array.shape[1] != gdalimage.rows or array.shape[2] != gdalimage.cols or array.shape[0] != gdalimage.bands:
        raise ValueError("Array shape does not match metadata: "
                         f"array shape {array.shape}, "
                         f"expected (bands={gdalimage.bands}, rows={gdalimage.rows}, cols={gdalimage.cols})")

    with gdal.GetDriverByName('ENVI').Create(path, gdalimage.cols, gdalimage.rows, gdalimage.bands, dtype) as out_ds:
        out_ds.SetGeoTransform(gdalimage.geotransform)
        out_ds.SetProjection(gdalimage.projection)

        for band in range(1, gdalimage.bands + 1):
            out_ds.GetRasterBand(band).WriteArray(array[band - 1])
            out_ds.GetRasterBand(band).SetDescription(gdalimage.band_description[band - 1])
            if len(gdalimage.band_nodata) >= 1:
                if gdalimage.band_nodata[band - 1] is not None:
                    out_ds.GetRasterBand(band).SetNoDataValue(gdalimage.band_nodata[band - 1])

        out_ds.FlushCache()

    return GdalImage(path)


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

    new_array = np.concatenate((array, row_band_array), axis=0)

    gdalimage.bands += 1
    gdalimage.band_description.append("Waterfall Row Band (Created By SPLASH)")
    gdalimage.band_nodata.append(None)

    return new_array


def save_image_envi_add_wr(array, gdalimage, row_band_array, output_path):
    updated_array = _add_waterfall_band_to_image(array, gdalimage, row_band_array)
    to_envi(updated_array, gdalimage, output_path)


def save_crop_warp(target_gdalImage, ref_gdalImage, homography_matrix, outfolder):

    # a sample warped band to get the extents of the image
    sample_warped_band = cv2.warpPerspective(target_gdalImage.read(int(target_gdalImage.bands/2)),
                                             homography_matrix,
                                             (ref_gdalImage.cols, ref_gdalImage.rows))[..., None]
    xmin, xmax, ymin, ymax = 0, sample_warped_band.shape[1], 0, sample_warped_band.shape[0]

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
    out_gdalimage = to_envi(target_warped_bands, target_gdalImage_warped, target_gdalImage_warped.path, output_folder=outfolder)

    return out_gdalimage

def set_zeros_base_ref(ref_gimg, target_gimg, dtype = np.uint16, output_folder = None):

    # Check if the two arrays are the same size
    if ref_gimg.shape[1:] != target_gimg.shape[1:]:
        raise ValueError("arrays are not the same size!")

    # Load the first image
    ref_arr = ref_gimg.read()
    target_arr = target_gimg.read()
    target_arr = target_arr[None,...] if target_arr.ndim == 2 else target_arr

    # Find indices where values are zero and setting it to zero
    try:
        zero_indices = np.nanmean(ref_arr,0) == 0
        target_arr[:, zero_indices] = 0
        out_gimage = to_envi(target_arr, target_gimg, target_gimg.path, dtype, output_folder=output_folder)
    except Exception as e:
        raise RuntimeError(f"Failed: {e}")

    return out_gimage


def warp_extent_res(gdalimage_input: GdalImage,
                    gdalimage_target: GdalImage,
                    extension_string: str,
                    resampling_algorithm: str = "near",
                    output_folder = None) -> str:
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

    format = "GTiff" if gdalimage_input.path.endswith("tif") else "ENVI"

    if format == "ENVI":
        output_filename = os.path.join(output_folder, os.path.basename(gdalimage_input.path) + f'_{extension_string}')
    else:
        output_filename = os.path.join(output_folder, os.path.basename(gdalimage_input.path).split('.')[0] + f'_{extension_string}')

    # remoev if file already exists
    remove_envi_file(output_filename)

    gdalimage_input_crs = gdalimage_input.crs
    gdalimage_target_crs = gdalimage_target.crs

    xmin, ymin, xmax, ymax = gdalimage_target.xmin, gdalimage_target.ymin, gdalimage_target.xmax, gdalimage_target.ymax
    x_res_target, y_res_target = gdalimage_target.x_res, gdalimage_target.y_res

    # subprocess.run([
    #     "gdalwarp",
    #     "-of", "ENVI",
    #     "-te", str(xmin), str(ymin), str(xmax), str(ymax),
    #     "-te_srs", gdalimage_target_crs,
    #     "-tr", str(x_res_target), str(abs(y_res_target)),
    #     "-t_srs", gdalimage_target_crs,
    #     "-s_srs", gdalimage_input_crs,
    #     "-r", resampling_algorithm,
    #     gdalimage_input.path,
    #     output_filename,
    #     "-overwrite"
    # ], check=True)
    #
    # print(f"{gdalimage_input.path} warped to {gdalimage_target_crs}: {output_filename}")

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

    gdal.Warp(destNameOrDestDS=output_filename, srcDSOrSrcDSTab=gdalimage_input.path, options=warp_options)

    print(f"{gdalimage_input.path} warped to {gdalimage_target_crs}: {output_filename}")

    return GdalImage(output_filename)

