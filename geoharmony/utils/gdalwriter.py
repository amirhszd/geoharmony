import os

import cv2
import numpy as np
from osgeo import gdal, gdal_array
from tqdm import tqdm

from geoharmony.utils.gdalimage import GdalImage


def remove_envi_file(path):
    """
    Remove an ENVI file and its associated header file.
    """
    for filename in (path, path + ".hdr", path + ".aux.xml"):
        if filename.startswith("/vsimem/"):
            if gdal.VSIStatL(filename) is not None:
                gdal.Unlink(filename)
        elif os.path.isfile(filename):
            os.remove(filename)


def _output_path(path, output_folder):
    return (
        os.path.join(output_folder, os.path.basename(path))
        if output_folder is not None
        else path
    )


def _output_type(dtype, image):
    if dtype is None:
        return image.dtype_classes["gdal"]
    code = gdal_array.NumericTypeCodeToGDALTypeCode(np.dtype(dtype))
    if code is None or code == gdal.GDT_Unknown:
        raise TypeError(f"Unsupported output dtype: {dtype}")
    return code


def _write_metadata(dataset, image, count, coefficients=None):
    dataset.SetGeoTransform(image.geotransform)
    dataset.SetProjection(image.projection)
    # Keep wavelength metadata, trimming the auxiliary row entry if present.
    metadata = (
        (image.ds.GetMetadata("ENVI") or {})
        if image.bands == image.ds.RasterCount
        else {}
    )
    for key in ("wavelength", "fwhm", "bbl"):
        value = metadata.get(key)
        if value:
            values = value.strip("{} ").split(",")
            if len(values) >= count:
                dataset.SetMetadataItem(
                    key, "{" + ",".join(values[:count]) + "}", "ENVI"
                )
            elif len(values) == count - 1:
                # Row-augmented ENVI inputs commonly list wavelengths for
                # spectral bands only. Keep that list through intermediate stages.
                dataset.SetMetadataItem(key, "{" + ",".join(values) + "}", "ENVI")
    for key in ("wavelength_units", "wavelength units"):
        if key in metadata:
            dataset.SetMetadataItem(key, metadata[key], "ENVI")
    for index in range(count):
        band = dataset.GetRasterBand(index + 1)
        if index < len(image.band_description):
            band.SetDescription(image.band_description[index])
        if index < len(image.band_nodata) and image.band_nodata[index] is not None:
            nodata = image.band_nodata[index]
            if coefficients is not None:
                nodata = nodata * coefficients[index]
            band.SetNoDataValue(float(nodata))


def to_envi_scale(
    gdalimage,
    path,
    coefficients=None,
    dtype=None,
    drop_last_band=True,
    output_folder=None,
):
    """Export spectral bands without mutating the source metadata.

    ``coefficients`` multiply stored values back into their original units;
    ``None`` is the identity. The auxiliary acquisition-row band is dropped once.
    """
    count = gdalimage.bands - int(drop_last_band)
    if count < 1:
        raise ValueError("At least one spectral band is required.")
    scales = (
        np.ones(count, dtype=np.float64)
        if coefficients is None
        else np.asarray(coefficients, dtype=np.float64)
    )
    if (
        scales.ndim != 1
        or len(scales) not in (count, gdalimage.bands)
        or not np.isfinite(scales).all()
    ):
        raise ValueError(
            "Restoration coefficients must contain one finite value per output/source band."
        )
    scales = scales[:count]
    code = _output_type(dtype, gdalimage)
    destination = _output_path(path, output_folder)
    if destination == gdalimage.path:
        raise ValueError("Final export must use a different path from its source.")
    remove_envi_file(destination)
    dataset = gdal.GetDriverByName("ENVI").Create(
        destination, gdalimage.cols, gdalimage.rows, count, code
    )
    if dataset is None:
        raise RuntimeError(f"Could not create output: {destination}")
    _write_metadata(dataset, gdalimage, count, scales)
    for index in range(count):
        values = gdalimage.ds.GetRasterBand(index + 1).ReadAsArray()
        dataset.GetRasterBand(index + 1).WriteArray(
            values.astype(np.float64) * scales[index]
        )
    dataset.FlushCache()
    dataset = None
    return GdalImage(destination)


def to_envi(array, gdalimage, path, dtype=None, output_folder=None):
    """Write a band-first array with the source grid and spectral metadata."""
    array = np.asarray(array)
    expected = (gdalimage.bands, gdalimage.rows, gdalimage.cols)
    if array.shape != expected:
        raise ValueError(
            f"Array shape {array.shape} does not match metadata {expected}."
        )
    code = _output_type(dtype, gdalimage)
    destination = _output_path(path, output_folder)
    remove_envi_file(destination)
    dataset = gdal.GetDriverByName("ENVI").Create(
        destination, gdalimage.cols, gdalimage.rows, gdalimage.bands, code
    )
    if dataset is None:
        raise RuntimeError(f"Could not create output: {destination}")
    _write_metadata(dataset, gdalimage, gdalimage.bands)
    for index in range(gdalimage.bands):
        dataset.GetRasterBand(index + 1).WriteArray(array[index])
    dataset.FlushCache()
    dataset = None
    return GdalImage(destination)


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
    sample_warped_band = cv2.warpPerspective(
        target_gdalImage.read(int(target_gdalImage.bands / 2)),
        homography_matrix,
        (ref_gdalImage.cols, ref_gdalImage.rows),
    )[..., None]
    xmin, xmax, ymin, ymax = (
        0,
        sample_warped_band.shape[1],
        0,
        sample_warped_band.shape[0],
    )

    # warping and cropping the data to the extents
    target_warped_bands = []
    target_arr = target_gdalImage.read()
    for i in tqdm(range(target_gdalImage.bands)):
        target_warped_bands.append(
            cv2.warpPerspective(
                target_arr[i],
                homography_matrix,
                (ref_gdalImage.cols, ref_gdalImage.rows),
                flags=cv2.INTER_NEAREST,
            )[None, ymin : ymax + 1, xmin : xmax + 1]
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

    # write to envi
    out_gdalimage = to_envi(
        target_warped_bands,
        target_gdalImage_warped,
        target_gdalImage_warped.path,
        output_folder=outfolder,
    )

    return out_gdalimage


def set_zeros_base_ref(ref_gimg, target_gimg, dtype=None, output_folder=None):

    # Check if the two arrays are the same size
    if ref_gimg.shape[1:] != target_gimg.shape[1:]:
        raise ValueError("arrays are not the same size!")

    # Load the first image
    ref_arr = ref_gimg.read()
    target_arr = target_gimg.read()
    target_arr = target_arr[None, ...] if target_arr.ndim == 2 else target_arr

    # Find indices where values are zero and setting it to zero
    try:
        # The final acquisition-row plane defines HS support. Individual
        # zero-valued spectral measurements inside that footprint stay valid.
        zero_indices = ~np.isfinite(ref_arr[-1]) | (ref_arr[-1] == 0)
        target_arr[:, zero_indices] = 0
        out_gimage = to_envi(
            target_arr,
            target_gimg,
            target_gimg.path,
            dtype,
            output_folder=output_folder,
        )
    except Exception as e:
        raise RuntimeError(f"Failed: {e}")

    return out_gimage


def warp_extent_res(
    gdalimage_input: GdalImage,
    gdalimage_target: GdalImage,
    extension_string: str,
    resampling_algorithm: str = "near",
    output_folder=None,
    categorical_last_band=False,
) -> GdalImage:
    """Reproject onto the exact target grid, with optional categorical row IDs."""
    algorithms = {
        "near": gdal.GRA_NearestNeighbour,
        "bilinear": gdal.GRA_Bilinear,
        "cubic": gdal.GRA_Cubic,
        "cubicspline": gdal.GRA_CubicSpline,
        "lanczos": gdal.GRA_Lanczos,
        "average": gdal.GRA_Average,
        "mode": gdal.GRA_Mode,
        "max": gdal.GRA_Max,
        "min": gdal.GRA_Min,
        "med": gdal.GRA_Med,
        "q1": gdal.GRA_Q1,
        "q3": gdal.GRA_Q3,
        "sum": gdal.GRA_Sum,
        "rms": gdal.GRA_RMS,
    }
    if resampling_algorithm not in algorithms:
        raise ValueError(f"Unsupported resampling algorithm: {resampling_algorithm}")
    basename = os.path.basename(gdalimage_input.path)
    if basename.lower().endswith((".tif", ".tiff")):
        basename = os.path.splitext(basename)[0]
    output_filename = _output_path(basename + "_" + extension_string, output_folder)
    if output_filename == gdalimage_input.path:
        raise ValueError("Reprojection output must differ from input.")
    remove_envi_file(output_filename)
    destination = gdal.GetDriverByName("ENVI").Create(
        output_filename,
        gdalimage_target.cols,
        gdalimage_target.rows,
        gdalimage_input.bands,
        gdal.GDT_Float32
        if categorical_last_band
        else gdalimage_input.dtype_classes["gdal"],
    )
    if destination is None:
        raise RuntimeError(f"Could not create reprojected output: {output_filename}")
    _write_metadata(destination, gdalimage_input, gdalimage_input.bands)
    destination.SetGeoTransform(gdalimage_target.geotransform)
    destination.SetProjection(gdalimage_target.projection)
    if not categorical_last_band:
        code = gdal.ReprojectImage(
            gdalimage_input.ds,
            destination,
            gdalimage_input.projection,
            gdalimage_target.projection,
            algorithms[resampling_algorithm],
        )
        if code != gdal.CE_None:
            raise RuntimeError(f"Reprojection failed: {gdalimage_input.path}")
    else:
        source_rows = gdal.Translate(
            "", gdalimage_input.ds, format="MEM", bandList=[gdalimage_input.bands]
        )
        target_rows = gdal.GetDriverByName("MEM").Create(
            "", destination.RasterXSize, destination.RasterYSize, 1, gdal.GDT_Float32
        )
        target_rows.SetGeoTransform(destination.GetGeoTransform())
        target_rows.SetProjection(destination.GetProjection())
        gdal.ReprojectImage(
            source_rows,
            target_rows,
            source_rows.GetProjection(),
            target_rows.GetProjection(),
            gdal.GRA_NearestNeighbour,
        )
        rows = target_rows.ReadAsArray()
        source_support = source_rows.ReadAsArray() > 0
        destination.GetRasterBand(gdalimage_input.bands).WriteArray(rows)
        for index in range(gdalimage_input.bands - 1):
            # NaN nodata excludes only absent support (and declared nonzero
            # sentinels), so bilinear interpolation preserves measured zeros.
            source_band = gdal.Translate(
                "",
                gdalimage_input.ds,
                format="MEM",
                bandList=[index + 1],
                outputType=gdal.GDT_Float32,
            )
            values = source_band.ReadAsArray()
            valid = source_support & np.isfinite(values)
            nodata = gdalimage_input.band_nodata[index]
            if nodata is not None and nodata != 0:
                valid &= values != nodata
            source_band.GetRasterBand(1).WriteArray(np.where(valid, values, np.nan))
            source_band.GetRasterBand(1).SetNoDataValue(float("nan"))
            target_band = gdal.GetDriverByName("MEM").Create(
                "",
                destination.RasterXSize,
                destination.RasterYSize,
                1,
                gdal.GDT_Float32,
            )
            target_band.SetGeoTransform(destination.GetGeoTransform())
            target_band.SetProjection(destination.GetProjection())
            target_band.GetRasterBand(1).SetNoDataValue(float("nan"))
            target_band.GetRasterBand(1).Fill(float("nan"))
            gdal.ReprojectImage(
                source_band,
                target_band,
                source_band.GetProjection(),
                target_band.GetProjection(),
                algorithms[resampling_algorithm],
            )
            destination.GetRasterBand(index + 1).WriteArray(
                np.where(rows > 0, target_band.ReadAsArray(), 0)
            )
            destination.GetRasterBand(index + 1).DeleteNoDataValue()
    destination.FlushCache()
    destination = None
    return GdalImage(output_filename)
