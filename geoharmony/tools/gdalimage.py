import numpy as np
from osgeo import gdal, osr, gdal_array

envi_datatypes = {
    1: [np.uint8, "numpy.uint8"],
    2: [np.int16, "numpy.int16"],
    3: [np.int32, "numpy.int32"],
    4: [np.float32, "numpy.float32"],
    5: [np.float64, "numpy.float64"],
    6: [np.complex64, "numpy.complex64"],
    9: [np.complex128, "numpy.complex128"],
    12: [np.uint16, "numpy.uint16"],
    13: [np.uint32, "numpy.uint32"],
    14: [np.int64, "numpy.int64"],
    15: [np.uint64, "numpy.uint64"]
}


class GdalImage():
    def __init__(self, path: str):

        #### TODO redo the commenting here, this is a mess

        """
        Open an existing GDAL dataset and store its handle,
        metadata, and data type.

        Parameters
        ----------
        path : str
            Path to the raster file (e.g. ENVI .hdr/.img or GeoTIFF).
        """
        # initialize the parent class gdalmetadata
        self.gdalmetadata(path)

    def read(self, bands: list = None):
        """
        Reads bands from the dataset.
        - If bands is None, loads the entire array.
        - If bands is an int, loads that specific band (1-based index).
        - If bands is a list, loads the specified bands (1-based indices).
        """
        if bands is None:
            array = self.ds.ReadAsArray()
        elif isinstance(bands, int):
            array = self.ds.GetRasterBand(bands).ReadAsArray()
        elif isinstance(bands, (list, tuple)):
            band_array = []
            for band in bands:
                band_array.append(self.ds.GetRasterBand(band).ReadAsArray()[None, ...])
            array = np.concatenate(band_array, 0)
        else:
            raise ValueError("bands must be None, an int, or a list/tuple of ints")
        return array

    def read_display_image(self, bands: list):
        """
        Returns a display-ready uint8 image from the data cube, suitable for visualization in a notebook or GUI.

        The selected bands are arranged in the last dimension of the array, matching the convention used in most non-geospatial libraries.

        Args:
            bands (list): List of band indices (1-based) to include in the display image.

        Returns:
            np.ndarray: uint8 image array with bands in the last dimension.
        """
        arr = self.read(bands)
        arr_uint8 = self.to_uint8(arr)
        return arr_uint8

    @staticmethod
    def to_uint8(array: np.ndarray) -> np.ndarray:
        new_image = np.zeros_like(array)
        for i in range(array.shape[0]):
            x = array[i,...]
            min = np.percentile(x, 2)
            max = np.percentile(x, 98)
            band = (x - min) / (max - min) * 255
            new_image[i,...] = band
        return new_image.astype(np.uint8)

    def gdalmetadata(self, path: str):
        """
        Open an existing GDAL dataset and store its handle,
        metadata, and data type.

        Parameters
        ----------
        path : str
            Path to the raster file (e.g. ENVI .hdr/.img or GeoTIFF).
        """
        ds = gdal.Open(path, gdal.GA_ReadOnly)
        if ds is None:
            raise RuntimeError(f"Could not open dataset: {path}")
        self.path = path
        self.ds = ds

        self.geotransform = ds.GetGeoTransform()
        self.projection = ds.GetProjection()
        self.dtype = ds.GetRasterBand(1).DataType
        self.cols = ds.RasterXSize
        self.rows = ds.RasterYSize
        self.bands = ds.RasterCount

        # get image CRS string
        srs = osr.SpatialReference()
        srs.ImportFromWkt(self.projection)
        srs.AutoIdentifyEPSG()
        self.crs = srs.GetAuthorityCode("PROJCS") or srs.GetAuthorityCode("GEOGCS")
        self.crs = "EPSG:" + self.crs

        # get image x and y pixel size
        self.x_res = self.geotransform[1]
        self.y_res = self.geotransform[5]

        # get extents
        self.xmin = self.geotransform[0]
        self.ymax = self.geotransform[3]
        self.xmax = self.xmin + self.cols * self.x_res
        self.ymin = self.ymax + self.rows * self.y_res

        self.dtype_classes = {
            "gdal": self.dtype,
            "numpy": gdal_array.GDALTypeCodeToNumericTypeCode(self.dtype),
            "string": gdal.GetDataTypeName(self.dtype)
        }

        # grab dataset level metadata
        self.ds_metadata = {}
        for domain in ds.GetMetadataDomainList():
            self.ds_metadata[domain] = ds.GetMetadata(domain)

        # grab band level metadata
        self.band_description = []
        self.band_nodata = []
        self.band_stats = []
        for band in range(1, self.bands + 1):
            ds_band = ds.GetRasterBand(band)
            # band description
            self.band_description.append(ds_band.GetDescription())
            # band nodata
            self.band_nodata.append(ds_band.GetNoDataValue())
            # band statistics
            stats = ds_band.GetStatistics(False, True)
            self.band_stats.append({
                "min": stats[0],
                "max": stats[1],
                "mean": stats[2],
                "std": stats[3]
            })
        self.band_wavelengths_units = self.ds.GetMetadata()['wavelength_units'] if 'wavelength_units' in self.ds.GetMetadata() else None


class gdalwriter():
    def __init__(self, array, metadata):
        self.array = array
        self.metadata = metadata

    def to_envi(self, path):
        """
        Write an array to an ENVI file using GDAL.

        Parameters
        ----------
        array : numpy.ndarray
            The data array to write.
        path : str
            The output file path (should end with .hdr).
        metadata : gdalmetadata
            Metadata object containing geotransform, projection, etc.

        Returns
        -------
        None
        """

        # Check if array shape matches metadata
        if self.array.shape[1] != self.metadata.rows or self.array.shape[2] != self.metadata.cols or self.array.shape[0] != self.metadata.bands:
            raise ValueError("Array shape does not match metadata: "
                             f"array shape {self.array.shape}, "
                             f"expected (bands={self.metadata.bands}, rows={self.metadata.rows}, cols={self.metadata.cols})")

        self.path = path
        driver = gdal.GetDriverByName('ENVI')
        out_ds = driver.Create(self.path, self.metadata.cols, self.metadata.rows, self.metadata.bands, self.metadata.dtype_classes['gdal'])

        if out_ds is None:
            raise RuntimeError(f"Could not create dataset: {self.path}")

        out_ds.SetGeoTransform(self.metadata.geotransform)
        out_ds.SetProjection(self.metadata.projection)

        for band in range(1, self.metadata.bands + 1):
            out_ds.GetRasterBand(band).WriteArray(self.array[band - 1])
            out_ds.GetRasterBand(band).SetDescription(self.metadata.band_description[band - 1])
            if self.metadata.band_nodata[band - 1] is not None:
                out_ds.GetRasterBand(band).SetNoDataValue(self.metadata.band_nodata[band - 1])

        out_ds.FlushCache()
        out_ds = None
        print(f"ENVI file saved to: {self.path}")

    def remove_bands_from_metadata(self, bands_to_remove):
        """
        Remove specified band(s) and their information from a gdalmetadata object.

        Parameters
        ----------
        metadata : gdalmetadata
            The metadata object to update.
        bands_to_remove : list of int
            List of band indices (1-based) to remove.

        Returns
        -------
        None
        """
        # Convert to 0-based indices and sort descending to avoid index shift
        bands_to_remove = sorted([b - 1 for b in bands_to_remove], reverse=True)
        for b in bands_to_remove:
            if 0 <= b < self.metadata.bands:
                del self.metadata.band_description[b]
                del self.metadata.band_nodata[b]
        self.metadata.bands -= len(bands_to_remove)


    def _add_waterfall_band_to_image(self, row_band_array):
        """
        Add a waterfall row band to the existing array and save it as an ENVI file.

        Parameters
        ----------
        row_band_array : numpy.ndarray
            The array containing the waterfall row band data.

        Returns
        -------
        None
        """
        if self.array.shape[1:] != row_band_array.shape[1:]:
            raise ValueError("Row band array must have the shape of the original array.")

        new_array = np.concatenate((self.array, row_band_array[None, ...]), axis=0)

        # Update metadata for new band
        self.metadata.bands += 1
        self.metadata.band_description.append("Waterfall Row Band (Created By SPLASH)")
        self.metadata.band_nodata.append(None)
        self.metadata.band_wavelengths.append(None)

        # Write the new array to an ENVI file
        self.to_envi(new_array)

    def save_image_envi_add_wr(self, row_band_array, output_path):
        self._add_waterfall_band_to_image(row_band_array)
        self.to_envi(output_path)


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
        srcDSOrSrcDSTab=gdalimage_input.ds,
        options=warp_options
    )

    print(f"{gdalimage_input.path} warped to {gdalimage_target_crs}: {output_filename}")

    return GdalImage(output_filename)


def gdalimage_to_unit16(gdalimage_input, extension_string = "u16", is_mica = False):
    output_filename = gdalimage_input.path + f'_{extension_string}'
    input_dtype_dict = gdalimage_input.dtype_classes

    if "uint" in input_dtype_dict["string"].lower():
        ## work with what we have it would be uint16 at worst
        return gdalimage_input

    # get max of all bands
    maxvals = np.array([band_stats["max"] for band_stats in gdalimage_input.band_stats])

    # find the correction factor going to uint16
    scale_coeffs = np.iinfo(np.uint16).max/(maxvals)
    scale_coeffs[np.isinf(scale_coeffs)] = 0
    unscale_coeffs = 1/scale_coeffs
    unscale_coeffs[np.isinf(unscale_coeffs)] = 0

    scale_params = []
    for c, band_stats in enumerate(gdalimage_input.band_stats):
        if is_mica:
            scale_params.append((0, band_stats["max"], 0, band_stats["max"]*10000))  # Mica is scaled to 1000
        else:
            if c == gdalimage_input.bands - 1:  # skip the last band for vnir_swir
                scale_params.append((0, band_stats["max"], 0, band_stats["max"]))
            else:
                scale_params.append((0, band_stats["max"], 0, np.iinfo(np.uint16).max))

    # Use Translate to apply scale and change type to UInt16
    translate_options = gdal.TranslateOptions(
        outputType=gdal.GDT_UInt16,
        scaleParams=scale_params,
        format="ENVI"
    )

    # Run
    gdal.Translate(output_filename, gdalimage_input.ds, options=translate_options)

    return GdalImage(output_filename), scale_coeffs
