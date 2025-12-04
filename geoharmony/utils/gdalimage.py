import numpy as np
from osgeo import gdal, osr, gdal_array
from osgeo import gdalconst
import os
from rasterio.crs import CRS

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

    def unlink(self):
        gdal.Unlink(self.path)
        gdal.Unlink(self.path + ".hdr")
        gdal.Unlink(self.path + ".aux.xml")

        # # try removing if it exists locally
        # try: os.remove(self.path)
        # except: pass
        # try: os.remove(self.path + ".hdr")
        # except: pass
        # try: os.remove(self.path + ".aux.xml")
        # except: pass

    def read(self, bands: list = None, band_last = False):
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
                band_array.append(self.ds.GetRasterBand(band + 1).ReadAsArray()[None, ...])
            array = np.concatenate(band_array, 0)
        else:
            raise ValueError("bands must be None, an int, or a list/tuple of ints")

        return array if not band_last else np.moveaxis(array, 0, -1)

    def read_display_image(self, bands: list, band_last = False):
        """
        Returns a display-ready uint8 image from the data cube, suitable for visualization in a notebook or GUI.

        The selected bands are arranged in the last dimension of the array, matching the convention used in most non-geospatial libraries.

        Args:
            bands (list): List of band indices (1-based) to include in the display image.

        Returns:
            np.ndarray: uint8 image array with bands in the last dimension.
        """
        arr = self.read(bands, band_last= band_last)
        arr_uint8 = self.to_uint8(arr, band_last = band_last)
        return arr_uint8

    @staticmethod
    def to_uint8(array: np.ndarray, band_last = False) -> np.ndarray:
        array[array<0] = 0
        array = np.nan_to_num(array, 0)
        new_image = np.zeros_like(array)

        nbands = array.shape[-1] if band_last else array.shape[0]
        for i in range(nbands):
            x = array[...,i] if band_last else array[i,...]
            min = np.nanpercentile(x, 2)
            max = np.nanpercentile(x, 98)
            band = (x - min)*255 / (max - min)
            if band_last:
                new_image[...,i] = band
            else:
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
        self.shape = (self.bands, self.rows, self.cols)

        # get image CRS string
        crs_obj = CRS.from_wkt(self.projection)
        crs_code = crs_obj.to_string()  # e.g., 'EPSG:32618'
        self.crs = crs_code if crs_code is not None else None
        if self.crs is None:
            print(f"Warning: Could not determine CRS code from dataset projection: {self.path}")

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
            "string": gdal.GetDataTypeName(self.dtype),
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
            stats = ds_band.GetStatistics(True, False)
            if stats is None:
                stats = (0, 0, 0, 0)
            self.band_stats.append({
                "min": stats[0],
                "max": stats[1],
                "mean": stats[2],
                "std": stats[3]
            })
        self.band_wavelengths_units = self.ds.GetMetadata()['wavelength_units'] if 'wavelength_units' in self.ds.GetMetadata() else None

        return self

    def create_latlon_raster(self):
        # creating lat and lon rasters and then cropping them accordingly
        lon_values = np.linspace(self.xmin, self.xmax, self.cols)
        lat_values = np.linspace(self.ymin, self.ymax, self.rows)
        lon_raster, lat_raster = np.meshgrid(lon_values, lat_values)
        return lon_raster, lat_raster


    def get_wgs84_bounds(self):
        """
        Transforms the image's bounding box to WGS84 (EPSG:4326).
        Returns [[south, west], [north, east]] as latitude-longitude pairs.
        """

        if self.crs == "EPSG:4326":
            # If the image is already in WGS84, return bounds directly
            return [[self.ymin, self.xmin], [self.ymax, self.xmax]]

        # Set up source CRS from the dataset
        src_srs = osr.SpatialReference()
        src_srs.ImportFromEPSG(int(self.crs.split(":")[1]))
        # Define target CRS (WGS84)
        dst_srs = osr.SpatialReference()
        dst_srs.ImportFromEPSG(4326)

        # Create coordinate transformation
        transform = osr.CoordinateTransformation(src_srs, dst_srs)

        # Transform the lower-left corner easting/northing to lat/lon: need to flip the order
        lat_min, lon_min, _ = transform.TransformPoint(self.xmin, self.ymin)
        # Transform the upper-right corner (xmax, ymax)
        lat_max, lon_max, _ = transform.TransformPoint(self.xmax, self.ymax)

        # Return bounds in Leaflet-friendly [lat, lon] order
        return [[lat_min, lon_min], [lat_max, lon_max]]


def gdalimage_to_unit16(gdalimage_input,
                        extension_string = "u16",
                        is_mica = False,
                        output_folder = None):

    output_filename = os.path.join(output_folder, os.path.basename(gdalimage_input.path) + f'_{extension_string}')
    input_dtype_dict = gdalimage_input.dtype_classes

    if "uint" in input_dtype_dict["string"].lower():
        ## work with what we have it would be uint16 at worst
        return gdalimage_input, None

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
        format="ENVI",
    )

    # Run
    gdal.Translate(output_filename, gdalimage_input.ds, options=translate_options)

    return GdalImage(output_filename), scale_coeffs
