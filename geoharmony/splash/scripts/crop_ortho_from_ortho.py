from spectral.io import envi
import numpy as np
import numpy.ma as ma
import os
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import minmax_scale
import matplotlib.cm as cm
import matplotlib.widgets as widgets
from scipy import interpolate
import pyelastix
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import SSDMetric, CCMetric
import dipy.align.imwarp as imwarp
from dipy.data import get_fnames
from dipy.io.image import load_nifti_data
from dipy.segment.mask import median_otsu
from dipy.viz import regtools
from scipy.ndimage import gaussian_filter
import matplotlib
from scipy.stats import pearsonr
import cmocean
import cv2

def load_images_envi(mica_path, swir_path):

    mica_ds = envi.open(mica_path)
    mica_profile = mica_ds.metadata
    mica_wavelengths = [475,560,634,668,717]
    mica_arr = mica_ds.load()

    swir_ds = envi.open(swir_path)
    swir_profile = swir_ds.metadata
    try:
        swir_wavelengths = swir_profile["wavelength"]
        swir_wavelengths = np.array([float(i) for i in swir_wavelengths])
    except:
        swir_wavelengths = swir_profile["band names"]
        swir_wavelengths = np.array([float(i.split(" ")[0]) for i in swir_wavelengths])

    swir_arr = swir_ds.load()

    return (mica_arr, mica_profile, mica_wavelengths ), (swir_arr, swir_profile, swir_wavelengths)

def save_image_envi(mica_arr, mica_profile, mica_path, swir_profile, name = None):

    if name:
        output_name = mica_path.replace(".hdr", f"_{name}.hdr")
    else:
        output_name = mica_path.replace(".hdr", "_cropped.hdr")

    # replicating swir metadata except for the extents
    mica_profile["map info"] = swir_profile["map info"]
    envi.save_image(output_name, mica_arr, metadata=mica_profile, force=True)
    print("image saved to: " + output_name)

    return output_name

def create_lat_lon_rasters(profile):
    # creating lat and lon rasters and then cropping them accordingly
    lon_values = np.linspace(float(profile["map info"][3]),
                             float(profile["map info"][3]) + (int(profile["samples"])) * float(profile["map info"][5]),
                             int(profile["samples"]))
    lat_values = np.linspace(float(profile["map info"][4]),
                             float(profile["map info"][4]) - (int(profile["lines"])) * float(profile["map info"][6]),
                             int(profile["lines"]))
    lon_raster, lat_raster = np.meshgrid(lon_values, lat_values)

    return lon_raster, lat_raster

def main(swir_hdr, mica_hdr, name = None):

    # load images envi
    (mica_arr, mica_profile, mica_wavelengths),\
        (swir_arr, swir_profile, swir_wavelengths) = load_images_envi(mica_hdr, swir_hdr)

    # loading the extents of mica
    lon_raster_mica, lat_raster_mica = create_lat_lon_rasters(mica_profile)
    lon_raster_swir, lat_raster_swir = create_lat_lon_rasters(swir_profile)

    # cropping out mica based on swir
    swir_lon_min_max = [np.min(lon_raster_swir), np.max(lon_raster_swir)]
    swir_lat_min_max = [np.min(lat_raster_swir), np.max(lat_raster_swir)]
    mica_lon_cropped = (lon_raster_mica >= swir_lon_min_max[0]) & (lon_raster_mica <= swir_lon_min_max[1])
    mica_lat_cropped = (lat_raster_mica >= swir_lat_min_max[0]) & (lat_raster_mica <= swir_lat_min_max[1])
    mica_extent_bool = np.logical_and(mica_lat_cropped,mica_lon_cropped)
    indices = np.where(mica_extent_bool)
    ymin, ymax, xmin, xmax = indices[0].min(), indices[0].max(), indices[1].min(), indices[1].max()
    mica_arr_cropped = mica_arr[ymin:ymax + 1,xmin:xmax + 1]

    # put where things are zero
    mica_arr_cropped[swir_arr[...,0].squeeze() == 0] = 0

    # save out the new mica array based on the swir array
    output_hdr = save_image_envi(mica_arr_cropped, mica_profile, mica_hdr, swir_profile, name)

    return output_hdr


if __name__ == "__main__":

    mica_hdr = "/Volumes/T7/axhcis/Projects/NURI/data/20210723_tait_labsphere/1133/Micasense/NURI_micasense_1133_transparent_mosaic_stacked_warped.hdr"
    swir_hdr = "/Volumes/T7/axhcis/Projects/NURI/data/20210723_tait_labsphere/1133/SWIR/raw_1504_nuc_or_plusindices3_warped.hdr"
    main(swir_hdr, mica_hdr)







