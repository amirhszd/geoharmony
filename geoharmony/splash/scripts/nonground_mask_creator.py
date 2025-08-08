import numpy as np
import rasterio
from scipy.ndimage import minimum_filter, median_filter
import os
from sklearn.cluster import MiniBatchKMeans
from scipy.ndimage import binary_dilation
from scipy.signal import medfilt2d
import subprocess
from geoharmony.tools.gdalimage import GdalImage
from osgeo import gdal

from geoharmony.tools.gdalwriter import to_envi


def get_dsm_gimg(dsm_crop_gimg, kernel_size=30, output_folder=None):
    # Load DSM data

    metadata = dsm_crop_gimg.ds
    dsm_data = dsm_crop_gimg.read(1)
    # with rasterio.open(dsm_crop_gimg.path) as dsm:
    #     dsm_data = dsm.read(1)
    #     profile = dsm.profile  # Save the profile for writing output files

    # performing median filtering in the beginign to get rid of some the artifacts
    filtered_dsm = medfilt2d(dsm_data, kernel_size=29)
    filtered_dsm = minimum_filter(filtered_dsm, size=kernel_size, mode='reflect')
    filtered_dsm = medfilt2d(filtered_dsm, kernel_size=5)

    # Calculate the height map (HM = DSM - DTM)
    height_map_arr = dsm_data - filtered_dsm
    height_map_arr[height_map_arr < 0.5] = 0  # setting negative values and value smaller than 0.5 to zero
    height_map_arr[height_map_arr > 5] = 5  # cap the height at 5 meters
    height_map_arr = height_map_arr[None,]
    # Update profile for saving outputs
    # profile.update(dtype=rasterio.float32)

    # Create output file names
    base_name = os.path.splitext(os.path.basename(dsm_crop_gimg.path))[0]
    hm_filename = os.path.join(os.path.dirname(dsm_crop_gimg.path),f"{base_name}_height.tif")
    # # Write the Height Model (HM) to a new raster file
    # with rasterio.open(hm_filename, 'w', **profile) as dst:
    #     dst.write(height_map_arr.astype(np.float32), 1)
    #

    to_envi(height_map_arr.astype(np.float32), dsm_crop_gimg, hm_filename, dtype = np.float32, output_folder = output_folder)

    return height_map_arr

def generate_mask(height_map):
    # ingesting hm_filename and running kmeans to get ground vs non ground points, k = 2
    print("ok")
    hm_reshaped = height_map.reshape(-1, 1)
    kmeans = MiniBatchKMeans(n_clusters=2, random_state=0)
    kmeans.fit(hm_reshaped)
    labels = kmeans.labels_.reshape(height_map.shape)

    # choose the second label as the non ground object since they are not the primary
    nonground = labels == 1

    # choosing the second label and dilating those points to create a bigger mask
    dilated_labels = binary_dilation(nonground, structure=np.ones((10, 10))).astype(nonground.dtype)

    return dilated_labels

def crop_image_to_extent_gimg(ref_gimg, target_gimg, out_path, out_type = 'GTiff', output_folder = None):
    """
    Use GDAL to crop the smaller image based on the extent of the larger image.
    """
    # Build the GDAL command to crop the smaller image
    # gdal_command = [
    #     'gdalwarp',
    #     '-te', str(ref_gimg.xmin), str(ref_gimg.ymin), str(ref_gimg.xmax), str(ref_gimg.ymax),
    #     '-tr', str(ref_gimg.x_res), str(ref_gimg.y_res),
    #     '-overwrite',
    #     '-of', out_type,
    #     target_gimg.path,
    #     out_path
    # ]
    #
    # # Run the GDAL command using subprocess
    # subprocess.run(gdal_command, check=True)
    #
    out_path = os.path.join(output_folder, out_path)
    warp_options = gdal.WarpOptions(
        format=out_type,
        outputBounds=(ref_gimg.xmin, ref_gimg.ymin, ref_gimg.xmax, ref_gimg.ymax),
        xRes=ref_gimg.x_res,
        yRes=abs(ref_gimg.y_res))

    gdal.Warp(destNameOrDestDS=out_path, srcDSOrSrcDSTab=target_gimg.path, options=warp_options)

    return GdalImage(out_path)



def get_mask_gimg(mica_gimg, dsm_gimg, kernel_size=30, output_folder = None):

    # first step is to make sure they are of the same size
    dsm_cropped_path = os.path.basename(dsm_gimg.path).replace(".tif", "_cropped.tif")
    dsm_crop_gimg = crop_image_to_extent_gimg(mica_gimg, dsm_gimg, dsm_cropped_path, output_folder = output_folder)

    # get the height map
    height_map_arr = get_dsm_gimg(dsm_crop_gimg, kernel_size=kernel_size, output_folder = output_folder)

    # using the height map as it is for what we have
    dilated_labels = height_map_arr
    dilated_labels[dilated_labels > 0] = 1

    # performing binary opening to get rid of small stuff then dilating to make the map bigger
    dilated_labels = median_filter(dilated_labels, size=9)
    dilated_labels = binary_dilation(dilated_labels, iterations = 5)

    dilated_labels = 1 - dilated_labels # ground is 1, else is 0

    mask_image_path = dsm_gimg.path.replace(".tif", "_mask.tif")

    # # write out the labels file based on dsm file
    # with rasterio.open(dsm_crop_gimg.path) as src:
    #     profile = src.profile
    #     profile.update(dtype=rasterio.uint8)  # Update dtype for the label output
    #     del profile["nodata"]
    #     # Save the labels raster
    #     with rasterio.open(mask_image_path, 'w', **profile) as dst:
    #         dst.write(dilated_labels.astype(np.uint8), 1)

    out_gimage = to_envi(dilated_labels, dsm_crop_gimg, mask_image_path, dtype = np.uint8, output_folder = output_folder)

    return out_gimage