import numpy as np
import matplotlib.pyplot as plt
import rasterio
from scipy.ndimage import minimum_filter, median_filter
import os
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.ndimage import binary_opening, binary_dilation
from scipy.signal import medfilt2d
import subprocess

from geoharmony.tools.gdalimage import GdalImage


def get_dsm(file_path, kernel_size=30, plot=False):
    # Load DSM data
    with rasterio.open(file_path) as dsm:
        dsm_data = dsm.read(1)
        profile = dsm.profile  # Save the profile for writing output files

    # Apply minimum filter followed by a median filter
    from scipy.ndimage import median_filter
    # performing median filtering in the beginign to get rid of some the artifacts

    filtered_dsm = medfilt2d(dsm_data, kernel_size=29)
    filtered_dsm = minimum_filter(filtered_dsm, size=kernel_size, mode='reflect')
    filtered_dsm = medfilt2d(filtered_dsm, kernel_size=5)

    # Calculate the height map (HM = DSM - DTM)
    height_map = dsm_data - filtered_dsm
    height_map[height_map < 0.5] = 0  # setting negative values and value smaller than 0.5 to zero
    height_map[height_map > 5] = 5  # cap the height at 5 meters

    # Update profile for saving outputs
    profile.update(dtype=rasterio.float32)

    # Create output file names
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    hm_filename = os.path.join(os.path.dirname(file_path),f"{base_name}_height.tif")

    # Write the Height Model (HM) to a new raster file
    with rasterio.open(hm_filename, 'w', **profile) as dst:
        dst.write(height_map.astype(np.float32), 1)

    if plot:
        # Create a plot with 3 images (original DSM, filtered DSM, height map)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # Show original DSM on the first axis
        ax1.imshow(dsm_data, cmap='viridis', vmin=240)
        ax1.set_title("DSM")

        # Show filtered DSM (DTM) on the second axis
        ax2.imshow(filtered_dsm, cmap='viridis', vmin=240)
        ax2.set_title(f"DTM")

        # Show height map (DSM - DTM) on the third axis
        ax3.imshow(height_map, cmap='viridis', vmin=0, vmax=5)
        ax3.set_title(f"Height (DSM - DTM)")

        # Show the plot
        plt.show()
    return height_map




def get_dsm_gimg(dsm_crop_gimg, kernel_size=30):
    # Load DSM data
    with rasterio.open(dsm_crop_gimg.path) as dsm:
        dsm_data = dsm.read(1)
        profile = dsm.profile  # Save the profile for writing output files

    # performing median filtering in the beginign to get rid of some the artifacts
    filtered_dsm = medfilt2d(dsm_data, kernel_size=29)
    filtered_dsm = minimum_filter(filtered_dsm, size=kernel_size, mode='reflect')
    filtered_dsm = medfilt2d(filtered_dsm, kernel_size=5)

    # Calculate the height map (HM = DSM - DTM)
    height_map_arr = dsm_data - filtered_dsm
    height_map_arr[height_map_arr < 0.5] = 0  # setting negative values and value smaller than 0.5 to zero
    height_map_arr[height_map_arr > 5] = 5  # cap the height at 5 meters

    # Update profile for saving outputs
    profile.update(dtype=rasterio.float32)

    # Create output file names
    base_name = os.path.splitext(os.path.basename(dsm_crop_gimg.path))[0]
    hm_filename = os.path.join(os.path.dirname(dsm_crop_gimg.path),f"{base_name}_height.tif")

    # Write the Height Model (HM) to a new raster file
    with rasterio.open(hm_filename, 'w', **profile) as dst:
        dst.write(height_map_arr.astype(np.float32), 1)

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


def get_extent(image_path):
    """
    Get the extent (bounding box) of the image using rasterio.
    """
    with rasterio.open(image_path.split(".hdr")[0] if "hdr" in image_path else image_path) as src:
        bounds = src.bounds
        x_res, y_res = src.get_transform()[1], np.abs(src.get_transform()[-1])
    return bounds, x_res, y_res



##### todo you have a duplicate of this, the function warp_to_target_extent_res does the same replace it as it has been written twice
def crop_image_to_extent(ref_path, to_crop_path, out_path, out_type = 'GTiff'):
    """
    Use GDAL to crop the smaller image based on the extent of the larger image.
    """

    # Get the extent of the larger image
    bounds, x_res, y_res = get_extent(ref_path)
    xmin, ymin, xmax, ymax = bounds.left, bounds.bottom, bounds.right, bounds.top

    # Build the GDAL command to crop the smaller image
    gdal_command = [
        'gdalwarp',
        '-te', str(xmin), str(ymin), str(xmax), str(ymax),
        '-tr', str(x_res), str(y_res),
        '-overwrite',
        '-of', out_type,
        to_crop_path,
        out_path
    ]

    # Run the GDAL command using subprocess
    subprocess.run(gdal_command, check=True)

def crop_image_to_extent_gimg(ref_gimg, target_gimg, out_path, out_type = 'GTiff'):
    """
    Use GDAL to crop the smaller image based on the extent of the larger image.
    """
    # Build the GDAL command to crop the smaller image
    gdal_command = [
        'gdalwarp',
        '-te', str(ref_gimg.xmin), str(ref_gimg.ymin), str(ref_gimg.xmax), str(ref_gimg.ymax),
        '-tr', str(ref_gimg.x_res), str(ref_gimg.y_res),
        '-overwrite',
        '-of', out_type,
        target_gimg.path,
        out_path
    ]

    # Run the GDAL command using subprocess
    subprocess.run(gdal_command, check=True)

    return GdalImage(out_path)



def get_mask_gimg(mica_gimg, dsm_gimg, kernel_size=30):

    # first step is to make sure they are of the same size
    dsm_cropped_path = dsm_gimg.path.replace(".tif", "_cropped.tif")
    dsm_crop_gimg = crop_image_to_extent_gimg(mica_gimg, dsm_gimg, dsm_cropped_path)

    # get the height map
    # height_map_arr = get_dsm(dsm_crop_gimg, kernel_size=kernel_size, plot=False)
    height_map_arr = get_dsm_gimg(dsm_crop_gimg, kernel_size=kernel_size)

    # generate a mask for the non ground points and dilate it
    # dilated_labels = generate_mask(height_map)

    # using the height map as it is for what we have
    dilated_labels = height_map_arr
    dilated_labels[dilated_labels > 0] = 1

    # performing binary opening to get rid of small stuff then dilating to make the map bigger
    dilated_labels = median_filter(dilated_labels, size=9)
    dilated_labels = binary_dilation(dilated_labels, iterations = 5)

    dilated_labels = 1 - dilated_labels # ground is 1, else is 0

    mask_image_path = dsm_gimg.path.replace(".tif", "_mask.tif")

    # write out the labels file based on dsm file
    with rasterio.open(dsm_crop_gimg.path) as src:
        profile = src.profile
        profile.update(dtype=rasterio.uint8)  # Update dtype for the label output
        del profile["nodata"]
        # Save the labels raster
        with rasterio.open(mask_image_path, 'w', **profile) as dst:
            dst.write(dilated_labels.astype(np.uint8), 1)

    return GdalImage(mask_image_path)


# Example usage:
if __name__ == "__main__":

    # making sure the DSM and the micasense data are of the same size

    # Example usage
    micasense_path = '/Volumes/T7/axhcis/Projects/NURI/data/20240719_argetsingerGold/micasense/NURI_Gold_transparent_reflectance_all'  # Replace with the path to your larger image
    dsm_path = '/Volumes/T7/axhcis/Projects/NURI/data/20240719_argetsingerGold/micasense/NURI_Gold_dsm.tif'  # Replace with the path to your smaller image
    get_mask(micasense_path, dsm_path)
