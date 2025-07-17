from itertools import repeat
import copy
from spectral.io import envi
import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor
from skimage.metrics import normalized_mutual_information
from tqdm import tqdm
import argparse
from multiprocessing import Pool
import rasterio
import os
import torch.nn.functional as F
import torch
from scipy.ndimage import median_filter
import sys
from geoharmony.tools.gdalwriter import to_envi
from .utils import save_image_envi, load_image_envi_fast, load_image_envi, to_uint8
from geoharmony.tools.gdalimage import GdalImage

def shift_rows_from_model(hs_image_copy, model, y_old, x_old, n):
    """
    Shift rows of a hyperspectral image based on a linear regression model.

    Parameters:
    - hs_image_copy (numpy.ndarray): Copy of the original hyperspectral image.
    - model (sklearn.linear_model): Trained linear regression model.
    - x_old (array-like): Old x-coordinates of points to be shifted.
    - y_old (array-like): Old y-coordinates of points to be shifted.
    - n (int): Number of pixels to shift.
    - quality_raster (numpy.ndarray, optional): Quality raster for storing shift quality metrics.

    Returns:
    - hs_image_copy (numpy.ndarray): Shifted hyperspectral image.
    - x_new (numpy.ndarray): New x-coordinates of shifted points.
    - y_new (numpy.ndarray): New y-coordinates of shifted points.
    - quality_raster (numpy.ndarray): Updated quality raster.
    """

    pixel_values = hs_image_copy[y_old, x_old]

    # Get residuals from old values
    y_old_model = model.predict(x_old.reshape(-1, 1))
    y_res = y_old_model - y_old

    # Add residuals to the points to calculate new coordinates
    x_new = x_old.reshape(-1, 1) + n
    y_new_model = model.predict(x_new)
    y_new = np.round(y_new_model - y_res).astype(int)

    if len(y_new) < 10:
        return None, None, None

    # Check if new coordinates exceed image dimensions
    if (x_new.max() >= hs_image_copy.shape[1]) or (y_new.max() >= hs_image_copy.shape[0]):
        x_ins = np.where((x_new < hs_image_copy.shape[1]) & (x_new > 0))[0]
        y_ins = np.where((y_new < hs_image_copy.shape[0]) & (y_new > 0))[0]
        ins_indices = np.array(list(set(x_ins) & set(y_ins)))
        y_new = y_new[ins_indices]
        x_new = x_new.squeeze()[ins_indices]
        hs_image_copy[y_new, x_new] = pixel_values[ins_indices]
    else:
        hs_image_copy[y_new, x_new.squeeze()] = pixel_values

    return hs_image_copy, x_new, y_new


def calculate_mi_patch(hs_patch, mica_patch):
    """
    Calculate mutual information between a hyperspectral image and a Mica image patch.

    Parameters:
    - hs_patch (numpy.ndarray): Copy of the hyperspectral image.
    - mica_patch (numpy.ndarray): Mica image.
    - min_row (int): Minimum row index of the image patch.
    - max_row (int): Maximum row index of the image patch.
    - min_col (int): Minimum column index of the image patch.
    - max_col (int): Maximum column index of the image patch.

    Returns:
    - mi (float): Mutual information value.
    """

    if hs_patch is None:
        return np.nan

    # Convert patches to uint8 and then to int for compatibility with mutual information calculation
    hs_patch = to_uint8(hs_patch).astype(int)
    mica_patch = to_uint8(mica_patch).astype(int)

    # Calculate mutual information between the patches
    mi = normalized_mutual_information(hs_patch, mica_patch)
    return mi



def get_shift_from_mi_patch(hs_image, mica_image, micamask_image, y_old, x_old, n=3):
    """

    Determine pixel shift from mutual information (MI) between a hyperspectral image and a Mica image.

    Parameters:
    - hs_image (numpy.ndarray): Hyperspectral image.
    - mica_image (numpy.ndarray): Mica image.
    - y_old (array-like): Old y-coordinates of points offseted
    - x_old (array-like): Old x-coordinates of points offseted.
    - min_row (int): Minimum row index of the image patch.
    - max_row (int): Maximum row index of the image patch.
    - min_col (int): Minimum column index of the image patch.
    - max_col (int): Maximum column index of the image patch.
    - n (int, optional): Number of pixel shifts to consider.

    Returns:
    - model (sklearn.linear_model): Best linear regression model.
    - best_shift (int): Best pixel shift.
    - max_mi_quality (float): Maximum mutual information quality metric.
    """
    mi_quality_metrics = []  # List to store MI quality metrics for different shifts
    linear_models = []  # List to store linear regression models for different shifts

    # Check if there are enough points for regression
    if (len(x_old) < 5) or (len(y_old) < 5):
        return None, np.nan, np.nan

    # Fit RANSAC regression model to find the best linear fit
    model = RANSACRegressor(LinearRegression(), max_trials=20, residual_threshold=30).fit(x_old.reshape(-1, 1), y_old)

    # lets make sure we have enough number of inliers
    inlier_mask = model.inlier_mask_
    if inlier_mask.sum() < 5:
        return None, np.nan, np.nan

    # Calculate MI before shifting
    patch1 = hs_image[micamask_image]
    patch2 = mica_image[micamask_image]
    mi_before = calculate_mi_patch(patch1,
                                   patch2)

    # Generate pixel shifts
    pixel_shifts = np.arange(-n, +n + 1)

    # Iterate over pixel shifts
    for shift_value in pixel_shifts:
        hs_image_copy = copy.copy(hs_image)

        # Shift rows based on the regression model
        hs_image_copy, x_new, y_new = shift_rows_from_model(hs_image_copy, model, y_old, x_old, shift_value)

        if (hs_image_copy is None) or (x_new is None) or (y_new is None):
            linear_models.append(None)
            mi_quality_metrics.append(np.nan)
            continue

        # Calculate MI after shifting
        mi_after = calculate_mi_patch(hs_image_copy[micamask_image],
                                      mica_image[micamask_image])
        # Calculate MI quality metric
        mi_quality_metric = (mi_after - mi_before) / mi_before
        mi_quality_metrics.append(mi_quality_metric)
        linear_models.append(model)

    # Find the shift with the maximum MI quality metric
    if len(linear_models) > 0:
        argmax = np.argmax(mi_quality_metrics)
        return linear_models[argmax], pixel_shifts[argmax], mi_quality_metrics[argmax]
    else:
        return None, np.nan, np.nan


def medfilt3d(hs_arr_copy, kernel_size=3, use_torch = True):
    hs_arr_final = []
    # we are not messing with the waterfall band

    for band in tqdm(range(hs_arr_copy.shape[2] -  1), desc="performing band wise filtering",
                     position=0,
                     leave=True,):
        # using GPU
        if use_torch:
            hs_arr_final.append(medfilt2d_gpu(hs_arr_copy[..., band], kernel_size=kernel_size)[..., None])
        else:
            hs_arr_final.append(median_filter(hs_arr_copy[..., band], size=kernel_size)[...,None])

    # not messing with the last layer just because
    hs_arr_final.append(hs_arr_copy[...,-1][...,None])
    hs_arr_final = np.concatenate(hs_arr_final, 2)
    return hs_arr_final


def medfilt2d_gpu(image, kernel_size=3):
    """
    Apply a median filter to a 2D image using PyTorch.

    Parameters:
    - image (torch.Tensor): The input image tensor of shape (H, W) or (1, H, W).
    - kernel_size (int): The size of the median filter kernel (default: 3).

    Returns:
    - torch.Tensor: The filtered image tensor.
    """
    assert kernel_size % 2 == 1, "Kernel size must be odd."

    # Ensure the image has the shape (1, H, W)
    assert image.ndim == 2, "Image size 2d"

    # converting to torch.Tensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device == "cpu":
        print("Did not find cuda compatible GPU, running median filtering using torch with CPU. Still faster!")

    image = image.astype(np.float16)
    image = torch.from_numpy(image).float().to(device)
    image = image[None,]

    # Pad the image to handle borders
    pad_size = kernel_size // 2
    padded_image = F.pad(image, (pad_size, pad_size, pad_size, pad_size), mode='reflect')

    # Get all sliding windows of the image
    windows = padded_image.unfold(1, kernel_size, 1).unfold(2, kernel_size, 1)
    windows = windows.contiguous().view(-1, kernel_size * kernel_size)

    # Compute the median for each window
    medians = windows.median(dim=1).values

    # Reshape the result back to the image shape
    filtered_image = medians.view(image.shape[1], image.shape[2]).cpu().numpy()
    filtered_image = filtered_image.astype(np.uint16)

    return filtered_image

def shape_shift_mpp(_filename,
                    mica_filename,
                    hs_bands,
                    mica_band,
                    pixel_shift = 3, kernel_size = 3,
                    hs_waterfall_rows_band_number = -1,
                    use_torch = True,
                    num_threads = None,
                    mica_mask_filename = None):

    if num_threads is None:
        num_threads = os.cpu_count()

    # Load the hyperspectral and Mica images
    hs_arr, hs_profile = load_image_envi_fast(hs_filename)
    mica_arr, mica_profile = load_image_envi_fast(mica_filename)
    mica_mask_arr, mica_mask_profile = load_image_envi_fast(mica_mask_filename)

    # # Convert arrays to float16 to speed up processing
    # hs_arr = hs_arr.astype(np.float16)
    # mica_arr = mica_arr.astype(np.float16)

    # Extract georectified rows and the original array
    waterfall_rows = hs_arr[..., hs_waterfall_rows_band_number].squeeze()
    hs_arr_shapeshifted = copy.copy(hs_arr)

    # Generate image for hyperspectral bands
    hs_image = np.mean(hs_arr[..., hs_bands], axis=2)
    mica_image = mica_arr[..., mica_band].squeeze()  # Grabbing the last band in Mica
    mica_mask_arr = mica_mask_arr.squeeze().astype(bool)

    # List to save quality metrics
    quality_metrics = []

    def generate_index_chunks(total_length, chunk_size):
        return [np.arange(i, min(i + chunk_size, total_length)) for i in range(0, total_length, chunk_size)]

    row_value_chunks = generate_index_chunks(int(np.max(waterfall_rows)) + 1, num_threads)
    # row_value_chunks = generate_index_chunks(50 + 1, num_threads)

    pbar = tqdm(total = np.max(waterfall_rows),
                position=0,
                leave=True,
                desc = "Finding SS transformations...")
    results = []
    y_olds_all = []
    x_olds_all = []

    # Process each chunk of rows
    for row_value_chunk in row_value_chunks:
        hs_images, mica_images, micamask_images, y_olds_offset, x_olds_offset = [], [], [], [], []
        for row_value in row_value_chunk:

            y_old, x_old = np.where(waterfall_rows == row_value)

            if (row_value == 0) or ((len(x_old) < 5) or (len(y_old) < 5)):
                continue

            # grab the boundary box
            min_row, min_col = np.min(y_old), np.min(x_old)
            max_row, max_col = np.max(y_old), np.max(x_old)

            hs_images.append(hs_image[min_row:max_row + 1, min_col:max_col+1])
            mica_images.append(mica_image[min_row:max_row + 1, min_col:max_col+1])
            micamask_images.append(mica_mask_arr[min_row:max_row + 1, min_col:max_col + 1])
            y_olds_offset.append(y_old - y_old.min())
            x_olds_offset.append(x_old - x_old.min())
            y_olds_all.append(y_old)
            x_olds_all.append(x_old)


        if len(hs_images) == 0:
            continue

        args_list = list(zip(hs_images, mica_images, micamask_images, y_olds_offset, x_olds_offset, repeat(pixel_shift)))
        with Pool(num_threads) as pool:
            results.extend(pool.starmap(get_shift_from_mi_patch, args_list))

        pbar.update(num_threads)

    pbar = tqdm(total = len(results),
                position=0,
                leave=True,
                desc = "Applying SS transformation to all bands.")
    for c, result in enumerate(results):
        linear_model, shift, quality_metric = result
        if linear_model is not None:
            hs_arr_shapeshifted, _, _ = shift_rows_from_model(hs_arr_shapeshifted,
                                                                      linear_model,
                                                                      y_olds_all[c],
                                                                      x_olds_all[c],
                                                                      shift)
        pbar.update(1)

    print(f"Overall relative improvement: {np.nansum(quality_metrics):.5f}.")

    # pefroming median filtering to smooth out the old values
    hs_arr_shapeshifted = medfilt3d(hs_arr_shapeshifted, kernel_size=kernel_size, use_torch = use_torch )

    ss_qa_filename = hs_filename.replace(".hdr", "_ss_qa.hdr")
    ss_filename = hs_filename.replace(".hdr", "_ss.hdr")

    # Save the quality assurance image
    quality_raster_metadata = copy.copy(hs_profile)
    quality_raster_metadata["bands"] = "1"
    quality_raster_metadata["band names"] = "Error Band"
    del quality_raster_metadata["wavelength"]
    quality_raster = np.mean(np.abs(hs_arr_shapeshifted - hs_arr), axis = 2)
    # clipping quality raster to 2 and 98 percentile
    to_uint8 = lambda x: ((x - np.min(x)) / (np.max(x) - np.min(x)) * 255)
    quality_raster = to_uint8(quality_raster).astype("uint8")
    save_image_envi(quality_raster, quality_raster_metadata, ss_qa_filename, dtype = "uint8", ext="")
    del quality_raster

    # Save the SS image
    save_image_envi(hs_arr_shapeshifted, hs_profile, ss_filename, dtype = "uint16",ext="")

    return ss_filename, ss_qa_filename





def shape_shift_mpp_gimg(hs_gimg,
                         mica_gimg,
                         hs_bands,
                         mica_band,
                         pixel_shift = 3, kernel_size = 3,
                         hs_waterfall_rows_band_number = -1,
                         use_torch = True,
                         num_threads = None,
                         mica_mask_gimg = None):

    if num_threads is None:
        num_threads = os.cpu_count()

    # Load the hyperspectral and Mica images
    # hs_arr, hs_profile = load_image_envi_fast(hs_gimg)
    # mica_arr, mica_profile = load_image_envi_fast(mica_gimg)
    hs_arr, hs_profile = hs_gimg.read(band_last = True), hs_gimg.ds
    mica_arr, mica_profile = mica_gimg.read(band_last = True), mica_gimg.ds
    if mica_mask_gimg is None:
        # mica_mask_arr, mica_mask_profile = load_image_envi_fast(mica_mask_filename)
        mica_mask_arr, mica_mask_profile  = mica_mask_gimg.read(band_last = True), mica_mask_gimg.ds

    # # Convert arrays to float16 to speed up processing
    # hs_arr = hs_arr.astype(np.float16)
    # mica_arr = mica_arr.astype(np.float16)

    # Extract georectified rows and the original array
    waterfall_rows = hs_arr[..., hs_waterfall_rows_band_number].squeeze()
    hs_arr_shapeshifted = copy.copy(hs_arr)

    # Generate image for hyperspectral bands
    hs_image = np.mean(hs_arr[..., hs_bands], axis=2)
    mica_image = mica_arr[..., mica_band].squeeze()  # Grabbing the last band in Mica
    mica_mask_arr = mica_mask_arr.squeeze().astype(bool)

    # List to save quality metrics
    quality_metrics = []

    def generate_index_chunks(total_length, chunk_size):
        return [np.arange(i, min(i + chunk_size, total_length)) for i in range(0, total_length, chunk_size)]

    row_value_chunks = generate_index_chunks(int(np.max(waterfall_rows)) + 1, num_threads)
    # row_value_chunks = generate_index_chunks(50 + 1, num_threads)

    pbar = tqdm(total = np.max(waterfall_rows),
                position=0,
                leave=True,
                desc = "Finding SS transformations...")
    results = []
    y_olds_all = []
    x_olds_all = []

    # Process each chunk of rows
    for row_value_chunk in row_value_chunks:
        hs_images, mica_images, micamask_images, y_olds_offset, x_olds_offset = [], [], [], [], []
        for row_value in row_value_chunk:

            y_old, x_old = np.where(waterfall_rows == row_value)

            if (row_value == 0) or ((len(x_old) < 5) or (len(y_old) < 5)):
                continue

            # grab the boundary box
            min_row, min_col = np.min(y_old), np.min(x_old)
            max_row, max_col = np.max(y_old), np.max(x_old)

            hs_images.append(hs_image[min_row:max_row + 1, min_col:max_col+1])
            mica_images.append(mica_image[min_row:max_row + 1, min_col:max_col+1])
            micamask_images.append(mica_mask_arr[min_row:max_row + 1, min_col:max_col + 1])
            y_olds_offset.append(y_old - y_old.min())
            x_olds_offset.append(x_old - x_old.min())
            y_olds_all.append(y_old)
            x_olds_all.append(x_old)


        if len(hs_images) == 0:
            continue

        args_list = list(zip(hs_images, mica_images, micamask_images, y_olds_offset, x_olds_offset, repeat(pixel_shift)))
        with Pool(num_threads) as pool:
            results.extend(pool.starmap(get_shift_from_mi_patch, args_list))

        pbar.update(num_threads)

    pbar = tqdm(total = len(results),
                position=0,
                leave=True,
                desc = "Applying SS transformation to all bands.")
    for c, result in enumerate(results):
        linear_model, shift, quality_metric = result
        if linear_model is not None:
            hs_arr_shapeshifted, _, _ = shift_rows_from_model(hs_arr_shapeshifted,
                                                                      linear_model,
                                                                      y_olds_all[c],
                                                                      x_olds_all[c],
                                                                      shift)
        pbar.update(1)

    print(f"Overall relative improvement: {np.nansum(quality_metrics):.5f}.")

    # pefroming median filtering to smooth out the old values
    hs_arr_shapeshifted = medfilt3d(hs_arr_shapeshifted, kernel_size=kernel_size, use_torch = use_torch )

    ss_qa_filename = hs_gimg.path + "_ss_qa"
    ss_filename = hs_gimg.path + "_ss"

    # Save the quality assurance image
    # quality_raster_metadata = copy.copy(hs_profile)
    # quality_raster_metadata["bands"] = "1"
    # quality_raster_metadata["band names"] = "Error Band"
    # del quality_raster_metadata["wavelength"]
    # quality_raster = np.mean(np.abs(hs_arr_shapeshifted - hs_arr), axis = 2)
    # # clipping quality raster to 2 and 98 percentile
    # to_uint8 = lambda x: ((x - np.min(x)) / (np.max(x) - np.min(x)) * 255)
    # quality_raster = to_uint8(quality_raster).astype("uint8")
    # save_image_envi(quality_raster, quality_raster_metadata, ss_qa_filename, dtype = "uint8", ext="")

    # for quality raster
    quality_raster_metadata = copy.copy(hs_profile)
    quality_raster_metadata.bands = 1
    quality_raster_metadata.band_description = ["Error Band"]
    quality_raster_metadata.band_nodata = []
    quality_raster_metadata.band_stats = []
    quality_raster = np.mean(np.abs(hs_arr_shapeshifted - hs_arr), axis = 2)

    # clipping quality raster to 2 and 98 percentile
    to_uint8 = lambda x: ((x - np.min(x)) / (np.max(x) - np.min(x)) * 255)
    quality_raster = to_uint8(quality_raster).astype("uint8")
    to_envi(np.moveaxis(quality_raster, 2, 0),
            quality_raster_metadata,
            ss_qa_filename,
            dtype = "numpy.uint8")
    # save_image_envi(quality_raster, quality_raster_metadata, ss_qa_filename, dtype = "uint8", ext="")
    del quality_raster

    # Save the SS image
    # save_image_envi(hs_arr_shapeshifted, hs_profile, ss_filename, dtype = "uint16",ext="")
    to_envi(np.moveaxis(hs_arr_shapeshifted, 2, 0),
            hs_profile,
            ss_filename)

    return GdalImage(ss_filename), GdalImage(ss_qa_filename)



if __name__ == "__main__":

    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='Process some integers.')
        parser.add_argument('--hs_filename', type=str, help='Hyperspectral HDR filename. The data must include georectified rows and indices using Headwalls proprietary software on the last two bands.')
        parser.add_argument('--mica_filename', type=str, help='Mica HDR filename. The data must be downsampled and coregistered using the coregister.py code; otherwise, this code will fail.')
        parser.add_argument('--pixel_shift', type=int, default=3, help='Number of pixels to shift.')
        parser.add_argument('--kernel_size', type=int, default=3, help='Size of the kernel.')
        parser.add_argument('--use_torch', type=bool, default=True,
                            help='Enable Torch median filtering, will use CUDA GPU if available; otherwise CPU is used. (default: True).')
        parser.add_argument('--num_threads', type=int, default=os.cpu_count(),
                            help='Number of threads to use for parallel processing. Defaults to the number of CPU cores available.')

        args = parser.parse_args()
        hs_filename = args.hs_filename
        mica_filename = args.mica_filename
        pixel_shift = args.pixel_shift
        kernel_size = args.kernel_size
        use_torch = args.use_torch
        num_threads = args.num_threads

    else:
        # debug
        mica_filename = "/Volumes/T7/axhcis/Projects/NURI/NURI_micasense_1133_transparent_mosaic_stacked_warped_cropped.hdr"
        hs_filename = "/Volumes/T7/axhcis/Projects/NURI/raw_1504_nuc_or_plusindices3_warped.hdr"
        mica_dsm_filename = "/Volumes/T7/axhcis/Projects/NURI/data/20240719_argetsingerGold/micasense/NURI_Gold_dsm.tif"

    shape_shift_mpp(hs_filename,
                    mica_filename,
                    mica_dsm_mask,
                    pixel_shift = 3, kernel_size=3, use_torch=True, num_threads=8)