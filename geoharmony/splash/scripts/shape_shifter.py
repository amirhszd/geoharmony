import copy
import hashlib
import os
import tempfile
from multiprocessing import Pool

import joblib
import numpy as np
from scipy.ndimage import median_filter
from skimage.metrics import normalized_mutual_information
from sklearn.linear_model import LinearRegression, RANSACRegressor
from tqdm import tqdm

from geoharmony.utils.gdalwriter import to_envi


def to_uint8(x):
    x = np.asarray(x)
    finite = np.isfinite(x)
    result = np.zeros(x.shape, dtype=np.uint8)
    if finite.any():
        low, high = np.min(x[finite]), np.max(x[finite])
        if high > low:
            result[finite] = np.clip(
                (x[finite] - low) * 255 / (high - low), 0, 255
            ).astype(np.uint8)
    return result


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

    if len(y_new) <= 10:
        return None, None, None

    x_new = x_new.ravel().astype(int)
    valid = (
        (x_new >= 0)
        & (x_new < hs_image_copy.shape[1])
        & (y_new >= 0)
        & (y_new < hs_image_copy.shape[0])
    )
    if not valid.any():
        return None, None, None
    x_new, y_new = x_new[valid], y_new[valid]
    hs_image_copy[y_new, x_new] = pixel_values[valid]

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

    valid = np.isfinite(hs_patch) & np.isfinite(mica_patch)
    if valid.sum() < 2:
        return np.nan
    hs_patch = to_uint8(np.asarray(hs_patch)[valid]).astype(int)
    mica_patch = to_uint8(np.asarray(mica_patch)[valid]).astype(int)
    if np.ptp(hs_patch) == 0 or np.ptp(mica_patch) == 0:
        return np.nan

    # Calculate mutual information between the patches
    mi = normalized_mutual_information(hs_patch, mica_patch)
    return mi


def get_shift_from_mi_patch(
    hs_image, mica_image, micamask_image, y_old, x_old, n=3, random_state=None
):
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
    model = RANSACRegressor(
        LinearRegression(),
        max_trials=20,
        residual_threshold=30,
        random_state=random_state,
    ).fit(x_old.reshape(-1, 1), y_old)

    # lets make sure we have enough number of inliers
    inlier_mask = model.inlier_mask_
    if inlier_mask.sum() < 5:
        return None, np.nan, np.nan

    # Calculate MI before shifting
    if micamask_image is not None and (
        mica_image.shape == hs_image.shape
    ):  # this is to make sure
        patch1 = hs_image[micamask_image]
        patch2 = mica_image[micamask_image]
    else:
        patch1 = hs_image
        patch2 = mica_image

    # create a binary mask around the indices to grab a smaller area that the entire image patch
    # binary_mask = np.zeros_like(patch1)
    # binary_mask[y_old, x_old] = 1
    # # dilate that mask 3 time
    # from scipy import ndimage
    # binary_mask = ndimage.binary_dilation(binary_mask, iterations = 10)
    # mi_before = calculate_mi_patch(patch1[binary_mask],
    #                                patch2[binary_mask])

    mi_before = calculate_mi_patch(patch1, patch2)
    if not np.isfinite(mi_before) or mi_before <= 0:
        return None, np.nan, np.nan

    # Generate pixel shifts
    pixel_shifts = np.arange(-n, +n + 1)

    # Iterate over pixel shifts
    for shift_value in pixel_shifts:
        hs_image_copy = copy.copy(hs_image)

        # Shift rows based on the regression model
        hs_image_copy, x_new, y_new = shift_rows_from_model(
            hs_image_copy, model, y_old, x_old, shift_value
        )

        if (hs_image_copy is None) or (x_new is None) or (y_new is None):
            linear_models.append(None)
            mi_quality_metrics.append(np.nan)
            continue

        # Calculate MI after shifting
        if micamask_image is not None:
            mi_after = calculate_mi_patch(
                hs_image_copy[micamask_image], mica_image[micamask_image]
            )
        else:
            # mi_after = calculate_mi_patch(hs_image_copy[binary_mask], mica_image[binary_mask])
            mi_after = calculate_mi_patch(hs_image_copy, mica_image)

        # Calculate MI quality metric
        mi_quality_metric = (mi_after - mi_before) / mi_before
        mi_quality_metrics.append(mi_quality_metric)
        linear_models.append(model)

    # Find the shift with the maximum MI quality metric
    if len(linear_models) > 0 and not all(np.isnan(mi_quality_metrics)):
        argmax = np.nanargmax(mi_quality_metrics)
        if (
            mi_quality_metrics[argmax] > 0.0005
        ):  # if the improvement is bigger than 0.05% do it
            return (
                linear_models[argmax],
                pixel_shifts[argmax],
                mi_quality_metrics[argmax],
            )
        else:
            return None, np.nan, np.nan
    else:
        return None, np.nan, np.nan


def medfilt3d(hs_arr_copy, kernel_size=3):
    hs_arr_final = []
    # we are not messing with the waterfall band

    for band in tqdm(
        range(hs_arr_copy.shape[2] - 1),
        desc="performing band wise filtering",
        position=0,
        leave=True,
    ):
        hs_arr_final.append(
            median_filter(hs_arr_copy[..., band], size=kernel_size)[..., None]
        )

    # not messing with the last layer just because
    hs_arr_final.append(hs_arr_copy[..., -1][..., None])
    hs_arr_final = np.concatenate(hs_arr_final, 2)
    return hs_arr_final


def generate_index_chunks(min_val, max_val, chunk_size):
    return [
        np.arange(i, min(i + chunk_size, max_val))
        for i in range(min_val, max_val, chunk_size)
    ]


def _row_groups(rows):
    flat = rows.ravel()
    positions = np.flatnonzero(np.isfinite(flat) & (flat > 0))
    if not len(positions):
        return []
    if np.any(flat[positions] != np.floor(flat[positions])):
        raise ValueError(
            "Acquisition-row identifiers must be integers; use nearest-neighbor resampling."
        )
    positions = positions[np.argsort(flat[positions], kind="stable")]
    values, starts, counts = np.unique(
        flat[positions], return_index=True, return_counts=True
    )
    return [
        (int(value), np.divmod(positions[start : start + count], rows.shape[1]))
        for value, start, count in zip(values, starts, counts)
        if count >= 5
    ]


def _row_seed(seed, name, row):
    if seed is None:
        return None
    digest = int.from_bytes(hashlib.sha256(str(name).encode()).digest()[:4], "little")
    return int(
        np.random.SeedSequence([int(seed), digest, int(row)]).generate_state(1)[0]
    )


def _image_fingerprint(image):
    stat = os.stat(image.path) if not image.path.startswith("/vsimem/") else None
    return {
        "path": image.path,
        "size": stat.st_size if stat else None,
        "mtime_ns": stat.st_mtime_ns if stat else None,
        "shape": list(image.shape),
        "transform": list(image.geotransform),
        "projection": image.projection,
    }


def _save_models(path, payload):
    descriptor, temporary = tempfile.mkstemp(
        prefix=".models-", dir=os.path.dirname(path)
    )
    os.close(descriptor)
    try:
        joblib.dump(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def shape_shift_mpp_gimg(
    hs_gimg,
    mica_gimg,
    hs_bands,
    mica_band,
    pixel_shift=3,
    kernel_size=3,
    hs_waterfall_rows_band_number=-1,
    num_threads=None,
    mica_mask_gimg=None,
    output_folder=None,
    use_available_model=False,
    name=None,
    random_seed=None,
    model_folder=None,
):
    """Fit reproducible row corrections and apply them to every spectral band.

    The row plane defines image support; valid zero-valued measurements are
    retained. Models live on disk even when intermediate rasters use /vsimem.
    New caches include input/settings/row associations; legacy lists are rejected.
    """
    workers = os.cpu_count() if num_threads is None else int(num_threads)
    if workers < 1 or int(pixel_shift) != pixel_shift or pixel_shift < 0:
        raise ValueError(
            "Worker count must be positive and pixel_shift a nonnegative integer."
        )
    if kernel_size < 1 or int(kernel_size) != kernel_size or kernel_size % 2 != 1:
        raise ValueError("Median kernel size must be a positive odd integer.")
    hs_bands = [int(index) for index in hs_bands]
    row_index = hs_waterfall_rows_band_number % hs_gimg.bands
    if row_index != hs_gimg.bands - 1:
        raise ValueError("The acquisition-row plane must be the final input band.")
    if not hs_bands or any(index < 0 or index >= row_index for index in hs_bands):
        raise ValueError(
            "Scoring bands must be nonempty spectral-band indices, excluding the row plane."
        )
    if mica_band is None or not (-mica_gimg.bands <= int(mica_band) < mica_gimg.bands):
        raise ValueError("A valid single MSI band is required.")
    cache_folder = model_folder or output_folder or os.path.dirname(hs_gimg.path) or "."
    if str(cache_folder).startswith("/vsimem"):
        raise ValueError(
            "model_folder must be a disk directory when raster output uses /vsimem."
        )
    os.makedirs(cache_folder, exist_ok=True)
    cache_path = os.path.join(cache_folder, f"results_{name or 'hs'}.pkl")
    hs_arr = hs_gimg.read(band_last=True)
    mica_image = mica_gimg.read(int(mica_band) % mica_gimg.bands + 1).astype(np.float32)
    if mica_image.shape != hs_arr.shape[:2]:
        raise ValueError(
            "HS and MSI grids must have equal dimensions before ShapeShifter."
        )
    rows = hs_arr[..., row_index]
    support = np.isfinite(rows) & (rows > 0)
    groups = _row_groups(rows)
    scoring = hs_arr[..., hs_bands].astype(np.float32)
    scoring[~np.isfinite(scoring)] = np.nan
    scoring[~support] = np.nan
    for channel, index in enumerate(hs_bands):
        nodata = hs_gimg.band_nodata[index]
        if nodata is not None and nodata != 0:
            scoring[..., channel][scoring[..., channel] == nodata] = np.nan
    count = np.isfinite(scoring).sum(axis=2)
    mean = np.divide(
        np.nansum(scoring, axis=2),
        count,
        out=np.full(rows.shape, np.nan, np.float32),
        where=count > 0,
    )
    del scoring
    mica_nodata = mica_gimg.band_nodata[int(mica_band) % mica_gimg.bands]
    if mica_nodata is not None:
        mica_image[mica_image == mica_nodata] = np.nan
    mask = (
        mica_mask_gimg.read().squeeze().astype(bool)
        if mica_mask_gimg is not None
        else None
    )
    settings = {
        "schema": 1,
        "kernel_version": "float-support-v1",
        "hs": _image_fingerprint(hs_gimg),
        "msi": _image_fingerprint(mica_gimg),
        "bands": hs_bands,
        "mica_band": int(mica_band),
        "pixel_shift": int(pixel_shift),
        "median_kernel": int(kernel_size),
        "seed": random_seed,
        "mask": _image_fingerprint(mica_mask_gimg)
        if mica_mask_gimg is not None
        else None,
        "row_ids": [row for row, _ in groups],
    }
    if use_available_model:
        payload = joblib.load(cache_path)
        if (
            not isinstance(payload, dict)
            or payload.get("settings") != settings
            or not payload.get("complete")
        ):
            raise ValueError(
                "Saved ShapeShifter cache is legacy, incomplete, or incompatible with this input/configuration."
            )
        results = payload["results"]
        if len(results) != len(groups):
            raise ValueError("Saved model count does not match acquisition rows.")
    else:
        results = []
        pool = Pool(workers) if workers > 1 else None
        try:
            for start in tqdm(
                range(0, len(groups), workers), desc="Finding SS transformations"
            ):
                tasks = []
                for row, (yy, xx) in groups[start : start + workers]:
                    region = np.s_[yy.min() : yy.max() + 1, xx.min() : xx.max() + 1]
                    tasks.append(
                        (
                            mean[region],
                            mica_image[region],
                            mask[region] if mask is not None else None,
                            yy - yy.min(),
                            xx - xx.min(),
                            int(pixel_shift),
                            _row_seed(random_seed, name, row),
                        )
                    )
                results.extend(
                    pool.starmap(get_shift_from_mi_patch, tasks)
                    if pool is not None
                    else [get_shift_from_mi_patch(*task) for task in tasks]
                )
                _save_models(
                    cache_path,
                    {
                        "settings": settings,
                        "complete": len(results) == len(groups),
                        "results": results,
                    },
                )
            if not groups:
                _save_models(
                    cache_path, {"settings": settings, "complete": True, "results": []}
                )
        finally:
            if pool is not None:
                pool.close()
                pool.join()
    shifted = hs_arr.copy()
    for (_, (yy, xx)), (model, shift, quality) in zip(groups, results):
        if model is not None:
            candidate, _, _ = shift_rows_from_model(shifted, model, yy, xx, shift)
            if candidate is not None:
                shifted = candidate
    shifted = medfilt3d(shifted, kernel_size=int(kernel_size))
    # Median filtering must not manufacture data outside the acquisition support.
    shifted[shifted[..., -1] == 0, :-1] = 0
    difference = np.zeros(rows.shape, np.float32)
    for band in range(row_index):
        difference += (
            np.abs(shifted[..., band].astype(np.float32) - hs_arr[..., band])
            / row_index
        )
    quality = to_uint8(difference)
    qa_image = copy.copy(hs_gimg)
    qa_image.bands = 1
    qa_image.band_description = ["ShapeShifter radiometric difference (display QA)"]
    qa_image.band_nodata = [None]
    qa = to_envi(
        quality[None],
        qa_image,
        hs_gimg.path + "_ss_qa",
        dtype=np.uint8,
        output_folder=output_folder,
    )
    output = to_envi(
        np.moveaxis(shifted, 2, 0),
        hs_gimg,
        hs_gimg.path + "_ss",
        output_folder=output_folder,
    )
    return output, qa
