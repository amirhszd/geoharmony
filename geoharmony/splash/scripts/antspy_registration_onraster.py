"""SyN alignment using floating-point radiometry and categorical support."""

import copy
import os
import tempfile
from multiprocessing import Pool

import ants
import numpy as np
from cv2 import bilateralFilter
from tqdm import tqdm

from geoharmony.splash.scripts.shape_shifter import to_uint8
from geoharmony.utils.gdalwriter import to_envi, warp_extent_res


def apply_registration_to_band(
    hs_band, mica_image_uint8, transformlist, valid_mask=None
):
    """Warp measurements and their support together, retaining valid zeros.

    Normalized linear interpolation excludes missing samples without adding an
    unwarped quantization residual. No uint8 conversion touches spectral values.
    """
    band = np.asarray(hs_band, dtype=np.float32)
    valid = np.isfinite(band)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)
    fixed = ants.from_numpy(np.asarray(mica_image_uint8, dtype=np.float32))
    moving = ants.from_numpy(np.where(valid, band, 0).astype(np.float32))
    weights = ants.from_numpy(valid.astype(np.float32))
    numerator = ants.apply_transforms(
        fixed=fixed, moving=moving, transformlist=transformlist, interpolator="linear"
    ).numpy()
    denominator = ants.apply_transforms(
        fixed=fixed, moving=weights, transformlist=transformlist, interpolator="linear"
    ).numpy()
    output = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=denominator > 1e-6,
    )
    return output[..., None]


def _mean_image(cube, bands, image):
    bands = [int(index) for index in bands]
    if not bands or any(index < 0 or index >= cube.shape[-1] - 1 for index in bands):
        raise ValueError(
            "Elastic overlap bands must exclude the final acquisition-row plane."
        )
    values = cube[..., bands].astype(np.float32)
    valid = np.isfinite(values) & (cube[..., -1, None] > 0)
    for channel, index in enumerate(bands):
        nodata = image.band_nodata[index]
        if nodata is not None and nodata != 0:
            valid[..., channel] &= values[..., channel] != nodata
    count = valid.sum(axis=2)
    total = np.where(valid, values, 0).sum(axis=2)
    return np.divide(
        total, count, out=np.full(count.shape, np.nan, np.float32), where=count > 0
    )


def _single_band_metadata(image, description):
    result = copy.copy(image)
    result.bands = 1
    result.band_description = [description]
    result.band_nodata = [None]
    return result


def antspy_vnir_swir_gimg(
    vnir_gimg,
    swir_gimg,
    vnir_overlap_band_indices=None,
    swir_overlap_band_indices=None,
    output_folder=None,
    kernel=20,
    sigma_color=15,
    sigma_space=20,
    steps=8,
    flow_sigma=3,
    num_threads=8,
    random_seed=42,
    model_folder=None,
):
    """Register SWIR to VNIR; the final plane is a nearest-sampled row map.

    ``steps`` remains a deprecated compatibility argument; ``num_threads``
    controls band application workers. ITK registration threads are controlled
    by ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS before importing ANTs.
    """
    if int(num_threads) != num_threads or num_threads < 1:
        raise ValueError("num_threads must be a positive integer.")
    if kernel < 1 or int(kernel) != kernel or sigma_color <= 0 or sigma_space <= 0:
        raise ValueError(
            "Bilateral filter settings must be positive, with an integer kernel."
        )
    cache = model_folder or (
        output_folder
        if output_folder and not str(output_folder).startswith("/vsimem")
        else None
    )
    cache = cache or tempfile.mkdtemp(prefix="splash-ants-")
    if str(cache).startswith("/vsimem"):
        raise ValueError("ANTs transforms require a disk model_folder.")
    os.makedirs(cache, exist_ok=True)
    swir_grid = warp_extent_res(
        swir_gimg,
        vnir_gimg,
        "vnirres",
        "bilinear",
        output_folder=output_folder,
        categorical_last_band=True,
    )
    fixed_cube = vnir_gimg.read(band_last=True)
    moving_cube = swir_grid.read(band_last=True).astype(np.float32)
    fixed_mean = _mean_image(fixed_cube, vnir_overlap_band_indices, vnir_gimg)
    moving_mean = _mean_image(moving_cube, swir_overlap_band_indices, swir_grid)
    common = (
        np.isfinite(fixed_mean)
        & np.isfinite(moving_mean)
        & (fixed_cube[..., -1] > 0)
        & (moving_cube[..., -1] > 0)
    )
    if (
        common.sum() < 16
        or np.ptp(fixed_mean[common]) <= 0
        or np.ptp(moving_mean[common]) <= 0
    ):
        raise ValueError(
            "Elastic registration needs overlapping image support with intensity variation."
        )
    fixed_uint8 = bilateralFilter(
        to_uint8(fixed_mean), int(kernel), sigma_color, sigma_space
    )
    moving_uint8 = bilateralFilter(
        to_uint8(moving_mean), int(kernel), sigma_color, sigma_space
    )
    fixed_uint8[~common] = 0
    moving_uint8[~common] = 0
    fixed = ants.from_numpy(fixed_uint8)
    moving = ants.from_numpy(moving_uint8)
    transforms = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform="SyN",
        flow_sigma=flow_sigma,
        initial_transform="Identity",
        random_seed=random_seed,
        outprefix=os.path.join(cache, "swir_syn_"),
        verbose=False,
    )
    transformlist = transforms["fwdtransforms"]
    # IDs are transformed categorically, independently of the spectral bands.
    row_image = ants.from_numpy(moving_cube[..., -1].astype(np.float32))
    warped_rows = ants.apply_transforms(
        fixed=fixed,
        moving=row_image,
        transformlist=transformlist,
        interpolator="nearestNeighbor",
    ).numpy()
    warped_support = np.isfinite(warped_rows) & (warped_rows > 0)
    output = np.zeros(moving_cube.shape, np.float32)
    spectral_count = moving_cube.shape[-1] - 1
    pool = Pool(int(num_threads)) if num_threads > 1 else None
    try:
        for start in tqdm(
            range(0, spectral_count, int(num_threads)),
            desc="Applying elastic transforms",
        ):
            indices = range(start, min(start + int(num_threads), spectral_count))
            arguments = []
            for index in indices:
                valid = (moving_cube[..., -1] > 0) & np.isfinite(
                    moving_cube[..., index]
                )
                nodata = swir_grid.band_nodata[index]
                if nodata is not None and nodata != 0:
                    valid &= moving_cube[..., index] != nodata
                arguments.append(
                    (moving_cube[..., index], fixed_uint8, transformlist, valid)
                )
            bands = (
                pool.starmap(apply_registration_to_band, arguments)
                if pool is not None
                else [apply_registration_to_band(*a) for a in arguments]
            )
            for index, band in zip(indices, bands):
                output[..., index] = np.where(warped_support, band[..., 0], 0)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    output[..., -1] = np.where(warped_support, warped_rows, 0)
    metadata = copy.copy(swir_grid)
    metadata.band_nodata = [
        None
    ] * metadata.bands  # row map, not spectral zero, defines support
    registered = to_envi(
        np.moveaxis(output, 2, 0),
        metadata,
        swir_grid.path + "_ereg",
        dtype=np.float32,
        output_folder=output_folder,
    )
    grid = ants.create_warped_grid(
        moving,
        grid_directions=(True, True),
        transform=transformlist,
        fixed_reference_image=fixed,
    ).numpy()
    deformation = to_envi(
        to_uint8(grid)[None],
        _single_band_metadata(swir_grid, "Elastic deformation grid (display QA)"),
        swir_grid.path + "_ereg_deform",
        dtype=np.uint8,
        output_folder=output_folder,
    )
    registered_mean = _mean_image(output, swir_overlap_band_indices, registered)
    difference = np.abs(
        to_uint8(registered_mean).astype(np.float32) - moving_uint8.astype(np.float32)
    )
    final_mask = ~(warped_support & (fixed_cube[..., -1] > 0))
    difference[final_mask] = 0
    quality = to_envi(
        to_uint8(difference)[None],
        _single_band_metadata(swir_grid, "Elastic intensity difference (display QA)"),
        swir_grid.path + "_ereg_qa",
        dtype=np.uint8,
        output_folder=output_folder,
    )
    return registered, deformation, quality, final_mask
