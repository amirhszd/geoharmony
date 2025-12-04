from cv2 import bilateralFilter
import numpy as np
import ants
from tqdm import tqdm
import copy
from multiprocessing import Pool
from itertools import repeat
from geoharmony.utils.gdalwriter import to_envi
from geoharmony.utils.gdalimage import GdalImage
from geoharmony.utils.gdalwriter import warp_extent_res

def apply_transform(fixed, moving, transform):

    regis_bands = []
    for i in range(moving.shape[-1]):
        regis_bands.append(ants.apply_transforms(fixed=fixed, moving=moving,
                                              transformlist=transform['fwdtransforms'])[...,None])
    regis_bands = np.concatenate(regis_bands, 2)
    return regis_bands

def get_transform(fixed, moving):
    transform = ants.registration(fixed=fixed, moving=moving, type_of_transform='Elastic')
    return transform


def apply_registration_to_band(hs_band, mica_image_uint8, transformlist):

    # some of the bands have all zero values
    nonzeros = np.nonzero(hs_band)
    if len(np.nonzero(hs_band)[0]) == 0:
        return hs_band[...,None]

    min = np.min(hs_band[nonzeros])
    max = hs_band.max()

    # convert to 0-1
    to_uint8_simple = lambda x: (
            (x - np.min(x[np.nonzero(x)])) / (np.max(x[np.nonzero(x)]) - np.min(x[np.nonzero(x)])) * 255)
    band_uint8 = to_uint8_simple(hs_band)
    band_uint8[hs_band == 0] = 0
    # residual to add later
    band_uint8_res = band_uint8 - band_uint8.astype(np.uint8)
    # process image
    hs_band_ants = ants.from_numpy(band_uint8.astype(np.uint8), is_rgb=False, has_components=False)
    mica_image_ants = ants.from_numpy(mica_image_uint8, is_rgb=False, has_components=False)
    hs_band_ants_warped = ants.apply_transforms(fixed=mica_image_ants, moving=hs_band_ants,
                                                transformlist=transformlist).numpy()
    # adding the residual
    hs_band_ants_warped = hs_band_ants_warped + band_uint8_res
    # adding the min and max to the entire array
    hs_band_ants_warped_final = hs_band_ants_warped * (max - min) / 255 + min
    # apply the mask back to it
    hs_band_ants_warped_final[hs_band == 0] = 0
    hs_band_ants_warped_final = hs_band_ants_warped_final.astype(int)
    return hs_band_ants_warped_final[...,None]

def to_uint8(x):
    return ((x - np.min(x[np.nonzero(x)])) / (np.max(x[np.nonzero(x)]) - np.min(x[np.nonzero(x)])) * 255).astype(np.uint8)

def antspy_vnir_swir_gimg(vnir_gimg,
                          swir_gimg,
                          vnir_overlap_band_indices = None,
                          swir_overlap_band_indices = None,
                          output_folder = None,
                          kernel=20,
                          sigma_color=15,
                          sigma_space=20,
                          steps = 8,
                          flow_sigma = 3):

    # matching the extent and resolution of SWIR to VNIR
    swir_gimg_vnirres = warp_extent_res(swir_gimg,
                                        vnir_gimg,
                                       "vnirres",
                                       "bilinear", output_folder=output_folder)


    vnir_arr = vnir_gimg.read(band_last = True)
    swir_arr = swir_gimg_vnirres.read(band_last = True)

    # set inf values to nan in both vnir and swir
    vnir_arr[np.isinf(vnir_arr)] = np.nan
    swir_arr[np.isinf(swir_arr)] = np.nan

    # Calculate the mean of the specified hyperspectral bands
    mean_vnir = np.nanmean(vnir_arr[..., vnir_overlap_band_indices], 2)
    mean_swir = np.nanmean(swir_arr[..., swir_overlap_band_indices], 2)

    # convert to uint8
    mean_vnir_uint8 = to_uint8(mean_vnir)
    mean_swir_uint8 = to_uint8(mean_swir)

    # perform bilateral filtering to manipulate the general shape and not the fine texture
    mean_vnir_uint8 = bilateralFilter(mean_vnir_uint8, kernel, sigma_color, sigma_space)
    mean_swir_uint8 = bilateralFilter(mean_swir_uint8, kernel, sigma_color, sigma_space)

    # develop a unitifed mask for all three images
    final_mask = np.logical_or(mean_swir_uint8 == 0, mean_vnir_uint8 == 0) # just setting zero in data where its zero

    mean_vnir_uint8[final_mask] = 0
    mean_swir_uint8[final_mask] = 0

    # Convert images to ANTs format
    vnir_image_ants = ants.from_numpy(mean_vnir_uint8, is_rgb=False, has_components=False)
    swir_image_ants = ants.from_numpy(mean_swir_uint8, is_rgb=False, has_components=False)

    # Calculate the transformation matrix using ANTs registration
    print("Calculating the transformation!")
    mtx = ants.registration(fixed=vnir_image_ants,
                            moving=swir_image_ants,
                            type_of_transform="SyN",
                            flow_sigma=flow_sigma,
                            initial_transform="Identity",
                            verbose=1)

    # Prepare to apply the transformation on all channels
    num_bands = swir_arr.shape[-1]
    # num_bands = 20
    pbar = tqdm(total = num_bands,
                position=0,
                leave=True,
                desc = "Performing non rigid transformation on bands")

    # Define the step size and generate index chunks for parallel processing
    def generate_index_chunks(total_length, chunk_size):
        return [np.arange(i, min(i + chunk_size, total_length)) for i in range(0, total_length, chunk_size)]

    indices_chunks = generate_index_chunks(num_bands, steps)
    bands_warped = np.zeros_like(swir_arr)

    # Process the bands in chunks using multiprocessing
    for indices_chunk in indices_chunks:
        hs_bands = []
        for index in indices_chunk:
            hs_bands.append(swir_arr[...,index])
        args_list = list(zip(hs_bands, repeat(mean_vnir_uint8), repeat(mtx['fwdtransforms'])))
        # we wont do more than 8 at a time due to memory
        with Pool(steps) as pool:
            band = pool.starmap(apply_registration_to_band, args_list)
            bands_warped[...,indices_chunk] = np.concatenate(band,2)
        pbar.update(steps)

    # Saving the image
    swir_ereg_output_filename = swir_gimg_vnirres.path + "_ereg"
    to_envi(np.moveaxis(bands_warped, 2, 0),
            swir_gimg_vnirres,
            swir_ereg_output_filename,
            output_folder=output_folder)


    # creating deformation band
    mygr = ants.create_warped_grid(swir_image_ants)
    warped_grid = ants.create_warped_grid(mygr,
                                          grid_directions=(True, True),
                                           transform=mtx['fwdtransforms'],
                                          fixed_reference_image=vnir_image_ants).numpy()
    warped_grid = warped_grid.astype(np.uint8)
    deform_gimg = copy.copy(swir_gimg_vnirres)
    deform_gimg.bands = 1
    deform_gimg.band_description = ["Deformation Band"]
    deform_gimg.band_nodata = []
    deform_gimg.band_stats = []
    ereg_deform_output_filename = deform_gimg.path + "_ereg_deform"
    to_envi(warped_grid[None,...],
            deform_gimg,
            ereg_deform_output_filename,
            np.uint8,
            output_folder=output_folder)


    # calculating quality assurance raster
    bands_warped = np.mean(bands_warped[..., swir_overlap_band_indices], 2).squeeze()
    bands_warped_uint8 = to_uint8(bands_warped)
    quality_raster = np.abs(bands_warped_uint8 - mean_swir_uint8)
    quality_raster[final_mask] = 0
    quality_gimg = copy.copy(swir_gimg_vnirres)
    quality_gimg.bands = 1
    quality_gimg.band_description = ["Error Band"]
    quality_gimg.band_nodata = []
    quality_gimg.band_stats = []
    ereg_qa_output_filename = swir_gimg_vnirres.path + "_ereg_qa"
    to_envi(quality_raster[None,...],
            quality_gimg,
            ereg_qa_output_filename,
            np.uint8,
            output_folder=output_folder)

    return GdalImage(swir_ereg_output_filename), GdalImage(ereg_deform_output_filename), GdalImage(ereg_qa_output_filename), final_mask






