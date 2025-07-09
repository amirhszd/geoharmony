from cv2 import bilateralFilter
from spectral.io import envi
import numpy as np
import matplotlib.pyplot as plt
import dipy.align.imwarp as imwarp
from dipy.viz import regtools
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
import ants
from skimage.transform import rescale
import itk
from scipy.ndimage import binary_dilation, binary_closing
from scipy.signal import medfilt2d
from tqdm import tqdm
import copy
from multiprocessing import Pool
from itertools import repeat
from .shape_shifter import load_image_envi_fast, load_image_envi, save_image_envi
import sys
import argparse
from .nonground_mask_creator import crop_image_to_extent

def callback_CC(sdr, status):
    # Status indicates at which stage of the optimization we currently are
    # For now, we will only react at the end of each resolution of the scale
    # space
    if status == imwarp.RegistrationStages.SCALE_END:
        # get the current images from the metric
        wmoving = sdr.metric.moving_image
        wstatic = sdr.metric.static_image
        # draw the images on top of each other with different colors
        regtools.overlay_images(wmoving, wstatic, 'Warped moving', 'Overlay',
                                'Warped static')

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

def main(hs_hdr, mica_hdr,
         hs_bands,
         mica_band,
         kernel = 20,
         sigma_color = 15,
         sigma_space = 20,
         steps = 8,
         mica_mask_filename = None):

    # Load the hyperspectral and Mica images
    hs_arr, hs_profile, _ = load_image_envi(hs_hdr)
    mica_arr, mica_profile, _ = load_image_envi(mica_hdr)
    if mica_mask_filename:
        mica_mask_arr, mica_mask_profile = load_image_envi_fast(mica_mask_filename)
        mica_mask_arr = mica_mask_arr.squeeze().astype(bool)


    # Calculate the mean of the specified hyperspectral bands
    hs_image = np.mean(hs_arr[..., hs_bands], axis=2)
    mica_image = mica_arr[..., mica_band].squeeze()  # grabbing the last band in mica

    # Apply a low-pass filter using a bilateral filter
    mica_image_uint8 = to_uint8(mica_image)
    hs_image_uint8 = to_uint8(hs_image)
    if mica_mask_filename:
        mica_image_uint8[np.invert(mica_mask_arr)] = 0
        hs_image_uint8[np.invert(mica_mask_arr)] = 0

    mica_image_uint8 = bilateralFilter(mica_image_uint8, kernel , sigma_color, sigma_space)

    # Convert images to ANTs format
    hs_image_ants = ants.from_numpy(hs_image_uint8, is_rgb=False, has_components=False)
    mica_image_ants = ants.from_numpy(mica_image_uint8, is_rgb=False, has_components=False)

    # Calculate the transformation matrix using ANTs registration
    print("Calculating the transformation!")
    mtx = ants.registration(fixed=mica_image_ants,
                            moving=hs_image_ants,
                            type_of_transform="SyN")

    # # applying the transform on all channels
    # bands_warped = []
    # pbar = tqdm(total = hs_arr.shape[-1],
    #             position=0,
    #             leave=True,
    #             desc = "Performing non rigid transformation on bands")
    #
    # for i in range(hs_arr.shape[-1]):
    #     band = apply_registration_to_band(hs_arr, i, mica_image_ants, mtx)
    #     bands_warped.append(band[...,None])
    #     pbar.update(1)
    #
    # # saving the image
    # bands_warped = np.concatenate(bands_warped, 2)
    # output_hdr_filename = hs_hdr.replace(".hdr", "_ereg.hdr")
    # save_image_envi(bands_warped, hs_profile, output_hdr_filename)

    # Prepare to apply the transformation on all channels
    num_bands = hs_arr.shape[-1]
    # num_bands = 20
    pbar = tqdm(total = num_bands,
                position=0,
                leave=True,
                desc = "Performing non rigid transformation on bands")

    # Define the step size and generate index chunks for parallel processing
    def generate_index_chunks(total_length, chunk_size):
        return [np.arange(i, min(i + chunk_size, total_length)) for i in range(0, total_length, chunk_size)]

    indices_chunks = generate_index_chunks(num_bands, steps)
    bands_warped = np.zeros_like(hs_arr)

    # Process the bands in chunks using multiprocessing
    for indices_chunk in indices_chunks:
        hs_bands = []
        for index in indices_chunk:
            hs_bands.append(hs_arr[...,index])

        args_list = list(zip(hs_bands, repeat(mica_image_uint8), repeat(mtx['fwdtransforms'])))
        # we wont do more than 8 at a time due to memory
        with Pool(steps) as pool:
            band = pool.starmap(apply_registration_to_band, args_list)
            bands_warped[...,indices_chunk] = np.concatenate(band,2)

        pbar.update(steps)

    # Saving the image
    output_hdr_filename = hs_hdr.replace(".hdr", "_ereg.hdr")

    # TODO: USE THE FUNCTION FROM OUTSIDE WE HAVE MANY MULTIPLES OF THIS
    save_image_envi(bands_warped, hs_profile, output_hdr_filename, dtype = "uint16", ext= "")

    return output_hdr_filename


def main_vnir_swir(vnir_hdr,
                   swir_hdr,
                   kernel=20,
                   sigma_color=15,
                   sigma_space=20,
                   steps = 8,
                   flow_sigma = 3,
                   mica_mask_filename = None):

    # matching the extent and resolution of SWIR to VNIR
    swir_vnirres_hdr = swir_hdr.replace(".hdr", "_vnirres.hdr")
    crop_image_to_extent(vnir_hdr.replace(".hdr",""),
                         swir_hdr.replace(".hdr",""),
                         swir_vnirres_hdr.replace(".hdr",""),
                         "ENVI")

    # Load the two hyperspectral datasets
    vnir_arr, vnir_profile, vnir_wavelegnths = load_image_envi(vnir_hdr)
    swir_arr, swir_profile, swir_wavelegnths = load_image_envi(swir_vnirres_hdr)
    # removed this because it is not doing much
    # if mica_mask_filename:
    #     mica_mask_arr, mica_mask_profile = load_image_envi_fast(mica_mask_filename)
    #     mica_mask_arr = mica_mask_arr.squeeze().astype(bool)

    # Calculate the mean of the specified hyperspectral bands
    # find the overlap region between the two datasets
    overlap_vnir_bands = (vnir_wavelegnths > 900)
    overlap_swir_bands = swir_wavelegnths < np.max(vnir_wavelegnths)
    mean_vnir = np.mean(vnir_arr[..., overlap_vnir_bands], 2)
    mean_swir = np.mean(swir_arr[..., overlap_swir_bands], 2)

    # convert to uint8
    mean_vnir_uint8 = to_uint8(mean_vnir)
    mean_swir_uint8 = to_uint8(mean_swir)

    # perform bilateral filtering to manipulate the general shape and not the fine texture
    mean_vnir_uint8 = bilateralFilter(mean_vnir_uint8, kernel, sigma_color, sigma_space)
    mean_swir_uint8 = bilateralFilter(mean_swir_uint8, kernel, sigma_color, sigma_space)

    # develop a unitifed mask for all three images
    # final_mask = np.invert(np.logical_or(np.logical_or(mean_swir_uint8 == 0, mean_vnir_uint8 ==0),mica_mask_arr))
    final_mask = np.logical_or(mean_swir_uint8 == 0, mean_vnir_uint8 == 0) # just setting zero in data where its zero
    # if mica_mask_filename:
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
                            initial_transform="Identity")

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
    output_hdr_filename = swir_vnirres_hdr.replace(".hdr", "_ereg.hdr")
    save_image_envi(bands_warped, swir_profile, output_hdr_filename, dtype = "uint16", ext= "")

    # creating deformation band
    mygr = ants.create_warped_grid(swir_image_ants)
    warped_grid = ants.create_warped_grid(mygr, grid_directions=(True, True),
                                           transform=mtx['fwdtransforms'], fixed_reference_image=vnir_image_ants).numpy()
    warped_grid = warped_grid.astype(np.uint8)
    metadata = copy.copy(swir_profile)
    metadata["bands"] = "1"
    metadata["band names"] = "Deformation Band"
    try:
        del metadata["wavelength"]
    except:
        pass
    def_hdr_filename = swir_vnirres_hdr.replace(".hdr", "_ereg_deformation.hdr")
    save_image_envi(warped_grid, metadata, def_hdr_filename, dtype="uint8", ext="")

    # calculating quality assurance raster
    bands_warped = np.mean(bands_warped[..., overlap_swir_bands], 2)
    bands_warped_uint8 = to_uint8(bands_warped)
    quality_raster = np.abs(bands_warped_uint8 - mean_swir_uint8)
    quality_raster[final_mask] = 0
    metadata = copy.copy(swir_profile)
    metadata["bands"] = "1"
    metadata["band names"] = "Error Band"
    try:
        del metadata["wavelength"]
    except:
        pass
    qa_hdr_filename = swir_vnirres_hdr.replace(".hdr", "_ereg_qa.hdr")
    save_image_envi(quality_raster, metadata, qa_hdr_filename, dtype="uint8", ext="")

    return output_hdr_filename, def_hdr_filename, qa_hdr_filename, final_mask


def main_vnir_swir_test(vnir_hdr, swir_hdr, flow_sigma=3.0, syn_metric="mattes",
                        smoothing_mm=False, method="SyN", mica_mask_filename = None):

    import os
    # Match the extent and resolution of SWIR to VNIR
    swir_vnirres_hdr = swir_hdr.replace(".hdr", "_vnirres.hdr")

    # Load the two hyperspectral datasets
    vnir_arr, vnir_profile, vnir_wavelengths = load_image_envi(vnir_hdr)
    swir_arr, swir_profile, swir_wavelengths = load_image_envi(swir_vnirres_hdr)
    if mica_mask_filename:
        mica_mask_arr, mica_mask_profile = load_image_envi_fast(mica_mask_filename)
        mica_mask_arr = mica_mask_arr.squeeze()

    # Calculate the mean of overlapping hyperspectral bands
    overlap_vnir_bands = (vnir_wavelengths > 900)
    overlap_swir_bands = (swir_wavelengths < np.max(vnir_wavelengths))
    mean_vnir = np.mean(vnir_arr[..., overlap_vnir_bands], axis=2)
    mean_swir = np.mean(swir_arr[..., overlap_swir_bands], axis=2)

    # Convert to uint8
    mean_vnir_uint8 = to_uint8(mean_vnir)
    mean_swir_uint8 = to_uint8(mean_swir)

    # mask is inclusive
    final_mask = np.logical_or(mean_swir_uint8 == 0, mean_vnir_uint8 == 0)
    mean_vnir_uint8[final_mask] = 0
    mean_swir_uint8[final_mask] = 0

    # mask_ants = None
    final_mask = np.invert(np.logical_or(np.logical_or(mean_swir_uint8 == 0, mean_vnir_uint8 ==0),mica_mask_arr)).astype(np.uint8)
    mask_ants = ants.from_numpy(final_mask, is_rgb=False, has_components=False)

    # Convert VNIR and SWIR images to ANTs images
    vnir_image_ants = ants.from_numpy(mean_vnir_uint8, is_rgb=False, has_components=False)
    swir_image_ants = ants.from_numpy(mean_swir_uint8, is_rgb=False, has_components=False)

    # Register images with specified parameters
    mtx = ants.registration(
        fixed=vnir_image_ants,
        moving=swir_image_ants,
        type_of_transform='SyNOnly',
        flow_sigma=flow_sigma,
        syn_metric=syn_metric,
        smoothing_in_mm=smoothing_mm,
        mask_all_stages=True,
        initial_transform = 'Identity')

    # Apply the transformation to three selected bands for RGB representation
    bands_warped = []
    for i in [30, 60, 90]:  # Replace these indices with the desired bands
        band_warped = apply_registration_to_band(swir_arr[..., i], mean_vnir_uint8, mtx['fwdtransforms'])
        bands_warped.append(band_warped)
    bands_warped_rgb = np.dstack(bands_warped)

    # save the output band

    # also saving out the deformation as well as the before and after normalized MAE plot
    mygr = ants.create_warped_grid(swir_image_ants)
    mywarpedgrid = ants.create_warped_grid(mygr, grid_directions=(True, True),
                                           transform=mtx['fwdtransforms'], fixed_reference_image=vnir_image_ants).numpy()
    mywarpedgrid = mywarpedgrid.astype(np.uint8)
    metadata = copy.copy(swir_profile)
    metadata["bands"] = "1"
    metadata["band names"] = "Deformation Band"
    try:
        del metadata["wavelength"]
    except:
        pass
    output_hdr_filename = swir_vnirres_hdr.replace(".hdr", "_ereg_deformation.hdr")
    save_image_envi(mywarpedgrid, metadata, output_hdr_filename, dtype="uint8", ext="")

    # calculating quality assurance raster
    quality_raster = np.mean(np.abs(bands_warped - swir_arr[...,[30,60,90]]), axis = 2)
    quality_raster = to_uint8(quality_raster).astype("uint8")
    metadata = copy.copy(swir_profile)
    metadata["bands"] = "1"
    metadata["band names"] = "Error Band"
    try:
        del metadata["wavelength"]
    except:
        pass
    output_hdr_filename = swir_vnirres_hdr.replace(".hdr", "_ereg_qa.hdr")
    save_image_envi(quality_raster, metadata, output_hdr_filename, dtype="uint8", ext="")


    return bands_warped_rgb, mywarpedgrid, quality_raster


if __name__ == "__main__":
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='Antspy elastic registration technique.')
        parser.add_argument('--hs_hdr', type=str, help='Hyperspectral HDR filename.')
        parser.add_argument('--mica_hdr', type=str, help='Mica HDR filename.')
        parser.add_argument('--hs_bands', type=int, nargs='+', help='List of hyperspectral band indices to use.')
        parser.add_argument('--mica_band', type=int, help='Specific band index of the Mica image to use.')
        parser.add_argument('--kernel', type=int, default=20, help='Kernel size for the bilateral filter.')
        parser.add_argument('--sigma_color', type=int, default=15, help='Sigma color for the bilateral filter.')
        parser.add_argument('--sigma_space', type=int, default=20, help='Sigma space for the bilateral filter.')
        args = parser.parse_args()

        hs_hdr = args.hs_hdr
        mica_hdr = args.mica_hdr
        hs_bands = args.hs_bands
        mica_band = args.mica_band
        kernel = args.kernel
        sigma_color = args.sigma_color
        sigma_space = args.sigma_space

        main(hs_hdr, mica_hdr, hs_bands, mica_band, kernel, sigma_color, sigma_space)
    else:
        # debug
        hs_hdr = "/Volumes/T7/axhcis/Projects/NURI/raw_1504_nuc_or_plusindices3_warped_ss.hdr"
        mica_hdr = "/Volumes/T7/axhcis/Projects/NURI/data/20210723_tait_labsphere/1133/Micasense/NURI_micasense_1133_transparent_mosaic_stacked_warped_cropped.hdr"
        hs_bands = [1, 2, 3]  # replace with actual bands
        mica_band = 0  # replace with actual band

        main(hs_hdr, mica_hdr, hs_bands, mica_band)






