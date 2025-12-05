"""
SPLASH: Spatial eLAstic Harmonization technique

this version of splash performs elastic coregistration between VNIR-SWIR at the end instead
of each of them together.

Author: Amirhossein Hassanzadeh
Email: axhcis@rit.edu
Date: 07/24/2025

Description:
This script implements the SPLASH technique for harmonizing spatial and spectral data.
It involves steps such as image loading, coregistration, cropping, and shape shifting.
"""
from sympy import false

from scripts.shape_shifter import shape_shift_mpp_gimg
from geoharmony.rigid_coregisteration.launchers import coregister_gui
from geoharmony.utils.gdalwriter import *
import geoharmony.utils.gdalimage as gi
from scripts.nonground_mask_creator import *
import warnings
from scripts.antspy_registration_onraster import antspy_vnir_swir_gimg
import copy
from geoharmony.utils.gdalwriter import warp_extent_res
from geoharmony.utils.gdalwriter import to_envi_scale
from geoharmony.rigid_coregisteration.launchers import get_splash_inputs
import os

warnings.filterwarnings("ignore")

class Color:
    BLUE = '\033[94m'
    END = '\033[0m'

if __name__ == "__main__":

    # inputs = get_splash_inputs()
    #
    # vnir_path = inputs["vnir_path"]
    # swir_path = inputs["swir_path"]
    # mica_path = inputs["mica_path"]
    # mica_dsm_path = inputs["mica_dsm_path"]
    # out_folder = inputs["out_folder"]
    # save_to_memory = inputs["save_to_memory"]
    # num_threads = inputs["num_threads"]
    #
    # # coregistration parameters
    # use_available_homography = inputs["use_available_homography"]
    # coreg_vnir_band_indices = inputs["coreg_vnir_band_indices"]
    # coreg_swir_band_indices = inputs["coreg_swir_band_indices"]
    # coreg_mica_band_indices = inputs["coreg_mica_band_indices"]
    #
    # ## shapeshifter parameters
    # pixel_shift = inputs["pixel_shift"]
    # kernel_size = inputs["kernel_size"]
    # ss_vnir_band_indices = np.arange(inputs["ss_vnir_band_indices"][0], inputs["ss_vnir_band_indices"][1] + 1)
    # ss_swir_band_indices = np.arange(inputs["ss_swir_band_indices"][0], inputs["ss_swir_band_indices"][1] + 1)
    # ss_vnir_mica_band_index = inputs["ss_vnir_mica_band_index"]
    # ss_swir_mica_band_index = inputs["ss_vnir_mica_band_index"]
    #
    # # antspy registration parameters
    # ereg_vnir_overlap_band_indices = inputs["ereg_vnir_overlap_band_indices"]
    # ereg_swir_overlap_band_indices = inputs["ereg_swir_overlap_band_indices"]
    # antspy_kernel = inputs["elastic_kernel"]
    # sigma_color = inputs["elastic_sigma_color"]
    # sigma_space = inputs["elastic_sigma_space"]
    # steps = inputs["elastic_steps"]
    # flow_sigma = inputs["elastic_flow_sigma"]

    ### for debugging purposes
    ### the image paths and parameters
    vnir_path = "/Volumes/T7/axhcis/Projects/NURI/data_forpaper/tait/vnir/raw_0_rf_wr_or"
    swir_path = "/Volumes/T7/axhcis/Projects/NURI/data_forpaper/tait/swir/raw_1504_nuc_rf_wr_or"
    mica_path = "/Volumes/T7/axhcis/Projects/NURI/data_forpaper/tait/mica/NURI_micasense_1133_transparent_mosaic_stacked"
    mica_dsm_path = None
    out_folder = "/Volumes/T7/axhcis/Projects/NURI/data_forpaper/tait/out"
    save_to_memory = False # radio button to save to memory or not should output True or False
    num_threads = int(8) # None defualts to max cpu count available
    convert_to_uint16 = False

    # coregistration parameters
    use_available_homography = True
    coreg_vnir_band_indices = [25, 70, 115]
    coreg_swir_band_indices = [10, 40, 90]
    coreg_mica_band_indices = [0, 1, 2]

    ## shapeshifter parameters
    pixel_shift = 3
    kernel_size = 3
    ss_vnir_band_indices = np.arange(60, 77)
    ss_swir_band_indices = np.arange(1, 12)
    ss_vnir_mica_band_index = -2
    ss_swir_mica_band_index = -1

    # antspy registration parameters
    ereg_vnir_overlap_band_indices = np.arange(272 - 46, 272)
    ereg_swir_overlap_band_indices = np.arange(1, 12)
    antspy_kernel = 20
    sigma_color = 15
    sigma_space = 20
    steps = 8
    flow_sigma = 3

    # creating folder
    os.makedirs(out_folder, exist_ok=True)

    memory_folder = '/vsimem' if save_to_memory else out_folder

    # 0) lets see if we can save some space and processing time by doing things in uint space
    print(Color.BLUE + "------> Converting all data to UINT16 for faster processing..." + Color.END)
    mica_gimg = gi.GdalImage(mica_path)
    vnir_gimg = gi.GdalImage(vnir_path)
    swir_gimg = gi.GdalImage(swir_path)
    if convert_to_uint16:
        mica_gimg_u16, _ = gi.gdalimage_to_unit16(mica_gimg, is_mica=True, output_folder=memory_folder)
        vnir_gimg_u16, vnir_scale_coefficients = gi.gdalimage_to_unit16(vnir_gimg, output_folder=memory_folder)
        swir_gimg_u16, swir_scale_coefficients = gi.gdalimage_to_unit16(swir_gimg, output_folder=memory_folder)
    else:
        mica_gimg_u16 = mica_gimg
        vnir_gimg_u16 = vnir_gimg
        swir_gimg_u16 = swir_gimg
        vnir_scale_coefficients = None
        swir_scale_coefficients = None

    # # 1) upsample micasense data to VNIR and SWIR resolutions
    print(" ")
    print(Color.BLUE + "------> Performing coregistration on VNIR..." + Color.END)
    mica_gimg_u16_vnirres = warp_extent_res(mica_gimg_u16, vnir_gimg_u16, "vnirres", output_folder=memory_folder)
    vnir_gimg_u16_warped = coregister_gui(mica_gimg_u16_vnirres,
                                          coreg_mica_band_indices,
                                          vnir_gimg_u16,
                                          coreg_vnir_band_indices,
                                          memory_folder,
                                          out_folder,
                                          use_available_homography=use_available_homography,
                                          name="vnir")
    if memory_folder == '/vsimem':
        vnir_gimg_u16.unlink()

    print(" ")
    print(Color.BLUE + "------> Performing coregistration on SWIR..." + Color.END)
    mica_gimg_u16_swirres = warp_extent_res(mica_gimg_u16, swir_gimg_u16, "swirres", output_folder=memory_folder)
    swir_gimg_u16_warped = coregister_gui(mica_gimg_u16_swirres,
                                          coreg_mica_band_indices,
                                          swir_gimg_u16,
                                          coreg_swir_band_indices,
                                          memory_folder,
                                          out_folder,
                                          use_available_homography=use_available_homography,
                                          name="swir")
    if memory_folder == '/vsimem':
        swir_gimg_u16.unlink()
        mica_gimg_u16.unlink()

    ### OK!
    # print(" ")
    print(Color.BLUE + "------> Setting zeros in the mica data..." + Color.END)
    # setting zeros in the mica sense image so shapeshifter could do what it is supposed to; writing in place
    mica_gimg_u16_vnirres = set_zeros_base_ref(vnir_gimg_u16_warped, mica_gimg_u16_vnirres, output_folder = memory_folder)
    mica_gimg_u16_swirres = set_zeros_base_ref(swir_gimg_u16_warped, mica_gimg_u16_swirres, output_folder = memory_folder)
    ### OK!

    # whether DSM data is available or not
    if mica_dsm_path is not None:
        mica_dsm_gimg = gi.GdalImage(mica_dsm_path)
        mica_mask_gimg = get_mask_gimg(mica_gimg, mica_dsm_gimg, output_folder=memory_folder)
        mica_mask_gimg_vnirres = warp_extent_res(mica_mask_gimg, vnir_gimg_u16_warped, "vnirres", output_folder=memory_folder)
        mica_mask_gimg_swirres = warp_extent_res(mica_mask_gimg, swir_gimg_u16_warped, "swirres", output_folder=memory_folder)
        mica_mask_gimg_vnirres = set_zeros_base_ref(vnir_gimg_u16_warped, mica_mask_gimg_vnirres, output_folder = memory_folder)
        mica_mask_gimg_swirres = set_zeros_base_ref(swir_gimg_u16_warped, mica_mask_gimg_swirres, output_folder = memory_folder)
    else:
        mica_mask_gimg_vnirres, mica_mask_gimg_swirres = None, None
    ### OK!

    # # 3) run shapeshifter on both of the datasets based on the mica data
    print(" ")
    print(Color.BLUE + "------> Running Shape Shifter for VNIR..." + Color.END)
    vnir_gimg_u16_warped_ss, vnir_gimg_u16_warped_ss_qa = shape_shift_mpp_gimg(vnir_gimg_u16_warped,
                                                                               mica_gimg_u16_vnirres,
                                                                               pixel_shift=pixel_shift,
                                                                               kernel_size=kernel_size,
                                                                               hs_bands=ss_vnir_band_indices,
                                                                               mica_band=ss_vnir_mica_band_index,
                                                                               num_threads=num_threads,
                                                                               mica_mask_gimg=mica_mask_gimg_vnirres,
                                                                               output_folder=memory_folder)
    if memory_folder == '/vsimem':
        mica_gimg_u16_vnirres.unlink()
        vnir_gimg_u16_warped.unlink()
        if mica_dsm_path is not None:
            mica_mask_gimg_vnirres.unlink()


    print(Color.BLUE + "------> Running Shape Shifter for SWIR..." + Color.END)
    swir_gimg_u16_warped_ss, swir_gimg_u16_warped_ss_qa = shape_shift_mpp_gimg(swir_gimg_u16_warped,
                                                                               mica_gimg_u16_swirres,
                                                                               pixel_shift=pixel_shift,
                                                                               kernel_size=kernel_size,
                                                                               hs_bands=ss_swir_band_indices,
                                                                               mica_band=ss_swir_mica_band_index,
                                                                               num_threads=num_threads,
                                                                               mica_mask_gimg=mica_mask_gimg_swirres,
                                                                               output_folder=memory_folder)
    if memory_folder == '/vsimem':
        mica_gimg_u16_swirres.unlink()
        swir_gimg_u16_warped.unlink()
        if mica_dsm_path is not None:
            mica_mask_gimg_swirres.unlink()

    ### OK


    print(" ")
    print(Color.BLUE + "------> Running Antspy registration..." + Color.END)
    (swir_gimg_u16_warped_ss_ereg,
     swir_gimg_u16_warped_ss_ereg_def,
     swir_gimg_u16_warped_ss_ereg_qa,
     final_mask) = antspy_vnir_swir_gimg(vnir_gimg_u16_warped_ss,
                                         swir_gimg_u16_warped_ss,
                                         vnir_overlap_band_indices=ereg_vnir_overlap_band_indices,
                                         swir_overlap_band_indices=ereg_swir_overlap_band_indices,
                                         output_folder=memory_folder,
                                         kernel=antspy_kernel,
                                         sigma_color=sigma_color,
                                         sigma_space=sigma_space,
                                         steps=steps,
                                         flow_sigma=flow_sigma)
    ### OK!

    ###### ERROR PROPAGATION
    print(" ")
    print(Color.BLUE + "------> Error Propagation..." + Color.END)
    ## MATCHGING SWIR TO VNIR
    swir_gimg_u16_warped_ss_qa_vnirres = warp_extent_res(swir_gimg_u16_warped_ss_qa,
                                                         vnir_gimg_u16_warped_ss_qa,
                                                         "vnirres",
                                                         "bilinear",
                                                         output_folder=memory_folder)
    #### EREG: MATCHING SWIR TO VNIR
    swir_gimg_u16_warped_ss_ereg_qa_vnirres = warp_extent_res(swir_gimg_u16_warped_ss_ereg_qa,
                                                              vnir_gimg_u16_warped_ss_qa,
                                                              "vnirres",
                                                              "bilinear",
                                                              output_folder=memory_folder)
    #### EREG: MATCHING SWIR TO VNIR
    swir_gimg_u16_warped_ss_ereg_def_vnirres = warp_extent_res(swir_gimg_u16_warped_ss_ereg_def,
                                                               vnir_gimg_u16_warped_ss_qa,
                                                               "vnirres",
                                                               "bilinear",
                                                               output_folder=out_folder)
    if memory_folder == '/vsimem':
        swir_gimg_u16_warped_ss_ereg_def.unlink()
    ### OK!

    # save the two images with coefficients applied to make life easier
    # we want to write these to file as they are final images
    if convert_to_uint16:
        to_envi_scale(swir_gimg_u16_warped_ss_ereg,
                      swir_path + "_splash",
                      swir_scale_coefficients,
                      dtype=np.float64,
                      drop_last_band=True,
                      output_folder = out_folder)# this gets rid of the waterfall band
        to_envi_scale(vnir_gimg_u16_warped_ss,
                      vnir_path + "_splash",
                      vnir_scale_coefficients,
                      dtype=np.float64,
                      drop_last_band=True,
                      output_folder = out_folder)
        print(f"Final files are written to:\n"
              f" {swir_path}" + "_splash\n"
              f" {vnir_path}" + "_splash")
    else:
        print(f"Final files are written to:\n"
              f" {swir_gimg_u16_warped_ss_ereg.path}\n"
              f" {vnir_gimg_u16_warped_ss.path}")

    # this gets rid of the waterfall band

    # # loading all the images now and calculating the propagated error
    swir_ereg_qa = swir_gimg_u16_warped_ss_ereg_qa_vnirres.read()
    swir_ss_qa = swir_gimg_u16_warped_ss_qa_vnirres.read()
    vnir_ss_qa = vnir_gimg_u16_warped_ss_qa.read()
    swir_ereg_qa, swir_ss_qa, vnir_ss_qa = swir_ereg_qa.astype(float), swir_ss_qa.astype(float), vnir_ss_qa.astype(
        float)
    uncertainty = (swir_ereg_qa ** 2 + swir_ss_qa ** 2 + vnir_ss_qa) ** 0.5
    uncertainty[final_mask] = 0

    # for uncertainty raster
    output_filename = swir_path + "_splash_qa"
    to_uint8 = lambda x: ((x - np.min(x)) / (np.max(x) - np.min(x)) * 255)
    uncertainty = to_uint8(uncertainty)
    uncertainty_gimg = copy.copy(swir_gimg_u16_warped_ss_qa_vnirres)
    uncertainty_gimg.bands = 1
    uncertainty_gimg.band_description = ["SPLASH QA"]
    uncertainty_gimg.band_nodata = []
    uncertainty_gimg.band_stats = []
    to_envi(uncertainty[None, ...],
            uncertainty_gimg,
            output_filename,
            dtype=np.uint8,
            output_folder=out_folder)

    print(Color.BLUE + "------> All Done..." + Color.END)


