"""
SPLASH: Spatial eLAstic Harmonization technique

this version of splash performs elastic coregistration between VNIR-SWIR at the end instead
of each of them together.

Author: Amirhossein Hassanzadeh
Email: axhcis@rit.edu
Date: 06/15/2024

Description:
This script implements the SPLASH technique for harmonizing spatial and spectral data.
It involves steps such as image loading, coregistration, cropping, and shape shifting.

Usage:
    python splash.py --vnir_hdr <path_to_vnir_hdr> --swir_hdr <path_to_swir_hdr> --mica_hdr <path_to_mica_hdr> --outfolder <output_folder> [--manual_warping <True/False>] [--use_homography <True/False>] [--use_torch <True/False>] [--num_threads <num_threads>]

Parameters:
    vnir_hdr (str): Path to the VNIR data with the waterfall band in the last channel.
    swir_hdr (str): Path to the SWIR data with the waterfall band in the last channel.
    mica_hdr (str): Path to the Micasense data stacked layer in order of RGB-RE-NIR.
    outfolder (str): Directory to save intermediate and final results.
    manual_warping (bool, optional): Enable manual warping using GUI. Default is False (automatic warping).
    use_homography (bool, optional): Use available homography for coregistration. Default is True.
    use_torch (bool, opt    ional): Enable Torch median filtering. Will use CUDA GPU if available; otherwise, CPU is used. Default is True.
    num_threads (int, optional): Number of threads to use for parallel processing. Defaults to the number of CPU cores available.

Example:
    python splash.py --vnir_hdr /path/to/vnir.hdr --swir_hdr /path/to/swir.hdr --mica_hdr /path/to/mica.hdr --outfolder /path/to/output --manual_warping False --use_homography True --use_torch True --num_threads 4
"""
import geoharmony.tools.gdalwriter
from scripts.coregister import coregister_manual
from scripts.coregister import coregister_automatic
from scripts.shape_shifter import shape_shift_mpp
from scripts.antspy_registration_onraster import main_vnir_swir as antspy_registration_vnir_swir
from scripts.nonground_mask_creator import crop_image_to_extent
from geoharmony.tools.image_matchmaker.image_matchmaker import coregister_gui
from geoharmony.tools.gdalwriter import *
import argparse
import sys
from scripts.utils import *
import geoharmony.tools.gdalimage as gi
from scripts.nonground_mask_creator import *


class Color:
    BLUE = '\033[94m'
    END = '\033[0m'


def splash(vnir_hdr,
           swir_hdr,
           mica_hdr,
           mica_dsm_path,
           out_folder,
           coreg_vnir_band_indices = [25, 70, 115],
           coreg_swir_band_indices = [10, 40, 90],
           coreg_mica_band_indices=[0, 1, 2],
           pixel_shift=3,
           kernel_size=3,
           ss_vnir_band_indices = np.arange(60, 77),
           ss_swir_band_indices = np.arange(0,12),
           ss_vnir_mica_band_index = -2,
           ss_swir_mica_band_index = -1,
           use_torch=True,
           num_threads = None,
           manual_warping = False,
           use_available_homography=True):
    """
    Harmonizes spatial data using the SPLASH technique.

    Steps:
    1) Convert data to uint16 for faster processing.
    2) Upsample Micasense data to VNIR and SWIR resolutions.
    3) Perform coregistration of VNIR and SWIR data with Micasense data.
    4) Set zeros in Micasense image for shapeshifting.
    5) Run shapeshifter on VNIR and SWIR datasets.
    6) Perform Antspy registration.

    Parameters:
        vnir_hdr (str): Path to the VNIR data with the waterfall band in the last channel.
        swir_hdr (str): Path to the SWIR data with the waterfall band in the last channel.
        mica_hdr (str): Path to the Micasense data stacked layer in order of RGB-RE-NIR.
        out_folder (str): Directory to save intermediate and final results.
        coreg_vnir_band_indices (list): Indices for VNIR bands used in coregistration.
        coreg_swir_band_indices (list): Indices for SWIR bands used in coregistration.
        coreg_mica_band_indices (list): Indices for Mica bands used in coregistration.
        pixel_shift (int): Pixel shift value for shapeshifter. Default is 3.
        kernel_size (int): Kernel size for shapeshifter. Default is 3.
        ss_vnir_band_indices (array): Indices for VNIR bands used in shapeshifter.
        ss_swir_band_indices (array): Indices for SWIR bands used in shapeshifter.
        ss_vnir_mica_band_index (int): Mica band index for VNIR shapeshifter. Default is 1.
        ss_swir_mica_band_index (int): Mica band index for SWIR shapeshifter. Default is -1.
        use_torch (bool): Enable Torch median filtering, using CUDA GPU if available. Default is True.
        num_threads (int): Number of threads to use for parallel processing. Defaults to the number of CPU cores available.
        manual_warping (bool): Enable manual warping using GUI. Default is False (automatic warping).
        use_available_homography (bool): Use available homography for coregistration. Default is True.

    Returns:
        None
    """
    #### TODO NEED TO MAKE EVERYTHING GETS PASTED IN THE OUTPUT FOLDER INSTEAD OF WHERE T

    start_time = time.time()
    # Creating an output folder to put intermediate and final results in
    os.makedirs(out_folder, exist_ok=True)

    # 0) lets see if we can save some space and processing time by doing things in uint space
    print(Color.BLUE + "------> Converting all data to UINT16 for faster processing..." + Color.END)

    # load vnir, swir and mica image
    # todo change the name hdr to just filename for the new version
    # need to make sure they all get saved in memory
    mica_gimg = gi.GdalImage(mica_hdr.replace(".hdr", ""))
    vnir_gimg = gi.GdalImage(vnir_hdr.replace(".hdr", ""))
    swir_gimg = gi.GdalImage(swir_hdr.replace(".hdr", ""))
    mica_gimg_u16, _ = gi.gdalimage_to_unit16(mica_gimg, is_mica = True)
    vnir_gimg_u16, vnir_scale_coefficients = gi.gdalimage_to_unit16(vnir_gimg)
    swir_gimg_u16, swir_scale_coefficients = gi.gdalimage_to_unit16(swir_gimg)

    # # 1) upsample each of the datasets to match mica sense resolution
    print(" ")
    print(Color.BLUE + "------> Matching Micasense extent, resolution, and CRS to hyperspectral data..." + Color.END)
    mica_gimg_u16_vnirres = warp_extent_res(mica_gimg_u16, vnir_gimg_u16, "vnirres")
    mica_gimg_u16_swirres = warp_extent_res(mica_gimg_u16, swir_gimg_u16, "swirres")

    print(" ")
    print(Color.BLUE + "------> Performing coregistration..." + Color.END)
    vnir_gimg_u16_warped = coregister_gui(mica_gimg_u16_vnirres,
                                          coreg_mica_band_indices,
                                          vnir_gimg_u16,
                                          coreg_vnir_band_indices,
                                          out_folder,
                                          use_available_homography=use_available_homography,
                                          name = "vnir")
    swir_gimg_u16_warped = coregister_gui(mica_gimg_u16_swirres,
                                          coreg_mica_band_indices,
                                          swir_gimg_u16,
                                          coreg_swir_band_indices,
                                          out_folder,
                                          use_available_homography=use_available_homography,
                                          name = "swir")

    print(" ")
    print(Color.BLUE + "------> Setting zeros in the mica data..." + Color.END)
    # setting zeros in the mica sense image so shapeshifter could do what it is supposed to; writing in place
    mica_gimg_u16_vnirres = set_zeros_base_ref(vnir_gimg_u16_warped, mica_gimg_u16_vnirres)
    mica_gimg_u16_swirres = set_zeros_base_ref(swir_gimg_u16_warped, mica_gimg_u16_swirres)

    # vnir_hdr_warped = "/Volumes/T7/axhcis/Projects/NURI/data/20240719_argetsingerGold/out/raw_0_rd_wr_or_u16_warped.hdr"
    # swir_hdr_warped = "/Volumes/T7/axhcis/Projects/NURI/data/20240719_argetsingerGold/out/raw_21872_nuc_rd_wr_or_u16_warped.hdr"
    # mica_hdr_vnirres = "/Volumes/T7/axhcis/Projects/NURI/data/20240719_argetsingerGold/out/NURI_Gold_transparent_reflectance_all_u16_vnirres.hdr"
    # mica_hdr_swirres = "/Volumes/T7/axhcis/Projects/NURI/data/20240719_argetsingerGold/out/NURI_Gold_transparent_reflectance_all_u16_swirres.hdr"
    # mica_mask = "/Volumes/T7/axhcis/Projects/NURI/data/20240719_argetsingerGold/micasense/NURI_Gold_dsm_mask.tif"

    # this entire section should be optional
    if mica_dsm_path is not None:
        mica_dsm_gimg = gi.GdalImage(mica_dsm_path)
        mica_mask_gimg = get_mask_gimg(mica_gimg, mica_dsm_gimg)
        mica_mask_gimg_vnirres = warp_extent_res(mica_mask_gimg, vnir_gimg_u16_warped, "vnirres")
        mica_mask_gimg_swirres = warp_extent_res(mica_mask_gimg, swir_gimg_u16_warped, "swirres")
        set_zeros_inimage(vnir_gimg_u16_warped, mica_mask_gimg_vnirres)
        set_zeros_inimage(swir_gimg_u16_warped, mica_mask_gimg_swirres)
    # micamask_hdr_vnirres = "/Volumes/T7/axhcis/Projects/NURI/data/20240719_argetsingerGold/micasense/NURI_Gold_dsm_mask_vnirres.hdr"
    # micamask_hdr_swirres = "/Volumes/T7/axhcis/Projects/NURI/data/20240719_argetsingerGold/micasense/NURI_Gold_dsm_mask_swirres.hdr"


    # # 3) run shapeshifter on both of the datasets
    print(" ")
    print(Color.BLUE + "------> Running Shape Shifter..." + Color.END)
    vnir_hdr_warped_ss, vnir_hdr_warped_ss_qa = shape_shift_mpp(vnir_gimg_u16_warped,
                                                                swir_gimg_u16_warped,
                                                                pixel_shift=pixel_shift, kernel_size=kernel_size,
                                                                hs_bands=ss_vnir_band_indices,  # going with an average of 530-570 region
                                                                mica_band=ss_vnir_mica_band_index, use_torch= use_torch,
                                                                num_threads = num_threads,
                                                                mica_mask_filename = mica_mask_gimg_vnirres)

    swir_hdr_warped_ss, swir_hdr_warped_ss_qa = shape_shift_mpp(swir_hdr_warped,
                                                                mica_hdr_swirres,
                                                                pixel_shift=pixel_shift, kernel_size=kernel_size,
                                                                hs_bands=ss_swir_band_indices,
                                                                mica_band=ss_swir_mica_band_index, use_torch= use_torch, num_threads = num_threads,
                                                                mica_mask_filename = micamask_hdr_swirres)

    # vnir_hdr_warped_ss = "/Volumes/T7/axhcis/Projects/NURI/data/20240719_argetsingerGold/out/raw_0_rd_wr_or_u16_warped_ss.hdr"
    # vnir_hdr_warped_ss_qa = "/Volumes/T7/axhcis/Projects/NURI/data/20240719_argetsingerGold/out/raw_0_rd_wr_or_u16_warped_ss_qa.hdr"
    # swir_hdr_warped_ss = "/Volumes/T7/axhcis/Projects/NURI/data/20240719_argetsingerGold/out/raw_21872_nuc_rd_wr_or_u16_warped_ss.hdr"
    # swir_hdr_warped_ss_qa = "/Volumes/T7/axhcis/Projects/NURI/data/20240719_argetsingerGold/out/raw_21872_nuc_rd_wr_or_u16_warped_ss_qa.hdr"
    # mica_hdr_vnirres = "/Volumes/T7/axhcis/Projects/NURI/data/20240719_argetsingerGold/out/NURI_Gold_transparent_reflectance_all_u16_vnirres.hdr"
    # mica_hdr_swirres = "/Volumes/T7/axhcis/Projects/NURI/data/20240719_argetsingerGold/out/NURI_Gold_transparent_reflectance_all_u16_swirres.hdr"
    print(" ")
    print(Color.BLUE + "------> Running Antspy registration..." + Color.END)
    # # # 4) run antspy registration
    # vnir_hdr_warped_ss_reg = antspy_registration(vnir_hdr_warped_ss,
    #                                              mica_hdr_vnirres,
    #                                              hs_bands = ss_vnir_band_indices,
    #                                              mica_band = ss_vnir_mica_band_index,
    #                                              mica_mask_filename = micamask_hdr_vnirres)
    # swir_hdr_warped_ss_reg = antspy_registration(swir_hdr_warped_ss,
    #                                              mica_hdr_swirres,
    #                                              hs_bands = ss_swir_band_indices,
    #                                              mica_band=ss_swir_mica_band_index,
    #                                              mica_mask_filename = micamask_hdr_swirres)

    # 4) run antspy registration between vnir and swir and vnir with the mask
    swir_hdr_warped_ss_vnirres_ereg,\
        swir_hdr_warped_ss_vnirres_ereg_def,\
        swir_hdr_warped_ss_vnirres_ereg_qa,\
        final_mask = antspy_registration_vnir_swir(vnir_hdr_warped_ss,
                                                                  swir_hdr_warped_ss)

    # # lets combine the two maps by error propagation
    from scripts.utils import load_image_envi_fast
    # # getting this to match the vnir-res as well
    swir_hdr_warped_ss_qa_vnires = swir_hdr_warped_ss_qa.replace(".hdr", "_vnirres.hdr")
    crop_image_to_extent(vnir_hdr_warped_ss.replace(".hdr",""),
                         swir_hdr_warped_ss_qa.replace(".hdr",""),
                         swir_hdr_warped_ss_qa_vnires.replace(".hdr",""),
                         "ENVI")

    # save the two images with coefficients applied to make life easier
    scale_and_overwrite_envi(swir_hdr_warped_ss_vnirres_ereg, swir_hdr_warped_ss_vnirres_ereg.replace(".hdr", "_final.hdr"), swir_scale_coefficients, dtype =np.float64)
    scale_and_overwrite_envi(vnir_hdr_warped_ss, vnir_hdr_warped_ss.replace(".hdr", "_final.hdr"), vnir_scale_coefficients, dtype =np.float64)

    # # loading all the images now and calculating the propagated error
    swir_ereg_qa, swir_profile = load_image_envi_fast(swir_hdr_warped_ss_vnirres_ereg_qa)
    swir_ss_qa, _ = load_image_envi_fast(swir_hdr_warped_ss_qa_vnires)
    swir_ereg_qa = swir_ereg_qa.astype(float)
    swir_ss_qa = swir_ss_qa.astype(float)
    swir_qa = (swir_ereg_qa**2 + swir_ss_qa**2)**0.5
    swir_qa[final_mask] = 0

    from scripts.utils import to_uint8
    import copy
    from scripts.utils import save_image_envi
    swir_qa = to_uint8(swir_qa)
    metadata = copy.copy(swir_profile)
    metadata["bands"] = "1"
    metadata["band names"] = "SWIR final QA Image"
    def_hdr_filename = os.path.join(os.path.dirname(swir_hdr_warped_ss_vnirres_ereg_qa),"splash_qa.hdr")
    save_image_envi(swir_qa, metadata, def_hdr_filename, dtype="uint8", ext="")

    # lets read the header file again and rewrite it
    # read the metadata and add the scale to it
    # add_to_envi_header(vnir_hdr_warped_ss, "scale coefficients", vnir_scale_coefficients)
    # add_to_envi_header(swir_hdr_warped_ss_vnirres_ereg, "scale coefficients", swir_scale_coefficients)



def splash_anstpytest(vnir_hdr,
           swir_hdr,
           mica_hdr,
           mica_dsm_path,
           out_folder,
           coreg_vnir_band_indices = [25, 70, 115],
           coreg_swir_band_indices = [10, 40, 90],
           coreg_mica_band_indices=[0, 1, 2],
           pixel_shift=3,
           kernel_size=3,
           ss_vnir_band_indices = np.arange(60, 77),
           ss_swir_band_indices = np.arange(0,12),
           ss_vnir_mica_band_index = -2,
           ss_swir_mica_band_index = -1,
           use_torch=True,
           num_threads = None,
           manual_warping = False,
           use_available_homography=True):
    """
    Harmonizes spatial and spectral data using the SPLASH technique.

    Steps:
    1) Convert data to uint16 for faster processing.
    2) Upsample Micasense data to VNIR and SWIR resolutions.
    3) Perform coregistration of VNIR and SWIR data with Micasense data.
    4) Set zeros in Micasense image for shapeshifting.
    5) Run shapeshifter on VNIR and SWIR datasets.
    6) Perform Antspy registration.

    Parameters:
        vnir_hdr (str): Path to the VNIR data with the waterfall band in the last channel.
        swir_hdr (str): Path to the SWIR data with the waterfall band in the last channel.
        mica_hdr (str): Path to the Micasense data stacked layer in order of RGB-RE-NIR.
        out_folder (str): Directory to save intermediate and final results.
        coreg_vnir_band_indices (list): Indices for VNIR bands used in coregistration.
        coreg_swir_band_indices (list): Indices for SWIR bands used in coregistration.
        coreg_mica_band_indices (list): Indices for Mica bands used in coregistration.
        pixel_shift (int): Pixel shift value for shapeshifter. Default is 3.
        kernel_size (int): Kernel size for shapeshifter. Default is 3.
        ss_vnir_band_indices (array): Indices for VNIR bands used in shapeshifter.
        ss_swir_band_indices (array): Indices for SWIR bands used in shapeshifter.
        ss_vnir_mica_band_index (int): Mica band index for VNIR shapeshifter. Default is 1.
        ss_swir_mica_band_index (int): Mica band index for SWIR shapeshifter. Default is -1.
        use_torch (bool): Enable Torch median filtering, using CUDA GPU if available. Default is True.
        num_threads (int): Number of threads to use for parallel processing. Defaults to the number of CPU cores available.
        manual_warping (bool): Enable manual warping using GUI. Default is False (automatic warping).
        use_available_homography (bool): Use available homography for coregistration. Default is True.

    Returns:
        None
    """
    micamask_hdr_vnirres = "/Volumes/T7/axhcis/Projects/NURI/data/20240719_argetsingerGold/micasense/NURI_Gold_dsm_mask_vnirres.hdr"
    micamask_hdr_swirres = "/Volumes/T7/axhcis/Projects/NURI/data/20240719_argetsingerGold/micasense/NURI_Gold_dsm_mask_swirres.hdr"
    vnir_hdr_warped_ss = "/Volumes/T7/axhcis/Projects/NURI/data/20240719_argetsingerGold/out/raw_0_rd_wr_or_u16_warped_ss.hdr"
    swir_hdr_warped_ss = "/Volumes/T7/axhcis/Projects/NURI/data/20240719_argetsingerGold/out/raw_21872_nuc_rd_wr_or_u16_warped_ss.hdr"
    mica_hdr_vnirres = "/Volumes/T7/axhcis/Projects/NURI/data/20240719_argetsingerGold/out/NURI_Gold_transparent_reflectance_all_u16_vnirres.hdr"
    mica_hdr_swirres = "/Volumes/T7/axhcis/Projects/NURI/data/20240719_argetsingerGold/out/NURI_Gold_transparent_reflectance_all_u16_swirres.hdr"
    print(" ")
    import matplotlib.pyplot as plt

    def save_output_in_folders(default_output, test_params, vnir_hdr, swir_hdr):
        # Define the base directory to save all outputs
        base_dir = "/Volumes/T7/axhcis/Projects/NURI/data/20240719_argetsingerGold/elasticreg_hyperparams"
        os.makedirs(base_dir, exist_ok=True)

        # Save the default output in a 'default' folder
        default_folder = os.path.join(base_dir, "default")
        os.makedirs(default_folder, exist_ok=True)
        plt.imsave(os.path.join(default_folder, "default_output.png"), default_output)
        print(f"Saved default output in {default_folder}")

        # Loop through each parameter independently
        for param, values in test_params.items():
            # Create a folder for each parameter
            param_folder = os.path.join(base_dir, param)
            os.makedirs(param_folder, exist_ok=True)

            for value in values:
                # Set parameters for each run
                kwargs = {param: value}

                # Run the main function with the specified parameter and get the output array
                output_array = antspy_registration_vnir_swir_test(
                    vnir_hdr=vnir_hdr,
                    swir_hdr=swir_hdr,
                    **kwargs
                )

                # Save the output image with a descriptive filename
                output_filename = f"{param}_{value}.png"
                output_path = os.path.join(param_folder, output_filename)
                plt.imsave(output_path, output_array)
                print(f"Saved output for {param}={value} in {output_path}")

    # Example usage
    # Define the test parameters
    test_params = {
        # "flow_sigma": [1.0, 2.0, 5, 10],
        # "syn_metric": ["CC", "meansquares", "mattes", "demons"],
        # "smoothing_mm": [True, False],
        "method": ["Elastic", "ElasticSyN", "SyN", "SyNRA", "SyNOnly", "SyNCC", "SyNabp", "SyNBold", "SyNBoldAff", "SyNAggro"]
    }

    # you dont need to spit these out
    bands_warped_rgb, mywarpedgrid, quality_raster = main_vnir_swir(
        vnir_hdr=vnir_hdr_warped_ss,
        swir_hdr=swir_hdr_warped_ss,
        mica_mask_filename = micamask_hdr_vnirres
    )
    print("Generated default output.")


    # calculating combined quality raster with error porpagation
    # (SS error, )

    # Save all results in separate folders
    # save_output_in_folders(default_output, test_params, vnir_hdr_warped_ss, swir_hdr_warped_ss)
    print("Saved all parameter tuning results in separate folders.")



if __name__ == "__main__":

    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(
            description='Run the SPLASH technique for spatial harmonization and registration of hyperspectral images.')
        parser.add_argument('--vnir_hdr', type=str, required=True,
                            help='Path to VNIR HDR file')
        parser.add_argument('--swir_hdr', type=str, required=True,
                            help='Path to SWIR HDR file')
        parser.add_argument('--mica_hdr', type=str, required=True,
                            help='Path to MICA HDR file')
        parser.add_argument('--mica_dsm', type=str, required=True,
                            help='Path to MICA DSM file')
        parser.add_argument('--outfolder', type=str, required=True,
                            help='Output folder')
        parser.add_argument('--coreg_vnir_band_indices', nargs='+', type=int, default=[25, 70, 115],
                            help='VNIR band indices for coregistration')
        parser.add_argument('--coreg_swir_band_indices', nargs='+', type=int, default=[10, 40, 90],
                            help='SWIR band indices for coregistration')
        parser.add_argument('--coreg_mica_band_indices', nargs='+', type=int, default=[0, 1, 2],
                            help='MICA band indices for coregistration')
        parser.add_argument('--pixel_shift', type=int, default=3,
                            help='Pixel shift value for shapeshifter')
        parser.add_argument('--kernel_size', type=int, default=3,
                            help='Kernel size for shapeshifter')
        parser.add_argument('--ss_vnir_band_start', type=int, required=True,
                            help='Start index for VNIR shapeshifter bands')
        parser.add_argument('--ss_vnir_band_end', type=int, required=True,
                            help='End index for VNIR shapeshifter bands')
        parser.add_argument('--ss_vnir_mica_band', type=int, default=1,
                            help='MICA band index for VNIR shapeshifter')
        parser.add_argument('--ss_swir_band_start', type=int, required=True,
                            help='Start index for SWIR shapeshifter bands')
        parser.add_argument('--ss_swir_band_end', type=int, required=True,
                            help='End index for SWIR shapeshifter bands')
        parser.add_argument('--ss_swir_mica_band', type=int, default=-1,
                            help='MICA band index for SWIR shapeshifter')
        parser.add_argument('--use_torch', action="store_true",
                            help='Enable Torch median filtering')
        parser.add_argument('--num_threads', type=int, default=os.cpu_count(),
                            help='Number of threads for parallel processing')
        parser.add_argument('--manual_warping', action="store_true",
                            help='Enable manual warping using GUI')
        parser.add_argument('--use_homography', action="store_true",
                            help='Use available homography for coregistration')

        args = parser.parse_args()

        ss_vnir_band_indices = np.arange(args.ss_vnir_band_start, args.ss_vnir_band_end)
        ss_swir_band_indices = np.arange(args.ss_swir_band_start, args.ss_swir_band_end)


        # TODO you need to change the mica_dsm_fie
        splash(args.vnir_hdr,
               args.swir_hdr,
               args.mica_hdr,
               args.mica_dsm_file,
               args.outfolder,
               coreg_vnir_band_indices=args.coreg_vnir_band_indices,
               coreg_swir_band_indices=args.coreg_swir_band_indices,
               coreg_mica_band_indices=args.coreg_mica_band_indices,
               pixel_shift=args.pixel_shift,
               kernel_size=args.kernel_size,
               ss_vnir_band_indices=ss_vnir_band_indices,
               ss_swir_band_indices=ss_swir_band_indices,
               ss_vnir_mica_band_index=args.ss_vnir_mica_band,
               ss_swir_mica_band_index=args.ss_swir_mica_band,
               use_torch=args.use_torch,
               num_threads=args.num_threads,
               manual_warping=args.manual_warping,
               use_available_homography=args.use_homography)

    else:
        vnir_hdr = "/Volumes/T7/axhcis/Projects/NURI/test_data/step2_splash/vnir/raw_0_rd_wr_or_small.hdr"
        swir_hdr = "/Volumes/T7/axhcis/Projects/NURI/test_data/step2_splash/swir/raw_21872_nuc_rd_wr_or_small.hdr"
        mica_hdr = "/Volumes/T7/axhcis/Projects/NURI/test_data/step2_splash/micasense/NURI_Gold_transparent_reflectance_all_small.hdr"
        mica_dsm = "/Volumes/T7/axhcis/Projects/NURI/test_data/step2_splash/micasense/NURI_Gold_dsm_small.tif"
        out_folder = "/Volumes/T7/axhcis/Projects/NURI/test_data/step2_splash/out"
        use_torch = False
        num_threads = 1
        manual_warping = True
        use_homography = True
        # todo add an option to get rid of intermediate files
        # todo change back to ss_vnir_mica_band_index to 1 (green)
        splash(vnir_hdr,
               swir_hdr,
               mica_hdr,
               mica_dsm,
               out_folder,
               use_torch = use_torch,
               num_threads = num_threads,
               manual_warping = manual_warping,
               use_available_homography= use_homography)

