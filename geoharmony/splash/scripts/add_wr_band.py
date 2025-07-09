import numpy as np
import sys
import argparse
from .utils import load_image_envi, save_image_envi_add_wr


def main(waterfall_hdr):
    vnir_arr, vnir_profile, vnir_wavelengths = load_image_envi(waterfall_hdr)

    # creating a mesh grid
    rows_vector = np.arange(1, vnir_arr.shape[0] + 1)
    columns_vector = np.arange(1, vnir_arr.shape[1] + 1)
    rows, _ = np.meshgrid(rows_vector, columns_vector, indexing = "ij")
    rows = rows[...,None]

    rows_cols_ind_raster = np.concatenate([vnir_arr, rows], 2)

    output_path = waterfall_hdr.replace(".hdr","_wr.hdr")
    save_image_envi_add_wr(rows_cols_ind_raster, output_path, vnir_profile)

    headwall_stuff = []
    with open(waterfall_hdr, "r") as file:
        for line in file:
            if line.startswith(";"):
                headwall_stuff.append(line)

    headwall_stuff = "".join(headwall_stuff)
    with open(output_path, "a") as file:
        file.write(headwall_stuff)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(
            description='Add waterfall row band to the image file; this would later be processed by headwall propietry software and orthorectified as part of splash model.')
        parser.add_argument('--hs_hdr', type=str, required=True,
                            help='Path to the hyperspectral image header file.')
        args = parser.parse_args()
        waterfall_hdr = args.waterfall_hdr
    else:
        # Default header path if no arguments are provided
        waterfall_hdr = "/dirs/data/tirs/axhcis/Projects/NURI/Data/20210723_tait_labsphere_processed/1133/vnir/raw_0_rd.hdr"

    main(waterfall_hdr)