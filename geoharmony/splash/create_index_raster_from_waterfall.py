from spectral.io import envi
import numpy as np
import argparse
from geoharmony.tools.gdalwriter import save_image_envi_add_wr
from geoharmony.tools.gdalimage import GdalImage


def main(envi_file):
    envi_file_hdr = envi_file + ".hdr"
    hs_ds = envi.open(envi_file_hdr)
    hs_arr = np.moveaxis(hs_ds.load(), 2, 0)  # move bands to last dimension

    # creating a mesh grid
    rows_vector = np.arange(1, hs_arr.shape[1] + 1)
    columns_vector = np.arange(1, hs_arr.shape[2] + 1)
    rows, _ = np.meshgrid(rows_vector, columns_vector, indexing = "ij")
    rows = rows[None,...]

    output_path = envi_file + "_wr"
    save_image_envi_add_wr(hs_arr, GdalImage(envi_file), rows, output_path)
