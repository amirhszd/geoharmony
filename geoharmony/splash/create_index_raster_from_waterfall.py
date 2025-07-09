from spectral.io import envi
import numpy as np
import os

def load_image_envi(waterfall_path):
    vnir_ds = envi.open(waterfall_path)
    vnir_profile = vnir_ds.metadata
    vnir_wavelengths = vnir_profile["wavelength"]
    vnir_wavelengths = np.array([float(i) for i in vnir_wavelengths])
    vnir_arr = vnir_ds.load()

    return vnir_arr, vnir_profile, vnir_wavelengths

def save_image_envi(new_arr, new_path, old_profile):
    # replicating vnir metadata except the bands and wavelength
    metadata = {}
    for k, v in old_profile.items():
        if (k != "bands") or (k != "wavelength"):
            metadata[k] = old_profile[k]
    metadata["bands"] = str(int(old_profile["bands"]) + 1)
    added_wl = str(float(old_profile["wavelength"][-1]) + 2)
    old_profile["wavelength"].extend([added_wl])
    metadata["wavelength"] = old_profile["wavelength"]
    metadata["description"] = new_path

    data_types = {
        1: [np.uint8, "numpy.uint8"],
        2: [np.int16, "numpy.int16"],
        3: [np.int32, "numpy.int32"],
        4: [np.float32, "numpy.float32"],
        5: [np.float64, "numpy.float64"],
        6: [np.complex64, "numpy.complex64"],
        9: [np.complex128, "numpy.complex128"],
        12: [np.uint16, "numpy.uint16"],
        13: [np.uint32, "numpy.uint32"],
        14: [np.int64, "numpy.int64"],
        15: [np.uint64, "numpy.uint64"]
    }

    # new_arr = new_arr.astype(data_types[int(old_profile["data type"])][0])
    envi.save_image(new_path, new_arr, metadata=metadata, force=True,
                    interleave= old_profile["interleave"],
                    dtype = data_types[int(old_profile["data type"])][0],
                    ext = None)

    # copy the hdr file and just change the number of bands,


    print("image saved to: " + new_path)


def main(waterfall_hdr):
    vnir_arr, vnir_profile, vnir_wavelengths = load_image_envi(waterfall_hdr)

    # creating a mesh grid
    rows_vector = np.arange(1, vnir_arr.shape[0] + 1)
    columns_vector = np.arange(1, vnir_arr.shape[1] + 1)
    rows, columns = np.meshgrid(rows_vector, columns_vector, indexing = "ij")
    rows = rows[...,None]

    rows_cols_ind_raster = np.concatenate([vnir_arr, rows], 2)

    output_path = waterfall_hdr.replace(".hdr","_wr.hdr")
    save_image_envi(rows_cols_ind_raster, output_path, vnir_profile)

    headwall_stuff = []
    with open(waterfall_hdr, "r") as file:
        for line in file:
            if line.startswith(";"):
                headwall_stuff.append(line)

    headwall_stuff = "".join(headwall_stuff)
    with open(output_path, "a") as file:
        file.write(headwall_stuff)

if __name__ == "__main__":

    ##### THIS IS REPEATED!!!!
    waterfall_hdr = "/dirs/data/tirs/axhcis/Projects/NURI/Data/20210723_tait_labsphere_processed/1133/swir/raw_1504_nuc_rd.hdr"
    main(waterfall_hdr)