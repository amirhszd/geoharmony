import os
import cv2
import rasterio
from spectral.io import envi
from tqdm import tqdm
import time
import numpy as np

from geoharmony.tools.gdalimage import GdalImage, warp_extent_res


def to_uint8(x):
    if len(x[np.nonzero(x)]) > 0:
        return ((x - np.min(x[np.nonzero(x)])) / (np.max(x[np.nonzero(x)]) - np.min(x[np.nonzero(x)])) * 255).astype(np.uint8)
    else:
        return x.astype(np.uint8)


# def save_image_envi_add_wr(new_arr, new_path, old_profile):
#     # replicating vnir metadata except the bands and wavelength
#     metadata = {}
#     for k, v in old_profile.items():
#         if (k != "bands") or (k != "wavelength"):
#             metadata[k] = old_profile[k]
#     metadata["bands"] = str(int(old_profile["bands"]) + 1)
#     added_wl = str(float(old_profile["wavelength"][-1]) + 2)
#     old_profile["wavelength"].extend([added_wl])
#     metadata["wavelength"] = old_profile["wavelength"]
#     metadata["description"] = new_path
#
#
#     # new_arr = new_arr.astype(data_types[int(old_profile["data type"])][0])
#     envi.save_image(new_path, new_arr, metadata=metadata, force=True,
#                     interleave= old_profile["interleave"],
#                     dtype = envi_datatypes[int(old_profile["data type"])][0],
#                     ext = None)
#
#     # copy the hdr file and just change the number of bands,
#
#
#     print("image saved to: " + new_path)

def load_image_envi(waterfall_path, profile_only=False):
    """
    Load an image from an ENVI file.

    Parameters:
    - waterfall_path (str): Path to the ENVI file.

    Returns:
    - vnir_arr (numpy.ndarray): Loaded image data as a NumPy array.
    - vnir_profile (dict): Metadata/profile of the ENVI file.
    """
    hs_ds = envi.open(waterfall_path)  # Open the ENVI file
    hs_profile = hs_ds.metadata  # Extract metadata
    if not profile_only:
        vnir_arr = np.array(hs_ds.load())  # Load image data into a NumPy array
    else:
        vnir_arr = None

    try:
        # First attempt to get "wavelength"
        vnir_wavelengths = hs_profile["wavelength"]
        vnir_wavelengths = np.array([float(i) for i in vnir_wavelengths])
    except KeyError:
        try:
            # Second attempt to get "band names" if "wavelength" doesn't exist
            vnir_wavelengths = hs_profile["band names"]
            vnir_wavelengths = np.array([float(i) for i in vnir_wavelengths])
        except (KeyError, ValueError):
            # Set to None if both attempts fail
            vnir_wavelengths = None

    return vnir_arr, hs_profile, vnir_wavelengths

def load_image_envi_fast(waterfall_path, bands=None, hs = False):
    """
    Load an image from an ENVI file using rasterio. THIS IS MUCH FASTER THAN SPECTRAL ENVI.

    Parameters:
    - waterfall_path (str): Path to the ENVI file.

    Returns:
    - vnir_arr (numpy.ndarray): Loaded image data as a NumPy array.
    - vnir_profile (dict): Metadata/profile of the ENVI file.
    """
    vnir_ds = envi.open(waterfall_path)  # Open the ENVI file
    vnir_profile = vnir_ds.metadata

    img_path = waterfall_path.split(".hdr")[0] + '.img' if os.path.isfile(waterfall_path.split(".hdr")[0] + '.img') else waterfall_path.split(".hdr")[0]
    with rasterio.open(img_path) as src:
        if bands is None:
            hs_arr = np.moveaxis(src.read(),0,-1)
        else:
            # rasterio takes band numbers to gotta pass +1
            bands = bands + 1
            bands = list(bands)
            if hs:
                bands.append(int(vnir_profile['bands']))
            hs_arr = np.moveaxis(src.read(bands), 0, -1)
    return hs_arr, vnir_profile


###################


def save_image_envi(new_hs_arr, old_hs_profile, hs_hdr_path, dtype=None, ext = None):
    """
    Save a hyperspectral image to an ENVI file.

    Parameters:
    - new_hs_arr (numpy.ndarray): New hyperspectral image data.
    - old_hs_profile (dict): Metadata/profile of the original hyperspectral image.
    - hs_hdr_path (str): Path to save the ENVI file.
    - dtype (dtype, optional): Data type of the image data.

    Returns:
    - None
    """

    if ext == None:
        ext = ".img"
    else:
        ext = ""
    # Capture the start time
    start_time = time.time()

    # Update the description in the metadata
    old_hs_profile["description"] = hs_hdr_path.replace(".hdr","")

    # Save the new hyperspectral image to the ENVI file
    envi.save_image(hs_hdr_path, new_hs_arr, metadata=old_hs_profile, force=True, dtype=dtype, ext = ext)

    # Print a message indicating the successful saving of the image and time taken
    print(f"Image saved to: {hs_hdr_path} in {(time.time() - start_time) / 60:.2f} minutes")

def scale_and_overwrite_envi(hs_hdr, output_hs_hdr, coefficients, dtype = np.float64, ext= ""):
    """
    Save a hyperspectral image to an ENVI file.

    Parameters:
    - new_hs_arr (numpy.ndarray): New hyperspectral image data.
    - old_hs_profile (dict): Metadata/profile of the original hyperspectral image.
    - hs_hdr_path (str): Path to save the ENVI file.
    - dtype (dtype, optional): Data type of the image data.

    Returns:
    - None
    """

    ds = envi.open(hs_hdr)
    data = ds.load()
    scaled_data = np.zeros((ds.shape[0], ds.shape[1], ds.shape[2] - 1), dtype = dtype)
    for coefficient, band_n in zip(coefficients, range(ds.nbands - 1)):
        scaled_data[..., band_n] = (data[...,band_n]*coefficient).squeeze()

    # Save the new hyperspectral image to the ENVI file
    # drop the last band from it
    metadata = ds.metadata
    metadata['bands'] = str(int(metadata['bands']) - 1)
    try:
        del metadata['band names'][-1]
    except:
        del metadata['wavelength'][-1]
    envi.save_image(output_hs_hdr, scaled_data, metadata=metadata, force=True, dtype=dtype, ext= ext)

    # Print a message indicating the successful saving of the image and time taken
    print(f"Image saved to: {hs_hdr} ")


def set_zeros_inimage(hs_hdr, mica_hdr, dtype = "uint16"):
    """
    Sets zero values in the Micasense image based on zero values in the hyperspectral image.

    Parameters:
        hs_hdr (str): Path to the hyperspectral image header file.
        mica_hdr (str): Path to the Micasense image header file.
    """

    # Load the first image
    hs_arr, hs_profile = load_image_envi_fast(hs_hdr)
    mica_arr, mica_profile = load_image_envi_fast(mica_hdr)

    # Check if the two arrays are the same size
    if hs_arr.shape[:2] != mica_arr.shape[:2]:
        raise ValueError("arrays are not the same size!")

    # Find indices where values are zero and setting it to zero
    try:
        zero_indices = np.nanmean(hs_arr,-1) == 0
        mica_arr[zero_indices,: ] = 0
        save_image_envi(mica_arr, mica_profile, mica_hdr, ext = "", dtype=dtype)
    except:
        print("Did not find zeros in the reference image! Moving on.")



def load_mica_hdr_envi(mica_path, swir_path):

    mica_ds = envi.open(mica_path)
    mica_profile = mica_ds.metadata
    mica_wavelengths = [475,560,634,668,717]
    mica_arr = mica_ds.load()

    swir_ds = envi.open(swir_path)
    swir_profile = swir_ds.metadata
    swir_wavelengths = swir_profile["wavelength"]
    swir_wavelengths = np.array([float(i) for i in swir_wavelengths])
    swir_arr = swir_ds.load()

    return (mica_arr, mica_profile, mica_wavelengths ), (swir_arr, swir_profile, swir_wavelengths)


def load_mica_hdr_rasterio(mica_path, swir_path):

    with rasterio.open(mica_path.split(".hdr")[0]) as src:
        _, mica_profile, mica_wavelengths = load_image_envi(mica_path, profile_only=True)
        mica_arr = np.moveaxis(src.read(),0,-1)
    with rasterio.open(swir_path.split(".hdr")[0]) as src:
        _, swir_profile, swir_wavelengths = load_image_envi(swir_path, profile_only=True)
        swir_arr = np.moveaxis(src.read(),0,-1)

    return (mica_arr, mica_profile, mica_wavelengths ), (swir_arr, swir_profile, swir_wavelengths)


def load_images_rasterio(vnir_path, swir_path):
    with rasterio.open(vnir_path) as src:
        vnir_arr = src.read()
        vnir_profile = src.profile
        # vnir_wavelengths = np.array([float(i.split(" ")[0]) for i in src.descriptions])
        vnir_wavelengths = 1
    with rasterio.open(swir_path) as src:
        swir_arr = src.read()
        swir_profile = src.profile
        # swir_wavelengths = np.array([float(i.split(" ")[0]) for i in src.descriptions])
        swir_wavelengths = 1

    return (vnir_arr, vnir_profile, vnir_wavelengths), (swir_arr, swir_profile, swir_wavelengths)


def save_image_homography(swir_arr, swir_wavelengths, swir_path, vnir_arr, vnir_profile, M):
    swir_registered_bands = []
    for i in range(len(swir_wavelengths)):
        swir_registered_bands.append(
            cv2.warpPerspective(np.fliplr(swir_arr[i]), M, (vnir_arr.shape[2], vnir_arr.shape[1])))

    # save data
    import os
    # output_path = swir_path.replace(".tif", "_warped.tif")
    output_path = os.path.basename(swir_path.replace(".tif", "_warped.tif"))
    vnir_profile.update(count=len(swir_registered_bands))
    with rasterio.open(output_path, 'w', **vnir_profile) as dst:
        for i, band in enumerate(swir_registered_bands):
            dst.write_band(i + 1, band)

    from gdal_set_band_description import set_band_descriptions
    bands = [int(i) for i in range(1, len(swir_wavelengths) + 1)]
    names = swir_wavelengths.astype(str)
    band_desciptions = zip(bands, names)
    set_band_descriptions(output_path, band_desciptions)


    print("Registered Image Saved to " + output_path)
    return output_path


def save_crop_image_envi(target_arr, target_profile, target_path, ref_arr, ref_profile, M, dtype ="uint16"):


    target_wavelengths = target_profile["wavelength"] if "wavelength" in target_profile else target_profile["band names"]

    # a sample warped band to get the extents of the image
    sample_warped_band = cv2.warpPerspective(target_arr[..., int(target_arr.shape[-1] / 2)], M, (ref_arr.shape[1], ref_arr.shape[0]))[..., None]

    # Extract non-zero data extents
    xmin, xmax, ymin, ymax = 0, sample_warped_band.shape[1], 0, sample_warped_band.shape[0]

    # creating lat and lon rasters and then cropping them accordingly
    lon_values = np.linspace(float(ref_profile["map info"][3]),
                             float(ref_profile["map info"][3]) + (int(ref_profile["samples"])) * float(ref_profile["map info"][5]),
                             int(ref_profile["samples"]))
    lat_values = np.linspace(float(ref_profile["map info"][4]),
                             float(ref_profile["map info"][4]) - (int(ref_profile["lines"])) * float(ref_profile["map info"][6]),
                             int(ref_profile["lines"]))

    lon_raster, lat_raster = np.meshgrid(lon_values, lat_values)
    lon_raster_crop = lon_raster[ymin:ymax+1, xmin:xmax+1]
    lat_raster_crop = lat_raster[ymin:ymax+1, xmin:xmax+1]
    del lon_raster, lat_raster

    # warping and cropping the data to the extents
    swir_registered_bands = []
    for i in tqdm(range(len(target_wavelengths))):
        swir_registered_bands.append(
            cv2.warpPerspective(target_arr[...,i], M, (ref_arr.shape[1], ref_arr.shape[0]), flags= cv2.INTER_NEAREST)[ymin:ymax + 1, xmin:xmax + 1, None])

    # replicating vnir metadata except the bands and wavelength
    metadata = {}
    for k, v in ref_profile.items():
        metadata[k] = ref_profile[k]

    metadata["bands"] = str(len(target_wavelengths))
    del metadata["band names"]
    metadata["wavelength"] = [str(i) for i in target_wavelengths]
    metadata["map info"][3] = lon_raster_crop.min()
    metadata["map info"][4] = lat_raster_crop.max()
    metadata["description"] = target_path.replace(".hdr", "_warped.hdr")

    swir_registered_bands = np.concatenate(swir_registered_bands, 2)
    envi.save_image(target_path.replace(".hdr", "_warped.hdr"), swir_registered_bands, metadata=metadata, force=True, ext ="")

    print("image saved to: " + target_path.replace(".hdr", "_warped.hdr"))

    return target_path.replace(".hdr", "_warped.hdr")


def add_to_envi_header(hs_hdr_path, key, value):
    hs_meta = envi.read_envi_header(hs_hdr_path)
    hs_meta[key] = value
    envi.write_envi_header(hs_hdr_path, hs_meta)


if __name__ == "__main__":
    vnir_gi = GdalImage("/Volumes/T7/axhcis/Projects/NURI/test_data/step2_splash/vnir/raw_0_rd_wr_or_small")
    swir_gi = GdalImage("/Volumes/T7/axhcis/Projects/NURI/test_data/step2_splash/swir/raw_21872_nuc_rd_wr_or_small")
    warp_extent_res(swir_gi, vnir_gi, "ss")
