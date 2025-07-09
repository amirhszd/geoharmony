import json
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from .utils import save_crop_image_envi, load_mica_hdr_envi, load_mica_hdr_rasterio
from scipy.stats import pearsonr
import cmocean
import json
from tqdm import tqdm
from skimage import data, feature, transform, io, color
from skimage.transform import AffineTransform, warp
from skimage.measure import ransac

def to_uint8(image):
    new_image = np.zeros_like(image)
    min = np.percentile(image, 1)
    max = np.percentile(image, 98)
    for i in range(image.shape[-1]):
        x = image[...,i]
        band = (x - min) / (max - min) * 255
        new_image[...,i] = band
    return new_image.astype(np.uint8)


def init_figs(mica_arr,
              mica_bands,
              hs_arr,
              hs_bands):

    # picking a band close to
    mica_image = mica_arr[..., mica_bands]
    hs_image = hs_arr[..., hs_bands]

    # Create a figure with two subplots in a single row
    fig, axis = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    ax1, ax2 = axis.flatten()

    # convert that to uint8 for cv2
    plt.suptitle("Use the right mouse button to pick points; at least 4. \n"
                 "Close the figure when finished.")
    mica_image_uint8 = to_uint8(mica_image)
    hs_image_uint8 = to_uint8(hs_image)

    # Display the VNIR image on the left subplot
    ax1.imshow(mica_image_uint8, cmap=cmocean.cm.thermal)
    ax1.set_title('Mica Image')

    # Display the SWIR image on the right subplot
    ax2.imshow(hs_image_uint8, cmap=cmocean.cm.thermal)
    ax2.set_title('HSI Image')

    return fig, ax1, ax2, mica_image, mica_image_uint8, hs_image, hs_image_uint8


def calculate_homography_manual(mica_arr, mica_bands, hs_arr, hs_bands):
    global not_satisfied

    not_satisfied = True
    while not_satisfied:
        count_mica = 1
        count_hs = 1
        fig, ax1, ax2, mica_image, mica_image_uint8, hs_image, hs_image_uint8 = init_figs(mica_arr,
                                                                                          mica_bands,
                                                                                          hs_arr,
                                                                                          hs_bands)
        mica_points = []
        hs_points = []

        def on_click_mica(event):
            nonlocal count_mica
            if event.inaxes == ax1 and event.button == 3:  # Left mouse button clicked in mica subplot
                mica_points.append((event.xdata, event.ydata))
                ax1.plot(event.xdata, event.ydata, 'ro')  # Plot a red dot at the clicked point
                ax1.annotate(str(count_mica), (event.xdata, event.ydata), textcoords="offset points", xytext=(0,10), ha='center')
                count_mica += 1
                ax1.figure.canvas.draw_idle()

        def on_click_hs(event):
            nonlocal count_hs
            if event.inaxes == ax2 and event.button == 3:  # Left mouse button clicked in hs subplot
                hs_points.append((event.xdata, event.ydata))
                ax2.plot(event.xdata, event.ydata, 'ro')  # Plot a red dot at the clicked point
                ax2.annotate(str(count_hs), (event.xdata, event.ydata), textcoords="offset points", xytext=(0,10), ha='center')
                count_hs += 1
                ax2.figure.canvas.draw_idle()

        # Connect the mouse click events to the respective axes
        fig.canvas.mpl_connect('button_press_event', on_click_mica)
        fig.canvas.mpl_connect('button_press_event', on_click_hs)
        plt.show()

        # calculate homorgraphy based on points found
        # point passed to homography should be x, y order
        mica_points = np.array(mica_points)
        hs_points = np.array(hs_points)
        M, mask = cv2.findHomography(hs_points, mica_points, cv2.RANSAC, 5)


        # show the result and see if the use is satisfied
        fig, ax = plt.subplots(1,1, figsize=(6,6))
        def on_key(event):
            global not_satisfied
            if event.key == 'escape':  # Close figure if Escape key is pressed
                not_satisfied = False;
                plt.close(fig)
        ax.imshow(mica_image_uint8, alpha=0.5)
        ax.imshow(cv2.warpPerspective(hs_image_uint8, M, (mica_image.shape[1], mica_image.shape[0])),
                  alpha=0.5)
        ax.set_title('Overlay of Coregistered Image \n'
                     'if satisfied press Escape to save image\n'
                     'if NOT satisfied close the figure to restart.')
        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()

    return M


def calculate_homography_automatic(mica_arr, mica_bands, hs_arr, hs_bands):


    # picking a band and converting to uint8
    mica_image = mica_arr[..., mica_bands]
    hs_image = hs_arr[..., hs_bands]
    mica_image_uint8 = to_uint8(mica_image)
    hs_image_uint8 = to_uint8(hs_image)


    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(hs_image_uint8, None)
    keypoints2, descriptors2 = orb.detectAndCompute(mica_image_uint8, None)

    # Match descriptors using the Brute Force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract location of good matches
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography
    M, _ = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)

    return M


def coregister_manual(mica_path, mica_bands, hs_path, hs_bands, use_available_homography = False, name = ""):

    mica_bands = np.array(mica_bands)
    hs_bands = np.array(hs_bands)
    # homography file name
    homography_path = os.path.join(os.path.dirname(hs_path), f"homography_{name}.npy")

    # load images envi
    (mica_arr, mica_profile, _),\
        (hs_arr, hs_profile, _) = load_mica_hdr_rasterio(mica_path, hs_path)

    if os.path.exists(homography_path) and use_available_homography:
        M = np.load(homography_path)
    else:
        M = calculate_homography_manual(mica_arr, mica_bands, hs_arr, hs_bands)
        np.save(homography_path, M)

    output_hdr = save_crop_image_envi(hs_arr, hs_profile, hs_path, mica_arr, mica_profile, M)

    return output_hdr

def coregister_automatic(mica_path, mica_bands, hs_path, hs_bands, use_available_homography = False, name = None):

    mica_bands = np.array(mica_bands)
    hs_bands = np.array(hs_bands)

    # homography file name
    homography_path = os.path.join(os.path.dirname(hs_path), f"homography_{name}.npy")

    # load images envi
    (mica_arr, mica_profile, _),\
        (hs_arr, hs_profile, hs_wavelengths) = load_mica_hdr_envi(mica_path, hs_path)

    if os.path.exists(homography_path) and use_available_homography:
        M = np.load(homography_path)
    else:
        M = calculate_homography_automatic(mica_arr, mica_bands, hs_arr, hs_bands)
        np.save(homography_path, M)


    output_hdr = save_crop_image_envi(hs_arr, hs_wavelengths, hs_path, mica_arr, mica_profile, M)

    return output_hdr



if __name__ == "__main__":
    """
    IMPORORTANT: I AM PASSING FINDING THE HIGHLY CORRELLATED BAND OVER A SMALL AREA OF THE IMAGE, INSTEAD OF THE ENTIRE
    THING, BE USEFUL TO JUST DO IT IN ANOTHER FASHION?
    
    This code is now doing nearest neighbour interpolation!
    """

    mica_path = "/Volumes/T7/axhcis/Projects/DIRSIG/idaho_60x60km_landis_thermal/v1/SixtyMeterBands.img.hdr"
    mica_bands = [0,1,2]
    swir_path = "/Volumes/T7/axhcis/Projects/DIRSIG/idaho_60x60km_landis_thermal/v1/TIR.img.hdr"
    swir_bands = [0,1,2]
    coregister_manual(mica_path,
                      mica_bands,
                      swir_path,
                      swir_bands,
                      True)

