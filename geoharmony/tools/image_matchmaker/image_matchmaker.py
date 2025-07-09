# main.py
import matplotlib.pyplot as plt
import multiprocessing as mp
from get_points_app_v2 import run_get_points_app
from show_warped_app import run_show_warped_app
import numpy as np
import os
import cv2


def launch_get_points_app(img1, img2):
    mgr = mp.Manager()
    q = mgr.Queue()
    p = mp.Process(target=run_get_points_app, args=(img1, img2, q), daemon=True)
    p.start()
    return q, p

def launch_show_warped_app(image):
    mgr = mp.Manager()
    q = mgr.Queue()
    p = mp.Process(target=run_show_warped_app, args=(image, q), daemon=True)
    p.start()
    return q, p


def calc_homography(ref_points, target_points):
    # point passed to homography should be x, y order
    ref_points = np.array(ref_points)
    target_points = np.array(target_points)
    M, mask = cv2.findHomography(ref_points, target_points, cv2.RANSAC, 5)

    return M

    # run a small app that shows the target image warped to the reference image and shows the result
    # if the use likes it we return the homography matrix if not we restart the process
    ### APP2 BASICALLY WARPS THE TARGET IMAGE TO THE REFERENCE IMAGE USING THE HOMOGRAPHY MATRIX AND SHOWS IT TO THE USER

    # Convert to float32 for accurate blending
    mica_float = mica_image_uint8.astype(np.float32)
    hs_float = hs_warped.astype(np.float32)
    # Blend the two images with 50% transparency each
    composite = 0.5 * mica_float + 0.5 * hs_float
    # Clip and convert back to uint8
    composite_uint8 = np.clip(composite, 0, 255).astype(np.uint8)

    # launch the show warped app here
    q, dash_process = launch_get_points_app(ref_image, target_image)
    ref_pts, tgt_pts = q.get()
    dash_process.join()



    if not satisfied:
        # we need to go back to the point picking GUI again so this function should return either a homography matrix or None
        return None

    # if the user is satisfied we return the homography matrix and save oout the image


    ### write out saving out the warped image to file; incorporate in gdal image?

    return M

def get_points_gui(ref_image, target_image):
    """
    This function launches the Dash app for manual point picking.
    It uses multiprocessing to run the app in a separate process and returns the picked points.

    Parameters:
        ref_image (numpy.ndarray): The reference image for point picking.
        target_image (numpy.ndarray): The target image for point picking.

    Returns:
        tuple: A tuple containing two lists of points:
               - ref_pts: Points picked on the reference image.
               - tgt_pts: Points picked on the target image.

    """

    # launch GUI
    q, dash_process = launch_get_points_app(ref_image, target_image)

    # block until user clicks Submit
    ref_pts, tgt_pts = q.get()
    dash_process.join()

    return ref_pts, tgt_pts

def get_warped_gui(image):
    """
    This function launches the Dash app for manual point picking.
    It uses multiprocessing to run the app in a separate process and returns the picked points.

    Parameters:
        ref_image (numpy.ndarray): The reference image for point picking.
        target_image (numpy.ndarray): The target image for point picking.

    Returns:
        tuple: A tuple containing two lists of points:
               - ref_pts: Points picked on the reference image.
               - tgt_pts: Points picked on the target image.

    """

    # launch GUI
    q, dash_process = launch_show_warped_app(image)

    # block until user clicks Submit
    response = q.get()
    dash_process.join()

    return response


def coregister_gui(ref_gdalimage,
                      ref_bands,
                      target_gdalimage,
                      target_bands,
                      outfolder,
                      use_available_homography=False,
                      name=""):
    """
    Coregister two images using manual point picking.
    the target image is warped to match the reference image using the picked points.

    homorgraphy is saved everytime is ran so it can be resused later.
    name is the  extension added at the end of the warped and coregistered iamge.
    """
    # lets generate unint 8 images for display
    ref_arr_uint8 = ref_gdalimage.read_display_image()
    target_arr_uint8 = target_gdalimage.read_display_image()

    # homography file name
    homography_path = os.path.join(outfolder, f"homography_{name}.npy")

    if os.path.exists(homography_path) and use_available_homography:
        h_mat = np.load(homography_path)
    else:
        response =  False
        while not response:
            # run the point picking GUI and get the points
            ref_points, target_points = get_points_gui(ref_arr_uint8, target_arr_uint8)

            # calculate the homography matrix
            h_mat = calc_homography(ref_points, target_points)

            # warp the target image using the homography matrix
            tareget_arr_warped = cv2.warpPerspective(target_arr_uint8, h_mat, (ref_arr_uint8.shape[1], ref_arr_uint8.shape[0]))

            # blend the two images to show them with 50% transparency each
            mica_float = ref_arr_uint8.astype(np.float32)
            hs_float = tareget_arr_warped.astype(np.float32)
            composite = 0.5 * mica_float + 0.5 * hs_float
            composite_uint8 = np.clip(composite, 0, 255).astype(np.uint8)

            # send the new image to the GUI to show the warped image, if the response is True we move on if not we repeat the process
            response = get_warped_gui(composite_uint8)

    # if successful we save the homography matrix to file
    np.save(homography_path, h_mat)

    # save the warped image to file using GDALimage
    from geoharmony.tools.gdalwriter import save_crop_warp
    target_gdalimage_warped = save_crop_warp(target_gdalImage = target_gdalimage,
                                             ref_gdalImage = ref_gdalimage,
                                             homography_matrix= h_mat)

    # save the warped image to file




if __name__ == "__main__":
    # load your images
    image1 = plt.imread("/Users/amirhassanzadeh/Downloads/photo-1548407260-da850faa41e3.jpeg")
    image2 = plt.imread("/Users/amirhassanzadeh/Downloads/photo-1548407260-da850faa41e3.jpeg")

    # launch GUI
    q, dash_proc = launch_point_picker(image1, image2)

    # block until user clicks Submit
    ref_pts, tgt_pts = q.get()
    dash_proc.join()

    print("Reference points:", ref_pts)
    print("Target points:   ", tgt_pts)
    # continue with your workflow...