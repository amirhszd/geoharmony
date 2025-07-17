# main.py
import matplotlib.pyplot as plt
import multiprocessing as mp
from geoharmony.tools.image_matchmaker.get_points_app_v2 import run_get_points_app
from geoharmony.tools.image_matchmaker.show_warped_app import run_show_warped_app
import numpy as np
import os
import cv2
from geoharmony.tools.gdalwriter import save_crop_warp


def launch_get_points_app(img1, img2):
    mgr = mp.Manager()
    q = mgr.Queue()
    p = mp.Process(target=run_get_points_app, args=(img1, img2, q), daemon=True)
    p.start()
    return q, p

def launch_show_warped_app(image1, image2):
    mgr = mp.Manager()
    q = mgr.Queue()
    p = mp.Process(target=run_show_warped_app, args=(image1, image2, q), daemon=True)
    p.start()
    return q, p


def calc_homography(ref_points, target_points):

    ref_points = np.array(ref_points)
    target_points = np.array(target_points)
    M, mask = cv2.findHomography(ref_points, target_points, cv2.RANSAC, 5)

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

    # conver to x,y point pairs
    ref_pts = np.array([[ref_pt["x"], ref_pt["y"]] for ref_pt in ref_pts])
    tgt_pts = np.array([[tgt_pt["x"], tgt_pt["y"]] for tgt_pt in tgt_pts])

    return ref_pts, tgt_pts

def get_warped_gui(image1, image2):
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
    q, dash_process = launch_show_warped_app(image1, image2)

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
    ref_arr_uint8 = np.moveaxis(ref_gdalimage.read_display_image(ref_bands),0, 2)
    target_arr_uint8 = np.moveaxis(target_gdalimage.read_display_image(target_bands), 0, 2)

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
            target_arr_warped = cv2.warpPerspective(target_arr_uint8, h_mat, (ref_arr_uint8.shape[1], ref_arr_uint8.shape[0]))

            # send the new image to the GUI to show the warped image, if the response is True we move on if not we repeat the process
            response = get_warped_gui(ref_arr_uint8, target_arr_warped.astype(np.uint8))
        np.save(homography_path, h_mat)

    # save the warped image to file using GDALimage
    target_gdalimage_warped = save_crop_warp(target_gdalImage = target_gdalimage,
                                             ref_gdalImage = ref_gdalimage,
                                             homography_matrix= h_mat)

    return target_gdalimage_warped




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