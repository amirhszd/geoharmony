# main.py
import matplotlib.pyplot as plt
import multiprocessing as mp
from get_points_v2 import run_dash
import numpy as np
import os
import cv2


def calculate_homography_gui(ref_arr,
                             ref_bands,
                             ref_arr_uint8,
                             target_arr,
                             target_bands,
                             target_arr_uint8):

    ###  THIS IS WHERE WE RUN APP GET POINTS APP


    # point passed to homography should be x, y order
    ref_points = np.array(ref_points)
    target_points = np.array(target_points)
    M, mask = cv2.findHomography(ref_points, target_points, cv2.RANSAC, 5)


    #### THIS IS WHEN THE GET POINTS APP IS AND WE CLOSE IT


    ### WE SHOW APP 2

    ### APP2 BASICALLY WARPS THE TARGET IMAGE TO THE REFERENCE IMAGE USING THE HOMOGRAPHY MATRIX AND SHOWS IT TO THE USER
    ax.imshow(mica_image_uint8, alpha=0.5)
    ax.imshow(cv2.warpPerspective(hs_image_uint8, M, (mica_image.shape[1], mica_image.shape[0])),
              alpha=0.5)

    ### TWO BUTTON DOWN BELOW, GOOD TO GO! AND REPEAT! GOOD TO GO WOULD BE GREEN AND REPEAT WOULD BE RED

    ### IF USER IS SATISFRIED WE RETURN THE HOMOGRAPHY MATRIX, CLOSE APP2

    ## IF USE IS NOT SATIFIRTED WE CLOSE APP2 AND RESTART THE PROCESS

    return M


def coregister_manual(ref_gdalimage,
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
    # load the dada
    ref_arr = ref_gdalimage.read()
    target_arr = target_gdalimage.read()

    # lets generate unint 8 images for display
    ref_arr_uint8 = ref_gdalimage.read_display_image()
    target_arr_uint8 = target_gdalimage.read_display_image()

    # homography file name
    homography_path = os.path.join(outfolder, f"homography_{name}.npy")


    if os.path.exists(homography_path) and use_available_homography:
        M = np.load(homography_path)
    else:       # launch the point picker GUI
        M = calculate_homography_manual(ref_arr,
                                        ref_arr_uint8,
                                        target_arr,
                                        target_bands,
                                        target_arr_uint8)
        np.save(homography_path, M)





def launch_point_picker(img1, img2):
    mgr = mp.Manager()
    q   = mgr.Queue()
    p   = mp.Process(target=run_dash, args=(img1, img2, q), daemon=True)
    p.start()
    return q, p

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