
import multiprocessing as mp
import matplotlib.pyplot as plt
from show_warped_app import run_show_warped_app
from get_points_app_v2 import run_get_points_app


def launch_show_warped_app(image1, image2):
    mgr = mp.Manager()
    q = mgr.Queue()
    p = mp.Process(target=run_show_warped_app, args=(image1, image2, q), daemon=True)
    p.start()
    return q, p

def launch_get_points_app(img1, img2):
    mgr = mp.Manager()
    q = mgr.Queue()
    p = mp.Process(target=run_get_points_app, args=(img1, img2, q), daemon=True)
    p.start()
    return q, p

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


if __name__ == "__main__":

    img1 = plt.imread("/Users/amirhassanzadeh/Downloads/premium_photo-1677545183884-421157b2da02.jpeg")
    img2 = plt.imread("/Users/amirhassanzadeh/Downloads/photo-1548407260-da850faa41e3.jpeg")

    p1, p2 = get_points_gui(img1, img2)
    print(p1, p2)

    import numpy as np
    response = get_warped_gui(img1, np.random.random(img1.shape))

    print("Reference points:", response)