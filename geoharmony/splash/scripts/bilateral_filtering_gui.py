import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.ndimage import median_filter, binary_dilation
from cv2 import bilateralFilter
from spectral import envi

def load_image_envi(waterfall_path):
    vnir_ds = envi.open(waterfall_path)
    vnir_profile = vnir_ds.metadata
    vnir_arr = np.array(vnir_ds.load())

    return vnir_arr, vnir_profile
def gui(hs_hdr, hs_type):

    if hs_type == "swir":
        hs_bands = np.arange(0, 12)
    elif hs_type == "vnir":
        #TODO: define what bands vnir is
        hs_bands = np.arange(0, 12)

    # load the SWIR and mica
    hs_arr, hs_profile= load_image_envi(hs_hdr)
    hs_arr = np.mean(hs_arr[..., hs_bands], axis=2)
    to_uint8 = lambda x: ((x - np.min(x[np.nonzero(x)])) / (np.max(x[np.nonzero(x)]) - np.min(x[np.nonzero(x)])) * 255).astype(
            np.uint8)
    hs_image = to_uint8(hs_arr)
    output = bilateralFilter(hs_image, 20,150,10)


    fig, ax = plt.subplots(1, 2, figsize=(10, 8))
    plt.subplots_adjust(left=0.1, bottom=0.3)
    ax_hs = ax[0]
    ax_output = ax[1]
    ax_output.set_title('Output')
    hs_im = ax_hs.imshow(hs_image, cmap='gray')
    output_im = ax_output.imshow(output, cmap='gray')

    # Slider positions
    ax_hsf = plt.axes([0.2, 0.2, 0.5, 0.03])
    ax_micaf = plt.axes([0.2, 0.15, 0.5, 0.03])
    ax_qaf = plt.axes([0.2, 0.1, 0.5, 0.03])

    # Slider creation
    s_d = Slider(ax_hsf, 'Kernel Size', 0, 255, valinit=20)
    s_sigmac = Slider(ax_micaf, 'Sigma Color', 0, 255, valinit=150)
    s_sigmas = Slider(ax_qaf, 'Sigma Space', 0, 255, valinit=10)

    def update(val):
        d = s_d.val
        sigmacolor = s_sigmac.val
        sigmaspace = s_sigmas.val

        output = bilateralFilter(hs_image, int(d), sigmacolor, sigmaspace)
        output_im.set_data(output)
        fig.canvas.draw_idle()

    s_d.on_changed(update)
    s_sigmac.on_changed(update)
    s_sigmas.on_changed(update)

    # Link axes for zooming
    ax_hs.callbacks.connect('xlim_changed', lambda ax: ax_output.set_xlim(ax_hs.get_xlim()))
    ax_hs.callbacks.connect('ylim_changed', lambda ax: ax_output.set_ylim(ax_hs.get_ylim()))

    ax_output.callbacks.connect('xlim_changed', lambda ax: ax_hs.set_xlim(ax_output.get_xlim()))
    ax_output.callbacks.connect('ylim_changed', lambda ax: ax_hs.set_ylim(ax_output.get_ylim()))

    plt.show()

if __name__ == "__main__":
    hs_hdr = "/Volumes/T7/axhcis/Projects/NURI/raw_1504_nuc_or_plusindices3_warped_SS.hdr"
    gui(hs_hdr, "swir")

