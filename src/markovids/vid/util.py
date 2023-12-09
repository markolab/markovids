from scipy import signal
import numpy as np
import cv2

def video_montage(vids, ncols=2):
    nframes, height, width, nchannels = vids[0].shape
    for _vid in vids:
        _nframes, _height, _width, _nchannels = _vid.shape
        if (
            (_nframes != nframes)
            or (_width != width)
            or (_height != height)
            or (_nchannels != nchannels)
        ):
            raise RuntimeError("Video dimensions not consistent")

    dtype = vids[0].dtype
    nrows = np.ceil(len(vids) / ncols).astype("int")
    montage = np.zeros((nframes, height * nrows, width * ncols, nchannels), dtype=dtype)
    montage_height, montage_width = montage.shape[1:3]
    # loop from top left to bottom right
    col = 0
    row = 0
    for _vid in vids:
        montage[:, row : row + height, col : col + width, :] = _vid
        col += width
        if col >= montage_width:
            col = 0
            row += height
    return montage

def sos_filter(x, fps, tau=.01, order=3):
    sos = signal.butter(order, (1 / tau) / (fps / 2), btype="low", output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)

def lp_filter(x, sigma):
    return cv2.GaussianBlur(x, [0, 0], sigma, sigma)

def bp_filter(x, sigma1, sigma2, clip=True):
    return np.clip(
        lp_filter(x, sigma1) - lp_filter(x, sigma2),
        0 if clip == True else -np.inf,
        np.inf,
    )