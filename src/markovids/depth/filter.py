import cv2
import numpy as np
from tqdm.auto import tqdm


def clean_frames(
    frames,
    prefilter_space=(3,),
    prefilter_time=None,
    strel_tail=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    iters_tail=None,
    frame_dtype="uint8",
    strel_min=cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    iters_min=None,
    progress_bar=True,
):
    """
    Simple filtering, median filter and morphological opening

    Args:
        frames (3d np array): frames x r x c
        strel (opencv structuring element): strel for morph opening
        iters_tail (int): number of iterations to run opening

    Returns:
        filtered_frames (3d np array): frame x r x c

    """
    # seeing enormous speed gains w/ opencv
    filtered_frames = frames.copy().astype(frame_dtype)

    for i in tqdm(range(frames.shape[0]), disable=not progress_bar, desc="Cleaning frames"):
        if iters_min is not None and iters_min > 0:
            filtered_frames[i, ...] = cv2.erode(filtered_frames[i, ...], strel_min, iters_min)

        if prefilter_space is not None and np.all(np.array(prefilter_space) > 0):
            for j in range(len(prefilter_space)):
                filtered_frames[i, ...] = cv2.medianBlur(
                    filtered_frames[i, ...], prefilter_space[j]
                )

        if iters_tail is not None and iters_tail > 0:
            filtered_frames[i, ...] = cv2.morphologyEx(
                filtered_frames[i, ...], cv2.MORPH_OPEN, strel_tail, iters_tail
            )

    if prefilter_time is not None and np.all(np.array(prefilter_time) > 0):
        for j in range(len(prefilter_time)):
            filtered_frames = signal.medfilt(filtered_frames, [prefilter_time[j], 1, 1])

    return filtered_frames


# filters outliers over time
def time_hampel_filter(
    dat, length=10, mads=4, fill_value="median", correction=1.4826, inplace=True
):
    from scipy import ndimage

    nframes, rows, columns = dat.shape

    if inplace:
        use_dat = dat
    else:
        use_dat = dat.copy()

    med = ndimage.median_filter(use_dat.astype("float32"), size=(length, 1, 1))
    absdev = np.abs(use_dat - med)
    mad = ndimage.median_filter(absdev, size=(length, 1, 1))

    if correction is not None:
        mad *= correction

    absdev /= mad + 1e-3
    outliers = absdev > mads

    if fill_value == "median":
        use_dat[outliers] = med[outliers]
    elif isinstance(fill_value, (float, int)):
        use_dat[outliers] = fill_value

    return use_dat