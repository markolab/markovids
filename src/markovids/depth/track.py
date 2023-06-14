import numpy as np
import cv2
from tqdm.auto import tqdm


def get_largest_cc(frame):
    """Returns largest connected component blob in image
    Args:
        frame (3d numpy array): frames x r x c, uncropped mouse
        progress_bar (bool): display progress bar

    """
    _, output, stats, _ = cv2.connectedComponentsWithStats(frame, connectivity=4)
    szs = stats[:, -1]

    if len(szs) <= 1:
        return None
    else:
        foreground_obj = output == szs[1:].argmax() + 1
        return foreground_obj


def binary_fill_holes(frame, inplace=True):
    if inplace:
        des = frame
    else:
        des = frame.copy()
    contour, hier = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        cv2.drawContours(des, [cnt], 0, 255, -1)
    return des


def get_largest_contour(frame):
    cnts, _ = cv2.findContours(frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    tmp = np.array([cv2.contourArea(x) for x in cnts])
    if tmp.size == 0:
        cnt_img = None
    else:
        cnt = cnts[tmp.argmax()]
        cnt_img = cv2.drawContours(
            image=np.zeros(frame.shape, dtype="uint8"),
            contours=[cnt],
            contourIdx=-1,
            color=(255, 255, 255),
            thickness=cv2.FILLED,
        )
        cnt_img = (cnt_img != 0).astype("uint8")
    return cnt_img


roi_strels = {
    cv2.MORPH_OPEN: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
}


def get_roi(
    dat,
    depth_range=(40, 900),
    median_kernels=[],
    gradient_kernel=3,
    gaussian_kernels=[5],
    fill_holes=True,
    use_cc=True,
    roi_strels=roi_strels,
):
    results = np.zeros(dat.shape, dtype="bool")
    for i, _frame in tqdm(enumerate(dat), total=len(dat), desc="Computing ROI"):
        proc_frame = _frame.copy()

        for _kernel in median_kernels:
            proc_frame = cv2.medianBlur(proc_frame, _kernel)

        for _kernel in gaussian_kernels:
            ksize = int(_kernel * 3)
            proc_frame = cv2.GaussianBlur(proc_frame, [ksize] * 2, _kernel)

        if gradient_kernel is not None:
            gradient_x = np.abs(cv2.Sobel(proc_frame, cv2.CV_32F, 1, 0, ksize=gradient_kernel))
            gradient_y = np.abs(cv2.Sobel(proc_frame, cv2.CV_32F, 0, 1, ksize=gradient_kernel))
            proc_frame = (gradient_x + gradient_y) / 2.

        proc_frame = np.logical_and(
            proc_frame > depth_range[0], proc_frame < depth_range[1]
        ).astype("uint8")

        for k, v in roi_strels.items():
            proc_frame = cv2.morphologyEx(proc_frame, k, v)

        mask = get_largest_contour(proc_frame)
        if (mask is not None) and (fill_holes):
            start_frame = binary_fill_holes(mask)
        elif mask is not None:
            start_frame = mask
        else:
            start_frame = proc_frame
        if use_cc:
            start_frame = get_largest_cc(start_frame)
        results[i] = start_frame
    return results


default_strels = {cv2.MORPH_DILATE: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))}


def clean_roi(dat, strels=default_strels):
    # assumes we've received a binary mask
    results = np.zeros(dat.shape, dtype="bool")
    for i, _frame in tqdm(enumerate(dat), total=len(dat), desc="Cleaning mask"):
        new_frame = _frame.copy().astype("uint8")
        for k, v in strels.items():
            new_frame = cv2.morphologyEx(new_frame, k, v)
            results[i] = new_frame != 0
    return results
