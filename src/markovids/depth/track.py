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
            proc_frame = (gradient_x + gradient_y) / 2.0

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


default_pre_strels = {cv2.MORPH_DILATE: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))}
default_post_strels = {cv2.MORPH_ERODE: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))}
# default_post_strels = {}


def clean_roi(
    dat,
    pre_strels=default_pre_strels,
    post_strels=default_post_strels,
    fill_holes=True,
    use_cc=True,
    progress_bar=True
):
    # assumes we've received a binary mask
    results = np.zeros(dat.shape, dtype="bool")
    for i, _frame in tqdm(enumerate(dat), total=len(dat), desc="Cleaning mask", disable=not progress_bar):
        new_frame = _frame.copy().astype("uint8")
        for k, v in pre_strels.items():
            new_frame = cv2.morphologyEx(new_frame, k, v)
        if fill_holes:
            new_frame = binary_fill_holes(new_frame)
        if new_frame.sum() == 0:
            continue
        if use_cc:
            new_frame = get_largest_cc(new_frame)
        for k, v in post_strels.items():
            new_frame = cv2.morphologyEx(new_frame.astype("uint8"), k, v)
        results[i] = new_frame != 0
    return results


def crop_and_rotate_frames(frames, features, crop_size=(160, 160), progress_bar=True):
    nframes = frames.shape[0]
    cropped_frames = np.zeros((nframes, crop_size[0], crop_size[1]), frames.dtype)
    win = (crop_size[0] // 2, crop_size[1] // 2 + 1)
    border = (crop_size[1], crop_size[1], crop_size[0], crop_size[0])

    for i in tqdm(range(frames.shape[0]), disable=not progress_bar, desc="Rotating"):
        if np.any(np.isnan(features["centroid"][i, :])):
            continue

        # use_frame = np.pad(frames[i, ...], (crop_size, crop_size), 'constant', constant_values=0)
        use_frame = cv2.copyMakeBorder(frames[i, ...], *border, cv2.BORDER_CONSTANT, 0)

        rr = np.arange(
            features["centroid"][i, 1] - win[0], features["centroid"][i, 1] + win[1]
        ).astype("int16")
        cc = np.arange(
            features["centroid"][i, 0] - win[0], features["centroid"][i, 0] + win[1]
        ).astype("int16")

        rr = rr + crop_size[0]
        cc = cc + crop_size[1]

        if (
            np.any(rr >= use_frame.shape[0])
            or np.any(rr < 1)
            or np.any(cc >= use_frame.shape[1])
            or np.any(cc < 1)
        ):
            continue

        rot_mat = cv2.getRotationMatrix2D(
            (crop_size[0] // 2, crop_size[1] // 2),
            -np.rad2deg(features["orientation"][i]),
            1,
        )
        cropped_frames[i, :, :] = cv2.warpAffine(
            use_frame[rr[0] : rr[-1], cc[0] : cc[-1]],
            rot_mat,
            (crop_size[0], crop_size[1]),
        )

    return cropped_frames