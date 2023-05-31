import cv2
import numpy as np
from tqdm.auto import tqdm


def im_moment_features(IM):
    """
    Use the method of moments and centralized moments to get image properties

    Args:
        IM (2d numpy array): depth image

    Returns:
        Features (dictionary): returns a dictionary with orientation,
        centroid, and ellipse axis length

    """

    tmp = cv2.moments(IM)
    num = 2 * tmp["mu11"]
    den = tmp["mu20"] - tmp["mu02"]

    common = np.sqrt(4 * np.square(tmp["mu11"]) + np.square(den))

    if tmp["m00"] == 0:
        features = {"orientation": np.nan, "centroid": np.nan, "axis_length": [np.nan, np.nan]}
    else:
        features = {
            "orientation": -0.5 * np.arctan2(num, den),
            "centroid": [tmp["m10"] / tmp["m00"], tmp["m01"] / tmp["m00"]],
            "axis_length": [
                2 * np.sqrt(2) * np.sqrt((tmp["mu20"] + tmp["mu02"] + common) / tmp["m00"]),
                2 * np.sqrt(2) * np.sqrt((tmp["mu20"] + tmp["mu02"] - common) / tmp["m00"]),
            ],
        }

    return features


def get_frame_features(
    frames,
    frame_threshold=10,
    mask=np.array([]),
    mask_threshold=-30,
    use_cc=False,
    progress_bar=True,
):
    """
    Use image moments to compute features of the largest object in the frame

    Args:
        frames (3d np array)
        frame_threshold (int): threshold in mm separating floor from mouse

    Returns:
        features (dict list): dictionary with simple image features

    """

    features = []
    nframes = frames.shape[0]

    if type(mask) is np.ndarray and mask.size > 0:
        has_mask = True
    else:
        has_mask = False
        mask = np.zeros((frames.shape), "uint8")

    features = {
        "centroid": np.empty((nframes, 2)),
        "orientation": np.empty((nframes,)),
        "axis_length": np.empty((nframes, 2)),
    }

    for k, v in features.items():
        features[k][:] = np.nan

    for i in tqdm(range(nframes), disable=not progress_bar, desc="Computing moments"):
        frame_mask = frames[i, ...] > frame_threshold

        if use_cc:
            cc_mask = get_largest_cc((frames[[i], ...] > mask_threshold).astype("uint8")).squeeze()
            frame_mask = np.logical_and(cc_mask, frame_mask)

        if has_mask:
            frame_mask = np.logical_and(frame_mask, mask[i, ...] > mask_threshold)
        else:
            mask[i, ...] = frame_mask

        cnts, hierarchy = cv2.findContours(
            frame_mask.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        tmp = np.array([cv2.contourArea(x) for x in cnts])

        if tmp.size == 0:
            continue

        mouse_cnt = tmp.argmax()

        for key, value in im_moment_features(cnts[mouse_cnt]).items():
            features[key][i] = value

    return features, mask
