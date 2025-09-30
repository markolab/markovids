from scipy import signal
from tqdm.auto import tqdm
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


fill_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


def fill_holes(depth_map, mouse_height_threshold=30, fill_kernel=fill_kernel):
    # First, identify the mouse region using non-zero pixels
    mouse_region = depth_map > mouse_height_threshold
    expanded_mouse_region = cv2.dilate(
        mouse_region.astype(np.uint8), fill_kernel, iterations=2
    )

    # Now find holes WITHIN the expanded mouse region
    hole_mask = (depth_map == 0) & (expanded_mouse_region > 0)
    if not np.any(hole_mask):
        return depth_map
    else:
        return cv2.inpaint(depth_map, hole_mask.astype(np.uint8), 5, cv2.INPAINT_TELEA)


def sos_filter(x, fps, tau=0.01, order=3):
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


def crop_and_rotate_frames(frames, features, crop_size=(80, 80), progress_bar=True):
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


def compute_bground(
    avi_file,
    step_size=1500,
    agg_func=np.median,
    reader_kwargs={"threads": 2},
    save_dir="_bground",
    force=False,
):
    import tifffile
    from markovids.vid.io import get_bground
    import os
    import toml

    parameters = locals()
    parameters["undistorted"] = False
    basename = os.path.splitext(os.path.basename(avi_file))[0]
    path = os.path.dirname(avi_file)
    bground_path = os.path.join(path, save_dir, f"{basename}.tiff")
    toml_path = os.path.join(path, save_dir, f"{basename}.toml")
    os.makedirs(os.path.join(path, save_dir), exist_ok=True)

    if os.path.exists(bground_path) and not force:
        return tifffile.imread(bground_path)

    _bground = get_bground(
        avi_file, spacing=step_size, agg_func=agg_func, **reader_kwargs
    )
    _bground = _bground.astype("uint16")
    tifffile.imwrite(bground_path, _bground)
    with open(toml_path, "w") as f:
        toml.dump(parameters, f)

    return _bground