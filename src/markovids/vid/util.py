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


def sync_depth_videos(
    data_dir,
    save_dir="_proc",
    timestamp_kwargs={
        "merge_tolerance": 0.003,
        "multiplexed": False,
        "burn_in": 500,
        "return_full_sync_only": True,
        "use_timestamp_field": "device_timestamp_ref",
    },
    undistort=True,
    intrinsics_matrix=None,
    distortion_coeffs=None,
    preview_kwargs={
        "crf": 26,
        "vmin": 5,
        "vmax": 90,
        "cmap": cv2.COLORMAP_VIRIDIS,
        "ncols": 2,
    },
    bground_kwargs={
        "step_size": 1500,
        "agg_func": np.median,
        "reader_kwargs": {"threads": 5},
        "save_dir": "_bground",
        "force": False,
    },
    preview_inpaint=True,
    reader_kwargs={"threads": 4},
    timestamp_order=["frame_id", "frame_index", "device_timestamp", "system_timestamp"],
    batch_size=500,
    vid_camera_order=[
        "Lucid Vision Labs-HTP003S-001-224500508",
        "Lucid Vision Labs-HTP003S-001-223702048",
        "Lucid Vision Labs-HTP003S-001-223702266",
    ],
):
    from markovids.vid.io import (
        get_bground,
        read_timestamps_multicam,
        read_frames_multicam,
        MP4WriterPreview,
        AviReader,
        AviWriter,
    )
    import os
    import toml
    import copy
    import warnings

    parameters = locals()
    colormap_options = ["vmin", "vmax", "cmap"]
    colormap_kwargs = {}
    use_preview_kwargs = copy.deepcopy(preview_kwargs)
    for _option in colormap_options:
        val = use_preview_kwargs.pop(_option)
        if val:
            colormap_kwargs[_option] = val

    print(f"Processing {data_dir}...")
    # stage output, get metadata
    output_dir = os.path.join(data_dir, save_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    metadata = toml.load(os.path.join(data_dir, "metadata.toml"))
    cameras = sorted(list(metadata["cameras"].keys()))

    if (not undistort) or (intrinsics_matrix is None) or (distortion_coeffs is None):
        undistort = False
    else:
        print("Will undistort data")

    preview_ncols = preview_kwargs.pop("ncols")
    preview_nrows = np.ceil(len(cameras) / preview_ncols)

    # need paths to timestamps and avi
    ts_paths = {os.path.join(data_dir, f"{_cam}.txt"): _cam for _cam in cameras}
    avi_paths = [os.path.join(data_dir, f"{_cam}.avi") for _cam in cameras]
    ts, merged_ts = read_timestamps_multicam(
        ts_paths,
        **timestamp_kwargs,
    )

    # sort timestamps
    column_order = [timestamp_kwargs["use_timestamp_field"]]
    for _timestamp_type in timestamp_order:
        for _cam in cameras:
            column_order += [(_cam, _timestamp_type)]

    merged_ts = merged_ts[column_order]
    with open(os.path.join(output_dir, "timestamps.txt"), "w") as f:
        merged_ts.to_csv(f, header=True, index=False)

    fps = np.round(
        1 / merged_ts[timestamp_kwargs["use_timestamp_field"]].diff().median()
    )
    # check for background compute if need be
    load_dct = {}
    avi_writers = {}
    bgrounds = {}
    for _cam, _file in tqdm(zip(cameras, avi_paths), total=len(cameras)):
        bgrounds[_cam] = get_bground(_file, **bground_kwargs)
        if undistort:
            bgrounds[_cam] = cv2.undistort(
                bgrounds[_cam], intrinsics_matrix[_cam], distortion_coeffs[_cam]
            )
        frame_idx = merged_ts[
            (_cam, "frame_index")
        ].tolist()  # list of frame indices we need to write out
        total_frames = len(frame_idx)
        reader = AviReader(_file)
        basename = os.path.basename(_file)

        pixel_format = reader.pixel_format
        frame_size = reader.frame_size
        dtype = reader.dtype
        load_dct[_cam] = {"frame_size": frame_size, "dtype": dtype}
        if undistort:
            load_dct[_cam]["intrinsic_matrix"] = intrinsics_matrix[_cam]
            load_dct[_cam]["distortion_coeffs"] = distortion_coeffs[_cam]

        avi_writers[_cam] = AviWriter(
            os.path.join(output_dir, basename),
            fps=fps,
            pixel_format="gray",
            frame_size=frame_size,
            dtype=np.uint8,
        )
        reader.close()

    mp4_writer = MP4WriterPreview(
        os.path.join(output_dir, "depth_preview.mp4"),
        fps=fps,
        frame_size=(
            int(frame_size[0] * preview_nrows),
            int(frame_size[1] * preview_ncols),
        ),
        **use_preview_kwargs,
    )

    total_frames = len(merged_ts)
    dat_paths = {_file: _cam for _file, _cam in zip(avi_paths, cameras)}
    nbatches = total_frames // batch_size
    for _left_edge in tqdm(
        range(0, total_frames, batch_size), total=nbatches, desc="Frame batch"
    ):
        left_edge = _left_edge
        right_edge = min(_left_edge + batch_size, total_frames)
        use_ts = merged_ts.iloc[left_edge:right_edge]

        read_frames = {
            _cam: use_ts[(_cam, "frame_index")].astype("int32").to_list()
            for _cam in cameras
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame_batch = read_frames_multicam(
                dat_paths, read_frames, load_dct, progress_bar=False
            )

        for k, v in frame_batch.items():
            frame_batch[k] = np.clip(
                np.floor((bgrounds[k] - v.astype("float32")) / 4), 0, 255
            ).astype("uint8")

        for _cam, _writer in avi_writers.items():
            _writer.write_frames(frame_batch[_cam], progress_bar=False)

        # since we've written out the raw data, fill holes for preview if we asked for it
        if preview_inpaint:
            for k, v in frame_batch.items():
                for i in range(len(v)):
                    frame_batch[k][i] = fill_holes(v[i])

        montage_frames = video_montage(
            [frame_batch[_cam][..., None] for _cam in vid_camera_order], ncols=2
        )
        # montage_frames = apply_opencv_colormap_stack(montage_frames, **colormap_kwargs)
        mp4_writer.write_frames(
            montage_frames, frames_idx=range(left_edge, right_edge), progress_bar=False
        )

    # TODO: need metadata homeson...
    with open(os.path.join(output_dir, "sync_metadata.toml"), "w") as f:
        toml.dump(parameters, f)

    for _writer in avi_writers.values():
        _writer.close()
    mp4_writer.close()
    return None
