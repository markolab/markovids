from typing import Tuple, Optional
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import os
import numpy as np

default_win_kwargs = {"window": 20, "min_periods": 1, "center": True}


def hampel(df, scale=0.6745, threshold=3, replace=True, insert_nans=True, **kwargs):
    use_kwargs = default_win_kwargs | kwargs
    new_df = df.copy()
    meds = df.rolling(**use_kwargs).median()
    devs = (df - meds).abs()
    mads = devs.rolling(**use_kwargs).median()
    mads /= scale
    mads_dev = devs / mads

    # handles edges via min_periods etc.
    if insert_nans:
        new_df[np.logical_or(np.isnan(meds), np.isnan(mads_dev))] = np.nan
    if replace:
        new_df[mads_dev > threshold] = meds[mads_dev > threshold]
    else:
        new_df[mads_dev > threshold] = np.nan
    return new_df


def squash_conf(conf, gamma=2, min_cutoff=0.05):
    return np.where(conf > min_cutoff, conf**gamma, 0)


def savgol_filter_missing(x, window_length=7, poly_order=2):
    from scipy.signal import savgol_filter
    import pandas as pd

    proc_x = x.to_numpy()
    is_valid = np.isfinite(proc_x)
    proc_x[is_valid] = savgol_filter(proc_x[is_valid], window_length, poly_order)
    proc_x[~is_valid] = np.nan
    return pd.Series(data=proc_x, index=x.index)


def next_even_number(x):
    return (np.ceil(x / 2) * 2).astype("int")


def prev_even_number(x):
    return (np.floor(x / 2) * 2).astype("int")


def alternating_excitation_vid_preview(
    dat_paths: dict,
    ts_paths: dict,
    load_dct: dict,
    batch_size: int = int(1e2),
    overlap: int = int(10),
    bground_spacing: int = int(1e3),
    downsample: int = 2,
    spatial_bp: tuple = (0.0, 0.0),
    temporal_tau: float = 0.0,
    fluo_threshold_sig: float = 5.0,
    vid_montage_ncols: int = 3,
    nbatches: int = 1,
    burn_in: int = int(3e2),
    use_timestamp_field="device_timestamp_ref",
    vids: list = ["fluorescence", "reflectance", "merge"],
    reflect_cmap: str = "bone",
    fluo_cmap: str = "turbo",
    fluo_only_cmap: str = "magma",
    reflect_norm: tuple = (0, 255),
    fluo_norm: tuple = (6, 40),
    fluo_only_norm: tuple = (6, 30),
    # reflect_cmap=plt.matplotlib.colormaps.get_cmap("gray"),
    # fluo_cmap=plt.matplotlib.colormaps.get_cmap("turbo"),
    # fluo_only_cmap=plt.matplotlib.colormaps.get_cmap("magma"),
    # reflect_norm=plt.matplotlib.colors.Normalize(vmin=0, vmax=255),
    # fluo_norm=plt.matplotlib.colors.Normalize(vmin=6, vmax=40),  # in z units
    # fluo_only_norm=plt.matplotlib.colors.Normalize(vmin=6, vmax=30),  # in z units
    vid_paths: dict = {
        "reflectance": "reflectance.mp4",
        "fluorescence": "fluorescence.mp4",
        "merge": "merge.mp4",
    },
    save_path: str = "_proc",
) -> None:
    # TODO: assert that all cams have same frame size
    from markovids.vid.io import (
        get_bground,
        downsample_frames,
        read_timestamps_multicam,
        read_frames_multicam,
        MP4WriterPreview,
        pseudocolor_frames,
    )
    from markovids.vid.util import bp_filter, sos_filter, video_montage

    cameras = list(dat_paths.values())
    vid_montage_nrows = int(np.ceil(len(cameras) / vid_montage_ncols))
    width, height = load_dct[cameras[0]][
        "frame_size"
    ]  # assumes frames are all same size
    montage_width = (width // downsample) * vid_montage_ncols
    montage_height = (height // downsample) * vid_montage_nrows
    _, _, ts_fluo, ts_reflect = read_timestamps_multicam(
        ts_paths,
        use_timestamp_field=use_timestamp_field,
        merge_tolerance=0.001,
        return_equal_frames=True,
        return_full_sync_only=True,
        multiplexed=True,
        fill=False,
        burn_in=300,
    )

    fps = 1 / ts_fluo[use_timestamp_field].diff().median()
    total_frames = len(ts_fluo)  # everything is aligned to fluorescence

    # set up videos...
    vid_writers = {}

    # use the first filename?
    base_path = os.path.dirname(
        os.path.normpath(os.path.abspath(list(dat_paths.keys())[0]))
    )
    full_save_path = os.path.join(base_path, save_path)
    os.makedirs(full_save_path, exist_ok=True)

    for _vid in vids:
        vid_writers[_vid] = MP4WriterPreview(
            os.path.join(full_save_path, vid_paths[_vid]),
            frame_size=(montage_width, montage_height),
            fps=fps,
        )

    # get background
    use_frames_bground_fluo = ts_fluo.iloc[::bground_spacing].dropna()
    read_frames_bground_fluo = {
        _cam: use_frames_bground_fluo[(_cam, "frame_index")].astype("int32").to_list()
        for _cam in cameras
    }
    bground_fluo = {
        _cam: get_bground(
            _path,
            valid_range=None,
            median_kernels=[],
            agg_func=np.nanmean,
            use_frames=read_frames_bground_fluo[_cam],
            **load_dct[_cam],
        ).astype("uint8")
        for _path, _cam in tqdm(
            dat_paths.items(), total=len(dat_paths), desc="Computing background"
        )
    }
    use_bground_fluo = {
        _cam: downsample_frames(_bground[None, ...], downsample=downsample)[0]
        for _cam, _bground in bground_fluo.items()
    }

    # make a batch here...
    if (nbatches is None) or (nbatches <= 0):
        nbatches = total_frames // batch_size
    else:
        total_frames = min(batch_size * nbatches, total_frames)

    idx = 0
    for _left_edge in tqdm(
        range(0, total_frames, batch_size), total=nbatches, desc="Frame batch"
    ):
        left_edge = max(_left_edge - overlap, 0)
        right_edge = min(_left_edge + batch_size, total_frames)

        use_ts_fluo = ts_fluo.iloc[left_edge:right_edge]
        use_ts_reflect = ts_reflect.iloc[left_edge:right_edge]

        read_frames_fluo = {
            _cam: use_ts_fluo[(_cam, "frame_index")].astype("int32").to_list()
            for _cam in cameras
        }
        raw_dat_fluo = read_frames_multicam(
            dat_paths, read_frames_fluo, load_dct, downsample=downsample
        )
        read_frames_reflect = {
            _cam: use_ts_reflect[(_cam, "frame_index")].astype("int32").to_list()
            for _cam in cameras
        }
        raw_dat_reflect = read_frames_multicam(
            dat_paths, read_frames_reflect, load_dct, downsample=downsample
        )

        nframes, height, width = raw_dat_reflect[cameras[0]].shape

        vid_frames = {}
        for _vid in vids:
            vid_frames[_vid] = {}

        for _cam in tqdm(cameras, desc="Camera"):
            # reflect_frames = raw_dat_reflect[_cam].copy()
            # reflect_frames = np.zeros((nframes, height, width, 3), dtype="uint8")
            # for i in range(nframes):
            # reflect_frames[i] = (
            #     reflect_cmap(reflect_norm(raw_dat_reflect[_cam][i]))[..., :3] * 255
            # ).astype("uint8")
            reflect_frames = pseudocolor_frames(
                raw_dat_reflect[_cam],
                cmap=reflect_cmap,
                vmin=reflect_norm[0],
                vmax=reflect_norm[1],
            ).astype("uint8")

            if "reflectance" in vids:
                vid_frames["reflectance"][_cam] = reflect_frames.copy()

            fluo_frames = np.clip(
                raw_dat_fluo[_cam].astype("float64")
                - use_bground_fluo[_cam].astype("float64"),
                0,
                np.inf,
            )  # only want things brighter than background
            if "fluorescence" in vids:
                # simple zscore to put everything on relatively equal footing
                fluo_mu = fluo_frames.mean(axis=(1, 2), keepdims=True)
                fluo_std = fluo_frames.std(axis=(1, 2), keepdims=True)
                zfluo_frames = (fluo_frames - fluo_mu) / fluo_std
                # vid_frames["fluorescence"][_cam] = np.clip(
                #     (fluo_only_cmap(fluo_only_norm(zfluo_frames))[..., :3] * 255),
                #     0,
                #     255,
                # ).astype("uint8")
                vid_frames["fluorescence"][_cam] = pseudocolor_frames(
                    zfluo_frames,
                    cmap=fluo_only_cmap,
                    vmin=fluo_only_norm[0],
                    vmax=fluo_only_norm[1],
                ).astype("uint8")

            if (spatial_bp is not None) and (spatial_bp[0] > 0 or spatial_bp[1] > 0):
                for i in range(nframes):
                    fluo_frames[i] = bp_filter(
                        fluo_frames[i], *spatial_bp
                    )  # spatial bandpass
            if (temporal_tau is not None) and (temporal_tau > 0):
                fluo_frames = sos_filter(
                    fluo_frames, temporal_tau, fps
                )  # copy off these frames for fluorescence only...

            fluo_mu = fluo_frames.mean(axis=(1, 2), keepdims=True)
            fluo_std = fluo_frames.std(axis=(1, 2), keepdims=True)

            fluo_frames -= fluo_mu
            fluo_frames /= fluo_std

            # plt_fluo_frames = np.zeros((nframes, height, width, 3), dtype="uint8")
            # for i in range(nframes):
            # plt_fluo_frames[i] = (
            #     fluo_cmap(fluo_norm(fluo_frames[i]))[..., :3] * 255
            # ).astype("uint8")
            plt_fluo_frames = pseudocolor_frames(
                fluo_frames, cmap=fluo_cmap, vmin=fluo_norm[0], vmax=fluo_norm[1]
            )
            reflect_frames[fluo_frames > fluo_threshold_sig] = plt_fluo_frames[
                fluo_frames > fluo_threshold_sig
            ]

            if "merge" in vids:
                vid_frames["merge"][_cam] = reflect_frames

        for _vid in vids:
            extra_frames = _left_edge - left_edge  # chop off overhang
            montage_frames = video_montage(list(vid_frames[_vid].values()), ncols=3)[
                extra_frames:
            ]
            vid_writers[_vid].write_frames(
                montage_frames,
                frames_idx=range(idx, idx + len(montage_frames)),
                progress_bar=False,
            )

        idx += len(montage_frames)

    for _vid in vids:
        vid_writers[_vid].close()


# move to markovids and tie into cli, need to call from PACE
def alternating_excitation_vid_split(
    dat_paths: dict,
    ts_paths: dict,
    load_dct: dict,
    batch_size: int = int(1e2),
    nbatches: Optional[int] = None,
    save_path: str = "_proc",
    use_timestamp_field: str = "device_timestamp_ref",
) -> None:
    from markovids.vid.io import (
        read_timestamps_multicam,
        read_frames_multicam,
        AviWriter,
    )
    from markovids.vid.util import bp_filter, sos_filter, video_montage

    # TODO: assert that all cams have same frame size
    # TODO: construct filenames from metadata!!!
    cameras = list(dat_paths.values())

    _, _, ts_fluo, ts_reflect = read_timestamps_multicam(
        ts_paths,
        use_timestamp_field=use_timestamp_field,
        merge_tolerance=0.001,
        return_equal_frames=True,
        return_full_sync_only=True,
        multiplexed=True,
        fill=False,
        burn_in=300,
    )

    new_timestamp_order = [
        "frame_id",
        "frame_index",
        "device_timestamp",
        "system_timestamp",
    ]
    column_order = [use_timestamp_field]
    for _timestamp_type in new_timestamp_order:
        for _cam in cameras:
            column_order += [(_cam, _timestamp_type)]

    ts_fluo = ts_fluo[column_order]
    ts_reflect = ts_reflect[column_order]

    fps = 1 / ts_fluo[use_timestamp_field].diff().median()
    total_frames = len(ts_fluo)

    # use the first filename?
    base_path = os.path.dirname(
        os.path.normpath(os.path.abspath(list(dat_paths.keys())[0]))
    )
    full_save_path = os.path.join(base_path, save_path)
    os.makedirs(full_save_path, exist_ok=True)

    # make a batch here...
    if (nbatches is None) or (nbatches <= 0):
        nbatches = total_frames // batch_size
    else:
        total_frames = min(batch_size * nbatches, total_frames)

    idx = 0

    fluo_writers = {}
    reflect_writers = {}

    for _cam in cameras:
        if load_dct[_cam]["dtype"].itemsize == 2:
            pixel_format = "gray16le"
        elif load_dct[_cam]["dtype"].itemsize == 1:
            pixel_format = "gray"
        else:
            dtype = load_dct[_cam]["dtype"]
            raise RuntimeError(f"Can't map {dtype.itemsize} bytes to pixel_format")
        fluo_writers[_cam] = AviWriter(
            os.path.join(full_save_path, f"{_cam}-fluorescence.avi"),
            fps=fps,
            frame_size=load_dct[_cam]["frame_size"],
            dtype=load_dct[_cam]["dtype"],
            pixel_format=pixel_format,
        )
        reflect_writers[_cam] = AviWriter(
            os.path.join(full_save_path, f"{_cam}-reflectance.avi"),
            fps=np.round(fps),
            frame_size=load_dct[_cam]["frame_size"],
            pixel_format=pixel_format,
            dtype=load_dct[_cam]["dtype"],
        )

    with open(os.path.join(full_save_path, "timestamps-fluorescence.txt"), "w") as f:
        ts_fluo.to_csv(f, header=True, index=False)

    with open(os.path.join(full_save_path, "timestamps-reflectance.txt"), "w") as f:
        ts_reflect.to_csv(f, header=True, index=False)

    for _left_edge in tqdm(
        range(0, total_frames, batch_size), total=nbatches, desc="Frame batch"
    ):
        left_edge = _left_edge
        right_edge = min(_left_edge + batch_size, total_frames)

        use_ts_fluo = ts_fluo.iloc[left_edge:right_edge]
        use_ts_reflect = ts_reflect.iloc[left_edge:right_edge]

        # we ffill and bfill nans just for rendering...
        read_frames_fluo = {
            _cam: use_ts_fluo[(_cam, "frame_index")].astype("int32").to_list()
            for _cam in cameras
        }
        raw_dat_fluo = read_frames_multicam(dat_paths, read_frames_fluo, load_dct)
        read_frames_reflect = {
            _cam: use_ts_reflect[(_cam, "frame_index")].astype("int32").to_list()
            for _cam in cameras
        }
        raw_dat_reflect = read_frames_multicam(dat_paths, read_frames_reflect, load_dct)

        for _cam in cameras:
            fluo_writers[_cam].write_frames(raw_dat_fluo[_cam], progress_bar=False)
            reflect_writers[_cam].write_frames(
                raw_dat_reflect[_cam], progress_bar=False
            )

    for _cam in cameras:
        fluo_writers[_cam].close()
        reflect_writers[_cam].close()


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
        "cmap": "viridis",
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
    from markovids.vid.util import video_montage, fill_holes
    import os
    import toml
    import copy
    import warnings
    import cv2

    parameters = locals()
    write_frames_options = ["vmin", "vmax"]
    write_frames_kwargs = {}
    use_preview_kwargs = copy.deepcopy(preview_kwargs)
    for _option in write_frames_options:
        val = use_preview_kwargs.pop(_option)
        if val:
            write_frames_kwargs[_option] = val

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

    preview_ncols = use_preview_kwargs.pop("ncols")
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
        bgrounds[_cam] = compute_bground(_file, **bground_kwargs)
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
            montage_frames, frames_idx=range(left_edge, right_edge), progress_bar=False, **write_frames_kwargs
        )

    # TODO: need metadata homeson...
    with open(os.path.join(output_dir, "sync_metadata.toml"), "w") as f:
        toml.dump(parameters, f)

    for _writer in avi_writers.values():
        _writer.close()
    mp4_writer.close()
    return None


# https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]
