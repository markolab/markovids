from typing import Tuple, Optional
from markovids.vid.io import (
    get_bground,
    downsample_frames,
    read_timestamps_multicam,
    read_frames_multicam,
    AviWriter,
    MP4WriterPreview,
)
from markovids.vid.util import bp_filter, sos_filter, video_montage
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import os
import numpy as np

default_win_kwargs =  {"window": 20, "min_periods": 1, "center": True}
def hampel(df, scale=.6745, threshold=3, replace=True, insert_nans=True, **kwargs):
    use_kwargs = default_win_kwargs | kwargs
    new_df = df.copy()
    meds = df.rolling(**use_kwargs).median()
    devs = (df - meds).abs()
    mads = devs.rolling(**use_kwargs).median()
    mads /= scale
    mads_dev = devs / mads

    # handles edges via min_periods etc.
    if insert_nans:
        new_df[np.logical_or(np.isnan(meds),np.isnan(mads_dev))] = np.nan
    if replace:
        new_df[mads_dev>threshold] = meds[mads_dev>threshold]
    else:
        new_df[mads_dev > threshold] = np.nan
    return new_df

def squash_conf(conf, gamma=2, min_cutoff=0.05):
    return np.where(conf > min_cutoff, conf ** gamma, 0)

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
    batch_size: int=int(1e2),
    overlap: int=int(10),
    bground_spacing: int=int(1e3),
    downsample: int=2,
    spatial_bp: tuple=(1.0, 4.0),
    temporal_tau: float=0.1,
    fluo_threshold_sig: float=5.0,
    vid_montage_ncols: int=3,
    nbatches: int=1,
    burn_in: int=int(3e2),
    use_timestamp_field="device_timestamp_ref",
    vids: list=["fluorescence", "reflectance", "merge"],
    reflect_cmap=plt.matplotlib.colormaps.get_cmap("gray"),
    fluo_cmap=plt.matplotlib.colormaps.get_cmap("turbo"),
    fluo_only_cmap=plt.matplotlib.colormaps.get_cmap("magma"),
    reflect_norm=plt.matplotlib.colors.Normalize(vmin=0, vmax=255),
    fluo_norm=plt.matplotlib.colors.Normalize(vmin=6, vmax=40),  # in z units
    fluo_only_norm=plt.matplotlib.colors.Normalize(vmin=6, vmax=30),  # in z units
    vid_paths: dict={
        "reflectance": "reflectance.mp4",
        "fluorescence": "fluorescence.mp4",
        "merge": "merge.mp4",
    },
    save_path: str="_proc",
) -> None:
    # TODO: assert that all cams have same frame size

    cameras = list(dat_paths.values())
    vid_montage_nrows = int(np.ceil(len(cameras) / vid_montage_ncols))
    width, height = load_dct[cameras[0]][
        "frame_size"
    ]  # assumes frames are all same size
    montage_width = (width // downsample) * vid_montage_ncols
    montage_height = (height // downsample) * vid_montage_nrows
    _, _, ts_fluo, ts_reflect = read_timestamps_multicam(ts_paths, 
                                                        use_timestamp_field=use_timestamp_field, 
                                                        merge_tolerance=0.001, 
                                                        return_equal_frames=True, 
                                                        return_full_sync_only=True,
                                                        multiplexed=True,
                                                        fill=False,
                                                        burn_in=300)

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
        _cam: use_frames_bground_fluo[(_cam,"frame_index")].astype("int32").to_list()
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
            _cam: use_ts_fluo[(_cam,"frame_index")].astype("int32").to_list() for _cam in cameras
        }
        raw_dat_fluo = read_frames_multicam(
            dat_paths, read_frames_fluo, load_dct, downsample=downsample
        )
        read_frames_reflect = {
            _cam: use_ts_reflect[(_cam,"frame_index")].astype("int32").to_list() for _cam in cameras
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
            reflect_frames = np.zeros((nframes, height, width, 3), dtype="uint8")
            for i in range(nframes):
                reflect_frames[i] = (
                    reflect_cmap(reflect_norm(raw_dat_reflect[_cam][i]))[..., :3] * 255
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
                vid_frames["fluorescence"][_cam] = np.clip(
                    (fluo_only_cmap(fluo_only_norm(zfluo_frames))[..., :3] * 255),
                    0,
                    255,
                ).astype("uint8")

            for i in range(nframes):
                fluo_frames[i] = bp_filter(
                    fluo_frames[i], *spatial_bp
                )  # spatial bandpass
            fluo_frames = sos_filter(
                fluo_frames, temporal_tau, fps
            )  # copy off these frames for fluorescence only...

            fluo_mu = fluo_frames.mean(axis=(1, 2), keepdims=True)
            fluo_std = fluo_frames.std(axis=(1, 2), keepdims=True)

            fluo_frames -= fluo_mu
            fluo_frames /= fluo_std

            plt_fluo_frames = np.zeros((nframes, height, width, 3), dtype="uint8")
            for i in range(nframes):
                plt_fluo_frames[i] = (
                    fluo_cmap(fluo_norm(fluo_frames[i]))[..., :3] * 255
                ).astype("uint8")

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
    batch_size: int=int(1e2),
    nbatches: Optional[int]=None,
    save_path: str="_proc",
    use_timestamp_field: str="device_timestamp_ref"
) -> None:
    # TODO: assert that all cams have same frame size
    # TODO: construct filenames from metadata!!!
    cameras = list(dat_paths.values())
    # width, height = load_dct[cameras[0]][
    #     "frame_size"
    # ]  # assumes frames are all same size

    _, _, ts_fluo, ts_reflect = read_timestamps_multicam(ts_paths, 
                                                        use_timestamp_field=use_timestamp_field, 
                                                        merge_tolerance=0.001, 
                                                        return_equal_frames=True, 
                                                        return_full_sync_only=True,
                                                        multiplexed=True,
                                                        fill=False,
                                                        burn_in=300)

    new_timestamp_order = [
        "frame_id",
        "frame_index",
        "device_timestamp",
        "system_timestamp"
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
            _cam: use_ts_fluo[(_cam,"frame_index")].astype("int32").to_list() for _cam in cameras
        }
        raw_dat_fluo = read_frames_multicam(
            dat_paths, read_frames_fluo, load_dct
        )
        read_frames_reflect = {
            _cam: use_ts_reflect[(_cam,"_frame_index")].astype("int32").to_list() for _cam in cameras
        }
        raw_dat_reflect = read_frames_multicam(
            dat_paths, read_frames_reflect, load_dct
        )

        for _cam in cameras:
            fluo_writers[_cam].write_frames(raw_dat_fluo[_cam], progress_bar=False)
            reflect_writers[_cam].write_frames(
                raw_dat_reflect[_cam], progress_bar=False
            )

    for _cam in cameras:
        fluo_writers[_cam].close()
        reflect_writers[_cam].close()


# https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]