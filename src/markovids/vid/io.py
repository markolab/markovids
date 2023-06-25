import os
import numpy as np
import subprocess
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


class RawFileReader:
    def __init__(
        self,
        filepath,
        frame_size=(640, 480),
        dtype=np.dtype("<u2"),
        shape=None,
        intrinsic_matrix=None,
        distortion_coeffs=None,
    ):
        self.dtype = dtype
        self.filepath = filepath
        self.frame_size = frame_size
        self.npixels = np.prod(self.frame_size)
        self.distortion_coeffs = distortion_coeffs
        self.intrinsic_matrix = intrinsic_matrix
        self.get_file_info()

    def open(self):
        self.file_object = open(self.filepath, "rb")
        # self.mmap_obj = np.memmap(self.filepath, dtype=self.dtype, mode="r", offset=0, shape=self.dims)

    def get_frames(self, frame_range=None):
        skip_read = False
        if frame_range is None:
            # if frames is None load everything
            seek_point = 0
            read_points = self.nframes
            dims = (self.nframes, self.frame_size[1], self.frame_size[0])
        elif isinstance(frame_range, tuple):
            # if it's a tuple, assume it's left_edge, right_edge, turn into range
            seek_point = frame_range[0] * self.bytes_per_frame
            read_points = frame_range[1] * self.npixels
            dims = ((frame_range[1] - frame_range[0]) + 1, self.frame_size[1], self.frame_size[0])
        elif isinstance(frame_range, int):
            seek_point = frame_range * self.bytes_per_frame
            read_points = self.npixels
            dims = (self.frame_size[1], self.frame_size[0])
        elif isinstance(frame_range, range):
            _tmp = list(frame_range)
            seek_point = _tmp[0] * self.bytes_per_frame
            read_points = ((_tmp[-1] - _tmp[0]) + 1) * self.npixels
            dims = ((_tmp[-1] - _tmp[0]) + 1, self.frame_size[1], self.frame_size[0])
            # run through each element in the list
        elif isinstance(frame_range, list):
            nframes = len(frame_range)
            dat = np.zeros((nframes, self.frame_size[1], self.frame_size[0]), dtype=self.dtype)
            for i, _frame in enumerate(frame_range):
                dat[i] = self.get_frames(_frame)
            skip_read = True
        else:
            raise RuntimeError("Did not understand frame range type")

        if not skip_read:
            self.file_object.seek(int(seek_point))
            dat = np.fromfile(file=self.file_object, dtype=self.dtype, count=read_points).reshape(
                dims
            )

        return dat

    def close(self):
        self.file_object.close()

    def undistort_frames(self, frames, progress_bar=True):
        if (self.intrinsic_matrix is not None) and (self.distortion_coeffs is not None):
            for i, _frame in tqdm(
                enumerate(frames),
                total=len(frames),
                desc="Removing frame distortion",
                disable=not progress_bar,
            ):
                frames[i] = cv2.undistort(_frame, self.intrinsic_matrix, self.distortion_coeffs)
        return frames

    def get_file_info(self):
        self.total_bytes = os.stat(self.filepath).st_size
        self.bytes_per_frame = np.prod(self.frame_size) * self.dtype.itemsize
        self.nframes = int(os.stat(self.filepath).st_size / self.bytes_per_frame)
        self.dims = (self.nframes, self.frame_size[1], self.frame_size[0])


# simple command to pipe frames to an ffv1 file
def write_frames(
    filename,
    frames,
    threads=6,
    fps=30,
    pixel_format="gray16le",
    codec="ffv1",
    close_pipe=True,
    pipe=None,
    slices=24,
    slicecrc=1,
    frame_size=None,
    get_cmd=False,
):
    """
    Write frames to avi file using the ffv1 lossless encoder
    """

    # we probably want to include a warning about multiples of 32 for videos
    # (then we can use pyav and some speedier tools)

    if not frame_size and type(frames) is np.ndarray:
        frame_size = "{0:d}x{1:d}".format(frames.shape[2], frames.shape[1])
    elif not frame_size and type(frames) is tuple:
        frame_size = "{0:d}x{1:d}".format(frames[0], frames[1])

    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "fatal",
        "-framerate",
        str(fps),
        "-f",
        "rawvideo",
        "-s",
        frame_size,
        "-pix_fmt",
        pixel_format,
        "-i",
        "-",
        "-an",
        "-vcodec",
        codec,
        "-threads",
        str(threads),
        "-slices",
        str(slices),
        "-slicecrc",
        str(slicecrc),
        "-r",
        str(fps),
        filename,
    ]

    if get_cmd:
        return command

    if not pipe:
        pipe = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    for i in tqdm.tqdm(range(frames.shape[0])):
        pipe.stdin.write(frames[i, ...].astype("uint16").tostring())

    if close_pipe:
        pipe.stdin.close()
        return None
    else:
        return pipe


def get_raw_info(filename, dtype=np.dtype("<u2"), frame_size=(512, 424)):
    if isinstance(filename, np.memmap):
        dtype = filename.dtype
        total_bytes = np.prod(filename.shape) * dtype.itemsize
        frame_size = filename.shape[1:]
        bytes_per_frame = frame_size[0] * frame_size[1] * dtype.itemsize
        nframes = len(filename)
    else:
        total_bytes = os.stat(filename).st_size
        bytes_per_frame = frame_size[0] * frame_size[1] * dtype.itemsize
        nframes = int(os.stat(filename).st_size / bytes_per_frame)

    file_info = {
        "bytes": total_bytes,
        "nframes": nframes,
        "dims": frame_size,
        "bytes_per_frame": bytes_per_frame,
    }
    return file_info


def read_frames_raw(
    filename,
    frames=None,
    frame_size=(512, 424),
    dtype=np.dtype("<u2"),
    intrinsic_matrix=None,
    distortion_coeffs=None,
    progress_bar=True,
    **kwargs,
):
    # TODO, if any frame indices are a nan or less than 0, save to insert into array...

    vid_info = get_raw_info(filename, frame_size=frame_size, dtype=dtype)
    if vid_info["dims"] != frame_size:
        frame_size = vid_info["dims"]
    nframes = vid_info["nframes"]

    if frames is None:
        # if frames is None load everything
        frames = (0, nframes)
        use_range = range(*frames)
        offset = 0
        dims = (nframes, frame_size[1], frame_size[0])
    elif isinstance(frames, tuple):
        # if it's a tuple, assume it's left_edge, right_edge, turn into range
        left_edge, right_edge = frames
        offset = left_edge * vid_info["bytes_per_frame"]
        use_range = range(0, min((right_edge - left_edge) + 1, nframes))
        dims = (nframes - left_edge, frame_size[1], frame_size[0])
    elif isinstance(frames, range):
        # range and tuples are contiguous
        left_edge, right_edge = list(frames)[0], list(frames)[-1]
        offset = left_edge * vid_info["bytes_per_frame"]
        use_range = range(0, min((right_edge - left_edge) + 1, nframes))
        dims = (nframes - left_edge, frame_size[1], frame_size[0])
    elif isinstance(frames, list):
        # a list is NOT contiguous, simply pass it to the memmap
        use_range = frames
        offset = 0
        dims = (nframes, frame_size[1], frame_size[0])
    else:
        raise RuntimeError("Did not understand frame argument")

    if isinstance(filename, np.memmap):
        mmap_obj = filename
    else:
        mmap_obj = np.memmap(filename, dtype=dtype, mode="r", offset=offset, shape=dims)
    chunk = mmap_obj[use_range]

    if (intrinsic_matrix is not None) and (distortion_coeffs is not None):
        for i, _frame in tqdm(
            enumerate(chunk),
            total=len(chunk),
            desc="Removing frame distortion",
            disable=not progress_bar,
        ):
            chunk[i] = cv2.undistort(_frame, intrinsic_matrix, distortion_coeffs)

    return chunk


default_config = {"dtype": np.dtype("<u2"), "frame_size": (640, 480)}


def read_frames_multicam(
    paths: dict,
    frames: dict,
    config: dict = {},
    tick_period: float = 1e9,
    progress_bar: bool = True,
):
    # config should contain frame_size and numpy data type
    # PATHs should be dictionary where camera is key...
    # CONFIGs should be dictionary where camera is key...
    # make sure we support inflating frames with nans...
    dat = {}
    for _cam, _path in paths.items():
        try:
            use_config = config[_cam]
            use_config = use_config | default_config
        except KeyError:
            use_config = default_config
        dat[_cam] = read_frames_raw(
            _path, frames=frames[_cam], progress_bar=progress_bar, **use_config
        )

    return dat


def fill_timestamps(timestamps, use_timestamp_field="device_timestamp", period=0.01):
    # TODO: add timestamp check as well, not just capture number
    capture_diff = timestamps["capture_number"].diff()
    gaps = capture_diff > 1
    nmissing_frames = capture_diff[gaps] - 1

    new_timestamps = timestamps.copy()
    new_timestamps.index.name = "frame_index"
    # new_timestamps["is_pad"] = False
    new_timestamps = new_timestamps.reset_index().set_index("capture_number")[
        ["frame_index", use_timestamp_field]
    ]

    to_insert_index = []
    to_insert_values = []
    to_insert_array_loc = []
    for _index, _n_missing in nmissing_frames.items():
        new_idx = np.arange(-_n_missing, 0)
        cap_number = timestamps.loc[_index]["capture_number"]
        last_good_value = new_timestamps.loc[cap_number, use_timestamp_field]
        to_insert_index += (cap_number + new_idx).tolist()
        to_insert_values += (last_good_value + new_idx * period).tolist()
        to_insert_array_loc.append((new_timestamps.index.get_loc(cap_number) + 1, len(new_idx)))

    to_insert_values = {
        "frame_index": [np.nan] * len(to_insert_values),
        use_timestamp_field: to_insert_values,
    }
    insert_df = pd.DataFrame(to_insert_values, index=to_insert_index)
    insert_df.index.name = "capture_number"
    new_timestamps = pd.concat([new_timestamps, insert_df]).sort_index()
    new_timestamps.index = range(len(new_timestamps))  # should be contiguous anyhow...
    new_timestamps["frame_index"] = new_timestamps["frame_index"].astype("Int32")

    return new_timestamps


def read_timestamps(path, tick_period=1e9, fill=False, fill_kwargs={}):
    import io

    with open(path, "r") as table:
        buffer = io.StringIO("\n".join(line.strip() for line in table))
        df = pd.read_table(buffer, delimiter="\t")
    df.index.name = "frame_index"
    df["system_timestamp"] /= tick_period
    df["device_timestamp"] /= tick_period
    df = df.sort_index()
    if fill:
        df = fill_timestamps(df, **fill_kwargs)
    return df


def read_timestamps_multicam(
    path: dict, use_timestamp_field: str = "system_timestamp", merge_tolerance=0.0035, fill=True
):
    from functools import reduce

    cameras = list(path.keys())
    ts = {}
    for _cam in cameras:
        ts[_cam] = read_timestamps(
            path[_cam],
            fill=fill,
            fill_kwargs={"use_timestamp_field": use_timestamp_field},
        ).rename(columns={"frame_index": _cam})
    merged_ts = reduce(
        lambda left, right: pd.merge_asof(
            left,
            right,
            tolerance=merge_tolerance,
            on=use_timestamp_field,
            direction="nearest",
        ),
        ts.values(),
    )
    return ts, merged_ts


def write_frames_preview(
    filename,
    frames=np.empty((0,)),
    threads=6,
    fps=30,
    pixel_format="yuv420p",
    # pixel_format="rgb24",
    codec="h264",
    slices=24,
    slicecrc=1,
    frame_size=None,
    depth_min=0,
    depth_max=80,
    get_cmd=False,
    cmap="turbo",
    pipe=None,
    close_pipe=True,
    frame_range=None,
    crf=28,
    progress_bar=False,
):
    """
    Simple command to pipe frames to an ffv1 file.
    Writes out a false-colored mp4 video.
    Parameters
    ----------
    filename (str): path to file to write to.
    frames (np.ndarray): frames to write
    threads (int): number of threads to write video
    fps (int): frames per second
    pixel_format (str): format video color scheme
    codec (str): ffmpeg encoding-writer method to use
    slices (int): number of frame slices to write at a time.
    slicecrc (int): check integrity of slices
    frame_size (tuple): shape/dimensions of image.
    depth_min (int): minimum mouse depth from floor in (mm)
    depth_max (int): maximum mouse depth from floor in (mm)
    get_cmd (bool): indicates whether function should return ffmpeg command (instead of executing)
    cmap (str): color map to use.
    pipe (subProcess.Pipe): pipe to currently open video file.
    close_pipe (bool): indicates to close the open pipe to video when done writing.
    frame_range (range()): frame indices to write on video
    progress_bar (bool): If True, displays a TQDM progress bar for the video writing progress.
    Returns
    -------
    pipe (subProcess.Pipe): indicates whether video writing is complete.
    """

    font = cv2.FONT_HERSHEY_SIMPLEX
    # white = (255, 255,  255)
    txt_pos = (30, 30)

    if not np.mod(frames.shape[1], 2) == 0:
        frames = np.pad(frames, ((0, 0), (0, 1), (0, 0)), "constant", constant_values=0)

    if not np.mod(frames.shape[2], 2) == 0:
        frames = np.pad(frames, ((0, 0), (0, 0), (0, 1)), "constant", constant_values=0)

    if not frame_size and type(frames) is np.ndarray:
        frame_size = "{0:d}x{1:d}".format(frames.shape[2], frames.shape[1])
    elif not frame_size and type(frames) is tuple:
        frame_size = "{0:d}x{1:d}".format(frames[0], frames[1])

    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "fatal",
        "-threads",
        str(threads),
        "-framerate",
        str(fps),
        "-f",
        "rawvideo",
        "-s",
        frame_size,
        "-pix_fmt",
        pixel_format,
        "-i",
        "-",
        "-an",
        "-vcodec",
        codec,
        "-slices",
        str(slices),
        "-slicecrc",
        str(slicecrc),
        "-crf",
        str(crf),
        "-r",
        str(fps),
        filename,
    ]

    if get_cmd:
        return command

    if not pipe:
        pipe = subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE
        )

    # scale frames to appropriate depth ranges
    use_cmap = plt.get_cmap(cmap)
    for i in tqdm(
        range(frames.shape[0]),
        disable=not progress_bar,
        desc=f"Writing frames to {filename}",
    ):
        disp_img = frames[i, :].copy().astype("float32")
        disp_img = (disp_img - depth_min) / (depth_max - depth_min)
        disp_img[disp_img < 0] = 0
        disp_img[disp_img > 1] = 1
        disp_img = np.delete(use_cmap(disp_img), 3, 2) * 255
        disp_img = cv2.cvtColor(disp_img.astype("uint8"), cv2.COLOR_RGB2YUV_I420)

        if frame_range is not None:
            try:
                cv2.putText(
                    disp_img,
                    str(frame_range[i]),
                    txt_pos,
                    font,
                    1,
                    255,
                    2,
                    cv2.LINE_AA,
                )
            except (IndexError, ValueError):
                # len(frame_range) M < len(frames) or
                # txt_pos is outside of the frame dimensions
                print("Could not overlay frame number on preview on video.")
        try:
            pipe.stdin.write(disp_img.astype("uint8").tobytes())
        except BrokenPipeError:
            return disp_img

    if close_pipe:
        pipe.communicate()
        return None
    else:
        return pipe


def make_timebase_uniform(timestamps, frames, use_timestamp_field="system_timestamp", period=0.01):
    import pandas as pd

    # TODO: add timestamp check as well, not just capture number
    capture_diff = timestamps["capture_number"].diff()
    gaps = capture_diff > 1
    nmissing_frames = capture_diff[gaps] - 1
    new_timestamps = timestamps.set_index("capture_number")[use_timestamp_field]

    to_insert_index = []
    to_insert_values = []
    to_insert_array_loc = []
    for _index, _n_missing in nmissing_frames.items():
        new_idx = np.arange(-_n_missing, 0)
        cap_number = timestamps.loc[_index]["capture_number"]
        last_good_value = new_timestamps.loc[cap_number]
        to_insert_index += (cap_number + new_idx).tolist()
        to_insert_values += (last_good_value + new_idx * period).tolist()
        to_insert_array_loc.append((new_timestamps.index.get_loc(cap_number) + 1, len(new_idx)))

    insert_series = pd.Series(to_insert_values, index=to_insert_index, name=use_timestamp_field)
    insert_series.index.name = "capture_number"

    new_timestamps = pd.concat([new_timestamps, insert_series]).sort_index()

    new_timestamps = new_timestamps.rename("timestamp")
    new_timestamps.index.name = "frame_index"

    new_frames = frames.copy()
    total_frames, height, width = new_frames.shape

    shift = 0
    for _loc, _nframes in to_insert_array_loc:
        pad_array = np.zeros((_nframes, height, width))
        pad_array[:] = np.nan
        new_frames = np.concatenate(
            [new_frames[: _loc + shift], pad_array, new_frames[_loc + shift :]]
        )
        shift += _nframes

    return new_timestamps, new_frames


def get_bground(
    dat_path,
    spacing=500,
    frame_size=(640, 480),
    dtype=np.dtype("<u2"),
    agg_func=np.nanmean,
    valid_range=(1000, 2000),
    median_kernels=(3, 5),
    **kwargs,
):
    from scipy import interpolate

    vid_inf = get_raw_info(dat_path, dtype=dtype, frame_size=frame_size)
    use_frames = list(range(0, vid_inf["nframes"], spacing))
    bground_frames = read_frames_raw(
        dat_path, frames=use_frames, dtype=dtype, frame_size=frame_size, **kwargs
    ).astype("float32")
    nframes, height, width = bground_frames.shape
    bground_frames[bground_frames < valid_range[0]] = np.nan
    bground_frames[bground_frames > valid_range[1]] = np.nan
    bground = agg_func(bground_frames, axis=0)

    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    zz = bground
    valid_pxs = ~np.isnan(zz.ravel())

    newz = interpolate.griddata(
        (xx.ravel()[valid_pxs], yy.ravel()[valid_pxs]),
        zz.ravel()[valid_pxs],
        (xx, yy),
        method="nearest",
        fill_value=np.nan,
    )

    bground = newz
    # # interpolate nans and filter
    for _med in median_kernels:
        bground = cv2.medianBlur(bground, _med)

    return bground


def pixel_format_to_np_dtype(pixel_format: str):
    if pixel_format == "Coord3D_C16":
        np_dtype = np.dtype("<u2")

    return np_dtype
