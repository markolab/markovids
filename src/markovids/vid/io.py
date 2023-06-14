import os
import numpy as np
import subprocess
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm



def get_raw_info(filename, dtype=np.dtype("<u2"), frame_size=(512, 424)):
    bytes_per_frame = frame_size[0] * frame_size[1] * dtype.itemsize
    file_info = {
        "bytes": os.stat(filename).st_size,
        "nframes": int(os.stat(filename).st_size / bytes_per_frame),
        "dims": frame_size,
        "bytes_per_frame": bytes_per_frame,
    }
    return file_info


def read_frames_raw(
    filename,
    frames=None,
    frame_size=(512, 424),
    movie_dtype="<u2",
    intrinsic_matrix=None,
    distortion_coeffs=None,
    **kwargs,
):

    # TODO, if any frame indices are a nan or less than 0, save to insert into array...
    dtype = np.dtype(movie_dtype) 
    vid_info = get_raw_info(filename, frame_size=frame_size, dtype=dtype)

    if vid_info["dims"] != frame_size:
        frame_size = vid_info["dims"]

    if type(frames) is int:
        frames = [frames]
    elif not frames or (type(frames) is range) and len(frames) == 0:
        frames = range(0, vid_info["nframes"])

    seek_point = np.maximum(0, frames[0] * vid_info["bytes_per_frame"])
    read_points = len(frames) * frame_size[0] * frame_size[1]

    dims = (len(frames), frame_size[1], frame_size[0])
    with open(filename, "rb") as f:
        f.seek(int(seek_point))
        chunk = np.fromfile(file=f, dtype=dtype, count=read_points).reshape(dims)

    if (intrinsic_matrix is not None) and (distortion_coeffs is not None):
        for i, _frame in tqdm(enumerate(chunk), total=len(chunk), desc="Removing frame distortion"):
            chunk[i] = cv2.undistort(_frame, intrinsic_matrix, distortion_coeffs)

    return chunk


def fill_timestamps(
    timestamps, use_timestamp_field="device_timestamp", period=0.01
):
    
    # TODO: add timestamp check as well, not just capture number
    capture_diff = timestamps["capture_number"].diff()
    gaps = capture_diff > 1
    nmissing_frames = capture_diff[gaps] - 1
    
    new_timestamps = timestamps.copy()
    new_timestamps.index.name = "frame_index"
    # new_timestamps["is_pad"] = False
    new_timestamps = new_timestamps.reset_index().set_index("capture_number")[["frame_index", use_timestamp_field]]

    to_insert_index = []
    to_insert_values = []
    to_insert_array_loc = []
    for _index, _n_missing in nmissing_frames.items():
        new_idx = np.arange(-_n_missing, 0)
        cap_number = timestamps.loc[_index]["capture_number"]
        last_good_value = new_timestamps.loc[cap_number, use_timestamp_field]
        to_insert_index += (cap_number + new_idx).tolist()
        to_insert_values += (last_good_value + new_idx * period).tolist()
        to_insert_array_loc.append(
            (new_timestamps.index.get_loc(cap_number) + 1, len(new_idx))
        )

    to_insert_values = {"frame_index": [np.nan] * len(to_insert_values),
                        use_timestamp_field: to_insert_values}
    insert_df = pd.DataFrame(
        to_insert_values, index=to_insert_index
    )
    insert_df.index.name = "capture_number"
    new_timestamps = pd.concat([new_timestamps, insert_df]).sort_index()
    new_timestamps.index = range(len(new_timestamps)) # should be contiguous anyhow...
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


def read_timestamps_multicam(path: dict, use_timestamp_field: str = "system_timestamp", merge_tolerance=.0035, fill=True): 
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
            left, right, tolerance=merge_tolerance, on=use_timestamp_field, direction="nearest",
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
        pipe = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

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


def make_timebase_uniform(
    timestamps, frames, use_timestamp_field="system_timestamp", period=0.01
):
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
        to_insert_array_loc.append(
            (new_timestamps.index.get_loc(cap_number) + 1, len(new_idx))
        )

    insert_series = pd.Series(
        to_insert_values, index=to_insert_index, name=use_timestamp_field
    )
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