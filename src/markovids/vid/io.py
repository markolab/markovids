import os
import numpy as np


def get_raw_info(filename, bit_depth=16, frame_size=(512, 424)):
    bytes_per_frame = (frame_size[0] * frame_size[1] * bit_depth) / 8
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
    bit_depth=16,
    movie_dtype="<u2",
    **kwargs,
):
    vid_info = get_raw_info(filename, frame_size=frame_size, bit_depth=bit_depth)

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
        chunk = np.fromfile(file=f, dtype=np.dtype(movie_dtype), count=read_points).reshape(dims)

    return chunk
