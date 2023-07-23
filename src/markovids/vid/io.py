import os
import numpy as np
import subprocess
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from typing import Optional


class MP4WriterPreview:
    def __init__(
        self,
        filepath,
        frame_size=(640, 480),
        fps=100,
        pixel_format="yuv420p",
        codec="h264",
        threads=6,
        slices=25,
        slicecrc=1,
        crf=28,
        cmap="turbo",
        font=cv2.FONT_HERSHEY_SIMPLEX,
        txt_pos=(30, 30),
    ):
        ext = os.path.splitext(filepath)[1]
        if ext != ".mp4":
            raise RuntimeError("Must use mp4 container (extension must be mp4)")
        self.filepath = filepath
        self.fps = fps
        self.pixel_format = pixel_format
        self.codec = codec
        self.threads = threads
        self.slices = slices
        self.slicecrc = slicecrc
        self.frame_size = frame_size
        self.pipe = None
        self.crf = crf
        self.cmap = plt.get_cmap(cmap)  # only used for intensity images
        self.txt_pos = txt_pos
        self.font = font

    def open(self):
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "fatal",
            "-framerate",
            str(self.fps),
            "-f",
            "rawvideo",
            "-s",
            "{:d}x{:d}".format(*self.frame_size),
            "-pix_fmt",
            self.pixel_format,
            "-i",
            "-",
            "-an",
            "-vcodec",
            self.codec,
            "-threads",
            str(self.threads),
            "-slices",
            str(self.slices),
            "-slicecrc",
            str(self.slicecrc),
            "-crf",
            str(self.crf),
            "-r",
            str(self.fps),
            self.filepath,
        ]

        self.pipe = subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.STDOUT
        )

    def pseudocolor_frames(self, frames, vmin=0, vmax=100):
        pseudo_ims = np.zeros(
            (len(frames), int(self.frame_size[1] * 1.5), self.frame_size[0]), dtype="uint8"
        )
        for i, _img in enumerate(frames):
            disp_img = _img.copy().astype("float32")
            disp_img = (disp_img - vmin) / (vmax - vmin)
            disp_img[disp_img < 0] = 0
            disp_img[disp_img > 1] = 1
            disp_img = np.delete(self.cmap(disp_img), 3, 2) * 255
            disp_img = cv2.cvtColor(disp_img.astype("uint8"), cv2.COLOR_RGB2YUV_I420)
            pseudo_ims[i] = disp_img
        return pseudo_ims

    def inscribe_frame_number(self, frame, idx):
        cv2.putText(
            frame,
            str(idx),
            self.txt_pos,
            self.font,
            1,
            255,
            2,
            cv2.LINE_AA,
        )

    def write_frames(
        self, frames, frames_idx=None, progress_bar=True, vmin=0, vmax=100,
    ):  # may need to enforce endianness...
        if self.pipe is None:
            self.open()

        if frames_idx is None:
            frames_idx = range(len(frames))

        assert len(frames) == len(frames_idx)

        if frames.ndim == 3:
            write_frames = self.pseudocolor_frames(frames, vmin=vmin, vmax=vmax)
        elif frames.ndim == 4:
            write_frames = frames
        else:
            raise RuntimeError("Wrong number of dims in frame data")

        for _idx, _frame in tqdm(
            zip(frames_idx, write_frames), total=len(frames), disable=not progress_bar
        ):
            _tmp_frame = _frame.copy()
            self.inscribe_frame_number(_tmp_frame, _idx)
            self.pipe.stdin.write(_tmp_frame.tobytes())

    def close(self):
        self.pipe.stdin.close()
        self.pipe.wait()
        return None


class AviWriter:
    def __init__(
        self,
        filepath,
        frame_size=(640, 480),
        dtype=np.dtype("<u2"),
        fps=100,
        pixel_format="gray16le",
        codec="ffv1",
        threads=6,
        slices=25,
        slicecrc=1,
    ):
        ext = os.path.splitext(filepath)[1]
        if ext != ".avi":
            raise RuntimeError("Must use avi container (extension must be avi)")
        self.filepath = filepath
        self.fps = fps
        self.pixel_format = pixel_format
        self.codec = codec
        self.threads = threads
        self.slices = slices
        self.slicecrc = slicecrc
        self.frame_size = frame_size
        self.dtype = dtype
        self.pipe = None

    def open(self):
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "fatal",
            "-framerate",
            str(self.fps),
            "-f",
            "rawvideo",
            "-s",
            "{:d}x{:d}".format(*self.frame_size),
            "-pix_fmt",
            self.pixel_format,
            "-i",
            "-",
            "-an",
            "-vcodec",
            self.codec,
            "-threads",
            str(self.threads),
            "-slices",
            str(self.slices),
            "-slicecrc",
            str(self.slicecrc),
            "-r",
            str(self.fps),
            self.filepath,
        ]

        self.pipe = subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.STDOUT
        )

    def write_frames(
        self, frames, progress_bar=True
    ):  # may need to enforce endianness...
        if self.pipe is None:
            self.open()
        for i in tqdm(range(len(frames)), disable=not progress_bar):
            self.pipe.stdin.write(frames[i].astype(self.dtype).tobytes())

    def close(self):
        self.pipe.stdin.close()
        self.pipe.wait()
        return None


def AutoReader(filepath, **kwargs):
    ext = os.path.splitext(filepath)[1]
    if ext == ".dat":
        reader = RawFileReader(filepath, **kwargs)
    elif ext == ".avi":
        reader = AviReader(filepath, **kwargs)
    return reader

    
class RawFileReader:
    def __init__(
        self,
        filepath,
        frame_size=(640, 480),
        dtype=np.dtype("<u2"),
        intrinsic_matrix=None,
        distortion_coeffs=None,
    ):
        self.dtype = dtype
        self.filepath = filepath
        self.frame_size = frame_size
        self.npixels = np.prod(self.frame_size)
        self.distortion_coeffs = distortion_coeffs
        self.intrinsic_matrix = intrinsic_matrix
        self.file_object = None
        self.downsample = None
        self.get_file_info()

    def open(self):
        self.file_object = open(self.filepath, "rb")

    def get_frames(self, frame_range=None):
        if self.file_object is None:
            self.open()
        skip_read = False
        if frame_range is None:
            # if frames is None load everything
            seek_point = 0
            read_points = self.nframes
            dims = (self.nframes, self.frame_size[1], self.frame_size[0])
        elif isinstance(frame_range, tuple):
            # if it's a tuple, assume it's left_edge, right_edge, turn into range
            use_range = frame_range
            use_range[1] = min(use_range[1], self.nframes)
            seek_point = use_range[0] * self.bytes_per_frame
            read_points = use_range[1] * self.npixels
            dims = ((use_range[1] - use_range[0]) + 1, self.frame_size[1], self.frame_size[0])
        elif isinstance(frame_range, (int, np.integer)):
            seek_point = frame_range * self.bytes_per_frame
            read_points = self.npixels
            dims = (self.frame_size[1], self.frame_size[0])
        elif isinstance(frame_range, range):
            _tmp = np.asarray(list(frame_range))
            _tmp = _tmp[_tmp<self.nframes]
            seek_point = _tmp[0] * self.bytes_per_frame
            read_points = ((_tmp[-1] - _tmp[0]) + 1) * self.npixels
            dims = ((_tmp[-1] - _tmp[0]) + 1, self.frame_size[1], self.frame_size[0])
            # run through each element in the list
        elif isinstance(frame_range, (list, np.ndarray)):
            frame_range = np.asarray(frame_range)
            nframes = len(frame_range)
            dat = np.zeros((nframes, self.frame_size[1], self.frame_size[0]), dtype=self.dtype)
            for i, _frame in enumerate(frame_range[frame_range<self.nframes]):
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


class AviReader:
    def __init__(
        self,
        filepath,
        threads=6,
        slices=24,
        slicecrc=1,
        intrinsic_matrix=None,
        distortion_coeffs=None,
    ):
        self.filepath = filepath
        self.threads = threads
        self.slices = slices
        self.slicecrc = slicecrc
        self.intrinsic_matrix = intrinsic_matrix
        self.distortion_coeffs = distortion_coeffs
        self.get_file_info()

    def open(self):
        pass

    def close(self):
        pass

    def get_file_info(self):
        command = [
            "ffprobe",
            "-v",
            "fatal",
            "-show_entries",
            "stream=width,height,pix_fmt,r_frame_rate,bits_per_raw_sample,nb_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            self.filepath,
            "-sexagesimal",
        ]

        ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        out, err = ffmpeg.communicate()

        if err:
            print(err)
        out = out.decode().split("\n")
        self.frame_size = (int(out[0]), int(out[1]))
        self.pixel_format = out[2]
        self.fps = float(out[3].split("/")[0]) / float(out[3].split("/")[1])
        self.bit_depth = int(out[4])
        self.nframes = int(out[5])

        if self.bit_depth == 16:
            self.dtype = np.dtype("<u2")
        elif self.bit_depth == 8:
            self.dtype = np.dtype("<u1")
        else:
            raise RuntimeError(f"Cannot work with bit depth {self.bit_depth}")

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

    def get_frames(self, frame_range=None):
        import datetime
        list_order = None
        if frame_range is None:
            use_frames = np.arange(self.nframes).astype("int16")
            frame_select = [
                "-ss",
                str(datetime.timedelta(seconds=use_frames[0] / self.fps)),
                "-vframes",
                str(len(use_frames)),
            ]
        elif isinstance(frame_range, range):
            use_frame_range = np.asarray(list(frame_range))
            use_frame_range = use_frame_range[use_frame_range<self.nframes]
            frame_select = [
                "-ss",
                str(datetime.timedelta(seconds=use_frame_range[0] / self.fps)),
                "-vframes",
                str(len(use_frame_range)),
            ]
        elif isinstance(frame_range, (list, np.ndarray)):
            # NEED TO REORDER USING THE LIST ORDER
            use_frame_range = np.asarray(frame_range)
            use_frame_range = use_frame_range[use_frame_range<self.nframes]
            list_order = np.argsort(np.argsort(use_frame_range))
            list_string = "+".join([f"eq(n\,{_frame})" for _frame in use_frame_range])
            # list_string = 'eq(n\,1)'
            frame_select = [
                "-vf",
                f"select={list_string}",
                "-vsync",
                "0",
                "-vframes",
                str(len(use_frame_range)),
            ]
        elif isinstance(frame_range, int):
            frame_select = [
                "-ss",
                str(datetime.timedelta(seconds=frame_range / self.fps)),
                "-vframes",
                "1",
            ]
        else:
            raise RuntimeError("Did not understand frame range")

        command = (
            ["ffmpeg", "-loglevel", "fatal", "-i", self.filepath]
            + frame_select
            + [
                "-f",
                "image2pipe",
                "-s",
                "{:d}x{:d}".format(*self.frame_size),
                "-pix_fmt",
                self.pixel_format,
                "-threads",
                str(self.threads),
                "-slices",
                str(self.slices),
                "-slicecrc",
                str(self.slicecrc),
                "-vcodec",
                "rawvideo",
                "-",
            ]
        )

        pipe = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        out, err = pipe.communicate()
        if err:
            print("error", err)
            return None

        total_bytes = len(out)
        bytes_per_frame = (self.bit_depth / 8) * np.prod(self.frame_size)
        n_out_frames = int(total_bytes / bytes_per_frame)
        dat = np.frombuffer(out, dtype=self.dtype).reshape(
            (n_out_frames, self.frame_size[1], self.frame_size[0])
        )
        if list_order is not None:
            dat = dat[list_order]
        return dat


def downsample_frames(frames, downsample=4):
    # short circuit
    if downsample == 1:
        return frames
    nframes, height, width = frames.shape[:3]
    ds_frames = np.zeros(
        (nframes, height // downsample, width // downsample), dtype=frames.dtype
    )
    for i in range(len(frames)):
        ds_frames[i] = cv2.resize(frames[i], ds_frames.shape[1:][::-1])
    return ds_frames


default_config = {"dtype": np.dtype("<u2"), "frame_size": (640, 480)}


def read_frames_multicam(
    paths: dict,
    frames: dict,
    config: dict = {},
    tick_period: float = 1e9,
    progress_bar: bool = True,
    downsample: Optional[int] = None,
):
    # config should contain frame_size and numpy data type
    # PATHs should be dictionary where camera is key...
    # CONFIGs should be dictionary where camera is key...
    # make sure we support inflating frames with nans...
    dat = {}
    for _path, _cam in paths.items():

        try:
            use_config = default_config | config[_cam]
        except KeyError:
            use_config = default_config
        ext = os.path.splitext(_path)[1]

        reader = AutoReader(_path, **use_config)
        _dat = reader.open()
        _dat = reader.get_frames(frames[_cam])
        _dat = reader.undistort_frames(_dat)

        if downsample is not None:
            _dat = downsample_frames(_dat, downsample)

        dat[_cam] = _dat
        reader.close()

    return dat


def fill_timestamps(timestamps, use_timestamp_field="device_timestamp", capture_number="frame_id", period=0.01):
    # TODO: add timestamp check as well, not just capture number
    capture_diff = timestamps[capture_number].diff()
    gaps = capture_diff > 1
    nmissing_frames = capture_diff[gaps] - 1

    new_timestamps = timestamps.copy()
    new_timestamps.index.name = "frame_index"
    # new_timestamps["is_pad"] = False
    new_timestamps = new_timestamps.reset_index().set_index(capture_number)[
        ["frame_index", use_timestamp_field]
    ]

    to_insert_index = []
    to_insert_values = []
    to_insert_array_loc = []
    for _index, _n_missing in nmissing_frames.items():
        new_idx = np.arange(-_n_missing, 0)
        cap_number = timestamps.loc[_index][capture_number]
        last_good_value = new_timestamps.loc[cap_number, use_timestamp_field]
        to_insert_index += (cap_number + new_idx).tolist()
        to_insert_values += (last_good_value + new_idx * period).tolist()
        to_insert_array_loc.append((new_timestamps.index.get_loc(cap_number) + 1, len(new_idx)))

    to_insert_values = {
        "frame_index": [np.nan] * len(to_insert_values),
        use_timestamp_field: to_insert_values,
    }
    insert_df = pd.DataFrame(to_insert_values, index=to_insert_index)
    insert_df.index.name = capture_number
    new_timestamps = pd.concat([new_timestamps, insert_df]).sort_index()
    new_timestamps.index = new_timestamps.index.astype("int")
    # new_timestamps.index = range(len(new_timestamps))  # should be contiguous anyhow...
    new_timestamps["frame_index"] = new_timestamps["frame_index"].astype("Int32")

    return new_timestamps


def read_timestamps(path, tick_period=1e9, fill=False, fill_kwargs={}):
    import io

    with open(path, "r") as table:
        buffer = io.StringIO("\n".join(line.strip() for line in table))
        df = pd.read_table(buffer, delimiter="\t")
    df.index.name = "frame_index"
    try:
        df["frame_id"] = df["frame_id"].astype("int")
    except KeyError:
        pass
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

    ts = {}
    for _path, _cam in path.items():
        ts[_cam] = read_timestamps(
            _path,
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
    merged_ts.index = list(ts.values())[0].index

    return ts, merged_ts


def get_bground(
    dat_path,
    spacing=500,
    frame_size=(640, 480),
    dtype=np.dtype("<u2"),
    agg_func=np.nanmean,
    valid_range=(1000, 2000),
    median_kernels=(3, 5),
    interpolate_invalid=True,
    **kwargs,
):
    from scipy import interpolate

    ext = os.path.splitext(dat_path)[1]
    reader = AutoReader(dat_path, frame_size=frame_size, dtype=dtype, **kwargs)
    reader.open()
    use_frames = list(range(0, reader.nframes, spacing))
    bground_frames = reader.get_frames(use_frames).astype("float32")
    bground_frames = reader.undistort_frames(bground_frames)
    reader.close()

    if valid_range is not None: 
        bground_frames[bground_frames < valid_range[0]] = np.nan
        bground_frames[bground_frames > valid_range[1]] = np.nan
    bground = agg_func(bground_frames, axis=0)

    if valid_range is not None and interpolate_invalid:
        height, width = bground.shape
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
    elif pixel_format == "Mono8":
        np_dtype = np.dtype("<u1")
    else:
        raise RuntimeError("Did not understand pixel format!")

    return np_dtype
