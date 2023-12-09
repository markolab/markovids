import os
import numpy as np
import subprocess
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import toml
import warnings
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
        crf=23,
        cmap="turbo",
        font=cv2.FONT_HERSHEY_SIMPLEX,
        text_pos=(30, 30),
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
        
        # adjust frame size to be even width and height
        width, height = frame_size

        if np.mod(width, 2) != 0:
            width += 1
            pad_width = 1
        else:
            pad_width = 0
        
        if np.mod(height, 2) != 0:
            height += 1
            pad_height = 1
        else:
            pad_height = 0

        self.pad = (pad_width, pad_height)
        self.frame_size = (width, height)
        self.pipe = None
        self.crf = crf
        self.cmap = plt.get_cmap(cmap)  # only used for intensity images
        self.text_pos = text_pos
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

        self.pipe = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)


    def write_frames(
        self,
        frames,
        frames_idx=None,
        progress_bar=True,
        vmin=0,
        vmax=100,
        mark_frames=[],
        marker_color=[0, 0, 255],
        marker_size=.1,
        inscribe_frame_number=True,
    ):  # may need to enforce endianness...
        if self.pipe is None:
            self.open()

        if frames_idx is None:
            frames_idx = range(len(frames))

        assert len(frames) == len(frames_idx)

        if (self.pad[0] > 0) or (self.pad[1] > 0): 
            for i in range(len(frames)):
                frames[i] = cv2.copyMakeBorder(frames[i], 0, self.pad[1], 0, self.pad[0], cv2.BORDER_REFLECT)

        if frames.ndim == 3:
            write_frames = pseudocolor_frames(frames, vmin=vmin, vmax=vmax, cmap=self.cmap)
        elif frames.ndim == 4:
            write_frames = frames
        else:
            raise RuntimeError("Wrong number of dims in frame data")

        for _idx, _frame in tqdm(
            zip(frames_idx, write_frames), total=len(frames), disable=not progress_bar
        ):
            _tmp_frame = _frame.copy()
            if inscribe_frame_number:
                inscribe_text(_tmp_frame, str(_idx), font=self.font, text_pos=self.text_pos)
            if _idx in mark_frames:
                mark_frame(_tmp_frame, marker_color, marker_size) 
            _tmp_frame = cv2.cvtColor(_tmp_frame.astype("uint8"), cv2.COLOR_RGB2YUV_I420)
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

        self.pipe = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)

    def write_frames(self, frames, progress_bar=True):  # may need to enforce endianness...
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
        frame_size=None,
        dtype=None,
        intrinsic_matrix=None,
        distortion_coeffs=None,
    ):
        # attempt to retrieve metadata
        metadata = toml.load(os.path.join(os.path.dirname(filepath), "metadata.toml"))
        cam, ext = os.path.splitext(os.path.basename(filepath))

        if frame_size is None:
            frame_size = (metadata[cam]["Width"], metadata[cam]["Height"])

        if dtype is None:
            dtype = pixel_format_to_np_dtype(metadata[cam]["PixelFormat"])

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
            _tmp = _tmp[_tmp < self.nframes]
            if frame_range.step == 1:
                seek_point = _tmp[0] * self.bytes_per_frame
                read_points = ((_tmp[-1] - _tmp[0]) + 1) * self.npixels
                dims = ((_tmp[-1] - _tmp[0]) + 1, self.frame_size[1], self.frame_size[0])
            else:
                # if spacing is non-contiguous process as list
                dat = self.get_frames(_tmp)
                return dat
        elif isinstance(frame_range, (list, np.ndarray)):
            frame_range = np.asarray(frame_range)
            nframes = len(frame_range)
            dat = np.zeros((nframes, self.frame_size[1], self.frame_size[0]), dtype=self.dtype)
            for i, _frame in enumerate(frame_range[frame_range < self.nframes]):
                dat[i] = self.get_frames(_frame)
            return dat
        else:
            raise RuntimeError("Did not understand frame range type")

        self.file_object.seek(int(seek_point))
        dat = np.fromfile(file=self.file_object, dtype=self.dtype, count=read_points).reshape(dims)

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
        else:
            warnings.warn("No intrinsic matrix or distortion matrix, skipping undistortion")

        return frames

    def get_file_info(self):
        self.total_bytes = os.stat(self.filepath).st_size
        self.bytes_per_frame = np.prod(self.frame_size) * self.dtype.itemsize
        self.nframes = int(os.stat(self.filepath).st_size / self.bytes_per_frame)
        self.dims = (self.nframes, self.frame_size[1], self.frame_size[0])


# class AviReader:
#     def __init__(
#         self,
#         filepath,
#         threads=16,
#         slices=24,
#         slicecrc=1,
#         intrinsic_matrix=None,
#         distortion_coeffs=None,
#         **kwargs,
#     ):
#         self.filepath = filepath
#         self.threads = threads
#         self.slices = slices
#         self.slicecrc = slicecrc
#         self.intrinsic_matrix = intrinsic_matrix
#         self.distortion_coeffs = distortion_coeffs
#         self.get_file_info()

#     def open(self):
#         pass

#     def close(self):
#         pass

#     def get_file_info(self):
#         command = [
#             "ffprobe",
#             "-v",
#             "fatal",
#             "-show_entries",
#             "stream=width,height,pix_fmt,r_frame_rate,bits_per_raw_sample,nb_frames",
#             "-of",
#             "default=noprint_wrappers=1:nokey=1",
#             self.filepath,
#             "-sexagesimal",
#         ]

#         ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
#         out, err = ffmpeg.communicate()

#         if err:
#             print(err)
#         out = out.decode().split("\n")
#         self.frame_size = (int(out[0]), int(out[1]))
#         self.pixel_format = out[2]
#         self.fps = float(out[3].split("/")[0]) / float(out[3].split("/")[1])
#         self.bit_depth = int(out[4])
#         self.nframes = int(out[5])

#         if self.bit_depth == 16:
#             self.dtype = np.dtype("<u2")
#         elif self.bit_depth == 8:
#             self.dtype = np.dtype("<u1")
#         else:
#             raise RuntimeError(f"Cannot work with bit depth {self.bit_depth}")

#     def undistort_frames(self, frames, progress_bar=True):
#         if (self.intrinsic_matrix is not None) and (self.distortion_coeffs is not None):
#             for i, _frame in tqdm(
#                 enumerate(frames),
#                 total=len(frames),
#                 desc="Removing frame distortion",
#                 disable=not progress_bar,
#             ):
#                 frames[i] = cv2.undistort(_frame, self.intrinsic_matrix, self.distortion_coeffs)
#         return frames

#     def get_frames(self, frame_range=None):
#         import datetime
#         list_order = None
#         if frame_range is None:
#             use_frames = np.arange(self.nframes).astype("int16")
#             frame_select = [
#                 "-ss",
#                 str(datetime.timedelta(seconds=use_frames[0] / self.fps)),
#                 "-vframes",
#                 str(len(use_frames)),
#             ]
#         elif isinstance(frame_range, range):
#             # ASSUMES THE RANGE IS CONTIGUOUS!!!!
#             # TODO: add a check
#             use_frame_range = np.asarray(list(frame_range))
#             use_frame_range = use_frame_range[use_frame_range<self.nframes]
#             frame_select = [
#                 "-ss",
#                 str(datetime.timedelta(seconds=use_frame_range[0] / self.fps)),
#                 "-vframes",
#                 str(len(use_frame_range)),
#             ]
#         elif isinstance(frame_range, (list, np.ndarray)):
#             # NEED TO REORDER USING THE LIST ORDER
#             use_frame_range = np.asarray(frame_range)
#             use_frame_range = use_frame_range[use_frame_range<self.nframes]
#             list_order = np.argsort(np.argsort(use_frame_range))
#             list_string = "+".join([f"eq(n\,{_frame})" for _frame in use_frame_range])
#             # list_string = 'eq(n\,1)'
#             frame_select = [
#                 "-vf",
#                 f"select={list_string}",
#                 "-vsync",
#                 "0",
#                 "-vframes",
#                 str(len(use_frame_range)),
#             ]
#         elif isinstance(frame_range, int):
#             frame_select = [
#                 "-ss",
#                 str(datetime.timedelta(seconds=frame_range / self.fps)),
#                 "-vframes",
#                 "1",
#             ]
#         else:
#             raise RuntimeError("Did not understand frame range")

#         command = (
#             ["ffmpeg", "-loglevel", "fatal", "-i", self.filepath]
#             + frame_select
#             + [
#                 "-f",
#                 "image2pipe",
#                 "-s",
#                 "{:d}x{:d}".format(*self.frame_size),
#                 "-pix_fmt",
#                 self.pixel_format,
#                 "-threads",
#                 str(self.threads),
#                 "-slices",
#                 str(self.slices),
#                 "-slicecrc",
#                 str(self.slicecrc),
#                 "-vcodec",
#                 "rawvideo",
#                 "-",
#             ]
#         )

#         pipe = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
#         out, err = pipe.communicate()
#         if err:
#             print("error", err)
#             return None

#         total_bytes = len(out)
#         bytes_per_frame = (self.bit_depth / 8) * np.prod(self.frame_size)
#         n_out_frames = int(total_bytes / bytes_per_frame)
#         dat = np.frombuffer(out, dtype=self.dtype).reshape(
#             (n_out_frames, self.frame_size[1], self.frame_size[0])
#         )
#         if list_order is not None:
#             dat = dat[list_order]
#         return dat


class AviReader:
    def __init__(
        self,
        filepath,
        threads=16,
        intrinsic_matrix=None,
        distortion_coeffs=None,
        **kwargs,
    ):
        self.filepath = filepath
        self.threads = threads
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
        else:
            warnings.warn("No intrinsic matrix or distortion coefficients, skipping undistortion")
        return frames

    def get_frames(self, frame_range=None, fast_seek=False):
        import datetime

        list_order = None
        input_opts = []
        output_opts = []
        if frame_range is None:
            use_frames = np.arange(self.nframes).astype("int16")
            output_opts = [
                "-ss",
                str(datetime.timedelta(seconds=use_frames[0] / self.fps)),
                "-vframes",
                str(len(use_frames)),
            ]
        elif isinstance(frame_range, range):
            use_frame_range = np.asarray(list(frame_range))
            use_frame_range = use_frame_range[use_frame_range < self.nframes]
            if frame_range.step == 1:
                input_opts = [
                    "-ss",
                    str(datetime.timedelta(seconds=use_frame_range[0] / self.fps)),
                ]
                output_opts = [
                    "-vframes",
                    str(len(use_frame_range)),
                ]
            else:
                dat = self.get_frames(use_frame_range)
                return dat
        elif isinstance(frame_range, (list, np.ndarray)):
            use_frame_range = np.asarray(frame_range)
            use_frame_range = use_frame_range[use_frame_range < self.nframes]
            # 1) process as range if frames are contiguous
            # 2) if spacing is large grab individual frames
            # 3) if spacing is small process using the vf select filter
            # 4) with fast seek turned on use more ram to read the file then return indexed results...
            if (np.diff(use_frame_range) == 1).all():
                dat = self.get_frames(range(use_frame_range[0], use_frame_range[-1] + 1))
                return dat
            elif fast_seek:
                dat = self.get_frames(range(min(use_frame_range), max(use_frame_range) + 1))
                idx = use_frame_range - min(use_frame_range)
                return dat[idx]
            elif np.abs(np.diff(use_frame_range)).mean() >= 10:
                dat = []
                for _frame in use_frame_range:
                    dat.append(self.get_frames(int(_frame)).squeeze())
                return np.array(dat)
            else:
                list_order = np.argsort(np.argsort(use_frame_range))
                list_string = "+".join([f"eq(n\,{_frame})" for _frame in use_frame_range])
                output_opts = [
                    "-vf",
                    f"select={list_string}",
                    "-vsync",
                    "0",
                    "-vframes",
                    str(len(use_frame_range)),
                ]
        elif isinstance(frame_range, int):
            input_opts = [
                "-ss",
                str(datetime.timedelta(seconds=frame_range / self.fps)),
            ]
            output_opts = [
                "-vframes",
                "1",
            ]
        else:
            raise RuntimeError("Did not understand frame range")

        command = (
            ["ffmpeg", "-loglevel", "fatal"]
            + input_opts
            + ["-i", 
               self.filepath,
                "-threads",
                str(self.threads),
            ]
            + output_opts
            + [
                "-f",
                "image2pipe",
                "-s",
                "{:d}x{:d}".format(*self.frame_size),
                "-pix_fmt",
                self.pixel_format,
                 "-threads",
                str(self.threads),
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
    ds_frames = np.zeros((nframes, height // downsample, width // downsample), dtype=frames.dtype)
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
        if type(reader).__name__ == "AviReader":
            _dat = reader.get_frames(frames[_cam], fast_seek=True).copy()
        else:
            _dat = reader.get_frames(frames[_cam]).copy()
        _dat = reader.undistort_frames(_dat, progress_bar=progress_bar)

        if downsample is not None:
            _dat = downsample_frames(_dat, downsample)

        dat[_cam] = _dat
        reader.close()

    return dat


def fill_timestamps(
    timestamps, use_timestamp_field="device_timestamp", capture_number="frame_id", period=0.01
):
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
        df["frame_id"] = df["capture_number"].astype("int")
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


def inscribe_text(frame, text, font=cv2.FONT_HERSHEY_SIMPLEX, text_pos=(30, 30)):
    cv2.putText(
        frame,
        text,
        text_pos,
        font,
        1,
        (255,255,255),
        2,
        cv2.LINE_AA,
    )


def pseudocolor_frames(frames, vmin=0, vmax=100, cmap=plt.get_cmap("turbo")):
    nframes, height, width = frames.shape
    pseudo_ims = np.zeros(
        (nframes, height, width, 3), dtype="uint8"
    )
    for i, _img in enumerate(frames):
        disp_img = _img.copy().astype("float32")
        disp_img = (disp_img - vmin) / (vmax - vmin)
        disp_img[disp_img < 0] = 0
        disp_img[disp_img > 1] = 1
        disp_img = np.delete(cmap(disp_img), 3, 2) * 255
        pseudo_ims[i] = disp_img
    return pseudo_ims


def mark_frame(frame, marker_color, marker_size):
    # bottom right for now
    height, width, channels = frame.shape
    x1, y1 = width, height
    x2, y2 = int(x1 - width * marker_size), int(y1 - height * marker_size)

    cv2.rectangle(
        frame,
        (x2, y2),
        (x1, y1),
        marker_color,
        -1
    )


def inscribe_masks(frames, masks, colors={1: [255,0,0], 2: [0,0,255]}, alpha=.5):
    color_mask = np.zeros_like(frames)
    for _idx, _color in colors.items():
        mask_idx = (masks == _idx)
        color_mask[mask_idx, :] = _color
    use_frames = frames * (1 - alpha) + alpha * color_mask
    return use_frames.astype("uint8")


def get_bground(
    dat_path,
    spacing=500,
    frame_size=(640, 480),
    dtype=np.dtype("<u2"),
    agg_func=np.nanmean,
    valid_range=(1000, 2000),
    median_kernels=(3, 5),
    interpolate_invalid=True,
    use_frames=None,
    **kwargs,
):
    from scipy import interpolate

    ext = os.path.splitext(dat_path)[1]
    reader = AutoReader(dat_path, frame_size=frame_size, dtype=dtype, **kwargs)
    reader.open()
    if use_frames is None:
        use_frames = range(0, reader.nframes, spacing)
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


def format_intrinsics(intrinsics):
    intrinsic_matrix = {}
    distortion_coeffs = {}
    for k, v in intrinsics.items():
        intrinsic_matrix[k] = np.array(
            [
                [v["CalibFocalLengthX"], 0, v["CalibOpticalCenterX"]],
                [0, v["CalibFocalLengthY"], v["CalibOpticalCenterY"]],
                [0, 0, 1],
            ]
        )
        distortion_coeffs[k] = np.array([v["k1"], v["k2"], v["p1"], v["p2"], v["k3"]])
    return intrinsic_matrix, distortion_coeffs
