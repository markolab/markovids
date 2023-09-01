from typing import Tuple, Tuple
from markovids.vid.io import (
    format_intrinsics,
    get_bground,
    read_timestamps_multicam,
    read_frames_multicam,
    MP4WriterPreview,
)
from markovids.depth.plane import get_floor
from markovids.depth.io import load_segmentation_masks
from markovids.pcl.io import pcl_from_depth, depth_from_pcl_interpolate
from markovids.pcl.registration import (
    DepthVideoPairwiseRegister,
    correct_breakpoints,
    correct_breakpoints_extrapolate,
)
from tqdm.auto import tqdm
from scipy import ndimage
import os
import numpy as np
import toml
import cv2
import open3d as o3d
import h5py
import gc
import copy

o3d.utility.set_verbosity_level(o3d.utility.Error)


pcl_kwargs_default = {"project_xy": True}
registration_kwargs_default = {
    "max_correspondence_distance": 1.0,
    "fitness_threshold": 0.25,
    "reference_future_len": 100,
    "reference_history_len": 50,
    "cleanup_nbs": 5,
    "cleanup_nbs_combined": 15,
}


def convert_depth_to_pcl_and_register(
    data_dir: str,
    intrinsics_file: str,
    registration_dir: str = "_registration",
    segmentation_dir: str = "_segmentation_tau-5",
    background_spacing: int = 500,
    floor_range: Tuple[float, float] = (1300.0, 1600.0),
    timestamp_merge_tolerance=0.003,  # in seconds
    burn_frames: int = 500,
    valid_height_range: Tuple[float, float] = (10.0, 800.0),
    batch_size: int = 2000,
    batch_overlap: int = 150,
    voxel_down_sample: float = 1.0,
    pcl_kwargs: dict = {},
    registration_kwargs: dict = {},
):
    pcl_kwargs = pcl_kwargs_default | pcl_kwargs
    registration_kwargs = registration_kwargs_default | registration_kwargs

    registration_dir = os.path.join(data_dir, registration_dir)
    os.makedirs(registration_dir, exist_ok=True)

    session_name = os.path.basename(data_dir)
    intrinsics = toml.load(intrinsics_file)
    intrinsics_matrix, distortion_coeffs = format_intrinsics(intrinsics)

    cameras = list(intrinsics_matrix.keys())
    metadata = toml.load(os.path.join(data_dir, "metadata.toml"))
    z_scale = (
        1 / metadata["camera_metadata"][cameras[0]]["Scan3dCoordinateScale"]
    )  # assume this is common for now

    load_dct = {}

    for camera, cfg in metadata["camera_metadata"].items():
        load_dct[camera] = {}
        load_dct[camera]["intrinsic_matrix"] = intrinsics_matrix[camera]
        load_dct[camera]["distortion_coeffs"] = distortion_coeffs[camera]

    dat_paths_avi = {
        _cam: os.path.join(data_dir, f"{_cam}.avi")
        for _cam in cameras
        if os.path.exists(os.path.join(data_dir, f"{_cam}.avi"))
    }
    dat_paths_dat = {
        _cam: os.path.join(data_dir, f"{_cam}.dat")
        for _cam in cameras
        if os.path.exists(os.path.join(data_dir, f"{_cam}.dat"))
    }

    dat_paths = dat_paths_avi | dat_paths_dat
    multiframe_load_paths = {v: k for k, v in dat_paths.items()}
    ts_paths = {os.path.join(data_dir, f"{_cam}.txt"): _cam for _cam in cameras}

    bgrounds = {
        _cam: get_bground(_path, **load_dct[_cam], spacing=background_spacing)
        for _cam, _path in tqdm(
            dat_paths.items(), total=len(dat_paths), desc="Computing backgrounds"
        )
    }
    rois = {
        _cam: get_floor(bgrounds[_cam], floor_range=floor_range, dilations=10) for _cam in cameras
    }

    floor_distances = {}
    for _cam in tqdm(cameras, desc="Getting floor distances"):
        roi_floor = cv2.erode(
            rois[_cam], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=5
        )
        floor_distances[_cam] = np.nanmedian(bgrounds[_cam][roi_floor > 0].ravel())

    _, merged_ts = read_timestamps_multicam(ts_paths, merge_tolerance=timestamp_merge_tolerance)
    use_ts = merged_ts.dropna().iloc[burn_frames:]
    nframes = len(use_ts)

    str_dt = h5py.special_dtype(vlen=str)
    pcl_f = h5py.File(os.path.join(registration_dir, "pcls.hdf5"), "w")
    pcl_f.create_dataset(
        "xyz",
        (len(use_ts) * 4e3, 3),
        "float64",
        compression="lzf",
    )
    pcl_f.create_dataset(
        "frame_index",
        (len(use_ts) * 4e3,),
        "uint32",
        compression="lzf",
    )
    pcl_f.create_dataset("reference_node", (len(use_ts),), str_dt, compression="lzf")

    for k, v in registration_kwargs.items():
        if v is not None:
            pcl_f["xyz"].attrs[k] = v
        else:
            pcl_f["xyz"].attrs[k] = "None"
    pcl_f["xyz"].attrs["segmentation_dir"] = segmentation_dir

    save_metadata = copy.deepcopy(registration_kwargs)
    save_metadata["segmentation_dir"] = segmentation_dir
    save_metadata["z_scale"] = z_scale
    save_metadata["floor_distances"] = floor_distances
    save_metadata["camera_order"] = cameras
    save_metadata["timestamp_camera_primary"] = cameras[0]

    frame_batches = range(0, nframes, batch_size)

    all_bpoints = []
    pcl_count = 0
    last_bpoint_transform = None
    last_pcl = None
    last_reference_node = None

    for batch in tqdm(frame_batches, desc="Conversion to PCL and registration"):
        left_edge = max(batch - batch_overlap, 0)
        left_pad_size = batch - left_edge
        right_edge = min(batch + batch_size + batch_overlap, nframes)
        right_edge_no_pad = min(batch + batch_size, nframes)

        cur_ts = use_ts.iloc[left_edge:right_edge]
        read_frames = {_cam: cur_ts[_cam].astype("int32").to_list() for _cam in cameras}
        raw_dat = read_frames_multicam(
            multiframe_load_paths, read_frames, load_dct, progress_bar=False
        )
        roi_dats = {
            _cam: load_segmentation_masks(_path, read_frames[_cam], segmentation_dir)
            for _cam, _path in dat_paths.items()
        }

        # convert everything to point clouds
        pcls = {_cam: [] for _cam in cameras}
        for _cam in tqdm(cameras, desc="Converting to point clouds", disable=True):
            for _frame in range(len(cur_ts)):
                use_dat = raw_dat[_cam][_frame].copy().astype("float32")
                use_roi = roi_dats[_cam][_frame]
                bground_rem_dat = floor_distances[_cam] - use_dat
                invalid_mask = np.logical_or(
                    bground_rem_dat < valid_height_range[0], bground_rem_dat > valid_height_range[1]
                )
                invalid_mask = np.logical_or(invalid_mask, use_roi == 0)
                use_dat[invalid_mask] = np.nan
                use_pcl = pcl_from_depth(
                    use_dat,
                    intrinsics_matrix[_cam],
                    post_z_shift=floor_distances[_cam] / z_scale,
                    **pcl_kwargs,
                )
                pcls[_cam].append(use_pcl)

        # registration
        registration = DepthVideoPairwiseRegister(**registration_kwargs)

        registration.get_transforms(pcls, progress_bar=False)
        pcls_combined = registration.combine_pcls(pcls, progress_bar=False)

        # farthest point downsample???
        for i, _pcl in enumerate(pcls_combined):
            pcls_combined[i] = _pcl.remove_non_finite_points().voxel_down_sample(voxel_down_sample)

        pcls_combined = pcls_combined[left_pad_size : right_edge_no_pad - left_edge]
        registration.reference_node = registration.reference_node[
            left_pad_size : right_edge_no_pad - left_edge
        ]

        # we need to know when we have another cam shift
        cur_node = registration.reference_node[0]
        next_break = len(registration.reference_node)

        for i, _ref_node in enumerate(registration.reference_node):
            if _ref_node != cur_node:
                next_break = i # when we switch cams
                break
        
        # need to correct breaks across batch transitions...
        # STORE LAST TRANSFORM AND REFERENCE NODE, IF THEY MATCH APPLY TRANSFORM
        # IF THEY DON'T COMPUTE NEW ONE...
        batch_bpoint = False
        if (last_bpoint_transform is not None) and (
            last_reference_node == registration.reference_node[0]
        ):
            # simply apply the last transform if the reference node matches...
            # only apply transformation for the current cam
            for i in range(next_break):
                pcls_combined[i] = pcls_combined[i].transform(last_bpoint_transform)
        elif (last_pcl is not None) and (
            last_reference_node != registration.reference_node[0]
        ):
            # TODO: add to list of bpoint transitions for smoothing...
            # if not recompute and apply to all data...
            # theoretically we could also extrapolate
            use_transformation = np.eye(4)
            c0 = np.array(np.median(last_pcl.points, axis=0))
            c1 = np.array(np.median(pcls_combined[0].points, axis=0))
            df = c0 - c1
            df[2] = 0.0
            use_transformation[:3, 3] = df
            last_bpoint_transform = use_transformation
            # fix only up until the next break point
            # as a general rule 
            for i in range(next_break):
                pcls_combined[i] = pcls_combined[i].transform(use_transformation)
            batch_bpoint = True

        bpoints, frame_groups, bpoint_transforms = correct_breakpoints_extrapolate(
            pcls_combined,
            registration.reference_node,
            # registration_kwargs["max_correspondence_distance"],
            z_shift=True,  # I would set to false if we're not using extrapolate
        )
        
        # bpoints = []
        # bpoint_transforms = []
        if len(bpoints) > 0:
            last_bpoint_transform = bpoint_transforms[-1]
        
        last_pcl = pcls_combined[-1]
        last_reference_node = registration.reference_node[-1]

        _tmp = [np.asarray(pcl.points) for pcl in pcls_combined]
        xyz = np.concatenate(_tmp)
        npoints = xyz.shape[0]
        frame_index = cur_ts.index[
            left_pad_size : right_edge_no_pad - left_edge
        ].to_numpy()  # make sure we can make back to og timestamp

        bpoints = [frame_index[_bpoint] for _bpoint in bpoints]
        if batch_bpoint:
            bpoints = [
                frame_index[0]
            ] + bpoints  # add first index if we have a breakpoint in the first frame

        assert len(frame_index) == len(pcls_combined)

        pcl_idx = [
            np.array([idx] * len(pcl.points), dtype="uint32")
            for idx, pcl in zip(frame_index, pcls_combined)
        ]
        pcl_idx = np.concatenate(pcl_idx)

        if pcl_f["xyz"].shape[0] >= pcl_count + npoints:
            pcl_f["xyz"][pcl_count : pcl_count + npoints] = xyz
            pcl_f["frame_index"][pcl_count : pcl_count + npoints] = pcl_idx
        else:
            pcl_f["xyz"].resize(pcl_count + npoints, axis=0)
            pcl_f["frame_index"].resize(pcl_count + npoints, axis=0)
            pcl_f["xyz"][pcl_count : pcl_count + npoints, :] = xyz
            pcl_f["frame_index"][pcl_count : pcl_count + npoints, :] = pcl_idx

        pcl_f["reference_node"][batch:right_edge_no_pad] = registration.reference_node

        pcl_count += npoints
        all_bpoints += bpoints
        # all_reference_node += registration.reference_node
        del pcls_combined
        del raw_dat
        del roi_dats
        gc.collect()

    pcl_f.create_dataset("bpoints", data=np.array(all_bpoints).astype("uint32"))
    if pcl_f["xyz"].shape[0] > pcl_count:
        pcl_f["xyz"].resize(pcl_count, axis=0)
        pcl_f["frame_index"].resize(pcl_count, axis=0)

    # store other attrs for downstream processing?

    pcl_f.close()
    save_metadata["complete"] = True
    with open(os.path.join(registration_dir, "pcls.toml"), "w") as f:
        toml.dump(save_metadata, f)


def reproject_pcl_to_depth(
    registration_dir: str,
    intrinsics_file: str,
    crop_pad: int = 50,
    stitch_buffer: int = 400,
    batch_size: int = 2000,
    batch_overlap: int = 50,
    interpolation_distance_threshold: float = 1.75,
    interpolation_method: str = "nearest",
    interpolation_fill_value: float = np.nan,
    z_clip: float = 0,
    project_xy: bool = True,
    bpoint_smooth_window: Tuple[int, int] = (50, 50),
    bpoint_retain_window: Tuple[int, int] = (10, 10),
    smooth_kernel: Tuple[float, float, float] = (1.0, 0.75, 0.75),
    smooth_kernel_bpoint: Tuple[float, float, float] = (2.0, 1.5, 1.5),
    visualize_results: bool = True,
):
    # registration_dir = os.path.dirname(registration_file)
    session_name = os.path.normpath(os.path.dirname(registration_dir)).split(os.sep)[-1]
    print(f"Session: {session_name}")
    # session name is two levels down...
    intrinsics = toml.load(intrinsics_file)
    intrinsics_matrix, distortion_coeffs = format_intrinsics(intrinsics)

    cameras = list(intrinsics_matrix.keys())
    metadata = toml.load(os.path.join(registration_dir, "..", "metadata.toml"))
    fps = np.round(metadata["camera_metadata"][cameras[0]]["AcquisitionFrameRate"])
    z_scale = (
        1 / metadata["camera_metadata"][cameras[0]]["Scan3dCoordinateScale"]
    )  # assume this is common for now

    registration_file = os.path.join(registration_dir, f"{session_name}.hdf5")
    pcl_f = h5py.File(os.path.join(registration_dir, "pcls.hdf5"), "r")
    pcl_metadata = toml.load(os.path.join(registration_dir, "pcls.toml"))
    all_bpoints = pcl_f["bpoints"][()]
    pcl_coord_index = pcl_f["frame_index"][()]
    pcl_frame_index = np.unique(pcl_coord_index)
    npcls = len(pcl_frame_index)

    width, height = (
        metadata["camera_metadata"][cameras[0]]["Width"],
        metadata["camera_metadata"][cameras[0]]["Height"],
    )
    stitch_size = (
        width + 2 * stitch_buffer,
        height + 2 * stitch_buffer,
    )

    depth_f = h5py.File(registration_file, "w")
    depth_f.create_dataset(
        "frames",
        (npcls, stitch_size[1], stitch_size[0]),
        "uint16",
        compression="lzf",
    )

    depth_f.create_dataset(
        "frame_id",
        data=pcl_frame_index,
    )

    batches = range(0, len(pcl_frame_index), batch_size)
    max_proj = np.zeros((stitch_size[1], stitch_size[0]), dtype="uint16")

    for batch in tqdm(batches, desc="Reproject to depth images"):
        left_edge = batch
        right_edge = min(batch + batch_size, len(pcl_frame_index))

        # loop through indices to build a boolean index into pcl store
        read_pcls = pcl_frame_index[left_edge:right_edge]
        new_im = np.zeros((len(read_pcls), stitch_size[1], stitch_size[0]), dtype="uint16")
        reference_node = pcl_f["reference_node"][left_edge:right_edge]
        reference_node = [_.decode() for _ in reference_node]

        mask_array = (pcl_coord_index >= min(read_pcls)) & (pcl_coord_index <= max(read_pcls))
        adj = np.min(np.flatnonzero(mask_array))
        use_coord_index = pcl_coord_index[mask_array]

        floor_distances = pcl_metadata["floor_distances"]
        floor_distances = {k: float(v) for k, v in floor_distances.items()}

        for i, _pcl_idx in enumerate(read_pcls):
            pcl_read_idx = np.flatnonzero(_pcl_idx == use_coord_index) + adj
            xyz = pcl_f["xyz"][slice(pcl_read_idx[0], pcl_read_idx[-1])]
            _pcl = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
            if len(_pcl.points) < 10:
                _new_im = np.zeros((stitch_size[1], stitch_size[0]), dtype="uint16")
            else:
                _new_im = depth_from_pcl_interpolate(
                    _pcl,
                    intrinsics_matrix[reference_node[i]],
                    z_adjust=floor_distances[reference_node[i]],
                    post_z_shift=floor_distances[reference_node[i]],
                    width=stitch_size[0],
                    height=stitch_size[1],
                    distance_threshold=interpolation_distance_threshold,
                    fill_value=interpolation_fill_value,
                    interpolation_method=interpolation_method,
                    z_scale=z_scale,
                    z_clip=z_clip,
                    project_xy=project_xy,
                )
                _new_im[np.logical_or(np.isnan(_new_im), _new_im < 0)] = 0
                _new_im = _new_im.astype("uint16")
                _new_im[_new_im > 30000] = 0
                _new_im = np.nan_to_num(_new_im, 0)
            new_im[i] = _new_im
        # then map the matrix onto the h5 file...
        depth_f["frames"][
            slice(left_edge, right_edge), slice(0, stitch_size[1]), slice(0, stitch_size[0])
        ] = new_im
        # keep track of max projection for cropping
        new_max_proj = np.max(new_im, axis=0)
        max_proj = np.maximum(new_max_proj, max_proj)

    # no longer need pcls
    pcl_f.close()

    # now crop the data...
    yproj = np.max(max_proj, axis=1)
    xproj = np.max(max_proj, axis=0)

    ynonzero = np.flatnonzero(yproj > 0)
    xnonzero = np.flatnonzero(xproj > 0)
    ycrop = next_even_number(max(min(ynonzero) - crop_pad, 0)), prev_even_number(
        min(max(ynonzero) + crop_pad, max_proj.shape[0])
    )
    xcrop = next_even_number(max(min(xnonzero) - crop_pad, 0)), prev_even_number(
        min(max(xnonzero) + crop_pad, max_proj.shape[1])
    )

    depth_f.create_dataset(
        "frames_crop",
        (len(pcl_frame_index), ycrop[1] - ycrop[0], xcrop[1] - xcrop[0]),
        "uint16",
        compression="lzf",
    )

    original_size = depth_f["frames"].shape[1:]
    crop_size = depth_f["frames_crop"].shape[1:]

    print(f"Original size {original_size}, crop size {crop_size}")

    for batch in tqdm(batches, desc="Cropping frames"):
        left_edge = max(batch - batch_overlap, 0)
        left_noverlap = batch - left_edge
        right_edge = min(batch + batch_size, len(pcl_frame_index))
        cropped_frames = depth_f["frames"][
            left_edge:right_edge, ycrop[0] : ycrop[1], xcrop[0] : xcrop[1]
        ]
        cropped_frames = ndimage.gaussian_filter(cropped_frames, smooth_kernel)
        depth_f["frames_crop"][batch:right_edge] = cropped_frames[left_noverlap:]

    bpoint_frame_index = [np.flatnonzero(pcl_frame_index == _bpoint)[0] for _bpoint in all_bpoints]

    print(f"Found breakpoints at {bpoint_frame_index}")

    for _bpoint in tqdm(bpoint_frame_index, desc="Smoothing over breakpoints"):
        adj_bpoint_smooth_window = (
            min(bpoint_smooth_window[0], _bpoint),
            min(bpoint_smooth_window[1], len(depth_f["frames_crop"]) - _bpoint),
        )
        _tmp = depth_f["frames_crop"][
            _bpoint - adj_bpoint_smooth_window[0] : _bpoint + adj_bpoint_smooth_window[1]
        ]
        _tmp = ndimage.gaussian_filter(_tmp, smooth_kernel_bpoint)
        depth_f["frames_crop"][
            max(_bpoint - bpoint_retain_window[0], 0) : _bpoint + bpoint_retain_window[1]
        ] = _tmp[
            bpoint_smooth_window[0]
            - bpoint_retain_window[0] : bpoint_smooth_window[0]
            + bpoint_retain_window[1]
        ]

    crop_size = depth_f["frames_crop"].shape[1:]

    if visualize_results:
        writer = MP4WriterPreview(
            os.path.join(registration_dir, f"{session_name}.mp4"),
            frame_size=(crop_size[1], crop_size[0]),
            fps=fps,
            cmap="turbo",
        )
        writer.open()

        for batch in tqdm(batches, desc="Writing preview video"):
            right_edge = min(batch + batch_size, len(pcl_frame_index))
            _tmp = depth_f["frames_crop"][batch:right_edge]
            writer.write_frames(
                _tmp,
                progress_bar=False,
                frames_idx=list(range(batch, right_edge)),
                vmin=0,
                vmax=600,
            )

        writer.close()


def next_even_number(x):
    return (np.ceil(x / 2) * 2).astype("int")


def prev_even_number(x):
    return (np.floor(x / 2) * 2).astype("int")
