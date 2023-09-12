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
from markovids.pcl.io import (
    pcl_from_depth,
    depth_from_pcl_interpolate,
    trim_outliers,
    pcl_to_pxl_coords,
)
from markovids.pcl.registration import (
    DepthVideoPairwiseRegister,
    correct_breakpoints,
    correct_breakpoints_extrapolate,
)
from collections import defaultdict
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
        (len(use_ts) * 8e3, 3),
        "float64",
        maxshape=(None, 3),
        compression="lzf",
    )
    pcl_f.create_dataset(
        "frame_index",
        (len(use_ts) * 8e3,),
        "uint32",
        maxshape=(None,),
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

    pcl_count = 0
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

        _tmp = [np.asarray(pcl.points) for pcl in pcls_combined]
        xyz = np.concatenate(_tmp)
        npoints = xyz.shape[0]
        frame_index = cur_ts.index[
            left_pad_size : right_edge_no_pad - left_edge
        ].to_numpy()  # make sure we can make back to og timestamp

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
            pcl_f["frame_index"][pcl_count : pcl_count + npoints] = pcl_idx

        pcl_f["reference_node"][batch:right_edge_no_pad] = registration.reference_node

        pcl_count += npoints
        del pcls_combined
        del raw_dat
        del roi_dats
        gc.collect()

    # pcl_f.create_dataset("bpoints", data=np.array(all_bpoints).astype("uint32"))
    if pcl_f["xyz"].shape[0] > pcl_count:
        pcl_f["xyz"].resize(pcl_count, axis=0)
        pcl_f["frame_index"].resize(pcl_count, axis=0)

    pcl_f.close()
    save_metadata["complete"] = True
    with open(os.path.join(registration_dir, "pcls.toml"), "w") as f:
        toml.dump(save_metadata, f)

    # FIX BREAKPOINTS DOWNSTREAM, AVERAGE ACROSS CAMERA COMBINATIONS


def fix_breakpoints(
    pcl_file: str,
):
    pcl_f = h5py.File(pcl_file, "r+")
    # now we need to determine points where we switched references and correct...
    pcl_coord_idx = pcl_f["frame_index"][()]
    pcl_frame_idx = np.unique(pcl_coord_idx)
    npcls = len(pcl_frame_idx)

    target = pcl_f["reference_node"][0].decode()
    transforms = defaultdict(list)
    for i in tqdm(range(len(pcl_f["reference_node"])), desc="Finding breakpoints..."):
        source = pcl_f["reference_node"][i].decode()
        if source != target:
            transforms[(target, source)].append(pcl_frame_idx[i])
        target = source

    all_bpoints = np.concatenate(list(transforms.values()))
    all_bpoints = all_bpoints[all_bpoints.argsort()]

    # store for later, may need to nan out
    pcl_f.create_dataset("bpoints", data=all_bpoints, compression="lzf")

    transform_list = []
    for (target, source), pcl_idxs in tqdm(transforms.items(), desc="Getting transforms"):
        diffs = []
        frame_group = []

        for _idx in pcl_idxs:
            # target is - 1
            # source is 0
            next_frame = max(pcl_frame_idx)
            try:
                next_frame = np.min(all_bpoints[all_bpoints > _idx])
            except:
                next_frame = max(pcl_frame_idx)

            target_read_idx = np.flatnonzero(
                pcl_coord_idx == np.max(pcl_frame_idx[pcl_frame_idx < _idx])
            )
            source_read_idx = np.flatnonzero(pcl_coord_idx == _idx)

            # use neighboring frames to we don't mess up???
            source_xyz = pcl_f["xyz"][slice(source_read_idx[0], source_read_idx[-1])]
            source_xyz = source_xyz[~np.isnan(source_xyz).any(axis=1)]  # remove nans
            source_xyz = trim_outliers(source_xyz)

            target_xyz = pcl_f["xyz"][slice(target_read_idx[0], target_read_idx[-1])]
            target_xyz = target_xyz[~np.isnan(target_xyz).any(axis=1)]  # remove nans
            target_xyz = trim_outliers(target_xyz)

            diffs.append(np.median(target_xyz, axis=0) - np.median(source_xyz, axis=0))
            # inclusive left hand range
            # exclusive right hand (cuts into next transition)
            matches = pcl_frame_idx[
                np.logical_and(pcl_frame_idx >= _idx, pcl_frame_idx < next_frame)
            ]
            frame_group.append(matches)

        # alternatively we can get a different transform for each one
        # except we aggressively trim outliers
        diffs = np.array(diffs)
        use_transform = np.eye(4)
        use_transform[:3, 3] = np.median(diffs, axis=0)
        use_transform[2, 3] = 0

        for _group in frame_group:
            transform_list.append(
                {
                    "start": _group[0],
                    "stop": _group[-1],
                    "transform": use_transform,
                    "pcl_idxs": _group,
                }
            )

    idxsort = np.array([_["start"] for _ in transform_list]).argsort()
    sorted_transform_list = [transform_list[i] for i in idxsort]
    odometry = np.eye(4)

    for i in tqdm(range(len(sorted_transform_list)), desc="Correcting breakpoints"):
        odometry = odometry @ sorted_transform_list[i]["transform"]
        # align everything
        read_pcls = sorted_transform_list[i]["pcl_idxs"]
        mask_array = (pcl_coord_idx >= min(read_pcls)) & (pcl_coord_idx <= max(read_pcls))
        adj = np.min(np.flatnonzero(mask_array))
        use_coord_index = pcl_coord_idx[mask_array]
        for _pcl_idx in read_pcls:
            pcl_read_idx = np.flatnonzero(_pcl_idx == use_coord_index) + adj
            xyz = pcl_f["xyz"][slice(pcl_read_idx[0], pcl_read_idx[-1] + 1)]
            _pcl = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
            _pcl = _pcl.transform(odometry)
            pcl_f["xyz"][slice(pcl_read_idx[0], pcl_read_idx[-1] + 1)] = np.asarray(_pcl.points)


def reproject_pcl_to_depth(
    registration_file: str,
    intrinsics_file: str,
    stitch_buffer: int = 10,
    batch_size: int = 2000,
    batch_overlap: int = 50,
    interpolation_distance_threshold: float = 1.75,
    interpolation_method: str = "nearest",
    interpolation_fill_value: float = np.nan,
    z_clip: float = 0,
    project_xy: bool = True,
    smooth_kernel: Tuple[float, float, float] = (1.0, 0.75, 0.75),
    visualize_results: bool = True,
    centroid_outlier_trim_nsigma: int = 5,
):
    save_metadata = locals()
    save_metadata["complete"] = False

    # session_name = os.path.normpath(os.path.dirname(registration_dir)).split(os.sep)[-1]
    # print(f"Session: {session_name}")
    # session name is two levels down...
    registration_dir = os.path.dirname(registration_file)
    registration_fname, ext = os.path.splitext(registration_file)
    intrinsics = toml.load(intrinsics_file)
    intrinsics_matrix, distortion_coeffs = format_intrinsics(intrinsics)
    cameras = list(intrinsics_matrix.keys())
    metadata = toml.load(os.path.join(registration_dir, "..", "metadata.toml"))
    fps = np.round(metadata["camera_metadata"][cameras[0]]["AcquisitionFrameRate"])
    z_scale = (
        1 / metadata["camera_metadata"][cameras[0]]["Scan3dCoordinateScale"]
    )  # assume this is common for now

    pcl_f = h5py.File(os.path.join(registration_dir, "pcls.hdf5"), "r")
    pcl_metadata = toml.load(os.path.join(registration_dir, "pcls.toml"))
    pcl_coord_index = pcl_f["frame_index"][()]
    pcl_frame_index = np.unique(pcl_coord_index)
    npcls = len(pcl_frame_index)

    floor_distances = pcl_metadata["floor_distances"]
    floor_distances = {k: float(v) for k, v in floor_distances.items()}

    max_batch_size = int(1e7)
    max_pts = np.nanmax(pcl_f["xyz"][:1], axis=0)
    min_pts = np.nanmin(pcl_f["xyz"][:1], axis=0)
    for batch in tqdm(range(0, len(pcl_f["xyz"]), max_batch_size), desc="Getting max"):
        pts = pcl_f["xyz"][batch : batch + max_batch_size]
        u, v, z = pcl_to_pxl_coords(
            pts,
            intrinsics_matrix[cameras[0]],
            z_scale=z_scale,
            post_z_shift=floor_distances[cameras[0]],
        )
        new_pts = np.vstack([u, v, z]).T
        max_pts = np.maximum(max_pts, np.nanmax(new_pts, axis=0))
        min_pts = np.minimum(min_pts, np.nanmin(new_pts, axis=0))

    buffer = np.floor(np.minimum(min_pts[:2], np.array([0, 0]))).astype("int") * -1
    buffer += stitch_buffer
    stitch_size = (next_even_number(max_pts[0] + buffer[0]),
                   next_even_number(max_pts[1] + buffer[1]))

    print(f"Stitch size {stitch_size}")

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

        for i, _pcl_idx in enumerate(read_pcls):
            pcl_read_idx = np.flatnonzero(_pcl_idx == use_coord_index) + adj
            xyz = pcl_f["xyz"][slice(pcl_read_idx[0], pcl_read_idx[-1])]
            xyz = xyz[~np.isnan(xyz).any(axis=1)]  # remove nans
            xyz = trim_outliers(xyz, thresh=centroid_outlier_trim_nsigma)

            _pcl = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
            # in the merged point cloud it doesn't make sense to use camera-specific
            # parameters, default to using parameters from the first cam in the intrinsics
            # file
            if len(_pcl.points) < 10:
                _new_im = np.zeros((stitch_size[1], stitch_size[0]), dtype="uint16")
            else:
                _new_im = depth_from_pcl_interpolate(
                    _pcl,
                    intrinsics_matrix[
                        cameras[0]
                    ],  # use first camera across all frames for consistency
                    z_adjust=floor_distances[cameras[0]],
                    post_z_shift=floor_distances[cameras[0]],
                    width=stitch_size[0],
                    height=stitch_size[1],
                    distance_threshold=interpolation_distance_threshold,
                    fill_value=interpolation_fill_value,
                    interpolation_method=interpolation_method,
                    z_scale=z_scale,
                    z_clip=z_clip,
                    buffer=buffer,
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

    # no longer need pcls
    pcl_f.close()

    # now crop the data...
    # we can easily skip this by simply projecting min/max xy
    # print(f"Original size {original_size}, crop size {crop_size}")
    if visualize_results:
        writer = MP4WriterPreview(
            f"{registration_fname}.mp4",
            frame_size=stitch_size,
            fps=fps,
            cmap="turbo",
        )
        writer.open()
    else:
        writer = None

    for batch in tqdm(batches, desc="Smoothing frames"):
        left_edge = max(batch - batch_overlap, 0)
        left_noverlap = batch - left_edge
        right_edge = min(batch + batch_size, len(pcl_frame_index))
        smooth_frames = ndimage.gaussian_filter(
            depth_f["frames"][left_edge:right_edge], smooth_kernel
        )
        depth_f["frames"][batch:right_edge] = smooth_frames[left_noverlap:]

        if writer is not None:
            writer.write_frames(
                smooth_frames[left_noverlap:],
                progress_bar=False,
                frames_idx=list(range(batch, right_edge)),
                vmin=0,
                vmax=600,
            )

    if writer is not None:
        writer.close()

    save_metadata["complete"] = True
    with open(f"{registration_fname}.toml", "w") as f:
        toml.dump(save_metadata, f)


def next_even_number(x):
    return (np.ceil(x / 2) * 2).astype("int")


def prev_even_number(x):
    return (np.floor(x / 2) * 2).astype("int")
