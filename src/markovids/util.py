from typing import Tuple, Optional
from markovids.vid.io import (
    format_intrinsics,
    get_bground,
    read_timestamps_multicam,
    read_frames_multicam,
    MP4WriterPreview,
)
from markovids.depth.plane import get_floor
from markovids.depth.io import load_segmentation_masks
from markovids.depth.track import clean_roi
from markovids.depth.moments import im_moment_features
from markovids.pcl.io import (
    pcl_from_depth,
    depth_from_pcl_interpolate,
    trim_outliers,
    pcl_to_pxl_coords,
    project_world_coordinates,
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
import joblib
import pandas as pd
import warnings

o3d.utility.set_verbosity_level(o3d.utility.Error)


pcl_kwargs_default = {"project_xy": True}
registration_kwargs_default = {
    "max_correspondence_distance": 1.0,
    "fitness_threshold": 0.25,
    "reference_future_len": 100,
    "reference_history_len": 50,
    "cleanup_nbs": 5,
    "cleanup_nbs_combined": 15,
    "cleanup_sigma": 2.0,
    "cleanup_sigma_combined": 2.0,
}


def convert_depth_to_pcl_and_register(
    data_dir: str,
    intrinsics_file: str,
    background_spacing: int = 500,
    batch_overlap: int = 150,
    batch_size: int = 2000,
    burn_frames: int = 500,
    depth_frame_bilateral_filter: Tuple[float, float] = (
        3.0,
        3.0,
    ),  # unclear if we want this yet, slow...
    floor_range: Tuple[float, float] = (1300.0, 1600.0),
    pcl_kwargs: dict = {},
    pcl_floor_delta: bool = True,
    registration_algorithm: str = "pairwise",
    registration_dir: str = "_registration",
    registration_kwargs: dict = {},
    segmentation_dir: str = "_segmentation_tau-5",
    tail_filter_pixels: Optional[
        int
    ] = 21,  # scale of morphological opening filter to remove tail (None to skip)
    test_run_batches: int = -1,
    timestamp_merge_tolerance=0.003,  # in seconds
    valid_height_range: Tuple[float, float] = (10.0, 800.0),
    voxel_down_sample: float = 1.0,
):
    pcl_kwargs = pcl_kwargs_default | pcl_kwargs
    registration_kwargs = registration_kwargs_default | registration_kwargs
    registration_dir = os.path.join(data_dir, registration_dir)
    save_params = locals()

    if (tail_filter_pixels is not None) and (tail_filter_pixels > 0):
        tail_filter_strels = {
            cv2.MORPH_OPEN: cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (tail_filter_pixels, tail_filter_pixels)
            )
        }
        clean_roi_kwargs = {
            "pre_strels": tail_filter_strels,
            "post_strels": {},
            "fill_holes": False,
            "use_cc": False,
        }
    else:
        clean_roi_kwargs = None

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
        _cam: get_floor(bgrounds[_cam], floor_range=floor_range, dilations=10)
        for _cam in cameras
    }

    floor_distances = {}
    for _cam in tqdm(cameras, desc="Getting floor distances"):
        roi_floor = cv2.erode(
            rois[_cam],
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=5,
        )
        floor_distances[_cam] = np.nanmedian(bgrounds[_cam][roi_floor > 0].ravel())

    _, merged_ts = read_timestamps_multicam(
        ts_paths, merge_tolerance=timestamp_merge_tolerance
    )
    use_ts = merged_ts.dropna().iloc[burn_frames:]
    nframes = len(use_ts)

    str_dt = h5py.special_dtype(vlen=str)
    pcl_f = h5py.File(os.path.join(registration_dir, "pcls.hdf5"), "w")
    pcl_f.create_group("rois")
    pcl_f.create_group("floor_distances")

    for k, v in rois.items():
        pcl_f[f"rois/{k}"] = v
    for k, v in floor_distances.items():
        pcl_f[f"floor_distances/{k}"] = v

    # TODO: here insert room for nframes x 4 x 4 matrices for storing all transformations
    # this can be used for volume integration downstream without having to stash point clouds...
    #
    for _cam in cameras:
        pcl_f.create_dataset(
            f"transformations/{_cam}", (len(use_ts), 4, 4), "float64", compression="lzf"
        )

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
    pcl_f.create_dataset(
        "reference_frame_index", (len(use_ts),), "uint32", compression="lzf"
    )

    for k, v in registration_kwargs.items():
        if v is not None:
            pcl_f["xyz"].attrs[k] = v
        else:
            pcl_f["xyz"].attrs[k] = "None"
    pcl_f["xyz"].attrs["segmentation_dir"] = segmentation_dir

    save_metadata = copy.deepcopy(save_params)
    save_metadata["z_scale"] = z_scale
    save_metadata["floor_distances"] = floor_distances
    save_metadata["cameras"] = cameras
    save_metadata["timestamp_camera_primary"] = cameras[0]
    save_metadata["dat_paths"] = dat_paths
    save_metadata["ts_paths"] = ts_paths

    if test_run_batches <= 0:
        frame_batches = range(0, nframes, batch_size)
    else:
        frame_batches = range(0, batch_size * test_run_batches, batch_size)

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
        if clean_roi_kwargs is not None:
            print("Cleaning ROI")
            roi_dats = {
                _cam: clean_roi(_dat, progress_bar=False, **clean_roi_kwargs)
                for _cam, _dat in roi_dats.items()
            }

        # convert everything to point clouds
        pcls = {_cam: [] for _cam in cameras}
        for _cam in tqdm(cameras, desc="Converting to point clouds", disable=True):
            for _frame in range(len(cur_ts)):
                # bilateral filter depth frames??
                use_dat = raw_dat[_cam][_frame].copy().astype("float32")
                if depth_frame_bilateral_filter is not None:
                    use_dat = cv2.bilateralFilter(
                        use_dat, -1, *depth_frame_bilateral_filter
                    )
                use_roi = roi_dats[_cam][_frame]
                bground_rem_dat = floor_distances[_cam] - use_dat
                invalid_mask = np.logical_or(
                    bground_rem_dat < valid_height_range[0],
                    bground_rem_dat > valid_height_range[1],
                )
                invalid_mask = np.logical_or(invalid_mask, use_roi == 0)
                use_dat[invalid_mask] = np.nan
                use_pcl = pcl_from_depth(
                    use_dat,
                    intrinsics_matrix[_cam],
                    post_z_shift=floor_distances[_cam] / z_scale
                    if pcl_floor_delta
                    else None,
                    **pcl_kwargs,
                )
                use_pcl = use_pcl.remove_non_finite_points()
                pcls[_cam].append(use_pcl)

        # registration
        registration = DepthVideoPairwiseRegister(**registration_kwargs)
        if registration_algorithm == "multiway":
            print("Using multiway registration...")
            registration.get_transforms_multiway(pcls, progress_bar=False)
        elif registration_algorithm == "pairwise":
            print("Using pairwise registration...")
            registration.get_transforms_pairwise(pcls, progress_bar=False)
        else:
            raise RuntimeError(
                f"Did not understand registration algorithm {registration_algorithm}"
            )

        pcls_combined = registration.combine_pcls(pcls, progress_bar=False)

        # farthest point downsample???
        for i, _pcl in enumerate(pcls_combined):
            pcls_combined[i] = _pcl.remove_non_finite_points().voxel_down_sample(
                voxel_down_sample
            )

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

        pcl_f["reference_frame_index"][batch:right_edge_no_pad] = frame_index
        pcl_f["reference_node"][batch:right_edge_no_pad] = registration.reference_node
        for _cam in cameras:
            pcl_f[f"transformations/{_cam}"][
                batch:right_edge_no_pad
            ] = registration.transforms[_cam][
                left_pad_size : right_edge_no_pad - left_edge
            ]

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


def fix_breakpoints_single(
    pcl_file: str,
    transform_aggregate: bool = True,
    enforce_symmetry: bool = True,
):
    pcl_f = h5py.File(pcl_file, "r+")

    pcl_metadata = f"{os.path.splitext(pcl_file)[0]}.toml"
    pcl_metadata = toml.load(pcl_metadata)

    # now we need to determine points where we switched references and correct...
    pcl_coord_idx = pcl_f["frame_index"][()]
    # pcl_frame_idx = np.unique(pcl_coord_idx)
    pcl_frame_idx = pcl_f["reference_frame_index"][()]
    npcls = len(pcl_frame_idx)

    target = pcl_f["reference_node"][0].decode()
    transforms = defaultdict(list)
    for i in tqdm(range(len(pcl_f["reference_node"])), desc="Finding breakpoints..."):
        source = pcl_f["reference_node"][i].decode()
        if len(source) == 0:
            break
        if source != target:
            transforms[(target, source)].append(pcl_frame_idx[i])
        target = source

    all_bpoints = np.concatenate(list(transforms.values()))
    all_bpoints = all_bpoints[all_bpoints.argsort()]

    _, merged_ts = read_timestamps_multicam(
        pcl_metadata["ts_paths"],
        merge_tolerance=pcl_metadata["timestamp_merge_tolerance"],
    )
    use_ts = merged_ts.dropna().iloc[pcl_metadata["burn_frames"] :]

    intrinsics = toml.load(pcl_metadata["intrinsics_file"])
    intrinsics_matrix, distortion_coeffs = format_intrinsics(intrinsics)
    load_dct = {}
    for camera in pcl_metadata["cameras"]:
        load_dct[camera] = {}
        load_dct[camera]["intrinsic_matrix"] = intrinsics_matrix[camera]
        load_dct[camera]["distortion_coeffs"] = distortion_coeffs[camera]

    load_paths = {v: k for k, v in pcl_metadata["dat_paths"].items()}

    z_scale = float(pcl_metadata["z_scale"])
    valid_height_range = (
        float(pcl_metadata["valid_height_range"][0]),
        float(pcl_metadata["valid_height_range"][1]),
    )

    # store for later, may need to nan out
    pcl_f.create_dataset("bpoints", data=all_bpoints, compression="lzf")

    transform_list = []
    for (target, source), pcl_idxs in tqdm(
        transforms.items(), desc="Estimating transforms"
    ):
        # load in CURRENT FRAME AT THE CAMERA TRANSITION POINT
        # source is where we ended up, target is where we want to go...
        diffs = []
        frame_group = []
        for _idx in pcl_idxs:
            next_frame = max(pcl_frame_idx)
            try:
                next_frame = np.min(all_bpoints[all_bpoints > _idx])
            except:
                next_frame = max(pcl_frame_idx)

            dct = use_ts.loc[_idx][pcl_metadata["cameras"]].astype("int").to_dict()
            # dct = {k: [v] for k, v in dct.items()}
            masks = {
                _cam: load_segmentation_masks(
                    _path, dct[_cam], pcl_metadata["segmentation_dir"]
                )
                for _cam, _path in pcl_metadata["dat_paths"].items()
            }
            frames = read_frames_multicam(load_paths, dct, load_dct, progress_bar=False)
            # with frame, convert to pcls, run it down...

            # inclusive left hand range
            # exclusive right hand (cuts into next transition)
            matches = pcl_frame_idx[
                np.logical_and(pcl_frame_idx >= _idx, pcl_frame_idx < next_frame)
            ]
            frame_group.append(matches)

            source_floor_distance = float(pcl_metadata["floor_distances"][source])
            source_bground_rem = source_floor_distance - frames[source]
            source_invalid_mask = np.logical_or(
                source_bground_rem < valid_height_range[0],
                source_bground_rem > valid_height_range[1],
            )
            source_invalid_mask = np.logical_or(source_invalid_mask, masks[source] == 0)
            source_proj_data = frames[source].copy().astype("float")
            source_proj_data[source_invalid_mask] = np.nan

            source_pcl = pcl_from_depth(
                source_proj_data[0],
                intrinsics_matrix[source],
                post_z_shift=source_floor_distance / z_scale,
                **pcl_metadata["pcl_kwargs"],
            )

            target_floor_distance = float(pcl_metadata["floor_distances"][target])
            target_bground_rem = target_floor_distance - frames[target]
            target_invalid_mask = np.logical_or(
                target_bground_rem < valid_height_range[0],
                target_bground_rem > valid_height_range[1],
            )
            target_invalid_mask = np.logical_or(target_invalid_mask, masks[target] == 0)
            target_proj_data = frames[target].copy().astype("float")
            target_proj_data[target_invalid_mask] = np.nan

            target_pcl = pcl_from_depth(
                target_proj_data[0],
                intrinsics_matrix[target],
                post_z_shift=target_floor_distance / z_scale,
                **pcl_metadata["pcl_kwargs"],
            )

            # TODO: bail here if we don't have a large enough pcl

            source_xyz = trim_outliers(np.asarray(source_pcl.points))
            target_xyz = trim_outliers(np.asarray(target_pcl.points))

            diffs.append(
                np.nanmedian(target_xyz, axis=0) - np.nanmedian(source_xyz, axis=0)
            )
            

        # alternatively we can get a different transform for each one
        # except we aggressively trim outliers...
        diffs = np.array(diffs)

        if transform_aggregate:
            use_transform = np.eye(4)
            use_transform[:3, 3] = np.nanmedian(diffs, axis=0)
            use_transform[2, 3] = 0
            for _group in frame_group:
                transform_list.append(
                    {
                        "start": _group[0],
                        "stop": _group[-1],
                        "transform": use_transform,
                        "pcl_idxs": _group,
                        "pair": (target, source),
                    }
                )
        else:
            for _group, _diff in zip(frame_group, diffs):
                use_transform = np.eye(4)
                use_transform[:3, 3] = _diff
                use_transform[2, 3] = 0
                transform_list.append(
                    {
                        "start": _group[0],
                        "stop": _group[-1],
                        "transform": use_transform,
                        "pcl_idxs": _group,
                        "pair": (target, source),
                    }
                )

    joblib.dump(
        transform_list,
        os.path.join(os.path.dirname(pcl_file), "bpoint_transform_list_pre.p"),
    )

    # enforce symmetry???
    if transform_aggregate and enforce_symmetry:
        print("Enforcing symmetry in breakpoint fixes...")
        uniq_pairs = list(set([_["pair"] for _ in transform_list]))
        uniq_transform = {}

        for _pair in uniq_pairs:
            try:
                transform = [
                    _["transform"] for _ in transform_list if _["pair"] == _pair
                ][0]
                uniq_transform[_pair] = transform
            except Exception:
                pass

        new_transform = {}
        for (target, source), v in uniq_transform.items():
            try:
                reflection = np.linalg.inv(uniq_transform[(source, target)])
                new_transform[(target, source)] = np.nanmean(
                    np.stack([uniq_transform[(target, source)], reflection]), axis=0
                )
            except KeyError:
                new_transform[(target, source)] = uniq_transform[(target, source)]

        for i in range(len(transform_list)):
            transform_list[i]["transform"] = new_transform[transform_list[i]["pair"]]

    joblib.dump(
        transform_list,
        os.path.join(os.path.dirname(pcl_file), "bpoint_transform_list.p"),
    )
    idxsort = np.array([_["start"] for _ in transform_list]).argsort()
    sorted_transform_list = [transform_list[i] for i in idxsort]
    odometry = np.eye(4)

    for i in tqdm(range(len(sorted_transform_list)), desc="Correcting breakpoints"):
        odometry = odometry @ sorted_transform_list[i]["transform"]
        # align everything
        read_pcls = sorted_transform_list[i]["pcl_idxs"]
        mask_array = (pcl_coord_idx >= min(read_pcls)) & (
            pcl_coord_idx <= max(read_pcls)
        )
        adj = np.min(np.flatnonzero(mask_array))
        use_coord_index = pcl_coord_idx[mask_array]
        for _pcl_idx in read_pcls:
            pcl_read_idx = np.flatnonzero(_pcl_idx == use_coord_index) + adj
            xyz = pcl_f["xyz"][slice(pcl_read_idx[0], pcl_read_idx[-1] + 1)]
            _pcl = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
            _pcl = _pcl.transform(odometry)
            pcl_f["xyz"][slice(pcl_read_idx[0], pcl_read_idx[-1] + 1)] = np.asarray(
                _pcl.points
            )


def fix_breakpoints_combined(
    pcl_file: str,
    transform_aggregate: bool = True,
    enforce_symmetry: bool = True,
    min_npoints: int = 1000,
):
    pcl_f = h5py.File(pcl_file, "r+")
    # now we need to determine points where we switched references and correct...
    pcl_coord_idx = pcl_f["frame_index"][()]
    # pcl_frame_idx = np.unique(pcl_coord_idx)
    pcl_frame_idx = pcl_f["reference_frame_index"][()]
    npcls = len(pcl_frame_idx)

    target = pcl_f["reference_node"][0].decode()
    transforms = defaultdict(list)
    for i in tqdm(range(len(pcl_f["reference_node"])), desc="Finding breakpoints..."):
        source = pcl_f["reference_node"][i].decode()
        if (source != target) and (len(source) > 0):
            try:
                transforms[(target, source)].append(pcl_frame_idx[i])
            except IndexError:
                warnings.warn(f"Indexing issue finding breakpoints at index number {i}")
                break
            # DON'T FORGET TO REMOVE AFTER DEBUGGING
        target = source

    print(transforms)
    all_bpoints = np.concatenate(list(transforms.values()))
    all_bpoints = all_bpoints[all_bpoints.argsort()]

    # store for later, may need to nan out
    try:
        pcl_f.create_dataset("bpoints", data=all_bpoints, compression="lzf")
    except ValueError:
        del pcl_f["bpoints"]
        pcl_f.create_dataset("bpoints", data=all_bpoints, compression="lzf")

    # TODO: enable walking paths between cameras if we don't have a direct path
    # e.g. a-->b-->c, most useful for short videos...
    # G = nx.complete_graph(3)
    # nx.all_simple_paths(G,0,1)
    transform_list = []
    for (target, source), pcl_idxs in tqdm(
        transforms.items(), desc="Getting transforms"
    ):
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

            # inclusive left hand range
            # exclusive right hand (cuts into next transition)
            matches = pcl_frame_idx[
                np.logical_and(pcl_frame_idx >= _idx, pcl_frame_idx < next_frame)
            ]
            frame_group.append(matches)

            try:
                target_read_idx = np.flatnonzero(
                    pcl_coord_idx == np.max(pcl_frame_idx[pcl_frame_idx < _idx])
                )
            except ValueError as e:
                warnings.warn(
                    f"Unable to compute transform between {source} and {target} at {_idx}"
                )
                continue

            source_read_idx = np.flatnonzero(pcl_coord_idx == _idx)

            if (len(source_read_idx) == 0) or (len(target_read_idx) == 0):
                warnings.warn(
                    f"Unable to compute transform between {source} and {target} at {_idx}"
                )
                continue

            # use neighboring frames to we don't mess up???
            source_xyz = pcl_f["xyz"][slice(source_read_idx[0], source_read_idx[-1])]
            source_xyz = source_xyz[~np.isnan(source_xyz).any(axis=1)]  # remove nans
            source_xyz = trim_outliers(source_xyz)

            target_xyz = pcl_f["xyz"][slice(target_read_idx[0], target_read_idx[-1])]
            target_xyz = target_xyz[~np.isnan(target_xyz).any(axis=1)]  # remove nans
            target_xyz = trim_outliers(target_xyz)

            if (source_xyz.shape[0] < min_npoints) or (
                target_xyz.shape[0] < min_npoints
            ):
                warnings.warn(
                    f"Unable to compute transform between {source} and {target} at {_idx}"
                )
                continue

            # be careful since nans will propagate...
            diffs.append(
                np.nanmedian(target_xyz, axis=0) - np.nanmedian(source_xyz, axis=0)
            )

        diffs = np.array(diffs)
        # pack into homogeneous coordinates
        if transform_aggregate:
            use_transform = np.eye(4)
            use_transform[:3, 3] = np.nanmedian(diffs, axis=0)
            use_transform[2, 3] = 0
            for _group in frame_group:
                transform_list.append(
                    {
                        "start": _group[0],
                        "stop": _group[-1],
                        "transform": use_transform,
                        "pcl_idxs": _group,
                        "pair": (target, source),
                    }
                )
        else:
            for _group, _diff in zip(frame_group, diffs):
                use_transform = np.eye(4)
                use_transform[:3, 3] = _diff
                use_transform[2, 3] = 0
                transform_list.append(
                    {
                        "start": _group[0],
                        "stop": _group[-1],
                        "transform": use_transform,
                        "pcl_idxs": _group,
                        "pair": (target, source),
                    }
                )

    joblib.dump(
        transform_list,
        os.path.join(os.path.dirname(pcl_file), "bpoint_transform_list_pre.p"),
    )
    if transform_aggregate and enforce_symmetry:
        print("Enforcing symmetry in breakpoint fixes...")
        uniq_pairs = list(set([_["pair"] for _ in transform_list]))
        uniq_transform = {}

        for _pair in uniq_pairs:
            try:
                transform = [
                    _["transform"] for _ in transform_list if _["pair"] == _pair
                ][0]
                uniq_transform[_pair] = transform
            except IndexError:
                pass
        
        # TODO: add indirect paths here... 
        new_transform = {}
        for (target, source), v in uniq_transform.items():
            try:
                reflection = np.linalg.inv(uniq_transform[(source, target)])
                new_transform[(target, source)] = np.nanmean(
                    np.stack([uniq_transform[(target, source)], reflection]), axis=0
                )
            except KeyError:
                new_transform[(target, source)] = uniq_transform[(target, source)]

        for i in range(len(transform_list)):
            transform_list[i]["transform"] = new_transform[transform_list[i]["pair"]]

    joblib.dump(
        transform_list,
        os.path.join(os.path.dirname(pcl_file), "bpoint_transform_list.p"),
    )
    idxsort = np.array([_["start"] for _ in transform_list]).argsort()
    sorted_transform_list = [transform_list[i] for i in idxsort]
    odometry = np.eye(4)

    for i in tqdm(range(len(sorted_transform_list)), desc="Correcting breakpoints"):
        odometry = odometry @ sorted_transform_list[i]["transform"]
        # align everything
        read_pcls = sorted_transform_list[i]["pcl_idxs"]
        mask_array = (pcl_coord_idx >= min(read_pcls)) & (
            pcl_coord_idx <= max(read_pcls)
        )
        adj = np.min(np.flatnonzero(mask_array))
        use_coord_index = pcl_coord_idx[mask_array]
        for _pcl_idx in read_pcls:
            pcl_read_idx = np.flatnonzero(_pcl_idx == use_coord_index) + adj
            # skip if there's no point cloud to fix...
            if len(pcl_read_idx) == 0:
                continue
            xyz = pcl_f["xyz"][slice(pcl_read_idx[0], pcl_read_idx[-1] + 1)]
            _pcl = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
            _pcl = _pcl.transform(odometry)
            pcl_f["xyz"][slice(pcl_read_idx[0], pcl_read_idx[-1] + 1)] = np.asarray(
                _pcl.points
            )


def reproject_pcl_to_depth(
    registration_file: str,
    intrinsics_file: str,
    batch_overlap: int = 50,
    batch_size: int = 2000,
    centroid_outlier_trim_nsigma: int = 5,
    interpolation_distance_threshold: float = 1.75,
    interpolation_fill_value: float = np.nan,
    interpolation_method: str = "nearest",
    project_xy: bool = True,
    smooth_kernel: Tuple[float, float, float] = (1.0, 0.75, 0.75),
    stitch_buffer: int = 10,
    visualize_results: bool = True,
    z_clip: float = 0,
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
    max_pts = np.nanmax(pcl_f["xyz"][:100], axis=0)
    min_pts = np.nanmin(pcl_f["xyz"][:100], axis=0)
    max_pts = np.nan_to_num(max_pts, -np.inf)
    min_pts = np.nan_to_num(min_pts, +np.inf)
    for batch in tqdm(range(0, len(pcl_f["xyz"]), max_batch_size), desc="Getting max"):
        pts = pcl_f["xyz"][batch : batch + max_batch_size]
        u, v, z = pcl_to_pxl_coords(
            pts,
            intrinsics_matrix[cameras[0]],
            z_scale=z_scale,
            post_z_shift=floor_distances[cameras[0]],
        )
        new_pts = np.vstack([u, v, z]).T
        new_pts_min = new_pts.copy()
        new_pts_max = new_pts.copy()
        new_pts_min = np.nanmin(new_pts, axis=0)
        new_pts_max = np.nanmax(new_pts, axis=0)
        new_pts_min = np.nan_to_num(new_pts_min, +np.inf)
        new_pts_max = np.nan_to_num(new_pts_max, -np.inf)
        max_pts = np.maximum(max_pts, new_pts_max)
        min_pts = np.minimum(min_pts, new_pts_min)

    buffer = np.floor(np.minimum(min_pts[:2], np.array([0, 0]))).astype("int") * -1
    buffer += stitch_buffer
    stitch_size = (
        next_even_number(max_pts[0] + buffer[0]),
        next_even_number(max_pts[1] + buffer[1]),
    )

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
        new_im = np.zeros(
            (len(read_pcls), stitch_size[1], stitch_size[0]), dtype="uint16"
        )
        reference_node = pcl_f["reference_node"][left_edge:right_edge]
        reference_node = [_.decode() for _ in reference_node]

        mask_array = (pcl_coord_index >= min(read_pcls)) & (
            pcl_coord_index <= max(read_pcls)
        )
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
            slice(left_edge, right_edge),
            slice(0, stitch_size[1]),
            slice(0, stitch_size[0]),
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


def compute_scalars(
    registration_file: str,
    intrinsics_file: str,
    batch_size: int = 2000,
    scalar_diff_tau: float = 0.05,
    scalar_tau: float = 0.1,
    z_range: Tuple[float, float] = (5, 1000),
) -> pd.DataFrame:
    # load intrinsics
    intrinsics_matrix, distortion_coeffs = format_intrinsics(toml.load(intrinsics_file))

    # directory with registration data
    registration_dir = os.path.dirname(os.path.abspath(registration_file))

    # directory with raw data (one dir up)
    data_dir = os.path.dirname(registration_dir)

    # ensure timestamps line up with registration
    metadata = toml.load(os.path.join(data_dir, "metadata.toml"))
    registration_metadata = toml.load(os.path.join(registration_dir, "pcls.toml"))
    cameras = list(metadata["camera_metadata"].keys())
    head_camera = cameras[0]

    ts_paths = {
        os.path.join(data_dir, f"{_cam}.txt"): _cam
        for _cam in registration_metadata["cameras"]
    }
    _, merged_ts = read_timestamps_multicam(
        ts_paths, merge_tolerance=registration_metadata["timestamp_merge_tolerance"]
    )

    fps = float(metadata["camera_metadata"][head_camera]["AcquisitionFrameRate"])
    floor_distance = float(registration_metadata["floor_distances"][head_camera])

    registration_metadata_file = os.path.join(registration_dir, "pcls.toml")
    registration_metadata = toml.load(registration_metadata_file)

    cx = intrinsics_matrix[head_camera][0, 2]
    cy = intrinsics_matrix[head_camera][1, 2]
    fx = intrinsics_matrix[head_camera][0, 0]
    fy = intrinsics_matrix[head_camera][1, 1]

    with h5py.File(registration_file, "r") as f:
        nframes = len(f["frames"])
        frame_ids = f["frame_id"][()]

    centroid = np.full((nframes, 3), np.nan, dtype="float32")
    centroid_px = np.full((nframes, 2), np.nan, dtype="float32")
    orientation = np.full((nframes,), np.nan, dtype="float32")
    axis_length = np.full((nframes, 2), np.nan, dtype="float32")
    sigma = np.full((nframes, 3), np.nan, dtype="float32")
    batches = range(0, nframes, batch_size)
    # batches = range(0, batch_size * 2, batch_size) # for testing only...

    with h5py.File(registration_file, "r") as f:
        for _batch in tqdm(batches, desc="Computing scalars"):
            working_range = range(_batch, min(_batch + batch_size, nframes))
            frame_batch = f["frames"][working_range]
            for _id, _frame in zip(working_range, frame_batch):
                mouse_mask = np.logical_and(_frame > z_range[0], _frame < z_range[1])

                # use contour for getting centroid, etc.
                cnts, hierarchy = cv2.findContours(
                    mouse_mask.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                tmp = np.array([cv2.contourArea(x) for x in cnts])
                mouse_contour = cnts[tmp.argmax()]
                new_mask = np.zeros(mouse_mask.shape, dtype="uint8")
                new_mask = cv2.drawContours(
                    new_mask, [mouse_contour], -1, color=1, thickness=cv2.FILLED
                )
                mouse_mask = new_mask

                v, u = np.where(mouse_mask)
                z = _frame[v, u]
                # pack into array and project to world coordinates for centroid
                uvz = np.hstack([u[:, None], v[:, None], z[:, None]])
                xyz = project_world_coordinates(
                    uvz, floor_distance=floor_distance, cx=cx, cy=cy, fx=fx, fy=fy
                )
                centroid[_id] = np.nanmean(xyz, axis=0)
                sigma[_id] = np.nanstd(xyz, axis=0)

                # use contour moments here...
                features = im_moment_features(mouse_contour)
                # here, centroid orientation and axis length are in pixels
                centroid_px[_id] = features["centroid"]
                orientation[_id] = features["orientation"]
                axis_length[_id] = features["axis_length"]

    all_data = np.hstack(
        [centroid, centroid_px, sigma, orientation[:, None], axis_length]
    )
    all_columns = [
        "x_mean_mm",
        "y_mean_mm",
        "z_mean_mm",
        "x_mean_px",
        "y_mean_px",
        "x_std_mm",
        "y_std_mm",
        "z_std_mm",
        "orientation_rad",
        "axis_length_1_px",
        "axis_length_2_px",
    ]

    scalar_tau_samples = np.round(scalar_tau * fps).astype("int")
    scalar_diff_tau_samples = np.round(scalar_diff_tau * fps).astype("int")

    df_scalars = pd.DataFrame(all_data, columns=all_columns, index=frame_ids)
    df_scalars["orientation_rad"] = (
        np.unwrap(df_scalars["orientation_rad"], period=np.pi) + np.pi
    )
    df_scalars["timestamps"] = merged_ts.loc[df_scalars.index, "system_timestamp"]
    df_scalars = df_scalars.rolling(scalar_tau_samples, 1, True).mean()

    df_scalars_diff = df_scalars.diff().rolling(scalar_diff_tau_samples, 1, True).mean()
    # divide by period to convert diff into s^-1
    period = df_scalars["timestamps"].diff()
    df_scalars_diff = df_scalars_diff.div(period, axis=0)

    velocity_3d = np.sqrt(
        (df_scalars_diff[["x_mean_mm", "y_mean_mm", "z_mean_mm"]] ** 2).sum(axis=1)
    )
    velocity_2d = np.sqrt(
        (df_scalars_diff[["x_mean_mm", "y_mean_mm"]] ** 2).sum(axis=1)
    )
    velocity_z = df_scalars_diff["z_mean_mm"]

    df_scalars["velocity_2d_mm_s"] = velocity_2d
    df_scalars["velocity_3d_mm_s"] = velocity_3d
    df_scalars["velocity_z_mm_s"] = velocity_z
    df_scalars["velocity_position_angle_rad_s"] = np.arctan2(
        df_scalars_diff["y_mean_mm"], df_scalars_diff["x_mean_mm"]
    )
    df_scalars["velocity_orientation_rad_s"] = df_scalars_diff["orientation_rad"]
    df_scalars.index.name = "frame_id"

    return df_scalars
