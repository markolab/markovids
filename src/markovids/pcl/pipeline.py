import ast

import cv2
import tifffile
import toml
import os
import numpy as np
import joblib
import copy
import h5py
import warnings
import pandas as pd
from markovids import depth, vid, pcl, util
from markovids.pcl.post_processing import post_processing
from collections import defaultdict
from pathlib import Path

def load_config(config_path):
    """
    Loads configuration from a TOML file and performs post-processing.

    Args:
        config_path (str): The file path to the TOML configuration file.

    Returns:
        dict: The processed configuration dictionary, with 'index_conf_map' 
            keys converted to integers.
    Raises:
        IOError: If the config_path is None or the file does not exist.
    """
    
    if config_path is None or not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    
    with open(config_path, "r") as f:
        config = toml.load(f)
    
    if "index_conf_map" in config:
        config["index_conf_map"] = {int(k): v for k, v in config["index_conf_map"].items()}
        
    return config

def nan_safe_linalg_norm(arr1, arr2, axis=1):
    """
    Compute L2 norm between two arrays, handling NaN values safely.
    
    Args:
        arr1 (np.ndarray): First array of shape (n, m) or (n,). Represents multiple points.
        arr2 (np.ndarray): Second array of shape (m,) or (1, m). Represents a single reference point.
        axis (int): Axis along which to compute the norm. Default is 1.

    Returns:
        np.ndarray: L2 norms with NaNs excluded. If all values are NaN for a
                    particular row, the result for that row will be NaN.
    """
    
    # Ensure arr2 is broadcastable to arr1
    if arr1.shape[-1] != arr2.shape[-1]:
        raise ValueError("Trailing dimensions of arr1 and arr2 must match for broadcasting.")

    # Find the difference, broadcasting arr2 if necessary
    diff = arr1 - arr2

    # Identify valid (non-NaN) elements
    valid_mask = ~np.isnan(diff)

    # Initialize an array for norms with NaN as default
    norm_result = np.full(arr1.shape[0], np.nan)

    # Compute norms row-wise where valid data exists
    for i in range(arr1.shape[0]):
        row_mask = valid_mask[i]
        if np.any(row_mask):  # Only compute if there are valid values
            norm_result[i] = np.linalg.norm(diff[i][row_mask])

    return norm_result

def get_bground_vals(keyps, _cam, bground_by_cam, width=640, height=480):
    """
    Extracts background intensity values at specific keypoint coordinates.

    Args:
        keyps (np.ndarray): Array of keypoints with shape (..., 2 or more), 
            where the first two dimensions of the last axis are x and y.
        _cam (str): The camera identifier key used to index bground_by_cam.
        bground_by_cam (dict): Dictionary mapping camera keys to 2D image arrays.
        width (int): Maximum width for clipping coordinates. Defaults to 640.
        height (int): Maximum height for clipping coordinates. Defaults to 480.

    Returns:
        np.ndarray: An array of the same leading shape as keyps containing 
            background values, with NaNs where keypoints were invalid.
    """
    
    bground = bground_by_cam[_cam]
    
    # Get x,y coordinates
    xy = keyps[...,:2]
    
    # Floor and clip coordinates in one step
    xy_int = np.clip(xy.astype(np.int32), 0, [width-1, height-1])
    
    # Create mask for valid (non-NaN) coordinates
    valid_mask = ~np.isnan(xy).any(axis=-1)
    
    # Initialize background values array with NaN
    bground_vals = np.full((keyps.shape[0], keyps.shape[1]), np.nan)
    
    # Index background values using valid coordinates
    bground_vals[valid_mask] = bground[xy_int[valid_mask,0], xy_int[valid_mask,1]]

    
    return bground_vals


def registration_pipeline(
    config_path,
    use_data_dir,
    kpoints_save_dir="_kpoints_v0_3d",
    intrinsics_matrix=None,
    distortion_coefficients=None,
    bground_erode_px=60,
    min_confidence=0.4,
    z_scale=1.0,
    mp4_max_render_frames=None,
    mp4_renderer="vedo",
    mp4_burn_in=50,
    save_file="merged_keypoints.h5",
    alt_save_dir=None,
    meta_path = None,
    render=False,
    bundle_adjust=False,
    transforms_path=None
):
    """
    Executes the full 3D keypoint registration pipeline, including coordinate 
    projection, multi-camera merging, and post-processing.

    Args:
        config_path (str): Path to the TOML configuration file.
        use_data_dir (str): Base directory containing the session data.
        kpoints_save_dir (str): Subdirectory name for saving/loading keypoints.
        intrinsics_matrix (dict): Dictionary of 3x3 camera intrinsic matrices.
        distortion_coefficients (dict): Dictionary of camera distortion vectors.
        bground_erode_px (int): Pixel radius for eroding background masks.
        min_confidence (float): Confidence threshold for merging keypoints.
        z_scale (float): Scaling factor applied to the Z-dimension.
        mp4_max_render_frames (int, optional): Max frames to include in video.
        mp4_renderer (str): Visualization backend ('vedo' or 'matplotlib').
        mp4_burn_in (int): Number of initial frames to skip in the render.
        save_file (str): Name of the output H5 file.
        alt_save_dir (str, optional): Alternative path to save outputs.
        meta_path (str, optional): Path to metadata file if not in use_data_dir.
        render (bool): If True, generates an MP4 visualization.
        bundle_adjust (bool): If True, uses bundle adjustment for registration.

    Returns:
        None: Results are saved directly to H5 and TOML files in the output path.

    Raises:
        RuntimeError: If intrinsic or distortion matrices are missing.
        IOError: If critical configuration or metadata files are not found.
    """
    
    cfg = load_config(config_path)
    
    reference_camera = cfg["reference_camera"]

    noisy_keypoints = cfg["noisy_keypoints"]
    incl_kpoints_fit_transform = cfg["incl_kpoints_fit_transform"]
    plt_kpoints = cfg["plt_kpoints"]
    skeleton = [tuple(item) for item in cfg["skeleton"]]
    fps = cfg["fps"]
    index_conf_map = cfg["index_conf_map"]
    renderer_kwargs = cfg["renderer_kwargs"]

    proc_order = cfg["proc_order"]

    postprocessing_params = cfg["post_processing"]

    # Extract the list of keypoints (and remove from dict to keep it clean)
    incl_kpoints_post_processing = postprocessing_params.pop("incl_kpoints_post_processing")

    if "temporal_regularization" in postprocessing_params:
        postprocessing_params["temporal_regularization"]["fps"] = fps
        
    print(f"Loaded post-processing params from {config_path}")

    output_path = alt_save_dir if alt_save_dir else os.path.join(use_data_dir, kpoints_save_dir)
     
    os.makedirs(output_path, exist_ok=True)
    if (intrinsics_matrix is None) or (distortion_coefficients is None):
        raise RuntimeError(
            "Need intrinsics and distortion_coefficients dictionaries to continue"
        )

    # Camera intrinsics
    cx = intrinsics_matrix[reference_camera][0, 2]
    cy = intrinsics_matrix[reference_camera][1, 2]
    fx = intrinsics_matrix[reference_camera][0, 0]
    fy = intrinsics_matrix[reference_camera][1, 1]

    cameras = list(intrinsics_matrix.keys())

    metadata_path = use_data_dir if meta_path is None else meta_path
    metadata_file = os.path.join(metadata_path, "metadata.toml")

    try:
        metadata = toml.load(metadata_file)
    except FileNotFoundError as e:
        warnings.warn(f"Did not find metadata file {metadata_file}")
        return None

    width = metadata["camera_metadata"][reference_camera]["Width"]
    height = metadata["camera_metadata"][reference_camera]["Height"]

    bground_by_cam = {}
    floor_dist_cam = {}

    for camera in cameras:
        intrinsic_matrix = intrinsics_matrix[camera]
        distortion_coeff = distortion_coefficients[camera]
        
        bground_file = os.path.join(use_data_dir, "_bground", f"{camera}.tiff")

        bground = tifffile.imread(bground_file)
        bground = cv2.undistort(bground, intrinsic_matrix, distortion_coeff)
        bground_roi = depth.plane.get_floor(bground.astype("float"), dilations=0)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (bground_erode_px, bground_erode_px)
        )  # erode walls, etc.
        use_bground_roi = cv2.erode(
            bground_roi, kernel
        )  # erode so we only get a big chunk of the middle

        floor_distance = np.median(bground[use_bground_roi]) / 4.0 
        
        floor_dist_cam[camera] = floor_distance
        
        bground_by_cam[camera] = bground.T / 4.0

    # Load 3D keypoints, computed previously
    kpoints_metadata = toml.load(
        os.path.join(use_data_dir, "_proc", kpoints_save_dir, f"{cameras[0]}.toml")
    )

    nbody_parts = len(kpoints_metadata["node_names"])

    kpoints_dat = {}
    for _cam in cameras:
        kpoints_dat[_cam] = joblib.load(
            os.path.join(use_data_dir, "_proc", kpoints_save_dir, f"{_cam}.pkl.gz")
        )

    min_frames = min(data.shape[0] for data in kpoints_dat.values())
    for _cam in cameras:
        bground = bground_by_cam[_cam]
        kpoints_dat[_cam] = kpoints_dat[_cam][:min_frames]
        
        # Get x,y coordinates
        xy = kpoints_dat[_cam][...,:2]
        
        # Floor and clip coordinates in one step
        xy_int = np.clip(xy.astype(np.int32), 0, [width-1, height-1])
        
        # Create mask for valid (non-NaN) coordinates
        valid_mask = ~np.isnan(xy).any(axis=-1)
        
        # Initialize background values array with NaN
        bground_vals = np.full((min_frames, kpoints_dat[_cam].shape[1]), np.nan)
        
        # Index background values using valid coordinates
        bground_vals[valid_mask] = bground[xy_int[valid_mask,0], xy_int[valid_mask,1]]
        
        # Update z-coordinates
        kpoints_dat[_cam][...,2] = -1 * kpoints_dat[_cam][...,2] + bground_vals

    n_frames, nbody_parts, dims = kpoints_dat[reference_camera].shape

    # Convert to World coordinates
    kpoints_dat_conv = defaultdict(lambda: np.zeros((n_frames, nbody_parts, dims)))

    for _cam in cameras:
        cx = intrinsics_matrix[_cam][0, 2]
        cy = intrinsics_matrix[_cam][1, 2]
        fx = intrinsics_matrix[_cam][0, 0]
        fy = intrinsics_matrix[_cam][1, 1]
        
        _converted = pcl.io.project_world_coordinates(
            kpoints_dat[_cam][..., :3].reshape(-1, 3),
            floor_distance=None,
            cx=cx,
            cy=cy,
            fx=fx,
            fy=fy,
            z_scale=1.0,
        ).reshape(-1, nbody_parts, 3)
        
        kpoints_dat_conv[_cam][..., :3] = _converted
        kpoints_dat_conv[_cam][..., 3] = kpoints_dat[_cam][..., 3]

    # Only include subset of nodes for transform
    incl_kpoints_idx = [
        kpoints_metadata["node_names"].index(_incl)
        for _incl in incl_kpoints_fit_transform
    ]

    use_points = []
    use_points_cam = [reference_camera]
    use_points.append(
        kpoints_dat_conv[reference_camera][:, incl_kpoints_idx, :].reshape(-1, 4)
    )

    for _cam in cameras:
        if _cam == reference_camera:
            continue
        else:
            use_points.append(kpoints_dat_conv[_cam][:, incl_kpoints_idx, :].reshape(-1, 4))
            use_points_cam.append(_cam)

    use_points = np.array(use_points)
    has_nans = np.isnan(use_points).any(axis=(0, 2))
    
    conf_threshold = 0.65
    high_confidence = (use_points[:3, :, 3] > conf_threshold).all(axis=0)
    
    excl = has_nans | (~high_confidence)

    # excl = np.isnan(use_points[0]).any(axis=1)
    # for _points in use_points[1:]:
    #     excl |= np.isnan(_points).any(axis=1)

    nframes = len(kpoints_dat[cameras[0]])
    ref_index = cameras.index(reference_camera)

    if transforms_path is None:

        print("No transforms file provided. Estimating transforms using rigid registration...")

        if bundle_adjust:
            result_rigid = pcl.registration.bundle_adjust_rigid_fixed_structure(
                use_points[0][~excl, :3],
                use_points[1][~excl, :3],
                use_points[2][~excl, :3],
                weights_B=use_points[1][~excl, 3],
                weights_C=use_points[2][~excl, 3],
            )
        else:
            result_rigid = pcl.registration.estimate_transform(
                use_points[0][~excl, :3],
                use_points[1][~excl, :3],
                use_points[2][~excl, :3],
                weights_B=use_points[1][~excl, 3],
                weights_C=use_points[2][~excl, 3],
            )

        
        new_transforms = {}
        new_transforms[(use_points_cam[1], reference_camera)] = (
            result_rigid["B_to_A"]["R"],
            result_rigid["B_to_A"]["t"],
        )
        new_transforms[(reference_camera, reference_camera)] = np.eye(3), np.zeros((3,))
        new_transforms[(use_points_cam[2], reference_camera)] = (
            result_rigid["C_to_A"]["R"],
            result_rigid["C_to_A"]["t"],
        )

    else:
        new_transforms = toml.load(transforms_path)
        # Cast the loaded lists back into NumPy arrays
        new_transforms = {
            ast.literal_eval(k): (np.array(v[0]), np.array(v[1])) 
            for k, v in new_transforms.items()
        }
        print(f"Using transforms from file: {transforms_path}")

    use_dat = copy.deepcopy(kpoints_dat_conv)
    use_dat_edge = copy.deepcopy(kpoints_dat)

    # weight points based on proximity to edges
    for _cam in cameras:
        xy = use_dat_edge[_cam][..., [0, 1, 3]].reshape(-1, 3)
        edge_weighting = pcl.kpoints.edge_weight_map(xy[:, :2], xy[..., 2], edge_margin=25)
        xy[:, 2] = edge_weighting
        xy = xy.reshape(-1, nbody_parts, 3)
        use_dat[_cam][..., 3] = xy[..., 2]

    # Project points into common space, drop nans
    proj_points = np.full((len(cameras), nframes, nbody_parts, 4), fill_value=np.nan)
    for i, _cam in enumerate(cameras):
        proj_points[i] = use_dat[_cam].copy()
        R, t = new_transforms[(_cam, reference_camera)]
        _points = use_dat[_cam].reshape(-1, 4)
        _points = (R @ _points[:, :3].T).T + t
        proj_points[i][:, :, :3] = _points.reshape(-1, nbody_parts, 3)
        for _frame in range(nframes):
            rem = np.isnan(proj_points[i][_frame]).any(axis=-1)
            proj_points[i][_frame][rem, :] = np.nan

    for i, _cam in enumerate(cameras):
        if _cam == reference_camera:
            continue

        use_points = proj_points[i]
        ref_points = proj_points[ref_index]

        for _frame in range(nframes):
            # Confidence threshold for high-confidence keypoints
            confidence_threshold = 0.6

            # Find high-confidence keypoints in the reference camera
            ref_confidences = ref_points[_frame, :, 3]
            ref_high_conf_mask = ref_confidences >= confidence_threshold

            # Find high-confidence keypoints in the current camera
            use_confidences = use_points[_frame, :, 3]
            use_high_conf_mask = use_confidences >= confidence_threshold

            # Compute the intersection of high-confidence keypoints
            high_conf_mask = ref_high_conf_mask & use_high_conf_mask

            # Check if there are more than 4 high-confidence keypoints in the intersection
            if np.sum(high_conf_mask) > 4:
                valid_use_points = use_points[_frame][high_conf_mask, :3]
                valid_ref_points = ref_points[_frame][high_conf_mask, :3]

                # Compute the bias using the valid high-confidence keypoints
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    bias = np.nanmean(valid_use_points - valid_ref_points, axis=0)

                # Replace NaNs in bias with the most recent bias term
                bias[np.isnan(bias)] = 0

                # Apply the bias correction to all keypoints in the current frame
                proj_points[i][_frame, :, :3] -= bias[None, :]

    # merge data via a weighted average
    merge_method = "mixed"
    min_confidence = 0.4
    distance_threshold = 15
    merged_data = np.full((nframes, nbody_parts, 3), fill_value=np.nan)
    merged_conf = np.full((nframes, nbody_parts, 3), fill_value=np.nan)
    for i in range(len(cameras)):
        merged_conf[:, :, i] = proj_points[i, :, :, 3]
    for _frame in range(nframes):
        # soft threshold the weights
        # merged_conf[_frame] = proj_points
        if merge_method == "weighted":
            weights = proj_points[:, _frame, :, 3][..., None]
            weights = util.squash_conf_dynamic(
                weights, index_conf_map
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                weighted_average = np.nansum(
                    (proj_points[:, _frame, :, :3] * weights), axis=0
                ) / np.nansum(weights, axis=0)
            merged_data[_frame] = weighted_average
        elif merge_method == "max":
            for i in range(nbody_parts):
                try:
                    use_cam = np.nanargmax(proj_points[:, _frame, i, 3], axis=0)
                except ValueError:
                    continue
                merged_data[_frame, i, :] = proj_points[use_cam, _frame, i, :3]
        elif merge_method == "mixed":
            for i in range(nbody_parts):
                # Extract confidence values for this frame and body part
                confidences = proj_points[:, _frame, i, 3]

                # Find the most confident camera for this body part
                if np.all(np.isnan(confidences)) or np.nanmax(confidences) < min_confidence:
                    continue  # Skip if no valid predictions or all below the confidence threshold
                ref_cam = np.nanargmax(confidences)
                ref_point = proj_points[ref_cam, _frame, i, :3]

                # Compute L2 distances from all other cameras to the reference point
                distances = nan_safe_linalg_norm(
                    proj_points[:, _frame, i, :3], ref_point[None, :]
                )

                # Create a mask for valid points within the distance threshold
                valid_mask = (distances <= distance_threshold) & ~np.isnan(distances)

                # Skip merging if no valid points remain
                if not np.any(valid_mask):
                    continue

                # Use a weighted average of valid points
                valid_keypoints = proj_points[valid_mask, _frame, i, :3]
                valid_confidences = confidences[valid_mask]

                part_cutoff = index_conf_map.get(i, 0.05) 
                weights = np.where(valid_confidences > part_cutoff, valid_confidences**2, 0)
                weights = weights[:, None] 

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    weighted_average = np.nansum(
                        valid_keypoints * weights, axis=0
                    ) / np.nansum(weights, axis=0)

                # Assign the merged keypoint to the output array
                merged_data[_frame, i, :] = weighted_average
            

    all_keypoints = kpoints_metadata["node_names"]
    not_noisy_keypoints = list(set(all_keypoints).difference(noisy_keypoints))

    merged_data_proc = merged_data.copy() 
    merged_conf_proc = merged_conf.copy()

    if len(proc_order) >= 1:
        final_smoothed, final_conf = post_processing(
            merged_data,
            merged_conf,
            kpoints_metadata,
            postprocessing_params,
            incl_kpoints_post_processing,
            skeleton,
            proc_order=proc_order,
        )

        incl_kpoints_post_proc_idx = [
            kpoints_metadata["node_names"].index(_incl)
            for _incl in incl_kpoints_post_processing
        ]

        merged_data_proc[:, incl_kpoints_post_proc_idx] = final_smoothed
    else:
        print("No post processing will be done...")
        final_conf = merged_conf.copy()

    plt_kpoints_idx = [
        kpoints_metadata["node_names"].index(_incl) for _incl in plt_kpoints
    ]

    # save smoothed and raw after projecting into world coordinates (mm)...
    merged_data_proj_smooth = merged_data_proc.copy()

    merged_data_proj_raw = merged_data.copy()

    all_bgrounds = {}

    # region of interest and background files
    for _cam in cameras:
        bground_file = os.path.join(use_data_dir, "_bground", f"{_cam}.tiff")
        bground = tifffile.imread(bground_file)
        bground_roi = depth.plane.get_floor(bground.astype("float"), dilations=0)
        R, t = new_transforms[(_cam, reference_camera)]
        _tmp = np.vstack(np.where(bground_roi > 0))
        roi_points = _tmp.copy()
        roi_points[0, :] = _tmp[1, :]
        roi_points[1, :] = _tmp[0, :]
        all_bgrounds[_cam] = np.round((R[:2, :2] @ roi_points).T + t[:2]).astype("int")
    all_roi_points = np.concatenate(list(all_bgrounds.values()))
    all_roi_points = np.unique(all_roi_points, axis=0)
    all_roi_points_proj = pcl.io.project_world_coordinates(
        np.hstack([all_roi_points, np.zeros((all_roi_points.shape[0],1))]),
        floor_distance=floor_distance,
        cx=cx,
        cy=cy,
        fx=fx,
        fy=fy,
        z_scale=z_scale,
    )

    timestamp_path = os.path.join(use_data_dir, "_proc", "timestamps.txt")
    df = pd.read_csv(timestamp_path)
    df.columns = df.columns.str.replace(r"[()',]", "", regex=True).str.replace(" ", "_")

    with h5py.File(os.path.join(output_path, save_file), "w") as f:
        f.create_dataset(
            "merged_keypoints_smooth",
            data=merged_data_proj_smooth.astype("float32"),
            compression="gzip",
        )
        f.create_dataset(
            "merged_keypoints_raw",
            data=merged_data_proj_raw.astype("float32"),
            compression="gzip",
        )
        f.create_dataset(
            "merged_keypoints_confidence",
            data=merged_conf_proc.astype("float32"),
            compression="gzip",
        )
        f.create_dataset(
            "post_processing_confidence",
            data=final_conf.astype("float32"),
            compression="gzip",
        )
        f.create_dataset(
            "proj_point_conf",
            data=proj_points[..., 3].astype("float32"),
            compression="gzip",
        )

        # for _col in use_frames.columns:
        #     f.create_dataset(
        #         f"index/{_col}",
        #         data=use_frames[_col].to_numpy(),
        #         compression="gzip"
        #     )

        # f.create_dataset(f"index/frame_id",
        #                  data=use_frames.index.to_numpy(),
        #                  compression="gzip")
        
        f.create_dataset(
            "device_timestamp_ref", data=df["device_timestamp_ref"], compression="gzip"
        )
        f.create_dataset("roi", data=bground_roi, compression="gzip")
        f.create_dataset("roi_merged", data=all_roi_points_proj, compression="gzip")

    metadata["transforms"] = new_transforms
    metadata["transform_type"] = "rigid"
    metadata["reference_camera"] = reference_camera
    metadata["cameras"] = cameras
    metadata["camera_parameters"] = {
        "cx": cx,
        "cy": cy,
        "fx": fx,
        "fy": fy,
        "zscale": 4.0,
    }
    metadata["kpoints"] = kpoints_metadata
    metadata["transforms"] = {str(k): v for k, v in metadata["transforms"].items()}

    toml_path = os.path.join(output_path, f"{os.path.splitext(save_file)[0]}.toml")
    with open(
        toml_path,
        "w",
    ) as f:
        toml.dump(metadata, f, encoder=toml.TomlNumpyEncoder())

    if render:
        if mp4_max_render_frames is not None:
            max_render_frames = np.minimum(mp4_max_render_frames, nframes)
        else:
            max_render_frames = nframes
        arr_slice = slice(mp4_burn_in, max_render_frames)
        frame_ids = range(mp4_burn_in, max_render_frames)
        movie_file = f"{os.path.splitext(save_file)[0]}.mp4"

        if mp4_renderer == "matplotlib":
            pcl.viz.visualize_xyz_trajectories_to_mp4(
                merged_data_proj_smooth[arr_slice, plt_kpoints_idx],
                os.path.join(output_path,movie_file),
                fps=100, # REMOVE HARD-CODING!
                frame_ids=frame_ids,
                **renderer_kwargs,
            )
        elif mp4_renderer == "vedo":
            pcl.viz.visualize_xyz_trajectories_vedo(
                merged_data_proj_smooth[arr_slice, plt_kpoints_idx],
                os.path.join(output_path, movie_file),
                fps=100, # REMOVE HARD-CODING!
                frame_ids=frame_ids,
                **renderer_kwargs,
            )
        else:
            pass
