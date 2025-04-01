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

# defaults...
reference_camera = "Lucid Vision Labs-HTP003S-001-224500508"
incl_kpoints_fit_transform = [
    "back_bottom",
    "back_middle_lower",
    "back_middle_upper",
    "back_top",
    "left_hip",
    "right_hip",
    "left_shoulder",
    "right_shoulder",
]

plt_kpoints = [
    "tail_tip",
    "tail_middle",
    "tail_base",
    "back_bottom",
    "back_middle_lower",
    "back_middle_upper",
    "back_top",
    "left_hip",
    "right_hip",
    "left_shoulder",
    "right_shoulder",
]

noisy_keypoints = ["tail_tip", "tail_middle", "tail_base", "snout"]
smoothing = {
    "not_noisy": {"window_length": int(5), "poly_order": int(2)},
    "noisy": {"window_length": int(25), "poly_order": int(2)},
}
interpolate = 10
renderer_kwargs = {
    "trail_length": 5,
    "xlim": (-175, 225),
    "ylim": (-125, 275),
    "zlim": (-10, 105),
    # "elevation": 30,
    # "azimuth": 65,
}


def registration_pipeline(
    use_data_dir,
    kpoints_save_dir="_kpoints_v0_3d",
    reference_camera=reference_camera,
    intrinsics_matrix=None,
    distortion_coefficients=None,
    bground_erode_px=60,
    smoothing=smoothing,
    noisy_keypoints=noisy_keypoints,
    min_confidence=0.4,
    interpolate=10,
    z_scale=4.0,
    incl_kpoints_fit_transform=incl_kpoints_fit_transform,
    plt_kpoints=plt_kpoints,
    mp4_renderer="vedo",
    mp4_burn_in=50,
    save_file="merged_keypoints.h5",
):

    if (intrinsics_matrix is None) or (distortion_coefficients is None):
        raise RuntimeError(
            "Need intrinsics and distortion_coefficients dictionaries to continue"
        )

    cx = intrinsics_matrix[reference_camera][0, 2]
    cy = intrinsics_matrix[reference_camera][1, 2]
    fx = intrinsics_matrix[reference_camera][0, 0]
    fy = intrinsics_matrix[reference_camera][1, 1]

    cameras = list(intrinsics_matrix.keys())
    metadata = toml.load(os.path.join(use_data_dir, "metadata.toml"))
    bground_file = os.path.join(use_data_dir, "_bground", f"{reference_camera}.tiff")
    bground = tifffile.imread(bground_file)
    bground_roi = depth.plane.get_floor(bground.astype("float"), dilations=0)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (bground_erode_px, bground_erode_px)
    )  # erode walls, etc.
    use_bground_roi = cv2.erode(
        bground_roi, kernel
    )  # erode so we only get a big chunk of the middle
    floor_distance = np.median(bground[use_bground_roi])

    kpoints_metadata = toml.load(
        os.path.join(use_data_dir, kpoints_save_dir, f"{cameras[0]}.toml")
    )
    nbody_parts = len(kpoints_metadata["node_names"])

    ts_paths = {os.path.join(use_data_dir, f"{_cam}.txt"): _cam for _cam in cameras}
    ts, merged_ts = vid.io.read_timestamps_multicam(ts_paths, merge_tolerance=0.0035)

    kpoints_dat = {}
    for _cam in cameras:
        kpoints_dat[_cam] = joblib.load(
            os.path.join(use_data_dir, kpoints_save_dir, f"{_cam}.pkl.gz")
        )

    # only include fully sync'd data...
    use_frames = merged_ts.dropna()

    for _cam in cameras:
        kpoints_dat[_cam] = kpoints_dat[_cam][use_frames[_cam]]

    incl_kpoints_idx = [
        kpoints_metadata["node_names"].index(_incl)
        for _incl in incl_kpoints_fit_transform
    ]
    use_points = []
    use_points_cam = [reference_camera]
    use_points.append(
        kpoints_dat[reference_camera][:, incl_kpoints_idx, :].reshape(-1, 4)
    )
    for _cam in cameras:
        if _cam == reference_camera:
            continue
        else:
            use_points.append(kpoints_dat[_cam][:, incl_kpoints_idx, :].reshape(-1, 4))
            use_points_cam.append(_cam)

    excl = np.isnan(use_points[0]).any(axis=1)
    for _points in use_points[1:]:
        excl |= np.isnan(_points).any(axis=1)

    result_rigid = pcl.registration.bundle_adjust_rigid_fixed_structure(
        use_points[0][~excl, :3],
        use_points[1][~excl, :3],
        use_points[2][~excl, :3],
        weights_B=use_points[1][~excl, 3],
        weights_C=use_points[2][~excl, 3],
    )

    nframes = len(kpoints_dat[cameras[0]])

    ref_index = cameras.index(reference_camera)
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

    use_dat = copy.deepcopy(kpoints_dat)

    for _cam in cameras:
        xy = use_dat[_cam][..., [0, 1, 3]].reshape(-1, 3)
        edge_weighting = pcl.kpoints.edge_weight_map(xy[:, :2], edge_margin=100)
        xy[:, 2] *= edge_weighting
        xy = xy.reshape(-1, nbody_parts, 3)
        use_dat[_cam][..., 3] = xy[..., 2]

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

    # per-frame adjustment
    for i, _cam in enumerate(cameras):
        if i == reference_camera:
            continue
        use_points = proj_points[i]
        ref_points = proj_points[ref_index]
        for _frame in range(nframes):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                bias = np.nanmean(use_points[_frame] - ref_points[_frame], axis=0)[:3]
            # any nans should be replaced by most recent bias term...
            bias[np.isnan(bias)] = 0
            proj_points[i][_frame, :, :3] -= bias[None, :]

    merge_method = "weighted"
    merged_data = np.full((nframes, nbody_parts, 3), fill_value=np.nan)
    merged_conf = np.full((nframes, nbody_parts, 3), fill_value=np.nan)
    for i in range(len(cameras)):
        merged_conf[:, :, i] = proj_points[i, :, :, 3]
    for _frame in range(nframes):
        weights = proj_points[:, _frame, :, 3][..., None]
        weights = util.squash_conf(
            weights, min_cutoff=min_confidence
        )  # soft threshold the weights
        # merged_conf[_frame] = proj_points
        if merge_method == "weighted":
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

    all_keypoints = kpoints_metadata["node_names"]
    not_noisy_keypoints = list(set(all_keypoints).difference(noisy_keypoints))

    _test = pd.DataFrame(merged_data.reshape(-1, nbody_parts * 3))  # ONLY SMOOTH XYZ
    _test = util.hampel(_test, window=100, threshold=4, replace=False)
    if interpolate is not None:
        _test = _test.interpolate(
            method="linear",
            limit=interpolate,
            axis=0,
            limit_direction="both",
            limit_area="inside",
        )
    if smoothing is not None:
        for _noisy in noisy_keypoints:
            match = _test.filter(regex=_noisy, axis=1)
            _test[match] = _test.apply(
                lambda x: util.savgol_filter_missing(x, **smoothing["noisy"])
            )
        for _not_noisy in not_noisy_keypoints:
            match = _test.filter(regex=_not_noisy, axis=1)
            _test[match] = _test.apply(
                lambda x: util.savgol_filter_missing(x, **smoothing["not_noisy"])
            )

    merged_data_proc = _test.to_numpy().reshape(-1, nbody_parts, 3)
    plt_kpoints_idx = [
        kpoints_metadata["node_names"].index(_incl) for _incl in plt_kpoints
    ]

    # save smoothed and raw...
    merged_data_proj_smooth = pcl.io.project_world_coordinates(
        merged_data_proc.reshape(-1, 3),
        floor_distance=floor_distance,
        cx=cx,
        cy=cy,
        fx=fx,
        fy=fy,
        z_scale=z_scale,
    ).reshape(-1, nbody_parts, 3)
    merged_data_proj_raw = pcl.io.project_world_coordinates(
        merged_data.reshape(-1, 3),
        floor_distance=floor_distance,
        cx=cx,
        cy=cy,
        fx=fx,
        fy=fy,
        z_scale=z_scale,
    ).reshape(-1, nbody_parts, 3)

    all_bgrounds = {}

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

    timestamps = use_frames["system_timestamp"].to_numpy()
    with h5py.File(os.path.join(use_data_dir, kpoints_save_dir, save_file), "w") as f:
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
            data=merged_conf.astype("float32"),
            compression="gzip",
        )
        f.create_dataset(
            "timestamps", data=timestamps.astype("float64"), compression="gzip"
        )
        f.create_dataset("roi", data=bground_roi, compression="gzip")
        f.create_dataset("roi_merged", data=all_roi_points_proj, compression="gzip")

    metadata["transforms"] = new_transforms
    metadata["transform_type"] = "rigid"
    metadata["reference_camera"] = reference_camera
    metadata["camera_parameters"] = {
        "cx": cx,
        "cy": cy,
        "fx": fx,
        "fy": fy,
        "zscale": 4.0,
    }
    metadata["kpoints"] = kpoints_metadata
    metadata["transforms"] = {str(k): v for k, v in metadata["transforms"].items()}

    with open(
        os.path.join(
            use_data_dir, kpoints_save_dir, f"{os.path.splitext(save_file)[0]}.toml"
        ),
        "w",
    ) as f:
        toml.dump(metadata, f, encoder=toml.TomlNumpyEncoder())

    arr_slice = slice(mp4_burn_in, nframes)
    frame_ids = range(mp4_burn_in, nframes)
    movie_file = f"{os.path.splitext(save_file)[0]}.mp4"
    if mp4_renderer == "matplotlib":
        pcl.viz.visualize_xyz_trajectories_to_mp4(
            merged_data_proj_smooth[arr_slice, plt_kpoints_idx],
            os.path.join(use_data_dir, kpoints_save_dir, movie_file),
            fps=100,
            frame_ids=frame_ids,
            **renderer_kwargs,
        )
    elif mp4_renderer == "vedo":
        pcl.viz.visualize_xyz_trajectories_vedo(
            merged_data_proj_smooth[arr_slice, plt_kpoints_idx],
            os.path.join(use_data_dir, kpoints_save_dir, movie_file),
            fps=100,
            frame_ids=frame_ids,
            **renderer_kwargs,
        )
    else:
        pass
