import click
import os
import toml
import numpy as np
from markovids.util import (
    convert_depth_to_pcl_and_register,
    reproject_pcl_to_depth,
    fix_breakpoints_combined,
    fix_breakpoints_single,
    compute_scalars,
    alternating_excitation_vid_split,
    alternating_excitation_vid_preview,
    batch,
)
from markovids.vid.io import pixel_format_to_np_dtype, MP4WriterPreview
from markovids.vid.util import crop_and_rotate_frames
from tqdm.auto import tqdm


@click.group()
def cli():
    pass


# TODO: add check that segmentations are complete before proceeding...
# fmt: off
@cli.command(name="registration", context_settings={"show_default": True, "auto_envvar_prefix": "MARKOVIDS_REG"})
@click.argument("data_dir", type=click.Path(exists=True))
@click.option("--background-spacing", type=int, default=500, show_envvar=True)
@click.option("--batch-overlap", type=int, default=100, show_envvar=True)
@click.option("--batch-size", type=int, default=2000, show_envvar=True)
@click.option("--breakpoint-algorithm", type=click.Choice(["combined", "single"]), default="combined", show_envvar=True)
@click.option("--breakpoint-transform-aggregate", type=bool, default=True, show_envvar=True)
@click.option("--burn-frames", type=int, default=500, show_envvar=True)
@click.option("--floor-range", type=(float, float), default=(1300, 1600), show_envvar=True)
@click.option("--intrinsics-file",type=click.Path(exists=True), default="intrinsics.toml", show_envvar=True, help="Path to intrinsics file")
@click.option("--registration-algorithm", type=click.Choice(["pairwise", "multiway"]), default="pairwise", show_envvar=True)
@click.option("--registration-cleanup-nbs-combined", type=int, default=200, show_envvar=True) # previously 21
@click.option("--registration-cleanup-nbs", type=int, default=200, show_envvar=True) # previously 7
@click.option("--registration-cleanup-radius-combined", type=float, default=3.0, show_envvar=True)
@click.option("--registration-cleanup-radius", type=float, default=3.0, show_envvar=True)
@click.option("--registration-cleanup-sigma", type=float, default=1.5, show_envvar=True)
@click.option("--registration-cleanup-sigma-combined", type=float, default=1.5, show_envvar=True)
@click.option("--registration-dir", type=str, default="_registration", help="Directory for output", show_envvar=True)
@click.option("--registration-fitness-threshold", type=float, default=.25, show_envvar=True)
@click.option("--registration-max-correspondence-distance", type=float, default=1.0, show_envvar=True)
@click.option("--registration-reference-future-len", type=int, default=50, show_envvar=True)
@click.option("--registration-reference-history-len", type=int, default=25, show_envvar=True)
@click.option("--registration-tail-filter-pixels", type=int, default=21, show_envvar=True) # previously 17
@click.option("--registration-type", type=click.Choice(["p2p","p2pl","generalized"]), default="generalized", show_envvar=True)
@click.option("--reproject-batch-overlap", type=int, default=150, show_envvar=True)
@click.option("--reproject-batch-size", type=int, default=2000, show_envvar=True)
@click.option("--reproject-interpolation-distance-threshold", type=float, default=2.5, show_envvar=True)
@click.option("--reproject-interpolation-method", type=str, default="nearest", show_envvar=True)
@click.option("--reproject-smooth-kernel", type=(float, float, float), default=(1., .75, .75), show_envvar=True)
@click.option("--reproject-stitch-buffer", type=int, default=25, show_envvar=True)
@click.option("--segmentation-dir", type=str, default="_segmentation_tau-5", show_envvar=True)
@click.option("--test-run-batches", type=int, default=-1, show_envvar=True)
@click.option("--valid-height-range", type=(float, float), default=(20, 800), show_envvar=True) # previously (10, 800)
# fmt: on
def cli_registration(
    data_dir,
    background_spacing,
    batch_overlap,
    batch_size,
    breakpoint_algorithm,
    breakpoint_transform_aggregate,
    burn_frames,
    floor_range,
    intrinsics_file,
    registration_algorithm,
    registration_cleanup_nbs_combined,
    registration_cleanup_nbs,
    registration_cleanup_radius_combined,
    registration_cleanup_radius,
    registration_cleanup_sigma_combined,
    registration_cleanup_sigma,
    registration_dir,
    registration_fitness_threshold,
    registration_max_correspondence_distance,
    registration_reference_future_len,
    registration_reference_history_len,
    registration_tail_filter_pixels,
    registration_type,
    reproject_batch_overlap,
    reproject_batch_size,
    reproject_interpolation_distance_threshold,
    reproject_interpolation_method,
    reproject_smooth_kernel,
    reproject_stitch_buffer,
    segmentation_dir,
    test_run_batches,
    valid_height_range,
):
    # SAVE PARAMETERS
    cli_params = locals()
    store_dir = os.path.join(data_dir, registration_dir)

    if not os.path.exists(os.path.join(data_dir, segmentation_dir)):
        raise RuntimeError(
            f"Segmentation directory {os.path.join(data_dir, segmentation_dir)} does not exist"
        )

    if not os.path.exists(store_dir):
        os.makedirs(store_dir)
    else:
        pass

    registration_metadata_file = os.path.join(store_dir, "pcls.toml")
    if os.path.exists(registration_metadata_file):
        registration_metadata = toml.load(registration_metadata_file)
        registration_complete = registration_metadata["complete"]
    else:
        registration_complete = False

    if not registration_complete:
        with open(os.path.join(store_dir, "registration_parameters.toml"), "w") as f:
            toml.dump(cli_params, f)

        registration_kwargs = {
            "cleanup_nbs_combined": registration_cleanup_nbs_combined,
            "cleanup_nbs": registration_cleanup_nbs,
            "cleanup_radius_combined": registration_cleanup_radius_combined,
            "cleanup_radius": registration_cleanup_radius,
            "cleanup_sigma_combined": registration_cleanup_sigma_combined,
            "cleanup_sigma": registration_cleanup_sigma,
            "fitness_threshold": registration_fitness_threshold,
            "max_correspondence_distance": registration_max_correspondence_distance,
            "reference_future_len": registration_reference_future_len,
            "reference_history_len": registration_reference_history_len,
            "registration_type": registration_type,
        }

        print(f"Registering {data_dir}")

        convert_depth_to_pcl_and_register(
            data_dir,
            intrinsics_file,
            background_spacing=background_spacing,
            batch_overlap=batch_overlap,
            batch_size=batch_size,
            burn_frames=burn_frames,
            floor_range=floor_range,
            registration_algorithm=registration_algorithm,
            registration_dir=registration_dir,
            registration_kwargs=registration_kwargs,
            segmentation_dir=segmentation_dir,
            tail_filter_pixels=registration_tail_filter_pixels,
            test_run_batches=test_run_batches,
            valid_height_range=valid_height_range,
        )
    else:
        print("Registration already complete, skipping...")

    session_name = os.path.normpath(data_dir).split(os.sep)[-1]
    print(f"Session: {session_name}")
    reproject_file = os.path.join(data_dir, registration_dir, f"{session_name}.hdf5")
    reproject_metadata_file = os.path.join(
        data_dir, registration_dir, f"{session_name}.toml"
    )

    if os.path.exists(reproject_metadata_file):
        reproject_metadata = toml.load(reproject_metadata_file)
        reproject_complete = reproject_metadata["complete"]
    else:
        reproject_complete = False

    if not reproject_complete:
        print("Fixing breakpoints...")
        if breakpoint_algorithm == "combined":
            fix_breakpoints_combined(
                os.path.join(data_dir, registration_dir, "pcls.hdf5"),
                transform_aggregate=breakpoint_transform_aggregate,
            )
        elif breakpoint_algorithm == "single":
            fix_breakpoints_single(
                os.path.join(data_dir, registration_dir, "pcls.hdf5"),
                transform_aggregate=breakpoint_transform_aggregate,
            )
        print("Reprojecting data...")
        reproject_pcl_to_depth(
            reproject_file,
            intrinsics_file,
            batch_size=reproject_batch_size,
            batch_overlap=reproject_batch_overlap,
            stitch_buffer=reproject_stitch_buffer,
            interpolation_distance_threshold=reproject_interpolation_distance_threshold,
            interpolation_method=reproject_interpolation_method,
            smooth_kernel=reproject_smooth_kernel,
        )
    else:
        print("Skipping...")
        pass


# fmt: off
@cli.command(name="compute-scalars", context_settings={"show_default": True, "auto_envvar_prefix": "MARKOVIDS_REG"})
@click.argument("registration_file", type=click.Path(exists=True))
@click.option("--batch-size", type=int, default=3000, show_envvar=True)
@click.option("--intrinsics-file",type=click.Path(exists=True), default="intrinsics.toml", show_envvar=True, help="Path to intrinsics file")
@click.option("--scalar-diff-tau", type=float, default=.05, show_envvar=True)
@click.option("--scalar-dir", type=str, default="_scalars", help="Directory for output", show_envvar=True)
@click.option("--scalar-tau", type=float, default=.1, show_envvar=True)
@click.option("--z-range", type=(float, float), default=(5., 1000.), show_envvar=True)
# fmt: on
def cli_compute_scalars(
    registration_file,
    batch_size,
    intrinsics_file,
    # scalar_diff_tau,
    scalar_dir,
    # scalar_tau,
    z_range,
):
    cli_params = locals()

    registration_dir = os.path.dirname(os.path.abspath(registration_file))
    data_dir = os.path.dirname(registration_dir)
    os.makedirs(
        os.path.join(data_dir, scalar_dir), exist_ok=True
    )  # make directory for output
    df_scalars = compute_scalars(
        registration_file,
        intrinsics_file,
        batch_size=batch_size,
        # scalar_diff_tau=scalar_diff_tau,
        # scalar_tau=scalar_tau,
        z_range=z_range,
    )

    df_scalars.to_parquet(os.path.join(data_dir, scalar_dir, "scalars.parquet"))
    with open(os.path.join(data_dir, scalar_dir, "scalars.toml"), "w") as f:
        toml.dump(cli_params, f)


# fmt: off
@cli.command(name="crop-video", context_settings={"show_default": True, "auto_envvar_prefix": "MARKOVIDS_CROP"})
@click.argument("registration_file", type=click.Path(exists=True))
@click.option("--scalar-path", type=str, default="_scalars/scalars.parquet", help="Path to scalars", show_envvar=True)
@click.option("--batch-size", type=int, default=3000, show_envvar=True)
@click.option("--crop-size", type=(int, int), default=(180, 180), show_envvar=True)
@click.option("--flip-model", type=str, default=None, show_envvar=True)
@click.option("--flip-model-proba-smoothing", type=int, default=301, show_envvar=True)
@click.option("--output-dir", type=str, default="_crop")
@click.option("--preview-cmap", type=str, default="cubehelix")
@click.option("--preview-clims", type=(float, float), default=(0, 400))
# fmt: on
def cli_crop_video(
    registration_file,
    batch_size,
    scalar_path,
    flip_model,
    flip_model_proba_smoothing,
    crop_size,
    output_dir,
    preview_cmap,
    preview_clims,
):
    import pandas as pd
    import h5py
    import onnxruntime as ort
    from scipy import signal

    try:
        use_flip_model = os.path.exists(flip_model)
    except TypeError:
        use_flip_model = False

    cli_params = locals()

    if use_flip_model:
        print(f"Using flip model: {use_flip_model}")
        sess = ort.InferenceSession(flip_model, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name

        basename = os.path.splitext(flip_model)[0]
        metadata_fname = f"{basename}.toml"
        flip_model_metadata = toml.load(metadata_fname)
        flip_model_use_class = int(flip_model_metadata["classifier_categories"]["flip"])

        def get_flips(data):
            _, width, height = data.shape
            pred_onx = sess.run(None, {input_name: data.reshape(-1, width * height).astype("float32")})[1]
            return signal.medfilt(pred_onx[:,flip_model_use_class], flip_model_proba_smoothing) > .5


    registration_dir = os.path.dirname(os.path.abspath(registration_file))
    base_fname = os.path.basename(os.path.normpath(registration_file))
    data_dir = os.path.dirname(registration_dir)
    scalar_file = os.path.join(data_dir, scalar_path)
    scalars_df = pd.read_parquet(scalar_file, columns=["x_mean_px", "y_mean_px", "orientation_rad"])

    # scalar_metadata_file = os.path.splitext(scalar_file)[0] + ".toml"
    # scalars_metadata = toml.load(scalar_metadata_file)

    data_dir = os.path.dirname(registration_dir)
    os.makedirs(
        os.path.join(data_dir, output_dir), exist_ok=True
    )  # make directory for output
    output_file = os.path.join(data_dir, output_dir, base_fname)
    output_fname = os.path.splitext(output_file)[0]
    output_vid = f"{output_fname}.mp4"

    f = h5py.File(registration_file, "r")
    frames_dset = f["frames"]
    nframes = len(frames_dset)

    features = {
        "centroid": scalars_df[["x_mean_px", "y_mean_px"]].to_numpy(),
        "orientation": scalars_df["orientation_rad_unwrap"].to_numpy(),
    }

    crop_f = h5py.File(output_file, "w")
    crop_f.create_dataset(
        "cropped_frames",
        (nframes, crop_size[1], crop_size[0]),
        "uint16",
        compression="lzf",
    )

    writer = MP4WriterPreview(
        output_vid,
        frame_size=crop_size,
        cmap=preview_cmap,
    )

    read_batches = tqdm(
        batch(range(nframes), batch_size), total=int(np.ceil(nframes / batch_size))
    )
    for _batch in read_batches:
        # predict flips and fix orientation here...
        _features = {
            "centroid": features["centroid"][_batch],
            "orientation": features["orientation"][_batch],
        }
        use_frames = frames_dset[_batch]
        cropped_frames = crop_and_rotate_frames(
            use_frames, _features, crop_size=crop_size
        )
        if use_flip_model:
            flips = get_flips(cropped_frames)
            if len(flips) > 0:
                cropped_frames[flips] = np.rot90(cropped_frames[flips], k=2, axes=(1,2))
                
                orientation_arr = scalars_df["orientation_rad"].to_numpy().copy()
                orientation_arr[flips] += np.pi
                
                scalars_df["orientation_rad"] = orientation_arr
                scalars_df["orientation_rad_unwrap"] = (
                    np.unwrap(scalars_df["orientation_rad"], period=np.pi) + np.pi
                )
                scalars_df.to_parquet(scalar_file)
    
        crop_f["cropped_frames"][_batch] = cropped_frames
        writer.write_frames(
            cropped_frames,
            frames_idx=_batch,
            vmin=preview_clims[0],
            vmax=preview_clims[1],
            inscribe_frame_number=True,
        )

    with open(f"{output_fname}.toml", "w") as f:
        toml.dump(cli_params, f)

    writer.close()
    f.close()
    crop_f.close()


if __name__ == "__main__":
    cli()


# fmt: off
@cli.command(name="generate-qd-preview", context_settings={"show_default": True, "auto_envvar_prefix": "MARKOVIDS_QD_PREVIEW"})
@click.argument("input_dir", type=click.Path(exists=True))
@click.option("--nbatches", type=int, default=0, show_envvar=True)
@click.option("--batch-size",type=int, default=int(5e2), show_envvar=True)
@click.option("--overlap",type=int, default=int(5), show_envvar=True)
# fmt: on
def cli_generate_qd_preview(
    input_dir,
    nbatches,
    batch_size,
    overlap,
):
    cli_params = locals()
    metadata = toml.load(os.path.join(input_dir, "metadata.toml"))

    user_metadata = metadata["user_input"]
    cli_metadata = metadata["cli_parameters"]

    fname = "{}_{}_{}_pulse-widths-{}".format(
        user_metadata["subject"],
        user_metadata["session"],
        user_metadata["notes"],
        cli_metadata["hw_trigger_pulse_width"],
    )

    cameras = list(metadata["camera_metadata"].keys())

    load_dct = {}
    for camera, cfg in metadata["camera_metadata"].items():
        dtype = pixel_format_to_np_dtype(cfg["PixelFormat"])
        load_dct[camera] = {}
        load_dct[camera]["frame_size"] = (cfg["Width"], cfg["Height"])
        load_dct[camera]["dtype"] = dtype

    dat_paths = {os.path.join(input_dir, f"{_cam}.avi"): _cam for _cam in cameras}
    ts_paths = {os.path.join(input_dir, f"{_cam}.txt"): _cam for _cam in cameras}

    vid_paths = {
        "reflectance": f"{fname}_reflectance.mp4",
        "fluorescence": f"{fname}_fluorescence.mp4",
        "merge": f"{fname}_merge.mp4",
    }

    for _vid in vid_paths.values():
        if os.path.exists(_vid):
            raise RuntimeError(f"{_vid} already exists, bailing!")

    alternating_excitation_vid_preview(
        dat_paths,
        ts_paths,
        load_dct,
        nbatches=nbatches,
        batch_size=batch_size,
        overlap=overlap,
        vid_paths=vid_paths,
    )


# fmt: off
@cli.command(name="split-qd-videos", context_settings={"show_default": True, "auto_envvar_prefix": "MARKOVIDS_QD_PREVIEW"})
@click.argument("input_dir", type=click.Path(exists=True))
@click.option("--nbatches", type=int, default=0, show_envvar=True)
@click.option("--batch-size",type=int, default=int(5e2), show_envvar=True)
# fmt: on
def cli_split_qd_vids(
    input_dir,
    nbatches,
    batch_size,
):
    cli_params = locals()
    metadata = toml.load(os.path.join(input_dir, "metadata.toml"))
    cameras = list(metadata["camera_metadata"].keys())

    load_dct = {}
    for camera, cfg in metadata["camera_metadata"].items():
        dtype = pixel_format_to_np_dtype(cfg["PixelFormat"])
        load_dct[camera] = {}
        load_dct[camera]["frame_size"] = (cfg["Width"], cfg["Height"])
        load_dct[camera]["dtype"] = dtype

    dat_paths = {os.path.join(input_dir, f"{_cam}.avi"): _cam for _cam in cameras}
    ts_paths = {os.path.join(input_dir, f"{_cam}.txt"): _cam for _cam in cameras}

    alternating_excitation_vid_split(
        dat_paths,
        ts_paths,
        load_dct,
        nbatches=nbatches,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    cli()
