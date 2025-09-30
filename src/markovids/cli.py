import click
import os
import toml
import numpy as np
from markovids.util import (
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
    from markovids.util import compute_scalars
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
        frame_downsample = flip_model_metadata["frame_downsample"]
        flip_model_use_class = int(flip_model_metadata["classifier_categories"]["flip"])

        def get_flips(data):
            use_data = data[:,::frame_downsample, ::frame_downsample]
            _, width, height = use_data.shape
            pred_onx = sess.run(None, {input_name: use_data.reshape(-1, width * height).astype("float32")})[1]
            return signal.medfilt(pred_onx[:,flip_model_use_class], flip_model_proba_smoothing) > .5


    registration_dir = os.path.dirname(os.path.abspath(registration_file))
    base_fname = os.path.basename(os.path.normpath(registration_file))
    data_dir = os.path.dirname(registration_dir)
    scalar_file = os.path.join(data_dir, scalar_path)
    scalars_df = pd.read_parquet(scalar_file, columns=["x_mean_px", "y_mean_px", "orientation_rad", "orientation_rad_unwrap"])

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
        chunks=(10, crop_size[1], crop_size[0]) # enables more efficient read access
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
                flip_idx = np.array(list(_batch))[flips]
                orientation_arr[flip_idx] += np.pi

        crop_f["cropped_frames"][_batch] = cropped_frames
        writer.write_frames(
            cropped_frames,
            frames_idx=_batch,
            vmin=preview_clims[0],
            vmax=preview_clims[1],
            inscribe_frame_number=True,
        )

    if use_flip_model:            
        scalars_df["orientation_rad_unwrap"] = (
            np.unwrap(scalars_df["orientation_rad"], period=np.pi) + np.pi
        )
        scalars_df.to_parquet(scalar_file)

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
@click.option("--batch-size",type=int, default=int(1e2), show_envvar=True)
@click.option("--overlap",type=int, default=int(5), show_envvar=True)
@click.option("--downsample", type=int, default=int(1), show_envvar=True)
# fmt: on
def cli_generate_qd_preview(
    input_dir,
    nbatches,
    batch_size,
    overlap,
    downsample,
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
        downsample=downsample,
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



# fmt: off
@cli.command(name="sync-depth-video", context_settings={"show_default": True, "auto_envvar_prefix": "MARKOVIDS_DEPTH"})
@click.argument("data_dir", type=click.Path(exists=True))
@click.option("--force/--no-force", default=False, show_envvar=True, help="Overwrite save directory if found")
@click.option("--batch-size", type=int, default=500, show_envvar=True)
@click.option("--intrinsics-file", type=click.Path(exists=True), default="intrinsics.toml", show_envvar=True, help="Path to intrinsics file")
@click.option("--save-dir", type=str, default="_proc", show_envvar=True, help="Output directory name")
@click.option("--undistort/--no-undistort", default=True, show_envvar=True, help="Apply lens undistortion")
@click.option("--preview-inpaint/--no-preview-inpaint", default=True, show_envvar=True, help="Fill holes in preview video")
# timestamp_kwargs options
@click.option("--ts-merge-tolerance", type=float, default=0.003, show_envvar=True, help="Timestamp merge tolerance in seconds")
@click.option("--ts-multiplexed/--ts-not-multiplexed", default=False, show_envvar=True, help="Multiplexed timestamps")
@click.option("--ts-burn-in", type=int, default=500, show_envvar=True, help="Number of burn-in frames to skip")
@click.option("--ts-return-full-sync-only/--ts-allow-partial-sync", default=True, show_envvar=True, help="Only return fully synchronized frames")
@click.option("--ts-timestamp-field", type=str, default="device_timestamp_ref", show_envvar=True, help="Timestamp field to use for synchronization")
# preview_kwargs options  
@click.option("--preview-crf", type=int, default=26, show_envvar=True, help="Video compression quality (lower=better)")
@click.option("--preview-vmin", type=float, default=5, show_envvar=True, help="Minimum value for preview colormap")
@click.option("--preview-vmax", type=float, default=90, show_envvar=True, help="Maximum value for preview colormap")
@click.option("--preview-cmap", type=str, default="viridis", show_envvar=True, help="Colormap for preview video")
@click.option("--preview-ncols", type=int, default=2, show_envvar=True, help="Number of columns in preview montage")
# bground_kwargs options
@click.option("--bground-step-size", type=int, default=1500, show_envvar=True, help="Frame step size for background calculation")
@click.option("--bground-save-dir", type=str, default="_bground", show_envvar=True, help="Directory to save background images")
@click.option("--bground-threads", type=int, default=5, show_envvar=True, help="Number of threads for background reader")
@click.option("--bground-force/--no-bground-force", default=False, show_envvar=True, help="Force recompute background")
# other options
@click.option("--reader-threads", type=int, default=4, show_envvar=True, help="Number of threads for video reader")
# fmt: on
def cli_sync_depth_video(
    data_dir,
    force,
    batch_size,
    intrinsics_file,
    save_dir,
    undistort,
    preview_inpaint,
    ts_merge_tolerance,
    ts_multiplexed,
    ts_burn_in,
    ts_return_full_sync_only,
    ts_timestamp_field,
    preview_crf,
    preview_vmin,
    preview_vmax,
    preview_cmap,
    preview_ncols,
    bground_step_size,
    bground_save_dir,
    bground_threads,
    bground_force,
    reader_threads,
):
    from markovids.util import sync_depth_videos
    from markovids.vid.io import format_intrinsics
    import toml
    import numpy as np

    cli_params = locals()

    if os.path.exists(intrinsics_file):
        print(f"Loading intrinsics file {intrinsics_file}")
        intrinsics_matrix, distortion_coeffs = format_intrinsics(toml.load(intrinsics_file))
    else:
        print("No intrinsics provided, skipping undistortion...")
        intrinsics_matrix = None
        distortion_coeffs = None

    # Build timestamp_kwargs from CLI options
    timestamp_kwargs = {
        "merge_tolerance": ts_merge_tolerance,
        "multiplexed": ts_multiplexed,
        "burn_in": ts_burn_in,
        "return_full_sync_only": ts_return_full_sync_only,
        "use_timestamp_field": ts_timestamp_field,
    }

    # Build preview_kwargs from CLI options
    preview_kwargs = {
        "crf": preview_crf,
        "vmin": preview_vmin,
        "vmax": preview_vmax,
        "cmap": preview_cmap,
        "ncols": preview_ncols,
    }

    # Build bground_kwargs from CLI options
    bground_kwargs = {
        "step_size": bground_step_size,
        "agg_func": np.median,  # Fixed to median for now
        "reader_kwargs": {"threads": bground_threads},
        "save_dir": bground_save_dir,
        "force": bground_force,
    }

    # Build reader_kwargs
    reader_kwargs = {"threads": reader_threads}
    output_dir = os.path.join(data_dir, save_dir)
    if os.path.exists(output_dir) and (not force):
        print(f"Directory {output_dir} exists, bailing...")
        return None

    sync_depth_videos(
        data_dir,
        save_dir=save_dir,
        timestamp_kwargs=timestamp_kwargs,
        undistort=undistort,
        intrinsics_matrix=intrinsics_matrix,
        distortion_coeffs=distortion_coeffs,
        preview_kwargs=preview_kwargs,
        bground_kwargs=bground_kwargs,
        preview_inpaint=preview_inpaint,
        reader_kwargs=reader_kwargs,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    cli()
