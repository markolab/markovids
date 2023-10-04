import click
import os
import toml
from markovids.util import (
    convert_depth_to_pcl_and_register,
    reproject_pcl_to_depth,
    fix_breakpoints_combined,
    fix_breakpoints_single,
    compute_scalars,
)


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
@click.option("--registration-cleanup-nbs-combined", type=int, default=25, show_envvar=True)
@click.option("--registration-cleanup-nbs", type=int, default=7, show_envvar=True)
@click.option("--registration-cleanup-radius-combined", type=float, default=3.0, show_envvar=True)
@click.option("--registration-cleanup-radius", type=float, default=3.0, show_envvar=True)
@click.option("--registration-cleanup-sigma", type=float, default=2.0, show_envvar=True)
@click.option("--registration-cleanup-sigma-combined", type=float, default=2.0, show_envvar=True)
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
@click.option("--valid-height-range", type=(float, float), default=(10, 800), show_envvar=True)
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
            valid_height_range=valid_height_range,
        )
    else:
        print("Registration already complete, skipping...")

    session_name = os.path.normpath(data_dir).split(os.sep)[-1]
    print(f"Session: {session_name}")
    reproject_file = os.path.join(data_dir, registration_dir, f"{session_name}.hdf5")
    reproject_metadata_file = os.path.join(data_dir, registration_dir, f"{session_name}.toml")

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
@click.option("--z-threshold", type=float, default=5., show_envvar=True)
# fmt: on
def cli_compute_scalars(
    registration_file,
    batch_size,
    intrinsics_file,
    scalar_diff_tau,
    scalar_dir,
    scalar_tau,
    z_threshold,
):
    cli_params = locals()
    
    registration_dir = os.path.dirname(os.path.abspath(registration_file))
    data_dir = os.path.dirname(registration_dir)
    os.makedirs(os.path.join(data_dir, scalar_dir), exist_ok=True)  # make directory for output
    df_scalars = compute_scalars(
        registration_file,
        intrinsics_file,
        batch_size=batch_size,
        scalar_diff_tau=scalar_diff_tau,
        scalar_tau=scalar_tau,
        z_threshold=z_threshold,
    )

    df_scalars.to_parquet(os.path.join(data_dir, scalar_dir, "scalars.parquet"))
    with open(os.path.join(data_dir, scalar_dir, "scalars.toml"), "w") as f:
        toml.dump(cli_params, f)


if __name__ == "__main__":
    cli()
