import click
import os
from markovids.util import convert_depth_to_pcl_and_register, reproject_pcl_to_depth


@click.group()
def cli():
    pass


# TODO: change point cloud merging to reduce scale changes...
# fmt: off
@cli.command(name="registration", context_settings={"show_default": True, "auto_envvar_prefix": "MARKOVIDS_REG"})
@click.argument("data_dir", type=click.Path(exists=True))
@click.option("--registration-dir", type=str, default="_registration", help="Directory for output", show_envvar=True)
@click.option("--intrinsics-file",type=click.Path(exists=True), default="intrinsics.toml", show_envvar=True, help="Path to intrinsics file")
@click.option("--segmentation-dir", type=str, default="_segmentation_tau-5", show_envvar=True)
@click.option("--background-spacing", type=int, default=500, show_envvar=True)
@click.option("--batch-size", type=int, default=1000, show_envvar=True)
@click.option("--batch-overlap", type=int, default=150, show_envvar=True)
@click.option("--burn-frames", type=int, default=500, show_envvar=True)
@click.option("--valid-height-range", type=(float, float), default=(10, 800), show_envvar=True)
@click.option("--floor-range", type=(float, float), default=(1300, 1600), show_envvar=True)
@click.option("--registration-max-correspondence-distance", type=float, default=1.0, show_envvar=True)
@click.option("--registration-fitness-threshold", type=float, default=.25, show_envvar=True)
@click.option("--registration-reference-future-len", type=int, default=150, show_envvar=True)
@click.option("--registration-reference-history-len", type=int, default=75, show_envvar=True)
@click.option("--registration-cleanup-nbs", type=int, default=5, show_envvar=True)
@click.option("--registration-cleanup-nbs-combined", type=int, default=15, show_envvar=True)
@click.option("--reproject-stitch-buffer", type=int, default=400, show_envvar=True)
@click.option("--reproject-interpolation-distance-threshold", type=float, default=1.75, show_envvar=True)
@click.option("--reproject-interpolation-method", type=str, default="nearest", show_envvar=True)
@click.option("--reproject-smooth-kernel", type=(float, float, float), default=(1., .75, .75), show_envvar=True)
@click.option("--reproject-smooth-kernel-bpoint", type=(float, float, float), default=(2., 1.5, 1.5), show_envvar=True)
# fmt: on
def cli_registration(
    data_dir,
    registration_dir,
    intrinsics_file,
    segmentation_dir,
    background_spacing,
    batch_size,
    batch_overlap,
    burn_frames,
    valid_height_range,
    floor_range,
    registration_max_correspondence_distance,
    registration_fitness_threshold,
    registration_reference_future_len,
    registration_reference_history_len,
    registration_cleanup_nbs,
    registration_cleanup_nbs_combined,
    reproject_stitch_buffer,
    reproject_interpolation_distance_threshold,
    reproject_interpolation_method,
    reproject_smooth_kernel,
    reproject_smooth_kernel_bpoint,
):

    registration_kwargs = {
        "max_correspondence_distance": registration_max_correspondence_distance,
        "fitness_threshold": registration_fitness_threshold,
        "reference_future_len": registration_reference_future_len,
        "reference_history_len": registration_reference_history_len,
        "cleanup_nbs": registration_cleanup_nbs,
        "cleanup_nbs_combined": registration_cleanup_nbs_combined,
    }

    print(f"Processing {data_dir}")

    convert_depth_to_pcl_and_register(
        data_dir,
        intrinsics_file,
        registration_dir=registration_dir,
        segmentation_dir=segmentation_dir,
        background_spacing=background_spacing,
        batch_size=batch_size,
        batch_overlap=batch_overlap,
        burn_frames=burn_frames,
        valid_height_range=valid_height_range,
        floor_range=floor_range,
        registration_kwargs=registration_kwargs,
    )

    reproject_pcl_to_depth(
        os.path.join(data_dir, registration_dir),
        intrinsics_file,
        stitch_buffer=reproject_stitch_buffer,
        interpolation_distance_threshold=reproject_interpolation_distance_threshold,
        interpolation_method=reproject_interpolation_method,
        smooth_kernel=reproject_smooth_kernel,
        smooth_kernel_bpoint=reproject_smooth_kernel_bpoint,
    )


if __name__ == "__main__":
    cli()
