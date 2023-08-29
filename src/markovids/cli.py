import click
import os
from markovids.util import convert_depth_to_pcl_and_register, reproject_pcl_to_depth


@click.group()
def cli():
    pass


# fmt: off
@cli.command(name="registration", context_settings={"show_default": True})
@click.argument("data_dir", type=click.Path(exists=True), help="Directory with dat/avi files")
@click.option("--registration-dir", type=str, default="_registration", help="Directory for output", show_envvar=True)
@click.option("--intrinsics-file",type=click.Path(exists=True), default="intrinsics.toml", help="Path to intrinsics file")
@click.option("--segmentation-dir", type=str, default="_segmentation_tau-5", show_envvar=True)
@click.option("--background-spacing", type=int, default=500, show_envvar=True)
@click.option("--batch-size", type=int, default=2000, show_envvar=True)
@click.option("--batch-overlap", type=int, default=200, show_envvar=True)
@click.option("--burn-frames", type=int, default=200, show_envvar=True)
@click.option("--valid-height-range", type=(float, float), default=(20, 800), show_envvar=True)
@click.option("--floor-range", type=(float, float), default=(1300, 1600), show_envvar=True)
@click.option("--registration-max-correspondence-distance", type=float, default=1.0, show_envvar=True)
@click.option("--registration-fitness-threshold", type=float, default=.35, show_envvar=True)
@click.option("--registration-reference-future-len", type=int, default=300, show_envvar=True)
@click.option("--registration-reference-history-len", type=int, default=200, show_envvar=True)
@click.option("--registration-cleanup-nbs", type=int, default=9, show_envvar=True)
@click.option("--registration-cleanup-nbs-combined", type=int, default=15, show_envvar=True)
# fmt: on
def cli_registration(
    data_dir,
    registration_dir,
    intrinsics_file,
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
):
    registration_kwargs = {
        "max_correspondence_distance": registration_max_correspondence_distance,
        "fitness_threshold": registration_fitness_threshold,
        "reference_future_len": registration_reference_future_len,
        "reference_history_len": registration_reference_history_len,
        "cleanup_nbs": registration_cleanup_nbs,
        "cleanup_nbs_combined": registration_cleanup_nbs_combined,
    }

    convert_depth_to_pcl_and_register(
        data_dir,
        intrinsics_file,
        registration_dir=registration_dir,
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
        stitch_buffer=400,
        interpolation_distance_threshold=1.75,
        interpolation_method="nearest",
        smooth_kernel=(1.0, 0.75, 0.75),
        smooth_kernel_bpoint=(2.0, 1.5, 1.5),
    )


if __name__ == "__main__":
    cli()
