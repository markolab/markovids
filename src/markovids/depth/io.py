import os
import h5py
import numpy as np
from markovids.depth.track import clean_roi


def load_segmentation_masks(
    dat_path,
    read_frames,
    segmentation_dir="_segmentation-tau-5",
    clean_masks=True,
    clean_masks_args={},
    mouse_category=1,
):
    dirname = os.path.dirname(dat_path)
    basename = os.path.splitext(os.path.basename(dat_path))[0]
    segmentation_fname = os.path.join(dirname, segmentation_dir, f"{basename}.hdf5")
    if isinstance(read_frames, (int, np.integer)):
        with h5py.File(segmentation_fname, "r") as f:
            mask = f["/labels"][read_frames]
        mask = mask == mouse_category
        mask = mask[None, ...]
    elif isinstance(read_frames, (list, np.ndarray)):
        _frames = np.array(read_frames)
        slc = slice(_frames[0], _frames[-1] + 1)
        _frames_no_offset = _frames - _frames[0]
        with h5py.File(segmentation_fname, "r") as f:
            mask = f["/labels"][slc]
        mask = mask[_frames_no_offset] == mouse_category
    else:
        raise RuntimeError("Did not understand format of frames")
    if clean_masks:
        mask = clean_roi(mask, progress_bar=False, **clean_masks_args)
    return mask