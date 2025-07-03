import numpy as np
import cv2
from tqdm.auto import tqdm


def _fit3(points):
    """Fit a plane to 3 points (min number of points for fitting a plane)
    Args:
        points (2d numpy array): each row is a group of points,
        columns correspond to x,y,z
    Returns:
        plane (1d numpy array): linear plane fit-->a*x+b*y+c*z+d=0
    """
    a = points[1, :] - points[0, :]
    b = points[2, :] - points[0, :]
    # cross prod
    normal = np.array(
        [
            [a[1] * b[2] - a[2] * b[1]],
            [a[2] * b[0] - a[0] * b[2]],
            [a[0] * b[1] - a[1] * b[0]],
        ]
    )
    denom = np.sum(np.square(normal))
    if denom < np.spacing(1):
        plane = np.empty((4,))
        plane[:] = np.nan
    else:
        normal /= np.sqrt(denom)
        d = np.dot(-points[0, :], normal)
        plane = np.hstack((normal.flatten(), d))

    return plane


def fit_ransac(
    depth_image,
    depth_range=(650, 750),
    iters=1000,
    noise_tolerance=30,
    in_ratio=0.1,
    progress_bar=False,
    mask=None,
):
    """Naive RANSAC implementation for plane fitting
    Args:
        depth_image (2d numpy array): hxw, background image to fit plane to
        depth_range (tuple): min/max depth (mm) to consider pixels for plane
        iters (int): number of RANSAC iterations
        noise_tolerance (float): dist. from plane to consider a point an inlier
        in_ratio (float): frac. of points required to consider a plane fit good
    Returns:
        best_plane (1d numpy array): plane fit to data
    """
    use_points = np.logical_and(depth_image > depth_range[0], depth_image < depth_range[1])

    if mask is not None:
        use_points = np.logical_and(use_points, mask)

    xx, yy = np.meshgrid(np.arange(depth_image.shape[1]), np.arange(depth_image.shape[0]))

    coords = np.vstack(
        (
            xx[use_points].ravel(),
            yy[use_points].ravel(),
            depth_image[use_points].ravel(),
        )
    )
    coords = coords.T

    best_dist = np.inf
    best_num = 0
    best_plane = None

    npoints = np.sum(use_points)

    for i in tqdm(range(iters), disable=not progress_bar, desc="Finding plane"):
        sel = coords[np.random.choice(coords.shape[0], 3, replace=True), :]
        tmp_plane = _fit3(sel)

        if np.all(np.isnan(tmp_plane)):
            continue

        dist = np.abs(np.dot(coords, tmp_plane[:3]) + tmp_plane[3])
        inliers = dist < noise_tolerance
        ninliers = np.sum(inliers)

        if (ninliers / npoints) > in_ratio and ninliers > best_num and np.mean(dist) < best_dist:
            best_dist = np.mean(dist)
            best_num = ninliers
            best_plane = tmp_plane

    if best_plane is None:
        raise RuntimeError("Plane never fit")

    coords = np.vstack((xx.ravel(), yy.ravel(), depth_image.ravel())).T
    dist = np.abs(np.dot(coords, best_plane[:3]) + best_plane[3])

    return best_plane, dist

strel_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

def get_floor(
    bground_im,
    floor_range=(1300, 1600),
    noise_tolerance=50,
    median_kernels=(3, 5),
    floor_threshold=30,
    weights=(1., 0., 0.),
    dilate_element=strel_element,
    dilations=5,
):
    from skimage import measure
    from scipy import stats, ndimage

    result = fit_ransac(bground_im, depth_range=floor_range, noise_tolerance=noise_tolerance)
    _dist_im = result[1].reshape(*bground_im.shape).astype("float32")
    valid_pixels = ~np.isnan(_dist_im)

    # # interpolate nans and filter
    for _med in median_kernels:
        _dist_im = cv2.medianBlur(_dist_im, _med)

    dist_im = _dist_im

    bin_im = dist_im < floor_threshold
    label_im = measure.label(bin_im)
    region_properties = measure.regionprops(label_im)
    
    areas = np.zeros((len(region_properties),))
    extents = np.zeros_like(areas)
    dists = np.zeros_like(extents)

    # get the max distance from the center, area and extent

    center = np.array(bin_im.shape) / 2

    for i, props in enumerate(region_properties):
        areas[i] = props.area
        extents[i] = props.extent
        tmp_dists = np.sqrt(np.sum(np.square(props.coords - center), 1))
        dists[i] = tmp_dists.max()
    
    ranks = np.vstack(
        (
            stats.rankdata(-areas, method="max"),
            stats.rankdata(-extents, method="max"),
            stats.rankdata(dists, method="max"),
        )
    )
    weight_array = np.array(weights, "float32")
    shape_index = np.mean(
        np.multiply(ranks.astype("float32"), weight_array[:, np.newaxis]), 0
    ).argsort()

    roi = np.zeros(bin_im.shape, dtype="bool")
    roi[
        region_properties[shape_index[0]].coords[:, 0],
        region_properties[shape_index[0]].coords[:, 1],
    ] = 1

    roi = ndimage.binary_fill_holes(roi).astype("uint8")
    roi = cv2.dilate(roi, dilate_element, iterations=dilations)

    return roi
