from numba import jit
import warnings
import numpy as np
import cv2

def pack_params(*args):
    param_dct = {}
    params = ["x0", "y0", "sigma_x", "sigma_y", "theta", "amplitude", "offset"]
    for _param, _arg in zip(params, args):
        param_dct[_param] = _arg
    return param_dct

@jit(nopython=True)
def gaussian_2d(coords, x0, y0, sigma_x, sigma_y, theta, amplitude, offset):
    x, y = coords
    x_diff = x - x0
    y_diff = y - y0

    a = (np.cos(theta) ** 2) / (2 * sigma_x**2) + (np.sin(theta) ** 2) / (
        2 * sigma_y**2
    )
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x**2) + (np.cos(theta) ** 2) / (
        2 * sigma_y**2
    )

    return (
        amplitude * np.exp(-(a * x_diff**2 + 2 * b * x_diff * y_diff + c * y_diff**2))
        + offset
    )

# @jit(nopython=True)
def estimate_gaussian_moments(image):
    y, x = np.indices(image.shape)
    total = np.sum(image)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        x_mean = np.sum(x * image) / total
        y_mean = np.sum(y * image) / total

    x_diff = x - x_mean
    y_diff = y - y_mean

    cov_xx = np.sum(image * x_diff**2) / total
    cov_yy = np.sum(image * y_diff**2) / total
    cov_xy = np.sum(image * x_diff * y_diff) / total

    mu = np.array([x_mean, y_mean])
    cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])

    return mu, cov


def get_closest_blob(roi_image, min_size=5):

    if roi_image.dtype != np.uint8:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            use_roi_image = roi_image.copy()
            use_roi_image = (use_roi_image - use_roi_image.min()) / (
                use_roi_image.max() - use_roi_image.min()
            )
            use_roi_image = (use_roi_image * 255).astype("uint8")
    else:
        use_roi_image = roi_image

    block_size = max(11, int(min(use_roi_image.shape) / 10) * 2 + 1)  # Must be odd
    binary = cv2.adaptiveThreshold(
        use_roi_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        -2,
    )

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Filter contours by size
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_size]

    if not contours:
        return None

    # Get the blob closest to the center of the ROI
    center_y, center_x = roi_image.shape[0] // 2, roi_image.shape[1] // 2

    best_contour = None
    min_dist = float("inf")

    for cnt in contours:
        # Calculate moments to find centroid
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Calculate distance to ROI center
        dist = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)

        if dist < min_dist:
            min_dist = dist
            best_contour = cnt

    if best_contour is None:
        return None

    # Create a mask for the selected blob
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, [best_contour], 0, 255, -1)
    mask = mask > 0
    return mask


def fit_2d_gaussian_with_moments(image):
    from scipy.optimize import least_squares
    # Create coordinate grid
    y, x = np.indices(image.shape)
    coords = (x.ravel(), y.ravel())
    image_flat = image.ravel()

    # Moment-based estimates
    mu, cov = estimate_gaussian_moments(image)
    x0, y0 = mu[0], mu[1]
    sigma_x = np.sqrt(cov[0, 0])
    sigma_y = np.sqrt(cov[1, 1])

    cov_xx = cov[0, 0]
    cov_xy = cov[0, 1]
    cov_yy = cov[1, 1]

    theta = 0.5 * np.arctan2(2 * cov_xy, cov_xx - cov_yy)

    amplitude = image.max() - np.median(image)
    offset = np.median(image)

    initial_guess = [x0, y0, sigma_x, sigma_y, theta, amplitude, offset]
    initial_guess_dct = pack_params(*initial_guess)
    bounds = (
        [-1, -1, 0.5, 0.5, -np.pi, 0, 0],
        [
            image.shape[1],
            image.shape[0],
            100,
            100,
            np.pi,
            np.max(image) * 2,
            np.max(image),
        ],
    )

    def residuals(params):
        model = gaussian_2d(coords, *params)
        return model - image_flat

    try:
        result = least_squares(
            residuals, 
            x0=initial_guess, 
            bounds=bounds, 
            method="trf", 
            loss="linear",
        )
        if result.success:
            param_dct = pack_params(*result.x)
            return param_dct, initial_guess_dct
        else:
            return None, initial_guess_dct
    except (RuntimeError, ValueError) as e:
        return None, initial_guess_dct