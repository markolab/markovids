import numpy as np

def edge_weight_map(keypoints_xy, image_shape=(640, 480), edge_margin=50, mode='quadratic'):
    """
    Compute attenuation factors for keypoints based on proximity to image edges.

    Parameters:
        keypoints_xy: (N, 2) array of (x, y) keypoint coordinates
        image_shape: (H, W) of the depth frame
        edge_margin: pixels from edge to start full attenuation (e.g., 20 px)
        mode: 'linear', 'quadratic', or 'sigmoid' falloff

    Returns:
        edge_weights: (N,) array in [0, 1], 1 = fully trusted, 0 = near edge
    """
    w, h = image_shape
    x = keypoints_xy[:, 0]
    y = keypoints_xy[:, 1]

    # Distance from each edge
    left = x
    right = w - x
    top = y
    bottom = h - y
    min_edge_dist = np.minimum(np.minimum(left, right), np.minimum(top, bottom))

    norm = np.clip(min_edge_dist / edge_margin, 0, 1)

    if mode == 'linear':
        return norm
    elif mode == 'quadratic':
        return norm**2
    elif mode == 'sigmoid':
        return 1 / (1 + np.exp(-6 * (norm - 0.5)))
    else:
        raise ValueError(f"Unknown edge weight mode: {mode}")