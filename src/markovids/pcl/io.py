import open3d as o3d
import numpy as np


def pcl_from_depth(
    depth_image,
    intrinsic_matrix,
    z_scale=4.0,
    is_tensor=True,
    project_xy=True,
    post_scale=1e3,
):
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]

    height, width = depth_image.shape

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u = u.astype("float")
    v = v.astype("float")
    z = depth_image.astype("float") / z_scale
    valid_points = ~np.isnan(depth_image)

    if project_xy:
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
    else:
        x = u
        y = v

    xyz = np.array(
        [
            x[valid_points].ravel(),
            y[valid_points].ravel(),
            z[valid_points].ravel(),
        ]
    ).T

    xyz /= post_scale  # convert mm to m

    if is_tensor:
        pcl = o3d.t.geometry.PointCloud()
        pcl.point.positions = o3d.core.Tensor(xyz)
    else:
        pcl = o3d.geometry.PointCloud()
        pcl.points = xyz

    pcl.estimate_normals()
    return pcl


def depth_from_pcl(
    pcl,
    intrinsic_matrix,
    width=640,
    height=480,
    z_scale=4.0,
    project_xy=True,
    post_scale=1e3,
    z_clip=1e-3,
    z_adjust=None,
    fill_value=0.0,
    buffer=200.0,
):
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]

    try:
        points = pcl.point.positions.numpy().copy()
    except AttributeError:
        points = pcl.points.copy()

    depth_image = np.zeros((height, width), "float")
    depth_image[:] = fill_value

    points *= post_scale
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    if project_xy:
        u = (x * fx) / z + cx
        v = (y * fy) / z + cy
    else:
        u = x
        v = y

    z *= z_scale
    if z_adjust is not None:
        z = -z + z_adjust

    u += buffer
    v += buffer
    u = np.clip(np.round(u), 0, width - 1).astype("int")
    v = np.clip(np.round(v), 0, height - 1).astype("int")

    depth_image[v, u] = z
    depth_image[depth_image <= z_clip] = np.nan

    return depth_image
