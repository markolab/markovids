import open3d as o3d
import numpy as np


def pcl_from_depth(
    depth_image,
    intrinsic_matrix,
    estimate_normals=True,
    z_scale=4.0,
    is_tensor=False,
    project_xy=True,
    post_scale=None,
    post_z_shift=None,
    normal_radius=None,
    normal_nn=30,
):
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]

    height, width = depth_image.shape

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u = u.astype("float32")
    v = v.astype("float32")
    raw_z = depth_image.astype("float32") / z_scale

    if post_z_shift is not None:
        use_z = post_z_shift - raw_z
    else:
        use_z = raw_z

    if project_xy:
        x = (u - cx) * raw_z / fx
        y = (v - cy) * raw_z / fy
    else:
        x = u
        y = v

    valid_points = ~np.isnan(use_z) & (use_z > 0)
    xyz = np.array(
        [
            x[valid_points].ravel(),
            y[valid_points].ravel(),
            use_z[valid_points].ravel(),
        ]
    ).T

    if post_scale is not None:
        xyz /= post_scale  # convert mm to m

    if is_tensor:
        pcl = o3d.t.geometry.PointCloud()
        pcl.point.positions = o3d.core.Tensor(xyz)
    else:
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(xyz)

    if estimate_normals:
        pcl.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=normal_nn))
    return pcl


def depth_from_pcl(
    pcl,
    intrinsic_matrix,
    width=640,
    height=480,
    z_scale=4.0,
    project_xy=True,
    post_scale=None,
    z_clip=1e-3,
    z_adjust=None,
    fill_value=0.0,
    buffer=200.0,
    transform_correct=False,
    transform_neighborhood=1,
):
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]

    try:
        points = pcl.point.positions.numpy().copy()
    except AttributeError:
        points = np.asarray(pcl.points)

    depth_image = np.zeros((height, width), "float")
    depth_image[:] = fill_value

    if post_scale is not None:
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

    u += buffer
    v += buffer
    u = np.clip(u, 0, width - 1)
    v = np.clip(v, 0, height - 1)

    if transform_correct:
        # hack to correct for rounding errors post-transformation
        # maybe consider a small neighborhood??
        for i in np.arange(-transform_neighborhood, +transform_neighborhood + 1):
            for j in np.arange(-transform_neighborhood, +transform_neighborhood + 1):
                depth_image[np.round(v + i).astype("int"), np.round(u + j).astype("int")] = z
        # depth_image[np.floor(v).astype("int"), np.floor(u).astype("int")] = z
        # depth_image[np.ceil(v).astype("int"), np.ceil(u).astype("int")] = z
        # depth_image[np.floor(v).astype("int"), np.ceil(u).astype("int")] = z
        # depth_image[np.ceil(v).astype("int"), np.floor(u).astype("int")] = z
    else:
        depth_image[np.round(v).astype("int"), np.round(u).astype("int")] = z

    if z_adjust is not None:
        depth_image = -depth_image + z_adjust

    depth_image[depth_image <= z_clip] = np.nan

    return depth_image


# use this to speed up cropping...
def pcl_to_pxl_coords(
    xyz, intrinsic_matrix, project_xy=True, post_z_shift=None, z_scale=4.0, post_scale=None
):
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]

    if post_scale is not None:
        points *= post_scale

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    if post_z_shift is not None:
        z = -z + post_z_shift / z_scale

    if project_xy:
        u = (x * fx) / z + cx
        v = (y * fy) / z + cy
    else:
        u = x
        v = y

    z *= z_scale

    return u, v, z


def depth_from_pcl_interpolate(
    pcl,
    intrinsic_matrix,
    width=640,
    height=480,
    z_scale=4.0,
    project_xy=True,
    post_scale=None,
    post_z_shift=None,
    z_clip=1e-3,
    z_adjust=None,
    fill_value=0.0,
    buffer=200.0,
    distance_threshold=1.3,
    interpolation_method="linear",
):
    from scipy.interpolate.interpnd import _ndim_coords_from_arrays
    from scipy.spatial import cKDTree
    from scipy.interpolate import griddata

    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]

    try:
        points = pcl.point.positions.numpy().copy()
    except AttributeError:
        points = np.asarray(pcl.points)

    # only include valid points...
    points = points[~np.isnan(points).any(axis=1)]
    u, v, z = pcl_to_pxl_coords(
        points,
        intrinsic_matrix,
        project_xy=project_xy,
        post_z_shift=post_z_shift,
        z_scale=z_scale,
        post_scale=post_scale,
    )

    # tic = time.process_time()
    depth_image = np.full((height, width), np.nan)

    # if post_scale is not None:
    #     points *= post_scale

    # x = points[:, 0]
    # y = points[:, 1]
    # z = points[:, 2]

    # if post_z_shift is not None:
    #     z = -z + post_z_shift / z_scale

    # if project_xy:
    #     u = (x * fx) / z + cx
    #     v = (y * fy) / z + cy
    # else:
    #     u = x
    #     v = y

    # z *= z_scale

    if np.isscalar(buffer)
        u += buffer
        v += buffer
    elif len(buffer) == 2:
        u += buffer[0]
        v += buffer[1]

    u = np.clip(u, 1, width - 2)
    v = np.clip(v, 1, height - 2)
    # print(f"Step 1 {time.process_time() - tic}")
    # tic = time.process_time()

    min_x, max_x = int(np.floor(min(u))), int(np.ceil(max(u)))
    min_y, max_y = int(np.floor(min(v))), int(np.ceil(max(v)))
    xx, yy = np.meshgrid(np.arange(min_x, max_x), np.arange(min_y, max_y))

    # print(f"Step 2 {time.process_time() - tic}")
    # tic = time.process_time()
    # https://stackoverflow.com/questions/28506050/understanding-inputs-and-outputs-on-scipy-ndimage-map-coordinates
    # test jax version as welll, we're spending so much time interpolating...
    try:
        new_depth_image = griddata(
            (u, v), z, (xx, yy), method=interpolation_method, fill_value=fill_value
        )
    except Exception as e:
        print(e)
        return depth_image

    # print(f"Step 3 {time.process_time() - tic}")
    # tic = time.process_time()

    # alternatively, could mask the output...
    # https://stackoverflow.com/questions/32272004/interpolation-of-numpy-array-with-a-maximum-interpolation-distance

    # https://stackoverflow.com/questions/30655749/how-to-set-a-maximum-distance-between-points-for-interpolation-when-using-scipy
    tree = cKDTree(np.vstack([u, v]).T)  # feed this in from elsewhere...
    xi = _ndim_coords_from_arrays((xx, yy))  # can also precook this...
    dists, indexes = tree.query(xi)
    # print(f"Step 4 {time.process_time() - tic}")
    # tic = time.process_time()
    #     # Copy original result but mask missing values with NaNs
    new_depth_image[dists > distance_threshold] = np.nan
    depth_image[min_y:max_y, min_x:max_x] = new_depth_image
    if z_adjust is not None:
        depth_image = -depth_image + z_adjust
    depth_image[depth_image <= z_clip] = np.nan

    # print(f"Step 5 {time.process_time() - tic}")
    # tic = time.process_time()

    return depth_image


# also see https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py
def trim_outliers(points, thresh=3):
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)
    modified_z_score = 0.6745 * diff / med_abs_deviation
    return points[modified_z_score < thresh]