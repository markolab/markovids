import open3d as o3d
import numpy as np


def pcl_from_depth(
    depth_image,
    intrinsic_matrix,
    z_scale=4.0,
    is_tensor=False,
    project_xy=True,
    post_scale=None,
    normal_radius=None,
    normal_nn=30,
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

    if post_scale is not None:
        xyz /= post_scale  # convert mm to m

    if is_tensor:
        pcl = o3d.t.geometry.PointCloud()
        pcl.point.positions = o3d.core.Tensor(xyz)
    else:
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(xyz)

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
        for i in np.arange(-transform_neighborhood, +transform_neighborhood+1):
            for j in np.arange(-transform_neighborhood, +transform_neighborhood+1):
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



def depth_from_pcl_interpolate(
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

    # tic = time.process_time()
    depth_image = np.full((height, width), np.nan)

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
    u = np.clip(u, 1, width - 2)
    v = np.clip(v, 1, height - 2)
    # print(f"Step 1 {time.process_time() - tic}")
    # tic = time.process_time()
    
    min_x, max_x = int(np.floor(min(u))), int(np.ceil(max(u)))
    min_y, max_y = int(np.floor(min(v))), int(np.ceil(max(v)))
    xx, yy = np.meshgrid(np.arange(min_x, max_x), np.arange(min_y, max_y))
    
    # print(f"Step 2 {time.process_time() - tic}")
    # tic = time.process_time()
    
    try:
        new_depth_image = griddata(
            (u, v), z, (xx, yy), method=interpolation_method, fill_value=fill_value
        )
    except:
        return depth_image
    
    # print(f"Step 3 {time.process_time() - tic}")
    # tic = time.process_time()
    
    # https://stackoverflow.com/questions/30655749/how-to-set-a-maximum-distance-between-points-for-interpolation-when-using-scipydepth_image =
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