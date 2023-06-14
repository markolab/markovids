import open3d as o3d
import numpy as np
from typing import Optional, Union
from tqdm.auto import tqdm


default_t_criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(
    relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=int(1e3)
)

default_kernel = o3d.t.pipelines.registration.robust_kernel.RobustKernel(
    o3d.t.pipelines.registration.robust_kernel.RobustKernelMethod.TukeyLoss,
    2.0,
)
default_t_estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(
    default_kernel
)


def pairwise_registration(
    source: o3d.t.geometry.PointCloud,
    target: o3d.t.geometry.PointCloud,
    max_correspondence_distance: float = 0.005,
    criteria: o3d.t.pipelines.registration.ICPConvergenceCriteria = default_t_criteria,
    init_transformation: Optional[o3d.core.Tensor] = o3d.core.Tensor(
        np.identity(4)
    ),
    estimation: o3d.t.pipelines.registration.TransformationEstimation = default_t_estimation,
    compute_information=True,
):
    
    if init_transformation is None:
        init_transformation = np.eye(4)
        c0 = target.get_center().numpy()
        c1 = source.get_center().numpy()
        df = c0 - c1
        init_transformation[:3, 3] = df

    _result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance,
        o3d.core.Tensor(init_transformation),
        estimation,
        criteria,
    )

    dct = {
        "fitness": _result.fitness,
        "inlier_rmse": _result.inlier_rmse,
        "transformation": _result.transformation.numpy(),
    }

    if compute_information:
        information_icp = o3d.pipelines.registration.get_information_matrix(
            source, target, max_correspondence_distance * 1.4, _result.transformation
        )
        dct["information"] = information_icp.numpy()
    return dct



def icp_multiway_registration(
    point_clouds: dict,
    max_correspondence_distance: float = 0.005,
    criteria: o3d.t.pipelines.registration.ICPConvergenceCriteria = default_t_criteria,
):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))

    cams = list(point_clouds.keys())
    ncams = len(cams)
    for i in tqdm(range(ncams)):
        for j in tqdm(range(i + 1, ncams)):
            source = cams[i]
            target = cams[j]

            initial_shift = np.eye(4)
            c0 = np.mean(point_clouds[target].point.positions.numpy(), axis=0)
            c1 = np.mean(point_clouds[source].point.positions.numpy(), axis=0)
            df = c0 - c1
            initial_shift[:3, 3] = df

            _result = pairwise_registration(
                point_clouds[source],
                point_clouds[target],
                max_correspondence_distance=max_correspondence_distance,
                init_transformation=o3d.core.Tensor(initial_shift),
                criteria=criteria,
            )

            if j == i + 1:  # odometry case
                odometry = np.dot(_result["transformation"], odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry))
                )
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        i,
                        j,
                        _result["transformation"],
                        _result["information"],
                        uncertain=False,
                    )
                )
            else:
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        i,
                        j,
                        _result["transformation"],
                        _result["information"],
                        uncertain=True,
                    )
                )

    return pose_graph