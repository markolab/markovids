import open3d as o3d
import numpy as np
from tqdm.auto import tqdm
from typing import Union


default_criteria = (
    o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=int(1e3)
    )
)

default_t_criteria = (
    o3d.t.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=int(1e3)
    )
)


def pairwise_registration(
    source: Union[o3d.t.geometry.PointCloud, o3d.geometry.PointCloud],
    target: Union[o3d.t.geometry.PointCloud, o3d.geometry.PointCloud],
    max_correspondence_distance: float = 0.005,
    criteria: Union[
        o3d.t.pipelines.registration.ICPConvergenceCriteria,
        o3d.pipelines.registration.ICPConvergenceCriteria,
    ] = default_t_criteria,
    init_transformation: Union[o3d.core.Tensor, np.ndarray] = o3d.core.Tensor(np.identity(4)),
):
    kernel = o3d.t.pipelines.registration.robust_kernel.RobustKernel(
        o3d.t.pipelines.registration.robust_kernel.RobustKernelMethod.TukeyLoss,
        2.0,
    )
    _result = o3d.t.pipelines.registration.icp(
        source,
        target,
        max_correspondence_distance,
        init_transformation,
        o3d.t.pipelines.registration.TransformationEstimationPointToPlane(kernel),
        criteria,
    )

    dct = {
        "fitness": _result.fitness,
        "inlier_rmse": _result.inlier_rmse,
        "transformation": _result.transformation.numpy(),
    }
    information_icp = o3d.t.pipelines.registration.get_information_matrix(
        source, target, max_correspondence_distance * 1.4, dct["transformation"]
    )
    dct["information"] = information_icp.numpy()
    return dct


def icp_multiway_registration(
    point_clouds: dict,
    max_correspondence_distance: float = 0.005,
    criteria: Union[
        o3d.t.pipelines.registration.ICPConvergenceCriteria,
        o3d.pipelines.registration.ICPConvergenceCriteria,
    ] = default_t_criteria,
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


def optimize_pose_graph(
          pose_graph: o3d.pipelines.registration.PoseGraph,
          max_correspondence_distance: float=.005,
          edge_prune_threshold: float=.25,
          preference_loop_closure: float=.1,
          reference_node: int=0,
):
    
	option = o3d.pipelines.registration.GlobalOptimizationOption(
		max_correspondence_distance=max_correspondence_distance,
		edge_prune_threshold=edge_prune_threshold,
		preference_loop_closure=preference_loop_closure,
		reference_node=reference_node)
	with o3d.utility.VerbosityContextManager(
			o3d.utility.VerbosityLevel.Debug) as cm:
		o3d.pipelines.registration.global_optimization(
			pose_graph,
			o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
			o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
			option)
