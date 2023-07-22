import open3d as o3d
import numpy as np
from tqdm.auto import tqdm
from typing import Union, Optional


default_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
    relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=int(1e3)
)

default_estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint(
)


def pairwise_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    max_correspondence_distance: float = 0.005,
    criteria: o3d.pipelines.registration.ICPConvergenceCriteria = default_criteria,
    init_transformation: np.ndarray = np.identity(4),
    estimation: o3d.pipelines.registration.TransformationEstimation = default_estimation,
    compute_information=True,
):
    
    if init_transformation is None:
        init_transformation = np.eye(4)
        c0 = np.array(target.get_center())
        c1 = np.array(source.get_center())
        df = c0 - c1
        init_transformation[:3, 3] = df

    _result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance,
        init_transformation,
        estimation,
        criteria,
    )

    dct = {
        "fitness": _result.fitness,
        "inlier_rmse": _result.inlier_rmse,
        "transformation": np.array(_result.transformation),
    }

    if compute_information:
        information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, max_correspondence_distance * 1.4, dct["transformation"]
        )
        dct["information"] = np.array(information_icp)
    return dct


default_opt = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
default_opt.max_iteration = 1000
default_opt.max_iteration_lm = 100

def optimize_pose_graph(
    pose_graph: o3d.pipelines.registration.PoseGraph,
    max_correspondence_distance: float = 0.005,
    edge_prune_threshold: float = 0.25,
    preference_loop_closure: float = 0.1,
    reference_node: int = 0,
    criteria: o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria = default_opt
):
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance,
        edge_prune_threshold=edge_prune_threshold,
        preference_loop_closure=preference_loop_closure,
        reference_node=reference_node,
    )
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            default_opt,
            option,
        )
