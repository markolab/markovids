import open3d as o3d
import numpy as np
from tqdm.auto import tqdm
from typing import Union, Optional


default_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
    relative_fitness=1e-3, relative_rmse=1e-3, max_iteration=int(100)
)

default_estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint(
)


class DepthVideoPairwiseRegister:
    def __init__(
        self,
        max_correspondence_distance=1.0,
        min_npoints=2000,
        fitness_threshold=0.3,
        current_reference_weight=1.25,
        reference_debounce=0,
        pairwise_criteria=default_criteria,
        pairwise_estimation=default_estimation,
        cleanup_nbs=9,
        cleanup_radius=3.0,
        cleanup_nbs_combined=9,
        cleanup_radius_combined=3.0,
    ):
        self.pairwise_registration_options = {
            "max_correspondence_distance": max_correspondence_distance,
            "criteria": pairwise_criteria,
            "estimation": pairwise_estimation,
        }
        self.min_npoints = min_npoints
        self.fitness_threshold = fitness_threshold
        self.current_reference_weight = current_reference_weight
        self.cleanup_nbs = cleanup_nbs
        self.cleanup_radius = cleanup_radius
        self.cleanup_nbs_combined = cleanup_nbs_combined
        self.cleanup_radius_combined = cleanup_nbs_combined
        self.reference_debounce = reference_debounce

    def get_transforms(self, pcls):
        cams = list(pcls.keys())
        npcls = len(pcls[cams[0]])

        # initialize variables we need to keep around
        self.current_transform = {_cam: None for _cam in cams}
        self.transforms = {
            _cam: np.full((npcls, 4, 4), np.nan, dtype="float") for _cam in cams
        }
        self.fitness = {_cam: np.full((npcls,), np.nan, dtype="float") for _cam in cams}
        self.reference_node = []
        # reference_node = None

        # initialize a transform for each frame, nan = skip transform
        transforms = {
            _cam: np.full((npcls, 4, 4), np.nan, dtype="float") for _cam in cams
        }
        npoints = {_cam: len(pcls[_cam][0].points) for _cam in cams}
        reference_node = max(npoints, key=npoints.get)
        previous_reference_node_proposal = reference_node

        reference_debounce_count = 0
        for _frame in tqdm(range(npcls)):
            # if the target object is missing, simply reference camera against previous transform

            npoints = {_cam: len(pcls[_cam][_frame].points) for _cam in cams}
            if reference_node is not None:
                npoints[reference_node] *= self.current_reference_weight

            reference_node_proposal = max(npoints, key=npoints.get)
            if (reference_node_proposal != reference_node) and (
                reference_node_proposal == previous_reference_node_proposal
            ):
                reference_debounce_count += 1
            else:
                reference_debounce_count = 0
            previous_reference_node_proposal = reference_node_proposal

            if reference_debounce_count > self.reference_debounce:
                reference_node = reference_node_proposal
                reference_debounce_count = 0

            if (len(self.reference_node) > 0) and (
                reference_node != self.reference_node[-1]
            ):
                self.current_transform = {
                    _cam: None for _cam in cams  # reset if our reference switches...
                }

            self.reference_node.append(reference_node)
            target_pcl = pcls[reference_node][_frame]
            self.transforms[reference_node][_frame] = np.eye(4)  # set to identity

            nontarget_cams = [_cam for _cam in cams if _cam is not reference_node]

            for _cam in nontarget_cams:
                use_pcl = pcls[_cam][_frame]
                if len(use_pcl.points) < self.min_npoints:
                    continue

                
                dct = pairwise_registration(
                    use_pcl,
                    target_pcl,
                    init_transformation=self.current_transform[_cam],
                    compute_information=False,
                    **self.pairwise_registration_options
                )
                fitness1 = dct["fitness"]
                transform1 = dct["transformation"]
                
                dct = pairwise_registration(
                    use_pcl,
                    target_pcl,
                    init_transformation=None, # re-initialize if last transform sucked
                    compute_information=False,
                    **self.pairwise_registration_options
                )
                fitness2 = dct["fitness"]
                transform2 = dct["transformation"]
                
                if fitness1 > fitness2:
                    fitness = fitness1
                    transform = transform1
                else:
                    fitness = fitness2
                    transform = transform2
                
                self.fitness[_cam][_frame] = fitness
                if fitness > self.fitness_threshold:
                    self.current_transform[_cam] = transform
                    self.transforms[_cam][_frame] = transform

    def combine_pcls(self, pcls):
        cams = list(pcls.keys())
        npcls = len(pcls[cams[0]])
        pcls_combined = []

        for _frame in tqdm(range(npcls)):
            pcl_combined = o3d.geometry.PointCloud()
            for _cam in cams:
                use_transform = self.transforms[_cam][_frame]
                if np.isnan(use_transform).any():
                    continue
                use_pcl = copy.deepcopy(pcls[_cam][_frame]).transform(use_transform)
                if self.cleanup_nbs is not None:
                    use_pcl, ind = use_pcl.remove_radius_outlier(
                        self.cleanup_nbs, self.cleanup_radius
                    )
                    # pcl_combined += use_pcl
                pcl_combined += use_pcl

            if len(pcl_combined.points) < 10:
                continue

            if self.cleanup_nbs_combined is not None:
                pcl_combined, ind = pcl_combined.remove_radius_outlier(
                    self.cleanup_nbs_combined, self.cleanup_radius_combined
                )
            pcls_combined.append(pcl_combined)

        return pcls_combined


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
