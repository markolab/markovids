import open3d as o3d
import numpy as np
import copy
import pandas as pd
from tqdm.auto import tqdm
from typing import Union, Optional
from scipy import signal


default_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
    relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=int(100)
)

# NO SCALING
default_estimation_p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint(
    with_scaling=False
)
default_estimation_p2pl = o3d.pipelines.registration.TransformationEstimationPointToPlane(
    # with_scaling=False
)
default_estimation_general = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(
    epsilon=1e-3
)


# TODO need rules to prevent body parts from getting clipped due to
# sticking to the same camera...are we also unintentionally adding 0???
class DepthVideoPairwiseRegister:
    def __init__(
        self,
        max_correspondence_distance=1.0,
        min_npoints=2000,
        fitness_threshold=0.3,
        current_reference_weight=1.0,
        reference_debounce=0,
        reference_min_npoints=2000,
        reference_future_len=75,
        reference_history_len=25,
        reference_medfilt=21,
        reference_min_fraction=0.5,
        registration_type="p2p",
        # pairwise_criteria=default_criteria,
        # pairwise_estimation=default_estimation,
        cleanup_nbs=21,
        cleanup_radius=3.0,
        cleanup_nbs_combined=21,
        cleanup_radius_combined=3.0,
        z_shift=True,
        nsig=2,
    ):
        if registration_type.lower() == "p2p":
            pairwise_criteria = default_criteria
            pairwise_estimation = default_estimation_p2p
        elif registration_type.lower() == "p2pl":
            pairwise_criteria = default_criteria
            pairwise_estimation = default_estimation_p2pl
        elif registration_type.lower() == "generalized":
            pairwise_criteria = default_criteria
            pairwise_estimation = default_estimation_general
        else:
            raise RuntimeError(f"Did not understand registration type {registration_type}")

        self.pairwise_registration_options = {
            "max_correspondence_distance": max_correspondence_distance,
            "criteria": pairwise_criteria,
            "estimation": pairwise_estimation,
            "z_shift": z_shift,
        }
        self.min_npoints = min_npoints
        self.fitness_threshold = fitness_threshold
        self.current_reference_weight = current_reference_weight
        self.cleanup_nbs = cleanup_nbs
        self.cleanup_radius = cleanup_radius
        self.cleanup_nbs_combined = cleanup_nbs_combined
        self.cleanup_radius_combined = cleanup_radius_combined
        self.reference_debounce = reference_debounce
        self.reference_history = reference_history_len
        self.reference_future = reference_future_len
        self.reference_medfilt = reference_medfilt
        self.reference_min_npoints = reference_min_npoints
        self.reference_min_fraction = reference_min_fraction
        self.weights = None
        self.nsig = nsig
        # self.z_shift = z_shift

    def get_reference_node_weights(self, pcls):
        cams = list(pcls.keys())
        npoints = {_cam: np.array([len(_.points) for _ in pcls[_cam]]) for _cam in cams}
        # npoints_arr_raw = np.array(list(npoints.values())).T
        # boxcar_filter_history = np.ones((self.reference_history,)) / self.reference_history
        # boxcar_filter_future = (np.ones((self.reference_future,)) / self.reference_future)
        # npoints_mask = signal.medfilt((npoints_arr_raw > self.reference_min_npoints).astype("uint8"), (self.reference_medfilt,1))
        npoints_pd = {k: pd.Series(v) for k, v in npoints.items()}
        npoints_mu = {
            k: (
                v.rolling(self.reference_history, 1).mean().to_numpy()
                + v[::-1].rolling(self.reference_future, 1).mean().to_numpy()[::-1]
            )
            / 2.0
            for k, v in npoints_pd.items()
        }
        npoints_sig = {
            k: (
                v.rolling(self.reference_history, 1).std().to_numpy()
                + v[::-1].rolling(self.reference_future, 1).std().to_numpy()[::-1]
            )
            / 2.0
            for k, v in npoints_pd.items()
        }
        self.weights_mu = npoints_mu
        self.weights_plus_ci = {k: v + self.nsig * npoints_sig[k] for k, v in npoints_mu.items()}
        self.weights_minus_ci = {k: v - self.nsig * npoints_sig[k] for k, v in npoints_mu.items()}

        # filter the array with a causal boxcar filter to capture history
        # filter the reversed array with the same filter to capture future
        # mask to ensure we only take the reference if it has the sufficient number of points
        # weight_matrix = (signal.lfilter(boxcar_filter_history, 1, npoints_arr_raw, axis=0)
        #                + signal.lfilter(boxcar_filter_future, 1, npoints_arr_raw[::-1], axis=0)[::-1]) * npoints_mask
        # self.weights = {_cam: weight_matrix[:,i] for i, _cam in enumerate(cams)}
        self.npoints = npoints

    def get_transforms_pairwise(self, pcls, progress_bar=True):
        if self.weights is None:
            self.get_reference_node_weights(pcls)

        cams = list(pcls.keys())
        npcls = len(pcls[cams[0]])

        # initialize variables we need to keep around
        self.current_transform = {_cam: None for _cam in cams}
        self.transforms = {_cam: np.full((npcls, 4, 4), np.nan, dtype="float") for _cam in cams}
        self.fitness = {_cam: np.full((npcls,), np.nan, dtype="float") for _cam in cams}
        self.reference_node = []
        # reference_node = None

        # initialize a transform for each frame, nan = skip transform
        transforms = {_cam: np.full((npcls, 4, 4), np.nan, dtype="float") for _cam in cams}
        init_weights = {_cam: self.weights_mu[_cam][0] for _cam in cams}
        reference_node = max(init_weights, key=init_weights.get)
        previous_reference_node_proposal = reference_node

        reference_debounce_count = 0
        for _frame in tqdm(
            range(npcls), disable=not progress_bar, desc="Estimating transformations"
        ):
            # if the target object is missing, simply reference camera against previous transform

            # npoints = {_cam: len(pcls[_cam][_frame].points) for _cam in cams}
            npoints = {_cam: self.npoints[_cam][_frame] for _cam in cams}
            if _frame > 0:
                npoints_retain = {
                    _cam: self.npoints[_cam][_frame] / (self.npoints[_cam][_frame - 1] + 1e-3)
                    for _cam in cams
                }
            else:
                npoints_retain = {_cam: 1 for _cam in cams}

            weights_diff = {
                _cam: self.weights_minus_ci[_cam][_frame]
                - self.weights_plus_ci[reference_node][_frame]
                for _cam in cams
            }
            max_diff = max(weights_diff.values())
            # use smoothed weights to see if we cross threshold...
            if (
                (max_diff <= 0)
                and (npoints[reference_node] >= self.reference_min_npoints)
                and (npoints_retain[reference_node] >= self.reference_min_fraction)
            ):
                reference_node_proposal = reference_node
            else:
                weights_diff[reference_node] = -np.inf  # exclude ref.
                reference_node_proposal = max(weights_diff, key=npoints.get)

            # weights_minus_ci = {_cam: self.weights_minus_ci[_cam][_frame] for _cam in cams}
            # reference_node_proposal = max(weights, key=weights.get)

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

            if (len(self.reference_node) > 0) and (reference_node != self.reference_node[-1]):
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
                    **self.pairwise_registration_options,
                )
                fitness1 = dct["fitness"]
                transform1 = dct["transformation"]

                dct = pairwise_registration(
                    use_pcl,
                    target_pcl,
                    init_transformation=None,  # re-initialize if last transform sucked
                    compute_information=False,
                    **self.pairwise_registration_options,
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

    def get_transforms_multiway(self, pcls, progress_bar=True):
        if self.weights is None:
            self.get_reference_node_weights(pcls)

        cams = list(pcls.keys())
        npcls = len(pcls[cams[0]])

        # initialize variables we need to keep around
        self.current_transform = {_cam: None for _cam in cams}
        self.transforms = {_cam: np.full((npcls, 4, 4), np.nan, dtype="float") for _cam in cams}
        self.fitness = {_cam: np.full((npcls,), np.nan, dtype="float") for _cam in cams}
        self.reference_node = []
        # reference_node = None

        # initialize a transform for each frame, nan = skip transform
        transforms = {_cam: np.full((npcls, 4, 4), np.nan, dtype="float") for _cam in cams}
        init_weights = {_cam: self.weights_mu[_cam][0] for _cam in cams}
        reference_node = max(init_weights, key=init_weights.get)
        previous_reference_node_proposal = reference_node

        reference_debounce_count = 0
        for _frame in tqdm(
            range(npcls), disable=not progress_bar, desc="Estimating transformations"
        ):
            # if the target object is missing, simply reference camera against previous transform

            # npoints = {_cam: len(pcls[_cam][_frame].points) for _cam in cams}
            npoints = {_cam: self.npoints[_cam][_frame] for _cam in cams}
            if _frame > 0:
                npoints_retain = {
                    _cam: self.npoints[_cam][_frame] / (self.npoints[_cam][_frame - 1] + 1e-3)
                    for _cam in cams
                }
            else:
                npoints_retain = {_cam: 1 for _cam in cams}

            weights_diff = {
                _cam: self.weights_minus_ci[_cam][_frame]
                - self.weights_plus_ci[reference_node][_frame]
                for _cam in cams
            }
            max_diff = max(weights_diff.values())
            # use smoothed weights to see if we cross threshold...
            if (
                (max_diff <= 0)
                and (npoints[reference_node] >= self.reference_min_npoints)
                and (npoints_retain[reference_node] >= self.reference_min_fraction)
            ):
                reference_node_proposal = reference_node
            else:
                weights_diff[reference_node] = -np.inf  # exclude ref.
                reference_node_proposal = max(weights_diff, key=npoints.get)

            # weights_minus_ci = {_cam: self.weights_minus_ci[_cam][_frame] for _cam in cams}
            # reference_node_proposal = max(weights, key=weights.get)

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

            if (len(self.reference_node) > 0) and (reference_node != self.reference_node[-1]):
                self.current_transform = {
                    _cam: None for _cam in cams  # reset if our reference switches...
                }

            self.reference_node.append(reference_node)
            target_pcl = pcls[reference_node][_frame]
            self.transforms[reference_node][_frame] = np.eye(4)  # set to identity

            nontarget_cams = [_cam for _cam in cams if _cam is not reference_node]

            # consider edges to target odometry, others are loop closures...
            # alternative: FORM A LOOP starting with reference and ending with reference...
            #              would require estimating overlap for a large bank of cameras...not a bad idea
            #              since it should be much more robust than current solution...
            #              simply add up segmented pixels on each camera simultaneously, use to form
            #              a loop from reference node and back
            pose_graph = o3d.pipelines.registration.PoseGraph()
            odometry = np.identity(4)
            # reference is node 0
            pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
            pose_graph_cams = [reference_node]
            target_id = 0
            offset1 = 1
            
            for _cam_id, _cam1 in enumerate(nontarget_cams):
                use_pcl = pcls[_cam1][_frame]

                odometry = np.identity(4)
                if len(use_pcl.points) < self.min_npoints:
                    # print("skip")
                    continue

                dct1 = pairwise_registration(
                    use_pcl,
                    target_pcl,
                    init_transformation=self.current_transform[_cam1],
                    compute_information=True,
                    **self.pairwise_registration_options,
                )

                dct2 = pairwise_registration(
                    use_pcl,
                    target_pcl,
                    init_transformation=None,  # re-initialize if last transform sucked
                    compute_information=True,
                    **self.pairwise_registration_options,
                )

                if dct1["fitness"] > dct2["fitness"]:
                    dct = dct1
                else:
                    dct = dct2

                fitness = dct["fitness"]
                transform = dct["transformation"]
                information = dct["information"]

                self.fitness[_cam1][_frame] = fitness
                odometry = np.matmul(transform, odometry)

                # each new node is now offset + 1
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(odometry)
                )

                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        offset1, target_id, transform, information, uncertain=False
                    )
                )
                # so we can recover cam name post-optimization
                pose_graph_cams.append(_cam1)
                offset2 = 1
                for _cam2 in nontarget_cams[_cam_id + 1:]:
                    use_pcl2 = pcls[_cam2][_frame]
                    
                    # now target is use_pcl1
                    if len(use_pcl2.points) < self.min_npoints:
                        # print("skip2")
                        continue 

                    # loop closure
                    dct = pairwise_registration(
                        use_pcl2,
                        use_pcl,
                        init_transformation=None,
                        compute_information=True,
                        **self.pairwise_registration_options,
                    )
                    
                    fitness = dct["fitness"]
                    transform = dct["transformation"]
                    information = dct["information"] 

                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            offset1 + offset2, offset1, transform, information, uncertain=True
                        )
                    )

                    offset2 += 1 
                offset1 += 1

            # need effective backup in case of failure (1 camera only, e.g.)...
            optimize_pose_graph(pose_graph)
            for i, _cam in enumerate(pose_graph_cams):
                # any fitness checks??
                self.current_transform[_cam] = pose_graph.nodes[i].pose
                self.transforms[_cam][_frame] = pose_graph.nodes[i].pose

                # if fitness > self.fitness_threshold:
                #     self.current_transform[_cam] = transform
                #     self.transforms[_cam][_frame] = transform

    # LOOK INTO VOLUME INTEGRATE, NEED DEPTH IMGS + INTRINSICS + RELATIVE POSES...
    def combine_pcls(self, pcls, progress_bar=True):
        cams = list(pcls.keys())
        npcls = len(pcls[cams[0]])
        pcls_combined = []

        for _frame in tqdm(range(npcls), disable=not progress_bar, desc="Stitching PCLs"):
            pcl_combined = o3d.geometry.PointCloud()
            for _cam in cams:
                use_transform = self.transforms[_cam][_frame]
                if np.isnan(use_transform).any():
                    continue
                # I don't think this meaningfully copies the data...
                # use_pcl = copy.deepcopy(pcls[_cam][_frame]).transform(use_transform)
                use_pcl = pcls[_cam][_frame].transform(use_transform)

                if self.cleanup_nbs is not None:
                    cl, ind = use_pcl.remove_radius_outlier(self.cleanup_nbs, self.cleanup_radius)
                    use_pcl = use_pcl.select_by_index(ind)
                    # pcl_combined += use_pcl
                pcl_combined += use_pcl

            if len(pcl_combined.points) < 10:
                pcls_combined.append(pcl_combined)
            elif self.cleanup_nbs_combined is not None:
                cl, ind = pcl_combined.remove_radius_outlier(
                    self.cleanup_nbs_combined, self.cleanup_radius_combined
                )
                pcl_combined = pcl_combined.select_by_index(ind)
                pcls_combined.append(pcl_combined)
            else:
                pcls_combined.append(pcl_combined)

        return pcls_combined


def pairwise_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    max_correspondence_distance: float = 0.005,
    criteria: o3d.pipelines.registration.ICPConvergenceCriteria = default_criteria,
    init_transformation: np.ndarray = np.identity(4),
    estimation: o3d.pipelines.registration.TransformationEstimation = default_estimation_p2p,
    compute_information=True,
    z_shift=True,
):
    if init_transformation is None:
        init_transformation = np.eye(4)
        c0 = np.array(target.get_center())
        c1 = np.array(source.get_center())
        df = c0 - c1
        init_transformation[:3, 3] = df

    if type(estimation).__name__ in [
        "TransformationEstimationPointToPoint",
        "TransformationEstimationPointToPlane",
    ]:
        _result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            max_correspondence_distance,
            init_transformation,
            estimation,
            criteria,
        )
    elif type(estimation).__name__ == "TransformationEstimationForGeneralizedICP":
        _result = o3d.pipelines.registration.registration_generalized_icp(
            source,
            target,
            max_correspondence_distance,
            init_transformation,
            estimation,
            criteria,
        )
    else:
        RuntimeError(f"Did not understand registration type: {estimation.__name__}")

    transform = np.array(_result.transformation)
    if not z_shift:
        transform[2, 3] = 0

    dct = {
        "fitness": _result.fitness,
        "inlier_rmse": _result.inlier_rmse,
        "transformation": transform,
    }

    if compute_information:
        information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, max_correspondence_distance * 1.4, dct["transformation"]
        )
        dct["information"] = np.array(information_icp)
    return dct


default_opt = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
default_opt.max_iteration = 50
default_opt.max_iteration_lm = 50
default_opt.min_right_term = 1e-20
default_opt.min_relative_increment = 1e-20
default_opt.min_relative_residual_increment = 1e-20


def optimize_pose_graph(
    pose_graph: o3d.pipelines.registration.PoseGraph,
    max_correspondence_distance: float = 1.0,
    edge_prune_threshold: float = 0.25,
    preference_loop_closure: float = 0.1,
    reference_node: int = 0,
    criteria: o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria = default_opt,
):
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance,
        edge_prune_threshold=edge_prune_threshold,
        preference_loop_closure=preference_loop_closure,
        reference_node=reference_node,
    )
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            default_opt,
            option,
        )


def correct_breakpoints(
    combined_pcl,
    reference_nodes,
    z_shift=False,
    # criteria=default_criteria,
    # estimation=default_estimation,
    # max_correspondence_distance=1.0,
):
    breakpoints = []
    transforms = []

    # CHECK IDX
    for i, (_ref1, _ref2) in enumerate(zip(reference_nodes[:-1], reference_nodes[1:])):
        if _ref1 != _ref2:
            breakpoints.append(i + 1)

    frame_groups = []
    for _bpoint in breakpoints:
        cur_node = reference_nodes[_bpoint]
        next_break = len(reference_nodes[_bpoint:])
        for i, _ref_node in enumerate(reference_nodes[_bpoint:]):
            if _ref_node != cur_node:
                next_break = i
                break

        # extrapolate new positions
        # for now linear is probably fine...
        _tmp_transforms = []
        for j in range(0, 1):
            for k in range(1, 2):
                use_pcl = combined_pcl[_bpoint + j]
                target_pcl = combined_pcl[_bpoint - k]

                init_transformation = np.eye(4)
                c0 = np.array(np.median(target_pcl.points, axis=0))
                c1 = np.array(np.median(use_pcl.points, axis=0))
                df = c0 - c1
                if not z_shift:
                    df[2] = 0  # null out z shift if we don't want it...
                init_transformation[:3, 3] = df

                # potentially downweight any z-shifts, those should be 0...
                _tmp_transforms.append(init_transformation)

                # do some time filtering, check n+1, n+2, n+3 relative to n and n-1 after breakpoint...
                # dct = pcl.registration.pairwise_registration(
                #     use_pcl,
                #     target_pcl,
                #     init_transformation=init_transformation,
                #     compute_information=False,
                #     max_correspondence_distance=max_correspondence_distance,
                #     criteria=criteria,
                #     estimation=estimation,
                # )
                # _tmp_fitnesses.append(dct["fitness"])
                # _tmp_transforms.append(dct["transformation"])

        use_transform = np.median(_tmp_transforms, axis=0)
        _frame_group = range(_bpoint, _bpoint + next_break)
        frame_groups.append(_frame_group)
        transforms.append(use_transform)
        for _frame in _frame_group:
            combined_pcl[_frame] = combined_pcl[_frame].transform(use_transform)

    return breakpoints, frame_groups, transforms


def correct_breakpoints_extrapolate(
    combined_pcl,
    reference_nodes,
    extrapolate_history=5,
    poly_deg=1,
    z_shift=False,
):
    breakpoints = []
    transforms = []

    # CHECK IDX
    for i, (_ref1, _ref2) in enumerate(zip(reference_nodes[:-1], reference_nodes[1:])):
        if _ref1 != _ref2:
            breakpoints.append(i + 1)

    frame_groups = []
    for _bpoint in breakpoints:
        cur_node = reference_nodes[_bpoint]
        next_break = len(reference_nodes[_bpoint:])
        for i, _ref_node in enumerate(reference_nodes[_bpoint:]):
            if _ref_node != cur_node:
                next_break = i
                break

        # extrapolate new positions
        # for now linear is probably fine...
        use_pcl = combined_pcl[_bpoint]
        centroids = []
        for j in range(extrapolate_history, 0, -1):
            target_pcl = combined_pcl[_bpoint - j]
            centroids.append(np.array(np.median(target_pcl.points, axis=0)))

        centroid_array = np.array(centroids)  # should be t x 3 (x, y, z)
        idx_array = np.arange(extrapolate_history)
        # interpolate target position
        # new_point = idx_array[-1] + 1
        # c0 = np.zeros((3,), dtype="float")
        # for _axis in range(centroid_array.shape[1]):
        # p = np.polynomial.Polynomial.fit(idx_array, centroid_array[:,_axis], poly_deg)
        # c0[_axis] = p(new_point)

        # take median vel and use to project next point...
        df = np.median(np.diff(centroid_array, axis=0), axis=0)
        c0 = centroid_array[-1] + df

        # c0 is new target built through extrapolation, find diff with c1, position of current pcl
        use_transform = np.eye(4)
        c1 = np.array(np.median(use_pcl.points, axis=0))
        df = c0 - c1
        if not z_shift:
            df[2] = 0
        use_transform[:3, 3] = df

        # potentially downweight any z-shifts, those should be 0...
        _frame_group = range(_bpoint, _bpoint + next_break)
        frame_groups.append(_frame_group)
        transforms.append(use_transform)
        for _frame in _frame_group:
            combined_pcl[_frame] = combined_pcl[_frame].transform(use_transform)

    return breakpoints, frame_groups, transforms
