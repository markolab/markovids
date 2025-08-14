import numpy as np
import warnings
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, diags
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm.auto import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


@dataclass
class BoneConstraints:
    """Define bone connectivity and target lengths"""

    connections: List[Tuple[int, int]]  # Pairs of keypoint indices
    target_lengths: np.ndarray  # Target length for each connection
    length_stds: np.ndarray  # Standard deviation for each length
    keypoint_names: List[str]  # Names for index mapping

    def get_keypoint_index(self, name: str) -> int:
        """Get index for keypoint name"""
        return self.keypoint_names.index(name)

    def get_connection_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get arrays of connection indices for vectorized operations"""
        connections_array = np.array(self.connections)
        return connections_array[:, 0], connections_array[:, 1]


class BoneConstraintOptimizer:
    """
    Fast iterative bone constraint enforcement.
    Processes all frames in parallel using vectorized operations.
    """

    def __init__(self, constraints: BoneConstraints, iterations: int = 5, 
                 correction_rate: float = 0.5, violation_threshold: float = 2.0):
        """
        Args:
            constraints: Bone constraint definitions
            iterations: Number of constraint iterations
            correction_rate: How much to correct per iteration (0-1)
            violation_threshold: Z-score threshold for corrections (default 2.0 = 2 std deviations)
        """
        self.constraints = constraints
        self.iterations = iterations
        self.correction_rate = correction_rate
        self.violation_threshold = violation_threshold

    def compute_bone_lengths(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute lengths of all bones.

        Args:
            positions: (n_frames, n_keypoints, 3)

        Returns:
            (n_frames, n_bones) array of lengths
        """
        idx1, idx2 = self.constraints.get_connection_indices()
        p1 = positions[:, idx1]
        p2 = positions[:, idx2]
        return np.linalg.norm(p2 - p1, axis=2)

    def process_sequence(self, keypoints: np.ndarray, confidences: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process sequence with iterative bone constraints.

        Args:
            keypoints: (n_frames, n_keypoints, 3) array, may contain NaN
            confidences: (n_frames, n_keypoints) confidence scores

        Returns:
            Tuple of (corrected_keypoints, adjusted_confidences)
        """
        n_frames, n_keypoints, _ = keypoints.shape
        corrected = keypoints.copy()

        # Create valid mask
        valid_mask = ~np.isnan(keypoints[:, :, 0])

        # Get connection indices
        idx1, idx2 = self.constraints.get_connection_indices()
        n_bones = len(idx1)

        # Pre-compute target lengths and tolerances
        target_lengths = self.constraints.target_lengths[None, :]  # (1, n_bones)
        length_stds = self.constraints.length_stds[None, :]  # (1, n_bones)

        for iteration in range(self.iterations):
            # Compute all bone vectors and lengths at once
            p1 = corrected[:, idx1, :]  # (n_frames, n_bones, 3)
            p2 = corrected[:, idx2, :]  # (n_frames, n_bones, 3)

            bone_vectors = p2 - p1
            current_lengths = np.linalg.norm(bone_vectors, axis=2)  # (n_frames, n_bones)

            # Check which bones are valid (both endpoints present)
            valid_bones = valid_mask[:, idx1] & valid_mask[:, idx2]  # (n_frames, n_bones)

            # Compute which bones need correction (>violation_threshold std deviations)
            length_errors = current_lengths - target_lengths
            z_scores = np.abs(length_errors) / (length_stds + 1e-6)
            needs_correction = (z_scores > self.violation_threshold) & valid_bones

            # Skip if no corrections needed
            if not needs_correction.any():
                break

            # Vectorized correction for all bones at once
            for bone_idx in range(n_bones):
                # Get frames that need correction for this bone
                frames_to_correct = needs_correction[:, bone_idx]

                if not frames_to_correct.any():
                    continue

                kp1_idx, kp2_idx = idx1[bone_idx], idx2[bone_idx]

                # Get confidence weights for these keypoints
                conf1 = confidences[frames_to_correct, kp1_idx]
                conf2 = confidences[frames_to_correct, kp2_idx]

                # Weight inversely by confidence (move low-confidence points more)
                total_conf = conf1 + conf2 + 1e-8
                weight1 = 1.0 - conf1 / total_conf
                weight2 = 1.0 - conf2 / total_conf

                # Normalize weights so they sum to 1
                weight_sum = weight1 + weight2 + 1e-8
                weight1 = weight1 / weight_sum
                weight2 = weight2 / weight_sum

                # Get current positions for frames needing correction
                pos1 = corrected[frames_to_correct, kp1_idx, :]
                pos2 = corrected[frames_to_correct, kp2_idx, :]

                # Compute weighted midpoint
                midpoint = weight2[:, None] * pos1 + weight1[:, None] * pos2

                # Get direction and current length
                bone_vec = pos2 - pos1
                curr_len = current_lengths[frames_to_correct, bone_idx : bone_idx + 1]
                safe_len = np.maximum(curr_len, 1e-8)

                # Direction unit vector
                direction = bone_vec / safe_len

                # Target length for this bone
                target_len = target_lengths[0, bone_idx]

                # Compute new positions to achieve target length
                half_target = target_len / 2
                new_pos1 = midpoint - direction * (half_target * (1 + weight1[:, None] - weight2[:, None]))
                new_pos2 = midpoint + direction * (half_target * (1 + weight2[:, None] - weight1[:, None]))

                # Apply correction gradually
                alpha = self.correction_rate
                corrected[frames_to_correct, kp1_idx, :] = (1 - alpha) * pos1 + alpha * new_pos1
                corrected[frames_to_correct, kp2_idx, :] = (1 - alpha) * pos2 + alpha * new_pos2

        # Restore NaN for originally invalid keypoints
        corrected[~valid_mask] = np.nan

        # Calculate adjustments for confidence
        movements = np.nansum((corrected - keypoints) ** 2, axis=2) ** 0.5

        # Adjust confidence based on movement
        confidence_penalty = np.exp(-movements / 5.0)  # Decay over 5mm
        adjusted_conf = confidences * confidence_penalty
        adjusted_conf[~valid_mask] = 0.0

        return corrected, adjusted_conf


def estimate_bone_lengths_from_data(
    keypoints: np.ndarray,
    keypoint_names: List[str],
    bone_connections: List[Tuple[str, str]],
    confidences: np.ndarray = None,
    min_samples: int = 100,
    use_robust: bool = True,
) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """
    Estimate bone length statistics from noisy data.

    Args:
        keypoints: (n_frames, n_keypoints, 3) array
        keypoint_names: List of keypoint names corresponding to array indices
        bone_connections: List of (keypoint1_name, keypoint2_name) pairs
        confidences: Optional (n_frames, n_keypoints) confidence array
        min_samples: Minimum samples needed for reliable estimate
        use_robust: If True, use median/MAD; if False, use mean/std

    Returns:
        Dictionary mapping bone pairs to (length, std) tuples
    """
    n_frames, n_keypoints, _ = keypoints.shape

    # Convert bone names to indices
    bone_indices = []
    for bone in bone_connections:
        try:
            idx1 = keypoint_names.index(bone[0])
            idx2 = keypoint_names.index(bone[1])
            bone_indices.append((idx1, idx2, bone))
        except ValueError:
            print(f"Warning: Keypoint {bone[0]} or {bone[1]} not found in keypoint_names")
            continue

    # Collect measurements
    bone_measurements = {bone[2]: [] for bone in bone_indices}

    for frame_idx in range(n_frames):
        frame = keypoints[frame_idx]
        frame_conf = confidences[frame_idx] if confidences is not None else np.ones(n_keypoints)

        for idx1, idx2, bone_names in bone_indices:
            pos1 = frame[idx1]
            pos2 = frame[idx2]

            # Check validity
            if not (np.isnan(pos1).any() or np.isnan(pos2).any()):
                # Check confidence if available
                if confidences is not None:
                    if frame_conf[idx1] < 0.3 or frame_conf[idx2] < 0.3:
                        continue  # Skip low-confidence measurements

                bone_length = np.linalg.norm(pos2 - pos1)

                # Basic sanity check
                if 1.0 < bone_length < 200.0:  # Between 1mm and 200mm
                    bone_measurements[bone_names].append(bone_length)

    # Compute statistics for each bone
    bone_stats = {}

    for bone_names, lengths in bone_measurements.items():
        if len(lengths) < min_samples:
            print(f"Warning: Bone {bone_names[0]}-{bone_names[1]} has only {len(lengths)} samples")
            if len(lengths) < 10:
                continue

        lengths = np.array(lengths)

        if use_robust:
            # Robust statistics: median and MAD
            length_median = np.median(lengths)
            mad = np.median(np.abs(lengths - length_median))

            # Remove outliers
            inlier_mask = np.abs(lengths - length_median) < (3 * mad)
            if np.sum(inlier_mask) > min_samples / 2:
                lengths_clean = lengths[inlier_mask]
                length_final = np.median(lengths_clean)
                std_final = np.median(np.abs(lengths_clean - length_final)) * 1.4826
            else:
                length_final = length_median
                std_final = mad * 1.4826
        else:
            # Traditional mean/std
            length_mean = np.mean(lengths)
            length_std = np.std(lengths)

            # Remove outliers using z-score
            if length_std > 0:
                z_scores = np.abs((lengths - length_mean) / length_std)
                inlier_mask = z_scores < 3

                if np.sum(inlier_mask) > min_samples / 2:
                    lengths_clean = lengths[inlier_mask]
                    length_final = np.mean(lengths_clean)
                    std_final = np.std(lengths_clean)
                else:
                    length_final = length_mean
                    std_final = length_std
            else:
                length_final = length_mean
                std_final = 1.0  # Default std if no variation

        bone_stats[bone_names] = (length_final, std_final)

        n_outliers = len(lengths) - np.sum(inlier_mask) if "inlier_mask" in locals() else 0
        print(
            f"Bone {bone_names[0]:15s} -> {bone_names[1]:15s}: "
            f"{length_final:6.2f} ± {std_final:5.2f} mm "
            f"(n={len(lengths)}, outliers removed: {n_outliers})"
        )

    return bone_stats


def create_bone_constraints_from_data(
    keypoints: np.ndarray,
    keypoint_names: List[str],
    bone_connections: List[Tuple[str, str]],
    confidences: np.ndarray = None,
    method: str = "robust",
) -> BoneConstraints:
    """
    Create bone constraints from data using array format.

    Args:
        keypoints: (n_frames, n_keypoints, 3) array
        keypoint_names: List of all keypoint names
        bone_connections: List of (name1, name2) bone connections
        confidences: Optional (n_frames, n_keypoints) confidence array
        method: 'robust' (median/MAD) or 'mean' (mean/std)

    Returns:
        BoneConstraints object ready for optimization
    """
    # Estimate lengths
    bone_stats = estimate_bone_lengths_from_data(
        keypoints, keypoint_names, bone_connections, confidences=confidences, use_robust=(method == "robust")
    )

    # Convert to arrays
    connections = []
    target_lengths = []
    length_stds = []

    for bone in bone_connections:
        if bone in bone_stats:
            # Convert names to indices
            idx1 = keypoint_names.index(bone[0])
            idx2 = keypoint_names.index(bone[1])
            connections.append((idx1, idx2))

            target_lengths.append(bone_stats[bone][0])
            length_stds.append(bone_stats[bone][1])

    return BoneConstraints(
        connections=connections,
        target_lengths=np.array(target_lengths),
        length_stds=np.array(length_stds),
        keypoint_names=keypoint_names,
    )


class TemporalRegularization:
    """
    Optimize keypoint trajectories with temporal smoothness constraints.
    Uses scipy sparse matrices for efficiency with large sequences.
    """

    def __init__(self, fps=100, lambda_velocity=0.0, lambda_accel=0.01, lambda_jerk=1.0, lambda_snap=10.0):
        """
        Args:
            fps: Frame rate
            lambda_velocity: Weight for velocity penalty (usually 0 for animal tracking)
            lambda_accel: Weight for acceleration penalty (small value)
            lambda_jerk: Weight for jerk (3rd derivative) penalty (main smoothing)
            lambda_snap: Weight for snap (4th derivative) penalty (high-freq noise)
        """
        self.dt = 1.0 / fps
        self.lambda_v = lambda_velocity
        self.lambda_a = lambda_accel
        self.lambda_j = lambda_jerk
        self.lambda_s = lambda_snap

    def build_difference_matrices(self, n_frames: int) -> Dict[str, sparse.csr_matrix]:
        """Build sparse difference matrices for derivatives"""
        # First difference (velocity)
        if n_frames > 1:
            D1 = diags([1, -1], [0, 1], shape=(n_frames - 1, n_frames), format="csr")
            D1 = D1 / self.dt
        else:
            D1 = csr_matrix((0, n_frames))

        # Second difference (acceleration)
        if n_frames > 2:
            D2 = diags([1, -2, 1], [0, 1, 2], shape=(n_frames - 2, n_frames), format="csr")
            D2 = D2 / (self.dt**2)
        else:
            D2 = csr_matrix((0, n_frames))

        # Third difference (jerk)
        if n_frames > 3:
            D3 = diags([-1, 3, -3, 1], [0, 1, 2, 3], shape=(n_frames - 3, n_frames), format="csr")
            D3 = D3 / (self.dt**3)
        else:
            D3 = csr_matrix((0, n_frames))

        # Fourth difference (snap)
        if n_frames > 4:
            D4 = diags([1, -4, 6, -4, 1], [0, 1, 2, 3, 4], shape=(n_frames - 4, n_frames), format="csr")
            D4 = D4 / (self.dt**4)
        else:
            D4 = csr_matrix((0, n_frames))

        return {"D1": D1, "D2": D2, "D3": D3, "D4": D4}

    def optimize_trajectory(
        self,
        observations: np.ndarray,
        confidences: np.ndarray,
        mask: Optional[np.ndarray] = None,
        max_gap_fill: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimize trajectory to minimize measurement error and temporal derivatives.

        Args:
            observations: (n_frames, 3) array of observed positions
            confidences: (n_frames,) array of confidence scores
            mask: (n_frames,) boolean array, True for valid observations
            max_gap_fill: Maximum gap size to fill with interpolation (frames).
                         Larger gaps are preserved as NaN for later imputation.

        Returns:
            Tuple of (optimized_trajectory, propagated_confidences)
        """
        n_frames = len(observations)
        if mask is None:
            mask = ~np.isnan(observations).any(axis=1)

        # Check if we have any valid data
        if not np.any(mask):
            # No valid data, return NaNs
            return observations.copy(), np.zeros(n_frames)

        # Build difference matrices
        D = self.build_difference_matrices(n_frames)

        # Optimize each dimension separately
        optimized = np.zeros_like(observations)

        for dim in range(3):
            obs_dim = observations[:, dim].copy()

            # Find gaps and their sizes
            gaps = self._find_gaps(~np.isnan(obs_dim))

            # Only fill small gaps, preserve large ones
            obs_filled = obs_dim.copy()
            valid_indices = np.where(~np.isnan(obs_dim))[0]

            if len(valid_indices) == 0:
                # No valid data for this dimension
                optimized[:, dim] = np.nan
                continue
            elif len(valid_indices) == 1:
                # Only one valid point - can't interpolate reliably
                optimized[:, dim] = obs_dim.copy()
                continue

            # Fill only small gaps
            for gap_start, gap_end in gaps:
                gap_size = gap_end - gap_start

                if gap_size <= max_gap_fill:
                    # Small gap - fill with linear interpolation
                    if gap_start > 0 and gap_end < n_frames:
                        # Interior gap - interpolate
                        before_idx = gap_start - 1
                        after_idx = gap_end

                        if not np.isnan(obs_dim[before_idx]) and not np.isnan(obs_dim[after_idx]):
                            # Linear interpolation
                            for i in range(gap_start, gap_end):
                                alpha = (i - before_idx) / (after_idx - before_idx)
                                obs_filled[i] = obs_dim[before_idx] * (1 - alpha) + obs_dim[after_idx] * alpha
                    elif gap_start == 0 and gap_end < n_frames:
                        # Gap at beginning - use first valid value
                        first_valid = valid_indices[0]
                        if first_valid < max_gap_fill:
                            obs_filled[:first_valid] = obs_dim[first_valid]
                    elif gap_start > 0 and gap_end == n_frames:
                        # Gap at end - use last valid value
                        last_valid = valid_indices[-1]
                        if n_frames - last_valid - 1 < max_gap_fill:
                            obs_filled[last_valid + 1 :] = obs_dim[last_valid]
                # Large gaps remain as NaN

            # Create mask for filled values (both original valid and small gaps filled)
            filled_mask = ~np.isnan(obs_filled)

            # Build regularization matrix
            L = csr_matrix((n_frames, n_frames))

            if self.lambda_v > 0 and n_frames > 1:
                L = L + self.lambda_v * (D["D1"].T @ D["D1"])
            if self.lambda_a > 0 and n_frames > 2:
                L = L + self.lambda_a * (D["D2"].T @ D["D2"])
            if self.lambda_j > 0 and n_frames > 3:
                L = L + self.lambda_j * (D["D3"].T @ D["D3"])
            if self.lambda_s > 0 and n_frames > 4:
                L = L + self.lambda_s * (D["D4"].T @ D["D4"])

            # Build data term (weighted by confidence)
            # Use higher weight for observed points, lower for interpolated
            weights = np.zeros(n_frames)
            for i in range(n_frames):
                if not np.isnan(obs_dim[i]):
                    # Original data point
                    weights[i] = confidences[i]
                elif not np.isnan(obs_filled[i]):
                    # Interpolated point - use lower weight
                    weights[i] = 0.1 * confidences[i] if i < len(confidences) else 0.1
                else:
                    # Still NaN - zero weight
                    weights[i] = 0.0

            W = diags(weights, format="csr")

            # For solving, replace remaining NaNs with nearest valid value
            obs_for_solve = obs_filled.copy()
            if np.any(np.isnan(obs_for_solve)):
                # Still have NaNs (large gaps) - use nearest neighbor
                for i in range(n_frames):
                    if np.isnan(obs_for_solve[i]):
                        valid_dists = np.abs(valid_indices - i)
                        nearest_idx = valid_indices[np.argmin(valid_dists)]
                        obs_for_solve[i] = obs_dim[nearest_idx]

            # Solve (W + L)x = W * observations
            A = W + L
            b = W @ obs_for_solve

            # Solve using sparse solver
            try:
                solution = spsolve(A, b)

                # Restore NaN for large gaps (they should be imputed later)
                for gap_start, gap_end in gaps:
                    gap_size = gap_end - gap_start
                    if gap_size > max_gap_fill:
                        solution[gap_start:gap_end] = np.nan

                optimized[:, dim] = solution

            except Exception as e:
                # If solve fails, return original with small gaps filled
                warnings.warn(f"Sparse solve failed for dimension {dim}: {e}")
                optimized[:, dim] = obs_filled

        # Propagate confidence through temporal smoothing
        propagated_conf = self._propagate_confidence(confidences, mask, n_frames)

        return optimized, propagated_conf

    def _find_gaps(self, valid_mask: np.ndarray) -> list:
        """
        Find all gaps in a boolean mask.

        Args:
            valid_mask: Boolean array, True for valid data

        Returns:
            List of (start, end) tuples for each gap
        """
        gaps = []
        in_gap = False
        gap_start = 0

        for i, valid in enumerate(valid_mask):
            if not valid and not in_gap:
                gap_start = i
                in_gap = True
            elif valid and in_gap:
                gaps.append((gap_start, i))
                in_gap = False

        # Handle gap at end
        if in_gap:
            gaps.append((gap_start, len(valid_mask)))

        return gaps

    def _propagate_confidence(self, confidences: np.ndarray, mask: np.ndarray, n_frames: int) -> np.ndarray:
        """Propagate confidence based on distance from observations"""
        propagated_conf = np.zeros(n_frames)

        for t in range(n_frames):
            if mask[t]:
                # Original observation - keep confidence
                propagated_conf[t] = confidences[t]
            else:
                # Find nearest valid observation
                valid_indices = np.where(mask)[0]
                if len(valid_indices) > 0:
                    distances = np.abs(valid_indices - t)
                    nearest_idx = valid_indices[np.argmin(distances)]
                    nearest_dist = np.min(distances)

                    # Decay confidence based on distance
                    decay_factor = np.exp(-nearest_dist / 10.0)  # Decay over ~10 frames
                    propagated_conf[t] = confidences[nearest_idx] * decay_factor
                else:
                    propagated_conf[t] = 0.1  # Minimum confidence

        return propagated_conf


class SlidingWindowTemporalRegularization:
    """
    Sliding window approach for very long sequences.
    Processes in overlapping chunks for memory efficiency.
    """

    def __init__(
        self,
        fps=100,
        lambda_velocity=0.0,
        lambda_accel=0.01,
        lambda_jerk=1.0,
        lambda_snap=10.0,
        window_size=2000,
        overlap=200,
    ):
        """
        Args:
            fps: Frame rate
            lambda_velocity: Weight for velocity penalty
            lambda_accel: Weight for acceleration penalty
            lambda_jerk: Weight for jerk penalty
            lambda_snap: Weight for snap penalty
            window_size: Size of processing window
            overlap: Overlap between windows
        """
        self.dt = 1.0 / fps
        self.lambda_v = lambda_velocity
        self.lambda_a = lambda_accel
        self.lambda_j = lambda_jerk
        self.lambda_s = lambda_snap
        self.window_size = window_size
        self.overlap = overlap

        # Create base regularizer for windows
        self.base_regularizer = TemporalRegularization(fps, lambda_velocity, lambda_accel, lambda_jerk, lambda_snap)

    def optimize_trajectory(
        self, observations: np.ndarray, confidences: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process trajectory in sliding windows.

        Args:
            observations: (n_frames, n_dims) array
            confidences: (n_frames,) array
            mask: Optional (n_frames,) boolean array

        Returns:
            Tuple of (optimized_trajectory, propagated_confidences)
        """
        n_frames, n_dims = observations.shape
        if mask is None:
            mask = ~np.isnan(observations).any(axis=1)

        # If small enough, process directly
        if n_frames <= self.window_size:
            return self.base_regularizer.optimize_trajectory(observations, confidences, mask)

        # Process in windows
        optimized = np.zeros_like(observations)
        weights = np.zeros((n_frames, 1))

        stride = self.window_size - self.overlap

        for start in range(0, n_frames, stride):
            end = min(start + self.window_size, n_frames)

            # Process window
            window_result, window_conf = self.base_regularizer.optimize_trajectory(
                observations[start:end], confidences[start:end], mask[start:end]
            )

            # Create blending weights
            window_weight = np.ones((end - start, 1))

            # Taper weights at boundaries for smooth blending
            if start > 0:
                taper_len = min(self.overlap // 2, end - start)
                taper = np.linspace(0, 1, taper_len).reshape(-1, 1)
                window_weight[:taper_len] = taper

            if end < n_frames:
                taper_len = min(self.overlap // 2, end - start)
                taper = np.linspace(1, 0, taper_len).reshape(-1, 1)
                window_weight[-taper_len:] = taper

            # Accumulate weighted results
            optimized[start:end] += window_result * window_weight
            weights[start:end] += window_weight

            if end >= n_frames:
                break

        # Normalize by weights
        optimized = optimized / np.maximum(weights, 1e-8)

        # Propagate confidence
        propagated_conf = self.base_regularizer._propagate_confidence(confidences, mask, n_frames)

        return optimized, propagated_conf



class PoseAligner:
    """
    Handle centering and orientation alignment for pose sequences.
    """
    
    def __init__(self, keypoint_names, exclude_from_center=None, 
                 center_keypoint_weights=None, use_median_centering=False):
        """
        Initialize the aligner.
        
        Parameters:
        -----------
        keypoint_names : list
            List of keypoint names in order
        exclude_from_center : list or None
            List of keypoint names to exclude from centroid calculation
        center_keypoint_weights : dict or None
            Dictionary mapping keypoint names to weights for centroid calculation
        use_median_centering : bool
            If True, use median instead of weighted mean for centering
        """
        self.keypoint_names = keypoint_names
        self.use_median_centering = use_median_centering
        
        # If using median centering, default to excluding only tail and snout
        if use_median_centering:
            default_exclude = ['snout', 'tail_tip', 'tail_middle', 'tail_base']
        else:
            default_exclude = ['snout', 'left_ear', 'right_ear', 'tail_tip', 'tail_middle', 'tail_base']
        
        if exclude_from_center is not None:
            exclude_set = set(exclude_from_center)
        else:
            exclude_set = set(default_exclude)
        
        # Build list of keypoints to use for centering
        self.center_keypoints = [kp for kp in keypoint_names if kp not in exclude_set]
        
        # Set up weights for center keypoints
        if center_keypoint_weights is not None:
            self.center_weights = center_keypoint_weights
        else:
            # Default weights: back keypoints get higher weight
            self.center_weights = {}
            for kp in self.center_keypoints:
                if 'back' in kp.lower():
                    self.center_weights[kp] = 3.0  # Triple weight for back keypoints
                elif 'hip' in kp.lower() or 'shoulder' in kp.lower():
                    self.center_weights[kp] = 2.0  # Double weight for hips/shoulders
                else:
                    self.center_weights[kp] = 1.0  # Normal weight for others
        
        # Get indices for center keypoints
        self.center_indices = []
        for kp in self.center_keypoints:
            if kp in keypoint_names:
                self.center_indices.append(keypoint_names.index(kp))
        
        # Define anterior and posterior keypoints for orientation
        self.anterior_keypoints = ['snout', 'left_ear', 'right_ear', 'left_shoulder', 
                                   'right_shoulder', 'back_top']
        self.posterior_keypoints = ['tail_tip', 'tail_middle', 'tail_base', 'back_bottom',
                                    'left_hip', 'right_hip']
        
        # Get indices for anterior/posterior keypoints
        self.anterior_indices = [keypoint_names.index(kp) for kp in self.anterior_keypoints 
                                if kp in keypoint_names]
        self.posterior_indices = [keypoint_names.index(kp) for kp in self.posterior_keypoints 
                                 if kp in keypoint_names]
        
        # Define nearest neighbor relationships for initialization
        self.neighbor_map = {
            'left_ear': ['right_ear', 'back_top', 'left_shoulder'],
            'right_ear': ['left_ear', 'back_top', 'right_shoulder'],
            'snout': ['back_top', 'left_ear', 'right_ear'],
            'left_shoulder': ['right_shoulder', 'back_middle_upper', 'left_ear', 'back_top'],
            'right_shoulder': ['left_shoulder', 'back_middle_upper', 'right_ear', 'back_top'],
            'left_hip': ['right_hip', 'back_bottom', 'back_middle_lower'],
            'right_hip': ['left_hip', 'back_bottom', 'back_middle_lower'],
            'back_top': ['back_middle_upper', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder'],
            'back_middle_upper': ['back_top', 'back_middle_lower', 'left_shoulder', 'right_shoulder'],
            'back_middle_lower': ['back_middle_upper', 'back_bottom', 'left_hip', 'right_hip'],
            'back_bottom': ['back_middle_lower', 'tail_base', 'left_hip', 'right_hip'],
            'tail_base': ['back_bottom', 'tail_middle', 'left_hip', 'right_hip'],
            'tail_middle': ['tail_base', 'tail_tip', 'back_bottom'],
            'tail_tip': ['tail_middle', 'tail_base'],
        }
    
    def _compute_2d_centroid(self, keypoints):
        """
        Compute 2D centroid using stable body keypoints.
        """
        if self.use_median_centering:
            stable_kps = keypoints[self.center_indices]
            valid_mask = ~np.isnan(stable_kps[:, :2]).any(axis=1)
            
            if valid_mask.any():
                centroid_xy = np.median(stable_kps[valid_mask, :2], axis=0)
            else:
                valid_mask = ~np.isnan(keypoints[:, :2]).any(axis=1)
                if valid_mask.any():
                    centroid_xy = np.median(keypoints[valid_mask, :2], axis=0)
                else:
                    centroid_xy = np.array([0.0, 0.0])
        else:
            stable_kps = keypoints[self.center_indices]
            valid_mask = ~np.isnan(stable_kps[:, :2]).any(axis=1)
            
            if valid_mask.any():
                weights = np.array([self.center_weights[kp] for kp in self.center_keypoints])
                valid_weights = weights[valid_mask]
                valid_weights = valid_weights / valid_weights.sum()
                centroid_xy = np.average(stable_kps[valid_mask, :2], axis=0, weights=valid_weights)
            else:
                valid_mask = ~np.isnan(keypoints[:, :2]).any(axis=1)
                if valid_mask.any():
                    centroid_xy = np.nanmean(keypoints[valid_mask, :2], axis=0)
                else:
                    centroid_xy = np.array([0.0, 0.0])
        
        return np.array([centroid_xy[0], centroid_xy[1], 0])
    
    def _compute_2d_orientation(self, keypoints):
        """
        Compute 2D orientation angle from anterior-posterior axis.
        """
        anterior_kps = keypoints[self.anterior_indices, :2]
        anterior_valid = ~np.isnan(anterior_kps).any(axis=1)
        
        posterior_kps = keypoints[self.posterior_indices, :2]
        posterior_valid = ~np.isnan(posterior_kps).any(axis=1)
        
        if anterior_valid.sum() >= 1 and posterior_valid.sum() >= 1:
            anterior_mean = np.nanmean(anterior_kps[anterior_valid], axis=0)
            posterior_mean = np.nanmean(posterior_kps[posterior_valid], axis=0)
            direction = anterior_mean - posterior_mean
            angle = np.arctan2(direction[1], direction[0])
            return angle
        else:
            # Fallback to spine-based orientation
            spine_indices = [self.keypoint_names.index(kp) for kp in 
                           ['back_bottom', 'back_middle_lower', 'back_middle_upper', 'back_top']
                           if kp in self.keypoint_names]
            
            if len(spine_indices) >= 2:
                spine_kps = keypoints[spine_indices, :2]
                valid_mask = ~np.isnan(spine_kps).any(axis=1)
                
                if valid_mask.sum() >= 2:
                    valid_spine = spine_kps[valid_mask]
                    mean = np.mean(valid_spine, axis=0)
                    centered = valid_spine - mean
                    U, S, Vt = np.linalg.svd(centered)
                    direction = Vt[0]
                    angle = np.arctan2(direction[1], direction[0])
                    return angle
            
            return 0.0
    
    def _rotate_2d(self, keypoints, angle):
        """
        Rotate keypoints in 2D (xy plane).
        """
        cos_a = np.cos(-angle)
        sin_a = np.sin(-angle)
        
        rotated = keypoints.copy()
        rotated[:, 0] = keypoints[:, 0] * cos_a - keypoints[:, 1] * sin_a
        rotated[:, 1] = keypoints[:, 0] * sin_a + keypoints[:, 1] * cos_a
        
        return rotated
    
    def compute_alignment(self, keypoints_sequence, smooth_alignment=True, alignment_window=5):
        """
        Compute centroids and orientation angles.
        
        Returns:
        --------
        centroids : array (n_frames, 3)
        angles : array (n_frames,)
        """
        n_frames = keypoints_sequence.shape[0]
        
        centroids = np.zeros((n_frames, 3))
        angles = np.zeros(n_frames)
        
        for i in range(n_frames):
            frame = keypoints_sequence[i]
            centroids[i] = self._compute_2d_centroid(frame)
            centered = frame - centroids[i]
            angles[i] = self._compute_2d_orientation(centered)
        
        # Smooth alignment parameters if requested
        if smooth_alignment and n_frames > alignment_window:
            from scipy.ndimage import median_filter
            
            for dim in range(2):  # Only smooth x and y
                centroids[:, dim] = median_filter(centroids[:, dim], size=alignment_window)
            
            angles_unwrapped = np.unwrap(angles)
            angles_smooth = median_filter(angles_unwrapped, size=alignment_window)
            angles = np.mod(angles_smooth + np.pi, 2*np.pi) - np.pi
        
        self.centroids_ = centroids
        self.angles_ = angles
        
        return centroids, angles
    
    def transform(self, keypoints_sequence, centroids=None, angles=None):
        """
        Apply centering and orientation to keypoints.
        """
        n_frames = keypoints_sequence.shape[0]
        
        if centroids is None:
            centroids = self.centroids_
        if angles is None:
            angles = self.angles_
        
        transformed = np.zeros_like(keypoints_sequence)
        
        for i in range(n_frames):
            frame = keypoints_sequence[i]
            centered = frame - centroids[i]
            rotated = self._rotate_2d(centered, angles[i])
            transformed[i] = rotated
        
        return transformed
    
    def inverse_transform(self, transformed_keypoints, centroids=None, angles=None):
        """
        Transform keypoints back to original space.
        """
        n_frames = transformed_keypoints.shape[0]
        
        if centroids is None:
            centroids = self.centroids_
        if angles is None:
            angles = self.angles_
        
        original = np.zeros_like(transformed_keypoints)
        
        for i in range(n_frames):
            frame = transformed_keypoints[i]
            unrotated = self._rotate_2d(frame, -angles[i])
            original[i] = unrotated + centroids[i]
        
        return original


class PCAImputer:
    """
    PCA-based imputation for aligned pose data.
    """
    
    def __init__(self, n_components=10, n_iterations=10,
                 use_kmeans_sampling=False, n_clusters=1000, samples_per_cluster=1,
                 transform_z=False):
        """
        Initialize the PCA imputer.
        """
        self.n_components = n_components
        self.n_iterations = n_iterations
        self.use_kmeans_sampling = use_kmeans_sampling
        self.n_clusters = n_clusters
        self.samples_per_cluster = samples_per_cluster
        self.transform_z = transform_z
        self.pca_ = None
        self.n_keypoints_ = None
    
    def _transform_z_coords(self, data):
        """Apply signed sqrt transformation to Z coordinates."""
        n_keypoints = self.n_keypoints_
        n_coords = 3
        
        transformed = data.copy()
        for i in range(n_keypoints):
            z_idx = i * n_coords + 2
            z_vals = data[:, z_idx]
            transformed[:, z_idx] = np.sign(z_vals) * np.sqrt(np.abs(z_vals))
        
        return transformed
    
    def _inverse_transform_z_coords(self, data):
        """Inverse of signed sqrt transformation for Z coordinates."""
        n_keypoints = self.n_keypoints_
        n_coords = 3
        
        original = data.copy()
        for i in range(n_keypoints):
            z_idx = i * n_coords + 2
            z_vals = data[:, z_idx]
            original[:, z_idx] = np.sign(z_vals) * (z_vals ** 2)
        
        return original
    
    def impute(self, aligned_keypoints, aligner=None):
        """
        Impute missing keypoints using iterative PCA.
        
        Returns:
        --------
        imputed : array (n_frames, n_keypoints, 3)
            Imputed keypoints in aligned space
        """
        n_frames, n_keypoints, _ = aligned_keypoints.shape
        self.n_keypoints_ = n_keypoints
        
        flattened = aligned_keypoints.reshape(n_frames, -1)
        missing_mask = np.isnan(flattened)
        imputed = flattened.copy()
        
        # Initialize missing values with nearest neighbors if aligner provided
        if aligner is not None and hasattr(aligner, 'neighbor_map'):
            for frame_idx in range(n_frames):
                frame_data = aligned_keypoints[frame_idx]
                frame_missing = np.isnan(frame_data).any(axis=1)
                
                for kp_idx in range(n_keypoints):
                    if frame_missing[kp_idx]:
                        kp_name = aligner.keypoint_names[kp_idx]
                        if kp_name in aligner.neighbor_map:
                            for neighbor_name in aligner.neighbor_map[kp_name]:
                                if neighbor_name in aligner.keypoint_names:
                                    neighbor_idx = aligner.keypoint_names.index(neighbor_name)
                                    if not frame_missing[neighbor_idx]:
                                        for dim in range(3):
                                            imputed[frame_idx, kp_idx * 3 + dim] = frame_data[neighbor_idx, dim]
                                        break
        
        # Fill remaining NaNs with 0
        imputed[missing_mask] = 0
        
        # K-means sampling if requested
        pca_training_data = None
        if self.use_kmeans_sampling:
            complete_mask = ~missing_mask.any(axis=1)
            complete_frames = flattened[complete_mask]
            
            if len(complete_frames) > self.n_clusters:
                print(f"K-means sampling: clustering {len(complete_frames)} complete frames...")
                kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(complete_frames)
                
                if self.samples_per_cluster == 1:
                    pca_training_data = kmeans.cluster_centers_
                else:
                    sampled_frames = []
                    for cluster_id in range(self.n_clusters):
                        cluster_mask = cluster_labels == cluster_id
                        cluster_frames = complete_frames[cluster_mask]
                        
                        if len(cluster_frames) > 0:
                            n_samples = min(self.samples_per_cluster, len(cluster_frames))
                            sample_indices = np.random.RandomState(42 + cluster_id).choice(
                                len(cluster_frames), n_samples, replace=False
                            )
                            sampled_frames.extend(cluster_frames[sample_indices])
                    
                    pca_training_data = np.array(sampled_frames)
                print(f"Using {len(pca_training_data)} frames for PCA training")
        
        # Apply Z transformation if requested
        if self.transform_z:
            imputed = self._transform_z_coords(imputed)
            if pca_training_data is not None:
                pca_training_data = self._transform_z_coords(pca_training_data)
        
        # Iterative PCA imputation
        for iter_num in range(self.n_iterations):
            if pca_training_data is not None and iter_num == 0:
                self.pca_ = PCA(n_components=min(self.n_components, min(pca_training_data.shape) - 1))
                self.pca_.fit(pca_training_data)
                transformed = self.pca_.transform(imputed)
            else:
                self.pca_ = PCA(n_components=min(self.n_components, min(imputed.shape) - 1))
                transformed = self.pca_.fit_transform(imputed)
            
            reconstructed = self.pca_.inverse_transform(transformed)
            imputed[missing_mask] = reconstructed[missing_mask]
            
            if iter_num == self.n_iterations - 1:
                variance_explained = np.sum(self.pca_.explained_variance_ratio_)
                reconstruction_error = np.mean((imputed[~missing_mask] - reconstructed[~missing_mask])**2)
                print(f"PCA: {self.pca_.n_components_} components explain {variance_explained:.1%} of variance")
                print(f"PCA: Reconstruction MSE: {reconstruction_error:.6f}")
        
        # Inverse Z transformation
        if self.transform_z:
            imputed = self._inverse_transform_z_coords(imputed)
        
        return imputed.reshape(n_frames, n_keypoints, 3)
    
    def transform(self, aligned_keypoints):
        """Transform aligned keypoints to PCA space."""
        if self.pca_ is None:
            raise ValueError("PCA model not fitted yet. Run impute() first.")
        
        n_frames = aligned_keypoints.shape[0]
        flattened = aligned_keypoints.reshape(n_frames, -1)
        
        if self.transform_z:
            flattened = self._transform_z_coords(flattened)
        
        return self.pca_.transform(flattened)
    
    def inverse_transform(self, scores):
        """Transform PCA scores back to aligned keypoint space."""
        if self.pca_ is None:
            raise ValueError("PCA model not fitted yet. Run impute() first.")
        
        reconstructed = self.pca_.inverse_transform(scores)
        
        if self.transform_z:
            reconstructed = self._inverse_transform_z_coords(reconstructed)
        
        return reconstructed.reshape(-1, self.n_keypoints_, 3)


class PoseAwareImputer:
    """
    Complete pose-aware imputation pipeline combining alignment and PCA.
    """
    
    def __init__(self, keypoint_names, **kwargs):
        """
        Initialize the complete imputation pipeline.
        """
        aligner_params = {
            'exclude_from_center': kwargs.get('exclude_from_center', None),
            'center_keypoint_weights': kwargs.get('center_keypoint_weights', None),
            'use_median_centering': kwargs.get('use_median_centering', False)
        }
        
        pca_params = {
            'n_components': kwargs.get('n_components', 10),
            'n_iterations': kwargs.get('n_iterations', 10),
            'use_kmeans_sampling': kwargs.get('use_kmeans_sampling', False),
            'n_clusters': kwargs.get('n_clusters', 1000),
            'samples_per_cluster': kwargs.get('samples_per_cluster', 1),
            'transform_z': kwargs.get('transform_z', False)
        }
        
        self.aligner = PoseAligner(keypoint_names, **aligner_params)
        self.pca_imputer = PCAImputer(**pca_params)
        self.keypoint_names = keypoint_names
    
    def impute(self, keypoints_sequence, smooth_alignment=True, alignment_window=5):
        """
        Complete imputation pipeline: align, impute, and transform back.
        """
        # Compute alignment
        centroids, angles = self.aligner.compute_alignment(
            keypoints_sequence, smooth_alignment, alignment_window
        )
        
        # Transform to aligned space
        aligned = self.aligner.transform(keypoints_sequence, centroids, angles)
        
        # Impute missing values
        imputed_aligned = self.pca_imputer.impute(aligned, self.aligner)
        
        # Transform back to original space
        imputed = self.aligner.inverse_transform(imputed_aligned, centroids, angles)
        
        # Store alignment for downstream use
        self.centroids_ = centroids
        self.angles_ = angles
        
        return imputed
    
    def get_pca_scores(self, keypoints_sequence):
        """Get PCA scores for keypoints."""
        aligned = self.aligner.transform(keypoints_sequence)
        return self.pca_imputer.transform(aligned)
    
    @property
    def pca_components_(self):
        """Get PCA components from the fitted model."""
        if self.pca_imputer.pca_ is None:
            raise ValueError("PCA model not fitted yet. Run impute() first.")
        return self.pca_imputer.pca_.components_
    
    @property
    def pca_explained_variance_ratio_(self):
        """Get explained variance ratio from PCA."""
        if self.pca_imputer.pca_ is None:
            raise ValueError("PCA model not fitted yet. Run impute() first.")
        return self.pca_imputer.pca_.explained_variance_ratio_


def compute_imputation_confidence(was_imputed):
    """
    Compute per-keypoint confidence based on imputation patterns.

    Parameters:
    -----------
    was_imputed : array (n_frames, n_keypoints)
        Boolean mask, True where keypoint was imputed

    Returns:
    --------
    confidence : array (n_frames, n_keypoints)
        Confidence scores between 0 and 1
    """
    # Base confidence: inverse of fraction imputed in each frame
    frame_quality = 1.0 - was_imputed.mean(axis=1, keepdims=True)

    # Per-keypoint penalty: imputed keypoints get lower confidence
    keypoint_penalty = np.where(was_imputed, 0.2, 1.0)

    # Combined confidence: frame quality * keypoint penalty
    confidence = frame_quality * keypoint_penalty

    # Optional: smooth confidence across time to avoid jumps
    from scipy.ndimage import gaussian_filter1d

    confidence = gaussian_filter1d(confidence, sigma=2.5, axis=0)

    return np.clip(confidence, 0.01, 1.0)  # Keep minimum confidence of 0.01


def simple_smooth_imputed(keypoints, was_imputed, medfilt_kernel=int(3), sigma=2.0):
    """
    Simple Gaussian smoothing only on imputed keypoints.

    Parameters:
    -----------
    keypoints : array (n_frames, n_keypoints, 3)
        Keypoint positions
    was_imputed : array (n_frames, n_keypoints)
        Boolean mask, True where keypoint was imputed
    sigma : float
        Gaussian kernel standard deviation in frames

    Returns:
    --------
    smoothed : array
        Smoothed keypoints
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import medfilt

    smoothed = keypoints.copy()

    for kp in tqdm(range(keypoints.shape[1])):
        for dim in range(3):
            # Only smooth if there are imputed points
            if was_imputed[:, kp].any():
                # Smooth the entire trajectory
                if medfilt_kernel is not None:
                    use_data = medfilt(keypoints[:, kp, dim], medfilt_kernel)
                else:
                    use_data = keypoints[:, kp, dim]
                smooth_traj = gaussian_filter1d(use_data, sigma)

                # Only replace imputed points
                mask = was_imputed[:, kp]
                smoothed[mask, kp, dim] = smooth_traj[mask]

    return smoothed


def hampel_filter(keypoints, was_imputed, window_size=5, n_sigmas=3):
    """
    Apply Hampel filter to remove outliers from imputed keypoints.
    
    Parameters:
    -----------
    keypoints : array (n_frames, n_keypoints, 3)
        Keypoint positions
    was_imputed : array (n_frames, n_keypoints)
        Boolean mask, True where keypoint was imputed
    window_size : int
        Size of the sliding window (should be odd)
    n_sigmas : float
        Number of standard deviations for outlier detection
        
    Returns:
    --------
    filtered : array (n_frames, n_keypoints, 3)
        Filtered keypoints (only imputed ones are modified)
    outlier_mask : array (n_frames, n_keypoints)
        Boolean mask indicating detected outliers
    """
    n_frames, n_keypoints, n_dims = keypoints.shape
    filtered = keypoints.copy()
    outlier_mask = np.zeros((n_frames, n_keypoints), dtype=bool)
    
    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1
    half_window = window_size // 2
    
    # MAD to standard deviation conversion factor
    k = 1.4826
    
    for kp in range(n_keypoints):
        for dim in range(n_dims):
            signal = keypoints[:, kp, dim].copy()
            
            if np.all(np.isnan(signal)):
                continue
            
            for t in range(n_frames):
                # Only filter imputed points
                if not was_imputed[t, kp]:
                    continue
                
                # Define window bounds
                start = max(0, t - half_window)
                end = min(n_frames, t + half_window + 1)
                
                # Get window values (excluding NaNs)
                window_vals = signal[start:end]
                valid_vals = window_vals[~np.isnan(window_vals)]
                
                if len(valid_vals) < 3:
                    continue
                
                # Compute median and MAD
                median = np.median(valid_vals)
                mad = np.median(np.abs(valid_vals - median))
                
                # Detect outlier
                if mad > 0:
                    threshold = n_sigmas * k * mad
                    if np.abs(signal[t] - median) > threshold:
                        filtered[t, kp, dim] = median
                        outlier_mask[t, kp] = True
    
    return filtered, outlier_mask


def smooth_all_keypoints(
    kpoints, conf, max_gap_fill=5, use_sliding_window=False, sliding_window_threshold=10000, **kwargs
):
    smoothed_kpoints = np.zeros_like(kpoints)
    smoothed_conf = np.zeros_like(conf)
    n_frames, n_keypoints = kpoints.shape[:2]

    for i in tqdm(range(n_keypoints)):
        use_trajectory = kpoints[:, i, :]
        use_conf = conf[:, i]
        use_conf[np.isnan(use_conf)] = 0
        use_mask = ~np.isnan(use_trajectory[:, 0])

        if use_sliding_window and n_frames > 5000:
            regularizer = SlidingWindowTemporalRegularization(**kwargs)
        else:
            regularizer = TemporalRegularization(**kwargs)

        smoothed_kpoints[:, i, :], smoothed_conf[:, i] = regularizer.optimize_trajectory(
            use_trajectory,
            use_conf,
            use_mask,
            max_gap_fill=max_gap_fill,
        )
    return smoothed_kpoints, smoothed_conf


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