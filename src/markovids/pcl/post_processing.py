from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import markovids.pcl.kpoints as kpoints

from markovids.pcl.kpoints import BoneConstraints

from scipy.signal import savgol_filter

def apply_final_smoothing(keypoints, keypoint_names, window_length=5, polyorder=3, fps=100):
    """Light final smoothing pass with Savitzky-Golay filter"""
    smoothed = keypoints.copy()

    for kp_idx, kp_name in enumerate(keypoint_names):
        # Adaptive parameters based on keypoint
        if "tail_tip" in kp_name or "snout" in kp_name:
            _window_length = window_length   # ~50ms at 100fps
            _polyorder = polyorder - 1
        else:
            _window_length = window_length + 2  # ~70ms at 100fps
            _polyorder = polyorder

        for dim in range(3):
            trajectory = keypoints[:, kp_idx, dim]
            if not np.all(np.isnan(trajectory)):
                # Only smooth non-NaN portions
                smoothed[:, kp_idx, dim] = savgol_filter(
                    trajectory, _window_length, _polyorder, mode="nearest"  # Good edge handling
                )

    return smoothed


@dataclass
class PostProcessingConfig:
    """Configuration for post-processing parameters."""
    temporal_regularization: Dict[str, Any]
    bone_length_regularization: Dict[str, Any]
    align: Dict[str, Any]
    align_compute: Dict[str, Any]
    pca: Dict[str, Any]
    post_align_hampel: Dict[str, Any]
    post_align_imputed_smoothing: Dict[str, Any]
    post_align_sgolay: Dict[str, Any]


@dataclass
class ProcessingFlags:
    """Boolean flags for enabling/disabling processing steps."""
    constrain_bones: bool = True
    impute_pca: bool = True
    regularize_temporal: bool = True


class KeypointPostProcessor:
    """Handles post-processing of keypoint data with various smoothing and constraint operations."""
    
    def __init__(self, kpoints_metadata: Dict[str, Any], skeleton: Any):
        self.kpoints_metadata = kpoints_metadata
        self.skeleton = skeleton
    
    def process(
        self,
        merged_data: np.ndarray,
        merged_conf: np.ndarray,
        included_keypoints: List[str],
        config: PostProcessingConfig,
        flags: ProcessingFlags = ProcessingFlags()
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main post-processing pipeline for keypoint data.
        
        Args:
            merged_data: Raw keypoint data
            merged_conf: Confidence scores for keypoints
            included_keypoints: List of keypoint names to include in processing
            config: Post-processing configuration parameters
            flags: Boolean flags to enable/disable processing steps
            
        Returns:
            Tuple of (processed_keypoints, processed_confidence)
        """
        # Extract relevant keypoints and confidence
        keypoints, confidence = self._extract_keypoints(
            merged_data, merged_conf, included_keypoints
        )
        
        # Apply temporal regularization
        if flags.regularize_temporal:
            keypoints, confidence = self._apply_temporal_smoothing(
                keypoints, confidence, config.temporal_regularization
            )
        
        # Apply bone constraints
        bone_constraints = None
        if flags.constrain_bones:
            keypoints, confidence, bone_constraints = self._apply_bone_constraints(
                keypoints, confidence, included_keypoints, config
            )
        
        # Apply PCA imputation
        if flags.impute_pca:
            keypoints, confidence = self._apply_pca_imputation(
                keypoints, confidence, included_keypoints, config
            )
        
        # Final bone constraints (if enabled)
        if flags.constrain_bones:
            keypoints, confidence = self._apply_final_bone_constraints(
                keypoints, confidence, config, bone_constraints
            )
        
        # Final smoothing
        keypoints = self._apply_final_smoothing(
            keypoints, included_keypoints, config.post_align_sgolay
        )
        
        return keypoints, confidence
    
    def _extract_keypoints(
        self, 
        merged_data: np.ndarray, 
        merged_conf: np.ndarray, 
        included_keypoints: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract specified keypoints and their confidence scores."""
        keypoint_indices = [
            self.kpoints_metadata["node_names"].index(keypoint_name)
            for keypoint_name in included_keypoints
        ]
        
        extracted_keypoints = merged_data[:, keypoint_indices, :].copy()
        extracted_confidence = np.nanmax(merged_conf[:, keypoint_indices], axis=-1)
        
        return extracted_keypoints, extracted_confidence
    
    def _apply_temporal_smoothing(
        self, 
        keypoints: np.ndarray, 
        confidence: np.ndarray, 
        smoothing_params: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply temporal regularization to smooth keypoints over time."""
        return kpoints.smooth_all_keypoints(
            keypoints, confidence, **smoothing_params
        )
    
    def _apply_bone_constraints(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray,
        included_keypoints: List[str],
        config: PostProcessingConfig
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply bone length constraints to maintain anatomical consistency."""
        bone_constraints = kpoints.create_bone_constraints_from_data(
            keypoints, included_keypoints, self.skeleton, confidence
        )
        
        optimizer = kpoints.BoneConstraintOptimizer(
            bone_constraints, **config.bone_length_regularization
        )
        
        constrained_keypoints, constrained_confidence = optimizer.process_sequence(
            keypoints, confidence
        )
        
        # Combine confidence scores
        combined_confidence = self._combine_confidence_scores(
            confidence, constrained_confidence
        )
        
        return constrained_keypoints, combined_confidence, bone_constraints
    
    def _apply_pca_imputation(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray,
        included_keypoints: List[str],
        config: PostProcessingConfig
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply PCA-based imputation to fill missing keypoints."""
        aligner = kpoints.PoseAligner(
            keypoint_names=included_keypoints, **config.align
        )
        pca_imputer = kpoints.PCAImputer(**config.pca)
        
        # Align keypoints
        centroids, angles = aligner.compute_alignment(
            keypoints, **config.align_compute
        )
        aligned_keypoints = aligner.transform(keypoints, centroids, angles)
        
        # Impute missing values
        imputed_aligned_keypoints = pca_imputer.impute(aligned_keypoints, aligner)
        imputed_keypoints = aligner.inverse_transform(
            imputed_aligned_keypoints, centroids, angles
        )
        
        # Calculate imputation confidence
        was_imputed = np.isnan(keypoints).any(axis=-1)
        imputation_confidence = kpoints.compute_imputation_confidence(was_imputed)
        
        # Apply Hampel filter to remove outliers
        filtered_keypoints, outlier_mask = kpoints.hampel_filter(
            imputed_keypoints, was_imputed, **config.post_align_hampel
        )
        
        # Final smoothing of imputed regions
        smoothed_keypoints = kpoints.simple_smooth_imputed(
            filtered_keypoints, was_imputed, **config.post_align_imputed_smoothing
        )
        
        # Combine confidence scores
        combined_confidence = self._combine_confidence_scores(
            confidence, imputation_confidence
        )
        
        return smoothed_keypoints, combined_confidence
    
    def _apply_final_bone_constraints(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray,
        # included_keypoints: List[str],
        config: PostProcessingConfig,
        bone_constraints: BoneConstraints, 
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply bone constraints one final time after imputation."""
        # Reuse bone constraints from earlier step
        # bone_constraints = kpoints.create_bone_constraints_from_data(
        #     keypoints, self.included_keypoints, self.skeleton, confidence
        # )
        
        optimizer = kpoints.BoneConstraintOptimizer(
            bone_constraints, **config.bone_length_regularization
        )
        
        return optimizer.process_sequence(keypoints, confidence)
    
    def _apply_final_smoothing(
        self,
        keypoints: np.ndarray,
        included_keypoints: List[str],
        smoothing_params: Dict[str, Any]
    ) -> np.ndarray:
        """Apply final Savitzky-Golay smoothing to the keypoints."""
        return apply_final_smoothing(
            keypoints, included_keypoints, **smoothing_params
        )
    
    def _combine_confidence_scores(
        self, 
        conf1: np.ndarray, 
        conf2: np.ndarray,
        w1 : np.float64 = 0.5,
        w2 : np.float64 = 0.5
    ) -> np.ndarray:
        """
        Combine two confidence score arrays.
        
        Note: This implements a simple additive combination.
        Consider implementing proper confidence fusion methods in the future.
        """
        return (conf1 * w1 + conf2 * w2)


# Convenience function to maintain backward compatibility
def post_processing(
    merged_data: np.ndarray,
    merged_conf: np.ndarray,
    kpoints_metadata: Dict[str, Any],
    postprocessing_params: Dict[str, Any],
    incl_kpoints_post_processing: List[str],
    skeleton: Any,
    constrain_bones: bool = True,
    impute_pca: bool = True,
    regularize_temporal: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Legacy wrapper for the refactored post-processing functionality.
    
    This function maintains backward compatibility while using the new implementation.
    
    Args:
        merged_data: Raw keypoint data
        merged_conf: Confidence scores for keypoints
        kpoints_metadata: Metadata about keypoint structure
        postprocessing_params: Dictionary of processing parameters
        incl_kpoints_post_processing: List of keypoint names to include
        skeleton: Skeleton structure for bone constraints
        constrain_bones: Enable/disable bone constraint processing
        impute_pca: Enable/disable PCA imputation processing
        regularize_temporal: Enable/disable temporal regularization
        
    Returns:
        Tuple of (processed_keypoints, processed_confidence)
    """

    print("Applying post processing...")
    processor = KeypointPostProcessor(kpoints_metadata, skeleton)
    
    config = PostProcessingConfig(**postprocessing_params)
    flags = ProcessingFlags(
        constrain_bones=constrain_bones,
        impute_pca=impute_pca,
        regularize_temporal=regularize_temporal
    )
    
    return processor.process(
        merged_data,
        merged_conf,
        incl_kpoints_post_processing,
        config,
        flags
    )