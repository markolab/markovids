# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**markovids** is a Python package for processing 2D and 3D videos, particularly focused on animal behavior analysis using depth cameras and fluorescence imaging. The package handles multi-camera video synchronization, depth data processing, point cloud registration, and feature extraction from behavioral videos.

## Installation & Setup

```bash
# Install in development mode
pip install -e .

# Install with optional video processing dependencies
pip install -e .[vedo]
```

## CLI Commands

The package provides a CLI tool accessed via `markovids` command with the following subcommands:

### Core Processing Commands

```bash
# Compute tracking scalars from registration data
markovids compute-scalars REGISTRATION_FILE [OPTIONS]
# Options: --batch-size, --intrinsics-file, --scalar-dir, --z-range

# Crop video around tracked features 
markovids crop-video REGISTRATION_FILE [OPTIONS]
# Options: --scalar-path, --batch-size, --crop-size, --flip-model, --output-dir

# Generate preview videos for alternating excitation data
markovids generate-qd-preview INPUT_DIR [OPTIONS]
# Options: --nbatches, --batch-size, --overlap, --downsample

# Split alternating excitation videos into separate fluorescence/reflectance streams
markovids split-qd-videos INPUT_DIR [OPTIONS] 
# Options: --nbatches, --batch-size
```

### Environment Variables
Commands support environment variable configuration with prefix `MARKOVIDS_*`:
- `MARKOVIDS_REG_*` for compute-scalars
- `MARKOVIDS_CROP_*` for crop-video  
- `MARKOVIDS_QD_PREVIEW_*` for preview commands

## Code Architecture

### Package Structure
```
src/markovids/
├── cli.py              # Click-based CLI interface
├── util.py             # Core processing functions and utilities  
├── depth/              # Depth camera and point cloud processing
│   ├── filter.py       # Depth data filtering
│   ├── moments.py      # Image moment-based feature extraction
│   ├── plane.py        # Ground plane estimation
│   ├── track.py        # Object tracking algorithms
│   └── io.py           # Depth data I/O
├── pcl/                # Point cloud processing
│   ├── registration.py # Point cloud registration and alignment
│   ├── kpoints.py      # Keypoint detection and matching
│   ├── pipeline.py     # Processing pipelines
│   ├── fluo.py         # Fluorescence data processing
│   ├── viz.py          # Visualization utilities
│   └── io.py           # Point cloud I/O
└── vid/                # Video processing utilities
    ├── io.py           # Video I/O, multi-camera sync
    └── util.py         # Video filtering and processing
```

### Key Processing Flows

#### 1. Multi-Camera Video Synchronization
- `read_timestamps_multicam()`: Synchronizes timestamps across multiple cameras
- `read_frames_multicam()`: Reads synchronized frame data from multiple video files
- Handles alternating excitation protocols (fluorescence/reflectance)

#### 2. Depth-Based Feature Extraction  
- `get_frame_features()`: Extracts centroid, orientation, axis lengths from depth frames
- Uses image moments for robust feature computation
- Supports batch processing with configurable parameters

#### 3. Video Cropping and Registration
- Crops videos around tracked features using computed centroids/orientations
- Supports flip detection using ONNX models
- Outputs cropped videos and tracking metadata

#### 4. Point Cloud Registration
- Registers point clouds across time/cameras using keypoint matching
- Handles coordinate system transformations
- Integrates with depth processing pipeline

### Data Formats

- **HDF5 files**: Primary format for registration data and processed frames
- **TOML files**: Configuration and metadata storage
- **Parquet files**: Scalar/feature data storage (pandas DataFrames)
- **AVI/MP4**: Video input/output formats

### Key Dependencies

- **Core**: numpy, pandas, scipy, scikit-image, opencv-python-headless
- **Video**: tqdm for progress bars, matplotlib for visualization
- **ML**: onnxruntime for flip detection models
- **Optional**: vedo for 3D visualization, imageio[ffmpeg] for video encoding

### Missing Implementation

The `compute_scalars` function referenced in the CLI is not implemented. It should:
- Process registration files to extract tracking features
- Combine functionality from `depth/moments.py` for feature extraction
- Output pandas DataFrame with centroid, orientation, and other scalar features
- Handle batch processing of large video datasets

### Development Notes

- Use `hampel()` filter for outlier detection in time series data
- Video processing uses background subtraction and spatial/temporal filtering
- Multi-camera data requires careful timestamp alignment (merge_tolerance=0.001)
- Frame indexing is 0-based throughout the codebase