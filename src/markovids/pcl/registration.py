import numpy as np
import warnings
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from tqdm.auto import tqdm
from typing import Union, Optional


def estimate_rigid_transform(A, B):
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    U, _, Vt = np.linalg.svd(H)
    R_ = Vt.T @ U.T
    if np.linalg.det(R_) < 0:
        Vt[-1, :] *= -1
        R_ = Vt.T @ U.T
    t = centroid_B - R_ @ centroid_A
    return R_, t


# def estimate_rigid_transform(A, B):
#     A = A.reshape(-1, 3)
#     B = B.reshape(-1, 3)
#     centroid_A = A.mean(axis=0)
#     centroid_B = B.mean(axis=0)
#     AA = A - centroid_A
#     BB = B - centroid_B
#     H = BB.T @ AA
#     U, S, Vt = np.linalg.svd(H)
#     R = Vt.T @ U.T
#     if np.linalg.det(R) < 0:
#         Vt[2, :] *= -1
#         R = Vt.T @ U.T
#     t = centroid_A - R @ centroid_B
#     return R, t


def estimate_similarity_transform(A, B):
    assert A.shape == B.shape
    N = A.shape[0]

    # Compute centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # Center the points
    AA = A - centroid_A
    BB = B - centroid_B

    # Compute covariance matrix
    H = AA.T @ BB / N

    # SVD
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Reflection correction
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute scale
    var_A = np.var(AA, axis=0).sum()
    s = (S @ np.ones(3)) / var_A

    # Translation
    t = centroid_B - s * R @ centroid_A

    return s, R, t


def residuals_rigid(x, points_A, points_B, points_C, weights_B, weights_C):
    rv_B = x[0:3]
    t_B = x[3:6]
    rv_C = x[6:9]
    t_C = x[9:12]

    R_B = Rotation.from_rotvec(rv_B).as_matrix()
    R_C = Rotation.from_rotvec(rv_C).as_matrix()

    B_proj = (R_B @ points_A.T).T + t_B
    C_proj = (R_C @ points_A.T).T + t_C

    err_B = weights_B[:, None] * (B_proj - points_B)
    err_C = weights_C[:, None] * (C_proj - points_C)

    return np.hstack([err_B.ravel(), err_C.ravel()])


def bundle_adjust_rigid_fixed_structure(
    points_A,
    points_B,
    points_C,
    weights_B=None,
    weights_C=None,
    huber_delta=5.0,
    jac_sparsity=None,
    **kwargs,
):
    N = points_A.shape[0]

    if weights_B is None:
        weights_B = np.ones(N)
    if weights_C is None:
        weights_C = np.ones(N)

    # Estimate initial R, t (from B → A and C → A)
    R_B0, t_B0 = estimate_rigid_transform(points_A, points_B)
    R_C0, t_C0 = estimate_rigid_transform(points_A, points_C)
    rv_B0 = Rotation.from_matrix(R_B0).as_rotvec()
    rv_C0 = Rotation.from_matrix(R_C0).as_rotvec()
    x0 = np.hstack([rv_B0, t_B0, rv_C0, t_C0])

    result = least_squares(
        residuals_rigid,
        x0,
        args=(points_A, points_B, points_C, weights_B, weights_C),
        loss="huber",
        f_scale=huber_delta,
        jac_sparsity=jac_sparsity,
        **kwargs,
    )

    rv_B, t_B = result.x[0:3], result.x[3:6]
    rv_C, t_C = result.x[6:9], result.x[9:12]
    R_B = Rotation.from_rotvec(rv_B).as_matrix()
    R_C = Rotation.from_rotvec(rv_C).as_matrix()

    # Invert to get B → A and C → A
    R_B_inv = R_B.T
    t_B_inv = -R_B_inv @ t_B
    R_C_inv = R_C.T
    t_C_inv = -R_C_inv @ t_C

    return {
        "points_3d": points_A,
        "B_to_A": {"R": R_B_inv, "t": t_B_inv},
        "C_to_A": {"R": R_C_inv, "t": t_C_inv},
    }


def invert_similarity_transform(R, t, s):
    R_inv = R.T
    s_inv = 1.0 / s
    t_inv = -s_inv * R_inv @ t
    return R_inv, t_inv, s_inv


# -----------------------------
# Residuals: fixed structure + similarity
# -----------------------------
def residuals_similarity_fixed(x, points_A, points_B, points_C, weights_B, weights_C):
    rv_B = x[0:3]
    t_B = x[3:6]
    s_B = x[6]
    rv_C = x[7:10]
    t_C = x[10:13]
    s_C = x[13]

    R_B = Rotation.from_rotvec(rv_B).as_matrix()
    R_C = Rotation.from_rotvec(rv_C).as_matrix()

    B_proj = (s_B * (R_B @ points_A.T)).T + t_B
    C_proj = (s_C * (R_C @ points_A.T)).T + t_C

    err_B = weights_B[:, None] * (B_proj - points_B)
    err_C = weights_C[:, None] * (C_proj - points_C)

    return np.hstack([err_B.ravel(), err_C.ravel()])


def bundle_adjust_fixed_structure_similarity(
    points_A, points_B, points_C, weights_B=None, weights_C=None, huber_delta=5.0
):
    N = points_A.shape[0]

    if weights_B is None:
        weights_B = np.ones(N)
    if weights_C is None:
        weights_C = np.ones(N)

    # Initial guess using similarity estimation (B → A, C → A)
    s_B0, R_B0, t_B0 = estimate_similarity_transform(points_B, points_A)
    s_C0, R_C0, t_C0 = estimate_similarity_transform(points_C, points_A)
    rv_B0 = Rotation.from_matrix(R_B0).as_rotvec()
    rv_C0 = Rotation.from_matrix(R_C0).as_rotvec()

    x0 = np.hstack([rv_B0, t_B0, s_B0, rv_C0, t_C0, s_C0])

    result = least_squares(
        residuals_similarity_fixed,
        x0,
        args=(points_A, points_B, points_C, weights_B, weights_C),
        loss="huber",
        f_scale=huber_delta,
        method="trf",
        verbose=2,
    )

    rv_B = result.x[0:3]
    t_B = result.x[3:6]
    s_B = result.x[6]
    rv_C = result.x[7:10]
    t_C = result.x[10:13]
    s_C = result.x[13]

    R_B = Rotation.from_rotvec(rv_B).as_matrix()
    R_C = Rotation.from_rotvec(rv_C).as_matrix()

    R_B_inv, t_B_inv, s_B_inv = invert_similarity_transform(R_B, t_B, s_B)
    R_C_inv, t_C_inv, s_C_inv = invert_similarity_transform(R_C, t_C, s_C)

    return {
        "points_3d": points_A,
        "B_to_A": {"R": R_B_inv, "t": t_B_inv, "s": s_B_inv},
        "C_to_A": {"R": R_C_inv, "t": t_C_inv, "s": s_C_inv},
    }
