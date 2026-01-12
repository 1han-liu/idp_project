import math
import os
from typing import Iterable, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import median_filter
from skimage import color, feature, filters, measure, morphology


def get_angles_d(corner: str) -> Tuple[float, float, float]:
    angles_d_all = np.array(
        [
            [61.60038904, 82.59533203, 112.0661066],
            [38.2970794, 69.71231051, 84.1233756],
            [53.22106313, 59.96173898, 70.20060129],
        ],
        dtype=float,
    )
    corners = {"A": 0, "B": 1, "C": 2}
    idx = corners.get(str(corner).upper())
    if idx is None:
        raise ValueError(f"Unknown corner '{corner}', expected A/B/C.")
    angles_d = angles_d_all[idx]
    return float(angles_d[0]), float(angles_d[1]), float(angles_d[2])


def calc_angle_d(line1: Iterable[float], line2: Iterable[float]) -> float:
    v1 = np.asarray(line1, dtype=float)
    v2 = np.asarray(line2, dtype=float)
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0.0:
        return float("nan")
    cos_val = float(np.dot(v1, v2) / denom)
    cos_val = max(-1.0, min(1.0, cos_val))
    return math.degrees(math.acos(cos_val))


def _point_xy(point: Iterable[float]) -> Tuple[float, float]:
    if isinstance(point, dict):
        return float(point["x"]), float(point["y"])
    arr = np.asarray(point, dtype=float)
    return float(arr[0]), float(arr[1])


def swap_points(p1: Iterable[float], p2: Iterable[float]):
    return np.asarray(p2, dtype=float), np.asarray(p1, dtype=float)


def reorient_points(t: Iterable[float], p: Iterable[float]):
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    if t[1] > p[1]:
        return swap_points(t, p)
    if t[1] == p[1] and t[0] > p[0]:
        return swap_points(t, p)
    return t, p


def calc_normal(
    p1: Iterable[float],
    p2: Iterable[float],
) -> Tuple[np.ndarray, dict, dict]:
    x1, y1 = _point_xy(p1)
    x2, y2 = _point_xy(p2)
    v_te = np.array([x2 - x1, y2 - y1, 0.0], dtype=float)
    n3d = np.cross(v_te, np.array([0.0, 0.0, 1.0]))
    n2d = n3d[:2]
    norm = np.linalg.norm(n2d)
    if norm == 0:
        n2d_unit = np.array([0.0, 0.0])
    else:
        n2d_unit = n2d / norm
    vc = {"x": (x1 + x2) / 2.0, "y": (y1 + y2) / 2.0}
    seg_len = np.linalg.norm(v_te[:2])
    v = {
        "x": vc["x"] + seg_len * n2d_unit[0] / 10.0,
        "y": vc["y"] + seg_len * n2d_unit[1] / 10.0,
    }
    return n2d_unit, v, vc


def _objective(
    z: np.ndarray,
    m: np.ndarray,
    w: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    umw: float,
    vmw: float,
    umv: float,
) -> float:
    m3 = np.array([m[0], m[1], 0.0], dtype=float)
    wz = np.array([w[0], w[1], z[0]], dtype=float)
    uz = np.array([u[0], u[1], z[1]], dtype=float)
    vz = np.array([v[0], v[1], z[2]], dtype=float)

    umn_z = calc_angle_d(wz - m3, uz - m3)
    vmn_z = calc_angle_d(wz - m3, vz - m3)
    umv_z = calc_angle_d(uz - m3, vz - m3)

    if math.isnan(umn_z) or math.isnan(vmn_z) or math.isnan(umv_z):
        return 1e6

    vec = np.array([umn_z - umw, vmn_z - vmw, umv_z - umv], dtype=float)
    return float(np.linalg.norm(vec))


def _constraints_for_direction(direction: str):
    z0 = np.array([100.0, 100.0, 100.0], dtype=float)
    A = None
    b = None
    z_min = np.array([0.0, 0.0, 0.0], dtype=float)
    z_max = np.array([np.inf, np.inf, np.inf], dtype=float)

    if direction == "inwards":
        z0 = np.array([-100.0, 100.0, 100.0])
        A = np.array([[1.0, -1.0, 0.0], [1.0, 0.0, -1.0]])
        b = np.array([0.0, 0.0])
        z_min = np.array([-np.inf, 0.0, 0.0])
        z_max = np.array([0.0, np.inf, np.inf])
    elif direction == "extra_inwards":
        z0 = np.array([-100.0, -100.0, -100.0])
        A = np.array([[1.0, -1.0, 0.0], [1.0, 0.0, -1.0]])
        b = np.array([0.0, 0.0])
        z_min = np.array([-np.inf, -np.inf, -np.inf])
        z_max = np.array([0.0, 0.0, 0.0])
    elif direction == "outwards":
        z0 = np.array([100.0, 100.0, 100.0])
        A = np.array([[1.0, -1.0, 0.0], [1.0, 0.0, -1.0]])
        b = np.array([0.0, 0.0])
        z_min = np.array([0.0, 0.0, 0.0])
        z_max = np.array([np.inf, np.inf, np.inf])
    elif direction == "in-1-1":
        z0 = np.array([100.0, 100.0, 100.0])
        A = np.array([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]])
        b = np.array([0.0, 0.0])
        z_min = np.array([0.0, 0.0, 0.0])
        z_max = np.array([np.inf, np.inf, np.inf])
    elif direction == "in-1-2":
        z0 = np.array([100.0, 100.0, 100.0])
        A = np.array([[-1.0, 0.0, 1.0], [0.0, 1.0, -1.0]])
        b = np.array([0.0, 0.0])
        z_min = np.array([0.0, 0.0, 0.0])
        z_max = np.array([np.inf, np.inf, np.inf])
    elif direction == "in-2-1":
        z0 = np.array([100.0, 100.0, -100.0])
        A = np.array([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]])
        b = np.array([0.0, 0.0])
        z_min = np.array([0.0, 0.0, -np.inf])
        z_max = np.array([np.inf, np.inf, 0.0])
    elif direction == "in-2-2":
        z0 = np.array([100.0, -100.0, 100.0])
        A = np.array([[-1.0, 0.0, 1.0], [0.0, 1.0, -1.0]])
        b = np.array([0.0, 0.0])
        z_min = np.array([0.0, -np.inf, 0.0])
        z_max = np.array([np.inf, 0.0, np.inf])
    elif direction == "in-3-1":
        z0 = np.array([100.0, -100.0, -100.0])
        A = np.array([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]])
        b = np.array([0.0, 0.0])
        z_min = np.array([0.0, -np.inf, -np.inf])
        z_max = np.array([np.inf, 0.0, 0.0])
    elif direction == "in-3-2":
        z0 = np.array([100.0, -100.0, -100.0])
        A = np.array([[-1.0, 0.0, 1.0], [0.0, 1.0, -1.0]])
        b = np.array([0.0, 0.0])
        z_min = np.array([0.0, -np.inf, -np.inf])
        z_max = np.array([np.inf, 0.0, 0.0])
    elif direction == "in-4-1":
        z0 = np.array([-100.0, -100.0, -100.0])
        A = np.array([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]])
        b = np.array([0.0, 0.0])
        z_min = np.array([-np.inf, -np.inf, -np.inf])
        z_max = np.array([0.0, 0.0, 0.0])
    elif direction == "in-4-2":
        z0 = np.array([-100.0, -100.0, -100.0])
        A = np.array([[-1.0, 0.0, 1.0], [0.0, 1.0, -1.0]])
        b = np.array([0.0, 0.0])
        z_min = np.array([-np.inf, -np.inf, -np.inf])
        z_max = np.array([0.0, 0.0, 0.0])
    elif direction == "out-1-1":
        z0 = np.array([-100.0, -100.0, -100.0])
        A = np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
        b = np.array([0.0, 0.0])
        z_min = np.array([-np.inf, -np.inf, -np.inf])
        z_max = np.array([0.0, 0.0, 0.0])
    elif direction == "out-1-2":
        z0 = np.array([-100.0, -100.0, -100.0])
        A = np.array([[1.0, 0.0, -1.0], [0.0, -1.0, 1.0]])
        b = np.array([0.0, 0.0])
        z_min = np.array([-np.inf, -np.inf, -np.inf])
        z_max = np.array([0.0, 0.0, 0.0])
    elif direction == "out-2-1":
        z0 = np.array([-100.0, -100.0, 100.0])
        A = np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
        b = np.array([0.0, 0.0])
        z_min = np.array([-np.inf, -np.inf, 0.0])
        z_max = np.array([0.0, 0.0, np.inf])
    elif direction == "out-2-2":
        z0 = np.array([-100.0, 100.0, -100.0])
        A = np.array([[1.0, 0.0, -1.0], [0.0, -1.0, 1.0]])
        b = np.array([0.0, 0.0])
        z_min = np.array([-np.inf, 0.0, -np.inf])
        z_max = np.array([0.0, np.inf, 0.0])
    elif direction == "out-3-1":
        z0 = np.array([-100.0, 100.0, 100.0])
        A = np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
        b = np.array([0.0, 0.0])
        z_min = np.array([-np.inf, 0.0, 0.0])
        z_max = np.array([0.0, np.inf, np.inf])
    elif direction == "out-3-2":
        z0 = np.array([-100.0, 100.0, 100.0])
        A = np.array([[1.0, 0.0, -1.0], [0.0, -1.0, 1.0]])
        b = np.array([0.0, 0.0])
        z_min = np.array([-np.inf, 0.0, 0.0])
        z_max = np.array([0.0, np.inf, np.inf])
    elif direction == "out-4-1":
        z0 = np.array([100.0, 100.0, 100.0])
        A = np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
        b = np.array([0.0, 0.0])
        z_min = np.array([0.0, 0.0, 0.0])
        z_max = np.array([np.inf, np.inf, np.inf])
    elif direction == "out-4-2":
        z0 = np.array([100.0, 100.0, 100.0])
        A = np.array([[1.0, 0.0, -1.0], [0.0, -1.0, 1.0]])
        b = np.array([0.0, 0.0])
        z_min = np.array([0.0, 0.0, 0.0])
        z_max = np.array([np.inf, np.inf, np.inf])

    return z0, A, b, z_min, z_max


def recover_3d(
    m: Iterable[float],
    w: Iterable[float],
    u: Iterable[float],
    v: Iterable[float],
    corner: str,
    is_full: bool,
    direction: str,
):
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise ImportError("scipy is required for recover_3d") from exc

    umw, vmw, umv = get_angles_d(corner)

    m = np.asarray(m, dtype=float)
    w = np.asarray(w, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    z0, A, b, z_min, z_max = _constraints_for_direction(direction if not is_full else "")

    bounds = [
        (float(z_min[0]), float(z_max[0])),
        (float(z_min[1]), float(z_max[1])),
        (float(z_min[2]), float(z_max[2])),
    ]
    constraints = []
    if A is not None and b is not None:
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)

        def _ineq(z, A=A, b=b):
            return b - A.dot(z)

        constraints.append({"type": "ineq", "fun": _ineq})

    result = minimize(
        _objective,
        z0,
        args=(m, w, u, v, umw, vmw, umv),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"disp": False, "maxiter": 200},
    )

    z = result.x
    M = np.array([m[0], m[1], 0.0], dtype=float)
    W = np.array([w[0], w[1], z[0]], dtype=float)
    U = np.array([u[0], u[1], z[1]], dtype=float)
    V = np.array([v[0], v[1], z[2]], dtype=float)
    return M, W, U, V, result


def calc_hough_info(uv_struct: dict) -> Tuple[float, float, bool]:
    n = np.asarray(uv_struct["n"], dtype=float)
    theta = calc_angle_d(n, np.array([1.0, 0.0, 0.0])) * np.sign(n[1])
    rho = float(np.dot(n, np.asarray(uv_struct["t"], dtype=float)))
    is_opposite = float(np.dot(np.asarray(uv_struct["o"], dtype=float) - np.asarray(uv_struct["t"], dtype=float), n)) < 0
    return float(theta), rho, bool(is_opposite)


def calc_2d_3d_info(
    M: Iterable[float],
    W: Iterable[float],
    U: Iterable[float],
    V: Iterable[float],
    u_op_struct: dict,
    v_op_struct: dict,
    u_ad_struct: dict,
    v_ad_struct: dict,
    is_full: bool,
):
    uv_list = [np.asarray(U, dtype=float), np.asarray(V, dtype=float)]
    if is_full:
        uv_struct_list = [dict(u_op_struct), dict(v_op_struct)]
    else:
        uv_struct_list = [dict(u_ad_struct), dict(v_ad_struct)]

    M = np.asarray(M, dtype=float)
    W = np.asarray(W, dtype=float)
    for idx in range(2):
        n_3d = np.cross(W - M, uv_list[idx] - M)
        uv_struct_list[idx]["n_3d"] = n_3d / np.linalg.norm(n_3d)
        n_2d = np.asarray(uv_struct_list[idx]["n"], dtype=float)
        uv_struct_list[idx]["cos"] = abs(
            float(np.dot(uv_struct_list[idx]["n_3d"], n_2d))
        ) / (float(np.linalg.norm(uv_struct_list[idx]["n_3d"])) * float(np.linalg.norm(n_2d)))
        theta_0, rho_0, is_opposite = calc_hough_info(uv_struct_list[idx])
        uv_struct_list[idx]["theta_0"] = theta_0
        uv_struct_list[idx]["rho_0"] = rho_0
        uv_struct_list[idx]["is_opposite"] = is_opposite

    return uv_struct_list[0], uv_struct_list[1]


def calc_foot_point(m: Iterable[float], foot_ad_struct: dict) -> np.ndarray:
    t = np.asarray(foot_ad_struct["t"], dtype=float)
    e = np.asarray(foot_ad_struct["e"], dtype=float)
    m = np.asarray(m, dtype=float)
    mid = (t + e) / 2.0
    direction = t - m
    return mid + np.linalg.norm(e - t) * direction / np.linalg.norm(direction)


def calc_kernel_mask(I: np.ndarray, kernel: dict) -> np.ndarray:
    height, width = I.shape[:2]
    x, y = np.meshgrid(np.arange(1, width + 1), np.arange(1, height + 1))
    kernel_mask = np.ones((height, width), dtype=bool)
    k_c_cell = kernel.get("k_c_cell", [])
    k_o_cell = kernel.get("k_o_cell", [])
    num_corners = len(k_c_cell)
    for ii in range(num_corners):
        p1 = np.asarray(k_c_cell[ii], dtype=float)
        p2 = np.asarray(k_c_cell[(ii + 1) % num_corners], dtype=float)
        tau = p2 - p1
        n = np.array([tau[1], -tau[0]], dtype=float)
        o = np.asarray(k_o_cell[ii], dtype=float)
        sign_val = (p1[0] - o[0]) * n[0] + (p1[1] - o[1]) * n[1]
        mask_tmp = sign_val * ((p1[0] - x) * n[0] + (p1[1] - y) * n[1]) < 0
        kernel_mask = kernel_mask & mask_tmp
    return kernel_mask


def initialize_uv_struct(
    uv_struct: dict,
    dt_G: float,
    resolution: float,
    q2: float,
    r_diag: Iterable[float],
):
    line = {
        "point1": uv_struct["t"],
        "point2": uv_struct["e"],
        "theta": uv_struct.get("theta_0"),
        "rho": uv_struct.get("rho_0"),
    }
    uv_struct = dict(uv_struct)
    uv_struct["line"] = line
    x_G = np.array([0.0, 0.0, 0.0], dtype=float)
    uv_struct["EKF_G"] = create_EKF_G(dt_G, resolution, q2, r_diag, x_G)
    uv_struct["x_G_array"] = []
    uv_struct["dist_array"] = []
    uv_struct["dist_KF_array"] = []
    uv_struct["G_array"] = []
    uv_struct["G_KF_array"] = []
    return uv_struct


def create_EKF_G(
    dt_G: float,
    resolution: float,
    q2: float,
    r_diag: Iterable[float],
    x_G: Iterable[float],
):
    dt = float(dt_G)
    resolution = float(resolution)
    q2 = float(q2)
    x_G = np.asarray(x_G, dtype=float)
    r_diag = np.asarray(list(r_diag), dtype=float)

    F = np.array(
        [
            [1.0, dt, dt**2 / 2.0],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    Q = (
        np.array(
            [
                [dt**4 / 4.0, dt**3 / 2.0, dt**2 / 2.0],
                [dt**3 / 2.0, dt**2, dt],
                [dt**2 / 2.0, dt, 1.0],
            ],
            dtype=float,
        )
        * q2
        * resolution**2
    )
    R = np.diag(r_diag) * resolution**2

    return {
        "F": F,
        "Q": Q,
        "R": R,
        "x": x_G,
    }


def _resolve_image_path(image_file) -> str:
    if isinstance(image_file, dict):
        folder = image_file.get("folder")
        name = image_file.get("name")
        if folder is not None and name is not None:
            return os.path.join(folder, name)
    return str(image_file)


def _load_image(path: str) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img)


def find_edge_points(I: np.ndarray, theta: float, params_G: dict, kernel: dict) -> np.ndarray:
    gray = color.rgb2gray(I)
    thresh = filters.threshold_otsu(gray)
    t = max(1.0, float(thresh) * 1.2)
    gray_255 = gray * 255.0
    gray_255 = np.minimum(gray_255, t * 255.0)
    gray_255 = median_filter(gray_255, size=(7, 7), mode="reflect")
    gray_255 = median_filter(gray_255, size=(100, 100), mode="reflect")

    local_thresh = filters.threshold_local(gray_255, block_size=51)
    binary = gray_255 < local_thresh
    binary = morphology.binary_closing(~binary, morphology.disk(10))
    binary = morphology.binary_opening(binary, morphology.disk(10))

    I_wo_kernel = binary.copy()
    kernel_mask = calc_kernel_mask(binary, kernel)
    binary[kernel_mask] = True
    binary = morphology.binary_closing(binary, morphology.disk(100))

    labeled = measure.label(binary)
    props = measure.regionprops(labeled)
    if props:
        max_label = max(props, key=lambda p: p.area).label
        binary = labeled == max_label
    else:
        binary = np.zeros_like(binary, dtype=bool)

    binary = binary & I_wo_kernel
    edges = feature.canny(binary.astype(float), sigma=1.0)
    edges = morphology.binary_dilation(edges, morphology.disk(3))
    return edges.astype(bool)


def calc_masked_image(
    I: np.ndarray,
    point1: Iterable[float],
    point2: Iterable[float],
    n: Iterable[float],
    width: float,
    ratio: float,
):
    t = np.asarray(point1, dtype=float)
    e = np.asarray(point2, dtype=float)
    n = np.asarray(n, dtype=float)
    up = t + n * width
    dp = t - n * width
    s = e - t
    _ = s  # reserved for the optional mask2
    _ = ratio

    height, width_img = I.shape[:2]
    x, y = np.meshgrid(np.arange(1, width_img + 1), np.arange(1, height + 1))
    mask1 = ((up[0] - x) * n[0] + (up[1] - y) * n[1]) * (
        (dp[0] - x) * n[0] + (dp[1] - y) * n[1]
    ) < 0
    mask = mask1
    if I.ndim == 2:
        return I * mask, mask
    masked = I.copy()
    masked[~mask] = 0
    return masked, mask


def calc_theta_range(theta: float, delta_theta: float) -> np.ndarray:
    theta_range = np.linspace(theta - delta_theta, theta + delta_theta, 20)
    theta_range = np.sort((theta_range + 90) % 180 - 90)
    return theta_range


def hough(I: np.ndarray, theta: np.ndarray):
    raise NotImplementedError("hough translation pending")


def houghpeaks(Hs: np.ndarray, num_peak: int, nhood_size: Tuple[int, int]):
    raise NotImplementedError("houghpeaks translation pending")


def houghlines(
    I: np.ndarray,
    thetas: np.ndarray,
    rhos: np.ndarray,
    peaks,
    fill_gap: int,
    min_length: int,
):
    raise NotImplementedError("houghlines translation pending")


def reorient_line(line: dict) -> dict:
    p1, p2 = reorient_points(line["point1"], line["point2"])
    line = dict(line)
    line["point1"] = p1
    line["point2"] = p2
    return line


def update_line(
    image_file,
    params_G: dict,
    line: dict,
    t: Iterable[float],
    e: Iterable[float],
    n: Iterable[float],
    o: Iterable[float],
    is_opposite: bool,
    kernel: dict,
):
    old_line = line
    path = _resolve_image_path(image_file)
    I_orig = _load_image(path)
    I = I_orig.copy()

    I = find_edge_points(I, line["theta"], params_G, kernel)
    I, _ = calc_masked_image(I, line["point1"], line["point2"], n, params_G["width"], params_G["ratio"])

    Hs, thetas, rhos = hough(I, theta=calc_theta_range(line["theta"], params_G["delta_theta"]))
    rho_min = line["rho"] - params_G["width"] / params_G["width_divider"]
    rho_max = line["rho"] + params_G["width"] / params_G["width_divider"]
    rho_mask = (rhos < rho_min) | (rhos > rho_max)
    Hs = Hs.copy()
    Hs[rho_mask, :] = 0
    peaks = houghpeaks(Hs, params_G["num_peak"], nhood_size=(9, 1))
    lines = houghlines(I, thetas, rhos, peaks, fill_gap=5, min_length=7)

    dist2o = float("inf")
    line_cand = None
    t = np.asarray(t, dtype=float)
    n = np.asarray(n, dtype=float)
    o = np.asarray(o, dtype=float)
    for cand in lines:
        cand = reorient_line(cand)
        cand["point1"] = np.asarray([cand["point1"][0], cand["point1"][1], 0.0], dtype=float)
        cand["point2"] = np.asarray([cand["point2"][0], cand["point2"][1], 0.0], dtype=float)
        if np.linalg.norm(cand["point1"] - cand["point2"]) < params_G["len_min"]:
            continue
        dist2o_new = float(np.dot(o - line["point1"], n)) * (float(is_opposite) - 0.5) * 2.0
        if dist2o_new < dist2o:
            dist2o = dist2o_new
            line_cand = cand

    line = old_line if line_cand is None else line_cand
    dist = float(abs(np.dot(line["point1"] - t, n)))
    return line, dist, I_orig


def update_uv_struct(
    uv_struct: dict,
    image_file,
    ii: int,
    params_G: dict,
    kernel: dict,
):
    line, dist, I_orig = update_line(
        image_file,
        params_G,
        uv_struct["line"],
        uv_struct["t"],
        uv_struct["e"],
        uv_struct["n"],
        uv_struct["o"],
        uv_struct["is_opposite"],
        kernel,
    )
    uv_struct = dict(uv_struct)
    uv_struct["line"] = line
    uv_struct["dist"] = dist
    dist_array = list(uv_struct.get("dist_array", []))
    idx = max(int(ii) - 1, 0)
    if idx >= len(dist_array):
        dist_array.extend([None] * (idx - len(dist_array) + 1))
    dist_array[idx] = dist
    uv_struct["dist_array"] = dist_array
    return uv_struct, I_orig


def initialize_DSCGR(
    image_file,
    choose_is_full_fn=None,
    get_points_fn=None,
    choose_corner_fn=None,
    recover_3d_all_fn=None,
    save_path: Optional[str] = None,
):
    path = _resolve_image_path(image_file)
    I = _load_image(path)

    if choose_is_full_fn is None:
        raise NotImplementedError("choose_is_full function is required")
    if get_points_fn is None:
        raise NotImplementedError("get_points function is required")
    if choose_corner_fn is None:
        raise NotImplementedError("choose_corner function is required")
    if recover_3d_all_fn is None:
        raise NotImplementedError("recover_3d_all function is required")

    is_full = choose_is_full_fn(I)
    result = get_points_fn(I, is_full)
    (
        _fig_2d,
        _ax_2d,
        m,
        w,
        u,
        v,
        u_op_struct,
        v_op_struct,
        u_ad_struct,
        v_ad_struct,
        kernel,
    ) = result
    corner = choose_corner_fn(I)
    _fig_3d, _ax_3d, M, W, U, V = recover_3d_all_fn(I, m, w, u, v, corner, is_full)
    u_struct, v_struct = calc_2d_3d_info(
        M, W, U, V, u_op_struct, v_op_struct, u_ad_struct, v_ad_struct, is_full
    )
    uv_struct_list = [u_struct, v_struct]

    if save_path:
        from scipy.io import savemat

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        savemat(save_path, {"uv_struct_list": uv_struct_list, "kernel": kernel})

    return uv_struct_list, kernel

