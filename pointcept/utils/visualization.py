"""
Visualization Utils

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os

try:
    import open3d as o3d
except ImportError:
    o3d = None
import numpy as np
import torch

try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.clone().detach().cpu().numpy()
    assert isinstance(x, np.ndarray)
    return x


def feat_to_rgb_pca(feat, n_components=3):
    """
    Map high-dim features [N, C] to RGB [N, 3] via PCA, then min-max normalize to [0, 1].
    Falls back to feat_to_rgb_norm if sklearn.PCA is not available.
    """
    feat = to_numpy(feat)
    if feat.ndim != 2:
        feat = feat.reshape(feat.shape[0], -1)
    n, c = feat.shape
    if PCA is None or c < n_components:
        return feat_to_rgb_norm(feat)
    pca = PCA(n_components=n_components)
    rgb = pca.fit_transform(feat)
    rgb_min, rgb_max = rgb.min(axis=0), rgb.max(axis=0)
    rgb_range = rgb_max - rgb_min
    rgb_range[rgb_range < 1e-8] = 1.0
    rgb = (rgb - rgb_min) / rgb_range
    return rgb.astype(np.float64)


def feat_to_rgb_norm(feat):
    """
    Use first 3 dims of feat as RGB, min-max normalized to [0, 1]. Fast fallback.
    """
    feat = to_numpy(feat)
    if feat.ndim != 2:
        feat = feat.reshape(feat.shape[0], -1)
    rgb = feat[:, : min(3, feat.shape[1])]
    if rgb.shape[1] < 3:
        rgb = np.pad(rgb, ((0, 0), (0, 3 - rgb.shape[1])), mode="edge")
    rgb_min, rgb_max = rgb.min(axis=0), rgb.max(axis=0)
    rgb_range = rgb_max - rgb_min
    rgb_range[rgb_range < 1e-8] = 1.0
    rgb = (rgb - rgb_min) / rgb_range
    return rgb.astype(np.float64)


def get_point_cloud(coord, color=None, verbose=True):
    if not isinstance(coord, list):
        coord = [coord]
        if color is not None:
            color = [color]

    pcd_list = []
    for i in range(len(coord)):
        coord_ = to_numpy(coord[i])
        if color is not None:
            color_ = to_numpy(color[i])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coord_)
        pcd.colors = o3d.utility.Vector3dVector(
            np.zeros_like(coord_) if color is None else color_
        )
        pcd_list.append(pcd)
    if verbose:
        o3d.visualization.draw_geometries(pcd_list)
    return pcd_list


def get_line_set(coord, line, color=(1.0, 0.0, 0.0), verbose=True):
    coord = to_numpy(coord)
    line = to_numpy(line)
    colors = np.array([color for _ in range(len(line))])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(coord)
    line_set.lines = o3d.utility.Vector2iVector(line)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    if verbose:
        o3d.visualization.draw_geometries([line_set])
    return line_set


def save_point_cloud(coord, color=None, file_path="pc.ply", logger=None):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    coord = to_numpy(coord)
    if color is not None:
        color = to_numpy(color)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.colors = o3d.utility.Vector3dVector(
        np.ones_like(coord) if color is None else color
    )
    o3d.io.write_point_cloud(file_path, pcd)
    if logger is not None:
        logger.info(f"Save Point Cloud to: {file_path}")


def save_bounding_boxes(
    bboxes_corners, color=(1.0, 0.0, 0.0), file_path="bbox.ply", logger=None
):
    bboxes_corners = to_numpy(bboxes_corners)
    # point list
    points = bboxes_corners.reshape(-1, 3)
    # line list
    box_lines = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 0],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
    )
    lines = []
    for i, _ in enumerate(bboxes_corners):
        lines.append(box_lines + i * 8)
    lines = np.concatenate(lines)
    # color list
    color = np.array([color for _ in range(len(lines))])
    # generate line set
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_line_set(file_path, line_set)

    if logger is not None:
        logger.info(f"Save Boxes to: {file_path}")


def save_lines(
    points, lines, color=(1.0, 0.0, 0.0), file_path="lines.ply", logger=None
):
    points = to_numpy(points)
    lines = to_numpy(lines)
    colors = np.array([color for _ in range(len(lines))])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_line_set(file_path, line_set)

    if logger is not None:
        logger.info(f"Save Lines to: {file_path}")
