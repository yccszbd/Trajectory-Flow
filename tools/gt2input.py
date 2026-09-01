#!/usr/bin/env python3
"""
Mesh → Point-cloud batch sampler
支持命令行配置采样方式、点数、并行度
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import os
from typing import Union

import numpy as np
import open3d as o3d
import trimesh
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.utils import normalize_mesh, start_process_pool

# ------------------------------------------------------------------
# 采样策略 dispatcher
# ------------------------------------------------------------------

AXIS_NAME_TO_INDEX = {
    "x": 0,
    "y": 1,
    "z": 2,
    "0": 0,
    "1": 1,
    "2": 2,
}


def parse_axis(axis: Union[str, int]) -> int:
    axis_key = str(axis).lower()
    if axis_key not in AXIS_NAME_TO_INDEX:
        raise ValueError(f"Invalid axis: {axis}. Use x/y/z or 0/1/2.")
    return AXIS_NAME_TO_INDEX[axis_key]


def make_axis_density_weights(
    coords: np.ndarray,
    axis: Union[str, int] = "x",
    direction: str = "min_to_max",
    density_ratio: float = 100.0,
    power: float = 2.0,
) -> np.ndarray:
    """
    Build weights that increase along one axis.

    min_to_max: axis min side is sparse, axis max side is dense.
    max_to_min: axis max side is sparse, axis min side is dense.
    density_ratio is approximately dense_side / sparse_side.
    """
    axis_idx = parse_axis(axis)
    if density_ratio < 1.0:
        raise ValueError("--gradient_density_ratio must be >= 1.0")
    if power <= 0.0:
        raise ValueError("--gradient_power must be > 0.0")

    values = coords[:, axis_idx]
    v_min, v_max = values.min(), values.max()
    span = v_max - v_min
    if span <= 1e-12:
        return np.ones(coords.shape[0], dtype=np.float64)

    t = (values - v_min) / span
    if direction == "max_to_min":
        t = 1.0 - t
    elif direction != "min_to_max":
        raise ValueError("--gradient_direction must be min_to_max or max_to_min")

    t = np.clip(t, 0.0, 1.0) ** power
    weights = 1.0 + (density_ratio - 1.0) * t
    return weights.astype(np.float64)


def allocate_axis_gradient_counts(
    count: int,
    bins: int,
    direction: str = "min_to_max",
    density_ratio: float = 100.0,
    power: float = 2.0,
) -> np.ndarray:
    if bins <= 0:
        raise ValueError("--gradient_bins must be > 0")
    centers = (np.arange(bins, dtype=np.float64) + 0.5) / bins
    if direction == "max_to_min":
        centers = 1.0 - centers
    elif direction != "min_to_max":
        raise ValueError("--gradient_direction must be min_to_max or max_to_min")

    profile = 1.0 + (density_ratio - 1.0) * (centers ** power)
    raw = profile / profile.sum() * count
    counts = np.floor(raw).astype(np.int64)
    remainder = count - int(counts.sum())
    if remainder > 0:
        frac_order = np.argsort(-(raw - counts))
        counts[frac_order[:remainder]] += 1
    return counts


def sample_indices_by_axis_bins(
    points: np.ndarray,
    count: int,
    axis: Union[str, int] = "x",
    direction: str = "min_to_max",
    density_ratio: float = 100.0,
    power: float = 2.0,
    bins: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick exact per-bin quotas so point counts increase along the chosen axis."""
    axis_idx = parse_axis(axis)
    values = points[:, axis_idx]
    v_min, v_max = values.min(), values.max()
    if v_max - v_min <= 1e-12:
        indices = np.random.choice(len(points), count, replace=len(points) < count)
        return indices, np.array([count], dtype=np.int64)

    target_counts = allocate_axis_gradient_counts(count, bins, direction, density_ratio, power)
    edges = np.linspace(v_min, v_max, bins + 1)
    bin_ids = np.searchsorted(edges, values, side="right") - 1
    bin_ids = np.clip(bin_ids, 0, bins - 1)

    selected = []
    for bin_idx, target in enumerate(target_counts):
        if target <= 0:
            continue
        candidates = np.flatnonzero(bin_ids == bin_idx)
        if len(candidates) == 0:
            continue
        replace = len(candidates) < target
        chosen = np.random.choice(candidates, int(target), replace=replace)
        selected.append(chosen)

    if selected:
        selected_indices = np.concatenate(selected)
    else:
        selected_indices = np.empty(0, dtype=np.int64)

    shortfall = count - len(selected_indices)
    if shortfall > 0:
        weights = make_axis_density_weights(points, axis, direction, density_ratio, power)
        probs = weights / weights.sum()
        extra = np.random.choice(np.arange(len(points)), shortfall, replace=len(points) < shortfall, p=probs)
        selected_indices = np.concatenate([selected_indices, extra])
    elif shortfall < 0:
        selected_indices = np.random.choice(selected_indices, count, replace=False)

    np.random.shuffle(selected_indices)
    actual_counts = np.bincount(bin_ids[selected_indices], minlength=bins)
    return selected_indices, actual_counts


def sample_random_surface(mesh: trimesh.Trimesh, count: int) -> tuple[np.ndarray, np.ndarray]:
    """纯随机:随机面 + 随机重心坐标,一行 Open3D 搞定"""
    # trimesh → open3d
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mesh.vertices), triangles=o3d.utility.Vector3iVector(mesh.faces)
    )
    # 计算顶点法向(可选,但采样时会插值面法向)
    o3d_mesh.compute_vertex_normals()

    pcd = o3d_mesh.sample_points_uniformly(number_of_points=count)  # ← 核心一行
    pts = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)  # 已插值好
    return pts, normals


def sample_uniform_surface(mesh: trimesh.Trimesh, count: int) -> tuple[np.ndarray, np.ndarray]:
    """均匀表面采样(面积加权)"""
    pts, face_idx = trimesh.sample.sample_surface(mesh, count)
    normals = mesh.face_normals[face_idx]
    return pts, normals


def sample_height_gradient(
    mesh: trimesh.Trimesh,
    count: int,
    axis: Union[str, int] = "x",
    direction: str = "min_to_max",
    density_ratio: float = 100.0,
    power: float = 2.0,
    bins: int = 80,
    oversample: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """沿指定坐标轴从稀疏到密集的表面采样。"""
    try:
        candidate_count = max(count * oversample, count + bins * 100)
        candidate_pts, candidate_face_idx = trimesh.sample.sample_surface(mesh, candidate_count)
        selected_idx, actual_counts = sample_indices_by_axis_bins(
            candidate_pts,
            count,
            axis=axis,
            direction=direction,
            density_ratio=density_ratio,
            power=power,
            bins=bins,
        )
        pts = candidate_pts[selected_idx]
        normals = mesh.face_normals[candidate_face_idx[selected_idx]]
        axis_idx = parse_axis(axis)
        axis_values = pts[:, axis_idx]
        print(
            "Gradient sampling: "
            f"axis={axis}, direction={direction}, ratio={density_ratio:g}, power={power:g}, "
            f"bins={bins}, bin_counts={actual_counts.tolist()}, "
            f"point_axis_range=[{axis_values.min():.3f}, {axis_values.max():.3f}]"
        )
        return pts, normals
    except Exception as e:
        print(f"Gradient sampling failed: {e}, using uniform.")
        return sample_uniform_surface(mesh, count)


def sample_poisson_disk(mesh: trimesh.Trimesh, count: int) -> tuple[np.ndarray, np.ndarray]:
    """泊松盘采样(trimesh 自带)"""
    pts, face_idx = trimesh.sample.sample_surface_even(mesh, count)
    normals = mesh.face_normals[face_idx]
    return pts, normals


def sample_poisson_fps(mesh: trimesh.Trimesh, count: int) -> tuple[np.ndarray, np.ndarray]:
    """
    1. 先用泊松盘采 2*count 个点
    2. 再用最远点采样降回到 count
    """
    pts, face_idx = trimesh.sample.sample_surface_even(mesh, count * 2)
    # 简单最远点采样
    selected = [np.random.randint(0, len(pts))]
    dists = np.full(len(pts), np.inf)
    for _ in range(count - 1):
        last = selected[-1]
        dists = np.minimum(dists, np.linalg.norm(pts - pts[last], axis=1))
        selected.append(int(np.argmax(dists)))
    pts = pts[selected]
    normals = mesh.face_normals[face_idx[selected]]
    return pts, normals


def sample_point_cloud_data(
    points: np.ndarray,
    count: int,
    normals: np.ndarray = None,
    weights: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    点云采样逻辑：
    - 如果点数 > count: 随机无放回抽取 (Downsample)
    - 如果点数 < count: 随机有放回重采样补全 (Upsample)
    """
    total = len(points)

    # 1. 点数正好
    if total == count:
        return points, (normals if normals is not None else np.zeros((count, 3)))

    indices = np.arange(total)
    probs = None
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape[0] != total:
            raise ValueError("weights length must match points length")
        weights = np.maximum(weights, 0.0)
        weight_sum = weights.sum()
        if weight_sum > 0.0:
            probs = weights / weight_sum

    # 2. 点数过多 -> 随机下采样
    if total > count:
        selected_indices = np.random.choice(indices, count, replace=False, p=probs)

    # 3. 点数不足 -> 先全取，再随机补足
    else:
        # 计算还需要补多少个点
        gap = count - total
        # 随机选取 gap 个索引（允许重复）
        extra_indices = np.random.choice(indices, gap, replace=True, p=probs)
        # 拼接到原始索引后面
        selected_indices = np.concatenate([indices, extra_indices])

    # 获取采样后的点
    sampled_pts = points[selected_indices]

    # 处理法向 (如果有的话)
    if normals is not None and len(normals) == total:
        sampled_normals = normals[selected_indices]
    else:
        sampled_normals = np.zeros((count, 3))  # 如果没有法向，填充0

    return sampled_pts, sampled_normals


def sample_random_unit_vectors(count: int) -> np.ndarray:
    vectors = np.random.normal(0.0, 1.0, size=(count, 3))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors.astype(np.float64)


def sample_outlier_points(
    reference_points: np.ndarray,
    count: int,
    mode: str = "uniform_bbox",
    bbox_expand: float = 0.25,
    gaussian_scale: float = 0.15,
) -> np.ndarray:
    if count <= 0:
        return np.empty((0, 3), dtype=np.float64)
    if len(reference_points) == 0:
        raise ValueError("Cannot sample outliers without reference points.")

    ref_min = reference_points.min(axis=0)
    ref_max = reference_points.max(axis=0)
    center = (ref_min + ref_max) * 0.5
    extent = ref_max - ref_min
    diag = np.linalg.norm(extent) + 1e-12

    if mode == "uniform_bbox":
        half = extent * (0.5 + bbox_expand)
        half = np.maximum(half, diag * 1e-4)
        low = center - half
        high = center + half
        return np.random.uniform(low, high, size=(count, 3))

    if mode == "gaussian":
        base_idx = np.random.choice(len(reference_points), count, replace=True)
        noise = np.random.normal(0.0, diag * gaussian_scale, size=(count, 3))
        return reference_points[base_idx] + noise

    raise ValueError("--outlier_mode must be uniform_bbox or gaussian")


def append_outliers(
    points: np.ndarray,
    normals: np.ndarray,
    outlier_count: int,
    mode: str,
    bbox_expand: float,
    gaussian_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    if outlier_count <= 0:
        return points, normals

    outlier_pts = sample_outlier_points(points, outlier_count, mode, bbox_expand, gaussian_scale)
    outlier_normals = sample_random_unit_vectors(outlier_count)
    points = np.concatenate([points, outlier_pts], axis=0)
    normals = np.concatenate([normals, outlier_normals], axis=0)
    print(f"Added outliers: count={outlier_count}, mode={mode}, final_points={len(points)}")
    return points, normals


def write_point_cloud_file(
    out_file: Path,
    points: np.ndarray,
    normals: np.ndarray,
    with_normal: bool,
) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if with_normal:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud(str(out_file), pcd)


SAMPLER = {
    "random": sample_random_surface,
    "uniform": sample_uniform_surface,
    "poisson": sample_poisson_disk,
    "poisson_fps": sample_poisson_fps,
    "gradient": sample_height_gradient,
}


# ------------------------------------------------------------------
# 单文件处理
# ------------------------------------------------------------------
# def data_no_filter(
#     mesh_path: Path, input_path: Path, sample_num: int, sample_mode: str, with_normal: bool = True
# ) -> None:
#     print(f"[PID {os.getpid()}] Processing: {mesh_path}")
#
#     try:
#         mesh = trimesh.load_mesh(mesh_path, process=False, maintain_order=False, force="mesh")
#         if isinstance(mesh, trimesh.Scene):
#             if len(mesh.geometry) == 0:
#                 print(f"Empty scene: {mesh_path}")
#                 return
#             mesh = mesh.to_mesh()
#
#         mesh = normalize_mesh(mesh)
#
#         # 采样
#         pts, normals = SAMPLER[sample_mode](mesh, sample_num)
#
#         # 保存
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(pts)
#         if with_normal:  # ← 新增判断
#             pcd.normals = o3d.utility.Vector3dVector(normals)
#
#         out_file = input_path / f"{mesh_path.stem}.ply"
#         o3d.io.write_point_cloud(str(out_file), pcd)
#
#         print(f"Saved -> {out_file}")
#     except Exception as e:
#         print(f"Error on {mesh_path}: {e}")
def data_no_filter(
        mesh_path: Path,
        input_path: Path,
        sample_num: int,
        sample_mode: str,
        with_normal: bool = True,
        noise_ratio: float = 0.0,
        gradient_axis: str = "x",
        gradient_direction: str = "min_to_max",
        gradient_density_ratio: float = 100.0,
        gradient_power: float = 2.0,
        gradient_bins: int = 80,
        gradient_oversample: int = 50,
        outlier_ratio: float = 0.0,
        outlier_mode: str = "uniform_bbox",
        outlier_bbox_expand: float = 0.25,
        outlier_gaussian_scale: float = 0.15,
        split_outlier_files: bool = False,
) -> None:
    print(f"[PID {os.getpid()}] Processing: {mesh_path}")

    try:
        if not (0.0 <= outlier_ratio < 1.0):
            raise ValueError("--outlier_ratio must be in [0, 1)")
        outlier_num = int(round(sample_num * outlier_ratio))
        outlier_num = min(outlier_num, sample_num - 1)
        surface_sample_num = sample_num - outlier_num
        if outlier_num > 0:
            print(
                f"Outlier sampling enabled: surface={surface_sample_num}, "
                f"outlier={outlier_num}, total={sample_num}"
            )

        # [关键修改] 去掉 force='mesh'，允许加载点云
        # process=False 防止 trimesh 自动合并顶点或改变数据
        mesh = trimesh.load(mesh_path, process=False)

        # 处理 Scene
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) == 0: return
            # 尝试合并或者取第一个几何体
            mesh = list(mesh.geometry.values())[0]

        # 检查是否成功加载
        if mesh is None:
            print(f"Error: Failed to load {mesh_path}")
            return

        # 获取顶点数据
        # trimesh 的 PointCloud 对象属性通常是 vertices
        if hasattr(mesh, 'vertices'):
            pts = np.array(mesh.vertices)
        else:
            print(f"Error: No vertices found in {mesh_path}")
            return

        # -------------------------------------------------
        # 分支处理：Mesh vs PointCloud
        # -------------------------------------------------

        # 判断是否有面 (Faces) -> 有面就是 Mesh
        has_faces = hasattr(mesh, 'faces') and mesh.faces is not None and len(mesh.faces) > 0

        if has_faces:
            # --- Mesh 处理流程 ---
            # 1. 归一化 (使用你原来的工具或手动)
            try:
                mesh = normalize_mesh(mesh)
            except Exception:
                # 如果 normalize_mesh 失败，手动归一化
                p_min, p_max = mesh.vertices.min(0), mesh.vertices.max(0)
                center = (p_min + p_max) / 2
                scale = (p_max - p_min).max()
                mesh.apply_translation(-center)
                mesh.apply_scale(1.0 / (scale + 1e-8))

            # 2. 使用采样器
            if sample_mode == "gradient":
                sampled_pts, sampled_normals = sample_height_gradient(
                    mesh,
                    surface_sample_num,
                    axis=gradient_axis,
                    direction=gradient_direction,
                    density_ratio=gradient_density_ratio,
                    power=gradient_power,
                    bins=gradient_bins,
                    oversample=gradient_oversample,
                )
            else:
                sampled_pts, sampled_normals = SAMPLER[sample_mode](mesh, surface_sample_num)

        else:
            # --- Point Cloud 处理流程 ---
            print(f"  -> Detected PointCloud (no faces), using resampling.")

            # 1. 手动归一化 (PointCloud 对象可能没有 apply_scale 等方法，直接操作 numpy)
            p_min, p_max = pts.min(0), pts.max(0)
            center = (p_min + p_max) / 2
            scale = (p_max - p_min).max()
            pts = (pts - center) / (scale + 1e-8)

            # 尝试获取现有的法向 (如果有)
            original_normals = None
            if hasattr(mesh, 'vertex_normals') and len(mesh.vertex_normals) == len(pts):
                original_normals = np.array(mesh.vertex_normals)

            point_weights = None
            if sample_mode == "gradient":
                selected_idx, actual_counts = sample_indices_by_axis_bins(
                    pts,
                    surface_sample_num,
                    gradient_axis,
                    gradient_direction,
                    gradient_density_ratio,
                    gradient_power,
                    gradient_bins,
                )
                print(f"Gradient point-cloud resampling: bins={gradient_bins}, bin_counts={actual_counts.tolist()}")
                sampled_pts = pts[selected_idx]
                if original_normals is not None and len(original_normals) == len(pts):
                    sampled_normals = original_normals[selected_idx]
                else:
                    sampled_normals = np.zeros((surface_sample_num, 3))
            else:
                sampled_pts, sampled_normals = sample_point_cloud_data(
                    pts, surface_sample_num, original_normals, point_weights
                )

        # -------------------------------------------------
        # 后处理：加噪声 & 保存 (这部分逻辑保持不变)
        # -------------------------------------------------
        if noise_ratio > 0.0:
            min_xyz = np.min(sampled_pts, axis=0)
            max_xyz = np.max(sampled_pts, axis=0)
            diag = np.linalg.norm(max_xyz - min_xyz)
            sigma = diag * noise_ratio
            noise = np.random.normal(loc=0.0, scale=sigma, size=sampled_pts.shape)
            sampled_pts = sampled_pts + noise

        out_file = input_path / f"{mesh_path.stem}.ply"
        if split_outlier_files and outlier_num > 0:
            outlier_pts = sample_outlier_points(
                sampled_pts,
                outlier_num,
                outlier_mode,
                outlier_bbox_expand,
                outlier_gaussian_scale,
            )
            outlier_normals = sample_random_unit_vectors(outlier_num)
            outlier_file = input_path / f"{mesh_path.stem}_outliers.ply"
            write_point_cloud_file(out_file, sampled_pts, sampled_normals, with_normal)
            write_point_cloud_file(outlier_file, outlier_pts, outlier_normals, with_normal)
            print(f"Saved clean -> {out_file} ({len(sampled_pts)} points)")
            print(f"Saved outliers -> {outlier_file} ({len(outlier_pts)} points)")
            return

        sampled_pts, sampled_normals = append_outliers(
            sampled_pts,
            sampled_normals,
            outlier_num,
            outlier_mode,
            outlier_bbox_expand,
            outlier_gaussian_scale,
        )

        write_point_cloud_file(out_file, sampled_pts, sampled_normals, with_normal)
        print(f"Saved -> {out_file}")

    except Exception as e:
        import traceback
        print(f"Error on {mesh_path}: {e}")
        # traceback.print_exc()
# ------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Batch mesh → point-cloud sampler")
    parser.add_argument("--sample_mode", choices=SAMPLER.keys(), default="uniform", help="Sampling strategy")
    parser.add_argument("--sample_num", type=int, default=100_000, help="Number of points to sample")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel processes")
    parser.add_argument(
        "--gt_dir",
        type=str,
        default="/home/ycc/data/DiffusionUDF不同版本/data/shapenetCars100000/ground_truth",
        help="Directory containing ground-truth meshes",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/ycc/data/DiffusionUDF不同版本/data/shapenetCars100000/input",
        help="Directory to save point clouds",
    )

    parser.add_argument("--without_normal", action="store_true", help="If set, do NOT write normals to output PLY")
    parser.add_argument(
        "--noise_ratio",
        type=float,
        default=0.0,
        help="Noise ratio relative to bbox diagonal (e.g., 0.005 for 0.5%)"
    )
    parser.add_argument(
        "--gradient_axis",
        type=str,
        default="x",
        choices=AXIS_NAME_TO_INDEX.keys(),
        help="Axis for gradient sampling: x/y/z or 0/1/2",
    )
    parser.add_argument(
        "--gradient_direction",
        type=str,
        default="min_to_max",
        choices=("min_to_max", "max_to_min"),
        help="min_to_max means sparse at axis min and dense at axis max",
    )
    parser.add_argument(
        "--gradient_density_ratio",
        type=float,
        default=100.0,
        help="Approximate dense-side / sparse-side sampling density ratio",
    )
    parser.add_argument(
        "--gradient_power",
        type=float,
        default=2.0,
        help="Gradient curve power. Larger values concentrate more points near the dense side.",
    )
    parser.add_argument(
        "--gradient_bins",
        type=int,
        default=80,
        help="Number of axis slices used to enforce sparse-to-dense point counts.",
    )
    parser.add_argument(
        "--gradient_oversample",
        type=int,
        default=50,
        help="Candidate multiplier for mesh gradient sampling before per-bin selection.",
    )
    parser.add_argument(
        "--outlier_ratio",
        type=float,
        default=0.0,
        help="Fraction of final points replaced by outliers, e.g. 0.1 means 10% outliers.",
    )
    parser.add_argument(
        "--outlier_mode",
        type=str,
        default="uniform_bbox",
        choices=("uniform_bbox", "gaussian"),
        help="uniform_bbox samples random bbox points; gaussian offsets surface points by large noise.",
    )
    parser.add_argument(
        "--outlier_bbox_expand",
        type=float,
        default=0.25,
        help="BBox expansion for uniform_bbox outliers, relative to half extent.",
    )
    parser.add_argument(
        "--outlier_gaussian_scale",
        type=float,
        default=0.15,
        help="Gaussian outlier std relative to bbox diagonal.",
    )
    parser.add_argument(
        "--split_outlier_files",
        action="store_true",
        help="Save clean points and outlier-only points as two separate PLY files.",
    )
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_files = sorted(p for p in gt_dir.iterdir() if p.suffix.lower() in {".ply", ".obj", ".off", ".stl"})

    call_params = [
        (
            m,
            out_dir,
            args.sample_num,
            args.sample_mode,
            not args.without_normal,
            args.noise_ratio,
            args.gradient_axis,
            args.gradient_direction,
            args.gradient_density_ratio,
            args.gradient_power,
            args.gradient_bins,
            args.gradient_oversample,
            args.outlier_ratio,
            args.outlier_mode,
            args.outlier_bbox_expand,
            args.outlier_gaussian_scale,
            args.split_outlier_files,
        )
        for m in mesh_files
    ]

    start_process_pool(data_no_filter, call_params, args.workers)


if __name__ == "__main__":
    import os

    main()
