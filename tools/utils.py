import multiprocessing
import os
import random
import shutil
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import trimesh
from matplotlib import cm, colors
from pyhocon import ConfigTree
from scipy.spatial import cKDTree
from torch.backends import cudnn

from tools.logger import print_log


def get_aver(distances, face):
    return (distances[face[0]] + distances[face[1]] + distances[face[2]]) / 3.0


def remove_far(gt_pts, mesh, dis_trunc=0.1, is_use_prj=False):
    # gt_pts: trimesh
    # mesh: trimesh

    gt_kd_tree = cKDTree(gt_pts)
    distances, vertex_ids = gt_kd_tree.query(mesh.vertices, p=2, distance_upper_bound=dis_trunc)
    faces_remaining = []
    faces = mesh.faces

    if is_use_prj:
        normals = gt_pts.vertex_normals
        closest_points = gt_pts.vertices[vertex_ids]
        closest_normals = normals[vertex_ids]
        direction_from_surface = mesh.vertices - closest_points
        distances = direction_from_surface * closest_normals
        distances = np.sum(distances, axis=1)

    faces_remaining = [face for face in faces if get_aver(distances, face) < dis_trunc]
    # 对于faces里的每个face,计算其三个顶点对应的距离的平均值,如果小于阈值就保留该face
    mesh_cleaned = mesh.copy()
    mesh_cleaned.faces = faces_remaining
    mesh_cleaned.remove_unreferenced_vertices()

    return mesh_cleaned


def remove_outlier(gt_pts, q_pts, dis_trunc=0.003):
    # gt_pts: trimesh
    # mesh: trimesh

    gt_kd_tree = cKDTree(gt_pts)
    distances, _ = gt_kd_tree.query(q_pts, p=2, distance_upper_bound=dis_trunc)

    q_pts = q_pts[distances < dis_trunc]

    return q_pts


def extract_np(a, t):
    """
    根据t的值从a中取出对应的元素,并返回与t相同shape的结果。
    a: shape为[T],通常是时间步的参数列表
    t: shape为任意(如[B, N, 1]),表示每个样本对应的时间步索引
    返回:与t相同shape的a中对应元素
    """
    # 先确保t为np.int64类型
    t = t.astype(np.int64)
    # 展平t,方便索引
    t_flat = t.reshape(-1)
    # 从a中取出对应的值
    out = a[t_flat]
    # 恢复成t的原始shape
    out = out.reshape(*t.shape)
    return out


def extract(a, t):
    """
    PyTorch风格的extract函数。
    a: shape为[T],torch.Tensor,通常是时间步的参数列表
    t: shape为任意(如[B, N, 1]),torch.LongTensor,表示每个样本对应的时间步索引
    返回:与t相同shape的a中对应元素
    """
    t = t.long()
    t_flat = t.reshape(-1)
    out = a[t_flat]
    out = out.reshape(*t.shape)
    return out


def gen_coefficients_np(timesteps, schedule="increased", sum_scale=1):
    """
    使用 NumPy 生成一个系数数组。

    参数:
      timesteps (int): 要生成的总步数(系数数量).
      schedule (str): 系数的生成方式 ('increased', 'decreased', 'average').
      sum_scale (float): 对最终系数和的缩放比例.

    返回:
      np.ndarray: 生成的系数数组。
    """
    if schedule == "increased":
        # np.arange(1, timesteps + 1) 等效于 torch.linspace(1, timesteps, timesteps)
        x = np.arange(1, timesteps + 1, dtype=np.float32)
        scale = 0.5 * timesteps * (timesteps + 1)
        alphas = x / scale

    elif schedule == "decreased":
        # np.arange(timesteps, 0, -1) 直接生成递减序列,等效于 torch.flip
        x = np.arange(timesteps, 0, -1, dtype=np.float32)
        scale = 0.5 * timesteps * (timesteps + 1)
        alphas = x / scale

    elif schedule == "average":
        # np.full() 等效于 torch.full()
        alphas = np.full(timesteps, 1 / timesteps, dtype=np.float32)

    else:
        # 默认情况
        alphas = np.full(timesteps, 1 / timesteps, dtype=np.float32)

    # np.isclose() 是比较浮点数的标准做法,比直接比较更稳健
    # 它等效于检查 |a - b| < tolerance
    assert np.isclose(np.sum(alphas), 1.0), "内部生成的 alphas 总和应为 1"

    return alphas * sum_scale


def gen_coefficients(timesteps, schedule="increased", sum_scale=1):
    if schedule == "increased":
        x = torch.linspace(1, timesteps, timesteps, dtype=torch.float32)  # 从1-1000
        scale = 0.5 * timesteps * (timesteps + 1)  # 500 * 1001
        alphas = x / scale
    elif schedule == "decreased":
        x = torch.linspace(1, timesteps, timesteps, dtype=torch.float32)
        x = torch.flip(x, dims=[0])  # 从1000 - 1
        scale = 0.5 * timesteps * (timesteps + 1)
        alphas = x / scale
    elif schedule == "average":
        alphas = torch.full([timesteps], 1 / timesteps, dtype=torch.float32)
    else:
        alphas = torch.full([timesteps], 1 / timesteps, dtype=torch.float32)
    assert alphas.sum() - torch.tensor(1) < torch.tensor(1e-10)

    return alphas * sum_scale


def set_seed(seed):
    """

    :param seed:
    :return:
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def normalize_mesh(mesh):
    # unit to [-0.5, 0.5]
    bounds = mesh.extents
    if bounds.min() == 0.0:
        return None

    # translate to origin
    translation = (mesh.bounds[0] + mesh.bounds[1]) * 0.5
    translation = trimesh.transformations.translation_matrix(direction=-translation)
    mesh.apply_transform(translation)

    # scale to unit cube
    scale = 1.0 / bounds.max()
    scale_trafo = trimesh.transformations.scale_matrix(factor=scale)
    mesh.apply_transform(scale_trafo)
    return mesh


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    """Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    """
    kdtree = cKDTree(points_tgt)
    dist, idx = kdtree.query(points_src, workers=-1)  # workers=-1 means all workers will be used

    if normals_src is not None and normals_tgt is not None:
        normals_src = normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        #        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        #        # Handle normals that point into wrong direction gracefully
        #        # (mostly due to mehtod not caring about this in generation)
        #        normals_dot_product = np.abs(normals_dot_product)

        normals_dot_product = np.abs(normals_tgt[idx] * normals_src)
        normals_dot_product = normals_dot_product.sum(axis=-1)
    else:
        normals_dot_product = np.array([np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product


def get_threshold_percentage(dist, thresholds):
    """Evaluates a point cloud.
    Args:
        dist (numpy array): calculated distance
        thresholds (numpy array): threshold values for the F-score calculation
    """
    in_threshold = [(dist <= t).mean() for t in thresholds]
    return in_threshold


def eval_pointcloud(pointcloud, pointcloud_tgt, normals=None, normals_tgt=None, thresholds=(0.01, 0.005)):
    """Evaluates a point cloud.

    Args:
        pointcloud (numpy array): predicted point cloud
        pointcloud_tgt (numpy array): target point cloud
        normals (numpy array): predicted normals
        normals_tgt (numpy array): target normals
    """
    # Return maximum losses if pointcloud is empty

    pointcloud = np.asarray(pointcloud)
    pointcloud_tgt = np.asarray(pointcloud_tgt)

    # Completeness: how far are the points of the target point cloud
    # from thre predicted point cloud
    completeness, completeness_normals = distance_p2p(pointcloud_tgt, normals_tgt, pointcloud, normals)
    recall = get_threshold_percentage(completeness, thresholds)
    completeness2 = completeness**2

    completeness_mean = completeness.mean()
    completeness_median = np.median(completeness)
    completeness2_mean = completeness2.mean()
    completeness2_median = np.median(completeness2)
    completeness_normals = completeness_normals.mean()

    # Accuracy: how far are th points of the predicted pointcloud
    # from the target pointcloud
    accuracy, accuracy_normals = distance_p2p(pointcloud, normals, pointcloud_tgt, normals_tgt)
    precision = get_threshold_percentage(accuracy, thresholds)
    accuracy2 = accuracy**2

    accuracy_mean = accuracy.mean()
    accuracy_median = np.median(accuracy)
    accuracy2_mean = accuracy2.mean()
    accuracy2_median = np.median(accuracy2)
    accuracy_normals = accuracy_normals.mean()
    # print(completeness,accuracy,completeness2,accuracy2)
    # Chamfer distance
    chamferL2_mean = 0.5 * (completeness2_mean + accuracy2_mean)
    chamferL2_median = 0.5 * (completeness2_median + accuracy2_median)
    normals_correctness = 0.5 * completeness_normals + 0.5 * accuracy_normals
    chamferL1_mean = 0.5 * (completeness_mean + accuracy_mean)
    chamferL1_median = 0.5 * (completeness_median + accuracy_median)

    # F-Score
    F = [2 * precision[i] * recall[i] / (precision[i] + recall[i]) for i in range(len(precision))]
    return (
        normals_correctness,
        chamferL1_mean,
        chamferL1_median,
        chamferL2_mean,
        chamferL2_median,
        F[0],
        F[1],
    )


def start_process_pool(worker_function, parameters, num_processes):
    """
    多线程进行器
    """
    if len(parameters) > 0:
        if num_processes <= 1:
            print(f"Running loop for {worker_function!s} with {len(parameters)} calls on {num_processes} workers")
            results = []
            for c in parameters:
                return [worker_function(*c) for c in parameters]
        print(
            f"Running loop for {worker_function!s} with {len(parameters)} calls on {num_processes} subprocess workers"
        )
        with multiprocessing.Pool(processes=num_processes, maxtasksperchild=1) as pool:
            results = pool.starmap(worker_function, parameters)
            return results
    else:
        return None


def count_parameters(model):
    # count the number of parameters in a given model
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_model_forward_flops(model, num_points=1):
    """Estimate one model forward pass using the common Linear-layer FLOPs convention."""
    macs_per_point = sum(
        module.in_features * module.out_features
        for module in model.modules()
        if isinstance(module, torch.nn.Linear)
    )
    flops_per_point = 2 * macs_per_point
    return {
        "macs_per_point": macs_per_point,
        "flops_per_point": flops_per_point,
        "num_points": num_points,
        "flops": flops_per_point * num_points,
    }


def setCUDA(gpu):
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")  # 显式指定设备
    torch.set_default_dtype(torch.float32)  # 设置默认数据类型
    return device


def print_config_tree(config, logger, indent=0):
    """递归打印 ConfigTree 或字典类型的配置项"""
    for key, value in config.items():
        if isinstance(value, ConfigTree):  # 如果值是 ConfigTree 类型,递归打印
            print_log(f"{' ' * indent}'{key}':", logger)
            print_config_tree(value, logger, indent + 4)  # 增加缩进
        else:
            print_log(f"{' ' * indent}'{key}': {value}", logger)


def conf_log(conf, logger):
    """打印配置信息,支持嵌套 ConfigTree 打印"""
    print_config_tree(conf, logger)


def info_CD(chamfer_dist, xyz1, xyz2, tau=0.5, lam=1e-7):
    # 你的扩展输出是“平方L2”,先转成 L2(论文推荐 L1,但 L2 也OK)
    dist1, dist2, _, _ = chamfer_dist(xyz1, xyz2)
    d1 = torch.sqrt(dist1)  # [B, N1]
    d2 = torch.sqrt(dist2)  # [B, N2]

    # y->x
    term1_g2p = (d2 / tau).mean(dim=1)
    lse_g2p = torch.logsumexp(-d2 / tau, dim=1)
    l_g2p = term1_g2p - lam * lse_g2p

    # x->y
    term1_p2g = (d1 / tau).mean(dim=1)
    lse_p2g = torch.logsumexp(-d1 / tau, dim=1)
    l_p2g = term1_p2g - lam * lse_p2g

    loss = 0.5 * (l_g2p + l_p2g)
    return loss.mean()


def load_input(data_dir, dataname, k_neighbors=30):
    """
    加载点云 (.npy/.xyz/.ply),统一用 PCA(KNN) 估计法线。
    如果输入自带法线 -> 计算与 PCA 法线的角度误差并上色,返回带颜色的点云和误差统计。
    如果输入不含法线 -> 返回空点云。

    参数:
        data_dir : Path 或 str
            输入数据目录
        dataname : str
            文件名(不含后缀)
        k_neighbors : int
            PCA 法线估计邻居数

    返回:
        points : (N,3) ndarray
        normals : (N,3) ndarray or None
        normals_pca : (N,3) ndarray
        error_pcd : trimesh.points.PointCloud
        error_stats : dict (包含平均误差等统计)
    """
    file_base = Path(data_dir) / "input" / dataname
    points, normals = None, None

    # ---------- 1) NPY 文件 ----------
    if (npy_file := file_base.with_suffix(".npy")).exists():
        arr = np.load(npy_file)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError("NPY 文件需形状 (N, >=3)")
        points = arr[:, :3].astype(np.float32)
        if arr.shape[1] >= 6:
            normals = arr[:, 3:6].astype(np.float32)

    # ---------- 2) XYZ 文件 ----------
    elif (xyz_file := file_base.with_suffix(".xyz")).exists():
        arr = np.loadtxt(xyz_file)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError("XYZ 文件需形状 (N, >=3)")
        points = arr[:, :3].astype(np.float32)
        normals = None

    # ---------- 3) PLY 文件 ----------
    elif (ply_file := file_base.with_suffix(".ply")).exists():
        pcd = o3d.io.read_point_cloud(str(ply_file))
        points = np.asarray(pcd.points, dtype=np.float32)
        if pcd.has_normals():
            normals = np.asarray(pcd.normals, dtype=np.float32)

    else:
        raise FileNotFoundError(f"未找到 {dataname} 的数据文件,仅支持 .npy / .xyz / .ply")

    # ---------- PCA 法线估计 ----------
    pcd_pca = o3d.geometry.PointCloud()
    pcd_pca.points = o3d.utility.Vector3dVector(points)
    pcd_pca.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors))
    pcd_pca.orient_normals_consistent_tangent_plane(k_neighbors)
    normals_pca = np.asarray(pcd_pca.normals, dtype=np.float32)

    # ---------- 如果没有输入法线 ----------
    if normals is None:
        return points, None, normals_pca, trimesh.points.PointCloud(np.empty((0, 3))), {}

    # ---------- 有输入法线 -> 计算误差 ----------
    ni = normals.copy()
    npca = normals_pca.copy()
    ni /= np.linalg.norm(ni, axis=1, keepdims=True) + 1e-12
    npca /= np.linalg.norm(npca, axis=1, keepdims=True) + 1e-12

    cos_val = np.abs(np.sum(ni * npca, axis=1))
    cos_val = np.clip(cos_val, 0.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_val))

    # ---------- 统计指标 ----------
    mean_error = np.mean(angle_deg)
    median_error = np.median(angle_deg)
    std_error = np.std(angle_deg)
    max_error = np.max(angle_deg)
    error_stats = {
        "mean": float(mean_error),
        "median": float(median_error),
        "std": float(std_error),
        "max": float(max_error),
    }

    # ---------- 映射到颜色 ----------
    vmax_deg = 60.0
    norm = colors.Normalize(vmin=0.0, vmax=float(vmax_deg))
    cmap = cm.get_cmap("jet")
    col = cmap(norm(np.clip(angle_deg, 0.0, vmax_deg)))[:, :3]
    pcd_pca.colors = o3d.utility.Vector3dVector(col)
    error_pcd = trimesh.points.PointCloud(vertices=points, colors=(col * 255).astype(np.uint8))

    return points, normals, normals_pca, error_pcd, error_stats


def back_up_code(code_root, save_root):
    code_root = Path(code_root).resolve()
    save_root = Path(save_root).resolve()

    if not code_root.exists():
        raise FileNotFoundError(f"root doesnt exist: {code_root}")

    save_root.mkdir(parents=True, exist_ok=True)
    save_root.mkdir(exist_ok=True)

    include_names = ["models", "tools", "confs", "scripts", "run.py"]

    for name in include_names:
        src_path = code_root / name
        dst_path = save_root / name

        if not src_path.exists():
            print(f"not find: {src_path}")
            continue

        try:
            if src_path.is_dir():
                shutil.copytree(
                    src_path,
                    dst_path,
                    ignore=shutil.ignore_patterns(
                        "__pycache__", "*.pyc", "*.pt", "*.pth", "*.log", ".git", ".vscode", "runs", "outputs"
                    ),
                )
            else:
                shutil.copy2(src_path, dst_path)
        except Exception as e:
            print(f"backup failed {name}: {e}")

    print("backup complete.")
