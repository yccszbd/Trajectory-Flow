import argparse  # noqa: INP001
import math
from pathlib import Path

import numpy as np
import point_cloud_utils as pcu
import torch
import torch.nn.functional as F
import trimesh
from models.dataset import Dataset
from models.fields import CAPUDFNetwork
from pyhocon import ConfigFactory
from tools.log2csv import parse_log_to_table
from tools.logger import get_root_logger, print_log
from tools.slice import extract_udf_slice, surface_extraction_2d
from tools.surface_extraction import surface_extraction
from tools.utils import (
    back_up_code,
    conf_log,
    count_model_forward_flops,
    count_parameters,
    eval_pointcloud,
    normalize_mesh,
    remove_far,
    set_seed,
    setCUDA,
)

# from tools.slice import save_all_slice_views
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm


def extract_fields(bound_min, bound_max, resolution, query_func, grad_func):
    N = 32
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    g = np.zeros([resolution, resolution, resolution, 3], dtype=np.float32)
    # with torch.no_grad():
    for xi, xs in tqdm(enumerate(X), total=len(X), desc="Calculate field"):
        for yi, ys in enumerate(Y):
            for zi, zs in enumerate(Z):
                xx, yy, zz = torch.meshgrid(xs, ys, zs)

                pts = torch.cat(
                    [xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)],
                    dim=-1,
                ).cuda()

                grad = grad_func(pts).reshape(len(xs), len(ys), len(zs), 3).detach().cpu().numpy()
                val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                u[
                    xi * N : xi * N + len(xs),
                    yi * N : yi * N + len(ys),
                    zi * N : zi * N + len(zs),
                ] = val
                g[
                    xi * N : xi * N + len(xs),
                    yi * N : yi * N + len(ys),
                    zi * N : zi * N + len(zs),
                ] = grad

    return u, g


def extract_geometry(
    bound_min,
    bound_max,
    resolution,
    threshold,
    query_func,
    grad_func,
):
    print(f"Extracting mesh with resolution: {resolution}")
    u, g = extract_fields(bound_min, bound_max, resolution, query_func, grad_func)
    b_max = bound_max.detach().cpu().numpy()
    b_min = bound_min.detach().cpu().numpy()
    mesh = surface_extraction(u, g, threshold, b_max, b_min, resolution)

    return mesh


class Runner:
    def __init__(self, args, conf_path):
        set_seed(123)
        self.device = setCUDA(args.gpu)
        # Configuration
        self.conf_path = Path(conf_path)
        conf_text = self.conf_path.read_text()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.dir = args.dir
        self.dataname = args.dataname
        self.base_exp_dir = Path(self.conf["general.base_exp_dir"]) / args.dir / args.dataname
        self.base_exp_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = Dataset(self.conf, self.dir, self.dataname, stage=1)
        self.GT_points = torch.from_numpy(self.dataset.points).to(self.device)
        self.batch_size = self.conf.get_int("train.batch_size")

        self.datalength = self.conf.get_int("train.datalength")
        self.epochs_stage_1 = self.conf.get_int("train.epochs_stage_1")
        self.epochs_stage_2 = self.conf.get_int("train.epochs_stage_2")
        self.epochs = self.epochs_stage_1 + self.epochs_stage_2

        self.learning_rate_stage1 = self.conf.get_float("train.learning_rate_stage1")
        self.learning_rate_stage2 = self.conf.get_float("train.learning_rate_stage2")

        self.warm_up_end_stage1 = self.conf.get_int("train.warm_up_end_stage1")
        self.warm_up_end_stage2 = self.conf.get_int("train.warm_up_end_stage2")

        self.metric = self.conf.get_string("train.metric")
        self.epoch_step = 0

        # Training parameters
        self.time_sum = self.conf.get_int("train.time_sum")
        self.report_freq = self.conf.get_int("train.report_freq")
        self.batch_size = self.conf.get_int("train.batch_size")

        self.extra_points_rate = self.conf.get_int("train.extra_points_rate")
        self.noise_range = self.conf.get_float("train.noise_range")

        # Networks
        self.udf_network = CAPUDFNetwork(self.time_sum, **self.conf["model.udf_network"]).to(self.device)

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.udf_network.parameters(), lr=self.learning_rate_stage1)


    def train(self):

        save_root = self.base_exp_dir / "code"
        code_root = Path("./")
        back_up_code(code_root, save_root)

        log_dir = self.base_exp_dir / "log"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"{self.epochs}_{self.time_sum}epoch_.log"
        self.writer = SummaryWriter(log_dir=log_dir)
        logger = get_root_logger(log_file=log_file, name="outs")
        self.logger = logger
        if self.conf.get_string("train.load_ckpt") != "None":
            self.load_checkpoint(self.conf.get_string("train.load_ckpt"))
            self.extract_mesh(
                resolution=256,  
                threshold=0.005, 
                point_gt=self.GT_points,
                epoch_step=self.epoch_step,
                time_sum=self.time_sum,
            )
        print_log(
            f"Message: {self.conf.get_string('train.message')}",
            logger=self.logger,
        )
        print_log(
            f"Dataset: Bounding Box:{self.dataset.bbox} \n Center:{self.dataset.shape_center} \n Scale:{self.dataset.shape_scale}",
            logger=self.logger,
        )
        (self.base_exp_dir / "pointcloud").mkdir(parents=True, exist_ok=True)
        self.error_stats = self.dataset.error_stats
        self.error_pointcloud = self.dataset.error_pointcloud
        if len(self.error_pointcloud.vertices) == 0:
            print_log("No Normals loaded.", logger=self.logger)
        else:
            error_pointcloud_path = self.base_exp_dir / "pointcloud" / "pca_error_point_cloud.ply"
            self.error_pointcloud.export(error_pointcloud_path)
            print_log(
                f"pca Error point cloud saved at {error_pointcloud_path}.",
                logger=self.logger,
            )
            print_log("pca_Normal Error Stats:")
            print_log(f"Mean={self.error_stats['mean']:.6f}", logger=self.logger)
            print_log(f"Median={self.error_stats['median']:.6f}", logger=self.logger)
            print_log(f"Std={self.error_stats['std']:.6f}", logger=self.logger)
            print_log(f"Max={self.error_stats['max']:.6f}", logger=self.logger)

        n_parameters = count_parameters(self.udf_network)
        print_log(
            f"Number of parameters in UDF network: {n_parameters}",
            logger=self.logger,
        )
        model_flops = count_model_forward_flops(self.udf_network, self.dataset.train_num_points)
        print_log(
            "Model FLOPs for one CAPUDFNetwork.forward() "
            "(Linear layers only, 1 MAC = 2 FLOPs):",
            logger=self.logger,
        )
        print_log(
            f"MACs per point: {model_flops['macs_per_point']:,}",
            logger=self.logger,
        )
        print_log(
            f"FLOPs per point: {model_flops['flops_per_point']:,} "
            f"({model_flops['flops_per_point'] / 1e6:.6f} MFLOPs)",
            logger=self.logger,
        )
        print_log(
            f"FLOPs for {model_flops['num_points']:,} points: "
            f"{model_flops['flops']:,} ({model_flops['flops'] / 1e9:.6f} GFLOPs)",
            logger=self.logger,
        )
        iter_sum = self.epochs * self.datalength
        print_log(
            f"Number of iter in training: {iter_sum}",
            logger=self.logger,
        )
        print_log(f"Training {self.dataname} with Conf {args.conf}", logger=self.logger)
        conf_log(self.conf, logger=self.logger)

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
        while self.epoch_step < self.epochs:

            if self.epoch_step == self.epochs_stage_1:
                print_log("进入第二阶段训练", logger=self.logger)
                self.dataset.stage = 2

                all_extra_points, all_extra_normals = self.get_extra_points(
                    1000000,
                    self.noise_range,
                )

                idx = pcu.downsample_point_cloud_poisson_disk(
                    all_extra_points,
                    num_samples=int(self.extra_points_rate * self.dataset.points.shape[0]),
                )
                extra_points = all_extra_points[idx]
                extra_normals = all_extra_normals[idx]

                (estimate_error_pointcloud, estimate_error_stats) = self.dataset.get_new_points(
                    extra_points, extra_normals
                )
                estimate_error_pointcloud_path = (
                    self.base_exp_dir / "pointcloud" / f"estimate_error_cloud{self.epoch_step}_{self.time_sum}epoch.ply"
                )

                estimate_error_pointcloud.export(estimate_error_pointcloud_path)
                print_log(
                    f"Estimate_Error_Point cloud saved successfully at epoch {self.epoch_step}. File path: {estimate_error_pointcloud_path}",
                    logger=self.logger,
                )
                print_log("estimate_Normal Error Stats:")
                print_log(f"Mean={estimate_error_stats['mean']:.6f}", logger=self.logger)
                print_log(f"Median={estimate_error_stats['median']:.6f}", logger=self.logger)
                print_log(f"Std={estimate_error_stats['std']:.6f}", logger=self.logger)
                print_log(f"Max={estimate_error_stats['max']:.6f}", logger=self.logger)

                current_dense_cloud = self.dataset.points
                final_dense_cloud = trimesh.points.PointCloud(vertices=current_dense_cloud)
                dense_path = (
                    self.base_exp_dir
                    / "pointcloud"
                    / f"FINAL_DENSE_{self.epoch_step}_{self.time_sum}epoch.ply"
                )
                final_dense_cloud.export(dense_path)
                print_log(f"----> 文件名: FINAL_DENSE_{self.epoch_step}_{self.time_sum}epoch.ply", logger=self.logger)
                print_log(f"----> 点数量: {current_dense_cloud.shape[0]}", logger=self.logger)
                count = current_dense_cloud.shape[0]
                if count > 10000:
                    self.dataset.train_num_points = 10000

                self.dataloader = torch.utils.data.DataLoader(
                    self.dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True,
                    persistent_workers=True,
                )
            self.epoch_step += 1
            self.update_learning_rate(self.epoch_step, self.epochs_stage_1, self.epochs_stage_2)
            for _, data in enumerate(self.dataloader):
                (
                    _,
                    sample_time,
                    _,
                    time,
                    res,
                ) = (
                    data["sample"].to(self.device),
                    data["sample_time"].to(self.device),
                    data["sample_near"].to(self.device),
                    data["time"].to(self.device),
                    data["res"].to(self.device),
                )
                # sample_gaussian_moved
                sample_time.requires_grad = True
                pred_gradients = self.udf_network.gradient(sample_time, time)  # 4*5000x3
                pred_res_udf = self.udf_network.res_udf(sample_time, time)  # 4*5000x1
                pred_grad_norm = F.normalize(pred_gradients, dim=-1)  # 4*5000x3
                pred_res = pred_res_udf * pred_grad_norm
                loss_res = F.l1_loss(pred_res, res)
                losses = {
                    "res": loss_res,
                }

                # Calculate total loss
                total_loss = sum(losses.values())
                losses["total"] = total_loss

                # Log losses to tensorboard
                for loss_name, loss_value in losses.items():
                    self.writer.add_scalar(f"Loss/{loss_name}", loss_value.item(), self.epoch_step)

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

            if self.epoch_step % self.report_freq == 0:
                for loss_name, loss_value in losses.items():
                    self.writer.add_scalar(f"Loss/{loss_name}", loss_value.item(), self.epoch_step)
                # Create dynamic loss string from dictionary
                loss_str = ", ".join([f"loss:{name} = {value:.6e}" for name, value in losses.items()])
                print_log(
                    "{}_{} epoch:{:8>d} {} lr={:.6e}".format(
                        self.dataname,
                        self.time_sum,
                        self.epoch_step,
                        loss_str,
                        self.optimizer.param_groups[0]["lr"],
                    ),
                    logger=logger,
                )
            if self.epoch_step in {self.epochs_stage_1, self.epochs}:
                self.save_checkpoint()
                self.extract_mesh(
                    resolution=args.mcube_resolution,
                    threshold=0.005,
                    point_gt=self.GT_points,
                    epoch_step=self.epoch_step,
                    time_sum=self.time_sum,
                )

        parse_log_to_table(log_file)


    def extract_mesh(
        self,
        resolution=64,
        threshold=0.005,
        point_gt=None,
        epoch_step=0,
        time_sum=0,
    ):
        bound_min = torch.tensor(self.dataset.bbox[0], dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.bbox[1], dtype=torch.float32)
        out_dir = self.base_exp_dir / "mesh"
        out_dir.mkdir(parents=True, exist_ok=True)

        mesh = extract_geometry(
            bound_min,
            bound_max,
            resolution=resolution,
            threshold=threshold,
            query_func=lambda pts: self.udf_network.predict(pts, self.time_sum, self.metric)[0],
            grad_func=lambda pts: self.udf_network.predict(pts, self.time_sum, self.metric)[1],
        )
        if self.conf.get_float("train.far") > 0:
            mesh = remove_far(point_gt.detach().cpu().numpy(), mesh, self.conf.get_float("train.far"))

        mesh_path = out_dir / f"{epoch_step}_{time_sum}epoch_mesh_noise0.1.obj"
        mesh.export(mesh_path)

        print_log(
            f"Mesh saved successfully at epoch {epoch_step}_{time_sum}epoch. File path: {mesh_path}",
            logger=self.logger,
        )

    def gen_cube_pointcloud(self, epoch_step, time_sum):
        device = self.device  
        points_batch_size = 100000
        resolution = 64  
        bounding_box = 1.1  
        p = torch.linspace(-bounding_box / 2, bounding_box / 2, resolution)
        px, py, pz = torch.meshgrid([p, p, p])
        points = torch.stack([px, py, pz], 3)
        points = points.view(-1, 3)
        p_split = torch.split(points, points_batch_size)  
        perd_points_all = [] 
        for pi in p_split:
            samples = pi.clone().to(device)
            # samples = pi.clone().to(device)
            _, _, perd_points = self.udf_network.predict(samples, self.time_sum, self.metric) 
            perd_points_all.append(perd_points.squeeze(0).detach().cpu())

        perd_points_all = torch.cat(perd_points_all, dim=0)

        perd_points_np = perd_points_all.cpu().numpy()

        cube_pointcloud_path = self.base_exp_dir / "pointcloud" / f"cube_point_cloud{epoch_step}_{time_sum}epoch.ply"

        trimesh.Trimesh(vertices=perd_points_np, process=False).export(cube_pointcloud_path)
        print_log(
            f"Cube_Point cloud saved successfully at epoch {epoch_step}. File path: {cube_pointcloud_path}",
            logger=self.logger,
        )

    def update_learning_rate(self, epoch_step, epochs_stage_1, epochs_stage_2):

        lr_stage1 = self.learning_rate_stage1
        lr_stage2 = self.learning_rate_stage2
        warmup_stage1 = min(self.warm_up_end_stage1, epochs_stage_1)
        warmup_stage2 = min(self.warm_up_end_stage2, epochs_stage_2)

        min_lr_stage1 = lr_stage1 * 0.005
        min_lr_stage2 = lr_stage2 * 0.005

        if epoch_step <= epochs_stage_1:
            local_epoch = epoch_step
            max_epoch_this_stage = epochs_stage_1
            warmup_this_stage = warmup_stage1
            base_lr = lr_stage1
            min_lr = min_lr_stage1 

        else:
            local_epoch = epoch_step - epochs_stage_1
            max_epoch_this_stage = epochs_stage_2
            warmup_this_stage = warmup_stage2
            base_lr = lr_stage2
            min_lr = min_lr_stage2  

        if local_epoch < warmup_this_stage and warmup_this_stage > 0:
            lr_factor = float(local_epoch) / float(warmup_this_stage)
            lr = min_lr + (base_lr - min_lr) * lr_factor
        else:
            t = local_epoch - warmup_this_stage
            T = max_epoch_this_stage - warmup_this_stage

            if T <= 0:
                lr = base_lr
            else:
                cos_factor = 0.5 * (1.0 + math.cos(math.pi * t / T))
                lr = min_lr + (base_lr - min_lr) * cos_factor

        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def load_checkpoint(self, checkpoint_name):
        checkpoint_path = self.base_exp_dir / "checkpoints" / checkpoint_name
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )
        print(checkpoint_path)
        self.dataset.set_point_cloud(checkpoint["points"], checkpoint["normals"], checkpoint["stage"])
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
        self.udf_network.load_state_dict(checkpoint["udf_network_fine"])

        self.epoch_step = checkpoint["epoch_step"]
        print_log(
            f"Checkpoint loaded successfully at epoch {self.epoch_step}.",
            logger=self.logger,
        )

    def save_checkpoint(self):
        checkpoint = {
            "udf_network_fine": self.udf_network.state_dict(),
            "epoch_step": self.epoch_step,
            "points": self.dataset.points,
            "normals": self.dataset.normals,
            "stage": self.dataset.stage,
        }
        checkpoint_dir = self.base_exp_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            checkpoint,
            checkpoint_dir / f"ckpt_{self.epoch_step:0>6d}_{self.time_sum}epoch.pth",
        )
        print_log(
            f"Checkpoint saved successfully at epoch {self.epoch_step}.",
            logger=self.logger,
        )

    def get_extra_points(self, extra_points, noise_range):
        batch_size = 10000
        all_extra_points = []
        all_extra_normals = []
        num_collected_points = 0

        while num_collected_points < extra_points:
            current_batch_size = min(batch_size, extra_points - num_collected_points)
            current_indices = np.random.choice(self.dataset.points.shape[0], current_batch_size, replace=True)
            # Gaussian noise points
            sample_surface = self.dataset.points[current_indices]
            sample_sigmas = self.dataset.sigmas[current_indices]
            theta_guassian = 0.25 * noise_range
            noise = np.random.normal(0.0, 1.0, size=(current_indices.shape[0], 3)).astype(np.float32)
            sample = sample_surface + theta_guassian * sample_sigmas * noise
            sample = torch.from_numpy(sample).to(self.device).float()
            sample.requires_grad = True
            _, sample_normal, sample_near = self.udf_network.predict(sample, self.time_sum, self.metric)
            sample_near = sample_near.detach().cpu().numpy()
            sample_normal = sample_normal.detach().cpu().numpy()
            gt_kd_tree = self.dataset.kd_tree
            distances, _ = gt_kd_tree.query(sample_near, p=2, distance_upper_bound=0.008)
            sample_near = sample_near[distances < 0.008]
            sample_normal = sample_normal[distances < 0.008]
            all_extra_points.append(sample_near)
            all_extra_normals.append(sample_normal)
            num_collected_points += sample_near.shape[0]
        all_extra_points = np.concatenate(all_extra_points, axis=0)
        all_extra_normals = np.concatenate(all_extra_normals, axis=0)
        return all_extra_points, all_extra_normals

    def evaluate(self, sample_num=100000):
        data_dir = Path(self.conf.get_string("general.data_dir"))
        gt_base = data_dir / self.dir / "ground_truth" / self.dataname
        if (gt_base.with_suffix(".ply")).is_file():
            gt_file = gt_base.with_suffix(".ply")
        elif (gt_base.with_suffix(".obj")).is_file():
            gt_file = gt_base.with_suffix(".obj")
        elif (gt_base.with_suffix(".xyz")).is_file():
            gt_file = gt_base.with_suffix(".xyz")
        else:
            raise FileNotFoundError(f"找不到 ground truth 文件: {gt_base}.[ply|obj|xyz]")
        input_base = data_dir / self.dir / "input" / self.dataname
        input_file = None
        if (input_base.with_suffix(".ply")).is_file():
            input_file = input_base.with_suffix(".ply")
        elif (input_base.with_suffix(".obj")).is_file():
            input_file = input_base.with_suffix(".obj")
        elif (input_base.with_suffix(".xyz")).is_file():
            input_file = input_base.with_suffix(".xyz")

        if input_file is None:
            print_log(
                f"Warning: Could not find input file at {input_base}.[ply|obj|xyz]. 将回退到使用 GT 进行归一化。",
                logger=self.logger,
            )
        pred_file = self.base_exp_dir / "mesh" / f"{self.epoch_step}_{self.time_sum}epoch_mesh.obj"
        print_log(f"pred_file:{pred_file}", logger=self.logger)
        print_log(f"gt_file:{gt_file}", logger=self.logger)
        gt_mesh = trimesh.load_mesh(gt_file)
        pred_mesh = trimesh.load_mesh(pred_file)
        gt_mesh = normalize_mesh(gt_mesh)
        print_log(f"使用 input file 进行归一化: {input_file}", logger=self.logger)
        inputshape = trimesh.load(input_file)
        total_size = (inputshape.bounds[1] - inputshape.bounds[0]).max()
        centers = (inputshape.bounds[1] + inputshape.bounds[0]) / 2
        pred_mesh.apply_scale(total_size)
        pred_mesh.apply_translation(centers)

        # pred_mesh = normalize_mesh(pred_mesh)

        # sample point for rec
        pts_pred, idx = pred_mesh.sample(sample_num, return_index=True)
        normals_rec = pred_mesh.face_normals[idx]
        # sample point for gt
        pts_gt = None
        normals_gt = None
        if isinstance(gt_mesh, trimesh.PointCloud):
            sample_num = min(sample_num, gt_mesh.vertices.shape[0])
            idx = np.random.choice(gt_mesh.vertices.shape[0], sample_num, replace=False)
            pts_gt = gt_mesh.vertices[idx]
            normals_gt = None
        elif isinstance(gt_mesh, trimesh.Trimesh):
            pts_gt, idx = gt_mesh.sample(sample_num, return_index=True)
            normals_gt = gt_mesh.face_normals[idx]
        elif isinstance(gt_mesh, trimesh.Scene):

            combined = gt_mesh.to_geometry() 
            pts_gt, idx = combined.sample(sample_num, return_index=True)
            pts_gt = pts_gt.astype(np.float32)
            normals_gt = combined.face_normals[idx]

        (
            normals_correctness,
            chamferL1_mean,
            chamferL1_median,
            chamferL2_mean,
            chamferL2_median,
            f_score_001,
            f_score_0005,
        ) = eval_pointcloud(pts_pred, pts_gt, normals_rec, normals_gt)

        print_log(f"dataset:{self.dir}", logger=self.logger)
        print_log(f"dataname:{self.dataname}", logger=self.logger)
        print_log(f"time_sum:{self.time_sum}", logger=self.logger)
        print_log(
            f"normals_correctness:{normals_correctness * 100:.4g}%",
            logger=self.logger,
        )
        print_log(f"chamferL1_mean:{chamferL1_mean * 1000:.4e}", logger=self.logger)
        print_log(
            f"chamferL1_median:{chamferL1_median * 1000:.4e}",
            logger=self.logger,
        )
        print_log(f"chamferL2_mean:{chamferL2_mean * 1000:.4e}", logger=self.logger)
        print_log(
            f"chamferL2_median:{chamferL2_median * 1000:.4e}",
            logger=self.logger,
        )
        print_log(f"F_score_0.01:{f_score_001 * 100:.4g}%", logger=self.logger)
        print_log(f"F_score_0.005:{f_score_0005 * 100:.4g}%", logger=self.logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default="./confs/ndf.conf")
    parser.add_argument("--mcube_resolution", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dir", type=str, default="test")
    parser.add_argument("--dataname", type=str, default="demo")
    args = parser.parse_args()

    runner = Runner(args, args.conf)

    runner.train()
