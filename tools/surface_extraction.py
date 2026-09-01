import mcubes
import numpy as np
import torch
import trimesh
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


def surface_extraction(ndf, grad, threshold, b_max, b_min, resolution):
    v_all = []
    t_all = []
    v_num = 0
    for i in tqdm(range(resolution - 1), desc="Extracting surface"):
        for j in range(resolution - 1):
            for k in range(resolution - 1):
                ndf_loc = ndf[i : i + 2]
                ndf_loc = ndf_loc[:, j : j + 2, :]
                ndf_loc = ndf_loc[:, :, k : k + 2]
                if np.min(ndf_loc) > threshold:
                    continue
                grad_loc = grad[i : i + 2]
                grad_loc = grad_loc[:, j : j + 2, :]
                grad_loc = grad_loc[:, :, k : k + 2]

                res = np.ones((2, 2, 2))
                for ii in range(2):
                    for jj in range(2):
                        for kk in range(2):
                            if np.dot(grad_loc[0][0][0], grad_loc[ii][jj][kk]) < 0:
                                res[ii][jj][kk] = -ndf_loc[ii][jj][kk]
                            else:
                                res[ii][jj][kk] = ndf_loc[ii][jj][kk]

                if res.min() < 0:
                    vertices, triangles = mcubes.marching_cubes(res, 0.0)
                    # print(vertices)
                    # vertices -= 1.5
                    # vertices /= 128
                    vertices[:, 0] += i  # / resolution
                    vertices[:, 1] += j  # / resolution
                    vertices[:, 2] += k  # / resolution
                    triangles += v_num
                    # vertices =
                    # vertices[:,1] /= 3  # TODO
                    v_all.append(vertices)
                    t_all.append(triangles)

                    v_num += vertices.shape[0]
                    # print(v_num)

    v_all = np.concatenate(v_all)
    t_all = np.concatenate(t_all)
    # Create mesh
    v_all = v_all / (resolution - 1.0) * (b_max - b_min)[None, :] + b_min[None, :]

    mesh = trimesh.Trimesh(v_all, t_all, process=False)

    return mesh