import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree


def extract_udf_slice(
    bound_min,
    bound_max,
    resolution,
    slice_axis,
    slice_value,
    query_func,
    grad_func,
):
    val, grad, extent, axis_labels = extract_slice_fields(
        bound_min=bound_min,
        bound_max=bound_max,
        resolution=resolution,
        slice_axis=slice_axis,
        slice_value=slice_value,
        query_func=query_func,
        grad_func=grad_func,
    )

    return val, grad, extent, axis_labels


def extract_slice_fields(
    bound_min,
    bound_max,
    resolution,
    slice_axis,
    slice_value,
    query_func,
    grad_func,
):
    N = 64

    if slice_axis == "x":
        assert bound_min[0] <= slice_value <= bound_max[0], "slice_value 超出 bbox 范围"
        A = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)  # y
        B = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)  # z
        axis_labels = ("y", "z")
        extent = [bound_min[1], bound_max[1], bound_min[2], bound_max[2]]

    elif slice_axis == "y":
        assert bound_min[1] <= slice_value <= bound_max[1], "slice_value 超出 bbox 范围"
        A = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)  # x
        B = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)  # z
        axis_labels = ("x", "z")
        extent = [bound_min[0], bound_max[0], bound_min[2], bound_max[2]]

    elif slice_axis == "z":
        assert bound_min[2] <= slice_value <= bound_max[2], "slice_value 超出 bbox 范围"
        A = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)  # x
        B = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)  # y
        axis_labels = ("x", "y")
        extent = [bound_min[0], bound_max[0], bound_min[1], bound_max[1]]

    else:
        raise ValueError("slice_axis must be one of ['x', 'y', 'z']")

    val = np.zeros((resolution, resolution), dtype=np.float32)
    grad = np.zeros((resolution, resolution, 3), dtype=np.float32)

    for ai, a_chunk in enumerate(A):
        for bi, b_chunk in enumerate(B):
            aa, bb = torch.meshgrid(a_chunk, b_chunk, indexing="ij")

            if slice_axis == "x":
                pts = torch.stack(
                    [
                        torch.full_like(aa, fill_value=slice_value),
                        aa,
                        bb,
                    ],
                    dim=-1,
                )

            elif slice_axis == "y":
                pts = torch.stack(
                    [
                        aa,
                        torch.full_like(aa, fill_value=slice_value),
                        bb,
                    ],
                    dim=-1,
                )

            else:  # "z"
                pts = torch.stack(
                    [
                        aa,
                        bb,
                        torch.full_like(aa, fill_value=slice_value),
                    ],
                    dim=-1,
                )

            pts = pts.reshape(-1, 3).cuda()

            g = (
                grad_func(pts)
                .reshape(len(a_chunk), len(b_chunk), 3)
                .detach()
                .cpu()
                .numpy()
            )
            u = (
                query_func(pts)
                .reshape(len(a_chunk), len(b_chunk))
                .detach()
                .cpu()
                .numpy()
            )

            val[
                ai * N : ai * N + len(a_chunk),
                bi * N : bi * N + len(b_chunk),
            ] = u

            grad[
                ai * N : ai * N + len(a_chunk),
                bi * N : bi * N + len(b_chunk),
            ] = g

    return val, grad, extent, axis_labels


def _interp_zero_crossing(v1, v2, p1, p2, eps=1e-12):
    if abs(v1) < eps and abs(v2) < eps:
        return None
    if abs(v1) < eps:
        return np.asarray(p1, dtype=np.float64)
    if abs(v2) < eps:
        return np.asarray(p2, dtype=np.float64)
    if v1 * v2 > 0:
        return None

    t = v1 / (v1 - v2)
    return (1.0 - t) * np.asarray(p1, dtype=np.float64) + t * np.asarray(
        p2, dtype=np.float64
    )


def surface_extraction_2d(udf, grad, extent, eps=1e-10, far=None):
    """
    udf  : (H, W)
    grad : (H, W, 3)
    extent = [xmin, xmax, ymin, ymax]

    return:
        segments: [ (p0, p1), ... ]
        每个 p0/p1 都是物理坐标系下的 2D 点 [x, y]
    """
    H, W = udf.shape
    xmin, xmax, ymin, ymax = extent

    dx = (xmax - xmin) / (H - 1)
    dy = (ymax - ymin) / (W - 1)

    segments = []

    local_pts = [
        np.array([0.0, 0.0], dtype=np.float64),
        np.array([1.0, 0.0], dtype=np.float64),
        np.array([1.0, 1.0], dtype=np.float64),
        np.array([0.0, 1.0], dtype=np.float64),
    ]

    edge_defs = [
        (0, 1),  # bottom
        (1, 2),  # right
        (2, 3),  # top
        (3, 0),  # left
    ]

    corner_adj_edges = {
        0: (3, 0),
        1: (0, 1),
        2: (1, 2),
        3: (2, 3),
    }

    def to_world(i, j, p_local):
        x = xmin + (i + p_local[0]) * dx
        y = ymin + (j + p_local[1]) * dy
        return np.array([x, y], dtype=np.float64)

    for i in range(H - 1):
        for j in range(W - 1):
            d = np.array(
                [
                    udf[i, j],
                    udf[i + 1, j],
                    udf[i + 1, j + 1],
                    udf[i, j + 1],
                ],
                dtype=np.float64,
            )

            if far is not None and d.min() > far:
                continue

            g = np.array(
                [
                    grad[i, j],
                    grad[i + 1, j],
                    grad[i + 1, j + 1],
                    grad[i, j + 1],
                ],
                dtype=np.float64,
            )

            ref_idx = int(np.argmin(d))
            g_ref = g[ref_idx]

            phi = d.copy()
            for k in range(4):
                if np.dot(g_ref, g[k]) < 0:
                    phi[k] = -phi[k]

            phi[np.abs(phi) < eps] = eps

            if not (phi.min() < 0.0 and phi.max() > 0.0):
                continue

            edge_points = {}
            for e_idx, (a, b) in enumerate(edge_defs):
                p = _interp_zero_crossing(
                    phi[a], phi[b], local_pts[a], local_pts[b], eps=eps
                )
                if p is not None:
                    edge_points[e_idx] = p

            n_cross = len(edge_points)

            if n_cross == 2:
                pts = list(edge_points.values())
                p0 = to_world(i, j, pts[0])
                p1 = to_world(i, j, pts[1])
                segments.append((p0, p1))

            elif n_cross == 4:
                center_val = phi.mean()
                neg = phi < 0.0

                target_corners = np.where(neg if center_val >= 0.0 else ~neg)[0]

                if len(target_corners) == 2:
                    for c in target_corners:
                        e0, e1 = corner_adj_edges[int(c)]
                        p0 = to_world(i, j, edge_points[e0])
                        p1 = to_world(i, j, edge_points[e1])
                        segments.append((p0, p1))

    return segments


def filter_segments_2d(segments, points_np, axis_idx, slice_value, far=0.005):
    mask = np.abs(points_np[:, axis_idx] - slice_value) < far
    pts_2d = points_np[mask]
    pts_2d = pts_2d[:, [i for i in range(3) if i != axis_idx]]
    if len(pts_2d) == 0:
        return []
    tree = cKDTree(pts_2d)
    filtered = []
    for p0, p1 in segments:
        mid = (p0 + p1) / 2
        d, _ = tree.query(mid, distance_upper_bound=far)
        if d < far:
            filtered.append((p0, p1))
    return filtered


def render_UDF_image(
    val,
    extent,
    segments,
    save_path,
    vmin=0.0,
    vmax=0.1,
    levels_step=0.01,
    cmap="Blues_r",
    dpi=400,
    figsize=(6, 6),
):
    nx, ny = val.shape
    x = np.linspace(extent[0], extent[1], nx)
    y = np.linspace(extent[2], extent[3], ny)
    XX, YY = np.meshgrid(x, y, indexing="ij")

    levels = np.arange(vmin, vmax + 1e-8, levels_step)

    fig, ax = plt.subplots(figsize=figsize)
    ax.contourf(
        XX,
        YY,
        val,
        levels=levels,
        cmap=cmap,
        extend="both",
        antialiased=True,
    )
    ax.contour(XX, YY, val, levels=levels, colors="0.65", linewidths=0.7)
    for p0, p1 in segments:
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            color="k",
            linewidth=1.2,
            solid_capstyle="round",
        )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def render_gradient_image(
    val,
    grad,
    extent,
    axis,
    save_path,
    stride=8,
    cmap="YlOrRd",
    clim=(-0.05, 0.1),
    dpi=220,
    figsize=(6, 6),
):
    grad_plane_idx = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}[axis]

    nx, ny = val.shape
    x = np.linspace(extent[0], extent[1], nx)
    y = np.linspace(extent[2], extent[3], ny)
    XX, YY = np.meshgrid(x, y, indexing="ij")

    gx = grad[::stride, ::stride, grad_plane_idx[0]]
    gy = grad[::stride, ::stride, grad_plane_idx[1]]
    g_norm = np.sqrt(gx**2 + gy**2)
    g_norm = np.where(g_norm > 1e-10, g_norm, 1.0)
    gx = -gx / g_norm
    gy = -gy / g_norm

    QX = XX[::stride, ::stride]
    QY = YY[::stride, ::stride]
    QV = val[::stride, ::stride]

    arrow_len = 0.15 + QV * 1.5
    gx = gx * arrow_len
    gy = gy * arrow_len

    fig, ax = plt.subplots(figsize=figsize)
    ax.quiver(
        QX,
        QY,
        gx,
        gy,
        QV,
        cmap=cmap,
        clim=clim,
        scale=4.0,
        width=0.004,
        headwidth=4,
        headlength=5,
    )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout(pad=0.05)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def render_overlay_image(
    val,
    grad,
    extent,
    segments,
    axis,
    save_path,
    stride=8,
    vmin=0.0,
    vmax=0.1,
    levels_step=0.01,
    cmap="Blues_r",
    flow_cmap="YlOrRd",
    flow_clim=(-0.05, 0.1),
    dpi=400,
    figsize=(6, 6),
):
    grad_plane_idx = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}[axis]

    nx, ny = val.shape
    x = np.linspace(extent[0], extent[1], nx)
    y = np.linspace(extent[2], extent[3], ny)
    XX, YY = np.meshgrid(x, y, indexing="ij")

    gx = grad[::stride, ::stride, grad_plane_idx[0]]
    gy = grad[::stride, ::stride, grad_plane_idx[1]]
    g_norm = np.sqrt(gx**2 + gy**2)
    g_norm = np.where(g_norm > 1e-10, g_norm, 1.0)
    gx = -gx / g_norm
    gy = -gy / g_norm

    QX = XX[::stride, ::stride]
    QY = YY[::stride, ::stride]
    QV = val[::stride, ::stride]

    arrow_len = 0.15 + QV * 1.5
    gx = gx * arrow_len
    gy = gy * arrow_len

    levels = np.arange(vmin, vmax + 1e-8, levels_step)

    fig, ax = plt.subplots(figsize=figsize)
    ax.contourf(XX, YY, val, levels=levels, cmap=cmap, extend="both", antialiased=True)
    ax.contour(XX, YY, val, levels=levels, colors="0.65", linewidths=0.7)
    for p0, p1 in segments:
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            color="k",
            linewidth=1.2,
            solid_capstyle="round",
        )
    ax.quiver(
        QX,
        QY,
        gx,
        gy,
        QV,
        cmap=flow_cmap,
        clim=flow_clim,
        scale=4.0,
        width=0.004,
        headwidth=4,
        headlength=5,
        zorder=5,
    )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout(pad=0.05)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)