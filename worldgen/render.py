# -*- coding: utf-8 -*-
"""
render.py — 地图渲染

输出 3 张 PNG 到 output/ 目录：
1. world_map.png — 世界地图（海洋深度渐变 + 群系 + 省份边界 + 资源标注）
2. capacity.png  — 农业承载力热力图（浅黄 -> 深绿）
3. provinces.png — 省份划分检查图（每省随机色 + 白色边界）

约束：无弹窗，只 savefig；所有坐标为像素坐标；imshow 显式 extent=[0, W, H, 0]。
"""

import os

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

from . import provinces as _prov_mod


# ---------------------------------------------------------------------------
# 配色
# ---------------------------------------------------------------------------
BIOME_COLORS = np.array([
    [0.30, 0.60, 0.85],   # 0 ocean（会被海洋色覆盖）
    [0.95, 0.97, 1.00],   # 1 ice
    [0.55, 0.58, 0.52],   # 2 tundra
    [0.20, 0.42, 0.25],   # 3 taiga
    [0.25, 0.55, 0.28],   # 4 temperate forest
    [0.72, 0.68, 0.35],   # 5 grassland
    [0.85, 0.78, 0.50],   # 6 desert
    [0.58, 0.62, 0.32],   # 7 savanna
    [0.08, 0.40, 0.16],   # 8 rainforest
    [0.55, 0.55, 0.58],   # 9 alpine
], dtype=np.float64)


def _ocean_rgb(height):
    """海洋按深度从浅蓝到深蓝渐变。"""
    depth = np.clip(-np.minimum(height, 0.0), 0.0, 8000.0) / 8000.0
    r = 0.40 * (1 - depth) + 0.02 * depth
    g = 0.72 * (1 - depth) + 0.10 * depth
    b = 0.90 * (1 - depth) + 0.28 * depth
    return np.stack([r, g, b], axis=-1)


def _hillshade(height, land):
    """简单山体阴影，让地形更有立体感（仅陆地）。"""
    gx, gy = np.gradient(height.astype(np.float64))
    gx = np.clip(gx, -150.0, 150.0) / 150.0
    gy = np.clip(gy, -150.0, 150.0) / 150.0
    nx, ny, nz = -gx, -gy, 1.0
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / norm, ny / norm, nz / norm
    lx, ly, lz = 0.5, -0.5, 0.707  # 光照来自西北上方
    shade = nx * lx + ny * ly + nz * lz
    return np.clip(np.where(land, shade, 1.0), 0.0, 1.0)


def _world_rgb(height, biome):
    """世界地图 RGB（海洋深度 + 群系 + 山体阴影）。"""
    land = height >= 0
    ocean = _ocean_rgb(height)
    land_col = BIOME_COLORS[np.clip(biome, 0, 9).astype(np.int64)]
    shade = _hillshade(height, land)[..., None]
    land_col = land_col * (0.72 + 0.45 * shade)
    return np.clip(np.where(land[..., None], land_col, ocean), 0.0, 1.0)


def _boundary_mask(labels):
    """省份之间的边界掩膜（不含海岸线）。"""
    h, w = labels.shape
    right = np.zeros((h, w), dtype=bool)
    down = np.zeros((h, w), dtype=bool)
    right[:, :-1] = (labels[:, :-1] != labels[:, 1:]) & (labels[:, :-1] > 0) & (labels[:, 1:] > 0)
    down[:-1, :] = (labels[:-1, :] != labels[1:, :]) & (labels[:-1, :] > 0) & (labels[1:, :] > 0)
    return right | down


def _resource_info(p):
    """返回省份的主导资源名与颜色。"""
    n = {
        "Agri": p["farm"],
        "Oil": np.log10(max(p["oil"], 1.0)) / 5.0,
        "Iron": np.log10(max(p["iron"], 1.0)) / 5.0,
        "Coal": np.log10(max(p["coal"], 1.0)) / 5.0,
    }
    dom = max(n, key=n.get)
    colors = {"Agri": (0.1, 0.5, 0.1), "Oil": (0.7, 0.1, 0.1),
              "Iron": (0.35, 0.35, 0.35), "Coal": (0.2, 0.1, 0.05)}
    return dom, colors[dom]


def _importance(p):
    """省份资源重要性评分。"""
    oil_n = np.log10(max(p["oil"], 1.0)) / 5.0
    iron_n = np.log10(max(p["iron"], 1.0)) / 5.0
    coal_n = np.log10(max(p["coal"], 1.0)) / 5.0
    return (0.30 * p["farm"] + 0.10 * p["pasture"] + 0.05 * p["fish"]
            + 0.20 * oil_n + 0.20 * iron_n + 0.15 * coal_n)


# ---------------------------------------------------------------------------
# 三张图
# ---------------------------------------------------------------------------
def _render_world_map(height, biome, provs):
    W, H = height.shape[1], height.shape[0]
    rgb = _world_rgb(height, biome)

    fig = plt.figure(figsize=(W / 100.0, H / 100.0), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb, extent=[0, W, H, 0], interpolation='nearest')
    ax.set_axis_off()

    # 省份边界：细黑线，透明度 0.3
    labels = _prov_mod.label_map()
    if labels is not None:
        bnd = _boundary_mask(labels)
        overlay = np.zeros((H, W, 4), dtype=np.float64)
        overlay[..., 3] = np.where(bnd, 0.3, 0.0)
        ax.imshow(overlay, extent=[0, W, H, 0], interpolation='nearest')

    # 资源标注：前 50 个最重要的资源省
    top = sorted(provs, key=_importance, reverse=True)[:50]
    for p in top:
        cx, cy = p["center"]
        y = H - cy  # 像素行 -> 数据坐标（imshow extent 翻转 y）
        dom, color = _resource_info(p)
        ax.plot(cx, y, marker='o', markersize=3.2, color=color,
                markeredgecolor='black', markeredgewidth=0.3)
        ax.annotate(dom, (cx, y), xytext=(0, 4), textcoords='offset points',
                    fontsize=4, color=color, ha='center', va='bottom')

    fig.savefig(os.path.join('output', 'world_map.png'))
    plt.close(fig)


def _ocean_overlay(land, color):
    """生成海洋覆盖 RGBA（海洋不透明、陆地透明），用于盖回多边形延伸到海洋的部分。"""
    h, w = land.shape
    overlay = np.zeros((h, w, 4), dtype=np.float64)
    overlay[~land, 0] = color[0]
    overlay[~land, 1] = color[1]
    overlay[~land, 2] = color[2]
    overlay[~land, 3] = 1.0
    return overlay


def _polygon_data(p, H):
    """把省份多边形顶点（像素坐标 x=列,y=行）转成数据坐标 (x, H-y)。"""
    xs = [pt[0] for pt in p["polygon"]]
    ys = [H - pt[1] for pt in p["polygon"]]
    return xs, ys


def _render_capacity(height, provs):
    W, H = height.shape[1], height.shape[0]
    land = height >= 0
    cmap = plt.get_cmap('YlGn')

    fig = plt.figure(figsize=(W / 100.0, H / 100.0), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)

    # 每个省份用 matplotlib Polygon 填充，颜色由该省 farm 潜力决定（浅黄 -> 深绿）
    for p in provs:
        if len(p["polygon"]) < 3:
            continue
        xs, ys = _polygon_data(p, H)
        ax.fill(xs, ys, facecolor=cmap(p["farm"]), edgecolor='none')

    # 盖回海洋（多边形是裁剪到地图边界的 Voronoi 区域，会延伸到海洋）
    ax.imshow(_ocean_overlay(land, (0.04, 0.12, 0.35)),
              extent=[0, W, H, 0], interpolation='nearest', origin='upper')

    ax.set_axis_off()
    fig.savefig(os.path.join('output', 'capacity.png'))
    plt.close(fig)


def _render_provinces(height, provs):
    W, H = height.shape[1], height.shape[0]
    land = height >= 0
    n = len(provs)

    # 黄金角分配色相，保证相邻省份颜色区分（确定性）
    hues = (np.arange(n, dtype=np.float64) * 0.61803398875 + 0.15) % 1.0
    sat = np.full(n, 0.62)
    val = np.full(n, 0.92)
    colors = hsv_to_rgb(np.stack([hues, sat, val], axis=1))

    fig = plt.figure(figsize=(W / 100.0, H / 100.0), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)

    # 每省随机色填充 + 白色细边界
    for i, p in enumerate(provs):
        if len(p["polygon"]) < 3:
            continue
        xs, ys = _polygon_data(p, H)
        ax.fill(xs, ys, facecolor=colors[i], edgecolor='white', linewidth=0.4)

    ax.imshow(_ocean_overlay(land, (0.13, 0.18, 0.32)),
              extent=[0, W, H, 0], interpolation='nearest', origin='upper')

    ax.set_axis_off()
    fig.savefig(os.path.join('output', 'provinces.png'))
    plt.close(fig)


def render_all(height, biome, provs, temp, rain):
    """渲染三张地图。"""
    os.makedirs('output', exist_ok=True)
    _render_world_map(height, biome, provs)
    _render_capacity(height, provs)
    _render_provinces(height, provs)


if __name__ == "__main__":
    import time
    from . import config, terrain, climate, provinces as prov, resources

    t0 = time.time()
    height = terrain.generate()
    temp, rain, biome = climate.generate(height)
    provs = prov.generate(height, biome)
    resources.assign(provs, height, temp, rain, biome)
    render_all(height, biome, provs, temp, rain)
    print("rendered 3 PNGs, elapsed {:.1f}s".format(time.time() - t0))
