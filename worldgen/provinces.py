# -*- coding: utf-8 -*-
"""
provinces.py — Voronoi + 分水岭省份划分

流程：
1. 密度函数（承载力 + 地形复杂度 + 沿海加成）
2. 加权泊松盘采样撒种子点（密度高的地方点多）
3. Lloyd 松弛 2 次（Voronoi 质心迭代）
4. 离散 Voronoi 划分（最近邻，用 cKDTree 实现）
5. scipy.ndimage.watershed_ift 沿山脊修正边界

输出：provinces 列表，每项含 id / polygon / center / area / biome。
polygon 与 center 均为像素坐标（x=列, y=行）。
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, watershed_ift, binary_dilation, gaussian_filter
from scipy.spatial import cKDTree

from . import config

# 最近一次 generate() 的省份标签图（0 = 海洋/无省份），供 render 使用
_LABELS = None


def label_map():
    """返回最近一次生成的省份标签图（像素 -> 省份 id，0 表示无省份）。"""
    return _LABELS


def _density(height, biome):
    """密度函数：承载力 + 地形复杂度 + 沿海加成。"""
    land = height >= 0
    cap_weights = np.array([0.0, 0.05, 0.15, 0.40, 0.80, 0.50, 0.05, 0.50, 1.00, 0.10])
    capacity = cap_weights[np.clip(biome, 0, 9).astype(np.int64)]
    gx = np.gradient(np.maximum(height, 0.0), axis=1)
    gy = np.gradient(np.maximum(height, 0.0), axis=0)
    slope = np.hypot(gx, gy)
    complexity = np.clip(slope / 250.0, 0.0, 1.0) * 0.2
    dist = distance_transform_edt(land)  # 陆地到最近海洋的距离
    coastal = 0.25 * np.exp(-dist / 100.0)
    density = capacity + complexity + coastal
    return np.where(land, density, 0.0)


def _weighted_poisson_disk(density, land, n, rng, min_dist):
    """按密度加权的泊松盘采样（拒绝采样 + 最小距离约束）。"""
    h, w = density.shape
    flat = density.ravel()
    total = float(flat.sum())
    if total <= 0:
        ys, xs = np.nonzero(land)
        if len(ys) == 0:
            return []
        idx = rng.choice(len(ys), size=min(n, len(ys)), replace=False)
        return list(zip(ys[idx].tolist(), xs[idx].tolist()))
    probs = flat / total
    n_candidates = n * 40
    idx = rng.choice(h * w, size=n_candidates, p=probs, replace=True)
    cand_y = (idx // w).astype(np.int64)
    cand_x = (idx % w).astype(np.int64)
    points = []
    md2 = min_dist * min_dist
    for k in range(n_candidates):
        if len(points) >= n:
            break
        y = int(cand_y[k])
        x = int(cand_x[k])
        ok = True
        for py, px in points:
            if (py - y) * (py - y) + (px - x) * (px - x) < md2:
                ok = False
                break
        if ok:
            points.append((y, x))
    if len(points) < n:
        ys, xs = np.nonzero(land)
        need = n - len(points)
        if len(ys) > 0:
            idx2 = rng.choice(len(ys), size=min(need, len(ys)), replace=False)
            for j in idx2:
                points.append((int(ys[j]), int(xs[j])))
    return points[:n]


def _lloyd(points, land, iters):
    """Lloyd 松弛：把每个种子移到其 Voronoi 胞元的质心。"""
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    ys, xs = np.nonzero(land)
    land_coords = np.c_[ys.astype(np.float64), xs.astype(np.float64)]
    pts = np.asarray(points, dtype=np.float64)
    tree = cKDTree(pts)
    for _ in range(iters):
        _, lab = tree.query(land_coords)
        new_pts = np.empty_like(pts)
        for i in range(len(pts)):
            sel = lab == i
            if sel.any():
                new_pts[i, 0] = ys[sel].mean()
                new_pts[i, 1] = xs[sel].mean()
            else:
                new_pts[i] = pts[i]
        pts = new_pts
        tree = cKDTree(pts)
    return pts


def _merge_small(labels, land, min_area):
    """把面积过小的省份并入共享边界最长的相邻省份，清除碎屑。"""
    ids, counts = np.unique(labels[land], return_counts=True)
    struct = np.ones((3, 3), dtype=bool)
    for pid, cnt in zip(ids, counts):
        if cnt >= min_area or pid == 0:
            continue
        mask = labels == pid
        nb = binary_dilation(mask, structure=struct)
        nbr = labels[nb & ~mask]
        nbr = nbr[(nbr != pid) & (nbr != 0)]
        if len(nbr) == 0:
            continue
        uniq, cnts = np.unique(nbr, return_counts=True)
        target = uniq[np.argmax(cnts)]
        labels[mask] = target
    return labels


def _extract(labels, height, biome, land):
    """从标签图提取省份信息。"""
    from scipy.ndimage import binary_erosion, generate_binary_structure

    struct = generate_binary_structure(2, 2)  # 8 连通
    provinces = []
    ids = np.unique(labels)
    ids = ids[ids > 0]
    for pid in ids:
        mask = labels == pid
        area = int(mask.sum())
        if area == 0:
            continue
        ys, xs = np.nonzero(mask)
        cx = float(xs.mean())
        cy = float(ys.mean())
        b = biome[mask]
        biome_code = int(np.bincount(b.ravel(), minlength=10).argmax())
        # 边界像素，按质心角度排序得到多边形（x=列, y=行）
        boundary = mask & ~binary_erosion(mask, structure=struct)
        yb, xb = np.nonzero(boundary)
        if len(yb) >= 3:
            ang = np.arctan2(yb - cy, xb - cx)
            order = np.argsort(ang)
            poly = [(float(xb[i]), float(yb[i])) for i in order]
        else:
            poly = [(float(x), float(y)) for y, x in zip(ys, xs)]
        provinces.append({
            "id": int(pid),
            "polygon": poly,
            "center": (cx, cy),  # 像素坐标 (x=列, y=行)
            "area": area,
            "biome": config.BIOME_NAMES[biome_code],
        })
    return provinces


def generate(height, biome):
    """划分省份。"""
    h, w = height.shape
    land = height >= 0
    rng = np.random.default_rng(config.SEED + 3)
    n = config.NUM_PROVINCES

    density = _density(height, biome)
    land_count = int(land.sum())
    if land_count == 0:
        return []

    # 过采样种子，以抵消后续合并碎屑，最终接近 NUM_PROVINCES
    n_seeds = int(round(n * 1.8))
    # 最小间距：按平均胞元间距缩放
    min_dist = max(6, int(round(np.sqrt(land_count / max(n_seeds, 1)) * 0.7)))
    points = _weighted_poisson_disk(density, land, n_seeds, rng, min_dist)
    points = _lloyd(points, land, 2)

    # 最近陆地索引（用于把种子吸附到陆地）
    _, land_idx = distance_transform_edt(~land, return_indices=True)

    # 种子取整并吸附到陆地，去重后写入标记
    seen = set()
    markers = np.zeros((h, w), dtype=np.int32)
    label = 1
    for y, x in points:
        yy = int(round(y))
        xx = int(round(x))
        yy = min(max(yy, 0), h - 1)
        xx = min(max(xx, 0), w - 1)
        if not land[yy, xx]:
            yy = int(land_idx[0, yy, xx])
            xx = int(land_idx[1, yy, xx])
        if (yy, xx) in seen:
            continue
        seen.add((yy, xx))
        markers[yy, xx] = label
        label += 1

    # 成本图：海拔作为成本（低=谷、高=脊），叠加中波长平滑噪声，使平坦低地也能
    # 被自然切分（近似均匀），而山地仍以山脊为主；海洋设为极大值充当分水岭屏障
    smooth = gaussian_filter(rng.random((h, w)), sigma=25.0)
    smooth = (smooth - smooth.mean()) / (smooth.std() + 1e-9)
    cost = np.where(land, height.astype(np.float64) + 8000.0 + smooth * 350.0, 65535.0)
    cost = np.clip(cost, 0.0, 65535.0).astype(np.uint16)
    labels = watershed_ift(cost, markers)
    labels = np.where(land, labels, 0)

    # 清除过小的碎屑省份
    labels = _merge_small(labels, land, min_area=100)

    # 缓存标签图供 render 使用
    global _LABELS
    _LABELS = labels

    return _extract(labels, height, biome, land)


if __name__ == "__main__":
    import time
    from . import terrain, climate

    t0 = time.time()
    height = terrain.generate()
    _, _, biome = climate.generate(height)
    provs = generate(height, biome)
    areas = np.array([p["area"] for p in provs], dtype=np.float64)
    print("provinces: {}".format(len(provs)))
    if len(provs):
        print("area: min={:.0f} max={:.0f} mean={:.0f}".format(
            areas.min(), areas.max(), areas.mean()))
        print("total land covered by provinces: {:.2f}".format(
            float(areas.sum()) / float((height >= 0).sum())))
    for p in provs[:5]:
        print("  id={:>3d} center=({:.0f},{:.0f}) area={} biome={} poly_len={}".format(
            p["id"], p["center"][0], p["center"][1], p["area"], p["biome"], len(p["polygon"])))
    print("elapsed: {:.1f}s".format(time.time() - t0))
