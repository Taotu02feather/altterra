# -*- coding: utf-8 -*-
"""
provinces.py — 纯 Voronoi 省份划分

流程：
1. 密度函数（承载力 + 地形复杂度 + 沿海加成）
2. 加权泊松盘采样撒种子点（密度高的地方点多）
3. Lloyd 松弛 5 次（Voronoi 质心迭代）
4. 纯 Voronoi 划分（scipy.spatial.Voronoi），直接得到凸多边形
5. 面积过滤：面积 < 中位数 10% 的省份合并到邻居

输出：provinces 列表，每项含 id / polygon / center / area / biome。
polygon 为 Voronoi 区域顶点（像素坐标，x=列, y=行），center 为像素坐标。
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree, Voronoi, ConvexHull

from . import config

# 最近一次 generate() 的省份标签图（0 = 海洋/无省份），供 render 使用
_LABELS = None


def label_map():
    """返回最近一次生成的省份标签图（像素 -> 省份 id，0 表示无省份）。"""
    return _LABELS


def _density(height, biome):
    """密度函数：承载力 + 地形复杂度 + 沿海加成。"""
    land = height >= 0
    # 承载力权重跨度较大，使肥沃区省份小而贫瘠区省份大（面积比约 20~100）
    cap_weights = np.array([0.0, 0.01, 0.05, 0.30, 0.85, 0.45, 0.01, 0.45, 2.00, 0.03])
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


def _lloyd(points, land, iters, weights=None):
    """Lloyd 松弛：把每个种子移到其 Voronoi 胞元的（密度加权）质心。

    weights 为逐像素密度场；按密度加权质心，使胞元面积 ∝ 1/密度（保留大小差异）。
    """
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    ys, xs = np.nonzero(land)
    land_coords = np.c_[ys.astype(np.float64), xs.astype(np.float64)]
    if weights is None:
        w = np.ones(len(ys), dtype=np.float64)
    else:
        w = weights[ys, xs].astype(np.float64)
    pts = np.asarray(points, dtype=np.float64)
    tree = cKDTree(pts)
    for _ in range(iters):
        _, lab = tree.query(land_coords)
        new_pts = np.empty_like(pts)
        for i in range(len(pts)):
            sel = lab == i
            if sel.any():
                ws = w[sel]
                sw = float(ws.sum())
                if sw > 0:
                    new_pts[i, 0] = float((ys[sel] * ws).sum() / sw)
                    new_pts[i, 1] = float((xs[sel] * ws).sum() / sw)
                else:
                    new_pts[i, 0] = float(ys[sel].mean())
                    new_pts[i, 1] = float(xs[sel].mean())
            else:
                new_pts[i] = pts[i]
        pts = new_pts
        tree = cKDTree(pts)
    return pts


def _filter_by_area(points, land):
    """删除陆地面积极小的种子（面积 < 中位数 10%），让邻居吸收其区域。"""
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    ys, xs = np.nonzero(land)
    land_coords = np.c_[ys.astype(np.float64), xs.astype(np.float64)]
    pts = np.asarray(points, dtype=np.float64)
    tree = cKDTree(pts)
    _, lab = tree.query(land_coords)
    counts = np.bincount(lab, minlength=len(pts)).astype(np.float64)
    pos = counts[counts > 0]
    if len(pos) == 0:
        return pts
    threshold = max(1.0, 0.10 * float(np.median(pos)))
    keep = counts >= threshold
    if keep.all():
        return pts
    return pts[keep]


def _voronoi_finite_polygons(vor, radius):
    """把 2D Voronoi 的无界区域重构为有界多边形，返回 (regions, vertices)。"""
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                t1 = vor.points[p2] - vor.points[p1]
                t1 = t1 / (np.linalg.norm(t1) + 1e-12)
                n = np.array([-t1[1], t1[0]])
                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v2] + direction * radius
                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())
            if v2 >= 0:
                t1 = vor.points[p2] - vor.points[p1]
                t1 = t1 / (np.linalg.norm(t1) + 1e-12)
                n = np.array([-t1[1], t1[0]])
                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v1] + direction * radius
                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())
        vs = np.asarray([new_vertices[v] for v in new_region])
        # 用凸包排序，保证得到严格凸多边形（角点区域比角度排序更稳健）
        if len(vs) >= 3:
            try:
                hull = ConvexHull(vs)
                new_region = [new_region[i] for i in hull.vertices]
            except Exception:
                c = vs.mean(axis=0)
                angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
                new_region = [new_region[i] for i in np.argsort(angles)]
        new_regions.append(new_region)
    return new_regions, np.asarray(new_vertices)


def _clip_polygon_rect(poly, xmin, ymin, xmax, ymax):
    """Sutherland-Hodgman 多边形裁剪到矩形。poly: (n,2) [x,y]。"""
    poly = np.asarray(poly, dtype=np.float64)
    if poly.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)

    def inside(p, e):
        x, y = p
        if e == 0:
            return x >= xmin
        if e == 1:
            return x <= xmax
        if e == 2:
            return y >= ymin
        return y <= ymax

    def inter(p1, p2, e):
        x1, y1 = p1
        x2, y2 = p2
        if e in (0, 1):
            x = xmin if e == 0 else xmax
            t = (x - x1) / (x2 - x1)
            return np.array([x, y1 + t * (y2 - y1)])
        y = ymin if e == 2 else ymax
        t = (y - y1) / (y2 - y1)
        return np.array([x1 + t * (x2 - x1), y])

    result = poly
    for e in range(4):
        if result.shape[0] == 0:
            return result
        new = []
        prev = result[-1]
        prev_in = inside(prev, e)
        for p in result:
            cur_in = inside(p, e)
            if cur_in:
                if not prev_in:
                    new.append(inter(prev, p, e))
                new.append(p)
            elif prev_in:
                new.append(inter(prev, p, e))
            prev = p
            prev_in = cur_in
        result = np.asarray(new, dtype=np.float64)
    return result


def _dedupe_polygon(poly):
    """移除相邻重复顶点（含首尾闭合重复），避免退化多边形。"""
    if poly.shape[0] == 0:
        return poly
    keep = [0]
    for i in range(1, poly.shape[0]):
        if np.abs(poly[i] - poly[keep[-1]]).max() > 1e-6:
            keep.append(i)
    poly = poly[keep]
    if poly.shape[0] > 1 and np.abs(poly[0] - poly[-1]).max() <= 1e-6:
        poly = poly[:-1]
    return poly


def _voronoi_polygons(points, w, h):
    """对每个种子点求 Voronoi 多边形（裁剪到 [0,w]x[0,h]，顶点为像素坐标 [x,y]）。"""
    pts_xy = np.asarray(points, dtype=np.float64)[:, ::-1]  # (y,x) -> (x,y)
    vor = Voronoi(pts_xy)
    radius = float(max(w, h)) * 4.0
    regions, vertices = _voronoi_finite_polygons(vor, radius)
    polys = []
    for region in regions:
        poly = vertices[region] if len(region) > 0 else np.zeros((0, 2))
        poly = _clip_polygon_rect(poly, 0.0, 0.0, float(w), float(h))
        poly = _dedupe_polygon(poly)
        polys.append([(float(x), float(y)) for x, y in poly])
    return polys


def _rasterize(points, land, biome, h, w):
    """离散 Voronoi 标签图（仅陆地）+ 每个种子的陆地面积与群系众数。"""
    n = len(points)
    labels = np.zeros((h, w), dtype=np.int32)
    areas = np.zeros(n, dtype=np.int64)
    biome_codes = np.zeros(n, dtype=np.int64)
    if n == 0:
        return labels, areas, biome_codes
    ys, xs = np.nonzero(land)
    land_coords = np.c_[ys.astype(np.float64), xs.astype(np.float64)]
    tree = cKDTree(np.asarray(points, dtype=np.float64))
    _, lab = tree.query(land_coords)
    labels[ys, xs] = (lab + 1).astype(np.int32)  # id 从 1 开始
    areas = np.bincount(lab, minlength=n).astype(np.int64)
    for i in range(n):
        m = lab == i
        if m.any():
            vals = biome[ys[m], xs[m]]
            biome_codes[i] = int(np.bincount(vals, minlength=10).argmax())
    return labels, areas, biome_codes


def generate(height, biome):
    """划分省份（纯 Voronoi，无分水岭）。"""
    h, w = height.shape
    land = height >= 0
    rng = np.random.default_rng(config.SEED + 3)
    n = config.NUM_PROVINCES

    density = _density(height, biome)
    land_count = int(land.sum())
    if land_count == 0:
        return []

    # 1. 加权泊松盘撒种子（略过采样，供面积过滤后仍接近 NUM_PROVINCES）
    n_seeds = int(round(n * 1.05))
    min_dist = max(6, int(round(np.sqrt(land_count / max(n_seeds, 1)) * 0.35)))
    points = _weighted_poisson_disk(density, land, n_seeds, rng, min_dist)
    if len(points) == 0:
        return []

    # 2. Lloyd 松弛 5 次（密度加权；用 density^2.5 强化密度对比，使面积比落到 20~100）
    weights = density ** 2.5
    points = _lloyd(points, land, 5, weights=weights)

    # 3. 面积过滤（面积 < 中位数 10% 的删除），再松弛收尾
    points = _filter_by_area(points, land)
    points = _lloyd(points, land, 3, weights=weights)

    # 4. 纯 Voronoi 多边形（裁剪到地图边界，顶点为像素坐标）
    polys = _voronoi_polygons(points, w, h)

    # 5. 离散 Voronoi 标签图（仅陆地）+ 面积 + 群系
    labels, areas, biome_codes = _rasterize(points, land, biome, h, w)

    # 缓存标签图供 render / resources 使用
    global _LABELS
    _LABELS = labels

    # 6. 组装省份列表
    provs = []
    for i in range(len(points)):
        provs.append({
            "id": i + 1,
            "polygon": polys[i],  # Voronoi 顶点 (x=列, y=行)
            "center": (float(points[i, 1]), float(points[i, 0])),  # (x, y) 像素坐标
            "area": int(areas[i]),
            "biome": config.BIOME_NAMES[int(biome_codes[i])],
        })
    return provs


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
        print("area: min={:.0f} max={:.0f} median={:.0f} mean={:.0f} ratio={:.1f}".format(
            areas.min(), areas.max(), np.median(areas), areas.mean(),
            areas.max() / areas.min()))
        print("total land covered by provinces: {:.2f}".format(
            float(areas.sum()) / float((height >= 0).sum())))
    for p in provs[:5]:
        print("  id={:>3d} center=({:.0f},{:.0f}) area={} biome={} poly_len={}".format(
            p["id"], p["center"][0], p["center"][1], p["area"], p["biome"], len(p["polygon"])))
    print("elapsed: {:.1f}s".format(time.time() - t0))
