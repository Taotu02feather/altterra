# -*- coding: utf-8 -*-
"""
resources.py — 资源三层系统（农业 / 牧业 / 渔业 + 矿产）

输入：provinces, height, temp, rain, biome
输出：就地修改 provinces，每项加入 farm / pasture / fish / oil / iron / coal。

- 矿产 reserve 用对数分布，跨 3~4 个数量级（10 ~ 100000）。
- 石油：低频噪声 × 沉积盆地掩膜（低海拔 + 平坦）。
- 铁矿：低频噪声 × 造山带加成（高海拔 + 梯度大）。
- 煤矿：低频噪声 × 古盆地（中低海拔 + 平坦）。
"""

import numpy as np
from scipy.ndimage import distance_transform_edt

from . import config, terrain
from . import provinces as _prov_mod


def _slope(height):
    gx = np.gradient(np.maximum(height, 0.0), axis=1)
    gy = np.gradient(np.maximum(height, 0.0), axis=0)
    return np.hypot(gx, gy)


def _farm_potential(height, temp, rain, slope):
    """农业潜力：温度、降水、坡度综合（0~1）。"""
    land = height >= 0
    temp_ok = np.exp(-((temp - 22.0) / 15.0) ** 2)
    rain_ok = np.exp(-((rain - 0.65) / 0.4) ** 2)
    slope_ok = np.clip(1.0 - slope / 250.0, 0.0, 1.0)
    return np.where(land, temp_ok * rain_ok * slope_ok, 0.0)


def _pasture_potential(height, temp, rain, slope):
    """牧业潜力：半干旱草地最佳（0~1）。"""
    land = height >= 0
    rain_ok = np.exp(-((rain - 0.30) / 0.25) ** 2)
    temp_ok = np.exp(-((temp - 14.0) / 16.0) ** 2)
    slope_ok = np.clip(1.0 - slope / 400.0, 0.0, 1.0)
    return np.where(land, rain_ok * temp_ok * slope_ok, 0.0)


def _fish_potential(height):
    """渔业潜力：沿海（近海大陆架），内陆为 0（0~1）。"""
    land = height >= 0
    dist = distance_transform_edt(land)  # 陆地到最近海洋的距离
    return np.where(land, np.exp(-dist / 40.0), 0.0)


def _mineral_oil(height, slope):
    """石油：低频噪声 × 沉积盆地（低海拔 + 平坦）。"""
    land = height >= 0
    noise = terrain.fbm(height.shape, 3, config.SEED + 101, base_cells=5)
    basin = np.clip(1.0 - np.maximum(height, 0.0) / 1500.0, 0.0, 1.0)
    flat = np.clip(1.0 - slope / 80.0, 0.0, 1.0)
    return np.where(land, noise * basin * flat, 0.0)


def _mineral_iron(height, slope):
    """铁矿：低频噪声 × 造山带（高海拔 + 梯度大）。"""
    land = height >= 0
    noise = terrain.fbm(height.shape, 3, config.SEED + 102, base_cells=5)
    orogenic = np.clip((np.maximum(height, 0.0) - 1200.0) / 3000.0, 0.0, 1.0)
    orogenic = orogenic * np.clip(slope / 120.0, 0.0, 1.0)
    return np.where(land, noise * orogenic, 0.0)


def _mineral_coal(height, slope):
    """煤矿：低频噪声 × 古盆地（中低海拔 + 平坦）。"""
    land = height >= 0
    noise = terrain.fbm(height.shape, 3, config.SEED + 103, base_cells=5)
    basin = np.clip(1.0 - np.maximum(height, 0.0) / 2000.0, 0.0, 1.0)
    flat = np.clip(1.0 - slope / 100.0, 0.0, 1.0)
    return np.where(land, noise * basin * flat, 0.0)


def _aggregate_mean(field, labels):
    """按省份标签求均值。"""
    flat_lab = labels.ravel()
    flat_field = field.ravel()
    n = int(flat_lab.max()) + 1
    sums = np.bincount(flat_lab, weights=flat_field, minlength=n)
    counts = np.bincount(flat_lab, minlength=n).astype(np.float64)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    return means


def _to_reserve(scores):
    """把省份矿床得分映射为对数分布的储量（10 ~ 100000）。"""
    mx = float(scores.max()) if scores.size else 0.0
    if mx <= 0:
        return np.full_like(scores, 10.0)
    norm = scores / mx  # 最大省份归一化为 1
    return 10.0 ** (1.0 + 4.0 * np.clip(norm, 0.0, 1.0))


def assign(provinces, height, temp, rain, biome):
    """给每个省份分配资源（就地修改 provinces）。"""
    labels = _prov_mod.label_map()
    if labels is None or len(provinces) == 0:
        return

    slope = _slope(height)
    farm_p = _farm_potential(height, temp, rain, slope)
    past_p = _pasture_potential(height, temp, rain, slope)
    fish_p = _fish_potential(height)

    farm_m = _aggregate_mean(farm_p, labels)
    past_m = _aggregate_mean(past_p, labels)
    fish_m = _aggregate_mean(fish_p, labels)

    oil_scores = _aggregate_mean(_mineral_oil(height, slope), labels)
    iron_scores = _aggregate_mean(_mineral_iron(height, slope), labels)
    coal_scores = _aggregate_mean(_mineral_coal(height, slope), labels)

    oil_r = _to_reserve(oil_scores)
    iron_r = _to_reserve(iron_scores)
    coal_r = _to_reserve(coal_scores)

    for p in provinces:
        pid = p["id"]
        p["farm"] = float(farm_m[pid])
        p["pasture"] = float(past_m[pid])
        p["fish"] = float(fish_m[pid])
        p["oil"] = float(oil_r[pid])
        p["iron"] = float(iron_r[pid])
        p["coal"] = float(coal_r[pid])


if __name__ == "__main__":
    import time
    from . import provinces, climate

    t0 = time.time()
    height = terrain.generate()
    temp, rain, biome = climate.generate(height)
    provs = provinces.generate(height, biome)
    assign(provs, height, temp, rain, biome)

    print("provinces: {}".format(len(provs)))
    for key in ["farm", "pasture", "fish", "oil", "iron", "coal"]:
        vals = np.array([p[key] for p in provs])
        print("  {:>8s}: min={:.3f} max={:.3f} mean={:.3f}".format(
            key, vals.min(), vals.max(), vals.mean()))
    # 资源最丰富的前 3 省
    top = sorted(provs, key=lambda p: p["farm"] + p["oil"] + p["iron"] + p["coal"], reverse=True)[:3]
    for p in top:
        print("  top id={} biome={} farm={:.2f} oil={:.0f} iron={:.0f} coal={:.0f}".format(
            p["id"], p["biome"], p["farm"], p["oil"], p["iron"], p["coal"]))
    print("elapsed: {:.1f}s".format(time.time() - t0))
