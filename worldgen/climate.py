# -*- coding: utf-8 -*-
"""
climate.py — 温度、降水与群系分类

输入：height 二维数组
输出：temp、rain、biome 三个二维数组（biome 为 int8，值域 0~9）

温度：纬向温度（赤道热、极地冷）+ 海拔递减率 + 小噪声
降水：纬度带（赤道/温带多雨、副热带/极地少雨）+ 沿海加成 + 地形抬升加成
群系：Whittaker 图简化分类
"""

import numpy as np
from scipy.ndimage import distance_transform_edt

from . import config


def _temperature(height):
    """纬向 + 海拔温度场。"""
    h, w = height.shape
    lat = np.linspace(-np.pi / 2, np.pi / 2, h)[:, None]  # 弧度，-90°~90°
    rng = np.random.default_rng(config.SEED + 11)
    noise = rng.normal(0.0, 1.2, (h, w)).astype(np.float32)
    land_height = np.maximum(height, 0.0)  # 海洋温度按海平面算
    # 纬向项：赤道约 TEMP_BASE，极地约 -12°C（额外极地冷却保证极地冰雪带）
    lat_term = config.TEMP_BASE * np.cos(lat) - 12.0 * (1.0 - np.cos(lat))
    temp = lat_term - config.LAPSE_RATE * land_height / 1000.0 + noise
    return temp.astype(np.float32)


def _rainfall(height):
    """简化降水模型（纬度带 + 沿海 + 地形）。"""
    h, w = height.shape
    land = height >= 0
    lat = np.linspace(-np.pi / 2, np.pi / 2, h)[:, None]
    lat_deg = np.degrees(lat)

    # 1. 纬度带：赤道雨带、温带雨带（±50°）、副热带干（±28°）、极地干（±80°）
    eq = np.exp(-(lat_deg / 15.0) ** 2)
    tmp = np.exp(-((lat_deg - 50) / 15.0) ** 2) + np.exp(-((lat_deg + 50) / 15.0) ** 2)
    sub = np.exp(-((lat_deg - 28) / 12.0) ** 2) + np.exp(-((lat_deg + 28) / 12.0) ** 2)
    pol = np.exp(-((lat_deg - 80) / 10.0) ** 2) + np.exp(-((lat_deg + 80) / 10.0) ** 2)
    band = np.clip(0.10 + 0.70 * eq + 0.16 * tmp - 0.11 * sub - 0.08 * pol, 0.02, 1.0)
    rain = np.broadcast_to(band, (h, w)).astype(np.float32).copy()

    # 2. 沿海加成：距海岸越近降水越多（edt 计算到最近背景即海洋的距离）
    dist_coast = distance_transform_edt(land)
    rain = rain + 0.30 * np.exp(-dist_coast / 70.0)

    # 3. 地形抬升加成（迎风坡用梯度近似）
    surf = np.maximum(height, 0.0)
    gx = np.gradient(surf, axis=1)
    gy = np.gradient(surf, axis=0)
    slope = np.hypot(gx, gy)
    rain = rain + 0.15 * np.clip(slope / 400.0, 0.0, 1.0)

    rain = np.clip(rain, 0.0, 1.0)
    # 海洋降水置满（不影响 biome，仅作水分来源的语义）
    rain = np.where(land, rain, 1.0)
    return rain.astype(np.float32)


def _classify_biome(height, temp, rain):
    """Whittaker 图简化群系分类。"""
    land = height >= 0
    t = temp
    r = rain
    conds = [
        ~land,                        # 0 海洋
        (t < -12) | (height > 4200),  # 1 冰盖（极地或极高海拔）
        height > 2800,                # 9 高山
        t < 0,                        # 2 苔原
        t < 8,                        # 3 泰加林
        (t < 18) & (r < 0.30),        # 5 温带草原
        t < 18,                       # 4 温带森林
        (t < 24) & (r < 0.25),        # 6 沙漠
        (t < 24) & (r < 0.45),        # 7 稀树草原
        t < 24,                       # 4 亚热带森林
        r < 0.25,                     # 6 热带沙漠
        r < 0.50,                     # 7 热带稀树草原
        True,                         # 8 热带雨林
    ]
    choices = [0, 1, 9, 2, 3, 5, 4, 6, 7, 4, 6, 7, 8]
    return np.select(conds, choices, default=0).astype(np.int8)


def generate(height):
    """生成气候三件套：temp、rain、biome。"""
    temp = _temperature(height)
    rain = _rainfall(height)
    biome = _classify_biome(height, temp, rain)
    return temp, rain, biome


if __name__ == "__main__":
    import time
    from . import terrain

    t0 = time.time()
    height = terrain.generate()
    temp, rain, biome = generate(height)
    land = height >= 0
    print("temp : min={:.1f} max={:.1f} mean={:.1f}".format(temp.min(), temp.max(), temp.mean()))
    print("rain : min={:.2f} max={:.2f} mean={:.2f}".format(rain.min(), rain.max(), rain.mean()))
    print("biome value range: {} ~ {}".format(biome.min(), biome.max()))
    for code in sorted(np.unique(biome).tolist()):
        frac = float((biome == code).mean())
        print("  biome {:>2d} {:>16s}: {:.3f}".format(code, config.BIOME_NAMES[code], frac))
    # 按纬度打印一条经线剖面，验证气候带
    col = height.shape[1] // 2
    print("latitude profile (every 64 rows, col={}):".format(col))
    for i in range(0, height.shape[0], 64):
        print("  lat={:+5.0f}  temp={:+.1f}  rain={:.2f}  biome={}".format(
            -90 + 180 * i / height.shape[0], temp[i, col], rain[i, col], config.BIOME_NAMES[int(biome[i, col])]))
    print("elapsed: {:.1f}s".format(time.time() - t0))
