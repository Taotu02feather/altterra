# -*- coding: utf-8 -*-
"""
terrain.py — 高度场生成与水力侵蚀

流程：
1. 大尺度大陆掩膜（低频 FBM + 域扭曲 + smoothstep）
2. 多尺度 FBM 细节（6 个八度）
3. 组合得到原始高度场（height = 大陆掩膜 × 细节噪声）
4. 简化 D8 水力侵蚀（流量累积 + 下切），迭代 EROSION_ITERATIONS 次

输出：height 二维数组，shape=(WORLD_SIZE, WORLD_SIZE)，值域约 -8000 ~ 6000。
海洋为负值，陆地为正值，海平面为 0。
"""

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter, distance_transform_edt

from . import config

try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:  # pragma: no cover - numba 未安装时回退到纯 numpy
    _HAS_NUMBA = False


# ---------------------------------------------------------------------------
# 值噪声（可平铺、可连续采样）
# ---------------------------------------------------------------------------
def _value_noise_grid(seed, size):
    """生成 size×size 的随机晶格（值噪声的格点值）。"""
    rng = np.random.default_rng(seed)
    return rng.random((size, size))


def _sample_grid(grid, y, x):
    """在晶格坐标 (y, x) 处做双线性采样，带平铺（wrap）。

    y、x 为同形状数组，单位 = 晶格单元；返回同形状数组。
    """
    size = grid.shape[0]
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    fy = y - y0
    fx = x - x0
    y0 = np.mod(y0, size)
    x0 = np.mod(x0, size)
    y1 = np.mod(y0 + 1, size)
    x1 = np.mod(x0 + 1, size)
    v00 = grid[y0, x0]
    v01 = grid[y0, x1]
    v10 = grid[y1, x0]
    v11 = grid[y1, x1]
    top = v00 * (1.0 - fx) + v01 * fx
    bot = v10 * (1.0 - fx) + v11 * fx
    return top * (1.0 - fy) + bot * fy


def fbm(shape, octaves, seed, base_cells=4, gain=0.5, lacunarity=2.0):
    """分形布朗运动（FBM），返回值归一化到约 [0, 1]。

    base_cells：最低频八度跨越整幅图的晶格数。
    """
    h, w = shape
    yy = np.linspace(0.0, float(base_cells), h, endpoint=False)
    xx = np.linspace(0.0, float(base_cells), w, endpoint=False)
    y, x = np.meshgrid(yy, xx, indexing='ij')
    result = np.zeros((h, w), dtype=np.float64)
    amplitude = 1.0
    norm = 0.0
    freq = 1.0
    for i in range(octaves):
        cells = max(1, int(round(base_cells * (lacunarity ** i))))
        grid = _value_noise_grid(seed + i, cells)
        result += amplitude * _sample_grid(grid, y * freq, x * freq)
        norm += amplitude
        amplitude *= gain
        freq *= lacunarity
    return result / norm


def _fbm_warped(shape, octaves, seed, base_cells, warp_strength, warp_cells=2):
    """域扭曲 FBM：用低频噪声扭曲采样坐标，让大陆形状更自然。"""
    h, w = shape
    yy = np.linspace(0.0, float(base_cells), h, endpoint=False)
    xx = np.linspace(0.0, float(base_cells), w, endpoint=False)
    y, x = np.meshgrid(yy, xx, indexing='ij')
    warp_y = fbm(shape, 2, seed + 1000, base_cells=warp_cells)
    warp_x = fbm(shape, 2, seed + 2000, base_cells=warp_cells)
    y_w = y + (warp_y - 0.5) * warp_strength
    x_w = x + (warp_x - 0.5) * warp_strength
    result = np.zeros((h, w), dtype=np.float64)
    amplitude = 1.0
    norm = 0.0
    freq = 1.0
    for i in range(octaves):
        cells = max(1, int(round(base_cells * (2.0 ** i))))
        grid = _value_noise_grid(seed + i, cells)
        result += amplitude * _sample_grid(grid, y_w * freq, x_w * freq)
        norm += amplitude
        amplitude *= 0.5
        freq *= 2.0
    return result / norm


def _smoothstep(x, edge0, edge1):
    """三次 Hermite 平滑过渡。"""
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------------------
# D8 水力侵蚀
# ---------------------------------------------------------------------------
def _compute_receiver(height):
    """计算每个格子的汇流方向（指向 8 邻居中最低者），返回扁平化的接收者索引。

    洼地（最低邻居不低于自身）返回 -1。
    """
    h, w = height.shape
    padded = np.pad(height, 1, mode='edge')
    nbrs = np.stack([
        padded[0:-2, 0:-2], padded[0:-2, 1:-1], padded[0:-2, 2:],
        padded[1:-1, 0:-2],                     padded[1:-1, 2:],
        padded[2:, 0:-2],   padded[2:, 1:-1],   padded[2:, 2:],
    ], axis=0)
    lowest = nbrs.min(axis=0)
    arg = nbrs.argmin(axis=0)
    flat = np.arange(h * w, dtype=np.int64).reshape(h, w)
    offsets = np.array([-w - 1, -w, -w + 1, -1, 1, w - 1, w, w + 1], dtype=np.int64)
    receiver = flat + offsets[arg]
    sink = lowest >= height
    receiver = np.where(sink, -1, receiver)
    return receiver.ravel()


if _HAS_NUMBA:

    @njit(cache=True)
    def _flow_accumulate(receiver, height_flat):
        """按海拔从高到低的拓扑序累加流量（D8）。"""
        n = receiver.shape[0]
        order = np.argsort(-height_flat)
        flow = np.ones(n, dtype=np.float64)
        for k in range(n):
            i = order[k]
            r = receiver[i]
            if r >= 0 and r != i:
                flow[r] += flow[i]
        return flow

else:
    def _flow_accumulate(receiver, height_flat):  # pragma: no cover
        # 纯 numpy 回退：返回全 1，由调用方改用坡度下切（见 _erode_thermal）
        return np.ones(receiver.shape[0], dtype=np.float64)


def _box_blur(arr):
    """3×3 均值模糊（边界用最近邻填充）。"""
    return uniform_filter(arr, size=3, mode='nearest')


def _erode_d8(height, land, iterations):
    """D8 流量累积 + 下切侵蚀（仅作用于陆地）。"""
    h = height.astype(np.float64).copy()
    for _ in range(iterations):
        receiver = _compute_receiver(h)
        flow = _flow_accumulate(receiver, h.ravel()).reshape(h.shape)
        incision = np.log1p(flow) * 1.2  # 流量越大下切越深
        h = h - np.where(land, incision, 0.0)
        blurred = _box_blur(h)  # 仅在陆地上做轻微平滑，防止数值发散
        h = np.where(land, blurred, h)
    return h


def _erode_thermal(height, land, iterations):  # pragma: no cover - 无 numba 时的回退
    """坡度驱动的简化侵蚀（纯 numpy 回退方案）。"""
    h = height.astype(np.float64).copy()
    for _ in range(iterations):
        gx = np.gradient(h, axis=1)
        gy = np.gradient(h, axis=0)
        slope = np.hypot(gx, gy)
        h = h - np.where(land, np.clip(slope, 0.0, 80.0) * 0.03, 0.0)
        blurred = _box_blur(h)
        h = np.where(land, blurred, h)
    return h


def _erode(height, land):
    """侵蚀入口：有 numba 用 D8，否则回退到坡度侵蚀。"""
    if _HAS_NUMBA:
        return _erode_d8(height, land, config.EROSION_ITERATIONS)
    return _erode_thermal(height, land, config.EROSION_ITERATIONS)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _continental_mask(size, seed):
    """大尺度大陆掩膜：低频域扭曲 FBM + 高斯平滑。"""
    continental = _fbm_warped(
        (size, size), octaves=3, seed=seed,
        base_cells=3, warp_strength=1.2, warp_cells=2,
    )
    return gaussian_filter(continental, sigma=4.0)  # 平滑去除碎斑


def generate():
    """生成高度场。"""
    size = config.WORLD_SIZE
    seed = config.SEED

    # 1. 大陆掩膜（调节阈值使海洋占比约 70%）
    continental = _continental_mask(size, seed)
    threshold = np.percentile(continental, config.SEA_LEVEL_PERCENTILE)
    landmask = _smoothstep(continental, threshold - 0.015, threshold + 0.015)
    land = landmask > 0.5

    # 2. 多尺度 FBM 细节（标准化为近似标准正态，便于控制山地范围）
    detail = fbm((size, size), config.NUM_OCTAVES, seed + 5000, base_cells=8)
    detail = (detail - detail.mean()) / (detail.std() + 1e-9)

    # 3. 组合高度
    elev = np.clip((landmask - 0.5) * 2.0, 0.0, 1.0)  # 海岸 0 -> 内陆 1
    # 基础低地随内陆略抬升；山地集中在细节噪声的高值区（形成山脉）
    mountain = np.clip((detail - 0.2) / 2.0, 0.0, 1.0) ** 1.4
    land_height = elev * 250.0 + mountain * 5400.0 * (0.2 + 0.8 * elev)
    # 距离变换：edt 计算到最近背景(False)的距离，故取反得到「距最近陆地」
    dist_land = distance_transform_edt(~land)
    ocean_depth = np.clip(-80.0 - 250.0 * np.sqrt(dist_land) - 20.0 * dist_land, -8000.0, 0.0)

    height = np.where(land, land_height, ocean_depth)

    # 4. 水力侵蚀
    height = _erode(height, land)

    # 裁剪到目标值域
    height = np.clip(height, -8000.0, 6000.0)
    return height.astype(np.float32)


if __name__ == "__main__":
    import time
    from scipy import ndimage

    t0 = time.time()
    h = generate()
    land = h >= 0
    labels, n = ndimage.label(land)
    sizes = ndimage.sum(land, labels, range(1, n + 1)) if n else np.array([])
    print("height : min={:.1f} max={:.1f} mean={:.1f}".format(h.min(), h.max(), h.mean()))
    print("land fraction : {:.3f}".format(float(land.mean())))
    print("continents    : {}".format(n))
    print("largest landmass size: {}".format(int(sizes.max()) if n else 0))
    print("elapsed       : {:.1f}s".format(time.time() - t0))
