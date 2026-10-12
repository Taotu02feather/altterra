# -*- coding: utf-8 -*-
"""
terrain.py — 高度场生成与水力侵蚀（梯度噪声 + 岛屿）

流程：
1. 大尺度大陆掩膜（低频域扭曲梯度噪声 + 轻度平滑）
2. 多尺度梯度噪声 FBM 细节（NUM_OCTAVES 个八度）
3. 组合得到大陆基础高度场（height = 大陆掩膜 × 细节噪声 + 细碎丘陵）
4. 岛屿生成（线性火山岛链 + 散布群岛 + 珊瑚环礁）
5. 简化 D8 水力侵蚀（仅作用于大陆，岛屿年龄太短不受侵蚀）
6. 叠加火山岛、盖章珊瑚环礁

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
# 梯度噪声（Perlin 风格，哈希选梯度，向量化）
# ---------------------------------------------------------------------------
# 8 个单位梯度方向：4 个轴向 + 4 个对角（对角已归一化到单位长度）
_GRAD_DIRS = np.array([
    [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
    [0.7071067811865476, 0.7071067811865476],
    [-0.7071067811865476, 0.7071067811865476],
    [0.7071067811865476, -0.7071067811865476],
    [-0.7071067811865476, -0.7071067811865476],
], dtype=np.float64)


def _hash2(ix, iy, seed):
    """splitmix64 风格 2D 哈希：从 (ix, iy, seed) 得到伪随机 uint64（向量化）。

    溢出环绕是哈希的预期行为，用 errstate 关闭溢出警告。
    """
    with np.errstate(over='ignore'):
        x = ix.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
        x += iy.astype(np.uint64) * np.uint64(0xBF58476D1CE4E5B9)
        x += np.uint64(seed) * np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(30)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27)
        x *= np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    return x


def _grad(ix, iy, seed):
    """格点 (ix, iy) 处的单位梯度向量 (gx, gy)（哈希选 8 方向之一，向量化）。"""
    h = _hash2(ix, iy, seed)
    idx = (h & np.uint64(7)).astype(np.int64)
    return _GRAD_DIRS[idx, 0], _GRAD_DIRS[idx, 1]


def _gradient_noise(y, x, cells, seed):
    """在晶格坐标 (y, x)（单位=晶格单元）处计算梯度噪声，值域约 [-1, 1]。

    对每个采样点：四角点积 + 5 次 smoothstep 双线性插值，全程向量化。
    """
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = x - x0
    fy = y - y0
    x0 = np.mod(x0, cells)
    y0 = np.mod(y0, cells)
    x1 = np.mod(x0 + 1, cells)
    y1 = np.mod(y0 + 1, cells)

    # 5 次 smoothstep：t = 6t^5 - 15t^4 + 10t^3
    u = fx * fx * fx * (fx * (fx * 6.0 - 15.0) + 10.0)
    v = fy * fy * fy * (fy * (fy * 6.0 - 15.0) + 10.0)

    g00x, g00y = _grad(x0, y0, seed)
    g10x, g10y = _grad(x1, y0, seed)
    g01x, g01y = _grad(x0, y1, seed)
    g11x, g11y = _grad(x1, y1, seed)

    d00 = g00x * fx + g00y * fy
    d10 = g10x * (fx - 1.0) + g10y * fy
    d01 = g01x * fx + g01y * (fy - 1.0)
    d11 = g11x * (fx - 1.0) + g11y * (fy - 1.0)

    top = d00 + u * (d10 - d00)
    bot = d01 + u * (d11 - d01)
    return top + v * (bot - top)


def fbm(shape, octaves, seed, base_cells=4, gain=None, lacunarity=None):
    """分形布朗运动（FBM），基于梯度噪声，返回值归一化到约 [0, 1]。

    base_cells：最低频八度跨越整幅图的晶格数。
    gain / lacunarity：默认读 config（GAIN / LACUNARITY）。
    """
    if gain is None:
        gain = config.GAIN
    if lacunarity is None:
        lacunarity = config.LACUNARITY
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
        result += amplitude * _gradient_noise(y * freq, x * freq, cells, seed + i)
        norm += amplitude
        amplitude *= gain
        freq *= lacunarity
    result /= norm
    # 梯度噪声值域约 [-1, 1]，映射到 [0, 1]
    return (result + 1.0) * 0.5


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

    gain = config.GAIN
    lacunarity = config.LACUNARITY
    result = np.zeros((h, w), dtype=np.float64)
    amplitude = 1.0
    norm = 0.0
    freq = 1.0
    for i in range(octaves):
        cells = max(1, int(round(base_cells * (lacunarity ** i))))
        result += amplitude * _gradient_noise(y_w * freq, x_w * freq, cells, seed + i)
        norm += amplitude
        amplitude *= gain
        freq *= lacunarity
    result /= norm
    return (result + 1.0) * 0.5


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
# 岛屿生成（线性火山岛链 + 散布群岛 + 珊瑚环礁）
# ---------------------------------------------------------------------------
def _add_gaussian(field, cy, cx, amp, sigma):
    """在 field 上以 (cy, cx) 为中心叠加高斯隆起（仅局部窗口内计算）。"""
    h, w = field.shape
    r = int(3.0 * sigma) + 1
    y0 = max(0, int(cy) - r)
    y1 = min(h, int(cy) + r + 1)
    x0 = max(0, int(cx) - r)
    x1 = min(w, int(cx) + r + 1)
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
    field[y0:y1, x0:x1] += amp * np.exp(-d2 / (sigma * sigma))


def _raise_seamount(height, cy, cx, top_depth, sigma):
    """把 (cy,cx) 附近海床抬升到 top_depth（高斯形态，用 max 累积，绝不超出浅海）。

    中心抬升到 top_depth，向外平滑衰减回原深度，所以不会在近海抬成高山。
    """
    h, w = height.shape
    r = int(3.0 * sigma) + 1
    y0 = max(0, int(cy) - r)
    y1 = min(h, int(cy) + r + 1)
    x0 = max(0, int(cx) - r)
    x1 = min(w, int(cx) + r + 1)
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
    g = np.exp(-d2 / (2.0 * sigma * sigma))  # 中心 1 → 边缘 0
    original = height[y0:y1, x0:x1]
    target = top_depth * g + original * (1.0 - g)  # 中心 top_depth，边缘回原深
    height[y0:y1, x0:x1] = np.maximum(original, target)


def _raise_platform(height, cy, cx, top_depth, radius, edge_sigma):
    """平顶浅滩（guyot）：半径 radius 内抬升到 top_depth，边缘平滑过渡回原深。"""
    h, w = height.shape
    r = int(radius + 3.0 * edge_sigma) + 2
    y0 = max(0, int(cy) - r)
    y1 = min(h, int(cy) + r + 1)
    x0 = max(0, int(cx) - r)
    x1 = min(w, int(cx) + r + 1)
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    s = np.clip((d - radius) / edge_sigma, 0.0, 1.0)
    s = s * s * (3.0 - 2.0 * s)  # 0 内 → 1 外
    original = height[y0:y1, x0:x1]
    target = top_depth * (1.0 - s) + original * s
    height[y0:y1, x0:x1] = np.maximum(original, target)


def _stamp_atoll(height, cy, cx, r_in, r_out, rim_h, lagoon_d, angle, ratio):
    """把珊瑚环礁盖章到 height（绝对值写入，椭圆 + 随机朝向）。

    环脊（正值露出海面）、潟湖（负值浅水）、环外浅滩平滑过渡到深海。
    """
    h, w = height.shape
    margin = int(r_out) + 30
    y0 = max(0, int(cy) - margin)
    y1 = min(h, int(cy) + margin + 1)
    x0 = max(0, int(cx) - margin)
    x1 = min(w, int(cx) + margin + 1)
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    dy = yy - cy
    dx = xx - cx
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rx = dx * cos_a + dy * sin_a
    ry = -dx * sin_a + dy * cos_a
    d = np.sqrt(rx * rx + (ry / ratio) * (ry / ratio))

    rim = (d >= r_in) & (d <= r_out)
    lagoon = d < r_in
    s = np.clip((d - r_out) / 25.0, 0.0, 1.0)  # 环外浅滩 → 深海过渡
    s = s * s * (3.0 - 2.0 * s)
    outer_depth = -10.0 + (-8000.0 + 10.0) * s
    new = np.where(lagoon, -lagoon_d, np.where(rim, rim_h, outer_depth))
    region = d <= r_out + 25.0
    sub = height[y0:y1, x0:x1]
    sub[region] = new[region]
    height[y0:y1, x0:x1] = sub


def _sample_ocean_point(ocean, dist_land, rng, n, min_dist):
    """在海洋中随机采样 n 个点（距陆地至少 min_dist 像素，保证远洋）。"""
    candidate = ocean & (dist_land > min_dist)
    ys, xs = np.nonzero(candidate)
    if len(ys) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    idx = rng.choice(len(ys), size=min(n, len(ys)), replace=False)
    return np.stack([ys[idx].astype(np.float64), xs[idx].astype(np.float64)], axis=1)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _continental_mask(size, seed):
    """大尺度大陆掩膜：低频域扭曲梯度噪声 + 轻度高斯平滑。"""
    continental = _fbm_warped(
        (size, size), octaves=3, seed=seed,
        base_cells=3, warp_strength=1.2, warp_cells=2,
    )
    return gaussian_filter(continental, sigma=2.5)  # 轻度平滑，保留海岸线细节


def generate_islands(height, sea_level, seed):
    """在 height 上直接盖章生成岛屿（线性火山岛链 + 散布群岛 + 珊瑚环礁）。

    参数：
        height：侵蚀后的高度场（海平面以下为海洋），原地修改。
        sea_level：海平面高度（本代码库固定为 0.0）。
        seed：随机种子（由 config.SEED 派生，保证可复现）。
    """
    h, w = height.shape
    ocean = height < sea_level
    dist_land = distance_transform_edt(ocean)
    rng = np.random.default_rng(seed)

    # ---- 形态A：线性火山岛链（夏威夷 / 日本 / 阿留申）----
    n_chains = int(rng.integers(6, 11))  # 6~10 条链
    for _ in range(n_chains):
        start = _sample_ocean_point(ocean, dist_land, rng, 1, min_dist=60)
        if start.shape[0] == 0:
            continue
        cy, cx = start[0]
        angle = rng.uniform(0.0, 2.0 * np.pi)
        n_nodes = int(rng.integers(7, 16))  # 7~15 个火山节点
        spacing = rng.uniform(35.0, 60.0)
        dir_y, dir_x = np.sin(angle), np.cos(angle)
        perp_y, perp_x = -dir_x, dir_y
        for i in range(n_nodes):
            jitter = rng.uniform(-0.9, 0.9) * spacing  # 30% 垂直偏移抖动
            ny = cy + dir_y * spacing * i + perp_y * jitter
            nx = cx + dir_x * spacing * i + perp_x * jitter
            taper = 1.0 - 0.6 * (i / max(1, n_nodes - 1))  # 热点漂移：越远越矮
            amp = rng.uniform(600.0, 1600.0) * taper * rng.uniform(0.8, 1.2)
            sigma = rng.uniform(12.0, 28.0)
            _raise_seamount(height, ny, nx, -400.0, rng.uniform(40.0, 70.0))
            _add_gaussian(height, ny, nx, amp, sigma)

    # ---- 形态B：散布群岛（印尼 / 菲律宾）----
    density_noise = fbm((h, w), 2, seed + 777, base_cells=6)
    n_arch = int(rng.integers(3, 6))  # 3~5 个群岛中心
    for _ in range(n_arch):
        center = _sample_ocean_point(ocean, dist_land, rng, 1, min_dist=50)
        if center.shape[0] == 0:
            continue
        cy, cx = center[0]
        radius = rng.uniform(100.0, 180.0)
        n_bumps = int(rng.integers(14, 28))  # 14~27 个隆起
        _raise_platform(height, cy, cx, -300.0, radius, 50.0)
        for _ in range(n_bumps):
            rr = radius * 0.85 * np.sqrt(rng.random())
            th = rng.uniform(0.0, 2.0 * np.pi)
            ny = cy + rr * np.sin(th)
            nx = cx + rr * np.cos(th)
            iy = int(np.clip(ny, 0, h - 1))
            ix = int(np.clip(nx, 0, w - 1))
            if density_noise[iy, ix] < rng.random():  # 低频噪声调制密度
                continue
            _add_gaussian(height, ny, nx, rng.uniform(500.0, 1200.0), rng.uniform(8.0, 18.0))

    # ---- 形态C：珊瑚环礁（热带浅海）----
    lat = np.linspace(-np.pi / 2, np.pi / 2, h)
    temp_approx = 42.0 * np.cos(lat) - 12.0  # 与 climate.py 一致的纬向温度（海平面）
    tropical = temp_approx[:, None] > 20.0
    cand = ocean & tropical & (dist_land > 30)
    ys, xs = np.nonzero(cand)
    n_atolls = int(rng.integers(50, 121))
    if len(ys) > 0:
        idx = rng.choice(len(ys), size=min(n_atolls, len(ys)), replace=False)
        for iy, ix in zip(ys[idx], xs[idx]):
            _stamp_atoll(height, float(iy), float(ix),
                         rng.uniform(4.0, 8.0), rng.uniform(8.0, 15.0),
                         rng.uniform(5.0, 20.0), rng.uniform(5.0, 20.0),
                         rng.uniform(0.0, np.pi), rng.uniform(0.65, 1.0))


def generate():
    """生成高度场（梯度噪声大陆 + 火山岛链 + 散布群岛 + 珊瑚环礁）。"""
    size = config.WORLD_SIZE
    seed = config.SEED

    # 1. 大陆掩膜（调节阈值使海洋占比约 68%）
    continental = _continental_mask(size, seed)
    threshold = np.percentile(continental, config.SEA_LEVEL_PERCENTILE)
    landmask = _smoothstep(continental, threshold - 0.015, threshold + 0.015)
    land = landmask > 0.5

    # 2. 多尺度梯度噪声 FBM 细节（标准化为近似标准正态）
    detail = fbm((size, size), config.NUM_OCTAVES, seed + 5000, base_cells=8)
    detail = (detail - detail.mean()) / (detail.std() + 1e-9)

    # 3. 大陆基础高度（海岸 0 -> 内陆抬升；山地集中在噪声高值区；全陆地叠加细碎丘陵）
    elev = np.clip((landmask - 0.5) * 2.0, 0.0, 1.0)
    mountain = np.clip((detail - 0.2) / 2.0, 0.0, 1.0) ** 1.4
    hills = detail * 0.5 + 0.5  # 0~1 细碎丘陵纹理，让低地不再平坦
    land_height = (elev * 250.0
                   + mountain * 5200.0 * (0.2 + 0.8 * elev)
                   + hills * 160.0 * (0.2 + 0.8 * elev))
    dist_land = distance_transform_edt(~land)
    ocean_depth = np.clip(-80.0 - 250.0 * np.sqrt(dist_land) - 20.0 * dist_land, -8000.0, 0.0)

    height = np.where(land, land_height, ocean_depth)

    # 4. 水力侵蚀（仅作用于大陆）
    height = _erode(height, land)

    # 5. 生成并盖章岛屿（岛龄短，不受侵蚀）：火山岛链 + 散布群岛 + 珊瑚环礁
    generate_islands(height, 0.0, seed + 9000)

    # 裁剪到目标值域
    height = np.clip(height, -8000.0, 6000.0)
    return height.astype(np.float32)


if __name__ == "__main__":
    import time
    from scipy import ndimage

    t0 = time.time()
    h = generate()
    land = h >= 0
    ocean_frac = float((~land).mean())
    labels, n = ndimage.label(land)
    sizes = ndimage.sum(land, labels, range(1, n + 1)) if n else np.array([])
    total_land = int(land.sum())

    # 大陆 = 面积 >= 0.5% 地图像素的连通域；岛屿 = 其余小连通域
    continent_threshold = int(0.005 * h.size)
    continent_size = int(sizes[sizes >= continent_threshold].sum()) if sizes.size else 0
    island_frac = (total_land - continent_size) / total_land if total_land else 0.0

    if n:
        comp_size = np.zeros(n + 1, dtype=np.int64)
        comp_size[1:] = sizes
        island_mask = land & (comp_size[labels] < continent_threshold)
    else:
        island_mask = np.zeros_like(land)
    island_mean = float(h[island_mask].mean()) if island_mask.any() else 0.0

    print("height        : min={:.1f} max={:.1f} mean={:.1f}".format(h.min(), h.max(), h.mean()))
    print("ocean fraction: {:.3f}".format(ocean_frac))
    print("land fraction : {:.3f}".format(1.0 - ocean_frac))
    print("continents    : {}".format(n))
    print("largest landmass size: {}".format(int(sizes.max()) if n else 0))
    print("island / land : {:.3f}".format(island_frac))
    print("island mean height: {:.1f}".format(island_mean))
    print("elapsed       : {:.1f}s".format(time.time() - t0))
