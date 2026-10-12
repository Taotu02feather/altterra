# -*- coding: utf-8 -*-
"""config.py — 全局配置参数。"""

WORLD_SIZE = 1024          # 世界边长（像素）
SEED = 42                  # 随机种子
NUM_PROVINCES = 200        # 省份数量
SEA_LEVEL_PERCENTILE = 70  # 海平面分位数（海洋占比约 70%）
NUM_OCTAVES = 6            # 细节噪声八度数
EROSION_ITERATIONS = 50    # 侵蚀迭代次数
TEMP_BASE = 30             # 赤道海平面基准温度（°C）
LAPSE_RATE = 6.5           # 温度递减率（°C / 1000m）

# 群系编码 -> 名称
BIOME_NAMES = {
    0: "ocean", 1: "ice", 2: "tundra", 3: "taiga", 4: "temperate_forest",
    5: "grassland", 6: "desert", 7: "savanna", 8: "rainforest", 9: "alpine",
}


if __name__ == "__main__":
    # 简单测试：打印配置
    for k, v in sorted(vars().items()):
        if k.isupper():
            print("{} = {}".format(k, v))
