# -*- coding: utf-8 -*-
"""main.py — AltTerra 一键生成入口。运行 `python main.py` 输出 3 张 PNG。"""

import argparse
import time
from worldgen import config, terrain, climate, provinces, resources, render


def main():
    parser = argparse.ArgumentParser(description="AltTerra 地图生成")
    parser.add_argument("--seed", type=int, default=None,
                        help="覆盖 config.SEED 的随机种子")
    args = parser.parse_args()

    seed = config.SEED
    if args.seed is not None:
        config.SEED = args.seed
        seed = args.seed
    print(f"当前 SEED: {seed}")

    t0 = time.time()
    print("[1/5] 生成高度场...")
    height = terrain.generate()
    print("  height: min={:.1f}, max={:.1f}, mean={:.1f}".format(
        height.min(), height.max(), height.mean()))
    print("[2/5] 生成气候...")
    temp, rain, biome = climate.generate(height)
    print("[3/5] 划分省份...")
    provs = provinces.generate(height, biome)
    print("[4/5] 分配资源...")
    resources.assign(provs, height, temp, rain, biome)
    print("[5/5] 渲染地图...")
    render.render_all(height, biome, provs, temp, rain)
    print("完成，耗时 {:.1f} 秒".format(time.time() - t0))


if __name__ == "__main__":
    main()
