# AltTerra

架空世界地形与战略地图程序化生成器（类似钢铁雄心/维多利亚的大战略地图）。

目标世界是"另一个地球"，非真实地球，但具有地质与气候的真实感。

## 快速开始

```bash
pip install -r requirements.txt
python main.py
```

运行后会在 `output/` 目录生成 3 张 PNG：

- `world_map.png` — 世界地图（海洋深度渐变 + 气候群系 + 省份边界 + 资源标注）
- `capacity.png`  — 农业承载力热力图（浅黄 -> 深绿）
- `provinces.png` — 省份划分检查图（每省随机色 + 白色边界）

## 模块

| 模块 | 说明 |
|------|------|
| `worldgen/terrain.py`   | 高度场生成（FBM 大陆掩膜 + D8 水力侵蚀） |
| `worldgen/climate.py`   | 温度 / 降水 / 群系分类（Whittaker 简化版） |
| `worldgen/provinces.py` | Voronoi + 分水岭省份划分 |
| `worldgen/resources.py` | 资源三层系统（农业/牧业/渔业 + 石油/铁矿/煤矿） |
| `worldgen/render.py`    | 地图渲染（matplotlib Agg，无弹窗） |

## 技术栈

Python 3.10+，仅依赖 numpy / scipy / matplotlib（numba 可选，未安装时自动回退纯 numpy）。

所有噪声函数均由 numpy 自行实现，未使用 noise / PIL / geopandas / shapely 等库。

## 配置

在 `worldgen/config.py` 中可调整世界边长、种子、省份数量、海平面占比、侵蚀迭代次数等参数。

