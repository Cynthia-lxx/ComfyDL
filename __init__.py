# ComfyDL - 插件入口
#
# ComfyUI 加载插件时会 import 本包；启动时自动扫描已注册节点并在控制台打印
# 数量与分类统计。节点数量不硬编码，随后续开发自动更新。

import os
import sys

# ComfyUI loads custom-node packages under synthetic module names, which breaks
# relative imports (``from ..src.d2lcore ...``). Make the plugin root importable
# so the fallback ``from src.d2lcore.torch import ...`` paths resolve.
_PLUGIN_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, _PLUGIN_ROOT)

from collections import Counter

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


def _print_startup_info():
    """启动时动态统计并打印已注册节点数量与分类（不硬编码数量）。"""
    try:
        total = len(NODE_CLASS_MAPPINGS)
        display = len(NODE_DISPLAY_NAME_MAPPINGS)
        cats = Counter()
        unresolved = []
        for node_id, cls in NODE_CLASS_MAPPINGS.items():
            try:
                # V3 节点的 CATEGORY 是惰性 classproperty：首次访问会触发
                # GET_SCHEMA() 解析。单个节点的 schema 异常不应拖垮整张统计表，
                # 因此逐类兜底，把失败项归入 d2l/Unknown 并单独列出。
                cats[getattr(cls, "CATEGORY", "d2l/Unknown")] += 1
            except Exception as e:  # noqa: BLE001 - 单个节点问题不应影响统计
                cats["d2l/Unknown"] += 1
                unresolved.append(f"{node_id} ({e})")
        print(f"[ComfyDL] 已注册 {total} 个节点（显示名 {display} 个），共 {len(cats)} 个分类：")
        for cat in sorted(cats):
            print(f"  {cat}: {cats[cat]}")
        for item in unresolved:
            print(f"[ComfyDL]   分类解析失败: {item}")
    except Exception as e:  # 统计失败不影响插件加载
        print(f"[ComfyDL] 启动节点统计失败: {e}")


_print_startup_info()
