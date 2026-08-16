# ComfyDL - 插件入口
#
# ComfyUI 加载插件时会 import 本包；启动时自动扫描已注册节点并在控制台打印
# 数量与分类统计。节点数量不硬编码，随后续开发自动更新。

from collections import Counter

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


def _print_startup_info():
    """启动时动态统计并打印已注册节点数量与分类（不硬编码数量）。"""
    try:
        total = len(NODE_CLASS_MAPPINGS)
        display = len(NODE_DISPLAY_NAME_MAPPINGS)
        cats = Counter(
            getattr(cls, "CATEGORY", "ComfyDL/Unknown")
            for cls in NODE_CLASS_MAPPINGS.values()
        )
        print(f"[ComfyDL] 已注册 {total} 个节点（显示名 {display} 个），共 {len(cats)} 个分类：")
        for cat in sorted(cats):
            print(f"  {cat}: {cats[cat]}")
    except Exception as e:  # 统计失败不影响插件加载
        print(f"[ComfyDL] 启动节点统计失败: {e}")


_print_startup_info()
