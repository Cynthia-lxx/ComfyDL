#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ComfyDL 文档统计同步脚本
========================
从当前 NODE_CLASS_MAPPINGS 实时统计各 CATEGORY 节点数量，并同步到四个文档：
FUNCTIONS.md / FUNCTIONS_zh.md / README.md / README_zh.md
（对应 Developer_Guidelines.txt 第 6 条：每次改动必须同步更新四个文档）。

节点数量不硬编码：动态统计注册表，后续新增节点/类别会自动更新。

覆盖位置：
  1) FUNCTIONS(.zh).md 章节标题：`## N. d2l / <Cat> (N nodes)` / `## N. ComfyUI / <cat> ...`
  2) FUNCTIONS(.zh).md 附录分类表：`| <category> | N | ...`（d2l/<Cat> 或核心分类名）
  3) FUNCTIONS(.zh).md 附录总数行：`**N nodes** across M categories` / `共 **N 个节点**，分属 M 个类别`
  4) README(.zh).md 概述总数句与类别表格（整块重建）

用法:
    python comfydl/_update_readme.py             # 同步（写回文档）
    python comfydl/_update_readme.py --check     # 只校验，不写文件；存在差异退出码 1
退出码:
    0 = 无差异(或已同步); 1 = check 模式存在差异 / 执行失败
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True  # 不写入 .pyc

_ROOT = Path(__file__).resolve().parent            # comfydl/
_WORKSPACE = _ROOT.parent                          # ComfyDL_UI/
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

import comfydl  # noqa: E402  触发插件入口启动统计

# CATEGORY -> 中文名（README 表格使用）
_CAT_ZH = {
    'd2l/CV Models': 'CV 模型',
    'd2l/Datasets': '数据集',
    'd2l/Device Utils': '设备工具',
    'd2l/GAN': 'GAN',
    'd2l/Model Utils': '模型工具',
    'd2l/NLP Models': 'NLP 模型',
    'd2l/NLP Utils': 'NLP 工具',
    'd2l/ObjectDetection': '目标检测',
    'd2l/Segmentation': '语义分割',
    'd2l/Tensor Basic': '张量基础',
    'd2l/TorchOps': '张量运算',
    'd2l/Visualization': '可视化',
    'd2l/_Legacy/Model Utils': '已弃用 · 模型工具',
    'd2l/_Legacy/NLP Models': '已弃用 · NLP 模型',
    'd2l/_Legacy/Tensor Basic': '已弃用 · 张量基础',
    'utilities': 'Comfy 实用工具',
    'image/color': 'Comfy 图像/颜色',
    'image/transform': 'Comfy 图像/变换',
    'image': 'Comfy 图像',
}
# CATEGORY -> 英文名（README 英文表格使用；缺省回退为分类叶子名）
_CAT_EN = {
    'utilities': 'utilities (Comfy core)',
    'image/color': 'image/color (Comfy core)',
    'image/transform': 'image/transform (Comfy core)',
    'image': 'image (Comfy core)',
    'd2l/_Legacy/Model Utils': 'd2l/_Legacy/Model Utils',
    'd2l/_Legacy/NLP Models': 'd2l/_Legacy/NLP Models',
    'd2l/_Legacy/Tensor Basic': 'd2l/_Legacy/Tensor Basic',
}
# 分类说明（英文 / 中文，与 FUNCTIONS 附录描述一致）
_CAT_DESC_EN = {
    'd2l/CV Models': 'CNN fundamentals & model construction',
    'd2l/Datasets': 'Dataset download, load, preview & stats',
    'd2l/Device Utils': 'GPU/CPU device utilities',
    'd2l/GAN': 'GAN training updates',
    'd2l/Model Utils': 'Model info, mode, forward, layers, params, clone & persistence',
    'd2l/NLP Models': 'RNN/GRU/RNNLM, attention & Seq2Seq building blocks',
    'd2l/NLP Utils': 'Text tokenization & vocabularies',
    'd2l/ObjectDetection': 'Anchor boxes, IoU, NMS',
    'd2l/Segmentation': 'VOC semantic segmentation tools',
    'd2l/Tensor Basic': 'Tensor I/O, conv, transpose, broadcast, activation',
    'd2l/TorchOps': 'Loss, optimization, metrics',
    'd2l/Visualization': 'Plots, charts & bounding box visualization',
    'd2l/_Legacy/Model Utils': 'Deprecated (soft-archived): Model Mode; use the core Training Mode',
    'd2l/_Legacy/NLP Models': 'Deprecated (soft-archived): Add & Norm, Transformer Encoder Block/Encoder',
    'd2l/_Legacy/Tensor Basic': 'Deprecated (soft-archived): Broadcast, Reshape, Activation',
    'utilities': 'MessageBox, NoOp pass-through & benchmark timer',
    'image/color': 'Grayscale, normalize & brightness/contrast/saturation',
    'image/transform': 'Arbitrary-angle rotation + canvas expand',
    'image': 'Per-channel image batch statistics',
}
_CAT_DESC_ZH = {
    'd2l/CV Models': 'CNN 基础与模型构建',
    'd2l/Datasets': '数据集下载、加载、预览与统计',
    'd2l/Device Utils': 'GPU/CPU 设备查询',
    'd2l/GAN': 'GAN 训练更新',
    'd2l/Model Utils': '模型信息、模式、前向、层结构、参数、克隆与存取',
    'd2l/NLP Models': 'RNN/GRU/RNNLM、注意力与 Seq2Seq 构件',
    'd2l/NLP Utils': '文本分词与词表',
    'd2l/ObjectDetection': '锚框、IoU、NMS',
    'd2l/Segmentation': 'VOC 语义分割工具',
    'd2l/Tensor Basic': '张量 I/O、卷积、转置、广播、激活函数',
    'd2l/TorchOps': '损失、优化、评估指标',
    'd2l/Visualization': '图表与边界框可视化',
    'd2l/_Legacy/Model Utils': '已弃用（软归档）：Model Mode，纯 TENSOR 图改用核心 Training Mode',
    'd2l/_Legacy/NLP Models': '已弃用（软归档）：Add & Norm、Transformer Encoder Block/Encoder',
    'd2l/_Legacy/Tensor Basic': '已弃用（软归档）：Broadcast、Reshape、Activation',
    'utilities': 'MessageBox、NoOp 空操作与基准计时',
    'image/color': '灰度、归一化与亮度/对比度/饱和度',
    'image/transform': '任意角度旋转 + 画布扩展',
    'image': '图像批次逐通道统计',
}

# 正则
_HDR_RE = re.compile(r'^\| (Category|类别|分类) \|')
_SEP_RE = re.compile(r'^\|[\s:\-|]+\|$')
_DATA_RE = re.compile(r'^\| \*\*.+?\*\* \| \d+ \|')


def collect_stats():
    """实时统计注册表：{CATEGORY: count}。"""
    cats = Counter(
        getattr(cls, 'CATEGORY', 'd2l/Unknown')
        for cls in comfydl.NODE_CLASS_MAPPINGS.values()
    )
    return cats, sum(cats.values()), len(cats)


def header_to_category(label):
    """章节标题里的分类标签 -> 注册表分类键。

    'd2l / Datasets' -> 'd2l/Datasets'；'ComfyUI / image/color' -> 'image/color'。
    """
    if ' / ' not in label:
        return None
    prefix, rest = label.split(' / ', 1)
    prefix, rest = prefix.strip(), rest.strip()
    if prefix == 'd2l':
        return 'd2l/' + rest
    if prefix == 'ComfyUI':
        return rest
    return None


def process_functions_lines(lines, cats, total, num_cats, zh, changes):
    """处理 FUNCTIONS(.zh).md：章节标题、附录分类表、附录总数行。"""
    new_lines = []
    for line in lines:
        updated = line
        # 1) 章节标题：## N. d2l / <Cat> (N nodes) / ## N. ComfyUI / <cat> ...
        if zh:
            m = re.match(r'^(## \d+\. )(.+?)（(\d+) 个节点）$', line)
        else:
            m = re.match(r'^(## \d+\. )(.+?) \((\d+) nodes\)$', line)
        if m:
            cat_key = header_to_category(m.group(2))
            cnt = cats.get(cat_key)
            if cnt is not None and cnt != int(m.group(3)):
                updated = m.group(1) + m.group(2) + (
                    f'（{cnt} 个节点）' if zh else f' ({cnt} nodes)'
                )
                changes.append(f'章节标题数量: {cat_key} {m.group(3)} -> {cnt}')
        # 2) 附录分类表：| <category> | N |
        m = re.match(r'^\| ([^|]+?) \| (\d+) \|', line)
        if m:
            key, cur = m.group(1).strip(), int(m.group(2))
            cnt = cats.get(key)
            if cnt is not None and cnt != cur:
                updated = f'| {key} | {cnt} |' + line[len(m.group(0)):]
                changes.append(f'附录表数量: {key} {cur} -> {cnt}')
        # 3) 附录总数行（比较后再替换，保留前后缀）
        if zh:
            m = re.search(r'(共 \*\*)(\d+)( 个节点\*\*，分属 )(\d+)( 个类别)', line)
            if m and (int(m.group(2)) != total or int(m.group(4)) != num_cats):
                updated = (line[:m.start()] + m.group(1) + str(total) + m.group(3)
                           + str(num_cats) + m.group(5) + line[m.end():])
                changes.append(f'附录总数: -> {total} 个节点 / {num_cats} 个类别')
        else:
            m = re.search(r'(\*\*)(\d+)( nodes\*\* across )(\d+)( categories)', line)
            if m and (int(m.group(2)) != total or int(m.group(4)) != num_cats):
                updated = (line[:m.start()] + m.group(1) + str(total) + m.group(3)
                           + str(num_cats) + m.group(5) + line[m.end():])
                changes.append(f'appendix total: -> {total} nodes / {num_cats} categories')
        new_lines.append(updated)
    return new_lines


def process_readme_lines(lines, cats, zh, changes):
    """处理 README(.zh).md：概述总数句 + 整块重建类别表格。"""
    total, num = sum(cats.values()), len(cats)
    new_lines = []
    in_table = False
    old_rows = []
    for line in lines:
        if in_table:
            if _SEP_RE.match(line):
                continue                        # 跳过旧分隔行（新表自带）
            if _DATA_RE.match(line):
                old_rows.append(line)           # 暂存旧数据行用于比较
                continue
            # 表格结束：若旧数据行与统计不一致则重建
            desc = _CAT_DESC_ZH if zh else _CAT_DESC_EN
            names = _CAT_ZH if zh else _CAT_EN
            col1 = '类别' if zh else 'Category'
            col2 = '节点数' if zh else 'Count'
            col3 = '说明' if zh else 'Description'
            new_rows = []
            for cat in sorted(cats):
                leaf = cat.split('/', 1)[1] if '/' in cat else cat
                new_rows.append(f'| **{names.get(cat, leaf)}** | {cats[cat]} | {desc.get(cat, "")} |')
            if old_rows != new_rows:
                new_lines.append(f'| {col1} | {col2} | {col3} |')
                new_lines.append('|---|---|---|')
                new_lines.extend(new_rows)
                changes.append(f'README 类别表格已重建为 {num} 类')
            else:
                new_lines.extend(old_rows)
            in_table = False
            new_lines.append(line)              # 表格后的行（如空行）
            continue
        # 表格外：更新概述总数句（比较后再替换，保留前后缀）
        if zh:
            m = re.search(r'(提供 \*\*)(\d+)( 个节点\*\*，涵盖 )(\d+)( 个类别)', line)
            if m and (int(m.group(2)) != total or int(m.group(4)) != num):
                line = (line[:m.start()] + m.group(1) + str(total) + m.group(3)
                        + str(num) + m.group(5) + line[m.end():])
                changes.append('README 概述总数句已更新')
        else:
            m = re.search(r'(provides \*\*)(\d+)( nodes\*\* across )(\d+)( categories)', line)
            if m and (int(m.group(2)) != total or int(m.group(4)) != num):
                line = (line[:m.start()] + m.group(1) + str(total) + m.group(3)
                        + str(num) + m.group(5) + line[m.end():])
                changes.append('README 概述总数句已更新')
        if _HDR_RE.match(line):
            in_table = True                     # 表格开始（表头行不保留，重建）
            continue
        new_lines.append(line)
    return new_lines


def sync_file(path, cats, total, num_cats, zh, only_check):
    """同步单个文档，返回该文件的差异数量。"""
    if not path.exists():
        print(f'[SKIP] 未找到: {path}')
        return 0
    lines = path.read_text(encoding='utf-8').splitlines()
    changes = []
    if path.name.startswith('FUNCTIONS'):
        new_lines = process_functions_lines(lines, cats, total, num_cats, zh, changes)
    else:  # README / README_zh
        new_lines = process_readme_lines(lines, cats, zh, changes)
    if changes:
        if not only_check:
            path.write_text('\n'.join(new_lines) + '\n', encoding='utf-8')
            print(f'[UPDATED] {path.name} ({len(changes)} 处)')
        else:
            print(f'[DIFF] {path.name} ({len(changes)} 处)')
        for c in changes:
            print(f'    - {c}')
    else:
        print(f'[OK] {path.name} 与注册表一致')
    return len(changes)


def main():
    ap = argparse.ArgumentParser(description='ComfyDL 文档统计同步/校验')
    ap.add_argument('--check', action='store_true', help='只校验不写文件，有差异退出码 1')
    args = ap.parse_args()

    cats, total, num_cats = collect_stats()
    print(f'[INFO] 当前注册表: {total} 个节点 / {num_cats} 个分类')

    targets = [
        (_ROOT / 'FUNCTIONS.md', False),
        (_ROOT / 'FUNCTIONS_zh.md', True),
        (_ROOT / 'README.md', False),
        (_ROOT / 'README_zh.md', True),
    ]
    total_diff = 0
    for path, zh in targets:
        total_diff += sync_file(path, cats, total, num_cats, zh, args.check)

    if args.check:
        if total_diff:
            print(f'[RESULT] 存在 {total_diff} 处差异（见上）')
            sys.exit(1)
        print('[RESULT] 全部文档与注册表一致')
        sys.exit(0)
    if total_diff:
        print(f'[RESULT] 已同步 {total_diff} 处差异')
    else:
        print('[RESULT] 无需同步')
    sys.exit(0)


if __name__ == '__main__':
    main()
