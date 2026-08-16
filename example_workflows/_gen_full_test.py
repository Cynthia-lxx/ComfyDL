#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ComfyDL 全节点测试工作流生成器
==============================
遍历当前 NODE_CLASS_MAPPINGS，为每个节点按 INPUT_TYPES 生成 widget 默认值，
输出一个包含全部已注册节点的 ComfyUI 工作流 JSON（不连线，links 为空），
落到 example_workflows/ComfyDL_AllNodes_Generated.json。

- 节点数量不硬编码：动态遍历注册表，后续新增节点会自动包含。
- 生成后可运行 `python _validate_wf.py example_workflows/ComfyDL_AllNodes_Generated.json` 校验。

用法:
    python example_workflows/_gen_full_test.py
退出码:
    0 = 成功; 1 = 失败
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True  # 不写入 .pyc

_WORKSPACE = Path(__file__).resolve().parent.parent.parent  # f:/Dev/ComfyDL_Refs
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

import ComfyDL  # noqa: E402

# ComfyUI 视为 widget 的内置标量类型（其余类型走输入槽位）
_WIDGET_TYPES = {'INT', 'FLOAT', 'STRING', 'BOOLEAN'}

# 输出目录：与脚本同目录（ComfyDL/example_workflows/）
_OUT_FILE = Path(__file__).resolve().parent / 'ComfyDL_AllNodes_Generated.json'


def gen_default(param, type_spec):
    """从 INPUT_TYPES 的类型标注生成默认 widget 值（与 cdl_smoke_tests 同思路）。"""
    t = type_spec[0]
    meta = type_spec[1] if len(type_spec) > 1 else {}
    if t == 'INT':
        return meta.get('default', 1)
    if t == 'FLOAT':
        return meta.get('default', 0.5)
    if t == 'BOOLEAN':
        return meta.get('default', True)
    if t == 'STRING':
        return meta.get('default', '')
    if isinstance(t, list):  # combo 下拉
        return t[0]
    return None  # cdlTensor / cdlModel / IMAGE ... 走输入槽


def is_widget(type_spec):
    t = type_spec[0]
    if isinstance(t, list):  # combo 下拉
        return True
    return t in _WIDGET_TYPES


def input_specs(node_cls):
    it = node_cls.INPUT_TYPES()
    return (it.get('required') or {}, it.get('optional') or {})


def class_attr(node_cls, name, default=None):
    v = getattr(node_cls, name, default)
    if callable(v):
        try:
            return v()
        except TypeError:
            return v
    return v


def build_node(node_cls, nid, x, y):
    """为单个节点类生成 ComfyUI UI 格式的节点 dict。"""
    required, optional = input_specs(node_cls)
    ordered = list(required.items()) + list(optional.items())

    inputs = []
    widgets = []
    for pname, spec in ordered:
        if is_widget(spec):
            widgets.append(gen_default(pname, spec))
        else:
            inputs.append({'name': pname, 'type': spec[0], 'link': None})

    return_types = class_attr(node_cls, 'RETURN_TYPES', ()) or ()
    return_names = class_attr(node_cls, 'RETURN_NAMES', None)
    outputs = []
    for i, rt in enumerate(return_types):
        name = return_names[i] if return_names and i < len(return_names) else f'output_{i}'
        outputs.append({'name': name, 'type': rt, 'links': None, 'slot_index': i})

    return {
        'id': nid,
        'type': node_cls.__name__,
        'pos': [x, y],
        'size': [315, 200],
        'flags': {},
        'order': nid,
        'mode': 0,
        'inputs': inputs,
        'outputs': outputs,
        'properties': {'Node name for S&R': node_cls.__name__},
        'widgets_values': widgets,
    }


def main():
    mappings = ComfyDL.NODE_CLASS_MAPPINGS
    if not mappings:
        print('[FAIL] NODE_CLASS_MAPPINGS 为空')
        sys.exit(1)

    nodes = []
    nid = 1
    y = 0
    cat_count = Counter()
    for name in sorted(mappings):
        cls = mappings[name]
        node = build_node(cls, nid, 50, y)
        nodes.append(node)
        cat_count[getattr(cls, 'CATEGORY', 'ComfyDL/Unknown')] += 1
        y += 240
        nid += 1

    wf = {
        'version': 0.4,
        'last_node_id': nid - 1,
        'last_link_id': 0,
        'nodes': nodes,
        'links': [],
        'groups': [],
        'config': {},
        'extra': {},
    }

    _OUT_FILE.write_text(json.dumps(wf, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'[OK] 已生成 {len(nodes)} 个节点 -> {_OUT_FILE}')
    for cat in sorted(cat_count):
        print(f'  {cat}: {cat_count[cat]}')
    sys.exit(0)


if __name__ == '__main__':
    main()
