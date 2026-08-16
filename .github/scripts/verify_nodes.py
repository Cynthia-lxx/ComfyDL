"""ComfyDL CI verification.

Per README, deployment = clone into ComfyUI's custom_nodes + `pip install -r
requirements.txt`. Importing the ComfyDL package (as ComfyUI does at startup)
triggers node registration. This script:

1. Asserts that nodes register cleanly (class/display mappings are non-empty
   and have equal size).
2. Asserts that every non-archived example workflow references only
   registered node types.
"""

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ComfyDL  # noqa: E402  # triggers node registration, like ComfyUI startup
from ComfyDL.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

# --- 1. Node registration sanity -----------------------------------------
if not NODE_CLASS_MAPPINGS:
    raise SystemExit("FAIL: no nodes registered after importing ComfyDL")

if len(NODE_CLASS_MAPPINGS) != len(NODE_DISPLAY_NAME_MAPPINGS):
    raise SystemExit(
        "FAIL: NODE_CLASS_MAPPINGS (%d) != NODE_DISPLAY_NAME_MAPPINGS (%d)"
        % (len(NODE_CLASS_MAPPINGS), len(NODE_DISPLAY_NAME_MAPPINGS))
    )

print(f"OK: {len(NODE_CLASS_MAPPINGS)} nodes registered")

categories = {}
for cls in NODE_CLASS_MAPPINGS.values():
    cat = getattr(cls, "CATEGORY", "ComfyDL/Unknown")
    categories[cat] = categories.get(cat, 0) + 1
for cat in sorted(categories):
    print(f"  {cat}: {categories[cat]}")

# --- 2. Example workflows reference only registered types ------------------
missing = {}
for path in sorted(glob.glob("example_workflows/**/*.json", recursive=True)):
    if "_archived" in path:  # archived workflows are intentional legacy
        continue
    with open(path, encoding="utf-8") as fh:
        workflow = json.load(fh)
    for node in workflow.get("nodes", []):
        node_type = node.get("type")
        if node_type and node_type not in NODE_CLASS_MAPPINGS:
            missing.setdefault(path, []).append(node_type)

if missing:
    for path, types in missing.items():
        print(f"FAIL: {path} references unknown node types: {sorted(set(types))}")
    raise SystemExit(1)

print("OK: all example workflows reference registered node types")
