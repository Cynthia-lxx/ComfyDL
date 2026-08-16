import os
import glob

nodes_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nodes')

replacements = [
    ('CATEGORY = "d2lcore/', 'CATEGORY = "ComfyDL/'),
    ('"cdl ', '"'),
    ('"CDL_TENSOR"', '"cdlTensor"'),
    ('"CDL_MODEL"', '"cdlModel"'),
    ('"CDL_VOCAB"', '"cdlVocab"'),
    ('"CDL_DATALOADER"', '"cdlDataloader"'),
    ('"CDL_BBOX"', '"cdlBbox"'),
]

for path in glob.glob(os.path.join(nodes_dir, '*.py')):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    old_text = text
    for old, new in replacements:
        text = text.replace(old, new)
    if text != old_text:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f'Updated: {os.path.basename(path)}')
    else:
        print(f'No changes: {os.path.basename(path)}')
