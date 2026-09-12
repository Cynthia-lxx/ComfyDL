![Banner](./banner.png)
<div align="center">
<h1>ComfyDL</h1>
<p>DeepLearning is just a few clicks away!</p>
  <a href="./README_zh.md">中文版本 / Chinese</a>
</div>


---

## What's This?

**ComfyDL** lets you build deep learning workflows — from CNNs to BERT and beyond — by connecting nodes in ComfyUI, not by writing code. Built on a foundation inspired by the `d2l` codebase, it keeps evolving as we develop more useful nodes — visual, educational, and great for rapid prototyping. Drag, connect, and see results instantly.

**ComfyDL has its own GUI version:** [ComfyDL_UI](https://github.com/Cynthia-lxx/ComfyDL_UI) ships ComfyDL as a built-in node pack on top of a ComfyUI runtime fork — see [Installation](#installation).

## Examples in Action

> **Note:** The example workflows that used to ship with ComfyDL have been **archived**. They predate the
> ongoing node refactor, so they no longer match the current node set and are no longer offered as workflow
> templates in the UI. They are kept for reference under
> [`example_workflows/_archived/`](./example_workflows/_archived), and a fresh set of runnable examples will be
> published once the refactor is complete.

### The Mysterious "?"
A tiny node with a toggle that does... something. Try it and see.

![The Mysterious ?](./assets/example_what.png)

---

## Installation

There are two ways to run ComfyDL: the GUI build, or the classic manual install into an
existing ComfyUI.

### Option 1 — the GUI build: [ComfyDL_UI](https://github.com/Cynthia-lxx/ComfyDL_UI)

[ComfyDL_UI](https://github.com/Cynthia-lxx/ComfyDL_UI) is a runtime fork of ComfyUI that
already ships ComfyDL as a built-in node pack. No `custom_nodes` step, no separate dependency
install, no manual wiring — only the fork itself:

```bash
git clone https://github.com/Cynthia-lxx/ComfyDL_UI
cd ComfyDL_UI
penv\Scripts\python.exe main.py     # Windows
penv/bin/python main.py             # Linux / macOS
```

Its README covers the details: <https://github.com/Cynthia-lxx/ComfyDL_UI#readme>.

### Option 2 — the classic manual install into an existing ComfyUI

> **This option requires ComfyUI.** If you don't have it, download from: [https://github.com/Comfy-Org/ComfyUI](https://github.com/Comfy-Org/ComfyUI)

1. Navigate to your `custom_nodes` folder:
   
   <img src="./assets/1.png" alt="custom_nodes folder location" width="400" />

2. Clone this repository:
   ```bash
   git clone https://github.com/Cynthia-lxx/ComfyDL ./ComfyDL
   ```
3. Install dependencies:
   ```bash
   pip install -r ./ComfyDL/requirements.txt
   ```
4. Restart ComfyUI. **You should see the new ComfyDL nodes appear in the node menu.**


---

## Function Overview

The built-in node library ships **177 nodes** across 32 categories — **109 provided by ComfyDL**
plus 68 ComfyUI core nodes (`Network & Layers` → `Activation` 14 + `Basic` 8 + `Normalization` 7
+ `Regularization` 1 + `Training` 11 + `Pooling` 2 + `Convolution` 2; `model` → `loaders` 7
+ `merging` 11 + `latent` 2 + `conditioning` 2; `3d` → `Preview 3D` 1) that were added on top of it:

| Category | Count | Description |
|---|---|---|
| **CV Models** | 5 | CNN fundamentals & model construction |
| **Datasets** | 10 | Dataset download, load, preview & stats |
| **Device Utils** | 3 | GPU/CPU device utilities |
| **GAN** | 2 | GAN training updates |
| **Model Utils** | 7 | Model info, mode, forward, layers, params, clone & persistence |
| **NLP Models** | 13 | RNN/GRU/RNNLM, attention & Seq2Seq building blocks |
| **NLP Utils** | 5 | Text tokenization & vocabularies |
| **ObjectDetection** | 10 | Anchor boxes, IoU, NMS |
| **Segmentation** | 4 | VOC semantic segmentation tools |
| **Tensor Basic** | 5 | Tensor I/O, conv, transpose, broadcast, activation |
| **TorchOps** | 10 | Loss, optimization, metrics |
| **Visualization** | 13 | Plots, charts & bounding box visualization |
| **d2l/_Legacy/Model Utils** | 1 | Deprecated (soft-archived): Model Mode; use the core Training Mode |
| **d2l/_Legacy/NLP Models** | 3 | Deprecated (soft-archived): Add & Norm, Transformer Encoder Block/Encoder |
| **d2l/_Legacy/Tensor Basic** | 3 | Deprecated (soft-archived): Broadcast, Reshape, Activation |
| **image (Comfy core)** | 1 | Per-channel image batch statistics |
| **image/color (Comfy core)** | 3 | Grayscale, normalize & brightness/contrast/saturation |
| **image/transform (Comfy core)** | 1 | Arbitrary-angle rotation + canvas expand |
| **utilities (Comfy core)** | 4 | MessageBox, NoOp pass-through & benchmark timer |
| **conversion** | 6 |  |

> The 20 categories above are the ones ComfyDL itself provides; the other 12 categories of the
> shipped library (`Network & Layers/*`, `model/*` and `3d`, 68 nodes in total) are pure ComfyUI
> core categories.
>
> For the complete node reference, see **[FUNCTIONS.md](./FUNCTIONS.md)** (English) or **[FUNCTIONS_zh.md](./FUNCTIONS_zh.md)** (中文).

---

## Repository Layout

```
ComfyDL/
├── src/d2lcore/     # D2L-inspired core implementation — a reference layer (torch.py, ...)
├── nodes/           # ComfyUI node definitions (thin mapping layer)
│                    #   incl. self-developed model_utils.py, image_tools.py &
│                    #   nlp model wrappers (model_nlp.py, model_attention.py, model_seq2seq.py)
└── example_workflows/  # Archived sample workflow JSONs (see _archived/, no longer offered as templates)
```

> **Note:** A mirror copy of `src/d2lcore/` also exists at the repository root as `d2lcore/` (outside the plugin folder). They must be kept in sync whenever the D2L core logic is modified.

---

## License

This project is licensed under the **GNU General Public License v3.0** (or any later version) —
see the [LICENSE](./LICENSE) file for details. ComfyDL is distributed as the node pack of the
[ComfyDL_UI](https://github.com/Cynthia-lxx/ComfyDL_UI) project and follows the same license;
upstream ComfyUI keeps its own copyright and maintainers.

---

## Special Thanks

ComfyDL stands on the shoulders of the incredible **`d2l`** (Dive into Deep Learning) community.

- **Codebase:** We heavily reference and adapt implementations from the [`d2l-pytorch`](https://github.com/dsgiitr/d2l-pytorch) repository. Its clear, textbook-grade code serves as a high-quality reference and starting point for many of our nodes — and continues to guide the development of new ones. We are deeply grateful to all contributors who made this resource available under the permissive **MIT-0** license.
- **Inspiration:** The design and pedagogical philosophy behind this project are fundamentally inspired by the book **《Dive into Deep Learning》** (《动手学深度学习》PyTorch版), which offers one of the most accessible and practical paths to mastering deep learning.

This project would not exist without their vision and generosity. We encourage everyone to explore the original book and repository:

- **Book (Chinese):** https://zh.d2l.ai/
- **GitHub Repository:** https://github.com/dsgiitr/d2l-pytorch

---

![Meow~](./assets/neko.jpg)
