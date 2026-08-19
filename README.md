![Banner](./banner.png)
<div align="center">
<h1>ComfyDL</h1>
<p>DeepLearning is just a few clicks away!</p>
</div>
[中文版本 / Chinese](./README_zh.md)

---

## What's This?

**ComfyDL** lets you build deep learning workflows — from CNNs to BERT and beyond — by connecting nodes in ComfyUI, not by writing code. Built on a foundation inspired by the `d2l` codebase, it keeps evolving as we develop more useful nodes — visual, educational, and great for rapid prototyping. Drag, connect, and see results instantly.

---

## Installation

> **This project requires ComfyUI.** If you don't have it, download from: [https://github.com/Comfy-Org/ComfyUI](https://github.com/Comfy-Org/ComfyUI)

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

ComfyDL provides **105 nodes** across 14 categories:

| Category | Count | Description |
|---|---|---|
| **CV Models** | 5 | CNN fundamentals & model construction |
| **Datasets** | 10 | Dataset download, load, preview & stats |
| **Device Utils** | 3 | GPU/CPU device utilities |
| **GAN** | 2 | GAN training updates |
| **Image Tools** | 9 | Resize, normalize, flip, rotate, crop, adjust, blur & stats |
| **Misc** | 3 | Windows MessageBox, NoOp pass-through & timing |
| **Model Utils** | 8 | Model info, mode, forward, layers, params, clone & persistence |
| **NLP Models** | 16 | RNN/GRU/RNNLM, attention & Seq2Seq model building blocks |
| **NLP Utils** | 5 | Text tokenization & vocabularies |
| **ObjectDetection** | 10 | Anchor boxes, IoU, NMS |
| **Segmentation** | 4 | VOC semantic segmentation tools |
| **Tensor Basic** | 8 | Tensor I/O, conv, transpose, broadcast, reshape, activation |
| **TorchOps** | 10 | Loss, optimization, metrics |
| **Visualization** | 12 | Plots, charts & bounding box visualization |

> For the complete node reference, see **[FUNCTIONS.md](./FUNCTIONS.md)** (English) or **[FUNCTIONS_zh.md](./FUNCTIONS_zh.md)** (中文).

---

## Repository Layout

```
ComfyDL/
├── src/d2lcore/     # D2L-inspired core implementation — a reference layer (torch.py, ...)
├── nodes/           # ComfyUI node definitions (thin mapping layer)
│                    #   incl. self-developed model_utils.py, image_tools.py &
│                    #   nlp model wrappers (model_nlp.py, model_attention.py, model_seq2seq.py)
└── example_workflows/  # Sample workflow JSONs
```

> **Note:** A mirror copy of `src/d2lcore/` also exists at the repository root as `d2lcore/` (outside the plugin folder). They must be kept in sync whenever the D2L core logic is modified.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.

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
