![Banner](./banner.png)
<div align="center">
<h1>ComfyDL</h1>
<p>深度学习，只需几次点击！</p>
</div>
[English Version / 英文版](./README.md)

---

## 这是什么？

**ComfyDL** 让你通过连接 ComfyUI 中的节点来构建深度学习工作流——从 CNN 到 BERT——而无需编写代码。基于 `d2l` 代码库，它可视化、富有教育意义，非常适合快速原型开发。拖拽、连接，即刻看到结果。

---

## 安装

> **此项目需要 ComfyUI。** 如果你还没有，请从以下地址下载：[https://github.com/Comfy-Org/ComfyUI](https://github.com/Comfy-Org/ComfyUI)

1. 进入你的 `custom_nodes` 文件夹：

   <img src="./assets/1.png" alt="custom_nodes 文件夹位置" width="400" />

2. 克隆本仓库：
   ```bash
   git clone https://github.com/Cynthia-lxx/ComfyDL ./ComfyDL
   ```
3. 安装依赖：
   ```bash
   pip install -r ./ComfyDL/requirements.txt
   ```
4. 重启 ComfyUI。**你应该能在节点菜单中看到 ComfyDL 新增的节点。**

---

## 功能概览

ComfyDL 提供 **71 个节点**，涵盖 10 个类别：

| 类别 | 节点数 | 示例 |
|---|---|---|
| **CV 模型** | 5 | LeNet、ResNet-18、ResNeXt 块、2D 卷积 |
| **NLP 工具** | 5 | 分词、词汇构建/编码/解码 |
| **目标检测** | 10 | Box IOU / NMS、锚框生成、多框目标/检测 |
| **语义分割** | 4 | VOC 色表、标签映射、随机裁剪 |
| **张量运算** | 17 | 线性回归、掩码 softmax、准确率、BLEU、SGD、conv2d、转置、激活函数 |
| **可视化** | 13 | 绘图、图表（柱状/饼图/散点/直方/面积图）、热力图、图像网格、边界框 |
| **数据集** | 10 | Fashion-MNIST、Bananas、VOC、下载、DataLoader 预览与统计 |
| **GAN** | 2 | 判别器/生成器更新步骤 |
| **设备工具** | 3 | GPU 信息、尝试 GPU |
| **杂项** | 2 | 消息框、无操作透传 |

> 完整节点参考，请参阅 **[FUNCTIONS.md](./FUNCTIONS.md)**（英文）或 **[FUNCTIONS_zh.md](./FUNCTIONS_zh.md)**（中文）。

---

## 许可证

本项目基于 **MIT 许可证** 授权——详见 [LICENSE](./LICENSE) 文件。

---

## 特别鸣谢

ComfyDL 站在卓越的 **`d2l`**（动手学深度学习）社区的肩膀之上。

- **代码库：** 我们大量参考并借鉴了 [`d2l-pytorch`](https://github.com/dsgiitr/d2l-pytorch) 仓库的实现。其清晰、教科书级别的代码构成了许多节点的骨干。我们深深感谢所有在宽松的 **MIT-0** 许可证下贡献这一资源的开发者们。
- **灵感来源：** 本项目从设计到教学理念，都深受 **《动手学深度学习》**（PyTorch 版）一书的启发，它为掌握深度学习提供了最易于理解且实用的路径之一。

没有他们的远见与慷慨，就不会有本项目。我们鼓励大家去探索原书和仓库：

- **书籍（中文）：** https://zh.d2l.ai/
- **GitHub 仓库：** https://github.com/dsgiitr/d2l-pytorch

---

> **免责声明：** 本文档由 AI 翻译自英文原版 [README.md](./README.md)，可能存在不准确之处。如有歧义，请以英文原版为准。

![喵~](./assets/neko.jpg)