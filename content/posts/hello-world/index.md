---
title: "你好，世界！— 博客正式上线"
date: 2026-02-17
draft: false
summary: "AI 学习笔记博客正式上线！在这里我会分享 AI 基础知识、大语言模型实战经验和实用工具教程。"
categories: ["AI 基础"]
tags: ["公告", "博客"]
weight: 1
---

## 欢迎来到 AI 学习笔记！

大家好！这是我的第一篇博客文章。作为一名 AI 培训讲师，我一直想找一个地方把教学内容沉淀下来，方便大家随时查阅复习。现在，这个博客就是这个地方了。

## 这里会分享什么？

### 1. AI 基础知识

从最基本的概念讲起：

- 什么是神经网络？
- Transformer 架构是怎么工作的？
- Tokenization（分词）为什么重要？

### 2. 大语言模型 (LLM) 实战

动手操作，不只是理论：

```python
# 一个简单的例子：用 nanoGPT 生成文本
python sample.py --out_dir=out-xiyouji --start="悟空道"
```

### 3. 工具教程

实用工具的上手指南，包括 Ollama、HuggingFace、PyTorch 等。

### 4. 教学视频

课程录像和技术讲解视频也会嵌入到文章中。

## Shortcode 演示

本博客支持多种富媒体嵌入方式，方便在文章中展示各类内容。

### B 站视频嵌入

在 Markdown 中使用以下语法即可嵌入 B 站视频：

```markdown
{{</* bilibili id="BVxxxxxxxx" title="视频标题" */>}}
```

### 微信二维码

```markdown
{{</* wechat-qr src="/images/qrcode.png" caption="扫码关注公众号" */>}}
```

### HTML5 视频

```markdown
{{</* video src="video.mp4" type="html5" title="演示视频" */>}}
```

## 写在最后

这个博客完全开源，使用 Hugo + PaperMod 主题搭建，部署在 GitHub Pages 上，**零成本运行**。

如果你也想搭建类似的博客，欢迎参考本站源码。后续我也会写一篇详细的搭建教程。

期待与你在这里交流，一起学习 AI！
