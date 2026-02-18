---
title: "LLM 全流程可视化：逐步拆解大语言模型的每一步计算"
date: 2026-02-18
draft: false
summary: "用一个真实的 4.3M 参数模型（nanoGPT + 西游记），逐步展示从输入「悟空道」到输出新字符的完整数据流。所有数值都是真实计算结果，不是模拟。"
categories: ["LLM"]
tags: ["nanoGPT", "可视化", "Transformer", "Embedding", "Attention", "教学"]
weight: 3
ShowToc: true
TocOpen: true
---

## 引言

当你在 ChatGPT 输入"请解释量子力学"，它是怎么一步步生成回答的？

今天我们用一个 **可以完全看透的小模型**，把这个过程从头到尾拆给你看。不是示意图，不是模拟数据——是直接打开模型内部，提取每一步的真实计算结果。

**我们的工具：**
- **模型：** 用《西游记》全文训练的 nanoGPT（4.3M 参数，4 层 Transformer）
- **输入：** "悟空道"
- **目标：** 看模型如何预测下一个字

> 这个模型和 GPT-4 使用完全相同的架构（Transformer），区别只在规模。理解了这个小模型，就理解了所有大模型的本质。

---

## 全局架构总览

在深入每一步之前，先看完整的数据流路径：

<div style="max-width: 620px; margin: 1.5em auto; font-size: 0.95em;">

<div style="border: 2px solid #4CAF50; border-radius: 8px; padding: 12px 16px; text-align: center; background: rgba(76,175,80,0.05);">
<strong>输入文本</strong><br>"悟空道"
</div>

<div style="text-align: center; font-size: 1.2em; color: #888; margin: 4px 0;">↓</div>

<div style="border: 2px solid #FF9800; border-radius: 8px; padding: 10px 16px; background: rgba(255,152,0,0.05);">
<strong>① Tokenization</strong> — 字符 → Token ID（词表: 4487）
</div>

<div style="text-align: center; font-size: 1.2em; color: #888; margin: 4px 0;">↓</div>

<div style="border: 2px solid #FF9800; border-radius: 8px; padding: 10px 16px; background: rgba(255,152,0,0.05);">
<strong>② Embedding 嵌入</strong> — Token ID → 向量（256 维）<br>
<span style="font-size: 0.85em; color: #888;">Token Embedding (4487×256) + Position Embedding (128×256)</span>
</div>

<div style="text-align: center; font-size: 1.2em; color: #888; margin: 4px 0;">↓</div>

<div style="border: 2px solid #2196F3; border-radius: 8px; padding: 10px 16px; background: rgba(33,150,243,0.05);">
<strong>③ Transformer × 4 层</strong> — 神经网络核心<br>
<div style="margin: 8px 0 0 16px; font-size: 0.9em; line-height: 1.6;">
Block 0: Attention(4头) → MLP(256→1024→256)<br>
Block 1: Attention(4头) → MLP(256→1024→256)<br>
Block 2: Attention(4头) → MLP(256→1024→256)<br>
Block 3: Attention(4头) → MLP(256→1024→256)
</div>
</div>

<div style="text-align: center; font-size: 1.2em; color: #888; margin: 4px 0;">↓</div>

<div style="border: 2px solid #9C27B0; border-radius: 8px; padding: 10px 16px; background: rgba(156,39,176,0.05);">
<strong>④ 输出层</strong> — 256 维 → Logits（4487 个）<br>
<span style="font-size: 0.9em;">Logits → ÷Temperature → Softmax → 概率分布</span>
</div>

<div style="text-align: center; font-size: 1.2em; color: #888; margin: 4px 0;">↓</div>

<div style="border: 2px solid #4CAF50; border-radius: 8px; padding: 10px 16px; background: rgba(76,175,80,0.05);">
<strong>⑤ 采样</strong> — Top-K 过滤 → 按概率随机选择
</div>

<div style="text-align: center; font-size: 1.2em; color: #888; margin: 4px 0;">↓</div>

<div style="border: 2px solid #4CAF50; border-radius: 8px; padding: 10px 16px; text-align: center; background: rgba(76,175,80,0.05);">
<strong>⑥ 输出新字符</strong> — Token ID → 解码 → 文字<br>
<span style="font-size: 0.9em;">↻ 拼回输入，重复以上过程（自回归）</span>
</div>

</div>

接下来逐步拆解每个环节，展示真实的中间数据。

---

## 阶段 1：原始文本输入

人类输入的文字，对计算机来说只是一串**字节**。先看看每个字符的 UTF-8 编码：

| 字符 | UTF-8 (十六进制) | UTF-8 (二进制) |
|:----:|:----------------:|:--------------:|
| 悟 | `E6 82 9F` | `11100110 10000010 10011111` |
| 空 | `E7 A9 BA` | `11100111 10101001 10111010` |
| 道 | `E9 81 93` | `11101001 10000001 10010011` |

**中文每个字占 3 个字节，英文每个字母只占 1 个字节。** 这是 UTF-8 编码的特点。

---

## 阶段 2：Tokenization — 文字变数字

LLM 不直接处理字节，而是将文本转换为 **Token ID**（数字编号）。

我们的模型使用**字符级**分词——每个字符就是一个 token，词表大小 4487 个唯一字符。

| 字符 | 查表操作 | Token ID |
|:----:|:-------:|:--------:|
| 悟 | `stoi['悟']` | 1342 |
| 空 | `stoi['空']` | 2784 |
| 道 | `stoi['道']` | 3915 |

结果：`"悟空道"` → `[1342, 2784, 3915]`

> **stoi** 是 "String TO Integer" 的缩写，就是一个字符→数字的查找表，在数据准备阶段从训练语料中自动构建。

### Token 的大小不是固定的

| 分词方式 | 一个 Token 代表什么 |
|:--------:|:-------------------:|
| 字符级（本模型） | 1 个字符 = 1 个 token |
| BPE（GPT-4） | 1 个 token ≈ 3-4 个英文字母 或 1-2 个汉字 |

Token ID 本身用 16-bit 整数存储（2 字节），但它代表的原文可长可短。

---

## 阶段 3：Token ID → 二进制 → 张量

Token ID 是整数，计算机用**二进制**存储。然后打包成 PyTorch 张量（Tensor）——这才是进入模型的真正输入。

| 字符 | Token ID | 二进制 (16-bit) |
|:----:|:--------:|:---------------:|
| 悟 | 1342 | `0000 0101 0011 1110` |
| 空 | 2784 | `0000 1010 1110 0000` |
| 道 | 3915 | `0000 1111 0100 1011` |

```python
torch.tensor([[1342, 2784, 3915]])   # 形状: [1, 3] (batch=1, seq_len=3)
```

**数据变换路径：** 文字世界 → 数字世界 → 二进制 → 张量 (Tensor)

---

## 阶段 4：Embedding — 给每个 Token 一张"数字身份证"

每个 Token ID 在 **Embedding 矩阵**中查找对应的一行向量。矩阵形状：`(4487 × 256)`——词表中每个字符都有一个 256 维的向量表示。

### Token Embedding（语义嵌入）

直接调用 `model.transformer.wte(idx)`，即 `nn.Embedding` 查表：

| 字符 | Token ID | 向量（前 6 维 / 共 256 维） |
|:----:|:--------:|:---------------------------:|
| 悟 | 1342 | `[-0.039, +0.010, -0.097, -0.018, -0.007, -0.060, ...]` |
| 空 | 2784 | `[+0.083, -0.061, -0.018, +0.049, +0.016, +0.015, ...]` |
| 道 | 3915 | `[+0.070, -0.041, +0.034, +0.092, +0.072, +0.032, ...]` |

### Position Embedding（位置编码）

让模型知道字的顺序。调用 `model.transformer.wpe(pos)`：

| 位置 | 向量（前 6 维 / 共 256 维） |
|:----:|:---------------------------:|
| 0 | `[+0.053, -0.065, +0.032, -0.008, -0.039, -0.024, ...]` |
| 1 | `[-0.010, -0.038, +0.014, +0.032, -0.019, +0.020, ...]` |
| 2 | `[+0.018, -0.038, +0.027, -0.070, -0.012, -0.066, ...]` |

**两者相加** → 每个 token 的初始表示，输出形状 `[1, 3, 256]`。

---

## 阶段 4B：Embedding 是怎么来的？

**一开始是随机数，通过训练逐渐学出有意义的值。**

### 训练前后对比

| 指标 | 随机初始化 | 训练后 | 变化 |
|:----:|:---------:|:------:|:----:|
| 标准差 | 0.0200 | 0.0604 | 3.0x |
| 值域 | [-0.093, 0.105] | [-0.453, 0.386] | 扩大 |
| 向量 L2 范数（均值） | 0.320 | 0.916 | 2.9x |

训练后，向量空间扩大了约 3 倍，常用字的向量更大更"自信"，罕见字的向量较小。

### 语义相似度的涌现

**同义词组：说、道、言、曰**（都表示"说话"）

| 字符对 | 随机初始化 | 训练后 | 变化 |
|:------:|:---------:|:------:|:----:|
| 说 vs 道 | -0.003 | **+0.240** | 靠近 ↑ |
| 说 vs 言 | -0.021 | **+0.368** | 靠近 ↑ |
| 道 vs 曰 | -0.055 | **+0.175** | 靠近 ↑ |

**场景词组：山、水、天**（都是自然/场景词）

| 字符对 | 随机初始化 | 训练后 | 变化 |
|:------:|:---------:|:------:|:----:|
| 山 vs 水 | -0.033 | **+0.394** | 靠近 ↑ |
| 山 vs 天 | +0.032 | **+0.358** | 靠近 ↑ |

**跨组对比**（"说话"类 vs "自然"类）

| 字符对 | 随机初始化 | 训练后 | 变化 |
|:------:|:---------:|:------:|:----:|
| 说 vs 山 | +0.054 | **-0.113** | 远离 ↓ |
| 说 vs 水 | -0.013 | **-0.065** | 远离 ↓ |

**关键发现：** 训练前所有向量都是随机噪声（相似度接近 0）。训练后，语义相关的字自动靠近，不相关的字保持距离甚至远离。**没有人告诉模型这些字"意思相近"——这个语义结构完全是从数据中涌现的。**

### 从预训练到对齐

大型 LLM 的 Embedding 经历多个训练阶段：

1. **随机初始化** → 所有向量都是噪声，无任何语义
2. **预训练（本模型所在阶段）** → 从语料中学到共现关系（说≈道，山≈水≈天）
3. **指令微调 (SFT)** → 学会"问题→回答"的模式，Embedding 微调
4. **RLHF/DPO 对齐** → 学会人类偏好，有害词的向量被推远

每个阶段都在现有基础上继续调整，不是重新来过。

---

## 阶段 5：Transformer — LLM 的核心

数据现在要穿过 **4 层 Transformer Block**。每一层都包含 Self-Attention（注意力）和 MLP（前馈网络）。

### Transformer 架构

<div style="max-width: 560px; margin: 1.5em auto; font-size: 0.9em;">

<div style="border: 1px solid #4CAF50; border-radius: 6px; padding: 8px 12px; text-align: center; background: rgba(76,175,80,0.08);">
<strong>输入</strong> (1×3×256)
</div>
<div style="text-align: center; color: #888; margin: 2px 0;">↓</div>

<div style="border: 1px solid #2196F3; border-radius: 6px; padding: 8px 12px; margin-bottom: 4px; background: rgba(33,150,243,0.05);">
<strong>Block 0</strong>: LayerNorm → Attention (4 头×64 维) → 残差连接<br>
<span style="margin-left: 4.2em;"></span>LayerNorm → MLP (256→1024→256) → 残差连接
</div>
<div style="text-align: center; color: #888; margin: 2px 0;">↓</div>

<div style="border: 1px solid #2196F3; border-radius: 6px; padding: 8px 12px; margin-bottom: 4px; background: rgba(33,150,243,0.05);">
<strong>Block 1</strong>: LayerNorm → Attention (4 头×64 维) → 残差连接<br>
<span style="margin-left: 4.2em;"></span>LayerNorm → MLP (256→1024→256) → 残差连接
</div>
<div style="text-align: center; color: #888; margin: 2px 0;">↓</div>

<div style="border: 1px solid #2196F3; border-radius: 6px; padding: 8px 12px; margin-bottom: 4px; background: rgba(33,150,243,0.05);">
<strong>Block 2</strong>: LayerNorm → Attention (4 头×64 维) → 残差连接<br>
<span style="margin-left: 4.2em;"></span>LayerNorm → MLP (256→1024→256) → 残差连接
</div>
<div style="text-align: center; color: #888; margin: 2px 0;">↓</div>

<div style="border: 1px solid #2196F3; border-radius: 6px; padding: 8px 12px; background: rgba(33,150,243,0.05);">
<strong>Block 3</strong>: LayerNorm → Attention (4 头×64 维) → 残差连接<br>
<span style="margin-left: 4.2em;"></span>LayerNorm → MLP (256→1024→256) → 残差连接
</div>
<div style="text-align: center; color: #888; margin: 2px 0;">↓</div>

<div style="border: 1px solid #FF9800; border-radius: 6px; padding: 8px 12px; text-align: center; background: rgba(255,152,0,0.08);">
<strong>Final LayerNorm</strong> → 输出 (1×3×256)
</div>

</div>

### 逐层数据变化

| 层 | 操作 | 输出形状 | 均值 | 标准差 | 最大值 |
|:--:|:----:|:-------:|:----:|:-----:|:-----:|
| 输入 | Embedding + Position | [1, 3, 256] | -0.0004 | 0.0604 | 0.2147 |
| Block 0 | Attn + MLP | [1, 3, 256] | 0.0217 | 1.2726 | 4.0916 |
| Block 1 | Attn + MLP | [1, 3, 256] | 0.0098 | 1.8565 | 5.4841 |
| Block 2 | Attn + MLP | [1, 3, 256] | 0.0031 | 2.2551 | 7.2026 |
| Block 3 | Attn + MLP | [1, 3, 256] | -0.0009 | 2.2807 | 7.4441 |
| 输出 | Final LayerNorm | [1, 3, 256] | -0.0004 | 1.3093 | 3.8789 |

观察标准差变化：数据经过每一层都在被变换和精炼，信息逐层浓缩。

### Self-Attention：模型在"看"什么？

第 0 层、第 0 个注意力头的 **Attention 矩阵**（3×3）。每一行表示该位置对前面所有位置的关注程度（总和=1.0）：

<div style="max-width: 360px; margin: 1em auto;">
<table style="width: 100%; text-align: center; border-collapse: collapse;">
<tr style="background: rgba(33,150,243,0.1);">
<th style="padding: 8px; border: 1px solid #ddd;"></th>
<th style="padding: 8px; border: 1px solid #ddd;">悟</th>
<th style="padding: 8px; border: 1px solid #ddd;">空</th>
<th style="padding: 8px; border: 1px solid #ddd;">道</th>
</tr>
<tr>
<td style="padding: 8px; border: 1px solid #ddd; font-weight: bold; background: rgba(33,150,243,0.1);">悟</td>
<td style="padding: 8px; border: 1px solid #ddd; background: rgba(255,0,0,0.25); font-weight: bold;">1.00</td>
<td style="padding: 8px; border: 1px solid #ddd; color: #ccc;">-</td>
<td style="padding: 8px; border: 1px solid #ddd; color: #ccc;">-</td>
</tr>
<tr>
<td style="padding: 8px; border: 1px solid #ddd; font-weight: bold; background: rgba(33,150,243,0.1);">空</td>
<td style="padding: 8px; border: 1px solid #ddd; background: rgba(255,0,0,0.18);">0.56</td>
<td style="padding: 8px; border: 1px solid #ddd; background: rgba(255,0,0,0.12);">0.44</td>
<td style="padding: 8px; border: 1px solid #ddd; color: #ccc;">-</td>
</tr>
<tr>
<td style="padding: 8px; border: 1px solid #ddd; font-weight: bold; background: rgba(33,150,243,0.1);">道</td>
<td style="padding: 8px; border: 1px solid #ddd; background: rgba(255,0,0,0.16);">0.51</td>
<td style="padding: 8px; border: 1px solid #ddd; background: rgba(255,0,0,0.10);">0.32</td>
<td style="padding: 8px; border: 1px solid #ddd; background: rgba(255,0,0,0.06);">0.18</td>
</tr>
</table>
</div>

**关键设计——因果掩码（Causal Mask）：** 每个字只能看到自己和前面的字，不能偷看后面的。矩阵右上角的 `-` 就是被遮住的位置。

从这个矩阵可以看出：生成"道"后面的字时，模型最关注的是"悟"（0.51），其次是"空"（0.32），最后才是"道"自己（0.18）。模型"知道"悟空是主语，正在说话。

---

## 阶段 6：输出层 — 从向量到概率

Transformer 输出的 256 维向量，经过最后一个线性层 `lm_head`，变成 4487 个分数（**logits**），每个分数对应词表中的一个字符。

### 完整链条

<div style="max-width: 480px; margin: 1em auto; font-size: 0.95em;">

<div style="border: 1px solid #888; border-radius: 6px; padding: 8px 12px; text-align: center;">
隐藏状态（256 维）
</div>
<div style="text-align: center; color: #888; margin: 2px 0;">↓ lm_head 线性变换</div>

<div style="border: 1px solid #888; border-radius: 6px; padding: 8px 12px; text-align: center;">
Logits（4487 个原始分数）
</div>
<div style="text-align: center; color: #888; margin: 2px 0;">↓ ÷ temperature (0.8)</div>

<div style="border: 1px solid #888; border-radius: 6px; padding: 8px 12px; text-align: center;">
缩放后的 Logits
</div>
<div style="text-align: center; color: #888; margin: 2px 0;">↓ Softmax: exp(x_i) / Σexp(x_j)</div>

<div style="border: 1px solid #9C27B0; border-radius: 6px; padding: 8px 12px; text-align: center; background: rgba(156,39,176,0.05);">
<strong>概率分布</strong>（4487 个概率，总和 = 1.0）
</div>

</div>

### Top-10 候选字符

<div style="max-width: 520px; margin: 1em auto;">

<div style="display: flex; align-items: center; margin: 4px 0;">
<span style="width: 100px; text-align: right; margin-right: 8px; font-size: 0.9em;"><strong>1.</strong> ： (ID=4483)</span>
<div style="flex: 1; background: #eee; border-radius: 4px; height: 22px; overflow: hidden;">
<div style="width: 98.6%; height: 100%; background: #4CAF50; border-radius: 4px;"></div>
</div>
<span style="width: 56px; text-align: right; margin-left: 8px; font-size: 0.9em;"><strong>98.6%</strong></span>
</div>

<div style="display: flex; align-items: center; margin: 4px 0;">
<span style="width: 100px; text-align: right; margin-right: 8px; font-size: 0.9em;"><strong>2.</strong> ， (ID=4481)</span>
<div style="flex: 1; background: #eee; border-radius: 4px; height: 22px; overflow: hidden;">
<div style="width: 4%; height: 100%; background: #2196F3; border-radius: 4px; min-width: 3px;"></div>
</div>
<span style="width: 56px; text-align: right; margin-left: 8px; font-size: 0.9em;">0.4%</span>
</div>

<div style="display: flex; align-items: center; margin: 4px 0;">
<span style="width: 100px; text-align: right; margin-right: 8px; font-size: 0.9em;"><strong>3.</strong> 士 (ID=833)</span>
<div style="flex: 1; background: #eee; border-radius: 4px; height: 22px; overflow: hidden;">
<div style="width: 3%; height: 100%; background: #2196F3; border-radius: 4px; min-width: 3px;"></div>
</div>
<span style="width: 56px; text-align: right; margin-left: 8px; font-size: 0.9em;">0.3%</span>
</div>

<div style="display: flex; align-items: center; margin: 4px 0;">
<span style="width: 100px; text-align: right; margin-right: 8px; font-size: 0.9em;"><strong>4.</strong> 人 (ID=106)</span>
<div style="flex: 1; background: #eee; border-radius: 4px; height: 22px; overflow: hidden;">
<div style="width: 1%; height: 100%; background: #2196F3; border-radius: 4px; min-width: 3px;"></div>
</div>
<span style="width: 56px; text-align: right; margin-left: 8px; font-size: 0.9em;">0.1%</span>
</div>

<div style="display: flex; align-items: center; margin: 4px 0;">
<span style="width: 100px; text-align: right; margin-right: 8px; font-size: 0.9em;"><strong>5-10.</strong> 了是我...</span>
<div style="flex: 1; background: #eee; border-radius: 4px; height: 22px; overflow: hidden;">
<div style="width: 1%; height: 100%; background: #ccc; border-radius: 4px; min-width: 3px;"></div>
</div>
<span style="width: 56px; text-align: right; margin-left: 8px; font-size: 0.9em;">< 0.1%</span>
</div>

</div>

模型以 98.6% 的概率认为"悟空道"后面应该是冒号"："——这说明模型学到了一个很强的语言模式：**"某某道"是说话的引导，后面应该跟冒号和引号。**

---

## 阶段 7：采样 — 概率变字符

用 `torch.multinomial` 从概率分布中随机采样一个 token。就像一个加权骰子——概率越高的字符越可能被选中。

**反向解码路径：**

| 步骤 | 操作 | 结果 |
|:----:|:----:|:----:|
| 1 | 概率采样 | 从 4487 个候选中选中 |
| 2 | Token ID | 4483 |
| 3 | 二进制 | `0001 0001 1000 0011` |
| 4 | 查表 itos | `itos[4483]` |
| 5 | 输出字符 | **：**（概率: 98.6%） |

序列更新：`"悟空道"` + `"："` → `"悟空道："`

---

## 阶段 8：自回归循环

把新生成的字符拼回输入序列，重复上述全部过程，继续生成。这就是**自回归（Autoregressive）**——用自己的输出作为下一步的输入。

<div style="max-width: 480px; margin: 1.5em auto; border: 2px solid #FF9800; border-radius: 8px; padding: 16px; text-align: center; background: rgba(255,152,0,0.05);">
<div>输入序列 → Embedding → Transformer → Logits</div>
<div style="display: flex; justify-content: space-between; align-items: center; margin: 8px 0;">
<span style="font-size: 1.5em;">↑</span>
<span style="font-size: 1.5em;">↓</span>
</div>
<div>采样新 Token ← 概率分布</div>
<div style="margin-top: 8px; font-size: 0.85em; color: #888;">↻ 循环重复</div>
</div>

生成 10 个 token 后的结果：

> **悟空道：不知之人，是我师父**

这段文字完全是模型**编出来**的——不是从原文中检索的，而是基于学到的统计模式一个字一个字生成的。每次运行结果都不同，因为采样有随机性。

---

## 总结：5 条核心洞察

### 完整数据流

> 文字 → UTF-8 字节 → Token ID → 二进制 → 张量 → Embedding 查表 → Transformer (×4层) → Logits (原始分数) → Softmax 概率 → 采样 → Token ID → 查表 itos → 文字

### 关键洞察

1. **LLM 的全部输入/输出都是数字**，"理解"源自海量数据的统计规律
2. **Embedding 不是人类定义的**，而是训练中自动学习的分布式表示——同义词自动靠近
3. **Transformer 的注意力机制**让模型能"关注"上下文中的相关信息——我们可以直接看到 Attention 矩阵
4. **生成是逐字进行的**（自回归），每次只预测下一个 token
5. **Temperature 和 Top-K** 控制了输出的"创造性" vs "确定性"

### 这不是黑盒

本文所有数据都来自**真实的模型计算**——直接调用 `model.py` 中的各个子模块（`wte`、`wpe`、`Block`、`ln_f`、`lm_head`），用 `torch.no_grad()` 逐步提取中间张量。所有数值都是真实计算结果，不是模拟或预设数据。

同样的架构、同样的计算过程，在 GPT-4 中以完全相同的方式运行，只是维度从 256 变成了 12288，层数从 4 变成了 96+。

---

## SLM vs LLM：小模型与大模型

| 对比项 | 本文的 demo 模型 (SLM) | GPT-4 级别 (LLM) |
|:------:|:---------------------:|:----------------:|
| 参数量 | 4.3M（430 万） | ~1.8T（1.8 万亿） |
| 层数 | 4 | 96~120 |
| 维度 | 256 | 12288+ |
| Attention 矩阵 | 3×3 = 9 个值 | 128K×128K = 160 亿个值 |
| 能力 | 模仿西游记文风 | 对话、推理、编程、翻译 |
| 可观测性 | 完全透明 | 可追踪但人脑无法理解 |

**架构完全相同，区别只在规模。** 这就是用小模型做教学的价值——用一个可以完全看透的模型，建立对大模型工作原理的直觉。

---

## 动手试试

如果你想亲自运行这个可视化演示：

```bash
# 环境准备
source ~/ai-lab-venv/bin/activate
cd ~/nanoGPT

# 交互模式（按回车逐步推进）
python demo_llm_pipeline.py

# 自动播放
python demo_llm_pipeline.py --auto

# 指定模型和输入
python demo_llm_pipeline.py --model xiyouji-big --start "唐僧道" --gen_tokens 50

# Shakespeare 模型
python demo_llm_pipeline.py --model shakespeare --auto
```

可选模型：

| 参数 | 模型 | 默认 prompt |
|:----:|:----:|:----------:|
| `xiyouji`（默认） | 西游记字符级 | "悟空道" |
| `xiyouji-big` | 西游记大模型 | "悟空道" |
| `shakespeare` | Shakespeare 字符级 | "ROMEO:" |
| `shakespeare-big` | Shakespeare 大模型 | "ROMEO:" |
