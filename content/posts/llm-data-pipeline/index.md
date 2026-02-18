---
title: "从文本到模型：LLM 数据处理全流程详解"
date: 2026-02-17
draft: false
summary: "深入了解大语言模型 (LLM) 的数据处理流程：从原始文本到 Tokenization，从 Embedding 到 Transformer，一步步拆解 LLM 的工作原理。基于 nanoGPT 实战项目。"
categories: ["LLM"]
tags: ["nanoGPT", "Tokenization", "Embedding", "Transformer", "教程"]
weight: 2
ShowToc: true
TocOpen: true
---

## 引言

你有没有好奇过，ChatGPT 是怎么"读懂"你输入的文字，又是怎么"写出"回答的？

今天我们用一个可以完全看透的小模型 —— [nanoGPT](https://github.com/karpathy/nanoGPT)，来拆解大语言模型 (LLM) 从输入到输出的完整数据流。

**核心观点：LLM 的架构和原理与我们的小模型完全相同，区别只在规模。** 理解了小模型，就理解了大模型的本质。

## 全流程概览

<div style="max-width: 640px; margin: 1.5em auto; font-size: 0.95em;">

<div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: center; gap: 6px 4px; line-height: 1.8;">
<span style="border: 2px solid #4CAF50; border-radius: 6px; padding: 4px 10px; background: rgba(76,175,80,0.08);"><strong>"悟空道"</strong><br><span style="font-size:0.8em;color:#888;">文字</span></span>
<span style="color:#aaa;">→</span>
<span style="border: 2px solid #FF9800; border-radius: 6px; padding: 4px 10px; background: rgba(255,152,0,0.06);"><strong>UTF-8 编码</strong><br><span style="font-size:0.8em;color:#888;">字节</span></span>
<span style="color:#aaa;">→</span>
<span style="border: 2px solid #FF9800; border-radius: 6px; padding: 4px 10px; background: rgba(255,152,0,0.06);"><strong>Token ID</strong><br><span style="font-size:0.8em;color:#888;">数字</span></span>
<span style="color:#aaa;">→</span>
<span style="border: 2px solid #2196F3; border-radius: 6px; padding: 4px 10px; background: rgba(33,150,243,0.06);"><strong>Embedding</strong><br><span style="font-size:0.8em;color:#888;">高维向量</span></span>
<span style="color:#aaa;">→</span>
<span style="border: 2px solid #2196F3; border-radius: 6px; padding: 4px 10px; background: rgba(33,150,243,0.06);"><strong>Transformer</strong><br><span style="font-size:0.8em;color:#888;">神经网络</span></span>
<span style="color:#aaa;">→</span>
<span style="border: 2px solid #9C27B0; border-radius: 6px; padding: 4px 10px; background: rgba(156,39,176,0.06);"><strong>Softmax</strong><br><span style="font-size:0.8em;color:#888;">概率</span></span>
<span style="color:#aaa;">→</span>
<span style="border: 2px solid #9C27B0; border-radius: 6px; padding: 4px 10px; background: rgba(156,39,176,0.06);"><strong>采样</strong><br><span style="font-size:0.8em;color:#888;">选择</span></span>
<span style="color:#aaa;">→</span>
<span style="border: 2px solid #4CAF50; border-radius: 6px; padding: 4px 10px; background: rgba(76,175,80,0.08);"><strong>"："</strong><br><span style="font-size:0.8em;color:#888;">新文字</span></span>
</div>

</div>

接下来我们逐步拆解每个环节。

---

## 第一步：Tokenization — 把文字变成数字

神经网络不能直接处理文字，第一步就是把文字转换成数字。这个过程叫 **Tokenization（分词/标记化）**。

### 字符级分词

最简单的方式：每个字符就是一个 token。

```
"悟空道" → 每个字一个编号 → [1342, 2784, 3915]
词表大小：Shakespeare = 65 个字符，西游记 ≈ 3000+ 个字符
```

### BPE 分词 — GPT 系列使用

把常见的字符组合合并成一个 token，效率更高：

```
"Hello World" → ["Hello", " World"] → [9906, 2159]
词表大小：GPT-2 = 50,257 个 token
```

### 中英文 Token 效率差异

一个有趣的发现：**同样含义的内容，中文需要约 1.4-1.5 倍的 token。**

| 文本 | Token 数量 |
|------|-----------|
| The quick brown fox jumps over the lazy dog | 9 |
| 敏捷的棕色狐狸跳过了懒狗（同义） | 13 |

这意味着在 ChatGPT 等模型中，中文消耗更多 token（花更多钱）。这也是为什么我们在 nanoGPT 中用**字符级**分词来训练西游记 —— 对中文更公平。

---

## 第二步：Embedding — 给每个 Token 一张"数字身份证"

模型不能直接用 Token ID（一个整数）做计算，需要把它转换成一个**高维向量**（一组浮点数）。

<div style="max-width: 520px; margin: 1em auto; font-size: 0.95em;">
<div style="border: 2px solid #FF9800; border-radius: 8px; padding: 10px 16px; text-align: center; background: rgba(255,152,0,0.05);">
<strong>Token ID: 1342</strong>（"悟"）
</div>
<div style="text-align: center; font-size: 1.2em; color: #888; margin: 4px 0;">↓ <span style="font-size: 0.75em;">在 Embedding 矩阵中查找第 1342 行</span></div>
<div style="border: 2px solid #2196F3; border-radius: 8px; padding: 10px 16px; text-align: center; background: rgba(33,150,243,0.05);">
<strong>向量:</strong> <code>[-0.039, +0.010, -0.097, +0.048, ..., +0.031]</code><br>
<span style="font-size: 0.85em; color: #888;">256 维浮点数</span>
</div>
</div>

### Embedding 是怎么来的？

**一开始是随机数，通过训练逐渐学出有意义的值。**

```python
# 创建 Embedding 矩阵
wte = nn.Embedding(vocab_size, n_embd)   # 如 (4487, 256)

# 随机初始化：均值=0, 标准差=0.02
torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

训练前，所有向量都是随机噪声。训练后，语义相关的字自动靠近：

| 字符对 | 训练前（随机） | 训练后 | 说明 |
|--------|--------------|--------|------|
| 说 vs 言 | -0.014 | +0.368 | 同义词靠近 |
| 说 vs 道 | -0.137 | +0.240 | 同义词靠近 |
| 山 vs 水 | -0.003 | +0.394 | 场景词聚类 |
| 说 vs 山 | -0.102 | -0.113 | 不相关保持距离 |

**这个语义结构完全是从数据中"涌现"的，没有人告诉模型这些字"意思相近"。**

---

## 第三步：Transformer — 核心处理引擎

Embedding 之后的向量进入 Transformer，这是模型的核心。

<div style="max-width: 520px; margin: 1.5em auto; font-size: 0.95em;">

<div style="border: 2px solid #4CAF50; border-radius: 8px; padding: 12px 16px; text-align: center; background: rgba(76,175,80,0.05);">
<strong>Token Embedding + Position Embedding</strong><br>
<span style="font-size: 0.85em; color: #888;">每个 token 的初始向量表示</span>
</div>

<div style="text-align: center; font-size: 1.2em; color: #888; margin: 4px 0;">↓</div>

<div style="border: 2px solid #2196F3; border-radius: 8px; padding: 14px 16px; background: rgba(33,150,243,0.05);">
<strong>Transformer Block × N</strong>（N 层叠加）
<div style="margin: 10px 0 0 0; border: 1px solid rgba(33,150,243,0.3); border-radius: 6px; padding: 10px 14px; background: rgba(33,150,243,0.03);">
<div style="margin-bottom: 4px;"><strong>LayerNorm</strong></div>
→ <strong>Self-Attention</strong> — 让每个字"看到"前面的字<br>
→ 残差连接
</div>
<div style="margin: 8px 0 0 0; border: 1px solid rgba(33,150,243,0.3); border-radius: 6px; padding: 10px 14px; background: rgba(33,150,243,0.03);">
<div style="margin-bottom: 4px;"><strong>LayerNorm</strong></div>
→ <strong>MLP（前馈网络）</strong> — "思考"和"记忆知识"<br>
→ 残差连接
</div>
</div>

<div style="text-align: center; font-size: 1.2em; color: #888; margin: 4px 0;">↓</div>

<div style="border: 2px solid #9C27B0; border-radius: 8px; padding: 12px 16px; text-align: center; background: rgba(156,39,176,0.05);">
<strong>LayerNorm → Linear</strong><br>
<span style="font-size: 0.85em; color: #888;">输出概率分布</span>
</div>

</div>

### Self-Attention：让模型学会"关注"

Self-Attention 让模型在生成下一个字时，能够关注前面所有相关的字：

**Attention 矩阵：**

<div style="max-width: 400px; margin: 1em auto;">
<table style="border-collapse: collapse; width: 100%; text-align: center; font-size: 0.95em;">
<tr>
<th style="padding: 8px; border: 1px solid #ddd;"></th>
<th style="padding: 8px; border: 1px solid #ddd;">悟</th>
<th style="padding: 8px; border: 1px solid #ddd;">空</th>
<th style="padding: 8px; border: 1px solid #ddd;">道</th>
</tr>
<tr>
<td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">悟</td>
<td style="padding: 8px; border: 1px solid #ddd; background: rgba(33,150,243,0.6); color: #fff;"><strong>1.00</strong></td>
<td style="padding: 8px; border: 1px solid #ddd; color: #ccc;">·</td>
<td style="padding: 8px; border: 1px solid #ddd; color: #ccc;">·</td>
</tr>
<tr>
<td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">空</td>
<td style="padding: 8px; border: 1px solid #ddd; background: rgba(33,150,243,0.35);">0.56</td>
<td style="padding: 8px; border: 1px solid #ddd; background: rgba(33,150,243,0.25);">0.44</td>
<td style="padding: 8px; border: 1px solid #ddd; color: #ccc;">·</td>
</tr>
<tr>
<td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">道</td>
<td style="padding: 8px; border: 1px solid #ddd; background: rgba(33,150,243,0.32);">0.51</td>
<td style="padding: 8px; border: 1px solid #ddd; background: rgba(33,150,243,0.18);">0.32</td>
<td style="padding: 8px; border: 1px solid #ddd; background: rgba(33,150,243,0.10);">0.18</td>
</tr>
</table>
<div style="font-size: 0.8em; color: #888; text-align: center; margin-top: 6px;">
每行表示该字的注意力分布：「悟」只看自己；「道」最关注「悟」
</div>
</div>

**关键设计：因果掩码（Causal Masking）**—— 每个位置只能看到前面的字，不能偷看后面的。这保证了模型学会"预测下一个字"而不是"抄答案"。

---

## 第四步：从 Logits 到概率

Transformer 的输出经过线性变换，得到词表中每个字符的"原始分数"（logits）：

<div style="max-width: 520px; margin: 1em auto; font-size: 0.95em;">
<div style="border: 2px solid #2196F3; border-radius: 8px; padding: 10px 16px; text-align: center; background: rgba(33,150,243,0.05);">
<strong>隐藏状态</strong>（256 维）
</div>
<div style="text-align: center; font-size: 1.2em; color: #888; margin: 4px 0;">↓ <span style="font-size: 0.75em;">lm_head 线性变换</span></div>
<div style="border: 2px solid #9C27B0; border-radius: 8px; padding: 10px 16px; text-align: center; background: rgba(156,39,176,0.05);">
<strong>Logits</strong>（4487 维）<br>
<code>[+2.3, -1.5, +0.8, ..., +5.1, ..., +3.7]</code><br>
<span style="font-size: 0.85em; color: #888;">每个值对应一个候选字符</span>
</div>
</div>

然后通过 **Softmax** 转换为概率分布：

<div style="max-width: 520px; margin: 1em auto; font-size: 0.95em;">
<table style="border-collapse: collapse; width: 100%; text-align: center; font-size: 0.95em;">
<tr>
<th style="padding: 8px; border: 1px solid #ddd;"></th>
<th style="padding: 8px; border: 1px solid #ddd;">候选 1</th>
<th style="padding: 8px; border: 1px solid #ddd;">候选 2</th>
<th style="padding: 8px; border: 1px solid #ddd;">候选 3</th>
<th style="padding: 8px; border: 1px solid #ddd;">候选 4</th>
</tr>
<tr>
<td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Logits</td>
<td style="padding: 8px; border: 1px solid #ddd;">+5.1</td>
<td style="padding: 8px; border: 1px solid #ddd;">+3.7</td>
<td style="padding: 8px; border: 1px solid #ddd;">+2.3</td>
<td style="padding: 8px; border: 1px solid #ddd;">-1.5</td>
</tr>
<tr>
<td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Softmax</td>
<td style="padding: 8px; border: 1px solid #ddd; background: rgba(76,175,80,0.35);"><strong>76.4%</strong></td>
<td style="padding: 8px; border: 1px solid #ddd; background: rgba(76,175,80,0.18);">18.8%</td>
<td style="padding: 8px; border: 1px solid #ddd; background: rgba(76,175,80,0.06);">4.7%</td>
<td style="padding: 8px; border: 1px solid #ddd; background: rgba(76,175,80,0.02);">0.1%</td>
</tr>
</table>
</div>

### Temperature：控制创造力

Temperature 控制生成的随机性：

```
temperature = 0.5  → 几乎只选最可能的字（保守、重复）
temperature = 0.8  → 适中（推荐值）
temperature = 1.5  → 非常随机（可能出现不连贯）
```

### Top-K：过滤离谱选项

只从概率最高的前 K 个候选中采样，防止选到极不可能的字。

---

## 第五步：自回归生成

LLM 每次只生成**一个** token，然后把它拼回输入，继续生成下一个：

<div style="max-width: 520px; margin: 1em auto; font-size: 0.95em;">

<div style="border: 1px solid #ddd; border-radius: 6px; padding: 10px 16px; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center;">
<span>输入 <code>[悟, 空, 道]</code></span>
<span style="color: #4CAF50; font-weight: bold;">→ 预测 "："</span>
</div>

<div style="border: 1px solid #ddd; border-radius: 6px; padding: 10px 16px; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center;">
<span>输入 <code>[悟, 空, 道, ：]</code></span>
<span style="color: #4CAF50; font-weight: bold;">→ 预测 "师"</span>
</div>

<div style="border: 1px solid #ddd; border-radius: 6px; padding: 10px 16px; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center;">
<span>输入 <code>[悟, 空, 道, ：, 师]</code></span>
<span style="color: #4CAF50; font-weight: bold;">→ 预测 "父"</span>
</div>

<div style="text-align: center; color: #888; font-size: 0.9em;">↻ 重复 N 次…</div>

</div>

这就是为什么 LLM 的输出是逐字"蹦出来"的 —— 它真的是一个字一个字生成的。

---

## SLM vs LLM：小模型与大模型

我们训练的 nanoGPT 模型只有 **4.3M（430 万）参数**，严格来说是一个 SLM（Small Language Model）。而 GPT-4 有约 **1.8T（1.8 万亿）参数** —— 是我们的 40 万倍。

但关键点是：**架构完全相同。**

```python
# nanoGPT demo 模型 (4.3M)
GPTConfig(n_layer=4, n_head=4, n_embd=256, vocab_size=4487)

# GPT-2 Small (124M)
GPTConfig(n_layer=12, n_head=12, n_embd=768, vocab_size=50257)

# GPT-3 (175B)
GPTConfig(n_layer=96, n_head=96, n_embd=12288, vocab_size=50257)
```

**规模的量变引起能力的质变：**

| 参数量 | 能力 |
|--------|------|
| 4M | 模仿文风，经常语句不通 |
| 100M | 能写出通顺的句子 |
| 10B | 能进行简单对话和问答 |
| 100B+ | 涌现出推理、编程、数学等"智能" |

这就是 **Scaling Law（缩放定律）**—— 模型越大、数据越多，能力就越强，某些能力甚至会在特定规模突然"涌现"。

---

## 动手试一试

如果你有一台 Linux 机器，可以用 nanoGPT 亲手训练一个语言模型：

```bash
# 1. 克隆项目
git clone https://github.com/karpathy/nanoGPT.git && cd nanoGPT

# 2. 准备数据
python data/shakespeare_char/prepare.py

# 3. 训练（CPU 上几分钟）
python train.py config/train_shakespeare_char.py

# 4. 生成文本
python sample.py --out_dir=out-shakespeare-char
```

训练日志中的 `loss` 会从 ~4.17（随机猜测）逐渐下降。当 loss 降到 1.0-2.0 时，模型就能生成像模像样的 Shakespeare 风格文本了。

---

## 总结

LLM 的核心数据流可以用一句话概括：

> **文字 → 数字 → 向量 → 注意力 + 变换 → 概率 → 采样 → 新文字**

每一步都是确定性的数学运算，没有任何"魔法"。LLM 的"智能"源于从海量数据中学到的统计模式 —— Embedding 捕捉语义关系，Attention 捕捉上下文依赖，MLP 存储和变换知识。

理解了这个流程，你就掌握了理解所有现代 LLM（GPT-4、Claude、DeepSeek）的基础。因为它们的架构，和我们这个小小的 nanoGPT，本质上是一样的。
