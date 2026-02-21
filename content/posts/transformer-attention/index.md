---
title: "从加减乘除到预测下一个字：Attention 机制零基础拆解"
date: 2026-02-21
draft: false
summary: "用零基础也能懂的语言，拆解 Transformer 的核心：点积、缩放、Softmax、QKV、多头注意力、因果遮罩。既讲人类设计了什么，也讲机器自己学到了什么。"
categories: ["LLM"]
tags: ["Transformer", "Attention", "Softmax", "QKV", "教程"]
weight: 4
ShowToc: true
TocOpen: true
---

## 引言

当你问 ChatGPT "猫为什么喜欢纸箱"，它回答了一大段话。这段话是怎么"想"出来的？

核心秘密只有一个：**注意力机制（Attention）**。它是 GPT、BERT、Claude、DeepSeek——所有现代大语言模型的心脏。

今天这篇文章，我们从最基础的数学开始，一步步讲清楚 Attention 的完整运作方式。**不需要任何前置知识**，但我们也不会回避数学。

> **关于"解释"的声明：** 本文会大量使用"寻找""关注""传递"这样的拟人化表述，这是为了帮助理解。但请始终记住：模型内部没有任何"理解"或"意图"，只有矩阵乘法和数值优化。文中会明确标注哪些是人类的设计、哪些是机器涌现的行为、哪些是我们至今无法解释的现象。

---

## 本文导读

全文分十章，沿着一条线索展开——**从最简单的数学工具，一步步组装出完整的注意力机制。**

<div style="max-width: 600px; margin: 1.5em auto; font-size: 0.93em; line-height: 1.9;">

<div style="border-left: 3px solid #FF9800; padding-left: 14px; margin-bottom: 10px;">
<strong>数学基础（第一~四章）</strong><br>
<span style="color: #888;">① 向量与相似度 → ② 点积 → ③ 缩放 ÷√d → ④ Softmax 归一化</span><br>
先把零件认全：什么是向量？怎么衡量相似？为什么要缩放？如何把分数变成权重？
</div>

<div style="border-left: 3px solid #2196F3; padding-left: 14px; margin-bottom: 10px;">
<strong>核心机制（第五~六章）</strong><br>
<span style="color: #888;">⑤ Q、K、V 三路变换 → ⑥ 完整计算示例</span><br>
用零件组装引擎：为什么需要三个矩阵？一个 token 走完 Attention 全流程是什么样的？
</div>

<div style="border-left: 3px solid #9C27B0; padding-left: 14px; margin-bottom: 10px;">
<strong>工程设计（第七~八章）</strong><br>
<span style="color: #888;">⑦ 因果遮罩与 KV Cache → ⑧ 多头注意力</span><br>
让引擎跑起来：为什么 GPT 只能往前看？为什么要切成 12 份？训练和推理有什么区别？
</div>

<div style="border-left: 3px solid #4CAF50; padding-left: 14px;">
<strong>全景与反思（第九~十章）</strong><br>
<span style="color: #888;">⑨ 从 Attention 到预测下一个词 → ⑩ 人类设计了什么，机器涌现了什么</span><br>
退一步看全貌：Attention 在整个模型中处于什么位置？哪些是我们能解释的，哪些至今是黑箱？
</div>

</div>

本文所有例子都使用我们在[前两篇文章](/ai-blog/posts/llm-pipeline-visual/)中训练的**西游记 nanoGPT 模型**——你已经见过"悟空道"如何变成向量，今天我们看它如何在 Attention 中流动。

---

## 第一章：什么是"相似"？

### 从生活说起

你去图书馆找书。心里想着"机器学习入门"，书架上每本书都有标签。你的大脑在做的事：**把你的需求和每本书的标签做比较，挑最匹配的。**

Transformer 做的事情完全一样。只不过"需求"和"标签"都变成了**一串数字**——也就是向量。

### 向量是什么？

LLM 把一个词变成一串数字（向量），你可以想象成一个雷达图：

```
          皇室感
            ↑
      0.9 ──┼── "国王"
      0.8 ──┼── "王后"
      0.1 ──┼── "汽车"
            │
  ──────────┼──────────→ 性别感
       女← 0 →男

"国王" ≈ (皇室感=0.9, 男性=0.8)
"王后" ≈ (皇室感=0.8, 女性=0.7)
"汽车" ≈ (皇室感=0.1, 中性=0.0)
```

真实的 LLM 里不是 2 维，是 768 维甚至更多。但原理一样——**方向接近的向量，语义接近。**

---

## 第二章：点积——最朴素的相似度计算

### 公式

两个向量 A 和 B，把对应位置的数字**相乘再相加**：

```
A = [a₁, a₂, a₃]
B = [b₁, b₂, b₃]

点积 = a₁×b₁ + a₂×b₂ + a₃×b₃
```

### 为什么这能衡量相似度？

核心直觉：**同号相乘得正，异号相乘得负。**

```
A = [ 0.9,  0.8]   ← "国王"
B = [ 0.8,  0.7]   ← "王后"

点积 = 0.9×0.8 + 0.8×0.7 = 0.72 + 0.56 = 1.28   ← 大正数，相似！

A = [ 0.9,  0.8]   ← "国王"
C = [-0.7,  0.0]   ← "汽车"

点积 = 0.9×(-0.7) + 0.8×0.0 = -0.63 + 0 = -0.63  ← 负数，不相似
```

**规律：两个向量方向越一致，点积越大。** 这就是余弦相似度的核心思想——不关心向量有多长，只关心它们是否指向同一个方向。

---

## 第三章：缩放——为什么要除以 √d

### 问题：维度越高，点积数值越大

假设向量是 d 维的，每个分量大约在 -1 到 1 之间。点积是 d 个乘积的和：

```
维度 d = 4:    点积 ≈ 几个单位      比如 3.2
维度 d = 768:  点积 ≈ 几十甚至上百   比如 42.7
```

加了更多项，总和自然更大。

### 大数值会怎样？

点积算完后，下一步要做 Softmax（下一章讲）。Softmax 里有指数运算 eˣ：

```
如果分数是 [1.0, 2.0, 3.0]:
  e¹ = 2.7,  e² = 7.4,  e³ = 20.1    ← 差距温和，每个词都有权重

如果分数是 [10, 20, 30]:
  e¹⁰ = 22026,  e²⁰ = 4.9亿,  e³⁰ = 10万亿  ← 最大值碾压一切！
```

分数太大 → Softmax 输出接近 one-hot（只有最大的那个接近 1，其余全趋近 0）→ 模型几乎只看一个词，丧失了综合多个词信息的能力 → 梯度趋近零，训练卡死。

### 解决方案：除以 √d

```
scaled_score = 点积 / √d
```

因为点积的标准差大约是 √d，除以它刚好把分数拉回到方差≈1 的舒适区间：

```
维度 d = 768,  √768 ≈ 27.7

原始点积 ≈ 42.7
缩放后   ≈ 42.7 / 27.7 ≈ 1.54   ← 回到合理范围
```

**这就是 "Scaled" 的含义——一个数学上的温度调节器。**

> **这是人类的设计。** 除以 √d 不是模型自己发现的，是 2017 年 Google 论文 *"Attention Is All You Need"* 的作者凭数学直觉写进去的。模型不知道为什么要除，它只是按公式执行。

---

## 第四章：Softmax——把分数变成权重

### 问题：原始分数没有统一尺度

经过缩放后，我们得到了一组相关性分数：

```
分数: [0.2, 3.8, 1.5, 0.9]
```

这些数字的总和不是 1，可正可负，没法直接当"百分比"用。

### Softmax 三步走

```
第1步：对每个分数取指数 eˣ（把所有值变为正数）
第2步：求和
第3步：每个值除以总和（归一化，使总和 = 1）
```

算一下：

```
分数:  [1.0,  2.0,  3.0]

eˣ:    [2.72, 7.39, 20.09]

总和 = 2.72 + 7.39 + 20.09 = 30.20

softmax = [2.72/30.20, 7.39/30.20, 20.09/30.20]
        = [0.09, 0.24, 0.67]

总和 = 1.0 ✓
```

### 为什么用 eˣ 而不是直接除以总和？

```
直接归一化: [1, 2, 3] → [1/6, 2/6, 3/6] = [0.17, 0.33, 0.50]  差距不大
Softmax:    [1, 2, 3] → [0.09, 0.24, 0.67]                       差距更明显
```

两个原因：

1. **负数问题：** 直接除以总和，如果有负分数，结果没有意义。eˣ 保证输出永远为正。
2. **放大区分度：** 指数运算天然放大差异，让高分更高、低分更低。模型可以更果断地"决定"该关注谁。

> **这也是人类的选择。** Softmax 是概率论和信息论中常用的函数，研究者选择它是因为它有优良的数学性质（可微分、输出为概率分布）。不是唯一选择——还有 sparsemax、硬注意力等替代方案——但 Softmax 是目前最常用的。

---

## 第五章：Q、K、V——注意力的三个角色

到这里，我们已经有了衡量相似度和归一化的全部数学工具。现在的问题是：**拿什么和什么比较？**

### 最朴素的想法

直接拿每个词的向量互相做点积？可以。但有一个根本问题——

一个向量要同时做三件事：
- 表达"我需要什么信息"
- 表达"我能被什么搜索到"
- 表达"命中后，我要传递什么内容"

这三个目标往往矛盾。一个向量不可能同时朝三个方向。

### 人类的解决方案：三个矩阵

这是 Transformer 设计者**有意识的工程选择**。他们在模型里放了三个权重矩阵 Wq、Wk、Wv，把同一个输入变换成三个不同的向量：

```
一个 token 的 embedding（比如 256 维的 x）

Q = x × Wq    ← 一次矩阵乘法，得到一串新的数字
K = x × Wk    ← 同样大小的矩阵，不同的参数，得到另一串数字
V = x × Wv    ← 再一次，又得到一串不同的数字
```

### 矩阵乘法在做什么？

不需要知道矩阵乘法的细节。你只需要理解一件事：

**矩阵乘法 = 信息的重新组合。**

原始的 x 里混着这个词的所有信息。乘以不同的矩阵，就像戴上不同的滤镜，从同一张照片里提取不同的侧面：

<div style="max-width: 480px; margin: 1em auto; font-size: 0.93em;">
<div style="text-align: center; margin-bottom: 6px; font-weight: bold;">同一个 x（原材料）</div>
<div style="display: flex; flex-direction: column; gap: 6px;">
<div style="border: 2px solid #FF9800; border-radius: 6px; padding: 8px 14px; background: rgba(255,152,0,0.05);">× Wq → <strong>Q</strong>：从 x 中提取"发出信号"的侧面</div>
<div style="border: 2px solid #2196F3; border-radius: 6px; padding: 8px 14px; background: rgba(33,150,243,0.05);">× Wk → <strong>K</strong>：从 x 中提取"接收匹配"的侧面</div>
<div style="border: 2px solid #9C27B0; border-radius: 6px; padding: 8px 14px; background: rgba(156,39,176,0.05);">× Wv → <strong>V</strong>：从 x 中提取"内容传递"的侧面</div>
</div>
</div>

**关键：三个矩阵的具体数值不是人类写的，是训练出来的。** 人类只决定了"用三个矩阵做变换"这个结构，至于每个矩阵具体做什么样的变换——那是模型在几十亿次训练中自己摸索出来的。

### 从模型的视角看：没有"查询""键""值"

这里必须澄清一件事。

Q、K、V 这些名字——Query（查询）、Key（键）、Value（值）——是**人类取的**，为了帮助人类自己理解。模型内部没有这些概念。

模型看到的全部真相：

```
输入: x    （一串数字）

q = x × Wq    （一次矩阵乘法，得到另一串数字）
k = x × Wk    （一次矩阵乘法，得到另一串数字）
v = x × Wv    （一次矩阵乘法，得到另一串数字）

score = q × kᵀ / √d    （数字乘数字，除以一个常数）
weight = softmax(score) （指数运算 + 归一化）
output = weight × v     （数字乘数字，加起来）
```

训练过程中发生了什么？

```
初始: Wq Wk Wv 全是随机数字，模型的预测和随机猜没区别

训练 100 万步后:
  某些参数组合碰巧让预测更准 → 梯度下降强化这个方向
  模型开始发现: "悟" 后面经常跟 "空"，"行者" 后面经常跟 "道"
  （但模型不知道"悟空"是名字，它只是在拟合统计规律。）

训练 10 亿步后:
  涌现出人类都没预料到的模式
  某些注意力头专门让相邻词互相关注
  某些头专门让远距离的词互相关联
  某些头做的事情人类至今无法解释
```

这就像看蚂蚁搬食物——人类说"蚂蚁知道路线"，但蚂蚁只是在跟随化学梯度。结果相似，内在机制完全不同。

### 每个 token 共用同一套矩阵

一个重要事实：**同一层里，所有 token 用的是同一个 Wq、同一个 Wk、同一个 Wv。**

```
句子: "悟空道"

Q_悟 = x_悟 × Wq
Q_空 = x_空 × Wq     ← 同一个 Wq
Q_道 = x_道 × Wq

K_悟 = x_悟 × Wk
K_空 = x_空 × Wk     ← 同一个 Wk
K_道 = x_道 × Wk
```

Q、K、V 之间的差异完全来自输入 x 的不同，不是来自矩阵的不同。

为什么共享？三个原因：

**参数效率：** 如果每个位置都有自己的矩阵，参数量会膨胀上千倍。共享一个 Wq，所有位置都能用。

**泛化能力：** 一个 Wq 从所有位置的数据中学习，训练更充分。如果每个位置单独学，数据太稀疏。

**长度无关：** 推理时输入多长都行，同一个矩阵照用不误。

---

## 第六章：Attention 的完整计算——一个例子

现在把所有组件组合起来。以处理 `"悟空道"` 中的 `"道"` 为例：

### 第 1 步：生成 Q、K、V

```
x_悟, x_空, x_道    ← 三个 token 的 embedding

每个 x 分别乘以 Wq, Wk, Wv:
Q_道 = x_道 × Wq
K_悟 = x_悟 × Wk,  K_空 = x_空 × Wk,  K_道 = x_道 × Wk
V_悟 = x_悟 × Wv,  V_空 = x_空 × Wv,  V_道 = x_道 × Wv
```

### 第 2 步：Q 和所有 K 做点积

```
score_1 = Q_道 · K_悟 = 0.3    （小数字）
score_2 = Q_道 · K_空 = 2.1    （大数字）
score_3 = Q_道 · K_道 = 0.8    （中等）
```

### 第 3 步：缩放

```
假设维度 d = 64, √64 = 8

scaled = [0.3/8, 2.1/8, 0.8/8] = [0.04, 0.26, 0.10]
```

### 第 4 步：Softmax 归一化

```
e^0.04 = 1.04,  e^0.26 = 1.30,  e^0.10 = 1.11

总和 = 3.45

weights = [1.04/3.45, 1.30/3.45, 1.11/3.45]
        = [0.30, 0.38, 0.32]
```

### 第 5 步：加权求和 V（Aggregate）

```
output = 0.30 × V_悟 + 0.38 × V_空 + 0.32 × V_道
```

这个 output 是一个新的向量——它**不再是某一个字，而是融合了上下文中相关字的信息**。

对 "道" 来说，它现在"吸收"了 "空" 的信息（权重最大）。人类会解读为"道 知道了前面说的是悟空"——但更准确地说，只是某些数字被加权混合了。

### 完整公式

```
                    Q × Kᵀ
Attention(Q,K,V) = softmax( ─────── ) × V
                      √d
```

<div style="max-width: 620px; margin: 1em auto; font-size: 0.93em;">
<div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: center; gap: 6px 4px; line-height: 1.8;">
<span style="border: 2px solid #FF9800; border-radius: 6px; padding: 6px 12px; background: rgba(255,152,0,0.05); text-align: center;"><strong>Q 和 K 做点积</strong><br><span style="font-size:0.8em; color:#888;">算相似度</span></span>
<span style="color:#aaa; font-size:1.2em;">→</span>
<span style="border: 2px solid #FF9800; border-radius: 6px; padding: 6px 12px; background: rgba(255,152,0,0.05); text-align: center;"><strong>除以 √d 缩放</strong><br><span style="font-size:0.8em; color:#888;">防止爆炸</span></span>
<span style="color:#aaa; font-size:1.2em;">→</span>
<span style="border: 2px solid #2196F3; border-radius: 6px; padding: 6px 12px; background: rgba(33,150,243,0.05); text-align: center;"><strong>Softmax 归一化</strong><br><span style="font-size:0.8em; color:#888;">变成权重</span></span>
<span style="color:#aaa; font-size:1.2em;">→</span>
<span style="border: 2px solid #9C27B0; border-radius: 6px; padding: 6px 12px; background: rgba(156,39,176,0.05); text-align: center;"><strong>加权求和 V</strong><br><span style="font-size:0.8em; color:#888;">得到输出</span></span>
</div>
</div>

> **设计 vs 涌现的分界线：** 这四步的结构（点积→缩放→softmax→加权求和）是人类设计的。但每一步里的具体数值（Wq、Wk、Wv 的参数）是训练出来的。人类搭了舞台，机器自己编了剧本。

---

## 第七章：因果遮罩——GPT 只能往前看

前面的例子为了简单，省略了一个关键细节：**GPT 是自回归模型，每个 token 只能看到它自己和它前面的 token。**

### 遮罩的实现

在 Q·K 算完分数之后、Softmax 之前，把"未来位置"的分数设成负无穷：

```
句子: "八戒道大师兄"

处理 "道" 时:

原始分数:  [1.2,  3.8,  2.1,  0.9,  0.3,  2.5]
             八    戒    道    大    师    兄
                              ↑ 后面三个是未来字

加遮罩后:  [1.2,  3.8,  2.1,  -∞,   -∞,   -∞ ]

Softmax 后: [0.15, 0.55, 0.30,  0,    0,    0  ]
                                 ↑ e^(-∞)=0，未来字权重自动归零
```

**"道" 永远看不到后面的 "大师兄"。** 无论训练还是推理，这个遮罩都在。这是 GPT 架构的硬规则。

### 训练时：完整句子 + 遮罩 = 高效学习

你可能会问：训练时已经有了完整答案，预测还有什么意义？

关键在于**遮罩保证了每个位置看不到答案**：

```
输入:     八    戒    道    大    师    兄
          ↓     ↓     ↓     ↓     ↓     ↓
模型预测:  戒?   道?   大?   师?   兄?   ，?
正确答案:  戒    道    大    师    兄    ，

位置 3（预测 "大"）: 只能看到 [八, 戒, 道]，看不到 大
位置 5（预测 "兄"）: 只能看到 [八, 戒, 道, 大, 师]，看不到 兄
```

每个位置的处境和推理时一模一样——都看不到答案。但因为所有位置是**并行计算**的，效率比逐个处理高得多：

```
逐个处理: 6 次前向传播才处理完一句话
带遮罩并行: 1 次前向传播，同时得到 6 个预测
```

效率提升了 6 倍，而效果完全等价。

> **遮罩也是人类的设计。** 它直接体现了"从左到右生成"这个设计目标。BERT 等双向模型就没有这个遮罩——它们可以前后都看。不同的模型架构，对应不同的设计选择。

### 推理时：Prefill + KV Cache

当你给 ChatGPT 输入一长段 prompt 时，模型也是带着遮罩一次性处理的。但**只有最后一个位置的输出有用**——因为模型要预测的是最后一个字后面的字。

前面位置的 K 和 V 不会浪费——它们被缓存下来（KV Cache），后续生成新 token 时直接复用：

```
阶段一（Prefill）:
  整个 prompt 并行处理 → 缓存所有位置的 K、V → 预测第一个新字

阶段二（Decode）:
  每次只处理 1 个新 token
  新 token 的 Q × 缓存的所有 K → 注意力权重
  权重 × 缓存的所有 V → 输出
  → 预测下一个字... 重复直到结束
```

这就是为什么你用 ChatGPT 时，**提交后等一下，然后文字就快速蹦出来**——Prefill（处理 prompt）慢，Decode（逐个生成）快。

---

## 第八章：多头注意力——为什么要切成 12 份

### 单头的局限

假设只有一个 attention head，处理这句话：

```
"悟空拿起金箍棒向那妖怪打去"

处理 "打" 时，模型可能需要同时关注:
  - "悟空"（谁在打？主语，距离远）
  - "金箍棒"（用什么打？工具）
  - "妖怪"（打谁？宾语，距离近）
```

但只有一组权重。**必须妥协——什么都关注一点，什么都不够专注。**

### 解决方案：切分 + 并行

把 768 维的空间切成 12 份，每份 64 维，各自独立做一遍完整的 Attention：

```
x（768维）切成 12 份:

  Head 1  (维度 1~64):    有自己的 Wq₁ Wk₁ Wv₁
  Head 2  (维度 65~128):  有自己的 Wq₂ Wk₂ Wv₂
  Head 3  (维度 129~192): 有自己的 Wq₃ Wk₃ Wv₃
  ...
  Head 12 (维度 705~768): 有自己的 Wq₁₂ Wk₁₂ Wv₁₂
```

每个 head 独立计算自己的注意力权重，然后把 12 个输出拼接回 768 维：

<div style="max-width: 500px; margin: 1em auto; font-size: 0.93em;">
<div style="display: flex; align-items: center; gap: 8px;">
<div style="flex: 1; display: flex; flex-direction: column; gap: 4px;">
<div style="border: 1px solid var(--border); border-radius: 6px; padding: 6px 12px; background: rgba(33,150,243,0.05); font-size: 0.9em;">Head 1 输出 (64维)</div>
<div style="border: 1px solid var(--border); border-radius: 6px; padding: 6px 12px; background: rgba(33,150,243,0.05); font-size: 0.9em;">Head 2 输出 (64维)</div>
<div style="border: 1px solid var(--border); border-radius: 6px; padding: 6px 12px; background: rgba(33,150,243,0.05); font-size: 0.9em;">Head 3 输出 (64维)</div>
<div style="text-align: center; color: #888;">...</div>
<div style="border: 1px solid var(--border); border-radius: 6px; padding: 6px 12px; background: rgba(33,150,243,0.05); font-size: 0.9em;">Head 12 输出 (64维)</div>
</div>
<div style="font-size: 1.2em; color: #888;">→</div>
<div style="border: 2px solid #9C27B0; border-radius: 8px; padding: 10px 14px; background: rgba(156,39,176,0.05); text-align: center;">
<strong>拼接</strong><br><span style="font-size: 0.85em;">768维</span><br><span style="font-size: 0.85em; color: #888;">× Wo</span>
</div>
<div style="font-size: 1.2em; color: #888;">→</div>
<div style="border: 2px solid #4CAF50; border-radius: 8px; padding: 10px 14px; background: rgba(76,175,80,0.05); text-align: center;">
<strong>最终输出</strong><br><span style="font-size: 0.85em;">768维</span>
</div>
</div>
</div>

最后还有一个 Wo 矩阵（768×768），用来让不同 head 的结果互相混合。

### 参数量没有增加

```
单头:     Wq 大小 768×768 = 589,824 个参数
12 头:    12 × (768×64)   = 589,824 个参数   ← 总量一样！
```

参数量相同，但效果好得多——因为 12 个独立的小空间比 1 个大空间更灵活。每个 head 可以毫无顾忌地给某个位置高权重，因为其他 head 会覆盖其他关系。

### 涌现出的分工

研究者事后观察 GPT-2 的 12 个 head，发现它们自发地分化出了不同的模式：

```
某个 head: 总是让相邻的字互相关注
某个 head: 总是让远距离的词互相关联
某个 head: 均匀关注所有前面的字
某个 head: 做的事情人类解释不了——但去掉它模型就变差
```

**这些"分工"不是人类编程进去的。** 人类只提供了"12 个独立子空间"这个结构。至于每个子空间最终学会关注什么——那完全是训练过程中，为了让预测更准而自发涌现的。

就像一个公司招了 12 个员工，没有给他们分配具体岗位，只是说"你们合作把预测做准"。结果他们自发地有人看语法、有人看指代、有人看搭配——因为这种分工让团队整体表现最好。

> **人类设计 vs 机器涌现：** "切成 12 份并行计算"是人类的架构设计。"每个 head 负责什么"是训练涌现的结果。人类甚至不能完全解释每个 head 在做什么——有些 head 的行为至今是黑箱。

---

## 第九章：从 Attention 到预测下一个字

Attention 只是 Transformer 的一个组件。从 Attention 输出到最终预测，还有一段路。

### Transformer 的一层

<div style="max-width: 520px; margin: 1em auto; font-size: 0.93em;">

<div style="text-align: center; margin-bottom: 4px; font-weight: bold;">输入 x</div>
<div style="text-align: center; font-size: 1.2em; color: #888; margin: 2px 0;">↓</div>

<div style="border: 2px solid #2196F3; border-radius: 8px; padding: 10px 16px; background: rgba(33,150,243,0.05); margin-bottom: 2px;">
<strong>Multi-Head Attention</strong> ← 上下文信息融合（本文讲的全部内容）<br>
<span style="font-size: 0.85em; color: #888;">+ 残差连接 + 归一化</span>
</div>

<div style="text-align: center; font-size: 1.2em; color: #888; margin: 2px 0;">↓</div>

<div style="border: 2px solid #FF9800; border-radius: 8px; padding: 10px 16px; background: rgba(255,152,0,0.05); margin-bottom: 2px;">
<strong>Feed-Forward Network (FFN, 两层全连接)</strong> ← 对每个位置独立做非线性变换<br>
<span style="font-size: 0.85em; color: #888;">+ 残差连接 + 归一化</span>
</div>

<div style="text-align: center; font-size: 1.2em; color: #888; margin: 2px 0;">↓</div>
<div style="text-align: center; font-weight: bold;">输出 x'（和输入同样大小的向量）</div>

</div>

一个 Transformer 由多层这样的结构堆叠：GPT-2 有 12 层，GPT-3 有 96 层，我们的西游记模型有 4 层。

研究者观察到（又是事后解释）：

```
低层:   学到的模式比较简单——相邻字关联、基本搭配
中层:   开始出现语义级别的模式——短语结构、常见组合
高层:   出现更抽象的模式——长距离关联、上下文推断

但这不是绝对的分界。每一层的行为高度复杂，
很多模式跨层分布，人类只能解释其中一小部分。
```

### 最后一步：预测头

经过 N 层 Transformer 后，最后一个 token 位置的输出向量要映射到词汇表上。以我们的西游记模型为例（词汇表 4487 个字），也可以对照 GPT-2（词汇表 50257 个 token）：

```
最后一个 token "道" 的隐藏状态 h（256维）
    ↓
线性层: h × W_output (256 → 4487)
    ↓
得到 4487 个原始分数（logits）:
  "："  → 8.7     ← 最高
  "大"  → 6.2
  "了"  → 5.1
  "的"  → 3.4
  ...其余 4483 个字 → 都很低
    ↓
Softmax（又见面了！）
    ↓
概率分布:
  "："  → 0.42
  "大"  → 0.18
  "了"  → 0.11
  ...
    ↓
采样或取最大值 → 输出: "："
```

在[前一篇文章](/ai-blog/posts/llm-pipeline-visual/)中，我们的模型输入"悟空道"确实预测出了"："——因为在《西游记》原文中，"悟空道"后面大概率就是冒号加引号，然后开始说话。模型从数据中学到了这个模式。

### 两次 Softmax 的区别

全流程中出现了两次 Softmax，它们的作用完全不同：

| | Attention 内部的 Softmax | 预测层的 Softmax |
|---|---|---|
| 输入 | Q·K 的相似度分数 | 4487 个 logits |
| 输出 | 注意力权重（关注哪些字） | 词汇表概率分布（选哪个字） |
| 问题 | "该关注句子里的哪些位置？" | "下一个字是词汇表里的哪个？" |

数学形式完全相同，但在流程中的角色不同。Softmax 是一个通用的数学工具，在不同地方解决不同的"把分数变概率"问题。

---

## 第十章：总结——人类设计了什么，机器涌现了什么

### 全景回顾

<div style="max-width: 640px; margin: 1.5em auto; font-size: 0.95em;">

<div style="border: 2px solid #4CAF50; border-radius: 8px; padding: 12px 16px; text-align: center; background: rgba(76,175,80,0.05);">
<strong>输入 token</strong><br>x（embedding 向量）
</div>

<div style="text-align: center; font-size: 1.2em; color: #888; margin: 4px 0;">↓</div>

<div style="border: 2px solid #FF9800; border-radius: 8px; padding: 10px 16px; background: rgba(255,152,0,0.05);">
<strong>① 三路变换</strong><br>Q = x×Wq, K = x×Wk, V = x×Wv<br>
<span style="font-size: 0.85em; color: #888;">同一个 x 通过三个不同矩阵，提取三个不同侧面</span>
</div>

<div style="text-align: center; font-size: 1.2em; color: #888; margin: 4px 0;">↓</div>

<div style="border: 2px solid #FF9800; border-radius: 8px; padding: 10px 16px; background: rgba(255,152,0,0.05);">
<strong>② 点积 + 缩放</strong><br>score = Q·Kᵀ / √d<br>
<span style="font-size: 0.85em; color: #888;">衡量每对 token 之间的相关性，除以 √d 防止数值爆炸</span>
</div>

<div style="text-align: center; font-size: 1.2em; color: #888; margin: 4px 0;">↓</div>

<div style="border: 2px solid #2196F3; border-radius: 8px; padding: 10px 16px; background: rgba(33,150,243,0.05);">
<strong>③ 因果遮罩</strong><br>未来位置的分数 → -∞<br>
<span style="font-size: 0.85em; color: #888;">GPT 只能往前看，不能偷看未来的字</span>
</div>

<div style="text-align: center; font-size: 1.2em; color: #888; margin: 4px 0;">↓</div>

<div style="border: 2px solid #2196F3; border-radius: 8px; padding: 10px 16px; background: rgba(33,150,243,0.05);">
<strong>④ Softmax 归一化</strong><br>分数 → 权重（总和=1）<br>
<span style="font-size: 0.85em; color: #888;">eˣ 保证正数 + 放大差异，归一化变成概率</span>
</div>

<div style="text-align: center; font-size: 1.2em; color: #888; margin: 4px 0;">↓</div>

<div style="border: 2px solid #9C27B0; border-radius: 8px; padding: 10px 16px; background: rgba(156,39,176,0.05);">
<strong>⑤ 加权求和 V</strong><br>output = weights × V<br>
<span style="font-size: 0.85em; color: #888;">从被关注的字中提取信息，融合成上下文感知的新表示</span>
</div>

<div style="text-align: center; font-size: 1.2em; color: #888; margin: 4px 0;">↓</div>

<div style="border: 2px solid #9C27B0; border-radius: 8px; padding: 10px 16px; background: rgba(156,39,176,0.05);">
<strong>⑥ 多头拼接</strong><br>12 个 head 的输出拼接 → ×Wo<br>
<span style="font-size: 0.85em; color: #888;">不同子空间的分析结果融合在一起</span>
</div>

<div style="text-align: center; font-size: 1.2em; color: #888; margin: 4px 0;">↓ × N 层</div>

<div style="border: 2px solid #4CAF50; border-radius: 8px; padding: 10px 16px; text-align: center; background: rgba(76,175,80,0.05);">
<strong>⑦ 线性层 + Softmax → 预测下一个 token</strong>
</div>

</div>

### 三条线索

读完全文，希望你带走三个层次的理解：

**人类设计的（架构）：**
- 把输入分成 Q、K、V 三路
- 用点积衡量相似度
- 除以 √d 缩放
- 用 Softmax 归一化
- 多头并行 + 拼接
- 因果遮罩，只看前文

**机器学到的（参数）：**
- Wq、Wk、Wv 的具体数值
- 每个 head 关注什么模式
- 不同层学到什么深度的特征
- 所有这些，都是为了一个目标：**让预测下一个 token 的概率最大化**

**人类无法完全解释的（涌现）：**
- 为什么某些 head 会自发地"分工"
- 某些 head 的行为模式至今是黑箱
- 模型在大规模训练后表现出的"推理""类比""创造"能力——到底是真正的理解，还是极其精妙的统计模式匹配？这仍然是一个开放问题

---

## 后记

2017 年，Google 的八位研究员发表了一篇论文，标题只有五个词：*"Attention Is All You Need"*——注意力就是你需要的全部。

那时候没有人预料到后来发生的事情。

这篇论文描述的机制——点积、缩放、Softmax、多头——数学上并不复杂，任何学过线性代数的本科生都能看懂。但就是这些简单的运算，堆叠到足够的规模后，涌现出了让所有人震惊的能力：它能写诗、能编程、能推理、能翻译，甚至能做一些看起来像"创造"的事情。

这件事本身就是一个深刻的哲学谜题。

我们在文中反复标注了一条分界线：人类设计了什么，机器学到了什么。现在让我们退到更远的距离来看这条线——

**人类提供的其实很少。** 点积不是为语言发明的，它是 19 世纪的数学工具。Softmax 来自统计力学。矩阵乘法有几百年的历史。Transformer 的创新不在于发明了新数学，而在于**把几个旧零件用一种特定的方式组装在了一起**。

**机器发现的却出奇地多。** 没有人教模型什么是主语、什么是隐喻、什么是因果推理。这些模式全部是从"预测下一个 token"这一个目标中自发涌现的。一个只被要求做"完形填空"的系统，学会了远超完形填空的能力。

这就引出了一个根本性的问题：**"预测下一个词"和"理解语言"之间，到底差了什么？**

如果一个系统能完美预测任何上下文中的下一个词，它是否必然已经理解了语言？或者说，是否存在一种不需要"理解"就能完美预测的方式？这个问题目前没有人能回答。

也许最诚实的立场是这样的：Transformer 是一面镜子。我们在里面看到了"智能"的影子，但我们甚至不确定自己完全理解什么是"智能"。我们造出了一台能力远超预期的机器，但我们对它为什么如此有效的理解，远远落后于它的表现。

这不是终点。这是一个起点——理解智能本质的起点。

不过至少有一件事是确定的：**读到这里的你，已经知道这台机器的每一个齿轮是怎么转的了。** 这比世界上绝大多数人都走得更远。当你下一次和 ChatGPT 对话时，你知道它不是在"思考"——它在做矩阵乘法、算点积、过 Softmax、加权求和。但你也知道，就是这些简单的运算，叠加到一起之后，产生了某种我们还无法完全命名的东西。

那是什么？也许下一个时代的人会告诉我们。
