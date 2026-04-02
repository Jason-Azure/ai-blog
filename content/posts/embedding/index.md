---
title: "当数字学会了远近亲疏——从查表到 Embedding 的一步跨越"
date: 2026-04-01
draft: false
summary: "A=65, B=66——这些数字是死的。「大漠孤烟直，长河落日圆」——这些文字是活的。从莫尔斯电码到 GPT 的编码史中，有一步跨越改变了一切：数字不再是编号，而是坐标。它们学会了远近、方向和意思。这一步叫 Embedding。"
categories: ["AI 基础"]
tags: ["Embedding", "Word2Vec", "向量", "语义空间", "多模态", "CLIP", "MLP", "认知", "分布式表示"]
weight: 49
ShowToc: true
TocOpen: true
---

## 从上一篇的结尾说起

上一篇 [《计算机只懂 0 和 1》](/ai-blog/posts/ascii-to-token/) 里，我们走完了一条 200 年的编码之路：莫尔斯电码 → ASCII → Unicode → BPE。一路走下来，结论很清晰：

**计算机一直在做同一件事——把符号变成数字。**

但文章快结尾时，我留了一个尾巴：

> Embedding 用 768 个数字代表 1 个子词。

那篇文章没有展开。因为展开之后，你会发现——这一步和前面所有的步骤，有一个**本质的不同**。

---

## 一、编号的困境

### 65 号和 66 号，是近还是远？

ASCII 说 A=65, B=66, C=67。Unicode 说"你"=20320，"好"=22909。BPE 说 "the"=1820，"cat"=9246。

这些数字有一个共同的特点：**它们只是编号，不是意思。**

65 和 66 挨着，但 A 和 B 的"意思"并不比 A 和 Z 更接近。"你"是 20320，"我"是 25105——差了近 5000，但"你"和"我"的语义距离，远比"你"和"镉"（38221）近得多。

这就是编号的根本局限：**数字之间的大小关系，和符号之间的意义关系，毫无关联。**

用一个类比来说：

> 你的身份证号是 11010219900307，你邻居的是 11010219900308。号码只差 1，但你们可能一个是诗人，一个是程序员，思想隔了十万八千里。
>
> 而你远在千里之外的一个陌生人，身份证号和你差了几亿，却恰好和你读同样的书、想同样的问题。
>
> **编号不反映关系。**

对人来说这不是问题——你看到"A"就知道它是"A"，不需要通过 65 这个数字去"理解"它。但对计算机来说，数字是它唯一能操作的东西。如果数字不携带意义，计算机就永远在做**盲目的搬运**，而不是**有意义的计算**。

![编号 vs 坐标：身份证号不反映两个人的关系，但地图上的坐标可以](01_id_vs_coordinate.png)

<div style="text-align: center; font-size: 0.85em; color: #888; margin-top: -10px; margin-bottom: 20px;">▲ 左：编号——数值大小和意义无关。右：坐标——距离就是关系</div>

---

### 最粗暴的方案：One-Hot

计算机科学家们最早想到的办法，简单粗暴：给每个词一个**独立的维度**。

如果词表有 50000 个词，就造一个 50000 维的空间。每个词是一个向量，只有自己对应的那一维是 1，其余全是 0。

```text
"猫" = [0, 0, 0, ..., 1, ..., 0, 0, 0]  ← 第 3721 维是 1
"狗" = [0, 0, 0, ..., 0, ..., 1, ..., 0]  ← 第 8456 维是 1
"桌" = [0, 0, 0, ..., 0, ..., 0, ..., 1]  ← 第 22017 维是 1
```

这叫 **One-Hot 编码**（独热编码）。

有三个致命问题：

**1. 维度爆炸。** 词表有 5 万词，每个词就是一个 5 万维的向量。存储和计算代价巨大。

**2. 没有"近远"。** "猫"和"狗"的距离 = "猫"和"桌"的距离 = "猫"和"量子力学"的距离。所有词之间的距离**完全相同**（都是 √2）。

**3. 无法泛化。** 模型在训练中见过"猫坐在沙发上"，但遇到"狗坐在沙发上"时完全没有借鉴——因为"猫"和"狗"在 One-Hot 空间里毫无关联。

> 这就是 Bengio 在 2003 年那篇开创性论文里指出的**维度灾难（curse of dimensionality）**：语言中的词汇量太大，如果每个词都是独立的维度，需要指数级的训练数据才能覆盖所有可能的词组合。
>
> Bengio, Y. et al. (2003). *A Neural Probabilistic Language Model*. JMLR.

我们需要一种根本不同的思路。

<div style="background: rgba(76,175,80,0.08); border-left: 4px solid #4CAF50; padding: 12px 16px; margin: 20px 0; border-radius: 0 6px 6px 0;">

**一句话记住：** 编号是死的——65 和 66 挨着，不代表 A 和 B 有关系。One-Hot 更糟——所有词两两等距。我们需要让数字**自己学会远近**。

</div>

---

## 二、从编号到坐标：Embedding 的诞生

### 一句改变历史的话

1957 年，英国语言学家 John Rupert Firth 写下了一句看似平常的话：

> **"You shall know a word by the company it keeps."**
>
> （要了解一个词的意思，看看它和谁在一起就行了。）
>
> — Firth, J.R. (1957). *A Synopsis of Linguistic Theory*.

这句话后来被称为**分布假说（distributional hypothesis）**，成为了整个 Embedding 思想的哲学基石。

想想看，这和我们学语言的方式一模一样。你第一次遇到"獬豸"这个词，不认识。但如果你反复在这些上下文中看到它：

```text
"獬豸能辨曲直"
"古代法官头戴獬豸冠"
"獬豸是传说中的神兽"
```

你不需要任何人给你下定义，就能大致理解：它是一种和"正义"、"法律"有关的传说动物。

**你是通过"它和谁在一起"来学会"它是什么意思"的。**

Firth 的洞察在 1957 年只是一个语言学假说。半个多世纪后，机器学习用数学把它变成了现实。

---

### Word2Vec：让数字学会远近

2013 年，Google 的 Tomáš Mikolov 和同事做了一件事，彻底改变了 NLP 的走向。

他们的想法极其简单：**让每个词不再是一个编号，而是一组可以训练的数字。训练目标就是 Firth 说的那个——用一个词去预测它周围的词。**

```text
训练语料：
  "... the cat sat on the mat ..."

对于 "sat" 这个词，模型要学会预测：
  前面可能是 "cat"
  后面可能是 "on"

如果模型能做到这一点，那 "sat" 的向量就必须
编码足够多的信息——它是动词、它表示动作、
它的主语通常是有生命的东西...
```

训练结束后，每个词不再是 50000 维的 One-Hot 向量，而是一个 **300 维的稠密向量**——300 个实数。

这 300 个数字不是人类指定的。它们是模型**自己学出来的**。

> Mikolov, T. et al. (2013). *Efficient Estimation of Word Representations in Vector Space*. arXiv:1301.3781.

---

### king − man + woman ≈ queen

然后，一件惊人的事情被发现了。

如果你把这些训练出来的向量做算术运算：

```text
vec("king") − vec("man") + vec("woman") ≈ vec("queen")
```

**国王减去男性，加上女性，约等于女王。**

这不是任何人告诉模型的。没有人标注过"king 和 queen 的关系等于 man 和 woman 的关系"。模型只是在读文本、预测下一个词，然后这些关系就**自己浮现了**。

![Word2Vec 类比：king-man+woman≈queen 的向量算术](02_king_queen_analogy.png)

<div style="text-align: center; font-size: 0.85em; color: #888; margin-top: -10px; margin-bottom: 20px;">▲ 向量空间中，从 man 到 woman 的方向 ≈ 从 king 到 queen 的方向</div>

更让人惊讶的是，这种关系**遍布整个向量空间**：

| 类比 | 向量算术 |
|------|----------|
| 国家-首都 | Paris − France + Japan ≈ Tokyo |
| 动词时态 | walking − walked + swam ≈ swimming |
| 比较级 | bigger − big + cold ≈ colder |

**方向就是关系。距离就是相似度。**

Mikolov 的第二篇论文系统地验证了这些发现，并发现负采样（negative sampling）这个训练技巧是让它成功的关键。

> Mikolov, T. et al. (2013). *Distributed Representations of Words and Phrases and their Compositionality*. NeurIPS.

### 为什么算术能表达语义？

这不是魔法。2016 年，Princeton 的 Sanjeev Arora 从数学上解释了原因：如果文本生成过程可以建模为一个"随机游走"，而每个词的出现概率和它的上下文向量的内积有关，那么 **对数共现概率自然会形成线性结构**。

换句话说：Word2Vec 学到的不是"意思"——它学到的是**共现模式的对数结构**。而人类语言中的语义关系，恰好在这种对数结构里表现为**线性关系**。

> Arora, S. et al. (2016). *A Latent Variable Model Approach to PMI-based Word Embeddings*. TACL.

这是一个深刻的发现：**语义关系之所以能用向量算术表达，是因为语言的统计结构本身就具有这种几何性质。** 这不是 Word2Vec 的巧合，而是人类语言的数学本性。

<div style="background: rgba(76,175,80,0.08); border-left: 4px solid #4CAF50; padding: 12px 16px; margin: 20px 0; border-radius: 0 6px 6px 0;">

**一句话记住：** Embedding 把编号变成了坐标。编号只是标签，坐标有远近。Word2Vec 证明了：**让数字通过训练学会远近亲疏，它们就自动发现了语义结构。**

</div>

---

## 三、"大漠孤烟直，长河落日圆"——人类的 Embedding

### 同一个词，在不同的人"脑"里，是不同的向量

让我们暂时离开计算机，想一个更基本的问题：

**人类是怎么"理解"一个词的？**

看这首王维的诗：

> **大漠孤烟直，长河落日圆。**

十个字。每个字你在小学就认识了。"大"、"漠"、"孤"、"烟"、"直"——没有一个生僻字。

但这十个字组合在一起，你"看到"了什么？

你看到了一幅画面：无边的沙漠，一缕炊烟笔直升起；黄河奔流，一轮落日浑圆地悬在地平线上。

你感受到了一种情绪：苍凉、壮阔、孤独。

你甚至体会到了一种审美判断："直"和"圆"这两个最朴素的几何词，用在这里为什么如此精妙——因为在无边的水平荒原上，唯一的垂直线是那缕烟，唯一的完美弧形是那轮日。极简的形状对极阔的空间。

**一个"直"字，在你的认知里，不是字典里"不弯曲"这三个字。它携带了视觉画面、空间构图、情感色彩、审美直觉——几十个维度的信息。**

再看马致远的小令：

> **枯藤老树昏鸦，小桥流水人家，古道西风瘦马。**

九个名词，没有一个动词。但你读完立刻看到了一幅完整的画面，感受到了深重的漂泊和乡愁。

"枯"不只是"干枯"。在你的认知中，"枯"连接着衰败、时间流逝、生命终结、秋冬、萧条——一整个**语义星系**。"老"也是。"昏"也是。它们各自携带的不是一个定义，而是一张语义网络。当九个这样的词排列在一起时，它们的语义网络**叠加、共振**，产生了远超字面意思的效果。

**这就是人类的 Embedding。**

每个词在你的大脑中，不是一个编号，而是一团丰富的联想——视觉的、听觉的、情感的、经验的。你的人生阅历越丰富，每个词在你脑中的表示就越**厚**、越**多维**。

一个孩子读"大漠孤烟直"，可能只理解字面意思：沙漠里有一缕烟是直的。
一个成年人读同一句，看到的是画面、孤独和壮美。
一个诗人读同一句，还能感受到王维在句式对仗中的克制——他不写"苍凉的大漠中一缕孤独的烟笔直升起"，他只说五个字，留白给读者。

**同一个词，在不同人的"向量空间"中，维度不同。**

这和 Embedding 的核心思想惊人地一致：

| 人类认知 | Embedding |
|---------|-----------|
| 一个词在你脑中的全部联想 | 一个词的向量表示 |
| 人生阅历越丰富，理解越深 | 训练数据越多，向量越精确 |
| 语境改变理解 | 上下文改变 Embedding（Transformer） |
| "近义词"直觉上感觉"近" | 近义词在向量空间中距离近 |
| 诗人用词的"精准"感 | 向量在语义空间中的位置恰到好处 |

---

### 那么，LLM 的 Embedding 在多大程度上接近人类的理解？

直说吧：**还差很远。但比你想象的近。**

LLM 的 Embedding 没有视觉、没有身体、没有情感体验。它的"大漠"向量不来自亲眼看到沙漠的经历，而来自读过的几百万个包含"大漠"的句子。

但正因为它读过的"大漠"的句子足够多、足够多样——

```text
"大漠孤烟直" — 和 孤独、壮阔、边塞 共现
"穿越大漠"   — 和 旅行、艰辛、戈壁 共现
"大漠沙如雪" — 和 苍白、寒冷、月光 共现
```

——它的"大漠"向量最终编码了一种**统计意义上的"理解"**：在向量空间中，"大漠"离"戈壁"近，离"苍凉"近，离"边塞"近，离"超市"远。

这不是真正的理解。但它是一种**功能性的理解**——在实际任务中，它表现得**仿佛**理解了。

> 这正是哲学家 Stevan Harnad 在 1990 年提出的 **符号接地问题（Symbol Grounding Problem）**：纯粹从符号到符号的映射，能不能产生"意义"？如果 Embedding 从未"见过"沙漠，它的"大漠"向量里有意义吗？
>
> Harnad, S. (1990). *The Symbol Grounding Problem*. Physica D, 42(1-3), 335-346.
>
> 这个问题至今没有定论。但 Embedding 的实际表现，至少迫使我们重新思考"理解"这个词的定义边界。

<div style="background: rgba(76,175,80,0.08); border-left: 4px solid #4CAF50; padding: 12px 16px; margin: 20px 0; border-radius: 0 6px 6px 0;">

**一句话记住：** 人类用一辈子的经历训练自己的 Embedding——每个词在你脑中不是定义，而是一团丰富的联想。LLM 用万亿个字训练它的。**方式不同，做的是同一件事：让数字学会远近亲疏。**

</div>

---

## 四、拆开 Embedding 矩阵：一张"活"的查表

### 它说白了就是一张表

好了，让我们回到工程层面。把诗意放下，看看 Embedding 在代码里到底是什么。

在 Karpathy 的 microgpt 中（200 行纯 Python 的 GPT 实现），Embedding 的初始化只有一行：

```python
state_dict = {
    'wte': matrix(vocab_size, n_embd),  # Token Embedding 表
    'wpe': matrix(block_size, n_embd),  # Position Embedding 表
    ...
}
```

在 nanoGPT 的 PyTorch 实现中：

```python
self.transformer = nn.ModuleDict(dict(
    wte = nn.Embedding(config.vocab_size, config.n_embd),   # Token Embedding
    wpe = nn.Embedding(config.block_size, config.n_embd),   # Position Embedding
    ...
))
```

`nn.Embedding` 是什么？**它就是一张表。** 一个 `vocab_size × n_embd` 的矩阵。

```text
Embedding 矩阵（Token Embedding Table）:

         维度1  维度2  维度3  ...  维度768
token 0: [0.02, -0.15, 0.08, ..., 0.11]
token 1: [0.31,  0.05, -0.22, ..., 0.04]
token 2: [-0.07, 0.18,  0.13, ..., -0.09]
...
token 50256: [0.14, -0.03, 0.21, ..., 0.06]
```

当你输入一个 token（比如 token ID = 1820），模型做的事情就是：**去第 1820 行，把那 768 个数字取出来。**

```python
tok_emb = state_dict['wte'][token_id]  # 就是查第 token_id 行
```

**这就是一次查表操作。和 ASCII 查表在结构上完全一样。**

| | ASCII | Embedding |
|---|-------|-----------|
| 输入 | 字符"A" | Token ID 1820 |
| 查表 | 得到数字 65 | 得到 768 个数字 |
| 数字含义 | 无意义的编号 | **有意义的坐标** |
| 表的来源 | 人类设计的 | **训练学出来的** |

**区别只有一个：ASCII 表是人写的，Embedding 表是训练出来的。**

就这一个区别，让数字从"编号"变成了"坐标"——有了远近，有了方向，有了意思。

---

### Embedding 占了多少参数？

Embedding 矩阵在整个模型中占的参数比例是多少？让我们用实际数字算一算。

```text
microgpt (Karpathy 的 200 行 GPT):
  词表大小 = 27（26 个字母 + 1 个特殊符号）
  嵌入维度 = 16
  Token Embedding: 27 × 16 = 432 个参数
  Position Embedding: 16 × 16 = 256 个参数
  总 Embedding: 688 / 4192 = 16.4%

nanoGPT (Shakespeare 字符级):
  词表大小 = 65
  嵌入维度 = 384
  Token Embedding: 65 × 384 = 24,960
  Position Embedding: 256 × 384 = 98,304
  总 Embedding: ~123K / 810K = ~15%

GPT-2 (1.24 亿参数):
  词表大小 = 50,257
  嵌入维度 = 768
  Token Embedding: 50,257 × 768 = 38,597,376
  Position Embedding: 1024 × 768 = 786,432
  总 Embedding: ~39M / 124M = ~32%

GPT-3 (1750 亿参数):
  词表大小 = 50,257
  嵌入维度 = 12,288
  Token Embedding: 50,257 × 12,288 = 617,558,016
  Position Embedding: 2048 × 12,288 = 25,165,824
  总 Embedding: ~643M / 175,000M = ~0.4%
```

一个有意思的趋势：**模型越大，Embedding 占的比例越小，但绝对值越大。**

GPT-2 中 Embedding 占了近三分之一的参数——这意味着这个模型"大量的知识"其实就存在那张查表里。

而 GPT-3 中 Embedding 只占 0.4%，剩下的 99.6% 参数都在 Transformer 的层里——那些才是"处理"和"思考"的部分。

> **有趣的设计：weight tying。** 在 GPT-2 和 nanoGPT 中，Token Embedding 矩阵和最后的输出层（lm_head）**共享权重**：
>
> ```python
> self.transformer.wte.weight = self.lm_head.weight
> ```
>
> 也就是说，**把 token 变成向量的矩阵**和**把向量变回 token 概率的矩阵**是同一张表！输入用它查表，输出用它的转置做投影。这个优雅的设计叫 **weight tying**，不仅节省参数，还强制了一种对称性：如果两个 token 的 Embedding 向量接近，它们在输出层被选中的概率也接近。
>
> Inan, H., Khosravi, K., & Socher, R. (2017). *Tying Word Vectors and Word Classifiers*. ICLR.

<div style="background: rgba(76,175,80,0.08); border-left: 4px solid #4CAF50; padding: 12px 16px; margin: 20px 0; border-radius: 0 6px 6px 0;">

**一句话记住：** Embedding 矩阵就是一张查表——和 ASCII 表结构完全相同。**区别只有一个：ASCII 表是人写的，Embedding 表是训练出来的。** 就这一个区别，让数字从编号变成了坐标。

</div>

---

## 五、Embedding 与 MLP：谁记知识，谁做推理？

### 一个根本性的分工

Transformer 里有两个核心组件不断交替：**Attention** 和 **MLP（前馈网络）**。而在它们之前，有 Embedding 作为入口。

这三者的关系，可以用一个类比来理解：

> **Embedding 是字典**——你查一个词，得到它的基本含义。
>
> **Attention 是阅读理解**——你根据上下文，决定这个词在这里具体指什么。
>
> **MLP 是知识库和推理引擎**——你根据理解到的信息，做判断、做推理、给出结论。

让我们看看在 Transformer 的一次前向传播中，信息是怎么流动的：

```text
Token "银行"
    ↓
[Embedding 查表] → 得到"银行"的基础向量
                    （此时它携带"银行"的全部可能含义——
                     金融机构？河岸？数据库？）
    ↓
[Attention 层]   → 看到上下文是"我去银行取钱"
                    → 把向量往"金融机构"方向偏移
                    （如果上下文是"坐在河银行上钓鱼"，
                     就会往"河岸"方向偏移）
    ↓
[MLP 层]         → 利用存储的知识进行推理
                    → "取钱 → 需要银行卡 → 可能排队..."
                    → 修改向量，为下一步预测做准备
    ↓
[Attention 层]   → 继续看上下文...
    ↓
[MLP 层]         → 继续推理...
    ↓
... 重复 N 层 ...
    ↓
[lm_head]        → 用 Embedding 矩阵的转置（weight tying）
                    把最终向量投影回词表，选出下一个词
```

### MLP 里存储了什么？

2024 年的一项重要研究发现，**MLP 层是 Transformer 中主要的"知识存储"位置**。具体来说：

- **Embedding 层**存储的是**词汇级别的基础语义**——每个词的"字典义"
- **MLP 层**存储的是**世界知识和推理模式**——"巴黎是法国的首都"、"水在 100°C 沸腾"这类事实

> Geva, M. et al. (2021). *Transformer Feed-Forward Layers Are Key-Value Memories*. EMNLP.
>
> 这篇论文证明了 MLP 的每一层可以被解读为一个**键-值记忆网络**：第一个线性层的行向量是"键"（匹配模式），第二个线性层的列向量是"值"（要写入的信息）。

所以信息在 Embedding 和 MLP 之间的互动是这样的：

```text
Embedding: "我知道每个词大概是什么意思"
            ↓ （提供基础向量）
Attention:  "我知道这些词在这个上下文里各自指什么"
            ↓ （利用上下文消歧）
MLP:        "我知道这个世界怎么运作的，
             基于上下文，我来推理下一步"
            ↓ （注入知识，修改表示）
... 反复交替 ...
```

**Embedding 给出起点，Attention 定位语境，MLP 注入知识。** 三者缺一不可。

如果把它类比为人类阅读：

- **Embedding** 相当于你认识这些字——你知道"大"、"漠"、"孤"、"烟"各自的基本意思
- **Attention** 相当于你读懂了句子——你知道"孤"在这里修饰的是"烟"，而"大"修饰的是"漠"
- **MLP** 相当于你的知识和想象力被激活——你知道大漠是什么样的地方，能在脑中构建出那幅苍凉的画面

**一首好诗之所以能在你脑中产生画面，不只是因为你认识每个字（Embedding），也不只是因为你读懂了句子结构（Attention），更因为你有足够的知识和想象力去"补全"字面之外的信息（MLP）。**

<div style="background: rgba(76,175,80,0.08); border-left: 4px solid #4CAF50; padding: 12px 16px; margin: 20px 0; border-radius: 0 6px 6px 0;">

**一句话记住：** Embedding 是字典，Attention 是阅读理解，MLP 是知识库。三者交替协作：**Embedding 给出起点，Attention 定位语境，MLP 注入知识。**

</div>

---

## 六、多语言：中文和英文在同一个空间里吗？

### 一个令人困惑的事实

直觉上，中文和英文是完全不同的语言系统。字形不同、语法不同、文化背景不同。

但在现代多语言模型（如 GPT-4、Claude）的 Embedding 空间中：

```text
vec("猫") 和 vec("cat") 的距离，
比 vec("猫") 和 vec("经济学") 的距离近得多。
```

**不同语言的同义词，在向量空间中被映射到了相近的位置。**

这是怎么做到的？

### 早期方案：跨语言对齐

最早的多语言 Embedding 方法很直觉：先分别训练中文和英文的 Word2Vec，得到两个独立的向量空间；然后用一个"翻译词典"（比如 5000 对中英文翻译对）作为"锚点"，学一个**旋转矩阵**，把中文空间"旋转"到和英文空间对齐。

```text
中文空间          旋转矩阵 W          英文空间
  "猫" ──────────── × W ──────────→ 靠近 "cat"
  "狗" ──────────── × W ──────────→ 靠近 "dog"
  "桌子" ─────────── × W ──────────→ 靠近 "table"
```

> Conneau, A. et al. (2018). *Word Translation Without Parallel Data*. ICLR.
>
> 这篇论文更进一步：证明了即使没有翻译词典，仅靠对抗训练（GAN）也能自动发现这个旋转矩阵——说明不同语言的 Embedding 空间有**固有的结构相似性**。

### 现代方案：从一开始就混着训练

GPT-4、Claude 等现代 LLM 不需要对齐。因为它们从训练的第一天起就**同时读中文、英文、法文、日文...** 的文本。

```text
训练数据里同时有：
  "The cat sat on the mat."
  "猫坐在垫子上。"
  "Le chat est assis sur le tapis."

模型学到的 Embedding 空间自然就是多语言的：
  "cat"、"猫"、"chat" → 空间中的相近位置
```

**但有一个不对称的问题。** 还记得上一篇提到的吗？

> 英文 "the cat" 通常是 2 个 token，中文 "猫" 可能是 1-3 个 token（取决于 BPE 分词器的训练语料比例）。

这意味着中文在 token 层面的 Embedding 空间中，是以**更碎片化**的方式存在的。英文的常见词往往有完整的 Embedding 向量，而中文的某些字被切成了字节级别的 token，需要经过多层 Transformer 才能重新组合出完整的语义。

> 这也解释了为什么 LLM 在中文上的表现通常略逊于英文——不是因为中文"更难"，而是因为编码效率不对等。同样的信息，中文需要更多的 token 来表达，模型需要"更努力"才能理解。

---

### 一个深刻的问题：不同语言的人，"想"的是同一个东西吗？

如果中文"猫"和英文"cat"在向量空间中位置相近，是不是说明不同语言的使用者在"想"同一个概念时，大脑的表示也类似？

神经科学的研究发现了惊人的证据：

> Huth, A. et al. (2016) 的 fMRI 研究显示，英语使用者和中文使用者在听同一个故事的翻译版本时，大脑的语义表征在高层次上**高度相似**。
>
> 也就是说，"猫"和"cat"激活的大脑区域模式，确实是类似的。语言是不同的编码，但编码的对象——**那个概念**——可能是共通的。

这和 Embedding 的发现是一致的：**不同的编码系统（不同的语言），在经过充分训练后，趋向于发现相似的底层结构。**

<div style="background: rgba(76,175,80,0.08); border-left: 4px solid #4CAF50; padding: 12px 16px; margin: 20px 0; border-radius: 0 6px 6px 0;">

**一句话记住：** 中文"猫"和英文"cat"在向量空间中是邻居——不是因为有人标注了对应关系，而是因为它们在各自语言中出现的上下文是相似的。**语言不同，但数字学会的远近亲疏是相通的。**

</div>

---

## 七、超越文字：万物皆可 Embedding

### 图片有 Embedding 吗？

有。而且方式出人意料的相似。

**Vision Transformer (ViT)** 把一张图片切成 16×16 的小块（patch），每个小块就像一个"视觉词元"。每个 patch 被投影成一个向量——这就是图片的 Embedding。

```text
文本: "猫坐在垫子上"
      ↓ tokenize
      ["猫", "坐在", "垫", "子上"]
      ↓ Embedding 查表
      [vec₁, vec₂, vec₃, vec₄]  ← 每个是 768 维向量

图片: 一张猫的照片
      ↓ 切成 14×14 = 196 个 patch
      [patch₁, patch₂, ..., patch₁₉₆]
      ↓ 线性投影（就是一次矩阵乘法）
      [vec₁, vec₂, ..., vec₁₉₆]  ← 每个也是 768 维向量
```

**从这一步开始，文本和图片的处理方式完全相同**——都是一系列向量，送入 Transformer 做 Attention + MLP。

### CLIP：让文字和图片住进同一个空间

2021 年，OpenAI 做了一件更大胆的事：**训练一个模型，让文字的 Embedding 和图片的 Embedding 住在同一个向量空间里。**

```text
训练数据：4 亿个 (图片, 文字描述) 配对

训练目标：
  - "一只猫坐在垫子上" 的文字向量
  - 一张猫坐在垫子上的图片向量
  → 这两个向量应该很近

  - "一只猫坐在垫子上" 的文字向量
  - 一张赛车的图片向量
  → 这两个向量应该很远
```

训练完成后：

```text
cos_similarity(
    CLIP_text("a photo of a cat"),
    CLIP_image(猫的照片)
) ≈ 0.85

cos_similarity(
    CLIP_text("a photo of a cat"),
    CLIP_image(赛车照片)
) ≈ 0.12
```

**文字和图片可以直接用向量距离来比较。** 你可以用文字搜图片，用图片搜文字——因为它们在同一个空间里。

> Radford, A. et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. ICML.

### ImageBind：六种模态，一个空间

Meta 在 2023 年更进一步，用 **ImageBind** 把**六种模态**——文字、图片、音频、深度图、热力图、IMU 传感器数据——全部映射到同一个 Embedding 空间。

```text
同一个空间里住着：
  - "狗叫声" 的音频向量
  - "一只狗" 的图片向量
  - "dog" 的文字向量
  - 狗的热成像图的向量

它们在空间中是邻居。
```

> Girdhar, R. et al. (2023). *ImageBind: One Embedding Space To Bind Them All*. CVPR.

![多模态 Embedding 空间：文字、图片、音频映射到同一个向量空间](03_multimodal_embedding.png)

<div style="text-align: center; font-size: 0.85em; color: #888; margin-top: -10px; margin-bottom: 20px;">▲ 不同模态的数据，经过各自的编码器，被映射到同一个向量空间中</div>

这意味着一件惊人的事：

**"猫"这个概念，无论是一个中文字、一个英文单词、一张照片、还是一段猫叫声——在向量空间中，它们都指向相近的位置。**

仿佛存在一个**独立于任何具体模态的"概念空间"**，而文字、图片、声音只是进入这个空间的不同入口。

---

### 柏拉图表示假说：所有模型都在趋向同一个现实？

2024 年，MIT 的 Minyoung Huh 和 Phillip Isola 提出了一个大胆的假说：

> **不同的神经网络，用不同的数据、不同的目标训练，最终学到的表示正在趋向一致——趋向现实世界的统计结构本身。**
>
> Huh, M. & Isola, P. (2024). *The Platonic Representation Hypothesis*. ICML.

他们的证据：

- 更大的语言模型，其内部表示和视觉模型的内部表示**越来越对齐**
- 更大的视觉模型，其内部表示和语言模型的内部表示也**越来越对齐**
- 这种趋同在**不同架构、不同训练集、不同模态**之间都在发生

他们的解释：所有数据——文字、图片、声音——都是底层现实的**投影**。如果一个模型足够强大，训练数据足够多，它最终会"透过"数据的表面形式，逼近那个底层的统计结构。

**就像柏拉图的洞穴寓言：我们看到的文字、图片、声音都是洞壁上的影子。但所有影子来自同一个"真实"。足够好的 Embedding 空间，就是对那个"真实"的逼近。**

这是一个哲学性的假说，远未被证实。但它提供了一个诱人的视角：Embedding 不仅仅是一种工程技巧——它可能在做一件更深刻的事：**用数学逼近现实的结构本身。**

<div style="background: rgba(76,175,80,0.08); border-left: 4px solid #4CAF50; padding: 12px 16px; margin: 20px 0; border-radius: 0 6px 6px 0;">

**一句话记住：** 文字、图片、声音——不同的入口，同一个空间。"猫"这个概念，无论你写它、拍它、还是听它叫——在足够好的 Embedding 中，它们都是邻居。**数字学会的远近亲疏，超越了任何一种模态。**

</div>

---

## 八、从静态到动态：Embedding 的进化

### Word2Vec 的局限：一词一义

Word2Vec 给每个词**一个**固定向量。但人类语言充满了多义词：

```text
"苹果" → 水果？公司？
"打"   → 打人？打电话？打酱油？打车？
"bank" → 银行？河岸？
```

在 Word2Vec 中，"bank" 只有一个向量——它是"银行"和"河岸"的某种混合。这显然不对。

### ELMo (2018)：让向量随语境变化

Matthew Peters 提出了一个关键改进：用双向 LSTM 在每个具体的上下文中**重新计算**每个词的表示。

```text
"I went to the bank to deposit money."
  → "bank" 的向量 ≈ [金融机构方向]

"I sat on the bank of the river."
  → "bank" 的向量 ≈ [河岸方向]
```

**同一个词，不同的上下文，不同的向量。**

> Peters, M. et al. (2018). *Deep Contextualized Word Representations*. NAACL.

### Transformer：Embedding 只是起点

在今天的 Transformer 架构中，这个进化更加彻底。

Token Embedding 矩阵给出的只是**初始向量**——一个词的"字典义"。然后经过 N 层 Transformer（Attention + MLP），每一层都在**修改**这个向量：

```text
层 0:  "bank" = [基础向量——所有义项的混合]
层 1:  注意到 "deposit money" → 向量偏移到 [金融机构]
层 2:  注意到 "went to" → 确认是 [具体地点]
层 3:  注意到 "I" → 标记为 [第一人称经历]
...
层 96: 最终向量 = 高度精炼的、语境化的表示
```

**Embedding 查表只是第一步。** 真正的"理解"发生在后续的 96 层里。Embedding 提供了原材料，Transformer 对它们进行加工。

这也是为什么前面说 GPT-3 的 Embedding 只占 0.4% 参数——**真正的"智能"不在查表那一步，而在查表之后的处理中。**

但没有那个起点，一切都无法开始。就像你必须先认识"大"、"漠"、"孤"、"烟"这些字，才有可能理解"大漠孤烟直"。

<div style="background: rgba(76,175,80,0.08); border-left: 4px solid #4CAF50; padding: 12px 16px; margin: 20px 0; border-radius: 0 6px 6px 0;">

**一句话记住：** Word2Vec 给每个词一个"身份"——固定不变。Transformer 给每个词一个"语境中的自我"——随上下文而变。**Embedding 只是起点，96 层 Transformer 才是让数字学会远近亲疏的全过程。**

</div>

---

## 九、压缩即理解：Embedding 的信息论视角

### 从 50000 维到 768 维，强制学习

回到我们系列的核心命题：**压缩即智能。**

One-Hot 把每个词表示为 50000 维的向量。Embedding 把它压缩到 768 维。

50000 → 768。压缩了 **65 倍**。

这意味着什么？

信息论告诉我们：**无损压缩的前提是发现数据中的结构。** 如果 50000 个词真的是 50000 个完全独立的概念，那就不可能压缩。之所以能压到 768 维，是因为这些词之间存在大量的**冗余和关联**。

- "猫"和"狗"共享很多属性（动物、宠物、有毛、四条腿...），所以它们的向量可以共享很多维度
- "走"和"跑"共享运动语义，只在速度维度上有差异
- "快乐"和"悲伤"的向量可能在很多维度上接近（都是情感、都是形容词），但在"正面/负面"这个维度上方向相反

**压缩迫使模型发现这些结构。** 如果不发现语义上的共性，就无法用 768 个数字区分 50000 个词。这和你在 [信息论文章](/ai-blog/posts/see-math-extra-information-theory/) 中读到的 Shannon 的核心思想一脉相承：压缩 = 预测 = 理解。

> Johnson-Lindenstrauss 引理从数学上保证了：高维空间中的点，可以被投影到低得多的维度中，同时**几乎保持它们之间的距离关系**。只要目标维度足够（对数级别），距离的失真就可以控制在任意小的范围内。
>
> Johnson, W. & Lindenstrauss, J. (1984). *Extensions of Lipschitz mappings into a Hilbert space*.
>
> Embedding 的成功，部分是因为这个数学保证：768 维已经足以保持 50000 个词之间的绝大部分距离关系。

<div style="background: rgba(76,175,80,0.08); border-left: 4px solid #4CAF50; padding: 12px 16px; margin: 20px 0; border-radius: 0 6px 6px 0;">

**一句话记住：** 50000 维压缩到 768 维——65 倍的压缩迫使模型发现词与词之间的共性。**压缩即智能，Embedding 是这五个字最生动的注脚。**

</div>

---

## 十、用 microgpt 亲眼看到 Embedding

回到我们的教学体系。在 microgpt 中——那个只有 200 行、零依赖的 GPT 实现——你可以亲眼看到 Embedding 矩阵是什么样的：

```python
# microgpt 中的 Embedding 初始化
n_embd = 16        # 嵌入维度只有 16！
vocab_size = 27     # 26 个字母 + 1 个特殊符号
block_size = 16     # 最大上下文长度

state_dict = {
    'wte': matrix(vocab_size, n_embd),  # 27 × 16 = 432 个参数
    'wpe': matrix(block_size, n_embd),  # 16 × 16 = 256 个参数
    ...
}
```

训练前，这 432 个数字是随机的。训练 1000 步之后（约 4 分钟），它们自动组织成了**有结构的向量**：

```text
训练前：每个字母的 16 维向量是随机噪声
训练后：
  - 元音字母（a, e, i, o, u）的向量彼此接近
  - 常见辅音（s, t, n, r）的向量形成一个簇
  - 罕见字母（q, x, z）被推到空间的边缘
```

这发生在一个只有 **4192 个参数**的模型中——连 GPT-2 参数量的万分之一都不到。但 Embedding 的核心机制已经在工作了：**数字通过训练，学会了反映字母之间的使用规律。**

```python
# 在 nanoGPT 中运行 Embedding 可视化 demo
# 此脚本加载训练好的模型，展示 Embedding 最近邻
source ~/ai-lab-venv/bin/activate
cd ~/nanoGPT
python demo_embedding_viz.py
```

---

<div style="border: 2px solid #FF9800; border-radius: 8px; padding: 15px 20px; margin: 20px 0; background: rgba(255,152,0,0.04);">

**本篇小结**

**一、编号是死的** — A=65, B=66，数字之间的大小和意义无关。One-Hot 编码让所有词两两等距，无法泛化。

**二、Embedding 把编号变成坐标** — Firth 1957 年的"看它和谁在一起"被 Word2Vec 翻译成了矩阵乘法。king − man + woman ≈ queen 不是魔法，是语言统计结构的几何性质。

**三、人类也在做同样的事** — "大漠孤烟直"之所以能激起画面和情感，是因为每个词在你的认知中不是定义，而是一团多维的联想。你的人生阅历，就是你的训练数据。

**四、说到底就是一张表** — Embedding 矩阵就是查找表，但这张表是训练出来的。GPT-2 中它占 32% 参数，GPT-3 中只占 0.4%——模型越大，"思考"的部分越多，"查表"的比例越小。

**五、三者协作** — Embedding 给出起点（认识字），Attention 定位语境（读懂句子），MLP 注入知识（调动想象力）。

**六、语言不是障碍** — 中英文的同义词在向量空间中自动成为邻居。不同的编码系统，趋向于发现相似的底层结构。

**七、万物皆可 Embedding** — 文字、图片、音频映射到同一个空间后，"猫"这个概念无论写它、拍它、还是听它叫，在向量空间中都是邻居。

**八、从静态到动态** — Word2Vec 一词一义，Transformer 让每个词的向量随上下文而变。Embedding 只是起点。

**九、压缩即理解** — 50000 维压到 768 维，这 65 倍的压缩迫使模型发现语义结构。这是"压缩即智能"最生动的注脚。

</div>

## 写在最后

在这个系列的 [开篇语](/ai-blog/posts/opening-essay/) 里，我写了五个字：**压缩即智能。**

Embedding 是这五个字最生动的注脚。

50000 个词，每个词本来需要一个独立的维度来表示。但如果你硬逼它只用 768 个数字来区分所有的词——它就不得不去发现：哪些词是近的，哪些词是远的，哪些词之间有规律。

**"猫"和"狗"之所以在向量空间中是邻居，不是因为任何人告诉模型它们是相似的，而是因为在人类写下的万亿个字中，它们总是出现在相似的上下文中。**

Firth 在 1957 年说"看看它和谁在一起"。六十多年后，这句话被翻译成了矩阵乘法和梯度下降。

而王维在 1200 年前写下的"大漠孤烟直"，之所以能在我们心中激起画面和情感，也是因为"大漠"、"孤"、"烟"、"直"这些词在我们的认知中，不是字典里的定义，而是丰富的、多维的、经过一生的阅读和体验训练出来的**向量**。

人类用一辈子的经历训练自己的 Embedding。LLM 用万亿个字训练它的。方式不同，但最终做的是同一件事：

**让数字学会远近亲疏。**

---

> **参考文献**
>
> 1. Firth, J.R. (1957). *A Synopsis of Linguistic Theory 1930-1955*. Studies in Linguistic Analysis. — "You shall know a word by the company it keeps."
> 2. Bengio, Y. et al. (2003). *A Neural Probabilistic Language Model*. JMLR, 3, 1137-1155. — 第一个用神经网络学习词的分布式表示的语言模型。
> 3. Mikolov, T. et al. (2013a). *Efficient Estimation of Word Representations in Vector Space*. arXiv:1301.3781. — Word2Vec 的提出。
> 4. Mikolov, T. et al. (2013b). *Distributed Representations of Words and Phrases and their Compositionality*. NeurIPS. — 负采样、词组向量、类比发现。
> 5. Pennington, J., Socher, R. & Manning, C.D. (2014). *GloVe: Global Vectors for Word Representation*. EMNLP. — 将全局共现统计与局部上下文预测统一。
> 6. Arora, S. et al. (2016). *A Latent Variable Model Approach to PMI-based Word Embeddings*. TACL. — 数学解释 Word2Vec 线性类比的原因。
> 7. Peters, M. et al. (2018). *Deep Contextualized Word Representations*. NAACL. — ELMo，第一个上下文化 Embedding。
> 8. Conneau, A. et al. (2018). *Word Translation Without Parallel Data*. ICLR. — 无平行语料的跨语言对齐。
> 9. Harnad, S. (1990). *The Symbol Grounding Problem*. Physica D, 42, 335-346. — Embedding 能否产生"意义"？
> 10. Geva, M. et al. (2021). *Transformer Feed-Forward Layers Are Key-Value Memories*. EMNLP. — MLP 层作为知识存储。
> 11. Radford, A. et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. ICML. — CLIP，文字和图片共享向量空间。
> 12. Girdhar, R. et al. (2023). *ImageBind: One Embedding Space To Bind Them All*. CVPR. — 六种模态，一个空间。
> 13. Huh, M. & Isola, P. (2024). *The Platonic Representation Hypothesis*. ICML. — 所有模型趋向同一个表示。
> 14. Inan, H. et al. (2017). *Tying Word Vectors and Word Classifiers*. ICLR. — Weight tying 技巧。
> 15. Johnson, W. & Lindenstrauss, J. (1984). *Extensions of Lipschitz Mappings into a Hilbert Space*. Contemporary Mathematics. — 高维到低维投影保持距离。
> 16. Karpathy, A. (2024). *microgpt*. GitHub Gist. — 200 行纯 Python GPT 实现。
>
> **推荐阅读**
>
> - Jay Alammar (2019). [*The Illustrated Word2Vec*](https://jalammar.github.io/illustrated-word2vec/) — 最好的 Word2Vec 可视化教程。
> - Christopher Olah (2014). [*Deep Learning, NLP, and Representations*](https://colah.github.io/posts/2014-07-NLP-RNNs-Representations/) — 从表示学习的角度理解 NLP。

---

<div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #e0e0e0; font-size: 14px; color: #999; line-height: 1.8;">

💡 相关文章

- [计算机只懂 0 和 1——从莫尔斯电码到 GPT 的编码简史](/ai-blog/posts/ascii-to-token/) — 本文的前篇
- [AI 的数学语言（一）：向量——万物皆可数字化](/ai-blog/posts/math-for-ai-1-vectors/) — 什么是向量
- [AI 的数学语言（二）：点积——相似度的量化](/ai-blog/posts/math-for-ai-2-dot-product/) — 余弦相似度
- [看见数学（十四）：高维空间——直觉失效的地方](/ai-blog/posts/see-math-14-high-dimensions/) — 为什么 768 维不会爆炸
- [看见数学（番外）：信息论——从电报到 GPT 的一条暗线](/ai-blog/posts/see-math-extra-information-theory/) — 压缩即智能
- [Shannon 没有想到的事——当信息论遇上有限算力](/ai-blog/posts/epiplexity/) — 有限的学习者能从数据中学到多少
- [从一个取反说起——计算机如何从「只会加法」走到「AI 写诗」](/ai-blog/posts/gates-to-gpt/) — 7 层抽象

博客：https://Jason-Azure.github.io/ai-blog/

微信公众号：AI-lab学习笔记

</div>
