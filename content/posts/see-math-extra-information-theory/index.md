---
title: "看见数学（番外）：信息论——从电报到 GPT 的一条暗线"
date: 2026-03-26
draft: false
summary: "Shannon 1948 年证明了一件事：压缩 = 预测 = 理解。76 年后，我们用万亿参数的神经网络去逼近他的定理。这是贯穿「看见数学」所有篇章的那条暗线。"
categories: ["看见数学"]
tags: ["信息论", "Shannon", "熵", "交叉熵", "压缩", "看见数学", "数学思维"]
weight: 46
ShowToc: true
TocOpen: true
---

> **系列导航**
>
> <div style="max-width: 660px; margin: 0.5em 0; font-size: 0.93em; line-height: 1.9;">
> <div style="border-left: 3px solid #ccc; padding-left: 12px; margin-bottom: 6px; padding: 8px 12px; color: #888;">
> 第一幕（5 篇）+ 第二幕（5 篇）+ 第三幕（6 篇）<a href="/ai-blog/tags/看见数学/" style="color: #888;">→ 查看全部 16 篇</a></div>
> <div style="border-left: 3px solid #FF9800; padding-left: 12px; background: rgba(255,152,0,0.05); padding: 8px 12px; border-radius: 0 4px 4px 0;">
> <strong>▸ 番外篇（本文）：信息论——从电报到 GPT 的一条暗线</strong></div>
> </div>

---

## 写在前面：为什么还要继续？

十六篇走完了。从结绳记事到梯度下降，从一万年前的绳结到今天的神经网络。

我以为旅程结束了。

但有一件事一直困扰我。博客的 [开篇语](/ai-blog/posts/opening-essay/) 写了四个字：**压缩即智能**。这四个字贯穿了我写的每一篇文章——讲 [MoE](/ai-blog/posts/moe-architecture/) 时我说"用更少的参数做更多的事"是一种压缩，讲 [知识蒸馏](/ai-blog/posts/knowledge-distillation/) 时我说"蒸馏就是极端压缩"，讲 [梯度下降](/ai-blog/posts/see-math-15-gradient/) 时我说训练的本质是寻找最优的模式表达。

但我从来没有正面回答过一个问题：**"压缩即智能"这个说法，从哪来的？有人证明过吗？**

答案是：有。1948 年，一个叫 Claude Shannon 的人，用一篇论文证明了它。

这条线藏在 GPT 训练时用的损失函数里，藏在 token 的编码方式里，藏在温度参数的名字里，藏在你读过的每一篇「看见数学」的背后。它无处不在，但我一直没有把它单独拎出来看。

这条线叫**信息论**。

这不是第十七篇。这是一篇番外——因为它不属于任何一幕，它是贯穿所有幕的那根弦。

---

## 一、一封电报的成本

### 1838 年，Morse 的印刷厂

1838 年。Samuel Morse 正在设计电报码。

他面对一个极其实际的问题：每个字母都要用电信号的"点"和"划"来表示，发电报按时间收费。发得越长，花得越多。

**怎样编码，才能让一封电报最便宜？**

Morse 做了一件聪明的事。他没有坐在书房里空想——他跑到印刷厂，去数铅字盒里每个字母的库存量。道理很简单：印刷厂备的铅字越多，说明这个字母在日常文本中出现得越频繁。

结果一目了然：

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(255,152,0,0.06); border: 1px solid rgba(255,152,0,0.2);">

| 字母 | 使用频率 | 莫尔斯电码 | 符号数 |
|------|---------|-----------|-------|
| **E** | 12.7% | `·` | 1 |
| **T** | 9.1% | `—` | 1 |
| **A** | 8.2% | `· —` | 2 |
| **Z** | 0.07% | `— — · ·` | 4 |
| **Q** | 0.10% | `— — · —` | 4 |

**E** 是英语中最常用的字母，所以它的编码最短——只有一个点。**Q** 几乎不用，编码就长得多。

</div>

Morse 不知道什么叫"信息论"。但他凭直觉做到了一件事：

**越常见的东西，用越短的编码。越罕见的东西，用越长的编码。**

这不只是省钱的技巧。110 年后，一个人会证明：这是宇宙的基本法则之一。

### 1948 年，贝尔实验室

快进到 1948 年。新泽西州，贝尔实验室。

32 岁的 Claude Shannon 发表了一篇论文：*A Mathematical Theory of Communication*——《通信的数学理论》。

在此之前，"信息"是一个模糊的日常用语。人们说"这条消息信息量很大"，就像说"这道菜味道不错"——是一种感觉，无法精确衡量。没有人能回答：

- 一条消息到底包含**多少**信息？
- 信息可以被测量吗？就像重量可以用千克、温度可以用度？
- 一条消息**最少**需要多少符号来表达，不多也不少？

Shannon 一个人，回答了所有这些问题。

就像牛顿给了"力"一个精确定义（F = ma），Shannon 给了"信息"一个精确定义。在此之后，信息不再是一种感觉。它有了单位——**bit**（比特）。

---

## 二、信息的度量——熵

### 什么是"信息"？

Shannon 的定义出人意料地简单：

<div style="max-width: 520px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: #FFF9C4; border: 2px solid #FF9800; text-align: center;">

**信息 = 意外程度 = 不确定性的消除**

</div>

"明天太阳会升起"——信息量约等于零，因为你本来就知道它会升起。

"明天有一颗陨石会撞击地球"——信息量极高，因为你完全没预料到。

一个事件越不可能发生，当它真的发生时，你就越惊讶。而这份惊讶，就是它所携带的信息。

Shannon 把这个直觉变成了公式：

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; border: 2px solid #E91E63; background: rgba(233,30,99,0.04);">

事件 x 的**信息量**（自信息）：

**I(x) = log₂(1 / P(x)) = −log₂ P(x)**

- 概率 = 1（必然发生）→ 信息量 = 0 bit
- 概率 = 1/2（抛硬币）→ 信息量 = 1 bit
- 概率 = 1/8 → 信息量 = 3 bit

</div>

为什么是对数？想象你在猜一个 8 面骰子的结果。朋友告诉你答案——你获得了 3 bit 的信息。为什么是 3？因为你需要问 3 个"是/否"问题：

> "是 5 或以上吗？" → "是 3 或 4 吗？" → "是 3 吗？"

三次二选一，就够了。每次二选一 = 1 bit。所以 8 种可能性 = log₂(8) = 3 bit。

**1 bit = 一次公平的"是/否"判断所消除的不确定性。** 这就是信息的原子。

如果你读过 [第十三篇「概率」](/ai-blog/posts/see-math-13-probability/)，你会发现我们在那里讨论的"不确定性"——在信息论的语言里，有了精确的度量单位。

### 熵：平均的惊讶程度

知道了单个事件的信息量之后，Shannon 问了一个更深的问题：

对于一个信息源——比如英语、比如天气预报、比如股票价格——它**平均**每发出一个符号，携带多少信息？

这个"平均信息量"，就是**熵**（Entropy）：

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; border: 2px solid #E91E63; background: rgba(233,30,99,0.04);">

**H(X) = −∑ P(xᵢ) log₂ P(xᵢ)**

熵 = 每种可能结果的「概率 × 惊讶程度」之和 = **平均惊讶程度**

</div>

"熵"这个名字借自热力学——这不是巧合。Shannon 当年去请教数学家冯·诺依曼该用什么名字，冯·诺依曼说："叫它'熵'吧，反正没有人真正理解熵是什么，所以在辩论中你总会占上风。"

这是一个玩笑。但 Shannon 的信息熵和热力学的热力学熵，确实有深层的数学同构——我们后面讲"温度"参数时会再碰到它。

### 三种方式理解熵

{{< sandbox src="entropy-calculator.html" title="🎛️ 交互演示：拖动滑块，看熵如何随概率分布变化" height="520" >}}

**第一种：猜谜游戏**

熵 = 平均需要问多少个"是/否"问题才能确定结果。

- 公平硬币：H = 1 bit（1 个问题够了）
- 公平 8 面骰：H = 3 bit（3 个问题）
- 不公平硬币（99% 正面）：H ≈ 0.08 bit（几乎不用问，你已经知道答案了）

**第二种：压缩极限**

这是熵最深刻的含义。Shannon 的**信源编码定理**证明了：

> 任何无损压缩方法，平均每个符号所需的最小 bit 数 = 熵。
>
> 你不可能压得比熵更小——除非你愿意丢失信息。

这就是 ZIP 文件的理论极限。也是 LLM 的理论极限。

**第三种：模式的度量**

- 熵最高 = 完全随机 = 无模式可利用 = **不可压缩**
- 熵最低 = 完全确定 = 全是模式 = **极度可压缩**
- 真实世界的数据 = 介于两者之间 = **有模式，但不完全可预测**

语言就是这样的数据。"q" 后面几乎一定跟 "u"——这是一个模式。"the" 是最常见的英语单词——这也是模式。这些模式降低了英语的熵，让语言变得可压缩、可预测。

### Shannon 的人肉实验

1951 年，Shannon 做了一个非常巧妙的实验。

他给实验对象看一段英文，然后遮住下一个字母，让他们猜：

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(33,150,243,0.06); border: 1px solid rgba(33,150,243,0.2);">

已知：**T H E R E &ensp; I S &ensp; N O &ensp; R E V E R ___**

请猜下一个字母。

大多数人的第一反应：**S**。

正确。"REVERS" 后面几乎一定是 "E"（completing "REVERSE"）。

</div>

Shannon 记录每个人猜对所需的次数——第一次就猜对，还是试了三四次才中。通过这些数据，他推算出了英语的熵：

**英语 ≈ 1.0 − 1.5 bit/字符**

对比一下：如果 26 个字母加空格完全随机出现，熵 = log₂(27) ≈ 4.75 bit/字符。

也就是说，**英语比随机乱码可以被压缩 3 到 4 倍**——因为语言中充满了规律和冗余。

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(255,152,0,0.06); border: 1px solid rgba(255,152,0,0.2);">

**等一下——这个实验是不是很眼熟？**

让一个人根据前文，预测下一个字符。

这不就是 GPT 在做的事情吗？

Shannon 在 1951 年用**人类**当语言模型。今天我们用 **Transformer** 做同样的事。

区别只是：GPT 的"实验对象"读过万亿 token 的文本，上下文窗口有百万 token。而人类只有短期记忆的 7±2 个项目。

但**任务完全一样**：根据已知的上文，预测下一个符号。

</div>

Shannon 甚至在论文中展示了不同"阶"的英语近似——看看它们像不像你调低 GPT 温度参数后的输出：

| 阶 | 生成方式 | 示例 |
|---|---------|------|
| **0 阶** | 完全随机 | XFOML RXKHRJFFJUJ ZLPWCF |
| **1 阶** | 字母频率 | OCRO HLI RGWR NMIELWIS |
| **2 阶** | 二元组统计 | ON IE ANTSOUTINYS ARE T |
| **3 阶** | 三元组统计 | IN NO IST LAT WHEY CRATI |
| **词级 2 阶** | 词频二元组 | THE HEAD AND IN FRONTAL ATTACK ON AN ENGLISH WRITER |

从上到下，每一阶都捕获了更多的语言结构——从噪音逐渐变成像样的英语。这恰恰是从 n-gram 模型到神经网络语言模型的演进路线。Shannon 在 1951 年一张表里，预演了 NLP 七十年的进化。

---

## 三、暗线浮现——从熵到 GPT

### 交叉熵：GPT 在优化什么？

当你训练一个 GPT 模型，最核心的那一行代码是什么？

```python
loss = cross_entropy(model_output, target)
```

**交叉熵**（Cross-Entropy）。一个信息论概念。

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; border: 2px solid #E91E63; background: rgba(233,30,99,0.04);">

真实分布 P 和模型分布 Q 之间的交叉熵：

**H(P, Q) = −∑ P(x) log Q(x)**

在语言模型里：
- **P** = 训练数据（下一个 token 的正确答案，概率 = 1）
- **Q** = 模型预测的概率分布

对于单个 token，简化为：

**Loss = −log Q(正确答案)**

</div>

直觉很直接：如果模型给正确答案分配了很高的概率（Q = 0.9），loss 就很小（0.15）。如果模型把正确答案的概率压得很低（Q = 0.01），loss 就很大（6.64）。

训练就是反复调整模型参数，让它给正确答案更高的概率——让 loss 降下来。

但为什么偏偏是**交叉熵**？数学上有无数种衡量"模型猜得准不准"的方法。为什么 Shannon 七十多年前定义的这个指标，成了深度学习的默认选择？

因为交叉熵有一个美妙的性质。它可以分解为：

**H(P, Q) = H(P) + D_KL(P ‖ Q)**

- **H(P)** 是数据本身的真实熵——这是一个常数，和你的模型无关
- **D_KL(P ‖ Q)** 是 KL 散度——衡量模型和现实之间的"差距"，永远 ≥ 0

所以**最小化交叉熵 = 最小化模型和现实的差距**。当 KL 散度降到 0 时，模型完美地拟合了数据——交叉熵就等于真实熵，不可能再低了。

还记得 Morse 吗？他让常见字母用短编码。GPT 训练时做的事情本质上一样：**让模型给更可能出现的 token 分配更高的概率**。

Morse 在压缩电报。GPT 在压缩语言。中间隔了 186 年，用的是同一个原理。

### 困惑度：用惊讶衡量模型好坏

研究者用一个叫**困惑度**（Perplexity）的指标来衡量语言模型的质量。它其实就是熵换了一种写法：

**PPL = 2^{H(P,Q)}**

直觉：模型在每一步平均面临多少个"等可能的选择"。

- PPL = 1：模型对每个 token 都完全确定（不可能的完美）
- PPL = 10：每步在约 10 个 token 之间犹豫
- PPL = 50000：在整个词表里瞎猜（什么都没学到）

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(33,150,243,0.06); border: 1px solid rgba(33,150,243,0.2);">

| 模型 | 困惑度 | 年代 |
|------|-------|------|
| N-gram 基线 | ~140 | 1990s |
| LSTM | ~60 | 2016 |
| Transformer | ~35 | 2018 |
| GPT-2 | ~18 | 2019 |
| 现代 LLM | <10 | 2024+ |

从 140 到 10——模型对语言的"惊讶程度"大幅下降。

或者用我们的话说：模型的**压缩能力**越来越逼近语言的真实熵。

</div>

### KL 散度：串联蒸馏、对齐和一切

如果你读过上一篇 [「知识蒸馏」](/ai-blog/posts/knowledge-distillation/)，你已经见过 KL 散度了——蒸馏的核心损失函数就是它。

**D_KL(P ‖ Q) = ∑ P(x) log(P(x) / Q(x))**

KL 散度衡量的是：**如果现实是 P，而你却用 Q 来编码——你会多浪费多少 bit？**

它不是一个真正的"距离"（因为不对称：D_KL(P‖Q) ≠ D_KL(Q‖P)），但它衡量的是两个分布之间的"信息差距"。

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(156,39,176,0.06); border-left: 4px solid #9C27B0;">

**KL 散度在 LLM 中无处不在：**

| 场景 | KL 散度怎么用 |
|------|-------------|
| **训练** | 最小化 H(P,Q) = 最小化 D_KL + 常数 |
| **知识蒸馏** | 让学生模型的分布逼近老师：D_KL(Teacher ‖ Student) |
| **RLHF** | 追求奖励，但不能偏离原始模型太远：Reward − β·D_KL(π_θ ‖ π_ref) |
| **DPO** | KL 约束直接内置在优化目标里 |

上一篇里我们讲蒸馏时说：学生学的是"老师的犹豫"。现在你知道这个犹豫的数学名字了——**它就是老师输出的概率分布**，学生通过最小化 KL 散度来模仿它。

</div>

### 温度——名字的由来

上一篇我们还提到了一个参数：**温度**（Temperature）。

$$\text{softmax}(z_i / T)$$

为什么叫"温度"？因为这个公式和物理学中的**玻尔兹曼分布**是同一个数学结构：

| 统计力学 | LLM |
|---------|-----|
| 能量 Eᵢ | 负 logit −zᵢ |
| 温度 T | 温度 T |
| 低温 → 粒子冻结在最低能态 | 低温 → 贪心选概率最高的 token |
| 高温 → 粒子在所有能态间均匀分布 | 高温 → 在所有 token 间近似均匀采样 |

Shannon 的信息熵和热力学的热力学熵，名字相同绝非巧合——它们是同一个数学对象在不同领域的投影。

温度直接控制的是**输出分布的熵**。T 低，分布尖锐，熵低，输出确定；T 高，分布平坦，熵高，输出随机。

---

## 四、压缩即智能——从直觉到定理

### 预测 = 压缩

这是整篇文章最核心的一节。也是博客开篇语的理论根基。

Shannon 的信源编码定理告诉我们，给定一个概率模型，可以用算术编码（Arithmetic Coding）把数据压缩到：

**L = −∑ log₂ P(xₙ | x₁, ..., xₙ₋₁) bit**

这个公式——**和交叉熵损失完全一样**。

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; border: 2px solid #E91E63; background: rgba(233,30,99,0.04);">

**一个语言模型就是一个压缩器。它的交叉熵损失就是它的压缩率。**

更好的预测
⟺ 更低的交叉熵
⟺ 更好的压缩
⟺ 对数据结构更完整的捕获

这不是类比。这是数学等价。

</div>

2024 年，DeepMind 发表了一篇论文，标题直截了当：*Language Modeling Is Compression*。他们用 Chinchilla 70B 模型做通用压缩器：

- 压缩文本——超过 gzip 和 bz2
- 压缩图片——超过 PNG
- 压缩音频——超过 FLAC

一个在文本上训练的语言模型，居然比专门设计的图片压缩算法和音频压缩算法还会压缩。

这不是魔法。这是因为**压缩的本质就是发现模式**。如果一个模型强大到能发现数据中的统计规律——无论这些数据是文字、像素还是声波——它就能把这些规律用来压缩数据。

### 回到开篇语

[开篇语](/ai-blog/posts/opening-essay/) 里写的"压缩即智能"——现在可以被精确表述了：

Shannon 的信源编码定理证明：

**最优压缩 = 最优预测 = 对数据中所有规律的完整提取**

Marcus Hutter（AIXI 理论的提出者，通用人工智能理论的奠基者之一）设立了 Hutter Prize——悬赏压缩 1GB 的维基百科。他的理由很直接：

> "对文本的压缩等价于对文本的理解。"

每一代获奖者都是通过构建更好的预测模型来实现更好的压缩的。

Ilya Sutskever（OpenAI 联合创始人）说过一句广为流传的话：

> "如果你把数据压缩得足够好，你就必然提取了其中的所有知识。"

这句话现在可以被精确化了：一个模型的交叉熵损失越低，它从数据中提取的统计结构就越完整。Shannon 在 1948 年就写下了这个等式。我们花了 76 年，才用万亿参数的神经网络去逼近它。

### Tokenization 就是压缩

一个你可能没注意到的事实：GPT 用的分词算法 BPE（Byte Pair Encoding），最初是 1994 年作为**数据压缩算法**被提出的。

BPE 的过程很简单：

1. 从单个字符开始
2. 找到最频繁的相邻字符对，合并成一个新 token
3. 重复，直到词表达到目标大小

如果你读过 [第一篇「结绳记事」](/ai-blog/posts/see-math-1-counting/)，你可能还记得我们在那篇里用 Python 跑了 OpenAI 的 tokenizer，看中文和英文是怎么被切成 token 的。当时我们关注的是"一一对应"这个思想。现在我们知道了更深一层的东西：

tokenizer 不只是在"切词"。它在做**压缩**——用 Morse 的思路：**常见的模式，用更短的编码**。

所以 GPT 的整个工作流程，本质上是一个**两级压缩系统**：

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(76,175,80,0.06); border-left: 4px solid #4CAF50;">

**第一级（BPE tokenization）**：字典压缩——把频繁出现的字符组合替换成更短的 token

**第二级（Transformer 预测）**：统计压缩——给更可能出现的下一个 token 分配更高的概率

从头到尾，从 tokenizer 到 loss function，整个 LLM 训练流程都是信息论。

</div>

---

## 五、一条暗线——从 1838 到 2025

把时间拉远，你会看到一条异常清晰的线。

{{< sandbox src="timeline.html" title="⏳ 从电报到 GPT：信息论的 187 年暗线" height="850" >}}

从 Morse 的电报码到 GPT 的 softmax，核心原理从未改变：

1. 建模数据的统计规律
2. 给更可能的模式分配更短的表示
3. 这**同时是**最优通信，**也是**最优理解

Morse 1838 年凭直觉做到了第 2 条——E 用最短的编码。Shannon 1948 年用数学证明了为什么。Transformer 2017 年用注意力机制和交叉熵损失，在万亿数据上实践了它。

最有趣的对比是 Shannon 的 1951 年实验和今天的 GPT：

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(33,150,243,0.06); border: 1px solid rgba(33,150,243,0.2);">

| | **Shannon 的实验（1951）** | **现代 LLM** |
|---|---|---|
| 预测者 | 人类 | Transformer |
| 输入 | 前面的字符 | 前面的 token |
| 输出 | 排序的猜测 | 完整概率分布 |
| 上下文窗口 | 工作记忆（7±2 项） | 4K ~ 1M+ token |
| 训练数据 | 一生的阅读 | 万亿 token |
| 测量什么 | 英语的熵 | 交叉熵损失 |
| 本质 | 人类做语言模型 | 神经网络做语言模型 |

**任务完全相同：根据上文，预测下一个符号。**

</div>

Shannon 用人做了这件事。我们用硅片做同样的事。但底层的数学——熵、交叉熵、概率分布——完全一样。

---

## 六、回望——那根贯穿所有篇章的弦

回头看「看见数学」的十六篇，信息论是那条一直在暗处运行的线索。

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(156,39,176,0.06); border-left: 4px solid #9C27B0;">

| 篇章 | 表面的主题 | 信息论的暗线 |
|------|----------|------------|
| [第 1 篇：结绳记事](/ai-blog/posts/see-math-1-counting/) | 一一对应，抽象 | 最早的编码方式 |
| [第 7 篇：指数](/ai-blog/posts/see-math-7-exponential/) | 指数增长 | 信息量 = **log**(1/P)——对数是信息的度量 |
| [第 8 篇：三角函数](/ai-blog/posts/see-math-8-waves/) | 圆与波 | 傅里叶变换 = 信号的频率分解 = 一种压缩 |
| [第 13 篇：概率](/ai-blog/posts/see-math-13-probability/) | 拥抱不确定 | 概率是信息论的基础语言 |
| [第 14 篇：高维](/ai-blog/posts/see-math-14-high-dimensions/) | 超越想象力 | 高维空间中的编码效率 |
| [第 15 篇：梯度下降](/ai-blog/posts/see-math-15-gradient/) | 数学会学习 | 训练 = 最小化交叉熵 = 最大化压缩效率 |
| [第 16 篇：终章](/ai-blog/posts/see-math-16-finale/) | 数学是望远镜 | 信息论是望远镜的光学理论 |

</div>

如果数学是人类的望远镜，信息论就是这架望远镜的光学原理。

它不告诉你看到什么。它告诉你：任何望远镜都有分辨率的极限。这个极限就是**熵**。你能做的最好的事，就是无限逼近这个极限——用更大的镜片（更多的参数），更精确的磨制（更好的训练方法），更长的曝光时间（更多的数据）。

Shannon 在 1948 年画出了极限的形状。七十六年来，从 n-gram 到 LSTM 到 Transformer，我们一直在向那个极限靠近。

### 最后

Shannon 1948 年的论文，标题叫 *A Mathematical Theory of **Communication***。

Communication。沟通。

七十六年后，我们用他的理论造出了人类历史上最强大的"沟通"工具——大语言模型。它能用几十种语言和你对话，能帮你写代码写文章，能回答你几乎所有的问题。

但 Shannon 的理论也留下了一个它自己无法回答的问题。

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; border: 2px solid #E91E63; background: rgba(233,30,99,0.04);">

当一个系统完美地压缩了人类的语言——它理解了语言吗？

Shannon 会说：从信息论的角度，"压缩"和"理解"是同一个操作。预测就是压缩，压缩就是建模，建模就是提取所有的结构和规律。

但我们还是会追问：一个完美的压缩器，和一个真正理解语言的心灵——它们之间，差的是什么？

也许差的不是数学。也许差的是某种数学尚未发明出来的东西。

或者——也许什么都不差。

</div>

Shannon 的论文最后一页，致谢了他在贝尔实验室的同事们。他感谢他们提供了"有益的建议和批评"。

七十六年后的今天，一个被训练在万亿 token 上的 Transformer 模型，在逼近他论文里推导出的理论极限。

他大概不会觉得意外。毕竟他在 1951 年就用人做了同样的实验——只不过，那时候的"语言模型"是坐在实验室里的人类志愿者，上下文窗口只有他们的短期记忆，训练数据是他们活到那时为止读过的所有文字。

本质上，没有任何区别。

---

<div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #e0e0e0; font-size: 0.9em; color: #888; line-height: 1.8;">

**📚 引用与延伸阅读**

- Shannon, C.E. (1948). *A Mathematical Theory of Communication*. Bell System Technical Journal.
- Shannon, C.E. (1951). *Prediction and Entropy of Printed English*. Bell System Technical Journal.
- Hinton, G. et al. (2015). *Distilling the Knowledge in a Neural Network*. arXiv:1503.02531.
- Delétang, G. et al. (2024). *Language Modeling Is Compression*. ICLR 2024.
- Hutter Prize: [prize.hutter1.net](https://prize.hutter1.net/)

**💡 相关文章**

- [压缩即智能——开篇语](/ai-blog/posts/opening-essay/)
- [看见数学（十三）：概率——拥抱不确定](/ai-blog/posts/see-math-13-probability/)
- [看见数学（十五）：梯度下降——数学会学习](/ai-blog/posts/see-math-15-gradient/)
- [看见数学（十六）：终章——数学是人类的望远镜](/ai-blog/posts/see-math-16-finale/)
- [当模型学会「偷师」——知识蒸馏、版权战争与学习的边界](/ai-blog/posts/knowledge-distillation/)

《看见数学》系列 — 从结绳记事到 AI，看见数学之美。

本文首发于「AI 学习笔记」博客：https://Jason-Azure.github.io/ai-blog/

微信公众号：AI-lab学习笔记

系列文章完整列表见 [标签：看见数学](/ai-blog/tags/看见数学/)
</div>
