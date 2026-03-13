---
title: "中文 vs 英文：大语言模型的语言鸿沟与技术突围"
date: 2026-03-03
draft: false
summary: "为什么 GPT-3 的中文只占训练数据的 0.1%？为什么同样一句话中文要花 13 倍的 Token？国内大模型是怎么用 15 万词表逆转这个劣势的？从训练语料、Tokenizer 到 Embedding，一篇讲透中英文 LLM 的底层差异。"
categories: ["LLM"]
tags: ["Tokenization", "Embedding", "中文NLP", "nanoGPT", "DeepSeek", "Qwen", "多语言"]
weight: 17
ShowToc: true
TocOpen: true
---

## 引言

你有没有好奇过：ChatGPT 回答英文问题总是又快又好，但中文有时候会"翻译腔"？国内的 DeepSeek、Qwen 在中文上明显更流畅，它们做了什么不同的事？

答案藏在三个层面：**训练数据的语言比例**、**Tokenizer 的设计**、**Embedding 的效率**。

这不是一个简单的"中文语料多一点就行"的问题。从 26 个英文字母到数千个汉字，从 50K 词表到 150K 词表，每一步都涉及根本性的技术权衡。今天我们把这条链路彻底拆开。

> **本文脉络：** 先看数据（谁用了多少中文语料），再看 Tokenizer（同一句话被拆成多少 Token），最后看 Embedding（大词表对模型参数的影响）。三层叠加，才是中英文 LLM 差异的完整图景。

---

## 一、多语言能力从哪来？原生语料，不是翻译

一个常见的误解：GPT-4 能回答中文，是因为它学了大量的英汉翻译对照。

**事实恰好相反。** 所有主流 LLM 的多语言能力都来自**直接用目标语言的原生语料训练**，而非翻译。

为什么不能靠翻译？

- **翻译腔**：翻译文本带有源语言的思维模式（"这是一个...的情况"），模型学到的不是地道中文
- **知识缺失**：大量中文特有知识不存在英文原文——《西游记》原文、中国法律法规、网络热梗
- **规模瓶颈**：高质量翻译语料的量级远不及原生网页语料

但这里有一个有趣的现象：**跨语言迁移（Cross-lingual Transfer）**。

> 模型在大量英文语料上学到的推理能力、逻辑结构、世界知识，会部分"迁移"到其他语言上。多语言 Embedding 空间中，不同语言的相似概念会被映射到**相近的向量位置**。

这也是为什么 GPT-3 虽然只有 0.1% 的中文训练数据，却仍然能回答一些中文问题。不是因为它"学过"足够多的中文，而是因为它在英文中学到的**世界知识结构**可以跨语言复用。

但"能回答"和"回答得好"之间，差距巨大。这就引出了核心问题：**各家模型到底用了多少中文数据？**

---

## 二、训练语料的语言比例：数据决定能力上限

### GPT 系列：英文为王

GPT-3 的论文（Brown et al., 2020）公开了训练数据的语言分布，这是目前唯一有精确数字的主流模型：

| 语言 | 占比 |
|------|------|
| **英文** | **92.65%** |
| 法语 | 1.82% |
| 德语 | 1.47% |
| 西班牙语 | 0.77% |
| 意大利语 | 0.61% |
| 日语 | 0.11% |
| **中文** | **0.10%** |
| 韩语 | 0.02% |
| 其他 110+ 种语言 | ~3.45% |

**中文只占 0.1%**——是英文的 **926 分之一**。

GPT-4 和 GPT-4o 没有公开语言分布数据。但从 GPT-4 在多语言 MMLU 基准测试上的表现看——它在 26 种语言中的 24 种上超过了 GPT-3.5 的英文成绩——推测多语言数据比例有大幅提升，但英文仍然是绝对主导。

### 国内模型：中英双语并重

国内主流大模型的训练策略明显不同——**有意识地大幅提高中文语料比例**：

| 模型 | 开发团队 | 训练数据量 | 中文占比（推测） | 英文占比 |
|------|---------|-----------|----------------|---------|
| **Qwen2.5** | 阿里 | 18 万亿 tokens | ~40-50% | ~40-50% |
| **DeepSeek-V3** | 深度求索 | 14.8 万亿 tokens | ~40%+ | ~50% |
| **GLM-4** | 智谱 AI | ~10 万亿+ tokens | ~50%+ | ~40% |
| **Baichuan 2** | 百川智能 | 2.6 万亿 tokens | ~55% | ~35% |

几个关键观察：

1. **中文普遍占 40-55%**，远超 GPT-3 的 0.1%
2. 都是**中英双语为主**，再加少量其他语言
3. **代码语料**也占相当比例（10-20%），因为代码能提升逻辑推理能力
4. DeepSeek-V2 技术报告明确提到"中文 tokens 比英文多约 12%"——刻意加重中文

> **一句话总结**：GPT-3 是"英文模型顺便支持中文"，国内模型是"中英双语原生模型"。这是设计理念的根本差异。

---

## 三、Tokenizer：同一句话，Token 数量差 13 倍

训练语料的比例决定了模型"见过多少中文"，但还有一个更底层的问题：**模型怎么切分中文文本？**

这就是 Tokenizer（分词器）的战场。

### 核心问题：一个汉字 = 几个 Token？

用不同的 Tokenizer 处理同一句中文 **"人工智能是未来的发展方向"**（12 个汉字），结果天差地别：

| Tokenizer | 词表大小 | Token 数 | 切分方式 |
|-----------|---------|---------|---------|
| **GPT-2** | 50,257 | **22** | 每个汉字拆成 2-3 个字节 token |
| **GPT-4** (cl100k) | 100,277 | **13** | 大部分汉字单独编码 |
| **GPT-4o** (o200k) | 200,019 | **6** | 常见词组合并（"人工""智能""未来"） |
| **Qwen2** | 151,643 | **5** | "人工智能"整词编码 |
| **DeepSeek-V2** | 100,000 | **5** | "人工智能"整词编码 |
| **GLM-4** | 151,329 | **6** | "人工智能""发展方向"整词编码 |

GPT-2 对同样一句中文要花 **22 个 Token**，而 Qwen2 只需要 **5 个**。

### Token 效率的全面对比

为了更直观地理解差异，我们用"每个汉字消耗多少 Token"来衡量：

| Tokenizer | 每汉字 Token 数 | 每英文字母 Token 数 | 中/英成本比 |
|-----------|----------------|-------------------|-----------|
| **GPT-2** | 2.11 | 0.16 | **13.3x** |
| **GPT-4** (cl100k) | 1.12 | 0.16 | **7.1x** |
| **GPT-4o** (o200k) | 0.56 | 0.15 | **3.7x** |
| **Qwen2** | 0.49 | 0.16 | **3.1x** |
| **DeepSeek-V2** | 0.46 | 0.16 | **2.9x** |
| **GLM-4** | 0.62 | 0.21 | **3.0x** |

<div style="max-width: 640px; margin: 1.5em auto; font-size: 0.95em;">
<div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: center; gap: 6px 4px; line-height: 1.8;">
<span style="border: 2px solid #f44336; border-radius: 6px; padding: 4px 10px; background: rgba(244,67,54,0.08);"><strong>GPT-2</strong><br><span style="font-size:0.8em;color:#888;">13.3x 成本</span></span>
<span style="color:#aaa;">→</span>
<span style="border: 2px solid #FF9800; border-radius: 6px; padding: 4px 10px; background: rgba(255,152,0,0.06);"><strong>GPT-4</strong><br><span style="font-size:0.8em;color:#888;">7.1x 成本</span></span>
<span style="color:#aaa;">→</span>
<span style="border: 2px solid #FF9800; border-radius: 6px; padding: 4px 10px; background: rgba(255,152,0,0.06);"><strong>GPT-4o</strong><br><span style="font-size:0.8em;color:#888;">3.7x 成本</span></span>
<span style="color:#aaa;">→</span>
<span style="border: 2px solid #4CAF50; border-radius: 6px; padding: 4px 10px; background: rgba(76,175,80,0.08);"><strong>Qwen / DeepSeek</strong><br><span style="font-size:0.8em;color:#888;">~3x 成本</span></span>
</div>
<p style="text-align: center; font-size: 0.85em; color: #888; margin-top: 8px;">中文 Token 效率演进：从 13 倍差距到 3 倍</p>
</div>

两个关键结论：

1. **中文在任何 Tokenizer 下都比英文"贵"**——这是 UTF-8 编码的物理限制（一个汉字 = 3 字节 vs 一个英文字母 = 1 字节）
2. 但从 GPT-2 到 Qwen2，**中文的 Token 效率提升了 4.3 倍**——这是 Tokenizer 设计的胜利

### 为什么 GPT-2 对中文这么"浪费"？

根本原因：**GPT-2 的 Tokenizer 几乎只在英文上训练**。

Tokenizer 的核心算法是 **BPE（Byte Pair Encoding，字节对编码）**。它的工作方式：

```text
第 1 步：从 256 个基础字节开始（覆盖所有可能的字节值）
第 2 步：统计训练语料中最频繁出现的字节对
第 3 步：把最频繁的字节对合并为新 token
第 4 步：重复第 2-3 步，直到达到目标词表大小
```

当训练语料 93% 是英文时：
- "th" 出现频率极高 → 被合并为一个 token
- "the" → 进一步合并
- "ing", "tion", "ment" → 全部有专属 token

但中文在训练语料中只占 0.1%：
- "人"的 UTF-8 字节序列 `[0xe4, 0xba, 0xba]` 出现频率太低
- 远远排不到合并队列前面
- 结果每个汉字被拆成 2-3 个字节 token

> **一个类比**：想象你在整理图书馆，你有 50,000 个标签。如果 93% 的书是英文，你自然会给英文书创建精细分类（"科幻/赛博朋克"、"科幻/太空歌剧"）。中文书那一小架子？全贴一个"外文"标签就完了。

---

## 四、国内大模型的技术突围

### 解法 1：扩大词表 + 中文语料重新训练 BPE

这是最核心的改进。对比各家词表规模：

```text
GPT-2:       50,257 tokens   → 中文 1 字 ≈ 2-3 tokens
LLaMA 1/2:   32,000 tokens   → 中文 1 字 ≈ 2-3 tokens（更小词表，中文更惨）
LLaMA 3:    128,256 tokens   → 中文有改善，但中文不在官方支持语言中
GPT-4o:     200,019 tokens   → 中文 1 字 ≈ 0.5 tokens ✓
Qwen2.5:    151,643 tokens   → 中文 1 字 ≈ 0.5 tokens ✓
DeepSeek-V3: 128,000 tokens  → 中文 1 字 ≈ 0.5 tokens ✓
GLM-4:      151,329 tokens   → 中文 1 字 ≈ 0.6 tokens ✓
```

**做法**：在包含大量中文的语料上**重新训练 BPE**，让高频中文词汇（"我们"、"因为"、"人工智能"）被合并为单个 token。

词表从 50K 扩到 150K，增加的那 10 万个位置，主要就是给中文（和其他非英语语言）的常见字词留的。

### 解法 2：字节级 BPE (BBPE) + 智能预分词

现代中文优化 Tokenizer 的标准架构：

```text
原始文本
   ↓
预分词 (Pre-tokenization)
   ├── 中文：按字/词边界切分
   ├── 英文：按空格 + 标点切分
   └── 代码：按语法符号切分
   ↓
字节级 BPE (BBPE)
   ↓
Token ID 序列
```

**Qwen2 的方案**比较典型：
- 基于 tiktoken 框架（与 OpenAI 相同的底层库）
- 151,643 个 token 的大词表
- 中文常用字基本都有独立 token
- 高频中文词组（"学习"、"可以"、"人工智能"）合并为单 token
- 罕见字回退到字节序列，保证 **零 OOV**（不会遇到未知字符）

**GLM-4** 的方案更有意思——它把中文词表和 OpenAI 的 cl100k_base 词表**合并**：
- 先单独在中文和多语言语料上学习 token
- 再与 OpenAI 的英文词表合并
- 最终 151,329 个 token，兼顾中英文效率

### 解法 3：词表扩展方案（Chinese-LLaMA 的路线）

如果你已经有一个英文为主的基座模型（比如 LLaMA），又想增加中文能力怎么办？

崔一鸣团队（2023）提出的 Chinese-LLaMA 方案：

1. 在 LLaMA 的 32K 词表基础上，**额外添加 20,000 个中文 token**（总共约 50K）
2. 用 120GB 中文文本做**二次预训练**
3. 再用百万级中文指令数据做微调

效果：中文编码效率从 ~2.5 tokens/字 降到 ~1.0 tokens/字，在 C-Eval 基准上达到"数倍参数量模型"的水平。

> 但这种"后补"方案终究不如从头训练的原生双语模型。中文 token 的 Embedding 没有经过充分预训练，深层语义理解有差距。

---

## 五、Embedding 层：大词表的代价

词表变大，解决了 Token 效率问题，但带来新的挑战：**Embedding 矩阵膨胀**。

### 参数量对比

Embedding 层的参数量 = 词表大小 × 嵌入维度：

| 模型 | 词表大小 | 嵌入维度 | Embedding 参数量 | 占总参数比例 |
|------|---------|---------|-----------------|------------|
| GPT-2 (117M) | 50,257 | 768 | 38.6M | ~33% |
| nanoGPT 西游记 | ~4,000 | 256 | 1.0M | ~30% |
| Qwen2-7B | 151,643 | 4,096 | 621M | ~8% |
| DeepSeek-V3 (671B) | 128,000 | 7,168 | 917M | ~0.1% |

两个有趣的规律：

1. **小模型中，Embedding 是参数大户**——nanoGPT 训练西游记时，仅 Embedding 就占了总参数的 30%。这也是我们在 VM 上亲身体验到的：西游记模型比 Shakespeare 模型更难训练，因为 4000+ 的词表导致 Embedding 层参数暴增
2. **大模型中，Embedding 占比反而很小**——Qwen2-7B 的 621M Embedding 参数只占总体 8%，到 DeepSeek-V3 的 671B 总参数，Embedding 只占 0.1%

### 工程优化手段

为了控制大词表带来的开销，工业界有几种常用策略：

**Weight Tying（权重共享）**：输入 Embedding 矩阵和输出 lm_head（最后一层的分类矩阵）**共享同一组权重**。原本需要两个 [vocab_size × hidden_dim] 的矩阵，现在只需要一个。这是最常用的优化，GPT-2、Qwen2 都使用了。

**实际计算开销可控**：Embedding 层的"计算"其实只是查表（lookup），不涉及矩阵乘法。一个 Token 进来，直接取对应行的向量就走了。真正的计算瓶颈在 Transformer 的 Attention 和 MLP 层。

> **所以大词表的代价主要是显存，不是算力。** 这也是为什么 150K 词表在现代 GPU 上完全可行——多出来的 Embedding 参数只多占几百 MB 显存，但换来的中文编码效率提升是 4 倍以上。

---

## 六、跨语言迁移：为什么 GPT-3 只有 0.1% 中文数据却能说中文？

这是一个令人着迷的问题。GPT-3 的中文训练数据只有 0.1%，但它确实能回答中文问题——虽然质量远不如英文。这怎么做到的？

### 三个迁移机制

**1. 共享字节表示**

即使中英文在语义层面完全不同，在字节层面有少量重叠。数字（"2024"）、标点、英文借词在中英文本中都出现，这些共享 token 就像两种语言之间的"桥梁"，帮助模型建立跨语言的关联。

**2. 锚点文本**

训练数据中存在少量"中英文混合"的内容——维基百科的跨语言链接、学术论文的中英对照、代码注释。这些文本虽然量小，但为模型提供了关键的**对齐信号**，让它知道"人工智能"和"artificial intelligence"指向同一个概念。

**3. 结构普遍性**

Transformer 的 Attention 机制学到的某些模式是跨语言的：
- 主语-谓语的依赖关系
- 否定词对语义的反转
- 列举结构的并列关系

这些语法结构在不同语言中有相似性，模型学会的英文语法模式可以部分迁移到中文。

### 但迁移有上限

GPT-4 的多语言 MMLU 测试显示，它在 24/26 种语言上超过了 GPT-3.5 的英文成绩。这说明当模型规模足够大时，跨语言迁移能力确实很强。

然而，**原生双语训练的模型在中文任务上仍然显著优于"英文为主 + 跨语言迁移"的模型**。这就是为什么 DeepSeek、Qwen 在中文评测中通常优于同等规模的 GPT 或 LLaMA。

> 跨语言迁移是一个了不起的**涌现现象**，但不能替代充足的原生语料训练。

---

## 七、总览：从 nanoGPT 到 GPT-4o 的完整图景

让我们把所有模型放在一张表里，看看中英文 LLM 的全貌：

| 模型 | 词表大小 | 训练数据量 | 中文占比 | 每汉字 Token 数 | Tokenizer 类型 |
|------|---------|-----------|---------|----------------|---------------|
| nanoGPT Shakespeare | 65 | ~1M 字符 | 0% | - | 字符级 |
| nanoGPT 西游记 | ~4,000 | ~2M 字符 | 100% | 1.0 | 字符级 |
| GPT-2 | 50,257 | ~40B tokens | <0.1% | 2.11 | 字节级 BPE |
| GPT-3 | 50,257 | 300B tokens | 0.10% | ~2.1 | 字节级 BPE |
| LLaMA 1/2 | 32,000 | 1-2T tokens | 极少 | ~2.5 | SentencePiece BPE |
| LLaMA 3 | 128,256 | 15T+ tokens | 少量 | ~1.0 | 字节级 BPE |
| GPT-4 | 100,277 | 未公开 | 未公开 | 1.12 | 字节级 BPE |
| GPT-4o | 200,019 | 未公开 | 未公开 | 0.56 | 字节级 BPE |
| Baichuan 2 | 125,696 | 2.6T tokens | ~55% | ~0.50 | SentencePiece BPE |
| DeepSeek-V2 | 100,000 | 8.1T tokens | >50% | 0.46 | BBPE |
| DeepSeek-V3 | 128,000 | 14.8T tokens | ~40%+ | ~0.45 | BBPE |
| Qwen2 | 151,643 | 7-12T tokens | ~45% | 0.49 | BBPE |
| Qwen2.5 | 151,643 | 18T tokens | ~45% | ~0.49 | BBPE |
| GLM-4 | 151,329 | ~10T+ tokens | ~50%+ | 0.62 | BBPE + cl100k 合并 |

从 65 个字符的 Shakespeare 词表到 200,019 个 token 的 GPT-4o 词表，跨越了 **3,077 倍**。这个跨度就是理解中英文 LLM 差异的关键。

---

## 八、动手验证：在我们的 VM 上感受差异

我们的 AI Lab VM 恰好有完整的教学链条来体验这些差异：

### Shakespeare vs 西游记：nanoGPT 的亲身对比

```bash
# 激活虚拟环境
source ~/ai-lab-venv/bin/activate
cd ~/nanoGPT

# Shakespeare 字符级：65 个字符的词表，训练快速收敛
python sample.py --out_dir=out-shakespeare-char

# 西游记字符级：4000+ 个字符的词表，训练明显更慢
python sample.py --out_dir=out-xiyouji
```

你会直观地感受到：
- Shakespeare 模型在几千步就能生成像样的英文
- 西游记模型需要更多训练步数才能生成通顺的中文
- 根本原因：**词表从 65 → 4000+，Embedding 参数暴增 60 倍**

### 用 Ollama 体验生产级中文模型

```bash
# DeepSeek R1 1.5B — 专业中文 Tokenizer + 大规模中文训练
ollama run deepseek-r1:1.5b "用一句话解释什么是 Tokenizer"

# Qwen3 0.6B — 阿里 15 万词表 + 18T tokens 训练
ollama run qwen3:0.6b "用一句话解释什么是 Tokenizer"
```

从 nanoGPT 的 4000 词表到 DeepSeek 的 128,000 词表，这就是"教学模型"到"生产模型"的距离。

---

## 总结

中英文 LLM 的差异不是单一因素造成的，而是**三层叠加**：

<div style="max-width: 640px; margin: 1.5em auto; font-size: 0.95em;">
<div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: center; gap: 6px 4px; line-height: 1.8;">
<span style="border: 2px solid #4CAF50; border-radius: 6px; padding: 4px 10px; background: rgba(76,175,80,0.08);"><strong>训练语料</strong><br><span style="font-size:0.8em;color:#888;">0.1% → 50%</span></span>
<span style="color:#aaa;">×</span>
<span style="border: 2px solid #2196F3; border-radius: 6px; padding: 4px 10px; background: rgba(33,150,243,0.06);"><strong>Tokenizer</strong><br><span style="font-size:0.8em;color:#888;">50K → 150K 词表</span></span>
<span style="color:#aaa;">×</span>
<span style="border: 2px solid #9C27B0; border-radius: 6px; padding: 4px 10px; background: rgba(156,39,176,0.06);"><strong>Embedding</strong><br><span style="font-size:0.8em;color:#888;">权重共享 + 查表</span></span>
<span style="color:#aaa;">=</span>
<span style="border: 2px solid #FF9800; border-radius: 6px; padding: 4px 10px; background: rgba(255,152,0,0.06);"><strong>中文能力</strong><br><span style="font-size:0.8em;color:#888;">从勉强到原生</span></span>
</div>
</div>

- **语料层**：GPT-3 的 0.1% 中文 vs 国内模型的 40-55% 中文，决定了模型"见过多少中文世界"
- **Tokenizer 层**：从 GPT-2 的 13.3 倍成本到 Qwen2 的 3.1 倍成本，决定了每个 Token 承载多少中文语义
- **Embedding 层**：150K 词表的大 Embedding 矩阵只多占几百 MB 显存，换来 4 倍以上的中文效率，这笔账怎么算都划算

而我们在 VM 上用 nanoGPT 训练 Shakespeare 和西游记的经历，恰好是这个宏大故事的微缩版：**65 个字符 vs 4000 个字符的词表差异，就是英文 LLM 和中文 LLM 技术鸿沟的缩影。**

理解了这三层差异，你就理解了为什么中文世界需要自己的大模型——不只是"数据主权"的叙事，更是**底层技术架构的必然要求**。

---

> **参考文献**
>
> - Brown et al. (2020). *Language Models are Few-Shot Learners* (GPT-3). arXiv: 2005.14165
> - OpenAI (2023). *GPT-4 Technical Report*. arXiv: 2303.08774
> - Yang et al. (2024). *Qwen2 Technical Report*. arXiv: 2407.10671
> - Qwen Team (2024). *Qwen2.5 Technical Report*. arXiv: 2412.15115
> - DeepSeek-AI (2024). *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*. arXiv: 2405.04434
> - DeepSeek-AI (2024). *DeepSeek-V3 Technical Report*. arXiv: 2412.19437
> - GLM Team (2024). *ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools*. arXiv: 2406.12793
> - Yang et al. (2023). *Baichuan 2: Open Large-scale Language Models*. arXiv: 2309.10305
> - Cui et al. (2023). *Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca*. arXiv: 2304.08177
> - Conneau et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale* (XLM-R). arXiv: 1911.02116
