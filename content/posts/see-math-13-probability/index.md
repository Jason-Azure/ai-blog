---
title: '看见数学（十三）：概率——拥抱不确定'
date: 2026-03-20
draft: false
summary: '概率不是"猜"——概率是用数学管理无知。赌徒的信件催生了概率论，贝叶斯牧师教会了 AI 如何"更新信念"，而 GPT 每写一个字，都是在从概率分布里抽样。'
categories: ["看见数学"]
tags: ["数学思维", "概率", "贝叶斯", "softmax", "看见数学"]
weight: 42
ShowToc: true
TocOpen: true
---

> **系列导航**
>
> <div style="max-width: 660px; margin: 0.5em 0; font-size: 0.93em; line-height: 1.9;">
> <div style="border-left: 3px solid #ccc; padding-left: 12px; margin-bottom: 6px; padding: 8px 12px; color: #888;">
> 第一幕（5 篇）+ 第二幕（5 篇）<a href="/ai-blog/tags/看见数学/" style="color: #888;">→ 查看全部</a></div>
> <div style="border-left: 3px solid #ccc; padding-left: 12px; margin-bottom: 6px; padding: 8px 12px; color: #888;">
> ▹ 第十一篇：向量——给万物一个坐标</div>
> <div style="border-left: 3px solid #ccc; padding-left: 12px; margin-bottom: 6px; padding: 8px 12px; color: #888;">
> ▹ 第十二篇：矩阵——空间的变形术</div>
> <div style="border-left: 3px solid #FF9800; padding-left: 12px; margin-bottom: 6px; background: rgba(255,152,0,0.05); padding: 8px 12px; border-radius: 0 4px 4px 0;">
> <strong>▸ 第十三篇（本文）：概率——拥抱不确定</strong></div>
> <div style="border-left: 3px solid #ccc; padding-left: 12px; margin-bottom: 6px; padding: 8px 12px; color: #888;">
> ▹ 第十四篇：高维——超越想象力</div>
> <div style="border-left: 3px solid #ccc; padding-left: 12px; margin-bottom: 6px; padding: 8px 12px; color: #888;">
> ▹ 第十五篇：梯度下降——数学会学习</div>
> <div style="border-left: 3px solid #ccc; padding-left: 12px; padding: 8px 12px; color: #888;">
> ▹ 第十六篇：终章——数学是人类的望远镜</div>
> </div>

---

## 第一章：赌桌上诞生的数学

1654 年，法国贵族梅雷骑士遇到一个问题：

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(255,152,0,0.06); border: 1px solid rgba(255,152,0,0.2);">

<div style="font-weight: bold; margin-bottom: 12px; color: #FF9800; font-size: 1.05em;">赌徒的困惑</div>

```text
两人赌博，约定先赢 3 局者拿走全部赌注。

现在 A 赢了 2 局，B 赢了 1 局。
比赛因故中断。

问：赌注该怎么分？
```

</div>

梅雷骑士写信给数学家帕斯卡。帕斯卡又写信给费马。两个人的通信，催生了**概率论**。

他们的回答：不是看"已经发生了什么"，而是看"接下来可能发生什么"。

```text
A 只差 1 局就赢。B 还差 2 局。

最多再打 2 局：
  情况 1：A 赢 → A 胜（概率 1/2）
  情况 2：B 赢, A 赢 → A 胜（概率 1/4）
  情况 3：B 赢, B 赢 → B 胜（概率 1/4）

A 获胜的概率 = 3/4，B 获胜的概率 = 1/4

所以赌注应该按 3:1 分配。
```

**概率论的出发点不是"预测未来"，而是"在不确定中做出合理的决策"。**

> **一句话记住：** 概率不是给赌徒用的。概率是在"不知道结果"的时候，做出"最不坏的选择"的数学工具。

---

## 第二章：概率的直觉——从频率到信念

概率是什么？有两种理解方式：

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(33,150,243,0.06); border: 1px solid rgba(33,150,243,0.2);">

<div style="font-weight: bold; margin-bottom: 12px; color: #2196F3; font-size: 1.05em;">两种概率观</div>

| 视角 | 含义 | 例子 |
|------|------|------|
| **频率派** | 重复实验无穷次，事件发生的比例 | 抛硬币 10000 次，大约 5000 次正面 → P = 0.5 |
| **贝叶斯派** | 你对一件事的**信念程度** | "我觉得明天下雨的概率是 60%" |

</div>

频率派的概率很"客观"——需要可以重复的实验。但现实中很多事不可重复：明天会下雨吗？这个病人有多大可能康复？这封邮件是垃圾邮件的概率有多大？

贝叶斯派说：**概率是你的信念，而且信念可以更新。**

```text
早上起来，你觉得今天下雨的概率是 30%（基于天气预报）。
走出门，看到乌云密布 → 你更新信念：概率上升到 70%。
又看到邻居带了伞 → 再更新：概率上升到 80%。

每一条新信息，都让你的判断更准确。
```

这种"看到新证据就更新概率"的方法，叫做**贝叶斯更新**。

<div style="max-width: 640px; margin: 1.5em auto; padding: 15px 20px; border-radius: 8px; background: rgba(76,175,80,0.06); border-left: 4px solid #4CAF50;">

**贝叶斯的故事：** 托马斯·贝叶斯（1701-1761）是一位英国长老会牧师。他的论文在他去世后才由朋友发表。这位牧师大概没想到，他的方法会成为 21 世纪 AI 的基石——垃圾邮件过滤、医学诊断、自动驾驶、语言模型，全部建立在贝叶斯思想之上。

</div>

> **一句话记住：** 频率派说"概率是客观事实"，贝叶斯派说"概率是可以更新的信念"。AI 用的是贝叶斯思想——每看到一个新词，就更新对下一个词的"信念"。

---

## 第三章：条件概率——"知道了 A，B 会变吗？"

概率最强大的工具是**条件概率**：在已知某件事发生的条件下，另一件事发生的概率。

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(76,175,80,0.06); border: 1px solid rgba(76,175,80,0.2);">

<div style="font-weight: bold; margin-bottom: 12px; color: #4CAF50; font-size: 1.05em;">条件概率的直觉</div>

```text
一个班 40 个学生。
  戴眼镜的：20 人
  学编程的：10 人
  又戴眼镜又学编程的：8 人

P(学编程) = 10/40 = 25%

但如果你已经知道这个学生戴眼镜呢？
P(学编程 | 戴眼镜) = 8/20 = 40%

"知道他戴眼镜"这个信息，
  让"他学编程的概率"从 25% 提升到了 40%。

这就是条件概率——新信息改变概率。
```

</div>

**条件概率写作 P(B|A)，读作"在 A 发生的条件下，B 发生的概率"。**

这不是什么抽象概念——你每天都在用条件概率：

```text
P(堵车 | 周一早高峰) >> P(堵车 | 周日凌晨)
P(迟到 | 没设闹钟)   >> P(迟到 | 设了三个闹钟)
```

你的大脑时刻在做条件概率的计算，只是没写成公式。

---

## 第四章：连接 AI——GPT 的每个字都是概率

现在来看概率在 AI 里的核心角色。

**GPT 生成文字的过程，就是反复做一件事：给定前面的所有词，预测下一个词的概率分布。**

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; border: 2px solid #E91E63; background: rgba(233,30,99,0.04);">

<div style="font-weight: bold; margin-bottom: 12px; font-size: 1.1em; color: #E91E63; text-align: center;">GPT = 条件概率机器</div>

```text
输入："今天天气真"

GPT 计算：P(下一个词 | "今天天气真") =

  好     → 0.45  (45%)
  不错   → 0.25  (25%)
  差     → 0.10  (10%)
  热     → 0.08  (8%)
  冷     → 0.05  (5%)
  ...其他 → 0.07  (7%)

然后从这个概率分布中"抽样"一个词。
假设抽到了"好"。

接着计算：P(下一个词 | "今天天气真好") =
  ，     → 0.35
  啊     → 0.20
  ！     → 0.15
  ...

如此循环，一个字一个字地生成。
```

</div>

这里有一个你在 [第七篇（指数爆炸）](/ai-blog/posts/see-math-7-exponential/) 里学过的关键工具：**softmax 函数**。

```text
神经网络输出的是"原始分数"（logits）：
  好: 3.2,  不错: 2.1,  差: 0.8,  热: 0.5, ...

softmax 把它们变成概率（加起来等于 1）：
  好: 0.45,  不错: 0.25,  差: 0.10,  热: 0.08, ...

softmax 用的是什么？指数函数 e^x！
→ 分数高的词被指数放大，分数低的词被压缩
→ "赢家通吃"效应
```

还记得第八篇说的 **temperature**（温度）吗？

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(33,150,243,0.06); border: 1px solid rgba(33,150,243,0.2);">

<div style="font-weight: bold; margin-bottom: 12px; color: #2196F3; font-size: 1.05em;">温度控制"创造力"</div>

| 温度 | 效果 | 适合 |
|------|------|------|
| 低 (0.1-0.3) | 概率分布很"尖"，几乎总是选最高分的词 | 翻译、代码——需要准确 |
| 中 (0.7-1.0) | 概率分布适度平坦，有一定随机性 | 聊天——自然但不乱来 |
| 高 (1.5-2.0) | 概率分布很"平"，低概率词也有机会被选中 | 创意写作——需要意外 |

</div>

**Temperature 不是改变"模型的知识"，而是改变"从概率分布中抽样"的方式。** 同一个模型，温度不同，输出完全不同。这就是为什么 ChatGPT 有时候很严谨，有时候很"跳脱"。

> **一句话记住：** GPT 不"知道"下一个字是什么。它计算每个词的条件概率，然后掷骰子。它不是在"思考"，而是在做概率抽样。

---

## 第五章：概率的古老智慧

概率思维在中国文化里有深远的根基。

<div style="max-width: 640px; margin: 1.5em auto; padding: 15px 20px; border-radius: 8px; background: rgba(76,175,80,0.06); border-left: 4px solid #4CAF50;">

**《孙子兵法》的概率思维：** "知彼知己，百战不殆"——不是说一定赢，而是说**赢的概率极高**。"不殆"不是"必胜"，是"不会有危险"。孙子的战略思想本质上就是概率思维：增大胜率，减小败率。

</div>

<div style="max-width: 640px; margin: 1.5em auto; padding: 15px 20px; border-radius: 8px; background: rgba(255,152,0,0.06); border-left: 4px solid #FF9800;">

**诸葛亮的"锦囊妙计"：** 不是未卜先知，而是**穷举可能的情况，提前为每种情况准备对策**——这就是条件概率的思维。"如果敌军从水路来（条件 A），则打开第一个锦囊（策略 B₁）。如果敌军从陆路来（条件 C），则打开第二个锦囊（策略 B₂）。"

</div>

<div style="max-width: 640px; margin: 1.5em auto; padding: 15px 20px; border-radius: 8px; background: rgba(156,39,176,0.06); border-left: 4px solid #9C27B0;">

**"谋事在人，成事在天"：** 这不是宿命论，而是对概率本质的深刻理解——你只能控制概率（谋事在人），不能控制结果（成事在天）。好的决策不是保证好的结果，而是让好的结果**更可能**发生。

</div>

---

## 动手实验

### 实验：模拟 GPT 的"下一个词预测"

```python
import random

# 简化的"语言模型"：给定前文，预测下一个词的概率分布
model = {
    "今天": {"天气": 0.4, "我": 0.3, "是": 0.2, "很": 0.1},
    "今天天气": {"真": 0.5, "不": 0.3, "很": 0.2},
    "今天天气真": {"好": 0.5, "不错": 0.25, "差": 0.1, "热": 0.15},
}

def sample(probs, temperature=1.0):
    """从概率分布中抽样（带温度控制）"""
    import math
    words = list(probs.keys())
    # 应用温度
    logits = [math.log(p) / temperature for p in probs.values()]
    # softmax
    max_l = max(logits)
    exps = [math.exp(l - max_l) for l in logits]
    total = sum(exps)
    adjusted = [e / total for e in exps]
    # 抽样
    r = random.random()
    cumsum = 0
    for word, prob in zip(words, adjusted):
        cumsum += prob
        if r < cumsum:
            return word
    return words[-1]

# 不同温度的生成效果
for temp in [0.3, 1.0, 2.0]:
    print(f"\n温度 = {temp}:")
    for i in range(5):
        text = "今天"
        while text in model:
            next_word = sample(model[text], temperature=temp)
            text += next_word
        print(f"  {text}")
```

---

## 本篇小结

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; border: 2px solid #FF9800; background: rgba(255,152,0,0.04);">

<div style="font-weight: bold; margin-bottom: 12px; font-size: 1.05em;">这篇文章讲了什么？</div>

**一、概率论诞生于赌桌**
- 帕斯卡和费马的通信，不是为了赢钱，而是为了在不确定中做合理决策

**二、频率 vs 贝叶斯**
- 频率派看重复实验的比例，贝叶斯派看可更新的信念
- AI 用的是贝叶斯思想

**三、条件概率 = 新信息改变概率**
- P(B|A)：知道 A，B 的概率就变了
- 你每天都在做条件概率计算

**四、GPT = 条件概率机器**
- 给定前文，计算每个词的概率，然后抽样
- softmax 把分数变成概率，temperature 控制随机性

**五、概率的古老智慧**
- "知彼知己，百战不殆" = 增大胜率
- 好的决策不保证好的结果，只让好的结果更可能

</div>

---

## 下一篇预告

向量可以有 2 维、3 维……但 AI 里的向量动辄 768 维、几千维。

人类能直觉理解的空间最多到 3 维。超过 3 维的空间叫**高维空间**。

高维空间有很多反直觉的性质：几乎所有点都在"表面"上、随机的两个向量几乎总是"差不多垂直"、数据在高维里变得极度稀疏……

这些不是数学家的游戏——它们直接影响 AI 的设计。理解高维，就理解了为什么深度学习需要那么多数据，为什么降维那么重要。

下一篇：**看见数学（十四）：高维——超越想象力**

---

<div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #e0e0e0; font-size: 0.9em; color: #888; line-height: 1.8;">

**《看见数学》系列** — 从结绳记事到 AI，看见数学之美。<br>
本文首发于「AI 学习笔记」博客：https://Jason-Azure.github.io/ai-blog/<br>
微信公众号：AI-lab学习笔记<br>
系列文章完整列表见 [标签：看见数学](/ai-blog/tags/看见数学/)

</div>
