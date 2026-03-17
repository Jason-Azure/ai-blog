---
title: '看见数学（十一）：向量——给万物一个坐标'
date: 2026-03-18
draft: false
summary: '向量不只是"有方向的箭头"。向量是用一组数来描述一个事物的方法——一杯咖啡、一个游戏角色、一个词的含义。而"国王 - 男人 + 女人 = 女王"，是向量最惊艳的表演。第三幕开篇。'
categories: ["看见数学"]
tags: ["数学思维", "向量", "点积", "余弦相似度", "词嵌入", "看见数学"]
weight: 40
ShowToc: true
TocOpen: true
---

> 前两幕，我们学会了描述静止的世界（数、方程、坐标），也学会了描述变化的世界（函数、指数、波、微积分）。
>
> **第三幕的主题是：看不见的世界。**
>
> 高维空间、概率分布、梯度场——这些东西你摸不到、看不见，但 AI 恰恰就在这些"看不见的世界"里运行。
>
> 数学帮你看见它们。

> **系列导航**
>
> <div style="max-width: 660px; margin: 0.5em 0; font-size: 0.93em; line-height: 1.9;">
> <div style="border-left: 3px solid #ccc; padding-left: 12px; margin-bottom: 6px; padding: 8px 12px; color: #888;">
> 第一幕（5 篇）+ 第二幕（5 篇）<a href="/ai-blog/tags/看见数学/" style="color: #888;">→ 查看全部</a></div>
> <div style="border-left: 3px solid #FF9800; padding-left: 12px; margin-bottom: 6px; background: rgba(255,152,0,0.05); padding: 8px 12px; border-radius: 0 4px 4px 0;">
> <strong>▸ 第十一篇（本文）：向量——给万物一个坐标 【第三幕开篇】</strong></div>
> <div style="border-left: 3px solid #ccc; padding-left: 12px; margin-bottom: 6px; padding: 8px 12px; color: #888;">
> ▹ 第十二篇：矩阵——空间的变形术</div>
> <div style="border-left: 3px solid #ccc; padding-left: 12px; margin-bottom: 6px; padding: 8px 12px; color: #888;">
> ▹ 第十三篇：概率——拥抱不确定</div>
> <div style="border-left: 3px solid #ccc; padding-left: 12px; margin-bottom: 6px; padding: 8px 12px; color: #888;">
> ▹ 第十四篇：高维——超越想象力</div>
> <div style="border-left: 3px solid #ccc; padding-left: 12px; margin-bottom: 6px; padding: 8px 12px; color: #888;">
> ▹ 第十五篇：梯度下降——数学会学习</div>
> <div style="border-left: 3px solid #ccc; padding-left: 12px; padding: 8px 12px; color: #888;">
> ▹ 第十六篇：终章——数学是人类的望远镜</div>
> </div>

---

## 第一章：一杯咖啡有多少"维度"？

你走进一家咖啡店。

服务员问："您想喝什么样的咖啡？"

你说："温度高一点，甜度低一点，浓度大一点。"

不知不觉间，你用了**三个数**来描述一杯咖啡：

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(255,152,0,0.06); border: 1px solid rgba(255,152,0,0.2);">

<div style="font-weight: bold; margin-bottom: 12px; color: #FF9800; font-size: 1.05em;">一杯咖啡的"坐标"</div>

```text
你的咖啡 = [温度: 85°C, 甜度: 2/10, 浓度: 8/10]

用数学的写法：
  咖啡A = [85, 2, 8]

这三个数组成的列表，就是一个向量。
```

</div>

在 [第四篇（坐标革命）](/ai-blog/posts/see-math-4-coordinates/) 里，我们学过：两个数 (x, y) 可以确定平面上的一个点。三个数 (x, y, z) 可以确定空间中的一个点。

现在我们扩展这个思想：**任意多个数，可以描述任意复杂的事物。**

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(33,150,243,0.06); border: 1px solid rgba(33,150,243,0.2);">

<div style="font-weight: bold; margin-bottom: 12px; color: #2196F3; font-size: 1.05em;">万物皆向量</div>

| 事物 | 用什么数描述 | 维度 |
|------|------------|------|
| 一杯咖啡 | [温度, 甜度, 浓度] | 3 |
| 一个游戏角色 | [攻击, 防御, 速度, HP, MP] | 5 |
| 一张脸 | [眼距, 鼻长, 脸宽, 肤色, ...] | 几十 |
| 一首歌 | [节奏, 旋律性, 能量, 情绪, ...] | 几十 |
| 一个词的含义 | [维度₁, 维度₂, ..., 维度₇₆₈] | 768 |
| 一张图片 | [像素₁, 像素₂, ...] | 数百万 |

</div>

**向量 = 一组有序的数。** 就这么简单。

它可以是 2 个数（地图坐标），可以是 5 个数（游戏角色属性），也可以是 768 个数（词向量）。个数就是"维度"。

> **一句话记住：** 向量不是一个高深的概念。向量就是"一串数字"——用来描述一个事物的多个方面。你每天都在和向量打交道，只是没人告诉你它叫向量。

---

## 第二章：向量的加减法——直觉篇

向量最直觉的操作是**加法**。

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(76,175,80,0.06); border: 1px solid rgba(76,175,80,0.2);">

<div style="font-weight: bold; margin-bottom: 12px; color: #4CAF50; font-size: 1.05em;">向量加法 = 合力</div>

```text
想象你和朋友一起推一辆车：

  你推：向右 3 步，向前 4 步  →  [3, 4]
  朋友推：向右 1 步，向前 2 步  →  [1, 2]

  合起来：[3+1, 4+2] = [4, 6]

  车最终：向右 4 步，向前 6 步

向量加法就是"每一维分别相加"
```

</div>

向量减法也很自然——它表示"差异"：

```text
北京的气候 = [年均温: 12°C, 降水: 580mm, 日照: 2600h]
上海的气候 = [年均温: 16°C, 降水: 1200mm, 日照: 1900h]

差异 = 上海 - 北京 = [+4°C, +620mm, -700h]

意思：上海比北京暖 4 度、多雨 620mm、少晒 700 小时
```

**向量减法告诉你"两个事物差在哪里"。**

---

## 第三章：点积——"两件事有多像？"

这是向量最重要的操作。没有之一。

**点积（dot product）衡量的是：两个向量有多"相似"。**

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(156,39,176,0.06); border: 1px solid rgba(156,39,176,0.2);">

<div style="font-weight: bold; margin-bottom: 12px; color: #9C27B0; font-size: 1.05em;">点积 = 逐项相乘再求和</div>

```text
向量 A = [1, 2, 3]
向量 B = [4, 5, 6]

点积 = 1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32

就是"对应位置相乘，然后全加起来"
```

</div>

点积为什么能衡量相似度？

直觉上：如果两个向量在每一个维度上都"同方向、同大小"，乘起来都是正数，加起来就很大。如果方向相反，乘起来是负数，加起来就小甚至是负的。

更精确地说，有一个叫**余弦相似度**的公式：

```text
余弦相似度 = 点积 / (|A| × |B|)

结果范围：-1 到 1
  1  = 完全相同方向（非常像）
  0  = 完全无关
 -1  = 完全相反
```

<div style="max-width: 640px; margin: 1.5em auto; padding: 15px 20px; border-radius: 8px; background: rgba(76,175,80,0.06); border-left: 4px solid #4CAF50;">

**中国古语的"相似度"：** 《**易经·系辞**》说"**方以类聚，物以群分**"——相似的东西会聚在一起。向量的余弦相似度就是这句话的数学表达：相似的向量（类聚的事物），在高维空间里靠得近。

</div>

> **一句话记住：** 点积 = "这两个东西有多像"。它是 AI 最核心的计算——Attention 机制的核心就是算词与词之间的点积（还记得第五篇吗？QK^T 就是在算点积）。

---

## 第四章：连接 AI——"国王 - 男人 + 女人 = 女王"

现在来看向量在 AI 里最惊艳的表演。

2013 年，Google 的研究者用一个叫 **Word2Vec** 的模型，把每个词映射成一个向量（几百维的数列表）。

训练完成后，他们发现了一个震撼世界的现象：

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; border: 2px solid #E91E63; background: rgba(233,30,99,0.04);">

<div style="font-weight: bold; margin-bottom: 12px; font-size: 1.1em; color: #E91E63; text-align: center;">国王 - 男人 + 女人 ≈ 女王</div>

```text
向量("国王") - 向量("男人") + 向量("女人") ≈ 向量("女王")

翻译成人话：
  从"国王"里减去"男性"的成分
  加上"女性"的成分
  得到的向量最接近的词是——"女王"

这意味着：
  "国王"和"女王"之间的关系
  与
  "男人"和"女人"之间的关系
  在向量空间里是同一个"方向"——
  那个方向代表了"性别"这个概念
```

</div>

**没有人教 AI "什么是性别"。** AI 只是读了大量的文本，自动学到了——"国王"和"男人"有某种关系，"女王"和"女人"有同样的关系。这种关系被编码成了向量空间中的一个**方向**。

更多例子：

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(33,150,243,0.06); border: 1px solid rgba(33,150,243,0.2);">

<div style="font-weight: bold; margin-bottom: 12px; color: #2196F3; font-size: 1.05em;">向量算术 = 语义推理</div>

| 向量运算 | 结果 | 编码的概念 |
|---------|------|----------|
| 巴黎 - 法国 + 日本 | ≈ 东京 | "首都" |
| 好 - 坏 + 慢 | ≈ 快 | "反义" |
| 游泳 - 游泳过去式 + 跑步 | ≈ 跑步过去式 | "时态变换" |

</div>

**向量不只是"一串数字"。它们捕捉到了概念之间的关系。**

如果你想深入了解词向量的运算、余弦相似度的计算、以及 AI 怎么把词变成向量，可以看我的 [《AI 的数学语言（一）：用数字画地图》](/ai-blog/posts/math-for-ai-1-vectors/) 和 [《AI 的数学语言（二）：向量的加减法》](/ai-blog/posts/math-for-ai-2-dot-product/)——那两篇有更详细的代码演示和数学推导。

> **一句话记住：** AI 把每个词变成向量后，词的"含义"就变成了可以加减乘除的数学对象。"国王 - 男人 + 女人 = 女王"不是 AI "理解"了性别，而是向量空间自然编码了语义关系。

---

## 第五章：你一直在用向量

在结束之前，让我指出一件你可能没意识到的事：

**你的大脑一直在做"向量运算"。**

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(255,152,0,0.06); border: 1px solid rgba(255,152,0,0.2);">

<div style="font-weight: bold; margin-bottom: 12px; color: #FF9800; font-size: 1.05em;">大脑的"向量运算"</div>

| 你做的事 | 其实是什么 |
|---------|----------|
| "这两首歌风格很像" | 在"风格向量"空间里，两首歌的点积很大 |
| "这个人和我很合得来" | 在"性格向量"空间里，两人的余弦相似度高 |
| "这道菜缺点什么" | 在"味觉向量"空间里，和你期望的向量有差距 |
| "这部电影让我想起了另一部" | 两部电影的"特征向量"在某些维度上相近 |

</div>

数学没有发明新的东西。数学只是把你大脑已经在做的事——**精确化、量化、可计算**。

---

## 动手实验

### 实验：词向量的相似度

```python
import numpy as np

# 简化的 5 维词向量
words = {
    '国王': np.array([0.9, 0.8, 0.1, 0.3, 0.7]),
    '女王': np.array([0.8, 0.2, 0.1, 0.3, 0.7]),
    '男人': np.array([0.2, 0.9, 0.1, 0.2, 0.5]),
    '女人': np.array([0.1, 0.1, 0.1, 0.2, 0.5]),
    '汽车': np.array([0.1, 0.5, 0.9, 0.8, 0.1]),
}

# 余弦相似度
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("余弦相似度（越接近 1 = 越像）：")
print("─" * 35)
for a, b in [('国王','女王'), ('国王','男人'), ('国王','汽车')]:
    sim = cosine_sim(words[a], words[b])
    bar = "█" * int(sim * 20)
    print(f"  {a} ↔ {b}: {sim:.3f}  {bar}")

# 向量算术
print("\n国王 - 男人 + 女人 ≈ ?")
result = words['国王'] - words['男人'] + words['女人']
best_word, best_sim = '', -1
for w, v in words.items():
    if w not in ['国王', '男人', '女人']:
        sim = cosine_sim(result, v)
        if sim > best_sim:
            best_word, best_sim = w, sim
print(f"  最接近的词: {best_word} (相似度 {best_sim:.3f})")
```

---

## 本篇小结

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; border: 2px solid #FF9800; background: rgba(255,152,0,0.04);">

<div style="font-weight: bold; margin-bottom: 12px; font-size: 1.05em;">这篇文章讲了什么？</div>

**一、向量 = 一组有序的数**
- 用来描述一个事物的多个方面
- 咖啡 [温度, 甜度, 浓度]，词 [768 个维度]

**二、向量加法 = 合力，向量减法 = 差异**
- 加法：每一维分别相加
- 减法：告诉你两个事物差在哪

**三、点积 = 衡量相似度**
- 逐项相乘再求和。余弦相似度范围 -1 到 1
- Attention 的核心就是算点积

**四、AI 的惊艳表演**
- 国王 - 男人 + 女人 ≈ 女王
- 向量自然编码了语义关系

**五、你的大脑一直在做向量运算**
- "这两首歌风格很像" = 点积大
- 数学只是把直觉精确化

</div>

---

## 下一篇预告

向量描述了"一个事物"。但如果你想**同时变换一大堆向量**呢？

比如——把一个图形旋转 45 度、把一张图片放大两倍、把一个词向量从"原始含义"空间投影到"查询/键/值"空间？

做这件事的工具叫**矩阵**。

矩阵不是"一堆数排成方块"。矩阵是一个**变换器**——它可以拉伸、旋转、投影整个空间。而神经网络的每一层，就是一次矩阵变换。

下一篇：**看见数学（十二）：矩阵——空间的变形术**

---

<div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #e0e0e0; font-size: 0.9em; color: #888; line-height: 1.8;">

**《看见数学》系列** — 从结绳记事到 AI，看见数学之美。<br>
本文首发于「AI 学习笔记」博客：https://Jason-Azure.github.io/ai-blog/<br>
微信公众号：AI-lab学习笔记<br>
系列文章完整列表见 [标签：看见数学](/ai-blog/tags/看见数学/)

</div>
