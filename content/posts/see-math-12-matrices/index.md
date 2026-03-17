---
title: '看见数学（十二）：矩阵——空间的变形术'
date: 2026-03-19
draft: false
summary: '矩阵不是"一堆数排成方块"。矩阵是一个变换器——它可以旋转、缩放、投影整个空间。神经网络的每一层，就是一次矩阵变换。而两千年前的《九章算术》，已经在用矩阵解方程了。'
categories: ["看见数学"]
tags: ["数学思维", "矩阵", "线性变换", "Transformer", "看见数学"]
weight: 41
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
> <div style="border-left: 3px solid #FF9800; padding-left: 12px; margin-bottom: 6px; background: rgba(255,152,0,0.05); padding: 8px 12px; border-radius: 0 4px 4px 0;">
> <strong>▸ 第十二篇（本文）：矩阵——空间的变形术</strong></div>
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

## 第一章：一台"空间变形机"

上一篇我们说，向量是"一组数"，可以描述任何事物——一杯咖啡、一个词的含义、一张脸。

现在问一个新问题：**如果你想同时改变一大堆向量，怎么办？**

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(255,152,0,0.06); border: 1px solid rgba(255,152,0,0.2);">

<div style="font-weight: bold; margin-bottom: 12px; color: #FF9800; font-size: 1.05em;">想象你是一个画家</div>

```text
你画了一只猫。猫是由很多点（向量）组成的。

现在你想：
  把猫放大 2 倍      → 所有点的坐标 ×2
  把猫旋转 45 度     → 所有点重新计算位置
  把猫翻转成镜像     → 所有 x 坐标取反

每一种变换，都是对"所有向量"施加同一种操作。
做这件事的数学工具，就是矩阵。
```

</div>

**矩阵 = 一台空间变形机。** 你输入一个向量（一个点的位置），它输出变换后的向量（新位置）。

但它不只变换一个点——它**同时变换整个空间里的所有点**。

```text
向量: [1, 2, 3]                ← 一行数字，描述一个事物

矩阵: [[2, 0, 0],             ← 一张数字表格
        [0, 2, 0],
        [0, 0, 2]]

矩阵 × 向量 = [2, 4, 6]       ← 每个维度都放大了 2 倍
```

这个矩阵做的事情很简单——**把所有东西放大 2 倍**。但矩阵能做的事远不止这个。

> **一句话记住：** 向量是"一个事物"。矩阵是"对事物的一种变换"。向量是名词，矩阵是动词。

---

## 第二章：旋转、缩放、翻转——矩阵的三板斧

让我们看看矩阵能做什么样的变换。为了直观，先在 2D 平面上演示：

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(33,150,243,0.06); border: 1px solid rgba(33,150,243,0.2);">

<div style="font-weight: bold; margin-bottom: 12px; color: #2196F3; font-size: 1.05em;">三种经典变换</div>

```text
原始点: [1, 0]（向右 1 步）

【缩放】放大 3 倍
  矩阵 = [[3, 0],     结果 = [3, 0]
           [0, 3]]     → 向右 3 步

【旋转】逆时针 90°
  矩阵 = [[0, -1],    结果 = [0, 1]
           [1,  0]]    → 向上 1 步（转了 90°）

【翻转】左右镜像
  矩阵 = [[-1, 0],    结果 = [-1, 0]
           [ 0, 1]]    → 向左 1 步（镜像）
```

</div>

**关键洞察：不同的矩阵 = 不同的变换方式。** 矩阵里的数字决定了空间会怎么"变形"。

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(76,175,80,0.06); border: 1px solid rgba(76,175,80,0.2);">

<div style="font-weight: bold; margin-bottom: 12px; color: #4CAF50; font-size: 1.05em;">生活中的矩阵变换</div>

| 你做的事 | 其实是矩阵变换 |
|---------|--------------|
| 手机照片旋转 90° | 旋转矩阵 |
| Photoshop 水平翻转 | 翻转矩阵 |
| 地图放大缩小 | 缩放矩阵 |
| 3D 游戏里视角变化 | 透视投影矩阵 |
| 抖音特效"瘦脸" | 非均匀缩放矩阵 |

</div>

你每天都在用矩阵变换，只是软件帮你做了数学。

> **一句话记住：** 每次你在手机上旋转、放大照片，背后都有一个矩阵在工作。矩阵就是"空间变形的指令"。

---

## 第三章：矩阵乘法——变换的叠加

如果一个矩阵是"一次变换"，那**两个矩阵相乘**是什么？

答案出奇简单：**先做一次变换，再做一次变换。**

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(156,39,176,0.06); border: 1px solid rgba(156,39,176,0.2);">

<div style="font-weight: bold; margin-bottom: 12px; color: #9C27B0; font-size: 1.05em;">矩阵乘法 = 变换的叠加</div>

```text
A = 旋转 90° 的矩阵
B = 放大 2 倍的矩阵

A × B = "先放大 2 倍，再旋转 90°"

一步到位！

这就是为什么矩阵乘法的定义
看起来那么奇怪（行×列求和）——
因为它在算"两次变换叠加后的效果"。
```

</div>

这就回答了一个很多人学线性代数时的困惑：**为什么矩阵乘法不是"对应位置相乘"？**

因为矩阵乘法不是在"合并两张表格"——它是在**组合两次变换**。行乘列求和的规则，恰好让"变换的叠加"在数学上成立。

<div style="max-width: 640px; margin: 1.5em auto; padding: 15px 20px; border-radius: 8px; background: rgba(76,175,80,0.06); border-left: 4px solid #4CAF50;">

**为什么矩阵乘法不满足交换律？** A × B ≠ B × A。因为"先旋转再放大"和"先放大再旋转"可能得到不同结果——就像"先穿袜子再穿鞋"和"先穿鞋再穿袜子"完全不同。变换的**顺序**很重要。

</div>

> **一句话记住：** 矩阵乘法不是"数字的计算技巧"，而是"两次变换的合成"。理解了这一点，线性代数的大门就打开了。

---

## 第四章：连接 AI——Transformer 里的矩阵

现在来看矩阵在 AI 里最核心的用法。

还记得 [上一篇](/ai-blog/posts/see-math-11-vectors/) 说的吗？AI 把每个词变成一个向量（768 维的数列表）。

但这个向量只是"原始含义"。Transformer 要做的事情是：**从不同角度重新审视每个词。**

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; border: 2px solid #E91E63; background: rgba(233,30,99,0.04);">

<div style="font-weight: bold; margin-bottom: 12px; font-size: 1.1em; color: #E91E63; text-align: center;">QKV 投影 = 矩阵变换</div>

```text
一个词的原始向量 x = [0.2, 0.5, 0.1, ..., 0.8]  (768维)

Transformer 用三个矩阵，把 x 变成三个不同的向量：

  Q = Wq × x     "我在找什么？"（查询）
  K = Wk × x     "我能提供什么？"（键）
  V = Wv × x     "我的内容是什么？"（值）

Wq、Wk、Wv 是三个不同的矩阵（变换器）。
同一个词，从三个不同角度被"审视"。
```

</div>

**这就是矩阵在 AI 里最核心的角色：投影（projection）。**

投影是什么？想象你拿着一个立体的地球仪，从正面拍一张照片——你得到一个 2D 的平面图。这就是把 3D 空间"投影"到 2D。信息有损失，但换来了一个特定视角。

同样，QKV 矩阵把每个词从"768 维的通用含义空间"投影到"查询空间"、"匹配空间"、"内容空间"——每个空间关注词义的不同侧面。

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(33,150,243,0.06); border: 1px solid rgba(33,150,243,0.2);">

<div style="font-weight: bold; margin-bottom: 12px; color: #2196F3; font-size: 1.05em;">神经网络 = 矩阵变换链</div>

```text
输入 → [矩阵变换₁] → [激活函数] → [矩阵变换₂] → [激活函数] → ... → 输出

神经网络的每一层：
  1. 矩阵变换：把向量投影到新空间
  2. 激活函数：加入非线性（第二幕里学过的函数！）

GPT-4 可能有上百层。
= 上百次矩阵变换 + 上百次非线性弯折。
= 把原始数据一步步变换到"能预测下一个词"的形态。
```

</div>

如果你想深入了解矩阵乘法的具体计算过程和 Attention 的完整公式推导，可以看我的 [《AI 的数学语言（三）：矩阵》](/ai-blog/posts/math-for-ai-3-matrices/) 和 [《AI 的数学语言（四）：矩阵乘法与 AI》](/ai-blog/posts/math-for-ai-4-matmul/)——那两篇有详细的手算演示和代码实现。

> **一句话记住：** 神经网络做的事情，本质上就是"一层层矩阵变换"。每一层矩阵把数据空间"揉"成新的形状，直到揉成 AI 需要的样子。

---

## 第五章：中国古代的"矩阵"

矩阵不是西方的发明。

中国人用矩阵解方程，比欧洲早了**一千五百年**。

<div style="max-width: 640px; margin: 1.5em auto; padding: 15px 20px; border-radius: 8px; background: rgba(76,175,80,0.06); border-left: 4px solid #4CAF50;">

**《九章算术》第八章：方程术**（约公元前 100 年）

> "今有上禾三秉，中禾二秉，下禾一秉，实三十九斗……"

翻译成现代语言：

3x + 2y + z = 39

用**算筹**（小棍子）在地上摆成方阵来求解——这就是最早的"矩阵消元法"，比欧洲的高斯消元法早了约 1800 年。

</div>

"矩阵"这个中文词本身就很有意思：

```text
矩 = 方形的尺（《说文解字》："矩，规矩之矩"）
阵 = 排列、阵法

矩阵 = 方方正正排列的数阵
```

古人用算筹在地上排出一个"方阵"来解方程，和今天我们在计算机里用矩阵做变换，思想一脉相承。

<div style="max-width: 640px; margin: 1.5em auto; padding: 15px 20px; border-radius: 8px; background: rgba(255,152,0,0.06); border-left: 4px solid #FF9800;">

**从算筹到 GPU：** 两千年前的数学家把算筹一根根摆在桌上做"矩阵运算"。今天的 GPU（图形处理器）每秒做几万亿次矩阵乘法。工具变了，数学没变。

NVIDIA 的 A100 GPU 的核心卖点不是"跑游戏更流畅"——而是它内置了专门的 **Tensor Core**，专门加速矩阵乘法。AI 时代的芯片，本质上是一台**超级矩阵计算机**。

</div>

---

## 动手实验

### 实验：矩阵变换的效果

```python
import numpy as np

# 原始点（一个三角形的三个顶点）
triangle = np.array([
    [0, 0],   # 原点
    [1, 0],   # 右边
    [0.5, 1], # 上方
])

# 缩放矩阵：放大 2 倍
scale = np.array([[2, 0],
                  [0, 2]])

# 旋转矩阵：逆时针旋转 90 度
rotate = np.array([[0, -1],
                   [1,  0]])

# 翻转矩阵：水平翻转
flip = np.array([[-1, 0],
                 [ 0, 1]])

for name, matrix in [("缩放2倍", scale), ("旋转90°", rotate), ("水平翻转", flip)]:
    result = triangle @ matrix.T   # 每个点做矩阵变换
    print(f"\n{name}:")
    for i, (orig, new) in enumerate(zip(triangle, result)):
        print(f"  点{i}: {orig} → {new}")

# 组合变换：先旋转再放大
combined = scale @ rotate    # 矩阵乘法 = 变换叠加
result = triangle @ combined.T
print(f"\n先旋转90°再放大2倍（组合变换）:")
for i, (orig, new) in enumerate(zip(triangle, result)):
    print(f"  点{i}: {orig} → {new}")
```

---

## 本篇小结

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; border: 2px solid #FF9800; background: rgba(255,152,0,0.04);">

<div style="font-weight: bold; margin-bottom: 12px; font-size: 1.05em;">这篇文章讲了什么？</div>

**一、矩阵 = 空间变形机**
- 输入一个向量，输出变换后的向量
- 不同的矩阵 = 不同的变换方式

**二、旋转、缩放、翻转——矩阵的三板斧**
- 手机旋转照片、地图缩放、游戏视角——全是矩阵变换

**三、矩阵乘法 = 变换的叠加**
- A × B = 先做 B 变换，再做 A 变换
- 不满足交换律，因为变换顺序很重要

**四、AI 的核心操作就是矩阵变换**
- QKV 投影 = 三次矩阵变换
- 神经网络 = 一层层的矩阵变换链

**五、《九章算术》的方程术**
- 中国人用算筹做"矩阵消元"，比高斯消元法早 1800 年
- 从算筹到 GPU，工具变了，数学没变

</div>

---

## 下一篇预告

矩阵变换是"确定的"——给定输入，输出完全确定。

但现实世界充满了**不确定性**。明天会不会下雨？这个邮件是不是垃圾邮件？AI 生成下一个词时，到底选哪个？

描述"不确定性"的数学工具叫**概率**。

概率不是"猜"——概率是**用数学管理无知**。你不知道明天是否下雨，但你可以说"下雨的概率是 60%"，然后据此做出最优决策。AI 的每一次预测，都是在做概率判断。

下一篇：**看见数学（十三）：概率——拥抱不确定**

---

<div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #e0e0e0; font-size: 0.9em; color: #888; line-height: 1.8;">

**《看见数学》系列** — 从结绳记事到 AI，看见数学之美。<br>
本文首发于「AI 学习笔记」博客：https://Jason-Azure.github.io/ai-blog/<br>
微信公众号：AI-lab学习笔记<br>
系列文章完整列表见 [标签：看见数学](/ai-blog/tags/看见数学/)

</div>
