---
title: '看见数学（三）：未知数 x——给"不知道"取个名字'
date: 2026-03-17
draft: false
summary: 'x 不可怕，它只是一个"待填空"。代数的核心不是解方程，而是——你可以用一个符号表示你还不知道的东西，然后用逻辑推出它。而 AI 的训练，本质上就是在同时求解几十亿个 x。'
categories: ["看见数学"]
tags: ["数学思维", "代数", "方程", "未知数", "参数", "看见数学"]
weight: 32
ShowToc: true
TocOpen: true
---

> 上一篇 [《零的发明》](/ai-blog/posts/see-math-2-zero/) 里，我们让数轴从正数延伸到了负无穷。人类终于有了一条完整的数轴。
>
> 但到目前为止，我们做的都是同一件事：**算已知的数。**
>
> 3 + 5 = 8。已知 3，已知 5，算出 8。
>
> 可是，如果你**不知道**其中一个数呢？

> **系列导航**
>
> <div style="max-width: 660px; margin: 0.5em 0; font-size: 0.93em; line-height: 1.9;">
> <div style="border-left: 3px solid #ccc; padding-left: 12px; margin-bottom: 6px; padding: 8px 12px; color: #888;">
> ▹ <a href="/ai-blog/posts/see-math-1-counting/" style="color: #888;">第一篇：结绳记事——人类第一次抽象</a></div>
> <div style="border-left: 3px solid #ccc; padding-left: 12px; margin-bottom: 6px; padding: 8px 12px; color: #888;">
> ▹ <a href="/ai-blog/posts/see-math-2-zero/" style="color: #888;">第二篇：零的发明——最伟大的"无"</a></div>
> <div style="border-left: 3px solid #FF9800; padding-left: 12px; margin-bottom: 6px; background: rgba(255,152,0,0.05); padding: 8px 12px; border-radius: 0 4px 4px 0;">
> <strong>▸ 第三篇（本文）：未知数 x——给"不知道"取个名字</strong></div>
> <div style="border-left: 3px solid #ccc; padding-left: 12px; margin-bottom: 6px; padding: 8px 12px; color: #888;">
> ▹ <a href="/ai-blog/posts/see-math-4-coordinates/" style="color: #888;">第四篇：坐标革命——笛卡尔的天才之桥</a></div>
> <div style="border-left: 3px solid #ccc; padding-left: 12px; padding: 8px 12px; color: #888;">
> ▹ <a href="/ai-blog/posts/see-math-5-equations/" style="color: #888;">第五篇：方程的力量——自然界的源代码</a></div>
> </div>

---

## 第一章：一道一千年前的题

公元 820 年左右，巴格达。

一位波斯学者坐在"智慧之家"（Bayt al-Hikma）里，正在写一本书。

他叫**花拉子米**（al-Khwarizmi）。这本书的名字翻译成中文大概是：《还原与对消的计算法则》。

书名里的阿拉伯语单词 **al-jabr**（还原），后来传入欧洲，变成了一个你一定听过的词——

**Algebra。代数。**

花拉子米在书里讨论了一类问题：

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(255,152,0,0.06); border: 1px solid rgba(255,152,0,0.2);">

<div style="font-weight: bold; margin-bottom: 12px; color: #FF9800; font-size: 1.05em;">花拉子米的原题（意译）</div>

```text
"一个财产，它的十分之一加上它自身，等于 110 第纳尔。
 问：这个财产是多少？"
```

用今天的符号写：

```text
x + x/10 = 110
```

答案：x = 100

</div>

注意——花拉子米那个时代，**没有 x，没有 +，没有 =。** 他用的全是文字描述。上面那道题在原文里是用一大段阿拉伯语写的。

但核心思想已经在了：

**有一个数，我不知道它是多少，但我知道它满足一个条件。我要找到它。**

这就是代数的灵魂。

---

## 第二章：x 是什么？

让我们暂时忘掉数学课本上的 x，从头想这件事。

你在日常生活中，其实**每天都在用 x**——只是你没意识到。

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(33,150,243,0.06); border: 1px solid rgba(33,150,243,0.2);">

<div style="font-weight: bold; margin-bottom: 12px; color: #2196F3; font-size: 1.05em;">你每天都在"解方程"</div>

| 你遇到的问题 | 翻译成数学 |
|-------------|-----------|
| "我手机掉哪了？" | 手机位置 = x，根据线索求解 |
| "这个月还能花多少钱？" | 工资 - 已花 - x = 0，求 x |
| "几点出门才不会迟到？" | 出发时间 x + 通勤 40 分钟 = 9:00 |
| "要买多少个鸡蛋够做蛋糕？" | 每个蛋糕 3 个 × 蛋糕数 = x |
| "晚饭 AA 制每人多少钱？" | 总价 ÷ 人数 = x |

</div>

看到了吗？

**x 就是"我现在不知道，但我想知道"的那个东西。**

给"不知道"取个名字——这就是 x 的全部含义。

你可以叫它 x，可以叫它 y，可以叫它"某某"，可以叫它"那个数"。名字不重要，重要的是这个**思维动作**：

> **承认"我不知道"，把它当作一个对象来处理，然后用逻辑去推。**

这就是代数思维的本质。

<div style="max-width: 640px; margin: 1.5em auto; padding: 15px 20px; border-radius: 8px; background: rgba(76,175,80,0.06); border-left: 4px solid #4CAF50;">

**想一想：** 侦探破案，其实就是在"解方程"——凶手是谁（x），根据已知线索（条件），一步步缩小范围，最终锁定答案。数学家和侦探的思维方式是一样的。

</div>

---

## 第三章：等号——一架天平

在继续之前，我们需要认真看看一个你从小学就认识的符号：

## **=**

等号。

你可能觉得它太简单了，没什么好说的。但等号可能是人类发明的**最深刻的符号之一**。

等号的发明人是英国数学家**罗伯特·雷科德**（Robert Recorde），1557 年。他在书里写道：

> "我选择两条平行的等长线段来表示'相等'，因为没有什么东西比它们更相等了。"

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(156,39,176,0.06); border: 1px solid rgba(156,39,176,0.2);">

<div style="font-weight: bold; margin-bottom: 12px; color: #9C27B0; font-size: 1.05em;">等号 = 天平</div>

```text
         ┌─────────────────────────────┐
         │         方程 = 天平           │
         └─────────────────────────────┘

              左边          右边
            ┌──────┐      ┌──────┐
            │ x + 3 │      │  10  │
            └──┬───┘      └──┬───┘
               │              │
         ══════╧══════════════╧══════
                    ▲
                  支点
              （等号 = ）

         天平两边一样重 → 等号成立
         在左边做什么操作，右边也必须做
         这样天平才能保持平衡
```

</div>

**方程就是天平。等号就是支点。两边必须保持平衡。**

这个理解至关重要，因为"解方程"的全部操作都基于一条规则：

> **你在等号左边做的任何操作，必须同时在右边也做一次。**

来看看怎么用这个规则"解"方程：

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(255,152,0,0.06); border: 1px solid rgba(255,152,0,0.2);">

<div style="font-weight: bold; margin-bottom: 12px; color: #FF9800; font-size: 1.05em;">解方程 = 保持天平平衡</div>

```text
题目：  x + 3 = 10
目标：  把 x 单独留在左边

第一步：两边同时减 3（保持平衡）

        x + 3 - 3 = 10 - 3
        x         = 7       ✓

验证：  7 + 3 = 10  ✓  天平平衡！
```

```text
再来一个：  2x + 5 = 13

第一步：两边同时减 5
        2x = 8

第二步：两边同时除以 2
        x = 4

验证：  2 × 4 + 5 = 13  ✓
```

</div>

整个过程就是：**一步一步拆掉 x 身上的"包袱"，把它解放出来。**

注意：你不是在"猜"x 是什么。你是在**推理**——每一步都有逻辑依据，每一步天平都保持平衡。

> **一句话记住：** 解方程不是"算"出来的。解方程是**推理**出来的——像剥洋葱一样，一层一层剥掉 x 身上的运算，直到 x 赤裸裸地站在你面前。

---

## 第四章：鸡兔同笼——两个未知数

一个 x 的方程你学会了。那如果有**两个**不知道的东西呢？

这就是中国古代最经典的数学题：

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(33,150,243,0.06); border: 1px solid rgba(33,150,243,0.2);">

<div style="font-weight: bold; margin-bottom: 12px; color: #2196F3; font-size: 1.05em;">鸡兔同笼（《孙子算经》，约 5 世纪）</div>

```text
原文："今有雉兔同笼，上有三十五头，下有九十四足。
      问雉兔各几何？"

翻译：笼子里有鸡和兔子，
      头一共 35 个，
      脚一共 94 只。
      鸡几只？兔几只？
```

</div>

古人怎么做这道题？

他们用的是一种巧妙的"假设法"——假设所有动物都是鸡，那应该有 35 × 2 = 70 只脚，但实际上有 94 只脚，多了 24 只。每用一只兔子替换一只鸡就多 2 只脚，所以兔子 = 24 ÷ 2 = 12 只。

很聪明。但**每道题都要想一个新花招**。

现在用代数来做：

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(76,175,80,0.06); border: 1px solid rgba(76,175,80,0.2);">

<div style="font-weight: bold; margin-bottom: 12px; color: #4CAF50; font-size: 1.05em;">方程组：通用的万能钥匙</div>

```text
设：鸡 = x 只，兔 = y 只

条件一（头）：  x  +  y = 35
条件二（脚）：  2x + 4y = 94

从条件一：x = 35 - y
代入条件二：2(35 - y) + 4y = 94
            70 - 2y + 4y = 94
            2y = 24
            y = 12   → 兔 12 只
            x = 23   → 鸡 23 只

验证：23 + 12 = 35 ✓（头对了）
      23×2 + 12×4 = 46 + 48 = 94 ✓（脚也对了）
```

</div>

两种方法都能解，但代数的方法有一个巨大的优势：

**它是通用的。**

假设法每道题都要想新花招。方程组的方法**换了题目也一样用**——管你是鸡兔同笼、牛马同圈、还是苹果橘子混装，步骤完全相同：

1. 给不知道的量取名字（x, y）
2. 把条件翻译成方程
3. 用逻辑推出答案

这就是代数的威力：**把"巧妙"变成"机械"，把"灵感"变成"流程"。**

> **一句话记住：** 代数的伟大之处，不在于它能解难题，而在于它让**任何人**都能用**同一套方法**解**任何题**。这就是符号化和通用化的力量。

---

## 第五章：符号的压缩力

还记得第一篇讲的吗？**数学是压缩语言。**

x 的发明是压缩史上的一次巨大飞跃。看看同一道题在不同时代的写法：

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(255,152,0,0.06); border: 1px solid rgba(255,152,0,0.2);">

<div style="font-weight: bold; margin-bottom: 12px; color: #FF9800; font-size: 1.05em;">同一道题，越来越短</div>

```text
古巴比伦（公元前 2000 年）：
"我有一个长方形的田。长比宽多 7 步。
 面积是 60 平方步。问长和宽各是多少？"
                                    → 3 行文字

花拉子米（公元 820 年）：
"一个东西，它的平方加上它的十倍等于三十九"
                                    → 1 行文字

今天：
x² + 10x = 39
                                    → 1 行公式，12 个字符
```

</div>

**从 3 行文字到 12 个字符。** 信息没少，但体积压缩了十几倍。

而且，不只是更短了。符号让你能做文字做不到的事：

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(156,39,176,0.06); border: 1px solid rgba(156,39,176,0.2);">

<div style="font-weight: bold; margin-bottom: 12px; color: #9C27B0; font-size: 1.05em;">符号能做，文字做不到的事</div>

| 能力 | 文字描述 | 符号公式 |
|------|---------|---------|
| **一眼看出结构** | "一个数的平方加上这个数的十倍" | x² + 10x |
| **机械化操作** | 需要重新理解语义 | 直接移项、合并、化简 |
| **通用化** | 每道题重新描述 | 换个系数就是新题 |
| **传递跨语言** | 中文/阿拉伯语/拉丁语各不同 | x² + 10x = 39 全世界都看得懂 |

</div>

最后一行尤其重要。

**数学符号是人类第一种真正的"通用语言"。**

一个中国学生、一个巴西学生、一个尼日利亚学生——他们可能一个字的日常语言都不通，但看到 x² + 10x = 39，他们理解的是**完全相同**的东西。

这就是压缩的极致——不只是压缩了长度，还压缩掉了语言和文化的差异。

> **一句话记住：** 符号化不只是"更短"。它让知识变成了**跨越语言、跨越时代、跨越文化**的通用格式。x² + 10x = 39，古巴比伦人看得懂，你也看得懂。

---

## 第六章：连接 AI——几十亿个 x

现在，让我们把目光从一个 x 移向 AI。

一个方程有一个未知数 x。鸡兔同笼有两个未知数 x 和 y。

那神经网络呢？

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; border: 2px solid #E91E63; background: rgba(233,30,99,0.04);">

<div style="font-weight: bold; margin-bottom: 12px; font-size: 1.05em; color: #E91E63;">AI 有多少个"x"？</div>

```text
一道鸡兔同笼题：           2 个未知数 (x, y)
一组工程方程：              几十个未知数
一个天气预报模型：          几千个未知数

GPT-2（2019）：            1.5 亿个未知数
GPT-3（2020）：            1750 亿个未知数
GPT-4（2023）：            据传超过 1 万亿个未知数
```

</div>

这些"未知数"在 AI 里有一个名字：**参数**（parameters），也叫**权重**（weights）。

**训练一个 AI 模型，就是在"解方程"——只不过，方程的数量和未知数的数量都是天文级别的。**

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(76,175,80,0.06); border: 1px solid rgba(76,175,80,0.2);">

<div style="font-weight: bold; margin-bottom: 12px; color: #4CAF50; font-size: 1.05em;">鸡兔同笼 vs 训练 AI</div>

| | 鸡兔同笼 | 训练 GPT |
|---|---------|---------|
| **未知数** | x = 鸡, y = 兔 | w₁, w₂, ..., w₁₇₅₀₀₀₀₀₀₀₀₀ |
| **条件（数据）** | 头 = 35, 脚 = 94 | 几万亿个词的文本 |
| **方程** | 2 个 | 数不清（每个词的预测都是一个条件） |
| **解法** | 手算几步 | 梯度下降，迭代几十万次 |
| **目标** | 精确解 | 足够好的近似解 |

</div>

来看一个具体的例子。

我们实验室里有一个叫 **microgpt** 的迷你模型。它只有 **4192 个参数**——对 GPT 来说微不足道，但对理解原理来说刚刚好。

```bash
azureuser@ai-lab:~$ cd ~/microgpt && python3 microgpt.py
```

训练开始后，你会看到：

```text
step 0, loss = 3.367  ← 刚开始，什么都不会，loss 很高
step 100, loss = 2.741
step 200, loss = 2.456
step 500, loss = 2.112
step 1000, loss = 1.924  ← 训练完成，loss 下降了

生成的名字: Jame, Kel, Mara, Trey ...
```

这个过程在做什么？

**调整 4192 个参数（x₁, x₂, ..., x₄₁₉₂），让模型对数据的预测越来越准确。**

每一步都在"解方程"。只不过不是一步求出精确解，而是**一点一点逼近**一个"足够好"的答案。

就像你在一个黑暗的房间里找东西——你看不见它在哪，但你每次伸手都能感觉到"比刚才近了一点"还是"远了一点"。走几千步之后，你就摸到了。

> **一句话记住：** AI 训练 = 求解天文数量的未知数。代数教会了人类"给未知数取名字"，AI 把这个思想推到了极致——几十亿个 x，用数学方法一起求解。

---

## 第七章：从 x 到 x 思维

这篇文章的最后，我想说一个可能让你意外的观点：

**x 不只是一个数学符号。x 是一种思维方式。**

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(33,150,243,0.06); border: 1px solid rgba(33,150,243,0.2);">

<div style="font-weight: bold; margin-bottom: 12px; color: #2196F3; font-size: 1.05em;">x 思维 = 处理"不确定"的能力</div>

```text
日常生活中的 x 思维：

  "如果明天下雨(x)，我就带伞"
  → x 是一个可能但不确定的事件，你根据它做计划

  "这条路堵了，换一条"
  → 原计划（方程）不成立，你调整条件重新解

  "我不知道对方怎么想"
  → 对方的想法是 x，你根据言行（条件）去推测

  "如果我每月存 x 元，几年后能买房？"
  → 这就是方程：12x × 年数 = 首付
```

</div>

小孩子的世界里，大多数事情是"已知"的：太阳会升起，妈妈会做饭，1+1=2。

成年人的世界里，大多数事情是"未知"的：明天股市涨不涨？这个项目能不能按时完成？我该不该换工作？

**处理"未知"是成人世界最核心的能力。而代数——给"不知道"取个名字，然后用逻辑去推理——正是这种能力的数学形式。**

数学不只教你算数。数学教你**如何面对未知**。

<div style="max-width: 640px; margin: 1.5em auto; padding: 20px; border-radius: 8px; border: 2px solid #FF9800; background: rgba(255,152,0,0.04);">

<div style="font-weight: bold; margin-bottom: 12px; font-size: 1.05em; color: #FF9800;">这篇的核心：</div>

数学的第三次飞跃（前两次是"抽象"和"零"）：

**敢于面对"不知道"，给它一个名字，然后用逻辑把它推出来。**

这就是代数。这就是 x。

</div>

---

## 动手实验

### 实验一：Python 解方程

```python
# 不需要任何库！纯逻辑

# 鸡兔同笼：x + y = 35, 2x + 4y = 94
# 用代入法求解

total_heads = 35
total_feet = 94

# 从条件一：x = total_heads - y
# 代入条件二：2(total_heads - y) + 4y = total_feet
# 化简：2*total_heads - 2y + 4y = total_feet
# 化简：2y = total_feet - 2*total_heads

y = (total_feet - 2 * total_heads) // 2  # 兔
x = total_heads - y                       # 鸡

print(f"鸡: {x} 只")
print(f"兔: {y} 只")
print(f"验证 — 头: {x + y}, 脚: {2*x + 4*y}")

# 输出：
# 鸡: 23 只
# 兔: 12 只
# 验证 — 头: 35, 脚: 94
```

### 实验二：给 AI 的"x"拍个快照

```python
# 看看 microgpt 训练前后，参数（x）是怎么变化的
# 这需要在 ai-lab-venv 环境里运行

# 简化演示：一个只有 3 个参数的"超迷你网络"
import random

# 三个参数（未知数），随机初始化
params = [random.uniform(-1, 1) for _ in range(3)]
print(f"训练前的参数（随机）: ")
for i, p in enumerate(params):
    print(f"  x{i+1} = {p:.4f}")

# 模拟训练：参数被一步步调整（梯度下降的简化演示）
target = [0.5, -0.3, 0.8]  # 假设"正确答案"
learning_rate = 0.1

print(f"\n模拟训练过程：")
for step in range(10):
    # 计算"差距"（loss）
    loss = sum((p - t) ** 2 for p, t in zip(params, target))
    if step % 3 == 0:
        print(f"  第 {step} 步, loss = {loss:.4f}")
    # 调整参数：往"正确答案"的方向靠近一点
    params = [p + learning_rate * (t - p) for p, t in zip(params, target)]

print(f"\n训练后的参数: ")
for i, p in enumerate(params):
    print(f"  x{i+1} = {p:.4f}  (目标: {target[i]})")

# 输出类似：
# 训练前的参数（随机）:
#   x1 = 0.7234
#   x2 = -0.8912
#   x3 = 0.1456
#
# 模拟训练过程：
#   第 0 步, loss = 0.7531
#   第 3 步, loss = 0.3891
#   第 6 步, loss = 0.2011
#   第 9 步, loss = 0.1040
#
# 训练后的参数:
#   x1 = 0.5134  (目标: 0.5)
#   x2 = -0.3073  (目标: -0.3)
#   x3 = 0.7826  (目标: 0.8)
```

**看到了吗？训练 = 让一堆"x"从随机值慢慢逼近"正确值"。**

---

## 本篇小结

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; border: 2px solid #FF9800; background: rgba(255,152,0,0.04);">

<div style="font-weight: bold; margin-bottom: 12px; font-size: 1.05em;">这篇文章讲了什么？</div>

**一、x = 给"不知道"取个名字**
- 花拉子米（公元 820 年）发明了代数 (al-jabr)
- x 不可怕，它只是"待填空"，你每天都在用

**二、方程 = 天平**
- 等号是支点，两边必须保持平衡
- 解方程 = 一层层剥掉 x 身上的运算

**三、代数把"灵感"变成了"流程"**
- 古人每道题要想新花招
- 代数提供了通用的方法：取名字 → 列方程 → 推理

**四、符号是跨越时空的通用语言**
- 从三行文字到 12 个字符
- x² + 10x = 39 全世界都看得懂

**五、AI 训练 = 求解几十亿个 x**
- 神经网络的参数就是 x
- 训练就是让参数从随机值逼近"好的值"

**六、x 思维 = 处理"未知"的能力**
- 数学不只教你算数，教你面对未知
- 敢于面对"不知道"，用逻辑去推——这就是代数思维

</div>

---

## 下一篇预告

我们现在有了数（从负无穷到正无穷），有了 x（可以表示任何未知的量），有了方程（描述 x 必须满足的条件）。

但数是"看不见"的。方程是"读不出画面"的。

**如果我们能把方程"画出来"呢？**

1637 年，一个法国哲学家躺在床上看天花板上的苍蝇，想到了一个天才的主意：**用两个数来确定一个点的位置。**

这个人叫笛卡尔。他的发明——坐标系——从此把代数和几何**永远焊在了一起**。

下一篇：**[看见数学（四）：坐标革命——笛卡尔的天才之桥](/ai-blog/posts/see-math-4-coordinates/)**

---

<div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #e0e0e0; font-size: 0.9em; color: #888; line-height: 1.8;">

**《看见数学》系列** — 从结绳记事到 AI，看见数学之美。<br>
本文首发于「AI 学习笔记」博客：https://Jason-Azure.github.io/ai-blog/<br>
微信公众号：AI-lab学习笔记<br>
系列文章完整列表见 [标签：看见数学](/ai-blog/tags/看见数学/)

</div>
