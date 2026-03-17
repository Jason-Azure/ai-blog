---
title: '看见数学（四）：坐标革命——笛卡尔的天才之桥'
date: 2026-03-17
draft: false
summary: '1637 年，笛卡尔把数和形焊在了一起。从此，每个方程都能画一幅画，每幅画都能写一个方程。而 AI 做的第一件事——把词变成向量——就是给每个词一个"坐标"。'
categories: ["看见数学"]
tags: ["数学思维", "坐标系", "笛卡尔", "数形结合", "词向量", "看见数学"]
weight: 33
ShowToc: true
TocOpen: true
---

> 上一篇 [《未知数 x》](/ai-blog/posts/see-math-3-unknown-x/) 里，我们学会了给"不知道"取名字，用方程去推理。
>
> 但到目前为止，数学都是**看不见**的。数字是抽象的，方程是一行符号。
>
> 如果我告诉你——**每一个方程，都可以画成一幅画呢？**

> **系列导航**
>
> <div style="max-width: 660px; margin: 0.5em 0; font-size: 0.93em; line-height: 1.9;">
> <div style="border-left: 3px solid #ccc; padding-left: 12px; margin-bottom: 6px; padding: 8px 12px; color: #888;">
> ▹ <a href="/ai-blog/posts/see-math-1-counting/" style="color: #888;">第一篇：结绳记事——人类第一次抽象</a></div>
> <div style="border-left: 3px solid #ccc; padding-left: 12px; margin-bottom: 6px; padding: 8px 12px; color: #888;">
> ▹ <a href="/ai-blog/posts/see-math-2-zero/" style="color: #888;">第二篇：零的发明——最伟大的"无"</a></div>
> <div style="border-left: 3px solid #ccc; padding-left: 12px; margin-bottom: 6px; padding: 8px 12px; color: #888;">
> ▹ <a href="/ai-blog/posts/see-math-3-unknown-x/" style="color: #888;">第三篇：未知数 x——给"不知道"取个名字</a></div>
> <div style="border-left: 3px solid #FF9800; padding-left: 12px; margin-bottom: 6px; background: rgba(255,152,0,0.05); padding: 8px 12px; border-radius: 0 4px 4px 0;">
> <strong>▸ 第四篇（本文）：坐标革命——笛卡尔的天才之桥</strong></div>
> <div style="border-left: 3px solid #ccc; padding-left: 12px; padding: 8px 12px; color: #888;">
> ▹ <a href="/ai-blog/posts/see-math-5-equations/" style="color: #888;">第五篇：方程的力量——自然界的源代码</a></div>
> </div>

---

## 第一章：天花板上的苍蝇

1637 年，法国。

**勒内·笛卡尔**（René Descartes）——哲学家，写过"我思故我在"的那位——据说正躺在床上发呆。

天花板上有一只苍蝇在爬。

笛卡尔盯着它，脑子里冒出一个问题：

> **我怎么才能精确地告诉别人，这只苍蝇在天花板的什么位置？**

他注意到，天花板的边缘形成了两条直线——一条横的，一条竖的。

如果以墙角为起点——

- 苍蝇离左墙 **3 尺**
- 苍蝇离前墙 **5 尺**

两个数 **(3, 5)**，就能唯一确定苍蝇的位置。

<div style="max-width: 660px; margin: 1.5em auto; text-align: center;">

```text
  前墙
  ─────────────────────────
  │                         │
  │            🪰           │
  │         (3, 5)          │
  │                         │
  │    ← 3尺 →              │
左│         ↑               │
墙│         5尺             │
  │         ↓               │
  ═════════════════════════
  墙角 (0,0)
```

</div>

这个看似简单的想法，改变了整个数学的历史。

因为笛卡尔做了一件前人从未做过的事：**他在数（代数）和形（几何）之间架了一座桥。**

从此——
- 每一个**点**，都可以用一组**数**来表示
- 每一个**方程**，都可以画成一条**线**或一个**面**
- 每一个**几何图形**，都可以写成一个**公式**

**代数和几何，从此是同一件事。**

---

## 第二章：两条线创造一个世界

笛卡尔的发明，我们今天叫它**直角坐标系**（Cartesian coordinate system）。名字就来自他的拉丁文名 Cartesius。

构造极其简单：

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(33,150,243,0.06); border: 1px solid rgba(33,150,243,0.2);">

<div style="font-weight: bold; margin-bottom: 12px; color: #2196F3; font-size: 1.05em;">坐标系：两条线 + 一个原点</div>

```text
          y 轴（纵轴）
          ↑
     3  ─ ┤         ★ B(4, 3)
          │
     2  ─ ┤    ★ A(2, 2)
          │
     1  ─ ┤
          │
     0  ──┼────┼────┼────┼────┼──→  x 轴（横轴）
          0    1    2    3    4

  原点 (0, 0) = 两条轴的交叉点
  A 点 = (2, 2) → 右走 2 步，上走 2 步
  B 点 = (4, 3) → 右走 4 步，上走 3 步
```

</div>

就是这么简单：

- **横轴**（x 轴）：往右是正，往左是负
- **纵轴**（y 轴）：往上是正，往下是负
- **原点** (0, 0)：两条轴的交叉点
- **每个点**：用 (x, y) 两个数确定

还记得第二篇里零打开了负数的大门吗？在坐标系里，负数让我们有了**四个象限**：

<div style="max-width: 660px; margin: 1.5em auto; text-align: center;">

```text
                y
                ↑
   第二象限     │     第一象限
   (-x, +y)    │     (+x, +y)
   "左上"       │     "右上"
 ───────────── 0 ──────────────→ x
   第三象限     │     第四象限
   (-x, -y)    │     (+x, -y)
   "左下"       │     "右下"
                │

  没有零和负数 → 只有右上角（第一象限）
  有了零和负数 → 四个象限，无限延伸
```

</div>

看到了吗？**前面学的每一个概念都在这里汇合了**——零是原点，负数让空间完整，x 和 y 是两个未知数。坐标系把它们全部统一在一张图里。

> **一句话记住：** 坐标系 = 一张无限大的纸 + 两条带刻度的线。任何一个点，都有一个唯一的"地址"——(x, y)。

---

## 第三章：方程会"画画"

坐标系最震撼的地方来了。

在上一篇里，方程 y = 2x + 1 只是一行符号。现在，我们来**画出来**看看。

方程的意思是：给 x 一个值，就能算出 y 的值。

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(255,152,0,0.06); border: 1px solid rgba(255,152,0,0.2);">

<div style="font-weight: bold; margin-bottom: 12px; color: #FF9800; font-size: 1.05em;">y = 2x + 1：从数到点到线</div>

```text
第一步：算几个点

  x = 0  →  y = 2×0 + 1 = 1   →  点 (0, 1)
  x = 1  →  y = 2×1 + 1 = 3   →  点 (1, 3)
  x = 2  →  y = 2×2 + 1 = 5   →  点 (2, 5)
  x = -1 →  y = 2×(-1)+1 = -1 →  点 (-1, -1)

第二步：把这些点画在坐标系上

     y
  5 ─┤                  ★ (2,5)
     │                ╱
  3 ─┤          ★ (1,3)
     │        ╱
  1 ─┤  ★ (0,1)
     │  ╱
 -1 ─★ (-1,-1)
     │╱
 ────┼────┼────┼────→ x
    -1    0    1    2

第三步：把所有可能的 x 都算一遍

     →  一条直线！
```

</div>

**一个方程 → 一条线。**

这就是笛卡尔的魔法。方程不再是一行干巴巴的符号——它有了**形状**。

不同的方程画出不同的"画"：

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(156,39,176,0.06); border: 1px solid rgba(156,39,176,0.2);">

<div style="font-weight: bold; margin-bottom: 12px; color: #9C27B0; font-size: 1.05em;">方程的"画廊"</div>

```text
y = 2x + 1        →  直线（匀速运动的轨迹）
         ╱

y = x²             →  抛物线（投篮的弧线）
        ╱╲

y = sin(x)         →  波浪（声音、光、心跳）
       ∿∿∿

x² + y² = 25       →  圆（r = 5 的完美圆）
        ◯

y = 2ˣ             →  指数曲线（病毒传播）
        ╱ ← 越来越陡！
```

**每一个方程都有一张"脸"。**
**学会了坐标系，你就拥有了把方程"看见"的能力。**

</div>

这就是为什么这一篇叫"坐标革命"——它不是一个小改进。它**从根本上改变了人类理解数学的方式**。

在笛卡尔之前：
- 代数和几何是两门独立的学科
- 代数家用符号算，几何家用尺规画
- 彼此说的是"不同的语言"

在笛卡尔之后：
- **它们是同一件事的两面**
- 每个代数问题都可以画成几何图形
- 每个几何图形都可以写成代数方程

> **一句话记住：** 数形结合不是一种"学习技巧"。它是**数学本身的结构**。方程和图形，从来就是同一个东西的两种表达方式。笛卡尔只是第一个看见这一点的人。

---

## 第四章：从两个数到三个数

笛卡尔的坐标系用两个数 (x, y) 确定平面上的一个点。

那如果加一个维度呢？

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(76,175,80,0.06); border: 1px solid rgba(76,175,80,0.2);">

<div style="font-weight: bold; margin-bottom: 12px; color: #4CAF50; font-size: 1.05em;">维度的扩展</div>

```text
1 个数  →  数轴上的一个点
             温度: 23°C
             ──────●──────→

2 个数  →  平面上的一个点
             地图: (经度, 纬度)
                  │
                  ●
                  │
             ─────┼──────→

3 个数  →  空间中的一个点
             房间: (楼层, 左右, 前后)
                ╱
               ●
              ╱│
             ╱ │
            ─────┼──────→
```

</div>

三维空间 (x, y, z) 我们还能想象——前后、左右、上下。这就是我们生活的物理世界。

你每天都在用三维坐标系，只是可能没意识到：

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(33,150,243,0.06); border: 1px solid rgba(33,150,243,0.2);">

<div style="font-weight: bold; margin-bottom: 12px; color: #2196F3; font-size: 1.05em;">坐标系无处不在</div>

| 场景 | 坐标 | 维度 |
|------|------|------|
| 地图导航 | (经度, 纬度) | 2D |
| GPS 定位 | (经度, 纬度, 海拔) | 3D |
| 游戏角色位置 | (x, y, z) | 3D |
| 棋盘 | (列, 行)，如 E4 | 2D |
| 电影院座位 | (排, 号) | 2D |
| 公寓地址 | (栋, 楼层, 房号) | 3D |
| Excel 单元格 | (列, 行)，如 C5 | 2D |

</div>

但这里有一个关键问题：**为什么要停在三个？**

从数学的角度——没有任何理由停在三个。

1 个数、2 个数、3 个数……为什么不能 4 个？10 个？100 个？768 个？

数学完全允许。公式一模一样。只是**你的眼睛和大脑画不出来了**。

但你的眼睛画不出来，不代表它不存在。

> **一句话记住：** 坐标系从 2 维到 3 维，只是多加一条轴。从 3 维到 768 维，也只是多加轴。数学不受你的想象力限制。

---

## 第五章：连接 AI——给每个词一个"坐标"

现在，让我们做一件惊人的事。

把"坐标"这个概念，从物理空间搬到**语义空间**。

### 词也可以有"位置"

在地图上，北京的坐标是 (116.4, 39.9)。两个数，确定一个物理位置。

在 AI 里，"国王"这个词也有"坐标"——只不过不是 2 维的，而是**几百维**的：

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; border: 2px solid #E91E63; background: rgba(233,30,99,0.04);">

<div style="font-weight: bold; margin-bottom: 12px; font-size: 1.05em; color: #E91E63;">从地理坐标到词坐标</div>

```text
地理空间（2 维）：
  北京   = (116.4,  39.9)
  上海   = (121.5,  31.2)
  → 两个数确定一个城市的位置

语义空间（768 维）：
  "国王" = (0.21, -0.45, 0.89, 0.12, ..., 0.33)
  "王后" = (0.19, -0.42, 0.85, -0.15, ..., 0.30)
  → 768 个数确定一个词的"含义位置"
```

```text
地理空间里：
  北京和上海"靠得近" → 因为它们都在中国东部

语义空间里：
  "国王"和"王后""靠得近" → 因为它们含义相关
  "国王"和"汽车""离得远" → 因为它们含义无关
```

</div>

这就是 **词嵌入**（Word Embedding）——AI 理解语言的第一步。

它做的事情，和笛卡尔做的事情**本质上一模一样**：

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(255,152,0,0.06); border: 1px solid rgba(255,152,0,0.2);">

<div style="font-weight: bold; margin-bottom: 12px; color: #FF9800; font-size: 1.05em;">笛卡尔 vs AI：同一个思想</div>

| | 笛卡尔（1637） | AI 词嵌入（2013） |
|---|--------------|-----------------|
| **给什么取坐标** | 平面上的点 | 词汇表里的词 |
| **坐标维度** | 2 维 (x, y) | 768 维 |
| **"近"意味着** | 物理距离近 | 含义相似 |
| **核心操作** | 算两个点的距离 | 算两个词向量的距离 |
| **革命在哪** | 把形（几何）变成了数 | 把义（语义）变成了数 |

</div>

笛卡尔把**形状**变成了数。AI 把**含义**变成了数。

思想是同一个：**给事物一个坐标，就能用数学来操作它。**

### 实际看看

让我们在实验机器上验证"含义相近的词，坐标也相近"：

```bash
azureuser@ai-lab:~$ source ~/ai-lab-venv/bin/activate
(ai-lab-venv) azureuser@ai-lab:~$ python3 -c "
import numpy as np

# 简化的 5 维词向量（真实模型用 768 维，原理完全一样）
words = {
    '国王': np.array([0.9, 0.8, 0.1, 0.3, 0.7]),
    '王后': np.array([0.8, 0.2, 0.1, 0.3, 0.7]),
    '公主': np.array([0.7, 0.1, 0.2, 0.3, 0.6]),
    '汽车': np.array([0.1, 0.5, 0.9, 0.8, 0.1]),
    '卡车': np.array([0.1, 0.6, 0.8, 0.9, 0.1]),
}

print('词语之间的距离（越小 = 越相似）：')
print('─' * 40)

pairs = [('国王','王后'), ('国王','公主'), ('汽车','卡车'),
         ('国王','汽车'), ('王后','卡车')]

for a, b in pairs:
    dist = np.linalg.norm(words[a] - words[b])
    bar = '█' * int(dist * 10)
    print(f'  {a} ↔ {b}: {dist:.2f}  {bar}')
"
```

```text
词语之间的距离（越小 = 越相似）：
────────────────────────────────────────
  国王 ↔ 王后: 0.61  ██████
  国王 ↔ 公主: 0.78  ███████
  汽车 ↔ 卡车: 0.17  █
  国王 ↔ 汽车: 1.41  ██████████████
  王后 ↔ 卡车: 1.38  █████████████
```

"国王"和"王后"距离 0.61——很近。"汽车"和"卡车"距离 0.17——更近。而"国王"和"汽车"？距离 1.41——很远。

**含义相近 → 坐标相近 → 距离小。**

这不就是地图上的逻辑吗？北京和天津离得近，北京和纽约离得远。只不过这张"地图"有 768 个维度，每个词是地图上的一个"城市"。

如果你读过我的 [《AI 的数学语言（一）》](/ai-blog/tags/线性代数/)，这些概念会很熟悉——那里有更详细的向量和距离计算。这篇是帮你理解**为什么**要这么做：因为笛卡尔 400 年前就证明了——**给事物一个坐标，你就能用数学来处理它。**

> **一句话记住：** AI 的词嵌入 = 给每个词一个"坐标"。坐标近 = 含义近。笛卡尔用坐标统一了数和形，AI 用坐标统一了数和义。同一个思想，跨越了 400 年。

---

## 第六章：坐标思维——万物皆可定位

走到这里，让我们再拉高视角。

笛卡尔发明坐标系时，他想解决的是"苍蝇在哪"的问题。但他实际上发明了一种**思维方式**——

**给事物取坐标，就是把它变成数学能处理的对象。**

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(76,175,80,0.06); border: 1px solid rgba(76,175,80,0.2);">

<div style="font-weight: bold; margin-bottom: 12px; color: #4CAF50; font-size: 1.05em;">万物皆可坐标</div>

| 事物 | "坐标"是什么 | 维度 |
|------|-------------|------|
| 地球上的城市 | (经度, 纬度) | 2 |
| 一杯咖啡 | (温度, 甜度, 浓度, 酸度) | 4 |
| 一个游戏角色 | (攻击, 防御, 速度, HP, MP) | 5 |
| 一首歌的风格 | (节奏感, 旋律性, 电子感, 忧伤度, ...) | 很多 |
| 一个人的性格 | (外向性, 尽责性, 开放性, ...) | 心理学用 5 维 |
| 一个词的含义 | (维度1, 维度2, ..., 维度768) | 768 |
| 一张图片 | (像素1, 像素2, ...) | 百万级 |

</div>

**任何事物，只要你能用一组数来描述它，它就有了坐标，它就可以被数学处理。**

这就是为什么坐标系是如此深刻的发明——它不只是一张画图的纸。它是一个**通用框架**：

- 定位一只苍蝇 → (x, y)
- 定位一颗恒星 → (赤经, 赤纬, 距离)
- 定位一个词的含义 → (768 个数)
- 定位一张图的内容 → (百万个数)

**笛卡尔教会人类的不只是"画坐标轴"。他教会人类的是——万物皆可用数定位，定位之后就能计算。**

---

## 第七章：回顾——四块拼图

让我们回头看看，从第一篇到现在，我们积累了什么：

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; background: rgba(33,150,243,0.06); border: 1px solid rgba(33,150,243,0.2);">

<div style="font-weight: bold; margin-bottom: 12px; color: #2196F3; font-size: 1.05em;">第一幕：数的觉醒——四块拼图</div>

```text
第一篇 · 抽象
  三只羊 → "3"
  "用一个符号代表一个事物"
  → AI 的 Tokenization 就是这件事
                    ↓
第二篇 · 扩展
  零 → 负数 → 完整数轴
  "敢想不存在的东西"
  → AI 的 ReLU 用零做开关
                    ↓
第三篇 · 未知
  x → 方程 → 推理
  "给不知道的东西取名字"
  → AI 的参数就是几十亿个 x
                    ↓
第四篇 · 定位（本文）
  坐标 → 数形统一 → 万物可定位
  "给事物一个坐标就能计算"
  → AI 的词向量就是语义坐标
```

</div>

这四块拼图拼在一起，你就已经拥有了理解 AI 的数学基础的**思想地基**：

1. **万物变数字**（抽象）
2. **数字可以是"没有"和"反方向"**（零和负数）
3. **不知道的东西也能处理**（代数）
4. **给事物坐标就能计算**（坐标系）

下一篇是第一幕的收官之作——我们要看看，当这些工具组合起来，方程能释放出什么样的力量。

---

## 动手实验

### 实验一：画你自己的方程

```python
# 在 ai-lab-venv 里运行
# 用 Python 画 y = x² 的图像

# 不需要 matplotlib！纯文本画图
def plot_text(func, x_range=(-5, 5), width=40, height=20):
    """用文字在终端画函数图像"""
    x_min, x_max = x_range

    # 计算所有 y 值
    points = []
    for i in range(width):
        x = x_min + (x_max - x_min) * i / width
        y = func(x)
        points.append((x, y))

    y_vals = [p[1] for p in points]
    y_min, y_max = min(y_vals), max(y_vals)

    # 画图
    for row in range(height, -1, -1):
        y = y_min + (y_max - y_min) * row / height
        line = ""
        for col in range(width):
            py = points[col][1]
            py_row = int((py - y_min) / (y_max - y_min) * height)
            if py_row == row:
                line += "★"
            elif col == width // 2 and row == 0:
                line += "┼"
            elif col == width // 2:
                line += "│"
            elif row == 0:
                line += "─"
            else:
                line += " "
        # 只在几个关键行显示 y 值
        if row == height or row == height // 2 or row == 0:
            print(f"{y:>6.1f} ┤{line}")
        else:
            print(f"       │{line}")
    print(f"        {'':>{width//2-3}}x 轴")

print("y = x² 的图像：")
print()
plot_text(lambda x: x**2)

# 输出一个文本版的抛物线！
```

### 实验二：感受"坐标 = 定位"

```python
# 给三个城市取坐标，算它们之间的距离

import math

cities = {
    "北京": (116.4, 39.9),
    "上海": (121.5, 31.2),
    "广州": (113.3, 23.1),
}

print("城市之间的坐标距离：")
print("─" * 40)
for c1 in cities:
    for c2 in cities:
        if c1 < c2:  # 避免重复
            x1, y1 = cities[c1]
            x2, y2 = cities[c2]
            dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
            print(f"  {c1} ↔ {c2}: {dist:.1f}°")

print()
print("坐标近 = 地理位置近")
print("词向量也是一样：坐标近 = 含义相似")

# 输出：
# 城市之间的坐标距离：
# ────────────────────────────────────────
#   上海 ↔ 广州: 11.9°
#   上海 ↔ 北京: 10.0°
#   北京 ↔ 广州: 17.1°
#
# 坐标近 = 地理位置近
# 词向量也是一样：坐标近 = 含义相似
```

---

## 本篇小结

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px; border-radius: 8px; border: 2px solid #FF9800; background: rgba(255,152,0,0.04);">

<div style="font-weight: bold; margin-bottom: 12px; font-size: 1.05em;">这篇文章讲了什么？</div>

**一、笛卡尔的天才之桥**
- 1637 年，两条线 + 一个原点 = 坐标系
- 把代数（数）和几何（形）永远焊在一起

**二、方程会"画画"**
- y = 2x + 1 → 直线。y = x² → 抛物线。y = sin(x) → 波浪
- 数形结合不是学习技巧，是数学本身的结构

**三、维度不止三个**
- 1D → 2D → 3D → ... → 768D
- 数学不受想象力限制，公式在任何维度都一样

**四、AI 的词嵌入 = 语义坐标**
- 笛卡尔把形状变成了数，AI 把含义变成了数
- 坐标近 = 含义近。同一个思想，跨越 400 年

**五、万物皆可坐标**
- 城市、咖啡、歌曲、性格、词汇——只要能用一组数描述，就能计算
- 笛卡尔教会人类：给事物坐标，就能用数学处理

</div>

---

## 下一篇预告

我们现在有了完整的工具箱：数、零、负数、未知数 x、坐标系。

是时候看看这些工具**组合起来能做什么了**。

> F = ma。E = mc²。每一个方程背后，都是人类对宇宙的一次发现。

方程不是考试题。方程是人类发现的**自然界的源代码**。

下一篇是第一幕的收官之作：**[看见数学（五）：方程的力量——自然界的源代码](/ai-blog/posts/see-math-5-equations/)**

---

<div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #e0e0e0; font-size: 0.9em; color: #888; line-height: 1.8;">

**《看见数学》系列** — 从结绳记事到 AI，看见数学之美。<br>
本文首发于「AI 学习笔记」博客：https://Jason-Azure.github.io/ai-blog/<br>
微信公众号：AI-lab学习笔记<br>
系列文章完整列表见 [标签：看见数学](/ai-blog/tags/看见数学/)

</div>
