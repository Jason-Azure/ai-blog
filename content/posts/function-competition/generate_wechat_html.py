#!/usr/bin/env python3
"""
Generate WeChat HTML article for 函数的竞赛.
Reads _img_data.py for base64 image URIs, outputs ~/wechat-articles/42-function-competition.html
"""

import os
import sys

# Load image data
img_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_img_data.py')
img_ns = {}
exec(open(img_data_path).read(), img_ns)
IMGS = img_ns['IMGS']

# Chapter title colors rotate
COLORS = ['#FF9800', '#2196F3', '#4CAF50', '#9C27B0', '#607D8B', '#E91E63']

def img(name, alt=""):
    return f'<img src="{IMGS[name]}" alt="{alt}" style="max-width: 100%; height: auto; border-radius: 6px; margin: 10px 0; display: block;">'

def h2(title, color_idx):
    c = COLORS[color_idx % len(COLORS)]
    return f'<h2 style="font-size: 20px; font-weight: bold; color: #333; margin-top: 30px; margin-bottom: 15px; padding-left: 12px; border-left: 4px solid {c};">{title}</h2>'

def p(text):
    return f'<p style="margin: 15px 0; font-size: 16px; line-height: 1.75; color: #333;">{text}</p>'

def blockquote(text):
    return f'''<section style="background: rgba(96,125,139,0.06); border-left: 4px solid #607D8B; padding: 12px 16px; margin: 20px 0; border-radius: 0 4px 4px 0;">
<p style="margin: 0; font-size: 15px; line-height: 1.75; color: #555;">{text}</p>
</section>'''

def insight_box(text, color='#4CAF50'):
    r, g, b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
    return f'''<section style="background: rgba({r},{g},{b},0.06); border-left: 4px solid {color}; padding: 12px 16px; margin: 20px 0; border-radius: 0 4px 4px 0;">
<p style="margin: 0; font-size: 15px; line-height: 1.75; color: #555;">{text}</p>
</section>'''

def colored_box(text, color, title=None):
    r, g, b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
    title_html = f'<p style="font-weight: bold; margin-bottom: 12px; color: {color}; font-size: 1.05em;">{title}</p>' if title else ''
    return f'''<section style="background: rgba({r},{g},{b},0.06); border: 1px solid rgba({r},{g},{b},0.2); border-radius: 8px; padding: 15px 20px; margin: 20px 0;">
{title_html}{text}
</section>'''

def pre_block(lines, bg='#f6f8fa'):
    """Build a pre block with br/ at end of each line (except last)."""
    br_lines = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            br_lines.append(line + '<br/>')
        else:
            br_lines.append(line)
    content = '\n'.join(br_lines)
    return f'<pre style="background: {bg}; padding: 12px 15px; border-radius: 6px; font-family: \'Courier New\', Consolas, monospace; font-size: 14px; line-height: 1.6; overflow-x: auto; border: 1px solid #e0e0e0; color: #333;">{content}</pre>'

def hr():
    return '<section style="border-top: 1px solid #e0e0e0; margin: 25px 0;"></section>'

def td(text, is_header=False):
    tag = 'th' if is_header else 'td'
    bg = ' background-color: #f9f9f9; font-weight: bold;' if is_header else ''
    return f'<{tag} style="padding: 8px 12px; border: 1px solid #e0e0e0; color: #333;{bg}">{text}</{tag}>'


# ============================================================
# Build HTML
# ============================================================

html_parts = []

# Header
html_parts.append('''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>函数的竞赛：人类试过的所有方法，和神经网络胜出的原因</title>
</head>
<body style="margin: 0; padding: 0; background-color: #f5f5f5;">
<section style="max-width: 100%; margin: 0 auto; padding: 20px; font-family: -apple-system, BlinkMacSystemFont, \'Helvetica Neue\', \'PingFang SC\', \'Microsoft YaHei\', sans-serif; font-size: 16px; line-height: 1.75; color: #333; background-color: #ffffff;">''')

# Title
html_parts.append('<h1 style="font-size: 24px; font-weight: bold; text-align: center; color: #333; margin-bottom: 10px; padding-bottom: 10px;">函数的竞赛：人类试过的所有方法，和神经网络胜出的原因</h1>')

# Subtitle
html_parts.append('<p style="text-align: center; font-size: 14px; color: #999; margin-bottom: 25px; padding-bottom: 15px; border-bottom: 2px solid #e0e0e0;">人类 400 年来发明了无数拟合函数的方法&mdash;&mdash;泰勒级数、傅里叶级数、多项式、样条、核方法。每一种都精妙绝伦。但当任务变成&ldquo;在万亿维空间中学习生成规律&rdquo;，只有一个选手能站到终点。这不是选择题，这是淘汰赛。</p>')

# ---- Navigation overview (gradient intro box) ----
html_parts.append('''<section style="max-width: 100%; margin: 20px 0; padding: 20px 24px; border-radius: 10px; background: linear-gradient(135deg, rgba(233,30,99,0.06), rgba(33,150,243,0.06)); border: 1px solid rgba(233,30,99,0.15);">
<p style="font-weight: bold; margin-bottom: 10px; color: #E91E63; font-size: 1.1em;">&#x1F4D6; 导读</p>
<p style="margin: 10px 0; font-size: 16px; line-height: 1.75; color: #333;">这不是一篇&ldquo;什么是神经网络&rdquo;的科普。</p>
<p style="margin: 10px 0; font-size: 16px; line-height: 1.75; color: #333;">这篇文章要回答的问题是：<strong>数学世界里有那么多精妙的工具，凭什么偏偏选了神经网络来做 AI？</strong></p>
<p style="margin: 10px 0; font-size: 16px; line-height: 1.75; color: #333;">我们将检阅人类 400 年来发明的函数拟合方法&mdash;&mdash;泰勒级数、傅里叶级数、多项式插值、样条曲线、核方法&mdash;&mdash;像一场淘汰赛一样，逐一看清它们的优势与致命缺陷。最后你会发现：<strong>不是人类&ldquo;选择&rdquo;了神经网络，而是只有神经网络满足所有条件。</strong></p>
<p style="margin: 10px 0; font-size: 16px; line-height: 1.75; color: #333;">灵感来源：Emergent Garden 的精彩视频 <em>Watching Neural Networks Learn</em>。</p>
<p style="font-size: 0.9em; color: #888; margin-top: 12px; line-height: 1.7;">&① 万物皆是函数 &rarr; &② 泰勒级数 &rarr; &③ 傅里叶级数 &rarr; &④ 多项式与样条 &rarr; &⑤ 核方法与 SVM &rarr; &⑥ 神经网络 &rarr; &⑦ 终极对比</p>
</section>''')

html_parts.append(hr())

# ================================================================
# Chapter 1: 万物皆是函数
# ================================================================
html_parts.append(h2('第一章：万物皆是函数 &#x1F3A8;', 0))

html_parts.append(p('你拍一张照片，手机里发生了什么？'))
html_parts.append(p('<strong>每个像素</strong>接收一个坐标 (x, y)，输出一个颜色 (R, G, B)。这就是一个函数：'))
html_parts.append(blockquote('f(x, y) &rarr; (R, G, B)'))
html_parts.append(p('你问 ChatGPT 一个问题，它做了什么？'))
html_parts.append(p('<strong>接收一串文字</strong>（上文），输出下一个最可能的词。这也是一个函数：'))
html_parts.append(blockquote('f(&ldquo;今天天气&rdquo;) &rarr; &ldquo;很好&rdquo;'))
html_parts.append(p('天气预报、股票预测、医学诊断、自动驾驶&mdash;&mdash;<strong>所有这些任务，本质上都是在求解一个函数</strong>。'))

html_parts.append(img('function-blackbox.png', '万物皆函数：输入→黑盒→输出'))

html_parts.append(p('问题来了：<strong>这个函数 f，我们不知道它长什么样。</strong>'))
html_parts.append(p('我们只有一堆输入-输出的样本（数据），需要找到一个函数来&ldquo;拟合&rdquo;这些数据&mdash;&mdash;让它在没见过的输入上也能给出合理的输出。'))
html_parts.append(p('<strong>这就是函数拟合问题。人类为此探索了 400 年。</strong>'))

html_parts.append(colored_box(
    p('<strong>关键设问：</strong> 400 年来，数学家发明了各种精妙的方法来逼近未知函数。泰勒、傅里叶、拉格朗日、贝塞尔、SVM&hellip;&hellip;每一种都在自己的领域里璀璨夺目。但当我们需要一个&ldquo;通用学习机器&rdquo;时，为什么最终胜出的是神经网络？') +
    p('让我们一个一个来看。'),
    '#FF9800'
))

html_parts.append(hr())

# ================================================================
# Chapter 2: 泰勒级数
# ================================================================
html_parts.append(h2('第二章：泰勒级数&mdash;&mdash;局部的完美主义者 &#x1F535;', 1))

html_parts.append('<h3 style="font-size: 17px; font-weight: bold; color: #555; margin-top: 25px; margin-bottom: 10px;">1715 年的天才想法</h3>')
html_parts.append(p('布鲁克&middot;泰勒（Brook Taylor）在 1715 年提出了一个优美的想法：'))
html_parts.append(blockquote('<strong>在一个点附近，任何&ldquo;光滑&rdquo;的函数都可以用多项式来逼近。</strong>'))

html_parts.append(p('公式长这样：'))

html_parts.append('''<section style="max-width: 100%; margin: 15px 0; padding: 15px 20px; border-radius: 8px; background: rgba(33,150,243,0.06); border: 1px solid rgba(33,150,243,0.15); text-align: center; font-size: 1.05em;">
<p style="margin: 0; color: #333;">f(x) &asymp; f(a) + f&prime;(a)(x&minus;a) + f&Prime;(a)(x&minus;a)&sup2;/2! + f&prime;&prime;&prime;(a)(x&minus;a)&sup3;/3! + &hellip;</p>
</section>''')

html_parts.append(p('直觉翻译：<strong>站在点 a 上，用这个点的函数值、斜率、曲率&hellip;&hellip;一层层叠加，像搭积木一样拼出函数的形状。</strong>'))
html_parts.append(p('阶数越高，逼近越精确&mdash;&mdash;至少在 a 附近是这样。'))

html_parts.append('<h3 style="font-size: 17px; font-weight: bold; color: #555; margin-top: 25px; margin-bottom: 10px;">看看效果</h3>')

html_parts.append(img('taylor-approximation.gif', '泰勒级数逼近 sin(x)：阶数增加时从中心扩展，但远处发散'))

html_parts.append(p('动图展示了 sin(x) 的泰勒展开从 1 阶到 15 阶的过程。注意看：'))
html_parts.append(p('&bull; <strong>绿色区域</strong>（收敛区）里，逼近精度惊人'))
html_parts.append(p('&bull; <strong>离开中心点越远</strong>，曲线开始疯狂偏离'))
html_parts.append(p('&bull; 15 阶时，中心附近已经完美重合，但两端飞到了天上'))

html_parts.append('<h3 style="font-size: 17px; font-weight: bold; color: #555; margin-top: 25px; margin-bottom: 10px;">泰勒的成绩单</h3>')

taylor_report = '''<p style="margin: 10px 0; font-size: 16px; line-height: 1.75; color: #333;">&check; <strong>优点：</strong></p>
<p style="margin: 5px 0 5px 15px; font-size: 15px; line-height: 1.75; color: #333;">&bull; 数学优美，推导简洁</p>
<p style="margin: 5px 0 5px 15px; font-size: 15px; line-height: 1.75; color: #333;">&bull; 在展开点附近精度极高</p>
<p style="margin: 5px 0 5px 15px; font-size: 15px; line-height: 1.75; color: #333;">&bull; 物理学的核心工具（力学、电磁学、量子力学处处用到）</p>
<p style="margin: 5px 0 5px 15px; font-size: 15px; line-height: 1.75; color: #333;">&bull; 可以用有限的导数信息重建函数</p>
<p style="margin: 15px 0 10px 0; font-size: 16px; line-height: 1.75; color: #333;">&cross; <strong>致命缺陷：</strong></p>
<p style="margin: 5px 0 5px 15px; font-size: 15px; line-height: 1.75; color: #333;">&bull; <strong>收敛半径有限</strong>&mdash;&mdash;离开展开点就崩溃</p>
<p style="margin: 5px 0 5px 15px; font-size: 15px; line-height: 1.75; color: #333;">&bull; <strong>全局拟合无能为力</strong>&mdash;&mdash;想逼近一个定义在整个实数轴上的函数？没门</p>
<p style="margin: 5px 0 5px 15px; font-size: 15px; line-height: 1.75; color: #333;">&bull; <strong>高维扩展困难</strong>&mdash;&mdash;二维的泰勒展开已经很复杂，万维？不可能</p>'''

html_parts.append(colored_box(taylor_report, '#2196F3'))

html_parts.append(p('泰勒级数是&ldquo;局部思维&rdquo;的极致。它像一个显微镜&mdash;&mdash;在一个点上看得无比清晰，但视野极其有限。'))
html_parts.append(insight_box('<strong>对 LLM 来说</strong>：语言模型需要理解万亿维度的全局规律，而泰勒只能看一个点的邻域。第一个选手，淘汰。', '#E91E63'))

html_parts.append(hr())

# ================================================================
# Chapter 3: 傅里叶级数
# ================================================================
html_parts.append(h2('第三章：傅里叶级数&mdash;&mdash;频率的魔法师 &#x1F7E2;', 2))

html_parts.append('<h3 style="font-size: 17px; font-weight: bold; color: #555; margin-top: 25px; margin-bottom: 10px;">1807 年的革命</h3>')
html_parts.append(p('约瑟夫&middot;傅里叶（Joseph Fourier）在研究热传导时，发现了一个惊人的事实：'))
html_parts.append(blockquote('<strong>任何周期函数，都可以写成正弦波和余弦波的叠加。</strong>'))
html_parts.append(p('这听起来不可思议&mdash;&mdash;<strong>一个锯齿形的方波，竟然能用光滑的正弦波拼出来？</strong>'))
html_parts.append(p('能！只要你愿意叠加足够多的波。'))

html_parts.append('<h3 style="font-size: 17px; font-weight: bold; color: #555; margin-top: 25px; margin-bottom: 10px;">与泰勒的本质区别</h3>')
html_parts.append(p('泰勒在一个&ldquo;点&rdquo;附近展开，傅里叶在&ldquo;全局&rdquo;用波去拼。这是两种完全不同的哲学：'))

html_parts.append(pre_block([
    '泰勒：站在一个点，向外扩张      &rarr; 局部 &rarr; 全局（常常失败）',
    '傅里叶：用全局的波，拼出细节    &rarr; 全局 &rarr; 局部（通过高频）',
]))

html_parts.append('<h3 style="font-size: 17px; font-weight: bold; color: #555; margin-top: 25px; margin-bottom: 10px;">看看效果</h3>')

html_parts.append(img('fourier-gibbs.gif', '方波的傅里叶逼近：注意阶跃处永远消不掉的过冲（吉布斯现象）'))

html_parts.append(p('动图展示了方波的傅里叶逼近从 1 项到 50 项的过程。注意看：'))
html_parts.append(p('&bull; 随着项数增加，整体形状越来越接近方波'))
html_parts.append(p('&bull; <strong>但在跳变点处，总有一个约 9% 的过冲永远消不掉</strong>&mdash;&mdash;这就是著名的<strong>吉布斯现象（Gibbs Phenomenon）</strong>'))
html_parts.append(p('&bull; 即使用无穷多项，跳变点的过冲也不会消失！'))

html_parts.append('<h3 style="font-size: 17px; font-weight: bold; color: #555; margin-top: 25px; margin-bottom: 10px;">傅里叶的成绩单</h3>')

fourier_report = '''<p style="margin: 10px 0; font-size: 16px; line-height: 1.75; color: #333;">&check; <strong>优点：</strong></p>
<p style="margin: 5px 0 5px 15px; font-size: 15px; line-height: 1.75; color: #333;">&bull; <strong>全局逼近</strong>&mdash;&mdash;不像泰勒那样局限于一个点</p>
<p style="margin: 5px 0 5px 15px; font-size: 15px; line-height: 1.75; color: #333;">&bull; 信号处理的基石&mdash;&mdash;MP3、JPEG、5G 通信、MRI 成像全靠它</p>
<p style="margin: 5px 0 5px 15px; font-size: 15px; line-height: 1.75; color: #333;">&bull; 数学理论完备（Parseval 定理、卷积定理）</p>
<p style="margin: 5px 0 5px 15px; font-size: 15px; line-height: 1.75; color: #333;">&bull; 快速算法（FFT）使大规模计算成为可能</p>
<p style="margin: 15px 0 10px 0; font-size: 16px; line-height: 1.75; color: #333;">&cross; <strong>致命缺陷：</strong></p>
<p style="margin: 5px 0 5px 15px; font-size: 15px; line-height: 1.75; color: #333;">&bull; <strong>吉布斯现象</strong>&mdash;&mdash;对不连续函数永远有过冲</p>
<p style="margin: 5px 0 5px 15px; font-size: 15px; line-height: 1.75; color: #333;">&bull; <strong>高维失效</strong>&mdash;&mdash;从 1D 到 1000D，需要的基函数数量指数爆炸</p>
<p style="margin: 5px 0 5px 15px; font-size: 15px; line-height: 1.75; color: #333;">&bull; <strong>不能自动学习</strong>&mdash;&mdash;基函数（sin/cos）是固定的，参数需要解析计算</p>
<p style="margin: 5px 0 5px 15px; font-size: 15px; line-height: 1.75; color: #333;">&bull; <strong>非周期信号需要拓展处理</strong>（DFT/STFT/小波）</p>'''

html_parts.append(colored_box(fourier_report, '#4CAF50'))

html_parts.append('<h3 style="font-size: 17px; font-weight: bold; color: #555; margin-top: 25px; margin-bottom: 10px;">一个来自视频的关键洞察：频谱偏差</h3>')
html_parts.append(p('Emergent Garden 的视频中展示了一个有趣现象：<strong>神经网络在学习目标函数时，总是先学会低频成分，再慢慢学习高频细节。</strong> 这被称为&ldquo;频谱偏差（Spectral Bias）&rdquo;。'))
html_parts.append(p('这恰好说明了傅里叶视角的价值&mdash;&mdash;即使在神经网络内部，频率依然是理解学习过程的关键语言。傅里叶没有赢得比赛，但它的思想渗透在了赢家的每一步训练中。'))
html_parts.append(insight_box('<strong>对 LLM 来说</strong>：语言不是周期信号，文本的&ldquo;维度&rdquo;是词汇表大小（数万到十万维），傅里叶的基函数数量会爆炸。第二个选手，淘汰。', '#E91E63'))

html_parts.append(hr())

# ================================================================
# Chapter 4: 多项式与样条
# ================================================================
html_parts.append(h2('第四章：多项式与样条&mdash;&mdash;曲线的裁缝 &#x1F7E3;', 3))

html_parts.append('<h3 style="font-size: 17px; font-weight: bold; color: #555; margin-top: 25px; margin-bottom: 10px;">多项式插值：精确但危险</h3>')
html_parts.append(p('拉格朗日（Lagrange）证明了一个优美的定理：'))
html_parts.append(blockquote('<strong>n 个数据点，恰好能唯一确定一个 n&minus;1 次多项式通过所有点。</strong>'))
html_parts.append(p('这听起来完美&mdash;&mdash;有多少数据就用多高的多项式，精确通过每一个点。但问题来了&hellip;&hellip;'))

html_parts.append('<h3 style="font-size: 17px; font-weight: bold; color: #555; margin-top: 25px; margin-bottom: 10px;">龙格现象：多项式的噩梦</h3>')
html_parts.append(p('1901 年，卡尔&middot;龙格（Carl Runge）用一个简单的函数 f(x) = 1/(1+25x&sup2;) 击碎了高阶多项式的美梦：'))
html_parts.append(blockquote('<strong>当插值点数增加时，多项式在边缘处疯狂振荡，误差不减反增！</strong>'))

html_parts.append(img('runge-vs-spline.gif', '龙格现象：多项式拟合（左）vs 样条拟合（右）'))

html_parts.append(p('动图从 3 个点到 21 个点，对比两种方法：'))
html_parts.append(p('&bull; <strong>左图（多项式）</strong>：随着点数增加，边缘振荡越来越剧烈，完全失控'))
html_parts.append(p('&bull; <strong>右图（样条）</strong>：始终平稳地贴合原函数，没有失控'))

html_parts.append('<h3 style="font-size: 17px; font-weight: bold; color: #555; margin-top: 25px; margin-bottom: 10px;">样条的智慧：分而治之</h3>')
html_parts.append(p('样条曲线（Spline）的思路极其朴素：'))
html_parts.append(blockquote('<strong>别用一条高阶多项式通吃，把曲线切成小段，每段用低阶多项式（通常是三次），接合处保证光滑。</strong>'))
html_parts.append(p('这就像一个好裁缝&mdash;&mdash;不用一整块布裁出衣服，而是分片裁剪再缝合。每一片都简单可控，缝合处平滑自然。'))
html_parts.append(p('贝塞尔曲线（B&eacute;zier Curve）是样条思想的明星应用：'))
html_parts.append(p('&bull; <strong>Photoshop</strong> 的钢笔工具'))
html_parts.append(p('&bull; <strong>字体设计</strong>（TrueType/OpenType 字体的每个字母）'))
html_parts.append(p('&bull; <strong>工业设计</strong>（汽车曲面、飞机机翼）'))
html_parts.append(p('&bull; <strong>动画</strong>（运动路径插值）'))

html_parts.append('<h3 style="font-size: 17px; font-weight: bold; color: #555; margin-top: 25px; margin-bottom: 10px;">多项式与样条的成绩单</h3>')

spline_report = '''<p style="margin: 10px 0; font-size: 16px; line-height: 1.75; color: #333;">&check; <strong>优点：</strong></p>
<p style="margin: 5px 0 5px 15px; font-size: 15px; line-height: 1.75; color: #333;">&bull; 样条拟合稳定，没有龙格现象</p>
<p style="margin: 5px 0 5px 15px; font-size: 15px; line-height: 1.75; color: #333;">&bull; 在 2D/3D 曲线拟合中无可替代</p>
<p style="margin: 5px 0 5px 15px; font-size: 15px; line-height: 1.75; color: #333;">&bull; 计算高效，理论成熟</p>
<p style="margin: 5px 0 5px 15px; font-size: 15px; line-height: 1.75; color: #333;">&bull; 工业设计和计算机图形学的基石</p>
<p style="margin: 15px 0 10px 0; font-size: 16px; line-height: 1.75; color: #333;">&cross; <strong>致命缺陷：</strong></p>
<p style="margin: 5px 0 5px 15px; font-size: 15px; line-height: 1.75; color: #333;">&bull; <strong>维度诅咒</strong>&mdash;&mdash;从 2D 到 1000D，需要的控制点数量指数爆炸</p>
<p style="margin: 5px 0 5px 20px; font-size: 14px; line-height: 1.75; color: #666;">2D 曲面：100&times;100 = 10,000 个控制点</p>
<p style="margin: 5px 0 5px 20px; font-size: 14px; line-height: 1.75; color: #666;">100D：100<sup>100</sup> = 10<sup>200</sup> 个控制点&mdash;&mdash;比宇宙原子数还多</p>
<p style="margin: 5px 0 5px 15px; font-size: 15px; line-height: 1.75; color: #333;">&bull; <strong>不能自动学习</strong>&mdash;&mdash;控制点位置需要人工指定或预设</p>
<p style="margin: 5px 0 5px 15px; font-size: 15px; line-height: 1.75; color: #333;">&bull; <strong>不能做生成</strong>&mdash;&mdash;它只能内插，不能创造新数据</p>'''

html_parts.append(colored_box(spline_report, '#9C27B0'))

html_parts.append(insight_box('<strong>对 LLM 来说</strong>：GPT-4 的输入空间是 128,000 个 token &times; 100,000 词汇 = 天文数字维度。样条在这个维度下需要的参数量超出宇宙能承载的范围。第三个选手，淘汰。', '#E91E63'))

html_parts.append(hr())

# ================================================================
# Chapter 5: 核方法与 SVM
# ================================================================
html_parts.append(h2('第五章：核方法与 SVM&mdash;&mdash;高维的魔术 &#x1F518;', 4))

html_parts.append('<h3 style="font-size: 17px; font-weight: bold; color: #555; margin-top: 25px; margin-bottom: 10px;">核技巧：天才的迂回</h3>')
html_parts.append(p('到了 1990 年代，机器学习的明星是支持向量机（SVM）。它的核心思想极其巧妙：'))
html_parts.append(blockquote('<strong>在原始空间中无法线性分类的数据，映射到更高维的空间后，可能就能用一个平面一刀切开。</strong>'))
html_parts.append(p('举个例子：'))

html_parts.append(img('kernel-svm.png', '核方法：2D 不可分数据 → 映射到 3D → 超平面分类'))

html_parts.append(p('&bull; <strong>左图</strong>：二维平面上，两类数据（蓝色和粉色）套在一起，画不出一条直线分开它们'))
html_parts.append(p('&bull; <strong>中间</strong>：核技巧把 (x&#x2081;, x&#x2082;) 映射到 (x&#x2081;, x&#x2082;, x&#x2081;&sup2;+x&#x2082;&sup2;)，加了一个维度'))
html_parts.append(p('&bull; <strong>右图</strong>：在三维空间中，两类数据被一个平面（绿色）干净利落地分开了'))
html_parts.append(p('这就是&ldquo;核技巧（Kernel Trick）&rdquo;&mdash;&mdash;用数学上的映射代替真正的高维计算，优雅而高效。'))

html_parts.append('<h3 style="font-size: 17px; font-weight: bold; color: #555; margin-top: 25px; margin-bottom: 10px;">SVM 的黄金时代</h3>')
html_parts.append(p('在 2000 年代，SVM 统治了机器学习竞赛。它有严格的数学基础（统计学习理论、VC 维），在手写数字识别、文本分类等任务上表现优异。'))

html_parts.append('<h3 style="font-size: 17px; font-weight: bold; color: #555; margin-top: 25px; margin-bottom: 10px;">核方法的成绩单</h3>')

kernel_report = '''<p style="margin: 10px 0; font-size: 16px; line-height: 1.75; color: #333;">&check; <strong>优点：</strong></p>
<p style="margin: 5px 0 5px 15px; font-size: 15px; line-height: 1.75; color: #333;">&bull; 理论优雅，有坚实的数学保障（最大间隔、泛化边界）</p>
<p style="margin: 5px 0 5px 15px; font-size: 15px; line-height: 1.75; color: #333;">&bull; 小数据上表现好，不容易过拟合</p>
<p style="margin: 5px 0 5px 15px; font-size: 15px; line-height: 1.75; color: #333;">&bull; 可解释性强（支持向量就是决策依据）</p>
<p style="margin: 5px 0 5px 15px; font-size: 15px; line-height: 1.75; color: #333;">&bull; 核技巧避免了真正的高维计算</p>
<p style="margin: 15px 0 10px 0; font-size: 16px; line-height: 1.75; color: #333;">&cross; <strong>致命缺陷：</strong></p>
<p style="margin: 5px 0 5px 15px; font-size: 15px; line-height: 1.75; color: #333;">&bull; <strong>计算复杂度 O(n&sup2;)~O(n&sup3;)</strong>&mdash;&mdash;数据量超过十万就崩溃</p>
<p style="margin: 5px 0 5px 15px; font-size: 15px; line-height: 1.75; color: #333;">&bull; <strong>不能做&ldquo;生成&rdquo;</strong>&mdash;&mdash;SVM 只能分类和回归，不能输出一段话、一张图</p>
<p style="margin: 5px 0 5px 15px; font-size: 15px; line-height: 1.75; color: #333;">&bull; <strong>不能端到端学习特征</strong>&mdash;&mdash;需要人工设计特征（特征工程），模型本身不学特征</p>
<p style="margin: 5px 0 5px 15px; font-size: 15px; line-height: 1.75; color: #333;">&bull; <strong>不能增量学习</strong>&mdash;&mdash;新数据来了要重新训练全部</p>'''

html_parts.append(colored_box(kernel_report, '#607D8B'))

html_parts.append('<h3 style="font-size: 17px; font-weight: bold; color: #555; margin-top: 25px; margin-bottom: 10px;">关键对比：分类 vs 生成</h3>')
html_parts.append(p('这是核方法被淘汰的最根本原因&mdash;&mdash;<strong>LLM 需要的是生成，不是分类。</strong>'))

html_parts.append(pre_block([
    'SVM 做的事：    输入一封邮件  &rarr; 输出一个标签（垃圾/非垃圾）',
    'LLM 做的事：    输入一段话    &rarr; 输出下一段话（创造新内容）',
]))

html_parts.append(p('分类是从有限选项中选一个。生成是在无穷可能中创造一个。'))
html_parts.append(p('SVM 是一个优秀的裁判，但它不会写诗。'))
html_parts.append(insight_box('<strong>对 LLM 来说</strong>：训练数据是数万亿 token，SVM 的 O(n&sup3;) 复杂度让它连启动都做不到。更根本的是，SVM 不能生成。第四个选手，淘汰。', '#E91E63'))

html_parts.append(hr())

# ================================================================
# Chapter 6: 神经网络
# ================================================================
html_parts.append(h2('第六章：神经网络&mdash;&mdash;为什么是它？ &#x1F497;', 5))

html_parts.append(p('四个选手全部淘汰。现在我们来看最后一个&mdash;&mdash;<strong>神经网络</strong>。'))

html_parts.append('<h3 style="font-size: 17px; font-weight: bold; color: #555; margin-top: 25px; margin-bottom: 10px;">万能逼近定理：理论底气</h3>')
html_parts.append(p('1989 年，Cybenko 和 Hornik 分别证明了：'))
html_parts.append(blockquote('<strong>只要一个隐藏层足够宽，带非线性激活函数的前馈神经网络可以逼近任何连续函数。</strong>'))
html_parts.append(p('这就是<strong>万能逼近定理（Universal Approximation Theorem）</strong>。它给了神经网络一张理论&ldquo;入场券&rdquo;&mdash;&mdash;任何你想拟合的函数，原则上它都能拟合。'))
html_parts.append(p('但这还不够。泰勒级数也能逼近任何光滑函数&mdash;&mdash;理论上可以，实际上做不到。神经网络凭什么不一样？'))

html_parts.append('<h3 style="font-size: 17px; font-weight: bold; color: #555; margin-top: 25px; margin-bottom: 10px;">看看神经网络怎么学的</h3>')

html_parts.append(img('nn-learning.gif', '神经网络的学习过程：先捕获低频特征，再逐步精炼高频细节'))

html_parts.append(p('动图展示了一个神经网络从随机初始化到拟合复杂函数的过程：'))
html_parts.append(p('&bull; <strong>Step 0</strong>：一条杂乱的噪声线'))
html_parts.append(p('&bull; <strong>Step 200</strong>：已经学到了函数的&ldquo;大致走向&rdquo;（低频成分）'))
html_parts.append(p('&bull; <strong>Step 2000</strong>：开始捕获高频细节'))
html_parts.append(p('&bull; <strong>Step 5000</strong>：几乎完美重合'))
html_parts.append(p('注意 <strong>频谱偏差（Spectral Bias）</strong>：网络先学低频、再学高频。这不是缺点&mdash;&mdash;这是一种<strong>隐式正则化</strong>，帮助模型避免过拟合，先抓本质规律再抓细节。'))

html_parts.append('<h3 style="font-size: 17px; font-weight: bold; color: #555; margin-top: 25px; margin-bottom: 10px;">五个独一无二的优势</h3>')
html_parts.append(p('为什么前面四个选手做不到的事，神经网络做到了？因为它同时具备了五个关键特性：'))

nn_advantages = '''<p style="margin: 15px 0 8px 0; font-size: 16px; line-height: 1.75; color: #333;"><strong>&#10102; 可扩展性（Scalability）</strong></p>
<p style="margin: 5px 0 5px 0; font-size: 15px; line-height: 1.75; color: #333;">参数量可以从 4,000（microgpt）到 1.8 万亿（GPT-4），性能随规模平滑提升。这就是 <strong>Scaling Laws</strong>&mdash;&mdash;不是&ldquo;越大越好&rdquo;的经验主义，而是有数学规律的幂律关系。</p>
<p style="margin: 5px 0 5px 0; font-size: 15px; line-height: 1.75; color: #333;">其他方法？泰勒加阶数只在局部有用，傅里叶加项数会遇到吉布斯现象，多项式加阶数会遇到龙格现象，SVM 加数据会遇到 O(n&sup3;) 的墙。</p>

<p style="margin: 20px 0 8px 0; font-size: 16px; line-height: 1.75; color: #333;"><strong>&#10103; 自动学习（Automatic Learning）</strong></p>
<p style="margin: 5px 0 5px 0; font-size: 15px; line-height: 1.75; color: #333;">泰勒需要你手动算导数，傅里叶需要你解析计算系数，样条需要你选择控制点，SVM 需要你设计特征。</p>
<p style="margin: 5px 0 5px 0; font-size: 15px; line-height: 1.75; color: #333;">神经网络？<strong>给它数据和一个损失函数，梯度下降自动找到所有参数。</strong> 不需要人设计基函数，不需要先验知识，不需要手工特征工程。</p>

<p style="margin: 20px 0 8px 0; font-size: 16px; line-height: 1.75; color: #333;"><strong>&#10104; 高维友好（High-Dimensional Friendly）</strong></p>
<p style="margin: 5px 0 5px 0; font-size: 15px; line-height: 1.75; color: #333;">这是最关键的一点。前面每个方法都被&ldquo;维度诅咒&rdquo;击败了。神经网络为什么能绕过？</p>
<p style="margin: 5px 0 5px 0; font-size: 15px; line-height: 1.75; color: #333;">因为<strong>真实数据不是均匀分布在整个高维空间中的，而是集中在低维流形（manifold）上</strong>。想象一张照片的所有像素值&mdash;&mdash;理论上有 256<sup>百万</sup> 种组合，但真正有意义的图片只占极小的一部分。</p>
<p style="margin: 5px 0 5px 0; font-size: 15px; line-height: 1.75; color: #333;">神经网络通过层层变换，自动发现这些低维结构。再加上<strong>参数共享</strong>（卷积网络共享卷积核，Transformer 共享注意力头），参数量远小于&ldquo;暴力覆盖&rdquo;全空间所需的量。</p>

<p style="margin: 20px 0 8px 0; font-size: 16px; line-height: 1.75; color: #333;"><strong>&#10105; 表示学习（Representation Learning）</strong></p>
<p style="margin: 5px 0 5px 0; font-size: 15px; line-height: 1.75; color: #333;">泰勒用多项式基，傅里叶用正弦基，SVM 用核函数&mdash;&mdash;这些基函数都是<strong>人类预先选定的</strong>。</p>
<p style="margin: 5px 0 5px 0; font-size: 15px; line-height: 1.75; color: #333;">神经网络自动学习中间表示（embedding）。一个词被映射到一个高维向量，这个向量捕获了语义、语法、情感等多层信息&mdash;&mdash;<strong>这些&ldquo;特征&rdquo;是模型自己发现的，不是人设计的。</strong></p>
<p style="margin: 5px 0 5px 0; font-size: 15px; line-height: 1.75; color: #333;">这就是为什么同一个架构能做翻译、写诗、编程、做数学&mdash;&mdash;它能自动学习不同任务需要的表示。</p>

<p style="margin: 20px 0 8px 0; font-size: 16px; line-height: 1.75; color: #333;"><strong>&#10106; 生成能力（Generation）</strong></p>
<p style="margin: 5px 0 5px 0; font-size: 15px; line-height: 1.75; color: #333;">核方法能分类，不能生成。样条能插值，不能创造。</p>
<p style="margin: 5px 0 5px 0; font-size: 15px; line-height: 1.75; color: #333;">神经网络可以输出任意复杂的高维结构&mdash;&mdash;一段话（GPT）、一张图（DALL-E）、一段音频（Whisper）、一段视频（Sora）。</p>'''

html_parts.append(colored_box(
    nn_advantages +
    '\n' + blockquote('<strong>生成，是从&ldquo;理解规律&rdquo;到&ldquo;应用规律&rdquo;的跨越。只有神经网络做到了。</strong>'),
    '#E91E63'
))

html_parts.append('<h3 style="font-size: 17px; font-weight: bold; color: #555; margin-top: 25px; margin-bottom: 10px;">来自视频的关键洞察：激活函数的选择</h3>')
html_parts.append(p('Emergent Garden 的视频展示了不同激活函数对拟合效果的影响：'))
html_parts.append(p('<strong>ReLU（修正线性单元）</strong>：最常用的激活函数。它把负数变成零，正数保持不变。结果是<strong>分段线性逼近</strong>&mdash;&mdash;像用折线去拼曲线。简单、高效、计算快，但存在频谱偏差（学高频慢）。'))
html_parts.append(p('<strong>SIREN（sin 激活）</strong>：Matthew Tancik 等人提出，用正弦函数作为激活函数。效果惊人&mdash;&mdash;平滑地拟合高频细节，连毛发纹理都能捕获。但训练不稳定，需要精心初始化。'))
html_parts.append(p('<strong>Fourier Features（傅里叶特征）</strong>：在输入端用正弦函数编码坐标，解决 ReLU 的频谱偏差问题。这是一个绝妙的混合&mdash;&mdash;<strong>傅里叶的思想嫁接到了神经网络的框架中</strong>。'))
html_parts.append(p('看到了吗？<strong>傅里叶级数没有赢得比赛，但它的精髓被神经网络吸收了。</strong> 历史上被淘汰的选手，并没有真正消失&mdash;&mdash;它们的思想活在了冠军的 DNA 里。'))

html_parts.append(hr())

# ================================================================
# Chapter 7: 终极对比
# ================================================================
html_parts.append(h2('第七章：一张表终结所有比较', 0))

html_parts.append(p('现在，让我们把六种方法放在一起做一次终极对比。'))

html_parts.append('<h3 style="font-size: 17px; font-weight: bold; color: #555; margin-top: 25px; margin-bottom: 10px;">雷达图总览</h3>')

html_parts.append(img('method-radar.png', '六种函数拟合方法的雷达图对比'))

html_parts.append(p('一眼就能看到：<strong>神经网络是唯一在高维可扩展性、自动学习、生成能力三个维度上得分为 5 的方法。</strong> 但它的可解释性和计算效率是所有方法中最差的。'))

html_parts.append('<h3 style="font-size: 17px; font-weight: bold; color: #555; margin-top: 25px; margin-bottom: 10px;">详细对比表</h3>')

# Build comparison table
table_headers = ['维度', '泰勒级数', '傅里叶级数', '多项式/样条', '核方法/SVM', '神经网络']
table_data = [
    ['诞生年代', '1715', '1807', '1795/1946', '1992/1995', '1943/1989'],
    ['逼近方式', '点展开', '频率叠加', '点插值/分段', '核映射', '层层变换'],
    ['全局精度', '&#11088;&#11088;', '&#11088;&#11088;&#11088;&#11088;', '&#11088;&#11088;&#11088;&#11088;', '&#11088;&#11088;&#11088;&#11088;', '&#11088;&#11088;&#11088;&#11088;&#11088;'],
    ['高维扩展', '&#11088;', '&#11088;&#11088;', '&#11088;', '&#11088;&#11088;&#11088;', '&#11088;&#11088;&#11088;&#11088;&#11088;'],
    ['自动学习', '&#11088;', '&#11088;', '&#11088;', '&#11088;&#11088;&#11088;', '&#11088;&#11088;&#11088;&#11088;&#11088;'],
    ['生成能力', '&#11088;', '&#11088;', '&#11088;', '&#11088;', '&#11088;&#11088;&#11088;&#11088;&#11088;'],
    ['计算效率', '&#11088;&#11088;&#11088;&#11088;&#11088;', '&#11088;&#11088;&#11088;&#11088;', '&#11088;&#11088;&#11088;&#11088;', '&#11088;&#11088;', '&#11088;&#11088;'],
    ['可解释性', '&#11088;&#11088;&#11088;&#11088;&#11088;', '&#11088;&#11088;&#11088;&#11088;', '&#11088;&#11088;&#11088;&#11088;', '&#11088;&#11088;&#11088;&#11088;', '&#11088;'],
    ['杀手锏', '局部高精度', '频域分析', '曲线设计', '小样本分类', '高维生成'],
    ['致命伤', '全局崩溃', '吉布斯+维度', '维度诅咒', 'O(n&sup3;)+不能生成', '黑盒+耗能'],
    ['今日角色', '物理推导', '信号处理', 'CAD/字体', '生物信息学', 'AI/LLM'],
]

table_html = '<table style="width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 14px;">\n<thead>\n<tr>'
for h in table_headers:
    table_html += td(h, is_header=True)
table_html += '</tr>\n</thead>\n<tbody>\n'
for row in table_data:
    table_html += '<tr>'
    table_html += td(f'<strong>{row[0]}</strong>')
    for cell in row[1:]:
        table_html += td(cell)
    table_html += '</tr>\n'
table_html += '</tbody>\n</table>'
html_parts.append(table_html)

html_parts.append('<h3 style="font-size: 17px; font-weight: bold; color: #555; margin-top: 25px; margin-bottom: 10px;">没有&ldquo;最好&rdquo;，只有&ldquo;最适合&rdquo;</h3>')

html_parts.append(p('<strong>泰勒级数</strong>在物理推导中不可替代&mdash;&mdash;牛顿力学、广义相对论的线性化、量子微扰论，离了它寸步难行。'))
html_parts.append(p('<strong>傅里叶级数</strong>在信号处理中不可替代&mdash;&mdash;你听的每一首 MP3、看的每一张 JPEG、打的每一通 5G 电话，都在用傅里叶。'))
html_parts.append(p('<strong>样条曲线</strong>在工业设计中不可替代&mdash;&mdash;你手机屏幕上每一个字母、汽车车身的每一条曲线，都是贝塞尔曲线。'))
html_parts.append(p('<strong>核方法</strong>在小样本场景中依然强大&mdash;&mdash;基因组分类、蛋白质功能预测，SVM 至今在用。'))
html_parts.append(p('<strong>但当任务变成&ldquo;在万亿维空间中，从万亿数据中，学习并生成复杂规律&rdquo;&mdash;&mdash;只有神经网络满足所有条件。</strong>'))

# Core insight box (orange 2px border)
html_parts.append('''<section style="border: 2px solid #FF9800; border-radius: 8px; padding: 15px 20px; margin: 20px 0; background: rgba(255,152,0,0.04);">
<p style="font-weight: bold; margin-bottom: 12px; font-size: 1.1em; color: #FF9800;">&#x1F3AF; 核心洞察</p>
<p style="margin: 0; font-size: 16px; line-height: 1.75; color: #333;">LLM 不是&ldquo;选择&rdquo;了神经网络&mdash;&mdash;是只有神经网络同时满足了五个条件：可扩展、可学习、高维友好、能表示学习、能生成。这不是选择题，这是淘汰赛。其他方法都在某个维度上碰到了不可逾越的墙。</p>
</section>''')

html_parts.append(hr())

# ================================================================
# 结语
# ================================================================
html_parts.append(h2('结语：400 年的接力赛', 1))

html_parts.append(p('回到 Emergent Garden 的视频标题：<em>Watching Neural Networks Learn</em>&mdash;&mdash;看着神经网络学习。'))
html_parts.append(p('你现在知道了，当你看着那些神经网络逐步拟合目标函数的动画时，你看到的不仅仅是一个算法的运行。'))
html_parts.append(p('你看到的是：'))
html_parts.append(blockquote('<strong>人类 400 年数学探索的最新一章。</strong>'))
html_parts.append(p('泰勒打下了逼近论的地基，傅里叶发现了频率的语言，拉格朗日和贝塞尔建造了曲线的工具箱，Vapnik 和 Cortes 探索了高维的魔术&mdash;&mdash;每一步都不白走。'))
html_parts.append(p('神经网络之所以能胜出，不是因为它&ldquo;更好&rdquo;，而是因为它站在了所有前人的肩膀上。'))
html_parts.append(p('<strong>ReLU 里有分段逼近的智慧，Fourier Features 里有傅里叶的回声，Scaling Laws 里有统计学的积累，梯度下降里有微积分的力量。</strong>'))
html_parts.append(p('没有谁被真正淘汰。他们都活在冠军的 DNA 里。'))

html_parts.append(hr())

# ================================================================
# 延伸阅读
# ================================================================
html_parts.append('''<section style="max-width: 100%; margin: 20px 0; padding: 20px 24px; border-radius: 8px; background: rgba(233,30,99,0.04); border: 1px solid rgba(233,30,99,0.12);">
<p style="margin: 0 0 12px 0; font-size: 16px; line-height: 1.75; color: #333;"><strong>&#x1F4DA; 延伸阅读</strong></p>
<p style="margin: 5px 0; font-size: 15px; line-height: 1.75; color: #333;">&bull; 为什么矩阵和激活函数就能涌现智能？&mdash;&mdash;万能近似定理</p>
<p style="margin: 5px 0; font-size: 15px; line-height: 1.75; color: #333;">&bull; 为什么 AI 离不开线性？&mdash;&mdash;傅里叶分析与维度诅咒</p>
<p style="margin: 5px 0; font-size: 15px; line-height: 1.75; color: #333;">&bull; 为什么把模型做大就能变聪明？&mdash;&mdash;Scaling Laws</p>
<p style="margin: 15px 0 0 0; font-size: 16px; line-height: 1.75; color: #333;"><strong>&#x1F3AC; 灵感来源</strong></p>
<p style="margin: 5px 0; font-size: 15px; line-height: 1.75; color: #333;">本文灵感来自 Emergent Garden 的精彩视频 <em>Watching Neural Networks Learn</em>，强烈推荐观看。</p>
</section>''')

# ================================================================
# Footer
# ================================================================
html_parts.append('''<section style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #e0e0e0; font-size: 14px; color: #999; line-height: 1.8;">
<p style="margin: 5px 0;">博客：https://Jason-Azure.github.io/ai-blog/</p>
<p style="margin: 5px 0;">微信公众号：AI-lab学习笔记</p>
</section>''')

# Close HTML
html_parts.append('''
</section>
</body>
</html>''')

# ============================================================
# Write output
# ============================================================
output_path = os.path.expanduser('~/wechat-articles/42-function-competition.html')
os.makedirs(os.path.dirname(output_path), exist_ok=True)

html_content = '\n\n'.join(html_parts)
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"Generated: {output_path}")
print(f"File size: {os.path.getsize(output_path):,} bytes")
print(f"Lines: {html_content.count(chr(10)) + 1}")
