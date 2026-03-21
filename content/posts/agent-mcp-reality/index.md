---
title: "AI Agent 热潮冷思考：当我们拆掉所有包装之后"
date: 2026-03-23
draft: false
summary: "Agent、MCP、Manus、Coze、Skills……AI 热词一个接一个。但拆掉所有包装之后，你会发现一个令人不安的事实：它们和二十年前的中间件，本质上没什么不同。真正改变的从来不是工具，而是我们对'该做什么'的回答。"
categories: ["AI 观点"]
tags: ["Agent", "MCP", "Manus", "Coze", "AI 应用", "中间件", "范式转变", "冷思考"]
weight: 10
ShowToc: true
TocOpen: true
---

<div style="max-width: 680px; margin: 1.5em auto; padding: 20px 24px; border-radius: 10px; background: linear-gradient(135deg, rgba(233,30,99,0.06), rgba(33,150,243,0.06)); border: 1px solid rgba(233,30,99,0.15);">

<div style="font-weight: bold; margin-bottom: 10px; color: #E91E63; font-size: 1.1em;">📖 导读</div>

这篇文章不是 AI Agent 的入门教程——那种文章已经太多了。

这是一篇**拆解文**。

我要拆掉 Agent、MCP、Manus、Coze、Skills 这些热词的包装，看看里面到底装着什么。然后回答一个没人敢问的问题：**这些东西，是 AI 的未来，还是 AI 的过渡期废料？**

如果你正在创业做 Agent、正在公司里推 AI 落地、或者只是看着满屏的 AI 新名词感到焦虑——这篇文章可能会让你安心，也可能让你更焦虑。

<div style="font-size: 0.9em; color: #888; margin-top: 12px; line-height: 1.7;">
① 热词地图 → ② Agent 到底是什么 → ③ MCP 的真相 → ④ Manus 启示录 → ⑤ 中间件宿命 → ⑥ 真正该问的问题
</div>

</div>

---

## 第一章：一张热词地图——你现在看到的 AI 世界 🗺️

### 2025-2026 年，如果你关注 AI，一定被这些词轰炸过

打开任何一个科技媒体或者朋友圈，你会看到：

<div style="max-width: 660px; margin: 1.5em auto; padding: 16px 20px; border-radius: 8px; background: rgba(255,152,0,0.06); border: 1px solid rgba(255,152,0,0.2);">

> "XX 公司发布了全新的 **AI Agent** 平台！"
>
> "**MCP** 将改变 AI 与世界的连接方式！"
>
> "**Manus** 一夜爆火，通用 Agent 时代来了！"
>
> "用 **Coze/Dify** 零代码搭建你的 AI 应用！"
>
> "**Function Calling** 升级为 **Tool Use**，再进化为 **Skills**！"

</div>

看完之后你的感受大概是：**每个词都认识，但连在一起就懵了。**

这不怪你。因为这些词之间的关系，连很多从业者都没理清楚。让我先画一张地图。

### 一张从内到外的架构图

在之前的《AI 全景定位》中，我们建立了这个框架。现在让我们重新审视它，聚焦到"LLM 之上"的这一层——也就是当前所有热词诞生的地方：

```text
┌─────────────────────────────────────────────────────────────┐
│                    你看到的 AI 产品                           │
│  ChatGPT / Claude / 豆包 / Kimi / ...                       │
├─────────────────────────────────────────────────────────────┤
│                  ↑ 这一层是所有热词的战场 ↑                   │
│                                                             │
│   ┌─────────┐  ┌────────┐  ┌──────────┐  ┌────────────┐   │
│   │  Agent  │  │  MCP   │  │  Skills  │  │    RAG     │   │
│   │ (项目   │  │ (连接  │  │ (标准化  │  │ (查资料   │   │
│   │  经理)  │  │  接口) │  │  流程)   │  │  再回答)  │   │
│   └────┬────┘  └───┬────┘  └────┬─────┘  └─────┬──────┘   │
│        │           │            │               │           │
│        └───────────┴────────────┴───────────────┘           │
│                          │                                   │
├──────────────────────────┼───────────────────────────────────┤
│                    LLM（大语言模型）                          │
│            GPT-4 / Claude / DeepSeek / Qwen                 │
│              ← 这一层才是真正的引擎                           │
├─────────────────────────────────────────────────────────────┤
│                  Transformer 架构                            │
├─────────────────────────────────────────────────────────────┤
│                    深度学习                                   │
├─────────────────────────────────────────────────────────────┤
│                    机器学习                                   │
└─────────────────────────────────────────────────────────────┘
```

**一句话：** 所有热词——Agent、MCP、Skills、RAG——都是 **LLM 之上的应用层**。它们不是 AI 本身，它们是 AI 的**使用方式**。

---

## 第二章：Agent 到底是什么？——拆掉第一层包装 🔧

### 最诚实的定义

一个 AI Agent，拆到最底层，就是这个循环：

```text
while True:
    观察环境
    → 把观察送给 LLM
    → LLM 决定下一步做什么
    → 执行动作（调 API / 点按钮 / 写文件）
    → 把执行结果再送给 LLM
    → LLM 决定：任务完了？还是继续？
```

**就这些。**

不管是 Manus、Devin、Claude Computer Use、还是你用 Coze 搭的机器人——**拆开来看都是这个循环。**

<div style="max-width: 660px; margin: 1.5em auto; padding: 16px 20px; border-radius: 8px; background: rgba(33,150,243,0.06); border: 1px solid rgba(33,150,243,0.15);">

**Agent = 循环调用 LLM 做决策的程序**

不是魔法。不是通用人工智能。是一个 while 循环。

</div>

### 和传统软件有什么区别？

你可能会问：这和传统软件有什么本质区别？传统软件不也是"观察→判断→执行"的循环吗？

<div style="max-width: 660px; margin: 1.5em auto; padding: 16px 20px; border-radius: 8px; background: rgba(76,175,80,0.06); border: 1px solid rgba(76,175,80,0.15);">

| | 传统软件 | AI Agent |
|--|--|--|
| **"判断"由谁做** | 人写的 if-else 规则 | LLM 根据上下文生成决策 |
| **能处理的情况** | 只能处理预设的情况 | 理论上能处理任何自然语言描述的情况 |
| **出错时的表现** | 要么崩溃，要么按错误路径走 | 不会崩溃，但可能"自信地犯错" |
| **可预测性** | 100%——相同输入必定相同输出 | 概率性——相同输入可能不同输出 |

</div>

区别是真实的，但没有很多人说的那么大。**Agent 的本质仍然是软件——只是"判断"这一步从硬编码的规则变成了 LLM 的概率性推理。**

### 当前 Agent 的真实能力边界

先看产品图谱：

<div style="max-width: 660px; margin: 1.5em auto; padding: 16px 20px; border-radius: 8px; background: rgba(156,39,176,0.06); border: 1px solid rgba(156,39,176,0.15);">

**编程 Agent（最成熟的品类）：**

| 产品 | 做什么 | 为什么能用 |
|------|-------|----------|
| Claude Code | 终端里写代码、改代码、跑测试 | 开发者**实时审查**每一步 |
| Cursor | IDE 里的 AI 编程助手 | 修改在你眼皮底下，不满意就撤销 |
| GitHub Copilot Agent | 从 Issue 出发自动写 PR | 有 code review 环节兜底 |
| Devin | 号称"AI 软件工程师" | 实际测试效果远低于宣传 |

**通用 Agent（最有话题性的品类）：**

| 产品 | 做什么 | 现实 |
|------|-------|------|
| Manus | 浏览网页、建网站、做 PPT | 被 Meta 收购后商业化 |
| Claude Computer Use | 截屏→分析→点鼠标操作电脑 | Beta，延迟高，操作不精确 |
| OpenAI Operator | 浏览器自动化 | 有限场景可用 |

**构建平台（最接地气的品类）：**

| 平台 | 定位 | 适合 |
|------|------|------|
| Coze（扣子） | 字节跳动的 Agent 构建平台 | 零代码搭建，发布到豆包/飞书 |
| Dify | 开源 Agentic Workflow Builder | 技术团队自部署 |
| FastGPT | 开源知识库 + Agent | 偏知识库问答 |

</div>

**一个残酷的事实：** 真正在生产环境中大量使用的 Agent，都有一个共同特征——**人在循环中（Human-in-the-loop）**。编程 Agent 之所以最成熟，恰恰因为开发者可以即时审查和修正每一步输出。

完全自主的"通用 Agent"？目前还停留在 demo 阶段。

---

## 第三章：MCP 的真相——USB-C 还是又一根转接线？ 🔌

### 先说 MCP 是什么

MCP（Model Context Protocol）是 Anthropic 在 2024 年 11 月发布的一个开源协议。官方比喻是：**AI 的 USB-C 接口**。

在 MCP 之前，每个 AI 应用想连接外部工具，都要自己写一套集成代码：

```text
之前：N 个 AI 应用 × M 个工具 = N×M 个集成
之后：N 个 AI 应用 → MCP 协议 ← M 个工具 = N+M 个集成
```

这个思路没有问题。它解决的是**标准化**问题。

### MCP 的技术本质

拆开来看，MCP 的架构是这样的：

<div style="max-width: 660px; margin: 1.5em auto; padding: 16px 20px; border-radius: 8px; background: rgba(33,150,243,0.06); border: 1px solid rgba(33,150,243,0.15);">

```text
MCP Host（AI 应用，如 Claude Desktop）
  ├── MCP Client 1 ←→ MCP Server A（文件系统）
  ├── MCP Client 2 ←→ MCP Server B（数据库）
  └── MCP Client 3 ←→ MCP Server C（搜索引擎）
```

每个 MCP Server 暴露三种东西：
1. **Tools（工具）：** AI 可调用的函数（查数据库、发邮件、搜索）
2. **Resources（资源）：** 提供上下文的数据源（文件内容、记录）
3. **Prompts（提示词）：** 可复用的交互模板

底层协议是 JSON-RPC 2.0，传输用 STDIO（本地）或 HTTP（远程）。

</div>

截至 2026 年 3 月，已经有 **80+ 个客户端** 支持 MCP——包括 Claude、ChatGPT、VS Code、Cursor、JetBrains 等。

### 拆掉包装之后

但是——让我们诚实地说：

<div style="max-width: 660px; margin: 1.5em auto; padding: 16px 20px; border-radius: 8px; background: rgba(244,67,54,0.06); border: 1px solid rgba(244,67,54,0.15);">

**MCP 不是技术革命，是标准化革命。**

它的底层仍然是：
1. 用 JSON 描述工具的参数和功能
2. LLM 决定调用哪个工具
3. 执行工具，把结果返回给 LLM

这就是 2023 年 OpenAI 推出的 **Function Calling** 的本质——只不过穿上了"协议"的外衣。

</div>

**Function Calling → Tool Use → MCP 的演化：**

```text
2023.06  Function Calling (OpenAI)
         → 模型输出结构化的函数调用请求
         → 每家 API 格式不同

2024.11  MCP (Anthropic)
         → 统一协议，工具只需实现一次
         → 动态发现工具、有状态连接

2025+    80+ 客户端采纳
         → 事实标准正在形成
```

MCP 的价值是**生态效应**，不是技术突破。就像 USB 没有发明数据传输，但它统一了接口。问题是：这个标准能撑多久？

---

## 第四章：Manus 启示录——一夜爆火背后的真相 💥

### 事件回顾

2025 年 3 月，一个叫 Manus 的 AI Agent 产品一夜之间刷爆了所有科技媒体和朋友圈：

<div style="max-width: 660px; margin: 1.5em auto; padding: 16px 20px; border-radius: 8px; background: rgba(255,152,0,0.06); border: 1px solid rgba(255,152,0,0.2);">

1. 创始人季逸超发布了令人震惊的 demo 视频——Agent 自主浏览网页、创建 PPT、搭建网站
2. 采用**邀请码制**，制造稀缺感，全网 FOMO（错失恐惧症）
3. Twitter 和微信同时刷屏
4. 三天之后，开源社区在 **3 小时内**做出了替代品 **OpenManus**（GitHub 55.4k 星）

</div>

### 三个值得深思的问题

<div style="max-width: 660px; margin: 1.5em auto; padding: 16px 20px; border-radius: 8px; background: rgba(156,39,176,0.06); border: 1px solid rgba(156,39,176,0.15);">

**问题 1：为什么 demo 总是比真实使用好看？**

所有 Agent 的 demo 都展示的是 "happy path"——精心挑选的任务、受控的环境、预先测试过的流程。真实场景中的不确定性（网页结构突然变了、弹窗挡住了按钮、网络超时）会让 Agent 频繁失败。

Claude Computer Use 的官方文档自己承认：*"延迟可能过高"*，*"Claude 在输出坐标时可能犯错或产生幻觉"*。

**问题 2：如果 3 小时就能做出替代品，壁垒在哪里？**

OpenManus 用 3 小时复制了 Manus 的核心功能。这说明：**Agent 的技术壁垒极低。** 真正的核心能力——理解、推理、生成——全在底层的 LLM 里。Agent 框架只是一层薄薄的胶水代码。

**问题 3：Devin 的前车之鉴**

2024 年 3 月，Devin 号称"世界第一个 AI 软件工程师"，估值 20 亿美元，定价 $500/月。社区实测后发现：**成功率远低于宣传，需要大量人工干预。** 这成了 "demo vs reality" 问题的经典案例。

</div>

---

## 第五章：中间件宿命——AI 热词的历史归宿 ⏳

### 如果你经历过互联网历史，这一切似曾相识

让我带你做一个时间旅行：

<div style="max-width: 660px; margin: 1.5em auto; padding: 16px 20px; border-radius: 8px; background: rgba(255,152,0,0.06); border: 1px solid rgba(255,152,0,0.2);">

**2000 年代：** 企业软件的热词是 **ESB（企业服务总线）**。每个厂商都说："你需要一个 ESB 来连接所有系统！"结果呢？微服务架构出来后，ESB 被淘汰了。

**2005 年代：** Web 开发的热词是 **SOA（面向服务架构）** 和 **CORBA**。无数公司买了昂贵的 SOA 中间件。结果呢？REST API 出来后，简单到不需要中间件了。

**2015 年代：** 移动开发的热词是 **跨平台框架**——Cordova、PhoneGap、React Native。"一次编写，到处运行！"结果呢？原生开发和 PWA 各自发展，中间层不断被挤压。

**2025 年代：** AI 的热词是 **Agent 框架、MCP、LangChain、RAG 中间件**。"你需要 XX 来连接 LLM 和你的业务！"

</div>

**看到规律了吗？**

<div style="max-width: 660px; margin: 1.5em auto; padding: 16px 20px; border-radius: 8px; background: rgba(33,150,243,0.06); border: 1px solid rgba(33,150,243,0.15);">

**中间件的生命周期：**

```text
阶段 1：新技术出现，与现有系统不兼容
阶段 2：中间件诞生，充当"翻译官"
阶段 3：中间件标准化，成为"热词"
阶段 4：底层平台强大到不再需要中间层
阶段 5：中间件被吸收或淘汰
```

Agent 框架和 MCP 目前处于**阶段 2-3**。

但如果 LLM 本身变得足够强——能直接理解你的需求、直接调用工具、直接完成任务——**中间层还有存在的必要吗？**

</div>

### 一个激进但值得思考的预测

<div style="max-width: 660px; margin: 1.5em auto; padding: 16px 20px; border-radius: 8px; background: rgba(244,67,54,0.06); border: 1px solid rgba(244,67,54,0.15);">

**现在的 Agent 热潮，本质上是 LLM 能力不足时期的过渡产品。**

为什么需要 MCP？因为 LLM 不能直接访问你的数据库。
为什么需要 Agent 框架？因为 LLM 不能直接规划和执行多步任务。
为什么需要 RAG？因为 LLM 的知识有截止日期、没有你公司的内部数据。

但如果——

- LLM 的上下文窗口大到可以装下你所有的文档？
- LLM 原生支持工具调用，不需要中间协议？
- LLM 的推理能力强到可以自主规划复杂任务？

那些"中间层"就会像 ESB 一样，安静地退出历史舞台。

</div>

我不是说它们**现在**没用。它们在当前阶段有真实的价值。但如果你把整个职业生涯押在"做 Agent 框架"上，请记住 ESB 厂商们的命运。

---

## 第六章：当我们拆掉代码的限制——该做什么？ 🤔

### 技术最终回归到"人的决策"

这是这篇文章最重要的一章。

在做了这么多技术分析之后，我发现最核心的问题不是技术问题，而是——

<div style="max-width: 660px; margin: 1.5em auto; padding: 16px 20px; border-radius: 8px; background: rgba(255,152,0,0.06); border: 1px solid rgba(255,152,0,0.2);">

**当 AI 能帮你做越来越多的事，"该做什么"这个问题反而变得更重要了。**

</div>

过去，技术能力是瓶颈。你想做一个网站，需要学 HTML、CSS、JavaScript。你想分析数据，需要学 SQL、Python、统计学。技术限制了你能做什么。

现在，AI 可以帮你写代码、做 PPT、搜索信息、分析数据。**技术瓶颈正在消失。** 但一个新的问题浮出水面：

> 你到底想用它来做什么？

<div style="max-width: 660px; margin: 1.5em auto; padding: 16px 20px; border-radius: 8px; background: rgba(76,175,80,0.06); border: 1px solid rgba(76,175,80,0.15);">

**Agent 可以自动浏览 100 个网页。** 但该浏览哪 100 个？这个决策需要人来做。

**MCP 可以连接你的所有系统。** 但连接之后要解决什么问题？这个决策需要人来做。

**RAG 可以让 AI 检索你的知识库。** 但知识库里该放什么、怎么组织？这个决策需要人来做。

**LLM 可以生成任何文本。** 但该生成什么、给谁看、解决什么需求？这个决策需要人来做。

</div>

### 真正的业务场景无法被自动化

这是我观察到的另一个事实：**越是关键的业务决策，越需要人的判断。**

<div style="max-width: 660px; margin: 1.5em auto; padding: 16px 20px; border-radius: 8px; background: rgba(33,150,243,0.06); border: 1px solid rgba(33,150,243,0.15);">

| 可以自动化 | 不能自动化 |
|--|--|
| 写代码 | 决定写什么产品 |
| 分析数据 | 决定看什么指标、设什么目标 |
| 搜索信息 | 判断信息的可信度和相关性 |
| 生成报告 | 决定报告给谁看、用来做什么决策 |
| 自动回复客户 | 在没有先例的情况下判断是否破例 |
| 翻译文档 | 决定是否进入某个市场 |

</div>

Agent 的确让"执行"变得更快更便宜了。但"决策"这一步——**理解场景、权衡利弊、承担后果**——仍然需要人。

而且有一个反直觉的现象：**当执行变得越来越容易时，决策的质量反而变得更加重要。** 因为 AI 可以帮你快速做错的事——只要你指错了方向，AI 会高效地在错误的道路上全速前进。

### 我的观点：别追工具，追问题

<div style="max-width: 660px; margin: 1.5em auto; padding: 16px 20px; border-radius: 8px; background: rgba(156,39,176,0.06); border: 1px solid rgba(156,39,176,0.15);">

**三个建议：**

**1. 不要问"怎么用 Agent"，而要问"我有什么问题"。** 很多人是先看到了工具，再去找问题。应该反过来。

**2. 判断 Agent 产品的真实价值，只需一个问题：** "去掉 LLM 之后还剩什么？"如果答案是"一个普通的自动化脚本"——那 Agent 的壁垒就很薄，随时可能被底层 LLM 的进化吞掉。

**3. 投资你的判断力，而不是工具使用技巧。** MCP 可能活 3 年，LangChain 可能活 5 年。但理解"什么问题值得解决、什么数据值得关注、什么决策不能让 AI 做"——这种判断力永远不会过时。

</div>

---

## 结语：热潮总会退去，问题永远在

<div style="max-width: 660px; margin: 1.5em auto; padding: 20px 24px; border-radius: 10px; border: 2px solid #FF9800; background: rgba(255,152,0,0.04);">

2000 年，所有人都在谈论 "dot-com"。大部分公司死了，但互联网留下了。

2017 年，所有人都在谈论 "区块链改变一切"。大部分项目消失了，但加密技术留下了。

2025 年，所有人都在谈论 "AI Agent 改变一切"。

大部分 Agent 框架会消失。但 LLM 的能力会留下——它们会被吸收进操作系统、进浏览器、进每一个应用的基础设施里。到那时，没有人会再说"我在用一个 Agent"，就像今天没有人说"我在用一个 REST API"一样。

**那时候真正留下的是什么？**

是你对"该做什么"的回答。

Agent 不会思考你的人生应该怎么活。MCP 不会告诉你公司的下一个战略方向。RAG 不会帮你判断什么事情值得做。

这些，始终是人的工作。

而且是越来越重要的工作。

</div>

---

<div style="max-width: 680px; margin: 1.5em auto; padding: 20px 24px; border-radius: 8px; background: rgba(233,30,99,0.04); border: 1px solid rgba(233,30,99,0.12);">

**📚 延伸阅读**

- MCP 官方文档：[modelcontextprotocol.io](https://modelcontextprotocol.io/introduction)
- OpenManus（Manus 开源替代）：[GitHub](https://github.com/FoundationAgents/OpenManus)（55.4k ⭐）
- 前文：**[AI 全景定位：从概念迷雾到清晰地图](/ai-blog/posts/ai-landscape/)** —— 建立完整的 AI 概念框架
- 前文：**[DeepSeek-R1：一个模型如何学会思考](/ai-blog/posts/deepseek-r1-thinking/)** —— 模型层面的真正突破
- 前文：**[MoE：671B 参数只用 37B 的秘密](/ai-blog/posts/moe-architecture/)** —— 架构层面的真正创新
- 下一篇预告：**从 RLHF 到 GRPO** —— 大模型训练的三次革命

**📌 后续文章暂存清单（规划中）：**
- 从 RLHF 到 GRPO：三次革命
- 蒸馏：大模型如何"教"小模型
- Attention 的进化：从标准注意力到 MLA
- Transformer 会被取代吗？—— Mamba 与混合架构

</div>

<div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #e0e0e0; font-size: 0.9em; color: #888;">

博客：https://Jason-Azure.github.io/ai-blog/

微信公众号：AI-lab学习笔记

</div>
