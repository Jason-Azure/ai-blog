# AI Blog 维护指南

## 项目概况

- **站点地址:** https://Jason-Azure.github.io/ai-blog/
- **仓库地址:** https://github.com/Jason-Azure/ai-blog
- **本地路径:** ~/ai-blog/
- **技术栈:** Hugo 0.146.0 + PaperMod 主题 + GitHub Pages
- **部署方式:** push 到 main → GitHub Actions 自动构建 → GitHub Pages
- **语言:** 中文 (zh-CN)
- **主题模式:** 自动切换明/暗色 (defaultTheme = "auto")

## 架构全景

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户访问博客                              │
│              https://Jason-Azure.github.io/ai-blog/             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    GitHub Pages 托管
                           │
┌──────────────────────────┴──────────────────────────────────────┐
│                     GitHub Actions CI/CD                         │
│              .github/workflows/hugo.yml                         │
│    push main → hugo --gc --minify → 部署到 GitHub Pages         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────────┐
│                     Hugo 静态站点生成                            │
│                                                                  │
│  hugo.toml (配置) + content/ (内容) + layouts/ (模板)            │
│       + themes/PaperMod/ (主题) → public/ (构建输出)             │
└─────────────────────────────────────────────────────────────────┘
```

### 外部服务依赖

| 服务 | 地址 | 用途 |
|------|------|------|
| GitHub Pages | github.io | 静态站点托管 |
| GitHub Actions | .github/workflows/ | CI/CD 自动构建部署 |
| Twikoo (Vercel) | https://ai-blog2026.vercel.app | 留言板后端 |
| MongoDB Atlas | cloud.mongodb.com | Twikoo 数据存储（免费 M0） |
| 不蒜子 | busuanzi.ibruce.info | 访问量/访客统计 |
| jsDelivr CDN | cdn.jsdelivr.net | Twikoo 前端 JS |

## 目录结构

```
~/ai-blog/
├── hugo.toml                          # 站点配置（语言、菜单、主题参数）
├── CLAUDE.md                          # 本文件 — 项目维护指南
├── .github/workflows/hugo.yml         # CI/CD 自动部署
├── .gitmodules                        # PaperMod 子模块声明
├── content/
│   ├── about/index.md                 # 关于页面
│   ├── search/index.md                # 搜索页面（PaperMod 内置）
│   └── posts/                         # 文章目录
│       ├── hello-world/index.md       # 博客上线公告
│       ├── llm-data-pipeline/index.md # LLM 数据处理全流程
│       └── llm-pipeline-visual/index.md # LLM 全流程可视化
├── layouts/
│   ├── index.html                     # 首页模板（自定义 6 区块着陆页）
│   ├── partials/
│   │   ├── comments.html              # Twikoo 评论系统集成
│   │   ├── extend_head.html           # 首页专属 CSS（暗色模式+响应式）
│   │   └── extend_footer.html         # 不蒜子统计脚本（仅首页）
│   └── shortcodes/
│       ├── bilibili.html              # B 站视频嵌入
│       ├── video.html                 # 通用视频 (iframe / HTML5)
│       ├── wechat-qr.html            # 微信二维码卡片
│       └── sandbox.html               # 交互式代码沙盒 iframe
├── static/images/
│   └── wechat-group-qr.jpg           # 微信群二维码
└── themes/PaperMod/                   # 主题 (git submodule，勿直接修改)
```

## 首页区块结构 (layouts/index.html)

首页为自定义着陆页，包含 6 个区块：

```
┌─────────────────────────────────────────┐
│  1. 欢迎语 (homeInfoParams)              │  复用 PaperMod home_info.html
├─────────────────────────────────────────┤
│  2. 企业 AI 框架全景图                    │  三列流程: 输入→核心系统→输出
│     12 个术语 pill，悬停显示 CSS tooltip   │  纯 CSS，零 JS
├─────────────────────────────────────────┤
│  3. 统计栏                               │  日期 + 不蒜子 PV/UV
├─────────────────────────────────────────┤
│  4. 微信群二维码卡片                      │  左图右文布局
├─────────────────────────────────────────┤
│  5. 留言板 (Twikoo)                      │  游客可留言，无需登录
├─────────────────────────────────────────┤
│  6. 最近文章列表                          │  复用 PaperMod 分页逻辑
└─────────────────────────────────────────┘
```

### AI 框架全景图术语

| 术语 | Tooltip 解释 |
|------|-------------|
| LLM | 大语言模型 — 理解和生成自然语言的核心引擎 |
| Agent | 智能代理 — 自主规划、调用工具完成复杂目标 |
| RAG | 检索增强生成 — 让 AI 查询外部知识库 |
| MCP | 模型上下文协议 — AI 与外部工具的连接标准 |
| Skills | 技能插件 — 赋予 AI 专业能力的可复用模块 |
| Prompt Engineering | 提示词工程 — 精心设计指令引导高质量输出 |
| Fine-tuning | 微调 — 用领域数据定制模型 |
| Embedding | 向量嵌入 — 文本转数字向量，理解语义 |
| Vector DB | 向量数据库 — 语义向量存储与检索 |
| Token | 分词单元 — LLM 处理文本的最小单位 |
| CoT | 思维链推理 — 逐步推理提高准确性 |
| Transformer | 变换器 — 现代 LLM 的核心神经网络架构 |

## 评论系统 (Twikoo)

### 当前配置

```toml
# hugo.toml
comments = true
[params.twikoo]
  envId = "https://ai-blog2026.vercel.app"
  lang = "zh-CN"
```

### 组件架构

```
用户留言 → 博客前端 (twikoo.min.js v1.6.44)
               ↓
         Vercel 云函数 (https://ai-blog2026.vercel.app)
               ↓
         MongoDB Atlas (免费 M0 集群)
```

### 管理面板

在博客留言板区域点击 ⚙️ 齿轮图标，输入管理员密码登录。可配置：
- `REQUIRED_FIELDS`: 必填项（当前设为 `nick` 仅昵称必填）
- 垃圾评论过滤 (Akismet)
- 邮件通知 (SMTP)
- 评论审核开关

### Twikoo 维护

- **Vercel 项目:** 登录 vercel.com 查看部署状态
- **MongoDB:** 登录 cloud.mongodb.com 查看数据
- **环境变量:** Vercel Settings → Environment Variables → `MONGODB_URI`

## 流量统计 (不蒜子)

- **脚本:** `//busuanzi.ibruce.info/busuanzi/2.3/busuanzi.pure.mini.js`
- **加载位置:** `extend_footer.html`，仅首页加载 (`{{ if .IsHome }}`)
- **显示指标:** 总访问 PV (`busuanzi_value_site_pv`) + 访客 UV (`busuanzi_value_site_uv`)

## CSS 样式说明 (extend_head.html)

仅在首页加载 (`{{ if .IsHome }}`)，使用 PaperMod CSS 变量实现自动暗色模式：

| CSS 变量 | 亮色模式 | 暗色模式 | 用途 |
|----------|---------|---------|------|
| `--primary` | rgb(30,30,30) | rgb(218,218,219) | pill 背景、文字 |
| `--theme` | rgb(255,255,255) | rgb(29,30,32) | pill 文字、页面背景 |
| `--entry` | rgb(255,255,255) | rgb(46,46,51) | 卡片背景 |
| `--border` | rgb(238,238,238) | rgb(51,51,51) | 边框 |
| `--secondary` | rgb(108,108,108) | rgb(155,156,157) | 次要文字 |
| `--code-bg` | rgb(245,245,245) | rgb(55,56,62) | AI 核心区域背景 |

响应式断点: `@media (max-width: 768px)` — 三列→垂直堆叠，箭头旋转 90°

## 导航菜单

| 菜单项 | URL | 权重(排序) |
|--------|-----|-----------|
| 文章 | /posts/ | 10 |
| 分类 | /categories/ | 20 |
| 标签 | /tags/ | 30 |
| 搜索 | /search/ | 40 |
| 关于 | /about/ | 50 |

## 写新文章

### 标准流程

```bash
cd ~/ai-blog

# 1. 创建文章目录（用英文短横线命名）
mkdir -p content/posts/my-new-post

# 2. 创建 index.md（见下方模板）

# 3. 本地预览
hugo server
# 访问 http://localhost:1313/ai-blog/

# 4. 发布
git add content/posts/my-new-post/
git commit -m "Add post: 文章标题"
git push
# 约 30 秒后自动上线
```

### 文章 Front Matter 模板

```yaml
---
title: "文章标题"
date: 2026-02-18
draft: false
summary: "一两句话描述，显示在列表页和 SEO 中"
categories: ["AI 基础"]
tags: ["标签1", "标签2"]
weight: 1            # 显示顺序
ShowToc: true        # 显示目录导航（长文推荐开启）
TocOpen: true        # 目录默认展开
# cover:
#   image: "cover.jpg"  # 封面图（放在同目录下）
#   alt: "封面描述"
---
```

### 可用分类 (categories)

当前已使用: `AI 基础`, `LLM`

建议分类体系:
- `AI 基础` — 入门概念、原理科普
- `LLM` — 大语言模型相关
- `工具` — AI 工具教程
- `视频` — 视频内容合集

### 文章中的图片

```markdown
<!-- 方式一：放在文章同目录下 -->
![描述](image.png)

<!-- 方式二：放在 static/images/ 下 -->
![描述](/ai-blog/images/image.png)
```

## Shortcodes 使用

### B 站视频

```markdown
{{</* bilibili id="BV1xxxxxxxxx" title="可选标题" */>}}
```

参数:
- `id` (必填): B 站视频 BV 号
- `title` (可选): 视频下方显示的标题

### 通用视频

```markdown
<!-- iframe 嵌入（默认） -->
{{</* video src="https://example.com/embed/xxx" title="视频标题" */>}}

<!-- HTML5 原生视频 -->
{{</* video src="/ai-blog/videos/demo.mp4" type="html5" title="演示" poster="thumb.jpg" */>}}
```

参数:
- `src` (必填): 视频 URL 或文件路径
- `type` (可选): `iframe`(默认) 或 `html5`
- `title` (可选): 视频标题
- `poster` (可选, 仅 html5): 封面图

### 微信二维码

```markdown
{{</* wechat-qr src="/ai-blog/images/qrcode.png" caption="扫码关注公众号" */>}}
```

参数:
- `src` (必填): 二维码图片路径
- `caption` (可选): 图片下方说明文字
- `alt` (可选): 图片 alt 文本，默认"扫码关注"

### 代码沙盒

```markdown
{{</* sandbox src="https://codesandbox.io/embed/xxx" title="在线演示" height="600" */>}}
```

参数:
- `src` (必填): iframe URL
- `title` (可选): 标题栏文字，默认"在线演示"
- `height` (可选): iframe 高度像素，默认 500

## 博客管理方式

### 方式 1：SSH 到 VM + Claude Code

```bash
ssh azureuser@20.10.135.83
work    # 进入 tmux 会话
cc      # 启动 Claude Code，可以对话式管理博客
```

适合：写新文章、改模板、调配置、调样式等复杂操作。

### 方式 2：GitHub 网页直接编辑

打开 https://github.com/Jason-Azure/ai-blog ，在浏览器中操作：

- **写新文章:** `content/posts/` → Add file → Create new file → 输入 `新目录名/index.md` → 粘贴 Markdown → Commit
- **编辑文章:** 点开 `.md` 文件 → 铅笔图标 → 在线编辑 → Commit
- **上传图片:** 进入文章目录 → Add file → Upload files

Commit 后 GitHub Actions 自动构建部署，约 30 秒上线。

适合：简单改文字、写新文章、不需要预览的快速操作。

### 方式 3：本地电脑 Git + Hugo（推荐日常使用）

```powershell
# 首次克隆（只需一次）
git clone --recurse-submodules https://github.com/Jason-Azure/ai-blog.git
cd ai-blog

# 安装 Hugo（Windows 用 winget 或 scoop）
winget install Hugo.Hugo.Extended

# 写文章
mkdir content\posts\my-new-post
# 用 VS Code 编辑 content\posts\my-new-post\index.md

# 本地预览
hugo server
# 浏览器打开 http://localhost:1313/ai-blog/

# 满意后发布
git add .
git commit -m "Add post: 文章标题"
git push
```

适合：本地实时预览、批量操作、用熟悉的编辑器写作。

### 方式选择建议

| 场景 | 推荐方式 |
|------|---------|
| 改模板/样式/配置 | 方式 1（SSH + Claude Code） |
| 快速改几个字 | 方式 2（GitHub 网页） |
| 日常写长文章 | 方式 3（本地 Git + Hugo） |
| 上传图片/二维码 | 方式 2 或 3 |

## 常用操作

### 本地预览

```bash
cd ~/ai-blog && hugo server
# 访问 http://localhost:1313/ai-blog/
```

### 发布更新

```bash
cd ~/ai-blog
git add -A && git commit -m "描述" && git push
```

### 构建检查（不启动 server）

```bash
cd ~/ai-blog && hugo --minify
# 输出在 public/ 目录
```

### 更新 PaperMod 主题

```bash
cd ~/ai-blog
git submodule update --remote themes/PaperMod
hugo server  # 验证无报错
git add themes/PaperMod && git commit -m "Update PaperMod theme" && git push
```

### 检查 GitHub Actions 构建状态

```bash
gh run list --repo Jason-Azure/ai-blog --limit 3
# 查看失败日志
gh run view <run-id> --repo Jason-Azure/ai-blog --log-failed
```

### 更新微信群二维码

```bash
# 从 Windows 电脑传图到 VM
scp chatgroup.jpg azureuser@20.10.135.83:~/ai-blog/static/images/wechat-group-qr.jpg

# 在 VM 上提交
cd ~/ai-blog
git add static/images/wechat-group-qr.jpg
git commit -m "Update WeChat group QR code"
git push
```

## GitHub Actions 工作流 (.github/workflows/hugo.yml)

```
触发: push main 或手动 workflow_dispatch
    ↓
Build (ubuntu-latest):
    安装 Hugo 0.146.0 → checkout (含 submodule) → hugo --gc --minify → 上传 artifact
    ↓
Deploy:
    部署到 GitHub Pages 环境
    ↓
~30 秒后上线
```

## 待优化功能

### Cloudflare CDN 加速（国内访问）

如果国内访问慢，可以免费添加 Cloudflare CDN:
1. 注册 Cloudflare，添加自定义域名
2. 在 GitHub Pages Settings 中配置 custom domain
3. 在 `hugo.toml` 中更新 `baseURL`

## 注意事项

- `themes/PaperMod/` 是 git submodule，不要直接修改其中的文件
- 自定义样式和模板覆盖放在 `layouts/` 下，Hugo 会优先使用项目级文件
- 图片尽量压缩后再放入，推荐 WebP 格式
- `hugo.toml` 中 `baseURL` 必须与实际部署地址匹配
- Front Matter 中 `draft: true` 的文章不会发布（本地 `hugo server -D` 可预览）
- Twikoo 的 Vercel 域名已固定为 `ai-blog2026.vercel.app`，无需随部署变化
- 不蒜子统计为第三方免费服务，偶尔可能不可用，不影响博客本身
