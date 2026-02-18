# AI Blog 维护指南

## 项目概况

- **站点地址:** https://Jason-Azure.github.io/ai-blog/
- **仓库地址:** https://github.com/Jason-Azure/ai-blog
- **本地路径:** ~/ai-blog/
- **技术栈:** Hugo 0.146.0 + PaperMod 主题 + GitHub Pages
- **部署方式:** push 到 main → GitHub Actions 自动构建 → GitHub Pages

## 目录结构

```
~/ai-blog/
├── hugo.toml                          # 站点配置（语言、菜单、主题参数）
├── .github/workflows/hugo.yml         # CI/CD 自动部署
├── content/
│   ├── about/index.md                 # 关于页面
│   ├── search/index.md                # 搜索页面（PaperMod 内置）
│   └── posts/                         # 文章目录
│       ├── hello-world/index.md
│       └── llm-data-pipeline/index.md
├── layouts/
│   ├── shortcodes/                    # 自定义 shortcodes
│   │   ├── bilibili.html             # B 站视频嵌入
│   │   ├── video.html                # 通用视频 (iframe / HTML5)
│   │   ├── wechat-qr.html            # 微信二维码卡片
│   │   └── sandbox.html              # 交互式代码沙盒 iframe
│   └── partials/
│       └── comments.html             # Giscus 评论（预留，需启用）
├── static/images/                     # 图片资源
└── themes/PaperMod/                   # 主题 (git submodule，勿直接修改)
```

## 写新文章

### 标准流程

```bash
cd ~/ai-blog

# 1. 创建文章目录（用英文短横线命名）
mkdir -p content/posts/my-new-post

# 2. 创建 index.md（见下方模板）

# 3. 本地预览
hugo server

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

## 待启用功能

### Giscus 评论系统

1. 在 GitHub 仓库 Settings → Features → 勾选 Discussions
2. 访问 https://giscus.app/ ，填入 `Jason-Azure/ai-blog`，获取 `repoID` 和 `categoryID`
3. 编辑 `hugo.toml`，取消 `[params.giscus]` 部分的注释，填入对应值
4. push 即可生效

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
