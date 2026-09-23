# copiioai.cn 纯静态官网 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `copiioai.cn` 成为一个纯静态产品介绍页，使其真实地不构成「具有舆论属性或社会动员能力的信息服务」，从而免除公安备案所需的安全评估。

**Architecture:** 新建独立 COS 桶承载静态文件，经腾讯云 CDN（自定义域名 + SSL 证书）对外服务，DNS 从 Cloudflare 迁至 DNSPod。`.cn` 与 `.com` 由此彻底分家——`.cn` 只有一个独立的、不含任何应用文件的源站。站点本身是手写 HTML，无构建步骤，无任何对 `api.*` 的调用。

**Tech Stack:** 手写 HTML/CSS（无 npm、无 bundler）、Bash 验收脚本、腾讯云 COS + CDN + DNSPod

**Spec:** `docs/superpowers/specs/2026-09-23-cn-static-site-design.md`

## Global Constraints

以下护栏违反任意一条，整个方案前功尽弃——违反即意味着 `.cn` 从"官网"退化为"静态壳"。每个任务的要求都隐含包含本节。

- **不得出现任何表单或输入元素**（`<form>`、`<input>`、`<textarea>`、`<select>`）——表单即构成收集个人信息
- **不得放知识库、社区、对话的任何截图或示例内容**——会破坏"本站无 UGC"的表述
- **不得有任何指向应用的深链接**（`/app`、`/login`、`/community`、`/signup`、`/sessions`、`/knowledge`、`/auth`、`/api`）
- **不得有任何对 `api.*` 的调用**，包括埋点——页面只要还在调 API，就不是静态站
- **不得引入 GA / YouTube / Unsplash 等任何境外依赖**；资源一律本地化
- **`copiioai.com` 全程不碰**——本次改动只作用于 `.cn`
- **`frontend/cn-site/` 目录内只放需要上传的文件**，开发工具（`verify.sh`）放目录外
- 文案中文单语，不出现"数据不出境""不与任何第三方共享""等保/ISO/SOC 2 认证"类表述

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `frontend/cn-site-verify.sh` | 验收脚本。A 组本地检查 + B 组线上检查。**先写它，再写页面** |
| `frontend/cn-site/index.html` | 唯一页面。内联 CSS，零外部请求，含备案号占位节点 |
| `frontend/cn-site/privacy-policy.html` | 本站专属隐私政策：不收集个人信息 + 访问日志说明 |
| `frontend/cn-site/robots.txt` | 爬虫规则 |
| `frontend/cn-site/sitemap.xml` | 站点地图 |
| `frontend/cn-site/assets/favicon.svg` | 图标（本地，不外链） |
| `docs/superpowers/runbooks/2026-09-23-cn-site-deploy.md` | 部署清单：COS 桶 + CDN + 证书 + 日志（控制台操作） |
| `docs/superpowers/runbooks/2026-09-23-cn-site-switch.md` | 切换清单：DNS 迁移 10 步 + 回滚（控制台操作） |

`verify.sh` 刻意放在站点目录**之外**：站点目录内只有需要上传的文件，因此任何上传方式（包括手滑的整目录同步）都不可能把开发工具带进桶里。Task 1 的 A6 检查依赖这个前提。

---

## Task 1: 验收脚本（A 组本地检查）

先写测试，再写实现。此时站点尚不存在，A 组必然全红——这正是要确认的。

**Files:**
- Create: `frontend/cn-site-verify.sh`

**Interfaces:**
- Consumes: 无
- Produces: 可执行脚本 `frontend/cn-site-verify.sh`。用法 `./frontend/cn-site-verify.sh [BASE_URL]`；无参数只跑 A 组，带 URL 加跑 B 组。退出码 = 失败项数（0 表示全绿）。输出每行格式 `  PASS|FAIL|WARN  <检查项>`。Task 2/3 依赖 A1–A6，Task 5 依赖 B1–B3。

- [ ] **Step 1: 写脚本**

创建 `frontend/cn-site-verify.sh`：

```bash
#!/usr/bin/env bash
#
# copiioai.cn 静态站验收脚本
#
# 用途：验证 copiioai.cn 确实是纯静态站点——不含用户功能、不调用 API、无境外依赖。
# 这既是备案材料的可执行依据，也是防止将来改坏的回归防线。
#
# 用法：
#   ./frontend/cn-site-verify.sh                         # 只跑 A 组（本地文件检查）
#   ./frontend/cn-site-verify.sh https://copiioai.cn     # A 组 + B 组（线上检查）
#
# 退出码：失败项数量。0 表示全绿。
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE_DIR="$REPO_ROOT/cn-site"
BASE_URL="${1:-}"

FAILED=0
WARNED=0

pass() { printf '  PASS  %s\n' "$1"; }
fail() { printf '  FAIL  %s\n' "$1"; FAILED=$((FAILED + 1)); }
warn() { printf '  WARN  %s\n' "$1"; WARNED=$((WARNED + 1)); }
hr()   { printf '\n--- %s ---\n' "$1"; }

# grep 快捷方式：递归、显示行号、忽略二进制、限定 html
G() { grep -rInoE "$1" "$SITE_DIR" --include='*.html' 2>/dev/null; }

# 备案号占位节点是否已填入真实号码
PENDING_RE='data-pending="true"'

# ---------------------------------------------------------------- A 组

hr "A 组：本地静态检查（$SITE_DIR）"

if [ ! -d "$SITE_DIR" ]; then
  fail "站点目录不存在：$SITE_DIR"
  printf '\n失败 %d 项\n' "$FAILED"
  exit "$FAILED"
fi

# A1 无表单与输入
hits=$(G '<form|<input|<textarea|<select')
if [ -n "$hits" ]; then
  fail "A1 发现表单或输入元素（构成收集个人信息）"
  printf '%s\n' "$hits"
else
  pass "A1 无表单与输入"
fi

# A2a 资源零外链：src 属性一定是加载资源，不得出现绝对 URL
hits=$(G 'src="[^"]*://[^"]*"')
if [ -n "$hits" ]; then
  fail "A2a 资源 src 指向外部地址"
  printf '%s\n' "$hits"
else
  pass "A2a 资源 src 全部本地"
fi

# A2b 超链接与 link[href] 仅允许白名单域名
bad=$(grep -rhoE 'href="https?://[^"]*"' "$SITE_DIR" --include='*.html' 2>/dev/null \
  | sed -E 's/^href="//; s/"$//' | sort -u \
  | grep -vE '^https?://(www\.)?copiioai\.(com|cn)(/|$)' \
  | grep -vE '^https?://beian\.miit\.gov\.cn(/|$)' \
  | grep -vE '^https?://beian\.mps\.gov\.cn(/|$)' \
  | grep -v '^$')
if [ -n "$bad" ]; then
  fail "A2b 存在非白名单域名（境外依赖或未预期外链）"
  printf '%s\n' "$bad"
else
  pass "A2b 外链仅限白名单（copiioai.com/cn、备案查询站）"
fi

# A2c 内联 CSS 不得引用外部资源
hits=$(grep -rInoE "url\([^)]*https?://" "$SITE_DIR" --include='*.html' 2>/dev/null)
if [ -n "$hits" ]; then
  fail "A2c 内联样式引用了外部资源"
  printf '%s\n' "$hits"
else
  pass "A2c 内联样式无外部资源"
fi

# A3 无 API 调用（含埋点）
hits=$(G 'fetch\(|XMLHttpRequest|sendBeacon|navigator\.sendBeacon|axios|api\.copiioai')
if [ -n "$hits" ]; then
  fail "A3 页面存在 API 调用——只要还在调 API 就不是静态站"
  printf '%s\n' "$hits"
else
  pass "A3 无任何 API 调用"
fi

# A4 无应用路由深链接
hits=$(G 'href="[^"]*/(app|login|signup|community|sessions|knowledge|auth|api)(/|"|\?)')
if [ -n "$hits" ]; then
  fail "A4 存在指向应用的深链接"
  printf '%s\n' "$hits"
else
  pass "A4 无应用路由深链接"
fi

# A5 备案号占位节点存在
miss=""
grep -q 'id="icp-license"'    "$SITE_DIR/index.html" 2>/dev/null || miss="$miss icp-license"
grep -q 'id="police-license"' "$SITE_DIR/index.html" 2>/dev/null || miss="$miss police-license"
if [ -n "$miss" ]; then
  fail "A5 页脚缺少备案号占位节点：$miss"
else
  pass "A5 备案号占位节点存在"
fi

# A5b 备案号尚未填入时给出提醒（不算失败——备案获批前本就没有号码）
if grep -q "$PENDING_RE" "$SITE_DIR/index.html" 2>/dev/null; then
  warn "A5b 页脚备案号仍为占位状态。备案获批后必须填入（见计划末尾 T1/T2）"
fi

# A6 站点标识存在（B2 依赖它确认线上内容就是本站文件）
if grep -q 'name="generator" content="copiioai-cn-site"' "$SITE_DIR/index.html" 2>/dev/null; then
  pass "A6 站点标识存在"
else
  fail "A6 缺少 <meta name=\"generator\" content=\"copiioai-cn-site\">"
fi

# A7 verify.sh 不在站点目录内（防止被误传到桶里）
if [ -e "$SITE_DIR/verify.sh" ]; then
  fail "A7 verify.sh 位于站点目录内，会被上传到桶"
else
  pass "A7 开发工具不在站点目录内"
fi

# ---------------------------------------------------------------- B 组

if [ -n "$BASE_URL" ]; then
  hr "B 组：线上检查（$BASE_URL）"

  # B1/B3 应用路径必须 404，且不得重定向
  for p in /app /login /community /signup /sessions /knowledge /auth /api; do
    read -r code loc < <(curl -s -o /dev/null --max-time 15 \
      -w '%{http_code} %{redirect_url}' "$BASE_URL$p")
    case "$code" in
      404) pass "B1 $p → 404" ;;
      301|302|307|308)
        fail "B3 $p 重定向到 $loc —— 把人送进应用与直接提供服务无实质区别" ;;
      *)
        fail "B1 $p 返回 $code（应为 404）" ;;
    esac
  done

  # B2 首页可达且确实是本站文件
  if curl -s --max-time 15 "$BASE_URL/" | grep -q 'copiioai-cn-site'; then
    pass "B2 首页 200 且为本站文件"
  else
    fail "B2 首页不可达，或内容不是本站文件"
  fi
else
  hr "B 组：已跳过（未提供 BASE_URL）"
fi

# ---------------------------------------------------------------- 汇总

printf '\n========================================\n'
if [ "$FAILED" -eq 0 ]; then
  printf '全部通过'
  [ "$WARNED" -gt 0 ] && printf '（%d 项提醒）' "$WARNED"
  printf '\n'
else
  printf '失败 %d 项' "$FAILED"
  [ "$WARNED" -gt 0 ] && printf '，另有 %d 项提醒' "$WARNED"
  printf '\n'
fi
printf '========================================\n'

exit "$FAILED"
```

- [ ] **Step 2: 赋予执行权限**

```bash
chmod +x frontend/cn-site-verify.sh
```

- [ ] **Step 3: 运行，确认全红**

Run: `./frontend/cn-site-verify.sh`
Expected: `FAIL 站点目录不存在：.../cn-site`，退出码 1

- [ ] **Step 4: 提交**

```bash
git add frontend/cn-site-verify.sh
git commit -m "test(cn-site): 验收脚本 A 组——静态站护栏的可执行检查"
```

---

## Task 2: `index.html` 主页面

让 Task 1 的 A1–A6 转绿。

**Files:**
- Create: `frontend/cn-site/index.html`
- Create: `frontend/cn-site/assets/favicon.svg`

**Interfaces:**
- Consumes: 无
- Produces: `frontend/cn-site/index.html`。必须含 `<meta name="generator" content="copiioai-cn-site">`（A6 与 B2 依赖）、页脚节点 `id="icp-license"` 与 `id="police-license"`（A5 依赖）、首屏注释位 `<!-- T3: 小程序码 -->`（计划末尾 T3 依赖）。

- [ ] **Step 1: 建目录与图标**

```bash
mkdir -p frontend/cn-site/assets
```

创建 `frontend/cn-site/assets/favicon.svg`：

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" role="img" aria-label="CopiioAI">
  <rect width="64" height="64" rx="14" fill="#1a56db"/>
  <path d="M40.5 24.8a11 11 0 1 0 0 14.4" fill="none" stroke="#fff" stroke-width="6" stroke-linecap="round"/>
  <circle cx="44" cy="32" r="4" fill="#fff"/>
</svg>
```

- [ ] **Step 2: 写页面**

创建 `frontend/cn-site/index.html`：

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="generator" content="copiioai-cn-site">
<title>CopiioAI — AI 专利情报检索与分析</title>
<meta name="description" content="CopiioAI 面向专利工作提供 AI 情报能力：专利检索、查重、审查历史分析、同族分析与报告生成。数据来源覆盖 USPTO、CNIPA、EPO、JPO。">
<link rel="canonical" href="https://copiioai.cn/">
<link rel="icon" href="assets/favicon.svg" type="image/svg+xml">
<style>
  :root{
    --ink:#0d1117; --ink-2:#39414d; --ink-3:#6b7480;
    --line:#e3e6ea; --bg:#fff; --bg-2:#f6f8fa;
    --accent:#1a56db; --accent-ink:#fff; --maxw:1040px;
  }
  *{box-sizing:border-box}
  body{
    margin:0;background:var(--bg);color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
    font-size:16px;line-height:1.75;-webkit-text-size-adjust:100%;
  }
  .wrap{max-width:var(--maxw);margin:0 auto;padding:0 24px}
  a{color:var(--accent)}

  header{border-bottom:1px solid var(--line)}
  .bar{display:flex;align-items:center;gap:12px;height:64px}
  .bar svg{width:28px;height:28px;flex:none}
  .brand{font-weight:600;letter-spacing:.2px}
  .bar nav{margin-left:auto;display:flex;gap:24px;font-size:14px}
  .bar nav a{color:var(--ink-2);text-decoration:none}
  .bar nav a:hover{color:var(--accent)}

  .hero{padding:88px 0 72px;text-align:center}
  .eyebrow{font-size:13px;letter-spacing:.12em;color:var(--ink-3);margin:0 0 16px}
  h1{font-size:clamp(30px,5vw,46px);line-height:1.25;margin:0 0 20px;letter-spacing:-.02em}
  .lede{font-size:clamp(16px,2vw,19px);color:var(--ink-2);max-width:640px;margin:0 auto 36px}
  .cta{display:inline-block;background:var(--accent);color:var(--accent-ink);
       padding:14px 32px;border-radius:8px;text-decoration:none;font-weight:500}
  .cta:hover{background:#1746b0}
  .cta-note{margin:16px 0 0;font-size:14px;color:var(--ink-3)}

  section{padding:64px 0;border-top:1px solid var(--line)}
  h2{font-size:24px;margin:0 0 28px;letter-spacing:-.01em}
  .grid{display:grid;gap:20px;grid-template-columns:repeat(auto-fit,minmax(220px,1fr))}
  .card{border:1px solid var(--line);border-radius:10px;padding:22px 20px}
  .card h3{font-size:16px;margin:0 0 8px}
  .card p{margin:0;font-size:14px;color:var(--ink-2)}

  .tags{display:flex;flex-wrap:wrap;gap:10px;margin-top:4px}
  .tag{border:1px solid var(--line);border-radius:999px;padding:6px 16px;
       font-size:14px;color:var(--ink-2);background:var(--bg-2)}

  ol.steps{margin:0;padding-left:22px}
  ol.steps li{margin-bottom:10px;color:var(--ink-2)}

  footer{border-top:1px solid var(--line);background:var(--bg-2);
         padding:40px 0;font-size:14px;color:var(--ink-3)}
  .foot{display:flex;flex-wrap:wrap;gap:12px 28px;align-items:center}
  .foot a{color:var(--ink-2);text-decoration:none}
  .foot a:hover{color:var(--accent);text-decoration:underline}
  .foot .sep{margin-left:auto}
  .licenses{margin-top:18px;display:flex;flex-wrap:wrap;gap:8px 20px}
  .licenses a{color:var(--ink-3)}
  @media (max-width:640px){
    .bar nav{display:none}
    .hero{padding:56px 0 48px}
    .foot .sep{margin-left:0}
  }
</style>
</head>
<body>

<header>
  <div class="wrap bar">
    <svg viewBox="0 0 64 64" aria-hidden="true">
      <rect width="64" height="64" rx="14" fill="#1a56db"/>
      <path d="M40.5 24.8a11 11 0 1 0 0 14.4" fill="none" stroke="#fff" stroke-width="6" stroke-linecap="round"/>
      <circle cx="44" cy="32" r="4" fill="#fff"/>
    </svg>
    <span class="brand">CopiioAI</span>
    <nav>
      <a href="#features">核心能力</a>
      <a href="#sources">数据来源</a>
      <a href="#usage">使用方式</a>
    </nav>
  </div>
</header>

<main>

<div class="hero wrap">
  <p class="eyebrow">AI 专利情报平台</p>
  <h1>专利情报，一问即得</h1>
  <p class="lede">
    用自然语言提问，直接得到检索结果、审查历史、同族信息与分析报告。
    面向专利代理人、企业 IP 负责人与研发工程师。
  </p>
  <a class="cta" href="https://copiioai.com/">前往 CopiioAI 使用</a>
  <p class="cta-note">也可在微信小程序中使用</p>
  <!-- T3: 小程序上线后，在此处加入小程序码区块（<img src="assets/weapp-qr.png" alt="微信小程序码">） -->
</div>

<section id="features">
  <div class="wrap">
    <h2>核心能力</h2>
    <div class="grid">
      <div class="card">
        <h3>专利检索</h3>
        <p>按技术方案描述检索，不依赖关键词组合的准确度。</p>
      </div>
      <div class="card">
        <h3>专利查重</h3>
        <p>对拟申请方案做在先文献比对，辅助可专利性初判。</p>
      </div>
      <div class="card">
        <h3>审查历史分析</h3>
        <p>还原审查过程中的意见与答复脉络，理解权利要求如何被界定。</p>
      </div>
      <div class="card">
        <h3>同族分析</h3>
        <p>跨法域梳理同族成员与各自的法律状态。</p>
      </div>
      <div class="card">
        <h3>报告生成</h3>
        <p>将检索与分析结果整理为可直接使用的文档。</p>
      </div>
    </div>
  </div>
</section>

<section id="sources">
  <div class="wrap">
    <h2>数据来源</h2>
    <p style="color:var(--ink-2);margin:0 0 20px">
      数据均来自各国专利主管机关公开的专利文献。
    </p>
    <div class="tags">
      <span class="tag">USPTO 美国专利商标局</span>
      <span class="tag">CNIPA 中国国家知识产权局</span>
      <span class="tag">EPO 欧洲专利局</span>
      <span class="tag">JPO 日本特许厅</span>
    </div>
  </div>
</section>

<section id="usage">
  <div class="wrap">
    <h2>使用方式</h2>
    <ol class="steps">
      <li>直接提问：用一句话描述你要找的技术方案。</li>
      <li>上传文件：提交 PDF、DOCX 或 XML 格式的专利文档。</li>
      <li>粘贴链接：给出专利页地址，由系统读取。</li>
      <li>批量任务：一次提交多篇文献，等待分析结果。</li>
    </ol>
  </div>
</section>

</main>

<footer>
  <div class="wrap">
    <div class="foot">
      <span>北京酷彼智能科技有限公司</span>
      <a href="privacy-policy.html">隐私政策</a>
      <a href="mailto:support@copiioai.com">support@copiioai.com</a>
      <span class="sep">© 2026 CopiioAI</span>
    </div>
    <div class="licenses">
      <!-- T1: ICP 备案获批后，填入备案号并把 data-pending 改为 false -->
      <span id="icp-license" data-pending="true"></span>
      <!-- T2: 公安备案获批后，填入备案号并把 data-pending 改为 false -->
      <a id="police-license" data-pending="true" href="https://beian.mps.gov.cn/"></a>
    </div>
  </div>
</footer>

</body>
</html>
```

- [ ] **Step 3: 运行验收，确认 A 组转绿**

Run: `./frontend/cn-site-verify.sh`
Expected: A1、A2a、A2b、A2c、A3、A4、A5、A6、A7 全部 `PASS`；A5b 出现一条 `WARN`（备案号仍为占位，符合预期）；汇总为"全部通过（1 项提醒）"，退出码 0

- [ ] **Step 4: 提交**

```bash
git add frontend/cn-site/index.html frontend/cn-site/assets/favicon.svg
git commit -m "feat(cn-site): 静态介绍页主页面——零外部请求、无表单、无 API 调用"
```

---

## Task 3: 附属文件

补齐隐私政策、爬虫规则、站点地图，并让新文件同样受 A 组约束。

**Files:**
- Create: `frontend/cn-site/privacy-policy.html`
- Create: `frontend/cn-site/robots.txt`
- Create: `frontend/cn-site/sitemap.xml`

**Interfaces:**
- Consumes: Task 1 的 A 组检查（新文件会自动被 `--include='*.html'` 覆盖）
- Produces: 无下游依赖

- [ ] **Step 1: 写隐私政策**

创建 `frontend/cn-site/privacy-policy.html`：

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="generator" content="copiioai-cn-site">
<title>隐私政策 — CopiioAI</title>
<meta name="description" content="copiioai.cn 本站仅提供产品介绍，不提供账号功能，不收集个人信息。">
<link rel="canonical" href="https://copiioai.cn/privacy-policy.html">
<link rel="icon" href="assets/favicon.svg" type="image/svg+xml">
<style>
  :root{--ink:#0d1117;--ink-2:#39414d;--ink-3:#6b7480;--line:#e3e6ea;--bg:#fff;--accent:#1a56db;--maxw:760px}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
    font-size:16px;line-height:1.85;-webkit-text-size-adjust:100%}
  .wrap{max-width:var(--maxw);margin:0 auto;padding:0 24px}
  a{color:var(--accent)}
  header{border-bottom:1px solid var(--line)}
  .bar{display:flex;align-items:center;height:64px;gap:12px}
  .brand{font-weight:600}
  .bar a{margin-left:auto;font-size:14px;color:var(--ink-2);text-decoration:none}
  main{padding:56px 0 72px}
  h1{font-size:clamp(26px,4vw,34px);margin:0 0 8px;letter-spacing:-.01em}
  .updated{color:var(--ink-3);font-size:14px;margin:0 0 40px}
  h2{font-size:18px;margin:36px 0 12px}
  p,li{color:var(--ink-2)}
  ul{padding-left:22px}
  footer{border-top:1px solid var(--line);background:#f6f8fa;padding:28px 0;
    font-size:14px;color:var(--ink-3)}
</style>
</head>
<body>

<header>
  <div class="wrap bar">
    <span class="brand">CopiioAI</span>
    <a href="index.html">返回首页</a>
  </div>
</header>

<main class="wrap">
  <h1>隐私政策</h1>
  <p class="updated">更新日期：2026 年 9 月 23 日</p>

  <h2>一、本页说明的范围</h2>
  <p>
    本政策适用于 <strong>copiioai.cn</strong> 域名下的页面。本站是 CopiioAI 的产品介绍页，
    <strong>不提供账号注册、登录、对话、上传等任何需要提交信息的功能</strong>。
  </p>

  <h2>二、我们不收集的信息</h2>
  <p>本站不设置任何表单、输入框或提交入口，因此不会收集：</p>
  <ul>
    <li>姓名、手机号、邮箱等联系方式</li>
    <li>身份证件信息</li>
    <li>您在本站填写的任何内容</li>
  </ul>

  <h2>三、服务器访问日志</h2>
  <p>
    根据《中华人民共和国网络安全法》第二十一条关于网络日志留存的要求，本站服务器会
    自动记录访问日志，内容包括访问时间、来源 IP 地址、请求的页面、浏览器类型等。
  </p>
  <p>
    该日志<strong>仅用于网络安全防护与依法配合监管</strong>，保留期限不少于六个月，
    不用于用户画像、广告投放或任何商业分析目的。
  </p>

  <h2>四、第三方</h2>
  <p>
    本站页面不加载任何第三方资源，不嵌入第三方统计、广告或视频服务，因此不会有
    您的访问数据被第三方获取。
  </p>

  <h2>五、产品服务的数据处理</h2>
  <p>
    使用 CopiioAI 产品（网页端、浏览器扩展、微信小程序）时的数据处理规则，
    不适用本政策，请以对应产品内的隐私说明为准。
  </p>

  <h2>六、您的权利</h2>
  <p>
    如您对本站的日志记录有查询、更正或删除的需求，或对本政策有任何疑问，请联系：
    <a href="mailto:support@copiioai.com">support@copiioai.com</a>。
  </p>

  <h2>七、政策更新</h2>
  <p>本政策如有变更，我们会在本页面更新并注明更新日期。</p>
</main>

<footer>
  <div class="wrap">北京酷彼智能科技有限公司 · © 2026 CopiioAI</div>
</footer>

</body>
</html>
```

- [ ] **Step 2: 写 robots.txt**

创建 `frontend/cn-site/robots.txt`：

```text
User-agent: *
Allow: /

Sitemap: https://copiioai.cn/sitemap.xml
```

- [ ] **Step 3: 写 sitemap.xml**

创建 `frontend/cn-site/sitemap.xml`：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://copiioai.cn/</loc>
  </url>
  <url>
    <loc>https://copiioai.cn/privacy-policy.html</loc>
  </url>
</urlset>
```

- [ ] **Step 4: 运行验收，确认仍全绿**

Run: `./frontend/cn-site-verify.sh`
Expected: A1–A7 全部 `PASS`（新加的 `privacy-policy.html` 自动纳入检查且不违规），退出码 0

若 A2b 报出 `sitemap.xml` 相关项——不会，因为 `--include='*.html'` 只扫 HTML。若报出 privacy-policy 的外链，检查是否误加了非白名单链接。

- [ ] **Step 5: 确认站点目录内容即为上传清单**

Run: `find frontend/cn-site -type f | sort`
Expected:
```
frontend/cn-site/assets/favicon.svg
frontend/cn-site/index.html
frontend/cn-site/privacy-policy.html
frontend/cn-site/robots.txt
frontend/cn-site/sitemap.xml
```
这五个文件**就是**要上传到桶里的全部内容，不多不少。

- [ ] **Step 6: 提交**

```bash
git add frontend/cn-site/privacy-policy.html frontend/cn-site/robots.txt frontend/cn-site/sitemap.xml
git commit -m "feat(cn-site): 隐私政策、robots 与 sitemap"
```

---

## Task 4: 部署清单（COS + CDN + 证书 + 日志）

控制台操作，无代码。产出物是一份可执行清单，每步带验证命令。

**Files:**
- Create: `docs/superpowers/runbooks/2026-09-23-cn-site-deploy.md`

**Interfaces:**
- Consumes: Task 3 的五个静态文件
- Produces: 一个已配好、可访问的 CDN 测试地址。Task 5 的切换依赖它

- [ ] **Step 1: 写清单文档**

创建 `docs/superpowers/runbooks/2026-09-23-cn-site-deploy.md`：

````markdown
# copiioai.cn 静态站部署清单

- 日期：2026-09-23
- 对应设计：`docs/superpowers/specs/2026-09-23-cn-static-site-design.md` §4.3、§5
- 性质：控制台操作，**不修改任何线上 DNS**。本清单只负责把新站准备好

---

## 前置

- [ ] 上传前先跑：`./frontend/cn-site-verify.sh` —— **A 组必须全绿**，这是不会把应用产物或境外依赖带进桶的保证

## D1 新建 COS 桶

- [ ] 新建存储桶，地域选**广州或上海**
- [ ] 权限：**公有读私有写**
- [ ] 记录桶名：`____________________`（含 APPID 后缀）

## D2 上传静态文件

- [ ] 上传 `frontend/cn-site/` 下的**全部且仅有的五个文件**：
      `index.html`、`privacy-policy.html`、`robots.txt`、`sitemap.xml`、`assets/favicon.svg`

> ⚠️ 不要上传 `frontend/cn-site-verify.sh`——它在站点目录之外，本就不该出现在上传源里。

- [ ] 开启**静态网站托管**：默认首页设 `index.html`
- [ ] **默认 404 页留空**，不要指向 `index.html`

> 404 页**不要**指向 `index.html`：那会让 `/app` 等路径返回首页内容，`curl` 拿到 200，
> 与验收脚本 B1 的期望（必须 404）冲突。留空即可让 COS 返回标准 404。

- [ ] 验证桶内清单：`aws s3 ls` 或控制台对象列表 —— **必须恰好是那五个文件**

## D3 绑定 CDN 加速域名

- [ ] 为该桶添加**自定义 CDN 加速域名**：`copiioai.cn`
- [ ] 源站类型：COS 源站，指向 D1 建的桶
- [ ] 回源协议：HTTPS
- [ ] 记录腾讯云为此分配的**加速 CNAME**（形如 `copiioai.cn.cdn.dnsv1.com`）：`____________________`

## D4 SSL 证书（对应设计 §7.2 N1）

- [ ] 在 CDN 侧配置 SSL 证书
- [ ] **确认续期方式是自动的**

> ⚠️ 这是本次迁移最容易埋雷的一步。当前 `.cn` 的证书是 Cloudflare 自动续期的
> 90 天 Let's Encrypt 证书（2026-12-21 到期）。迁出后若续期靠人工，
> **90 天后站点会静默失去 HTTPS**——没人会发现，直到用户报错或备案审查时打开。
>
> 若确认腾讯云侧是自动续期：在下方打勾。若不是：设一个 90 天周期的日历提醒。
>
> - [ ] 已确认证书续期为自动

## D5 访问日志（对应设计 §5）

- [ ] 开通 **CDN 访问日志**
- [ ] 投递目标：**专用日志桶**（不要用站点桶）
- [ ] 确认日志字段包含**真实客户端 IP**
- [ ] 为该日志桶配置**生命周期规则：保留 12 个月**，到期自动转归档或删除

> 注意：CDN 日志生成有**小时级延迟**。刚访问完看不到记录不代表失败。

## D6 配置 DNS（先不切 NS）

- [ ] 在 DNSPod 新建 `copiioai.cn` 域名，**先不动现有 NS**
- [ ] 建好全部解析记录：
      - `copiioai.cn` → **D3 记录的那个加速 CNAME**
        （不是"CDN 加速域名"本身——`copiioai.cn` 指向 `copiioai.cn` 是循环解析）
      - **把现有 Cloudflare 里的每一条记录都抄过来**，逐条核对，不要漏
- [ ] 记录当前 NS 与全部解析，截图存档：`____________________`

## D7 验证（在切 DNS 之前）

- [ ] 用 CDN 的测试域名访问：`curl -sI https://<CDN测试域名>/` → 200
- [ ] 在**测试域名**上跑 B 组：
      `./frontend/cn-site-verify.sh https://<CDN测试域名>`
      预期：B1 八条路径全部 404、B2 首页为本站文件

> 若 B1 出现 200 而非 404，回到 D2 检查静态网站托管的 404 配置。

- [ ] 确认日志桶在访问后（等待小时级延迟）出现了记录，且 IP 是**你的真实公网 IP**
      而非 CDN 回源 IP

> 这一条是整个留存义务的关键验证。拿 `curl ifconfig.me` 的结果与日志里的 IP 比对。

## 完成后

- [ ] 提交本清单的填写结果（桶名、测试域名、证书续期确认情况）到仓库
- [ ] 进入 `2026-09-23-cn-site-switch.md`
````

- [ ] **Step 2: 提交**

```bash
git add docs/superpowers/runbooks/2026-09-23-cn-site-deploy.md
git commit -m "docs(cn-site): 部署清单——COS 桶、CDN、证书续期确认、日志留存"
```

---

## Task 5: 切换清单（DNS 迁移 + 回滚）

**Files:**
- Create: `docs/superpowers/runbooks/2026-09-23-cn-site-switch.md`

**Interfaces:**
- Consumes: Task 4 产出的 CDN 测试地址与已配好的桶
- Produces: 完成迁移的 `.cn`

- [ ] **Step 1: 写清单文档**

创建 `docs/superpowers/runbooks/2026-09-23-cn-site-switch.md`：

````markdown
# copiioai.cn 切换清单

- 日期：2026-09-23
- 对应设计：`docs/superpowers/specs/2026-09-23-cn-static-site-design.md` §4.4、§4.5
- 前置：`2026-09-23-cn-site-deploy.md` 已全部完成
- 原则：**只动 `.cn`。`copiioai.com` 全程不碰**

---

## 切换前准备

- [ ] **T-24h：把 `.cn` 的 TTL 调到 60 秒**
      不做这一步的话，下面第 6 步和第 9 步的变更与回滚都要等原 TTL 过期

- [ ] 记录当前状态截图存档：
      - NS：`____________________`
      - `copiioai.cn` 的解析记录：`____________________`
      - 回滚时需要还原成的样子

- [ ] 确认 DNSPod 侧已建好全部记录（部署清单 D6）

---

## 10 步切换

第 **6** 步与第 **9** 步刻意拆开：站点替换风险最高但回滚是**秒级**；
DNS 权威迁移风险低但回滚要 **24–48 小时**。
把慢回滚的那步放在站点已验证正确之后，两类问题就不会在同一个窗口里互相掩盖。

- [ ] **1.** 新桶已就绪，五个文件在位
- [ ] **2.** CDN 加速域名 + SSL 证书已配好
- [ ] **3.** 测试域名已验证通过（部署清单 D7）
- [ ] **4.** `.cn` 的 TTL 已降至 60 秒且已生效
- [ ] **5.** 当前 NS 与解析已截图存档
- [ ] **6.** **在 Cloudflare 里把 `.cn` 的 CNAME 改指新 CDN**
      ← 站点内容在此刻切换。**这是风险最高的一步，也是回滚最快的一步**

      验证：
      ```bash
      curl -sI https://copiioai.cn/ | head -5      # 应 200
      ./frontend/cn-site-verify.sh https://copiioai.cn
      ```
- [ ] **7.** **B 组必须全绿**（B1 八条路径 404、B2 首页为本站文件）。
      未全绿则执行下面的「回滚 A」，**不要继续**

      > 到这一步为止，站点已经是新的、正确的、且 `.cn` 已指向它。
      > 如果想在此收工、改天再切 NS——完全可以。那时站点已是稳态，
      > 第 8–10 步纯粹是收尾。

- [ ] **8.** DNSPod 侧记录已就绪（重复确认，不漏 CNAME 之外的记录）
- [ ] **9.** **切换 NS 到 DNSPod**
- [ ] **10.** 验证：

      ```bash
      nslookup -type=NS copiioai.cn          # 应显示 DNSPod 的 NS
      curl -sI https://copiioai.cn/ | head -5
      ./frontend/cn-site-verify.sh https://copiioai.cn
      ```

      另需确认**证书是新的那张**（不再是 Cloudflare 签发）：
      ```bash
      echo | openssl s_client -connect copiioai.cn:443 -servername copiioai.cn 2>/dev/null | openssl x509 -noout -issuer -dates
      ```

---

## 回滚

| 出问题的步骤 | 回滚动作 | 生效速度 |
|---|---|---|
| 第 6 步之后站点异常 | **回滚 A**：CNAME 改回旧桶 | 秒级（TTL 已调小） |
| 第 9 步之后解析异常 | **回滚 B**：NS 切回 Cloudflare | **24–48 小时** |

- [ ] **回滚 A**：在 Cloudflare 把 `.cn` 的 CNAME 改回原值，恢复原状
- [ ] **回滚 B**：把 NS 切回原 registrar 的 NS，按切换前截图的记录还原

> ⚠️ **回滚边界**：回滚**只允许回到旧桶**（即恢复原状）。
> **不得**把 `.cn` 指向任何应用产物或应用入口以"临时救急"——
> 那正是设计文档 §1.3 条件 3 禁止的做法，会把"官网"变成"静态壳"。

---

## 切换后

- [ ] 观察 24 小时，确认站点可用、证书有效、日志有记录
- [ ] 确认日志中的 IP 仍是真实用户 IP（NS 切换不应影响这一点，但要复核）
- [ ] 记录切换完成时间：`____________________`
````

- [ ] **Step 2: 提交**

```bash
git add docs/superpowers/runbooks/2026-09-23-cn-site-switch.md
git commit -m "docs(cn-site): 切换清单——10 步迁移与双通道回滚"
```

---

## 完成后的时序待办

这三项**不在本次实现范围内**，但必须在对应事件发生时执行。页脚没有备案号本身就是备案例行检查项。

| # | 待办 | 触发条件 | 落点 |
|---|---|---|---|
| T1 | 页脚填入 ICP 备案号 | ICP 备案获批 | `index.html` 的 `id="icp-license"`，填文本并把 `data-pending` 改为 `false` |
| T2 | 页脚填入公安备案号 | 公安备案获批 | `index.html` 的 `id="police-license"`，填文本并把 `data-pending` 改为 `false` |
| T3 | 首屏加入小程序码 | 微信小程序上线 | `index.html` 首屏 `<!-- T3: 小程序码 -->` 注释位；图片放 `assets/weapp-qr.png` |

每项完成后跑一次 `./frontend/cn-site-verify.sh`（A5b 的 WARN 应消失），重新上传对应文件到桶。

---

## 自检记录

- **规格覆盖**：设计 §3（页面）→ Task 2、3；§4（源站与切换）→ Task 4、5；§5（日志）→ Task 4 D5；§6（验收）→ Task 1；§7.2（N1）→ Task 4 D4；§8（时序待办）→ 本文件末节
- **命名一致性**：`copiioai-cn-site`（A6/B2 的标识）、`icp-license` / `police-license`（A5 与 T1/T2）、`data-pending`（A5b）三处标识在 Task 1、2 与待办表中一致
- **一处规格修正**：设计 §3.1 未提及站点标识 meta 与 A7 检查。A6 为支撑 B2 的必要补充，A7 为保障"站点目录只有可上传文件"这一前提的必要补充，两者均已写入 Task 1
````
