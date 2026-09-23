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

# grep 快捷方式：递归、显示行号、忽略二进制，扫描站点目录下全部文件。
# 刻意不加 --include 过滤：站点目录里的任何文件（.js/.css/.json/…）都不能夹带违规内容，
# 只查 *.html 会让放在同目录的 app.js、cfg.json 完全逃过检查。
#
# 传参纪律（本机踩过的坑，务必遵守）：pattern 只能作为**唯一的** "$1" 传给 grep。
# 本机 MINGW64 的 GNU grep 3.0 在选项串与 pattern 分开经 argv 传入时会丢 pattern 而恒返回空
# （实测：grep "$opts" "$@" DIR 恒空；grep -rInoE -i PATTERN 恒空；grep -rInoE "$1" DIR 正常）。
# 因此这里把选项直接写死在函数里、pattern 走单一 "$1"，不做任何选项透传。
G()  { grep -rInoE "$1" "$SITE_DIR" 2>/dev/null || true; }   # 大小写敏感
GI() { grep -rInoEi "$1" "$SITE_DIR" 2>/dev/null || true; }  # 大小写不敏感

# 备案号占位节点是否已填入真实号码
PENDING_RE='data-pending="true"'

# 全站 URL 白名单（A2b 与 A2d 共用）
#
# 刻意收窄到**精确路径**，而不是整个域名：
# 放行 `copiioai.(com|cn)` 的任意路径时，注入 `https://copiioai.com/chat` 或
# `https://copiioai.cn/workspace` 也能全绿（A4 的词表里没有 chat/workspace）。
# 而"不得有任何指向应用的深链接"是本项目第一红线，不能只靠一份有限的词表来守。
# 放行清单只保留四类：
#   1) .com 的根路径（CTA 出口）——注意允许 `copiioai.com` 后面**不**跟路径，
#      `?`/`#` 是两种合法的收尾；任何斜杠开头的更深路径一律不放行
#   2) .cn 的已知静态页（收敛到枚举，不是任意路径）
#   3) 备案查询站
#   4) XML 命名空间：sitemap.xml 的 www.sitemaps.org 与 favicon.svg 的 www.w3.org。
#      这两个是语法要求的 URI 标识符，不发起任何网络请求；换任何写法都会让
#      XML 解析器不认命名空间。收敛到各自文档规定的精确版本路径。
WL_RE='^https?://(copiioai\.com/?([?#].*)?$|copiioai\.cn/(|index\.html|privacy-policy\.html|sitemap\.xml|robots\.txt)$|beian\.miit\.gov\.cn/|beian\.mps\.gov\.cn/|www\.sitemaps\.org/schemas/sitemap/0\.9$|www\.w3\.org/2000/svg$)'

# 全部绝对 URL 的提取（A2d 用；A2b 只取 href=，两者共用同一套字符类）
#
# 逐条说明这些字符类为什么必须这么写：
#   - `<` `>` 必须排除：否则 sitemap.xml 的 `<loc>https://x/</loc>` 会把 `</loc>` 一起吞进来
#   - 引号、空白必须排除：否则会吞掉结束引号与后面的标签
#   - `)` 必须排除：否则内联 CSS 的 `url(...)` 会把右括号吞进来
#   - 末尾标点要在提取后用 sed 剥掉：`见 https://x/y.` 这类句末句号不是 URL 的一部分
#   - 协议相对 URL（`//host/path`）没有 scheme，上面两条都抓不到，由 A2a/A2d 单独处理
URL_EXTRACT_RE='https?://[^<>"'"'"'[:space:])]+'

strip_trailing_punct() { sed -E 's/[.,;:!?]+$//'; }

# ---------------------------------------------------------------- A 组

hr "A 组：本地静态检查（$SITE_DIR）"

if [ ! -d "$SITE_DIR" ]; then
  fail "站点目录不存在：$SITE_DIR"
  printf '\n失败 %d 项\n' "$FAILED"
  exit "$FAILED"
fi

# A1 无表单与输入（大小写不敏感：<FORM> 同样是表单）
# contenteditable 必须一并拦截：它是真实的可编辑输入面，与被拦截的 <input>/<textarea>
# 在"页面是否具备收集个人信息的能力"这一点上等价，漏掉它等于给红线留了后门
# （<div contenteditable="true"> 与空字符串写法 contenteditable="" 都算）。
# 刻意不匹配 type=：本页有合法的 <link rel="icon" type="image/svg+xml"> 与 <style>，
# 一加 type= 就会立刻误报，属于假阳性陷阱。
hits=$(GI '<form|<input|<textarea|<select|contenteditable' || true)
if [ -n "$hits" ]; then
  fail "A1 发现表单、输入元素或可编辑区域（构成收集个人信息）"
  printf '%s\n' "$hits"
else
  pass "A1 无表单、输入与可编辑区域"
fi

# A2a 资源零外链：src 属性一定是加载资源，不得出现绝对 URL，也不得用协议相对写法
# 两种写法都要拦：
#   1) 有 scheme 的绝对 URL —— `src="https://evil.example/x.js"`（匹配 `://`）
#   2) 协议相对 URL —— `src="//evil.example/x.js"`，它**没有** scheme，
#      只查 `://` 会完全放行；浏览器会按当前页面的协议补全它，效果与外链等同
# 正常相对路径（`assets/favicon.svg`、`privacy-policy.html`）不以 `//` 开头，不受影响。
# 容忍双引号、单引号、无引号三种属性写法
hits=$(GI 'src[[:space:]]*=[[:space:]]*["'"'"']?([^"'"'"'[:space:]>]*://|//)' || true)
if [ -n "$hits" ]; then
  fail "A2a 资源 src 指向外部地址（含协议相对写法 //host/…）"
  printf '%s\n' "$hits"
else
  pass "A2a 资源 src 全部本地"
fi

# A2b 超链接与 link[href] 仅允许白名单**精确路径**（白名单见文件头 WL_RE）
bad=$(grep -rhoiE 'href[[:space:]]*=[[:space:]]*["'"'"']?https?://[^"'"'"'[:space:]>]*' \
    "$SITE_DIR" 2>/dev/null \
  | sed -E 's/^[^h]*href[[:space:]]*=[[:space:]]*["'"'"']?//; s/["'"'"']$//' \
  | strip_trailing_punct | sort -u \
  | grep -vE "$WL_RE" \
  | grep -v '^$' || true)
if [ -n "$bad" ]; then
  fail "A2b 存在非白名单 URL（境外依赖、未预期外链或应用深链接）"
  printf '%s\n' "$bad"
else
  pass "A2b 外链仅限白名单（copiioai.com 根路径、.cn 已知静态页、备案查询站）"
fi

# A2c 内联 CSS 不得引用外部资源
hits=$(grep -rInoE "url\([^)]*https?://" "$SITE_DIR" 2>/dev/null || true)
if [ -n "$hits" ]; then
  fail "A2c 内联样式引用了外部资源"
  printf '%s\n' "$hits"
else
  pass "A2c 内联样式无外部资源"
fi

# A2d 站点内**全部**绝对 URL 都要过白名单（不限 href=）
#
# 为什么必须有这一条：href= 之外还有大量能把访问者带走的载体，而它们都不含 `href=`——
#   - robots.txt 与 sitemap.xml 没有 href 也没有 src，A2b/A2a 对它们**完全失明**
#     （实测：把 robots.txt 换成 `Sitemap: https://evil.example/…` 后 A 组全绿、退出码 0）
#   - `<meta http-equiv="refresh" content="0;url=https://…">` 用的是 content 属性
#   - `window.location='https://…'` 是脚本里的字符串字面量
#   - XML 命名空间 `xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"` 也在其中，
#     它是合法且在白名单里的，所以本检查不会误报
# 做法：把站点目录下**所有文件**的全部 `https?://…` 抽出来，逐条过 WL_RE。
hits=$(grep -rInoE "$URL_EXTRACT_RE" "$SITE_DIR" 2>/dev/null \
  | sed -E 's/^[^:]*:[0-9]+://' | strip_trailing_punct | sort -u \
  | grep -vE "$WL_RE" || true)
if [ -n "$hits" ]; then
  fail "A2d 站点内存在非白名单绝对 URL（含 robots/sitemap/跳转指令/脚本字符串）"
  printf '%s\n' "$hits"
else
  pass "A2d 站点内全部绝对 URL 均在白名单内"
fi

# A3 无 API 调用（含埋点）
hits=$(G 'fetch\(|XMLHttpRequest|sendBeacon|navigator\.sendBeacon|axios|api\.copiioai' || true)
if [ -n "$hits" ]; then
  fail "A3 页面存在 API 调用——只要还在调 API 就不是静态站"
  printf '%s\n' "$hits"
else
  pass "A3 无任何 API 调用"
fi

# A4 无应用路由深链接
hits=$(G 'href="[^"]*/(app|login|signup|community|sessions|knowledge|auth|api)(/|"|\?)' || true)
if [ -n "$hits" ]; then
  fail "A4 存在指向应用的深链接"
  printf '%s\n' "$hits"
else
  pass "A4 无应用路由深链接"
fi

# A5 备案号占位节点存在——**两个页面都查**
# 多数属地要求全站每个页面的页脚都展示备案号。只查 index.html 时，
# 监管人员打开隐私政策页看不到备案号，而脚本依然全绿（空转）。
miss=""
for page in index.html privacy-policy.html; do
  for node in icp-license police-license; do
    grep -q "id=\"$node\"" "$SITE_DIR/$page" 2>/dev/null || miss="$miss $page:$node"
  done
done
if [ -n "$miss" ]; then
  fail "A5 页脚缺少备案号占位节点：$miss"
else
  pass "A5 两个页面的备案号占位节点均存在"
fi

# A5b 备案号尚未填入时给出提醒（不算失败——备案获批前本就没有号码）
# 与 A5 同步查两个页面：只查 index.html 的话，隐私政策页漏填不会触发任何提醒。
pending_pages=""
for page in index.html privacy-policy.html; do
  grep -q "$PENDING_RE" "$SITE_DIR/$page" 2>/dev/null && pending_pages="$pending_pages $page"
done
if [ -n "$pending_pages" ]; then
  warn "A5b 页脚备案号仍为占位状态（$pending_pages）。备案获批后必须填入（见 runbook 的 T1/T2）"
fi

# A6 站点标识存在（B2 依赖它确认线上内容就是本站文件）
if grep -q 'name="generator" content="copiioai-cn-site"' "$SITE_DIR/index.html" 2>/dev/null; then
  pass "A6 站点标识存在"
else
  fail "A6 缺少 <meta name=\"generator\" content=\"copiioai-cn-site\">"
fi

# A7 站点目录内不得有任何开发工具/文档（防止被误传到桶里）
# 只守一个精确文件名挡不住改名与变体（cn-site-verify.sh、verify.sh.bak、_verify.sh 都会漏过），
# 因此改为整类拦截：点开头 / 以 .sh 结尾 / 名字含 verify / 以 .md 结尾。
[ -d "$SITE_DIR" ] || exit 0
devfiles=$(find "$SITE_DIR" -type f -printf '%f\n' 2>/dev/null \
  | grep -iE '(^\.|\.sh$|verify|\.md$)' | sort -u || true)
if [ -n "$devfiles" ]; then
  fail "A7 站点目录内存在开发工具/文档，会被上传到桶"
  printf '%s\n' "$devfiles"
else
  pass "A7 开发工具不在站点目录内"
fi

# ---------------------------------------------------------------- B 组

if [ -n "$BASE_URL" ]; then
  hr "B 组：线上检查（$BASE_URL）"

  # B1/B3 应用路径必须 404，且不得重定向
  # /chat 与 /workspace 是 A4 词表之外的常见应用路径：词表守的是本地文件里的 href，
  # 线上路径集守的是"桶里真的没有应用产物"，两者覆盖的载体不同，都要有。
  for p in /app /login /signup /community /sessions /knowledge /auth /api /chat /workspace; do
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

  # B2 首页可达且确实是本站文件——**状态码必须真的是 200**
  # 只 grep 内容是不够的：一个 302 到首页的响应同样会带着首页正文返回，
  # grep 照过，但它其实把访问者送到了别处。因此显式断言状态码。
  b2_code=$(curl -s -o /dev/null --max-time 15 -w '%{http_code}' "$BASE_URL/")
  b2_body=$(curl -s --max-time 15 "$BASE_URL/")
  if [ "$b2_code" != "200" ]; then
    fail "B2 首页返回 $b2_code（应为 200）"
  elif ! printf '%s' "$b2_body" | grep -q 'copiioai-cn-site'; then
    fail "B2 首页内容不是本站文件（缺 copiioai-cn-site 标识）"
  else
    pass "B2 首页 200 且为本站文件"
  fi

  # B2b 隐私政策页同样必须 200——它承载"本站不收集个人信息"的对外声明，
  # 404 或跳走会让这份声明在线上不可达，也让 A2b 放行的 canonical 成为死链。
  pp_code=$(curl -s -o /dev/null --max-time 15 -w '%{http_code}' "$BASE_URL/privacy-policy.html")
  if [ "$pp_code" = "200" ]; then
    pass "B2b 隐私政策页 200"
  else
    fail "B2b 隐私政策页返回 $pp_code（应为 200）"
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
