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

# ---------------------------------------------------------------- A 组

hr "A 组：本地静态检查（$SITE_DIR）"

if [ ! -d "$SITE_DIR" ]; then
  fail "站点目录不存在：$SITE_DIR"
  printf '\n失败 %d 项\n' "$FAILED"
  exit "$FAILED"
fi

# A1 无表单与输入（大小写不敏感：<FORM> 同样是表单）
hits=$(GI '<form|<input|<textarea|<select' || true)
if [ -n "$hits" ]; then
  fail "A1 发现表单或输入元素（构成收集个人信息）"
  printf '%s\n' "$hits"
else
  pass "A1 无表单与输入"
fi

# A2a 资源零外链：src 属性一定是加载资源，不得出现绝对 URL
# 容忍双引号、单引号、无引号三种属性写法
hits=$(GI 'src[[:space:]]*=[[:space:]]*["'"'"']?[^"'"'"'[:space:]>]*://' || true)
if [ -n "$hits" ]; then
  fail "A2a 资源 src 指向外部地址"
  printf '%s\n' "$hits"
else
  pass "A2a 资源 src 全部本地"
fi

# A2b 超链接与 link[href] 仅允许白名单域名
bad=$(grep -rhoiE 'href[[:space:]]*=[[:space:]]*["'"'"']?https?://[^"'"'"'[:space:]>]*' \
    "$SITE_DIR" 2>/dev/null \
  | sed -E 's/^[^h]*href[[:space:]]*=[[:space:]]*["'"'"']?//; s/["'"'"']$//' | sort -u \
  | grep -vE '^https?://(www\.)?copiioai\.(com|cn)(/|$)' \
  | grep -vE '^https?://beian\.miit\.gov\.cn(/|$)' \
  | grep -vE '^https?://beian\.mps\.gov\.cn(/|$)' \
  | grep -v '^$' || true)
if [ -n "$bad" ]; then
  fail "A2b 存在非白名单域名（境外依赖或未预期外链）"
  printf '%s\n' "$bad"
else
  pass "A2b 外链仅限白名单（copiioai.com/cn、备案查询站）"
fi

# A2c 内联 CSS 不得引用外部资源
hits=$(grep -rInoE "url\([^)]*https?://" "$SITE_DIR" 2>/dev/null || true)
if [ -n "$hits" ]; then
  fail "A2c 内联样式引用了外部资源"
  printf '%s\n' "$hits"
else
  pass "A2c 内联样式无外部资源"
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
