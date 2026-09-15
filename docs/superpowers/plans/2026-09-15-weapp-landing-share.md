# 小程序分享收敛为纯品牌卡片 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把小程序分享卡片的内容固定为品牌（标题 + 配图），使其不再携带用户的会话内容。

**Architecture:** 分享载荷的纯逻辑抽到 `src/utils/share.ts`（零资源依赖，可被 node 直载并测试）；配图 `import` 留在页面以保证资源被打包进 `dist/`；配图由一次性 Python 脚本生成并连同脚本一起提交。

**Tech Stack:** Taro 4.1 + React（小程序端）、TypeScript、node:test（测试）、Pillow 12.2（配图生成）

## Global Constraints

- 分支：`feat/weapp`
- 分享载荷**不得携带任何会话内容**（`sessionTitle`、`msgs` 一律不得进入分享链路）
- 分享标题固定为 `CopiioAI 专利情报，一问即得`
- 分享 `path` 固定为 `/pages/chat/index`
- 配图尺寸 500×400（微信标准 5:4），底色 `#f6f8fa`
- `src/utils/share.ts` **不得 import 任何资源**（含图片）——它必须能被 `node --test` 直接加载
- **不得修改** `frontend/weapp/tsconfig.test.json`
- 分享入口任何时候保留（不调 `hideShareMenu`）

---

## 文件结构

| 动作 | 文件 | 职责 |
|---|---|---|
| 增 | `frontend/weapp/src/utils/share.ts` | 分享载荷纯逻辑：品牌常量 + `buildShareCard()` |
| 增 | `frontend/weapp/src/utils/share.test.mjs` | 钉住「载荷不携带会话内容」的回归测试 |
| 增 | `frontend/weapp/scripts/make-share-card.py` | 生成分享卡配图，参数写死，可重跑 |
| 增 | `frontend/weapp/src/assets/share-card.png` | 配图产物（由上一行的脚本生成） |
| 改 | `frontend/weapp/src/pages/chat/index.tsx:261-277` | 接线；移除 `sessionTitle` 参与分享 |

---

## Task 1: 分享载荷纯模块（TDD）

**Files:**
- Create: `frontend/weapp/src/utils/share.ts`
- Test: `frontend/weapp/src/utils/share.test.mjs`

**Interfaces:**
- Consumes: 无
- Produces: `SHARE_TITLE: string`、`SHARE_PATH: string`、`buildShareCard(imagePath: string): { title: string; imageUrl: string }`

- [ ] **Step 1: 写失败的测试**

创建 `frontend/weapp/src/utils/share.test.mjs`：

```js
import test from 'node:test'
import assert from 'node:assert/strict'
import { SHARE_PATH, SHARE_TITLE, buildShareCard } from './share.js'

test('载荷字段固定：只有 title 与 imageUrl', () => {
  // 修复前的 bug：title 拼进 sessionTitle，把用户首问原文带到了群聊卡片上。
  // 这条钉住"不夹带字段"——将来若有人把会话数据塞进载荷，key 集合会变，测试红。
  assert.deepEqual(Object.keys(buildShareCard('x')).sort(), ['imageUrl', 'title'])
})

test('标题恒为品牌文案，与会话无关', () => {
  assert.equal(buildShareCard('x').title, SHARE_TITLE)
  assert.equal(SHARE_TITLE, 'CopiioAI 专利情报，一问即得')
})

test('imageUrl 原样透传调用方传入的配图路径', () => {
  assert.equal(buildShareCard('/assets/share-card.png').imageUrl, '/assets/share-card.png')
})

test('path 指向落地页', () => {
  assert.equal(SHARE_PATH, '/pages/chat/index')
})
```

- [ ] **Step 2: 运行测试确认它失败**

```bash
cd frontend/weapp && npx tsc -p tsconfig.test.json && node --test dist-test/src/utils/share.test.mjs
```

预期：**`tsc` 退出码为 0 且无输出** —— 实测确认 `allowJs` 下 `./share.js` 解析失败**不会**升级为 TS 错误，所以 `tsc` 这一步是绿灯。红灯出现在 `node --test` 层：

```
Error [ERR_MODULE_NOT_FOUND]: Cannot find module
  '...\frontend\weapp\dist-test\src\utils\share.js'
```

此刻 `share.ts` 尚不存在，这就是本轮的 RED。

**不要为了让 `tsc` 报错去改 `tsconfig.test.json`** —— 那是 Global Constraints 明令禁止的。红灯落在 `node --test` 层属正常，如实记录即可。

- [ ] **Step 3: 写最小实现**

创建 `frontend/weapp/src/utils/share.ts`：

```ts
/**
 * 分享载荷：标题与配图固定为品牌，不携带任何会话内容。
 *
 * 本模块**刻意不 import 任何资源** —— 它要能被 `node --test` 直接加载。
 * tsconfig.test.json 只编译不拷资源，若此处 import 图片，编译产物里的
 * require('../assets/share-card.png') 会指向不会生成的 dist-test/assets/，
 * 运行期直接 MODULE_NOT_FOUND（类型层面是合法的，见 types/assets.d.ts）。
 * 故配图路径由调用方（页面）import 后作为参数传入。
 *
 * 设计依据：docs/superpowers/specs/2026-09-15-weapp-landing-share-design.md
 */

export const SHARE_TITLE = 'CopiioAI 专利情报，一问即得'
export const SHARE_PATH = '/pages/chat/index'

/** 品牌卡片载荷：标题与配图固定，不携带任何会话内容。 */
export function buildShareCard(imagePath: string) {
  return { title: SHARE_TITLE, imageUrl: imagePath }
}
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd frontend/weapp && npx tsc -p tsconfig.test.json && node --test dist-test/src/utils/share.test.mjs
```

预期：PASS，`# pass 4` / `# fail 0`

> 注意产物路径是 `dist-test/**src**/utils/`（源码树结构被保留），不是 `dist-test/utils/`。

- [ ] **Step 5: 提交**

```bash
git add frontend/weapp/src/utils/share.ts frontend/weapp/src/utils/share.test.mjs
git commit -m "feat(weapp): 分享载荷纯模块 —— 品牌常量 + buildShareCard (需求: 分享不携带会话内容)"
```

---

## Task 2: 生成分享卡配图

**Files:**
- Create: `frontend/weapp/scripts/make-share-card.py`
- Create（脚本产物）: `frontend/weapp/src/assets/share-card.png`

**Interfaces:**
- Consumes: `frontend/weapp/src/assets/brand-logo.png`（已存在，480×546 透明底）
- Produces: `frontend/weapp/src/assets/share-card.png`（500×400），供 Task 3 以 `import` 引用

- [ ] **Step 1: 写生成脚本**

创建 `frontend/weapp/scripts/make-share-card.py`：

```python
"""生成小程序分享卡配图（500×400，微信标准 5:4）。

用法：cd frontend/weapp && python scripts/make-share-card.py
产物：src/assets/share-card.png

参数全部写死在此处，改文案或配色后重跑即可。脚本与产物一并提交。
设计依据：docs/superpowers/specs/2026-09-15-weapp-landing-share-design.md
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CANVAS_W, CANVAS_H = 500, 400
BG = '#f6f8fa'          # 小程序 --c-bg，与端内一致
TEXT_COLOR = '#1f2328'  # 小程序 --c-text
SLOGAN = '专利情报，一问即得'
FONT_SIZE = 28
LOGO_W = 200            # 品牌图缩放后宽度
GAP = 20                # 品牌图与标语间距

# 按平台候选，取第一个存在的。分享卡是离线一次性产物，不参与构建链路。
FONT_CANDIDATES = (
    'C:/Windows/Fonts/msyhbd.ttc',                              # 微软雅黑 Bold
    '/System/Library/Fonts/PingFang.ttc',                       #  macOS 苹方
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',      # Linux Noto CJK
)

ROOT = Path(__file__).resolve().parent.parent
SRC_LOGO = ROOT / 'src' / 'assets' / 'brand-logo.png'
OUT = ROOT / 'src' / 'assets' / 'share-card.png'


def load_font(size):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise SystemExit(f'找不到可用的中文字体，候选路径：{FONT_CANDIDATES}')


def main():
    logo = Image.open(SRC_LOGO).convert('RGBA')
    logo_h = round(LOGO_W * logo.height / logo.width)
    logo = logo.resize((LOGO_W, logo_h), Image.LANCZOS)

    font = load_font(FONT_SIZE)
    # 用 bbox 而非 font size 量文字高度：字体自带行距会让垂直居中偏下。
    bbox = font.getbbox(SLOGAN)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    top = (CANVAS_H - (logo_h + GAP + text_h)) // 2

    canvas = Image.new('RGB', (CANVAS_W, CANVAS_H), BG)
    # 第三个参数是 mask：品牌图有透明通道，必须走 alpha 合成，否则会糊成黑块。
    canvas.paste(logo, ((CANVAS_W - LOGO_W) // 2, top), logo)

    draw = ImageDraw.Draw(canvas)
    # 减去 bbox 左上角偏移，让文字按视觉外框居中，而不是按字体基线框。
    draw.text(
        ((CANVAS_W - text_w) // 2 - bbox[0], top + logo_h + GAP - bbox[1]),
        SLOGAN,
        font=font,
        fill=TEXT_COLOR,
    )

    canvas.save(OUT, optimize=True)
    print(f'wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB, {canvas.size[0]}x{canvas.size[1]})')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 运行脚本生成配图**

```bash
cd frontend/weapp && python scripts/make-share-card.py
```

预期输出类似：`wrote E:\...\src\assets\share-card.png (XX.X KB, 500x400)`

- [ ] **Step 3: 校验产物尺寸**

```bash
cd frontend/weapp && python -c "from PIL import Image; im = Image.open('src/assets/share-card.png'); print(im.size, im.mode); assert im.size == (500, 400), im.size; print('OK')"
```

预期：`(500, 400) RGB` 然后 `OK`

- [ ] **Step 4: 人工过目配图**

打开 `frontend/weapp/src/assets/share-card.png` 肉眼确认：品牌图居中、标语在其下方、四周留白均匀、无裁切、透明底已正确合成为 `#f6f8fa`。

**若视觉不达标，调 `LOGO_W` / `GAP` / `FONT_SIZE` 后重跑 Step 2，不要带着不合格的图往下走。**

- [ ] **Step 5: 提交**

```bash
git add frontend/weapp/scripts/make-share-card.py frontend/weapp/src/assets/share-card.png
git commit -m "feat(weapp): 分享卡配图生成脚本与产物 (500x400 品牌卡)"
```

---

## Task 3: 接线到对话页

**Files:**
- Modify: `frontend/weapp/src/pages/chat/index.tsx:261-277`（分享区块）与文件顶部 import 区

**Interfaces:**
- Consumes: `SHARE_PATH`、`buildShareCard`（Task 1）；`share-card.png`（Task 2）
- Produces: 无（终端改动）

- [ ] **Step 1: 加两行 import**

在 `frontend/weapp/src/pages/chat/index.tsx` 顶部 import 区追加（与既有 `import BrandBlock from '../../components/BrandBlock'` 同一区域）：

```tsx
import shareCard from '../../assets/share-card.png'
import { SHARE_PATH, buildShareCard } from '../../utils/share'
```

- [ ] **Step 2: 替换分享区块**

把 `frontend/weapp/src/pages/chat/index.tsx:261-271` 的注释与两个 hook 整体替换为：

```tsx
  // 分享：只做转发卡片，path 指向首页，标题与配图固定为品牌。
  // 不做分享特定会话——会话有归属校验，转发出去对方只会看到「会话不存在」；
  // 也不能把 sessionTitle 写进标题——那是用户首问原文，会随卡片进群聊/朋友圈。
  // 详见 docs/superpowers/specs/2026-09-15-weapp-landing-share-design.md
  useShareAppMessage(() => ({ ...buildShareCard(shareCard), path: SHARE_PATH }))

  useShareTimeline(() => ({ ...buildShareCard(shareCard), query: '' }))
```

**保持紧随其后的 `Taro.showShareMenu` effect（原 :273-277）原样不动。**

- [ ] **Step 3: 确认 sessionTitle 已脱离分享链路**

```bash
cd frontend/weapp && grep -n "sessionTitle" src/pages/chat/index.tsx
```

预期：**不应再出现**在 `useShareAppMessage` / `useShareTimeline` 内。`sessionTitle` 仍会出现在 :771 附近的标题栏渲染处（`title={sessionTitle || '新对话'}`）——**这是对的，不要动它**，该状态另有用途。

- [ ] **Step 4: 跑全量测试**

```bash
cd frontend/weapp && npm test
```

预期：PASS，`# pass 32`（原有 28 条 + Task 1 新增 4 条）/ `# fail 0`

- [ ] **Step 5: 跑构建**

```bash
cd frontend/weapp && npm run build:weapp
```

预期：构建成功，且内置的 `check:syntax` / `check:appid` 两道检查均通过。

- [ ] **Step 6: 确认配图被打包进 dist**

```bash
cd frontend/weapp && ls -la dist/assets/share-card.png
```

预期：文件存在。**若不存在，说明 import 写法有问题，配图不会被发到小程序包内**——必须回到 Step 1 排查，不要继续。

- [ ] **Step 7: 提交**

```bash
git add frontend/weapp/src/pages/chat/index.tsx
git commit -m "feat(weapp): 分享卡片收敛为纯品牌内容 —— 标题不再取会话标题, 补品牌配图"
```

---

## 交付后的验收（不在代码任务内，须真机执行）

代码任务只保证「载荷正确」和「配图进包」。**以下必须在真机完成**，因为微信开发者工具的分享卡片预览与真机表现不一致，且 `imageUrl` 加载失败时微信会静默回退到当前页面截图（泄露问题原样复活）：

1. 在小程序里进行一段**有内容的对话**
2. 点右上角「…」→ 转发给好友
3. 确认卡片**配图是品牌图、标题是品牌文案**
4. 若配图回退了（显示对话截图），说明 `imageUrl` 的引用方式在真机上不被接受，需另换引用方式并重走一轮

微信开发者工具里看不到这条路径的真实表现，**只有真机算数**。
