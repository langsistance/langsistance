# 小程序分享收敛为纯品牌卡片

- 日期：2026-09-15（brainstorm 定稿）
- 状态：设计已确认，待实现
- 分支：`feat/weapp`
- 修订：**推翻** `2026-09-11-weapp-m3-design.md:228` 的「卡片标题取当前提问/会话标题」
- 依据：`frontend/weapp/src/pages/chat/index.tsx:261-277`（现状实现）

---

## 1. 范围

| # | 事项 | 本轮 | 说明 |
|---|---|---|---|
| 1 | 分享卡标题固定为品牌文案 | ✅ | 不再取 `sessionTitle`，见 §2 |
| 2 | 分享卡配图固定为品牌图 | ✅ | 新增 `imageUrl`，见 §4 |
| 3 | 分享载荷逻辑抽为可测模块 | ✅ | 附回归测试，见 §3、§5 |
| — | 分享特定会话 / 报告给外部人 | ❌ | 与 M3 §5 同结论：会话有归属校验，转发出去对方只会看到「会话不存在」 |
| — | 结果页（`pages/results`）分享 | ❌ | 本轮不做 |
| — | 分享埋点 / 回调统计 | ❌ | 本轮不做 |
| — | 中途隐藏分享入口 | ❌ | 分享入口任何时候保留，只收敛卡片内容（见 §3.3） |

---

## 2. 为什么修订 M3 §5

M3 spec 原文（`2026-09-11-weapp-m3-design.md:228`）：

> `Taro.useShareAppMessage` → 转发给好友，**卡片标题取当前提问/会话标题**，`path` 指向首页

这条已按原样实现（`chat/index.tsx:263-266, 268-271`）。但落地后暴露出两个泄露口子：

1. **标题带用户提问原文**。`sessionTitle` 就是首条消息前 60 字符（`chat/index.tsx:548`，`MAX_TITLE_LEN` 定义于 `components/RenameModal/index.tsx:14`）。用户首问若含专利号、公司名或具体诉求，这段文字会直接出现在群聊/朋友圈的卡片上。
2. **配图未设 `imageUrl`**。微信在 `imageUrl` 缺省时取**当前页面截图**作缩略图。从一个进行中的对话分享出去，卡片配图就是对话内容本身。

两条叠加，分享这个本该是增长渠道的动作变成了内容外泄通道，且**用户无感知**——他点的是「转发」，不会预期标题和配图带了什么。

M3 当时的判断（"仅转发卡片对提审够用"）依然成立，本轮不改结论，只改**卡片内容**：入口保留，内容恒定。

---

## 3. 前端设计（`frontend/weapp`）

### 3.1 文件清单

| 动作 | 文件 | 说明 |
|---|---|---|
| 增 | `src/assets/share-card.png` | 分享卡配图，500×400，见 §4 |
| 增 | `scripts/make-share-card.py` | Pillow 生成脚本，参数写死，可重跑 |
| 增 | `src/utils/share.ts` | 分享载荷纯逻辑，**不 import 任何资源** |
| 增 | `src/utils/share.test.mjs` | 回归测试，见 §5 |
| 改 | `src/pages/chat/index.tsx` | 接线；删掉 `sessionTitle` 参与分享 |
| — | `tsconfig.test.json` | **不改**，见 §3.4 |

### 3.2 接口

```ts
// src/utils/share.ts
export const SHARE_TITLE = 'CopiioAI 专利情报，一问即得'
export const SHARE_PATH = '/pages/chat/index'

/** 品牌卡片载荷：标题与配图固定，不携带任何会话内容。 */
export function buildShareCard(imagePath: string) {
  return { title: SHARE_TITLE, imageUrl: imagePath }
}
```

**`buildShareCard` 为何不含 `path`**：两个 hook 的返回结构本就不同——

- `useShareAppMessage` → `{ title, path, imageUrl }`
- `useShareTimeline` → `{ title, query, imageUrl }`（path 隐式为当前页，传了无效）

故共用的核心只承载两条不变量（标题固定、配图固定），页面各自补自己那一个字段：

```tsx
import shareCard from '../../assets/share-card.png'
import { SHARE_PATH, buildShareCard } from '../../utils/share'

useShareAppMessage(() => ({ ...buildShareCard(shareCard), path: SHARE_PATH }))
useShareTimeline(() => ({ ...buildShareCard(shareCard), query: '' }))
```

`Taro.showShareMenu` 保持现状不变（`chat/index.tsx:273-277`），分享入口任何时候都在。

### 3.3 `share.ts` 为何不 import 资源

两条约束把它逼到了这个位置：

- **资源必须被 `import` 才会拷进 `dist/`**。证据：`dist/assets/brand-logo.png` 即 `components/BrandBlock/index.tsx` import 的产物。纯字符串路径不会被 bundler 识别，图不会进包。
- **但 `node --test` 加载不了 PNG**。注意障碍不在类型层——`types/assets.d.ts:11` 已声明 `declare module '*.png'`，import 图片类型合法。卡在运行期：`tsconfig.test.json` 只编译不拷资源，编译产物里的 `require('../assets/share-card.png')` 会指向不存在的文件（`dist-test/assets/` 不会被生成），`node --test` 直接 MODULE_NOT_FOUND。**类型过得去，运行过不去。**

故边界划为：`share.ts` 保持零资源依赖、纯函数、可被 node 直接加载；图片 import 留在页面（页面本就要打包）。配图路径由页面作为参数传入。

### 3.4 测试构建为何不用改 `tsconfig.test.json`

该文件的 `include` 白名单里已有两条通配：

```
"src/utils/**/*.ts",
"src/utils/**/*.mjs",
```

`share.ts` 与 `share.test.mjs` 落在 `src/utils/` 下会被自动纳入。这也是把逻辑放 `utils/` 而非 `pages/` 的硬理由——`pages/` 不在白名单内。测试沿用既有相对 import 写法（`import { ... } from './share.js'`，参见 `src/utils/results.test.mjs:3-10`）。

---

## 4. 配图生成（`scripts/make-share-card.py`）

| 项 | 值 | 依据 |
|---|---|---|
| 画布 | 500×400 | 微信标准 5:4 |
| 底色 | `#f6f8fa` | 小程序 `--c-bg`（`src/app.scss:9`） |
| 品牌图 | `src/assets/brand-logo.png` 缩放居中 | 复用现有素材，不重做 logo |
| 标语 | `专利情报，一问即得` | 与 `components/BrandBlock/index.tsx` 同一句 |
| 字体 | `msyhbd.ttc`（微软雅黑 Bold），色 `#1f2328` | 对齐 `BrandBlock/index.scss` 的 600 字重 + `--c-text`（`src/app.scss:12`） |

- 参数写死在脚本内，**脚本与产物一并提交**，改文案/配色时重跑即可
- 边缘留足留白：微信可能对卡片做圆角裁切
- 生成后先人工过目确认视觉，再提交 PNG（不直接落库）
- 放 `scripts/` 沿用该目录既有检查脚本的位置约定，语言为 Python（Pillow 12.2 已在环境内）

---

## 5. 测试

`src/utils/share.test.mjs`，三条断言钉住「分享载荷不携带会话内容」：

```js
test('载荷字段固定：只有 title 与 imageUrl', () => {
  // 修复前的 bug：title 拼进 sessionTitle，把用户首问原文带到了群聊卡片上。
  // 这条钉住"不夹带字段"——将来若有人把会话数据塞进载荷，key 集合会变，测试红。
  assert.deepEqual(Object.keys(buildShareCard('x')).sort(), ['imageUrl', 'title'])
})

test('标题恒为品牌文案，与会话无关', () => {
  assert.equal(buildShareCard('x').title, SHARE_TITLE)
  assert.equal(SHARE_TITLE, 'CopiioAI 专利情报，一问即得')
})

test('path 指向落地页', () => {
  assert.equal(SHARE_PATH, '/pages/chat/index')
})
```

第二条同时断言 `SHARE_TITLE` **等于字面量**——否则若有人改了常量本身，前一条会跟着一起变绿，等于没测。

---

## 6. 风险

### 6.1 配图加载失败会回退到页面截图（无法用代码消除）

**若 `imageUrl` 在真机上加载失败，微信回退到当前页面截图，泄露问题原样复活。** 这是平台行为，代码侧无法兜底。

且微信开发者工具的分享卡片预览与真机表现**并不一致**，工具里看着对不代表真机对。因此 §7 验收动作必须在真机上、且在**有对话内容的页面**执行。

若真机确认回退，说明 `imageUrl` 的引用写法有问题，需换引用方式（届时另议）。

### 6.2 文案变更需重新发版

标语是**烤进 PNG 的像素**，改文案 = 重新生成图片 = 重新发版。这是选择"配图带标语"方案时已接受的取舍。

### 6.3 字体差异（不影响功能）

图内标语用微软雅黑渲染，小程序内 slogan 走真机系统字体（iOS 苹方 / Android 思源），会有细微字形差异。因卡片图是静态图，不会错乱。

### 6.4 用户预期偏差（产品选择）

在有对话的页面点转发，卡片显示品牌内容而非当前对话。用户可能预期"转发这个对话"。这是本轮明确的产品取舍，非缺陷。

---

## 7. 验收标准

1. `npm test`（`frontend/weapp`）通过 —— 含新增 3 条断言
2. `npm run build:weapp` 通过 —— 含既有 `check:syntax` / `check:appid`
3. 构建产物中存在 `dist/assets/share-card.png`
4. **真机验证**：在一个**有对话内容**的页面点转发，确认卡片**配图为品牌图、标题为品牌文案**，而非对话截图或会话标题
5. 确认 `chat/index.tsx` 中 `sessionTitle` 不再参与分享链路
