# 检索结果浏览器本地持久化设计（localStorage）

**日期:** 2026-08-14
**分支:** feature/search-results-split-view
**前置:** 分屏视图 + 内嵌 PDF 阅读器 + 空页兜底修复（e48891e）

---

## 一、背景

用户检索后的结果集目前只存在于前端内存（React Context），刷新/关标签页即丢失；`?set=` 参数指向的是内存状态而非持久化实体。经讨论（对照 Eureka 的工作区模式、评估 MySQL/Redis 方案、考虑数据量与用户行为），采用**浏览器 localStorage 持久化**：

- 同设备同浏览器刷新/隔天/关标签页重开 → 结果可恢复
- 零服务器成本、零运维；数据是用户自己的检索结果，无隐私外溢
- 不做后端持久化（跨设备回访、分享留待未来真实需求出现）

## 二、目标

1. 流式检索结束后，裁剪版结果集写入 localStorage（键 `copiioai_results`）
2. 结果页刷新（URL 带 `?set=X`）→ 从 localStorage 恢复该结果集渲染
3. 聊天页刷新（会话水合后）→ 按"用户消息文本匹配"恢复对应结果卡片
4. 上限 100 个结果集，超出丢最旧；存储不可用（无痕/被禁/配额满）时静默降级回内存模式，绝不报错

## 三、反目标

- 不做后端持久化（MySQL conversations.messages 不加 results；不建 result_sets 表）
- 不存 CSV/XLSX 完整 artifact（体积大，继续走 SSE 直发）
- 不存任何凭据/ token
- 不改 SSE 通道、不改后端任何代码

## 四、设计

### 4.1 存储结构（localStorage，单键）

```jsonc
// copiioai_results
{
  "sets": {
    "<setId>": {            // 即 json artifact_id，如 96f0b14e…-json
      "source": "uspto",
      "columns": [/* 裁剪后 */],
      "rows": [/* 裁剪后 */]
    }
  },
  "index": [                 // setId → 会话消息的关联线索，用于聊天页恢复
    { "setId": "<setId>", "sessionId": "…|null", "queryText": "用户原始提问", "savedAt": 1710000000000 },
    … 最多 200 条
  ]
}
```

- 裁剪复用现有 `pruneResultsForPersistence`（50 行上限、摘要 500 字、丢 role=text 列）→ 单集 20-40KB
- 容量核算：100 集 × 40KB ≈ 4MB，贴近 localStorage 5MB 上限 → 写入前检查并丢最旧直到放下（见 4.3）

### 4.2 模块与函数（新 `lib/resultsStore.js`，纯函数 + 注入 storage，node 可测）

```js
loadResultsStore(storage?)        // 读取+JSON.parse；异常→空 store
persistResultsSet(store, results, meta) // 不可变更新：写 sets[sid] + index 头部追加
                                       // 超 100 集丢最旧；返回新 store（不落盘）
saveResultsStore(storage, store)  // 落盘；QuotaExceeded → 逐次丢最旧重试；SecurityError → 静默
restoreResultsInMessages(messages, store) // 水合消息恢复：index 中 queryText 命中 user 消息
                                          // → 紧随其后的 assistant 消息挂 results（未命中不动）
buildStoredMessage(setId, store)  // 构造 {id:'stored-'+setId, role:'assistant', results:…} 合成消息
```

`meta = { sessionId, queryText, savedAt }`。同 queryText 重复检索时 index 出现多条，恢复时取 savedAt 最新。

### 4.3 写入时机与配额策略

- 写入点：`useChatStream` 的 `artifact_end` 同步解码处——把 `decodeArtifactChunksToResults` 的完整返回值（现在只取了 `.setId`）保留为局部变量，解码成功后立即 `persistResultsSet + saveResultsStore`（40KB JSON.stringify 同步写，耗时 <5ms，可接受）
- 配额满：`saveResultsStore` 从最旧 set 开始逐个丢弃重试，直到写成功或 store 空（空则放弃写入）
- localStorage 不可用：try/catch 全部静默 → 行为等同现状（内存模式）

### 4.4 恢复点（读）

1. **结果页**（`results/page.tsx`）：`resolveActiveResultsMessage` 增加第三层查找——内存精确匹配 → **localStorage 按 setId 命中 → 合成消息** → 内存最新兜底
2. **聊天页水合**（`chat/page.tsx` getSession 成功后）：`setMessages(restoreResultsInMessages(loaded, store))`
3. **结果页水合**（`results/page.tsx` 同款）：同样应用 `restoreResultsInMessages`

### 4.5 Next.js 约束（SSR 水合安全）

- **store 读取必须两阶段**：`const [store, setStore] = useState(null)` + `useEffect(() => setStore(loadResultsStore()), [])`——首渲染（服务端静态页/客户端水合）都用 `null`（不参与解析），effect 后才读 localStorage 触发第二阶段渲染。**禁止** `useState(() => loadResultsStore())` 惰性初始化（服务端与客户端首渲染不一致 → 水合错误）
- `resultsStore.js` 内对 `typeof window === 'undefined'` 守卫，SSR/静态构建返回空 store
- 静态导出（output: export）场景下水合前不触 storage

## 五、错误处理

| 场景 | 行为 |
|---|---|
| 无痕/禁用站点数据 | load/save 抛异常 → 空 store / 丢弃写入，静默降级内存模式 |
| 配额满 | 丢最旧重试；全丢光则放弃写入 |
| JSON 损坏（手改/半写） | parse 失败 → 空 store |
| set 不在 store（超 100 被挤掉/清数据） | 结果页走内存兜底或空态（现状行为） |
| Safari ITP 7 天未访问清除 | 数据消失，重跑检索即可（可接受） |

## 六、测试

`lib/resultsStore.test.mjs`（node，注入 fake storage）：
- persist：新集写入、超 100 丢最旧、index 头部追加
- save：配额满丢最旧重试、storage 抛异常静默
- restore：queryText 命中 → 紧随的 assistant 消息挂 results；未命中不动；同 queryText 多条取 savedAt 最新
- buildStoredMessage 结构正确
- prune 集成：写入的 rows ≤50、text 列被丢

`lib/results.test.mjs`：`resolveActiveResultsMessage` 扩展第三层（store 命中 → 合成消息）

## 七、影响面

| 文件 | 改动 |
|---|---|
| `frontend/nextjs/lib/resultsStore.js` | 新增（~100 行） |
| `frontend/nextjs/lib/resultsStore.test.mjs` | 新增 |
| `frontend/nextjs/lib/results.js` | `resolveActiveResultsMessage` 增 store 参数（~15 行） |
| `frontend/nextjs/lib/results.test.mjs` | 增 2-3 用例 |
| `frontend/nextjs/lib/useChatStream.ts` | 保留完整解码结果 + 写入（~15 行） |
| `frontend/nextjs/app/app/(auth)/results/page.tsx` | store 读取 + 三层解析 + 水合恢复（~15 行） |
| `frontend/nextjs/app/app/(auth)/chat/page.tsx` | 水合恢复（~5 行） |

后端零改动。前端需 `npm run build` 部署。
