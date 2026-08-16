# 方案 B：CPC 语义扩展（伪相关反馈）实施计划

日期：2026-08-16
状态：待确认

## 1. 目标与原理

**目标**：修复关键词检索的召回天花板——"词面不重叠就召回不了"（典型：提问"独立控制 RGB 颜色输出"找不到 ERP Power 的"LED driver for integrated human-centric lighting"，因为标题不含 RGB/independent 等词）。

**原理**：不猜词，猜分类。

```
用户提问（中文）
    │ bge-m3 embedding（已有基建）
    ▼
提问向量  ←─余弦相似度─→  CPC 分类标题向量（本地预嵌入缓存）
    │
    ▼
Top-K CPC 码（如 H05B45/20 — LED 驱动电路）
    │
    ├─ 路线 A：检索式追加 CPC 字段限定 → 类内宽词检索（直接捞出 ERP Power 族）
    ├─ 路线 B：CPC 符号作为裸词 OR 进检索式
    └─ 路线 C：CPC 标题词注入缺失方向 prompt → LLM 用分类语言生成补充检索式
```

CPC（Cooperative Patent Classification）是数据驱动的官方分类体系，约 26 万条符号，每条带标题/定义。用"语义找类 + 类内检索"模拟智慧芽语义检索的效果，全程零领域词表硬编码——满足通用性约束。

## 2. 前置验证（Phase 0，0.5 天）——决定三条路线

`scripts/uspto_cpc_probe.py` 已存在（探测 USPTO applications/search 是否支持 CPC 字段限定查询），**需要在服务器上跑一次**：

```bash
USPTO_API_KEY=... python scripts/uspto_cpc_probe.py
```

| 探测结果 | 结论 | 集成路线 |
|---|---|---|
| probe 2/3 返回 200 且 count>0 | CPC 字段限定可用 | **路线 A**（最优） |
| probe 4 可用（2/3 不行） | CPC 符号可作裸词 | **路线 B** |
| 全部失败 | 无法用 CPC 约束检索 | **路线 C**（词汇扩展兜底） |
| probe 6 无 CPC 数据 | 候选 CPC 需另找数据源 | 改用 CPC 标题词扩展（路线 C） |

## 3. 数据管线（Phase 1，1 天）

**数据源**：CPC Scheme XML（EPO 官方月度发布，cooperativepatentclassification.org 公开下载，~50MB）。解析出 符号+标题 清单（仅保留有独立标题的条目，约 10-15 万条）。

**新模块 `sources/long_task/cpc_semantic.py`**：
- `load_cpc_titles() -> list[(code, title)]` —— 本地 JSON 加载（数据文件随代码库提交或部署时放置，不运行时下载）
- `ensure_cpc_vectors() -> np.ndarray` —— 标题批量嵌入（复用 `knowledge.get_embeddings_batch`），结果缓存到本地 `.npy`（float16，体积减半、排序精度损失可忽略）；首次运行构建，之后秒级加载
- `match_cpc_codes(query_embedding, top_k=8) -> list[(code, title, score)]` —— numpy 点积暴力搜索，毫秒级
- 防御性降级：数据文件缺失 / embedding 失败 → 返回空列表，功能自动关闭

**资源估算**：
| 项 | 量级 |
|---|---|
| 磁盘 | XML ~50MB + JSON ~15MB + 向量缓存 ~60-300MB（按精度档位） |
| 内存 | <500MB（可只加载主组标题档位则 ~60MB） |
| 计算 | 纯 CPU numpy；匹配 ~5-50ms/次；首次嵌入构建一次（约 15 万标题 × bge-m3，分批跑，一次性成本 ~数元） |
| 新增依赖 | numpy（若未在依赖中） |
| 新服务 | 无 |

**档位策略**（先粗后细）：
- v1：只嵌入"主组"标题（H05B45/00 级，约 1.4 万条，60MB）——粗定位领域
- v2（可选）：命中主组后按需嵌入其子组（数千条）细定位

## 4. 集成（Phase 2，1-1.5 天）

### 4.1 通用部分（与路线无关）

缺失方向推断增加 CPC 上下文：`_maybe_append_missing_directions` 拿到池子标题的同时，把 `match_cpc_codes(提问)` 的结果（码 + 分类标题）一并传给推断 LLM：

> 该技术主题对应的专利分类：H05B45/20 - Circuit arrangements for operating light-emitting diodes …

让 LLM 用"分类语言"生成补充检索式。**零 API 格式风险**——只改 prompt 输入。

### 4.2 按路线注入检索式（自动二轮执行）

- **路线 A**：自动二轮追加 CPC 字段限定检索式（字段语法按 probe 结果），如 `("LED" OR driver OR current) AND <cpc字段>:"H05B45/20"`
- **路线 B**：自动二轮追加 `(LED OR driver OR color) AND H05B45/20` 裸词式
- **路线 C**：CPC 标题词直接并入概念词库

### 4.3 开关与状态

- env `REACT_CPC_EXPANSION`（默认 0 关，灰度）
- 所有 per-turn 标志进 `create_agent` 重置块（沿用池复用教训）
- 每次匹配记日志（`cpc_semantic.log`：top codes + 耗时）

## 5. 测试（TDD，穿插在各 Phase）

- `test_cpc_semantic.py`：解析样例 XML/JSON、匹配排序正确、向量缓存读写、数据缺失降级、embedding 失败降级
- `test_react_tools.py`：CPC 上下文注入缺失方向调用、自动二轮 CPC 检索式生成、开关关闭跳过、失败回退
- 全量回归基线：26 个环境性预存失败不变

## 6. 验收标准（Phase 3，0.5 天）

线上同款提问「控制放大器，独立控制 RGB 颜色输出」：
1. `cpc_semantic.log` 显示匹配出 H05B45 族（LED 驱动电路）等正确分类
2. 自动二轮检索式带 CPC 限定且命中 >0
3. **结果池出现 ERP Power 族（US11882632B2 等）或同域专利**——这是本次改造的终极判据
4. 无回归：其他提问（非 LED 域）CPC 匹配同样正常（通用性）

## 7. 风险与回退

| 风险 | 应对 |
|---|---|
| probe 全失败（API 不支持 CPC 字段） | 路线 C：纯词汇扩展，不依赖 API 格式 |
| CPC 数据文件体积/首次嵌入成本超预期 | 主组档位 v1 先行（60MB/1.4 万条） |
| 分类匹配错误域（提问歧义） | 只影响补充检索式，主链路不变；阈值过滤低分码 |
| embedding 服务抖动 | 全部防御性降级，CPC 功能关闭不影响现有链路 |

## 8. 总改动量

| Phase | 内容 | 改动量 |
|---|---|---|
| 0 | 服务器跑 probe 定路线 | 0.5 天 |
| 1 | 数据管线 + 匹配模块 + 测试 | 1 天 |
| 2 | 集成（prompt 上下文 + 自动二轮检索式）+ 测试 | 1-1.5 天 |
| 3 | 线上验收 + 调参 | 0.5 天 |
| **合计** | | **3-3.5 天** |

文件改动预估：新模块 2 个（cpc_semantic.py + 数据文件），改动 2-3 个（search_query_builder / react_tools / general_agent 重置块），新测试 1 个文件。
