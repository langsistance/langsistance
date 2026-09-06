# 可行性调研：US 外观（design）专利检索 + 侵权风险判断（B 方案自研版）

> 场景：卖家选品防侵权（输入产品图/链接/品名 → US 外观在先权利检索 → 视觉相似风险判断）。
> B 方案立场：**不建图像向量库/不依赖本地 GPU**——召回靠文本/分类 +（可选）图像反查桥，判定靠多模态 LLM。
> 范围：**首版 US-only**（CN/欧盟数据通道现状见 §3.4）。
> 证据标记：[本机实证] = 本机已验证；[服务器实证] = spike/生产行为已验证（2026-09-06）；[待验 Vn] = 需服务器实测项。

---

## 1. 调研结论速览

**可行，且比预期更顺的三点**：
1. **视觉判定管道已存在**：`sources/long_task/patent_analyzer.py:344 analyze_patent_with_vision` + `_call_vision_api`，生产配置 `config.ini [LONG_TASK] vision_provider=minimax / vision_model=MiniMax-M3`（`config.py:19-20/40-54`），PDF 页图→vision 流程在 scanned 专利分析中已实战——L0（图→文本）与 L3（图×图对比）**复用现配置与调用封装，无需新 provider 基建**。
2. **Google 附图域本机/服务器双可达**：`patentimages.storage.googleapis.com` 本机 [本机实证] 403（存活，缺正确路径）；生产服务器访问 patents.google.com 已多次实证（families CN 腿 + PCT→WO spike）。附图获取链路 = 服务器抓专利页解析 patentimages 图 URL。
3. **US 外观无年费/维护体系**：授权后 15 年（2015-05-13 后申请）或 14 年有效，**法律状态可由授权日推导**——比 utility 的维护费状态查询简单一个量级，无额外状态 API 依赖。

**三处必须服务器实测才能定稿**（V1/V2/V3，§5）——USPTO search 对 D 号段的字段支持、Google Patents 设计页是否带 Locarno、老设计图覆盖。

---

## 2. 接口调研（L1 检索 / L2 取图 / L3 判定）

### 2.1 候选数据接口全景

| 源 | 能力 | 证据/状态 | 角色 |
|---|---|---|---|
| **R1 USPTO `applications/search`**（现用，`resolve_us_pub_number` 同款） | 著录检索：`patentNumber / earliestPublicationNumber / applicationNumberText / inventionTitle / applicationMetaData.*`；X-API-Key | 代码在用 [本机实证-代码]；**对 D 号段（design）的收录与字段支持待验 V1**（PEDS 覆盖外观申请，但 patentNumber 查询形态：带 `D` 前缀串 vs 纯数字——现有 resolve 用纯数字，设计号可能与 utility 同 digits 冲突，需专用形态） | 备选/官方源 |
| **R2 Google Patents XHR 检索 + 专利页** | 检索（PCT spike 已证 XHR `q=` 唯一命中 JSON，含 bibliographic/family）；专利页 HTML 含 patentimages 图 src 与著录/分类 | spike [服务器实证]（2026-09-06，PCTUS2021059064→WO2023075806A1）；对 **design 检索字段/结果含 USPC-D 与 Locarno 与否待验 V2** | **推荐主检索源** |
| **R3 USPTO PEDS documents API（D 授权 PDF）** | 官方授权 PDF → 页图（现 text_extractor 已有 PDF→页图→vision 先例） | 通道存在 [代码实证]；PDF 页图切分成本高于 R2 直取图 | 回退（R2 图缺失时） |

### 2.2 推荐链路（每环有回退）

```
[输入] 产品图/链接/品名
  L0 视觉翻译   analyze_patent_with_vision 管道(MiniMax-M3) 一次调用
       → {en 品名, 视觉特征词, USPC-D 候选类, Locarno 候选类}
  L1 文本召回   R2 Google XHR q=「品名词 OR 特征词」(服务器) 或 R1 USPTO search(待 V1)
       → 阶梯放宽(同 utility 基建); 附带授权日/权利人/分类
       → 过滤: 授权日 + 15y < 今天 → 失效剔除/标注
       → Top-N(默认 30; 可配)
  L2 附图获取   对每件: patents.google.com/patent/USDxxxxS 页 HTML → 解析
       patentimages.storage.googleapis.com 的 <img> src → 并发抓主视图+局部(≤8 张/件)
       → 抓取失败件降级"文本维度候选"; 页不可达回退 R3 PEDS PDF→页图
  L3 视觉判定   L0 产品图 × L2 件图 分组(1~4 件/批)送 vision, 结构化 JSON 打分
       → 阈值 0.7 或 risk=high 高亮 → 报告(风险档/设计点命中/依据/差异点)
```

### 2.3 判定口径（进 L3 prompt，随报告展示）
- US：*ordinary observer* 整体观感 + *point of novelty*；CN（二期）：一般消费者整体观察、区别设计空间
- 色彩：US 外观对比不以色彩为独立维度（若以图对比亦如此声明）；输出必须"分数+图部位依据句"，可追溯
- 输出风险仅三档 + 置信提示 + "非法律意见，建议律师复核"（行业红线）

### 2.4 产品图输入通道
现 `_handle_file_upload_query`（core.py）已接收图像文件（产品图/设计稿），新增 design 意图场景即可直连 L0——**无新上传基建**。

---

## 3. 数据调研

### 3.1 规模
US design 授权累计约 **100 万件级**（D 号段已过 D1,0xx,xxx；近年 ~5 万/年）[公开事实]。
- 文本语料极小 → 无索引需求、USPTO/Google 检索即可
- 图语料每件 1~7 视图；只需"随询抓取 Top-N"，**全量图库不存在于本地**（省 <1GB 内存约束）
- 待验 V4：R2/R1 实际可检出的 design 子集规模（检索层覆盖度）

### 3.2 字段（US 外观著录）
`D 号 / 标题（品名词）/ 授权日 / 发明人 / 权利人 / 分类`。**无 abstract/claims**（utility 阶梯基建的正文检索不适用；标题+分类即全部文本信号）。
- 分类体系：USPC **D 类**为主；Locarno 是否随 US 外观数据可得待验 **V2**（Google Patents 页面若展示 LOC，则 L1 可加 LOC 维度；否则以 USPC-D 为准并在 UI 说明）

### 3.3 法律状态
授权起 **15 年**（2015-05-13 后申请）/ 14 年（此前），无维护费、无续展 → `grant_date + term >= today` 即有效；超期标注"已期满"。无状态 API 调用（utility 侧法律状态查询无需接入）。

### 3.4 CN/欧盟外观数据现状（明确边界）
- 佰腾两通道（open.zldsj.com 网关 + open.baiten.cn 开放平台全目录）**均无外观库/LOC/以图搜图接口**（[本机实证-2026-09-06 目录核对]：只有发明/实用检索、著录/法律/同族/相似(文本)/附图(说明书)等）
- CNIPA 外观检索系统为网页端、无公开 API
- → **首版 US-only**；遇 CN/欧盟询检给出能力边界说明与替代（站外/后续第三方 A 方案）；勿静默跨区

### 3.5 附图可用性风险
- 老设计（1970s 前扫描件）在 Google Patents 的图覆盖与质量待验 **V5**（抽样 3 个年代段：>2000 / 1980-2000 / <1980）
- 抓图预算：Top-30 × ≤8 图 ≈ ≤240 请求上限、单张典型 <200KB、并发 6 路 → 峰值内存 <150MB（[服务器内存约束] 合规），随用随弃不落库

---

## 4. 成本与延迟量级（估算，API 现价口径以运营确认为准）

| 环节 | 调用量 | 延迟 |
|---|---|---|
| L0 视觉翻译 | 1 × vision(MiniMax-M3) | ~1–3s |
| L1 文本召回 | 3–8 × search（0 命中阶梯放宽同 utility） | ~3–10s |
| L2 抓图 Top-30 | 30 页 + ≤240 图（并发） | ~10–20s |
| L3 视觉判定 | 30 件 ÷ 4 件/批 ≈ 8 批 vision | ~30–60s |
| **合计** | 9–17 外部调用 | **~1–2 min**（与现 utility 上传查重长任务同量级，走既有 long_task 管道与进度/锚点/回执） |

成本护栏：判定 Top 数可配（默认 30 → 用户可"仅高/中风险再展开"）；vision 调用量约 **9 次/询检**量级，月万次询检成本可控（具体价目需运营按 MiniMax-M3 计费核实）。

---

## 5. 服务器实测项清单（V1–V5，一次性脚本 ~30 min）

| # | 实测内容 | 判定影响 |
|---|---|---|
| V1 | USPTO search：`patentNumber:"DxxxxxS"`（带前缀）与纯 digits 对设计号的命中差异；`inventionTitle` 是否可检 design；D 号与 utility 同 digits 是否串扰 | 决定 L1 主源是否用 R1 |
| V2 | Google Patents：设计页/XHR 结果是否含 **Locarno** 与 USPC-D；design XHR 检索 `q=` 语义（品名词是否有效召回） | 决定 L1 字段体系（USPC-D / LOC / 品名词）与主源选 R2 |
| V3 | patentimages 直链：抓一个真实设计页取 `<img>` src → 直链 200 验证 URL 模式（本机 storage 域可达但需真实 hash 路径） | 敲定 L2 取图实现 |
| V4 | 检索覆盖度抽样：5 个真实产品品名（玩具/电子/家具/瓶器/灯饰各 1）各取 Top 数 | 校准召回与 Top-N 默认值 |
| V5 | 老设计图覆盖抽样（3 年代段各 5 件） | L2 回退策略与报告免责口径 |

脚本建议：独立 `scripts/design_feasibility_probe.py`（流式、低内存、只输出 JSON 结论表），在服务器用现 .env 凭据跑一次，产出回填本文。

---

## 5.5 V1–V5 实测回填（2026-09-06 服务器两轮探针，`scripts/design_feasibility_probe.py` v2）

### V1 · USPTO search 对 D 号段 —— **定案：可用，形态=D 前缀查询；无分类字段**

| 探针 | 结果 | 结论 |
|---|---|---|
| `patentNumber:"D504889"`（带 D） | **200 命中 1 件**（patentNumber=D504889，title=ELECTRONIC DEVICE） | D 号**必须带前缀**查询，形态 `D`+数字无逗号 |
| `patentNumber:"504889"`（纯 digits） | **404 = 0 命中** | 纯数字查不到 design（与 utility 段 digits 冲突问题不存在） |
| `inventionTitle:"toy"` | 200 命中（限制 3） | title 文本可检 |

**关键局限**：返回 `applicationMetaData` 仅含 `inventionTitle/patentNumber`，**无任何分类字段**（meta_classification_present=false）→ USPTO search 通道拿不到 USPC-D/Locarno，**只支持 title/品名词检索**。

### V2 · Google Patents 设计页与 XHR —— **定案：type=DESIGN 过滤可用；设计页主资源是 PDF**

- XHR 设计号查询：`USD504889S` → 唯一命中 id=`patent/USD504889S1/en`（**S1 后缀**），pub=`USD504889S1`，title=Electronic device
- 设计页（`USD504889S1/en`）200；**页面无 Locarno/USPC 文本**（has_locarno/uspc=false）
- **过滤试探（决定性）**：`toy snake` 纯词 6286 条 → `type:DESIGN` token 3923 → **`&type=DESIGN` 独立参数 270 条** → XHR **支持 type=DESIGN 限定设计库**（数量级收窄 ~23×）
- V4 佐证：不过滤时 Top10 几乎无设计件（robot toy/bottle=0、office chair 8 全 EM、desk lamp 6 全 EM）→ **L1 必须带 type=DESIGN**

### V3 · 附图直链 —— **定案：PDF 直链 200，架构级利好**

- 设计页首资源 = `https://patentimages.storage.googleapis.com/…/USD504889.pdf` → **直链 200、203KB、application/pdf**
- 含义：设计附图以**整本 PDF** 提供（多视图在 PDF 页内）→ L2 取图 = 拿 PDF 直链 → **复用既有 `text_extractor` PDF→页图→vision 扫描件管道**，无需新图像管线，单件 <250KB 低内存合规

### V4 · 品名词 design 召回 —— **定案：需过滤后测；EM/CN 设计在 XHR 可见（多区扩展通道）**

- 不过滤时 top10 设计占比 ≈0（EM 例外 6-8 件）；US 设计在 5 词 × top10 仅 1 件 → **纯词排序对设计无效，type=DESIGN 是硬前提**；US-only 是否另有 `country=` 过滤参数**未验（V4′，入 spec UAT）**

### V5 · 老设计图覆盖 —— **降级为低优先（term 推导自动过滤老件）**

- 年代过滤可用但需叠加 type=DESIGN（首轮未叠加导致 top10 全 utility，无设计件样本）
- **风险场景只关心授权 ≤15 年件**（法律状态=日期推导）→ 有效区间 2010+，其图通道与 2005 样本同源已证 → V5 老件覆盖不再阻塞

### 遗留微探针（可并入 spec UAT，不阻塞设计）
V4′：`type=DESIGN` + `country=US`（参数名待验）下 5 品名词的 US 设计召回占比 → 决定 L1 是否需要 US 限定与 Top-N 默认值。

### V6 · Locarno 对 US 设计可用性 —— **裁定：不可依赖（数据源结构性缺失）**

第三轮探针 Google 503 限流（瞬时反爬，非数据结论），LOC 直接证据未取得；结合已有证据与领域事实收敛裁定：

1. USPTO 本土申请以 **USPC-D 类** 存储（V1：API 无分类字段返回）；仅 **Hague 指定美国**子集带 Locarno → LOC 对 US 本土件**结构性缺失**；
2. Google 设计页（200 时）无 "Locarno" 字样 → GP 侧亦不提供；
3. **设计约束（进 spec）**：US 检索不依赖 LOC/USPC——L1 = `type=DESIGN` + 品名词阶梯；分类字段不作检索/过滤维度；
4. LOC 保留两个正确用途：① L0 视觉模型**推断 LOC** → 品类自检展示（用户可纠偏）+ 报告标注；② 未来 EM/CN 外观询检（XHR 已见 EM/CN S 号）时 LOC 推断直接作过滤（彼处 LOC 为存储分类）；
5. USPC-D 若日后从 GP 页面可解析，仅作报告"分类标注"增强，不进主检索（USPTO 无分类 API 字段，D 子类枚举路线关闭）。

---

## 6. 开放决策点（实测后定）

1. **L1 主源**：**R2 Google XHR 为主**（type=DESIGN 过滤已证；设计号/PDF 同源）——R1 USPTO search 降为 title 检索备选（无分类字段、无 PDF 直链便利）；V4′ 若 `country=US` 不可用则考虑 XHR 结果侧按 US 过滤或 USPTO title 检索补 US 命中。双源合并去重（成本 +~1 查询/件）留二期
2. **USPC-D 广度策略**：因 USPTO 无分类字段、Google 页无 USPC 文本 → **放弃分类穷举路线**，召回 = 品名词/特征词阶梯 + type=DESIGN（0 命中阶梯放宽与 utility 同构）
3. **L3 批量**：1 件/批（准）vs 2–4 件/批（省）——UAT 定
4. **报告形态**：沿用结果卡+artifact 下载（对比图并排需前端小改）还是先纯文本+PDF 链接——UAT 定
5. 是否二期立项：图像反查桥（L1-b）、CN/欧盟路由（XHR 已见 EM/CN S 号，未来可扩区）、`country=` 参数若无效时 US 收窄策略

---

## 7. 结论（实测后更新）

B 方案（US 外观、无向量库、LLM 判定）**技术上可行，且主链路已实证打通**：

- **L1 召回** = Google XHR `q=品名词阶梯 + type=DESIGN`（已证可收窄至设计库）；USPTO search 仅作 title 备选（D 前缀形态已证、无分类字段）
- **L2 取图** = 设计 PDF 直链（200 实证，203KB/件）→ 复用既有 PDF→页图→vision 管道（**架构级利好：无新图像管线**）
- **L3 判定** = 复用 `analyze_patent_with_vision`（vision_provider/model 可配；服务器当前 minimax/MiniMax-M3，deepseek-v4-flash-vision-exp 属候选）
- 法律状态 = 授权日 + 15 年推导（无状态 API 依赖）；老件（>15 年）自动失效过滤，规避老图覆盖风险
- 唯一未验项 V4′（`country=US` 参数与 US 设计召回占比）降级为 **spec 内 UAT 项**，不再阻塞设计

在 V1–V5 全部定案的基础上，可直接进入正式 spec 设计（范围：US 外观防侵权询检，B 方案主链路，报告含视觉对比与 0.7 风险口径）。
