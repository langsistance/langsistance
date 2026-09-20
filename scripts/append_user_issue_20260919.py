# -*- coding: utf-8 -*-
"""08-19/09-19 批次日志分析追加器。

把第 21 位样本(11714190467530210252)的三表内容追加进
user_issue_analyze.xlsx。行样式从上一既有行复制,不改动任何旧内容。
"""
import copy
import os
import shutil

import openpyxl

BASE = r"E:\online\酷彼智能\copiioai\patent\用户增长"
MAIN = os.path.join(BASE, "user_issue_analyze.xlsx")
VIEW = os.path.join(BASE, "user_issue_analyze - 视图.xlsx")

# ── 用户问题分析 ────────────────────────────────────────────────────────────
Q_USER = "11714190467530210252"
Q_PROCESS = """【用户场景】新注册用户(QQ 邮箱 854173195@qq.com，auth:signup 2026-09-19 08:41:28 成功，Firebase uid B9xfMlWvVOdrgblTBl6b1XCb0UN2，内部本地 uid 11714190467530210252，第 21 位样本)。注册流程压缩到极短：08:41:28 signUp→08:41:29 signInWithPassword→Session 建立→08:41:30 选场景 scenes=[1] 并置 onboarded→08:41:30.003 发起首问——注册到首问**不到 1 秒**(同批最早的 9408… 为 3 秒，本样本 0.7 秒是已记录批次中最快之一)。更关键的是**首问形态**：不是"帮我找 XX 专利"这类试探性开口，而是携一份**四要素结构化的检索任务书**——「请检索：同时包含下列要素的专利文献：①微环谐振器；②冗余/备份微环；③检测微环失效并切换到备份环；④微环用于波长路由或光交换（而非调制器）」，并显式用括号排除调制器用途。这种"编号列举要素 + 排除项括号"的写法是 FTO/查新作业的典型输入格式，画像指向发明人本人或 IP 从业者（研发方向为硅光/光交换芯片），不是品类级买家式的模糊搜索。

【提问时间线与逻辑链】
1. 08:41:30 首问四要素检索，新建会话 sess_a0c17582d606；08:41:50 search_interpretation 语义解读把①微环谐振器映射为容错型硅光波长选择路由/交换架构（工作环+冗余环耦合至同一或可重构波导路径、监测端口/光功率监测器/谐振失锁检测判定主环状态、热调谐或电光调谐切换），players=Intel/Cisco/IBM/HPE/Oracle——域翻译准确；search_rewrite mode=semantic 生成 8 条英文阶梯 + 6 条中文阶梯（ti/ab/clm 三级、每组配直译词版与载体词版）。
2. 08:41:53-56 佰腾 CN 侧跑 4 条阶梯：ti 级两条均 0 记录(gateway 0 records)，ab 级两条合计 35 条（rows=10 candidates=20 pages=2 total=57 / rows=10 candidates=15 pages=2 total=15），clm 级 total=11891 收 20 条——**自动补跑阶梯 gained=55**。
3. USPTO 侧 3 条阶梯仅③（松档 `("microring resonator" OR "ring resonator")`）返回 200/gained=20；①②两条 404（日志被截断到 60 字符无法逐字复原，但两条均属 ≥3 AND 的连接级形态）。
4. 08:42:10 dead_filter_diag——**美方 15 条被判定失效**，状态清一色程序性文献：3 条 Provisional Application Expired、2 条 RO PROCESSING COMPLETED-PLACED IN STORAGE，granted 全 False。同批后续再滤 137 条（Abandoned--Failure to Respond / Patent Expired Due to NonPayment of Maintenance Fee）。
5. 08:42:18 relevance scoring 50 条（1.3s）→ 08:42:20 语义重排候选=28 → patent_search_result us_hits=20 cn_hits=55 total=75。
6. **08:42:20-08:42:50 空转段**：LLM 在 30 秒内连发 6 次 `search_patent_by_key_word`（08:42:20/21/22/23/25/26），每次 relevance scoring candidates=0、family seeds=0，relevance_pool gate 记 push=2、parseable=0、applies=False，全部零产出——其中两次（08:42:20:867、08:42:21:886）是 404。
7. 08:42:25-48 grounded_interpretation 触发（pool=56 scored=47 / 后 pool=155 scored=95），归纳出三条主线：核心光子器件与冗余拓扑层 / 失效监测与切换控制层 / 波长路由与光交换应用层，players=西安电子科技大学、华为技术有限公司、Ayar Labs, Inc.、中国科学院半导体研究所、东南大学——**从数据归纳而非预置词表，是需求#35 追求的正确形态**。
8. 08:42:57 第二轮 dual_patent_search，US 侧换载体词阶梯（add-drop filter / all-pass ring / racetrack resonator）**仍 404**，随即 `auto-ladder budget exhausted (used=3/3)`，2 条未试阶梯被跳过；CN 侧 clm 载体词版 total=29884 收 20 条 → us_hits=0 cn_hits=20 total=20。
9. 08:43:34 stored_conversation_patent_ids count=82 retrievable=82（**需求#29 正样本**：面板下发的 82 个号码全部可回读）；08:43:34 首轮助手回答 2581 字。首轮全程约 5 分钟，含 2 次语义重排与 2 轮家族探针（均 seeds=0）。
10. 08:48:11 追问「US12253745B2符合吗」（会话 sess_ba38d2e894c2，距首轮结束后约 4.5 分钟）：number_parse 把 US12253745 判为 ambiguous（**纯数字无法区分授权号与申请号**），lookups 取到 '12253745' 与 '122537452' 两条腿；08:48:30 number_resolve candidates=['US12253745B2'] merged=2 legs=USPTO(nums=12253745,122537452) — USPTO 2 hits；relevance scoring 2 条。
11. 08:48:34-37 按号取件成功：GET documents 200（doc_count=54、spec_count=1），LLM 选中 index=51 的 SPEC，下载 70531 字节 DOCX、抽出 66020 字——08:49:09 assistant_len=1048，_emit_patent_ids_to_frontend ids=['12253745','18363489']。
12. 08:49:42 用户**逐字重复同一提问**（query_id 7xyr1jga1ol，会话承接 sess_ba38d2e894c2，agent 复用池内实例）：number_resolve 结果与上轮完全一致（merged=2）、relevance scoring 2 条、无新的取件动作，08:50:15 assistant_len=772 收尾。

【系统响应核验（逐条对照日志）】
- 支持到的：四要素检索被完整承接并交付 75 条双库候选、82 个可回读号码、三条数据归纳主线；按号取件管道跑通（指定申请号的说明书下载+抽取+蒸馏）；需求#29 的可回读性再次实证（retrievable=82=count）；answer 文案未见复述通道失败叙事（未触犯需求#26）。
- 未支持到的：①**美方侧近乎全灭**——本批 37 次 USPTO 请求仅 9 次 200，首轮 6 条美方阶梯只 1 条通、可用活体仅 20 条，第二轮美方直接 0 条；137 条被程序性文献（临时申请/未授权公开）占掉；②**号码解析的命中集不含目标号码本身**——2 条候选中用户看到的题名是 LEAK DETECTOR，助手据此答"不符合"，而真正切题的记录（US18363489，题名 Silicon Photonic Device With Backup Light Paths）是被同一次解析**顺带**带出来的、系统并未识别它才是用户正在追的对象；③**用户连问三遍未获增量**——同一问题三次（08:48、08:49），系统每次用同一路径重跑，答案措辞微调而结论未澄清；④**空转浪费**——首轮 6 次零产出调用，且自动补跑把"已 404 的阶梯"计进尝试预算，导致两条最松的合规阶梯被跳过。"""

Q_SUPPORT = """部分支持——"检索"达到交付水位，"按号判定"未达到。

已支持：
① 四要素结构化检索被准确承接。语义解读把"微环+冗余+失效切换+波长路由"映射到容错型硅光波长选择/交换架构（players=Intel/Cisco/IBM/HPE/Oracle），8 条英文 + 6 条中文阶梯按紧→松排列，直译词版与载体词版成对生成，说明生成侧对概念词库的使用是正确的。
② 双源检索闭环跑通并交付。CN 4 条阶梯 55 条、US 1 条阶梯 20 条，经 relevance scoring(50) + 语义重排收敛后 patent_search_result total=75；面板下发 82 个号码且 **retrievable=82**（需求#29 正样本，用户后续可回读全部候选）。
③ 检索式透明化。专利检索式写入答案，用户可复制复现——需求#25 的能力在本轮可见。
④ 数据驱动的主线归纳。grounded_interpretation 从命中集归纳出"核心光子器件与冗余拓扑层 / 失效监测与切换控制层 / 波长路由与光交换应用层"三条主线与 5 位玩家（西安电子科技大学、华为、Ayar Labs、中科院半导体所、东南大学），未预置领域词，恰是需求#35 要求的通用机制。
⑤ 按号取件管道可用。US12253745B2 这一问触发了完整的 documents→SPEC 选取→DOCX 下载(70531B)→文本抽取(66020 字)→蒸馏链路，说明"按号取全文"这条路是通的。

未支持到（用户诉求的核心部分）：
① **号码命中集不含目标号码本身**。number_resolve 用「两位腿」查询（12253745 与 122537452）走 USPTO **全文检索**（`fetch_by_numbers` 以 `"12253745" OR "122537452"` 为 q、并以 sort=_score 排序），返回的是"相关度最高的 2 条"，不保证包含该号码自身的记录。结果用户看到的是题名 LEAK DETECTOR，结论"不符合"建立在一条与提问主题无关的记录上——这是**答案的事实基础错位**，而非措辞问题。
② 用户重复提问 3 次（08:48、08:49 逐字相同）未获增量。系统三次走完全相同的路径、给出同一结论的不同措辞，既没有自检"我这轮命中集里到底有没有这个号"，也没有反问澄清（如"你查的是不是 Silicon Photonic Device With Backup Light Paths 这一族"）。
③ 切题记录就在同一次解析里却未被识别。US18363489 具备①主硅环+备用硅环、②冗余光路、③光功率监测器检测并切换 RF/热调谐，只因④用途是**光调制器**而不满足排除项。助手按记录边界表述"不符合"在事实层面成立，但用户真正想知道的是"这件专利和我要找的是不是一回事"，系统未把"同一次解析里出现了一件高度贴题的设备"这一信号回给用户。
④ 号码问的上下文隔离：首轮 82 条候选池未被用于这一问的判定，"符合吗"的语义被窄化为对孤立号码做题名比对，而不是对"该号 vs 首轮四要素候选集"做关系判断。
⑤ 检索过程浪费与预算误用：首轮 6 次 `search_patent_by_key_word` 零产出（push=2 但 parseable=0，其中 2 次 404）；自动补跑阶梯把"已 404 的检索式"计为一次尝试（used=3/3），使两条最松、最可能合规的阶梯变体被跳过（`2 untried ladder queries skipped`）——**美方预算被前两条 404 消耗在"确定不会成功"的形态上**。
⑥ 美方源数据质量：首轮 15 条（后续累计 137 条）被死件过滤器判定为程序性文献（Provisional Application Expired / RO PROCESSING COMPLETED-PLACED IN STORAGE / Abandoned / 未缴年费终止），granted 全 False；第二轮美方 0 条可用。用户拿到的美方候选里真正"活体"的比例很低。"""

Q_ADVICE = """1. **按号解析必须做"中靶"校验**（最高优先，见需求#36）。`fetch_by_numbers` 走的是全文检索 + `sort=_score`（`sources/long_task/recall_sources.py:159`），返回的是相关度榜而非号码榜，因此"按号查"可能返回一堆不含该号的记录。修法：解析结果先与查询号码做一次确定性匹配（申请号/专利号任一字段归一化后相等），把命中集拆成「中靶记录」与「相关但非该号」两组；前者排首并标注，后者仅在无中靶时作补充且**必须显式声明"未取得该号码本身的记录"**。当前 `_lookup_number_candidates`（react_tools.py:3364）只按"有无条目"判断主源是否命中（`if len(merged) > before: break`），没有任何中靶判定。
2. **同问重复检测与主动澄清**（需求#38）。同一会话内对同一号码/同一诉求重复提问 ≥2 次时，系统应先做自检（本轮命中集构成 / 是否含目标号码 / 与上一轮答案的差异），给出增量信息或反问，而不是换措辞重念同一结论。重复提问是本项目已记录的流失前兆（1545… 同问 3 次、7577… 同诉求 4 次），本轮再添一例。
3. **阶梯预算按"结果"而非"发起"计费**（需求#37）。`_auto_run_patent_ladder`（react_tools.py:3015）在 `used_map[source] = used + 1` 处即扣额，未区分"语法被拒 404"与"真零命中"。建议：404 形态（`_flatten_query_for_uspto` 重试后仍 404）不计额，只对"回传 0 条"扣额；并把"本请求已 404 过的形态"记入 `_tried_queries`，避免同形态重发再耗一次。本轮 `used=3/3` 时被跳过的两条恰是最松、最可能命中的合规阶梯。
4. **要素命中视图**（需求#39）。用户以"① ② ③ ④"列举要素时，判定类回答应按要素逐项给"命中/部分命中/本轮记录未见"，而不是只复述题名。切题记录 US18363489 已具备①②③（主环+备环、冗余光路、光功率监测切换），仅④因"光调制器"不满足——这一结构对用户极有价值（说明该技术路线确实存在、只是用途分支不同），当前答复把它降格成了"另一个申请号"。
5. **美方程序性文献过滤前移**（需求#40）。137 条被死件过滤器拦下说明数据源里临时申请/未授权公开占比很高；建议在 `prune()`/`ranked()` 之前就把 Provisional 挡在打分窗口外（避免"滤掉后又要重发检索式找活体"），并在活体不足时优先执行最松阶梯——而这条依赖第 3 项的预算修法。
6. **首轮空转收敛**。首轮 30 秒内 6 次 `search_patent_by_key_word` 全部零产出（parseable=0）。建议：同一工具连续 2 次零产出后不再放行后续调用，或把"零产出"作为观察信号回灌给模型（当前 observe 为空串，模型无从得知自己做错了什么）。"""

# ── 用户运营分析 ────────────────────────────────────────────────────────────
OPS_ROWS = [
    (
        "新增样本10(09-19 08:41,第21位样本-四要素查新/注册即用)",
        "第 21 位样本（新注册 QQ 邮箱 854173195@qq.com，内部 uid 11714190467530210252），"
        "2026-09-19 08:41-08:50。**注册即用极限值**：signup→首问 0.7 秒，是已记录批次中最快；"
        "注册后先选场景(scenes=[1])再提问，说明 onboarding 流程未构成摩擦。"
        "**首问形态是新的**：不是品类词摸底，而是携一份四要素结构化的检索任务书"
        "（①微环谐振器 ②冗余/备份微环 ③检测失效并切备环 ④波长路由/光交换，且**显式用括号排除调制器**）"
        "——编号列举+排除项的写法是 FTO/查新作业的标准输入格式，画像不像品类买家，"
        "更像硅光/光交换芯片方向的发明人或 IP 从业者。"
        "行为链：75 条双库候选检索（~5 分钟）→ 面板 82 号可回读 → 45 秒后转按号追问"
        "→ **同一问题连问 3 遍**（08:48、08:49 逐字重复）→ 无新动作收尾。",
    ),
    (
        "新增样本10-交互密度与流失信号",
        "本批仅 1 位用户、3 次提问，但**交互密度高**：单轮内 6 条美方阶梯、4 条中文阶梯、"
        "2 轮语义重排、2 轮家族探针、1 次按号取件（54 个文档中选 1 篇 SPEC、66020 字）。"
        "流失风险信号明确：**同一提问逐字重复 3 次**是本项目记录的流失前兆"
        "（1545… 同问 3 次、7577… 同诉求 4 次），且本轮重复发生在**系统已交付了 82 个候选号码的前提下**"
        "——说明候选清单的交付达成了，但在用户真正想确认的那一件上没有得到答案。"
        "值得注意：用户没有放弃，而是回到面板/会话继续问同一号，属于"
        "「不满意但仍在场」状态，是**可挽回**的信号窗口。",
    ),
    (
        "新增样本10-该批美方源质量",
        "37 次 USPTO 请求仅 9 次 200（约 24%）。死件过滤器两轮共滤 137 条，状态清一色程序性文献："
        "Provisional Application Expired、RO PROCESSING COMPLETED-PLACED IN STORAGE、"
        "Abandoned--Failure to Respond to an Office Action、Patent Expired Due to NonPayment of Maintenance Fee，"
        "granted 全部 False。首轮美方 6 条阶梯只 1 条通（可用活体 20 条），第二轮美方直接 0 条。"
        "该批用户的美方获得感因此显著低于中文侧——与 [[uspto-search-404-zero-hits]]、"
        "[[uspto-score-sort-pathology]] 记录的问题同源，但本轮的新形态是"
        "**自动补跑把已 404 的阶梯计入尝试预算(used=3/3)，使两条最松的合规阶梯被直接跳过**。",
    ),
]

# ── 需求列表 ────────────────────────────────────────────────────────────────
REQ_ROWS = [
    (
        "36",
        "按号解析的中靶校验（按号查询的命中集必须显式区分「中靶记录（号码自身）」与"
        "「相关但非该号」；当前 fetch_by_numbers 走 USPTO 全文检索 + sort=_score 返回相关度榜，"
        "不保证含该号码自身，_lookup_number_candidates 仅以「有无条目」判定主源命中、无中靶判定；"
        "无中靶时须明确告知「未取得该号码本身的记录」，禁止据非中靶记录的题名作答）",
        "1/24:11714190467530210252(2026-09-19 按号问 US12253745B2，number_resolve merged=2 却返回 "
        "LEAK DETECTOR 题名；用户三次重复提问未获澄清)",
        "高",
        "高",
        "高",
        "P0",
    ),
    (
        "37",
        "检索阶梯预算按结果计费（语法被拒/404 形态不得扣减重试额度，仅「真零命中」扣额；"
        "同请求内同形态重发不再计一次；松档阶梯因预算耗尽被跳过时须在答复中如实说明）",
        "1/24:11714190467530210252(2026-09-19 美方 3 连 404 后即 used=3/3，"
        "2 条未试的松档阶梯被跳过，次轮美方 us_hits=0)",
        "高",
        "高",
        "高",
        "P1",
    ),
    (
        "38",
        "同问重复检测与主动澄清（同一会话内对同一号码/同一诉求重复提问 ≥2 次时，"
        "作答前先自检：本轮命中集构成 / 是否含目标号码 / 与上一轮答案的差异；"
        "有增量则给增量，无增量则反问澄清，禁止换措辞重念同一结论）",
        "1/24:11714190467530210252(2026-09-19 08:48 与 08:49 逐字重复「US12253745B2符合吗」，"
        "系统两次同路径重跑、结论未澄清)",
        "高",
        "中",
        "高",
        "P0",
    ),
    (
        "39",
        "判定类回答的要素命中视图（用户以① ② ③ ④列举要素并问「符合吗」时，"
        "须对目标记录逐要素标注命中/部分命中/本轮记录未见，而非仅复述题名；"
        "同一轮解析中出现的其他号码若高度贴题，须显式提示「本次解析同时返回了…」）",
        "1/24:11714190467530210252(2026-09-19 同次解析返回的 US18363489 具备①②③要素、"
        "仅④因光调制器用途不满足，答复仅将其降格为「另一个申请号」)",
        "中",
        "高",
        "高",
        "P1",
    ),
    (
        "40",
        "美方程序性文献过滤与活体保底（Provisional / 未授权公开 / RO 归档件在进入打分窗口前即剔除，"
        "避免占页后再耗检索式找活体；活体不足时优先执行最松档阶梯，"
        "并在答复中如实给出「可用活体 N 条」的口径）",
        "1/24:11714190467530210252(2026-09-19 死件过滤器两轮共滤 137 条，granted 全 False；"
        "首轮美方可用活体仅 20 条、次轮 0 条)",
        "中",
        "高",
        "中",
        "P1",
    ),
]


def _clone_row_style(ws, src_row: int, dst_row: int) -> None:
    for c in range(1, ws.max_column + 1):
        s = ws.cell(src_row, c)
        d = ws.cell(dst_row, c)
        if s.has_style:
            d._style = copy.copy(s._style)


def main() -> None:
    shutil.copy2(MAIN, os.path.join(BASE, "user_issue_analyze - bak20260920b.xlsx"))
    wb = openpyxl.load_workbook(MAIN)

    ws = wb["用户问题分析"]
    r = ws.max_row + 1
    _clone_row_style(ws, ws.max_row, r)
    for c, v in enumerate([Q_USER, Q_PROCESS, Q_SUPPORT, Q_ADVICE], 1):
        ws.cell(r, c, v)
    ws.cell(r, 1).alignment = copy.copy(ws.cell(r - 1, 1).alignment)
    print("用户问题分析 +1 → R%d" % r)

    ws = wb["用户运营分析"]
    for dim, concl in OPS_ROWS:
        r = ws.max_row + 1
        _clone_row_style(ws, ws.max_row, r)
        ws.cell(r, 1, dim)
        ws.cell(r, 2, concl)
    print("用户运营分析 +%d" % len(OPS_ROWS))

    ws = wb["需求列表"]
    for seq, need, ev, urg, imp, val, pri in REQ_ROWS:
        r = ws.max_row + 1
        _clone_row_style(ws, ws.max_row, r)
        for c, v in enumerate([seq, need, ev, urg, imp, val, pri], 1):
            ws.cell(r, c, v)
    print("需求列表 +%d → R%d" % (len(REQ_ROWS), ws.max_row))

    wb.save(MAIN)
    shutil.copy2(MAIN, VIEW)
    print("saved + synced view")


if __name__ == "__main__":
    main()
