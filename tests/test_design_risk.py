from sources.design.design_risk import (
    effective_until, is_expired, risk_of_score, aggregate,
    build_report_md, build_digest, DesignCandidate, JudgeVerdict)


def test_term_2015_boundary():
    assert effective_until("2015-05-13") == "2030-05-13"   # 15y
    assert effective_until("2015-05-12") == "2029-05-12"   # 14y
    assert effective_until("2010-01-01") == "2024-01-01"


def test_term_leap_day_safe():
    assert effective_until("2016-02-29") == "2031-02-28"   # 加年后 2/29 不存在


def test_is_expired_boundary():
    # 语义: 届满当日(含)仍有效; 次日才过期。
    # 分界点落在 14y 期(收 grant<2015-05-13): grant 2012-09-01 → eff 2026-09-01。
    assert is_expired("2012-08-31", "2026-09-01") is True   # eff 2026-08-31 → 已过
    assert is_expired("2012-09-01", "2026-09-01") is False  # 届满当日仍有效
    assert is_expired("2012-09-02", "2026-09-01") is False


def test_risk_of_score():
    assert risk_of_score(0.70) == "high"
    assert risk_of_score(0.69) == "medium"
    assert risk_of_score(0.45) == "medium"
    assert risk_of_score(0.44) == "low"


def test_aggregate_splits_and_flags():
    active = [DesignCandidate("USD1A", "Toy snake", "2023-01-01", status="active")]
    expired = [DesignCandidate("USD9Z", "Toy snake old", "2008-01-01", status="expired")]
    verdicts = [JudgeVerdict("USD1A", 0.82, "high",
                             (("整体轮廓", 0.8, "分节走向一致"),), "蛇头近似", "尾部开关位置不同")]
    agg = aggregate(active, verdicts, expired, today="2026-09-01")
    assert agg["high"] == [verdicts[0]]
    assert agg["expired_hits"] == expired
    assert agg["medium"] == [] and agg["low"] == []


def test_report_and_digest_structures():
    active = [DesignCandidate("USD1A", "Toy snake", "2023-01-01", assignee="ACME")]
    verdicts = [JudgeVerdict("USD1A", 0.82, "high", (("整体轮廓", 0.8, "x"),),
                             "basis", "diff")]
    agg = aggregate(active, verdicts, [], today="2026-09-01")
    md = build_report_md("product.jpg", agg)
    assert "USD1A" in md and "0.82" in md and "ACME" in md
    assert "非法律意见" in md
    dig = build_digest("product.jpg", agg)
    assert dig["target"] == "product.jpg"
    assert dig["type"] == "file"
    assert "USD1A" in dig["result_ids"] and len(dig["result_ids"]) <= 50


def test_aggregate_scores_medium_not_in_high():
    # score 0.5 落 medium 档, 不入 high。
    active = [DesignCandidate("USD2B", "Mug", "2023-01-01", status="active")]
    verdicts = [JudgeVerdict("USD2B", 0.50, "medium", (("手柄", 0.5, "n"),),
                             "basis", "diff")]
    agg = aggregate(active, verdicts, [], today="2026-09-01")
    assert agg["medium"] == [verdicts[0]]
    assert agg["high"] == []


def test_aggregate_dedup_keeps_highest_and_expired_not_in_result_ids():
    # 同 d_number 去重保留最高 score; suppressed(phase out) 件不进 aggregate 判定。
    # 夹具收敛: 单件候选 (aggregate 不去重候选, 断言意图仍在同 D 号两件 verdict 留高)。
    active = [DesignCandidate("USD3C", "Lamp", "2023-01-01", status="active")]
    low_v = JudgeVerdict("USD3C", 0.50, "medium", (("灯罩", 0.5, "n"),),
                         "basis", "diff")
    high_v = JudgeVerdict("USD3C", 0.85, "high", (("灯罩", 0.85, "y"),),
                          "basis", "diff")
    expired = [DesignCandidate("USD9Z", "old", "2008-01-01", status="expired")]
    agg = aggregate(active, [low_v, high_v], expired, today="2026-09-01")
    assert agg["high"] == [high_v]
    assert agg["medium"] == []
    dig = build_digest("product.jpg", agg)
    assert "USD3C" in dig["result_ids"]
    assert "USD9Z" not in dig["result_ids"]
