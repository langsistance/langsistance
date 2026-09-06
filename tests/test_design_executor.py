"""test_design_executor: execute_design_clearance terminal mapping (design P1 T7b).

Offline coverage follows the repository convention that celery task *bodies* are
not invoked without a broker (test_failure_terminal_state tests the equivalent
family terminal helper directly, not the full bound task).  Here we cover the two
terminal helpers that carry the executor's mapping semantics:

- success ``_design_complete`` → ``set_task_completed`` fires once with an
  anchor_payload derived from the pipeline digest (mirror of families); and
- ``_design_exception_hard_stop`` → ``_notify_terminal_failure`` + MySQL failed
  marker fire exactly once (the single notify/analytics exit), and the user queue
  is completed when a user_id is present.

The real-network ``_google_fetch`` path and router→task dispatch need a
broker/Redis/MySQL and are covered by the manual smoke steps in task-7 report.
"""
import unittest
from unittest import mock

import celery_worker  # noqa: E402  (imports cleanly off-server, broker not used)


def _tid():
    return "lt_design_1"


# ── success: _design_complete persists digest → set_task_completed + anchor ──

class TestDesignComplete(unittest.TestCase):
    def test_writes_set_task_completed_anchor_from_digest(self):
        result = {"report_md": "# ok",
                  "digest": {"target": "pirate ship mug",
                             "result_ids": ["USD9"], "totals": {}}}
        seen = {}

        def fake_set(tid, files, patent_ids=None, anchor_payload=None):
            seen["files"] = files
            seen["patent_ids"] = patent_ids
            seen["anchor"] = anchor_payload

        with mock.patch(
                "sources.long_task.status_manager.set_task_completed",
                side_effect=fake_set), \
             mock.patch("sources.long_task.user_queue.complete_user_task") as _cq:
            celery_worker._design_complete(_tid(), result,
                                           {"source": "us_design"}, "user-7")
        _cq.assert_called_once()
        self.assertEqual(seen["files"], [])
        self.assertEqual(seen["patent_ids"], ["USD9"])
        anchor = seen["anchor"]
        self.assertIsNotNone(anchor)
        self.assertEqual(anchor["source"], "us_design")
        self.assertEqual(anchor["target"], "pirate ship mug")
        self.assertIn("USD9", anchor["result_ids"])

    def test_no_user_id_skips_queue_completion(self):
        with mock.patch(
                "sources.long_task.status_manager.set_task_completed") as _st, \
             mock.patch("sources.long_task.user_queue.complete_user_task") as _cq:
            celery_worker._design_complete(_tid(), {"digest": {}}, {}, "")
        _st.assert_called_once()
        _cq.assert_not_called()


# ── failure: single terminal exit pairs notify + MySQL once ──

class TestDesignFailedTerminal(unittest.TestCase):
    def test_pairs_notify_and_mysql_once_with_queue_when_user(self):
        calls = {"notify": [], "mysql": [], "queue": []}

        def fn(tid, err): calls["notify"].append((tid, err))
        def mq(tid, phase, progress, **kw):
            calls["mysql"].append((tid, phase, progress))
        def cq(uid, tid): calls["queue"].append((uid, tid))

        with mock.patch("celery_worker._notify_terminal_failure",
                        side_effect=fn), \
             mock.patch("celery_worker._update_mysql_progress",
                        side_effect=mq), \
             mock.patch("sources.long_task.user_queue.complete_user_task",
                        side_effect=cq):
            celery_worker._design_exception_hard_stop(_tid(), "u9", "需要澄清")
        self.assertEqual(len(calls["notify"]), 1)
        self.assertEqual(len(calls["mysql"]), 1)
        self.assertEqual(calls["mysql"][0][1:], ("failed", 0))
        self.assertEqual(calls["notify"][0], (_tid(), "需要澄清"))
        self.assertEqual(len(calls["queue"]), 1)

    def test_empty_user_skips_queue_but_still_pairs_terminal(self):
        calls = {"notify": [], "mysql": []}
        with mock.patch("celery_worker._notify_terminal_failure",
                        side_effect=lambda tid, e: calls["notify"].append(tid)), \
             mock.patch("celery_worker._update_mysql_progress",
                        side_effect=lambda tid, ph, pr, **kw:
                        calls["mysql"].append((tid, ph, pr))):
            celery_worker._design_exception_hard_stop(_tid(), "", "服务暂不可用")
        self.assertEqual(len(calls["notify"]), 1)
        self.assertEqual(calls["mysql"][0][1:], ("failed", 0))


# ── core plug point: design-clearance intent detector (additive, pure) ──

from sources.design.clearance_intent import (
    has_design_cue, design_clearance_intent)  # noqa: E402 — firebase-free module


class TestDesignClearanceIntentDetector(unittest.TestCase):
    def test_image_required(self):
        # image is the hinge: query-only, no upload, cannot drive visual clearance
        self.assertFalse(design_clearance_intent("外观专利比对", []))
        self.assertFalse(design_clearance_intent("外观专利比对", None))

    def test_design_cue_required(self):
        # image present but no appearance-design cue word → not clearance-routed
        self.assertFalse(design_clearance_intent("帮我看看这张图", ["p.jpg"]))

    def test_image_plus_cue_matches(self):
        self.assertTrue(
            design_clearance_intent("请做外观侵权比对", ["pic.png", " b.jpg "]))

    def test_has_design_cue_domain_terms(self):
        self.assertTrue(has_design_cue("此产品的外观设计专利情况"))
        self.assertTrue(has_design_cue("检查这杯子的设计专利"))
        self.assertFalse(has_design_cue("帮我查这个号码的审查流程"))


if __name__ == "__main__":
    unittest.main()
