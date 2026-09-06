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
        result = {"report_md": "# ok report body",
                  "digest": {"target": "pirate ship mug",
                             "result_ids": ["USD9"], "totals": {}}}
        seen = {}

        def fake_set(tid, files, patent_ids=None, anchor_payload=None):
            seen["files"] = files
            seen["patent_ids"] = patent_ids
            seen["anchor"] = anchor_payload

        def fake_update(tid, phase, progress, step_msg, status='running',
                        **extra):
            seen["summary_kw"] = extra

        with mock.patch(
                "sources.long_task.status_manager.set_task_completed",
                side_effect=fake_set), \
             mock.patch("sources.long_task.status_manager.update_task_status",
                        side_effect=fake_update) as _up, \
             mock.patch("sources.long_task.user_queue.complete_user_task") as _cq:
            celery_worker._design_complete(_tid(), result,
                                           {"source": "us_design"}, "user-7")
        _cq.assert_called_once()
        # I1: report_md is persisted to the sticky result_summary so the
        # completed-conversation digest (task_messages.build_result_digest)
        # carries the report text rather than a bare "任务已完成".
        _up.assert_called_once()
        self.assertEqual(seen["summary_kw"].get("result_summary"),
                         "# ok report body")
        self.assertEqual(seen["files"], [])
        self.assertEqual(seen["patent_ids"], ["USD9"])
        anchor = seen["anchor"]
        self.assertIsNotNone(anchor)
        self.assertEqual(anchor["source"], "us_design")
        self.assertEqual(anchor["target"], "pirate ship mug")
        self.assertIn("USD9", anchor["result_ids"])

    def test_no_report_md_skips_summary_persist(self):
        with mock.patch(
                "sources.long_task.status_manager.set_task_completed") as _st, \
             mock.patch("sources.long_task.status_manager.update_task_status"
                        ) as _up, \
             mock.patch("sources.long_task.user_queue.complete_user_task") as _cq:
            celery_worker._design_complete(_tid(), {"digest": {}}, {}, "")
        _st.assert_called_once()
        _cq.assert_not_called()
        _up.assert_not_called()   # 无 report → 不写 result_summary

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


# ── T8 core plug point: seller three-condition gate (pure, broker-free) ──

from sources.design.clearance_intent import (
    is_image_file, uploaded_image_refs_of, seller_design_clearance_gate,
)  # noqa: E402


class TestSellerDesignClearanceGate(unittest.TestCase):
    """Truth table: image-file ∧ scene=="seller" ∧ appearance cue → route.

    Non-hit cases (缺 scene / 非 seller scene / 无 cue / 非图) must all be False
    so the upload carries on its original patent path unchanged.
    """

    IMG = [{"filename": "mug.png", "path": "/up/a.png", "content_type": "image/png"}]
    DOC = [{"filename": "spec.pdf", "path": "/up/b.pdf", "content_type": "application/pdf"}]

    def test_all_three_true_routes(self):
        self.assertTrue(
            seller_design_clearance_gate("seller", self.IMG, "帮忙做外观侵权比对"))

    def test_missing_scene_false(self):
        self.assertFalse(
            seller_design_clearance_gate("", self.IMG, "外观侵权比对"))
        self.assertFalse(
            seller_design_clearance_gate(None, self.IMG, "外观侵权比对"))

    def test_non_seller_scene_false(self):
        self.assertFalse(
            seller_design_clearance_gate("pro", self.IMG, "外观侵权比对"))
        self.assertFalse(
            seller_design_clearance_gate("sellerX", self.IMG, "外观侵权比对"))

    def test_missing_cue_false(self):
        self.assertFalse(
            seller_design_clearance_gate("seller", self.IMG, "看看这张图片"))

    def test_non_image_file_false(self):
        self.assertFalse(
            seller_design_clearance_gate("seller", self.DOC, "外观侵权比对"))
        self.assertFalse(
            seller_design_clearance_gate("seller", [], "外观侵权比对"))

    def test_cue_can_come_from_filename(self):
        self.assertTrue(
            seller_design_clearance_gate(
                "seller", [{"filename": "外观专利比对.png", "path": "/up/x.png"}],
                "帮我查这个"))

    # pure image-file suffix detection
    def test_is_image_file_extensions(self):
        self.assertTrue(is_image_file("shot.PNG"))
        self.assertTrue(is_image_file("a.jpg"))
        self.assertTrue(is_image_file("b.webp"))
        self.assertFalse(is_image_file("doc.pdf"))
        self.assertFalse(is_image_file("doc.docx"))
        self.assertFalse(is_image_file(""))

    def test_uploaded_image_refs_of_filters(self):
        refs = [{"filename": "mug.png", "path": "/u/m.png"},
                {"filename": "spec.pdf", "path": "/u/spec.pdf"},
                {"filename": "noext", "path": "/u/noext"}]
        got = uploaded_image_refs_of(refs)
        self.assertEqual(got, ["/u/m.png"])


# ── T8: resume dispatch maps design_clearance to its own executor ──

class TestDispatchDesignClearanceResume(unittest.TestCase):
    """_dispatch_queued_task task_type table: design_clearance → executor."""

    def test_dispatch_branches_for_design_clearance(self):
        # read-only probe that the resume dispatch no longer lets a
        # design_clearance row fall through to execute_patent_analysis.
        import inspect
        import celery_worker as _cw
        body = inspect.getsource(_cw._dispatch_queued_task)
        self.assertIn("elif task_type == 'design_clearance':", body)
        self.assertIn("execute_design_clearance.delay", body)


# ── I2: L2 pdf_fetch seam is real asyn-a-browser-crush, not the sync default ──

class TestDesignPdfFetchSeam(unittest.TestCase):
    def test_placeholder_never_falls_back_to_sync_default(self):
        # design_pipeline drives design_image.fetch_design_pdf(pdf_fetch=_pdf_fetch).
        # The un-wired placeholder must raise (fail closed), so L2 never silently
        # uses design_image._fetch_pdf_http on an idle/offline run.
        import asyncio
        from sources.design.design_pipeline import _pdf_fetch

        async def probe():
            try:
                await _pdf_fetch("https://patentimages.storage.googleapis.com/x.pdf")
                return "no-error"
            except RuntimeError as e:
                return str(e)

        self.assertIn("not wired", asyncio.run(probe()))

    def test_executor_real_fetcher_is_async_fail_open_byte_tuple(self):
        # _design_pdf_fetch: async, answers the design_image pdf_fetch contract
        # (status:int, body:bytes) and degrades network errors to (0, b"").
        import asyncio
        import inspect
        self.assertTrue(inspect.iscoroutinefunction(
            celery_worker._design_pdf_fetch))
        import httpx as _httpx

        class _Resp:
            status_code = 200
            content = b"%PDF-1.4"
        with mock.patch.object(_httpx, "AsyncClient") as _ac:
            async def __aenter__(_self):
                return mock.MagicMock(get=mock.AsyncMock(return_value=_Resp()))
            async def __aexit__(_self, *a):
                return False
            _ac.return_value.__aenter__ = __aenter__
            _ac.return_value.__aexit__ = __aexit__
            got = asyncio.run(celery_worker._design_pdf_fetch(
                "https://patentimages.storage.googleapis.com/x.pdf"))
        self.assertEqual(got, (200, b"%PDF-1.4"))

    def test_executor_wires_pdf_fetch_into_pipeline(self):
        # The executor's _run replaces design_pipeline._pdf_fetch with the real
        # async fetcher (mirror of the _google_fetch seam).  Read-only probe of
        # the source wiring keeps this broker-free.
        import inspect
        src = inspect.getsource(celery_worker.execute_design_clearance)
        self.assertIn("design_pipeline._pdf_fetch = _design_pdf_fetch", src)


# ── core plug point: design-clearance intent detector (additive, pure) ──
