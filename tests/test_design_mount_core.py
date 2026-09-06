#!/usr/bin/env python3
"""Behavioral target tests for the Task-8 core mount: resume/claim dispatch by type.

Review flag (Important): the c03bccf ``design_clearance`` "core mount" shipped pure
gate unit tests (``seller_design_clearance_gate`` + ``has_design_cue``) and executor
terminal-helper tests, but nothing proofed the branch at the dispatch/submit seam —
that a route decision collapses to ``task_type='design_clearance'`` and re-enters the
right executor rather than the default ``execute_patent_analysis``.

Because the full live-upload handler lives inside a heavy DB/Redis/OCR closure
(``_handle_file_upload_query``) that also cannot currently be imported offline — see
coverage-boundary note at the bottom — the reachable, broker-free seam here is the
companion *resume/claim* dispatcher ``celery_worker._dispatch_queued_task``: it reads
the stored ``long_tasks.task_type`` (exactly the ``'design_clearance'`` value the core
upload handler writes on a three-condition hit) and must send the queued task to
``execute_design_clearance`` — never to ``execute_patent_analysis``.  Mirror of the
family/prosecution resume tests and the terminal-helper style in test_design_executor.
"""
import json
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# celery_worker imports cleanly off-server (broker not touched at import time).
import celery_worker

from sources.design.clearance_intent import (
    has_design_cue,
    is_image_file,
    seller_design_clearance_gate,
    uploaded_image_refs_of,
)


def _row(overrides):
    row = {
        "input_params": json.dumps({
            "query": "比对这款产品是否外观侵权",
            "patent_ids": [],
            "patent_source": "us_design",
            "product_text": "比对这款产品是否外观侵权",
            "product_image_refs": ["/tmp/shot.png"],
            "source": "us_design",
        }),
        "session_id": "sess_r",
        "scene_id": None,
        "task_type": "design_clearance",
    }
    row.update(overrides)
    return row


class TestDispatchQueuedDesignClearance(unittest.TestCase):
    """task_type=='design_clearance' stored row ⇒ execute_design_clearance, not the
    default patent_analysis — so the Task-8 mount survives pauses on the right worker."""

    def test_design_clearance_rows_resume_to_design_executor(self):
        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value.fetchone.return_value = \
            _row({})
        delay_seen = {}

        def _cap(target):
            def w(**kw):
                delay_seen[target] = kw
            return w

        with patch(
                "sources.knowledge.knowledge.get_db_connection",
                return_value=conn), \
             patch("celery_worker.execute_design_clearance.delay",
                   side_effect=_cap("executor")) as d1, \
             patch("celery_worker.execute_patent_analysis.delay") as d2:
            celery_worker._dispatch_queued_task("lt_design_resume", "user-1")

        d1.assert_called_once()
        d2.assert_not_called()  # never falls back to the default analysis path
        kw = delay_seen["executor"]
        self.assertEqual(kw["task_id"], "lt_design_resume")
        self.assertEqual(kw["params"]["product_image_refs"], ["/tmp/shot.png"])
        self.assertEqual(kw["params"]["source"], "us_design")
        self.assertEqual(kw["params"]["product_text"],
                         "比对这款产品是否外观侵权")
        # image refs stay refs; they are never smuggled in as patent_ids
        self.assertEqual(kw["params"]["patent_ids"], [])

    def test_queued_design_row_without_image_preserves_product_text(self):
        # product_text field absent in a hand-queued row ⇒ falls back to query text
        # (bytes-for-bytes the resume path's T8 semantics).
        stored = json.loads(_row({})["input_params"])
        del stored["product_text"]
        row = _row({})
        row["input_params"] = json.dumps(stored)
        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value.fetchone.return_value = row
        seen = {}

        def cap(**kw):
            seen["target"] = kw["params"]

        with patch(
                "sources.knowledge.knowledge.get_db_connection",
                return_value=conn), \
             patch("celery_worker.execute_design_clearance.delay",
                   side_effect=cap), \
             patch("celery_worker.execute_patent_analysis.delay") as d2:
            celery_worker._dispatch_queued_task("lt_d2", "user-1")
        d2.assert_not_called()
        self.assertEqual(seen["target"]["product_text"],
                         stored["query"])


class TestGateMappingStaysConsistent(unittest.TestCase):
    """The three-condition gate inputs that the live upload maps to task_type live in
    the pure helper; these guard T8's controller ruling without the heavy route."""

    IMAGES = [{"filename": "shot.jpg", "path": "/u/shot.jpg",
               "content_type": "image/jpeg"}]

    def test_hit_when_image_seller_and_cue(self):
        img = {"filename": "产品图.jpg", "path": "/u/产品图.jpg",
               "content_type": "image/jpeg"}
        # query cue + image + seller ⇒ route
        self.assertTrue(
            seller_design_clearance_gate("seller", [img], "请做外观侵权比对"))
        # no query cue but an image *filename* carrying the cue still counts ∨
        # over file refs — an image batch is never silently dropped.
        cuefile = {"filename": "外观侵权产品.jpg", "path": "/u/x.jpg",
                   "content_type": "image/jpeg"}
        self.assertTrue(seller_design_clearance_gate("seller", [cuefile],
                                                     "查查这产品"))

    def test_miss_when_any_of_three_conditions_is_absent(self):
        img = {"filename": "shot.jpg", "path": "/u/shot.jpg",
               "content_type": "image/jpeg"}
        # image present but scene not seller
        self.assertFalse(seller_design_clearance_gate("", [img],
                                                      "外观侵权比对"))
        # scene seller + cue but NO image
        self.assertFalse(
            seller_design_clearance_gate("seller", [], "外观侵权比对"))
        # scene seller + image but NO cue (query nor filename)
        pdf = {"filename": "spec.pdf", "path": "/u/spec.pdf",
               "content_type": "application/pdf"}
        self.assertFalse(seller_design_clearance_gate("seller", [img],
                                                      "帮我查一下"))
        # non-image ref never routes even with cue + seller
        self.assertFalse(seller_design_clearance_gate("seller", [pdf],
                                                      "外观侵权比对"))


if __name__ == "__main__":
    unittest.main()
