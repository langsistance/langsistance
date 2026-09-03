# -*- coding: utf-8 -*-
"""[MODEL] role-override layer (2026-09).

A single config section lets every expensive/visible model role switch to
another provider+model pair without touching the role's original config
location.  Rules under test:
- provider+model must be set TOGETHER — a half-pair is ignored.
- vision: [MODEL] vision_* pair overrides [LONG_TASK] vision_provider/model
  inside get_long_task_config (all celery vision sites read it).
- stream: [MODEL] stream_* pair > [PROSECUTION] streaming_* > ('', '').
- all-empty behaviour is identical to before (backward compatible).
"""
import os
import tempfile
import unittest


def _write_ini(section_lines: list) -> str:
    """Write a temp config.ini body; return its path."""
    body = "\n".join(section_lines) + "\n"
    fd, path = tempfile.mkstemp(suffix=".ini", prefix="gsd_model_test_")
    try:
        os.write(fd, body.encode("utf-8"))
    finally:
        os.close(fd)
    return path


def _cleanup(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


from sources.long_task.config import (  # noqa: E402
    get_long_task_config,
    get_model_overrides,
    get_streaming_provider_model,
    _override_pair,
)

_BASE_LONG = [
    "[LONG_TASK]",
    "provider_family = deepseek",
    "vision_enabled = true",
    "vision_provider = minimax",
    "vision_model = MiniMax-M3",
]


class TestGetModelOverrides(unittest.TestCase):
    def test_no_model_section_returns_empty_pairs(self):
        path = _write_ini(_BASE_LONG)
        try:
            ov = get_model_overrides(path)
            for role in ("chat", "interpret", "stream", "vision"):
                self.assertEqual(ov[f"{role}_provider"], "")
                self.assertEqual(ov[f"{role}_model"], "")
        finally:
            _cleanup(path)

    def test_full_pairs_parsed(self):
        path = _write_ini(_BASE_LONG + [
            "[MODEL]",
            "chat_provider = deepseek",
            "chat_model = deepseek-v4-flash",
            "vision_provider = deepseek",
            "vision_model = deepseek-v4-flash-vision-exp",
        ])
        try:
            ov = get_model_overrides(path)
            self.assertEqual(ov["chat_provider"], "deepseek")
            self.assertEqual(ov["chat_model"], "deepseek-v4-flash")
            self.assertEqual(ov["vision_model"], "deepseek-v4-flash-vision-exp")
            self.assertEqual(ov["interpret_provider"], "")
        finally:
            _cleanup(path)

    def test_half_pair_is_ignored_by_resolver(self):
        path = _write_ini(_BASE_LONG + [
            "[MODEL]",
            "chat_provider = deepseek",  # model missing → no override
        ])
        try:
            ov = get_model_overrides(path)
            provider, model = _override_pair(ov, "chat",
                                             fallback_provider="openrouter",
                                             fallback_model="gpt-x")
            self.assertEqual((provider, model), ("openrouter", "gpt-x"))
        finally:
            _cleanup(path)


class TestVisionOverride(unittest.TestCase):
    def test_long_task_defaults_kept_without_override(self):
        path = _write_ini(_BASE_LONG)
        try:
            cfg = get_long_task_config(path)
            self.assertEqual(cfg["vision_provider"], "minimax")
            self.assertEqual(cfg["vision_model"], "MiniMax-M3")
        finally:
            _cleanup(path)

    def test_model_vision_pair_overrides_long_task(self):
        path = _write_ini(_BASE_LONG + [
            "[MODEL]",
            "vision_provider = deepseek",
            "vision_model = deepseek-v4-flash-vision-exp",
        ])
        try:
            cfg = get_long_task_config(path)
            self.assertEqual(cfg["vision_provider"], "deepseek")
            self.assertEqual(cfg["vision_model"], "deepseek-v4-flash-vision-exp")
        finally:
            _cleanup(path)

    def test_model_vision_half_pair_keeps_long_task(self):
        path = _write_ini(_BASE_LONG + [
            "[MODEL]",
            "vision_model = deepseek-v4-flash-vision-exp",  # provider missing
        ])
        try:
            cfg = get_long_task_config(path)
            self.assertEqual(cfg["vision_provider"], "minimax")
            self.assertEqual(cfg["vision_model"], "MiniMax-M3")
        finally:
            _cleanup(path)


class TestStreamingProviderModel(unittest.TestCase):
    def test_legacy_prosecution_section_wins_without_model_pair(self):
        path = _write_ini(_BASE_LONG + [
            "[PROSECUTION]",
            "streaming_provider = openrouter",
            "streaming_model = openai/gpt-5.6-terra",
        ])
        try:
            provider, model = get_streaming_provider_model(path)
            self.assertEqual(provider, "openrouter")
            self.assertEqual(model, "openai/gpt-5.6-terra")
        finally:
            _cleanup(path)

    def test_model_stream_pair_overrides_prosecution(self):
        path = _write_ini(_BASE_LONG + [
            "[PROSECUTION]",
            "streaming_provider = openrouter",
            "streaming_model = openai/gpt-5.6-terra",
            "[MODEL]",
            "stream_provider = deepseek",
            "stream_model = deepseek-v4-flash",
        ])
        try:
            provider, model = get_streaming_provider_model(path)
            self.assertEqual(provider, "deepseek")
            self.assertEqual(model, "deepseek-v4-flash")
        finally:
            _cleanup(path)

    def test_unset_returns_empty_pair(self):
        path = _write_ini(_BASE_LONG)
        try:
            provider, model = get_streaming_provider_model(path)
            self.assertEqual((provider, model), ("", ""))
        finally:
            _cleanup(path)


if __name__ == "__main__":
    unittest.main()
