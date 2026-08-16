"""Logger resilience: an unwritable log directory must not crash."""
import logging
import unittest
from unittest.mock import patch


class TestLoggerFallback(unittest.TestCase):
    def test_file_handler_failure_falls_back_to_stderr(self):
        from sources.logger import Logger

        def _boom(*args, **kwargs):
            raise PermissionError("read-only log dir")

        with patch("logging.FileHandler", side_effect=_boom):
            lg = Logger("unwritable.log")
        # constructor must not raise
        self.assertTrue(lg.enabled)
        self.assertIsNotNone(lg.logger)
        # logging still works through the fallback stream handler
        lg.log("still works")
        self.assertTrue(any(
            isinstance(h, logging.StreamHandler) for h in lg.logger.handlers))


if __name__ == "__main__":
    unittest.main()
