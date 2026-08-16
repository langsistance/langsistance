"""The build script's .env loader must strip quoted values and never
override already-set variables."""
import importlib.util
import os
import tempfile
import unittest


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "build_cpc_vectors_script", "scripts/build_cpc_vectors.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestBuildScriptEnvLoader(unittest.TestCase):
    def setUp(self):
        self._module = _load_script_module()
        os.environ.pop("_CPC_TEST_QUOTED_KEY", None)
        os.environ.pop("_CPC_TEST_DQ_KEY", None)
        os.environ.pop("_CPC_TEST_PLAIN_KEY", None)

    def tearDown(self):
        for key in ("_CPC_TEST_QUOTED_KEY", "_CPC_TEST_DQ_KEY",
                    "_CPC_TEST_PLAIN_KEY"):
            os.environ.pop(key, None)

    def test_strips_single_and_double_quotes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, ".env")
            with open(path, "w", encoding="utf-8") as f:
                f.write("_CPC_TEST_QUOTED_KEY='sk-abc123'\n"
                        '_CPC_TEST_DQ_KEY="https://x/v1"\n'
                        "_CPC_TEST_PLAIN_KEY=plain\n")
            self._module._load_env(path)
        self.assertEqual(os.environ["_CPC_TEST_QUOTED_KEY"], "sk-abc123")
        self.assertEqual(os.environ["_CPC_TEST_DQ_KEY"], "https://x/v1")
        self.assertEqual(os.environ["_CPC_TEST_PLAIN_KEY"], "plain")

    def test_never_overrides_existing_values(self):
        os.environ["_CPC_TEST_QUOTED_KEY"] = "existing"
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, ".env")
            with open(path, "w", encoding="utf-8") as f:
                f.write("_CPC_TEST_QUOTED_KEY='from-file'\n")
            self._module._load_env(path)
        self.assertEqual(os.environ["_CPC_TEST_QUOTED_KEY"], "existing")


if __name__ == "__main__":
    unittest.main()
