"""The build script's .env loader must strip quoted values and never
override already-set variables; the --groups flag must select the right
titles source and vector output per CPC tier."""
import importlib.util
import os
import tempfile
import unittest
from unittest.mock import patch


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


class TestBuildScriptGroups(unittest.TestCase):
    """--groups selects the CPC tier: titles source + vector output."""

    def setUp(self):
        self._module = _load_script_module()

    def _run_main(self, argv):
        with patch.object(self._module, "load_cpc_titles",
                          return_value=[{"code": "H05B45/20",
                                         "title": "Colour control"}]) \
                as mock_titles, \
             patch("sources.knowledge.knowledge.get_embeddings_batch",
                   return_value=[[1.0, 0.0]]), \
             patch("numpy.save") as mock_save, \
             patch.object(self._module.sys, "argv",
                          ["build_cpc_vectors.py"] + argv):
            rc = self._module.main()
        return rc, mock_titles, mock_save

    def test_groups_sub_reads_subgroup_titles(self):
        rc, mock_titles, mock_save = self._run_main(["--groups", "sub"])
        self.assertEqual(rc, 0)
        titles_path = mock_titles.call_args[0][0]
        self.assertTrue(titles_path.replace("\\", "/")
                        .endswith("cpc_titles_subgroups.json"))
        self.assertTrue(mock_save.call_args[0][0].replace("\\", "/")
                        .endswith("cpc_title_vectors_sub.npy"))

    def test_groups_main_keeps_main_tier(self):
        rc, mock_titles, mock_save = self._run_main(["--groups", "main"])
        self.assertEqual(rc, 0)
        titles_path = mock_titles.call_args[0][0]
        self.assertTrue(titles_path.replace("\\", "/")
                        .endswith("cpc_titles_main_groups.json"))
        self.assertTrue(mock_save.call_args[0][0].replace("\\", "/")
                        .endswith("cpc_title_vectors.npy"))

    def test_default_is_sub(self):
        rc, _, mock_save = self._run_main([])
        self.assertEqual(rc, 0)
        self.assertTrue(mock_save.call_args[0][0].replace("\\", "/")
                        .endswith("cpc_title_vectors_sub.npy"))

    def test_unknown_groups_rejected(self):
        with patch.object(self._module.sys, "argv",
                          ["build_cpc_vectors.py", "--groups", "nope"]):
            with self.assertRaises(SystemExit):
                self._module.main()


if __name__ == "__main__":
    unittest.main()
