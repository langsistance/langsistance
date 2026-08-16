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


class TestEmbedToMemmap(unittest.TestCase):
    """The streaming build must write batches into a memmap as they
    arrive (.tmp file + atomic rename) — accumulating the sub tier's
    ~150k x 1024 embeddings as Python float lists needs ~5GB and gets
    the process OOM-killed."""

    def setUp(self):
        self._module = _load_script_module()

    def test_streams_batches_into_final_npy(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "vec.npy")

            def fake_embed(chunk):
                return [[int(t[1]), 0.5] for t in chunk]

            rows, dim = self._module._embed_to_memmap(
                ["t0", "t1", "t2", "t3", "t4"], target, batch=2,
                float16=False, get_embeddings=fake_embed)
            arr = np.load(target)
        self.assertEqual((rows, dim), (5, 2))
        self.assertEqual(arr.shape, (5, 2))
        self.assertEqual(arr.tolist(),
                         [[0.0, 0.5], [1.0, 0.5], [2.0, 0.5],
                          [3.0, 0.5], [4.0, 0.5]])
        self.assertFalse(os.path.exists(target + ".tmp"))

    def test_writes_batches_as_they_arrive(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "vec.npy")
            state = {"calls": 0, "tmp_seen": []}

            def fake_embed(chunk):
                if state["calls"] == 1:  # second batch: first is on disk
                    state["tmp_seen"].append(
                        os.path.exists(target + ".tmp"))
                state["calls"] += 1
                return [[1.0, 2.0] for _ in chunk]

            self._module._embed_to_memmap(
                ["t0", "t1", "t2", "t3"], target, batch=1,
                float16=False, get_embeddings=fake_embed)
        # the memmap file existed before the last batch was embedded —
        # proof that nothing accumulates in RAM
        self.assertEqual(state["tmp_seen"], [True])

    def test_failure_keeps_existing_target_intact(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "vec.npy")
            np.save(target, np.zeros((1, 2)))

            def boom(chunk):
                raise RuntimeError("api down")

            with self.assertRaises(RuntimeError):
                self._module._embed_to_memmap(
                    ["t0", "t1"], target, batch=1, float16=False,
                    get_embeddings=boom)
            arr = np.load(target)
            self.assertEqual(arr.tolist(), [[0.0, 0.0]])
            self.assertFalse(os.path.exists(target + ".tmp"))


class TestBuildScriptGroups(unittest.TestCase):
    """--groups selects the CPC tier: titles source + vector output."""

    def setUp(self):
        self._module = _load_script_module()

    def _run_main(self, argv):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            titles_path = os.path.join(tmp, "titles.json")
            vectors_path = os.path.join(tmp, "vectors.npy")
            with patch.object(self._module, "cpc_paths_for_level",
                              return_value=(titles_path, vectors_path)) \
                    as mock_paths, \
                 patch.object(self._module, "load_cpc_titles",
                              return_value=[
                                  {"code": "H05B45/20", "title": "Colour"},
                                  {"code": "H05B45/21", "title": "Temp"}]), \
                 patch("sources.knowledge.knowledge.get_embeddings_batch",
                       return_value=[[1.0, 0.0], [0.0, 1.0]]), \
                 patch.object(self._module.sys, "argv",
                              ["build_cpc_vectors.py"] + argv):
                rc = self._module.main()
            arr = np.load(vectors_path)
        return rc, mock_paths, arr

    def test_groups_sub_selects_sub_tier(self):
        rc, mock_paths, arr = self._run_main(["--groups", "sub"])
        self.assertEqual(rc, 0)
        self.assertEqual(mock_paths.call_args[0][0], "sub")
        self.assertEqual(arr.tolist(), [[1.0, 0.0], [0.0, 1.0]])

    def test_groups_main_selects_main_tier(self):
        rc, mock_paths, arr = self._run_main(["--groups", "main"])
        self.assertEqual(rc, 0)
        self.assertEqual(mock_paths.call_args[0][0], "main")
        self.assertEqual(arr.tolist(), [[1.0, 0.0], [0.0, 1.0]])

    def test_default_is_sub(self):
        rc, mock_paths, _ = self._run_main([])
        self.assertEqual(rc, 0)
        self.assertEqual(mock_paths.call_args[0][0], "sub")

    def test_unknown_groups_rejected(self):
        with patch.object(self._module.sys, "argv",
                          ["build_cpc_vectors.py", "--groups", "nope"]):
            with self.assertRaises(SystemExit):
                self._module.main()


if __name__ == "__main__":
    unittest.main()
