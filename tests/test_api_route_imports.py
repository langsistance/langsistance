from pathlib import Path
import unittest


class TestApiRouteImports(unittest.TestCase):

    def test_api_routes_do_not_import_beyond_top_level_package(self):
        api_routes_dir = Path(__file__).resolve().parents[1] / "api_routes"
        offenders = []

        for path in api_routes_dir.glob("*.py"):
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if "from .." in line or "import .." in line:
                    offenders.append(f"{path.name}:{line_number}:{line.strip()}")

        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
