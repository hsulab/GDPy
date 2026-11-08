import json
import tempfile
import unittest
from pathlib import Path

from gdpx.utils.parser import parse_input_file


class InputParserTests(unittest.TestCase):
    def test_missing_explicit_input_reports_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            for suffix in (".yaml", ".json"):
                missing = Path(directory) / ("missing" + suffix)
                for source in (missing, str(missing)):
                    with self.subTest(source=source):
                        with self.assertRaises(FileNotFoundError) as exc:
                            parse_input_file(source)
                        self.assertEqual(exc.exception.filename, str(missing))

    def test_nested_literals_and_existing_includes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            included = root / "included.json"
            included.write_text('{"steps": 200}')
            missing = str(root / "future-output.json")
            config = {"provider": "vasp", "output": missing,
                      "include": str(included), "items": [missing, str(included)]}
            expected = {"provider": "vasp", "output": missing,
                        "include": {"steps": 200}, "items": [missing, {"steps": 200}]}
            for suffix in (".yaml", ".json"):
                with self.subTest(suffix=suffix):
                    source = root / ("runtime" + suffix)
                    source.write_text(json.dumps(config))
                    self.assertEqual(parse_input_file(source), expected)

    def test_in_memory_configuration_keeps_literal_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            config = [{"output": str(Path(directory) / "future.yaml")}]
            self.assertEqual(parse_input_file(config), config)
