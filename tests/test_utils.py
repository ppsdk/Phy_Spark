from __future__ import annotations

import unittest

from physground.utils import nested_get, parse_binary, parse_choice


class ParserTests(unittest.TestCase):
    def test_choice_parser_requires_an_option(self) -> None:
        self.assertEqual(parse_choice("Final answer: C"), "C")
        self.assertIsNone(parse_choice("Because the object accelerates"))

    def test_binary_parser_uses_terminal_answer(self) -> None:
        self.assertEqual(parse_binary("Reasoning. Final answer: No"), 0)
        self.assertEqual(parse_binary("plausible"), 1)
        self.assertIsNone(parse_binary("uncertain"))

    def test_nested_get(self) -> None:
        value = nested_get({"outer": [{"target": 7}]}, ["target"])
        self.assertEqual(value, 7)


if __name__ == "__main__":
    unittest.main()
