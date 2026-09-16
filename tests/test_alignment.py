from __future__ import annotations

import unittest

from physground.alignment import answer_boundary, common_prefix_length


class AlignmentTests(unittest.TestCase):
    def test_common_prefix_length(self) -> None:
        self.assertEqual(common_prefix_length([1, 2, 3], [1, 2, 4, 5]), 2)

    def test_answer_boundary_for_strict_prefix(self) -> None:
        self.assertEqual(answer_boundary([1, 2, 3], [1, 2, 3, 4]), 3)

    def test_answer_boundary_handles_template_divergence(self) -> None:
        self.assertEqual(answer_boundary([1, 2, 9], [1, 2, 8, 4]), 2)

    def test_rejects_no_shared_prefix(self) -> None:
        with self.assertRaisesRegex(ValueError, "no common"):
            answer_boundary([1], [2, 3])

    def test_rejects_full_encoding_without_answer_tokens(self) -> None:
        with self.assertRaisesRegex(ValueError, "answer-side"):
            answer_boundary([1, 2], [1, 2])


if __name__ == "__main__":
    unittest.main()
