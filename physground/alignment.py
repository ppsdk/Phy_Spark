from __future__ import annotations

from collections.abc import Sequence


def common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    """Return the number of identical leading token ids in two sequences."""
    length = 0
    for left_id, right_id in zip(left, right):
        if int(left_id) != int(right_id):
            break
        length += 1
    return length


def answer_boundary(prompt_ids: Sequence[int], full_ids: Sequence[int]) -> int:
    """Locate the first answer-side token shared safely across chat templates.

    ``prompt_ids`` is rendered with ``add_generation_prompt=True`` and
    ``full_ids`` includes the assistant answer. Most templates make the former a
    strict prefix of the latter. For remote-code templates that render the
    assistant header differently, the longest common prefix remains the only
    causal region guaranteed not to contain answer tokens.
    """
    boundary = common_prefix_length(prompt_ids, full_ids)
    if boundary <= 0:
        raise ValueError("Prompt and full chat encodings have no common token prefix")
    if boundary >= len(full_ids):
        raise ValueError("Full chat encoding does not contain answer-side tokens")
    return boundary
