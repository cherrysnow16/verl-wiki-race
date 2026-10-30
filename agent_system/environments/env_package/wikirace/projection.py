import re
from typing import List, Optional, Sequence


_INTEGER_RE = re.compile(r"\b\d+\b")


def _parse_one(text: str, neighbors: Optional[Sequence[str]]):
    """Parse a single response into (action, valid), matching the original
    ``LLM_wikirace.parse_answer`` when ``neighbors`` is available.

    Order:
      1. last integer that is a valid index into ``neighbors``  -> valid=1
      2. last neighbor title mentioned in the response          -> valid=0
      3. fallback: first neighbor (0)                           -> valid=0

    When ``neighbors`` is None (no choice context, e.g. standalone demos),
    the last integer is used and no range/title check is done.
    """
    text = str(text)
    numbers = _INTEGER_RE.findall(text)

    if neighbors is None:
        if numbers:
            return int(numbers[-1]), 1
        return 0, 0

    n = len(neighbors)

    # 1. last integer inside the valid range
    for num in reversed(numbers):
        choice = int(num)
        if 0 <= choice < n:
            return choice, 1

    # 2. last neighbor title mentioned in the response
    mentioned = sorted(
        (nb for nb in neighbors if nb in text),
        key=lambda nb: text.index(nb),
    )
    if mentioned:
        return neighbors.index(mentioned[-1]), 0

    # 3. fallback: first neighbor
    return 0, 0


def wikirace_projection(actions: List[str], neighbors_list=None):
    """Convert model responses into WikiRace link indices.

    Mirrors ``LLM_wikirace.parse_answer``: use the last integer that is a valid
    index; otherwise the last neighbor title mentioned in the response;
    otherwise 0. Pass ``neighbors_list`` (per-item list of available choices)
    to enable the range/title checks; without it the last integer is used.

    Returns ``(actions, valids)``. ``valid`` is 1 only when a numeric in-range
    choice was parsed; the title/zero fallbacks are marked 0 so they feed the
    invalid-action penalty.
    """
    parsed_actions = []
    valids = []
    for i, text in enumerate(actions):
        neighbors = None if neighbors_list is None else neighbors_list[i]
        action, valid = _parse_one(text, neighbors)
        parsed_actions.append(action)
        valids.append(valid)

    return parsed_actions, valids
