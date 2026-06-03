import re
from typing import List


def wikirace_projection(actions: List[str]):
    """
    A function to process the actions.
    actions: the list of actions to be processed, it is a list of strings.
    Expected format:
        <think>some reasoning...</think><action>3</action>
    WikiRace action: integer index into available_choices list (0-based).
    """
    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i]

        has_think = "<think>" in original_str and "</think>" in original_str

        # Attempt to extract the integer within <action>...</action>
        m = re.search(r'<action>\s*(\d+)\s*</action>', original_str, re.IGNORECASE)
        if m:
            actions[i] = int(m.group(1))
            valids[i] = 1 if has_think else 0
        else:
            # If no valid <action> tag, fall back to the first digit sequence found
            nums = re.findall(r'\d+', original_str)
            actions[i] = int(nums[0]) if nums else 0

    return actions, valids
