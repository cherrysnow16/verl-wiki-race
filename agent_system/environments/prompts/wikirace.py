__all__ = ["WIKIRACE_SYSTEM_PROMPT", "WIKIRACE_TEMPLATE_NO_HIS", "WIKIRACE_TEMPLATE"]

# Mirrors LLM_wikirace/prompts.py (system message + game_prompt) verbatim.
WIKIRACE_SYSTEM_PROMPT = "You are a helpful assistant helping play the Wikipedia link game."

WIKIRACE_TEMPLATE_NO_HIS = """
You are playing a game where you start at Wikipedia page "{current_page}" and want to reach page "{target_page}" by clicking links.
So far, you have visited the following pages in order:

{current_page}

You see the following possible links from the current page:

{choices_list}

Which link should you click to get closer to the target? Reply with the number of your choice (0 to {max_choice_num}).
"""

WIKIRACE_TEMPLATE = """
You are playing a game where you start at Wikipedia page "{current_page}" and want to reach page "{target_page}" by clicking links.
So far, you have visited the following pages in order:

{history}

You see the following possible links from the current page:

{choices_list}

Which link should you click to get closer to the target? Reply with the number of your choice (0 to {max_choice_num}).
"""
