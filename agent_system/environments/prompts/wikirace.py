__all__ = ["WIKIRACE_TEMPLATE_NO_HIS", "WIKIRACE_TEMPLATE"]

WIKIRACE_TEMPLATE_NO_HIS = """You are playing the WikiRace game. Your goal is to navigate from the current Wikipedia page to the target page by following links.

Current page: {current_page}
Target page:  {target_page}

Available links from the current page:
{choices_list}

Now it is your turn. First reason step-by-step inside <think> </think> tags about which link is most likely to lead toward the target page. Then output your chosen link number inside <action> </action> tags.
"""

WIKIRACE_TEMPLATE = """You are playing the WikiRace game. Your goal is to navigate from the starting Wikipedia page to the target page by following links.

Target page:  {target_page}
Pages visited so far: {history}
Current page: {current_page}
Step: {step_count}

Available links from the current page:
{choices_list}

Now it is your turn. First reason step-by-step inside <think> </think> tags about which link is most likely to lead toward the target page. Then output your chosen link number inside <action> </action> tags.
"""
