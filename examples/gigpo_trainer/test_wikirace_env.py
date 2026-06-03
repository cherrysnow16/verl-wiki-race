"""
Lightweight env-only test for WikiRace integration.
No GPU required. Validates envs.py, projection.py, and env_manager plumbing.

Run from verl-agent root:
    python examples/gigpo_trainer/test_wikirace_env.py
"""

import sys
import os
import ray

WIKIRACE_SRC = "/workspace/wiki-run/wikirace/src"
if WIKIRACE_SRC not in sys.path:
    sys.path.insert(0, WIKIRACE_SRC)

sys.path.insert(0, "/workspace/wiki-run/verl-agent")

# ---------------------------------------------------------------------------
# 1. projection (no Ray, no graph)
# ---------------------------------------------------------------------------
from agent_system.environments.env_package.wikirace.projection import wikirace_projection

def test_projection():
    cases = [
        ("<think>go to X</think><action>2</action>",  2, 1),
        ("<action>5</action>",                         5, 0),  # no think
        ("no tags at all",                             0, 0),
        ("<think>t</think><action>  10  </action>",   10, 1),
        ("<think>t</think><action>abc</action>",        0, 0),  # non-digit
    ]
    actions_in = [c[0] for c in cases]
    actions_out, valids = wikirace_projection(actions_in)
    for i, (_, exp_a, exp_v) in enumerate(cases):
        assert actions_out[i] == exp_a, f"case {i}: action {actions_out[i]} != {exp_a}"
        assert valids[i]      == exp_v, f"case {i}: valid  {valids[i]}  != {exp_v}"
    print("  [OK] projection")

# ---------------------------------------------------------------------------
# 2. WikiRaceMultiProcessEnv  (2 envs, group_n=1)
# ---------------------------------------------------------------------------
from agent_system.environments.env_package.wikirace.envs import build_wikirace_envs
from LLM_wikirace.data_utils import load_wikigraph

def test_envs(wikigraph_ref, splits_path):
    env_kwargs = {
        "difficulty":     "easy",
        "max_choices":    10,
        "max_path_length": 5,
        "wikirace_src":   WIKIRACE_SRC,
        "wikigraph_ref":  wikigraph_ref,
        "custom_splits_path": splits_path,
        "train_ratio":    0.8,
    }
    envs = build_wikirace_envs(
        seed=42, env_num=2, group_n=1,
        resources_per_worker={"num_cpus": 0.1},
        is_train=False,
        env_kwargs=env_kwargs,
    )

    # reset
    state_list, info_list = envs.reset()
    assert len(state_list) == 2
    for s, info in zip(state_list, info_list):
        assert s.status == "in progress"
        assert len(s.available_choices) > 0
        assert not info["won"]
    print(f"  [OK] reset  — pages: {[s.current_page for s in state_list]}")
    print(f"               targets: {[s.target_page  for s in state_list]}")

    # step (always pick choice 0)
    actions = [0, 0]
    state_list2, rewards, dones, infos = envs.step(actions)
    assert len(state_list2) == 2
    for r, d in zip(rewards, dones):
        assert isinstance(r, float)
        assert isinstance(d, bool)
    print(f"  [OK] step   — rewards: {rewards}  dones: {dones}")
    print(f"               pages after step: {[s.current_page for s in state_list2]}")

    envs.close()
    print("  [OK] close")

# ---------------------------------------------------------------------------
# 3. prompt templates
# ---------------------------------------------------------------------------
from agent_system.environments.prompts.wikirace import (
    WIKIRACE_TEMPLATE_NO_HIS, WIKIRACE_TEMPLATE,
)

def test_prompts(state):
    choices_list = "\n".join(f"{i}: {t}" for i, t in enumerate(state.available_choices))
    obs_no_his = WIKIRACE_TEMPLATE_NO_HIS.format(
        current_page=state.current_page,
        target_page=state.target_page,
        choices_list=choices_list,
    )
    obs_his = WIKIRACE_TEMPLATE.format(
        current_page=state.current_page,
        target_page=state.target_page,
        history=" -> ".join(state.history),
        step_count=state.step_number,
        choices_list=choices_list,
    )
    assert state.current_page in obs_no_his
    assert state.target_page  in obs_no_his
    assert state.current_page in obs_his
    assert state.target_page in obs_his
    print("  [OK] prompt templates")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== WikiRace env-only test ===\n")

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=4, num_gpus=0)

    # Load WikiGraph once
    print("Loading WikiGraph...")
    wg = load_wikigraph()
    wikigraph_ref = ray.put(wg)
    del wg
    print("WikiGraph ready.\n")

    splits_path = "/workspace/wiki-run/wikirace/data/splits/easy.csv"

    print("1. Projection")
    test_projection()

    print("2. Env (reset + step + close)")
    test_envs(wikigraph_ref, splits_path)

    print("3. Prompt templates")
    # build one state for prompt test
    from agent_system.environments.env_package.wikirace.envs import build_wikirace_envs
    env_kwargs2 = {
        "difficulty": "easy", "max_choices": 10, "max_path_length": 5,
        "wikirace_src": WIKIRACE_SRC, "wikigraph_ref": wikigraph_ref,
        "custom_splits_path": splits_path, "train_ratio": 0.8,
    }
    _e = build_wikirace_envs(seed=0, env_num=1, group_n=1,
                              resources_per_worker={"num_cpus": 0.1},
                              is_train=False, env_kwargs=env_kwargs2)
    states, _ = _e.reset()
    test_prompts(states[0])
    _e.close()

    ray.shutdown()
    print("\n=== All tests passed ===\n")
