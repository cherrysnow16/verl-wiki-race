from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import ray

VERL_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", default="easy", choices=("easy", "medium", "hard"))
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--max-choices", type=int, default=10)
    parser.add_argument("--max-path-length", type=int, default=30)
    parser.add_argument("--wikirace-src", type=Path, default=None)
    return parser.parse_args()


def add_import_paths(wikirace_src: "Path | None"):
    # verl-agent root so agent_system.* imports resolve.
    sys.path.insert(0, str(VERL_ROOT))

    # LLM_wikirace source: CLI flag > WIKIRACE_SRC env var > vendored engine.
    src = wikirace_src or os.environ.get("WIKIRACE_SRC")
    if not src:
        src = VERL_ROOT / "agent_system" / "environments" / "env_package" / "wikirace" / "vendor" / "src"
    src = str(Path(src).expanduser().resolve())
    sys.path.insert(0, src)
    return src


def main():
    args = parse_args()
    wikirace_src = add_import_paths(args.wikirace_src)

    from LLM_wikirace.data_utils import load_wikigraph
    from agent_system.environments.env_package.wikirace.envs import build_wikirace_envs
    from agent_system.environments.env_package.wikirace.projection import wikirace_projection

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=4, num_gpus=0)

    print("Loading WikiGraph...")
    wikigraph_ref = ray.put(load_wikigraph())

    env_kwargs = {
        "difficulty": args.difficulty,
        "max_choices": args.max_choices,
        "max_path_length": args.max_path_length,
        "wikigraph_ref": wikigraph_ref,
        "train_ratio": 0.8,
    }
    if wikirace_src:
        env_kwargs["wikirace_src"] = wikirace_src

    envs = build_wikirace_envs(
        seed=0,
        env_num=1,
        group_n=1,
        is_train=False,
        resources_per_worker={"num_cpus": 0.1},
        env_kwargs=env_kwargs,
    )

    for _ in range(args.episodes):
        print("\nReset")
        states, infos = envs.reset()
        for i, s in enumerate(states):
            print(f"[{i}] game={infos[i]['game_id']} '{s.current_page}' -> '{s.target_page}'")
            for j, title in enumerate(s.available_choices[:5]):
                print(f"    {j}: {title}")

        # Projection demo: a numeric answer (valid=1) and a no-number answer (valid=0).
        nb = states[0].available_choices
        actions, valids = wikirace_projection(["I choose 0", "no number here"], [nb, nb])
        print(f"\nprojection demo: actions={actions}, valids={valids}")

        print("\nStep")
        for t in range(args.steps):
            print(f"step {t + 1}")
            states, rewards, dones, infos = envs.step([0] * envs.num_processes)
            for i, s in enumerate(states):
                print(f"[{i}] reward={rewards[i]:.1f} done={dones[i]} "
                      f"status={infos[i]['status']} page='{s.current_page}'")

    envs.close()
    ray.shutdown()


if __name__ == "__main__":
    main()
