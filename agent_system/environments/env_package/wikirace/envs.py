import os
import sys
import gymnasium as gym
import ray
import numpy as np


class WikiRaceWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds its own WikiRaceEngine instance and the full splits table
    so that reset(game_id) can switch games without recreating the actor.
    """

    def __init__(self, wikigraph_ref, game_id, difficulty, seed,
                 max_choices, max_path_length, splits_path, wikirace_src):
        # Lazy import: wikirace lives outside the verl-agent package tree.
        if wikirace_src not in sys.path:
            sys.path.insert(0, wikirace_src)

        from LLM_wikirace.game_engine import WikiRaceEngine
        from LLM_wikirace.data_utils import load_test_pages

        # Ray auto-resolves ObjectRef arguments, so wikigraph_ref is already
        # the WikiGraph object when __init__ is called.
        self.game_engine = WikiRaceEngine(
            wikigraph_ref,
            max_choices=max_choices,
            max_path_length=max_path_length,
            game_seed=seed,
        )
        self.test_pages = load_test_pages(splits_path)
        self.current_state = None
        self._set_pages(game_id)

    def _set_pages(self, game_id):
        self.game_id = game_id
        self.start_page = self.test_pages[game_id][0]
        self.target_page = self.test_pages[game_id][1]

    def reset(self, game_id=None):
        """Reset the environment, optionally switching to a different game."""
        if game_id is not None:
            self._set_pages(game_id)
        self.current_state = self.game_engine.initialize_game(
            inference_engine_name="verl-agent",
            current_page=self.start_page,
            target_page=self.target_page,
            game_id=self.game_id,
        )
        return self.current_state, self._make_info()

    def step(self, action):
        """Execute a step in the environment"""
        if self.current_state.status != "in progress":
            return self.current_state, 0.0, True, self._make_info()

        # Clamp out-of-range index to 0 rather than crashing.
        n = len(self.current_state.available_choices)
        if not (0 <= action < n):
            action = 0

        self.current_state = self.game_engine.step(self.current_state, action)
        done = self.current_state.status in ("completed", "failed")
        reward = 10.0 if self.current_state.status == "completed" else 0.0
        return self.current_state, reward, done, self._make_info()

    def _make_info(self):
        s = self.current_state
        return {
            "won":         s.status == "completed",
            "game_id":     s.game_id,
            "status":      s.status,
            "step_number": s.step_number,
            "num_choices": len(s.available_choices),
        }


class WikiRaceMultiProcessEnv(gym.Env):
    """
    Ray-based parallel environment wrapper for WikiRace.
    - env_num: Number of distinct games per rollout batch
    - group_n: Number of replicas per game (for GRPO and GiGPO)
    - env_kwargs: Parameters for configuring WikiRace (see build_wikirace_envs)
    """

    def __init__(self,
                 seed=0,
                 env_num=1,
                 group_n=1,
                 resources_per_worker={"num_cpus": 0.1},
                 is_train=True,
                 env_kwargs=None):
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        if env_kwargs is None:
            env_kwargs = {}

        self.env_num = env_num
        self.group_n = group_n
        self.num_processes = env_num * group_n
        self._rng = np.random.RandomState(seed)

        difficulty      = env_kwargs.get("difficulty", "easy")
        max_choices     = env_kwargs.get("max_choices", 50)
        max_path_length = env_kwargs.get("max_path_length", 30)
        wikirace_src    = env_kwargs.get("wikirace_src", "/workspace/wiki-run/wikirace/src")
        wikigraph_path  = env_kwargs.get("wikigraph_path", None)
        train_ratio     = env_kwargs.get("train_ratio", 0.8)

        custom_splits = env_kwargs.get("custom_splits_path", None)
        if custom_splits:
            splits_path = custom_splits
        else:
            wikirace_root = os.path.dirname(wikirace_src)
            splits_path = os.path.join(wikirace_root, "data", "splits", f"{difficulty}.csv")

        # Lazy import of wikirace modules
        if wikirace_src not in sys.path:
            sys.path.insert(0, wikirace_src)
        from LLM_wikirace.data_utils import load_wikigraph
        from LLM_wikirace.wikirace_env import get_possible_game_ids

        # Accept a pre-loaded ref so train and val envs can share one WikiGraph.
        if env_kwargs.get("wikigraph_ref") is not None:
            wikigraph_ref = env_kwargs["wikigraph_ref"]
        else:
            wikigraph_ref = ray.put(load_wikigraph(wikigraph_path))

        # Split game IDs into train / val pools
        all_ids = get_possible_game_ids(difficulty)
        split_idx = int(len(all_ids) * train_ratio)
        self.game_ids = all_ids[:split_idx] if is_train else all_ids[split_idx:]

        # Workers need wikirace_src + verl-agent root on PYTHONPATH so that
        # Ray can deserialise the WikiGraph ObjectRef argument before __init__ runs.
        extra_paths = [wikirace_src] + [
            p for p in sys.path if p and p not in ("", ".", wikirace_src)
        ]
        existing_pypath = os.environ.get("PYTHONPATH", "")
        if existing_pypath:
            extra_paths.append(existing_pypath)

        # Create Ray remote actors instead of processes
        env_worker = ray.remote(
            **resources_per_worker,
            runtime_env={"env_vars": {"PYTHONPATH": ":".join(extra_paths)}},
        )(WikiRaceWorker)
        self.workers = []
        for i in range(self.num_processes):
            init_gid = self.game_ids[i % len(self.game_ids)]
            worker = env_worker.remote(
                wikigraph_ref, init_gid, difficulty,
                seed + i, max_choices, max_path_length,
                splits_path, wikirace_src,
            )
            self.workers.append(worker)

    def step(self, actions):
        """
        Perform step in parallel.
        :param actions: list of int, length must equal self.num_processes
        :return: state_list, reward_list, done_list, info_list
        """
        assert len(actions) == self.num_processes

        # Send step commands to all workers
        futures = [w.step.remote(a) for w, a in zip(self.workers, actions)]

        # Collect results
        results = ray.get(futures)
        state_list, reward_list, done_list, info_list = [], [], [], []
        for state, reward, done, info in results:
            state_list.append(state)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
        return state_list, reward_list, done_list, info_list

    def reset(self):
        """
        Perform reset in parallel.
        Samples env_num game IDs and repeats each group_n times for GRPO/GiGPO.
        :return: (state_list, info_list)
        """
        replace = self.env_num > len(self.game_ids)
        game_ids = self._rng.choice(self.game_ids, size=self.env_num, replace=replace)

        # Repeat game_id for environments in the same group
        game_ids = np.repeat(game_ids, self.group_n).tolist()

        # Send reset commands to all workers
        futures = [w.reset.remote(gid) for w, gid in zip(self.workers, game_ids)]

        # Collect results
        results = ray.get(futures)
        state_list, info_list = [], []
        for state, info in results:
            state_list.append(state)
            info_list.append(info)
        return state_list, info_list

    def close(self):
        """Close all Ray actors."""
        try:
            if hasattr(self, "workers") and ray.is_initialized():
                for worker in self.workers:
                    ray.kill(worker)
        except Exception:
            pass

    def __del__(self):
        self.close()


def build_wikirace_envs(seed=0,
                        env_num=1,
                        group_n=1,
                        resources_per_worker={"num_cpus": 0.1},
                        is_train=True,
                        env_kwargs=None):
    """
    Externally exposed constructor function to create parallel WikiRace environments.
    - seed: Random seed for game sampling
    - env_num: Number of distinct games per batch
    - group_n: Number of environment replicas under the same game (for GRPO and GiGPO)
    - is_train: Determines the game ID pool used (train/val split)
    - env_kwargs: {difficulty, max_choices, max_path_length, wikirace_src,
                   wikigraph_path, wikigraph_ref, train_ratio, custom_splits_path}
    """
    return WikiRaceMultiProcessEnv(
        seed, env_num, group_n, resources_per_worker, is_train, env_kwargs
    )
