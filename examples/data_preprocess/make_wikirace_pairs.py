"""Generate WikiRace train/val (start, target) pairs disjoint from the benchmark eval splits.

- Every page that appears in easy/medium/hard (as start or target) is excluded from both roles.
- Pairs are stratified by shortest-path distance (default 2..8).
- Val targets are disjoint from train targets.
- Degree floors mirror the benchmark: start out-degree >= min over benchmark starts,
  target in-degree >= min over benchmark targets.

Outputs (benchmark CSV format "0,1" header + start,target), plus *_meta.csv with distance and titles:
  vendor/data/splits/train_pairs.csv, val_pairs.csv
"""
import argparse
import csv
import os
import random
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

VENDOR = os.path.join(os.path.dirname(__file__), "..", "..", "agent_system", "environments",
                      "env_package", "wikirace", "vendor")
VENDOR = os.path.abspath(VENDOR)
sys.path.insert(0, os.path.join(VENDOR, "src"))
from LLM_wikirace.data_utils import load_wikigraph, load_test_pages  # noqa: E402

G = None  # shared with forked workers
START_OK = None


def starts_by_distance(args):
    """BFS to one target; return {distance: random eligible start at that distance}."""
    target, dists, seed = args
    d = G.get_shortest_path_arrays(target)["dist_matrix"][0]
    rng = np.random.default_rng(seed)
    out = {}
    for k in dists:
        cand = np.flatnonzero((d == k) & START_OK)
        if cand.size:
            out[k] = int(rng.choice(cand))
    return target, out


def main():
    global G, START_OK
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_dist", type=int, default=2)
    ap.add_argument("--max_dist", type=int, default=8)
    ap.add_argument("--train_per_dist", type=int, default=300)
    ap.add_argument("--val_per_dist", type=int, default=30)
    ap.add_argument("--pairs_per_target", type=int, default=2)
    ap.add_argument("--max_targets", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=os.path.join(VENDOR, "data", "splits"))
    args = ap.parse_args()

    t0 = time.time()
    G = load_wikigraph(os.path.join(VENDOR, "data", "wikidata"))
    n = len(G.row_ptr) - 1
    out_deg = np.diff(G.row_ptr)
    in_deg = np.bincount(G.col_indices, minlength=n)

    eval_pairs = []
    for sp in ["easy", "medium", "hard"]:
        eval_pairs += [(int(s), int(t)) for s, t in load_test_pages(os.path.join(VENDOR, "data", "splits", f"{sp}.csv"))]
    eval_nodes = {x for p in eval_pairs for x in p}
    min_start_out = int(min(out_deg[s] for s, _ in eval_pairs))
    min_target_in = int(min(in_deg[t] for _, t in eval_pairs))
    print(f"graph n={n} loaded {time.time() - t0:.0f}s | eval pairs={len(eval_pairs)} nodes={len(eval_nodes)} "
          f"| floors: start out-deg>={min_start_out}, target in-deg>={min_target_in}", flush=True)

    # Node ids index a sparse id space; only ids with a title are real pages.
    titled = np.zeros(n, dtype=bool)
    titled[np.fromiter((int(k) for k in G.id_to_title), dtype=np.int64)] = True
    banned = np.zeros(n, dtype=bool)
    banned[list(eval_nodes)] = True
    START_OK = (out_deg >= max(1, min_start_out)) & titled & ~banned
    target_pool = np.flatnonzero((in_deg >= max(1, min_target_in)) & (out_deg >= 1) & titled & ~banned)
    print(f"eligible starts={int(START_OK.sum())}, eligible targets={len(target_pool)}", flush=True)
    rng = random.Random(args.seed)
    targets = rng.sample(target_pool.tolist(), min(args.max_targets, len(target_pool)))

    dists = list(range(args.min_dist, args.max_dist + 1))
    quota = {"val": {k: args.val_per_dist for k in dists}, "train": {k: args.train_per_dist for k in dists}}
    chosen = {"val": [], "train": []}
    used_starts = set()
    done = lambda: all(v == 0 for q in quota.values() for v in q.values())

    with Pool(args.workers) as pool:
        jobs = ((t, dists, args.seed + i) for i, t in enumerate(targets))
        for i, (target, by_d) in enumerate(pool.imap_unordered(starts_by_distance, jobs, chunksize=4)):
            # Fill val first so val targets are a separate set; a target goes wholly to one side.
            for side in ("val", "train"):
                need = sorted((k for k in by_d if quota[side][k] > 0 and by_d[k] not in used_starts),
                              key=lambda k: -quota[side][k] / max(1, args.train_per_dist if side == "train" else args.val_per_dist))
                if need:
                    for k in need[: args.pairs_per_target]:
                        chosen[side].append((by_d[k], target, k))
                        used_starts.add(by_d[k])
                        quota[side][k] -= 1
                    break
            if (i + 1) % 500 == 0:
                print(f"{i + 1} targets {time.time() - t0:.0f}s remaining={ {s: sum(q.values()) for s, q in quota.items()} }", flush=True)
            if done():
                pool.terminate()
                break

    for side in ("train", "val"):
        rows = chosen[side]
        rng.shuffle(rows)
        with open(os.path.join(args.out_dir, f"{side}_pairs.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([0, 1])
            w.writerows((s, t) for s, t, _ in rows)
        with open(os.path.join(args.out_dir, f"{side}_pairs_meta.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["start", "target", "dist", "start_title", "target_title"])
            w.writerows((s, t, k, G.get_node_title(s), G.get_node_title(t)) for s, t, k in rows)
        print(f"{side}: {len(rows)} pairs, by distance {dict(sorted(Counter(k for *_, k in rows).items()))}, "
              f"unique targets {len({t for _, t, _ in rows})}", flush=True)

    tr_t = {t for _, t, _ in chosen["train"]}
    va_t = {t for _, t, _ in chosen["val"]}
    all_nodes = {x for side in chosen.values() for s, t, _ in side for x in (s, t)}
    print(f"checks: train/val target overlap={len(tr_t & va_t)}, overlap with eval nodes={len(all_nodes & eval_nodes)}, "
          f"unfilled quota={ {s: {k: v for k, v in q.items() if v} for s, q in quota.items()} } total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
