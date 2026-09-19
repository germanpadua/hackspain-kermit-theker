"""Comparative evaluation for the orderpick demo (H6).

Runs seeded episodes across scenarios and perception modes and reports
every episode — no cherry-picking. The evaluator reads simulator truth;
the controller never does (vision mode perceives through the wrist camera;
oracle is the labeled debug baseline).

  MUJOCO_GL=egl .venv/bin/python -m orderpick.evaluate \
      --seeds 7:8 --modes oracle,vision --scenarios nominal --output runs/eval

Metrics per episode: fulfilled, delivered, verified content, exact-match,
misses/retries/shortfalls, drops, sim vs wall time, failure reason.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from warehouse.persistence import RunLog

from .controller import Controller
from .demo import _slot_map
from .sim import CellSim

SCENARIOS = ["nominal", "reserve_empty", "obs_glitch", "obs_occluded",
             "park_blocked", "reception_blocked", "grasp_slip"]


def run_episode(seed, scenario, perception, retries):
    sim = CellSim(seed=seed, scenario=scenario)
    ctl = Controller(sim, perception=perception)
    ctl.max_retries = retries
    order = [(l["sku"], l["qty"]) for l in sim.catalog["order"]["lines"]]
    t0 = time.time()
    for _ in range(15):
        ctl.spin()
    fulfilled = ctl.fulfill(order, _slot_map(sim))
    delivered = ctl.deliver_tray() if (fulfilled or ctl._placed) else False
    wall = time.time() - t0

    kinds = Counter(e["kind"] for e in ctl.events)
    content = ctl._tray_contents()
    exact = delivered and all(
        content.get(s, 0) == q for s, q in order) and \
        sum(content.values()) == sum(q for _s, q in order)
    tray = sim.truth_body_pos("tray")
    rc = sim.config["reception"]
    on_table = (abs(tray[0] - rc["x_m"]) < rc["size_xy_m"][0] / 2 + 0.05
                and abs(tray[1] - rc["y_m"]) < rc["size_xy_m"][1] / 2 + 0.05
                and abs(tray[2] - rc["surface_z_m"]) < 0.12)
    # a piece lost = logged piece_lost or ended far from tray
    reason = None
    if not exact:
        for k in ("order_shortfall", "reserve_fail", "reception_unavailable",
                  "park_unavailable", "bin_unreadable", "cart_error",
                  "tray_approach_fail", "tray_slip"):
            if kinds.get(k):
                reason = k
                break
        else:
            reason = "incomplete" if not fulfilled else "not_delivered"
    ep = {
        "seed": seed, "scenario": scenario, "perception": perception,
        "retries": retries, "fulfilled": fulfilled, "delivered": delivered,
        "on_table": bool(on_table), "content": content, "exact": exact,
        "reason": reason, "sim_time_s": round(float(sim.data.time), 1),
        "wall_time_s": round(wall, 1),
        "pick_miss": kinds.get("pick_miss", 0),
        "place_miss": kinds.get("place_miss", 0),
        "piece_lost": kinds.get("piece_lost", 0),
        "rescued": kinds.get("unit_picked", 0)
        - sum(1 for e in ctl.events
              if e["kind"] == "unit_picked" and not e.get("rescued")),
        "obs_invalid": kinds.get("obs_invalid", 0),
        "replenished": kinds.get("reserve_advanced", 0),
        "events": ctl.events,
    }
    sim.close()
    return ep


def summarize(eps):
    n = len(eps)
    agg = {
        "episodes": n,
        "exact_deliveries": sum(1 for e in eps if e["exact"]),
        "fulfilled": sum(1 for e in eps if e["fulfilled"]),
        "delivered": sum(1 for e in eps if e["delivered"]),
        "false_success": sum(1 for e in eps if e["fulfilled"]
                             and not e["exact"]),
        "pick_miss": sum(e["pick_miss"] for e in eps),
        "place_miss": sum(e["place_miss"] for e in eps),
        "piece_lost": sum(e["piece_lost"] for e in eps),
        "rescued": sum(e["rescued"] for e in eps),
        "obs_invalid": sum(e["obs_invalid"] for e in eps),
        "replenished": sum(e["replenished"] for e in eps),
        "reasons": dict(Counter(e["reason"] for e in eps if e["reason"])),
        "sim_time_s": round(sum(e["sim_time_s"] for e in eps), 1),
        "wall_time_s": round(sum(e["wall_time_s"] for e in eps), 1),
    }
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="7")
    ap.add_argument("--modes", default="oracle,vision")
    ap.add_argument("--scenarios", default="nominal")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--no-retry", action="store_true",
                    help="vision without grasp retries (comparison arm)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    a, _, b = args.seeds.partition(":")
    seeds = list(range(int(a), int(b) + 1)) if b else [int(a)]
    modes = args.modes.split(",")
    scenarios = args.scenarios.split(",")

    run = RunLog(args.output or "runs",
                 {"demo": "orderpick-eval", "seeds": seeds,
                  "modes": modes, "scenarios": scenarios})
    out = run.path
    episodes = []
    for scenario in scenarios:
        for mode in modes:
            retries = 0 if (args.no_retry and mode == "vision") \
                else args.retries
            for seed in seeds:
                ep = run_episode(seed, scenario, mode, retries)
                ep.pop("events") if False else None
                print(f"[{scenario}/{mode}/s{seed}] exact={ep['exact']} "
                      f"deliv={ep['delivered']} reason={ep['reason']} "
                      f"wall={ep['wall_time_s']:.0f}s", flush=True)
                run.write_json(f"ep-{scenario}-{mode}-s{seed}.json", ep)
                episodes.append(ep)
    summary = {f"{s}+{m}": summarize([e for e in episodes
                                     if e["scenario"] == s
                                     and e["perception"] == m])
               for s in scenarios for m in modes}
    run.write_json("summary.json", summary)
    print(json.dumps(summary, indent=1))
    print("eval dir:", out)


if __name__ == "__main__":
    raise SystemExit(main())
