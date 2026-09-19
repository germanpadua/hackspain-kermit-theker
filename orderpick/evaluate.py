"""Comparative evaluation for the orderpick demo (H6).

Runs seeded episodes across scenarios and perception modes and reports
every episode — no cherry-picking. Product vision and oracle-assisted
logistics are labeled separately. Final verification is independent.

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
from .contracts import Phase
from .orders import load_recipe, perception_scope, slot_map
from .scene import load_config
from .sim import SCENARIOS, CellSim
from .verification import assess_delivery, verify_delivery


def run_episode(seed, scenario, perception, retries, recipe_path=None):
    sim = CellSim(seed=seed, scenario=scenario, recipe_path=recipe_path)
    try:
        return _run_episode(sim, perception, retries, recipe_path)
    finally:
        sim.close()


def _run_episode(sim, perception, retries, recipe_path):
    ctl = Controller(sim, perception=perception)
    ctl.max_retries = retries
    recipe = sim.recipe
    order = recipe.lines
    t0 = time.monotonic()
    for _ in range(15):
        ctl.spin()
    fulfilled = ctl.fulfill(order, slot_map(sim))
    if not ctl.blocked:
        ctl.spin(50)
        if assess_delivery(sim, order).kit_exact:
            ctl.set_state(Phase.KIT_PREPARED)
    delivered = ctl.deliver_tray() if not ctl.blocked and fulfilled else False
    assessment = verify_delivery(sim, order)
    ctl.set_state(Phase.KIT_READY if assessment.exact else (
        Phase.PERCEPTION_STOP if ctl.blocked else Phase.ORDER_INCOMPLETE))
    wall = time.monotonic() - t0
    kinds = Counter(e["kind"] for e in ctl.events)
    states = Counter(e["state"] for e in ctl.events if e["kind"] == "process_state")
    exact = assessment.exact
    reason = None
    if not exact:
        for k in ("pick_unverified", "order_shortfall", "reserve_fail", "reception_unavailable",
                  "park_unavailable", "bin_unreadable", "cart_error",
                  "tray_approach_fail", "tray_slip"):
            if kinds.get(k):
                reason = k
                break
        else:
            reason = ("wrong_compartment" if assessment.misplaced_units else
                      "incomplete" if not fulfilled else "not_delivered")
    ep = {
        "seed": sim.seed, "scenario": sim.scenario, "perception": perception,
        "scope": perception_scope(perception), "recipe_id": recipe.id, "order": order,
        "retries": retries, "fulfilled": fulfilled, "delivered": delivered,
        "on_table": assessment.on_table, "content": assessment.content, "exact": exact,
        "assessment": assessment.as_dict(),
        "verification_window_s": 0.5,
        "state": sim.process_state.value,
        "state_counts": dict(states),
        "incomplete": not exact,
        "perception_stop": ctl.blocked,
        "missing_units": sum(assessment.missing.values()),
        "misplaced_units": len(assessment.misplaced_units),
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
    return ep


def summarize(eps):
    n = len(eps)
    agg = {
        "episodes": n,
        "exact_deliveries": sum(1 for e in eps if e["exact"]),
        "incomplete_orders": sum(e["incomplete"] for e in eps),
        "perception_stops": sum(e["perception_stop"] for e in eps),
        "missing_units": sum(e["missing_units"] for e in eps),
        "misplaced_units": sum(e["misplaced_units"] for e in eps),
        "state_counts": dict(sum((Counter(e["state_counts"]) for e in eps), Counter())),
        "fulfilled": sum(1 for e in eps if e["fulfilled"]),
        "delivered": sum(1 for e in eps if e["delivered"]),
        "false_success": sum(1 for e in eps if e["fulfilled"] and e["delivered"]
                             and not e["exact"]),
        "physical_drops": sum(len(e["assessment"]["dropped_units"]) for e in eps),
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
    ap.add_argument("--recipe", type=Path)
    args = ap.parse_args()

    a, _, b = args.seeds.partition(":")
    try:
        seeds = list(range(int(a), int(b) + 1)) if b else [int(a)]
    except ValueError:
        ap.error("Seeds must be an integer or an inclusive START:END range")
    modes = args.modes.split(",")
    scenarios = args.scenarios.split(",")
    if not seeds or args.retries < 0:
        ap.error("Need at least one seed and nonnegative retries")
    if any(m not in ("oracle", "vision") for m in modes):
        ap.error("Modes must be oracle or vision")
    if any(s not in SCENARIOS for s in scenarios):
        ap.error(f"Scenarios must be chosen from {SCENARIOS}")

    try:
        recipe = load_recipe(load_config("catalog"), args.recipe)
    except (ValueError, OSError) as error:
        ap.error(str(error))
    run = RunLog(args.output or "runs",
                 {"demo": "orderpick-eval", "seeds": seeds,
                  "modes": modes, "scenarios": scenarios, "recipe_id": recipe.id,
                  "order": recipe.lines, "workstation": recipe.workstation,
                  "scope": {mode: perception_scope(mode) for mode in modes}})
    out = run.path
    episodes = []
    for scenario in scenarios:
        for mode in modes:
            retries = 0 if (args.no_retry and mode == "vision") \
                else args.retries
            for seed in seeds:
                ep = run_episode(seed, scenario, mode, retries, args.recipe)
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
    return 0 if all(e["exact"] for e in episodes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
