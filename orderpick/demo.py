"""HackSpain'26 THEKER demo: autonomous industrial order preparation.

Rail-mounted cart carries a Franka Panda with a parallel gripper and an
onboard order tray. The controller fulfils a multi-SKU order: drive to the
shelf slot, perceive pieces, pick units, deposit in the tray, expose the
reserve bin when the front tote empties, drive to reception, deliver the
tray by its handle and verify the content.

  MUJOCO_GL=egl .venv/bin/python -m orderpick.demo --seed 7 \
      --perception vision --headless --run-dir runs/demo

Perception modes:
  oracle  – debug/integration: the controller reads piece poses from sim
            truth (labeled; never mixed with vision results).
  vision  – rendered wrist camera: hue segmentation of the declared piece
            head markers + depth back-projection through the calibrated
            camera model. The controller only ever sees unit positions.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .controller import Controller
from .sim import CellSim
from warehouse.persistence import RunLog


def _slot_map(sim):
    cat = sim.catalog
    slot_map = {}
    for e in cat["bins"]:
        x = sim.config["shelf"]["slots"][e["slot"]]["x_m"]
        slot_map.setdefault(e["sku"], (x, []))[1].append(e["id"])
    for v in slot_map.values():
        v[1].sort(key=lambda b: 0 if "front" in b else 1)
    return slot_map


def main():
    ap = argparse.ArgumentParser(description="orderpick demo")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--perception", choices=["oracle", "vision"],
                    default="vision")
    ap.add_argument("--headless", action="store_true",
                    help="no native viewer (viewer is separate anyway)")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--video", action="store_true",
                    help="record overview+wrist PNG frames to <run>/frames")
    ap.add_argument("--scenario", default="nominal",
                    help="CellSim scenario (nominal, reception_blocked, ...)")
    args = ap.parse_args()

    sim = CellSim(seed=args.seed, scenario=args.scenario)
    ctl = Controller(sim, perception=args.perception)
    if args.run_dir:
        run = Path(args.run_dir)
        run.mkdir(parents=True, exist_ok=True)
    else:
        run = RunLog("runs", {"demo": "orderpick",
                              "perception": args.perception,
                              "seed": args.seed}).path
    frames = run / "frames"
    if args.video:
        frames.mkdir(exist_ok=True)

    order = [(l["sku"], l["qty"]) for l in sim.catalog["order"]["lines"]]
    t0 = time.time()
    for _ in range(15):
        ctl.spin()

    n_frame = 0

    def snap():
        nonlocal n_frame
        if args.video and n_frame % 25 == 0:
            import cv2
            cv2.imwrite(str(frames / f"o{n_frame:04d}.png"),
                        cv2.cvtColor(sim.render("overview"),
                                     cv2.COLOR_RGB2BGR))
        n_frame += 1

    # wrap the scheduler's spin to grab frames cheaply
    orig_spin = ctl.skills.spin
    ctl.skills.spin = lambda n=1: (orig_spin(n), snap())[0]

    ok = ctl.fulfill(order, _slot_map(sim))
    print(f"fulfill={ok}")
    delivered = ctl.deliver_tray() if ok or ctl._placed else False
    print(f"deliver={delivered}")

    wall = time.time() - t0
    sim_t = float(sim.data.time)
    content = ctl._tray_contents()
    result = {
        "seed": args.seed, "perception": args.perception,
        "order": order, "fulfilled": ok, "delivered": delivered,
        "tray_content": content,
        "verified": delivered and content == {s: q for s, q in order},
        "sim_time_s": round(sim_t, 1), "wall_time_s": round(wall, 1),
        "events": ctl.events,
    }
    (run / "result.json").write_text(json.dumps(result, indent=2))
    for e in ctl.events:
        print("  ", e)
    print("content:", content)
    print(f"sim={sim_t:.0f}s wall={wall:.0f}s run={run}")
    sim.close()
    return 0 if result["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
