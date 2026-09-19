"""Assembly-station kitting demo. Product vision; oracle-assisted logistics."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import mujoco.viewer

from warehouse.persistence import RunLog

from .controller import Controller
from .orders import load_recipe, perception_scope, slot_map
from .sim import SCENARIOS, CellSim
from .verification import verify_delivery


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--perception", choices=["oracle", "vision"],
                    default="vision")
    ap.add_argument("--headless", action="store_true",
                    help="disable the native viewer")
    ap.add_argument("--scenario", choices=SCENARIOS, default="nominal")
    ap.add_argument("--recipe", type=Path, help="JSON recipe; defaults to catalog order")
    ap.add_argument("--run-dir", default="runs", help="parent of a unique run directory")
    ap.add_argument("--video", action="store_true",
                    help="save overview PNG sequence (not encoded video)")
    args = ap.parse_args()

    sim = CellSim(seed=args.seed, scenario=args.scenario)
    viewer = None
    try:
        try:
            recipe = load_recipe(sim.catalog, args.recipe)
        except (ValueError, OSError) as error:
            ap.error(str(error))
        ctl = Controller(sim, perception=args.perception)
        scope = perception_scope(args.perception)
        run = RunLog(args.run_dir, {
            "demo": "orderpick", "perception": args.perception, "seed": args.seed,
            "scenario": args.scenario, "recipe_id": recipe.id,
            "order": recipe.lines, "workstation": recipe.workstation, "scope": scope})
        frames = run.path / "frames"
        if args.video:
            frames.mkdir()
        if not args.headless:
            viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
        n_frame = 0
        orig_spin = ctl.skills.spin
        started = time.monotonic()

        def spin(n=10):
            nonlocal n_frame
            orig_spin(n)
            if viewer is not None:
                if not viewer.is_running():
                    raise KeyboardInterrupt
                viewer.sync()
            if args.video and n_frame % 5 == 0:
                cv2.imwrite(str(frames / f"o{n_frame:06d}.png"),
                            cv2.cvtColor(sim.render("overview"), cv2.COLOR_RGB2BGR))
            n_frame += 1

        ctl.skills.spin = spin
        fulfilled = delivered = False
        interrupted = False
        try:
            for _ in range(15):
                ctl.spin()
            fulfilled = ctl.fulfill(recipe.lines, slot_map(sim))
            if not ctl.blocked and (fulfilled or ctl._placed):
                delivered = ctl.deliver_tray()
        except KeyboardInterrupt:
            interrupted = True
        assessment = verify_delivery(sim, recipe.lines)
        result = {
            "seed": args.seed, "perception": args.perception, "scope": scope,
            "recipe_id": recipe.id, "order": recipe.lines,
            "fulfilled": fulfilled, "delivered": delivered, "interrupted": interrupted,
            "tray_content": assessment.content, "verified": assessment.exact and not interrupted,
            "assessment": assessment.as_dict(),
            "verification_window_s": 0.5,
            "sim_time_s": round(float(sim.data.time), 1),
            "wall_time_s": round(time.monotonic() - started, 1), "events": ctl.events}
        run.write_json("result.json", result)
        print(f"kit={recipe.id} verified={result['verified']} content={assessment.content}")
        print(f"sim={result['sim_time_s']}s wall={result['wall_time_s']}s run={run.path}")
        return 0 if result["verified"] else 1
    finally:
        if viewer is not None:
            viewer.close()
        sim.close()


if __name__ == "__main__":
    raise SystemExit(main())
