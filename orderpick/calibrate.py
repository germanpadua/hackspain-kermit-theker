"""Measure rendered front-bin XY localization error without manipulating parts."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from orderpick.controller import Controller
from orderpick.sim import CellSim
from warehouse.persistence import RunLog


def measure(seed):
    sim = CellSim(seed=seed)
    try:
        ctl = Controller(sim, perception="vision")
        ctl._bin_row = {}
        for _ in range(15):
            ctl.spin()
        rows = []
        for entry in sim.catalog["bins"]:
            if entry["depth_row"] != 0:
                continue
            ctl.skills.drive(sim.config["shelf"]["slots"][entry["slot"]]["x_m"])
            status, units = ctl._perceive_bin(entry["id"], entry["sku"])
            truth = [sim.truth_body_pos(name) for name in sim.piece_qadr
                     if sim.piece_bin[name] == entry["id"]]
            unmatched = list(units)
            errors = []
            for point in truth:
                if not unmatched:
                    break
                index = min(range(len(unmatched)),
                            key=lambda i: np.linalg.norm(unmatched[i][:2] - point[:2]))
                prediction = unmatched.pop(index)
                errors.append(round(float(np.linalg.norm(prediction[:2] - point[:2])) * 1000, 3))
            rows.append({"seed": seed, "bin": entry["id"], "sku": entry["sku"],
                         "status": status, "expected_units": len(truth),
                         "detected_units": len(units), "xy_errors_mm": errors})
        return rows
    finally:
        sim.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 8, 9])
    parser.add_argument("--output", type=Path, default=Path("runs"))
    args = parser.parse_args()
    run = RunLog(args.output, {"benchmark": "front-bin-localization", "seeds": args.seeds})
    rows = [row for seed in args.seeds for row in measure(seed)]
    errors = [error for row in rows for error in row["xy_errors_mm"]]
    matched = all(row["status"] == "ok" and row["detected_units"] == row["expected_units"]
                  for row in rows)
    summary = {
        "regions": len(rows), "matched_counts": matched,
        "observations": dict(Counter(row["status"] for row in rows)),
        "mean_xy_mm": round(float(np.mean(errors)), 3) if errors else None,
        "max_xy_mm": max(errors) if errors else None,
        "rows": rows,
    }
    run.write_json("localization.json", summary)
    print(json.dumps(summary, indent=2))
    print("calibration dir:", run.path)
    return 0 if matched and errors and max(errors) < 5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
