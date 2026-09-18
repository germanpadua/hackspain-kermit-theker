import argparse
import json
from pathlib import Path
import time

import numpy as np

from warehouse.contracts import Level, State
from warehouse.runtime import Experiment, SCENARIOS, validate_modes


class Evaluator:
    def __init__(self):
        self.previous_position = None
        self.previous_time = None
        self.bin_speed_m_s = 0.0
        self.delivered = False
        self.repicked = False
        self.drop_detected = False

    def sample(self, experiment):
        sim, controller = experiment.sim, experiment.controller
        if controller.request is None:
            return
        bin_id = controller.request.bin_id
        if bin_id not in sim.present:
            return
        position = sim.data.mocap_pos[sim.mocap_id(bin_id)].copy()
        timestamp = float(sim.data.time)
        if self.previous_time is not None and timestamp > self.previous_time:
            self.bin_speed_m_s = float(np.linalg.norm(position - self.previous_position) / (timestamp - self.previous_time))
        self.previous_position, self.previous_time = position, timestamp
        self.drop_detected |= bool(position[2] < 0.12)
        if controller.state == State.WAIT_OPERATOR:
            self.delivered |= bool(sim.attached_bin_id is None and np.linalg.norm(position - sim.config["reception_xyz_m"]) < 0.004)
        if controller.state == State.PICK_AT_RECEPTION and sim.attached_bin_id == bin_id:
            self.repicked = True

    def result(self, experiment):
        self.sample(experiment)
        sim, controller, inventory = experiment.sim, experiment.controller, experiment.inventory
        request = controller.request
        bin = inventory.bins[request.bin_id]
        home_slot = inventory.slots[bin.home_slot_id]
        position = sim.data.mocap_pos[sim.mocap_id(bin.id)]
        error = float(np.linalg.norm(position - home_slot.nominal_pick_xyz_m)) if bin.id in sim.present else None
        coherent = bin.current_location == bin.home_slot_id and bin.status == "AVAILABLE" and home_slot.reserved_by is None
        truth_success = bool(self.delivered and self.repicked and controller.confirmation_count == 1
                             and error is not None and error < 0.004 and self.bin_speed_m_s < 0.008
                             and not self.drop_detected and not sim.attached_bin_id and coherent)
        success = bool(controller.state == State.COMPLETE and truth_success)
        expected_failures = {"missing": "BIN_MISSING_OR_POSE_UNKNOWN", "blocked_return": "RETURN_SLOT_OCCUPIED", "pick_timeout": "MOTION_TIMEOUT"}
        expected = expected_failures.get(sim.scenario)
        integrity_preserved = home_slot.reserved_by == request.id and bin.status == "INCIDENT"
        exception_handled = bool(expected and controller.failure_reason == expected and integrity_preserved and not success)
        if sim.scenario == "blocked_return":
            exception_handled &= bin.current_location == "reception" and sim.attached_bin_id is None
        stock = inventory.observations.get(bin.id)
        expected_level = Level.UNKNOWN if sim.scenario == "unknown_stock" else Level.EMPTY if sim.scenario == "empty" else Level.LOW
        stock_correct = stock.level == expected_level if stock else None
        elapsed = time.monotonic() - experiment.log.started
        idle_wall = experiment.request_started_wall - experiment.log.started
        waiting = controller.state == State.WAIT_OPERATOR
        wait_sim = controller.wait_sim_s + (sim.data.time - controller.wait_started_sim if waiting else 0)
        wait_wall = controller.wait_wall_s + (time.monotonic() - controller.wait_started_wall if waiting else 0)
        return {
            **experiment.log.metadata,
            "request_id": request.id, "sku": request.sku, "state": controller.state,
            "success": success, "exception_handled_correctly": exception_handled,
            "false_success": bool(controller.state == State.COMPLETE and not truth_success),
            "failure_reason": controller.failure_reason, "expected_failure": expected,
            "sim_time_s": float(sim.data.time), "active_sim_s": float(sim.data.time - wait_sim),
            "operator_wait_sim_s": float(wait_sim), "wall_time_s": elapsed,
            "active_wall_s": elapsed - wait_wall - idle_wall, "operator_wait_wall_s": wait_wall,
            "startup_and_idle_wall_s": idle_wall,
            "retries": controller.retries, "pose_error_m": error, "final_bin_speed_m_s": self.bin_speed_m_s,
            "delivered_verified": self.delivered, "reception_repick_verified": self.repicked,
            "logical_drop_detected": self.drop_detected,
            "physical_support_verified": None, "physical_drops": None,
            "stock_level": stock.level if stock else None,
            "stock_observation_source": stock.source if stock else None,
            "stock_correct": stock_correct,
            "false_empty": bool(stock and stock.level == Level.EMPTY and expected_level != Level.EMPTY),
            "camera_error": experiment.camera_error, "history": controller.history,
        }


def parse_seeds(value):
    try:
        start, stop = (int(part) for part in value.split(":"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use START:STOP, with STOP exclusive") from error
    if start < 0 or not start < stop or stop - start > 1000:
        raise argparse.ArgumentTypeError("Seeds must satisfy 0 <= START < STOP, maximum 1000 episodes")
    return range(start, stop)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate integration only: ORACLE + IDEALIZED; simulated operator")
    parser.add_argument("--seeds", type=parse_seeds, default=range(3))
    parser.add_argument("--perception", choices=("oracle", "vision"), default="oracle")
    parser.add_argument("--transport", choices=("idealized", "contact"), default="idealized")
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=["nominal"])
    parser.add_argument("--offset-mm", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=Path("runs"))
    args = parser.parse_args(argv)
    try:
        validate_modes(args.perception, args.transport)
    except ValueError as error:
        parser.error(str(error))
    if not np.isfinite(args.offset_mm) or abs(args.offset_mm) > 100:
        parser.error("--offset-mm must be finite and within +/-100 mm")
    results = []
    batch_path = None
    for scenario in args.scenarios:
        for seed in args.seeds:
            with Experiment(seed=seed, scenario=scenario, output=args.output,
                            offset_m=args.offset_mm / 1000 if scenario == "lateral" else 0.0) as experiment:
                experiment.start()
                evaluator = Evaluator()
                while not experiment.controller.done:
                    experiment.tick(experiment.simulated_operator_events())
                    evaluator.sample(experiment)
                result = evaluator.result(experiment)
                experiment.log.write_json("result.json", result)
                if batch_path is None:
                    batch_path = experiment.log.path / "batch.json"
                results.append({**result, "run_path": str(experiment.log.path)})
                print(f"seed={seed} scenario={scenario} success={result['success']} exception_ok={result['exception_handled_correctly']} reason={result['failure_reason']}")
    summary = {
        "perception": "ORACLE", "transport": "IDEALIZED", "operator_source": "SIMULATED",
        "episodes": len(results), "task_successes": sum(item["success"] for item in results),
        "exceptions_handled": sum(item["exception_handled_correctly"] for item in results),
        "false_successes": sum(item["false_success"] for item in results),
        "false_empty": sum(item["false_empty"] for item in results),
        "results": results,
    }
    batch_path.write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}))
    print(f"Batch: {batch_path}")
    good = all((item["success"] or item["exception_handled_correctly"])
               and not item["false_success"] and item["stock_correct"] is not False for item in results)
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
