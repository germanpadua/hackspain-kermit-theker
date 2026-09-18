from dataclasses import replace
import json

import pytest

from warehouse.contracts import Level, Observations, OperatorEvent, State
from warehouse.evaluate import Evaluator
from warehouse.runtime import Experiment


def run_cycle(experiment):
    experiment.start()
    evaluator = Evaluator()
    for _ in range(12000):
        experiment.tick(experiment.simulated_operator_events())
        evaluator.sample(experiment)
        if experiment.controller.done:
            return evaluator.result(experiment)
    pytest.fail("Controller never reached a terminal state")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_three_cycles_from_reset(tmp_path, seed):
    with Experiment(seed=seed, output=tmp_path) as experiment:
        result = run_cycle(experiment)
        assert result["success"], result
        assert result["reception_repick_verified"]
        assert result["pose_error_m"] < 0.004
        assert result["stock_level"] == Level.LOW
        assert result["stock_correct"]
        assert result["physical_support_verified"] is None
        assert experiment.inventory.alerts["box-1"]["active"]
        events = [json.loads(line) for line in (experiment.log.path / "events.jsonl").read_text().splitlines()]
        assert sum(event["type"] == "IDEALIZED_ATTACH" for event in events) == 2
        assert sum(event["type"] == "IDEALIZED_DETACH" for event in events) == 2
        assert events[-1]["payload"]["state"] == "COMPLETE"
        snapshot = json.loads((experiment.log.path / "inventory.json").read_text())
        assert snapshot["slots"]["A1"]["reserved_by"] is None


@pytest.mark.parametrize("scenario,reason,location", [
    ("missing", "BIN_MISSING_OR_POSE_UNKNOWN", "A1"),
    ("blocked_return", "RETURN_SLOT_OCCUPIED", "reception"),
    ("pick_timeout", "MOTION_TIMEOUT", "A1"),
])
def test_failures_preserve_reservation(tmp_path, scenario, reason, location):
    with Experiment(scenario=scenario, output=tmp_path) as experiment:
        result = run_cycle(experiment)
        assert not result["success"]
        assert result["failure_reason"] == reason
        assert result["exception_handled_correctly"]
        assert not result["false_success"]
        assert experiment.inventory.bins["box-1"].current_location == location
        assert experiment.inventory.slots["A1"].reserved_by == experiment.controller.request.id


@pytest.mark.parametrize("offset", [-0.02, -0.005, 0.005, 0.02])
def test_oracle_lateral_correction(tmp_path, offset):
    with Experiment(offset_m=offset, output=tmp_path) as experiment:
        result = run_cycle(experiment)
        assert result["success"], result
        assert experiment.inventory.slots["A1"].nominal_pick_xyz_m == (0.0, 0.2, 0.38)


def test_double_confirmation_and_reservation_while_waiting(tmp_path):
    with Experiment(output=tmp_path) as experiment:
        experiment.start()
        for _ in range(5000):
            experiment.tick()
            if experiment.controller.state == State.WAIT_OPERATOR:
                break
        assert experiment.controller.state == State.WAIT_OPERATOR
        request = experiment.controller.request
        assert experiment.inventory.slots["A1"].reserved_by == request.id
        assert experiment.inventory.bins["box-1"].current_location == "reception"
        event = OperatorEvent(request.id, source="SIMULATED")
        experiment.tick([event, event])
        assert experiment.controller.confirmation_count == 1
        assert experiment.controller.state == State.OBSERVE_STOCK
        experiment.tick([event])
        assert experiment.controller.confirmation_count == 1
        assert experiment.inventory.observations["box-1"].level == Level.LOW


@pytest.mark.parametrize("scenario,level", [("empty", Level.EMPTY), ("unknown_stock", Level.UNKNOWN)])
def test_camera_empty_and_unknown(tmp_path, scenario, level):
    with Experiment(scenario=scenario, output=tmp_path) as experiment:
        result = run_cycle(experiment)
        assert result["success"]
        assert result["stock_level"] == level
        assert result["stock_correct"]
        assert ("box-1" in experiment.inventory.alerts) == (level == Level.EMPTY)


def test_stale_pose_cannot_start_pick(tmp_path):
    with Experiment(output=tmp_path) as experiment:
        experiment.start()
        for _ in range(2000):
            experiment.tick()
            if experiment.controller.state == State.OBSERVE:
                break
        pose = experiment.perception.observe_bin("box-1")
        experiment.controller.step_controller(experiment.sim.snapshot(), Observations(pose=replace(pose, timestamp=-1)))
        assert experiment.controller.state == State.FAILED
        assert experiment.sim.attached_bin_id is None


def test_extension_blocks_horizontal_transport(tmp_path):
    with Experiment(output=tmp_path) as experiment:
        snapshot = replace(experiment.sim.snapshot(), axes_xyz_m=(0, 0.2, 0.38))
        with pytest.raises(ValueError, match="RETRACTED"):
            experiment.motion.command((0.5, 0.2, 0.38), snapshot)
        with pytest.raises(ValueError, match="RETRACTED"):
            experiment.motion.command((0, 0.2, 0.45), snapshot)
        experiment.motion.command((0, 0.2, 0.45), snapshot, docking=True)


@pytest.mark.parametrize("mode", [{"transport": "contact"}, {"perception": "vision"}])
def test_unimplemented_modes_never_fallback_silently(tmp_path, mode):
    with pytest.raises(ValueError, match="only implements"):
        Experiment(output=tmp_path, **mode)


def test_reset_is_fresh_and_preserves_prior_logs(tmp_path):
    with Experiment(output=tmp_path) as first:
        run_cycle(first)
        old_path = first.log.path
        old_events = (old_path / "events.jsonl").read_bytes()
    with Experiment(output=tmp_path) as second:
        assert second.log.path != old_path
        assert second.inventory.observations == {}
        assert second.inventory.slots["A1"].reserved_by is None
        assert (old_path / "events.jsonl").read_bytes() == old_events


def test_human_wait_does_not_trigger_active_timeout(tmp_path):
    with Experiment(output=tmp_path) as experiment:
        experiment.start()
        for _ in range(5000):
            experiment.tick()
            if experiment.controller.state == State.WAIT_OPERATOR:
                break
        assert experiment.controller.state == State.WAIT_OPERATOR
        experiment.sim.data.time += 300
        event = OperatorEvent(experiment.controller.request.id, source="SIMULATED")
        experiment.tick([event])
        assert experiment.controller.state == State.OBSERVE_STOCK
        assert experiment.controller.wait_sim_s >= 300


@pytest.mark.parametrize("sku", ["TUERCAS", "ARANDELAS"])
def test_other_catalog_locations(tmp_path, sku):
    with Experiment(output=tmp_path) as experiment:
        experiment.start(sku)
        evaluator = Evaluator()
        while not experiment.controller.done:
            experiment.tick(experiment.simulated_operator_events())
            evaluator.sample(experiment)
        assert evaluator.result(experiment)["success"]
        bin = experiment.inventory.bins[experiment.controller.request.bin_id]
        assert bin.current_location == bin.home_slot_id


def test_lateral_zero_offset_is_not_replaced_by_default(tmp_path):
    with Experiment(scenario="lateral", offset_m=0.0, output=tmp_path) as experiment:
        assert experiment.sim.offset_m == 0.0


def test_idle_does_not_advance_episode_clock(tmp_path):
    with Experiment(output=tmp_path) as experiment:
        for _ in range(10):
            experiment.tick()
        assert experiment.sim.data.time == 0


def test_false_success_is_detected_independently(tmp_path):
    with Experiment(output=tmp_path) as experiment:
        experiment.start()
        experiment.controller.state = State.COMPLETE
        result = Evaluator().result(experiment)
        assert result["false_success"]
        assert not result["success"]
