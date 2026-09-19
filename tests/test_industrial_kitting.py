"""Manufacturing recipes and physical poka-yoke verification."""

from unittest.mock import patch

import mujoco
import numpy as np
import pytest

from orderpick.controller import Controller
from orderpick import evaluate
from orderpick.contracts import Phase
from orderpick.orders import compartment_for, load_recipe, validate_tray_recipe
from orderpick.scene import ROOT, load_config
from orderpick.sim import CellSim
from orderpick.verification import assess_delivery, verify_delivery


@pytest.fixture
def cell():
    sim = CellSim()
    yield sim
    sim.close()


def supported_gear(sim, x=.079, y=0):
    reception = sim.config["reception"]
    tray = np.array([reception["x_m"], reception["y_m"], reception["surface_z_m"] + .001])
    sim.data.qpos[sim.tray_qadr:sim.tray_qadr + 7] = [*tray, 1, 0, 0, 0]
    adr = sim.piece_qadr["piece_bin-a-front_0"]
    sim.data.qpos[adr:adr + 7] = [*(tray + [x, y, .012]), 1, 0, 0, 0]
    mujoco.mj_forward(sim.model, sim.data)
    sim.step(600)


def test_wrong_compartment_rejected_with_correct_total_and_physical_support(cell):
    supported_gear(cell, x=-.079)
    result = verify_delivery(cell, [("ENGRANAJE", 1)])
    assert result.content == {"ENGRANAJE": 1}
    assert not result.missing and not result.extra
    assert result.physically_delivered
    assert result.misplaced_units == ["piece_bin-a-front_0"]
    assert result.compartments["0"] == {"ENGRANAJE": 1}
    assert not result.kit_exact and not result.exact


def test_controller_flags_cannot_change_independent_result(cell):
    supported_gear(cell)
    expected = assess_delivery(cell, [("ENGRANAJE", 1)])
    assert expected.exact
    ctl = Controller(cell)
    ctl.blocked = True
    ctl._order = {"ESPARRAGO": 100}
    ctl._placed_skus = ["RODAMIENTO"] * 20
    ctl._placed = [(0, 0)] * 20
    assert assess_delivery(cell, [("ENGRANAJE", 1)]) == expected


def test_delivery_outside_zone_rejected_even_when_contact_support_exists(cell):
    supported_gear(cell)
    cell.data.qpos[cell.tray_qadr] += .07
    cell.data.qpos[cell.piece_qadr["piece_bin-a-front_0"]] += .07
    mujoco.mj_forward(cell.model, cell.data)
    cell.step(100)
    result = assess_delivery(cell, [("ENGRANAJE", 1)])
    assert result.table_support_n > .5
    assert not result.on_table and not result.exact


@pytest.mark.parametrize("file,units", [
    ("recipe-production.json", 4), ("recipe-service.json", 2)])
def test_recipe_drives_work_order_and_does_not_change_stock(file, units):
    sim = CellSim(recipe_path=ROOT / "config" / file)
    try:
        assert sum(qty for _, qty in sim.recipe.lines) == units
        assert len(sim.piece_sku) == 5
        assert sim.catalog["order"]["id"] == sim.recipe.id
        assert sim.model.geom("work_order").id >= 0
        assert sim.model.body("open_transmission_housing").id >= 0
    finally:
        sim.close()


def test_recipe_over_capacity_rejected_before_motion():
    config, catalog = load_config("cell"), load_config("catalog")
    catalog["order"]["lines"] = [{"sku": "ENGRANAJE", "qty": 3}]
    with pytest.raises(ValueError, match="capacity"):
        validate_tray_recipe(config, catalog, load_recipe(catalog))


def test_picking_uses_assigned_compartments_independent_of_recipe_order(cell):
    controller = Controller(cell)
    for sku in ("ESPARRAGO", "ENGRANAJE", "RODAMIENTO", "ENGRANAJE"):
        x, y = controller._next_comp(sku)
        index = compartment_for(cell.config, x, y)
        assert cell.config["tray"]["compartment_skus"][index] == sku
        assert (x, y) not in controller._placed
        controller._placed.append((x, y))


def test_divider_is_not_a_compartment(cell):
    boundary = (cell.config["tray"]["size_xyz_m"][0]
                - 2 * cell.config["tray"]["wall_m"]) / 6
    assert compartment_for(cell.config, boundary, 0) is None


def test_unreadable_pick_stops_without_touching_other_skus(cell):
    controller = Controller(cell, perception="vision")
    with (patch.object(controller.skills, "drive", return_value=True) as drive,
          patch.object(controller, "_perceive_bin", return_value=("unknown", []))):
        assert not controller.fulfill(
            [("ENGRANAJE", 1), ("RODAMIENTO", 1)],
            {"ENGRANAJE": (0, ["bin-a-front"]), "RODAMIENTO": (.55, ["bin-b"])})
    assert controller.blocked
    assert cell.process_state == Phase.PERCEPTION_STOP
    assert not controller._confirmed_empty
    drive.assert_called_once()


def test_real_camera_occlusion_reports_unknown():
    sim = CellSim(scenario="obs_occluded")
    try:
        controller = Controller(sim, perception="vision")
        controller._bin_row = {}
        for _ in range(15):
            controller.spin()
        assert sim.model.geom("camera_occluder").id >= 0
        assert controller._perceive_bin("bin-a-front", "ENGRANAJE") == ("unknown", [])
    finally:
        sim.close()


def test_empty_front_retains_physical_reserve():
    sim = CellSim(scenario="front_empty")
    try:
        assert "piece_bin-a-front_0" not in sim.piece_sku
        assert sum(bin_id == "bin-a-reserve" for bin_id in sim.piece_bin.values()) == 2
        assert sim.model.body("bin-a-front").id >= 0
    finally:
        sim.close()


def test_benchmark_wall_time_includes_independent_verification(cell):
    clock = [0.0]
    verify = evaluate.verify_delivery

    def timed_verify(sim, order):
        clock[0] += 5
        return verify(sim, order)

    with (patch.object(Controller, "fulfill", return_value=False),
          patch.object(Controller, "spin"),
          patch.object(evaluate.time, "monotonic", side_effect=lambda: clock[0]),
          patch.object(evaluate, "verify_delivery", side_effect=timed_verify)):
        episode = evaluate._run_episode(cell, "oracle", 2, None)
    assert episode["wall_time_s"] == 5
    assert episode["state"] == "ORDER_INCOMPLETE"
    assert not episode["exact"]


def test_contact_diagnostic_render_reuses_sim_renderer(cell):
    controller = Controller(cell)
    with patch("orderpick.skills.cv2.imwrite", return_value=True) as write:
        controller.skills._dump_frame()
    assert write.call_args.args[1].shape == (480, 640, 3)
    assert (640, 480, False) in cell.renderers


def test_empty_tray_and_handle_do_not_create_product_detections(cell):
    controller = Controller(cell, perception="vision")
    controller.spin(150)
    assert controller._perceive_tray() == ("empty", [])


def test_tray_vision_covers_valid_pieces_near_inner_wall(cell):
    cell.step(500)
    tray = cell.truth_body_pos("tray")
    adr = cell.piece_qadr["piece_bin-a-front_0"]
    cell.data.qpos[adr:adr + 7] = [*(tray + [.08, -.075, .012]), 1, 0, 0, 0]
    mujoco.mj_forward(cell.model, cell.data)
    cell.step(500)
    controller = Controller(cell, perception="vision")
    status, hits = controller._perceive_tray()
    assert status == "ok"
    assert [sku for sku, _ in hits] == ["ENGRANAJE"]
    assert controller._make_place_verify("ENGRANAJE", None)()


def test_visual_pick_uses_measured_marker_height_and_keeps_pinch_limits(cell):
    controller = Controller(cell, perception="vision")
    controller._bin_row = {}
    controller.spin(150)
    status, points = controller._perceive_bin("bin-a-front", "ENGRANAJE")
    assert status == "ok" and len(points) == 1
    actual = cell.truth_body_pos("piece_bin-a-front_0")
    assert abs(points[0][2] - actual[2]) < .002
    assert controller.skills.pick_piece(points[0], None, pinch_band=(.007, .0155))
    assert cell.truth_body_pos("piece_bin-a-front_0")[2] > actual[2] + .05
