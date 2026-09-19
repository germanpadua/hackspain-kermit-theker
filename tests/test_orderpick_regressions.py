"""Safety decisions, visual localization and independent delivery checks."""

import json
from unittest.mock import patch

import mujoco
import numpy as np
import pytest

from orderpick.controller import Controller
from orderpick.orders import load_recipe, slot_map
from orderpick.perception import WristVision, _merge_units
from orderpick.sim import CellSim
from orderpick.verification import assess_delivery, verify_delivery
from warehouse import persistence


@pytest.fixture
def sim():
    cell = CellSim(seed=7)
    yield cell
    cell.close()


@pytest.fixture
def camera(sim):
    vision = WristVision(sim)
    rgb = np.full((80, 80, 3), 180, dtype=np.uint8)
    depth = np.full((80, 80), 0.49)
    with (patch.object(vision, "_frame", return_value=(rgb, depth)),
          patch.object(vision, "_extrinsics",
                       return_value=(np.array([0, 0, .5]), np.eye(3))),
          patch.object(vision, "_intrinsics", return_value=(40, 40, 40, 40))):
        yield vision, rgb, depth


def test_region_outside_camera_never_means_empty_or_free(camera):
    vision, _, _ = camera
    assert vision.observe_region((10, 0), (.07, .07), floor_z=0) == ("unknown", [])
    assert vision.surface_free((10, 0), (.07, .07), floor_z=0) == "unknown"


def test_visible_floor_is_empty_and_free(camera):
    vision, _, _ = camera
    assert vision.observe_region((0, 0), (.07, .07), floor_z=0) == ("empty", [])
    assert vision.surface_free((0, 0), (.07, .07), floor_z=0) == "free"


def test_partial_invalid_depth_is_unknown_despite_valid_center(camera):
    vision, _, depth = camera
    depth[:, 43:] = np.nan
    assert np.isfinite(depth[40, 40])
    assert vision.observe_region((0, 0), (.07, .07), floor_z=0) == ("unknown", [])
    assert vision.surface_free((0, 0), (.07, .07), floor_z=0) == "unknown"


def test_foreground_occlusion_is_unknown(camera):
    vision, _, depth = camera
    depth[:] = 0.3
    assert vision.observe_region((0, 0), (.07, .07), floor_z=0) == ("unknown", [])


def test_marker_uses_head_instead_of_base_centroid():
    hits = [("gear", np.array([0.008, -.041, .637])),
            ("gear", np.array([-.002, -.024, .662])),
            ("gear", np.array([.012, -.020, .662]))]
    units = _merge_units(hits)
    assert len(units) == 1
    np.testing.assert_allclose(units[0][1], [.005, -.022, .662])


def test_rendered_front_marker_has_subcentimeter_error(sim):
    ctl = Controller(sim, perception="vision")
    ctl._bin_row = {}
    for _ in range(15):
        ctl.spin()
    status, units = ctl._perceive_bin("bin-a-front", "ENGRANAJE")
    assert status == "ok" and len(units) == 1
    truth = sim.truth_body_pos("piece_bin-a-front_0")
    assert np.linalg.norm(units[0][:2] - truth[:2]) < .005
    assert ctl.vision.observe_bin("offscreen", (10, 0), (.09, .09),
                                  floor_z=.62) == ("unknown", [])


def test_parking_region_matches_physical_support(sim):
    ctl = Controller(sim, perception="vision")
    for _ in range(15):
        ctl.spin()
    assert ctl._surface_free("park") == "free"


@pytest.mark.parametrize("observation", [
    ("unknown", []),
    ("ok", []),
    ("ok", [np.array([0, 0, .63])]),
])
def test_unreadable_wrong_sku_or_failed_grasp_cannot_deplete_bin(sim, observation):
    ctl = Controller(sim, perception="vision")
    with (patch.object(ctl.skills, "drive", return_value=True),
          patch.object(ctl, "_perceive_bin", return_value=observation),
          patch.object(ctl.skills, "pick_piece", return_value=False),
          patch.object(ctl, "advance_reserve", return_value=True) as advance):
        assert not ctl.fulfill([("ENGRANAJE", 2)], slot_map(sim))
        advance.assert_not_called()
    assert "bin-a-front" not in ctl._confirmed_empty
    assert not any(e["kind"] == "front_depleted" for e in ctl.events)


def test_confirmed_empty_can_advance_reserve(sim):
    ctl = Controller(sim, perception="vision")
    with (patch.object(ctl.skills, "drive", return_value=True),
          patch.object(ctl, "_perceive_bin", return_value=("empty", [])),
          patch.object(ctl, "advance_reserve", return_value=True) as advance):
        assert not ctl.fulfill([("ENGRANAJE", 2)], slot_map(sim))
        advance.assert_called_once_with("bin-a-front", "bin-a-reserve")


def test_unknown_after_grasp_stops_without_place_or_next_sku(sim):
    ctl = Controller(sim, perception="vision")
    with (patch.object(ctl.skills, "drive", return_value=True) as drive,
          patch.object(ctl, "_perceive_bin",
                       side_effect=[("ok", [np.array([0, 0, .63])]), ("unknown", [])]),
          patch.object(ctl.skills, "pick_piece", return_value=True),
          patch.object(ctl.skills, "place_in_tray") as place):
        assert not ctl.fulfill([("ENGRANAJE", 1), ("RODAMIENTO", 1)], slot_map(sim))
        place.assert_not_called()
        drive.assert_called_once()
    assert ctl.blocked


def test_tray_visual_verification_requires_correct_sku_counts(sim):
    ctl = Controller(sim, perception="vision")
    ctl._placed_skus.append("ENGRANAJE")
    verify = ctl._make_tray_verify("RODAMIENTO")
    point = np.zeros(3)
    for status, hits, expected in [
        ("unknown", [], False),
        ("ok", [("ENGRANAJE", point), ("ENGRANAJE", point)], False),
        ("ok", [("ENGRANAJE", point), ("RODAMIENTO", point)], True),
        ("ok", [("ENGRANAJE", point), ("RODAMIENTO", point), ("ESPARRAGO", point)], False),
    ]:
        with patch.object(ctl, "_perceive_tray", return_value=(status, hits)):
            assert verify() is expected


def _place_test_kit(sim, extra=False, floating=False):
    rc = sim.config["reception"]
    pos = np.array([rc["x_m"], rc["y_m"], rc["surface_z_m"] + (0.08 if floating else .001)])
    sim.data.qpos[sim.tray_qadr:sim.tray_qadr + 7] = [*pos, 1, 0, 0, 0]
    for name, offset in [("piece_bin-a-front_0", [.079, 0, .01]),
                         *([("piece_bin-b_0", [-.079, 0, .01])] if extra else [])]:
        adr = sim.piece_qadr[name]
        sim.data.qpos[adr:adr + 7] = [*(pos + offset), 1, 0, 0, 0]
    mujoco.mj_forward(sim.model, sim.data)
    if not floating:
        sim.step(500)


def test_physical_evaluator_accepts_supported_exact_kit(sim):
    _place_test_kit(sim)
    result = assess_delivery(sim, [("ENGRANAJE", 1)])
    assert result.exact, result
    assert result.table_support_n > .5
    assert result.contained and result.stable and result.released
    assert verify_delivery(sim, [("ENGRANAJE", 1)]).exact


def test_brief_motion_inside_verification_window_is_not_success(sim):
    _place_test_kit(sim)
    velocity = int(sim.model.joint("tray_free").dofadr[0])
    step = sim.step
    samples = 0

    def transient_motion(count):
        nonlocal samples
        step(count)
        samples += 1
        sim.data.qvel[velocity] = .1 if samples == 2 else 0
        mujoco.mj_forward(sim.model, sim.data)

    with patch.object(sim, "step", side_effect=transient_motion):
        result = verify_delivery(sim, [("ENGRANAJE", 1)])
    assert not result.exact
    assert not result.stable


def test_extra_unit_is_not_success(sim):
    _place_test_kit(sim, extra=True)
    result = assess_delivery(sim, [("ENGRANAJE", 1)])
    assert result.extra == {"RODAMIENTO": 1}
    assert not result.exact


def test_hovering_above_table_is_not_delivery(sim):
    _place_test_kit(sim, floating=True)
    result = assess_delivery(sim, [("ENGRANAJE", 1)])
    assert result.content == {"ENGRANAJE": 1}
    assert not result.physically_delivered
    assert not result.exact


def test_moving_tray_is_not_success(sim):
    _place_test_kit(sim)
    velocity = int(sim.model.joint("tray_free").dofadr[0])
    sim.data.qvel[velocity] = .5
    mujoco.mj_forward(sim.model, sim.data)
    result = assess_delivery(sim, [("ENGRANAJE", 1)])
    assert not result.stable
    assert not result.exact


@pytest.mark.parametrize("lines", [
    [], [{"sku": "UNKNOWN", "qty": 1}], [{"sku": "ENGRANAJE", "qty": 0}],
    [{"sku": "ENGRANAJE", "qty": True}], [{"sku": "ENGRANAJE", "qty": 1.5}],
    [{"sku": "ENGRANAJE", "qty": 1}, {"sku": "ENGRANAJE", "qty": 1}],
])
def test_recipe_rejects_invalid_demand(sim, tmp_path, lines):
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"lines": lines}))
    with pytest.raises(ValueError):
        load_recipe(sim.catalog, recipe)


def test_recipe_variation_does_not_modify_inventory(sim, tmp_path):
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"id": "service", "lines": [{"sku": "RODAMIENTO", "qty": 1}]}))
    stock = dict(sim.piece_sku)
    assert load_recipe(sim.catalog, recipe).lines == [("RODAMIENTO", 1)]
    assert sim.piece_sku == stock


def test_code_hash_covers_orderpick_and_nested_recipes(tmp_path):
    (tmp_path / "orderpick").mkdir()
    (tmp_path / "config" / "recipes").mkdir(parents=True)
    source = tmp_path / "orderpick" / "controller.py"
    recipe = tmp_path / "config" / "recipes" / "kit.json"
    source.write_text("version = 1\n")
    recipe.write_text("{}")
    with patch.object(persistence, "ROOT", tmp_path):
        initial = persistence.code_version()["source_sha256"]
        source.write_text("version = 2\n")
        modified = persistence.code_version()["source_sha256"]
        assert initial != modified
        recipe.write_text('{"qty": 2}')
        assert persistence.code_version()["source_sha256"] != modified
