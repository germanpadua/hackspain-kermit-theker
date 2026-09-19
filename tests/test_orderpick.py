"""orderpick pipeline: contracts-level tests that don't run physics for
minutes — scene construction, inventory semantics, perception sanity.
Full physics episodes live in orderpick.demo / orderpick.evaluate."""
import numpy as np
import pytest

from orderpick.perception import _merge_units
from orderpick.sim import CellSim


def test_scene_builds_and_scatter_is_seeded():
    a = CellSim(seed=7)
    b = CellSim(seed=7)
    for name in a.piece_qadr:
        pa = a.data.qpos[a.piece_qadr[name]:a.piece_qadr[name] + 3]
        pb = b.data.qpos[b.piece_qadr[name]:b.piece_qadr[name] + 3]
        assert np.allclose(pa, pb), name
    a.close(); b.close()


def test_scene_scatter_varies_across_seeds():
    a = CellSim(seed=7)
    b = CellSim(seed=9)
    pa = a.data.qpos[a.piece_qadr["piece_bin-b_0"]:a.piece_qadr["piece_bin-b_0"] + 3]
    pb = b.data.qpos[b.piece_qadr["piece_bin-b_0"]:b.piece_qadr["piece_bin-b_0"] + 3]
    assert not np.allclose(pa, pb)
    a.close(); b.close()


def test_reserve_empty_scenario_has_no_reserve_pieces():
    sim = CellSim(seed=7, scenario="reserve_empty")
    assert not any("bin-a-reserve" in n for n in sim.piece_qadr)
    assert "piece_bin-a-front_0" in sim.piece_qadr
    sim.close()


def test_obstacle_scenarios_place_fixture_on_surface():
    for scen, name, key in (("park_blocked", "park_obstacle", "park"),
                            ("reception_blocked", "reception_obstacle",
                             "reception")):
        sim = CellSim(seed=7, scenario=scen)
        p = sim.truth_body_pos(name)
        cfg = sim.config[key]
        assert abs(p[0] - cfg["x_m"]) < 0.15
        assert p[2] > cfg["surface_z_m"]
        sim.close()


def test_glitch_frames_return_nan_depth():
    sim = CellSim(seed=7, scenario="obs_glitch")
    d = sim.render("wrist", depth=True)
    assert not np.isfinite(d).any()
    d = sim.render("wrist", depth=True)
    assert not np.isfinite(d).any()
    sim.close()


def test_merge_units_clusters_same_sku_xy():
    hits = [("A", np.array([0.0, 0.0, 0.6])),
            ("A", np.array([0.02, 0.03, 0.66])),   # same unit, depth noise
            ("A", np.array([0.20, 0.0, 0.6])),
            ("B", np.array([0.0, 0.01, 0.6]))]
    units = _merge_units(hits)
    assert len(units) == 3
    assert sorted(u[0] for u in units) == ["A", "A", "B"]


def test_perception_mode_rejects_unknown():
    from orderpick.controller import Controller
    sim = CellSim(seed=7)
    with pytest.raises(ValueError):
        Controller(sim, perception="magic")
    sim.close()
