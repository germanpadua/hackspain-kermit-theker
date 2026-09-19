"""Read-only physical evaluation, independent of controller bookkeeping."""

from collections import Counter
from dataclasses import asdict, dataclass, replace

import mujoco
import numpy as np

from .sim import CellSim


@dataclass(frozen=True)
class Delivery:
    content: dict[str, int]
    missing: dict[str, int]
    extra: dict[str, int]
    on_table: bool
    table_support_n: float
    released: bool
    stable: bool
    contained: bool
    physically_delivered: bool
    exact: bool
    dropped_units: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


def _speed(sim: CellSim, body_id: int) -> tuple[float, float]:
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(sim.model, sim.data, mujoco.mjtObj.mjOBJ_BODY,
                            body_id, velocity, 0)
    return float(np.linalg.norm(velocity[3:])), float(np.linalg.norm(velocity[:3]))


def assess_delivery(sim: CellSim, order: list[tuple[str, int]]) -> Delivery:
    tray_id = int(sim.model.body("tray").id)
    pos = sim.data.xpos[tray_id]
    rotation = sim.data.xmat[tray_id].reshape(3, 3)
    rc, tray = sim.config["reception"], sim.config["tray"]
    half = np.array(tray["size_xyz_m"][:2]) / 2
    footprint = np.abs(rotation[:2, :2]) @ half
    delta = np.abs(pos[:2] - [rc["x_m"], rc["y_m"]])
    on_table = bool(
        np.all(delta + footprint <= np.array(rc["size_xy_m"]) / 2 + 0.005)
        and abs(pos[2] - rc["surface_z_m"]) < 0.02
        and rotation[2, 2] > np.cos(np.deg2rad(10)))
    content: Counter[str] = Counter()
    inside_ids = set()
    dropped = []
    stable_pieces = True
    for name, sku in sim.piece_sku.items():
        bid = int(sim.model.body(name).id)
        rel = rotation.T @ (sim.data.xipos[bid] - pos)
        if sim.data.xpos[bid, 2] < 0.10:
            dropped.append(name)
        if (np.all(np.abs(rel[:2]) < half - tray["wall_m"] / 2)
                and 0 < rel[2] < tray["size_xyz_m"][2] - 0.01):
            content[sku] += 1
            inside_ids.add(bid)
            speed, angular = _speed(sim, bid)
            stable_pieces &= speed < 0.025 and angular < 0.2

    table_id = int(sim.model.geom("reception_top").id)
    gripper_ids = {int(sim.model.body(name).id)
                   for name in ("hand", "left_finger", "right_finger")}
    support = 0.0
    held = False
    supported_ids = set()
    force = np.zeros(6)
    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        b1, b2 = int(sim.model.geom_bodyid[g1]), int(sim.model.geom_bodyid[g2])
        mujoco.mj_contactForce(sim.model, sim.data, i, force)
        vertical = max(0.0, float(force[0])) * abs(float(contact.frame[2]))
        if tray_id in (b1, b2):
            if table_id in (g1, g2):
                support += vertical
            other = b2 if b1 == tray_id else b1
            if other in gripper_ids:
                held = True
            if other in inside_ids and vertical > 0.01:
                supported_ids.add(other)
        if (b1 in inside_ids and b2 in gripper_ids
                or b2 in inside_ids and b1 in gripper_ids):
            held = True

    speed, angular = _speed(sim, tray_id)
    stable = bool(speed < 0.025 and angular < 0.2 and stable_pieces)
    contained = inside_ids <= supported_ids
    delivered = bool(on_table and support > 0.5 and not held and stable and contained)
    wanted = Counter(dict(order))
    missing, extra = dict(wanted - content), dict(content - wanted)
    return Delivery(dict(content), missing, extra, on_table, round(support, 4),
                    not held, stable, contained, delivered,
                    delivered and not missing and not extra, dropped)


def verify_delivery(sim: CellSim, order: list[tuple[str, int]]) -> Delivery:
    samples = [assess_delivery(sim, order)]
    for _ in range(5):
        sim.step(round(0.1 / sim.model.opt.timestep))
        samples.append(assess_delivery(sim, order))
    final = samples[-1]
    steady = all(s.stable and s.content == final.content for s in samples)
    delivered = all(s.physically_delivered for s in samples) and steady
    return replace(final, stable=steady, physically_delivered=delivered,
                   exact=delivered and not final.missing and not final.extra)
