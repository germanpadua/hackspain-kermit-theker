"""Scene builder for the order-picking cell (MjSpec composition).

Layout (m/s/kg, Z up, X along the rail, +Y into the shelves):
- A horizontal rail at rail.y_m carrying an actuated cart (slide joint on X).
- The cart carries the Panda arm on a pedestal plus a recessed pocket holding
  the removable order tray.
- A shelf with several slots along X; each slot holds `depth` bins arranged
  front-to-back (row 0 = front). One slot uses depth 2 for reserve access.
- A parking stand for one empty bin, left of the first slot.
- A reception table at the far right end of the rail, clear of the slots.
- Cameras: fixed overview, fixed reception top view, wrist camera on the hand.
"""
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PANDA_XML = ROOT / "assets" / "menagerie" / "franka_emika_panda" / "panda.xml"

STEEL = (0.38, 0.43, 0.50, 1)
SHELF_GRAY = (0.42, 0.47, 0.52, 1)
BIN_COLORS = [(0.25, 0.50, 0.62, 1), (0.32, 0.55, 0.40, 1), (0.55, 0.45, 0.28, 1), (0.50, 0.40, 0.55, 1)]
TRAY_COLOR = (0.72, 0.62, 0.20, 1)
FLOOR_RGBA = (0.16, 0.19, 0.23, 1)
PARK_RGBA = (0.45, 0.40, 0.55, 1)
TABLE_RGBA = (0.55, 0.42, 0.30, 1)
QUAT_X_AXIS = [0.7071068, 0, 0.7071068, 0]  # +90 deg about Y: cylinder axis -> X


def load_config(name: str) -> dict:
    return json.loads((ROOT / "config" / f"{name}.json").read_text())


def bin_size(config: dict) -> np.ndarray:
    return np.asarray(config["bin"]["size_xyz_m"])


def bin_origin(config: dict, slot_id: str, row: int) -> np.ndarray:
    """Bottom-center position of a bin stored at (slot, row)."""
    slot = config["shelf"]["slots"][slot_id]
    y = config["shelf"]["front_y_m"] + bin_size(config)[1] / 2 + 0.003 \
        + row * config["shelf"]["bin_spacing_y_m"]
    return np.array([slot["x_m"], y, config["shelf"]["surface_z_m"]])


def tray_pocket_world(config: dict, cart_x: float) -> np.ndarray:
    px, py, pz = config["tray_pocket_xyz_m"]
    return np.array([cart_x + px, config["rail"]["y_m"] + py, pz])


def _walls(body, size_xyz, wall, height, rgba, prefix, open_front=False, lip_h=0.0):
    """Three-sided bin walls (x sides + back); front is open or a low lip."""
    x, y, _ = size_xyz
    fl = max(wall, 0.010)  # thicker floor plate resists impact tunneling
    body.add_geom(name=f"{prefix}_floor", type=mujoco.mjtGeom.mjGEOM_BOX,
                  pos=[0, 0, fl / 2], size=[x / 2, y / 2, fl / 2], rgba=list(rgba))
    for side in (-1, 1):
        body.add_geom(name=f"{prefix}_wallx{side}", type=mujoco.mjtGeom.mjGEOM_BOX,
                      pos=[side * (x / 2 - wall / 2), 0, height / 2],
                      size=[wall / 2, y / 2, height / 2], rgba=list(rgba))
    body.add_geom(name=f"{prefix}_wally1", type=mujoco.mjtGeom.mjGEOM_BOX,
                  pos=[0, y / 2 - wall / 2, height / 2],
                  size=[x / 2, wall / 2, height / 2], rgba=list(rgba))
    if not open_front:
        body.add_geom(name=f"{prefix}_wally-1", type=mujoco.mjtGeom.mjGEOM_BOX,
                      pos=[0, -y / 2 + wall / 2, height / 2],
                      size=[x / 2, wall / 2, height / 2], rgba=list(rgba))
    elif lip_h > 0:
        body.add_geom(name=f"{prefix}_lip", type=mujoco.mjtGeom.mjGEOM_BOX,
                      pos=[0, -y / 2 + wall / 2, wall + lip_h / 2],
                      size=[x / 2, wall / 2, lip_h / 2], rgba=list(rgba))


def _post_handle(body, spec_cfg, base_z, rim_z, y, rgba, prefix):
    """Mushroom pull handle: a thin post topped by a wider head disc.

    The gripper pinches the post just below the head; the head cannot pass
    between the pads, so lifting is mechanical rather than friction-only.
    """
    post_r = spec_cfg["post_r_m"]
    head_r = spec_cfg["head_r_m"]
    head_h = spec_cfg["head_h_m"]
    top_z = rim_z + spec_cfg["handle_rise_m"]
    post_top = top_z - head_h
    body.add_geom(name=f"{prefix}_post", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  pos=[0, y, (base_z + post_top) / 2],
                  size=[post_r, (post_top - base_z) / 2, 0], rgba=list(rgba))
    body.add_geom(name=f"{prefix}_head", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  pos=[0, y, post_top + head_h / 2],
                  size=[head_r, head_h / 2, 0], rgba=list(rgba))


def add_bin(spec, config, entry, rgba):
    b = config["bin"]
    size = b["size_xyz_m"]
    origin = bin_origin(config, entry["slot"], entry["depth_row"])
    body = spec.worldbody.add_body(name=entry["id"], pos=list(origin))
    body.add_freejoint(name=f"{entry['id']}_free")
    # Open-front pick tote: pieces sit in the front zone, the hand's tail
    # hangs over the shelf edge in free air while the fingers dive inside.
    _walls(body, size, b["wall_m"], size[2], rgba, entry["id"],
           open_front=True, lip_h=b["lip_h_m"])
    body.add_geom(name=f"{entry['id']}_mass", type=mujoco.mjtGeom.mjGEOM_BOX,
                  pos=[0, 0, 0.02], size=[0.05, 0.05, 0.02],
                  contype=0, conaffinity=0, mass=b["mass_kg"])
    return body


def add_piece(spec, sku_id, sku_info, name):
    """Pawn-shaped part: base, slim post, head brim.

    Parallel jaws pinch the exposed post just below the head, so the part
    is captured mechanically (the head can't slip through the pads) and
    cannot rotate out of a rim pinch. All geoms are box/cylinder pairs that
    rest and contact reliably (flat cylinder faces tunnel through thin
    floors, so the base collision is its inscribed box).
    """
    body = spec.worldbody.add_body(name=name, pos=[0, 0, -2])
    body.add_freejoint(name=f"{name}_free")
    friction = [1.6, 0.04, 0.004]
    rgb = list(sku_info["rgba"])
    r = sku_info["radius_m"] if "radius_m" in sku_info else sku_info["size_xy_m"][0] / 2
    base_h = 0.008; post_h = 0.022; head_h = 0.003
    ib = r * 0.7071
    body.add_geom(name=f"{name}_g", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[ib, ib, base_h / 2], pos=[0, 0, base_h / 2],
                  mass=sku_info["mass_kg"], friction=friction, rgba=rgb)
    body.add_geom(name=f"{name}_vis", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  pos=[0, 0, base_h / 2], size=[r, base_h / 2, 0],
                  contype=0, conaffinity=0, rgba=rgb)
    body.add_geom(name=f"{name}_neck", type=mujoco.mjtGeom.mjGEOM_BOX,
                  pos=[0, 0, base_h + post_h / 2],
                  size=[0.007, 0.007, post_h / 2],
                  friction=[2.0, 0.05, 0.005], rgba=[0.25, 0.25, 0.27, 1])
    body.add_geom(name=f"{name}_head", type=mujoco.mjtGeom.mjGEOM_BOX,
                  pos=[0, 0, base_h + post_h + head_h / 2],
                  size=[0.008, 0.008, head_h / 2],
                  friction=[2.0, 0.05, 0.005], rgba=[0.25, 0.25, 0.27, 1])
    body.add_geom(name=f"{name}_headvis", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  pos=[0, 0, base_h + post_h + head_h / 2],
                  size=[0.008, head_h / 2, 0],
                  contype=0, conaffinity=0, rgba=[0.25, 0.25, 0.27, 1])
    return body


def add_tray(spec, config, cart_body):
    t = config["tray"]
    size = t["size_xyz_m"]
    px, py, pz = config["tray_pocket_xyz_m"]
    lip = 0.010
    gap = 0.008  # clearance so the tray only touches lips under lateral load
    for side in (-1, 1):
        cart_body.add_geom(name=f"pocket_x{side}", type=mujoco.mjtGeom.mjGEOM_BOX,
                           pos=[px + side * (size[0] / 2 + gap + lip / 2), py, pz + lip / 2],
                           size=[lip / 2, size[1] / 2 + gap + lip, lip / 2], rgba=list(STEEL))
        cart_body.add_geom(name=f"pocket_y{side}", type=mujoco.mjtGeom.mjGEOM_BOX,
                           pos=[px, py + side * (size[1] / 2 + gap + lip / 2), pz + lip / 2],
                           size=[size[0] / 2 + gap + lip, lip / 2, lip / 2], rgba=list(STEEL))
    tray = spec.worldbody.add_body(name="tray", pos=[0, 0, 0])
    tray.add_freejoint(name="tray_free")
    _walls(tray, size, t["wall_m"], size[2], TRAY_COLOR, "tray")
    n = t["compartments"]
    inner = size[0] - 2 * t["wall_m"]
    for i in range(1, n):
        x = -inner / 2 + i * inner / n
        tray.add_geom(name=f"tray_div{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                      pos=[x, 0, size[2] / 2],
                      size=[t["wall_m"] / 2, size[1] / 2 - t["wall_m"], size[2] / 2],
                      rgba=list(TRAY_COLOR))
    _post_handle(tray, t, t["wall_m"], size[2], 0.0, (0.85, 0.75, 0.25, 1), "tray")
    tray.add_geom(name="tray_mass", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, 0.01],
                  size=[0.06, 0.05, 0.01], contype=0, conaffinity=0, mass=t["mass_kg"])
    return tray


def piece_name(bin_id: str, index: int) -> str:
    return f"piece_{bin_id}_{index}"


def build_spec(config: dict, catalog: dict) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.option.timestep = config["timestep_s"]
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    spec.compiler.autolimits = True

    world = spec.worldbody
    world.add_light(pos=[0.8, -1.2, 2.6], dir=[-0.15, 0.35, -1], diffuse=[0.9, 0.9, 0.9])
    world.add_light(pos=[0.8, 0.8, 2.2], dir=[0, -0.4, -1], diffuse=[0.55, 0.55, 0.55])
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                   size=[4, 4, 0.1], rgba=list(FLOOR_RGBA))

    rail = config["rail"]
    mid_x = (rail["x_min_m"] + rail["x_max_m"]) / 2
    world.add_geom(name="rail_beam", type=mujoco.mjtGeom.mjGEOM_BOX,
                   pos=[mid_x, rail["y_m"], 0.055],
                   size=[(rail["x_max_m"] - rail["x_min_m"]) / 2 + 0.15, 0.05, 0.055],
                   rgba=list(STEEL))

    cart = world.add_body(name="cart", pos=[config["cart"]["home_x_m"], rail["y_m"], 0])
    cart.add_joint(name="cart_x", type=mujoco.mjtJoint.mjJNT_SLIDE,
                   axis=[1, 0, 0], range=[rail["x_min_m"], rail["x_max_m"]],
                   damping=40.0, armature=1.0)
    px, py, pz = config["cart"]["plate_xyz_m"]
    plate_bottom = 0.115  # clears the rail beam (top at 0.11)
    cart.add_geom(name="cart_plate", type=mujoco.mjtGeom.mjGEOM_BOX,
                  pos=[0, 0, plate_bottom + pz / 2], size=[px / 2, py / 2, pz / 2],
                  mass=config["cart"]["mass_kg"], rgba=[0.60, 0.45, 0.15, 1])
    # two low skids beside the rail, clear of it (visual only)
    for side in (-1, 1):
        cart.add_geom(name=f"cart_skid{side}", type=mujoco.mjtGeom.mjGEOM_BOX,
                      pos=[0, side * (py / 2 - 0.015), 0.045],
                      size=[px / 2 - 0.06, 0.012, 0.02],
                      contype=0, conaffinity=0, rgba=list(STEEL))
    ax, ay, az = config["arm_base_xyz_m"]
    plate_top = plate_bottom + pz
    pedestal_h = az - plate_top
    cart.add_geom(name="pedestal", type=mujoco.mjtGeom.mjGEOM_BOX,
                  pos=[ax, ay, plate_top + pedestal_h / 2],
                  size=[0.055, 0.055, pedestal_h / 2], rgba=[0.30, 0.30, 0.32, 1],
                  mass=1.5)
    frame = cart.add_frame(pos=[ax, ay, az])
    arm_spec = mujoco.MjSpec.from_file(str(PANDA_XML))
    hand = next(b for b in arm_spec.bodies if b.name == "hand")
    hand.add_camera(name="wrist", pos=[0, 0, 0.115], xyaxes=[1, 0, 0, 0, -1, 0], fovy=70)
    # Fingertip adapters: the stock pads pinch a short piece only along its
    # top rim, so it rotates out of the grip under load. Two small pads per
    # finger protrude slightly past the stock face at two depths, giving a
    # stable two-line pinch with the lower line below the piece's CoM.
    for fname in ("left_finger", "right_finger"):
        fbody = next(b for b in arm_spec.bodies if b.name == fname)
        # channel jaw: center pad pinches the post, side nubs block the
        # post sliding out along the finger direction (x)
        fbody.add_geom(name=f"{fname}_pad0",
                       type=mujoco.mjtGeom.mjGEOM_BOX,
                       pos=[-0.0098, 0.0055, 0.050],
                       size=[0.0032, 0.009, 0.0032],
                       rgba=[0.15, 0.15, 0.16, 1],
                       friction=[2.0, 0.05, 0.005])
        for side in (-1, 1):
            fbody.add_geom(name=f"{fname}_nub{side}",
                           type=mujoco.mjtGeom.mjGEOM_BOX,
                           pos=[-0.012, 0.0055 + side * 0.008, 0.050],
                           size=[0.0028, 0.0016, 0.0032],
                           rgba=[0.15, 0.15, 0.16, 1],
                           friction=[2.0, 0.05, 0.005])
    spec.attach(arm_spec, frame=frame, prefix="")

    add_tray(spec, config, cart)

    shelf = config["shelf"]
    depth = shelf["depth_m"]
    surf_y = shelf["front_y_m"] + depth / 2
    for slot_id, slot in shelf["slots"].items():
        x = slot["x_m"]
        width = bin_size(config)[0] + 0.10
        world.add_geom(name=f"{slot_id}_board", type=mujoco.mjtGeom.mjGEOM_BOX,
                       pos=[x, surf_y, shelf["surface_z_m"] - 0.015],
                       size=[width / 2, depth / 2, 0.015], rgba=list(SHELF_GRAY))
        for side in (-1, 1):
            world.add_geom(name=f"{slot_id}_guide{side}", type=mujoco.mjtGeom.mjGEOM_BOX,
                           pos=[x + side * (width / 2 + 0.01), surf_y, shelf["surface_z_m"] + 0.015],
                           size=[0.01, depth / 2, 0.03], rgba=list(SHELF_GRAY))
        world.add_geom(name=f"{slot_id}_back", type=mujoco.mjtGeom.mjGEOM_BOX,
                       pos=[x, shelf["front_y_m"] + depth + 0.01, shelf["surface_z_m"] + 0.06],
                       size=[width / 2, 0.012, 0.12], rgba=list(SHELF_GRAY))
        for sx in (-1, 1):
            world.add_geom(name=f"{slot_id}_leg{sx}", type=mujoco.mjtGeom.mjGEOM_BOX,
                           pos=[x + sx * (width / 2 - 0.02), surf_y, (shelf["surface_z_m"] - 0.03) / 2],
                           size=[0.02, depth / 2 - 0.02, (shelf["surface_z_m"] - 0.03) / 2],
                           rgba=list(STEEL))

    pk = config["park"]
    world.add_geom(name="park_plate", type=mujoco.mjtGeom.mjGEOM_BOX,
                   pos=[pk["x_m"], pk["y_m"], pk["surface_z_m"] - 0.012],
                   size=[0.13, 0.12, 0.012], rgba=list(PARK_RGBA))
    world.add_geom(name="park_leg", type=mujoco.mjtGeom.mjGEOM_BOX,
                   pos=[pk["x_m"], pk["y_m"], pk["surface_z_m"] / 2 - 0.012],
                   size=[0.03, 0.03, pk["surface_z_m"] / 2 - 0.012], rgba=list(STEEL))
    for side in (-1, 1):
        world.add_geom(name=f"park_rim{side}", type=mujoco.mjtGeom.mjGEOM_BOX,
                       pos=[pk["x_m"] + side * 0.125, pk["y_m"], pk["surface_z_m"] + 0.008],
                       size=[0.006, 0.11, 0.008], rgba=list(PARK_RGBA))

    rc = config["reception"]
    world.add_geom(name="reception_top", type=mujoco.mjtGeom.mjGEOM_BOX,
                   pos=[rc["x_m"], rc["y_m"], rc["surface_z_m"] - 0.015],
                   size=[rc["size_xy_m"][0] / 2, rc["size_xy_m"][1] / 2, 0.015],
                   rgba=list(TABLE_RGBA))
    for sx in (-1, 1):
        for sy in (-1, 1):
            world.add_geom(name=f"table_leg{sx}{sy}", type=mujoco.mjtGeom.mjGEOM_BOX,
                           pos=[rc["x_m"] + sx * (rc["size_xy_m"][0] / 2 - 0.02),
                                rc["y_m"] + sy * (rc["size_xy_m"][1] / 2 - 0.02),
                                rc["surface_z_m"] / 2 - 0.02],
                           size=[0.02, 0.02, rc["surface_z_m"] / 2 - 0.02], rgba=list(STEEL))

    for i, entry in enumerate(catalog["bins"]):
        add_bin(spec, config, entry, BIN_COLORS[i % len(BIN_COLORS)])
        for k in range(entry["units"]):
            add_piece(spec, entry["sku"], catalog["skus"][entry["sku"]],
                      piece_name(entry["id"], k))

    # Scenario fixtures: real dynamic bodies resting off-cell on the floor
    # unless the scenario places them on a surface at episode start.
    for k, name in enumerate(("reception_obstacle", "park_obstacle")):
        body = world.add_body(name=name, pos=[-0.6 - 0.25 * k, -2.5, 0.05])
        body.add_freejoint(name=f"{name}_free")
        body.add_geom(name=f"{name}_g", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[0.06, 0.05, 0.05], mass=0.9, rgba=[0.80, 0.15, 0.15, 1])

    world.add_camera(name="overview", pos=[0.85, -1.75, 1.35],
                     xyaxes=[0.95, 0.31, 0, -0.20, 0.62, 0.75], fovy=55)
    world.add_camera(name="reception_cam",
                     pos=[rc["x_m"], rc["y_m"] - 0.02, rc["surface_z_m"] + 0.62],
                     xyaxes=[1, 0, 0, 0, 1, 0], fovy=55)

    act = spec.add_actuator()
    act.trntype = mujoco.mjtTrn.mjTRN_JOINT
    act.target = "cart_x"
    act.name = "cart_drive"
    act.set_to_position(kp=400, kv=60)
    return spec
