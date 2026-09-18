import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CAMERA_WIDTH = 480
CAMERA_HEIGHT = 360
CAMERA_HEIGHT_M = 1.15
CAMERA_FOVY_DEG = 45.0


def load_config(name: str) -> dict:
    return json.loads((ROOT / "config" / f"{name}.json").read_text())


def vector(values) -> str:
    return " ".join(str(float(value)) for value in values)


def box(parent, name, pos, dimensions, rgba, mass=0.1):
    return ET.SubElement(parent, "geom", name=name, type="box", pos=vector(pos),
                         size=vector(np.asarray(dimensions) / 2), rgba=vector(rgba), mass=str(mass))


def build_scene(config: dict, bins: list[dict]) -> str:
    if config["bin_size_xyz_m"] != [0.30, 0.22, 0.12]:
        raise ValueError("MVP supports only the validated 0.30 x 0.22 x 0.12 m geometry")
    root = ET.Element("mujoco", model="warehouse_ORACLE_IDEALIZED")
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(root, "option", timestep=str(config["timestep_s"]), integrator="implicitfast")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth=str(CAMERA_WIDTH), offheight=str(CAMERA_HEIGHT))
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "geom", contype="0", conaffinity="0")
    ET.SubElement(default, "joint", damping="20", armature="0.02")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0.6 -0.4 2.5", dir="0 0 -1", diffuse="0.8 0.8 0.8")
    ET.SubElement(world, "light", pos="1.3 0.4 2", dir="0 0 -1", diffuse="0.5 0.5 0.5")
    box(world, "floor", (0.5, 0, -0.035), (2.6, 2.1, 0.05), (0.13, 0.16, 0.20, 1))
    box(world, "rail", (0.55, -0.70, 0.055), (1.9, 0.06, 0.07), (0.3, 0.35, 0.4, 1))
    supports = {**config["slots"], "reception": config["reception_xyz_m"]}
    for name, (x, y, z) in supports.items():
        for side in (-1, 1):
            box(world, f"{name}_support_{side}", (x + side * 0.125, y, z - 0.055),
                (0.055, 0.29, 0.03), (0.40, 0.48, 0.53, 1))
        box(world, f"{name}_back", (x, y + 0.17, z - 0.06),
            (0.44, 0.025, 0.055), (0.32, 0.40, 0.45, 1))
    for x in (-0.24, 0.27, 0.80):
        box(world, f"upright_{x}", (x, 0.36, 0.48), (0.025, 0.04, 0.96), (0.38, 0.43, 0.5, 1))
    for side in (-1, 1):
        box(world, f"table_leg_{side}", (1.12 + side * 0.19, 0.3, 0.15),
            (0.035, 0.035, 0.30), (0.38, 0.43, 0.5, 1))
    xbody = ET.SubElement(world, "body", name="axis_x", gravcomp="1")
    ET.SubElement(xbody, "joint", name="x", type="slide", axis="1 0 0", range="-0.25 1.4")
    box(xbody, "mast", (0, -0.73, 0.53), (0.06, 0.06, 1.06), (0.8, 0.6, 0.15, 1), 1.0)
    zbody = ET.SubElement(xbody, "body", name="axis_z", gravcomp="1")
    ET.SubElement(zbody, "joint", name="z", type="slide", axis="0 0 1", range="0.1 1.05")
    box(zbody, "carrier", (0, -0.48, -0.034), (0.32, 0.30, 0.025), (0.65, 0.48, 0.1, 1), 0.6)
    ybody = ET.SubElement(zbody, "body", name="axis_y", gravcomp="1")
    ET.SubElement(ybody, "joint", name="y", type="slide", axis="0 1 0", range="-0.49 0.3")
    for side in (-1, 1):
        box(ybody, f"fork_{side}", (side * 0.065, -0.015, -0.006),
            (0.035, 0.25, 0.012), (0.85, 0.68, 0.22, 1), 0.15)
    actuators = ET.SubElement(root, "actuator")
    for axis in ("x", "z", "y"):
        ET.SubElement(actuators, "position", name=f"drive_{axis}", joint=axis, kp="1800", kv="100")
    for entry in bins:
        name = entry["id"]
        body = ET.SubElement(world, "body", name=name, mocap="true",
                             pos=vector(config["slots"][entry["home_slot_id"]]))
        box(body, f"{name}_base", (0, 0, 0.006), (0.30, 0.22, 0.012), (0.12, 0.62, 0.65, 1))
        for side in (-1, 1):
            box(body, f"{name}_wall_x_{side}", (side * 0.145, 0, 0.06),
                (0.01, 0.22, 0.12), (0.12, 0.28, 0.55, 1))
            box(body, f"{name}_wall_y_{side}", (0, side * 0.105, 0.06),
                (0.28, 0.01, 0.12), (0.12, 0.28, 0.55, 1))
            box(body, f"{name}_skid_{side}", (side * 0.125, 0, -0.02),
                (0.03, 0.20, 0.04), (0.2, 0.3, 0.4, 1))
        box(body, f"{name}_marker", (0, -0.111, 0.07), (0.09, 0.003, 0.045), (0.15, 0.9, 0.2, 1))
        for index in range(8):
            box(body, f"{name}_content_{index}", (-0.09 + 0.06 * (index % 4), -0.04 + 0.08 * (index // 4), 0.017),
                (0.048, 0.060, 0.01), (0.96, 0.32, 0.025, 1))
    blocker = ET.SubElement(world, "body", name="return_obstacle", mocap="true", pos="0 0 -5")
    box(blocker, "blocked_slot", (0, 0, 0.07), (0.29, 0.21, 0.14), (0.8, 0.12, 0.12, 1))
    ET.SubElement(world, "camera", name="front", pos="0.55 -1.9 0.65", xyaxes="1 0 0 0 0 1", fovy="45")
    rx, ry, _ = config["reception_xyz_m"]
    ET.SubElement(world, "camera", name="reception", pos=vector((rx, ry, CAMERA_HEIGHT_M)),
                  xyaxes="1 0 0 0 1 0", fovy=str(CAMERA_FOVY_DEG))
    return ET.tostring(root, encoding="unicode")


def stock_roi(config: dict) -> tuple[int, int, int, int]:
    plane_z = config["reception_xyz_m"][2] + 0.017
    focal_px = CAMERA_HEIGHT / (2 * np.tan(np.deg2rad(CAMERA_FOVY_DEG / 2)))
    scale = focal_px / (CAMERA_HEIGHT_M - plane_z)
    half_x, half_y = int(0.12 * scale), int(0.075 * scale)
    return CAMERA_WIDTH // 2 - half_x, CAMERA_HEIGHT // 2 - half_y, 2 * half_x, 2 * half_y
