"""Industrial context and readable signs; no changes to manipulation state."""

import cv2
import mujoco
import numpy as np

FRONT = [0.70710678, 0.70710678, 0, 0]
WHITE = [0.82, 0.85, 0.87, 1]
METAL = [0.38, 0.43, 0.47, 1]
BLUE = [0.10, 0.22, 0.32, 1]


def label(spec, body, name, pos, size, lines, front=True, quarter_turn=False):
    width, height = 1024, max(128, 104 * len(lines))
    image = np.full((height, width, 3), 232, dtype=np.uint8)
    for row, text in enumerate(lines):
        scale = min(1.8, 920 / max(1, cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0]))
        cv2.putText(image, text, (36, 80 + row * 104),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (28, 34, 38), 3, cv2.LINE_AA)
    ok, png = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Could not render industrial label")
    spec.assets[f"{name}.png"] = png.tobytes()
    spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D,
                     file=f"{name}.png")
    material = spec.add_material(name=name, texrepeat=[1, 1], emission=0.25)
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name
    body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos,
                  size=[size[0] / 2, size[1] / 2, 0.0002],
                  quat=FRONT if front else (
                      [0.70710678, 0, 0, 0.70710678] if quarter_turn else [1, 0, 0, 0]),
                  material=name, contype=0, conaffinity=0, mass=0)


def box(body, name, pos, size, rgba=METAL, collision=True):
    body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
                  pos=pos, size=size, rgba=rgba,
                  contype=int(collision), conaffinity=int(collision))


def add_context(spec, config, catalog):
    world = spec.worldbody
    cell = config["industrial"]
    x, y, z = cell["bench_xyz_m"]
    world.add_light(pos=[x, y - .2, 1.65], dir=[0, .1, -1],
                    diffuse=[.8, .8, .8], castshadow=False)
    box(world, "assembly_bench", [x, y, z - .025], [.36, .24, .025], WHITE)
    box(world, "assembly_cabinet", [x, y + .02, z / 2 - .03],
        [.32, .19, z / 2 - .03], BLUE)
    for side in (-1, 1):
        box(world, f"cabinet_handle_{side}", [x + side * .16, y - .179, .45],
            [.07, .007, .008], WHITE)
    housing = world.add_body(name="open_transmission_housing", pos=[x, y, z])
    box(housing, "housing_floor", [0, 0, .015], [.245, .145, .015], WHITE)
    for side in (-1, 1):
        box(housing, f"housing_side_{side}", [side * .237, 0, .073],
            [.012, .145, .043], WHITE)
        height = .008 if side < 0 else .043
        box(housing, f"housing_end_{side}", [0, side * .137, .03 + height],
            [.245, .012, height], WHITE)
    for j, cx in enumerate((-.11, .11)):
        housing.add_geom(name=f"housing_bearing_seat_{j}",
                         type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                         pos=[cx, 0, .033], size=[.069, .003, 0], rgba=BLUE)
        for k in range(24):
            a, b = k * 2 * np.pi / 24, (k + 1) * 2 * np.pi / 24
            housing.add_geom(name=f"housing_bore_{j}_{k}",
                             type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                             fromto=[cx + .06 * np.cos(a), .06 * np.sin(a), .049,
                                     cx + .06 * np.cos(b), .06 * np.sin(b), .049],
                             size=[.009, 0, 0], rgba=WHITE)
        for angle in np.linspace(0, 2 * np.pi, 6, endpoint=False):
            housing.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                             pos=[cx + .085 * np.cos(angle), .085 * np.sin(angle), .034],
                             size=[.007, .004, 0], rgba=[.1, .12, .14, 1])
    label(spec, world, "assembly_scope", [x, y - .242, z - .04], [.65, .07],
          ["OPEN REDUCER / ASSEMBLY OUT OF SCOPE"])

    for px in (-.65, 1.38, 2.4):
        box(world, f"gantry_{px}", [px, .73, .73], [.022, .022, .73], BLUE)
    box(world, "cell_header_frame", [.88, .74, 1.47], [1.56, .025, .14], BLUE)
    label(spec, world, "cell_header", [.88, .712, 1.49], [2.95, .20],
          [cell["title"], "RECIPE > PICK > REPLENISH > VERIFY > DELIVER"])

    order = catalog["order"]
    lines = [order["id"], order["workstation"]]
    lines += [f"{line['qty']} x {catalog['skus'][line['sku']]['reference']}"
              for line in order["lines"]]
    lines += ["KIT PREPARATION ONLY"]
    label(spec, world, "work_order", [2.0, .69, 1.16], [.66, .43], lines)
    label(spec, world, "reserve_notice", [0, .47, 1.12], [.43, .16],
          ["S1 / TWO-TOTE FIFO", "FRONT -> RESERVE"])
    label(spec, world, "process_scope", [.65, .60, 1.07], [.64, .23],
          ["CONTACT SIMULATION", "RGB-D PRODUCT MARKERS", "ORACLE LOGISTICS"])

    rc = config["reception"]
    label(spec, world, "kit_ready_sign",
          [rc["x_m"], rc["y_m"] - rc["size_xy_m"][1] / 2 - .001,
           rc["surface_z_m"] - .065], [.33, .10],
          [cell["delivery_zone"], cell["station"]])
    for side in (-1, 1):
        box(world, f"kit_ready_edge_{side}",
            [rc["x_m"] + side * (rc["size_xy_m"][0] / 2 - .006),
             rc["y_m"], rc["surface_z_m"] + .0003],
            [.005, rc["size_xy_m"][1] / 2, .0002], [.15, .60, .46, 1], False)
    pk = config["park"]
    label(spec, world, "empty_tote_sign",
          [pk["x_m"], pk["y_m"] - .121, pk["surface_z_m"] - .06],
          [.25, .08], ["EMPTY TOTE", "RETURN BUFFER"])
    for side in (-1, 1):
        box(world, f"safety_line_{side}", [.9, -.9 if side < 0 else .97, .001],
            [1.7, .025, .001], [.9, .67, .16, 1], False)
    label(spec, world, "floor_process", [.9, -.78, .003], [1.3, .18],
          ["AUTONOMOUS CELL / KEEP CLEAR"], front=False)
    box(world, "stack_light_post", [2.35, .70, 1.07], [.014, .014, .33])
    for i, color in enumerate(([.15, .6, .45, 1], [.8, .5, .1, 1], [.7, .18, .18, 1])):
        world.add_geom(name=f"stack_light_{i}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                       pos=[2.35, .70, 1.16 + i * .06], size=[.027, .026, 0],
                       rgba=color)
