"""MuJoCo adapter: the only object that owns MjModel/MjData.

The controller receives Snapshot/Observation objects from this layer and from
the perception layer; it never indexes qpos/xpos of world objects itself.
"""
import numpy as np
import mujoco

from warehouse.scene import load_config
from orderpick.contracts import Snapshot
from orderpick.scene import ROOT, bin_origin, bin_size, build_spec, piece_name, tray_pocket_world

HOME = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
STOW = np.array([0.35, -0.55, 0.0, -2.35, 0.0, 1.95, 0.75])
GRASP_OFFSET = np.array([0, 0, 0.1035])
DOWN = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)
# Fingers point -X (along the rail), closing axis +Y, hand body held at +Y side.
# Keeps the hand/wrist bulk outside the shelf footprint while fingers enter a bin.
DOWN_X = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float)


def yaw_mat(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ DOWN


class CellSim:
    def __init__(self, seed=7, scenario="nominal"):
        self.config = load_config("cell")
        self.catalog = load_config("catalog")
        self.seed = seed
        self.scenario = scenario
        spec = build_spec(self.config, self.catalog)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.renderer = None
        self.renderer_depth = None
        self.fault = None
        self._index()
        self._scatter(seed)
        self._apply_scenario()
        mujoco.mj_forward(self.model, self.data)

    def _index(self):
        m = self.model
        self.arm_qadr = np.array([int(m.joint(f"joint{i}").qposadr[0]) for i in range(1, 8)])
        self.arm_vadr = np.array([int(m.joint(f"joint{i}").dofadr[0]) for i in range(1, 8)])
        self.arm_range = np.stack([m.joint(f"joint{i}").range for i in range(1, 8)])
        self.fing_qadr = np.array([int(m.joint(f"finger_joint{i}").qposadr[0]) for i in (1, 2)])
        self.fing_vadr = np.array([int(m.joint(f"finger_joint{i}").dofadr[0]) for i in (1, 2)])
        self.cart_qadr = int(m.joint("cart_x").qposadr[0])
        self.cart_vadr = int(m.joint("cart_x").dofadr[0])
        self.hand_id = int(m.body("hand").id)
        self.bin_ids = {e["id"]: int(m.body(e["id"]).id) for e in self.catalog["bins"]}
        self.bin_qadr = {e["id"]: int(m.joint(f"{e['id']}_free").qposadr[0])
                         for e in self.catalog["bins"]}
        self.tray_id = int(m.body("tray").id)
        self.tray_qadr = int(m.joint("tray_free").qposadr[0])
        self.piece_qadr = {}
        self.piece_sku = {}
        self.piece_bin = {}
        for e in self.catalog["bins"]:
            for k in range(e["units"]):
                name = piece_name(e["id"], k)
                self.piece_qadr[name] = int(m.joint(f"{name}_free").qposadr[0])
                self.piece_sku[name] = e["sku"]
                self.piece_bin[name] = e["id"]
        self.gripper_act = int(m.actuator("actuator8").id)
        self.cart_act = int(m.actuator("cart_drive").id)
        self.arm_acts = np.array([int(m.actuator(f"actuator{i}").id) for i in range(1, 8)])
        self.obstacle_qadr = {n: int(m.joint(f"{n}_free").qposadr[0])
                              for n in ("reception_obstacle", "park_obstacle")}

    # ---------- initialization ----------
    def _scatter(self, seed):
        """Place pieces inside their bins with seeded pose jitter."""
        rng = np.random.default_rng(seed)
        v = self.catalog["variability"]
        d = self.data
        for e in self.catalog["bins"]:
            origin = bin_origin(self.config, e["slot"], e["depth_row"])
            wall = self.config["bin"]["wall_m"]
            sku = self.catalog["skus"][e["sku"]]
            n = e["units"]
            size = self.config["bin"]["size_xyz_m"]
            if sku["shape"] == "cylinder":
                rad = sku["radius_m"]
            else:
                rad = max(sku["size_xy_m"]) / 2
            margin = 0.006
            max_dx = size[0] / 2 - wall - rad - margin
            max_dy = size[1] / 2 - wall - rad - margin
            # Front-access band: pieces sit near the open front so the hand's
            # tail overhangs the shelf edge; |dy| keeps clear of the pull post.
            front_y = -0.033
            if n > 1:
                grid = [(-max_dx * 0.85, front_y), (max_dx * 0.85, front_y)]
            else:
                grid = [(0.0, front_y)]
            jlim_x = min(v["piece_xy_jitter_m"], max(0.0, max_dx / 2 - 0.004))
            jlim_y = min(0.004, max(0.0, front_y + max_dy - (2 * rad + 0.02)))
            for k in range(n):
                name = piece_name(e["id"], k)
                adr = self.piece_qadr[name]
                dx, dy = grid[k % len(grid)]
                jx = rng.uniform(-jlim_x, jlim_x)
                jy = rng.uniform(-jlim_y, jlim_y)
                yaw = rng.uniform(-v["piece_yaw_jitter_rad"], v["piece_yaw_jitter_rad"])
                d.qpos[adr:adr + 7] = [
                    origin[0] + dx + jx, origin[1] + dy + jy,
                    origin[2] + max(wall, 0.010) + 0.0005,
                    np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
                mscale = 1.0 + rng.uniform(-v["mass_jitter_frac"], v["mass_jitter_frac"])
                bid = int(self.model.body(name).id)
                self.model.body_mass[bid] *= mscale
        # tray rests on the cart plate inside the pocket lips
        tp = tray_pocket_world(self.config, self.config["cart"]["home_x_m"])
        d.qpos[self.tray_qadr:self.tray_qadr + 7] = [tp[0], tp[1],
                                                     tp[2] + self.config["tray"]["wall_m"] / 2,
                                                     1, 0, 0, 0]
        d.qpos[self.arm_qadr] = HOME
        d.qpos[self.fing_qadr] = [0.04, 0.04]
        d.qpos[self.cart_qadr] = self.config["cart"]["home_x_m"]
        d.ctrl[self.arm_acts] = HOME
        d.ctrl[self.gripper_act] = self.config["grasp"]["open_ctrl"]
        d.ctrl[self.cart_act] = self.config["cart"]["home_x_m"]

    def _apply_scenario(self):
        """Place the scenario fixture body on the target surface (init only)."""
        s = self.scenario
        if s == "reception_blocked":
            rc = self.config["reception"]
            self._set_obstacle("reception_obstacle",
                               [rc["x_m"], rc["y_m"], rc["surface_z_m"] + 0.05])
        elif s == "park_blocked":
            pk = self.config["park"]
            self._set_obstacle("park_obstacle",
                               [pk["x_m"], pk["y_m"], pk["surface_z_m"] + 0.05])

    def _set_obstacle(self, name, pos):
        adr = self.obstacle_qadr[name]
        self.data.qpos[adr:adr + 7] = [*pos, 1, 0, 0, 0]

    # ---------- commands (called by Motion/adapters) ----------
    def command_cart(self, x):
        self.data.ctrl[self.cart_act] = x

    def command_arm(self, q7):
        self.data.ctrl[self.arm_acts] = q7

    def command_gripper(self, ctrl):
        self.data.ctrl[self.gripper_act] = ctrl

    def step(self, n=1):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
        if not np.all(np.isfinite(self.data.qpos)):
            self.fault = "SIM_DIVERGED"

    # ---------- adapters ----------
    def grasp_point(self, data=None):
        d = data if data is not None else self.data
        hid = self.hand_id
        return d.xpos[hid] + d.xmat[hid].reshape(3, 3) @ GRASP_OFFSET

    def snapshot(self):
        d = self.data
        return Snapshot(
            sim_time_s=float(d.time),
            cart_x_m=float(d.qpos[self.cart_qadr]),
            cart_vel_m_s=float(d.qvel[self.cart_vadr]),
            arm_qpos=tuple(float(q) for q in d.qpos[self.arm_qadr]),
            arm_qvel=tuple(float(v) for v in d.qvel[self.arm_vadr]),
            gripper_open_m=float(np.mean(d.qpos[self.fing_qadr])),
            gripper_force_n=float(d.actuator_force[self.gripper_act]),
            ee_xyz_m=tuple(self.grasp_point()),
            ee_xmat=tuple(d.xmat[self.hand_id]),
            fault=self.fault)

    def render(self, camera, depth=False):
        w, h = (self.config["cameras"]["wrist_w"], self.config["cameras"]["wrist_h"]) \
            if camera == "wrist" else (self.config["cameras"]["overview_w"],
                                       self.config["cameras"]["overview_h"])
        if depth:
            if self.renderer_depth is None:
                self.renderer_depth = mujoco.Renderer(
                    self.model, height=max(h, 360), width=max(w, 640))
                self.renderer_depth.enable_depth_rendering()
            self.renderer_depth.update_scene(self.data, camera=camera)
            return self.renderer_depth.render().copy()
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=max(h, 360), width=max(w, 640))
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    # ---------- evaluator-only truth API ----------
    def truth_body_pos(self, name):
        return self.data.xpos[int(self.model.body(name).id)].copy()

    def truth_body_quat(self, name):
        return self.data.xquat[int(self.model.body(name).id)].copy()

    def truth_free_z(self, name):
        return float(self.data.qpos[self.piece_qadr[name] + 2]) if name in self.piece_qadr \
            else float(self.data.qpos[self.tray_qadr + 2])

    def contacts_of(self, body_name):
        gid = {int(g) for g in self.model.body(body_name).geomadr
               for g in range(int(g), int(g) + int(self.model.body(body_name).geomnum))}
        out = []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 in gid or c.geom2 in gid:
                other = c.geom2 if c.geom1 in gid else c.geom1
                out.append(other)
        return [self.model.geom(g).name or f"geom{g}" for g in out]

    def close(self):
        for r in (self.renderer, self.renderer_depth):
            if r is not None:
                r.close()
        self.renderer = self.renderer_depth = None
