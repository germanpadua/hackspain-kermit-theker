import mujoco
import numpy as np

from warehouse.contracts import PoseObservation, Snapshot
from warehouse.scene import CAMERA_HEIGHT, CAMERA_WIDTH, build_scene, load_config


class Simulation:
    def __init__(self, seed=7, transport="idealized", scenario="nominal", offset_m=None):
        if transport != "idealized":
            raise ValueError("CONTACT is not implemented or validated; select --transport idealized explicitly")
        self.config = load_config("scene")
        self.catalog = load_config("inventory")
        self.seed = seed
        self.scenario = scenario
        self.model = mujoco.MjModel.from_xml_string(build_scene(self.config, self.catalog["bins"]))
        self.data = mujoco.MjData(self.model)
        self.renderer = None
        self.attached_bin_id = None
        self.attachment_offset = None
        self.blocked_home = False
        self.fault = None
        self.frozen_axes = False
        self.present = {entry["id"] for entry in self.catalog["bins"]}
        self.joints = [self.model.joint(axis).qposadr[0] for axis in ("x", "y", "z")]
        self.dofs = [self.model.joint(axis).dofadr[0] for axis in ("x", "y", "z")]
        self.data.qpos[self.joints] = (0.0, self.config["retracted_y_m"], 0.20)
        self.command_axes(0.0, 0.20, self.config["retracted_y_m"])
        rng = np.random.default_rng(seed)
        if offset_m is None:
            offset_m = 0.01 if scenario == "lateral" else 0.0
        if not np.isfinite(offset_m) or abs(offset_m) > 0.1:
            raise ValueError("Scenario offset must be finite and within +/-100 mm")
        self.offset_m = offset_m if scenario != "lateral" else float(rng.choice((-1, 1))) * abs(offset_m)
        self.data.mocap_pos[self.mocap_id("box-1"), 0] += self.offset_m
        if scenario == "missing":
            self.present.remove("box-1")
            self.data.mocap_pos[self.mocap_id("box-1"), 2] = -5
        if scenario == "empty":
            self.set_visible_content("box-1", 0)
        mujoco.mj_forward(self.model, self.data)

    def mocap_id(self, bin_id):
        return int(self.model.body(bin_id).mocapid[0])

    def command_axes(self, x, z, y):
        self.data.ctrl[:] = (x, z, y)

    def snapshot(self):
        return Snapshot(float(self.data.time), tuple(self.data.qpos[self.joints]),
                        tuple(self.data.qvel[self.dofs]), self.attached_bin_id,
                        not self.blocked_home, self.fault)

    def step(self):
        if self.frozen_axes:
            self.data.time += self.model.opt.timestep
            return
        mujoco.mj_step(self.model, self.data)
        if self.attached_bin_id:
            self.data.mocap_pos[self.mocap_id(self.attached_bin_id)] = (
                self.data.qpos[self.joints] + self.attachment_offset
            )
        mujoco.mj_forward(self.model, self.data)
        if not np.all(np.isfinite(self.data.qpos)) or np.max(np.abs(self.data.qvel)) > 5:
            self.fault = "AXIS_INSTABILITY"

    def observe_oracle(self, bin_id):
        if bin_id not in self.present:
            return None
        pos = self.data.mocap_pos[self.mocap_id(bin_id)]
        return PoseObservation(bin_id, tuple(pos), 0.0, 1.0, float(self.data.time), "ORACLE")

    def attach(self, bin_id):
        observation = self.observe_oracle(bin_id)
        if not observation or self.attached_bin_id:
            raise ValueError("PICK_MISSING_OR_ALREADY_ATTACHED")
        offset = np.asarray(observation.xyz_m) - self.data.qpos[self.joints]
        clearance = self.config["pick_clearance_m"]
        if np.max(np.abs(offset - (0, 0, clearance))) > 0.006:
            raise ValueError("PICK_ALIGNMENT_OUT_OF_RANGE")
        self.attachment_offset = offset
        self.attached_bin_id = bin_id

    def detach(self):
        if not self.attached_bin_id:
            raise ValueError("PLACE_WITHOUT_BIN")
        self.attached_bin_id = None
        self.attachment_offset = None

    def set_visible_content(self, bin_id, patches):
        if not 0 <= patches <= 8:
            raise ValueError("Visual patches must be in [0, 8]; these are not stock units")
        for index in range(8):
            self.model.geom(f"{bin_id}_content_{index}").rgba[3] = float(index < patches)

    def confirm_material_removed(self, bin_id):
        patches = 0 if self.scenario == "empty" else 2
        self.set_visible_content(bin_id, patches)

    def inject_return_obstacle(self, home_xyz):
        self.blocked_home = True
        self.data.mocap_pos[self.mocap_id("return_obstacle")] = home_xyz
        mujoco.mj_forward(self.model, self.data)

    def render(self, camera="reception"):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=CAMERA_HEIGHT, width=CAMERA_WIDTH)
        self.renderer.update_scene(self.data, camera=camera)
        image = self.renderer.render().copy()
        if camera == "reception" and self.scenario == "unknown_stock":
            image[:] = 0
        return image

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
