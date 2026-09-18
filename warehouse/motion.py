import numpy as np

from warehouse.contracts import Snapshot, XYZ


class Motion:
    def __init__(self, sim):
        self.sim = sim
        self.config = sim.config
        self.goal = np.array(sim.snapshot().axes_xyz_m)
        self.setpoint = self.goal.copy()
        self.velocity = np.zeros(3)
        self.started_at = 0.0
        self.stable_steps = 0
        self.active = False
        self.error = None

    def command(self, xyz: XYZ, snapshot: Snapshot, docking=False):
        target = np.asarray(xyz, dtype=float)
        if not np.all(np.isfinite(target)) or np.any(target < (-0.25, -0.49, 0.1)) or np.any(target > (1.4, 0.3, 1.05)):
            raise ValueError("AXIS_TARGET_OUT_OF_RANGE")
        current = np.asarray(snapshot.axes_xyz_m)
        extended = max(current[1], target[1]) > self.config["retracted_y_m"] + 0.005
        lateral_move = abs(target[0] - current[0]) > 0.003
        vertical_move = abs(target[2] - current[2]) > 0.003
        if extended and (lateral_move or (vertical_move and not docking)):
            raise ValueError("TRANSPORT_REQUIRES_RETRACTED_FORK")
        if docking and (lateral_move or abs(target[1] - current[1]) > 0.003):
            raise ValueError("DOCKING_ONLY_ALLOWS_VERTICAL_MOTION")
        self.goal = target
        self.setpoint = current.copy()
        self.velocity[:] = 0
        self.started_at = snapshot.sim_time_s
        self.stable_steps = 0
        self.active = True
        self.error = None

    def tick(self, snapshot: Snapshot):
        if not self.active:
            return
        dt = self.config["timestep_s"] * self.config["control_steps"]
        acceleration = self.config["axis_acceleration_m_s2"]
        distance = self.goal - self.setpoint
        desired = np.sign(distance) * np.minimum(self.config["axis_speed_m_s"], np.sqrt(2 * acceleration * np.abs(distance)))
        self.velocity += np.clip(desired - self.velocity, -acceleration * dt, acceleration * dt)
        delta = self.velocity * dt
        reached = np.abs(delta) >= np.abs(distance)
        self.setpoint = np.where(reached, self.goal, self.setpoint + delta)
        self.velocity[reached] = 0
        x, y, z = self.setpoint
        self.sim.command_axes(x, z, y)
        position_ok = np.max(np.abs(np.asarray(snapshot.axes_xyz_m) - self.goal)) < 0.002
        velocity_ok = np.max(np.abs(snapshot.velocities_xyz_m_s)) < 0.008
        self.stable_steps = self.stable_steps + 1 if position_ok and velocity_ok else 0
        if self.stable_steps >= 5:
            self.active = False
        elif snapshot.sim_time_s - self.started_at > self.config["motion_timeout_s"]:
            self.error = "MOTION_TIMEOUT"
            self.stop(snapshot)

    def stop(self, snapshot):
        self.active = False
        self.goal = np.array(snapshot.axes_xyz_m)
        self.setpoint = self.goal.copy()
        self.velocity[:] = 0
        x, y, z = self.goal
        self.sim.command_axes(x, z, y)

    def motion_status(self):
        return "FAILED" if self.error else "MOVING" if self.active else "SETTLED"
