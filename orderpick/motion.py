"""Non-blocking motion runner: cart drives, joint-space arm moves, gripper.

Every command goes through a queued operation with a ramp profile, a settle
check and a simulated-time timeout. Nothing teleports; the cart joint is
position-actuated and the arm tracks interpolated joint setpoints.
"""
import numpy as np
import mujoco

from orderpick.sim import GRASP_OFFSET, STOW


class OpStatus:
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"


class Motion:
    """Drives ctrl[] of the cell. One active operation at a time."""

    def __init__(self, sim):
        self.sim = sim
        self.cfg = sim.config["motion"]
        self.op = None
        self.error = None

    @property
    def active(self):
        return self.op is not None

    @property
    def active_label(self):
        if isinstance(self.op, (_ArmMove, _ArmServo, _GripMove)):
            return self.op.label
        return ""

    # ---- public commands -------------------------------------------------
    def drive_cart(self, x_target):
        self.error = None
        self.op = _CartMove(self.sim, float(x_target))

    def move_arm(self, q_target, duration_s=None, label="", speed_scale=1.0):
        self.error = None
        self.op = _ArmMove(self.sim, np.asarray(q_target, dtype=float),
                           duration_s, label, speed_scale)

    def servo_to(self, pos, mat, speed_m_s=0.10, label="", dq_clip=0.025):
        """Cartesian straight-line move of the grasp point (resolved-rate)."""
        self.error = None
        self.op = _ArmServo(self.sim, np.asarray(pos, dtype=float),
                            np.asarray(mat, dtype=float), speed_m_s, label,
                            dq_clip)

    def grip(self, ctrl, duration_s=0.45, label=""):
        self.error = None
        self.op = _GripMove(self.sim, float(ctrl), duration_s, label)

    def stow(self):
        self.move_arm(STOW, duration_s=1.4, label="STOW")

    def stop(self):
        if isinstance(self.op, (_ArmMove, _ArmServo)):
            self.sim.command_arm(np.asarray(self.sim.data.qpos[self.sim.arm_qadr]))
        if isinstance(self.op, _CartMove):
            self.sim.command_cart(self.sim.data.qpos[self.sim.cart_qadr])
        self.op = None

    # ---- per-tick advance -------------------------------------------------
    def tick(self, snapshot):
        if self.error:
            # A latched error must release any queued op, not hold it forever —
            # otherwise run() spins indefinitely on an op that never ticks.
            self.op = None
            return
        if self.op is None:
            return
        status = self.op.tick(snapshot)
        if status == OpStatus.FAILED:
            self.error = self.op.error or "MOTION_FAILED"
            self.op = None
        elif status == OpStatus.DONE:
            self.op = None

    def status(self):
        return "FAILED" if self.error else "MOVING" if self.op else "SETTLED"


class _CartMove:
    def __init__(self, sim, target):
        self.sim = sim
        self.target = target
        rail = sim.config["rail"]
        if not rail["x_min_m"] - 1e-6 <= target <= rail["x_max_m"] + 1e-6:
            raise ValueError("CART_TARGET_OUT_OF_RANGE")
        self.speed = rail["speed_m_s"]
        self.accel = rail["accel_m_s2"]
        self.setpoint = float(sim.data.qpos[sim.cart_qadr])
        self.vel = 0.0
        self.stable = 0
        self.error = None
        self.t0 = None

    def tick(self, s):
        if self.t0 is None:
            self.t0 = s.sim_time_s
        dt = self.sim.config["timestep_s"] * self.sim.config["control_steps"]
        dist = self.target - self.setpoint
        desired = np.sign(dist) * min(self.speed, np.sqrt(2 * self.accel * abs(dist) + 1e-9))
        self.vel += np.clip(desired - self.vel, -self.accel * dt, self.accel * dt)
        step = self.vel * dt
        self.setpoint = self.target if abs(step) >= abs(dist) else self.setpoint + step
        if abs(step) >= abs(dist):
            self.vel = 0.0
        self.sim.command_cart(self.setpoint)
        pos_ok = abs(s.cart_x_m - self.target) < 0.003
        vel_ok = abs(s.cart_vel_m_s) < 0.01
        self.stable = self.stable + 1 if pos_ok and vel_ok else 0
        if self.stable >= self.sim.config["motion"]["settle_steps"]:
            return OpStatus.DONE
        if s.sim_time_s - self.t0 > self.sim.config["motion"]["move_timeout_s"] + 10:
            self.error = "CART_TIMEOUT"
            return OpStatus.FAILED
        return OpStatus.RUNNING


class _ArmMove:
    def __init__(self, sim, q_target, duration_s, label, speed_scale=1.0):
        self.sim = sim
        self.q_from = np.asarray(sim.data.qpos[sim.arm_qadr], dtype=float)
        self.q_target = np.asarray(q_target, dtype=float)
        span = float(np.max(np.abs(self.q_target - self.q_from)))
        rate = sim.config["motion"]["joint_speed_rad_s"] * speed_scale
        self.dur = max(0.5, duration_s if duration_s else span / rate + 0.6)
        self.t0 = None
        self.label = label
        self.stable = 0
        self.error = None

    def tick(self, s):
        if self.t0 is None:
            self.t0 = s.sim_time_s
        t = s.sim_time_s - self.t0
        a = min(1.0, t / self.dur)
        a = a * a * a * (10 - 15 * a + 6 * a * a)  # quintic ease
        self.sim.command_arm(self.q_from + a * (self.q_target - self.q_from))
        if a < 1.0:
            self.stable = 0
            return OpStatus.RUNNING
        err = np.max(np.abs(np.asarray(s.arm_qpos) - self.q_target))
        vel = np.max(np.abs(s.arm_qvel))
        self.stable = self.stable + 1 if err < 0.04 and vel < 0.05 else 0
        if self.stable >= self.sim.config["motion"]["settle_steps"]:
            return OpStatus.DONE
        if t > self.dur + self.sim.config["motion"]["move_timeout_s"]:
            self.error = "ARM_TIMEOUT" if not self.label else f"ARM_TIMEOUT:{self.label}"
            return OpStatus.FAILED
        return OpStatus.RUNNING


def _skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


class _ArmServo:
    """Resolved-rate Cartesian servo: the grasp point follows a straight line
    to the target, so the hand never sweeps through a wide joint-space arc."""

    def __init__(self, sim, pos, mat, speed, label, dq_clip=0.025):
        self.sim = sim
        self.pos = pos
        self.mat = mat
        self.speed = speed
        self.label = label
        self.dq_clip = dq_clip
        self.q_cmd = np.asarray(sim.data.qpos[sim.arm_qadr], dtype=float).copy()
        self.lam = 0.05
        self.stable = 0
        self.error = None
        self.t0 = None
        self.best_dist = None
        self.stall_t = None

    def tick(self, s):
        if self.t0 is None:
            self.t0 = s.sim_time_s
        sim = self.sim
        m, d = sim.model, sim.data
        hid = sim.hand_id
        R_h = d.xmat[hid].reshape(3, 3)
        p = d.xpos[hid] + R_h @ GRASP_OFFSET
        e_p = self.pos - p
        dist = float(np.linalg.norm(e_p))
        R = self.mat @ R_h.T
        e_r = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        dt = sim.config["timestep_s"] * sim.config["control_steps"]
        v = min(self.speed, 1.5 * dist + 0.015)
        dx = e_p / dist * v * dt if dist > 1e-9 else np.zeros(3)
        wr = 0.6
        e6 = np.concatenate([dx, wr * np.clip(e_r, -1.0, 1.0) * 0.12])
        jp = np.zeros((3, m.nv))
        jr = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jp, jr, hid)
        r = p - d.xpos[hid]
        jp = (jp - _skew(r) @ jr)[:, sim.arm_vadr]
        jr = jr[:, sim.arm_vadr]
        J = np.vstack([jp, wr * jr])
        dq = J.T @ np.linalg.solve(J @ J.T + self.lam**2 * np.eye(6), e6)
        self.q_cmd = np.clip(
            self.q_cmd + np.clip(dq, -self.dq_clip, self.dq_clip),
            sim.arm_range[:, 0], sim.arm_range[:, 1])
        sim.command_arm(self.q_cmd)
        if dist < 0.004 and np.linalg.norm(e_r) < 0.06:
            self.stable += 1
        else:
            self.stable = 0
        if self.stable >= sim.config["motion"]["settle_steps"]:
            return OpStatus.DONE
        # stall detection: no distance progress for 3 s -> wedged or at a
        # local minimum; bail so callers can react instead of timing out
        if self.best_dist is None or dist < self.best_dist - 0.001:
            self.best_dist = dist
            self.stall_t = s.sim_time_s
        if s.sim_time_s - self.stall_t > 3.0:
            self.error = ("SERVO_STALL" if not self.label
                          else f"SERVO_STALL:{self.label}")
            return OpStatus.FAILED
        if s.sim_time_s - self.t0 > sim.config["motion"]["move_timeout_s"]:
            self.error = "SERVO_TIMEOUT" if not self.label else f"SERVO_TIMEOUT:{self.label}"
            return OpStatus.FAILED
        return OpStatus.RUNNING


class _GripMove:
    def __init__(self, sim, ctrl, duration_s, label):
        self.sim = sim
        self.ctrl = ctrl
        self.dur = duration_s
        self.t0 = None
        self.error = None
        self.label = label

    def tick(self, s):
        if self.t0 is None:
            self.t0 = s.sim_time_s
        self.sim.command_gripper(self.ctrl)
        if s.sim_time_s - self.t0 >= self.dur:
            return OpStatus.DONE
        return OpStatus.RUNNING
