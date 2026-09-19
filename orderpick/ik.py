"""Damped-least-squares task-space IK over the 7 Panda arm joints.

Solved on a scratch MjData so physics is never perturbed. The cart joint is
held fixed (cart is settled before any arm move by construction).
"""
import numpy as np
import mujoco

from orderpick.sim import GRASP_OFFSET, HOME


def skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def rot_err(cur, des):
    R = des @ cur.T
    return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


class IKSolver:
    def __init__(self, sim):
        self.sim = sim
        self.scratch = mujoco.MjData(sim.model)

    def _jacobian(self, d, point_world):
        m = self.sim.model
        hid = self.sim.hand_id
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jp, jr, hid)
        r = point_world - d.xpos[hid]
        return (jp - skew(r) @ jr)[:, self.sim.arm_vadr], jr[:, self.sim.arm_vadr]

    def solve(self, target_pos, target_mat, weight_rot=0.8, iters=350, lam=0.08):
        m, d = self.sim.model, self.scratch
        d.qpos[:] = self.sim.data.qpos
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        hid = self.sim.hand_id
        e_p = np.array([np.inf]); e_r = np.array([np.inf])
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            p = d.xpos[hid] + d.xmat[hid].reshape(3, 3) @ GRASP_OFFSET
            e_p = target_pos - p
            e_r = rot_err(d.xmat[hid].reshape(3, 3), target_mat)
            if np.linalg.norm(e_p) < 3e-4 and np.linalg.norm(e_r) < 4e-3:
                break
            jp, jr = self._jacobian(d, p)
            J = np.vstack([jp, weight_rot * jr])
            err = np.concatenate([e_p, weight_rot * e_r])
            dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(6), err)
            dq += (np.eye(7) - np.linalg.pinv(J, rcond=1e-4) @ J) @ \
                (0.35 * (HOME - d.qpos[self.sim.arm_qadr]))
            step = np.clip(dq, -0.3, 0.3)
            d.qpos[self.sim.arm_qadr] = np.clip(d.qpos[self.sim.arm_qadr] + step,
                                                self.sim.arm_range[:, 0],
                                                self.sim.arm_range[:, 1])
            mujoco.mj_forward(m, d)
        return d.qpos[self.sim.arm_qadr].copy(), float(np.linalg.norm(e_p))


def rms(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
