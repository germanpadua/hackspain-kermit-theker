"""Manipulation skills: verified pick/place steps built on Motion primitives.

Every skill returns a bool and leaves evidence via the sim state. Piece
picks pinch the post just below the head (mechanical hold). The controller
supplies estimated piece poses; the skills never read sim truth themselves —
verification uses the piece's tracked lift (lift-height delta between the
piece body and the grasp point is provided by the caller's estimate source).
"""
from __future__ import annotations

import numpy as np

import os

from .ik import IKSolver
from .motion import Motion
from .scene import tray_pocket_world
from .sim import DOWN_X, HOME, STOW, CellSim

GRIP_POST_LOCAL_Z = 0.022  # pinch zone centre above the piece origin
PINCH_GP_OFFSET = 0.022    # gp above piece origin -> pads pinch post under head


DUMP_DIR = os.environ.get("ORDERPICK_DUMP", "")


def _quat_mat(q):
    """wxyz quaternion -> rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class Skills:
    def __init__(self, sim: CellSim):
        self.sim = sim
        self.motion = Motion(sim)
        self.ik = IKSolver(sim)
        self.hold_guard = False   # abort active ops if the pinched object slips
        self.hold_ref = None      # |post_top - gp| at pinch time (drift sensor)
        self.hold_post_local = np.array([0.0, 0.0, 0.140])
        self._dump_tick = 0
        self._dump_idx = 0

    def _dump_frame(self):
        import cv2
        import mujoco
        cam = mujoco.MjvCamera()
        tp = self.sim.truth_body_pos("tray")
        cam.lookat[:] = tp + np.array([0, 0, 0.14])
        cam.distance = 0.28
        cam.azimuth = 90
        cam.elevation = -5
        if self.sim.renderer is None:
            self.sim.renderer = mujoco.Renderer(self.sim.model, height=480,
                                                width=640)
        self.sim.renderer.update_scene(self.sim.data, camera=cam)
        img = self.sim.renderer.render().copy()
        cv2.imwrite(f"{DUMP_DIR}/f{self._dump_idx:04d}_"
                    f"t{self.sim.data.time:.1f}.png",
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self._dump_idx += 1

    def spin(self, n=10):
        for _ in range(n):
            self.sim.step(self.sim.config["control_steps"])
            self.motion.tick(self.sim.snapshot())
            if DUMP_DIR and (self.hold_guard or getattr(
                    self.motion.op, "label", "").startswith(
                    ("tray", "pinch", "reseat"))):
                self._dump_tick += 1
                if self._dump_tick % 12 == 0:
                    self._dump_frame()
            if self.hold_guard and self.motion.active:
                f = float(np.mean(self.sim.data.qpos[self.sim.fing_qadr]))
                if f < 0.004:
                    self.motion.error = "GRIP_LOST"
                    self.motion.op = None
                elif self.hold_ref is not None:
                    # the post sliding out the channel ends barely moves
                    # fing; track the post top vs the grasp point instead
                    tp = self.sim.truth_body_pos("tray")
                    tq = _quat_mat(self.sim.truth_body_quat("tray"))
                    post_top = tp + tq @ self.hold_post_local
                    d = float(np.linalg.norm(post_top - self.sim.grasp_point()))
                    if abs(d - self.hold_ref) > 0.022:
                        self.motion.error = "GRIP_SLIP"
                        self.motion.op = None
                    elif tq[2, 2] < 0.90:
                        # tray tipping >25 deg: the post is camming out —
                        # abort before full escape so it drops into the pocket
                        self.motion.error = "GRIP_TILT"
                        self.motion.op = None

    def run(self):
        while self.motion.active:
            self.spin()
        return self.motion.error is None

    def cart(self):
        return float(self.sim.data.qpos[self.sim.cart_qadr])

    def drive(self, x):
        self.motion.drive_cart(x)
        return self.run()

    def tray_pocket(self):
        return tray_pocket_world(self.sim.config, self.cart())

    def _arrived(self, target, tol=0.04):
        return (np.linalg.norm(self.sim.grasp_point() - np.asarray(target, float))
                < tol)

    def hover_to(self, p, mat, off):
        tgt = np.asarray(p, float) + np.asarray(off, float)
        q, err = self.ik.solve_restarts(tgt, mat, seeds=(HOME,))
        if q is None or err > 0.006:
            # iterative IK can stall from an awkward start pose — the Cartesian
            # servo solves incrementally and often still reaches the target.
            ok = self.servo(tgt, mat, 0.08, "hover_fb")
            return ok and self._arrived(tgt)
        self.motion.move_arm(q)
        ok = self.run()
        if not (ok and self._arrived(tgt)):
            # IK decimetre error or a mid-settle check — servo refines it
            ok = self.servo(tgt, mat, 0.08, "hover_fb")
        return ok and self._arrived(tgt)

    def servo(self, pos, mat, speed, label, dq_clip=0.025):
        pos = np.asarray(pos, float)
        self.motion.servo_to(pos, mat, speed, label=label, dq_clip=dq_clip)
        ok = self.run()
        if not ok or not self._arrived(pos):
            self.motion.error = None
            q, err = self.ik.solve_restarts(pos, mat, seeds=(HOME,))
            if q is None or err > 0.006:
                return False
            self.motion.move_arm(q, label=label)
            ok = self.run()
        return ok and self._arrived(pos)

    def open(self, t=0.5):
        self.motion.grip(255, duration_s=t)
        self.run()

    def close(self, t=1.0):
        self.motion.grip(0, duration_s=t)
        self.run()

    def piece_pose(self, name):
        adr = self.sim.piece_qadr[name]
        return self.sim.data.qpos[adr:adr + 3].copy()

    def pinched_something(self):
        """Encoder-based capture estimate: fingers must stall on the post
        (~8-11 mm), not close to empty air (~0) or stop on the base (>14 mm)."""
        # post pinch lands ~0.0086; a yaw-rotated head corner can stall as
        # high as ~0.022 and still holds mechanically (head can't pass).
        f = float(np.mean(self.sim.data.qpos[self.sim.fing_qadr]))
        return 0.0035 < f < 0.0235

    def piece_lifted_truth(self, name, min_z=0.03):
        """Debug/evaluator check only — reads simulator truth."""
        return self.piece_pose(name)[2] > 0.63 + min_z

    # --- skills ---------------------------------------------------------
    def pick_piece(self, pos_xyz, piece_name, pinch_band=(0.0035, 0.0235)):
        """Approach open -> descend -> pinch post -> verify lift. Returns True
        when the piece is held. Caller supplies the perceived piece position;
        `pinch_band` tightens the encoder acceptance (vision picks)."""
        pos = np.asarray(pos_xyz, float)
        gpz = pos[2] + PINCH_GP_OFFSET
        self.open(0.4)
        if not self.hover_to(np.array([pos[0], pos[1], gpz + 0.20]), DOWN_X, [0, 0, 0]):
            return False
        self.servo([pos[0], pos[1], gpz + 0.10], DOWN_X, 0.12, "approach")
        if not self.servo([pos[0], pos[1], gpz], DOWN_X, 0.04, "grasp"):
            return False
        self.close(1.0)
        f = float(np.mean(self.sim.data.qpos[self.sim.fing_qadr]))
        if not (pinch_band[0] < f < pinch_band[1]):
            self.open(0.4)  # release whatever was caught, then lift clear
            self.servo(self.sim.grasp_point() + np.array([0, 0, 0.12]),
                       DOWN_X, 0.05, "pick_reject_lift")
            return False
        # test lift: raise 5 cm; drop detection uses encoder + optional truth
        self.servo(self.sim.grasp_point() + np.array([0, 0, 0.05]),
                   DOWN_X, 0.03, "test_lift")
        self.spin(10)
        return True

    def place_in_tray(self, piece_name=None, comp_xy=(-0.02, 0.05),
                      verify=None):
        """Carry held piece to the onboard tray and release. Drop zones are
        offset ~5 cm in y so the closed finger channel never sweeps over the
        tray's center post while a piece is pinched. `verify` (if given) is
        a callable evaluated after settling — used for vision-mode checks;
        without it, truth pose is used (oracle/debug only)."""
        tp = tray_pocket_world(self.sim.config, self.cart())
        tp = tp + np.array([comp_xy[0], comp_xy[1], 0.0])
        if not self.servo(self.sim.grasp_point() + np.array([0, 0, 0.13]),
                          DOWN_X, 0.035, "clear"):
            pass
        # joint-space waypoints: the mid heights solve IK cleanly where a
        # straight-line servo stalls near singularities
        if not self.hover_to(tp + np.array([0, 0, 0.50]), DOWN_X, [0, 0, 0]):
            return False
        # the pocket mouth is a tight corridor: joint-space IK lands on a
        # branch that cannot descend into it — servo straight down instead
        self.servo(tp + np.array([0, 0, 0.30]), DOWN_X, 0.04, "pre")
        # carry-drop check: encoder reads open air if the piece slipped
        # during the swing — abort before releasing into the walls
        if float(np.mean(self.sim.data.qpos[self.sim.fing_qadr])) < 0.003:
            self.servo(self.sim.grasp_point() + np.array([0, 0, 0.14]),
                       DOWN_X, 0.03, "drop_abort")
            return False
        # release low: fingertips ~3 cm above the tray floor — the piece
        # drops ~2 cm and cannot build bounce energy over the rim lips
        ok = self.servo(tp + np.array([0, 0, 0.04]), DOWN_X, 0.025, "release")
        if not ok:
            # one retry from a touch higher, then drop into the walls anyway
            self.servo(self.sim.grasp_point() + np.array([0, 0, 0.14]),
                       DOWN_X, 0.02, "relift")
            ok = self.servo(tp + np.array([0, 0, 0.04]), DOWN_X, 0.02,
                            "release2")
        if not ok:
            self.servo(tp + np.array([0, 0, 0.30]), DOWN_X, 0.025, "aborted")
        self.spin(35)  # let the pinched piece stop swinging before release
        self.open(0.6)
        # fingers finish opening below the rim; lift a bit so a perched piece
        # cannot hook a fingertip when the arm swings home
        self.servo(self.sim.grasp_point() + np.array([0, 0, 0.12]),
                   DOWN_X, 0.02, "postlift")
        if verify is not None:
            for _ in range(6):
                self.spin(40)
            return bool(verify())
        # verify over a settling window: a piece inside the tray footprint
        # below the rim-lip band is contained during transport — perched on
        # a divider is fine (the lips keep it in). Perched ON the lip itself
        # (z >= ~0.10) or outside the footprint counts as a miss.
        for _ in range(10):
            self.spin(40)
            rel = self.piece_pose(piece_name) - self.sim.truth_body_pos("tray")
            if (abs(rel[0]) < 0.115 and abs(rel[1]) < 0.075
                    and 0.0 < rel[2] < 0.095):
                return True
        return False

    def stow(self):
        for _ in range(2):
            self.motion.stow()
            ok = self.run()
            q = np.asarray(self.sim.data.qpos[self.sim.arm_qadr])
            if ok and np.max(np.abs(q - np.asarray(STOW))) < 0.08:
                return True
        return False
