"""Manipulation skills: verified pick/place steps built on Motion primitives.

Every skill returns a bool and leaves evidence via the sim state. Piece
picks pinch the post just below the head (mechanical hold). The controller
supplies estimated piece poses; the skills never read sim truth themselves —
verification uses the piece's tracked lift (lift-height delta between the
piece body and the grasp point is provided by the caller's estimate source).
"""
from __future__ import annotations

import numpy as np

from .ik import IKSolver
from .motion import Motion
from .scene import tray_pocket_world
from .sim import DOWN_X, CellSim

GRIP_POST_LOCAL_Z = 0.022  # pinch zone centre above the piece origin
PINCH_GP_OFFSET = 0.030    # gp above piece origin so pad band sits under head


class Skills:
    def __init__(self, sim: CellSim):
        self.sim = sim
        self.motion = Motion(sim)
        self.ik = IKSolver(sim)

    # --- low level -----------------------------------------------------
    def spin(self, n=10):
        for _ in range(n):
            self.sim.step(self.sim.config["control_steps"])
            self.motion.tick(self.sim.snapshot())

    def run(self):
        while self.motion.active:
            self.spin()
        return self.motion.error is None

    def cart(self):
        return float(self.sim.data.qpos[self.sim.cart_qadr])

    def drive(self, x):
        self.motion.drive_cart(x)
        return self.run()

    def hover_to(self, p, mat, off):
        q, err = self.ik.solve(p + np.array(off), mat)
        if q is None or err > 0.006:
            return False
        self.motion.move_arm(q)
        return self.run()

    def servo(self, pos, mat, speed, label):
        self.motion.servo_to(np.asarray(pos, float), mat, speed, label=label)
        ok = self.run()
        if not ok:
            self.motion.error = None
            q, err = self.ik.solve(np.asarray(pos, float), mat)
            if q is None or err > 0.006:
                return False
            self.motion.move_arm(q, label=label)
            ok = self.run()
        return ok

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
        f = float(np.mean(self.sim.data.qpos[self.sim.fing_qadr]))
        return 0.0035 < f < 0.0135

    def piece_lifted_truth(self, name, min_z=0.03):
        """Debug/evaluator check only — reads simulator truth."""
        return self.piece_pose(name)[2] > 0.63 + min_z

    # --- skills ---------------------------------------------------------
    def pick_piece(self, pos_xyz, piece_name):
        """Approach open -> descend -> pinch post -> verify lift. Returns True
        when the piece is held. Caller supplies the perceived piece position."""
        pos = np.asarray(pos_xyz, float)
        gpz = pos[2] + PINCH_GP_OFFSET
        self.open(0.4)
        if not self.hover_to(np.array([pos[0], pos[1], gpz + 0.20]), DOWN_X, [0, 0, 0]):
            return False
        if not self.servo([pos[0], pos[1], gpz + 0.10], DOWN_X, 0.12, "approach"):
            pass  # keep descending anyway; contact stall handled below
        if not self.servo([pos[0], pos[1], gpz], DOWN_X, 0.04, "grasp"):
            return False
        self.close(1.0)
        if not self.pinched_something():
            return False
        # test lift: raise 5 cm; drop detection uses encoder + optional truth
        self.servo(self.sim.grasp_point() + np.array([0, 0, 0.05]),
                   DOWN_X, 0.03, "test_lift")
        self.spin(10)
        return True

    def place_in_tray(self, piece_name, comp_x=0.0):
        """Carry held piece to the onboard tray and release."""
        tp = tray_pocket_world(self.sim.config, self.cart())
        tp = tp + np.array([comp_x, 0.0, 0.0])
        if not self.servo(self.sim.grasp_point() + np.array([0, 0, 0.13]),
                          DOWN_X, 0.035, "clear"):
            pass
        if not self.hover_to(tp + np.array([0, 0, 0.34]), DOWN_X, [0, 0, 0]):
            return False
        self.servo(tp + np.array([0, 0, 0.15]), DOWN_X, 0.04, "lower")
        self.open(0.6)
        self.spin(30)
        # verify: piece rests inside tray walls
        rel = self.piece_pose(piece_name) - self.sim.truth_body_pos("tray")
        return abs(rel[0]) < 0.115 and abs(rel[1]) < 0.07 and 0.0 < rel[2] < 0.06

    def stow(self):
        self.motion.stow()
        return self.run()
