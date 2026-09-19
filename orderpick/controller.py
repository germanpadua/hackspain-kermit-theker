"""Autonomous order-preparation controller (oracle pose mode).

State machine over Skills: for each order line, drive to the slot, perceive
(oracle for now), pick with a bounded retry budget, deposit in the onboard
tray. When the front bin is empty, remove it to the park and pull the
reserve bin forward by its lip.

Verification stays sensor-shaped: captures are reported by the pinch-width
encoder and by tray/piece observations; controller never reads body truth.
"""
from __future__ import annotations

import numpy as np

from .ik import IKSolver
from .motion import Motion
from .perception import WristVision
from .scene import bin_origin, piece_name, tray_pocket_world
from .sim import DOWN, DOWN_X, CellSim
from .skills import Skills, _quat_mat

BIN_POST_GP_Z = 0.118      # gp above bin origin -> pads just under the head
# saddle handle: the pads descend beside the mast, slide UNDER a crossbar
# and lift it — the tray's weight rides on the pads' top faces
# (compression, not friction) while the centre post is only pinched for
# lateral location. TRAY_UNDER_Z clears the prongs; TRAY_POST_GP_Z is the
# seated pinch height used for the carry legs.
TRAY_UNDER_Z = 0.113
TRAY_POST_GP_Z = 0.1295


class Controller:
    def __init__(self, sim: CellSim, rng=None, perception="oracle"):
        self.sim = sim
        self.skills = Skills(sim)
        self.motion = self.skills.motion
        self.events = []
        self.max_retries = 2
        self._placed = []         # tray drop cells used — load balancing
        if perception not in ("oracle", "vision"):
            raise ValueError(f"unknown perception mode {perception!r}")
        self.perception = perception
        self.vision = WristVision(sim) if perception == "vision" else None

    def log(self, kind, **kw):
        self.events.append({"t": round(self.sim.data.time, 3), "kind": kind, **kw})

    def spin(self, n=10):
        self.skills.spin(n)

    def run(self):
        while self.motion.active:
            self.skills.spin()
        return self.motion.error is None

    # ---------- oracle pose source (debug/integration only) ----------
    def piece_pos_oracle(self, name):
        adr = self.sim.piece_qadr[name]
        return self.sim.data.qpos[adr:adr + 3].copy()

    def bin_pos_oracle(self, bin_id):
        adr = self.sim.bin_qadr[bin_id]
        return self.sim.data.qpos[adr:adr + 3].copy()

    # ---------- vision: rendered wrist camera, calibrated ----------
    def _perceive_bin(self, bin_id, sku):
        """Servo the wrist camera above the bin and observe. Returns
        (status, [(pos,)]): 'ok'/'empty'/'unknown'; poses are ordered
        front-most first. The controller never sees piece names — only
        perceived positions of the expected SKU."""
        e = next(b for b in self.sim.catalog["bins"] if b["id"] == bin_id)
        bo = bin_origin(self.sim.config, e["slot"], e["depth_row"])
        tgt = np.array([bo[0], bo[1] - 0.05, bo[2] + 0.33])
        row = self._bin_row.get(bin_id, e["depth_row"])
        if row != e["depth_row"]:
            bo = bin_origin(self.sim.config, e["slot"], row)
        self.skills.servo(tgt, DOWN_X, 0.10, "observe")  # partial reach ok
        half = (self.sim.config["bin"]["size_xyz_m"][0] / 2,
                self.sim.config["bin"]["size_xyz_m"][1] / 2)
        st, hits = self.vision.observe_bin(bin_id, bo[:2], half,
                                           floor_z=bo[2])
        units = []
        for s, p in sorted((h for h in hits if h[0] == sku),
                           key=lambda h: h[1][1]):
            p = np.asarray(p, float)
            p[2] = bo[2] + 0.0105   # perceived z is the head top; the pick
            units.append(p)          # needs the piece's base height
        return st, units

    def _perceive_tray(self):
        """Observe the onboard tray from above; returns (status, units)
        where units are world positions inside the tray floor band."""
        tp = self.skills.tray_pocket()
        self.skills.servo(tp + np.array([0, 0, 0.36]), DOWN_X, 0.08,
                          "observe_tray")
        st, hits = self.vision.observe_region(
            tp[:2], (0.115, 0.09), floor_z=tp[2], shrink=0.02, z_band=0.095)
        return st, hits

    # ---------- top level ----------
    def fulfill(self, order, slot_map):
        """order: list[(sku, qty)]; slot_map: {sku: (slot_x, [front,reserve])}.
        Returns True when every unit is verified in the tray."""
        self._order = dict(order)
        self._bin_sku = {b: sku for sku, (_x, bins) in slot_map.items()
                         for b in bins}
        self._bin_row = {b: i for sku, (_x, bins) in slot_map.items()
                         for i, b in enumerate(bins)}
        complete = True
        for sku, qty in order:
            slot_x, bins = slot_map[sku]
            need = qty
            for bin_id in bins:
                if need <= 0:
                    break
                if not self.skills.drive(slot_x):
                    self.log("cart_error", at=slot_x)
                    return False
                got = self._pick_units(bin_id, sku, need)
                need -= got
                if need > 0 and len(bins) > 1 and bin_id == bins[0]:
                    self.log("front_depleted", bin=bin_id, remaining=need)
                    if not self.advance_reserve(bin_id, bins[1]):
                        self.log("reserve_fail", bin=bins[1])
                        return False
                    # reserve physically sits at the front row now
                    self._bin_row[bins[1]] = 0
            if need > 0:
                # record the shortfall but still complete the other lines —
                # a partial order is INCOMPLETE, not a success
                self.log("order_shortfall", sku=sku, missing=need)
                complete = False
        return complete

    def _pick_units(self, bin_id, sku, want):
        got = 0
        tries = 0
        obs_retries = 0
        while got < want and tries < want * (self.max_retries + 1):
            tries += 1
            if self.perception == "vision":
                # re-observe EVERY attempt: the bin's content changed by the
                # previous pick, so cached positions would be stale
                st, units = self._perceive_bin(bin_id, sku)
                if st == "unknown":
                    self.log("obs_invalid", bin=bin_id)
                    obs_retries += 1
                    if obs_retries > 1:
                        self.log("bin_unreadable", bin=bin_id)
                        break
                    continue
                obs_retries = 0
                if st == "empty" or not units:
                    self.log("bin_seen_empty", bin=bin_id)
                    break
                pos = units[0]
                name = None     # the controller never sees piece names
                # ~2cm pose error means the pinch can land on the head brim
                # instead of the neck: fing width >0.0155 is a brim catch
                # (held but swings loose on the carry) — reject it
                if not self.skills.pick_piece(pos, None,
                                              pinch_band=(0.007, 0.0155)):
                    self.log("pick_miss", piece="?", bin=bin_id)
                    continue
                # verify the pick: lift the held unit clear of the bin's
                # z-band, then the aimed spot must read empty
                self.skills.servo(self.sim.grasp_point()
                                  + np.array([0, 0, 0.12]), DOWN_X, 0.04,
                                  "pick_verify_lift")
                st2, units2 = self._perceive_bin(bin_id, sku)
                if st2 == "ok" and any(
                        np.linalg.norm(u[:2] - pos[:2]) < 0.04
                        for u in units2):
                    # whatever the jaws hold, it is NOT that unit — drop it
                    self.skills.open(0.4)
                    self.log("pick_miss", piece="?", bin=bin_id,
                             why="unit_still_there")
                    continue
            else:
                # perceived units: oracle returns the remaining pieces
                names = [n for n in self.sim.piece_qadr
                         if self.sim.piece_bin.get(n) == bin_id
                         and self._piece_in_bin(n, bin_id)]
                if not names:
                    break
                name = sorted(names)[0]
                pos = self.piece_pos_oracle(name)
                if not self.skills.pick_piece(pos, name):
                    self.log("pick_miss", piece=name, bin=bin_id)
                    continue
            # drop at outer-compartment centres (x +-0.079, dividers at
            # +-0.040) with +-0.032 y offset (finger pads stay clear of the
            # pocket lips); the side is chosen to keep the tray's x-moment
            # ~0 — an unbalanced load torques the post out of the pinch
            comp = self._next_comp(tall=(sku == "ESPARRAGO"))
            verify = self._make_tray_verify() if self.vision else None
            if not self.skills.place_in_tray(name, comp_xy=comp,
                                             verify=verify):
                self.log("place_miss", piece=name, bin=bin_id)
                # rescue: re-observe the piece where it actually landed and
                # re-pick it — recoverable only while it rests near tray top
                if self.vision:
                    p2, near_tray = self._rescue_spot()
                else:
                    p2 = self.piece_pos_oracle(name)
                    tp = self.skills.tray_pocket()
                    near_tray = (abs(p2[0] - tp[0]) < 0.22
                                 and abs(p2[1] - tp[1]) < 0.20
                                 and p2[2] > tp[2] - 0.01)
                if near_tray and self.skills.pick_piece(p2, name):
                    v2 = self._make_tray_verify() if self.vision else None
                    if self.skills.place_in_tray(
                            name, comp_xy=(comp[0], -comp[1]), verify=v2):
                        self._placed.append((comp[0], -comp[1]))
                        got += 1
                        self.log("unit_picked", piece=name, bin=bin_id,
                                 rescued=True)
                        continue
                    self.log("place_miss", piece=name, bin=bin_id)
                    if self._piece_in_tray(name):
                        self._placed.append(comp)
                        got += 1
                        self.log("unit_picked", piece=name, bin=bin_id,
                                 rescued=True)
                elif self._piece_in_tray(name):
                    # bounced/perched pieces can settle inside after the
                    # verify window — count it instead of declaring it lost
                    self._placed.append(comp)
                    got += 1
                    self.log("unit_picked", piece=name, bin=bin_id,
                             rescued=True)
                else:
                    self.log("piece_lost", piece=name)
                continue
            self._placed.append(comp)
            got += 1
            self.log("unit_picked", piece=name, bin=bin_id)
        return got

    def _piece_in_tray(self, name):
        """Oracle: is THIS piece inside the tray containment band? Vision:
        did the observed unit count grow past the placed tally?"""
        if self.vision:
            for _ in range(3):
                self.spin(40)
            st, hits = self._perceive_tray()
            self.log("tray_observed", status=st, seen=len(hits))
            return st == "ok" and len(hits) >= len(self._placed) + 1
        tp = self.sim.truth_body_pos("tray")
        rel = self.piece_pos_oracle(name) - tp
        return (abs(rel[0]) < 0.115 and abs(rel[1]) < 0.09
                and -0.02 < rel[2] < 0.098)

    def _make_tray_verify(self):
        """Vision place check: the piece must be SEEN inside the tray floor
        band after settling — the observed unit count must grow by one."""
        expected = len(self._placed) + 1

        def verify():
            # re-observe across the settle window: a piece still bouncing
            # reads above the tray band on the first frames
            for _ in range(4):
                st, hits = self._perceive_tray()
                self.log("tray_observed", status=st, seen=len(hits))
                if st == "ok" and len(hits) >= expected:
                    return True
                self.spin(50)
            return False
        return verify

    def _rescue_spot(self):
        """Vision-mode rescue: look at the tray + cart surround; a unit that
        bounced out sits ON the cart/rim — outside the tray floor band."""
        tp = self.skills.tray_pocket()
        self.skills.servo(tp + np.array([0, 0, 0.40]), DOWN_X, 0.08,
                          "observe_rescue")
        st, hits = self.vision.observe_region(tp[:2], (0.30, 0.28),
                                              floor_z=None, shrink=0.0)
        if st != "ok":
            return None, False
        for _s, p in hits:
            p = np.asarray(p)
            if (abs(p[0] - tp[0]) < 0.115 and abs(p[1] - tp[1]) < 0.09
                    and p[2] < tp[2] + 0.055):
                continue  # properly inside — not the stray
            if p[2] > tp[2] - 0.01:
                p[2] = tp[2] + 0.0105   # perceived z is a top face
                return p, True
        return None, False

    def _next_comp(self, tall=False):
        """Next drop slot: each unit gets its own cell+row — stacking a drop
        on a previous piece tips it onto the rim and out. The slot order
        alternates x-sides so the tray's x-moment stays ~0 for the final
        mast pinch; tall pieces take the centre cells (walled both sides)."""
        if tall:
            seq = [(0.0, -0.03), (0.0, 0.03),
                   (-0.079, -0.03), (0.079, 0.03)]
        else:
            seq = [(-0.079, -0.03), (0.079, -0.03),
                   (0.079, 0.03), (-0.079, 0.03),
                   (0.0, -0.03), (0.0, 0.03)]
        for c in seq:
            if c not in self._placed:
                return c
        return seq[0]

    def _piece_in_bin(self, name, bin_id):
        p = self.piece_pos_oracle(name)
        b = self.bin_pos_oracle(bin_id)
        return abs(p[0] - b[0]) < 0.09 and abs(p[1] - b[1]) < 0.09 \
            and 0.0 < p[2] - b[2] < 0.10

    def _surface_free(self, kind):
        """Verify a surface is clear before placing something on it.
        Vision mode observes it with the wrist camera (magenta obstacle
        markers); oracle reads the fixture body (labeled debug path)."""
        cfg = self.sim.config[kind]
        c = np.array([cfg["x_m"], cfg["y_m"]])
        half = ((0.15, 0.15) if kind == "park"
                else (cfg["size_xy_m"][0] / 2, cfg["size_xy_m"][1] / 2))
        if self.vision:
            self.skills.servo(np.array([c[0], c[1], cfg["surface_z_m"] + 0.30]),
                              DOWN_X, 0.10, f"observe_{kind}")
            return self.vision.surface_free(c, half, cfg["surface_z_m"])
        obs = self.sim.truth_body_pos(f"{kind}_obstacle")
        return ("occupied" if abs(obs[0] - c[0]) < half[0]
                and abs(obs[1] - c[1]) < half[1]
                and obs[2] > cfg["surface_z_m"] - 0.02 else "free")

    # ---------- reserve advance ----------
    def advance_reserve(self, front_bin, reserve_bin):
        """Remove the empty front bin to the park, then pull the reserve bin
        forward by hooking its front lip."""
        sk = self.skills
        free = self._surface_free("park")
        if free != "free":
            self.log("park_unavailable", status=free)
            return False
        # 1) pinch front bin's post, lift, park
        bo = self.bin_pos_oracle(front_bin)
        gp = np.array([bo[0], bo[1] + 0.0865, bo[2] + BIN_POST_GP_Z])
        sk.open(0.4)
        if not sk.hover_to(gp, DOWN_X, [0, 0, 0.20]):
            return False
        if not sk.servo(gp + np.array([0, 0, 0.04]), DOWN_X, 0.05, "bin_center"):
            return False
        if not sk.servo(gp, DOWN_X, 0.02, "bin_grasp"):
            return False
        sk.close(1.0)
        if not self._pinch_ok(0.005, 0.019):
            self.log("bin_pinch_fail", bin=front_bin)
            return False
        sk.servo(self.sim.grasp_point() + np.array([0, 0, 0.20]), DOWN_X,
                 0.035, "bin_lift")
        # park plate is left of slot S1
        pk = self.sim.config["park"]
        if not sk.hover_to(np.array([pk["x_m"], pk["y_m"], pk["surface_z_m"] + 0.24]),
                           DOWN_X, [0, 0, 0]):
            return False
        if not sk.hover_to(np.array([pk["x_m"], pk["y_m"], pk["surface_z_m"] + 0.10]),
                           DOWN_X, [0, 0, 0]):
            return False  # hold the bin rather than releasing mid-path
        sk.open(0.6)
        self.spin(30)
        if not self._bin_on_park(front_bin):
            p = self.bin_pos_oracle(front_bin)
            self.log("bin_park_fail", bin=front_bin,
                     at=[round(float(v), 3) for v in p])
            return False
        self.log("bin_parked", bin=front_bin)
        # 2) hook reserve's front lip, pull to the front row
        bo2 = self.bin_pos_oracle(reserve_bin)
        gl = np.array([bo2[0], bo2[1] - 0.0865, bo2[2] + 0.011])
        if not sk.hover_to(gl, DOWN_X, [0, 0, 0.15]):
            return False
        sk.open(0.4)
        if not sk.servo(gl, DOWN_X, 0.03, "lip"):
            return False
        sk.close(0.8)
        if not sk.servo(self.sim.grasp_point() + np.array([0, -0.24, 0]),
                        DOWN_X, 0.03, "drag"):
            pass  # partial drag may still be enough — verify by pose
        sk.open(0.4)
        self.spin(20)
        # verify: reserve bin arrived near the front row
        b2 = self.bin_pos_oracle(reserve_bin)
        if not (b2[1] < 0.11):
            self.log("reserve_drag_fail", bin=reserve_bin, y=float(b2[1]))
            return False
        self.log("reserve_advanced", bin=reserve_bin, y=float(b2[1]))
        return True

    def _pinch_ok(self, lo, hi):
        f = float(np.mean(self.sim.data.qpos[self.sim.fing_qadr]))
        return lo < f < hi

    def _bin_on_park(self, bin_id):
        p = self.bin_pos_oracle(bin_id)
        pk = self.sim.config["park"]
        return abs(p[0] - pk["x_m"]) < 0.15 and abs(p[1] - pk["y_m"]) < 0.15 \
            and abs(p[2] - pk["surface_z_m"]) < 0.03

    # ---------- tray delivery ----------
    def deliver_tray(self):
        """Drive to reception with the tray in its pocket, then pinch the post
        and deposit the tray on the table."""
        sk = self.skills
        rc = self.sim.config["reception"]
        if not sk.stow():
            sk.stow()  # one retry: a failed stow leaves the arm mid-pose
        self.motion.drive_cart(rc["x_m"] - 0.20)
        if not self.run():
            self.log("cart_error", at="reception")
            return False
        self.log("cart_at_reception", cart=float(self.sim.data.qpos[self.sim.cart_qadr]))
        free = self._surface_free("reception")
        if free != "free":
            self.log("reception_unavailable", status=free)
            return False
        # pinch -> lift -> carry -> lower, retried end-to-end: a slipped mast
        # usually drops back onto the pocket lip near-vertical, so a fresh
        # pinch from the tray's true pose recovers it. A tray resting flat
        # (mast not vertical) cannot be re-pinched and fails after the budget.
        last_at = None
        for _cycle in range(3):
            tray = self.sim.truth_body_pos("tray")  # oracle path (H4 replaces)
            # the pocket lets the tray tilt a degree or two — aim the approach
            # at the mast's true top so the pinch stays centred
            tq = _quat_mat(self.sim.truth_body_quat("tray"))
            under = tray + tq @ np.array([0.0, 0.0, TRAY_UNDER_Z])
            seat = tray + tq @ np.array([0.0, 0.0, TRAY_POST_GP_Z])
            side = tq @ np.array([0.047, 0.0, 0.0])
            sk.open(0.55)
            # waypoint high over the tray first: a stowed arm near a
            # kinematic limit unfolds into a bad branch when asked to reach
            # under+side directly — the over-the-top waypoint fixes that
            sk.hover_to(tray + np.array([0.0, 0.0, 0.42]), DOWN_X,
                        [0, 0, 0])
            if not sk.hover_to(under + side, DOWN_X, [0, 0, 0.18]):
                self.log("tray_approach_fail", leg="hover",
                         hand=[round(float(v), 3) for v in
                               self.sim.grasp_point()],
                         err=sk.motion.error)
                sk.stow()  # unfold to a canonical pose before the next cycle
                continue
            # descend beside the mast (offset clears the saddle and prongs),
            # slide under the saddle, then pinch the centre post for location
            if not sk.servo(under + side, DOWN_X, 0.05, "tray_beside"):
                self.log("tray_approach_fail", leg="beside")
                continue
            if not sk.servo(under, DOWN_X, 0.02, "tray_under"):
                self.log("tray_approach_fail", leg="under",
                         hand=[round(float(v), 3) for v in
                               self.sim.grasp_point()])
                continue
            held = False
            for _attempt in range(3):
                sk.close(1.0)
                f = float(np.mean(self.sim.data.qpos[self.sim.fing_qadr]))
                if not self._pinch_ok(0.004, 0.030):
                    self.log("tray_pinch_reject", fing=round(f, 4))
                    sk.open(0.55)
                    continue
                # seat: raise the pads until the saddle rests on their tops —
                # contact stalls the servo before it fully converges
                if not sk.servo(seat, DOWN_X, 0.02, "tray_seat") \
                        and sk.motion.error != "SERVO_STALL":
                    self.log("tray_approach_fail", leg="seat")
                    break
                z0 = float(self.sim.truth_body_pos("tray")[2])
                if sk.servo(seat + np.array([0, 0, 0.05]), DOWN_X, 0.02,
                            "tray_tlift") \
                        and float(self.sim.truth_body_pos("tray")[2]) \
                        > z0 + 0.03:
                    f2 = float(np.mean(
                        self.sim.data.qpos[self.sim.fing_qadr]))
                    if 0.006 < f2 < 0.030:
                        held = True
                        break
                    self.log("tray_slip", fing=f2)
                else:
                    self.log("tray_slip", fing=f)
                sk.open(0.55)
                if not sk.servo(under, DOWN_X, 0.02, "tray_under"):
                    break
            if not held:
                continue
            # slow Cartesian legs: joint swings yank the pinched post loose.
            # Lift to full carry height over the pocket first, then translate —
            # the tray hangs ~8.5 cm below gp and must clear the table edge.
            sk.hold_guard = True
            post_top = (self.sim.truth_body_pos("tray")
                        + _quat_mat(self.sim.truth_body_quat("tray"))
                        @ np.array([0.0, 0.0, 0.140]))
            sk.hold_ref = float(np.linalg.norm(post_top
                                               - self.sim.grasp_point()))
            carry_z = rc["surface_z_m"] + 0.26
            ok = False
            for label, tgt in (("tray_lift", np.array([seat[0], seat[1],
                                                       carry_z])),
                               ("tray_carry", np.array([rc["x_m"], rc["y_m"],
                                                        carry_z])),
                               ("tray_lower", np.array([rc["x_m"], rc["y_m"],
                                       rc["surface_z_m"] + TRAY_POST_GP_Z
                                       + 0.01]))):
                # gentle accel on loaded legs: hard jerks slide the post out
                if not sk.servo(tgt, DOWN_X, 0.03 if "carry" not in label
                                else 0.05, label, dq_clip=0.012):
                    self.log("tray_carry_slip", leg=label,
                             err=sk.motion.error,
                             fing=float(np.mean(
                                 self.sim.data.qpos[self.sim.fing_qadr])),
                             tray=[round(float(v), 3) for v in
                                   self.sim.truth_body_pos("tray")])
                    break
            else:
                ok = True
            sk.hold_guard = False
            sk.hold_ref = None
            if not ok:
                # graceful abort: fly back over the pocket before releasing so
                # a still-pinched tray falls inside its cage, then re-attempt
                sk.servo(np.array([seat[0], seat[1], carry_z]), DOWN_X, 0.05,
                         "abort_back")
                sk.servo(np.array([seat[0], seat[1],
                                   seat[2] + 0.19]), DOWN_X, 0.03, "abort_low")
            sk.open(0.6)
            sk.hover_to(np.array([rc["x_m"], rc["y_m"],
                                  rc["surface_z_m"] + 0.16]),
                        DOWN_X, [0, 0, 0])
            self.spin(30)
            if not ok:
                self.log("tray_carry_slip",
                         fing=float(np.mean(
                             self.sim.data.qpos[self.sim.fing_qadr])))
                last_at = self.sim.truth_body_pos("tray")
                continue
            # verify: tray rests on the table, upright, released
            tp = self.sim.truth_body_pos("tray")
            qw = abs(float(self.sim.truth_body_quat("tray")[0]))
            on_table = (abs(tp[0] - rc["x_m"]) < 0.12
                        and abs(tp[1] - rc["y_m"]) < 0.12
                        and abs(tp[2] - rc["surface_z_m"]) < 0.04
                        and qw > 0.96)
            if on_table:
                # verify contents after release: units resting inside the tray
                # tallied by SKU and compared against the order
                content = self._tray_contents()
                self.log("tray_delivered",
                         at=[round(float(v), 3) for v in tp],
                         content=content,
                         verified=content == getattr(self, "_order", None))
                return True
            self.log("tray_delivery_fail",
                     at=[round(float(v), 3) for v in tp], qw=round(qw, 3))
            last_at = tp
        if last_at is None:
            self.log("tray_pinch_fail")
        return False

    def _tray_contents(self):
        """SKU tally of the pieces resting inside the tray (evaluator-visible
        truth — the controller's own accounting is via place verification)."""
        tp = self.sim.truth_body_pos("tray")
        tq = _quat_mat(self.sim.truth_body_quat("tray"))
        counts = {}
        for n in self.sim.piece_qadr:
            rel = tq.T @ (self.sim.truth_body_pos(n) - tp)
            if abs(rel[0]) < 0.115 and abs(rel[1]) < 0.09 \
                    and -0.02 < rel[2] < 0.10:
                sku = getattr(self, "_bin_sku", {}).get(
                    self.sim.piece_bin.get(n), "?")
                counts[sku] = counts.get(sku, 0) + 1
        return counts


def sim_gp_up(sim, dz):
    return sim.grasp_point() + np.array([0, 0, dz])
