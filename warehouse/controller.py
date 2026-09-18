from collections import deque
from dataclasses import asdict
import math
import time

import numpy as np

from warehouse.contracts import Observations, Snapshot, State


class Controller:
    def __init__(self, inventory, motion, actions, log, config):
        self.inventory = inventory
        self.motion = motion
        self.actions = actions
        self.log = log
        self.config = config
        self.state = State.IDLE
        self.request = None
        self.failure_reason = None
        self.plan = deque()
        self.pick_xyz = None
        self.confirmed = False
        self.confirmation_count = 0
        self.reception_pick_count = 0
        self.retries = 0
        self.wait_sim_s = 0.0
        self.wait_wall_s = 0.0
        self.wait_started_sim = None
        self.wait_started_wall = None
        self.home_stable_steps = 0
        self.history = [State.IDLE]

    @property
    def done(self):
        return self.state in (State.COMPLETE, State.FAILED)

    @property
    def home(self):
        bin = self.inventory.bins[self.request.bin_id]
        return self.inventory.slots[bin.home_slot_id].nominal_pick_xyz_m

    def event(self, event_type, snapshot, **payload):
        self.log.record(self.request.id if self.request else None, event_type,
                        snapshot.sim_time_s, payload, self.inventory)

    def request_material(self, sku, snapshot):
        if self.state != State.IDLE:
            raise ValueError("MVP accepts one request per run; reset explicitly for another cycle")
        self.request = self.inventory.request_material(sku)
        self.transition(State.RESERVE, snapshot)
        return self.request

    def transition(self, state, snapshot):
        previous = self.state
        if previous == State.WAIT_OPERATOR:
            self.wait_sim_s += snapshot.sim_time_s - self.wait_started_sim
            self.wait_wall_s += time.monotonic() - self.wait_started_wall
        self.state = state
        self.request.status = state
        self.history.append(state)
        if state == State.WAIT_OPERATOR:
            self.wait_started_sim = snapshot.sim_time_s
            self.wait_started_wall = time.monotonic()
        self.event("TRANSITION", snapshot, previous=previous, state=state, reason=self.failure_reason)

    def fail(self, reason, snapshot):
        if self.done:
            return
        self.failure_reason = reason
        self.plan.clear()
        self.motion.stop(snapshot)
        if self.request:
            self.inventory.fail(self.request)
            self.transition(State.FAILED, snapshot)

    def move(self, xyz, docking=False):
        return ("move", (tuple(xyz), docking))

    def pick_plan(self, xyz):
        x, y, z = xyz
        low = z - self.config["pick_clearance_m"]
        high = low + self.config["lift_m"]
        return deque([
            self.move((x, y, low)),
            ("attach", None),
            self.move((x, y, high), docking=True),
            self.move((x, self.config["retracted_y_m"], high)),
        ])

    def place_plan(self, xyz, location):
        x, y, z = xyz
        low = z - self.config["pick_clearance_m"]
        high = low + self.config["lift_m"]
        return deque([
            self.move((x, y, high)),
            self.move((x, y, low), docking=True),
            ("detach", location),
            self.move((x, self.config["retracted_y_m"], low)),
        ])

    def run_plan(self, snapshot):
        if self.motion.active:
            return False
        if not self.plan:
            return True
        kind, value = self.plan.popleft()
        if kind == "move":
            xyz, docking = value
            self.motion.command(xyz, snapshot, docking=docking)
        elif kind == "attach":
            self.actions.attach(self.request.bin_id)
            self.inventory.move(self.request, "carrier")
            if self.state == State.PICK_AT_RECEPTION:
                self.reception_pick_count += 1
            self.event("IDEALIZED_ATTACH", snapshot, bin_id=self.request.bin_id)
        elif kind == "detach":
            self.actions.detach()
            self.inventory.move(self.request, value)
            self.event("IDEALIZED_DETACH", snapshot, location=value)
        return False

    def valid_pose(self, observation, snapshot, target=None):
        if observation is None or observation.bin_id != self.request.bin_id:
            return False
        if observation.source != "ORACLE" or not math.isfinite(observation.confidence) or observation.confidence < 0.7:
            return False
        if not 0 <= snapshot.sim_time_s - observation.timestamp <= 0.2:
            return False
        if not np.all(np.isfinite(observation.xyz_m)) or not math.isfinite(observation.yaw_rad) or abs(observation.yaw_rad) > 0.02:
            return False
        return target is None or np.max(np.abs(np.array(observation.xyz_m) - target)) < 0.004

    def step_controller(self, snapshot: Snapshot, observations: Observations, operator_events=()):
        if self.done or self.state == State.IDLE:
            return
        try:
            self._step(snapshot, observations, operator_events)
        except ValueError as error:
            self.fail(str(error), snapshot)

    def _step(self, snapshot, observations, operator_events):
        if snapshot.fault:
            self.fail(snapshot.fault, snapshot)
            return
        self.motion.tick(snapshot)
        if self.motion.error:
            self.fail(self.motion.error, snapshot)
            return
        for operator in operator_events:
            accepted = (self.state == State.WAIT_OPERATOR and not self.confirmed
                        and operator.request_id == self.request.id and operator.type == "CONFIRM_REMOVED"
                        and operator.source in ("HUMAN", "SIMULATED"))
            if accepted:
                self.actions.confirm_material_removed(self.request.bin_id)
                self.confirmed = True
                self.confirmation_count += 1
                self.event("MATERIAL_REMOVED", snapshot, operator_source=operator.source,
                           representation="visual patches only; no mass or unit count")
                self.transition(State.OBSERVE_STOCK, snapshot)
            else:
                self.event("OPERATOR_EVENT_IGNORED", snapshot, operator=asdict(operator))
        state = self.state
        retracted = self.config["retracted_y_m"]
        clearance = self.config["pick_clearance_m"]
        lift = self.config["lift_m"]
        reception = self.config["reception_xyz_m"]
        if state == State.RESERVE:
            self.inventory.reserve(self.request)
            x, _, z = self.home
            self.plan = deque([self.move((x, retracted, z - clearance))])
            self.transition(State.APPROACH, snapshot)
        elif state == State.APPROACH and self.run_plan(snapshot):
            self.transition(State.OBSERVE, snapshot)
        elif state == State.OBSERVE:
            pose = observations.pose
            if not self.valid_pose(pose, snapshot):
                self.fail("BIN_MISSING_OR_POSE_UNKNOWN", snapshot)
            elif abs(pose.xyz_m[0] - self.home[0]) > 0.025 or np.max(np.abs(np.array(pose.xyz_m)[1:] - self.home[1:])) > 0.004:
                self.fail("POSE_OUTSIDE_MVP_ENVELOPE", snapshot)
            else:
                self.pick_xyz = pose.xyz_m
                self.event("POSE_OBSERVED", snapshot, observation=asdict(pose))
                x, _, z = self.pick_xyz
                self.plan = deque([self.move((x, retracted, z - clearance))])
                self.transition(State.ALIGN, snapshot)
        elif state == State.ALIGN and self.run_plan(snapshot):
            self.plan = self.pick_plan(self.pick_xyz)
            self.transition(State.EXTRACT, snapshot)
        elif state == State.EXTRACT and self.run_plan(snapshot):
            self.transition(State.VERIFY_PICK, snapshot)
        elif state == State.VERIFY_PICK:
            if snapshot.attached_bin_id != self.request.bin_id or not self.valid_pose(observations.pose, snapshot):
                self.fail("PICK_NOT_VERIFIED", snapshot)
            else:
                x, _, z = reception
                self.plan = deque([self.move((x, retracted, z - clearance + lift))])
                self.transition(State.TRANSFER, snapshot)
        elif state == State.TRANSFER and self.run_plan(snapshot):
            self.plan = self.place_plan(reception, "reception")
            self.transition(State.PLACE_AT_RECEPTION, snapshot)
        elif state == State.PLACE_AT_RECEPTION and self.run_plan(snapshot):
            if snapshot.attached_bin_id or not self.valid_pose(observations.pose, snapshot, reception):
                self.fail("RECEPTION_NOT_VERIFIED", snapshot)
            else:
                self.transition(State.WAIT_OPERATOR, snapshot)
        elif state == State.OBSERVE_STOCK and observations.stock is not None:
            stock = observations.stock
            if stock.bin_id != self.request.bin_id or not 0 <= snapshot.sim_time_s - stock.timestamp <= 0.2:
                self.fail("STOCK_OBSERVATION_INVALID", snapshot)
                return
            self.inventory.observe(stock)
            self.event("STOCK_OBSERVED", snapshot, observation=asdict(stock))
            if not snapshot.home_clear:
                self.fail("RETURN_SLOT_OCCUPIED", snapshot)
            elif not self.valid_pose(observations.pose, snapshot, reception):
                self.fail("RECEPTION_BIN_MISSING", snapshot)
            else:
                self.plan = self.pick_plan(observations.pose.xyz_m)
                self.transition(State.PICK_AT_RECEPTION, snapshot)
        elif state == State.PICK_AT_RECEPTION and self.run_plan(snapshot):
            if snapshot.attached_bin_id != self.request.bin_id:
                self.fail("RECEPTION_PICK_NOT_VERIFIED", snapshot)
            else:
                x, _, z = self.home
                self.plan = deque([self.move((x, retracted, z - clearance + lift))])
                self.transition(State.RETURN, snapshot)
        elif state == State.RETURN and self.run_plan(snapshot):
            if not snapshot.home_clear:
                self.fail("RETURN_SLOT_OCCUPIED", snapshot)
            else:
                location = self.inventory.bins[self.request.bin_id].home_slot_id
                self.plan = self.place_plan(self.home, location)
                self.transition(State.PLACE_HOME, snapshot)
        elif state == State.PLACE_HOME:
            if not snapshot.home_clear:
                self.fail("RETURN_SLOT_OCCUPIED", snapshot)
            elif self.run_plan(snapshot):
                self.transition(State.VERIFY_HOME, snapshot)
        elif state == State.VERIFY_HOME:
            valid = (snapshot.home_clear and snapshot.attached_bin_id is None
                     and self.valid_pose(observations.pose, snapshot, self.home)
                     and max(abs(v) for v in snapshot.velocities_xyz_m_s) < 0.008)
            if not valid:
                self.fail("HOME_NOT_VERIFIED", snapshot)
            else:
                self.home_stable_steps += 1
                if self.home_stable_steps >= 5:
                    self.inventory.complete(self.request, verified=True)
                    self.transition(State.COMPLETE, snapshot)
