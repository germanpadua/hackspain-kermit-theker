from dataclasses import asdict
import time

import cv2

from warehouse.contracts import Level, Observations, OperatorEvent, State, StockObservation
from warehouse.controller import Controller
from warehouse.inventory import Inventory
from warehouse.motion import Motion
from warehouse.perception import OraclePerception, estimate_stock
from warehouse.persistence import RunLog
from warehouse.scene import stock_roi
from warehouse.sim import Simulation


SCENARIOS = ("nominal", "lateral", "missing", "empty", "unknown_stock", "blocked_return", "pick_timeout")


def validate_modes(perception, transport):
    if perception != "oracle" or transport != "idealized":
        raise ValueError("This MVP only implements ORACLE + IDEALIZED. VISION pose and CONTACT transport are not validated or available.")


class Experiment:
    def __init__(self, seed=7, perception="oracle", transport="idealized", scenario="nominal",
                 output="runs", operator_source="SIMULATED", offset_m=None):
        validate_modes(perception, transport)
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")
        self.sim = Simulation(seed, transport, scenario, offset_m)
        self.inventory = Inventory.load()
        self.log = RunLog(output, {
            "seed": seed, "perception": "ORACLE", "transport": "IDEALIZED", "scenario": scenario,
            "offset_m": self.sim.offset_m, "operator_source": operator_source, "stock_source": "VISION_AREA",
            "content": "visual patches; not units, loose pieces or changing mass",
            "contact_validation": False, "timestep_s": self.sim.model.opt.timestep,
        })
        self.motion = Motion(self.sim)
        self.perception = OraclePerception(self.sim)
        self.controller = Controller(self.inventory, self.motion, self.sim, self.log, self.sim.config)
        self.before_captured = False
        self.obstacle_injected = False
        self.camera_error = None
        self.last_image = None
        self.request_started_wall = None
        self.log.record(None, "RUN_STARTED", 0.0, self.log.metadata, self.inventory)

    def start(self, sku="TORNILLOS"):
        request = self.controller.request_material(sku, self.sim.snapshot())
        self.request_started_wall = time.monotonic()
        return request

    def stock_observation(self, name):
        controller = self.controller
        bin_id = controller.request.bin_id
        timestamp = float(self.sim.data.time)
        try:
            image = self.sim.render("reception")
        except (RuntimeError, ValueError) as error:
            self.camera_error = str(error)
            controller.event("CAMERA_UNAVAILABLE", self.sim.snapshot(), reason=self.camera_error)
            return StockObservation(bin_id, Level.UNKNOWN, None, 0.0, timestamp, "CAMERA_UNAVAILABLE")
        self.last_image = image
        if not cv2.imwrite(str(self.log.path / f"{name}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
            raise OSError("Cannot save camera evidence")
        return estimate_stock(image, stock_roi(self.sim.config), bin_id, timestamp)

    def tick(self, operator_events=()):
        controller = self.controller
        if controller.done or controller.state == State.IDLE:
            return
        if controller.state == State.WAIT_OPERATOR and not self.before_captured:
            before = self.stock_observation("stock-before")
            self.inventory.observe(before)
            controller.event("STOCK_BASELINE", self.sim.snapshot(), observation=asdict(before))
            self.before_captured = True
        if controller.state == State.WAIT_OPERATOR and self.sim.scenario == "blocked_return" and not self.obstacle_injected:
            self.sim.inject_return_obstacle(controller.home)
            self.obstacle_injected = True
            controller.event("SCENARIO_INJECTION", self.sim.snapshot(), fixture="return obstacle")
        if controller.state == State.EXTRACT and self.sim.scenario == "pick_timeout":
            self.sim.frozen_axes = True
        pose = self.perception.observe_bin(controller.request.bin_id) if controller.request else None
        stock = self.stock_observation("stock-after") if controller.state == State.OBSERVE_STOCK else None
        controller.step_controller(self.sim.snapshot(), Observations(pose, stock), operator_events)
        if not controller.done:
            for _ in range(self.sim.config["control_steps"]):
                self.sim.step()
        if self.sim.data.time - controller.wait_sim_s > 180 and controller.state != State.WAIT_OPERATOR:
            controller.fail("EPISODE_TIMEOUT", self.sim.snapshot())

    def simulated_operator_events(self):
        controller = self.controller
        if controller.state == State.WAIT_OPERATOR and self.sim.data.time - controller.wait_started_sim >= 0.2:
            return [OperatorEvent(controller.request.id, source="SIMULATED")]
        return []

    def close(self):
        self.sim.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
