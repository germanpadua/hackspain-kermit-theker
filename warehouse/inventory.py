from dataclasses import asdict
import math
from uuid import uuid4

from warehouse.contracts import Bin, Level, Request, Slot, State, StockObservation
from warehouse.scene import load_config


class Inventory:
    def __init__(self, slots, bins, catalog, min_confidence=0.7):
        self.slots = {slot.id: slot for slot in slots}
        self.bins = {bin.id: bin for bin in bins}
        self.catalog = catalog
        self.min_confidence = min_confidence
        self.requests = {}
        self.observations = {}
        self.last_valid = {}
        self.alerts = {}

    @classmethod
    def load(cls):
        scene, config = load_config("scene"), load_config("inventory")
        slots = [Slot(name, tuple(xyz), tuple(scene["slot_size_xyz_m"])) for name, xyz in scene["slots"].items()]
        bins = [Bin(**entry, current_location=entry["home_slot_id"]) for entry in config["bins"]]
        return cls(slots, bins, config["catalog"], config["alert_min_confidence"])

    def request_material(self, sku):
        candidates = [bin for bin in self.bins.values() if bin.sku == sku and bin.status == "AVAILABLE"]
        if not candidates:
            raise ValueError(f"Unknown or unavailable SKU: {sku}")
        request = Request(uuid4().hex, sku, candidates[0].id)
        self.requests[request.id] = request
        return request

    def reserve(self, request):
        bin = self.bins[request.bin_id]
        slot = self.slots[bin.home_slot_id]
        if slot.reserved_by or bin.status != "AVAILABLE" or bin.current_location != bin.home_slot_id:
            raise ValueError("Bin or slot already reserved or unavailable")
        slot.reserved_by = request.id
        bin.status = "RESERVED"

    def move(self, request, location):
        bin = self.bins[request.bin_id]
        if self.slots[bin.home_slot_id].reserved_by != request.id:
            raise ValueError("Movement requires owned reservation")
        bin.current_location = location
        bin.status = "IN_TRANSIT" if location == "carrier" else "AT_RECEPTION" if location == "reception" else "RESERVED"

    def complete(self, request, verified):
        bin = self.bins[request.bin_id]
        if not verified or bin.current_location != bin.home_slot_id:
            raise ValueError("Cannot release reservation without verified return")
        if self.slots[bin.home_slot_id].reserved_by != request.id:
            raise ValueError("Reservation ownership mismatch")
        self.slots[bin.home_slot_id].reserved_by = None
        bin.status = "AVAILABLE"

    def fail(self, request):
        bin = self.bins[request.bin_id]
        if self.slots[bin.home_slot_id].reserved_by == request.id:
            bin.status = "INCIDENT"
        request.status = State.FAILED

    def observe(self, observation: StockObservation):
        if observation.bin_id not in self.bins:
            raise ValueError("Unknown bin observation")
        if not math.isfinite(observation.confidence) or not 0 <= observation.confidence <= 1 or not math.isfinite(observation.timestamp):
            raise ValueError("Invalid observation quality or timestamp")
        previous = self.observations.get(observation.bin_id)
        if previous and observation.timestamp <= previous.timestamp:
            return
        self.observations[observation.bin_id] = observation
        if observation.level == Level.UNKNOWN or observation.confidence < self.min_confidence:
            return
        self.last_valid[observation.bin_id] = observation
        alert = self.alerts.get(observation.bin_id)
        if observation.level in (Level.EMPTY, Level.LOW):
            self.alerts[observation.bin_id] = {
                "bin_id": observation.bin_id,
                "sku": self.bins[observation.bin_id].sku,
                "active": True,
                "level": observation.level,
                "first_seen_sim_s": alert["first_seen_sim_s"] if alert and alert["active"] else observation.timestamp,
                "last_seen_sim_s": observation.timestamp,
                "source": observation.source,
            }
        elif observation.level == Level.OK and alert:
            alert.update(active=False, resolved_sim_s=observation.timestamp, source=observation.source)

    def snapshot(self):
        return {
            "catalog": self.catalog,
            "slots": {key: asdict(value) for key, value in self.slots.items()},
            "bins": {key: asdict(value) for key, value in self.bins.items()},
            "requests": {key: asdict(value) for key, value in self.requests.items()},
            "observations": {key: asdict(value) for key, value in self.observations.items()},
            "last_valid_observations": {key: asdict(value) for key, value in self.last_valid.items()},
            "alerts": self.alerts,
        }
