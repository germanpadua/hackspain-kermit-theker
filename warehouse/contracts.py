from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


XYZ = tuple[float, float, float]


class State(StrEnum):
    IDLE = "IDLE"
    RESERVE = "RESERVE"
    APPROACH = "APPROACH"
    OBSERVE = "OBSERVE"
    ALIGN = "ALIGN"
    EXTRACT = "EXTRACT"
    VERIFY_PICK = "VERIFY_PICK"
    TRANSFER = "TRANSFER"
    PLACE_AT_RECEPTION = "PLACE_AT_RECEPTION"
    WAIT_OPERATOR = "WAIT_OPERATOR"
    OBSERVE_STOCK = "OBSERVE_STOCK"
    PICK_AT_RECEPTION = "PICK_AT_RECEPTION"
    RETURN = "RETURN"
    PLACE_HOME = "PLACE_HOME"
    VERIFY_HOME = "VERIFY_HOME"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class Level(StrEnum):
    EMPTY = "EMPTY"
    LOW = "LOW"
    OK = "OK"
    UNKNOWN = "UNKNOWN"


@dataclass
class Slot:
    id: str
    nominal_pick_xyz_m: XYZ
    size_xyz_m: XYZ
    reserved_by: str | None = None


@dataclass
class Bin:
    id: str
    sku: str
    home_slot_id: str
    current_location: str
    status: str = "AVAILABLE"


@dataclass(frozen=True)
class PoseObservation:
    bin_id: str
    xyz_m: XYZ
    yaw_rad: float
    confidence: float
    timestamp: float
    source: str


@dataclass(frozen=True)
class StockObservation:
    bin_id: str
    level: Level
    count_estimate: int | None
    confidence: float
    timestamp: float
    source: str


@dataclass
class Request:
    id: str
    sku: str
    bin_id: str
    status: State = State.IDLE


@dataclass(frozen=True)
class Event:
    request_id: str | None
    type: str
    sim_time_s: float
    wall_time_s: float
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Snapshot:
    sim_time_s: float
    axes_xyz_m: XYZ
    velocities_xyz_m_s: XYZ
    attached_bin_id: str | None
    home_clear: bool
    fault: str | None = None


@dataclass(frozen=True)
class OperatorEvent:
    request_id: str
    type: str = "CONFIRM_REMOVED"
    source: str = "HUMAN"


@dataclass(frozen=True)
class Observations:
    pose: PoseObservation | None = None
    stock: StockObservation | None = None
