"""Cell contracts; product observations coexist with oracle-assisted logistics."""
from dataclasses import dataclass, field
from enum import StrEnum


XYZ = tuple[float, float, float]


class Phase(StrEnum):
    ORDER_RECEIVED = "ORDER_RECEIVED"
    PICKING = "PICKING"
    TOTE_DEPLETED = "TOTE_DEPLETED"
    KIT_PREPARED = "KIT_PREPARED"
    KIT_READY = "KIT_READY"
    ORDER_INCOMPLETE = "ORDER_INCOMPLETE"
    PERCEPTION_STOP = "PERCEPTION_STOP"
    IDLE = "IDLE"
    PLAN = "PLAN"
    DRIVE = "DRIVE"
    OBSERVE_BIN = "OBSERVE_BIN"
    PICK_UNIT = "PICK_UNIT"
    PLACE_TRAY = "PLACE_TRAY"
    CONFIRM_EMPTY = "CONFIRM_EMPTY"
    RESERVE_PARK_CHECK = "RESERVE_PARK_CHECK"
    REMOVE_FRONT_BIN = "REMOVE_FRONT_BIN"
    PARK_BIN = "PARK_BIN"
    VERIFY_PARK = "VERIFY_PARK"
    OBSERVE_REVEALED = "OBSERVE_REVEALED"
    ADVANCE_RESERVE = "ADVANCE_RESERVE"
    VERIFY_ADVANCE = "VERIFY_ADVANCE"
    VERIFY_ORDER = "VERIFY_ORDER"
    STOW = "STOW"
    DRIVE_RECEPTION = "DRIVE_RECEPTION"
    CHECK_TABLE = "CHECK_TABLE"
    PICK_TRAY = "PICK_TRAY"
    PLACE_TABLE = "PLACE_TABLE"
    VERIFY_DELIVERY = "VERIFY_DELIVERY"
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    FAILED = "FAILED"


class BinStatus(StrEnum):
    SHELVED = "SHELVED"            # in its slot, front or reserve row
    IN_HAND = "IN_HAND"            # currently held by the gripper
    PARKED = "PARKED"              # moved to the empty-bin parking stand
    ADVANCED = "ADVANCED"          # reserve pulled to the front row
    INCIDENT = "INCIDENT"


class UnitStatus(StrEnum):
    SHELVED = "SHELVED"
    IN_HAND = "IN_HAND"
    IN_TRAY = "IN_TRAY"
    DELIVERED = "DELIVERED"
    DROPPED = "DROPPED"
    INCIDENT = "INCIDENT"


class FillLevel(StrEnum):
    EMPTY = "EMPTY"
    NONEMPTY = "NONEMPTY"
    UNKNOWN = "UNKNOWN"            # never equals EMPTY


@dataclass
class PieceObservation:
    piece_id_guess: str | None     # None for detections without identity
    sku: str
    xyz_m: XYZ
    yaw_rad: float
    confidence: float
    timestamp: float
    source: str                    # "VISION" | "ORACLE"


@dataclass
class BinFillObservation:
    bin_id: str
    level: FillLevel
    piece_count: int | None        # honest count only when fully visible; else None
    confidence: float
    timestamp: float
    source: str


@dataclass
class SurfaceObservation:
    name: str                      # e.g. "reception", "park"
    free: bool | None              # None when UNKNOWN
    confidence: float
    timestamp: float
    source: str


@dataclass
class Observations:
    pieces: list[PieceObservation] = field(default_factory=list)
    fill: BinFillObservation | None = None
    surfaces: list[SurfaceObservation] = field(default_factory=list)


@dataclass
class Snapshot:
    """Proprioception + timestamps only. No world-object poses."""
    sim_time_s: float
    cart_x_m: float
    cart_vel_m_s: float
    arm_qpos: tuple[float, ...]
    arm_qvel: tuple[float, ...]
    gripper_open_m: float          # tendon length m; 0.04 = fully open
    gripper_force_n: float
    ee_xyz_m: XYZ                  # grasp point, forward kinematics
    ee_xmat: tuple[float, ...]
    fault: str | None = None


@dataclass
class OrderLine:
    sku: str
    qty: int
    picked: int = 0
    delivered_units: list[str] = field(default_factory=list)


@dataclass
class Order:
    id: str
    lines: list[OrderLine]
    status: str = "ACTIVE"
    missing: dict[str, int] = field(default_factory=dict)
    incidents: list[str] = field(default_factory=list)


@dataclass
class Alert:
    kind: str
    detail: str
    sim_time_s: float
    active: bool = True


@dataclass(frozen=True)
class Event:
    order_id: str | None
    type: str
    sim_time_s: float
    wall_time_s: float
    payload: dict[str, object] = field(default_factory=dict)
