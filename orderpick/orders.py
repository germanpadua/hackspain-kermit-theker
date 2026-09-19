"""Recipe validation for assembly-station kitting."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Recipe:
    id: str
    workstation: str
    lines: list[tuple[str, int]]


def load_recipe(catalog: dict, path: Path | None = None) -> Recipe:
    raw = json.loads(path.read_text()) if path else catalog["order"]
    if not isinstance(raw, dict) or not isinstance(raw.get("lines"), list) or not raw["lines"]:
        raise ValueError("Recipe needs a nonempty list of lines")
    lines: list[tuple[str, int]] = []
    available = {b["sku"] for b in catalog["bins"]}
    for line in raw["lines"]:
        if not isinstance(line, dict):
            raise ValueError("Each order line needs sku and qty")
        sku, qty = line.get("sku"), line.get("qty")
        if not isinstance(sku, str) or sku not in available:
            raise ValueError(f"SKU has no configured source bin: {sku!r}")
        if type(qty) is not int or qty <= 0:
            raise ValueError("Quantities must be positive integers")
        if sku in {s for s, _ in lines}:
            raise ValueError(f"Duplicate SKU: {sku}")
        lines.append((sku, qty))
    return Recipe(str(raw.get("id", "KIT-DEMO")),
                  str(raw.get("workstation", "Montaje de reductores")), lines)


def slot_map(sim) -> dict:
    slots: dict[str, tuple[float, list[str]]] = {}
    entries = sorted(sim.catalog["bins"], key=lambda b: b["depth_row"])
    for entry in entries:
        x = sim.config["shelf"]["slots"][entry["slot"]]["x_m"]
        if entry["sku"] in slots and slots[entry["sku"]][0] != x:
            raise ValueError("Each SKU must occupy a single configured shelf slot")
        slots.setdefault(entry["sku"], (x, []))[1].append(entry["id"])
    return slots


def compartment_bounds(config: dict, index: int) -> tuple[float, float, float, float]:
    tray = config["tray"]
    width, depth, _ = tray["size_xyz_m"]
    wall, count = tray["wall_m"], tray["compartments"]
    inner = width - 2 * wall
    left = -inner / 2 + index * inner / count
    right = left + inner / count
    return (left + (wall / 2 if index else 0),
            right - (wall / 2 if index < count - 1 else 0),
            -depth / 2 + wall, depth / 2 - wall)


def compartment_for(config: dict, x: float, y: float) -> int | None:
    for i in range(config["tray"]["compartments"]):
        left, right, front, back = compartment_bounds(config, i)
        if left < x < right and front < y < back:
            return i
    return None


def validate_tray_recipe(config: dict, catalog: dict, recipe: Recipe) -> None:
    tray = config["tray"]
    skus, capacities = tray["compartment_skus"], tray["compartment_capacity"]
    if (len(skus) != tray["compartments"] or len(capacities) != len(skus)
            or len(set(skus)) != len(skus)
            or any(sku not in catalog["skus"] for sku in skus)
            or any(type(cap) is not int or cap <= 0 for cap in capacities)):
        raise ValueError("Tray needs one known SKU and positive capacity per compartment")
    capacity = dict(zip(skus, capacities))
    for sku, qty in recipe.lines:
        if qty > capacity.get(sku, 0):
            raise ValueError(f"Recipe exceeds tray capacity for {sku}")


def perception_scope(mode: str) -> dict:
    return {
        "product_pose": "RGBD_MARKER" if mode == "vision" else "ORACLE",
        "logistics_pose": "ORACLE",
        "tray_slip_guard": "ORACLE",
        "independent_verification": "SIMULATOR_TRUTH",
        "transport": "CONTACT",
        "geometry": "GRIPPABLE_PART_PROXIES",
        "human_interventions": 0,
    }
