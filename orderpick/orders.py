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


def perception_scope(mode: str) -> dict:
    return {
        "product_pose": "RGBD_MARKER" if mode == "vision" else "ORACLE",
        "logistics_pose": "ORACLE",
        "tray_slip_guard": "ORACLE",
        "transport": "CONTACT",
        "geometry": "GRIPPABLE_PART_PROXIES",
        "human_interventions": 0,
    }
