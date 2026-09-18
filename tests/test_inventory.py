from warehouse.contracts import Level, StockObservation
from warehouse.inventory import Inventory


def test_reservation_survives_delivery_and_failure():
    inventory = Inventory.load()
    request = inventory.request_material("TORNILLOS")
    inventory.reserve(request)
    inventory.move(request, "reception")
    assert inventory.slots["A1"].reserved_by == request.id
    assert inventory.bins[request.bin_id].home_slot_id == "A1"
    inventory.fail(request)
    assert inventory.slots["A1"].reserved_by == request.id
    assert inventory.bins[request.bin_id].status == "INCIDENT"


def test_unknown_preserves_alert_and_last_valid_observation():
    inventory = Inventory.load()
    for timestamp in (1.0, 2.0):
        inventory.observe(StockObservation("box-1", Level.LOW, None, 0.9, timestamp, "VISION_AREA"))
    assert len(inventory.alerts) == 1
    inventory.observe(StockObservation("box-1", Level.UNKNOWN, None, 0.0, 3.0, "VISION_AREA"))
    assert inventory.alerts["box-1"]["active"]
    assert inventory.last_valid["box-1"].timestamp == 2.0
    assert inventory.observations["box-1"].level == Level.UNKNOWN
    inventory.observe(StockObservation("box-1", Level.OK, None, 0.2, 4.0, "VISION_AREA"))
    assert inventory.alerts["box-1"]["active"]
    inventory.observe(StockObservation("box-1", Level.OK, None, 0.9, 5.0, "VISION_AREA"))
    assert not inventory.alerts["box-1"]["active"]


def test_reservation_is_exclusive():
    import pytest

    inventory = Inventory.load()
    first = inventory.request_material("TORNILLOS")
    second = inventory.request_material("TORNILLOS")
    inventory.reserve(first)
    with pytest.raises(ValueError, match="reserved"):
        inventory.reserve(second)
