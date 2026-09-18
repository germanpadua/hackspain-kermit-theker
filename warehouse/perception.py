import cv2
import numpy as np

from warehouse.contracts import Level, StockObservation


class OraclePerception:
    def __init__(self, sim):
        self.sim = sim

    def observe_bin(self, bin_id):
        return self.sim.observe_oracle(bin_id)


def estimate_stock(image, roi, bin_id, timestamp):
    unknown = StockObservation(bin_id, Level.UNKNOWN, None, 0.0, timestamp, "VISION_AREA")
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        return unknown
    x, y, width, height = roi
    if min(x, y) < 0 or min(width, height) <= 0 or x + width > image.shape[1] or y + height > image.shape[0]:
        return unknown
    hsv = cv2.cvtColor(image[y:y + height, x:x + width], cv2.COLOR_RGB2HSV)
    floor = cv2.inRange(hsv, (75, 60, 35), (105, 255, 255)) > 0
    material = cv2.inRange(hsv, (4, 110, 45), (27, 255, 255)) > 0
    visible_fraction = float(np.mean(floor | material))
    if visible_fraction < 0.85:
        return unknown
    occupied = float(np.mean(material))
    level = Level.EMPTY if occupied < 0.015 else Level.LOW if occupied < 0.25 else Level.OK
    return StockObservation(bin_id, level, None, min(0.95, visible_fraction), timestamp, "VISION_AREA")
