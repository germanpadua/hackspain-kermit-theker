"""Visual perception adapter: renders the wrist camera, segments catalog
piece colors, and back-projects blob centroids to world points using the
camera's calibrated pose and intrinsics.

The controller calls this instead of oracle poses in VISION mode. It never
reads object body positions — camera extrinsics (mjData cam_xpos/cam_xmat)
are the declared calibration, and catalog colors are declared markers.
"""
import cv2
import numpy as np

MIN_AREA_PX = 8          # a piece must cover at least this many pixels
DEPTH_MAX_M = 0.75        # wrist camera reads within this range
SAT_MIN = 0.32            # reject unsaturated (shadow/steel/bin wall) pixels
MERGE_M = 0.05            # same-SKU hits closer than this are one unit
# Rendered hues per SKU (calibrated on the actual render — scene lighting
# whitewashes the flat rgba toward cyan, so bands match the OBSERVED hue)
HUE_BANDS = {"ENGRANAJE": (0, 30), "ESPARRAGO": (35, 79),
             "RODAMIENTO": (80, 115)}


class WristVision:
    def __init__(self, sim):
        self.sim = sim
        self.cam_id = int(sim.model.camera("wrist").id)
        self.sku_hue = {sid: info.get("marker_hue", HUE_BANDS.get(sid))
                        for sid, info in sim.catalog["skus"].items()}
        if any(band is None for band in self.sku_hue.values()):
            raise ValueError("Every SKU needs a marker_hue calibration")

    def _frame(self):
        rgb = self.sim.render("wrist")
        depth = self.sim.render("wrist", depth=True)
        return rgb, depth

    def _intrinsics(self, h, w):
        fovy = np.deg2rad(float(self.sim.model.cam_fovy[self.cam_id]))
        fy = 0.5 * h / np.tan(fovy / 2.0)
        return fy, fy, w / 2.0, h / 2.0

    def _extrinsics(self):
        """Calibrated camera pose (world <- camera)."""
        pos = self.sim.data.cam_xpos[self.cam_id].copy()
        mat = self.sim.data.cam_xmat[self.cam_id].reshape(3, 3).copy()
        return pos, mat

    def _to_world(self, u, v, d, cam_pos, cam_mat, intr):
        fx, fy, cx, cy = intr
        # mujoco camera: looks down its -z, +x right, +y up (v flips)
        ray = np.array([(u - cx) * d / fx, -(v - cy) * d / fy, -d])
        return cam_pos + cam_mat @ ray

    def _region_visible(self, rgb, depth, center_xy, half_xy, floor_z, z_band):
        if floor_z is None:
            return True
        h, w = depth.shape
        fx, fy, cx, cy = self._intrinsics(h, w)
        pos, mat = self._extrinsics()
        readable = 0
        for dx in (-0.8, 0, 0.8):
            for dy in (-0.8, 0, 0.8):
                target = np.array([center_xy[0] + dx * half_xy[0],
                                   center_xy[1] + dy * half_xy[1], floor_z + 0.01])
                local = mat.T @ (target - pos)
                distance = -local[2]
                if distance <= 0 or distance > DEPTH_MAX_M:
                    return False
                u, v = cx + fx * local[0] / distance, cy - fy * local[1] / distance
                if not (0 <= u < w and 0 <= v < h):
                    return False
                d = self._depth_at(depth, u, v)
                if d is None or np.max(rgb[int(v), int(u)]) < 15:
                    continue
                observed = self._to_world(u, v, d, pos, mat, (fx, fy, cx, cy))
                if floor_z - 0.025 < observed[2] < floor_z + z_band + 0.025:
                    readable += 1
        return readable >= 8

    def observe_bin(self, bin_id, bin_center_xy, bin_half_xy, floor_z=None):
        """Return (status, points) where status is 'ok'|'empty'|'unknown'
        and points are world xyz of detected pieces inside the bin bounds.
        'empty' = bin interior visible with no units; 'unknown' = the bin's
        footprint is not in view at all (occluded or aimed elsewhere).
        The ROI is the bin INTERIOR: wall/rim pixels (which share the piece
        hue under scene lighting) are cut by shrinking the footprint and by
        a floor-relative z band when floor_z is given."""
        return self.observe_region(bin_center_xy, bin_half_xy,
                                   floor_z=floor_z, shrink=0.02)

    def observe_region(self, center_xy, half_xy, floor_z=None, shrink=0.0,
                       z_band=0.055):
        """Generic ROI observation: hue-segment catalog SKUs, back-project
        each blob centroid with median depth, keep hits inside the (shrunk)
        footprint and optional floor-relative z band (z_band above floor)."""
        in_half = (max(half_xy[0] - shrink, 0.01),
                   max(half_xy[1] - shrink, 0.01))
        rgb, depth = self._frame()
        h, w = rgb.shape[:2]
        cam_pos, cam_mat = self._extrinsics()
        intr = self._intrinsics(h, w)

        # sanity: the bin centre must project into the image at a readable
        # depth, otherwise the observation is invalid (UNKNOWN != EMPTY)
        d_mid = self._depth_at(depth, w / 2, h / 2)
        if d_mid is None or not self._region_visible(
                rgb, depth, center_xy, in_half, floor_z, z_band):
            return "unknown", []

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(float)
        sat = hsv[..., 1] / 255.0
        hits = []
        for sid, (h0, h1) in self.sku_hue.items():
            mask = ((hsv[..., 0] >= h0) & (hsv[..., 0] <= h1)
                    & (sat > SAT_MIN)).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                    np.ones((3, 3), np.uint8))
            n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
            for i in range(1, n):
                if stats[i, cv2.CC_STAT_AREA] < MIN_AREA_PX:
                    continue
                px = np.argwhere(lab == i)
                points = [self._to_world(x, y, d, cam_pos, cam_mat, intr)
                          for y, x in px[::max(1, len(px) // 100)]
                          if (d := self._depth_at(depth, x, y)) is not None]
                if not points:
                    continue
                world = np.median(points, axis=0)
                if (abs(world[0] - center_xy[0]) < in_half[0]
                        and abs(world[1] - center_xy[1]) < in_half[1]
                        and (floor_z is None or
                             floor_z - 0.02 < world[2] < floor_z + z_band)):
                    hits.append((sid, world))
        if not hits:
            return ("empty" if floor_z is not None else "unknown"), []
        return "ok", _merge_units(hits)

    def surface_free(self, center_xy, half_xy, floor_z):
        """Scenario fixtures are declared magenta markers: a hit inside the
        surface ROI means the surface is OCCUPIED. Returns 'free',
        'occupied', or 'unknown' (ROI not readable — never treated free)."""
        rgb, depth = self._frame()
        h, w = rgb.shape[:2]
        d_mid = self._depth_at(depth, w / 2, h / 2)
        if d_mid is None:
            return "unknown"
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(float)
        mask = ((hsv[..., 0] >= 140) & (hsv[..., 0] <= 175)
                & (hsv[..., 1] / 255.0 > 0.35)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cam_pos, cam_mat = self._extrinsics()
        intr = self._intrinsics(h, w)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 30:
                continue
            u, v = cent[i]
            px = np.argwhere(lab == i)
            ds = np.array([self._depth_at(depth, x, y)
                           for y, x in px[::max(1, len(px) // 40)]])
            ds = np.array([d for d in ds if d is not None])
            if len(ds) == 0:
                continue
            world = self._to_world(u, v, float(np.median(ds)),
                                   cam_pos, cam_mat, intr)
            if (abs(world[0] - center_xy[0]) < half_xy[0]
                    and abs(world[1] - center_xy[1]) < half_xy[1]
                    and world[2] > floor_z - 0.02):
                return "occupied"
        return ("free" if self._region_visible(
            rgb, depth, center_xy, half_xy, floor_z, 0.02) else "unknown")

    def _depth_at(self, depth, x, y):
        xi = int(np.clip(x, 0, depth.shape[1] - 1))
        yi = int(np.clip(y, 0, depth.shape[0] - 1))
        d = float(depth[yi, xi])
        if not np.isfinite(d) or d <= 0 or d > DEPTH_MAX_M:
            return None
        return d


def _merge_units(hits):
    """Cluster same-SKU detections within MERGE_M into a single unit
    (a pawn's colored head can fragment into several blobs)."""
    groups = []
    for sid, p in hits:
        for group_sid, points in groups:
            if group_sid == sid and np.linalg.norm(
                    np.asarray(p)[:2] - points[0][:2]) < MERGE_M:
                points.append(np.asarray(p, float))
                break
        else:
            groups.append((sid, [np.asarray(p, float)]))
    units = []
    for sid, points in groups:
        top = max(p[2] for p in points)
        head = [p for p in points if p[2] >= top - 0.006]
        units.append((sid, np.mean(head, axis=0)))
    return units
