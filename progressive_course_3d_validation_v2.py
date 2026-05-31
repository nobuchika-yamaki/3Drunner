#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Progressive 3D Course Validation v2 for an Integrated Autonomous Explorer
=====================================================================

This script implements a true 3D progressive course.

Coordinates
-----------
position = (x, y, z)

x: forward progression
y: lateral lane
z: height

No semantic hint module is used.

Central task
------------
The agent must move through a progressively complex 3D world while maintaining
viability. The world includes height changes, ledges, climbable steps, drops,
gaps, walls, slippery surfaces, slopes, hidden dangers, resources, rest sites,
and intervention zones requiring JUMP, CLIMB, DROP, BRAKE, SCAN, and PROBE.

Main outcome
------------
course_progress = max_x / (course_length - 1)

Core variants
-------------
full_core:
    Viability regulation + proto-valence Q + body-consequence learning.

no_q:
    Q is removed from action scoring and state trajectory.

damage_only_q:
    Q is reduced to recent pain/damage memory only.

no_body_model:
    Learned body-consequence prediction is not used.

frozen_predictor:
    The predictor exists but is not updated.

Example
-------
python3 -u ~/Desktop/progressive_course_3d_validation_v2.py \
  --mode full \
  --episodes 50 \
  --steps 1200 \
  --outdir ~/Desktop/progressive_course_3d_validation_n50 \
  --resume \
  2>&1 | tee ~/Desktop/progressive_course_3d_validation_n50.log
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import random
import statistics
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ACTIONS = [
    "MOVE_FWD", "MOVE_BACK", "MOVE_LEFT", "MOVE_RIGHT",
    "JUMP", "CLIMB", "DROP", "BRAKE", "REST", "SCAN", "PROBE",
]

MOVE_ACTIONS = {"MOVE_FWD", "MOVE_BACK", "MOVE_LEFT", "MOVE_RIGHT", "JUMP", "CLIMB", "DROP"}
ACTIVE_ACTIONS = {"MOVE_FWD", "MOVE_BACK", "MOVE_LEFT", "MOVE_RIGHT", "JUMP", "CLIMB", "DROP", "BRAKE", "PROBE"}

LEVELS = ["simple3d", "terrain3d", "gaps3d", "hidden3d", "full3d"]
VARIANTS = ["full_core", "no_q", "damage_only_q", "no_body_model", "frozen_predictor"]

SURFACE_MU = {
    "normal": 0.42,
    "slippery": 0.05,
    "rough": 0.75,
    "slope": 0.25,
    "bouncy": 0.34,
}

ZONE_NAMES = {
    0: "safe_flat",
    1: "resource_rest",
    2: "height_terrain",
    3: "gaps_drops",
    4: "walls_ledges",
    5: "hidden_risk",
    6: "intervention",
}


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def safe_mean(xs: Iterable[float], default: float = 0.0) -> float:
    vals = [float(x) for x in xs if x is not None and not math.isnan(float(x))]
    return sum(vals) / len(vals) if vals else default


def safe_sd(xs: Iterable[float], default: float = 0.0) -> float:
    vals = [float(x) for x in xs if x is not None and not math.isnan(float(x))]
    if len(vals) < 2:
        return default
    return statistics.stdev(vals)


def mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def append_csv(path: Path, fieldnames: List[str], row: Dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


class ProgressLogger:
    def __init__(self, outdir: Path) -> None:
        self.t0 = time.time()
        self.path = outdir / "run.log"
        mkdir(outdir)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(f"\n--- started {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")

    def log(self, msg: str) -> None:
        elapsed = time.time() - self.t0
        line = f"[{elapsed:8.1f}s] {msg}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


@dataclasses.dataclass
class Config:
    outdir: Path
    seed: int
    episodes: int
    steps: int
    length: int
    width: int
    height: int
    resume: bool
    trace_stride: int
    mode: str
    levels: List[str]
    variants: List[str]
    make_figures: bool


class Platform:
    __slots__ = ("exists", "kind", "hidden", "known", "surface", "mu", "slope", "depleted", "climbable")

    def __init__(self) -> None:
        self.exists = True
        self.kind = "EMPTY"
        self.hidden: Optional[str] = None
        self.known = True
        self.surface = "normal"
        self.mu = SURFACE_MU["normal"]
        self.slope = 0.0
        self.depleted = False
        self.climbable = False

    def effective(self) -> str:
        if not self.exists:
            return "VOID"
        if self.kind == "UNKNOWN" and self.known:
            return str(self.hidden)
        if self.kind == "UNKNOWN" and not self.known:
            return "UNKNOWN"
        return self.kind


class ProgressiveCourse3D:
    def __init__(self, cfg: Config, level: str, episode: int, seed: int) -> None:
        self.cfg = cfg
        self.level = level
        self.episode = episode
        self.rng = random.Random(seed)
        self.length = cfg.length
        self.width = cfg.width
        self.height = cfg.height
        self.start = (0, self.width // 2, 1)
        self.pos = self.start
        self.prev_pos = self.start
        self.max_x = 0
        self.max_z = self.start[2]
        self.visited = set([self.pos])
        self.t = 0

        self.columns: Dict[Tuple[int, int], Platform] = {}
        self.floor_z: Dict[Tuple[int, int], Optional[int]] = {}

        self.unknown_revealed = 0
        self.resources_collected = 0
        self.rest_uses = 0
        self.danger_entries = 0
        self.falls = 0
        self.fall_distance_total = 0.0
        self.slips = 0
        self.collisions = 0
        self.successful_jumps = 0
        self.failed_jumps = 0
        self.successful_climbs = 0
        self.failed_climbs = 0
        self.controlled_drops = 0
        self.successful_brakes = 0
        self.probes = 0
        self.safe_intervention_success = 0

        self._build()

    def zone_index(self, x: int) -> int:
        z = int(7 * x / max(1, self.length))
        return max(0, min(6, z))

    def zone_name(self, x: Optional[int] = None) -> str:
        if x is None:
            x = self.pos[0]
        return ZONE_NAMES[self.zone_index(x)]

    def level_allows_zone(self, zone: int) -> bool:
        if self.level == "simple3d":
            return zone <= 1
        if self.level == "terrain3d":
            return zone <= 2
        if self.level == "gaps3d":
            return zone <= 3
        if self.level == "hidden3d":
            return zone <= 5
        if self.level == "full3d":
            return zone <= 6
        return zone <= 1

    def in_xy(self, x: int, y: int) -> bool:
        return 0 <= x < self.length and 0 <= y < self.width

    def platform_at(self, x: int, y: int) -> Optional[Platform]:
        if not self.in_xy(x, y):
            return None
        z = self.floor_z.get((x, y))
        if z is None:
            return None
        p = self.columns.get((x, y))
        if p is None or not p.exists:
            return None
        return p

    def cell_at_pos(self, pos: Tuple[int, int, int]) -> Optional[Platform]:
        x, y, z = pos
        p = self.platform_at(x, y)
        if p is None:
            return None
        if self.floor_z[(x, y)] != z:
            return None
        return p

    def current_platform(self) -> Platform:
        p = self.cell_at_pos(self.pos)
        if p is None:
            # Should not happen after build/start guarantee.
            p = Platform()
        return p

    def _set_column(self, x: int, y: int, z: Optional[int], platform: Optional[Platform]) -> None:
        self.floor_z[(x, y)] = z
        if platform is not None:
            self.columns[(x, y)] = platform

    def _base_height(self, x: int, y: int, zone: int) -> int:
        center = self.width // 2
        lateral = abs(y - center)
        if zone <= 1:
            return 1
        if zone == 2:
            # Height terrain: rolling ledges and slopes.
            return max(0, min(self.height - 1, 1 + ((x // 8) % 3) - (1 if lateral == 2 and x % 9 < 4 else 0)))
        if zone == 3:
            return max(0, min(self.height - 1, 1 + ((x // 10) % 2)))
        if zone == 4:
            return max(0, min(self.height - 1, 1 + ((x // 7) % 4)))
        if zone == 5:
            return max(0, min(self.height - 1, 1 + ((x // 6) % 3)))
        if zone == 6:
            return max(0, min(self.height - 1, 1 + ((x // 5) % 4) - (1 if lateral == 2 else 0)))
        return 1

    def _build(self) -> None:
        center = self.width // 2

        for x in range(self.length):
            zone = self.zone_index(x)
            effective_zone = zone if self.level_allows_zone(zone) else 1

            for y in range(self.width):
                p = Platform()
                p.kind = "EMPTY"
                p.surface = "normal"
                p.mu = SURFACE_MU["normal"]
                p.slope = 0.0
                p.climbable = False
                z = self._base_height(x, y, effective_zone)

                r = self.rng.random()

                if effective_zone == 0:
                    if r < 0.025:
                        p.kind = "RESOURCE"

                elif effective_zone == 1:
                    if r < 0.065:
                        p.kind = "RESOURCE"
                    elif r < 0.105:
                        p.kind = "REST"

                elif effective_zone == 2:
                    rr = self.rng.random()
                    if rr < 0.22:
                        p.surface = "slippery"
                    elif rr < 0.38:
                        p.surface = "rough"
                    elif rr < 0.60:
                        p.surface = "slope"
                        p.slope = self.rng.uniform(0.10, 0.32)
                    else:
                        p.surface = "normal"
                    p.mu = SURFACE_MU[p.surface]
                    if r < 0.045:
                        p.kind = "RESOURCE"
                    elif r < 0.065:
                        p.kind = "REST"

                elif effective_zone == 3:
                    if r < 0.12 and y != center:
                        self._set_column(x, y, None, None)
                        continue
                    elif r < 0.19:
                        p.kind = "DANGER"
                    elif r < 0.25:
                        p.kind = "WALL"
                    if self.rng.random() < 0.15:
                        p.surface = "slippery"
                        p.mu = SURFACE_MU["slippery"]
                    if self.rng.random() < 0.04:
                        p.kind = "RESOURCE"

                elif effective_zone == 4:
                    # Ledges and climbable structures.
                    if r < 0.14:
                        p.kind = "WALL"
                    elif r < 0.22:
                        z = min(self.height - 1, z + 2)
                        p.climbable = True
                    elif r < 0.29 and y != center:
                        self._set_column(x, y, None, None)
                        continue
                    elif r < 0.35:
                        p.kind = "DANGER"
                    if self.rng.random() < 0.08:
                        p.kind = "REST"

                elif effective_zone == 5:
                    if r < 0.18:
                        p.kind = "UNKNOWN"
                        p.known = False
                        p.hidden = self.rng.choices(
                            ["RESOURCE", "DANGER", "REST", "EMPTY"],
                            weights=[0.22, 0.46, 0.14, 0.18],
                            k=1,
                        )[0]
                    elif r < 0.25:
                        p.kind = "DANGER"
                    elif r < 0.31 and y != center:
                        self._set_column(x, y, None, None)
                        continue
                    if self.rng.random() < 0.16:
                        p.surface = self.rng.choice(["slippery", "slope", "rough"])
                        p.mu = SURFACE_MU[p.surface]
                        p.slope = self.rng.uniform(0.08, 0.35) if p.surface == "slope" else 0.0

                elif effective_zone == 6:
                    if r < 0.11 and y != center:
                        self._set_column(x, y, None, None)
                        continue
                    elif r < 0.22:
                        p.kind = "WALL"
                        if self.rng.random() < 0.55:
                            p.climbable = True
                    elif r < 0.31:
                        p.kind = "DANGER"
                    elif r < 0.40:
                        p.kind = "UNKNOWN"
                        p.known = False
                        p.hidden = self.rng.choices(
                            ["RESOURCE", "DANGER", "REST", "EMPTY"],
                            weights=[0.18, 0.50, 0.12, 0.20],
                            k=1,
                        )[0]
                    if self.rng.random() < 0.25:
                        p.surface = self.rng.choice(["slippery", "slope", "rough", "normal"])
                        p.mu = SURFACE_MU[p.surface]
                        p.slope = self.rng.uniform(0.08, 0.40) if p.surface == "slope" else 0.0

                self._set_column(x, y, z, p)

        # Ensure central corridor is playable but still 3D and nontrivial.
        for x in range(self.length):
            zone = self.zone_index(x)
            y = center
            z = self._base_height(x, y, zone if self.level_allows_zone(zone) else 1)
            p = self.platform_at(x, y)
            if p is None:
                p = Platform()
                self._set_column(x, y, z, p)
            else:
                self.floor_z[(x, y)] = z
            p.exists = True
            if zone <= 2 and p.kind in {"WALL", "DANGER"}:
                p.kind = "EMPTY"
            if x in {max(3, self.length // 6), max(6, self.length // 3), max(8, self.length // 2)}:
                p.kind = "REST"
                p.known = True
                p.hidden = None
            if x in {max(4, self.length // 5), max(10, 2 * self.length // 5), max(12, 3 * self.length // 5)}:
                p.kind = "RESOURCE"
                p.known = True
                p.hidden = None

        # Force a few true 3D challenges in full routes.
        if self.length >= 40:
            for x in [int(self.length * 0.42), int(self.length * 0.58), int(self.length * 0.75)]:
                if 0 <= x < self.length:
                    y = center
                    p = self.platform_at(x, y)
                    if p is not None:
                        self.floor_z[(x, y)] = min(self.height - 1, self.floor_z[(x, y)] + 2)
                        p.climbable = True
                        p.kind = "EMPTY"
            for x in [int(self.length * 0.50), int(self.length * 0.70)]:
                if 0 <= x < self.length:
                    # Central gap that can be jumped or routed around.
                    self._set_column(x, center, None, None)

        # Start/goal
        sp = Platform()
        sp.kind = "REST"
        sp.surface = "normal"
        sp.mu = SURFACE_MU["normal"]
        self._set_column(0, center, 1, sp)
        self.pos = (0, center, 1)
        self.start = self.pos

        gp = Platform()
        gp.kind = "GOAL"
        gp.surface = "normal"
        gp.mu = SURFACE_MU["normal"]
        self._set_column(self.length - 1, center, self.floor_z.get((self.length - 1, center), 1) or 1, gp)

    def observe(self, radius: int = 2) -> List[Dict[str, Any]]:
        x0, y0, z0 = self.pos
        out = []
        for dx in range(-1, radius + 1):
            for dy in range(-radius, radius + 1):
                x = x0 + dx
                y = y0 + dy
                if not self.in_xy(x, y):
                    continue
                dist = abs(dx) + abs(dy)
                if dist > radius:
                    continue
                p = self.platform_at(x, y)
                z = self.floor_z.get((x, y))
                if p is None or z is None:
                    out.append({
                        "pos": (x, y, None),
                        "rel": (dx, dy, None),
                        "dist": dist,
                        "effective": "VOID",
                        "kind": "VOID",
                        "known": True,
                        "surface": "void",
                        "mu": 0.0,
                        "slope": 0.0,
                        "height_diff": -999,
                        "climbable": False,
                    })
                else:
                    out.append({
                        "pos": (x, y, z),
                        "rel": (dx, dy, z - z0),
                        "dist": dist,
                        "effective": p.effective(),
                        "kind": p.kind,
                        "known": p.known,
                        "hidden": p.hidden,
                        "surface": p.surface,
                        "mu": p.mu,
                        "slope": p.slope,
                        "height_diff": z - z0,
                        "climbable": p.climbable,
                        "depleted": p.depleted,
                    })
        return out

    def local_pressures(self) -> Dict[str, float]:
        obs = self.observe(radius=2)
        danger = unknown = resource = rest = gap = wall = climb = drop = vertical = forward_open = 0.0
        denom = 1e-9
        for o in obs:
            w = 1.0 / (1.0 + float(o["dist"]))
            denom += w
            eff = o["effective"]
            if eff == "DANGER":
                danger += w
            if eff == "UNKNOWN":
                unknown += w
            if eff == "RESOURCE" and not bool(o.get("depleted", False)):
                resource += w
            if eff == "REST":
                rest += w
            if eff == "VOID":
                gap += w
            if eff == "WALL":
                wall += w
            hd = float(o["height_diff"]) if o["height_diff"] != -999 else -999
            if hd > 1:
                climb += w
            if hd < -1 and hd != -999:
                drop += w
            if hd != -999:
                vertical += w * min(1.0, abs(hd) / max(1, self.height))
            if o["rel"][0] > 0 and eff not in {"DANGER", "WALL", "VOID"} and hd <= 1:
                forward_open += w

        return {
            "danger": clamp(danger / denom),
            "unknown": clamp(unknown / denom),
            "resource": clamp(resource / denom),
            "rest": clamp(rest / denom),
            "gap": clamp(gap / denom),
            "wall": clamp(wall / denom),
            "climb": clamp(climb / denom),
            "drop": clamp(drop / denom),
            "vertical": clamp(vertical / denom),
            "forward_open": clamp(forward_open / denom),
        }

    def scan(self) -> int:
        revealed = 0
        for o in self.observe(radius=2):
            pos = o["pos"]
            if pos[2] is None:
                continue
            p = self.platform_at(pos[0], pos[1])
            if p is not None and p.kind == "UNKNOWN" and not p.known:
                p.known = True
                revealed += 1
                self.unknown_revealed += 1
        return revealed

    def target_xy(self, action: str) -> Tuple[int, int]:
        x, y, z = self.pos
        if action in {"MOVE_FWD", "CLIMB", "DROP"}:
            return (x + 1, y)
        if action == "MOVE_BACK":
            return (x - 1, y)
        if action == "MOVE_LEFT":
            return (x, y - 1)
        if action == "MOVE_RIGHT":
            return (x, y + 1)
        if action == "JUMP":
            return (x + 2, y)
        return (x, y)

    def step(self, action: str, stability: float, fatigue: float) -> Dict[str, Any]:
        self.t += 1
        self.prev_pos = self.pos
        old_x, old_y, old_z = self.pos
        current = self.current_platform()

        out = {
            "action": action,
            "event": "none",
            "old_pos": self.prev_pos,
            "new_pos": self.pos,
            "zone": self.zone_name(),
            "old_z": old_z,
            "new_z": old_z,
            "height_diff": 0,
            "fall_distance": 0,
            "damage": 0.0,
            "gain": 0.0,
            "energy_cost": 0.0,
            "fatigue_delta": 0.0,
            "stability_delta": 0.0,
            "pain": 0.0,
            "progress_delta": 0.0,
            "vertical_progress_delta": 0.0,
            "slip": 0,
            "collision": 0,
            "fall": 0,
            "danger_entry": 0,
            "unknown_revealed": 0,
            "resource_collected": 0,
            "rest_recovery": 0,
            "jump_success": 0,
            "climb_success": 0,
            "drop_success": 0,
            "brake_success": 0,
            "probe": 0,
            "safe_intervention_success": 0,
            "surface": current.surface,
            "mu": current.mu,
            "slope": current.slope,
            "entered_kind": current.effective(),
        }

        if action not in ACTIONS:
            action = "SCAN"

        if action in MOVE_ACTIONS:
            tx, ty = self.target_xy(action)
            if not self.in_xy(tx, ty):
                out["event"] = "boundary_collision"
                out["collision"] = 1
                out["damage"] += 0.022
                out["pain"] += 0.20
                out["stability_delta"] -= 0.06
                self.collisions += 1
            else:
                target_platform = self.platform_at(tx, ty)
                target_z = self.floor_z.get((tx, ty))
                if target_platform is None or target_z is None:
                    # Void or gap.
                    if action == "JUMP":
                        out["event"] = "jump_to_void_failed"
                        self.failed_jumps += 1
                    else:
                        out["event"] = "fall_into_gap"
                    fall_distance = max(1, old_z + 1)
                    out["fall"] = 1
                    out["fall_distance"] = fall_distance
                    out["damage"] += 0.045 + 0.025 * fall_distance + 0.025 * fatigue
                    out["pain"] += 0.38 + 0.12 * fall_distance
                    out["stability_delta"] -= 0.16 + 0.04 * fall_distance
                    self.falls += 1
                    self.fall_distance_total += fall_distance
                else:
                    hd = target_z - old_z
                    out["height_diff"] = hd

                    moved = False
                    safe_intervention = 0

                    if action in {"MOVE_FWD", "MOVE_BACK", "MOVE_LEFT", "MOVE_RIGHT"}:
                        if target_platform.effective() == "WALL":
                            out["event"] = "wall_collision"
                            out["collision"] = 1
                            out["damage"] += 0.026
                            out["pain"] += 0.22
                            out["stability_delta"] -= 0.07
                            self.collisions += 1
                        elif hd > 1:
                            out["event"] = "ledge_blocked"
                            out["collision"] = 1
                            out["damage"] += 0.012
                            out["pain"] += 0.10
                            self.collisions += 1
                        elif hd < -1:
                            # uncontrolled drop
                            moved = True
                            fall_distance = abs(hd)
                            out["event"] = "uncontrolled_drop"
                            out["fall"] = 1
                            out["fall_distance"] = fall_distance
                            out["damage"] += 0.025 + 0.025 * fall_distance + 0.020 * fatigue
                            out["pain"] += 0.22 + 0.12 * fall_distance
                            out["stability_delta"] -= 0.12 + 0.045 * fall_distance
                            self.falls += 1
                            self.fall_distance_total += fall_distance
                        else:
                            moved = True
                            out["event"] = "move"

                    elif action == "CLIMB":
                        climb_risk = clamp(0.05 + 0.22 * fatigue + 0.20 * (1.0 - stability) + 0.08 * max(0, hd - 1))
                        if target_platform.effective() == "WALL" and not target_platform.climbable:
                            out["event"] = "climb_blocked_wall"
                            out["collision"] = 1
                            self.failed_climbs += 1
                        elif 1 <= hd <= 3 and self.rng.random() > climb_risk:
                            moved = True
                            out["event"] = "climb_success"
                            out["climb_success"] = 1
                            safe_intervention = 1
                            self.successful_climbs += 1
                        else:
                            out["event"] = "climb_failed"
                            out["damage"] += 0.018 + 0.025 * climb_risk
                            out["pain"] += 0.18 + 0.20 * climb_risk
                            out["stability_delta"] -= 0.10
                            self.failed_climbs += 1

                    elif action == "DROP":
                        if hd < 0:
                            moved = True
                            fall_distance = abs(hd)
                            out["event"] = "controlled_drop"
                            out["drop_success"] = 1
                            safe_intervention = 1
                            out["damage"] += 0.006 * max(0, fall_distance - 1)
                            out["pain"] += 0.05 * max(0, fall_distance - 1)
                            out["stability_delta"] -= 0.025 * fall_distance
                            self.controlled_drops += 1
                        elif hd <= 1 and target_platform.effective() != "WALL":
                            moved = True
                            out["event"] = "drop_forward"
                        else:
                            out["event"] = "drop_blocked"

                    elif action == "JUMP":
                        mid_x = old_x + 1
                        mid_void = (not self.in_xy(mid_x, old_y)) or self.platform_at(mid_x, old_y) is None
                        jump_risk = clamp(0.08 + 0.30 * fatigue + 0.26 * (1.0 - stability) + 0.08 * max(0, abs(hd) - 1))
                        if target_platform.effective() in {"WALL", "DANGER"} or abs(hd) > 2 or self.rng.random() < jump_risk:
                            out["event"] = "jump_failed"
                            out["fall"] = 1
                            out["damage"] += 0.045 + 0.040 * jump_risk + 0.012 * abs(hd)
                            out["pain"] += 0.32 + 0.28 * jump_risk
                            out["stability_delta"] -= 0.15
                            self.failed_jumps += 1
                        else:
                            moved = True
                            out["event"] = "jump_success"
                            out["jump_success"] = 1
                            safe_intervention = int(mid_void or abs(hd) > 0)
                            self.successful_jumps += 1

                    if moved:
                        self.pos = (tx, ty, target_z)
                        out["new_pos"] = self.pos
                        out["new_z"] = target_z
                        out["vertical_progress_delta"] = max(0, target_z - old_z)
                        self.max_z = max(self.max_z, target_z)
                        p = target_platform
                        eff = p.effective()

                        if p.kind == "UNKNOWN" and not p.known:
                            p.known = True
                            eff = p.effective()
                            out["unknown_revealed"] += 1
                            self.unknown_revealed += 1
                            out["event"] = "entered_unknown_" + eff

                        # Terrain and surface risk
                        slip_risk = clamp(
                            max(0.0, 0.18 - p.mu) * 2.2
                            + max(0.0, p.slope) * 1.0
                            + 0.30 * fatigue
                            + 0.25 * (1.0 - stability)
                            + 0.04 * max(0, abs(hd) - 1)
                        )
                        if action not in {"JUMP", "CLIMB", "DROP"} and self.rng.random() < slip_risk:
                            out["event"] = "slip"
                            out["slip"] = 1
                            out["damage"] += 0.018 + 0.030 * slip_risk
                            out["pain"] += 0.18 + 0.28 * slip_risk
                            out["stability_delta"] -= 0.10 + 0.12 * slip_risk
                            self.slips += 1

                        if eff == "DANGER":
                            out["event"] = "danger"
                            out["danger_entry"] = 1
                            out["damage"] += 0.050 + 0.020 * fatigue + 0.010 * max(0, abs(hd))
                            out["pain"] += 0.55
                            out["stability_delta"] -= 0.12
                            self.danger_entries += 1
                        elif eff == "RESOURCE" and not p.depleted:
                            out["event"] = "resource"
                            out["gain"] += 0.105
                            out["resource_collected"] = 1
                            p.depleted = True
                            self.resources_collected += 1
                        elif eff == "REST":
                            out["event"] = "rest_cell"
                            out["gain"] += 0.015
                            out["rest_recovery"] = 1
                        elif eff == "GOAL":
                            out["event"] = "goal"
                            out["gain"] += 0.200

                        out["surface"] = p.surface
                        out["mu"] = p.mu
                        out["slope"] = p.slope
                        out["entered_kind"] = eff
                        out["safe_intervention_success"] += safe_intervention
                        self.safe_intervention_success += safe_intervention

                        self.max_x = max(self.max_x, self.pos[0])
                        self.visited.add(self.pos)
                        out["progress_delta"] = max(0.0, self.pos[0] - old_x)

                    # Common movement cost
                    cp = self.current_platform()
                    terrain_cost = max(0.0, 0.13 - cp.mu) * 0.020 + max(0.0, cp.slope) * 0.018
                    vertical_cost = 0.006 * max(0, target_z - old_z) if target_z is not None else 0.0
                    action_cost = 0.020 * float(action == "JUMP") + 0.016 * float(action == "CLIMB")
                    out["energy_cost"] += 0.006 + terrain_cost + vertical_cost + action_cost
                    out["fatigue_delta"] += 0.003 + 0.45 * action_cost + max(0.0, cp.slope) * 0.010

        elif action == "BRAKE":
            cp = self.current_platform()
            out["event"] = "brake"
            out["energy_cost"] += 0.007
            out["fatigue_delta"] += 0.0015
            if cp.surface in {"slippery", "slope"} or cp.slope > 0.08:
                out["stability_delta"] += 0.060
                out["brake_success"] = 1
                out["safe_intervention_success"] = 1
                self.successful_brakes += 1
                self.safe_intervention_success += 1
            else:
                out["stability_delta"] += 0.010

        elif action == "REST":
            cp = self.current_platform()
            if cp.effective() == "REST":
                out["event"] = "rest"
                out["gain"] += 0.080
                out["fatigue_delta"] -= 0.060
                out["stability_delta"] += 0.060
                out["rest_recovery"] = 1
                self.rest_uses += 1
            else:
                out["event"] = "pause"
                out["gain"] += 0.004
                out["fatigue_delta"] -= 0.010
                out["stability_delta"] += 0.008

        elif action == "SCAN":
            out["event"] = "scan"
            revealed = self.scan()
            out["unknown_revealed"] = revealed
            out["gain"] += 0.006 * revealed
            out["energy_cost"] += 0.002
            out["fatigue_delta"] += 0.001

        elif action == "PROBE":
            cp = self.current_platform()
            out["event"] = "probe"
            out["probe"] = 1
            out["energy_cost"] += 0.010
            out["fatigue_delta"] += 0.003
            self.probes += 1
            out["unknown_revealed"] = self.scan()
            out["friction_observed"] = cp.mu + self.rng.gauss(0, 0.010)
            out["slope_observed"] = cp.slope + self.rng.gauss(0, 0.010)
            out["height_observed"] = self.pos[2]
            if cp.surface in {"slippery", "slope", "rough"} or self.zone_index(self.pos[0]) >= 3:
                out["safe_intervention_success"] = 1
                self.safe_intervention_success += 1

        out["damage"] = max(0.0, out["damage"])
        out["gain"] = max(0.0, out["gain"])
        out["pain"] = clamp(out["pain"])
        out["new_pos"] = self.pos
        out["new_z"] = self.pos[2]
        return out

    def progress(self) -> float:
        return self.max_x / max(1, self.length - 1)

    def zone_reached(self) -> int:
        return self.zone_index(self.max_x)


class ProtoValenceQ:
    def __init__(self) -> None:
        self.q = 0.0
        self.pain_memory = 0.0
        self.danger_memory = 0.0
        self.vertical_risk_memory = 0.0
        self.comfort_memory = 0.50
        self.action_possibility = 0.50
        self.q_history: List[float] = []
        self.damage_history: List[float] = []

    def update(self, viability: float, energy: float, fatigue: float, stability: float,
               pressures: Dict[str, float], outcome: Dict[str, Any]) -> float:
        pain = clamp(float(outcome.get("pain", 0.0)) + 2.4 * float(outcome.get("damage", 0.0)))
        vertical = clamp(
            pressures.get("gap", 0.0)
            + pressures.get("drop", 0.0)
            + pressures.get("climb", 0.0)
            + 0.30 * float(outcome.get("fall", 0.0))
        )
        danger = clamp(
            pressures.get("danger", 0.0)
            + 0.35 * pressures.get("gap", 0.0)
            + 0.25 * pressures.get("wall", 0.0)
            + 0.35 * pressures.get("vertical", 0.0)
            + 0.45 * float(outcome.get("danger_entry", 0.0))
            + 0.35 * float(outcome.get("fall", 0.0))
        )
        comfort = clamp(
            4.0 * float(outcome.get("gain", 0.0))
            + 0.45 * float(outcome.get("rest_recovery", 0.0))
            + 0.10 * float(outcome.get("resource_collected", 0.0))
            + 0.08 * float(outcome.get("safe_intervention_success", 0.0))
        )
        self.pain_memory = clamp(0.985 * self.pain_memory + 0.015 * pain)
        self.danger_memory = clamp(0.985 * self.danger_memory + 0.015 * danger)
        self.vertical_risk_memory = clamp(0.985 * self.vertical_risk_memory + 0.015 * vertical)
        self.comfort_memory = clamp(0.985 * self.comfort_memory + 0.015 * comfort)
        self.action_possibility = clamp(0.45 + 0.25 * energy + 0.22 * stability - 0.28 * fatigue - 0.25 * danger - 0.15 * vertical)

        q_raw = (
            0.18 * pain
            + 0.16 * danger
            + 0.13 * vertical
            + 0.14 * self.pain_memory
            + 0.12 * self.danger_memory
            + 0.10 * self.vertical_risk_memory
            + 0.12 * (1.0 - viability)
            + 0.11 * fatigue
            + 0.12 * (1.0 - stability)
            - 0.10 * self.comfort_memory
            - 0.08 * self.action_possibility
        )
        self.q = clamp(0.90 * self.q + 0.10 * clamp(q_raw))
        self.q_history.append(self.q)
        self.damage_history.append(float(outcome.get("damage", 0.0)))
        return self.q

    def damage_correlation(self) -> float:
        xs = self.q_history[-800:]
        ys = self.damage_history[-800:]
        if len(xs) < 10:
            return 0.0
        mx = safe_mean(xs)
        my = safe_mean(ys)
        vx = sum((x - mx) ** 2 for x in xs)
        vy = sum((y - my) ** 2 for y in ys)
        if vx <= 1e-12 or vy <= 1e-12:
            return 0.0
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        return cov / math.sqrt(vx * vy)


class OnlineLinearPredictor:
    def __init__(self, n_features: int, lr: float = 0.018, decay: float = 0.0004) -> None:
        self.w = [0.0 for _ in range(n_features)]
        self.lr = lr
        self.decay = decay
        self.n = 0
        self.errs: List[float] = []

    def predict(self, x: List[float]) -> float:
        return sum(w * xi for w, xi in zip(self.w, x))

    def update(self, x: List[float], y: float) -> float:
        pred = self.predict(x)
        err = y - pred
        self.n += 1
        self.errs.append(abs(err))
        for i in range(len(self.w)):
            self.w[i] = (1.0 - self.decay) * self.w[i] + self.lr * err * x[i]
        return err


class BodyConsequenceModel:
    def __init__(self) -> None:
        self.damage_model = OnlineLinearPredictor(18, lr=0.018)
        self.progress_model = OnlineLinearPredictor(18, lr=0.018)
        self.friction_samples: List[float] = []
        self.slope_samples: List[float] = []
        self.height_samples: List[float] = []
        self.safe_interventions = 0
        self.samples = 0

    def features(self, action: str, pressures: Dict[str, float], viability: float,
                 energy: float, fatigue: float, stability: float, mu: float,
                 slope: float, z: int) -> List[float]:
        return [
            1.0,
            float(action == "MOVE_FWD"),
            float(action == "JUMP"),
            float(action == "CLIMB"),
            float(action == "DROP"),
            float(action == "BRAKE"),
            float(action == "SCAN"),
            float(action == "PROBE"),
            pressures.get("danger", 0.0),
            pressures.get("unknown", 0.0),
            pressures.get("gap", 0.0),
            pressures.get("wall", 0.0),
            pressures.get("climb", 0.0),
            pressures.get("drop", 0.0),
            1.0 - viability,
            fatigue,
            1.0 - stability,
            max(0.0, 0.20 - mu) + slope + 0.08 * z,
        ]

    def predict(self, action: str, pressures: Dict[str, float], viability: float,
                energy: float, fatigue: float, stability: float, mu: float,
                slope: float, z: int) -> Tuple[float, float]:
        x = self.features(action, pressures, viability, energy, fatigue, stability, mu, slope, z)
        dmg = clamp(self.damage_model.predict(x), 0.0, 1.0)
        prog = clamp(self.progress_model.predict(x), 0.0, 2.0)
        return dmg, prog

    def update(self, action: str, pressures: Dict[str, float], viability: float,
               energy: float, fatigue: float, stability: float, outcome: Dict[str, Any]) -> None:
        x = self.features(
            action,
            pressures,
            viability,
            energy,
            fatigue,
            stability,
            float(outcome.get("mu", SURFACE_MU["normal"])),
            float(outcome.get("slope", 0.0)),
            int(outcome.get("new_z", 0)),
        )
        self.damage_model.update(x, float(outcome.get("damage", 0.0)))
        self.progress_model.update(x, float(outcome.get("progress_delta", 0.0)))
        self.samples += 1
        if "friction_observed" in outcome:
            self.friction_samples.append(float(outcome["friction_observed"]))
        if "slope_observed" in outcome:
            self.slope_samples.append(float(outcome["slope_observed"]))
        if "height_observed" in outcome:
            self.height_samples.append(float(outcome["height_observed"]))
        self.safe_interventions += int(outcome.get("safe_intervention_success", 0))

    def prediction_mae(self) -> float:
        return safe_mean(self.damage_model.errs[-800:] + self.progress_model.errs[-800:])

    def law_score(self) -> float:
        parts = []
        parts.append(clamp(self.samples / 350.0))
        if self.friction_samples:
            parts.append(clamp(safe_sd(self.friction_samples[-250:]) / 0.18))
        if self.slope_samples:
            parts.append(clamp(len(self.slope_samples[-250:]) / 90.0))
        if self.height_samples:
            parts.append(clamp(safe_sd(self.height_samples[-250:]) / 1.0))
        parts.append(clamp(self.safe_interventions / 35.0))
        return safe_mean(parts)


class Agent3D:
    def __init__(self, cfg: Config, course: ProgressiveCourse3D, variant: str, seed: int) -> None:
        self.cfg = cfg
        self.course = course
        self.variant = variant
        self.rng = random.Random(seed)

        self.viability = 1.0
        self.energy = 1.0
        self.fatigue = 0.0
        self.stability = 1.0
        self.damage_total = 0.0

        self.q = ProtoValenceQ()
        self.body = BodyConsequenceModel()

        self.steps = 0
        self.terminal = False
        self.goal_reached = False

        self.active_steps = 0
        self.rest_steps = 0
        self.scan_steps = 0
        self.probe_steps = 0
        self.regulatory_rest_steps = 0
        self.passive_shutdown_steps = 0

        self.recent_damage: deque = deque(maxlen=70)
        self.recent_actions: deque = deque(maxlen=70)
        self.failure_zone = "none"

    def q_for_decision(self) -> float:
        if self.variant == "no_q":
            return 0.0
        if self.variant == "damage_only_q":
            return clamp(self.q.pain_memory)
        return self.q.q

    def body_prediction(self, action: str, pressures: Dict[str, float], p: Platform) -> Tuple[float, float]:
        z = self.course.pos[2]
        if self.variant == "no_body_model":
            damage_prior = clamp(
                0.018 * float(action in MOVE_ACTIONS)
                + 0.045 * float(action == "JUMP")
                + 0.040 * float(action == "CLIMB")
                + 0.025 * float(action == "DROP")
                + 0.030 * float(action == "PROBE")
                + 0.32 * pressures.get("danger", 0.0)
                + 0.25 * pressures.get("gap", 0.0)
                + 0.17 * pressures.get("wall", 0.0)
                + 0.18 * pressures.get("drop", 0.0)
                + 0.12 * max(0.0, 0.20 - p.mu)
                + 0.10 * p.slope
                + 0.02 * z
            )
            progress_prior = clamp(
                1.0 * float(action == "MOVE_FWD")
                + 1.35 * float(action == "JUMP")
                + 0.80 * float(action == "CLIMB")
                + 0.65 * float(action == "DROP")
                - 0.55 * pressures.get("wall", 0.0)
                - 0.50 * pressures.get("gap", 0.0),
                0.0,
                2.0,
            )
            return damage_prior, progress_prior
        return self.body.predict(action, pressures, self.viability, self.energy, self.fatigue, self.stability, p.mu, p.slope, z)

    def forward_obstacle_features(self) -> Dict[str, float]:
        """Direct 3D features immediately ahead of the agent.

        This is intentionally separate from diffuse local pressures. Progressive
        courses often require a specific intervention at x+1 or x+2.
        """
        x, y, z = self.course.pos
        f = {"gap_ahead": 0.0, "wall_ahead": 0.0, "danger_ahead": 0.0,
             "climb_needed": 0.0, "drop_needed": 0.0, "jump_available": 0.0,
             "forward_safe": 0.0}
        tx, ty = x + 1, y
        if not self.course.in_xy(tx, ty):
            f["wall_ahead"] = 1.0
            return f

        p1 = self.course.platform_at(tx, ty)
        z1 = self.course.floor_z.get((tx, ty))
        if p1 is None or z1 is None:
            f["gap_ahead"] = 1.0
        else:
            hd = z1 - z
            eff = p1.effective()
            if eff == "WALL":
                f["wall_ahead"] = 1.0
            if eff == "DANGER":
                f["danger_ahead"] = 1.0
            if hd > 1:
                f["climb_needed"] = 1.0
            if hd < -1:
                f["drop_needed"] = 1.0
            if eff not in {"WALL", "DANGER"} and -1 <= hd <= 1:
                f["forward_safe"] = 1.0

        # Jump landing at x+2.
        lx, ly = x + 2, y
        if self.course.in_xy(lx, ly):
            p2 = self.course.platform_at(lx, ly)
            z2 = self.course.floor_z.get((lx, ly))
            if p2 is not None and z2 is not None:
                h2 = z2 - z
                if p2.effective() not in {"WALL", "DANGER"} and abs(h2) <= 2:
                    if f["gap_ahead"] or f["wall_ahead"] or f["climb_needed"]:
                        f["jump_available"] = 1.0
        return f

    def choose_action(self) -> Tuple[str, Dict[str, Any]]:
        pressures = self.course.local_pressures()
        direct = self.forward_obstacle_features()
        p = self.course.current_platform()
        q_dec = self.q_for_decision()
        scores: Dict[str, float] = {}

        for action in ACTIONS:
            pred_damage, pred_progress = self.body_prediction(action, pressures, p)

            progress_drive = 0.0
            if action == "MOVE_FWD":
                progress_drive += 0.46 + 0.18 * pressures.get("forward_open", 0.0)
                progress_drive -= 0.70 * direct["gap_ahead"]
                progress_drive -= 0.60 * direct["wall_ahead"]
                progress_drive -= 0.55 * direct["climb_needed"]
                progress_drive -= 0.45 * direct["drop_needed"]
                progress_drive -= 0.55 * direct["danger_ahead"]
                progress_drive += 0.20 * direct["forward_safe"]
            elif action == "JUMP":
                progress_drive += 0.10 + 0.34 * (pressures.get("gap", 0.0) + 0.6 * pressures.get("wall", 0.0))
                progress_drive += 1.05 * direct["jump_available"]
                progress_drive += 0.55 * direct["gap_ahead"]
                progress_drive += 0.35 * direct["wall_ahead"]
            elif action == "CLIMB":
                progress_drive += 0.10 + 0.34 * pressures.get("climb", 0.0)
                progress_drive += 1.15 * direct["climb_needed"]
            elif action == "DROP":
                progress_drive += 0.08 + 0.28 * pressures.get("drop", 0.0)
                progress_drive += 0.95 * direct["drop_needed"]
            elif action in {"MOVE_LEFT", "MOVE_RIGHT"}:
                progress_drive += 0.10 + 0.13 * (pressures.get("danger", 0.0) + pressures.get("gap", 0.0) + pressures.get("wall", 0.0))
                progress_drive += 0.35 * max(direct["gap_ahead"], direct["wall_ahead"], direct["danger_ahead"])
            elif action == "MOVE_BACK":
                progress_drive -= 0.22

            information_drive = 0.0
            if action == "SCAN":
                information_drive += 0.46 * pressures.get("unknown", 0.0)
                if pressures.get("unknown", 0.0) < 0.04:
                    information_drive -= 0.25
            if action == "PROBE":
                law_deficit = 1.0 - self.body.law_score()
                safe_body = float(self.viability > 0.55 and self.energy > 0.30 and self.stability > 0.35)
                vertical_need = pressures.get("climb", 0.0) + pressures.get("drop", 0.0) + pressures.get("gap", 0.0)
                information_drive += safe_body * (0.22 * law_deficit + 0.18 * vertical_need)
                if not safe_body:
                    information_drive -= 0.25

            recovery_drive = 0.0
            if action == "REST":
                recovery_drive += 0.55 * float(self.viability < 0.55)
                recovery_drive += 0.42 * float(self.energy < 0.42)
                recovery_drive += 0.42 * float(self.fatigue > 0.52)
                recovery_drive += 0.42 * float(self.stability < 0.45)
                if p.effective() == "REST":
                    recovery_drive += 0.25

            intervention_drive = 0.0
            if action == "BRAKE":
                intervention_drive += 0.22 * float(p.surface in {"slippery", "slope"} or p.slope > 0.08)
                intervention_drive += 0.15 * float(self.stability < 0.45)
            if action == "CLIMB":
                intervention_drive += 0.20 * pressures.get("climb", 0.0) + 0.70 * direct["climb_needed"]
            if action == "DROP":
                intervention_drive += 0.18 * pressures.get("drop", 0.0) + 0.55 * direct["drop_needed"]
            if action == "JUMP":
                intervention_drive += 0.65 * direct["jump_available"]

            risk_penalty = (
                1.28 * pred_damage
                + 0.68 * q_dec
                + 0.26 * self.fatigue
                + 0.24 * (1.0 - self.stability)
                + 0.12 * pressures.get("danger", 0.0)
                + 0.08 * pressures.get("vertical", 0.0)
            )

            score = (
                0.35 * pred_progress
                + progress_drive
                + information_drive
                + recovery_drive
                + intervention_drive
                - risk_penalty
            )

            if self.energy < 0.20 or self.stability < 0.18:
                if action in ACTIVE_ACTIONS:
                    score -= 0.65
                if action == "REST":
                    score += 0.55
            if self.viability < 0.22 and action in {"JUMP", "CLIMB", "DROP", "PROBE"}:
                score -= 0.60

            scores[action] = score

        recent_passive = sum(1 for x in list(self.recent_actions)[-20:] if x in {"REST", "SCAN"})
        recovered = self.viability > 0.78 and self.energy > 0.55 and self.fatigue < 0.40 and self.stability > 0.50
        if recovered and recent_passive >= 12:
            scores["REST"] -= 0.40
            scores["SCAN"] -= 0.30
            scores["MOVE_FWD"] += 0.20

        best = max(scores, key=scores.get)
        return best, {"pressures": pressures, "scores": scores, "q_decision": q_dec}

    def step(self) -> Dict[str, Any]:
        action, decision = self.choose_action()
        pressures = decision["pressures"]

        if action in ACTIVE_ACTIONS:
            self.active_steps += 1
        if action == "REST":
            self.rest_steps += 1
        if action == "SCAN":
            self.scan_steps += 1
        if action == "PROBE":
            self.probe_steps += 1

        outcome = self.course.step(action, self.stability, self.fatigue)

        damage = float(outcome.get("damage", 0.0))
        gain = float(outcome.get("gain", 0.0))
        energy_cost = float(outcome.get("energy_cost", 0.0))
        self.damage_total += damage

        self.energy = clamp(
            self.energy
            - energy_cost
            + 0.050 * float(outcome.get("resource_collected", 0))
            + 0.040 * float(outcome.get("rest_recovery", 0))
            + 0.003 * float(action in {"REST", "SCAN"}),
        )
        self.fatigue = clamp(self.fatigue + float(outcome.get("fatigue_delta", 0.0)))
        self.stability = clamp(
            self.stability
            + float(outcome.get("stability_delta", 0.0))
            + 0.025 * float(outcome.get("rest_recovery", 0))
            + 0.003 * float(action in {"REST", "SCAN"} and damage <= 1e-12),
        )

        basal = 0.0010
        self.viability = clamp(self.viability - basal - 0.92 * damage - 0.20 * energy_cost + gain)

        if self.variant == "no_q":
            q_val = 0.0
            self.q.q = 0.0
            self.q.q_history.append(0.0)
            self.q.damage_history.append(damage)
        elif self.variant == "damage_only_q":
            pain = clamp(float(outcome.get("pain", 0.0)) + 2.4 * damage)
            self.q.pain_memory = clamp(0.985 * self.q.pain_memory + 0.015 * pain)
            q_val = clamp(self.q.pain_memory)
            self.q.q = q_val
            self.q.q_history.append(q_val)
            self.q.damage_history.append(damage)
        else:
            q_val = self.q.update(self.viability, self.energy, self.fatigue, self.stability, pressures, outcome)

        if self.variant in {"full_core", "no_q", "damage_only_q"}:
            self.body.update(action, pressures, self.viability, self.energy, self.fatigue, self.stability, outcome)
        elif self.variant in {"no_body_model", "frozen_predictor"}:
            pass

        recent_damage = sum(self.recent_damage)
        bodily_need = (
            self.viability < 0.80
            or self.energy < 0.50
            or self.fatigue > 0.40
            or self.stability < 0.52
            or recent_damage > 0.025
        )
        local_need = (
            pressures.get("danger", 0.0) >= 0.08
            or pressures.get("unknown", 0.0) >= 0.08
            or pressures.get("resource", 0.0) >= 0.08
            or pressures.get("rest", 0.0) >= 0.08
            or pressures.get("gap", 0.0) >= 0.08
            or pressures.get("wall", 0.0) >= 0.08
            or pressures.get("climb", 0.0) >= 0.08
            or pressures.get("drop", 0.0) >= 0.08
        )
        if action in {"REST", "SCAN"}:
            if bodily_need or local_need or action == "SCAN":
                self.regulatory_rest_steps += 1
            else:
                self.passive_shutdown_steps += 1

        self.recent_damage.append(damage)
        self.recent_actions.append(action)
        self.steps += 1

        if self.viability <= 0.05:
            self.terminal = True
            self.failure_zone = self.course.zone_name()
        if self.course.pos[0] >= self.course.length - 1:
            self.goal_reached = True

        return {
            "t": self.steps,
            "level": self.course.level,
            "variant": self.variant,
            "action": action,
            "event": outcome.get("event", "none"),
            "zone": self.course.zone_name(),
            "x": self.course.pos[0],
            "y": self.course.pos[1],
            "z": self.course.pos[2],
            "course_progress": self.course.progress(),
            "zone_reached": self.course.zone_reached(),
            "viability": self.viability,
            "energy": self.energy,
            "fatigue": self.fatigue,
            "stability": self.stability,
            "q": q_val,
            "damage": damage,
            "gain": gain,
            "height_diff": outcome.get("height_diff", 0),
            "fall_distance": outcome.get("fall_distance", 0),
            "law_score": self.body.law_score(),
            "prediction_mae": self.body.prediction_mae(),
            "unknown_revealed": self.course.unknown_revealed,
            "resources_collected": self.course.resources_collected,
        }

    def run(self, episode: int, outdir: Path, trace_stride: int) -> Dict[str, Any]:
        trace_path = outdir / f"trace_episode_{episode:03d}.csv"
        trace_fields = [
            "episode", "t", "level", "variant", "action", "event", "zone",
            "x", "y", "z", "course_progress", "zone_reached", "viability",
            "energy", "fatigue", "stability", "q", "damage", "gain",
            "height_diff", "fall_distance", "law_score", "prediction_mae",
            "unknown_revealed", "resources_collected",
        ]

        for _ in range(self.cfg.steps):
            row = self.step()
            if self.steps % trace_stride == 0 or self.terminal or self.goal_reached:
                append_csv(trace_path, trace_fields, {"episode": episode, **row})
            if self.terminal or self.goal_reached:
                break
        return self.metrics(episode)

    def metrics(self, episode: int) -> Dict[str, Any]:
        total = max(1, self.steps)
        active_fraction = self.active_steps / total
        regulatory_rest_fraction = self.regulatory_rest_steps / total
        passive_shutdown_fraction = self.passive_shutdown_steps / total
        survival = float(not self.terminal)
        bounded_internal = float(self.energy > 0.05 and self.stability > 0.05 and self.fatigue < 0.97)
        progress = self.course.progress()

        living_engagement = clamp(
            0.50 * progress
            + 0.14 * active_fraction
            + 0.10 * (self.course.unknown_revealed / max(1.0, self.course.length * 0.55))
            + 0.08 * (self.course.resources_collected / max(1.0, self.course.length * 0.35))
            + 0.10 * clamp(self.body.law_score())
            + 0.08 * clamp((self.course.successful_climbs + self.course.successful_jumps + self.course.controlled_drops) / 15.0)
        )
        autonomous_established = int(
            survival
            and bounded_internal
            and self.viability > 0.10
            and progress > 0.35
            and active_fraction > 0.10
            and living_engagement > 0.14
            and passive_shutdown_fraction < 0.20
        )
        autonomy_proper = clamp(
            0.22 * survival
            + 0.17 * bounded_internal
            + 0.24 * living_engagement
            + 0.13 * active_fraction
            + 0.12 * (1.0 - passive_shutdown_fraction)
            + 0.12 * self.viability
        )

        failure_zone = self.failure_zone if self.failure_zone != "none" else self.course.zone_name(self.course.max_x)

        return {
            "episode": episode,
            "level": self.course.level,
            "variant": self.variant,
            "steps_completed": self.steps,
            "terminal": int(self.terminal),
            "goal_reached": int(self.goal_reached),
            "survival": survival,
            "autonomous_life_established": autonomous_established,
            "autonomy_proper": autonomy_proper,
            "bounded_internal": bounded_internal,
            "course_progress": progress,
            "max_x": self.course.max_x,
            "max_z": self.course.max_z,
            "zone_reached": self.course.zone_reached(),
            "failure_zone": failure_zone,
            "final_viability": self.viability,
            "final_energy": self.energy,
            "final_fatigue": self.fatigue,
            "final_stability": self.stability,
            "damage_total": self.damage_total,
            "active_fraction": active_fraction,
            "regulatory_rest_fraction": regulatory_rest_fraction,
            "passive_shutdown_fraction": passive_shutdown_fraction,
            "unknown_revealed": self.course.unknown_revealed,
            "resources_collected": self.course.resources_collected,
            "danger_entries": self.course.danger_entries,
            "falls": self.course.falls,
            "fall_distance_total": self.course.fall_distance_total,
            "slips": self.course.slips,
            "collisions": self.course.collisions,
            "successful_jumps": self.course.successful_jumps,
            "failed_jumps": self.course.failed_jumps,
            "successful_climbs": self.course.successful_climbs,
            "failed_climbs": self.course.failed_climbs,
            "controlled_drops": self.course.controlled_drops,
            "successful_brakes": self.course.successful_brakes,
            "safe_intervention_success": self.course.safe_intervention_success,
            "probes": self.course.probes,
            "q_final": self.q.q,
            "q_damage_corr": self.q.damage_correlation(),
            "prediction_mae": self.body.prediction_mae(),
            "law_score": self.body.law_score(),
            "body_model_samples": self.body.samples,
        }


EPISODE_FIELDS = [
    "episode", "level", "variant", "steps_completed", "terminal", "goal_reached",
    "survival", "autonomous_life_established", "autonomy_proper", "bounded_internal",
    "course_progress", "max_x", "max_z", "zone_reached", "failure_zone",
    "final_viability", "final_energy", "final_fatigue", "final_stability",
    "damage_total", "active_fraction", "regulatory_rest_fraction", "passive_shutdown_fraction",
    "unknown_revealed", "resources_collected", "danger_entries", "falls",
    "fall_distance_total", "slips", "collisions", "successful_jumps", "failed_jumps",
    "successful_climbs", "failed_climbs", "controlled_drops", "successful_brakes",
    "safe_intervention_success", "probes", "q_final", "q_damage_corr",
    "prediction_mae", "law_score", "body_model_samples",
]

SUMMARY_METRICS = [
    "survival", "autonomous_life_established", "autonomy_proper", "goal_reached",
    "course_progress", "zone_reached", "max_z", "final_viability", "damage_total",
    "active_fraction", "regulatory_rest_fraction", "passive_shutdown_fraction",
    "unknown_revealed", "resources_collected", "danger_entries", "falls",
    "fall_distance_total", "slips", "collisions", "successful_jumps",
    "successful_climbs", "controlled_drops", "safe_intervention_success",
    "q_damage_corr", "prediction_mae", "law_score", "body_model_samples",
]


def completed_episodes(path: Path) -> set:
    rows = read_csv_rows(path)
    out = set()
    for r in rows:
        try:
            out.add(int(float(r["episode"])))
        except Exception:
            pass
    return out


def run_condition(cfg: Config, level: str, variant: str, logger: ProgressLogger) -> None:
    cond_out = cfg.outdir / level / variant
    mkdir(cond_out)
    ep_path = cond_out / "episode_metrics.csv"
    done = completed_episodes(ep_path) if cfg.resume else set()

    logger.log(f"RUN level={level} variant={variant}")
    t0 = time.time()
    for ep in range(cfg.episodes):
        if ep in done:
            logger.log(f"skip level={level} variant={variant} episode={ep:03d}")
            continue

        seed = cfg.seed + 1000003 * ep + 9199 * LEVELS.index(level) + 30047 * VARIANTS.index(variant)
        course = ProgressiveCourse3D(cfg, level=level, episode=ep, seed=seed)
        agent = Agent3D(cfg, course=course, variant=variant, seed=seed + 173)
        metrics = agent.run(ep, cond_out, cfg.trace_stride)
        append_csv(ep_path, EPISODE_FIELDS, metrics)

        n_done = len(completed_episodes(ep_path))
        elapsed = time.time() - t0
        rate = elapsed / max(1, n_done)
        remaining = (cfg.episodes - n_done) * rate
        logger.log(
            f"done level={level} variant={variant} ep={ep:03d} "
            f"progress={metrics['course_progress']:.3f} zmax={metrics['max_z']} "
            f"est={metrics['autonomous_life_established']} dmg={metrics['damage_total']:.3f} "
            f"falls={metrics['falls']} law={metrics['law_score']:.3f} ETA~{remaining/60:.1f}min"
        )


def collect_all(cfg: Config) -> List[Dict[str, Any]]:
    rows = []
    for level in cfg.levels:
        for variant in cfg.variants:
            path = cfg.outdir / level / variant / "episode_metrics.csv"
            for r in read_csv_rows(path):
                rr = dict(r)
                rr["level"] = level
                rr["variant"] = variant
                rows.append(rr)
    return rows


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(str(r["level"]), str(r["variant"]))].append(r)

    out = []
    for (level, variant), gr in sorted(groups.items()):
        row = {"level": level, "variant": variant, "n": len(gr)}
        for m in SUMMARY_METRICS:
            vals = [to_float(r.get(m)) for r in gr]
            row[f"{m}_mean"] = safe_mean(vals)
            row[f"{m}_sd"] = safe_sd(vals)
        out.append(row)
    return out


def make_variant_deltas(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by = {(r["level"], r["variant"]): r for r in summary_rows}
    out = []
    for level in LEVELS:
        full = by.get((level, "full_core"))
        if not full:
            continue
        for variant in VARIANTS:
            if variant == "full_core":
                continue
            r = by.get((level, variant))
            if not r:
                continue
            row = {"level": level, "variant": variant}
            for m in SUMMARY_METRICS:
                row[f"delta_{m}_variant_minus_full"] = to_float(r.get(f"{m}_mean")) - to_float(full.get(f"{m}_mean"))
            out.append(row)
    return out


def make_paired_deltas(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by = {}
    for r in rows:
        by[(str(r["level"]), str(r["variant"]), int(float(r["episode"])))] = r

    out = []
    for level in LEVELS:
        eps = sorted({k[2] for k in by if k[0] == level})
        for ep in eps:
            full = by.get((level, "full_core", ep))
            if not full:
                continue
            for variant in VARIANTS:
                if variant == "full_core":
                    continue
                r = by.get((level, variant, ep))
                if not r:
                    continue
                row = {"level": level, "episode": ep, "variant": variant}
                for m in SUMMARY_METRICS:
                    row[f"delta_{m}_variant_minus_full"] = to_float(r.get(m)) - to_float(full.get(m))
                out.append(row)
    return out


def failure_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
    totals: Dict[Tuple[str, str], int] = defaultdict(int)
    for r in rows:
        key = (str(r["level"]), str(r["variant"]))
        totals[key] += 1
        zone = str(r.get("failure_zone", "none"))
        counts[(key[0], key[1], zone)] += 1

    out = []
    for (level, variant, zone), count in sorted(counts.items()):
        total = totals[(level, variant)]
        out.append({
            "level": level,
            "variant": variant,
            "failure_or_terminal_zone": zone,
            "count": count,
            "fraction": count / max(1, total),
        })
    return out


def metric(summary_rows: List[Dict[str, Any]], level: str, variant: str, name: str) -> float:
    for r in summary_rows:
        if r["level"] == level and r["variant"] == variant:
            return to_float(r.get(f"{name}_mean"))
    return 0.0


def write_outputs(cfg: Config, logger: ProgressLogger) -> None:
    logger.log("COLLECT outputs")
    rows = collect_all(cfg)
    if not rows:
        raise RuntimeError("No rows collected")

    write_csv(cfg.outdir / "all_episode_metrics.csv", rows, list(rows[0].keys()))
    summary_rows = summarize(rows)
    write_csv(cfg.outdir / "summary_by_level_variant.csv", summary_rows, list(summary_rows[0].keys()))

    full_rows = [r for r in summary_rows if r["variant"] == "full_core"]
    write_csv(cfg.outdir / "full_core_level_sweep.csv", full_rows, list(summary_rows[0].keys()))

    variant_deltas = make_variant_deltas(summary_rows)
    write_csv(cfg.outdir / "variant_deltas_vs_full_core.csv", variant_deltas, list(variant_deltas[0].keys()) if variant_deltas else ["level", "variant"])

    paired_deltas = make_paired_deltas(rows)
    write_csv(cfg.outdir / "paired_episode_deltas_vs_full_core.csv", paired_deltas, list(paired_deltas[0].keys()) if paired_deltas else ["level", "episode", "variant"])

    fail_rows = failure_summary(rows)
    write_csv(cfg.outdir / "failure_zone_summary.csv", fail_rows, list(fail_rows[0].keys()) if fail_rows else ["level", "variant"])

    if cfg.make_figures:
        make_figures(cfg, summary_rows, logger)

    write_report(cfg, summary_rows, variant_deltas, fail_rows, logger)


def make_figures(cfg: Config, summary_rows: List[Dict[str, Any]], logger: ProgressLogger) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.log(f"figures skipped: {type(e).__name__}: {e}")
        return

    figdir = cfg.outdir / "figures"
    mkdir(figdir)

    for m in ["course_progress", "max_z", "autonomous_life_established", "damage_total", "falls", "law_score", "q_damage_corr"]:
        vals = [metric(summary_rows, level, "full_core", m) for level in cfg.levels]
        fig = plt.figure(figsize=(8.5, 4.8))
        ax = fig.add_subplot(111)
        ax.plot(cfg.levels, vals, marker="o")
        ax.set_xlabel("3D course level")
        ax.set_ylabel(m)
        ax.set_title(f"Full core: {m}")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(figdir / f"full_core_3d_level_sweep_{m}.png", dpi=200)
        plt.close(fig)

    for m in ["course_progress", "max_z", "damage_total", "falls", "law_score", "q_damage_corr"]:
        mat = [[metric(summary_rows, level, variant, m) for variant in cfg.variants] for level in cfg.levels]
        fig = plt.figure(figsize=(10, 5.4))
        ax = fig.add_subplot(111)
        im = ax.imshow(mat, aspect="auto")
        ax.set_xticks(range(len(cfg.variants)))
        ax.set_xticklabels(cfg.variants, rotation=35, ha="right")
        ax.set_yticks(range(len(cfg.levels)))
        ax.set_yticklabels(cfg.levels)
        ax.set_title(m)
        fig.colorbar(im, ax=ax, label=m)
        fig.tight_layout()
        fig.savefig(figdir / f"heatmap_3d_{m}.png", dpi=200)
        plt.close(fig)

    logger.log("wrote figures")


def write_report(cfg: Config, summary_rows: List[Dict[str, Any]], variant_deltas: List[Dict[str, Any]],
                 failure_rows: List[Dict[str, Any]], logger: ProgressLogger) -> None:
    lines = []
    lines.append("Progressive 3D Course Validation")
    lines.append("=" * 36)
    lines.append("")
    lines.append("Purpose")
    lines.append("-------")
    lines.append("This validation uses a true 3D course with coordinates (x, y, z).")
    lines.append("The agent must progress forward while handling height changes, falls, climbs, jumps, drops, slippery surfaces, walls, gaps, and hidden risks.")
    lines.append("No semantic hint module is used.")
    lines.append("")
    lines.append("Design")
    lines.append("------")
    lines.append("Levels: " + ", ".join(cfg.levels))
    lines.append("Variants: " + ", ".join(cfg.variants))
    lines.append(f"Episodes per cell: {cfg.episodes}")
    lines.append(f"Steps per episode: {cfg.steps}")
    lines.append(f"Course shape: length={cfg.length}, width={cfg.width}, height={cfg.height}")
    lines.append("")
    lines.append("Full-core 3D level sweep")
    lines.append("------------------------")
    for level in cfg.levels:
        lines.append(
            f"{level}: progress={metric(summary_rows, level, 'full_core', 'course_progress'):.3f}, "
            f"max_z={metric(summary_rows, level, 'full_core', 'max_z'):.3f}, "
            f"established={metric(summary_rows, level, 'full_core', 'autonomous_life_established'):.3f}, "
            f"goal={metric(summary_rows, level, 'full_core', 'goal_reached'):.3f}, "
            f"damage={metric(summary_rows, level, 'full_core', 'damage_total'):.3f}, "
            f"falls={metric(summary_rows, level, 'full_core', 'falls'):.3f}, "
            f"law={metric(summary_rows, level, 'full_core', 'law_score'):.3f}, "
            f"qcorr={metric(summary_rows, level, 'full_core', 'q_damage_corr'):.3f}"
        )

    lines.append("")
    lines.append("Variant deltas vs full_core")
    lines.append("---------------------------")
    for r in variant_deltas:
        lines.append(
            f"{r['level']} | {r['variant']}: "
            f"d_progress={to_float(r.get('delta_course_progress_variant_minus_full')):.3f}, "
            f"d_max_z={to_float(r.get('delta_max_z_variant_minus_full')):.3f}, "
            f"d_est={to_float(r.get('delta_autonomous_life_established_variant_minus_full')):.3f}, "
            f"d_damage={to_float(r.get('delta_damage_total_variant_minus_full')):.3f}, "
            f"d_falls={to_float(r.get('delta_falls_variant_minus_full')):.3f}, "
            f"d_law={to_float(r.get('delta_law_score_variant_minus_full')):.3f}, "
            f"d_qcorr={to_float(r.get('delta_q_damage_corr_variant_minus_full')):.3f}"
        )

    lines.append("")
    lines.append("Interpretation guide")
    lines.append("--------------------")
    lines.append("1. course_progress is the main task outcome.")
    lines.append("2. max_z, falls, successful_climbs, successful_jumps, and controlled_drops verify that the task is genuinely 3D.")
    lines.append("3. no_q and damage_only_q test whether proto-valence affects body-state-dependent risk handling.")
    lines.append("4. no_body_model and frozen_predictor test whether body-consequence learning affects safe vertical intervention and progress.")
    lines.append("5. failure zones indicate where the agent stalls or terminates.")
    lines.append("")
    lines.append("Generated files")
    lines.append("---------------")
    lines.append("all_episode_metrics.csv")
    lines.append("summary_by_level_variant.csv")
    lines.append("full_core_level_sweep.csv")
    lines.append("variant_deltas_vs_full_core.csv")
    lines.append("paired_episode_deltas_vs_full_core.csv")
    lines.append("failure_zone_summary.csv")
    lines.append("figures/*.png if matplotlib is available")

    (cfg.outdir / "progressive_3d_course_validation_report.txt").write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "script": "progressive_course_3d_validation_v2.py",
        "semantic_hint": "none",
        "coordinates": ["x", "y", "z"],
        "levels": cfg.levels,
        "variants": cfg.variants,
        "episodes": cfg.episodes,
        "steps": cfg.steps,
        "length": cfg.length,
        "width": cfg.width,
        "height": cfg.height,
    }
    (cfg.outdir / "progressive_3d_course_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.log("wrote progressive_3d_course_validation_report.txt")


def run_all(cfg: Config, logger: ProgressLogger) -> None:
    logger.log("STEP 1/4 configuration")
    logger.log(f"levels={cfg.levels}")
    logger.log(f"variants={cfg.variants}")
    logger.log(f"episodes={cfg.episodes}; steps={cfg.steps}; length={cfg.length}; width={cfg.width}; height={cfg.height}")

    logger.log("STEP 2/4 running grid")
    for level in cfg.levels:
        for variant in cfg.variants:
            run_condition(cfg, level, variant, logger)

    logger.log("STEP 3/4 writing outputs")
    write_outputs(cfg, logger)
    logger.log("STEP 4/4 done")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="True 3D progressive course validation.")
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--mode", choices=["smoke", "quick", "full"], default="full")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--length", type=int, default=120)
    p.add_argument("--width", type=int, default=5)
    p.add_argument("--height", type=int, default=6)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--levels", type=str, default=",".join(LEVELS))
    p.add_argument("--variants", type=str, default=",".join(VARIANTS))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-figures", action="store_true")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    if args.mode == "smoke":
        episodes = 2
        steps = 300
        trace_stride = 10
    elif args.mode == "quick":
        episodes = 10
        steps = 800
        trace_stride = 20
    else:
        episodes = 30
        steps = 1200
        trace_stride = 25

    if args.episodes is not None:
        episodes = args.episodes
    if args.steps is not None:
        steps = args.steps

    levels = [x.strip() for x in args.levels.split(",") if x.strip()]
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    for x in levels:
        if x not in LEVELS:
            raise ValueError(f"Unknown level: {x}")
    for x in variants:
        if x not in VARIANTS:
            raise ValueError(f"Unknown variant: {x}")

    return Config(
        outdir=Path(args.outdir).expanduser(),
        seed=args.seed,
        episodes=episodes,
        steps=steps,
        length=args.length,
        width=args.width,
        height=args.height,
        resume=args.resume,
        trace_stride=trace_stride,
        mode=args.mode,
        levels=levels,
        variants=variants,
        make_figures=not args.no_figures,
    )


def main() -> int:
    args = parse_args()
    cfg = build_config(args)
    mkdir(cfg.outdir)
    logger = ProgressLogger(cfg.outdir)
    try:
        run_all(cfg, logger)
        return 0
    except KeyboardInterrupt:
        logger.log("interrupted")
        return 130
    except Exception as e:
        logger.log(f"ERROR: {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
