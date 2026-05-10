"""
AIDRA - Adaptive Intelligent Disaster Response Agent
Environment Model: Grid map, victims, hazards, roads, resources
"""

import random
import math
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


class CellType(Enum):
    SAFE = "safe"
    ROAD = "road"
    HIGH_RISK = "high_risk"
    BLOCKED = "blocked"
    MEDICAL_CENTER = "medical_center"
    BASE = "base"


class VictimSeverity(Enum):
    CRITICAL = 3
    MODERATE = 2
    MINOR = 1


class VictimStatus(Enum):
    WAITING = "waiting"
    IN_RESCUE = "in_rescue"
    RESCUED = "rescued"
    DECEASED = "deceased"


@dataclass
class Victim:
    id: int
    position: Tuple[int, int]
    severity: VictimSeverity
    status: VictimStatus = VictimStatus.WAITING
    survival_prob: float = 1.0
    time_waiting: int = 0
    assigned_ambulance: Optional[int] = None

    def decay_survival(self, time_step: int):
        """Survival probability decays over time based on severity."""
        decay_rates = {
            VictimSeverity.CRITICAL: 0.04,
            VictimSeverity.MODERATE: 0.015,
            VictimSeverity.MINOR: 0.005,
        }
        decay = decay_rates[self.severity] * time_step
        self.survival_prob = max(0.0, self.survival_prob - decay)
        self.time_waiting += time_step

    def __repr__(self):
        return f"V{self.id}({self.severity.name[:3]}, pos={self.position}, surv={self.survival_prob:.2f})"


@dataclass
class Ambulance:
    id: int
    position: Tuple[int, int]
    capacity: int = 2
    victims_onboard: List[int] = field(default_factory=list)
    is_available: bool = True
    current_route: List[Tuple[int, int]] = field(default_factory=list)
    route_index: int = 0
    mission: str = "idle"
    trips_completed: int = 0
    total_distance: int = 0
    risk_exposure: int = 0


@dataclass
class RescueTeam:
    id: int
    position: Tuple[int, int]
    is_available: bool = True
    current_location: Optional[Tuple[int, int]] = None
    mission: str = "idle"


class DisasterEnvironment:
    """
    10x10 grid environment representing an urban disaster zone.
    """

    GRID_SIZE = 10

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.grid: List[List[CellType]] = []
        self.victims: List[Victim] = []
        self.ambulances: List[Ambulance] = []
        self.rescue_teams: List[RescueTeam] = []
        self.medical_centers: List[Tuple[int, int]] = []
        self.base: Tuple[int, int] = (0, 0)
        self.time_step: int = 0
        self.event_log: List[Dict] = []
        self.medical_kits: int = 10
        self.dynamic_events: List[Dict] = []
        self._setup_environment()

    def _setup_environment(self):
        """Initialize the grid with zones, resources, and victims."""
        # Build base grid — all roads initially
        self.grid = [[CellType.ROAD] * self.GRID_SIZE for _ in range(self.GRID_SIZE)]

        # Set base at (0,0)
        self.base = (0, 0)
        self.grid[0][0] = CellType.BASE

        # Medical centers
        self.medical_centers = [(9, 9), (9, 0)]
        for mc in self.medical_centers:
            self.grid[mc[0]][mc[1]] = CellType.MEDICAL_CENTER

        # High-risk zones (fire / structural collapse)
        high_risk_cells = [(3, 4), (3, 5), (4, 4), (4, 5), (5, 4), (4, 3), (2, 5)]
        for cell in high_risk_cells:
            self.grid[cell[0]][cell[1]] = CellType.HIGH_RISK

        # Initially blocked roads
        blocked_cells = [(1, 3), (2, 3)]
        for cell in blocked_cells:
            self.grid[cell[0]][cell[1]] = CellType.BLOCKED

        # Safe zones
        safe_cells = [(0, 5), (0, 6), (1, 7), (2, 8), (7, 1), (8, 2)]
        for cell in safe_cells:
            self.grid[cell[0]][cell[1]] = CellType.SAFE

        # Victims: 2 critical, 2 moderate, 1 minor
        victim_data = [
            (1, (5, 7), VictimSeverity.CRITICAL),
            (2, (3, 2), VictimSeverity.CRITICAL),
            (3, (7, 6), VictimSeverity.MODERATE),
            (4, (1, 8), VictimSeverity.MODERATE),
            (5, (6, 1), VictimSeverity.MINOR),
        ]
        for vid, pos, sev in victim_data:
            decay_base = {
                VictimSeverity.CRITICAL: 0.65,
                VictimSeverity.MODERATE: 0.85,
                VictimSeverity.MINOR: 0.95,
            }
            self.victims.append(Victim(vid, pos, sev, survival_prob=decay_base[sev]))

        # Ambulances at base
        self.ambulances = [
            Ambulance(id=1, position=self.base),
            Ambulance(id=2, position=self.base),
        ]

        # Rescue team at base
        self.rescue_teams = [RescueTeam(id=1, position=self.base)]

        # Schedule dynamic events
        self.dynamic_events = [
            {"time": 8, "type": "road_block", "cell": (6, 7), "description": "Aftershock blocks road at (6,7)"},
            {"time": 15, "type": "road_block", "cell": (8, 5), "description": "Fire spreads, blocks (8,5)"},
            {"time": 12, "type": "risk_increase", "cell": (3, 3), "description": "Hazard zone expands to (3,3)"},
            {"time": 20, "type": "new_victim", "pos": (7, 8), "severity": VictimSeverity.MODERATE,
             "description": "New victim discovered at (7,8)"},
        ]

    def apply_dynamic_events(self):
        """Apply scheduled events at current time step."""
        triggered = []
        for event in self.dynamic_events:
            if event["time"] == self.time_step:
                if event["type"] == "road_block":
                    cell = event["cell"]
                    if self.grid[cell[0]][cell[1]] == CellType.ROAD:
                        self.grid[cell[0]][cell[1]] = CellType.BLOCKED
                        triggered.append(event)
                elif event["type"] == "risk_increase":
                    cell = event["cell"]
                    self.grid[cell[0]][cell[1]] = CellType.HIGH_RISK
                    triggered.append(event)
                elif event["type"] == "new_victim":
                    new_id = max(v.id for v in self.victims) + 1
                    v = Victim(new_id, event["pos"], event["severity"],
                               survival_prob=0.75)
                    self.victims.append(v)
                    triggered.append(event)
        return triggered

    def get_neighbors(self, pos: Tuple[int, int], allow_risk: bool = True) -> List[Tuple[int, int]]:
        """Get traversable neighboring cells."""
        r, c = pos
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE:
                cell = self.grid[nr][nc]
                if cell != CellType.BLOCKED:
                    if not allow_risk and cell == CellType.HIGH_RISK:
                        continue
                    neighbors.append((nr, nc))
        return neighbors

    def is_high_risk(self, pos: Tuple[int, int]) -> bool:
        return self.grid[pos[0]][pos[1]] == CellType.HIGH_RISK

    def manhattan_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_waiting_victims(self) -> List[Victim]:
        return [v for v in self.victims if v.status == VictimStatus.WAITING]

    def get_active_victims(self) -> List[Victim]:
        return [v for v in self.victims if v.status != VictimStatus.RESCUED]

    def tick(self):
        """Advance one time step — decay survival, apply events."""
        self.time_step += 1
        for v in self.get_waiting_victims():
            v.decay_survival(1)
        return self.apply_dynamic_events()

    def snapshot(self) -> Dict:
        """Return a serializable snapshot of the environment state."""
        return {
            "time_step": self.time_step,
            "grid": [[c.value for c in row] for row in self.grid],
            "victims": [
                {
                    "id": v.id,
                    "pos": v.position,
                    "severity": v.severity.name,
                    "status": v.status.name,
                    "survival_prob": round(v.survival_prob, 3),
                    "time_waiting": v.time_waiting,
                }
                for v in self.victims
            ],
            "ambulances": [
                {
                    "id": a.id,
                    "pos": a.position,
                    "available": a.is_available,
                    "onboard": a.victims_onboard,
                    "mission": a.mission,
                }
                for a in self.ambulances
            ],
            "medical_kits": self.medical_kits,
        }
