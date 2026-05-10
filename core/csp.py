"""
Constraint Satisfaction Problem (CSP) Module
Models resource allocation: ambulances, rescue team, medical kits
Solver: Backtracking + MRV + Forward Checking + Degree Heuristic
"""

import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from core.environment import Victim, Ambulance, RescueTeam, VictimSeverity, VictimStatus


@dataclass
class Assignment:
    victim_id: int
    ambulance_id: Optional[int]
    priority_rank: int
    kits_assigned: int
    justification: str = ""


class ResourceCSP:
    """
    CSP Variables: each victim → (ambulance_id, priority_rank, kits)
    Constraints:
      - Max 2 victims per ambulance
      - Each rescue team services 1 location at a time
      - Total kits ≤ available kits
      - Critical victims must be assigned before moderate/minor
    """

    def __init__(self, victims: List[Victim], ambulances: List[Ambulance],
                 rescue_teams: List[RescueTeam], medical_kits: int):
        self.victims = [v for v in victims if v.status == VictimStatus.WAITING]
        self.ambulances = ambulances
        self.rescue_teams = rescue_teams
        self.medical_kits = medical_kits
        self.backtrack_count = 0
        self.forward_check_prunes = 0
        self.solve_time_ms = 0.0
        self.stats = {}

    def _kit_need(self, severity: VictimSeverity) -> int:
        return {VictimSeverity.CRITICAL: 2, VictimSeverity.MODERATE: 2, VictimSeverity.MINOR: 1}[severity]

    # ── MRV: sort victims by fewest valid ambulance assignments ──────────────
    def _mrv_order(self, unassigned: List[Victim], current: Dict[int, List[int]]) -> List[Victim]:
        """MRV heuristic: select victim with fewest remaining valid ambulances."""
        def remaining_values(v):
            count = 0
            for a in self.ambulances:
                if len(current.get(a.id, [])) < a.capacity:
                    count += 1
            return count

        # Tie-break: degree heuristic (most-constrained victim → most severe)
        return sorted(unassigned, key=lambda v: (remaining_values(v), -v.severity.value))

    # ── Forward Checking ─────────────────────────────────────────────────────
    def _forward_check(self, amb_assignment: Dict[int, List[int]],
                       kits_used: int, remaining_victims: List[Victim]) -> bool:
        """Prune if remaining victims cannot all be assigned."""
        remaining_kits = self.medical_kits - kits_used
        for v in remaining_victims:
            if self._kit_need(v.severity) > remaining_kits:
                self.forward_check_prunes += 1
                return False
            has_slot = any(len(amb_assignment.get(a.id, [])) < a.capacity
                           for a in self.ambulances)
            if not has_slot:
                self.forward_check_prunes += 1
                return False
        return True

    # ── Backtracking Solver ───────────────────────────────────────────────────
    def _backtrack(self, unassigned: List[Victim],
                   amb_assignment: Dict[int, List[int]],
                   kits_used: int,
                   result: List[Assignment]) -> Optional[List[Assignment]]:

        if not unassigned:
            return result

        ordered = self._mrv_order(unassigned, amb_assignment)
        victim = ordered[0]
        remaining = ordered[1:]

        kit_need = self._kit_need(victim.severity)
        if kits_used + kit_need > self.medical_kits:
            return None

        # Try assigning to each ambulance
        ambulances_tried = sorted(
            self.ambulances,
            key=lambda a: len(amb_assignment.get(a.id, []))  # least loaded first
        )

        for amb in ambulances_tried:
            slots_used = len(amb_assignment.get(amb.id, []))
            if slots_used >= amb.capacity:
                continue

            # Assign
            if amb.id not in amb_assignment:
                amb_assignment[amb.id] = []
            amb_assignment[amb.id].append(victim.id)
            new_kits = kits_used + kit_need
            priority = len(result) + 1

            justification = (
                f"V{victim.id} ({victim.severity.name}) → Amb#{amb.id}. "
                f"Kit cost: {kit_need}. Priority rank: {priority}. "
                f"Survival prob: {victim.survival_prob:.2f}. "
            )
            if victim.severity == VictimSeverity.CRITICAL:
                justification += "Assigned first due to critical severity (MRV prioritized)."
            elif victim.severity == VictimSeverity.MODERATE:
                justification += "Assigned second tier; moderate urgency."
            else:
                justification += "Assigned last; minor severity, lower risk of death."

            assignment = Assignment(
                victim_id=victim.id,
                ambulance_id=amb.id,
                priority_rank=priority,
                kits_assigned=kit_need,
                justification=justification,
            )
            result.append(assignment)

            if self._forward_check(amb_assignment, new_kits, remaining):
                sub = self._backtrack(remaining, amb_assignment, new_kits, result)
                if sub is not None:
                    return sub

            # Undo
            result.pop()
            amb_assignment[amb.id].remove(victim.id)
            self.backtrack_count += 1

        return None

    def solve(self) -> Tuple[Optional[List[Assignment]], Dict]:
        """Solve the CSP and return assignments + statistics."""
        t0 = time.perf_counter()
        self.backtrack_count = 0
        self.forward_check_prunes = 0

        # Sort victims by severity first (critical first)
        sorted_victims = sorted(self.victims, key=lambda v: -v.severity.value)
        amb_assignment: Dict[int, List[int]] = {}
        result = self._backtrack(sorted_victims, amb_assignment, 0, [])

        elapsed = (time.perf_counter() - t0) * 1000
        self.solve_time_ms = round(elapsed, 3)

        self.stats = {
            "solved": result is not None,
            "victims_assigned": len(result) if result else 0,
            "backtrack_count": self.backtrack_count,
            "forward_check_prunes": self.forward_check_prunes,
            "solve_time_ms": self.solve_time_ms,
            "kits_used": sum(a.kits_assigned for a in result) if result else 0,
        }

        return result, self.stats

    def solve_without_heuristics(self) -> Tuple[Optional[List[Assignment]], Dict]:
        """Naive backtracking (no MRV, no forward checking) for comparison."""
        t0 = time.perf_counter()
        self.backtrack_count = 0
        self.forward_check_prunes = 0

        victims = sorted(self.victims, key=lambda v: v.id)  # naive order
        amb_assignment: Dict[int, List[int]] = {}

        def backtrack_naive(unassigned, assignments, kits_used):
            if not unassigned:
                return assignments
            victim = unassigned[0]
            remaining = unassigned[1:]
            kit_need = self._kit_need(victim.severity)
            if kits_used + kit_need > self.medical_kits:
                return None
            for amb in self.ambulances:
                if len(amb_assignment.get(amb.id, [])) >= amb.capacity:
                    continue
                if amb.id not in amb_assignment:
                    amb_assignment[amb.id] = []
                amb_assignment[amb.id].append(victim.id)
                result = backtrack_naive(
                    remaining,
                    assignments + [Assignment(victim.id, amb.id, len(assignments) + 1, kit_need)],
                    kits_used + kit_need,
                )
                if result:
                    return result
                amb_assignment[amb.id].remove(victim.id)
                self.backtrack_count += 1
            return None

        result = backtrack_naive(victims, [], 0)
        elapsed = (time.perf_counter() - t0) * 1000

        stats = {
            "solved": result is not None,
            "victims_assigned": len(result) if result else 0,
            "backtrack_count": self.backtrack_count,
            "forward_check_prunes": 0,
            "solve_time_ms": round(elapsed, 3),
            "kits_used": sum(a.kits_assigned for a in result) if result else 0,
            "variant": "naive (no heuristics)",
        }
        return result, stats
