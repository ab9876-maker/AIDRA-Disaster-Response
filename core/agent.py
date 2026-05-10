"""
AIDRA Agent Orchestrator
Integrates: Environment, Search, CSP, ML, Fuzzy/Bayesian, Dynamic Replanning
"""

import copy
import random
from typing import List, Dict, Optional, Tuple

from core.environment import (DisasterEnvironment, Victim, Ambulance,
                               VictimSeverity, VictimStatus, CellType)
from core.search import RoutePlanner, hill_climbing_route, simulated_annealing_route
from core.csp import ResourceCSP, Assignment
from core.uncertainty import FuzzyPrioritySystem, BayesianRiskEstimator
from ml.models import MLManager


class DecisionLog:
    def __init__(self):
        self.entries: List[Dict] = []

    def log(self, time_step: int, event_type: str, description: str,
            objective_prioritized: str, justification: str, details: Dict = None):
        self.entries.append({
            "time_step": time_step,
            "event_type": event_type,
            "description": description,
            "objective_prioritized": objective_prioritized,
            "justification": justification,
            "details": details or {},
        })

    def to_list(self) -> List[Dict]:
        return self.entries


class AIDRAAgent:
    """
    Adaptive Intelligent Disaster Response Agent
    Orchestrates all AI modules to solve the disaster response CCP.
    """

    def __init__(self, env: DisasterEnvironment, policy: str = "balanced"):
        self.env = env
        self.policy = policy  # 'fast', 'safe', 'balanced'
        self.planner = RoutePlanner(env)
        self.fuzzy = FuzzyPrioritySystem()
        self.bayesian = BayesianRiskEstimator()
        self.ml = MLManager()
        self.decision_log = DecisionLog()

        # Performance tracking
        self.victims_saved = 0
        self.victims_deceased = 0
        self.total_rescue_time = 0
        self.total_risk_exposure = 0
        self.total_distance = 0
        self.replanning_events = 0
        self.csp_stats = {}
        self.search_comparisons = []
        self.survival_estimates = {}
        self.rescue_log = []

        # State
        self.assignments: List[Assignment] = []
        self.ambulance_routes: Dict[int, Tuple[List, Dict]] = {}
        self.replan_triggered = False
        self._priority_order: List[Victim] = []

    # ── Phase 1: Initial Planning ─────────────────────────────────────────────

    def plan(self):
        """Full planning cycle: fuzzy prioritize → CSP allocate → route plan."""
        victims = self.env.get_waiting_victims()

        # Step 1: Fuzzy priority scoring + ML survival estimates
        priority_scores = []
        for v in victims:
            # ML survival prediction
            ml_result = self.ml.predict_survival(
                severity=v.severity.value,
                time_waiting=v.time_waiting,
                proximity_risk=0.7 if self.env.is_high_risk(v.position) else 0.2,
                kits=self.env.medical_kits / 10,
                team=1 if any(t.is_available for t in self.env.rescue_teams) else 0
            )
            ensemble_prob = ml_result["ensemble"]["probability"]
            self.survival_estimates[v.id] = {
                "ml_predictions": ml_result,
                "ensemble_survival_prob": ensemble_prob,
                "adjusted_survival_prob": ensemble_prob,
            }

            # Fuzzy priority
            fuzzy_result = self.fuzzy.compute_priority(
                severity=float(v.severity.value),
                survival_prob=v.survival_prob,
                time_waiting=float(v.time_waiting),
            )
            priority_scores.append((v, fuzzy_result["priority_score"], fuzzy_result))

        # Sort by fuzzy priority (descending)
        priority_scores.sort(key=lambda x: -x[1])
        self._priority_order = [p[0] for p in priority_scores]

        self.decision_log.log(
            self.env.time_step, "INITIAL_PLANNING",
            f"Fuzzy priority scores computed for {len(victims)} victims.",
            objective_prioritized="Victim Prioritization (Conflicting Obj 2)",
            justification=(
                "Fuzzy logic balances severity + survival probability + wait time. "
                "Critical victims with low survival probability receive highest urgency. "
                "This trades overall throughput for saving the most at-risk victims first."
            ),
            details={v.id: {"fuzzy_score": round(s, 2), "fuzzy_detail": d}
                     for v, s, d in priority_scores}
        )

        # Step 2: CSP resource allocation
        csp = ResourceCSP(
            victims=victims,
            ambulances=self.env.ambulances,
            rescue_teams=self.env.rescue_teams,
            medical_kits=self.env.medical_kits,
        )
        self.assignments, self.csp_stats = csp.solve()
        if self.assignments is None:
            self.assignments = []
        csp_naive_assignments, csp_naive_stats = csp.solve_without_heuristics()

        self.decision_log.log(
            self.env.time_step, "CSP_ALLOCATION",
            f"CSP solved: {len(self.assignments)} victims assigned.",
            objective_prioritized="Hard Resource Constraint",
            justification=(
                f"CSP with MRV+Forward Checking reduced backtracks to "
                f"{self.csp_stats['backtrack_count']} vs {csp_naive_stats['backtrack_count']} naive. "
                f"All hard constraints satisfied: ≤2 victims/ambulance, kits ≤ {self.env.medical_kits}. "
                f"Priority rank respects fuzzy ordering."
            ),
            details={"heuristic_stats": self.csp_stats,
                     "naive_stats": csp_naive_stats}
        )
        self.csp_stats["naive_comparison"] = csp_naive_stats

        # Step 3: Route planning with local search optimization
        hc_order, hc_stats = hill_climbing_route(self.env, self._priority_order,
                                                  self.env.base)
        sa_order, sa_stats = simulated_annealing_route(self.env, self._priority_order,
                                                        self.env.base)

        best_order = hc_order if hc_stats["final_cost"] <= sa_stats["final_cost"] else sa_order
        best_algo = "Hill Climbing" if hc_stats["final_cost"] <= sa_stats["final_cost"] else "SA"

        self.decision_log.log(
            self.env.time_step, "LOCAL_SEARCH_OPTIMIZATION",
            f"Victim visit order optimized via {best_algo}.",
            objective_prioritized="Minimize Average Rescue Time",
            justification=(
                f"HC final cost={hc_stats['final_cost']}, SA final cost={sa_stats['final_cost']}. "
                f"{best_algo} selected as it minimized total travel distance. "
                f"SA ran {sa_stats['iterations']} iterations, HC made {hc_stats['improvements']} swaps."
            ),
            details={"hc": hc_stats, "sa": sa_stats, "selected": best_algo}
        )

        # Step 4: Plan routes using A*/search for each ambulance
        amb_missions = self._build_ambulance_missions(best_order)
        self._plan_ambulance_routes(amb_missions)

        return self.assignments

    def _build_ambulance_missions(self, ordered_victims: List) -> Dict[int, List]:
        """Assign victims to ambulances based on CSP assignments, respecting priority order."""
        missions: Dict[int, List] = {a.id: [] for a in self.env.ambulances}
        assigned = {a.victim_id: a.ambulance_id for a in self.assignments}

        for v in ordered_victims:
            if v.id in assigned:
                amb_id = assigned[v.id]
                missions[amb_id].append(v)

        return missions

    def _plan_ambulance_routes(self, missions: Dict[int, List]):
        """For each ambulance, plan route to pick up victims then deliver to medical center."""
        for amb in self.env.ambulances:
            victim_list = missions.get(amb.id, [])
            if not victim_list:
                continue

            full_route = [amb.position]
            route_details = []

            current_pos = amb.position
            for v in victim_list:
                # Bayesian check on route safety
                bay_result = self.bayesian.estimate(
                    zone_type="high_risk_zone" if self.env.is_high_risk(v.position) else "moderate_zone",
                    evidence=["aftershock_detected"] if self.env.is_high_risk(v.position) else []
                )

                # Choose policy based on Bayesian risk
                route_policy = self.policy
                if bay_result["posterior_blocked"] > 0.6:
                    route_policy = "safe"
                    self.decision_log.log(
                        self.env.time_step, "ROUTE_POLICY_CHANGE",
                        f"Amb#{amb.id} → V{v.id}: Bayesian blockage risk={bay_result['posterior_blocked']:.2f}",
                        objective_prioritized="Risk Exposure (Conflicting Obj 1)",
                        justification=(
                            f"Bayesian posterior P(blocked)={bay_result['posterior_blocked']:.2f} > 0.6. "
                            "Switching to safe route (avoids high-risk zones). "
                            "Trade-off: longer route, lower risk exposure."
                        ),
                        details={"bayesian": bay_result}
                    )

                path, stats, all_comparisons = self.planner.plan_route(
                    current_pos, v.position, policy=route_policy
                )
                self.search_comparisons.append({
                    "from": current_pos, "to": v.position,
                    "amb_id": amb.id, "victim_id": v.id,
                    "comparisons": {k: v2[1] for k, v2 in all_comparisons.items()},
                    "selected": stats,
                })

                if path:
                    full_route.extend(path[1:])
                    route_details.append({
                        "type": "pickup",
                        "victim_id": v.id,
                        "path": path,
                        "stats": stats,
                        "risk_cells": stats.get("risk_cells", 0),
                    })
                    current_pos = v.position

            # Route to nearest medical center
            best_mc = min(self.env.medical_centers,
                          key=lambda mc: self.env.manhattan_distance(current_pos, mc))
            mc_path, mc_stats, mc_comparisons = self.planner.plan_route(
                current_pos, best_mc, policy=route_policy if 'route_policy' in dir() else self.policy
            )
            if mc_path:
                full_route.extend(mc_path[1:])
                route_details.append({
                    "type": "delivery",
                    "medical_center": best_mc,
                    "path": mc_path,
                    "stats": mc_stats,
                })

            amb.current_route = full_route
            self.ambulance_routes[amb.id] = (full_route, route_details)

    # ── Phase 2: Dynamic Execution & Replanning ───────────────────────────────

    def execute_step(self) -> Dict:
        """Execute one time step of the simulation."""
        events = self.env.tick()
        triggered_replanning = False

        for event in events:
            self.decision_log.log(
                self.env.time_step, "DYNAMIC_EVENT",
                event["description"],
                objective_prioritized="Dynamic Replanning",
                justification=(
                    f"Event '{event['type']}' triggered at t={self.env.time_step}. "
                    "Agent detecting environmental change..."
                )
            )

            if event["type"] in ("road_block", "risk_increase"):
                # Check if any current route is affected
                for amb in self.env.ambulances:
                    if not amb.is_available:
                        continue
                    blocked_cell = event.get("cell")
                    if blocked_cell and blocked_cell in amb.current_route:
                        self.replanning_events += 1
                        triggered_replanning = True
                        self.decision_log.log(
                            self.env.time_step, "REPLAN_TRIGGERED",
                            f"Amb#{amb.id} route blocked at {blocked_cell}. Replanning.",
                            objective_prioritized="Rescue Time vs. Risk (Conflicting Obj 1)",
                            justification=(
                                f"Blockage at {blocked_cell} invalidates current route. "
                                "Agent replanning: searches alternate path. "
                                "Trade-off: longer time but route stays clear."
                            )
                        )
                        # Replan from current ambulance position
                        self._replan_ambulance(amb)

            elif event["type"] == "new_victim":
                self.decision_log.log(
                    self.env.time_step, "NEW_VICTIM",
                    event["description"],
                    objective_prioritized="Overall Throughput (Conflicting Obj 2)",
                    justification=(
                        "New victim detected. Re-evaluating CSP allocation and priorities. "
                        "Agent must balance this victim's needs against ongoing rescues."
                    )
                )
                # Trigger a re-plan
                triggered_replanning = True
                self._replan_all()

        # Move ambulances along their routes
        for amb in self.env.ambulances:
            if amb.current_route and amb.route_index < len(amb.current_route) - 1:
                amb.route_index += 1
                new_pos = amb.current_route[amb.route_index]
                if self.env.is_high_risk(new_pos):
                    amb.risk_exposure += 1
                    self.total_risk_exposure += 1
                amb.position = new_pos
                amb.total_distance += 1

                # Check if reached a victim
                for v in self.env.victims:
                    if (v.status == VictimStatus.WAITING and
                            v.position == new_pos and
                            len(amb.victims_onboard) < amb.capacity):
                        v.status = VictimStatus.IN_RESCUE
                        amb.victims_onboard.append(v.id)
                        self.env.medical_kits -= self._kits_for_severity(v.severity)
                        self.decision_log.log(
                            self.env.time_step, "VICTIM_PICKUP",
                            f"Amb#{amb.id} picked up V{v.id} ({v.severity.name}) at {new_pos}",
                            objective_prioritized="Victim Prioritization",
                            justification=f"Victim {v.id} survival prob at pickup: {v.survival_prob:.2f}"
                        )

                # Check if at medical center
                if new_pos in self.env.medical_centers and amb.victims_onboard:
                    for vid in amb.victims_onboard:
                        v = next((x for x in self.env.victims if x.id == vid), None)
                        if v:
                            v.status = VictimStatus.RESCUED
                            self.victims_saved += 1
                            self.total_rescue_time += self.env.time_step
                            self.rescue_log.append({
                                "victim_id": v.id,
                                "severity": v.severity.name,
                                "rescue_time": self.env.time_step,
                                "survival_prob_at_rescue": round(v.survival_prob, 3),
                                "ambulance": amb.id,
                            })
                            self.decision_log.log(
                                self.env.time_step, "VICTIM_RESCUED",
                                f"V{v.id} ({v.severity.name}) delivered to medical center {new_pos}",
                                objective_prioritized="Victims Saved",
                                justification=f"Rescue complete. Survival probability: {v.survival_prob:.2f}"
                            )
                    amb.victims_onboard = []
                    amb.trips_completed += 1
                    # Return to base for next mission
                    amb.current_route = [new_pos, self.env.base]
                    amb.route_index = 0

        # Check deceased (survival → 0)
        for v in self.env.victims:
            if v.status == VictimStatus.WAITING and v.survival_prob <= 0:
                v.status = VictimStatus.DECEASED
                self.victims_deceased += 1
                self.decision_log.log(
                    self.env.time_step, "VICTIM_DECEASED",
                    f"V{v.id} ({v.severity.name}) deceased at t={self.env.time_step}",
                    objective_prioritized="N/A",
                    justification="Survival probability reached 0 before rescue."
                )

        return {
            "time_step": self.env.time_step,
            "events": events,
            "replanning_triggered": triggered_replanning,
            "victims_saved": self.victims_saved,
            "victims_deceased": self.victims_deceased,
        }

    def _kits_for_severity(self, severity: VictimSeverity) -> int:
        return {VictimSeverity.CRITICAL: 2, VictimSeverity.MODERATE: 2, VictimSeverity.MINOR: 1}[severity]

    def _replan_ambulance(self, amb: Ambulance):
        """Replan route for a single ambulance due to environmental change."""
        remaining_victims = [
            v for v in self.env.victims
            if v.status == VictimStatus.WAITING and v.id not in amb.victims_onboard
        ]
        if not remaining_victims:
            return

        # Re-run local search + route planning for remaining
        if remaining_victims:
            hc_order, _ = hill_climbing_route(self.env, remaining_victims, amb.position)
            new_route = [amb.position]
            pos = amb.position
            for v in hc_order:
                path, stats, _ = self.planner.plan_route(pos, v.position, policy="balanced")
                if path:
                    new_route.extend(path[1:])
                    pos = v.position

            best_mc = min(self.env.medical_centers,
                          key=lambda mc: self.env.manhattan_distance(pos, mc))
            mc_path, _, _ = self.planner.plan_route(pos, best_mc, policy="balanced")
            if mc_path:
                new_route.extend(mc_path[1:])

            amb.current_route = new_route
            amb.route_index = 0

    def _replan_all(self):
        """Full re-planning triggered by significant environment change."""
        self.plan()

    # ── Phase 3: Simulation Run ───────────────────────────────────────────────

    def run_simulation(self, max_steps: int = 35) -> Dict:
        """Full simulation run."""
        print(f"\n{'='*60}")
        print("  AIDRA - Adaptive Intelligent Disaster Response Agent")
        print(f"  Policy: {self.policy.upper()} | Max Steps: {max_steps}")
        print(f"{'='*60}\n")

        self.plan()
        snapshots = [self.env.snapshot()]

        for step in range(max_steps):
            result = self.execute_step()
            snapshots.append(self.env.snapshot())

            waiting = len(self.env.get_waiting_victims())
            rescued = sum(1 for v in self.env.victims if v.status == VictimStatus.RESCUED)
            total = len(self.env.victims)

            if rescued + self.victims_deceased >= total and step > 5:
                break

        return self.generate_report(snapshots)

    # ── Phase 4: Performance Report ───────────────────────────────────────────

    def generate_report(self, snapshots: List[Dict]) -> Dict:
        total_victims = len(self.env.victims)
        avg_rescue_time = self.total_rescue_time / max(self.victims_saved, 1)
        resource_utilization = {
            "ambulances_used": sum(1 for a in self.env.ambulances if a.trips_completed > 0),
            "total_ambulance_trips": sum(a.trips_completed for a in self.env.ambulances),
            "kits_remaining": self.env.medical_kits,
            "kits_used": 10 - self.env.medical_kits,
        }

        # Path optimality ratio (A* / BFS cost)
        optimality_ratios = []
        for comp in self.search_comparisons:
            astar_cost = comp["comparisons"].get("A*_balanced", {}).get("path_cost")
            bfs_cost = comp["comparisons"].get("BFS", {}).get("path_cost")
            if astar_cost and bfs_cost and bfs_cost > 0:
                optimality_ratios.append(astar_cost / bfs_cost)

        avg_optimality = sum(optimality_ratios) / len(optimality_ratios) if optimality_ratios else 1.0

        # Risk exposure score
        total_path_steps = sum(a.total_distance for a in self.env.ambulances)
        risk_score = self.total_risk_exposure / max(total_path_steps, 1)

        return {
            "summary": {
                "total_victims": total_victims,
                "victims_saved": self.victims_saved,
                "victims_deceased": self.victims_deceased,
                "victims_in_progress": total_victims - self.victims_saved - self.victims_deceased,
                "avg_rescue_time": round(avg_rescue_time, 2),
                "total_simulation_steps": self.env.time_step,
                "replanning_events": self.replanning_events,
                "policy_used": self.policy,
            },
            "kpis": {
                "victims_saved": self.victims_saved,
                "avg_rescue_time": round(avg_rescue_time, 2),
                "path_optimality_ratio": round(avg_optimality, 3),
                "resource_utilization": resource_utilization,
                "risk_exposure_score": round(risk_score, 4),
                "total_risk_cells_traversed": self.total_risk_exposure,
            },
            "ml_metrics": {
                "survival_models": self.ml.survival_metrics,
                "risk_models": self.ml.risk_metrics,
            },
            "csp_stats": self.csp_stats,
            "rescue_log": self.rescue_log,
            "survival_estimates": self.survival_estimates,
            "decision_log": self.decision_log.to_list(),
            "search_comparisons": self.search_comparisons,
            "priority_order": [{"id": v.id, "severity": v.severity.name} for v in self._priority_order],
            "snapshots": snapshots,
            "assignments": [
                {
                    "victim_id": a.victim_id,
                    "ambulance_id": a.ambulance_id,
                    "priority_rank": a.priority_rank,
                    "kits_assigned": a.kits_assigned,
                    "justification": a.justification,
                }
                for a in (self.assignments or [])
            ],
        }
