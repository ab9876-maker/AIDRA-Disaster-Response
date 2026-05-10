"""
Uncertainty Handling Module
1. Fuzzy Logic: Victim priority scoring under noisy/incomplete data
2. Bayesian Inference: Road blockage probability estimation
"""

import math
from typing import Dict, List, Tuple


# ─── Fuzzy Logic Engine ───────────────────────────────────────────────────────

class FuzzySet:
    """Trapezoidal membership function: (a, b, c, d)"""

    def __init__(self, a: float, b: float, c: float, d: float, label: str):
        self.a, self.b, self.c, self.d = a, b, c, d
        self.label = label

    def membership(self, x: float) -> float:
        a, b, c, d = self.a, self.b, self.c, self.d
        if x <= a or x >= d:
            return 0.0
        elif b <= x <= c:
            return 1.0
        elif a < x < b:
            return (x - a) / (b - a)
        else:
            return (d - x) / (d - c)


class FuzzyPrioritySystem:
    """
    Fuzzy rule base for victim rescue priority.
    Inputs: severity (1-3), survival_prob (0-1), time_waiting (0-30)
    Output: priority_score (0-100)

    Rules:
      IF severity=high AND survival_prob=low  → priority = CRITICAL (urgent)
      IF severity=high AND survival_prob=high → priority = HIGH
      IF severity=medium AND time_waiting=long → priority = HIGH
      IF severity=low                          → priority = LOW
      ... etc.
    """

    def __init__(self):
        # Severity fuzzy sets
        self.sev_low = FuzzySet(0.9, 1.0, 1.5, 1.6, "low")
        self.sev_medium = FuzzySet(1.5, 2.0, 2.0, 2.5, "medium")
        self.sev_high = FuzzySet(2.4, 2.7, 3.0, 3.1, "high")

        # Survival probability
        self.surv_low = FuzzySet(-0.1, 0.0, 0.3, 0.5, "low_surv")
        self.surv_medium = FuzzySet(0.3, 0.5, 0.6, 0.8, "med_surv")
        self.surv_high = FuzzySet(0.6, 0.75, 1.0, 1.1, "high_surv")

        # Time waiting
        self.wait_short = FuzzySet(-1, 0, 5, 10, "short_wait")
        self.wait_medium = FuzzySet(8, 12, 18, 22, "med_wait")
        self.wait_long = FuzzySet(18, 22, 30, 31, "long_wait")

        # Output crisp values for priority (centroid defuzz)
        self.output_levels = {
            "urgent": 90,
            "critical": 75,
            "high": 60,
            "medium": 40,
            "low": 20,
        }

    def _fuzzify(self, severity: float, survival_prob: float,
                  time_waiting: float) -> Dict:
        return {
            "sev_low": self.sev_low.membership(severity),
            "sev_med": self.sev_medium.membership(severity),
            "sev_high": self.sev_high.membership(severity),
            "surv_low": self.surv_low.membership(survival_prob),
            "surv_med": self.surv_medium.membership(survival_prob),
            "surv_high": self.surv_high.membership(survival_prob),
            "wait_short": self.wait_short.membership(time_waiting),
            "wait_med": self.wait_medium.membership(time_waiting),
            "wait_long": self.wait_long.membership(time_waiting),
        }

    def _apply_rules(self, f: Dict) -> Dict:
        """
        Rule base (15 rules, Mamdani min-AND).
        Returns strength for each output level.
        """
        rules = {
            "urgent": max(
                min(f["sev_high"], f["surv_low"]),             # R1: high severity + dying
                min(f["sev_high"], f["wait_long"]),            # R2: critical + long wait
            ),
            "critical": max(
                min(f["sev_high"], f["surv_med"]),             # R3: high severity + ok survival
                min(f["sev_med"], f["surv_low"], f["wait_long"]),  # R4: med+dying+long
            ),
            "high": max(
                min(f["sev_high"], f["surv_high"]),            # R5: high sev + good survival
                min(f["sev_med"], f["surv_low"]),              # R6: med + dying
                min(f["sev_med"], f["wait_long"]),             # R7: moderate + long wait
            ),
            "medium": max(
                min(f["sev_med"], f["surv_med"]),              # R8
                min(f["sev_med"], f["wait_med"]),              # R9
                min(f["sev_low"], f["surv_low"]),              # R10: low sev but dying
            ),
            "low": max(
                min(f["sev_low"], f["surv_high"]),             # R11
                min(f["sev_low"], f["wait_short"]),            # R12
                f["sev_low"] * f["surv_med"],                  # R13
            ),
        }
        return rules

    def _defuzzify(self, activations: Dict) -> float:
        """Weighted average (centroid) defuzzification."""
        numerator = sum(activations[k] * self.output_levels[k] for k in activations)
        denominator = sum(activations.values())
        if denominator == 0:
            return 50.0
        return numerator / denominator

    def compute_priority(self, severity: float, survival_prob: float,
                          time_waiting: float) -> Dict:
        """
        Returns priority score (0-100) and explanation.
        """
        fuzz = self._fuzzify(severity, survival_prob, time_waiting)
        activations = self._apply_rules(fuzz)
        score = self._defuzzify(activations)

        dominant_rule = max(activations, key=activations.get)
        explanation = (
            f"Fuzzy priority={score:.1f}/100. "
            f"Dominant rule: '{dominant_rule}' (strength={activations[dominant_rule]:.2f}). "
            f"Inputs: severity={severity}, survival={survival_prob:.2f}, wait={time_waiting}t"
        )

        return {
            "priority_score": round(score, 2),
            "dominant_rule": dominant_rule,
            "rule_activations": {k: round(v, 3) for k, v in activations.items()},
            "explanation": explanation,
        }


# ─── Bayesian Road Blockage Estimator ────────────────────────────────────────

class BayesianRiskEstimator:
    """
    Estimates road blockage probability using Bayes' theorem.
    Prior: base probability of blockage given disaster severity.
    Likelihood: evidence from aftershock sensors, fire proximity, structural damage.
    P(blocked | evidence) = P(evidence | blocked) * P(blocked) / P(evidence)
    """

    def __init__(self):
        # Prior probabilities of blockage per zone type
        self.priors = {
            "high_risk_zone": 0.65,
            "moderate_zone": 0.30,
            "safe_zone": 0.08,
        }

        # Likelihoods P(evidence | blocked)
        self.likelihoods = {
            "aftershock_detected": {"blocked": 0.80, "not_blocked": 0.20},
            "fire_nearby": {"blocked": 0.70, "not_blocked": 0.25},
            "structural_damage_report": {"blocked": 0.75, "not_blocked": 0.15},
            "traffic_flow_stopped": {"blocked": 0.90, "not_blocked": 0.05},
        }

    def estimate(self, zone_type: str, evidence: List[str]) -> Dict:
        """
        zone_type: 'high_risk_zone' | 'moderate_zone' | 'safe_zone'
        evidence: list of observed evidence keys
        Returns: posterior probability of blockage
        """
        prior = self.priors.get(zone_type, 0.3)
        p_blocked = prior
        p_not_blocked = 1.0 - prior

        for ev in evidence:
            if ev in self.likelihoods:
                ll_blocked = self.likelihoods[ev]["blocked"]
                ll_not_blocked = self.likelihoods[ev]["not_blocked"]
                p_blocked *= ll_blocked
                p_not_blocked *= ll_not_blocked

        # Normalize
        total = p_blocked + p_not_blocked
        posterior = p_blocked / total if total > 0 else prior

        explanation = (
            f"P(blocked|evidence)={posterior:.3f}. "
            f"Prior={prior}, Evidence={evidence}. "
        )
        if posterior > 0.7:
            explanation += "HIGH blockage risk → Agent should replan route."
        elif posterior > 0.4:
            explanation += "MODERATE risk → Monitor; prepare alternate route."
        else:
            explanation += "LOW risk → Route considered passable."

        return {
            "posterior_blocked": round(posterior, 3),
            "posterior_clear": round(1 - posterior, 3),
            "prior": prior,
            "evidence": evidence,
            "recommendation": "replan" if posterior > 0.6 else "proceed",
            "explanation": explanation,
        }

    def batch_estimate(self, roads: List[Dict]) -> List[Dict]:
        """Estimate blockage for multiple road segments."""
        results = []
        for road in roads:
            result = self.estimate(
                road.get("zone_type", "moderate_zone"),
                road.get("evidence", [])
            )
            result["road_id"] = road.get("id", "unknown")
            results.append(result)
        return results
