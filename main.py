"""
AIDRA - Adaptive Intelligent Disaster Response Agent
AIC-201 Complex Computing Problem
Instructor: Dr. Arshad Farhad | Semester 5-A

Run: python main.py
"""

import json
import numpy as np

from core.environment import DisasterEnvironment
from core.agent import AIDRAAgent


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def run_policy(policy: str, max_steps: int = 60) -> dict:
    env = DisasterEnvironment(seed=42)
    agent = AIDRAAgent(env, policy=policy)
    return agent.run_simulation(max_steps=max_steps)


def print_report(report: dict, policy: str):
    s = report["summary"]
    k = report["kpis"]
    print(f"\n{'─'*55}")
    print(f"  Policy: {policy.upper()}")
    print(f"{'─'*55}")
    print(f"  Victims Saved    : {s['victims_saved']}")
    print(f"  Victims Deceased : {s['victims_deceased']}")
    print(f"  Avg Rescue Time  : {k['avg_rescue_time']} steps")
    print(f"  Risk Exposure    : {k['total_risk_cells_traversed']} cells")
    print(f"  Path Optimality  : {k['path_optimality_ratio']}")
    print(f"  Replanning Events: {s['replanning_events']}")
    print(f"  Decision Log Size: {len(report['decision_log'])}")
    print()
    print("  ML Metrics (Survival Prediction):")
    for name, m in report["ml_metrics"]["survival_models"].items():
        print(f"    {name:15s} Acc={m['accuracy']:.2f}  "
              f"P={m['precision']:.2f}  R={m['recall']:.2f}  F1={m['f1']:.2f}")
    print()
    print("  Priority Order:")
    for v in report["priority_order"]:
        print(f"    V{v['id']} ({v['severity']})")
    print()
    print("  CSP Stats:")
    csp = report["csp_stats"]
    print(f"    Solved={csp['solved']}  Backtracks={csp['backtrack_count']}  "
          f"FwdPrunes={csp['forward_check_prunes']}  Time={csp['solve_time_ms']}ms")


def main():
    print("\n" + "="*55)
    print("  AIDRA — Adaptive Intelligent Disaster Response Agent")
    print("  AIC-201 | Complex Computing Problem")
    print("="*55)

    all_reports = {}

    for policy in ["fast", "safe", "balanced"]:
        report = run_policy(policy)
        all_reports[policy] = report
        print_report(report, policy)

    # Save combined report
    with open("all_reports.json", "w") as f:
        json.dump(all_reports, f, cls=NumpyEncoder, indent=2)
    print("\n  Full report saved to all_reports.json")
    print("  Open AIDRA_Dashboard.html in a browser to explore results.\n")


if __name__ == "__main__":
    main()
