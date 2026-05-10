# AIDRA — Adaptive Intelligent Disaster Response Agent
**AIC-201 Complex Computing Problem | Semester 5-A | Dr. Arshad Farhad**

A hybrid AI system that integrates Search, CSP, Machine Learning, Fuzzy Logic, and Bayesian reasoning to solve a dynamic disaster response problem.

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python main.py
```

Then open `AIDRA_Dashboard.html` in your browser for the interactive visualization.

---

## 🧠 AI Modules

| Module | Techniques |
|--------|-----------|
| Search & Planning | BFS, DFS, Greedy Best-First, A*, Hill Climbing, Simulated Annealing |
| Constraint Satisfaction | Backtracking + MRV + Forward Checking + Degree Heuristic |
| Machine Learning | kNN, Naive Bayes, MLP Perceptron |
| Uncertainty Handling | Fuzzy Logic (15-rule Mamdani), Bayesian Inference |
| Dynamic Replanning | Event-driven replan on road blocks, new victims, risk changes |

---

## 📁 Project Structure

```
AIDRA/
├── main.py                  # Entry point — runs all 3 policies
├── requirements.txt
├── AIDRA_Dashboard.html     # Interactive results dashboard
├── core/
│   ├── environment.py       # Grid, victims, ambulances, dynamic events
│   ├── search.py            # All search algorithms + RoutePlanner
│   ├── csp.py               # CSP solver with MRV + Forward Checking
│   ├── agent.py             # AIDRA orchestrator + decision log
│   └── uncertainty.py       # Fuzzy Logic + Bayesian estimator
└── ml/
    └── models.py            # kNN, Naive Bayes, MLP + evaluation metrics
```

---

## 📊 Problem Scenario

- **Grid**: 10×10 urban disaster map with high-risk zones, blocked roads, safe zones
- **Victims**: 5 initial (2 critical, 2 moderate, 1 minor) + 1 dynamic new victim at t=20
- **Resources**: 2 ambulances (capacity 2), 1 rescue team, 10 medical kits
- **Dynamic Events**: Road blocks at t=8, t=15; risk expansion at t=12; new victim at t=20

## ⚡ Conflicting Objectives

1. **Rescue Time vs. Risk Exposure** — faster routes go through hazard zones
2. **Victim Prioritization vs. Throughput** — critical cases delay rescue of others

---

## 📈 Key Results (Balanced Policy)

- Fuzzy priority scoring guides rescue order
- A* with risk_penalty=3.0 balances time/risk tradeoff
- CSP with MRV+FC: 0 backtracks vs naive baseline
- Bayesian posterior >0.6 triggers automatic safe-routing

---

## 🎥 Demo Video
[Link to be added]

## 🔗 LinkedIn Post
[Link to be added]
