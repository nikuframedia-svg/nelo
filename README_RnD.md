# ProdPlan 4.0 + SmartInventory — R&D Programme

## SIFIDE Classification
- **Type**: Applied Research + Experimental Development
- **Domain**: Industrial Planning & Scheduling with AI
- **TRL Start**: 4 (validated in lab)
- **TRL Target**: 6-7 (prototype in relevant environment)

---

## 1. Research Questions (RQs)

### RQ1: Hybrid APS with Dynamic Routing
> Can we design a scheduling engine that mixes classical APS heuristics with dynamic per-operation routing decisions and ML-predicted setup/throughput, and outperform baseline APS in makespan + stability?

**Technical Uncertainty**:
- Optimal scoring function for route selection is unknown
- Trade-off between exploration (new routes) and exploitation (proven routes)
- Setup prediction accuracy with limited historical data

### RQ2: Automated Industrial Suggestions
> Can we automatically generate human-readable suggestions (move operations, change routes, adjust shifts) that production directors consider credible and useful?

**Technical Uncertainty**:
- What makes a suggestion "credible" vs "naive"?
- How to rank suggestions by impact without full re-optimization?
- LLM hallucination risk in industrial context

### RQ3: Coupled Inventory-Production Optimization
> Can we jointly optimize coverage, risk, and OTD instead of treating inventory and production as separate modules?

**Technical Uncertainty**:
- Multi-objective formulation with conflicting goals
- Demand uncertainty propagation through capacity constraints
- Computational tractability for real-time decisions

### RQ4: Explainable AI Co-pilot
> Can we create a factory co-pilot that answers technical questions, justifies APS choices, and proposes scenarios without becoming a black box?

**Technical Uncertainty**:
- Faithful explanations vs plausible-but-wrong explanations
- Appropriate level of detail for different users
- Maintaining determinism while using probabilistic LLM

---

## 2. Work Packages

### WP1: APS Core + Routing Intelligence
- **Duration**: M1-M6
- **Lead**: Backend/Algorithm Team
- **Modules**: `routing_engine.py`, `setup_engine.py`
- **Experiments**:
  - E1.1: Fixed routing vs dynamic routing (makespan comparison)
  - E1.2: Rule-based setup vs ML-predicted setup (MAE, hours saved)

### WP2: What-If + Explainable AI
- **Duration**: M3-M9
- **Lead**: AI/NLP Team
- **Modules**: `explainability_engine.py`, enhanced `what_if_engine.py`
- **Experiments**:
  - E2.1: Suggestion usefulness study (N=20 suggestions, Likert scale)
  - E2.2: Explanation fidelity test (does explanation match actual reason?)

### WP3: Inventory-Production Coupling
- **Duration**: M4-M10
- **Lead**: Operations Research Team
- **Modules**: `inventory_optimization.py`, simulation engine
- **Experiments**:
  - E3.1: Coupled vs decoupled optimization (Monte Carlo, 1000 scenarios)
  - E3.2: Sensitivity analysis on demand variance

### WP4: Learning Scheduler + Experimentation
- **Duration**: M6-M12
- **Lead**: ML Team
- **Modules**: `learning_scheduler.py`, experiment infrastructure
- **Experiments**:
  - E4.1: Contextual bandits vs fixed heuristics (cumulative regret)
  - E4.2: Transfer learning across product families

---

## 3. Experiment Infrastructure

### Logging Schema
All experiments produce JSON logs with:
```json
{
  "experiment_id": "E1.1-2024-001",
  "timestamp": "2024-01-15T10:30:00Z",
  "hypothesis": "H1.1",
  "config": { "routing_mode": "dynamic", "scoring": "setup_aware" },
  "inputs": { "num_orders": 50, "num_machines": 12 },
  "outputs": {
    "makespan_h": 42.5,
    "setup_hours": 3.2,
    "otd_pct": 94.5
  },
  "baseline_outputs": {
    "makespan_h": 48.1,
    "setup_hours": 5.8,
    "otd_pct": 91.2
  },
  "delta_pct": { "makespan": -11.6, "setup": -44.8, "otd": +3.6 }
}
```

### Metrics Dashboard
- Real-time KPI tracking: Makespan, OTD, Setup Hours, Utilization
- Experiment comparison view
- Statistical significance tests (t-test, Mann-Whitney)

---

## 4. Module Overview

| Module | Purpose | RQ | Status |
|--------|---------|----| ------|
| `routing_engine.py` | Dynamic route selection per operation | Q1 | 🔬 Research |
| `setup_engine.py` | Setup time prediction (rules + ML) | Q1 | 🔬 Research |
| `learning_scheduler.py` | Logged decisions + learned policies | Q1, Q4 | 🔬 Research |
| `inventory_optimization.py` | Joint stock + capacity optimization | Q3 | 🔬 Research |
| `explainability_engine.py` | Decision justification + explanations | Q2, Q4 | 🔬 Research |
| `experiment_logger.py` | Structured logging for experiments | All | 🛠️ Infrastructure |

---

## 5. SIFIDE Documentation Requirements

For each experiment, we maintain:
1. **Hypothesis document** (what we expect, why)
2. **Experiment protocol** (inputs, procedure, metrics)
3. **Raw results** (JSON logs)
4. **Analysis report** (statistical tests, conclusions)
5. **Technical learnings** (what worked, what didn't, next steps)

This documentation structure supports SIFIDE audits by demonstrating:
- Systematic investigation
- Technical uncertainty
- Iterative refinement
- Novel contribution beyond state-of-the-art

---

## 6. Current Baseline (MVP)

The existing MVP serves as our **baseline** for all experiments:
- Fixed Route A scheduling
- Rule-based setup (family matrix)
- Decoupled inventory (ABC/XYZ only)
- Reactive suggestions (identify problems, don't propose solutions)

All improvements are measured against this baseline.

---

## 7. Next Milestones

| Milestone | Target Date | Deliverable |
|-----------|-------------|-------------|
| M1 | Week 2 | `routing_engine.py` with 3 scoring strategies |
| M2 | Week 4 | Experiment E1.1 completed (dynamic vs fixed routing) |
| M3 | Week 6 | `setup_engine.py` with ML predictor |
| M4 | Week 8 | `explainability_engine.py` integrated |
| M5 | Week 10 | User study E2.1 (suggestion usefulness) |
| M6 | Week 12 | `inventory_optimization.py` with simulation |

---

*This document is maintained as part of the SIFIDE R&D programme documentation.*



