# Backend Audit Report - Contract 15

## Summary

- **Total Python files**: 162
- **Total endpoints**: 150
- **Total classes**: 881
- **Total functions**: 2181
- **Parse errors**: 0

## Domains

### causal (5 files)
  - `causal/causal_effect_estimator.py`
  - `causal/causal_graph_builder.py`
  - `causal/complexity_dashboard_engine.py`
  - `causal/data_collector.py`
  - `causal/tests/test_causal.py`

### chat (2 files)
  - `chat/engine.py`
  - `chat/router.py`

### core (3 files)
  - `core/explainability/explainability_engine.py`
  - `core/optimization/scheduling_milp.py`
  - `core/setup_engine.py`

### dashboards (6 files)
  - `dashboards/capacity_projection.py`
  - `dashboards/cell_performance.py`
  - `dashboards/gantt_comparison.py`
  - `dashboards/machine_oee.py`
  - `dashboards/operator_dashboard.py`
  - `dashboards/utilization_heatmap.py`

### digital_twin (11 files)
  - `digital_twin/api_shi_dt.py`
  - `digital_twin/api_xai_dt.py`
  - `digital_twin/api_xai_dt_product.py`
  - `digital_twin/health_indicator_cvae.py`
  - `digital_twin/process_optimization.py`
  - `digital_twin/rul_estimator.py`
  - `digital_twin/rul_integration_scheduler.py`
  - `digital_twin/shi_dt.py`
  - `digital_twin/tests/test_rul_integration.py`
  - `digital_twin/xai_dt_geometry.py`
  - ... and 1 more

### duplios (24 files)
  - `duplios/api_compliance.py`
  - `duplios/api_duplios.py`
  - `duplios/api_gap_filling.py`
  - `duplios/api_pdm.py`
  - `duplios/api_trust_index.py`
  - `duplios/carbon_calculator.py`
  - `duplios/compliance_engine.py`
  - `duplios/compliance_models.py`
  - `duplios/compliance_radar.py`
  - `duplios/dpp_models.py`
  - ... and 14 more

### evaluation (3 files)
  - `evaluation/data_quality.py`
  - `evaluation/kpi_engine.py`
  - `evaluation/model_metrics.py`

### experiments (1 files)
  - `experiments/experiment_runner.py`

### explainability (1 files)
  - `explainability/explain.py`

### integration (1 files)
  - `integration/erp_mes_connector.py`

### inventory (1 files)
  - `inventory/inventory_engine.py`

### ml (3 files)
  - `ml/forecasting.py`
  - `ml/rul_models.py`
  - `ml/setup_models.py`

### ops_ingestion (6 files)
  - `ops_ingestion/api.py`
  - `ops_ingestion/data_quality.py`
  - `ops_ingestion/excel_parser.py`
  - `ops_ingestion/models.py`
  - `ops_ingestion/schemas.py`
  - `ops_ingestion/services.py`

### optimization (11 files)
  - `optimization/api_optimization.py`
  - `optimization/drl_scheduler/drl_scheduler_interface.py`
  - `optimization/drl_scheduler/drl_trainer.py`
  - `optimization/drl_scheduler/env_scheduling.py`
  - `optimization/evaluator.py`
  - `optimization/learning_scheduler.py`
  - `optimization/math_optimization.py`
  - `optimization/objectives.py`
  - `optimization/scheduling_models.py`
  - `optimization/solver_interface.py`
  - ... and 1 more

### planning (7 files)
  - `planning/capacity_planner.py`
  - `planning/chained_scheduler.py`
  - `planning/conventional_scheduler.py`
  - `planning/operator_allocator.py`
  - `planning/planning_engine.py`
  - `planning/planning_modes.py`
  - `planning/setup_optimizer.py`

### prodplan (2 files)
  - `prodplan/execution_log_models.py`
  - `prodplan/work_instructions.py`

### product_metrics (3 files)
  - `product_metrics/delivery_time_engine.py`
  - `product_metrics/product_classification.py`
  - `product_metrics/product_kpi_engine.py`

### project_planning (4 files)
  - `project_planning/project_kpi_engine.py`
  - `project_planning/project_load_engine.py`
  - `project_planning/project_model.py`
  - `project_planning/project_priority_optimization.py`

### quality (2 files)
  - `quality/api_prevention_guard.py`
  - `quality/prevention_guard.py`

### rd (8 files)
  - `rd/api.py`
  - `rd/causal_deep_experiments.py`
  - `rd/experiments_core.py`
  - `rd/reporting.py`
  - `rd/wp1_routing_experiments.py`
  - `rd/wp2_suggestions_eval.py`
  - `rd/wp3_inventory_capacity.py`
  - `rd/wp4_learning_scheduler.py`

### reporting (2 files)
  - `reporting/comparison_engine.py`
  - `reporting/report_generator.py`

### research (6 files)
  - `research/experiment_logger.py`
  - `research/explainability_engine.py`
  - `research/inventory_optimization.py`
  - `research/learning_scheduler.py`
  - `research/routing_engine.py`
  - `research/setup_engine.py`

### root (16 files)
  - `actions_engine.py`
  - `api.py`
  - `chains.py`
  - `command_parser.py`
  - `dashboards.py`
  - `data_loader.py`
  - `diagnose_features.py`
  - `feature_flags.py`
  - `ml_engine.py`
  - `models_common.py`
  - ... and 6 more

### scheduling (7 files)
  - `scheduling/api.py`
  - `scheduling/cpsat_models.py`
  - `scheduling/data_driven_durations.py`
  - `scheduling/drl_policy_stub.py`
  - `scheduling/heuristics.py`
  - `scheduling/milp_models.py`
  - `scheduling/types.py`

### scripts (1 files)
  - `scripts/seed_dev_data.py`

### shopfloor (2 files)
  - `shopfloor/api_work_instructions.py`
  - `shopfloor/work_instructions.py`

### simulation (5 files)
  - `simulation/tests/test_zdm_simulator.py`
  - `simulation/zdm/api_zdm.py`
  - `simulation/zdm/failure_scenario_generator.py`
  - `simulation/zdm/recovery_strategy_engine.py`
  - `simulation/zdm/zdm_simulator.py`

### smart_inventory (13 files)
  - `smart_inventory/api_mrp.py`
  - `smart_inventory/api_mrp_complete.py`
  - `smart_inventory/bom_engine.py`
  - `smart_inventory/demand_forecasting.py`
  - `smart_inventory/external_signals.py`
  - `smart_inventory/forecasting_engine.py`
  - `smart_inventory/iot_ingestion.py`
  - `smart_inventory/mrp_complete.py`
  - `smart_inventory/mrp_engine.py`
  - `smart_inventory/multi_warehouse_optimizer.py`
  - ... and 3 more

### tests (2 files)
  - `tests/test_bandits.py`
  - `tests/test_snr.py`

### tools (1 files)
  - `tools/backend_map.py`

### workforce_analytics (3 files)
  - `workforce_analytics/workforce_assignment_model.py`
  - `workforce_analytics/workforce_forecasting.py`
  - `workforce_analytics/workforce_performance_engine.py`

## API Endpoints

### digital_twin
- `POST /product/{revision_id}/analyze-scan` → `api_analyze_scan()` in `digital_twin/api_xai_dt.py`
- `GET /product/{revision_id}/conformance` → `api_get_conformance()` in `digital_twin/api_xai_dt.py`
- `GET /product/{revision_id}/conformance/summary` → `api_get_conformance_summary()` in `digital_twin/api_xai_dt.py`
- `GET /product/{revision_id}/golden-runs` → `api_get_golden_runs()` in `digital_twin/api_xai_dt.py`
- `POST /product/{revision_id}/golden-runs/compute` → `api_compute_golden_runs()` in `digital_twin/api_xai_dt.py`
- `GET /product/{revision_id}/suggest-params` → `api_suggest_params()` in `digital_twin/api_xai_dt.py`

### duplios
- `GET /analytics/carbon` → `api_get_carbon_analytics()` in `duplios/api_duplios.py`
- `GET /analytics/compliance` → `api_get_compliance_analytics()` in `duplios/api_duplios.py`
- `GET /dashboard` → `api_get_dashboard()` in `duplios/api_duplios.py`
- `POST /dpp` → `api_create_dpp()` in `duplios/api_duplios.py`
- `GET /dpp` → `api_list_dpp()` in `duplios/api_duplios.py`
- `GET /dpp-revision/{revision_id}` → `api_get_dpp_by_revision()` in `duplios/api_duplios.py`
- `GET /dpp/by-gtin/{gtin}` → `api_get_dpp_gtin()` in `duplios/api_duplios.py`
- `GET /dpp/{dpp_id}` → `api_get_dpp()` in `duplios/api_duplios.py`
- `PATCH /dpp/{dpp_id}` → `api_update_dpp()` in `duplios/api_duplios.py`
- `DELETE /dpp/{dpp_id}` → `api_delete_dpp()` in `duplios/api_duplios.py`
- `GET /dpp/{dpp_id}/carbon-breakdown` → `api_get_carbon_breakdown()` in `duplios/api_duplios.py`
- `GET /dpp/{dpp_id}/compliance` → `api_get_dpp_compliance()` in `duplios/api_duplios.py`
- `POST /dpp/{dpp_id}/publish` → `api_publish_dpp()` in `duplios/api_duplios.py`
- `GET /dpp/{dpp_id}/qrcode` → `api_get_qrcode()` in `duplios/api_duplios.py`
- `POST /dpp/{dpp_id}/recalculate` → `api_recalculate_dpp()` in `duplios/api_duplios.py`
- `GET /dpp/{dpp_id}/trust-breakdown` → `api_get_trust_breakdown()` in `duplios/api_duplios.py`
- `GET /export/csv` → `api_export_csv()` in `duplios/api_duplios.py`
- `GET /export/json` → `api_export_json()` in `duplios/api_duplios.py`
- `POST /identity/ingest` → `api_ingest_identity()` in `duplios/api_duplios.py`
- `POST /identity/verify` → `api_verify_identity()` in `duplios/api_duplios.py`
- `GET /items` → `api_list_items()` in `duplios/api_duplios.py`
- `POST /items` → `api_create_item()` in `duplios/api_duplios.py`
- `GET /items/by-sku/{sku}` → `api_get_item_by_sku()` in `duplios/api_duplios.py`
- `GET /items/{item_id}` → `api_get_item()` in `duplios/api_duplios.py`
- `GET /items/{item_id}/revisions` → `api_get_revisions()` in `duplios/api_duplios.py`
- `POST /items/{item_id}/revisions` → `api_create_revision()` in `duplios/api_duplios.py`
- `GET /materials` → `api_get_materials()` in `duplios/api_duplios.py`
- `GET /public/dpp/{slug}` → `api_get_public_dpp()` in `duplios/api_duplios.py`
- `GET /revisions/{revision_id}/bom` → `api_get_bom()` in `duplios/api_duplios.py`
- `POST /revisions/{revision_id}/bom` → `api_add_bom_line()` in `duplios/api_duplios.py`
- `GET /revisions/{revision_id}/identities` → `api_get_identities()` in `duplios/api_duplios.py`
- `GET /revisions/{revision_id}/lca` → `api_get_lca()` in `duplios/api_duplios.py`
- `POST /revisions/{revision_id}/lca/recalculate` → `api_recalculate_lca()` in `duplios/api_duplios.py`
- `POST /revisions/{revision_id}/release` → `api_release_revision()` in `duplios/api_duplios.py`
- `GET /revisions/{revision_id}/routing` → `api_get_routing()` in `duplios/api_duplios.py`
- `POST /revisions/{revision_id}/routing` → `api_add_routing_op()` in `duplios/api_duplios.py`
- `GET /view/{slug}` → `api_view_dpp_redirect()` in `duplios/api_duplios.py`

### root
- `GET /actions` → `list_actions()` in `api.py`
- `POST /actions/from-suggestion` → `create_action_from_suggestion_endpoint()` in `api.py`
- `GET /actions/pending/count` → `get_pending_count()` in `api.py`
- `POST /actions/propose` → `propose_new_action()` in `api.py`
- `GET /actions/{action_id}` → `get_action()` in `api.py`
- `POST /actions/{action_id}/approve` → `approve_action_endpoint()` in `api.py`
- `POST /actions/{action_id}/reject` → `reject_action_endpoint()` in `api.py`
- `GET /api/etl/status` → `api_etl_status()` in `api.py`
- `GET /bottleneck` → `get_bottleneck()` in `api.py`
- `GET /bottlenecks/` → `bottlenecks_stub()` in `api.py`
- `GET /causal/complexity` → `get_complexity_metrics()` in `api.py`
- `GET /causal/dashboard` → `get_causal_dashboard()` in `api.py`
- `GET /causal/effect/{treatment}/{outcome}` → `get_causal_effect()` in `api.py`
- `GET /causal/effects/outcome/{outcome}` → `get_all_effects_outcome()` in `api.py`
- `GET /causal/effects/treatment/{treatment}` → `get_all_effects_treatment()` in `api.py`
- `POST /causal/explain` → `explain_causal_question()` in `api.py`
- `GET /causal/graph` → `get_causal_graph()` in `api.py`
- `GET /causal/health` → `get_causal_health()` in `api.py`
- `GET /causal/insights` → `get_causal_insights()` in `api.py`
- `GET /causal/tradeoffs/{treatment}` → `get_tradeoff_analysis()` in `api.py`
- `GET /causal/variables` → `get_causal_variables()` in `api.py`
- `POST /chat` → `chat()` in `api.py`
- `POST /dashboards/capacity-projection` → `get_capacity_projection()` in `api.py`
- `GET /dashboards/cell-performance` → `get_cell_performance()` in `api.py`
- `GET /dashboards/gantt-comparison` → `get_gantt_comparison()` in `api.py`
- `GET /dashboards/machine-oee` → `get_machine_oee_dashboard()` in `api.py`
- `GET /dashboards/operator` → `get_operator_dashboard()` in `api.py`
- `GET /dashboards/summary` → `get_dashboards_summary()` in `api.py`
- `GET /dashboards/utilization-heatmap` → `get_utilization_heatmap()` in `api.py`
- `POST /digital-twin/adjust-plan` → `adjust_plan_with_rul_endpoint()` in `api.py`
- `GET /digital-twin/dashboard` → `get_digital_twin_dashboard()` in `api.py`
- `GET /digital-twin/health` → `get_digital_twin_health()` in `api.py`
- `GET /digital-twin/machine/{machine_id}` → `get_machine_health_detail()` in `api.py`
- `GET /digital-twin/machines` → `get_monitored_machines()` in `api.py`
- `GET /digital-twin/rul-penalties` → `get_rul_penalties_endpoint()` in `api.py`
- `GET /etl/status` → `etl_status()` in `api.py`
- `GET /health` → `health()` in `api.py`
- `GET /insights/generate` → `insights_generate()` in `api.py`
- `GET /inventory/` → `inventory_endpoint()` in `api.py`
- `GET /inventory/forecast/{sku}` → `get_inventory_forecast()` in `api.py`
- `POST /inventory/optimize` → `optimize_inventory()` in `api.py`
- `GET /inventory/rop/{sku}` → `get_inventory_rop()` in `api.py`
- `GET /inventory/stock` → `get_smart_inventory_stock()` in `api.py`
- `GET /inventory/suggestions` → `get_inventory_suggestions()` in `api.py`
- `GET /machines` → `get_machines()` in `api.py`
- `GET /orders` → `get_orders()` in `api.py`
- `GET /plan` → `get_plan()` in `api.py`
- `GET /plan/data_quality` → `get_plan_data_quality()` in `api.py`
- `GET /plan/kpis` → `get_plan_kpis()` in `api.py`
- `GET /plan/milp` → `get_plan_milp()` in `api.py`
- `GET /plan/suggestions` → `get_plan_suggestions()` in `api.py`
- `POST /planning/chained` → `execute_chained_planning()` in `api.py`
- `POST /planning/compare` → `compare_planning_modes()` in `api.py`
- `POST /planning/conventional` → `execute_conventional_planning()` in `api.py`
- `POST /planning/long-term` → `execute_long_term_planning()` in `api.py`
- `GET /planning/modes` → `list_planning_modes()` in `api.py`
- `POST /planning/short-term` → `execute_short_term_planning()` in `api.py`
- `GET /planning/v2/plano` → `planning_v2_plano()` in `api.py`
- `GET /product/classification` → `get_product_classification()` in `api.py`
- `POST /product/delivery-estimate` → `post_delivery_estimate()` in `api.py`
- `GET /product/delivery-estimates` → `get_all_delivery_estimates()` in `api.py`
- `GET /product/summary` → `get_product_summary()` in `api.py`
- `GET /product/type-kpis` → `get_product_type_kpis()` in `api.py`
- `GET /product/{article_id}/kpis` → `get_product_kpis()` in `api.py`
- `GET /projects` → `get_projects()` in `api.py`
- `GET /projects/priority-plan` → `get_project_priority_plan()` in `api.py`
- `POST /projects/recompute` → `recompute_project_plan()` in `api.py`
- `GET /projects/summary` → `get_projects_summary()` in `api.py`
- `GET /projects/{project_id}/kpis` → `get_project_kpis()` in `api.py`
- `GET /reports/algorithms` → `list_available_algorithms()` in `api.py`
- `POST /reports/compare` → `compare_scenarios_endpoint()` in `api.py`
- `POST /reports/compare-whatif` → `compare_whatif_scenario()` in `api.py`
- `GET /reports/current-metrics` → `get_current_metrics()` in `api.py`
- `POST /reports/scenario-summary` → `generate_scenario_summary()` in `api.py`
- `POST /reports/technical-explanation` → `get_technical_explanation()` in `api.py`
- `POST /research/drl-evaluate` → `evaluate_drl_policy()` in `api.py`
- `GET /research/drl-status` → `get_drl_status()` in `api.py`
- `POST /research/drl-train` → `train_drl_policy()` in `api.py`
- `GET /research/experiments` → `list_experiments()` in `api.py`
- `POST /research/explain` → `explain_decision()` in `api.py`
- `GET /research/inventory-simulation` → `run_inventory_simulation()` in `api.py`
- `GET /research/learning-status` → `get_learning_status()` in `api.py`
- `GET /research/plan-experimental` → `get_experimental_plan()` in `api.py`
- `POST /research/run-experiment` → `run_experiment()` in `api.py`
- `GET /shopfloor/machines` → `api_get_shopfloor_machines()` in `api.py`
- `GET /shopfloor/orders` → `api_get_shopfloor_orders()` in `api.py`
- `POST /shopfloor/orders/{order_id}/complete` → `api_complete_order()` in `api.py`
- `POST /shopfloor/orders/{order_id}/pause` → `api_pause_order()` in `api.py`
- `POST /shopfloor/orders/{order_id}/report` → `api_report_order_execution()` in `api.py`
- `POST /shopfloor/orders/{order_id}/start` → `api_start_order()` in `api.py`
- `GET /suggestions/` → `suggestions_endpoint()` in `api.py`
- `GET /system/model-sources` → `api_get_model_sources()` in `api.py`
- `POST /what-if/compare` → `compare_scenario()` in `api.py`
- `POST /what-if/describe` → `describe_scenario()` in `api.py`
- `GET /work-instructions/{revision_id}/{operation_id}` → `api_get_work_instructions()` in `api.py`
- `POST /work-instructions/{revision_id}/{operation_id}` → `api_create_work_instructions()` in `api.py`
- `POST /workforce/assign` → `post_workforce_assignment()` in `api.py`
- `POST /workforce/forecast` → `post_workforce_forecast()` in `api.py`
- `GET /workforce/performance` → `get_workforce_performance()` in `api.py`
- `GET /workforce/summary` → `get_workforce_summary()` in `api.py`
- `GET /workforce/{worker_id}/performance` → `get_worker_performance()` in `api.py`
- `GET /zdm/dashboard` → `get_zdm_dashboard()` in `api.py`
- `GET /zdm/health` → `get_zdm_health()` in `api.py`
- `GET /zdm/quick-check` → `zdm_quick_check()` in `api.py`
- `GET /zdm/recovery/{scenario_id}` → `get_zdm_recovery_plan()` in `api.py`
- `GET /zdm/scenarios` → `get_zdm_scenarios()` in `api.py`
- `POST /zdm/simulate` → `zdm_simulate()` in `api.py`

## Key Classes

### Engines & Estimators
- `AdvancedForecastEngine` in `smart_inventory/forecasting_engine.py` (bases: ForecastEngineBase)
- `BOMEngine` in `smart_inventory/bom_engine.py` (bases: none)
- `BaseRulEstimator` in `digital_twin/rul_estimator.py` (bases: RulEstimatorBase)
- `BayesianRULEstimator` in `ml/rul_models.py` (bases: RULEstimator)
- `BomValidationEngine` in `duplios/pdm_core.py` (bases: none)
- `CapacityModelEngine` in `optimization/math_optimization.py` (bases: none)
- `CausalEffectEstimator` in `causal/causal_effect_estimator.py` (bases: none)
- `CausalEngine` in `feature_flags.py` (bases: str, Enum)
- `CausalEstimatorBase` in `causal/causal_effect_estimator.py` (bases: ABC)
- `CevaeEstimator` in `rd/causal_deep_experiments.py` (bases: none)
- `ChatEngine` in `chat/engine.py` (bases: none)
- `ClassicalForecastEngine` in `smart_inventory/forecasting_engine.py` (bases: ForecastEngineBase)
- `DeepSurvRulEstimator` in `digital_twin/rul_estimator.py` (bases: RulEstimatorBase)
- `DeviationEngine` in `feature_flags.py` (bases: str, Enum)
- `DeviationEngineBase` in `digital_twin/xai_dt_geometry.py` (bases: ABC)
- `DmlCausalEstimator` in `causal/causal_effect_estimator.py` (bases: CausalEstimatorBase)
- `DragonnetEstimator` in `rd/causal_deep_experiments.py` (bases: none)
- `ECREngine` in `duplios/pdm_core.py` (bases: none)
- `ExplainabilityEngine` in `core/explainability/explainability_engine.py` (bases: none)
- `ExplainabilityEngine` in `digital_twin/shi_dt.py` (bases: none)
- `ExplainabilityEngine` in `research/explainability_engine.py` (bases: none)
- `ForecastEngine` in `feature_flags.py` (bases: str, Enum)
- `ForecastEngineBase` in `smart_inventory/forecasting_engine.py` (bases: ABC)
- `GoldenRunsEngine` in `optimization/math_optimization.py` (bases: none)
- `InventoryEngine` in `inventory/inventory_engine.py` (bases: none)
- `InventoryPolicyEngine` in `feature_flags.py` (bases: str, Enum)
- `KPIEngine` in `evaluation/kpi_engine.py` (bases: none)
- `MLRULEstimator` in `ml/rul_models.py` (bases: RULEstimator)
- `MRPCompleteEngine` in `smart_inventory/mrp_complete.py` (bases: none)
- `MRPEngine` in `smart_inventory/mrp_engine.py` (bases: none)
- `MRPFromOrdersEngine` in `smart_inventory/mrp_engine.py` (bases: none)
- `OlsCausalEstimator` in `causal/causal_effect_estimator.py` (bases: CausalEstimatorBase)
- `PDMGuardEngine` in `quality/prevention_guard.py` (bases: none)
- `PlanningEngine` in `planning/planning_engine.py` (bases: none)
- `PodDeviationEngine` in `digital_twin/xai_dt_geometry.py` (bases: DeviationEngineBase)
- `PredictiveGuardEngine` in `quality/prevention_guard.py` (bases: none)
- `RULEstimator` in `digital_twin/rul_estimator.py` (bases: BaseRulEstimator)
- `RULEstimator` in `ml/rul_models.py` (bases: ABC)
- `RULEstimatorConfig` in `digital_twin/rul_estimator.py` (bases: none)
- `ReleaseValidationEngine` in `duplios/pdm_core.py` (bases: none)
- `RevisionComparisonEngine` in `duplios/pdm_core.py` (bases: none)
- `RevisionWorkflowEngine` in `duplios/pdm_core.py` (bases: none)
- `RoutingEngine` in `research/routing_engine.py` (bases: none)
- `RoutingValidationEngine` in `duplios/pdm_core.py` (bases: none)
- `RulEngine` in `feature_flags.py` (bases: str, Enum)
- `RulEstimatorBase` in `digital_twin/rul_estimator.py` (bases: ABC)
- `SchedulerEngine` in `feature_flags.py` (bases: str, Enum)
- `SchedulerEngine` in `scheduling/types.py` (bases: str, Enum)
- `SetupEngine` in `core/setup_engine.py` (bases: none)
- `SetupEngine` in `research/setup_engine.py` (bases: none)
- `ShopfloorGuardEngine` in `quality/prevention_guard.py` (bases: none)
- `SimpleDeviationEngine` in `digital_twin/xai_dt_geometry.py` (bases: DeviationEngineBase)
- `StatisticalRULEstimator` in `ml/rul_models.py` (bases: RULEstimator)
- `TarnetEstimator` in `rd/causal_deep_experiments.py` (bases: none)
- `TestCausalEffectEstimator` in `causal/tests/test_causal.py` (bases: none)
- `TestRULEstimator` in `digital_twin/tests/test_rul_integration.py` (bases: none)
- `TestRecoveryStrategyEngine` in `simulation/tests/test_zdm_simulator.py` (bases: none)
- `TimePredictionEngineBase` in `optimization/math_optimization.py` (bases: none)
- `TimePredictionEngineML` in `optimization/math_optimization.py` (bases: TimePredictionEngineBase)
- `WorkInstructionExecutionEngine` in `shopfloor/work_instructions.py` (bases: none)
- `XAIEngine` in `feature_flags.py` (bases: str, Enum)

### Services
- `ComplianceRadarService` in `duplios/compliance_radar.py`
- `GapFillingLiteService` in `duplios/gap_filling_lite.py`
- `MRPService` in `smart_inventory/mrp_complete.py`
- `MathOptimizationService` in `optimization/math_optimization.py`
- `OpsIngestionService` in `ops_ingestion/services.py`
- `PDMService` in `duplios/pdm_core.py`
- `PreventionGuardService` in `quality/prevention_guard.py`
- `TrustIndexService` in `duplios/trust_index_service.py`
- `WorkInstructionService` in `prodplan/work_instructions.py`
- `WorkInstructionService` in `shopfloor/work_instructions.py`

### Models (Pydantic/SQLAlchemy)
- `ApproveActionRequest` in `api.py`
- `CapacityProjectionRequest` in `api.py`
- `CausalExplainRequest` in `api.py`
- `CellConfig` in `api.py`
- `ChainedPlanningRequest` in `api.py`
- `ChatQuery` in `api.py`
- `ChatRequest` in `chat/engine.py`
- `ChatResponse` in `chat/engine.py`
- `CompareRequest` in `api.py`
- `ConventionalPlanningRequest` in `api.py`
- `DRLTrainRequest` in `api.py`
- `DeliveryEstimateRequest` in `api.py`
- `ExperimentRequest` in `api.py`
- `ExplainRequest` in `api.py`
- `HealthIndexResponse` in `digital_twin/api_shi_dt.py`
- `KpiPayload` in `chat/engine.py`
- `LongTermPlanningRequest` in `api.py`
- `PlanningComparisonRequest` in `api.py`
- `ProjectPriorityRequest` in `api.py`
- `ProposeActionRequest` in `api.py`
- `RejectActionRequest` in `api.py`
- `ScenarioQuery` in `api.py`
- `ScenarioSummaryRequest` in `api.py`
- `SensorDataRequest` in `digital_twin/api_shi_dt.py`
- `ShopfloorStartRequest` in `api.py`
- `ShortTermPlanningRequest` in `api.py`
- `TechnicalExplainRequest` in `api.py`
- `WorkforceAssignRequest` in `api.py`
- `WorkforceForecastRequest` in `api.py`
- `ZDMSimulateRequest` in `api.py`
- ... and 184 more models

## R&D Work Packages

### WP Files
- `causal/complexity_dashboard_engine.py`
- `dashboards.py`
- `dashboards/capacity_projection.py`
- `dashboards/cell_performance.py`
- `dashboards/gantt_comparison.py`
- `dashboards/machine_oee.py`
- `dashboards/operator_dashboard.py`
- `dashboards/utilization_heatmap.py`
- `quality/api_prevention_guard.py`
- `quality/prevention_guard.py`
- `rd/api.py`
- `rd/causal_deep_experiments.py`
- `rd/experiments_core.py`
- `rd/reporting.py`
- `rd/wp1_routing_experiments.py`
- `rd/wp2_suggestions_eval.py`
- `rd/wp3_inventory_capacity.py`
- `rd/wp4_learning_scheduler.py`

### WP/Experiment Classes
- `CausalDeepExperiment` in `rd/causal_deep_experiments.py`
- `ExperimentConfig` in `experiments/experiment_runner.py`
- `ExperimentContext` in `models_common.py`
- `ExperimentCreatedResponse` in `rd/api.py`
- `ExperimentLog` in `research/experiment_logger.py`
- `ExperimentLogger` in `rd/experiments_core.py`
- `ExperimentLogger` in `research/experiment_logger.py`
- `ExperimentRequest` in `api.py`
- `ExperimentResult` in `experiments/experiment_runner.py`
- `ExperimentRunner` in `experiments/experiment_runner.py`
- `ExperimentStatus` in `models_common.py`
- `ExperimentStatus` in `rd/causal_deep_experiments.py`
- `ExperimentStatus` in `rd/experiments_core.py`
- `ExperimentSummary` in `models_common.py`
- `RDExperiment` in `rd/experiments_core.py`
- `RDExperimentCreate` in `rd/experiments_core.py`
- `RDExperimentUpdate` in `rd/experiments_core.py`
- `TestSIFIDEExperiments` in `optimization/tests/test_drl_scheduler.py`
- `WP1PolicyResult` in `rd/wp1_routing_experiments.py`
- `WP1RoutingExperiment` in `rd/wp1_routing_experiments.py`
- `WP1RoutingRequest` in `rd/wp1_routing_experiments.py`
- `WP1Summary` in `rd/reporting.py`
- `WP2BatchEvaluationRequest` in `rd/wp2_suggestions_eval.py`
- `WP2BatchEvaluationResult` in `rd/wp2_suggestions_eval.py`
- `WP2EvaluationRequest` in `rd/wp2_suggestions_eval.py`
- `WP2EvaluationResult` in `rd/wp2_suggestions_eval.py`
- `WP2Summary` in `rd/reporting.py`
- `WP3ComparisonRequest` in `rd/wp3_inventory_capacity.py`
- `WP3Experiment` in `rd/wp3_inventory_capacity.py`
- `WP3ScenarioRequest` in `rd/wp3_inventory_capacity.py`
- `WP3ScenarioResult` in `rd/wp3_inventory_capacity.py`
- `WP3Summary` in `rd/reporting.py`
- `WP4ExperimentResult` in `rd/wp4_learning_scheduler.py`
- `WP4RunRequest` in `rd/wp4_learning_scheduler.py`
- `WP4Summary` in `rd/reporting.py`

## Feature Coverage Check

- ✅ **Scheduling/APS**: 15 files, 42 classes, 30 functions
- ✅ **SmartInventory**: 26 files, 69 classes, 102 functions
- ✅ **Duplios/PDM**: 40 files, 23 classes, 54 functions
- ✅ **Digital Twin**: 22 files, 48 classes, 56 functions
- ✅ **Prevention Guard**: 4 files, 25 classes, 6 functions
- ✅ **R&D**: 26 files, 66 classes, 96 functions
- ✅ **Ops Ingestion**: 14 files, 3 classes, 23 functions
- ✅ **Work Instructions**: 5 files, 5 classes, 11 functions
- ✅ **Causal/Intelligence**: 22 files, 24 classes, 29 functions

## Potential Dead Code (needs manual verification)

- `clear_all()` in `actions_engine.py` (line 223)
- `get_action_history()` in `actions_engine.py` (line 611)
- `create_action_from_command()` in `actions_engine.py` (line 661)
- `to_causal_estimate()` in `causal/causal_effect_estimator.py` (line 209)
- `to_adjacency_matrix()` in `causal/causal_graph_builder.py` (line 153)
- `build_causal_dataset()` in `causal/data_collector.py` (line 78)
- `get_dataset_summary()` in `causal/data_collector.py` (line 329)
- `prepare_for_estimator()` in `causal/data_collector.py` (line 343)
- `list_available_treatments()` in `causal/data_collector.py` (line 376)
- `list_available_outcomes()` in `causal/data_collector.py` (line 381)
- `test_add_variable()` in `causal/tests/test_causal.py` (line 44)
- `test_add_relation()` in `causal/tests/test_causal.py` (line 58)
- `test_domain_knowledge()` in `causal/tests/test_causal.py` (line 68)
- `test_build_graph()` in `causal/tests/test_causal.py` (line 86)
- `test_learn_causal_graph()` in `causal/tests/test_causal.py` (line 97)
- `test_graph_navigation()` in `causal/tests/test_causal.py` (line 105)
- `test_estimate_effect_basic()` in `causal/tests/test_causal.py` (line 122)
- `test_effect_interpretation()` in `causal/tests/test_causal.py` (line 134)
- `test_confounder_identification()` in `causal/tests/test_causal.py` (line 146)
- `test_estimate_intervention()` in `causal/tests/test_causal.py` (line 157)
- `test_all_effects_for_outcome()` in `causal/tests/test_causal.py` (line 169)
- `test_all_effects_from_treatment()` in `causal/tests/test_causal.py` (line 180)
- `test_compute_metrics()` in `causal/tests/test_causal.py` (line 192)
- `test_generate_insights()` in `causal/tests/test_causal.py` (line 202)
- `test_insight_types()` in `causal/tests/test_causal.py` (line 211)
- `test_tradeoff_analysis()` in `causal/tests/test_causal.py` (line 220)
- `test_dashboard_to_dict()` in `causal/tests/test_causal.py` (line 230)
- `test_generate_data()` in `causal/tests/test_causal.py` (line 245)
- `test_causal_relationships_in_data()` in `causal/tests/test_causal.py` (line 260)
- `test_full_causal_workflow()` in `causal/tests/test_causal.py` (line 280)
- `test_causal_explanation()` in `causal/tests/test_causal.py` (line 307)
- `scheduler_skill()` in `chat/engine.py` (line 63)
- `inventory_skill()` in `chat/engine.py` (line 104)
- `duplios_skill()` in `chat/engine.py` (line 140)
- `digital_twin_skill()` in `chat/engine.py` (line 175)
- `rd_skill()` in `chat/engine.py` (line 211)
- `causal_skill()` in `chat/engine.py` (line 248)
- `greeting_skill()` in `chat/engine.py` (line 283)
- `general_skill()` in `chat/engine.py` (line 311)
- `get_session_context()` in `chat/engine.py` (line 421)
- `clear_session()` in `chat/engine.py` (line 425)
- `get_intent_description()` in `chat/router.py` (line 154)
- `explain_schedule_timing()` in `core/explainability/explainability_engine.py` (line 420)
- `explain_schedule_decision()` in `core/explainability/explainability_engine.py` (line 580)
- `load_historical()` in `core/setup_engine.py` (line 218)
- `estimate_setup_savings()` in `core/setup_engine.py` (line 493)
- `compute_sequence_setup()` in `core/setup_engine.py` (line 557)
- `build_gantt_comparison()` in `dashboards.py` (line 14)
- `build_heatmap_machine_load()` in `dashboards.py` (line 35)
- `build_annual_projection()` in `dashboards.py` (line 49)
- ... and 453 more

---
*Report generated by backend_map.py - Contract 15*