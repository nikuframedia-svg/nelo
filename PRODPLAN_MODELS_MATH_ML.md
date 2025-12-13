# ProdPlan 4.0 - Modelos Matemáticos e Machine Learning

## 📋 Resumo Executivo

O ProdPlan 4.0 utiliza uma combinação de:
- **Otimização Matemática**: MILP, CP-SAT, heurísticas
- **Machine Learning**: Forecasting, Deep Learning (CVAE), Reinforcement Learning (stubs)
- **Estatística Bayesiana**: Estimação de incerteza, Monte Carlo
- **Inferência Causal**: Grafos causais, estimação de efeitos

---

## 🏭 1. SCHEDULING (Planeamento de Produção)

### 1.1 Heurísticas de Dispatching
**Ficheiro:** `backend/scheduler.py`

```
Regras implementadas:
- FIFO: First In, First Out
- SPT: Shortest Processing Time
- EDD: Earliest Due Date (via priority + due_date sorting)

Algoritmo:
1. Ordenar ordens por (priority DESC, due_date ASC)
2. Para cada operação:
   start_time = max(machine_free, order_last_finish, prev_op_finish)
   end_time = start_time + duration
3. Atualizar machine_next_free e order_last_finish
```

### 1.2 MILP (Mixed-Integer Linear Programming)
**Ficheiro:** `backend/optimization/scheduling_models.py`

**Formulação Job-Shop:**
```
Variáveis:
- x[op,m] ∈ {0,1}: operação op atribuída à máquina m
- s[op] ∈ ℝ⁺: tempo de início da operação
- e[op] ∈ ℝ⁺: tempo de fim da operação
- C_max ∈ ℝ⁺: makespan

Restrições:
1. Atribuição única: Σ_m x[op,m] = 1, ∀ op
2. Tempo de processamento: e[op] ≥ s[op] + p[op,m] - M(1 - x[op,m])
3. Precedência: s[succ] ≥ e[pred], ∀ (pred,succ) ∈ Prec
4. Não-sobreposição (Big-M):
   e[op1] ≤ s[op2] + M(1-y) + M(2-x[op1,m]-x[op2,m])
   e[op2] ≤ s[op1] + My + M(2-x[op1,m]-x[op2,m])
5. Makespan: C_max ≥ e[op], ∀ op

Objetivo: min C_max (ou soma ponderada)
Solver: OR-Tools CBC/SCIP
```

### 1.3 CP-SAT (Constraint Programming)
**Ficheiro:** `backend/optimization/scheduling_models.py`

**Formulação:**
```
Variáveis:
- interval[op]: variável de intervalo (start, duration, end)
- machine[op]: índice da máquina atribuída
- optional_interval[op,m]: intervalo opcional por máquina

Restrições Globais:
- NoOverlap(intervals_on_machine): disjunção de intervalos
- Cumulative: para máquinas com capacidade > 1
- AddExactlyOne(presence_vars): exatamente uma máquina

Propagação: Constraint propagation + SAT learning
Solver: OR-Tools CP-SAT
```

### 1.4 DRL (Deep Reinforcement Learning) - Stub
**Ficheiro:** `backend/optimization/drl_scheduler.py`

```
MDP Formulation:
- State: (machine_states, operation_states, global_time)
- Action: operation_index to dispatch
- Reward: -tardiness - makespan_factor - setup_penalty + flow_bonus

Algoritmo: PPO/A2C/DQN (Stable-Baselines3)
Status: Stub para R&D
```

---

## 📈 2. FORECASTING (Previsão de Demanda)

**Ficheiro:** `backend/ml/forecasting.py`

### 2.1 Naive Forecaster
```
ŷ_t+h = y_t (último valor observado)
Intervalo: ŷ ± z * σ (σ = desvio padrão histórico)
```

### 2.2 Moving Average
```
ŷ_t+h = (1/k) Σ_{i=0}^{k-1} y_{t-i}
Intervalo: MA ± z * σ_window
k = window_size (default 7)
```

### 2.3 ETS (Exponential Smoothing)
```
Holt's Linear Method:
- Nível: L_t = α * y_t + (1-α) * (L_{t-1} + T_{t-1})
- Tendência: T_t = β * (L_t - L_{t-1}) + (1-β) * T_{t-1}
- Previsão: ŷ_{t+h} = L_t + h * T_t

Parâmetros: α=0.3, β=0.1
Biblioteca: statsmodels.tsa.holtwinters
```

### 2.4 ARIMA
```
ARIMA(p,d,q):
- AR(p): y_t = c + Σ_{i=1}^{p} φ_i * y_{t-i} + ε_t
- I(d): diferenciação d vezes
- MA(q): ε_t = Σ_{j=1}^{q} θ_j * ε_{t-j}

Default: ARIMA(1,1,1)
Intervalos: get_forecast().conf_int(alpha=0.05)
Biblioteca: statsmodels.tsa.arima.model
```

### 2.5 XGBoost
```
Features: Lag features (últimos n_lags valores)
Modelo: XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
Predição recursiva: usar predições anteriores como input
```

### 2.6 Transformer (Stub)
```
Arquitetura planejada:
- Temporal Fusion Transformer (TFT)
- Non-Stationary Transformers
Fallback atual: ETS
```

---

## 🤖 3. DIGITAL TWIN (Health Indicators + RUL)

### 3.1 CVAE (Conditional Variational Autoencoder)
**Ficheiro:** `backend/digital_twin/health_indicator_cvae.py`

**Arquitetura PyTorch:**
```
Input: x ∈ ℝ^64 (sensor features) + context (machine, op, product embeddings)

Encoder:
- x_concat = [x; emb_machine; emb_op; emb_product] ∈ ℝ^88
- h = LeakyReLU(BatchNorm(Linear(128))) → ... → ℝ^32
- μ = Linear(32 → 16)
- log σ² = Linear(32 → 16)

Reparametrization:
- ε ~ N(0, I)
- z = μ + σ * ε

Decoder:
- z_concat = [z; emb_machine; emb_op; emb_product] ∈ ℝ^40
- h = LeakyReLU(BatchNorm(Linear(...)))
- x̂ = Linear(h → 64)

Health Indicator Head:
- HI = Sigmoid(Linear(ReLU(Linear(z))))
- HI ∈ [0, 1]: 1 = saudável, 0 = crítico

Loss Function:
- L = L_recon + β * KL(q(z|x,c) || p(z)) + L_HI
- L_recon = MSE(x, x̂)
- KL = -0.5 * Σ(1 + log σ² - μ² - σ²)
```

### 3.2 RUL Estimator (Remaining Useful Life)
**Ficheiro:** `backend/digital_twin/rul_estimator.py`

**Modelo Exponencial:**
```
Degradação: HI(t) = HI_0 * exp(-λt)

Fitting (Least Squares):
- log(HI) = log(HI_0) - λt
- λ = -slope (regressão linear em log-scale)

RUL: T_RUL = ln(HI_current / HI_fail) / λ

Incerteza:
- σ_RUL ≈ RMSE / (λ * HI) * T_RUL * 0.3
- IC 95%: [T_RUL - 1.96*σ, T_RUL + 1.96*σ]
```

**Modelo Linear:**
```
Degradação: HI(t) = HI_0 - λt
RUL: T_RUL = (HI_current - HI_fail) / λ
```

**Monte Carlo (Gaussian Process Approximation):**
```
Para N=1000 amostras:
1. Perturbar λ: λ' = λ * N(1, 0.2)
2. Perturbar HI: HI' = HI + N(0, 0.05)
3. Calcular RUL': ln(HI' / HI_fail) / λ'
4. Estatísticas: mean, std, percentis 2.5% e 97.5%
```

### 3.3 RUL-Integrated Scheduling
**Ficheiro:** `backend/digital_twin/rul_integration_scheduler.py`

```
Penalização de Máquinas:
- Se RUL < threshold: penalty = 1 - (RUL / threshold)
- Penalty range: [0, 1]

Ajuste do Plano:
1. Calcular HI e RUL para cada máquina
2. Máquinas com status CRITICAL ou WARNING:
   - Redistribuir operações para máquinas alternativas
   - Sugerir manutenção preventiva
   - Reduzir carga em operações longas
3. Gerar RULAdjustedPlan com decisões documentadas
```

---

## 🛡️ 4. ZDM (Zero Disruption Manufacturing)

### 4.1 Geração de Cenários de Falha
**Ficheiro:** `backend/simulation/zdm/failure_scenario_generator.py`

```
Tipos de Falha:
- SUDDEN: P(failure) ~ Exponential(λ_RUL)
- GRADUAL: degradation_rate = 1 - RUL_normalized
- QUALITY: reject_rate ~ Beta(α, β)
- MATERIAL: P(shortage) baseado em ROP
- OPERATOR: P(absence) ~ histórico

Parâmetros por Tipo:
- duration_hours ~ LogNormal(μ, σ)
- severity ∈ [0, 1]
- quality_reject_rate ∈ [0, 0.5]
```

### 4.2 Simulação de Resiliência
**Ficheiro:** `backend/simulation/zdm/zdm_simulator.py`

```
Métricas de Impacto:
- downtime_hours: duração da paragem
- operations_delayed: # operações afetadas
- throughput_loss_pct = (ops_delayed / total_ops) * 100 * severity
- otd_impact_pct = (orders_at_risk / orders_impacted) * 100
- estimated_cost = downtime * €500/h + orders_at_risk * €200

Severity Score:
score = 0.3 * time_score + 0.3 * throughput_score + 0.4 * otd_score

Resilience Score = 100 - avg(severity_scores)
```

### 4.3 Estratégias de Recuperação
**Ficheiro:** `backend/simulation/zdm/recovery_strategy_engine.py`

```
Estratégias:
1. REROUTE: reencaminhar para máquinas alternativas
   - recovery_factor += 0.4 * rerouting_efficiency
2. ADD_OVERTIME: adicionar horas extra
   - recovery_factor += 0.3 * min(1, max_overtime / duration)
3. PRIORITY_SHUFFLE: repriorizar ordens VIP
   - recovery_factor += 0.2

Recovery Status:
- SUCCESS: recovery_factor >= 0.8
- PARTIAL: 0.4 <= recovery_factor < 0.8
- FAILED: recovery_factor < 0.4
```

---

## 📦 5. SMART INVENTORY (ROP Dinâmico)

### 5.1 ROP Clássico
**Ficheiro:** `backend/smart_inventory/rop_engine.py`

```
Fórmula:
ROP = μ_d * L + z * σ_d * √L

onde:
- μ_d = consumo médio diário (do forecast)
- σ_d = desvio padrão do consumo
- L = lead time (dias)
- z = quantil do nível de serviço (z_0.95 = 1.96)

Safety Stock:
SS = z * σ_d * √L

ROP Dinâmico (com sazonalidade):
ROP = μ_d(t) * L + z * σ_d(t) * √L + seasonal_adjustment(t)
```

### 5.2 Risco de Ruptura (Monte Carlo)
```
Simulação:
1. Gerar N=10000 amostras de consumo diário
   D_i ~ N(μ_d, σ_d), truncado em 0
2. Stock após 30 dias: S_30 = S_0 - Σ_{j=1}^{30} D_j
3. P(ruptura) = #{S_30 < 0} / N

Aproximação Analítica:
- E[consumo_30d] = μ_d * 30
- Var[consumo_30d] = σ_d² * 30
- P(ruptura) = Φ((0 - (S_0 - μ*30)) / (σ*√30))
```

---

## 🔗 6. CAUSAL ANALYSIS (Inferência Causal)

### 6.1 Grafo Causal
**Ficheiro:** `backend/causal/causal_graph_builder.py`

```
Estrutura DAG (Directed Acyclic Graph):
- Nós: variáveis (treatments, outcomes, confounders)
- Arestas: relações causais com strength e confidence

Variáveis de Tratamento:
- setup_frequency, batch_size, machine_load
- night_shifts, overtime_hours, maintenance_delay

Outcomes:
- energy_cost, makespan, tardiness, otd_rate
- machine_wear, failure_prob, operator_stress

Confounders:
- demand_volume, product_mix, seasonality
- machine_age, workforce_experience
```

### 6.2 Estimação de Efeitos Causais
**Ficheiro:** `backend/causal/causal_effect_estimator.py`

**Regression Adjustment:**
```
E[Y|do(T)] ≈ E[Y|T, Z]

Modelo OLS:
Y = β_0 + β_1*T + β_2*Z_1 + ... + β_n*Z_n + ε

Efeito Causal (ATE) = β_1

Estimação:
β = (X'X)^{-1} X'y  (least squares)
Var(β) = σ² * (X'X)^{-1}
σ² = ||residuals||² / (n - k)

Intervalo de Confiança:
β_1 ± t_{0.975, n-k} * SE(β_1)

P-value:
t = β_1 / SE(β_1)
p = 2 * (1 - CDF_t(|t|, n-k))
```

**Identificação de Confounders (Backdoor Criterion):**
```
Algoritmo:
1. Ancestrais do treatment: A_T
2. Pais do outcome (excluindo treatment): P_Y
3. Confounders = {v : v.type == CONFOUNDER ∧ (v ∈ A_T ∨ v ∈ Ancestrais(outcome))}
```

### 6.3 Complexity Dashboard
**Ficheiro:** `backend/causal/complexity_dashboard_engine.py`

```
Métricas de Complexidade:
- n_variables: número de variáveis no grafo
- n_relations: número de relações causais
- connectivity: n_relations / (n_variables * (n_variables - 1))
- avg_path_length: comprimento médio de caminhos causais
- complexity_score = f(n_vars, n_rels, connectivity, path_length)

Identificação de Trade-offs:
Para cada treatment:
  effects = estimate_all_effects(treatment)
  positive = [e for e in effects if e.direction == "positive"]
  negative = [e for e in effects if e.direction == "negative"]
  if positive and negative:
    trade_off = (treatment, positive, negative)

Leverage Points:
Variables com alto impacto em múltiplos outcomes positivos
```

---

## 📊 7. MÉTRICAS E KPIs

### 7.1 Métricas de Scheduling
```
Makespan: C_max = max(end_time[op]) - min(start_time[op])
Tardiness: T = Σ max(0, end_time[order] - due_date[order])
OTD Rate: % de ordens entregues a tempo
Utilização: Σ duration[m] / (C_max * n_machines) * 100%
Setup Time: Σ setup_time (quando op_code muda)
```

### 7.2 Métricas de Forecasting
```
MAPE: (1/n) Σ |y - ŷ| / |y| * 100
RMSE: √((1/n) Σ (y - ŷ)²)
MAE: (1/n) Σ |y - ŷ|
SNR (Signal-to-Noise): μ / σ (forecastability)
```

### 7.3 Métricas de Digital Twin
```
Health Index: HI ∈ [0, 1]
RUL: horas até falha (com IC 95%)
Status: HEALTHY (HI > 0.7), DEGRADED (0.5-0.7), WARNING (0.3-0.5), CRITICAL (< 0.3)
```

### 7.4 Métricas de Resiliência (ZDM)
```
Resilience Score: 100 - avg(severity_scores)
Recovery Rate: #SUCCESS / #scenarios * 100%
Avg Recovery Time: média de horas para recuperar
Avg Throughput Loss: média de % de perda
```

---

## 🧪 8. ALGORITMOS R&D (Stubs/TODO)

### 8.1 Deep Bayesian RUL
```
TODO: MC Dropout, HMC/VI para incerteza epistémica
Objetivo: Separar incerteza aleatória vs epistémica
```

### 8.2 Transformer Forecasting
```
TODO: Temporal Fusion Transformer
Objetivo: Superar ARIMA em séries não-estacionárias
```

### 8.3 MILP vs CP-SAT Benchmarks
```
TODO: Comparar solution quality, time, optimality gap
Hipótese: MILP melhor para < 100 ops, CP-SAT para > 100
```

### 8.4 DRL Scheduling
```
TODO: PPO com reward shaping
Objetivo: Generalizar para diferentes instâncias
```

### 8.5 DoWhy/EconML Integration
```
TODO: Double ML para CATE
Objetivo: Efeitos heterogéneos por contexto
```

---

## 📚 Referências

1. Pinedo, M. (2016). *Scheduling: Theory, Algorithms, and Systems*
2. Hyndman & Athanasopoulos (2021). *Forecasting: Principles and Practice*
3. Kingma & Welling (2014). *Auto-Encoding Variational Bayes*
4. Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*
5. Wu et al. (2022). *Non-stationary Transformers*
6. Google OR-Tools Documentation



