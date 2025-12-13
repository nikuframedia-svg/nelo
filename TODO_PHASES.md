TODO LIST (MVP – Phase 1):

[x] Create the basic project structure: /data and /backend.
[x] Implement data_loader.py:
    - Load the Excel file (production_os_data_MVP.xlsx) using pandas and openpyxl.
    - Create a DataBundle object (dict or dataclass) with DataFrames for each sheet.
    - Add caching so the Excel is only loaded once per process.
[x] Implement scheduler.py:
    - Build a simple APS scheduler:
      - Short-term planning only.
      - Use route_label "A" per article for now.
      - Sequence operations per order and per machine, avoiding resource overlap.
      - Compute start_time, end_time, duration_min for each operation.
      - Identify bottleneck machine (highest total duration).
    - Export the schedule as /data/production_plan.csv.
[x] Implement openai_client.py:
    - Read OPENAI_API_KEY from .env.
    - Implement a function ask_openai(system_prompt, user_prompt) -> str using gpt-4o-mini.
[x] Implement qa_engine.py:
    - Implement answer_question(message: str) -> str.
    - It should:
      - Load DataBundle (via data_loader).
      - For the question:
        - If it asks about “percurso”/“rota ART-XXX”: build a context string using routing + article_routes and call OpenAI to explain the routes in pt-PT.
        - If it asks about “gargalo”: read production_plan.csv, compute the bottleneck machine (highest total duration) and call OpenAI to explain why it is the bottleneck and suggest actions.
        - Else: call OpenAI with a short context (num orders, num machines, etc.) and answer generically.
[x] Implement api.py:
    - FastAPI app with endpoints:
      - POST /chat { "message": str } -> { "answer": str }
      - GET /plan -> JSON list of the current production plan (read production_plan.csv).
      - GET /bottleneck -> { "machine_id": str, "total_minutes": float } based on production_plan.csv.
      - (optional) GET /orders and GET /machines for debugging.
[x] Add a simple CLI entry point (if __name__ == "__main__") in scheduler.py to generate the plan.
[x] Add a simple CLI entry in qa_engine.py to test answer_question() from the terminal.


TODO LIST (Future – Phase 2, mark only with TODO comments):

[ ] Add “encadeado” (chained) planning mode (PLANEAMENTO ENCADEADO).
[ ] Add What-If scenario engine:
    - Natural language → scenario delta (new machine, new times, new shifts). *(MVP já gera ScenarioDelta via `/what-if/describe`).*
    - Re-run APS on baseline vs scenario and compare. *(Endpoint `/what-if/compare` já devolve métricas básicas – falta enriquecer com dashboards/relatórios.)*
[ ] Add ML predictions (local) for:
    - load forecasts
    - lead time
    - throughput
[ ] Add advanced dashboards (Gantt, heatmaps, before/after, annual projections).
[ ] Add ERP/MES integration via SQL Server / APIs (only as interfaces / stubs for now).

---

## Roadmap – Phase 2 (APS Inteligente On-Prem)

1. **PLANEAMENTO_ENCADEADO**
   - Introduzir modos `NORMAL` vs `ENCADEADO` no scheduler.
   - Modelar cadeias de máquinas/recursos com buffers, tempos de transporte e coerência upstream/downstream.
   - Expor comandos no QA/LLM para ativar/desativar encadeamentos e explicar diferenças.

2. **WHAT_IF_SCENARIO_ENGINE**
   - Motor para converter linguagem natural em `ScenarioDelta` estruturado (novas máquinas, tempos, turnos, routing).
   - Aplicar deltas ao `DataBundle`, recalcular plano baseline vs cenário e comparar métricas (throughput, carga, atrasos, OEE).
   - Endpoints dedicados `/what-if/*` para descrição e comparação, mais relatórios executivos.
   - *Estado atual:* `/what-if/describe` e `/what-if/compare` implementados no MVP.

3. **OFFLINE_ML_ENGINE**
   - Treinos offline (H2O AutoML / scikit-learn / XGBoost) para carga, lead time, throughput, anomalias.
   - Publicar `predict_*` APIs para alimentar scheduler (ex.: tempos previstos) e dashboards/projeções.

4. **LLM_LOCAL_GATEWAY**
   - Interface genérica de LLM para trocar OpenAI pelo modelo local (LLaMA/Mistral fine-tuned com LoRA).
   - Modo offline, outputs JSON/Markdown/resumos PT-PT.

5. **ADVANCED_DASHBOARDS**
   - Gantt comparativo baseline vs cenário, heatmaps de carga, projeções anuais, mapas de impacto, drill-down.
   - APIs específicas e integração com previsões ML.

6. **ERP_MES_CONNECTOR**
   - Interfaces para SQL Server, REST, SOAP, stored procedures e serviços Windows.
   - Suporte bidirecional (fetch ordens/BOMs/routing/stocks/skills, push plano, recolher estados MES).
   - Cada cliente terá adaptadores específicos.

> Todos os itens acima devem ser introduzidos como módulos/stubs com TODOs claros antes de implementar lógica completa.

---

## Test & Validation Checklist

1. **Backend**
   - `cd "/Users/martimnicolau/mvp geral" && python3 -m pip install -r backend/requirements.txt`
   - Confirmar `.env` com `OPENAI_API_KEY` e Excel em `data/production_os_data_MVP.xlsx`.
   - `python3 -m uvicorn backend.api:app --reload` e testar:
     - `curl http://127.0.0.1:8000/health`
     - `curl http://127.0.0.1:8000/plan`
     - `curl -X POST http://127.0.0.1:8000/what-if/describe -d '{"scenario":"..."}'`

2. **Frontend**
   - `cd "/Users/martimnicolau/mvp geral/factory-optimizer/frontend"`
   - `npm install`
   - Criar `.env` com `VITE_API_BASE_URL=http://127.0.0.1:8000`.
   - `npm run dev` e verificar:
     - Página What-If: “Descrever cenário” (preview do ScenarioDelta).
     - “Comparar cenário” (cartão baseline vs cenário).
     - Painéis Planeamento, Gargalos e Chat.

3. **Plano exportado**
   - `python3 backend/scheduler.py` gera/atualiza `data/production_plan.csv`.
   - Validar gargalo com `curl http://127.0.0.1:8000/bottleneck`.

