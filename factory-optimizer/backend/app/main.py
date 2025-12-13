from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv

from app.api import bottlenecks, chat, etl, insight, insights, inventory, planning, planning_chat, planning_v2, suggestions, technical_queries, whatif
from app.etl.loader import get_loader, run_startup_etl


load_dotenv()

app = FastAPI(title="ProdPlan 4.0 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(planning.router, prefix="/api/planning", tags=["Planning"])
app.include_router(planning_v2.router, prefix="/api/planning/v2", tags=["Planning V2"])
app.include_router(planning_chat.router, prefix="/api/planning/chat", tags=["Planning Chat"])
app.include_router(technical_queries.router, prefix="/api/technical", tags=["Technical Queries"])
app.include_router(bottlenecks.router, prefix="/api/bottlenecks", tags=["Bottlenecks"])
app.include_router(inventory.router, prefix="/api/inventory", tags=["Inventory"])
app.include_router(whatif.router, prefix="/api/whatif", tags=["What-If"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(suggestions.router, prefix="/api/suggestions", tags=["Suggestions"])
app.include_router(insight.router, prefix="/api/insight", tags=["Insight"])
app.include_router(insights.router, prefix="/api/insights", tags=["Insights"])
app.include_router(etl.router, prefix="/api", tags=["ETL"])


@app.on_event("startup")
async def startup_event():
    try:
        summary = run_startup_etl()
        if summary:
            get_loader().status.setdefault("startup_summary", summary)
    except Exception as exc:  # pylint: disable=broad-except
        get_loader().status.setdefault("startup_error", str(exc))


@app.get("/api/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

