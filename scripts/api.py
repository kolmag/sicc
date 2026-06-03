"""
scripts/api.py — thin FastAPI wrapper around the validated SICC RAG brain.

This module deliberately calls scripts.answer.answer() instead of duplicating
retrieval, prompting, model, or grounding logic. The current Streamlit app can
remain the validated baseline while a separate web UI talks to this API.
"""

from __future__ import annotations

import asyncio
import json
import logging
import pickle
import sqlite3
import time
import uuid
from collections.abc import AsyncIterator
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)

APP_NAME = "SICC API"
APP_VERSION = "0.1.0"
CHROMA_DB_PATH = "chroma_db"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DB_PATH = _PROJECT_ROOT / "data" / "supplier_portfolio.db"
_ML_DIR = _PROJECT_ROOT / "ml"

rag_answer = None
build_where_filter = None

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="Thin API layer for the validated SICC supplier intelligence pipeline.",
)

_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://web:3000",  # Docker Compose service name
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    family: Optional[str] = None
    risk: Optional[str] = None
    session_id: str = "api"


class ChatResponse(BaseModel):
    run_id: str
    elapsed_ms: int
    result: dict


class PortfolioRequest(BaseModel):
    question: str = Field(..., min_length=1)
    family: Optional[str] = None
    region: Optional[str] = None
    risk: Optional[str] = None


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _load_brain():
    global rag_answer, build_where_filter
    if rag_answer is None or build_where_filter is None:
        from scripts.answer import answer, build_where_filter as where_filter_builder

        rag_answer = answer
        build_where_filter = where_filter_builder
    return rag_answer, build_where_filter


def _run_rag(request: ChatRequest):
    answer_fn, where_filter_builder = _load_brain()
    where_filter = where_filter_builder(risk=request.risk, family=request.family)
    return answer_fn(
        question=request.question,
        db_path=CHROMA_DB_PATH,
        where_filter=where_filter,
        session_id=request.session_id,
    )


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "app": APP_NAME,
        "version": APP_VERSION,
        "brain": "scripts.answer.answer",
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    started = time.perf_counter()
    run_id = str(uuid.uuid4())
    result = await asyncio.to_thread(_run_rag, request)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return ChatResponse(
        run_id=run_id,
        elapsed_ms=elapsed_ms,
        result=result.model_dump(),
    )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    async def events() -> AsyncIterator[str]:
        started = time.perf_counter()
        run_id = str(uuid.uuid4())
        yield _sse("status", {"run_id": run_id, "message": "started"})
        yield _sse("status", {"run_id": run_id, "message": "running_sicc_brain"})

        try:
            result = await asyncio.to_thread(_run_rag, request)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            yield _sse(
                "result",
                {
                    "run_id": run_id,
                    "elapsed_ms": elapsed_ms,
                    "result": result.model_dump(),
                },
            )
            yield _sse("done", {"run_id": run_id})
        except Exception:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            logger.exception("chat_stream failed (run_id=%s)", run_id)
            yield _sse(
                "error",
                {
                    "run_id": run_id,
                    "elapsed_ms": elapsed_ms,
                    "message": "The SICC brain failed to process this request.",
                },
            )

    return StreamingResponse(events(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Supplier / ML data endpoints
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_ml_artifacts() -> dict | None:
    model_path = _ML_DIR / "model.pkl"
    shap_path = _ML_DIR / "shap_values.pkl"
    metrics_path = _ML_DIR / "model_metrics.json"
    if not model_path.exists():
        return None
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(shap_path, "rb") as f:
            shap_payload = pickle.load(f)
        sv = shap_payload["shap_values"]
        if isinstance(sv, np.ndarray) and sv.ndim == 3:
            sv = [sv[:, :, i] for i in range(sv.shape[2])]
        metrics: dict = {}
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
        return {
            "model": model,
            "shap_values": sv,
            "expected_value": shap_payload["expected_value"],
            "feature_names": shap_payload["feature_names"],
            "supplier_ids": shap_payload["supplier_ids"],
            "y_pred": shap_payload["y_pred"],
            "y_pred_proba": shap_payload["y_pred_proba"],
            "label_order": shap_payload["label_order"],
            "winner_name": shap_payload.get("winner_name", "RandomForest"),
            "metrics": metrics,
        }
    except Exception:
        return None


def _query_db(sql: str, params: tuple = ()) -> list[dict]:
    if not _DB_PATH.exists():
        return []
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _ml_prediction(ml: dict | None, supplier_id: str) -> dict:
    if ml is None or supplier_id not in ml["supplier_ids"]:
        return {"ml_prediction": None, "ml_confidence": None}
    idx = ml["supplier_ids"].index(supplier_id)
    class_idx = int(ml["y_pred"][idx])
    return {
        "ml_prediction": ml["label_order"][class_idx],
        "ml_confidence": round(float(ml["y_pred_proba"][idx][class_idx]), 4),
    }


@app.get("/suppliers/compare")
def compare_suppliers(ids: str = Query(..., description="Comma-separated supplier IDs, max 3")) -> dict:
    id_list = [i.strip() for i in ids.split(",") if i.strip()][:3]
    results = []
    ml = _load_ml_artifacts()
    for supplier_id in id_list:
        rows = _query_db("""
            SELECT r.*, s.name, s.country, s.region, s.city, s.product_family,
                   s.subcategory, s.certification, s.archetype, s.archetype_description,
                   s.primary_contact, s.account_manager, s.years_active, s.onboarding_date
            FROM risk_scores r JOIN suppliers s ON r.supplier_id = s.supplier_id
            WHERE r.supplier_id = ?
        """, (supplier_id,))
        if not rows:
            continue
        supplier = rows[0]
        kpis = _query_db(
            "SELECT * FROM supplier_kpis WHERE supplier_id = ? ORDER BY year_month DESC LIMIT 12",
            (supplier_id,),
        )
        pred = _ml_prediction(ml, supplier_id)
        shap_top: list[dict] = []
        if ml and supplier_id in ml["supplier_ids"]:
            idx = ml["supplier_ids"].index(supplier_id)
            class_idx = int(ml["y_pred"][idx])
            sv = ml["shap_values"][class_idx][idx]
            order = np.argsort(np.abs(sv))[::-1][:8]
            shap_top = [
                {"feature": ml["feature_names"][i], "value": round(float(sv[i]), 4)}
                for i in order
            ]
        results.append({
            "supplier": supplier,
            "kpis": kpis,
            "ml_prediction": pred["ml_prediction"],
            "ml_confidence": pred["ml_confidence"],
            "shap_top_features": shap_top,
        })
    return {"suppliers": results}


@app.post("/chat/portfolio")
async def chat_portfolio(request: PortfolioRequest) -> dict:
    started = time.perf_counter()
    run_id = str(uuid.uuid4())
    from scripts.portfolio_qa import run_portfolio_query
    result = await asyncio.to_thread(
        run_portfolio_query,
        request.question,
        request.family,
        request.region,
        request.risk,
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return {"run_id": run_id, "elapsed_ms": elapsed_ms, **result}


@app.get("/suppliers/sparklines")
def supplier_sparklines(months: int = Query(6, ge=3, le=24)) -> dict:
    """Last N months of PPM per supplier — compact for table sparklines."""
    rows = _query_db("""
        SELECT k.supplier_id, k.ppm_external, k.year_month
        FROM supplier_kpis k
        WHERE k.year_month >= (
            SELECT strftime('%Y-%m', date(MAX(year_month) || '-01', ?))
            FROM supplier_kpis
        )
        ORDER BY k.supplier_id, k.year_month
    """, (f"-{months - 1} months",))

    result: dict[str, list[float]] = {}
    for r in rows:
        sid = r["supplier_id"]
        if sid not in result:
            result[sid] = []
        result[sid].append(round(float(r["ppm_external"] or 0), 1))
    return {"sparklines": result}


@app.get("/suppliers")
def list_suppliers(
    risk: Optional[str] = Query(None, description="Filter by risk_label: red, amber, green"),
    family: Optional[str] = Query(None),
    spend_tier: Optional[str] = Query(None),
    single_source: Optional[bool] = Query(None),
    ml_mismatch: bool = Query(False, description="Only return suppliers where ML != rule-based label"),
) -> dict:
    rows = _query_db("""
        SELECT
            r.supplier_id, r.composite_risk_score, r.risk_label,
            r.avg_ppm_3m, r.avg_otd_3m, r.avg_audit_score_3m,
            r.avg_scar_count_3m, r.recommended_action,
            r.spend_tier, r.annual_spend_eur, r.single_source,
            r.qualification_status, r.strategic_importance,
            s.name, s.country, s.region, s.product_family,
            s.archetype, s.years_active, s.certification
        FROM risk_scores r
        JOIN suppliers s ON r.supplier_id = s.supplier_id
    """)

    ml = _load_ml_artifacts()
    result = []
    for row in rows:
        pred = _ml_prediction(ml, row["supplier_id"])
        row.update(pred)

        if risk and row["risk_label"].lower() != risk.lower():
            continue
        if family and row["product_family"] != family:
            continue
        if spend_tier and row["spend_tier"] != spend_tier:
            continue
        if single_source is not None:
            ss = str(row["single_source"]).lower() in ("true", "1", "yes")
            if ss != single_source:
                continue
        if ml_mismatch and pred["ml_prediction"] is not None:
            if pred["ml_prediction"].lower() == row["risk_label"].lower():
                continue

        result.append(row)

    result.sort(key=lambda r: r["composite_risk_score"] or 0, reverse=True)
    return {"suppliers": result, "total": len(result)}


@app.get("/suppliers/{supplier_id}")
def get_supplier(supplier_id: str) -> dict:
    rows = _query_db("""
        SELECT r.*, s.name, s.country, s.region, s.city, s.product_family,
               s.subcategory, s.certification, s.archetype, s.archetype_description,
               s.primary_contact, s.primary_contact_email, s.account_manager,
               s.years_active, s.onboarding_date
        FROM risk_scores r
        JOIN suppliers s ON r.supplier_id = s.supplier_id
        WHERE r.supplier_id = ?
    """, (supplier_id,))
    if not rows:
        raise HTTPException(status_code=404, detail="Supplier not found")
    supplier = rows[0]

    kpis = _query_db(
        "SELECT * FROM supplier_kpis WHERE supplier_id = ? ORDER BY year_month",
        (supplier_id,),
    )
    claims = _query_db(
        "SELECT * FROM claims WHERE supplier_id = ? ORDER BY creation_date DESC LIMIT 20",
        (supplier_id,),
    )
    audits = _query_db(
        "SELECT * FROM audits WHERE supplier_id = ? ORDER BY audit_date DESC LIMIT 10",
        (supplier_id,),
    )
    apqp = _query_db(
        "SELECT * FROM apqp_projects WHERE supplier_id = ? ORDER BY creation_date DESC LIMIT 10",
        (supplier_id,),
    )
    events = _query_db(
        "SELECT * FROM external_events WHERE supplier_id = ? ORDER BY event_date DESC LIMIT 15",
        (supplier_id,),
    )

    ml = _load_ml_artifacts()
    pred = _ml_prediction(ml, supplier_id)
    shap_top: list[dict[str, Any]] = []
    if ml and supplier_id in ml["supplier_ids"]:
        idx = ml["supplier_ids"].index(supplier_id)
        class_idx = int(ml["y_pred"][idx])
        sv = ml["shap_values"][class_idx][idx]
        feat_names = ml["feature_names"]
        order = np.argsort(np.abs(sv))[::-1][:12]
        shap_top = [
            {"feature": feat_names[i], "value": round(float(sv[i]), 4)}
            for i in order
        ]

    return {
        "supplier": supplier,
        "kpis": kpis,
        "claims": claims,
        "audits": audits,
        "apqp_projects": apqp,
        "external_events": events,
        "ml_prediction": pred["ml_prediction"],
        "ml_confidence": pred["ml_confidence"],
        "shap_top_features": shap_top,
    }


@app.get("/model/metrics")
def model_metrics() -> dict:
    ml = _load_ml_artifacts()
    if ml is None:
        raise HTTPException(status_code=503, detail="ML model not trained yet")
    m = ml["metrics"].get("winner_metrics", ml["metrics"])
    return {
        "winner_name": ml["winner_name"],
        "n_features": len(ml["feature_names"]),
        "n_suppliers": len(ml["supplier_ids"]),
        "accuracy": round(m.get("accuracy", 0), 4),
        "f1_macro": round(m.get("f1_macro", 0), 4),
        "auc_ovr": round(m.get("auc_ovr", 0), 4),
        "f1_red": round(m.get("f1_red", 0), 4),
        "f1_amber": round(m.get("f1_amber", 0), 4),
        "f1_green": round(m.get("f1_green", 0), 4),
    }


@app.get("/model/feature-importance")
def feature_importance(top_n: int = Query(20, ge=1, le=61)) -> dict:
    ml = _load_ml_artifacts()
    if ml is None:
        raise HTTPException(status_code=503, detail="ML model not trained yet")
    shap_red = ml["shap_values"][2]
    mean_abs = np.abs(shap_red).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n]
    return {
        "features": [
            {"feature": ml["feature_names"][i], "importance": round(float(mean_abs[i]), 5)}
            for i in order
        ]
    }


def main() -> None:
    import os

    import uvicorn

    # Bind to 0.0.0.0 by default so the service is reachable inside Docker
    # (the web container talks to http://api:8000). reload is opt-in for local dev.
    host = os.environ.get("SICC_API_HOST", "0.0.0.0")
    port = int(os.environ.get("SICC_API_PORT", "8000"))
    reload = os.environ.get("SICC_API_RELOAD", "").lower() in ("1", "true", "yes")

    uvicorn.run("scripts.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
