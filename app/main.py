"""FastAPI application for semantic navigation search."""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from app.data_loader import DataStore
from app.query import parser as query_parser
from app.query import router as query_router


LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format=LOG_FORMAT)
logger = logging.getLogger("semantic-nav-search")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data once before serving requests."""
    data_store = app.state.data_store
    data_store.load(os.getenv("DATA_DIR", "./data"))
    logger.info("data_loaded", extra={"records_loaded": len(data_store.get())})
    yield


app = FastAPI(title="Semantic Nav Search", lifespan=lifespan)
app.state.data_store = DataStore


class QueryRequest(BaseModel):
    """Request body for natural-language queries."""

    query: str


@app.post("/query")
async def query_endpoint(body: QueryRequest, request: Request) -> dict:
    """Parse a natural-language query, dispatch computation, and return JSON."""
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    started = time.perf_counter()
    parsed = await query_parser.parse(body.query)
    df = request.app.state.data_store.get()
    result = query_router.dispatch(parsed, df)
    latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
    status = "error" if "error" in result else "success"
    logger.info(
        "query_completed",
        extra={
            "query": body.query,
            "intent": parsed.intent,
            "params": parsed.params,
            "result_count": _result_count(result),
            "latency_ms": latency_ms,
        },
    )

    response = {"status": status, "query": body.query, "intent": parsed.intent, "result": result}
    if result.get("error") == "unknown_intent":
        raise HTTPException(status_code=422, detail=response)
    return response


def _result_count(result: dict) -> int | None:
    for key in ("total_points", "total_braking_events"):
        if key in result:
            return int(result[key])
    if "events" in result and isinstance(result["events"], list):
        return len(result["events"])
    if "points" in result and isinstance(result["points"], list):
        return len(result["points"])
    return None
