import pandas as pd

from app.handlers import aggregation, braking, filter_quality, geo_filter, time_slice
from app.query.contracts import ParsedQuery


INTENT_MAP = {
    "aggregation": aggregation.handle,
    "filter_quality": filter_quality.handle,
    "time_slice": time_slice.handle,
    "braking": braking.handle,
    "geo_filter": geo_filter.handle,
}


def dispatch(parsed: ParsedQuery, df: pd.DataFrame) -> dict:
    """Run the handler matching *parsed.intent* against *df*."""
    handler = INTENT_MAP.get(parsed.intent)
    if handler is None:
        return {
            "error": "unknown_intent",
            "message": f"Cannot handle intent: {parsed.intent}",
            "suggestion": (
                "Try asking about speed, position quality, time periods, braking, "
                "or geographic location"
            ),
        }
    return handler(df, parsed.params)
