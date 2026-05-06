"""Domain constants shared by parser and deterministic handlers."""

from __future__ import annotations


# GPS/NovAtel position type values used by the provided navigation export.
POOR_POS_TYPE = 19
GOOD_POS_TYPES = (53, 54, 56)

# The service reports user-facing timestamps in Moscow local civil time.
MOSCOW_UTC_OFFSET_HOURS = 3

KMH_FACTOR = 3.6

TIME_PERIODS = {
    "twilight": (16, 19),
    "dawn": (5, 8),
    "night": (22, 5),
    "daytime": (8, 18),
}

TIME_PERIOD_ALIASES = {
    "twilight": "twilight",
    "twighlight": "twilight",
    "сумерки": "twilight",
    "сумерках": "twilight",
    "dawn": "dawn",
    "рассвет": "dawn",
    "рассвете": "dawn",
    "night": "night",
    "ночь": "night",
    "ночью": "night",
    "daytime": "daytime",
    "day": "daytime",
    "день": "daytime",
    "днем": "daytime",
    "днём": "daytime",
}

DEFAULT_BRAKING_THRESHOLD = -2.0

GEO_REGIONS = {
    "m11": {"lat_min": 55.5, "lat_max": 60.0, "lon_min": 30.0, "lon_max": 37.5},
    "moscow": {"lat_min": 55.5, "lat_max": 56.0, "lon_min": 37.0, "lon_max": 38.0},
}

GEO_REGION_ALIASES = {
    "m11": "m11",
    "м11": "m11",
    "moscow": "moscow",
    "москва": "moscow",
}

# Approximate M11 polyline. The bounding box pre-filter keeps the route check cheap;
# the corridor then removes points that are in the box but far from the highway.
M11_ROUTE = [
    {"lat": 55.7522, "lon": 37.6156},
    {"lat": 56.8587, "lon": 35.9176},
    {"lat": 57.1459, "lon": 33.1126},
    {"lat": 58.5215, "lon": 31.2755},
    {"lat": 59.9343, "lon": 30.3351},
]
M11_CORRIDOR_KM = 15.0
