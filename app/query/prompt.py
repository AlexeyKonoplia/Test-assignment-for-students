"""Prompt template for intent classification."""

from __future__ import annotations


SYSTEM_INSTRUCTIONS = """
Ты классифицируешь вопросы пользователя по навигационным данным тягача.
Основной язык пользователей - русский, но английские запросы тоже нужно понимать.
Ты НЕ считаешь итоговый ответ и НЕ анализируешь CSV. Твоя задача - только выбрать intent и параметры.

Верни ТОЛЬКО валидный JSON. Без markdown, без комментариев, без пояснений.
""".strip()

RESPONSE_SCHEMAS = (
    '{"intent":"aggregation","params":{"metric":"max_speed|min_speed|avg_speed|max_acceleration|total_distance"}}',
    '{"intent":"filter_quality","params":{"quality":"poor|good"}}',
    '{"intent":"time_slice","params":{"period":"twilight|dawn|night|daytime"}}',
    '{"intent":"time_slice","params":{"period":"custom","start_hour":13,"start_minute":30,"end_hour":14,"end_minute":0}}',
    '{"intent":"braking","params":{"threshold":-2.0}}',
    '{"intent":"geo_filter","params":{"region":"м11|m11|moscow|москва"}}',
    '{"intent":"unknown","params":{"reason":"short reason"}}',
)

INTENT_RULES = {
    "aggregation": "вопросы про максимум/минимум/среднюю скорость, ускорение или дистанцию.",
    "filter_quality": (
        "вопросы про плохой/хороший GPS, GNSS, качество позиционирования, сбои, ошибки или fix."
    ),
    "time_slice": (
        "вопросы про время суток или конкретный интервал часов: сумерки, рассвет, ночь, день, "
        "twilight, dawn, night, daytime, с 10 до 11, с 13:30 до 14, в 14, from 10 to 11."
    ),
    "braking": "вопросы про резкое торможение, замедление, brake, braking, deceleration.",
    "geo_filter": "вопросы про М11/m11, Москву/Moscow или точки внутри региона.",
    "unknown": "если запрос не относится к навигационным данным или невозможно выбрать intent.",
}

EXAMPLES = (
    ("максималка", '{"intent":"aggregation","params":{"metric":"max_speed"}}'),
    ("какая была максимальная скорость?", '{"intent":"aggregation","params":{"metric":"max_speed"}}'),
    ("top speed", '{"intent":"aggregation","params":{"metric":"max_speed"}}'),
    ("средняя скорость", '{"intent":"aggregation","params":{"metric":"avg_speed"}}'),
    ("average speed", '{"intent":"aggregation","params":{"metric":"avg_speed"}}'),
    ("общий пробег", '{"intent":"aggregation","params":{"metric":"total_distance"}}'),
    ("total distance by bag", '{"intent":"aggregation","params":{"metric":"total_distance"}}'),
    ("где были проблемы с gps?", '{"intent":"filter_quality","params":{"quality":"poor"}}'),
    (
        "покажи все моменты, когда качество позиционирования было плохим",
        '{"intent":"filter_quality","params":{"quality":"poor"}}',
    ),
    ("where was GPS fixed", '{"intent":"filter_quality","params":{"quality":"good"}}'),
    ("хорошее качество позиционирования", '{"intent":"filter_quality","params":{"quality":"good"}}'),
    ("в сумерках", '{"intent":"time_slice","params":{"period":"twilight"}}'),
    ("на рассвете", '{"intent":"time_slice","params":{"period":"dawn"}}'),
    ("at dawn", '{"intent":"time_slice","params":{"period":"dawn"}}'),
    ("с 10 до 11", '{"intent":"time_slice","params":{"period":"custom","start_hour":10,"end_hour":11}}'),
    ("что происходило с 8 до 9 утра?", '{"intent":"time_slice","params":{"period":"custom","start_hour":8,"end_hour":9}}'),
    (
        "что происходило с 13:30 до 14?",
        '{"intent":"time_slice","params":{"period":"custom","start_hour":13,"start_minute":30,"end_hour":14,"end_minute":0}}',
    ),
    ("что было в 14?", '{"intent":"time_slice","params":{"period":"custom","start_hour":14,"end_hour":15}}'),
    ("from 22 to 5", '{"intent":"time_slice","params":{"period":"custom","start_hour":22,"end_hour":5}}'),
    ("резко тормозил", '{"intent":"braking","params":{"threshold":-2.0}}'),
    ("hard braking below minus three", '{"intent":"braking","params":{"threshold":-3.0}}'),
    ("когда тягач находился в пределах трассы М11?", '{"intent":"geo_filter","params":{"region":"м11"}}'),
    ("на м11", '{"intent":"geo_filter","params":{"region":"м11"}}'),
    ("points in Moscow", '{"intent":"geo_filter","params":{"region":"moscow"}}'),
)

PYTHON_OWNED_PARAMS_NOTE = """
Часы известных периодов и координатные границы регионов выставляет Python после классификации.
Для конкретного пользовательского интервала верни start_hour/end_hour как целые часы 0-24.
Если пользователь указал минуты, верни start_minute/end_minute как целые минуты 0-59.
Не добавляй lat/lon или вычисленные значения известных периодов самостоятельно.
""".strip()


def build_prompt(query: str) -> str:
    """Build a Russian-first prompt that asks the LLM for JSON classification only."""
    sections = [
        SYSTEM_INSTRUCTIONS,
        "Допустимые схемы ответа:\n" + "\n".join(RESPONSE_SCHEMAS),
        "Правила:\n" + "\n".join(f"- {intent}: {rule}" for intent, rule in INTENT_RULES.items()),
        "Примеры:\n" + _format_examples(),
        PYTHON_OWNED_PARAMS_NOTE,
        f"Запрос пользователя: {query}\nJSON:",
    ]
    return "\n\n".join(sections)


def _format_examples() -> str:
    return "\n".join(f"Пользователь: {user_query}\nJSON: {payload}" for user_query, payload in EXAMPLES)
