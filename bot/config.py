from dataclasses import dataclass
import os

from dotenv import load_dotenv


@dataclass(slots=True)
class Config:
    bot_token: str
    database_url: str
    gemini_api_key: str
    gemini_endpoint: str
    gemini_model: str
    gemini_ssl_verify: bool
    gemini_status_endpoint_template: str


def _to_bool(value: str, default: bool = True) -> bool:
    normalized = value.strip().lower()
    if not normalized:
        return default
    return normalized in {"1", "true", "yes", "y", "on"}


def load_config() -> Config:
    load_dotenv()
    bot_token = os.getenv("BOT_TOKEN", "")
    database_url = os.getenv("DATABASE_URL", "")

    if not bot_token:
        raise ValueError("BOT_TOKEN is required")
    if not database_url:
        raise ValueError("DATABASE_URL is required")

    return Config(
        bot_token=bot_token,
        database_url=database_url,
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_endpoint=os.getenv("GEMINI_ENDPOINT", "https://api.gen-api.ru/api/v1/networks/gemini-2-5-flash-lite"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"),
        gemini_ssl_verify=_to_bool(os.getenv("GEMINI_SSL_VERIFY", "true"), default=True),
        gemini_status_endpoint_template=os.getenv("GEMINI_STATUS_ENDPOINT_TEMPLATE", "https://api.gen-api.ru/api/v1/request/get/{request_id}"),
    )
