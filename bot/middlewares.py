from typing import Any, Awaitable, Callable, Dict

from aiogram import BaseMiddleware
from aiogram.types import TelegramObject

from bot.db import Database
from bot.services.formula_renderer import FormulaRenderer
from bot.services.gemini_client import GeminiClient


class DbSessionMiddleware(BaseMiddleware):
    def __init__(self, db: Database, llm: GeminiClient, renderer: FormulaRenderer) -> None:
        self._db = db
        self._llm = llm
        self._renderer = renderer

    async def __call__(
        self,
        handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: Dict[str, Any],
    ) -> Any:
        data["db"] = self._db
        data["llm"] = self._llm
        data["renderer"] = self._renderer
        return await handler(event, data)
