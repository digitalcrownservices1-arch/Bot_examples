import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from bot.config import load_config
from bot.db import Database
from bot.handlers.start import router as start_router
from bot.middlewares import DbSessionMiddleware
from bot.services.formula_renderer import FormulaRenderer
from bot.services.gemini_client import GeminiClient


async def main() -> None:
    logging.basicConfig(level=logging.INFO)

    config = load_config()
    db = Database(config.database_url)
    llm = GeminiClient(
        config.gemini_api_key,
        config.gemini_endpoint,
        config.gemini_model,
        config.gemini_ssl_verify,
        config.gemini_status_endpoint_template,
    )
    renderer = FormulaRenderer()
    await db.connect()

    bot = Bot(
        token=config.bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()
    dp.update.middleware(DbSessionMiddleware(db, llm, renderer))
    dp.include_router(start_router)

    try:
        await dp.start_polling(bot)
    finally:
        await db.close()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
