from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from bot.db import Teacher


def teacher_choice_keyboard(teachers: list[Teacher]) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=teacher.name, callback_data=f"teacher:{teacher.id}")]
            for teacher in teachers
        ]
    )
