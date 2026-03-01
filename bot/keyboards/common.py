from aiogram.types import KeyboardButton, ReplyKeyboardMarkup


def teacher_menu_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Сгенерировать задания")],
            [KeyboardButton(text="Мой пул заданий")],
        ],
        resize_keyboard=True,
    )


def student_menu_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Режим обучения")],
            [KeyboardButton(text="Режим тестирования")],
        ],
        resize_keyboard=True,
    )


def learning_after_answer_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Следующее задание")],
            [KeyboardButton(text="Завершить обучение")],
        ],
        resize_keyboard=True,
    )


def waiting_answer_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="Пропустить задание")]],
        resize_keyboard=True,
    )


def learning_incorrect_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Подсказка")],
            [KeyboardButton(text="Попробовать ещё раз")],
            [KeyboardButton(text="Пропустить задание")],
        ],
        resize_keyboard=True,
    )


def theory_keyboard(has_next_page: bool) -> ReplyKeyboardMarkup:
    if has_next_page:
        buttons = [[KeyboardButton(text="Следующая страница")]]
    else:
        buttons = [[KeyboardButton(text="Начать решение")]]

    buttons.append([KeyboardButton(text="Завершить обучение")])
    return ReplyKeyboardMarkup(keyboard=buttons, resize_keyboard=True)
