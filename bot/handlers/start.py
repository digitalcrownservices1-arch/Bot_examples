from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import BufferedInputFile, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from aiogram.exceptions import TelegramBadRequest

import base64
import io

from bot.db import Database, Student, Task, Teacher, TheoryPage, Topic
from bot.keyboards.common import (
    learning_after_answer_keyboard,
    learning_incorrect_keyboard,
    student_menu_keyboard,
    teacher_menu_keyboard,
    theory_keyboard,
    waiting_answer_keyboard,
)
from bot.services.formula_renderer import FormulaRenderer
from bot.services.gemini_client import GeminiClient

router = Router()


class StudentFlow(StatesGroup):
    choosing_topic = State()
    showing_theory = State()
    waiting_learning_answer = State()
    waiting_learning_retry_answer = State()
    waiting_testing_answer = State()
    learning_incorrect_options = State()


class TeacherCreateFlow(StatesGroup):
    waiting_topic = State()
    waiting_mode = State()
    waiting_count = State()
    reviewing_generated = State()


@router.message(Command("fileid"), F.photo)
async def show_photo_file_id(message: Message) -> None:
    await message.answer(f"file_id: <code>{message.photo[-1].file_id}</code>")


@router.message(Command("fileid"), F.document)
async def show_document_file_id(message: Message) -> None:
    if message.document and message.document.mime_type and message.document.mime_type.startswith("image/"):
        await message.answer(f"file_id: <code>{message.document.file_id}</code>")
        return
    await message.answer("Для команды /fileid отправьте изображение фото или документом.")


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext, db: Database) -> None:
    user = message.from_user
    if not user:
        return

    teacher = await db.get_teacher_by_telegram_id(user.id)
    if teacher:
        await state.clear()
        await message.answer(
            f"Здравствуйте, {teacher.name}! Вы вошли как преподаватель.",
            reply_markup=teacher_menu_keyboard(),
        )
        return

    student = await db.get_student_by_telegram_id(user.id)
    if student:
        await state.clear()
        await message.answer(
            f"Здравствуйте, {student.name}! Вы вошли как студент.",
            reply_markup=student_menu_keyboard(),
        )
        return

    await state.clear()
    await message.answer("Ваш Telegram ID не найден в базе.")


@router.message(F.text == "Сгенерировать задания")
async def teacher_start_generation(message: Message, state: FSMContext, db: Database, llm: GeminiClient) -> None:
    teacher = await _get_teacher_or_notify(message, db)
    if not teacher:
        return
    if not llm.enabled:
        await message.answer("LLM не настроена. Заполните GEMINI_API_KEY в .env")
        return

    topics = await db.list_topics()
    if not topics:
        await message.answer("В базе нет тем.")
        return

    await state.set_state(TeacherCreateFlow.waiting_topic)
    await state.update_data(teacher_id=teacher.id)
    await message.answer("Выберите тему:", reply_markup=_topics_keyboard(topics))


@router.callback_query(TeacherCreateFlow.waiting_topic, F.data.startswith("teacher_topic:"))
async def teacher_select_topic(callback: CallbackQuery, state: FSMContext, db: Database) -> None:
    if not callback.data or not callback.message:
        return

    topic_id = int(callback.data.split(":", 1)[1])
    topics = await db.list_topics()
    topic = next((item for item in topics if item.id == topic_id), None)
    if not topic:
        await callback.answer("Тема не найдена", show_alert=True)
        return

    await state.set_state(TeacherCreateFlow.waiting_mode)
    await state.update_data(topic_id=topic.id, topic_title=topic.title, topic_prompt=topic.llm_prompt)
    await callback.message.answer("Выберите режим задач:", reply_markup=_modes_keyboard())
    await callback.answer()


@router.callback_query(TeacherCreateFlow.waiting_mode, F.data.startswith("teacher_mode:"))
async def teacher_select_mode(callback: CallbackQuery, state: FSMContext) -> None:
    if not callback.data or not callback.message:
        return

    mode = callback.data.split(":", 1)[1]
    if mode not in {"learning", "testing"}:
        await callback.answer("Некорректный режим", show_alert=True)
        return

    await state.set_state(TeacherCreateFlow.waiting_count)
    await state.update_data(mode=mode)
    await callback.message.answer("Введите количество заданий (1-10):")
    await callback.answer()


@router.message(TeacherCreateFlow.waiting_count)
async def teacher_set_count(
    message: Message,
    state: FSMContext,
    llm: GeminiClient,
    renderer: FormulaRenderer,
    db: Database,
) -> None:
    value = (message.text or "").strip()
    if not value.isdigit():
        await message.answer("Нужно ввести число от 1 до 10.")
        return

    count = int(value)
    if count < 1 or count > 10:
        await message.answer("Количество должно быть от 1 до 10.")
        return

    data = await state.get_data()
    topic_title = str(data["topic_title"])
    topic_prompt = str(data.get("topic_prompt") or topic_title)
    mode = str(data["mode"])
    teacher_id = int(data["teacher_id"])
    topic_id = int(data["topic_id"])

    recent_task_texts = await db.list_recent_teacher_formulas(teacher_id, topic_id, mode, limit=10)
    forbidden_formulas = [_extract_formula_from_task_text(item) for item in recent_task_texts if _extract_formula_from_task_text(item)]

    await message.answer("Генерирую варианты, это может занять до минуты…")
    candidates: list[dict[str, str | bytes | None]] = []
    generated_formulas: list[str] = []
    for i in range(count):
        try:
            candidate = await _build_candidate(
                llm,
                renderer,
                topic_prompt,
                mode,
                i + 1,
                forbidden_formulas=forbidden_formulas + generated_formulas,
            )
            candidates.append(candidate)
            generated_formula = str(candidate.get("latex") or "").strip()
            if generated_formula:
                generated_formulas.append(generated_formula)
        except Exception as exc:  # noqa: BLE001
            await message.answer(f"Ошибка генерации кандидата #{i + 1}: {exc}")
            break

    if not candidates:
        await state.clear()
        await message.answer("Не удалось сгенерировать задания.", reply_markup=teacher_menu_keyboard())
        return

    await state.set_state(TeacherCreateFlow.reviewing_generated)
    await state.update_data(
        total_to_generate=len(candidates),
        generated_index=0,
        generated_candidates=candidates,
        forbidden_formulas=forbidden_formulas,
    )
    await _show_generated_candidate(message, state)


@router.callback_query(TeacherCreateFlow.reviewing_generated, F.data == "teacher_gen:regenerate")
async def teacher_regenerate(
    callback: CallbackQuery,
    state: FSMContext,
    llm: GeminiClient,
    renderer: FormulaRenderer,
) -> None:
    if not callback.message:
        return

    data = await state.get_data()
    topic_title = str(data["topic_title"])
    topic_prompt = str(data.get("topic_prompt") or topic_title)
    mode = str(data["mode"])
    generated_index = int(data.get("generated_index", 0))
    candidates = list(data.get("generated_candidates", []))

    if generated_index >= len(candidates):
        await callback.answer("Нет кандидата для перегенерации", show_alert=True)
        return

    forbidden_formulas = list(data.get("forbidden_formulas", []))
    already_generated = [
        str(item.get("latex") or "").strip()
        for idx, item in enumerate(candidates)
        if idx != generated_index and str(item.get("latex") or "").strip()
    ]

    try:
        candidates[generated_index] = await _build_candidate(
            llm,
            renderer,
            topic_prompt,
            mode,
            generated_index + 1,
            forbidden_formulas=forbidden_formulas + already_generated,
        )
    except Exception as exc:  # noqa: BLE001
        await callback.answer(f"Ошибка генерации: {exc}", show_alert=True)
        return

    await state.update_data(generated_candidates=candidates)
    await _show_generated_candidate(callback.message, state)
    await callback.answer()


@router.callback_query(TeacherCreateFlow.reviewing_generated, F.data == "teacher_gen:approve")
async def teacher_approve(callback: CallbackQuery, state: FSMContext, db: Database) -> None:
    if not callback.message:
        return

    data = await state.get_data()
    teacher_id = int(data["teacher_id"])
    topic_id = int(data["topic_id"])
    mode = str(data["mode"])
    generated_index = int(data.get("generated_index", 0))
    total_to_generate = int(data.get("total_to_generate", 1))

    candidate_text = str(data.get("candidate_text", ""))
    candidate_hint = data.get("candidate_hint")
    candidate_image_file_id = data.get("candidate_image_file_id")
    candidate_answer = data.get("candidate_answer")

    task_id = await db.create_task(
        topic_id=topic_id,
        teacher_id=teacher_id,
        mode=mode,
        task_text=candidate_text,
        task_hint_text=candidate_hint,
        task_answer_text=candidate_answer,
        task_image_file_id=candidate_image_file_id,
    )

    if task_id is None:
        await callback.message.answer(
            "Такой пример уже есть в вашей базе. Пожалуйста, перегенерируйте его.",
            reply_markup=_generated_review_keyboard(),
        )
        await callback.answer()
        return

    generated_index += 1
    if generated_index >= total_to_generate:
        await state.clear()
        await callback.message.answer(
            f"Готово! Добавлено задач: {generated_index}. Последняя задача ID: {task_id}",
            reply_markup=teacher_menu_keyboard(),
        )
        await callback.answer()
        return

    await state.update_data(generated_index=generated_index)
    await callback.message.answer(f"Задача #{generated_index} подтверждена (ID {task_id}).")
    await _show_generated_candidate(callback.message, state)
    await callback.answer()


@router.callback_query(TeacherCreateFlow.reviewing_generated, F.data == "teacher_gen:skip")
async def teacher_skip_candidate(callback: CallbackQuery, state: FSMContext) -> None:
    if not callback.message:
        return

    data = await state.get_data()
    generated_index = int(data.get("generated_index", 0))
    total_to_generate = int(data.get("total_to_generate", 1))

    generated_index += 1
    if generated_index >= total_to_generate:
        await state.clear()
        await callback.message.answer(
            "Готово! Генерация завершена. Пропущенный кандидат не добавлен в пул.",
            reply_markup=teacher_menu_keyboard(),
        )
        await callback.answer()
        return

    await state.update_data(generated_index=generated_index)
    await _show_generated_candidate(callback.message, state)
    await callback.answer()


@router.callback_query(TeacherCreateFlow.reviewing_generated, F.data == "teacher_gen:cancel")
async def teacher_cancel_generation(callback: CallbackQuery, state: FSMContext) -> None:
    if callback.message:
        await callback.message.answer("Генерация отменена.", reply_markup=teacher_menu_keyboard())
    await state.clear()
    await callback.answer()


@router.message(F.text == "Мой пул заданий")
async def teacher_pool(message: Message, state: FSMContext, db: Database) -> None:
    teacher = await _get_teacher_or_notify(message, db)
    if not teacher:
        return

    tasks = await db.list_teacher_tasks(teacher.id)
    if not tasks:
        await message.answer("Пул задач пока пуст.")
        return

    await state.update_data(teacher_pool_ids=[task.id for task in tasks], teacher_pool_list_page=0)
    await _send_pool_list(message, tasks, page=0)




@router.callback_query(F.data.startswith("pool_list_nav:"))
async def teacher_pool_list_nav(callback: CallbackQuery, state: FSMContext, db: Database) -> None:
    if not callback.data or not callback.message:
        return

    direction = callback.data.split(":", 1)[1]
    teacher = await _get_teacher_from_callback(callback, db)
    if not teacher:
        await callback.answer("Доступ запрещён", show_alert=True)
        return

    tasks = await db.list_teacher_tasks(teacher.id)
    if not tasks:
        await callback.answer("Пул пуст", show_alert=True)
        return

    data = await state.get_data()
    page = int(data.get("teacher_pool_list_page", 0))
    total_pages = max((len(tasks) - 1) // 10 + 1, 1)

    if direction == "next":
        page = min(page + 1, total_pages - 1)
    else:
        page = max(page - 1, 0)

    await state.update_data(teacher_pool_list_page=page)
    await _send_pool_list(callback.message, tasks, page=page, edit=True)
    await callback.answer()


@router.callback_query(F.data.startswith("pool_open:"))
async def teacher_pool_open(callback: CallbackQuery, state: FSMContext, db: Database) -> None:
    if not callback.data or not callback.message:
        return

    task_id = int(callback.data.split(":", 1)[1])
    teacher = await _get_teacher_from_callback(callback, db)
    if not teacher:
        await callback.answer("Доступ запрещён", show_alert=True)
        return

    tasks = await db.list_teacher_tasks(teacher.id)
    ids = [task.id for task in tasks]
    if task_id not in ids:
        await callback.answer("Задача не найдена", show_alert=True)
        return

    index = ids.index(task_id)
    await state.update_data(teacher_pool_ids=ids)
    await state.update_data(teacher_pool_current_id=task_id)
    await _send_pool_task(callback.message, tasks[index], index, len(tasks))
    await callback.answer()


@router.callback_query(F.data.startswith("pool_nav:"))
async def teacher_pool_nav(callback: CallbackQuery, state: FSMContext, db: Database) -> None:
    if not callback.data or not callback.message:
        return

    direction = callback.data.split(":", 1)[1]
    teacher = await _get_teacher_from_callback(callback, db)
    if not teacher:
        await callback.answer("Доступ запрещён", show_alert=True)
        return

    tasks = await db.list_teacher_tasks(teacher.id)
    if not tasks:
        await callback.answer("Пул пуст", show_alert=True)
        return

    data = await state.get_data()
    ids = data.get("teacher_pool_ids") or [task.id for task in tasks]
    current_id = int(data.get("teacher_pool_current_id", ids[0]))
    index = ids.index(current_id) if current_id in ids else 0

    if direction == "next":
        index = min(index + 1, len(ids) - 1)
    else:
        index = max(index - 1, 0)

    next_id = ids[index]
    task = next(item for item in tasks if item.id == next_id)
    await state.update_data(teacher_pool_current_id=next_id)
    await _send_pool_task(callback.message, task, index, len(ids))
    await callback.answer()


@router.callback_query(F.data == "pool_noop")
async def teacher_pool_noop(callback: CallbackQuery) -> None:
    await callback.answer()


@router.callback_query(F.data == "pool_back")
async def teacher_pool_back(callback: CallbackQuery, state: FSMContext, db: Database) -> None:
    if not callback.message:
        return
    teacher = await _get_teacher_from_callback(callback, db)
    if not teacher:
        await callback.answer("Доступ запрещён", show_alert=True)
        return

    tasks = await db.list_teacher_tasks(teacher.id)
    data = await state.get_data()
    page = int(data.get("teacher_pool_list_page", 0))
    await _send_pool_list(callback.message, tasks, page=page)
    await callback.answer()


@router.message(F.text == "Режим обучения")
async def student_learning_mode(message: Message, state: FSMContext, db: Database) -> None:
    student = await _get_student_or_notify(message, db)
    if not student:
        return

    topics = await db.list_topics()
    if not topics:
        await message.answer("В базе нет тем.")
        return

    await state.set_state(StudentFlow.choosing_topic)
    await state.update_data(pending_mode="learning")
    await message.answer("Выберите тему:", reply_markup=_student_topics_keyboard(topics))




@router.callback_query(StudentFlow.choosing_topic, F.data.startswith("student_topic:"))
async def student_select_topic(callback: CallbackQuery, state: FSMContext, db: Database) -> None:
    if not callback.data or not callback.message:
        return

    student = await _get_student_from_callback_or_notify(callback, db)
    if not student:
        return

    topic_id = int(callback.data.split(":", 1)[1])
    topics = await db.list_topics()
    topic = next((item for item in topics if item.id == topic_id), None)
    if not topic:
        await callback.answer("Тема не найдена", show_alert=True)
        return

    data = await state.get_data()
    pending_mode = str(data.get("pending_mode") or "")
    await state.update_data(selected_topic_id=topic_id, selected_topic_title=topic.title)

    if pending_mode == "learning":
        pages = await db.list_theory_pages(topic_id=topic_id)
        if not pages:
            await state.update_data(selected_topic_id=topic_id)
            await _send_learning_task(callback.message, state, db)
            await callback.answer()
            return

        await state.set_state(StudentFlow.showing_theory)
        await state.update_data(theory_index=0)
        await _send_theory_page(callback.message, pages, 0)
        await callback.answer()
        return

    solved_count = await db.count_student_answers_by_mode(student.id, "testing")
    if solved_count >= 10:
        await state.clear()
        await callback.message.answer("Вы уже завершили тестирование (10 из 10).", reply_markup=student_menu_keyboard())
        await callback.answer()
        return

    await state.update_data(selected_topic_id=topic_id)
    await _send_testing_task(callback.message, state, db, student)
    await callback.answer()


@router.message(StudentFlow.showing_theory, F.text == "Следующая страница")
async def next_theory_page(message: Message, state: FSMContext, db: Database) -> None:
    data = await state.get_data()
    selected_topic_id = int(data.get("selected_topic_id", 0)) or None
    pages = await db.list_theory_pages(topic_id=selected_topic_id)
    if not pages:
        await message.answer("Теория по выбранной теме пока не добавлена. Нажмите «Начать решение».", reply_markup=theory_keyboard(False))
        return

    index = int(data.get("theory_index", 0)) + 1
    if index >= len(pages):
        await state.update_data(theory_index=max(len(pages) - 1, 0))
        await message.answer("Теория закончилась. Нажмите «Начать решение».", reply_markup=theory_keyboard(False))
        return

    await state.update_data(theory_index=index)
    await _send_theory_page(message, pages, index)


@router.message(StudentFlow.showing_theory, F.text == "Начать решение")
async def start_solving_after_theory(message: Message, state: FSMContext, db: Database) -> None:
    data = await state.get_data()
    selected_topic_id = int(data.get("selected_topic_id", 0)) or None
    await _send_learning_task(message, state, db)




@router.callback_query(F.data == "learning:show_answer")
async def learning_show_answer(callback: CallbackQuery, state: FSMContext, renderer: FormulaRenderer) -> None:
    if not callback.message:
        return

    data = await state.get_data()
    answer_text = data.get("current_answer")
    if not answer_text:
        await callback.answer("Ответ для текущего задания недоступен", show_alert=True)
        return

    try:
        image_bytes = renderer.render_integral_image(str(answer_text))
    except Exception as exc:  # noqa: BLE001
        await callback.answer(f"Не удалось отрисовать ответ: {exc}", show_alert=True)
        return

    image = BufferedInputFile(image_bytes, filename="learning_answer.png")
    await callback.message.answer_photo(image, caption="Ответ к текущему заданию")
    await callback.answer()


@router.message(F.text == "Следующее задание")
async def student_next_learning_task(message: Message, state: FSMContext, db: Database) -> None:
    data = await state.get_data()
    selected_topic_id = int(data.get("selected_topic_id", 0)) or None
    await _send_learning_task(message, state, db)


@router.message(F.text == "Завершить обучение")
async def student_finish_learning(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer("Режим обучения завершён.", reply_markup=student_menu_keyboard())


@router.message(F.text == "Режим тестирования")
async def student_testing_mode(message: Message, state: FSMContext, db: Database) -> None:
    student = await _get_student_or_notify(message, db)
    if not student:
        return

    solved_count = await db.count_student_answers_by_mode(student.id, "testing")
    if solved_count >= 10:
        await state.clear()
        await message.answer("Вы уже завершили тестирование (10 из 10).", reply_markup=student_menu_keyboard())
        return

    topics = await db.list_topics()
    if not topics:
        await message.answer("В базе нет тем.")
        return

    await state.set_state(StudentFlow.choosing_topic)
    await state.update_data(pending_mode="testing")
    await message.answer("Выберите тему:", reply_markup=_student_topics_keyboard(topics))


@router.message(StudentFlow.waiting_learning_answer, F.photo)
@router.message(StudentFlow.waiting_learning_answer, F.document)
async def learning_answer_first_attempt(message: Message, state: FSMContext, db: Database, llm: GeminiClient) -> None:
    await _process_learning_attempt(message, state, db, llm, is_retry=False)


@router.message(StudentFlow.waiting_learning_retry_answer, F.photo)
@router.message(StudentFlow.waiting_learning_retry_answer, F.document)
async def learning_answer_retry_attempt(message: Message, state: FSMContext, db: Database, llm: GeminiClient) -> None:
    await _process_learning_attempt(message, state, db, llm, is_retry=True)


@router.message(StudentFlow.learning_incorrect_options, F.text == "Подсказка")
async def show_hint(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    hint = data.get("current_hint") or "Подсказка пока не добавлена для этого задания."
    await message.answer(str(hint), reply_markup=learning_incorrect_keyboard())


@router.message(StudentFlow.learning_incorrect_options, F.text == "Попробовать ещё раз")
async def retry_learning(message: Message, state: FSMContext) -> None:
    await state.set_state(StudentFlow.waiting_learning_retry_answer)
    await message.answer("Отправьте фото решения ещё раз.", reply_markup=waiting_answer_keyboard())


@router.message(StudentFlow.waiting_testing_answer, F.photo)
@router.message(StudentFlow.waiting_testing_answer, F.document)
async def testing_answer_photo(message: Message, state: FSMContext, db: Database, llm: GeminiClient) -> None:
    student = await _get_student_or_notify(message, db)
    if not student:
        return

    file_id = _extract_image_file_id(message)
    if not file_id:
        await message.answer("Нужно отправить изображение (фото или документ-картинку).")
        return

    task_id = await _get_task_id_or_reset(message, state)
    if task_id is None:
        return

    state_data = await state.get_data()
    expected_answer = str(state_data.get("current_answer") or "").strip()
    if not expected_answer:
        await message.answer("Не удалось найти эталонный ответ для проверки. Попробуйте другое задание.")
        return

    progress_message = await message.answer("⏳ Проверяю фото ответа…")
    check = await _check_student_answer(message, llm, file_id, expected_answer, f"Задание #{task_id}")
    await _finish_progress_message(progress_message, check)
    if check is None:
        return

    if check.verdict == "unreadable":
        await message.answer(
            "Не удалось уверенно распознать ответ на фото. Пожалуйста, перефотографируйте и отправьте ещё раз."
        )
        return

    await db.save_answer(student.id, task_id, "testing", answer_image_file_id=file_id, is_correct=(check.verdict == "correct"))

    solved_count = await db.count_student_answers_by_mode(student.id, "testing")
    if solved_count >= 10:
        await state.clear()
        await message.answer("Тестирование завершено: 10 из 10 задач отправлены.", reply_markup=student_menu_keyboard())
        return

    await _send_testing_task(message, state, db, student)


@router.message(StudentFlow.waiting_learning_answer, F.text == "Пропустить задание")
@router.message(StudentFlow.waiting_learning_retry_answer, F.text == "Пропустить задание")
@router.message(StudentFlow.learning_incorrect_options, F.text == "Пропустить задание")
@router.message(StudentFlow.waiting_testing_answer, F.text == "Пропустить задание")
async def skip_task(message: Message, state: FSMContext, db: Database) -> None:
    student = await _get_student_or_notify(message, db)
    if not student:
        return

    state_value = await state.get_state()
    task_id = await _get_task_id_or_reset(message, state)
    if task_id is None:
        return

    mode = "testing" if state_value == StudentFlow.waiting_testing_answer.state else "learning"
    await db.save_answer(student.id, task_id, mode, answer_image_file_id=None, is_correct=False, is_skipped=True)

    if mode == "learning":
        await state.clear()
        await message.answer("Задание пропущено.", reply_markup=learning_after_answer_keyboard())
        return

    solved_count = await db.count_student_answers_by_mode(student.id, "testing")
    if solved_count >= 10:
        await state.clear()
        await message.answer("Тестирование завершено: 10 из 10 задач обработаны.", reply_markup=student_menu_keyboard())
        return

    await _send_testing_task(message, state, db, student)


@router.message(StudentFlow.waiting_learning_answer)
@router.message(StudentFlow.waiting_learning_retry_answer)
@router.message(StudentFlow.learning_incorrect_options)
@router.message(StudentFlow.waiting_testing_answer)
async def waiting_photo_only(message: Message) -> None:
    await message.answer("Пожалуйста, отправьте фото ответа или нажмите «Пропустить задание».")


async def _build_candidate(
    llm: GeminiClient,
    renderer: FormulaRenderer,
    topic_prompt: str,
    mode: str,
    index: int,
    forbidden_formulas: list[str] | None = None,
) -> dict[str, str | bytes | None]:
    effective_prompt = topic_prompt
    if forbidden_formulas:
        forbidden_lines = "\n".join(f"- {formula}" for formula in forbidden_formulas[:10])
        effective_prompt = (
            f"{topic_prompt}\n\n"
            "Важно: не повторяй следующие 10 последних формул для этого преподавателя (в том числе в эквивалентной форме):\n"
            f"{forbidden_lines}\n"
            "Сгенерируй новый, отличный пример."
        )

    generated = await llm.generate_task(effective_prompt)
    image_bytes = renderer.render_integral_image(generated.latex_integral)
    text = f"Вычислите интеграл: {generated.latex_integral}"
    hint = _clean_student_text(generated.hint) if mode == "learning" else None
    answer = _clean_student_text(generated.answer)
    return {"text": text, "hint": hint, "answer": answer, "image_bytes": image_bytes, "latex": generated.latex_integral, "index": str(index)}


async def _show_generated_candidate(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    topic_title = str(data["topic_title"])
    mode = str(data["mode"])
    generated_index = int(data.get("generated_index", 0))
    total_to_generate = int(data["total_to_generate"])

    candidates = list(data.get("generated_candidates", []))
    if generated_index >= len(candidates):
        await message.answer("Кандидатов больше нет.", reply_markup=teacher_menu_keyboard())
        await state.clear()
        return

    candidate = candidates[generated_index]
    img = BufferedInputFile(candidate["image_bytes"], filename=f"candidate_{generated_index+1}.png")

    text = (
        f"Задание #{generated_index + 1}\n"
        f"Тема: {topic_title}\n"
        f"Режим: {'Обучение' if mode == 'learning' else 'Тестирование'}"
    )
    if mode == "learning":
        hint = _clean_student_text(str(candidate.get("hint") or ""))
        answer = _clean_student_text(str(candidate.get("answer") or ""))
        if hint:
            text += f"\nПодсказка: {hint}"
        if answer:
            text += f"\nОтвет: {answer}"
    sent = await message.answer_photo(img, caption=text, reply_markup=_generated_review_keyboard())

    file_id = sent.photo[-1].file_id if sent.photo else None
    await state.update_data(
        candidate_text=candidate["text"],
        candidate_hint=candidate["hint"],
        candidate_answer=candidate["answer"],
        candidate_image_file_id=file_id,
    )


def _student_topics_keyboard(topics: list[Topic]) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text=topic.title, callback_data=f"student_topic:{topic.id}")] for topic in topics]
    )


def _topics_keyboard(topics: list[Topic]) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text=topic.title, callback_data=f"teacher_topic:{topic.id}")] for topic in topics]
    )


def _modes_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Обучение", callback_data="teacher_mode:learning")],
            [InlineKeyboardButton(text="Тестирование", callback_data="teacher_mode:testing")],
        ]
    )


def _generated_review_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="✅ Подтвердить", callback_data="teacher_gen:approve")],
            [InlineKeyboardButton(text="🔁 Сгенерировать заново", callback_data="teacher_gen:regenerate")],
            [InlineKeyboardButton(text="⏭️ Пропустить", callback_data="teacher_gen:skip")],
            [InlineKeyboardButton(text="❌ Отменить", callback_data="teacher_gen:cancel")],
        ]
    )




def _learning_answer_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="Показать ответ", callback_data="learning:show_answer")]]
    )

def _pool_list_keyboard(tasks: list[Task], page: int, page_size: int = 10) -> InlineKeyboardMarkup:
    total_pages = max((len(tasks) - 1) // page_size + 1, 1)
    current_page = min(max(page, 0), total_pages - 1)

    start = current_page * page_size
    end = start + page_size
    current_tasks = tasks[start:end]

    rows = [
        [
            InlineKeyboardButton(
                text=f"#{task.id} | {task.topic_title} | {'обуч.' if task.mode == 'learning' else 'тест'}",
                callback_data=f"pool_open:{task.id}",
            )
        ]
        for task in current_tasks
    ]

    rows.append(
        [
            InlineKeyboardButton(text="⬅️", callback_data="pool_list_nav:prev"),
            InlineKeyboardButton(text=f"{current_page + 1}/{total_pages}", callback_data="pool_noop"),
            InlineKeyboardButton(text="➡️", callback_data="pool_list_nav:next"),
        ]
    )

    return InlineKeyboardMarkup(inline_keyboard=rows)


async def _send_pool_list(message: Message, tasks: list[Task], page: int, edit: bool = False) -> None:
    if edit:
        try:
            await message.edit_text("Ваш пул заданий:", reply_markup=_pool_list_keyboard(tasks, page=page))
            return
        except TelegramBadRequest:
            pass
    await message.answer("Ваш пул заданий:", reply_markup=_pool_list_keyboard(tasks, page=page))


def _pool_nav_keyboard(index: int, total: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="⬅️", callback_data="pool_nav:prev"),
                InlineKeyboardButton(text=f"{index + 1}/{total}", callback_data="pool_noop"),
                InlineKeyboardButton(text="➡️", callback_data="pool_nav:next"),
            ],
            [InlineKeyboardButton(text="↩️ К списку", callback_data="pool_back")],
        ]
    )


async def _send_pool_task(message: Message, task: Task, index: int, total: int) -> None:
    text = (
        f"Задание #{task.id}\n"
        f"Тема: {task.topic_title}\n"
        f"Режим: {'Обучение' if task.mode == 'learning' else 'Тестирование'}"
    )

    if not task.task_image_file_id:
        text += f"\n{task.task_text}"
    if task.task_image_file_id:
        await message.answer_photo(task.task_image_file_id, caption=text, reply_markup=_pool_nav_keyboard(index, total))
    else:
        await message.answer(text, reply_markup=_pool_nav_keyboard(index, total))


async def _send_learning_task(message: Message, state: FSMContext, db: Database, topic_id: int | None = None, **_: object) -> None:
    student = await _get_student_or_notify(message, db)
    if not student:
        return

    if topic_id is None:
        state_data = await state.get_data()
        topic_id = int(state_data.get("selected_topic_id", 0)) or None
    if topic_id is None:
        await message.answer("Сначала выберите тему.")
        return

    task = await db.get_next_task(student.id, student.teacher_id, "learning", topic_id=topic_id)
    if not task:
        await state.clear()
        await message.answer("Доступные задания для обучения закончились.", reply_markup=student_menu_keyboard())
        return

    await state.set_state(StudentFlow.waiting_learning_answer)
    await state.update_data(task_id=task.id, current_hint=task.task_hint_text, current_answer=task.task_answer_text, selected_topic_id=topic_id)
    await _send_task_with_prompt(message, task)


async def _send_testing_task(message: Message, state: FSMContext, db: Database, student: Student, topic_id: int | None = None, **_: object) -> None:
    if topic_id is None:
        state_data = await state.get_data()
        topic_id = int(state_data.get("selected_topic_id", 0)) or None
    if topic_id is None:
        await message.answer("Сначала выберите тему.")
        return

    task = await db.get_next_task(student.id, student.teacher_id, "testing", topic_id=topic_id)
    if not task:
        await state.clear()
        await message.answer(
            "Задания для тестирования закончились раньше лимита. Тестирование завершено.",
            reply_markup=student_menu_keyboard(),
        )
        return

    await state.set_state(StudentFlow.waiting_testing_answer)
    await state.update_data(task_id=task.id, current_answer=task.task_answer_text, selected_topic_id=topic_id)
    await _send_task_with_prompt(message, task)


async def _send_theory_page(message: Message, pages: list[TheoryPage], index: int) -> None:
    page = pages[index]
    text = f"{page.title}\n\n{page.text_content}"
    has_next = index < len(pages) - 1

    if page.image_file_id:
        await message.answer_photo(page.image_file_id, caption=text, reply_markup=theory_keyboard(has_next))
        return

    await message.answer(text, reply_markup=theory_keyboard(has_next))


async def _send_task_with_prompt(message: Message, task: Task) -> None:
    lines = [f"Тема: {task.topic_title}", f"Задание #{task.id}:"]

    if not task.task_image_file_id:
        lines.append(task.task_text)

    answer_button = waiting_answer_keyboard() if task.mode == "testing" else None
    if task.mode == "learning" and task.task_answer_text:
        lines.append("Чтобы посмотреть решение-ответ, нажмите кнопку ниже.")
        answer_button = _learning_answer_keyboard()

    lines.append("Пришлите фото ответа.")
    text = "\n".join(lines)

    if task.task_image_file_id:
        await message.answer_photo(task.task_image_file_id, caption=text, reply_markup=answer_button)
    else:
        await message.answer(text, reply_markup=answer_button)

    if task.mode == "learning":
        await message.answer("Отправьте фото с ответом или нажмите «Пропустить задание».", reply_markup=waiting_answer_keyboard())


async def _process_learning_attempt(message: Message, state: FSMContext, db: Database, llm: GeminiClient, is_retry: bool) -> None:
    student = await _get_student_or_notify(message, db)
    if not student:
        return

    file_id = _extract_image_file_id(message)
    if not file_id:
        await message.answer("Нужно отправить изображение (фото или документ-картинку).")
        return

    task_id = await _get_task_id_or_reset(message, state)
    if task_id is None:
        return

    state_data = await state.get_data()
    expected_answer = str(state_data.get("current_answer") or "").strip()
    if not expected_answer:
        await message.answer("Не удалось найти эталонный ответ для проверки. Попробуйте следующее задание.")
        return

    progress_message = await message.answer("⏳ Проверяю фото ответа…")
    check = await _check_student_answer(message, llm, file_id, expected_answer, f"Задание #{task_id}")
    await _finish_progress_message(progress_message, check)
    if check is None:
        return

    if check.verdict == "unreadable":
        await state.set_state(StudentFlow.waiting_learning_retry_answer if is_retry else StudentFlow.waiting_learning_answer)
        await message.answer(
            "Не удалось уверенно распознать ответ на фото. Пожалуйста, перефотографируйте и отправьте ещё раз.",
            reply_markup=waiting_answer_keyboard(),
        )
        return

    if check.verdict == "incorrect":
        await state.set_state(StudentFlow.learning_incorrect_options)
        await message.answer(
            f"Ответ неверный. {check.feedback}\n\nВыберите действие: подсказка, попробовать ещё раз или пропустить задание.",
            reply_markup=learning_incorrect_keyboard(),
        )
        return

    await db.save_answer(student.id, task_id, "learning", answer_image_file_id=file_id, is_correct=True)
    await state.clear()
    await message.answer("Отлично! Ответ верный 🎉", reply_markup=learning_after_answer_keyboard())


async def _get_task_id_or_reset(message: Message, state: FSMContext) -> int | None:
    state_data = await state.get_data()
    task_id = state_data.get("task_id")
    if not task_id:
        await state.clear()
        await message.answer("Не удалось определить текущее задание. Выберите режим заново.")
        return None
    return int(task_id)




def _extract_formula_from_task_text(task_text: str) -> str:
    prefix = "Вычислите интеграл:"
    text = task_text.strip()
    if text.startswith(prefix):
        return text[len(prefix):].strip()
    return text

def _clean_student_text(value: str) -> str:
    cleaned = value.strip()
    if cleaned.startswith("$") and cleaned.endswith("$") and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()
    cleaned = cleaned.replace("$", "")
    if cleaned.lower().startswith("подсказка:"):
        cleaned = cleaned.split(":", 1)[1].strip()
    return cleaned




async def _finish_progress_message(progress_message: Message, check_result) -> None:
    try:
        if check_result is not None and check_result.verdict == "unreadable":
            await progress_message.edit_text("📷 Фото не удалось распознать.")
        else:
            await progress_message.delete()
    except Exception:  # noqa: BLE001
        pass

async def _check_student_answer(
    message: Message,
    llm: GeminiClient,
    file_id: str,
    expected_answer: str,
    task_text: str,
):
    if not llm.enabled:
        await message.answer("Проверка ответов временно недоступна: LLM не настроена.")
        return None

    try:
        image_bytes, mime_type = await _download_telegram_image(message, file_id)
        image_base64 = base64.b64encode(image_bytes).decode("ascii")
        image_data_uri = f"data:{mime_type};base64,{image_base64}"
        return await llm.check_student_answer(image_data_uri, expected_answer=expected_answer, task_text=task_text)
    except Exception as exc:  # noqa: BLE001
        await message.answer(f"Не удалось проверить ответ через LLM: {exc}")
        return None


async def _download_telegram_image(message: Message, file_id: str) -> tuple[bytes, str]:
    telegram_file = await message.bot.get_file(file_id)
    destination = io.BytesIO()
    await message.bot.download_file(telegram_file.file_path, destination=destination)
    mime_type = "image/jpeg"
    if message.document and message.document.file_id == file_id and message.document.mime_type:
        mime_type = message.document.mime_type
    return destination.getvalue(), mime_type


def _extract_image_file_id(message: Message) -> str | None:
    if message.photo:
        return message.photo[-1].file_id
    if message.document and message.document.mime_type and message.document.mime_type.startswith("image/"):
        return message.document.file_id
    return None


async def _get_student_or_notify(message: Message, db: Database) -> Student | None:
    user = message.from_user
    if not user:
        return None
    student = await db.get_student_by_telegram_id(user.id)
    if not student:
        await message.answer("Доступ запрещён: студент не найден в базе. Обратитесь к преподавателю.")
        return None
    return student


async def _get_teacher_or_notify(message: Message, db: Database) -> Teacher | None:
    user = message.from_user
    if not user:
        return None
    teacher = await db.get_teacher_by_telegram_id(user.id)
    if not teacher:
        await message.answer("Доступ запрещён: команда доступна только преподавателю.")
        return None
    return teacher


async def _get_teacher_from_callback(callback: CallbackQuery, db: Database) -> Teacher | None:
    if not callback.from_user:
        return None
    return await db.get_teacher_by_telegram_id(callback.from_user.id)


async def _get_student_from_callback_or_notify(callback: CallbackQuery, db: Database) -> Student | None:
    if not callback.from_user:
        return None

    student = await db.get_student_by_telegram_id(callback.from_user.id)
    if not student:
        await callback.answer("Доступ запрещён: студент не найден в базе.", show_alert=True)
        return None
    return student
