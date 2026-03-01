from __future__ import annotations

from dataclasses import dataclass

import asyncpg


@dataclass(slots=True)
class Teacher:
    id: int
    name: str
    telegram_user_id: int


@dataclass(slots=True)
class Student:
    id: int
    name: str
    telegram_user_id: int
    group_id: int
    teacher_id: int


@dataclass(slots=True)
class Topic:
    id: int
    title: str
    llm_prompt: str


@dataclass(slots=True)
class Task:
    id: int
    topic_title: str
    mode: str
    task_text: str
    task_hint_text: str | None
    task_answer_text: str | None
    task_image_file_id: str | None


@dataclass(slots=True)
class TheoryPage:
    id: int
    page_order: int
    title: str
    text_content: str
    image_file_id: str | None


class Database:
    def __init__(self, dsn: str):
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=5)
        await self.pool.execute("ALTER TABLE tasks ADD COLUMN IF NOT EXISTS task_answer_text TEXT")

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    @property
    def pool(self) -> asyncpg.Pool:
        if not self._pool:
            raise RuntimeError("Database pool is not initialized")
        return self._pool

    async def get_teacher_by_telegram_id(self, user_id: int) -> Teacher | None:
        row = await self.pool.fetchrow(
            """
            SELECT id, name, telegram_user_id
            FROM teachers
            WHERE telegram_user_id = $1
            """,
            user_id,
        )
        if not row:
            return None
        return Teacher(**dict(row))

    async def get_student_by_telegram_id(self, user_id: int) -> Student | None:
        row = await self.pool.fetchrow(
            """
            SELECT s.id, s.name, s.telegram_user_id, s.group_id, g.teacher_id
            FROM students s
            JOIN groups g ON g.id = s.group_id
            WHERE s.telegram_user_id = $1
            """,
            user_id,
        )
        if not row:
            return None
        return Student(**dict(row))

    async def list_topics(self) -> list[Topic]:
        rows = await self.pool.fetch(
            """
            SELECT id, title, llm_prompt
            FROM topics
            ORDER BY id
            """
        )
        return [Topic(**dict(row)) for row in rows]

    async def list_theory_pages(self) -> list[TheoryPage]:
        rows = await self.pool.fetch(
            """
            SELECT id, page_order, title, text_content, image_file_id
            FROM theory_pages
            ORDER BY page_order, id
            """
        )
        return [TheoryPage(**dict(row)) for row in rows]

    async def get_next_task(self, student_id: int, teacher_id: int, mode: str) -> Task | None:
        row = await self.pool.fetchrow(
            """
            SELECT t.id, tp.title AS topic_title, t.mode, t.task_text, t.task_hint_text, t.task_answer_text, t.task_image_file_id
            FROM tasks t
            JOIN topics tp ON tp.id = t.topic_id
            WHERE t.teacher_id = $1
              AND t.mode = $2
              AND NOT EXISTS (
                  SELECT 1
                  FROM answers a
                  WHERE a.task_id = t.id AND a.student_id = $3
              )
            ORDER BY t.id
            LIMIT 1
            """,
            teacher_id,
            mode,
            student_id,
        )
        if not row:
            return None
        return Task(**dict(row))

    async def create_task(
        self,
        topic_id: int,
        teacher_id: int,
        mode: str,
        task_text: str,
        task_hint_text: str | None,
        task_answer_text: str | None,
        task_image_file_id: str | None,
    ) -> int:
        return int(
            await self.pool.fetchval(
                """
                INSERT INTO tasks (topic_id, teacher_id, mode, task_text, task_hint_text, task_answer_text, task_image_file_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id
                """,
                topic_id,
                teacher_id,
                mode,
                task_text,
                task_hint_text,
                task_answer_text,
                task_image_file_id,
            )
        )

    async def list_teacher_tasks(self, teacher_id: int) -> list[Task]:
        rows = await self.pool.fetch(
            """
            SELECT t.id, tp.title AS topic_title, t.mode, t.task_text, t.task_hint_text, t.task_answer_text, t.task_image_file_id
            FROM tasks t
            JOIN topics tp ON tp.id = t.topic_id
            WHERE t.teacher_id = $1
            ORDER BY t.id
            """,
            teacher_id,
        )
        return [Task(**dict(row)) for row in rows]

    async def save_answer(
        self,
        student_id: int,
        task_id: int,
        mode: str,
        answer_image_file_id: str | None,
        is_correct: bool,
        is_skipped: bool = False,
    ) -> None:
        await self.pool.execute(
            """
            INSERT INTO answers (student_id, task_id, mode, answer_image_file_id, is_correct, is_skipped)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (student_id, task_id) DO UPDATE
            SET answer_image_file_id = EXCLUDED.answer_image_file_id,
                is_correct = EXCLUDED.is_correct,
                is_skipped = EXCLUDED.is_skipped,
                created_at = NOW()
            """,
            student_id,
            task_id,
            mode,
            answer_image_file_id,
            is_correct,
            is_skipped,
        )

    async def count_student_answers_by_mode(self, student_id: int, mode: str) -> int:
        return int(
            await self.pool.fetchval(
                """
                SELECT COUNT(*)
                FROM answers
                WHERE student_id = $1 AND mode = $2
                """,
                student_id,
                mode,
            )
            or 0
        )
