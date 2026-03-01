CREATE TABLE IF NOT EXISTS teachers (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    telegram_user_id BIGINT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS groups (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    teacher_id INTEGER NOT NULL REFERENCES teachers(id)
);

CREATE TABLE IF NOT EXISTS topics (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL UNIQUE,
    llm_prompt TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS theory_pages (
    id SERIAL PRIMARY KEY,
    page_order INTEGER NOT NULL,
    title TEXT NOT NULL,
    text_content TEXT NOT NULL,
    image_file_id TEXT
);

CREATE TABLE IF NOT EXISTS students (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    telegram_user_id BIGINT NOT NULL UNIQUE,
    group_id INTEGER NOT NULL REFERENCES groups(id)
);

CREATE TABLE IF NOT EXISTS tasks (
    id SERIAL PRIMARY KEY,
    topic_id INTEGER NOT NULL REFERENCES topics(id),
    teacher_id INTEGER NOT NULL REFERENCES teachers(id),
    mode TEXT NOT NULL CHECK (mode IN ('learning', 'testing')),
    task_text TEXT NOT NULL,
    task_hint_text TEXT,
    task_image_file_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (topic_id, teacher_id, mode, task_text)
);

CREATE TABLE IF NOT EXISTS answers (
    id SERIAL PRIMARY KEY,
    task_id INTEGER NOT NULL REFERENCES tasks(id),
    student_id INTEGER NOT NULL REFERENCES students(id),
    mode TEXT NOT NULL CHECK (mode IN ('learning', 'testing')),
    answer_image_file_id TEXT,
    is_correct BOOLEAN NOT NULL,
    is_skipped BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (student_id, task_id)
);

CREATE TABLE IF NOT EXISTS tasks (
    id SERIAL PRIMARY KEY,
    topic_id INTEGER NOT NULL REFERENCES topics(id),
    teacher_id INTEGER NOT NULL REFERENCES teachers(id),
    mode TEXT NOT NULL CHECK (mode IN ('learning', 'testing')),
    task_text TEXT NOT NULL,
    task_hint_text TEXT,
    task_answer_text TEXT,
    task_image_file_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (topic_id, teacher_id, mode, task_text)
);