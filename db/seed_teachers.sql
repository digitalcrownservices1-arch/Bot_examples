INSERT INTO teachers (name, telegram_user_id)
VALUES
    ('Преподаватель1', 5397130405),
    ('Преподаватель2', 5248186952),
    ('Преподаватель3', 333333333)
ON CONFLICT (name) DO UPDATE
SET telegram_user_id = EXCLUDED.telegram_user_id;

INSERT INTO groups (name, teacher_id)
SELECT 'Группа1', t.id FROM teachers t WHERE t.name = 'Преподаватель1'
ON CONFLICT (name) DO NOTHING;

INSERT INTO groups (name, teacher_id)
SELECT 'Группа2', t.id FROM teachers t WHERE t.name = 'Преподаватель2'
ON CONFLICT (name) DO NOTHING;

INSERT INTO groups (name, teacher_id)
SELECT 'Группа3', t.id FROM teachers t WHERE t.name = 'Преподаватель3'
ON CONFLICT (name) DO NOTHING;

INSERT INTO topics (title, llm_prompt)
VALUES
    (
        'Метод замены переменной',
        'Ты — генератор задач по математическому анализу (интегралы). Сгенерируй 1 задачу по теме: метод замены переменной. Требования: 1) Дай ровно ОДИН пример на вычисление неопределённого интеграла, чтобы он решался именно методом из темы. 2) Верни результат СТРОГО в 3 строках и строго с такими префиксами: пример:, подсказка:, ответ:. 3) В строке "пример:" укажи только LaTeX-формулу интеграла (БЕЗ знака $...$), которую можно напрямую вставить в Python как raw string: r"...". Никаких слов и пояснений. Используй только команды, которые понимает matplotlib mathtext: \int, \frac, ^, _, \sqrt, \ln, \sin, \cos, \tan, e^{...}, \cdot, \left( \right), \,. НЕ используй \textbf, \text, \begin, \end, \displaystyle, \cases и т.п. 4) "подсказка:" — 1 короткое предложение (до 12 слов), как решать. 5) "ответ:" — итоговый результат интеграла в LaTeX (БЕЗ $...$), обязательно с "+ C". 6) Сложность уровня 1–2 курса. Запрещено: дополнительные строки, нумерации, пустые строки, любые символы кроме этих 3 строк.'
    ),
    (
        'Метод подведения под знак дифференциала',
        'Ты — генератор задач по математическому анализу (интегралы). Сгенерируй 1 задачу по теме: метод подведения под знак дифференциала. Требования: 1) Дай ровно ОДИН пример на вычисление неопределённого интеграла, чтобы он решался именно методом из темы. 2) Верни результат СТРОГО в 3 строках и строго с такими префиксами: пример:, подсказка:, ответ:. 3) В строке "пример:" укажи только LaTeX-формулу интеграла (БЕЗ знака $...$), которую можно напрямую вставить в Python как raw string: r"...". Никаких слов и пояснений. Используй только команды, которые понимает matplotlib mathtext: \int, \frac, ^, _, \sqrt, \ln, \sin, \cos, \tan, e^{...}, \cdot, \left( \right), \,. НЕ используй \textbf, \text, \begin, \end, \displaystyle, \cases и т.п. 4) "подсказка:" — 1 короткое предложение (до 12 слов), как решать. 5) "ответ:" — итоговый результат интеграла в LaTeX (БЕЗ $...$), обязательно с "+ C". 6) Сложность уровня 1–2 курса. Запрещено: дополнительные строки, нумерации, пустые строки, любые символы кроме этих 3 строк.'
    ),
    (
        'Метод интегрирования по частям',
        'Ты — генератор задач по математическому анализу (интегралы). Сгенерируй 1 задачу по теме: метод интегрирования по частям. Требования: 1) Дай ровно ОДИН пример на вычисление неопределённого интеграла, чтобы он решался именно методом из темы. 2) Верни результат СТРОГО в 3 строках и строго с такими префиксами: пример:, подсказка:, ответ:. 3) В строке "пример:" укажи только LaTeX-формулу интеграла (БЕЗ знака $...$), которую можно напрямую вставить в Python как raw string: r"...". Никаких слов и пояснений. Используй только команды, которые понимает matplotlib mathtext: \int, \frac, ^, _, \sqrt, \ln, \sin, \cos, \tan, e^{...}, \cdot, \left( \right), \,. НЕ используй \textbf, \text, \begin, \end, \displaystyle, \cases и т.п. 4) "подсказка:" — 1 короткое предложение (до 12 слов), как решать. 5) "ответ:" — итоговый результат интеграла в LaTeX (БЕЗ $...$), обязательно с "+ C". 6) Сложность уровня 1–2 курса. Запрещено: дополнительные строки, нумерации, пустые строки, любые символы кроме этих 3 строк.'
    )
ON CONFLICT (title) DO UPDATE
SET llm_prompt = EXCLUDED.llm_prompt;

INSERT INTO theory_pages (page_order, title, text_content, image_file_id)
VALUES
    (1, 'Теория: метод замены переменной', 'Если интеграл содержит сложную функцию и её производную, делаем замену u = g(x).', 'AgACAgIAAxkBAAIG6WmhM2003FM7TAQWuTef8e4JywKsAAJ0FmsbxDQJSb8I9lm_Y8YmAQADAgADeAADOgQ'),
    (2, 'Теория: метод по частям', 'Используем формулу ∫u dv = uv - ∫v du. Выбираем u так, чтобы du упрощался.', 'AgACAgIAAxkBAAIGYmmgm3uN_FKYSAABoH9C4KiRblx3_QACoBVrGyNj-Uh23NLY0op7SQEAAwIAA3kAAzoE')
ON CONFLICT DO NOTHING;

WITH student_seed AS (
    SELECT 'Группа1'::TEXT AS group_name, 742381892::BIGINT AS telegram_user_id, 1::INT AS student_no
    UNION ALL SELECT 'Группа1', 100000002, 2
    UNION ALL SELECT 'Группа2', 200000001, 1
    UNION ALL SELECT 'Группа3', 300000001, 1
)
INSERT INTO students (name, telegram_user_id, group_id)
SELECT
    'Студент' || regexp_replace(g.name, '\D', '', 'g') || '_' || ss.student_no,
    ss.telegram_user_id,
    g.id
FROM student_seed ss
JOIN groups g ON g.name = ss.group_name
ON CONFLICT (telegram_user_id) DO NOTHING;
