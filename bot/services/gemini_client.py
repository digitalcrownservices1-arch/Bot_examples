from __future__ import annotations

import asyncio
import json
import ssl
from dataclasses import dataclass
from typing import Any

import aiohttp


@dataclass(slots=True)
class GeneratedTask:
    latex_integral: str
    hint: str
    answer: str


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        endpoint: str,
        model: str,
        verify_ssl: bool = True,
        status_endpoint_template: str = "https://api.gen-api.ru/api/v1/request/get/{request_id}",
    ) -> None:
        self._api_key = api_key
        self._endpoint = endpoint
        self._model = model
        self._verify_ssl = verify_ssl
        self._status_endpoint_template = status_endpoint_template

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    async def generate_task(self, prompt: str) -> GeneratedTask:
        if not self.enabled:
            raise RuntimeError("GEMINI_API_KEY is not configured")

        if not prompt.strip():
            raise RuntimeError("LLM prompt is empty")

        payload = {
            "is_sync": True,
            "model": self._model,
            "temperature": 0.4,
            "max_tokens": 500,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        data = await self._post_json(self._endpoint, payload, headers, verify_ssl=self._verify_ssl)
        data = await self._resolve_async_if_needed(data, headers)
        content = self._extract_content(data)
        return self._parse_generation(content)

    async def _resolve_async_if_needed(self, data: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        status = str(data.get("status", "")).lower()
        request_id = data.get("request_id")

        if status in {"starting", "processing", "queued"} and request_id:
            for _ in range(30):
                status_url = self._status_endpoint_template.replace("{request_id}", str(request_id))
                polled = await self._post_json(status_url, {}, headers, verify_ssl=self._verify_ssl, use_get=True)
                polled_status = str(polled.get("status", "")).lower()
                if polled_status == "success":
                    return polled
                if polled_status in {"failed", "error"}:
                    raise RuntimeError(f"LLM request failed: {self._short(polled)}")
                await asyncio.sleep(1)

            raise RuntimeError("LLM request is still processing too long. Please retry.")

        return data

    async def _post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        verify_ssl: bool,
        use_get: bool = False,
    ) -> dict[str, Any]:
        connector = aiohttp.TCPConnector(ssl=verify_ssl)
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60), connector=connector) as session:
                if use_get:
                    async with session.get(url, headers=headers) as response:
                        body = await response.text()
                else:
                    async with session.post(url, json=payload, headers=headers) as response:
                        body = await response.text()

                if response.status >= 400:
                    raise RuntimeError(f"LLM error {response.status}: {body[:500]}")
                return json.loads(body)
        except aiohttp.ClientConnectorCertificateError:
            if verify_ssl:
                return await self._post_json(url, payload, headers, verify_ssl=False, use_get=use_get)
            raise RuntimeError(
                "SSL certificate verification failed for LLM endpoint. "
                "Set GEMINI_SSL_VERIFY=false or install proper CA certificates."
            )
        except ssl.SSLCertVerificationError:
            if verify_ssl:
                return await self._post_json(url, payload, headers, verify_ssl=False, use_get=use_get)
            raise RuntimeError(
                "SSL certificate verification failed for LLM endpoint. "
                "Set GEMINI_SSL_VERIFY=false or install proper CA certificates."
            )

    def _extract_content(self, data: dict[str, Any]) -> str:
        candidates: list[str] = []

        def walk(node: Any) -> None:
            if isinstance(node, str):
                if "пример:" in node.lower() and "подсказка:" in node.lower() and "ответ:" in node.lower():
                    candidates.append(node)
                return
            if isinstance(node, dict):
                for value in node.values():
                    walk(value)
                return
            if isinstance(node, list):
                for item in node:
                    walk(item)

        walk(data)

        if candidates:
            return candidates[0]

        raise RuntimeError(f"Unexpected LLM response format: {self._short(data)}")

    def _parse_generation(self, text: str) -> GeneratedTask:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 3:
            raise RuntimeError(f"Invalid LLM format: {text}")

        p = next((line for line in lines if line.lower().startswith("пример:")), "")
        h = next((line for line in lines if line.lower().startswith("подсказка:")), "")
        a = next((line for line in lines if line.lower().startswith("ответ:")), "")
        if not (p and h and a):
            raise RuntimeError(f"Missing expected fields in LLM response: {text}")

        integral_raw = p.split(":", 1)[1].strip()
        hint = h.split(":", 1)[1].strip()
        answer_raw = a.split(":", 1)[1].strip()

        integral = self._sanitize_latex(integral_raw)
        answer = self._sanitize_latex(answer_raw)
        if not integral or not answer:
            raise RuntimeError("Generated task has empty formula/answer")

        return GeneratedTask(latex_integral=integral, hint=hint, answer=answer)

    def _sanitize_latex(self, value: str) -> str:
        cleaned = value.strip()

        # unwrap optional markdown code fences
        if cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = cleaned.strip("`").strip()

        # remove optional raw-string wrappers: r"..." / r'...'
        if (cleaned.startswith('r"') and cleaned.endswith('"')) or (
            cleaned.startswith("r'") and cleaned.endswith("'")
        ):
            cleaned = cleaned[2:-1].strip()

        # remove optional $...$ wrappers
        if cleaned.startswith("$") and cleaned.endswith("$") and len(cleaned) >= 2:
            cleaned = cleaned[1:-1].strip()

        # sometimes model returns quoted latex
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (
            cleaned.startswith("'") and cleaned.endswith("'")
        ):
            cleaned = cleaned[1:-1].strip()

        return cleaned

    def _short(self, value: Any) -> str:
        raw = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
        return raw[:500]
