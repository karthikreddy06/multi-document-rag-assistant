"""
LLM Generation Module.
Interacts with Ollama (llama3.2) to synthesize accurate, grounded answers from retrieved context.
"""

import re
import time
from typing import Any, Iterator, List, Optional
import httpx
import ollama

from app.config import settings
from app.generation.prompts import build_rag_prompt
from app.models import RetrievedChunk
from app.utils.logger import setup_logger

logger = setup_logger("generation.generator")


class LLMGenerator:
    """Production LLM response generator utilizing Ollama or Cloud AI API."""

    CONTRADICTION_PATTERN = re.compile(
        r"^(?:"
        r"I'll\s+follow\s+the\s+(?:operational\s+)?guidelines[^\n\.]*[\.\:]\s*|"
        r"(?:I\s+(?:can(?:not|'t)|am\s+unable\s+to)\s+provide\s+[^\n\.]+\b(?:as\s+it\s+is\s+not\s+present|not\s+available|not\s+found)\s+in\s+the\s+provided\s+(?:context|documents?)\.?\s*)|"
        r"(?:This\s+information\s+is\s+not\s+available\s+in\s+the\s+provided\s+(?:context|documents?)\.?\s*)|"
        r"(?:The\s+provided\s+documents\s+describe\s+[^\n\.]+\bdo\s+not\s+contain\s+source\s+code[^\n\.]*\.?\s*)"
        r")"
        r"(?:(?:However|Here\s+is|Based\s+on)[^\n\:]*[\:\.]\s*)?",
        re.IGNORECASE
    )

    @classmethod
    def clean_contradictory_preambles(cls, text: str) -> str:
        """
        Removes contradictory preambles where an LLM states information is unavailable
        yet immediately provides that exact information in the following paragraph.
        """
        if not text:
            return text
        m = cls.CONTRADICTION_PATTERN.match(text)
        if m and m.end() < len(text.strip()):
            cleaned = text[m.end():].strip()
            if cleaned:
                return cleaned
        return text

    @classmethod
    def filter_stream_tokens(cls, token_stream: Iterator[str]) -> Iterator[str]:
        """
        Filters token stream to eliminate contradictory preambles like:
        'I cannot provide the code... However, here is the code: ...'
        Flushes immediately if no contradictory prefix is detected.
        """
        buffer = []
        buffer_str = ""
        buffering = True

        for token in token_stream:
            if not buffering:
                yield token
                continue

            buffer.append(token)
            buffer_str += token

            # If buffer starts in a way that is clearly NOT contradictory, flush immediately
            if len(buffer_str) >= 20:
                lower = buffer_str.lower().strip()
                starts_suspicious = (
                    lower.startswith("i can") or
                    lower.startswith("i am unable") or
                    lower.startswith("i cannot") or
                    lower.startswith("this information is not") or
                    lower.startswith("the provided documents describe") or
                    lower.startswith("i'll follow")
                )
                if not starts_suspicious:
                    for t in buffer:
                        yield t
                    buffer = []
                    buffering = False
                    continue

            # If contradictory preamble is followed by "However" or code block
            if ("\n\n" in buffer_str or "```" in buffer_str) and len(buffer_str) >= 40:
                cleaned = cls.clean_contradictory_preambles(buffer_str)
                if cleaned != buffer_str:
                    if cleaned:
                        yield cleaned
                    buffer = []
                    buffering = False
                    continue

            # Safety cap on buffering
            if len(buffer_str) > 250:
                cleaned = cls.clean_contradictory_preambles(buffer_str)
                yield cleaned
                buffer = []
                buffering = False

        if buffer:
            cleaned = cls.clean_contradictory_preambles(buffer_str)
            yield cleaned

    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        temperature: float = 0.0,
        timeout: Optional[float] = None,
        num_predict: Optional[int] = None,
        num_ctx: Optional[int] = None,
        provider: Optional[str] = None,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.provider = (provider or settings.llm_provider).lower()
        if self.provider == "groq":
            if model:
                self.model = model
            elif settings.llm_model != "llama3.2:1b":
                self.model = settings.llm_model
            else:
                self.model = "openai/gpt-oss-20b"
            self.api_url = api_url or settings.llm_api_url or "https://api.groq.com/openai/v1"
        elif self.provider == "cloud":
            if model:
                self.model = model
            elif settings.llm_model != "llama3.2:1b":
                self.model = settings.llm_model
            else:
                self.model = "gemini-3.6-flash"
            self.api_url = api_url or settings.llm_api_url or "https://generativelanguage.googleapis.com/v1beta/openai/"
        else:
            self.model = model or settings.llm_model
            self.api_url = api_url or settings.llm_api_url
        self.host = host or settings.ollama_host
        self.temperature = temperature
        self.timeout = timeout or settings.ollama_timeout
        self.num_predict = num_predict if num_predict is not None else settings.llm_num_predict
        self.num_ctx = num_ctx if num_ctx is not None else settings.llm_num_ctx
        self.api_key = api_key or settings.effective_llm_api_key

        if self.provider == "ollama":
            self.client = ollama.Client(host=self.host, timeout=self.timeout)
        else:
            self.client = None

    def _get_generation_options(
        self,
        num_predict: Optional[int] = None,
        num_ctx: Optional[int] = None,
    ) -> dict:
        """Construct optimized Ollama options limiting context and generation buffer."""
        return {
            "temperature": self.temperature,
            "num_predict": num_predict if num_predict is not None else self.num_predict,
            "num_ctx": num_ctx if num_ctx is not None else self.num_ctx,
            "stop": ["<|eot_id|>", "<|start_header_id|>", "\n\nQuestion:"],
        }

    def _cloud_generate(self, prompt: str, num_predict: Optional[int] = None) -> str:
        """Execute non-streaming completion call to OpenAI-compatible cloud REST endpoint using httpx."""
        url = self.api_url.rstrip("/")
        if not url.endswith("/chat/completions") and not url.endswith("/completions"):
            url = f"{url}/chat/completions"

        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            if self.provider == "cloud" or "googleapis" in self.api_url:
                headers["x-goog-api-key"] = self.api_key

        max_tokens = num_predict if num_predict is not None else self.num_predict
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }

        with httpx.Client(timeout=self.timeout) as http_client:
            resp = http_client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        choices = data.get("choices", [])
        if not choices:
            return ""
        choice = choices[0]
        if "message" in choice:
            return choice["message"].get("content", "").strip()
        elif "text" in choice:
            return choice["text"].strip()
        return ""

    def _cloud_generate_stream(self, prompt: str, num_predict: Optional[int] = None) -> Iterator[str]:
        """Execute streaming completion call to OpenAI-compatible cloud REST endpoint using httpx."""
        import json
        url = self.api_url.rstrip("/")
        if not url.endswith("/chat/completions") and not url.endswith("/completions"):
            url = f"{url}/chat/completions"

        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            if self.provider == "cloud" or "googleapis" in self.api_url:
                headers["x-goog-api-key"] = self.api_key

        max_tokens = num_predict if num_predict is not None else self.num_predict
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        with httpx.Client(timeout=self.timeout) as http_client:
            with http_client.stream("POST", url, headers=headers, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    line_str = line.strip()
                    if line_str.startswith("data: "):
                        data_part = line_str[6:].strip()
                        if data_part == "[DONE]":
                            break
                        try:
                            parsed = json.loads(data_part)
                            choices = parsed.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                token = delta.get("content") or ""
                                if not token and "text" in choices[0]:
                                    token = choices[0].get("text") or ""
                                if token:
                                    yield token
                        except Exception:
                            continue

    def generate_answer(
        self,
        question: str,
        chunks: List[RetrievedChunk],
        plan: Optional[Any] = None,
        max_retries: int = 3,
        num_predict: Optional[int] = None,
        num_ctx: Optional[int] = None,
    ) -> str:
        """
        Generate an answer from the retrieved chunks with retry handling for transient drops.
        Supports dynamic generation budgets tailored to query complexity.
        """
        if not question or not question.strip():
            return "Please provide a valid question."

        if not chunks:
            logger.info("No context chunks provided to LLM generator.")
            return "I do not have enough information in the provided documents to answer this question."

        if num_predict is None and plan and getattr(plan, "generation_budget", None):
            num_predict = plan.generation_budget.num_predict
        if num_ctx is None and plan and getattr(plan, "generation_budget", None):
            num_ctx = plan.generation_budget.num_ctx

        intent = str(getattr(plan, "intent", "")) if plan else ""
        q_lower = question.lower()

        # Handle IMAGE_UNDERSTANDING queries when VLM is absent
        if intent == "image_understanding" or re.search(r"\b(?:tell\s+me\s+about\s+(?:that\s+|the\s+)?image|what\s+is\s+in\s+the\s+image|describe\s+the\s+image|explain\s+the\s+image|about\s+that\s+photo)\b", q_lower):
            img_chunk = next((c for c in chunks if c.metadata.get("is_image") or c.metadata.get("format") == "image" or "[Image File:" in c.text), None)
            if img_chunk:
                fn = img_chunk.metadata.get("filename", "the image")
                return f"Visual understanding is currently unavailable in the local model pipeline for '{fn}'; OCR produced no usable text content."

        # Handle OCR_QUERY
        if intent == "ocr_query" or re.search(r"\b(?:what\s+text\s+is\s+in\s+the\s+image|extract\s+text\s+from\s+the\s+image|read\s+text\s+in\s+the\s+image)\b", q_lower):
            img_chunk = next((c for c in chunks if c.metadata.get("is_image") or c.metadata.get("format") == "image" or "[Image File:" in c.text), None)
            if img_chunk:
                fn = img_chunk.metadata.get("filename", "the image")
                if "Note: This is an image file" in img_chunk.text or img_chunk.metadata.get("is_placeholder"):
                    return f"No text content could be extracted from '{fn}' using OCR."

        # Handle DOCUMENT_LIST_QUERY
        if intent == "document_list_query" or re.search(r"\b(?:what\s+files\s+are\s+attached|list\s+attached\s+files|what\s+documents\s+are\s+in\s+this\s+chat|which\s+documents\s+are\s+attached|files\s+attached\s+to\s+this\s+chat)\b", q_lower):
            fns = []
            for c in chunks:
                fn = c.metadata.get("filename")
                if fn and fn not in fns:
                    fns.append(fn)
            if fns:
                lines = [f"The following {len(fns)} documents are attached to this chat session:"]
                for idx, fn in enumerate(fns, 1):
                    lines.append(f"{idx}. {fn}")
                return "\n".join(lines)

        # Deterministic refusal if code was requested but evidence contains no code
        is_code_req = bool(
            (plan and getattr(plan, "is_code_request", False)) or
            re.search(r"\b(?:give|show|what\s+is|display|write)\s+(?:me\s+)?(?:the\s+)?(?:code|source\s+code|script|implementation)\b", question.lower())
        )
        if is_code_req:
            has_code = any(
                c.metadata.get("content_type") == "code" or
                "```" in c.text or
                bool(re.search(r"\b(?:def\s+\w+|class\s+\w+|import\s+\w+|input\(|print\(|return\b|while\b|for\s+\w+\s+in|console\.log|printf|System\.out)\b", c.text))
                for c in chunks
            )
            if not has_code:
                return "The provided documents describe this item conceptually but do not contain source code for it."

        # Check if the query asks for an ordered list/sequence already resolved with 100% item coverage
        resolved_items = getattr(plan, "resolved_items", None) if plan else None
        if resolved_items and len(resolved_items) >= 2:
            is_list_query = bool(
                re.search(
                    r"\b(?:tell\s+me|give\s+me|list|what\s+are|show\s+me|which|except|all|every|display)\b",
                    q_lower
                )
            )
            if is_list_query:
                logger.info(f"Deterministically formatting answer for {len(resolved_items)} resolved items.")
                lines = [f"{it.identifier}. {it.title}" for it in resolved_items]
                return "\n".join(lines)

        effective_question = question
        if plan and getattr(plan, "is_follow_up", False) and getattr(plan, "query", None):
            effective_question = plan.query
        elif resolved_items and len(resolved_items) == 1:
            it = resolved_items[0]
            item_ref = f"Item {it.identifier}: {it.title}"
            if item_ref.lower() not in effective_question.lower():
                replaced = re.sub(
                    r"\b(?:the\s+)?(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last|\d+(?:st|nd|rd|th))\s+(?:one|program|item|section|step|rule)\b",
                    item_ref,
                    effective_question,
                    flags=re.IGNORECASE
                )
                if replaced != effective_question:
                    effective_question = replaced
                else:
                    effective_question = f"{effective_question} ({item_ref})"

        lines_per_doc = getattr(plan, "lines_per_doc", None) if plan else None
        prompt = build_rag_prompt(effective_question, chunks, lines_per_doc=lines_per_doc)
        pred = num_predict if num_predict is not None else self.num_predict
        ctx = num_ctx if num_ctx is not None else self.num_ctx
        logger.info(f"Generating answer using {self.model} ({self.provider}) with {len(chunks)} context chunks (num_predict={pred}, num_ctx={ctx})...")

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                if self.provider == "ollama":
                    response = self.client.generate(
                        model=self.model,
                        prompt=prompt,
                        options=self._get_generation_options(num_predict=num_predict, num_ctx=num_ctx),
                    )
                    answer = response.get("response", "").strip()
                else:
                    answer = self._cloud_generate(prompt, num_predict=num_predict)

                cleaned_ans = self.clean_contradictory_preambles(answer)
                if lines_per_doc:
                    cleaned_ans = self.enforce_exact_line_count(cleaned_ans, lines_per_doc)
                return cleaned_ans
            except Exception as e:
                last_error = e
                logger.warning(
                    f"LLM generation attempt {attempt}/{max_retries} failed for {self.model} ({self.provider}): {e}. "
                    f"{'Retrying...' if attempt < max_retries else 'No more retries.'}"
                )
                if attempt < max_retries:
                    time.sleep(0.5 * attempt)

        logger.error(f"Error during LLM generation with {self.model} ({self.provider}) after {max_retries} attempts: {last_error}")
        raise RuntimeError(
            f"LLM generation failed ({self.provider}): {last_error}."
        ) from last_error

    def generate_answer_stream(
        self,
        question: str,
        chunks: List[RetrievedChunk],
        plan: Optional[Any] = None,
        num_predict: Optional[int] = None,
        num_ctx: Optional[int] = None,
    ):
        """
        Generate answer as an iterable stream of token strings for immediate user feedback.
        Supports dynamic generation budgets tailored to query complexity.
        """
        if not question or not question.strip():
            yield "Please provide a valid question."
            return

        if not chunks:
            logger.info("No context chunks provided to LLM streaming generator.")
            yield "I do not have enough information in the provided documents to answer this question."
            return

        if num_predict is None and plan and getattr(plan, "generation_budget", None):
            num_predict = plan.generation_budget.num_predict
        if num_ctx is None and plan and getattr(plan, "generation_budget", None):
            num_ctx = plan.generation_budget.num_ctx

        intent = str(getattr(plan, "intent", "")) if plan else ""
        q_lower = question.lower()

        # Handle IMAGE_UNDERSTANDING queries when VLM is absent
        if intent == "image_understanding" or re.search(r"\b(?:tell\s+me\s+about\s+(?:that\s+|the\s+)?image|what\s+is\s+in\s+the\s+image|describe\s+the\s+image|explain\s+the\s+image|about\s+that\s+photo)\b", q_lower):
            img_chunk = next((c for c in chunks if c.metadata.get("is_image") or c.metadata.get("format") == "image" or "[Image File:" in c.text), None)
            if img_chunk:
                fn = img_chunk.metadata.get("filename", "the image")
                yield f"Visual understanding is currently unavailable in the local model pipeline for '{fn}'; OCR produced no usable text content."
                return

        # Handle OCR_QUERY
        if intent == "ocr_query" or re.search(r"\b(?:what\s+text\s+is\s+in\s+the\s+image|extract\s+text\s+from\s+the\s+image|read\s+text\s+in\s+the\s+image)\b", q_lower):
            img_chunk = next((c for c in chunks if c.metadata.get("is_image") or c.metadata.get("format") == "image" or "[Image File:" in c.text), None)
            if img_chunk:
                fn = img_chunk.metadata.get("filename", "the image")
                if "Note: This is an image file" in img_chunk.text or img_chunk.metadata.get("is_placeholder"):
                    yield f"No text content could be extracted from '{fn}' using OCR."
                    return

        # Handle DOCUMENT_LIST_QUERY
        if intent == "document_list_query" or re.search(r"\b(?:what\s+files\s+are\s+attached|list\s+attached\s+files|what\s+documents\s+are\s+in\s+this\s+chat|which\s+documents\s+are\s+attached|files\s+attached\s+to\s+this\s+chat)\b", q_lower):
            fns = []
            for c in chunks:
                fn = c.metadata.get("filename")
                if fn and fn not in fns:
                    fns.append(fn)
            if fns:
                lines = [f"The following {len(fns)} documents are attached to this chat session:"]
                for idx, fn in enumerate(fns, 1):
                    lines.append(f"{idx}. {fn}")
                yield "\n".join(lines)
                return

        # Deterministic refusal if code was requested but evidence contains no code
        is_code_req = bool(
            (plan and getattr(plan, "is_code_request", False)) or
            re.search(r"\b(?:give|show|what\s+is|display|write)\s+(?:me\s+)?(?:the\s+)?(?:code|source\s+code|script|implementation)\b", question.lower())
        )
        if is_code_req:
            has_code = any(
                c.metadata.get("content_type") == "code" or
                "```" in c.text or
                bool(re.search(r"\b(?:def\s+\w+|class\s+\w+|import\s+\w+|input\(|print\(|return\b|while\b|for\s+\w+\s+in|console\.log|printf|System\.out)\b", c.text))
                for c in chunks
            )
            if not has_code:
                yield "The provided documents describe this item conceptually but do not contain source code for it."
                return

        # Check if the query asks for an ordered list/sequence already resolved with 100% item coverage
        resolved_items = getattr(plan, "resolved_items", None) if plan else None
        if resolved_items and len(resolved_items) >= 2:
            is_list_query = bool(
                re.search(
                    r"\b(?:tell\s+me|give\s+me|list|what\s+are|show\s+me|which|except|all|every|display)\b",
                    q_lower
                )
            )
            if is_list_query:
                lines = [f"{it.identifier}. {it.title}" for it in resolved_items]
                yield "\n".join(lines)
                return

        effective_question = question
        if plan and getattr(plan, "is_follow_up", False) and getattr(plan, "query", None):
            effective_question = plan.query
        elif resolved_items and len(resolved_items) == 1:
            it = resolved_items[0]
            item_ref = f"Item {it.identifier}: {it.title}"
            if item_ref.lower() not in effective_question.lower():
                replaced = re.sub(
                    r"\b(?:the\s+)?(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last|\d+(?:st|nd|rd|th))\s+(?:one|program|item|section|step|rule)\b",
                    item_ref,
                    effective_question,
                    flags=re.IGNORECASE
                )
                if replaced != effective_question:
                    effective_question = replaced
                else:
                    effective_question = f"{effective_question} ({item_ref})"

        lines_per_doc = getattr(plan, "lines_per_doc", None) if plan else None
        prompt = build_rag_prompt(effective_question, chunks, lines_per_doc=lines_per_doc)
        pred = num_predict if num_predict is not None else self.num_predict
        ctx = num_ctx if num_ctx is not None else self.num_ctx
        logger.info(f"Streaming answer using {self.model} ({self.provider}) with {len(chunks)} context chunks (num_predict={pred}, num_ctx={ctx})...")

        def raw_token_stream():
            if self.provider == "ollama":
                stream = self.client.generate(
                    model=self.model,
                    prompt=prompt,
                    stream=True,
                    options=self._get_generation_options(num_predict=num_predict, num_ctx=num_ctx),
                )
                for chunk in stream:
                    token = chunk.get("response", "")
                    if token:
                        yield token
            else:
                yield from self._cloud_generate_stream(prompt, num_predict=num_predict)

        yield from self.filter_stream_tokens(raw_token_stream())

    @classmethod
    def enforce_exact_line_count(cls, text: str, target_lines: int) -> str:
        """
        Clamps each document section in multi-document response to at most target_lines concise lines/bullet points.
        """
        if not text or target_lines <= 0:
            return text

        pattern = re.compile(
            r"(\*\*(?:DOCUMENT|Document)\s+\d+:[^\*\n]+\*\*|###\s+(?:DOCUMENT|Document)\s+\d+:[^\n]+)",
            re.IGNORECASE
        )
        parts = pattern.split(text)

        if len(parts) <= 1:
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            if len(lines) > target_lines:
                return "\n".join(lines[:target_lines])
            return text

        result = []
        if parts[0].strip():
            result.append(parts[0].strip())

        i = 1
        while i < len(parts):
            header = parts[i].strip()
            body = parts[i + 1].strip() if i + 1 < len(parts) else ""
            i += 2

            raw_lines = [l.strip() for l in body.splitlines() if l.strip()]
            extracted = []

            for line in raw_lines:
                cleaned_l = re.sub(r"^[-*•\d+\.]+\s*", "", line).strip()
                if cleaned_l:
                    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned_l) if s.strip()]
                    for s in sents:
                        extracted.append(s)
                if len(extracted) >= target_lines:
                    break

            final_sents = extracted[:target_lines]
            formatted_body_lines = []
            for s in final_sents:
                if not s.endswith((".", "!", "?")):
                    s += "."
                formatted_body_lines.append(s)

            result.append(header)
            result.append("\n".join(formatted_body_lines))

        return "\n\n".join(result)

