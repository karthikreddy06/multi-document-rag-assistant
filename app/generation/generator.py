"""
LLM Generation Module.
Interacts with Ollama (llama3.2) to synthesize accurate, grounded answers from retrieved context.
"""

import time
from typing import List, Optional
import ollama

from app.config import settings
from app.generation.prompts import build_rag_prompt
from app.models import RetrievedChunk
from app.utils.logger import setup_logger

logger = setup_logger("generation.generator")


class LLMGenerator:
    """Production LLM response generator utilizing Ollama llama3.2."""

    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        temperature: float = 0.0,
        timeout: Optional[float] = None,
    ):
        self.model = model or settings.llm_model
        self.host = host or settings.ollama_host
        self.temperature = temperature
        self.timeout = timeout or settings.ollama_timeout
        self.client = ollama.Client(host=self.host, timeout=self.timeout)

    def generate_answer(
        self,
        question: str,
        chunks: List[RetrievedChunk],
        max_retries: int = 3,
    ) -> str:
        """
        Generate an answer from the retrieved chunks with retry handling for transient drops.
        """
        if not question or not question.strip():
            return "Please provide a valid question."

        if not chunks:
            logger.info("No context chunks provided to LLM generator.")
            return "I do not have enough information in the provided documents to answer this question."

        prompt = build_rag_prompt(question, chunks)
        logger.info(f"Generating answer using {self.model} with {len(chunks)} context chunks...")

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.generate(
                    model=self.model,
                    prompt=prompt,
                    options={
                        "temperature": self.temperature,
                    }
                )
                answer = response.get("response", "").strip()
                return answer
            except Exception as e:
                last_error = e
                logger.warning(
                    f"LLM generation attempt {attempt}/{max_retries} failed for {self.model}: {e}. "
                    f"{'Retrying...' if attempt < max_retries else 'No more retries.'}"
                )
                if attempt < max_retries:
                    time.sleep(0.5 * attempt)

        logger.error(f"Error during LLM generation with {self.model} after {max_retries} attempts: {last_error}")
        raise RuntimeError(
            f"LLM generation failed: {last_error}. Verify Ollama is running at {self.host} "
            f"and model '{self.model}' is installed."
        ) from last_error
