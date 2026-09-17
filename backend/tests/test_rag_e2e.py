"""
End-to-End Evaluation Test for RAG Application.
Validates all 11 evaluation queries specified in the requirements.
"""

import pytest
from app.main import RAGApplication

pytestmark = pytest.mark.llm


@pytest.fixture(scope="module")
def rag_app():
    app = RAGApplication()
    if app.vector_store.count() == 0:
        app.ingest()
    return app


def test_q1_all_skills(rag_app):
    answer = rag_app.answer_question("What are all my skills?")
    ans_lower = answer.lower()
    # Check that broad skills cover all key domains
    assert "programming" in ans_lower or "python" in ans_lower
    assert "machine learning" in ans_lower or "deep learning" in ans_lower
    assert "ai" in ans_lower or "generative ai" in ans_lower
    assert "backend" in ans_lower or "fastapi" in ans_lower
    assert "database" in ans_lower or "postgresql" in ans_lower
    assert "tools" in ans_lower or "git" in ans_lower


def test_q2_programming_languages(rag_app):
    answer = rag_app.answer_question("What programming languages do I know?")
    ans_lower = answer.lower()
    assert "python" in ans_lower
    assert "sql" in ans_lower
    assert "java" in ans_lower


def test_q3_backend_technologies(rag_app):
    answer = rag_app.answer_question("What backend technologies do I know?")
    ans_lower = answer.lower()
    assert "fastapi" in ans_lower
    assert "django" in ans_lower


def test_q4_databases(rag_app):
    answer = rag_app.answer_question("What databases do I know?")
    ans_lower = answer.lower()
    assert "postgresql" in ans_lower
    assert "mongodb" in ans_lower


def test_q5_projects_built(rag_app):
    answer = rag_app.answer_question("What projects have I built?")
    ans_lower = answer.lower()
    assert "traveltrack" in ans_lower
    assert "skillmatch" in ans_lower
    # TripTrack should not be listed as a separate project
    assert "triptrack is a separate project" not in ans_lower


def test_q6_tell_me_about_traveltrack(rag_app):
    answer = rag_app.answer_question("Tell me about TravelTrack.")
    ans_lower = answer.lower()
    assert "travel" in ans_lower
    assert "fastapi" in ans_lower or "mongodb" in ans_lower


def test_q7_tell_me_about_skillmatch(rag_app):
    answer = rag_app.answer_question("Tell me about SkillMatch.")
    ans_lower = answer.lower()
    assert "job" in ans_lower
    assert "django" in ans_lower or "postgresql" in ans_lower


def test_q8_authentication_in_traveltrack(rag_app):
    answer = rag_app.answer_question("What authentication did I use in TravelTrack?")
    ans_lower = answer.lower()
    assert "jwt" in ans_lower
    assert "bcrypt" in ans_lower


def test_q9_education(rag_app):
    answer = rag_app.answer_question("What is my education?")
    ans_lower = answer.lower()
    assert "saveetha" in ans_lower
    assert "b.tech" in ans_lower or "computer science" in ans_lower
    assert "8.27" in answer


def test_q10_certifications(rag_app):
    answer = rag_app.answer_question("What certifications do I have?")
    ans_lower = answer.lower()
    assert "oracle" in ans_lower
    assert "generative ai" in ans_lower or "java" in ans_lower


def test_q11_unrelated_information(rag_app):
    answer = rag_app.answer_question("What is my favorite movie and what musical instruments do I play?")
    ans_lower = answer.lower()
    decline_phrases = ["not have enough information", "not mentioned", "not provide", "not contain", "not available", "cannot provide", "can't provide"]
    assert any(phrase in ans_lower for phrase in decline_phrases)
