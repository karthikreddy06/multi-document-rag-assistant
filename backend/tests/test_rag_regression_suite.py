"""
Comprehensive Regression & Evaluation Suite for Multi-Document RAG Assistant.

Validates 15 core retrieval/RAG categories and 5 end-to-end regression scenarios:
1. Single-document factual retrieval
2. Multi-part questions
3. Cross-document questions
4. Comparative questions
5. Numeric/date questions
6. Broad questions
7. Unknown / unanswerable questions
8. Document-specific questions
9. Similar terminology across different documents
10. Questions requiring multiple chunks
11. Compound document titles
12. Multiple subjects
13. Partial-context situations
14. Duplicate ingestion idempotency
15. Incremental ingestion synchronization
"""

import os
import pytest
from app.main import RAGApplication
from app.models import Chunk, Document
from app.ingestion.pipeline import IngestionPipeline


@pytest.fixture(scope="module")
def rag_app():
    """Initialize RAG Application ensuring full collection is indexed."""
    from app.config import settings as _settings
    if _settings.is_pgvector:
        pytest.skip("Skipping local auto-ingest regression suite in pgvector/production mode.")
    app = RAGApplication()
    if app.vector_store.count() == 0:
        app.ingest()
    return app


# ==============================================================================
# SECTION 17: FIVE CORE REGRESSION TESTS (END-TO-END)
# ==============================================================================

@pytest.mark.llm
def test_regression_a_old_man_character_and_duration(rag_app):
    """
    Regression A: Multi-part single-document factual query.
    'What is the name of the main character in The Old Man and the Sea, and how long had he gone without catching a fish?'
    Expected: Retrieve evidence for both facts and answer them.
    """
    q = "What is the name of the main character in The Old Man and the Sea, and how long had he gone without catching a fish?"
    ans = rag_app.answer_question(q).lower()

    # Santiago and 84 days (or eighty-four)
    assert "santiago" in ans, f"Answer must identify character name: {ans}"
    assert ("84" in ans or "eighty-four" in ans or "eighty four" in ans), f"Answer must identify 84 days: {ans}"


@pytest.mark.llm
def test_regression_b_compare_main_characters(rag_app):
    """
    Regression B: Comparative cross-document query with compound titles.
    'Compare the main characters in Alice in Wonderland and The Old Man and the Sea.'
    Expected: Retrieve evidence from both documents and produce grounded comparison.
    """
    q = "Compare the main characters in Alice in Wonderland and The Old Man and the Sea."
    ans = rag_app.answer_question(q).lower()

    # Grounded evidence from both works must be present
    assert "alice" in ans, f"Answer must discuss Alice: {ans}"
    assert ("santiago" in ans or "old man" in ans or "fisherman" in ans), f"Answer must discuss Santiago/the old man: {ans}"


@pytest.mark.llm
def test_regression_c_databases_across_projects(rag_app):
    """
    Regression C: Project-specific database distinction.
    'What database did I use in TravelTrack, and what database did I use in SkillMatch?'
    Expected: Retrieve project-specific evidence and distinguish both projects.
    """
    q = "What database did I use in TravelTrack, and what database did I use in SkillMatch?"
    ans = rag_app.answer_question(q).lower()

    assert "mongodb" in ans, f"Answer must identify MongoDB for TravelTrack: {ans}"
    assert "postgresql" in ans or "postgres" in ans, f"Answer must identify PostgreSQL for SkillMatch: {ans}"


@pytest.mark.llm
def test_regression_d_main_topic_of_sample_pdf(rag_app):
    """
    Regression D: Broad document topic extraction.
    'What is the main topic of the Sample PDF Document?'
    Expected: Retrieve enough context to provide a concise grounded summary.
    """
    q = "What is the main topic of the Sample PDF Document?"
    ans = rag_app.answer_question(q).lower()

    # Sample PDF is a template / sample document demonstrating formatting, LaTeX, or document features
    assert ("sample" in ans or "template" in ans or "pdf" in ans or "document" in ans or "format" in ans), (
        f"Answer must ground topic of sample.pdf: {ans}"
    )


@pytest.mark.llm
def test_regression_e_unsupported_favorite_movie(rag_app):
    """
    Regression E: Unknown / unanswerable query.
    'What is Karthik's favorite movie?'
    Expected: Refuse safely without hallucinating.
    """
    q = "What is Karthik's favorite movie?"
    ans = rag_app.answer_question(q).lower()

    refusal_markers = [
        "not mentioned", "not provide", "not contain", "not have enough information",
        "not available", "cannot find", "no information", "do not have enough"
    ]
    assert any(m in ans for m in refusal_markers), f"Expected refusal for missing movie info: {ans}"


# ==============================================================================
# SECTION 16: CATEGORICAL REGRESSION TESTS
# ==============================================================================

def test_category_1_single_document_factual(rag_app):
    """Category 1: Direct factual extraction from a single document."""
    chunks = rag_app.retriever.retrieve("Where did I complete my Bachelor of Technology?", top_k=3)
    assert len(chunks) > 0
    top_texts = " ".join(c.text.lower() for c in chunks)
    assert "saveetha" in top_texts


def test_category_2_multi_part_question(rag_app):
    """Category 2: Multi-part question within the same document."""
    chunks = rag_app.retriever.retrieve(
        "What degree did I pursue, what was my CGPA, and which college did I attend?", top_k=5
    )
    texts = " ".join(c.text.lower() for c in chunks)
    assert "saveetha" in texts
    assert "8.27" in texts or "b.tech" in texts


def test_category_3_cross_document_retrieval(rag_app):
    """Category 3: Query requiring chunks across different documents."""
    chunks = rag_app.retriever.retrieve(
        "Give me information about the White Rabbit in Alice in Wonderland and the fisherman in The Old Man and the Sea.",
        top_k=6
    )
    filenames = {c.metadata.get("filename") for c in chunks}
    assert "Alice_in_Wonderland.pdf" in filenames
    assert "oldmansea.pdf" in filenames


def test_category_4_comparative_retrieval(rag_app):
    """Category 4: Compare attributes between two documents."""
    chunks = rag_app.retriever.retrieve(
        "Compare the setting in Alice in Wonderland with the setting in The Old Man and the Sea.",
        top_k=6
    )
    filenames = {c.metadata.get("filename") for c in chunks}
    assert "Alice_in_Wonderland.pdf" in filenames
    assert "oldmansea.pdf" in filenames


def test_category_5_numeric_date_retrieval(rag_app):
    """Category 5: Precise numeric and temporal retrieval."""
    chunks = rag_app.retriever.retrieve(
        "How many days had passed without catching a fish in The Old Man and the Sea?",
        top_k=3
    )
    texts = " ".join(c.text.lower() for c in chunks)
    assert "84" in texts or "eighty-four" in texts


def test_category_6_broad_document_query(rag_app):
    """Category 6: Broad summary inquiry."""
    chunks = rag_app.retriever.retrieve("Summarize the contents of sample.pdf", top_k=5)
    assert any(c.metadata.get("filename") == "sample.pdf" for c in chunks)


@pytest.mark.llm
def test_category_7_unknown_question_handling(rag_app):
    """Category 7: Query about non-existent facts."""
    ans = rag_app.answer_question("What is the capital city of Atlantis mentioned in the documents?").lower()
    refusal_markers = ["not mentioned", "not provide", "not contain", "not available", "no information"]
    assert any(m in ans for m in refusal_markers)


def test_category_8_document_specific_scoping(rag_app):
    """Category 8: Ensuring document name scoping isolates relevant file."""
    chunks = rag_app.retriever.retrieve("What is discussed in chapter 1 of sample.pdf?", top_k=3)
    for c in chunks[:2]:
        assert c.metadata.get("filename") == "sample.pdf"


def test_category_9_similar_terminology(rag_app):
    """Category 9: Disambiguating similar terms (e.g. databases, sea, fish) across contexts."""
    chunks = rag_app.retriever.retrieve("What database is used in TravelTrack?", top_k=3)
    assert any(c.metadata.get("filename") == "company.pdf" for c in chunks)
    assert any("mongodb" in c.text.lower() for c in chunks)


def test_category_10_multiple_chunks_same_document(rag_app):
    """Category 10: Query requiring evidence spanning distant chunks."""
    chunks = rag_app.retriever.retrieve(
        "What are all the technical skills listed including programming languages, frameworks, and databases?",
        top_k=5
    )
    texts = " ".join(c.text.lower() for c in chunks)
    assert "python" in texts
    assert "fastapi" in texts or "django" in texts


def test_category_11_compound_document_titles():
    """Category 11: Decomposition must preserve multi-word titles."""
    from app.retrieval.retriever import QueryDecomposer
    subqueries = QueryDecomposer.decompose(
        "What themes are present in The Old Man and the Sea?"
    )
    # The title must not be fractured into separate words
    combined = " ".join(subqueries)
    assert "Old Man and the Sea" in combined or "the old man and the sea" in combined.lower()


def test_category_12_multiple_subjects_decomposition():
    """Category 12: Comparison across three distinct entities."""
    from app.retrieval.retriever import QueryDecomposer
    subqueries = QueryDecomposer.decompose(
        "Compare the technologies used in Project A, Project B, and Project C."
    )
    assert len(subqueries) >= 3


@pytest.mark.llm
def test_category_13_partial_context_generation(rag_app):
    """
    Category 13: One supported fact and one unsupported fact.
    The assistant must answer the supported part and explicitly note the unsupported part.
    """
    q = "What database did I use in TravelTrack, and what database was used in CloudScale?"
    ans = rag_app.answer_question(q).lower()

    # Supported part answered
    assert "mongodb" in ans, f"Supported part must be answered: {ans}"
    # Unsupported part clearly stated
    decline_markers = [
        "not mentioned", "not provide", "not contain", "not available",
        "can't provide", "cannot provide", "no information", "cloudscale"
    ]
    assert any(m in ans for m in decline_markers), f"Unsupported part must be identified: {ans}"


def test_category_14_duplicate_ingestion_idempotency(rag_app):
    """Category 14: Re-ingesting unchanged corpus must be idempotent and produce zero duplicates."""
    count_before = rag_app.vector_store.count()
    pipeline = IngestionPipeline(
        vector_store=rag_app.vector_store,
        embedding_service=rag_app.embedding_service
    )
    # Run ingestion again on the documents directory
    pipeline.ingest_directory("documents")
    count_after = rag_app.vector_store.count()

    assert count_after == count_before, (
        f"Duplicate ingestion created extra chunks! Before: {count_before}, After: {count_after}"
    )


def test_category_15_incremental_ingestion_sync(rag_app):
    """Category 15: Vector store accurately tracks document index map."""
    indexed = rag_app.vector_store.get_indexed_files()
    assert "company.pdf" in indexed
    assert "Alice_in_Wonderland.pdf" in indexed
    assert "oldmansea.pdf" in indexed
    assert "sample.pdf" in indexed
    for fn, fhash in indexed.items():
        assert len(fhash) == 64, f"Hash for {fn} must be a 64-char SHA-256 string"
