"""
Regression Tests for Broad / Complete Document Questions (Phase 5).
Ensures natural language broad queries trigger DOCUMENT_WIDE retrieval strategy,
adequate chunk budgets, user_id preservation, and don't degrade normal factual queries.
"""

import pytest
from app.retrieval.query_understanding import QueryAnalyzer, QueryIntent
from app.retrieval.document_resolver import DocumentResolutionResult, ResolvedDocument
from app.retrieval.planner import RetrievalPlanner, RetrievalStrategy


BROAD_QUERIES = [
    "explain the pdf completely",
    "explain the document completely",
    "explain the file completely",
    "explain this PDF thoroughly",
    "explain the whole document",
    "explain the entire PDF",
    "walk me through the entire document",
    "describe the whole file",
    "explain everything in this PDF",
    "summarize everything",
    "tell me everything in this document",
    "give me a complete explanation",
    "give me a full explanation",
    "explain it in full",
    "explain it thoroughly",
    "explain the document in detail",
]

FACTUAL_QUERIES = [
    "what is the candidate's phone number?",
    "what email address is listed?",
    "when did the author graduate from college?",
    "which database is used for caching?",
    "what was the project revenue in 2023?",
]


@pytest.mark.parametrize("query", BROAD_QUERIES)
def test_broad_queries_classified_as_exhaustive_or_summarization(query: str):
    analysis = QueryAnalyzer.analyze(query)
    assert analysis.intent in (QueryIntent.EXHAUSTIVE, QueryIntent.SUMMARIZATION), (
        f"Query '{query}' was classified as {analysis.intent}, expected EXHAUSTIVE or SUMMARIZATION"
    )


@pytest.mark.parametrize("query", BROAD_QUERIES)
def test_broad_queries_select_document_wide_strategy(query: str):
    analysis = QueryAnalyzer.analyze(query)
    doc_res = DocumentResolutionResult(
        resolved_documents=[ResolvedDocument(filename="sample_doc.pdf", doc_id="doc_123")],
        is_strictly_targeted=True,
        chroma_where_filter={"$and": [{"user_id": "usr_isolated"}, {"filename": "sample_doc.pdf"}]},
    )
    plan = RetrievalPlanner.create_plan(analysis, doc_res)

    assert plan.strategy == RetrievalStrategy.DOCUMENT_WIDE, (
        f"Query '{query}' produced strategy {plan.strategy}, expected DOCUMENT_WIDE"
    )
    assert plan.candidate_k >= 50, f"Candidate k too small: {plan.candidate_k}"
    assert plan.final_top_k >= 25, f"Final top k too small: {plan.final_top_k}"
    assert plan.where_filter is not None
    assert "usr_isolated" in str(plan.where_filter), "User ID filter was stripped!"


@pytest.mark.parametrize("query", FACTUAL_QUERIES)
def test_factual_queries_remain_narrow_and_fast(query: str):
    analysis = QueryAnalyzer.analyze(query)
    # Factual queries should not be classified as EXHAUSTIVE
    assert analysis.intent not in (QueryIntent.EXHAUSTIVE, QueryIntent.SUMMARIZATION), (
        f"Factual query '{query}' was falsely classified as {analysis.intent}"
    )

    doc_res = DocumentResolutionResult(
        resolved_documents=[ResolvedDocument(filename="sample_doc.pdf", doc_id="doc_123")],
        is_strictly_targeted=True,
        chroma_where_filter={"$and": [{"user_id": "usr_isolated"}, {"filename": "sample_doc.pdf"}]},
    )
    plan = RetrievalPlanner.create_plan(analysis, doc_res)
    assert plan.strategy != RetrievalStrategy.DOCUMENT_WIDE
    assert plan.final_top_k <= 10
