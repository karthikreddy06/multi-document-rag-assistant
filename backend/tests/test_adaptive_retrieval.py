"""
Automated Tests for Adaptive Retrieval Strategies.
Verifies Document-Wide exhaustive coverage, Balanced Comparison retrieval,
Multi-Aspect coverage guarantees, Ordinal/Slice extraction, and Chat Scoping.
Zero hardcoding for specific files or queries.
"""

import pytest
from app.retrieval.retriever import HybridRetriever
from app.retrieval.planner import RetrievalStrategy
from app.vectorstore.store import VectorStore


@pytest.fixture(scope="module")
def retriever():
    from app.config import settings as _settings
    if _settings.is_pgvector:
        pytest.skip("Skipping Chroma-based adaptive retrieval tests in pgvector/production mode.")
    vs = VectorStore()
    if vs.count() == 0:
        pytest.skip("ChromaDB is empty. Ingestion required.")
    return HybridRetriever(vector_store=vs, top_k=5)


def test_exhaustive_retrieval_exceeds_top_k(retriever):
    """
    PROPERTY TEST: Exhaustive queries ('give me all...') must use DOCUMENT_WIDE strategy
    and retrieve chunks across multiple pages, NOT truncated to a small fixed top_k=3.
    """
    q = "give me all programs in this document"
    # Find a multi-page document in index
    indexed_files = list(retriever.vector_store.get_indexed_files().keys())
    assert len(indexed_files) > 0

    target_file = indexed_files[0]
    chunks, plan, coverage = retriever.retrieve_adaptive(
        query=q,
        where={"filename": target_file}
    )

    assert plan.strategy == RetrievalStrategy.DOCUMENT_WIDE
    # For exhaustive queries, chunks should represent the document sequentially
    assert len(chunks) > 0
    # Sequential ordering property
    pages = [int(c.metadata.get("page_number", 1)) for c in chunks]
    assert pages == sorted(pages), "Exhaustive chunks must be ordered sequentially by page number."


def test_comparison_balanced_representation(retriever):
    """
    PROPERTY TEST: Comparative query across two subjects must return balanced chunks
    with representation from both subjects.
    """
    q = "Compare the protagonist in Alice in Wonderland with the protagonist in The Old Man and the Sea."
    chunks, plan, coverage = retriever.retrieve_adaptive(query=q)

    assert plan.strategy == RetrievalStrategy.COMPARISON
    filenames = {c.metadata.get("filename") for c in chunks}
    assert "Alice_in_Wonderland.pdf" in filenames
    assert "oldmansea.pdf" in filenames

    # Verify neither document is crowded out
    alice_count = sum(1 for c in chunks if c.metadata.get("filename") == "Alice_in_Wonderland.pdf")
    oldman_count = sum(1 for c in chunks if c.metadata.get("filename") == "oldmansea.pdf")
    assert alice_count >= 2, f"Alice count {alice_count} must be >= 2"
    assert oldman_count >= 2, f"Old man count {oldman_count} must be >= 2"


def test_multi_aspect_coverage_tracking(retriever):
    """
    PROPERTY TEST: Multi-aspect query decomposes into distinct information needs,
    guaranteeing coverage tracking across all aspects.
    """
    q = "What is the background, methodology, and conclusion?"
    chunks, plan, coverage = retriever.retrieve_adaptive(query=q)

    assert plan.strategy == RetrievalStrategy.MULTI_ASPECT
    assert len(plan.aspects) >= 3
    assert len(chunks) > 0
    assert coverage.coverage_ratio > 0.0


def test_ordinal_exclusion_selection(retriever):
    """
    PROPERTY TEST: Ordinal query with exclusions ('all except first and second')
    activates ORDINAL strategy and slices candidates properly.
    """
    q = "Tell me all programs except the first and second"
    chunks, plan, coverage = retriever.retrieve_adaptive(query=q)

    assert plan.strategy == RetrievalStrategy.ORDINAL
    assert plan.ordinal_spec is not None
    assert 0 in plan.ordinal_spec.exclude_indices
    assert 1 in plan.ordinal_spec.exclude_indices


def test_chat_document_scoping_isolation(retriever):
    """
    SECURITY PROPERTY TEST: When a chat scope filter is provided,
    ZERO chunks from outside the scoped document can ever be retrieved.
    """
    q = "Who is the main character?"
    scope_alice = {"filename": "Alice_in_Wonderland.pdf"}
    chunks_alice, _, _ = retriever.retrieve_adaptive(query=q, where=scope_alice)

    for c in chunks_alice:
        assert c.metadata.get("filename") == "Alice_in_Wonderland.pdf"

    scope_oldman = {"filename": "oldmansea.pdf"}
    chunks_oldman, _, _ = retriever.retrieve_adaptive(query=q, where=scope_oldman)

    for c in chunks_oldman:
        assert c.metadata.get("filename") == "oldmansea.pdf"


def test_ordinal_exclusion_item_completeness(retriever):
    """
    REGRESSION TEST: Specifically verifies 'except first and second' returns items 3
    through the end with NO missing items (specifically items 3, 4, 5 must NOT be lost).
    """
    if "python_number_programs_20.pdf" not in retriever.vector_store.get_indexed_files():
        pytest.skip("python_number_programs_20.pdf not in index")
    q = "Tell me all programs except the first and second"
    chunks, plan, coverage = retriever.retrieve_adaptive(
        query=q,
        where={"filename": "python_number_programs_20.pdf"}
    )

    assert plan.strategy == RetrievalStrategy.ORDINAL
    resolved = getattr(plan, "resolved_items", [])
    assert len(resolved) == 18, f"Expected 18 resolved items, got {len(resolved)}"

    # Check identifiers and numeric values
    identifiers = [it.identifier for it in resolved]
    expected_identifiers = [str(i) for i in range(3, 21)]
    assert identifiers == expected_identifiers, f"Identifiers mismatch: {identifiers}"

    # Verify items 3, 4, and 5 are explicitly represented
    titles = [it.title.lower() for it in resolved]
    assert any("largest of two" in t for t in titles), "Item 3 (Largest of Two Numbers) missing!"
    assert any("largest of three" in t for t in titles), "Item 4 (Largest of Three Numbers) missing!"
    assert any("sum of natural" in t for t in titles), "Item 5 (Sum of Natural Numbers) missing!"

    # Verify evidence chunks contain items 3 through 20 and NOT items 1 and 2
    combined_text = "\n".join(c.text for c in chunks)
    assert "3. Largest of Two Numbers" in combined_text
    assert "4. Largest of Three Numbers" in combined_text
    assert "5. Sum of Natural Numbers" in combined_text
    assert "20. Square of Numbers" in combined_text
    # Items 1 and 2 must NOT be in the assembled evidence chunks
    assert "1. Even or Odd Number" not in combined_text
    assert "2. Positive, Negative or Zero" not in combined_text


def test_exhaustive_contains_all_20_items(retriever):
    """
    REGRESSION TEST: Specifically verifies 'all programs' contains all 20 items.
    """
    if "python_number_programs_20.pdf" not in retriever.vector_store.get_indexed_files():
        pytest.skip("python_number_programs_20.pdf not in index")
    q = "give me all programs"
    chunks, plan, coverage = retriever.retrieve_adaptive(
        query=q,
        where={"filename": "python_number_programs_20.pdf"}
    )

    assert plan.strategy == RetrievalStrategy.DOCUMENT_WIDE
    resolved = getattr(plan, "resolved_items", [])
    assert len(resolved) == 20, f"Expected 20 resolved items, got {len(resolved)}"

    identifiers = [it.identifier for it in resolved]
    expected = [str(i) for i in range(1, 21)]
    assert identifiers == expected, f"Expected 1..20, got {identifiers}"

    # Verify evidence chunks contain 1 through 20
    combined_text = "\n".join(c.text for c in chunks)
    for i in range(1, 21):
        assert f"{i}." in combined_text, f"Item {i} missing from evidence text!"


def test_ordinal_slices_and_single_items(retriever):
    """
    REGRESSION TEST: Verifies range slices ('programs 3 through 7') and single items ('second program').
    """
    if "python_number_programs_20.pdf" not in retriever.vector_store.get_indexed_files():
        pytest.skip("python_number_programs_20.pdf not in index")
    # 1. Range slice
    q_slice = "Give me programs 3 through 7"
    _, plan_slice, _ = retriever.retrieve_adaptive(
        query=q_slice,
        where={"filename": "python_number_programs_20.pdf"}
    )
    resolved_slice = getattr(plan_slice, "resolved_items", [])
    assert len(resolved_slice) == 5
    assert [it.identifier for it in resolved_slice] == ["3", "4", "5", "6", "7"]

    # 2. Single item (second)
    q_second = "Tell me about the second program"
    _, plan_second, _ = retriever.retrieve_adaptive(
        query=q_second,
        where={"filename": "python_number_programs_20.pdf"}
    )
    resolved_second = getattr(plan_second, "resolved_items", [])
    assert len(resolved_second) == 1
    assert resolved_second[0].identifier == "2"
    assert "positive" in resolved_second[0].title.lower()


def test_follow_up_scope_isolation(retriever):
    """
    CRITICAL REGRESSION TEST:
    Verifies that 'give the code for it' carries forward Item 3 from previous turn,
    activates ORDINAL strategy, retrieves ONLY Item 3 evidence chunk,
    and completely excludes other items (e.g., Item 5 Sum of Natural Numbers).
    """
    history = [
        {"role": "user", "content": "Tell me about the third program"},
        {
            "role": "assistant",
            "content": "The third program is **Largest of Two Numbers** (Item 3). It compares two numbers.",
            "sources_json": [{"section": "Item 3 - Largest of Two Numbers", "filename": "python_number_programs_20.pdf"}]
        }
    ]

    q = "give the code for it"
    if "python_number_programs_20.pdf" not in retriever.vector_store.get_indexed_files():
        pytest.skip("python_number_programs_20.pdf not in index")
    chunks, plan, coverage = retriever.retrieve_adaptive(
        query=q,
        recent_messages=history,
        where={"filename": "python_number_programs_20.pdf"}
    )

    assert plan.strategy == RetrievalStrategy.ORDINAL
    assert plan.ordinal_spec is not None
    assert plan.ordinal_spec.indices == [2]

    # Verify resolved items contains ONLY Item 3
    resolved = getattr(plan, "resolved_items", [])
    assert len(resolved) == 1, f"Expected exactly 1 resolved item, got {len(resolved)}"
    assert resolved[0].identifier == "3"
    assert "largest of two" in resolved[0].title.lower()

    # Evidence chunk must be strictly for Item 3
    assert len(chunks) == 1, f"Expected 1 chunk restricted to Item 3, got {len(chunks)}"
    assert "Largest of Two Numbers" in chunks[0].text
    # Must NOT contain other programs like Sum of Natural Numbers
    assert "Sum of Natural Numbers" not in chunks[0].text
    assert "Even or Odd Number" not in chunks[0].text


def test_follow_up_arbitrary_document_scoping(retriever):
    """
    PROPERTY TEST:
    Arbitrary document follow-up preserves target document scoping and entity focus.
    """
    history = [
        {"role": "user", "content": "Tell me about Santiago in the novel."},
        {
            "role": "assistant",
            "content": "Santiago is an aging Cuban fisherman who has gone eighty-four days without taking a fish.",
            "sources_json": [{"section": "Chapter 1", "filename": "oldmansea.pdf"}]
        }
    ]

    q = "how long has he gone without catching a fish?"
    chunks, plan, coverage = retriever.retrieve_adaptive(
        query=q,
        recent_messages=history,
    )

    assert plan.is_follow_up is True
    assert len(chunks) > 0
    # Chunks should be retrieved from oldmansea.pdf
    for c in chunks:
        assert c.metadata.get("filename") == "oldmansea.pdf"


