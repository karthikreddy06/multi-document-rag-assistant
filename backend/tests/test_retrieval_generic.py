"""
Automated tests for generic multi-document and multi-part retrieval behavior.
Validates general retrieval categories across heterogeneous documents without hardcoded document-specific logic.
"""

import pytest
from app.retrieval.retriever import HybridRetriever, QueryDecomposer
from app.vectorstore.store import VectorStore


@pytest.fixture(scope="module")
def retriever():
    vs = VectorStore()
    if vs.count() == 0:
        pytest.skip("ChromaDB collection is empty. Ingestion required.")
    return HybridRetriever(vector_store=vs, top_k=5)


# --- Query Decomposition Unit Tests ---

def test_query_decomposer_comparison():
    """Test generic decomposition of comparative queries."""
    q = "What is the difference between Document A and Document B?"
    subqueries = QueryDecomposer.decompose(q)
    assert len(subqueries) >= 3
    assert any("Document A" in sq for sq in subqueries)
    assert any("Document B" in sq for sq in subqueries)


def test_query_decomposer_comparison_attribute_distribution():
    """Test that comparing a shared attribute generates an attribute-paired need for each subject."""
    q = "Compare the main characters in Alice in Wonderland and The Old Man and the Sea."
    subqueries = QueryDecomposer.decompose(q)
    assert len(subqueries) >= 3

    # Must contain attribute for BOTH subjects
    has_alice_attr = any("main characters in Alice in Wonderland" in sq for sq in subqueries)
    has_oldman_attr = any("main characters in The Old Man and the Sea" in sq for sq in subqueries)

    assert has_alice_attr, "Subqueries must pair main characters with Alice in Wonderland."
    assert has_oldman_attr, "Subqueries must pair main characters with The Old Man and the Sea."

    # Must NOT generate a bare document title as an information need
    bare_title = "The Old Man and the Sea"
    assert bare_title not in subqueries, "Decomposer must NOT generate a bare document title when an attribute is requested."


def test_query_decomposer_comparison_compound_title_in_first_subject():
    """Test comparative query when the compound title containing 'and' is the first subject."""
    q = "Compare the themes of The Old Man and the Sea and Alice in Wonderland"
    subqueries = QueryDecomposer.decompose(q)
    assert len(subqueries) >= 3

    has_oldman_theme = any("themes of The Old Man and the Sea" in sq for sq in subqueries)
    has_alice_theme = any("themes of Alice in Wonderland" in sq for sq in subqueries)

    assert has_oldman_theme, "Must pair themes with The Old Man and the Sea."
    assert has_alice_theme, "Must pair themes with Alice in Wonderland."


def test_query_decomposer_comparison_based_on_attributes():
    """Test 'Compare A and B based on X and Y' variation."""
    q = "Compare Document A and Document B based on architecture and performance"
    subqueries = QueryDecomposer.decompose(q)
    assert len(subqueries) >= 3

    assert any("architecture and performance in Document A" in sq for sq in subqueries)
    assert any("architecture and performance in Document B" in sq for sq in subqueries)


def test_query_decomposer_comparison_differ_from():
    """Test 'How does X in A differ from X in B?' variation."""
    q = "How does the protagonist in Alice in Wonderland differ from the protagonist in The Old Man and the Sea?"
    subqueries = QueryDecomposer.decompose(q)
    assert len(subqueries) >= 3

    assert any("protagonist in Alice in Wonderland" in sq for sq in subqueries)
    assert any("protagonist in The Old Man and the Sea" in sq for sq in subqueries)


def test_query_decomposer_comparison_multi_subject():
    """Test comparative queries comparing more than two subjects."""
    q = "Compare the technologies used in System A, System B, and System C"
    subqueries = QueryDecomposer.decompose(q)
    assert len(subqueries) >= 4

    assert any("technologies used in System A" in sq for sq in subqueries)
    assert any("technologies used in System B" in sq for sq in subqueries)
    assert any("technologies used in System C" in sq for sq in subqueries)


def test_query_decomposer_comparison_scoped():
    """Test that comparative queries preserve introductory scope."""
    q = "In the provided documents, compare the main characters in Alice in Wonderland and The Old Man and the Sea."
    subqueries = QueryDecomposer.decompose(q)
    assert len(subqueries) >= 3

    assert any("In the provided documents" in sq and "Alice in Wonderland" in sq and "main characters" in sq for sq in subqueries)
    assert any("In the provided documents" in sq and "The Old Man and the Sea" in sq and "main characters" in sq for sq in subqueries)


def test_query_decomposer_conjunction():
    """Test generic decomposition of multi-clause conjunction queries with and without commas."""
    q1 = "What is the primary algorithm in Section 1, and how does the evaluation metric perform?"
    subqueries1 = QueryDecomposer.decompose(q1)
    assert len(subqueries1) >= 2

    q2 = "What is the primary algorithm in Section 1 and how does the evaluation metric perform?"
    subqueries2 = QueryDecomposer.decompose(q2)
    assert len(subqueries2) >= 2


def test_query_decomposer_scope_preservation():
    """Test that introductory scope (e.g. 'In Book X, ...') is preserved across decomposed subqueries."""
    q = "In Alice in Wonderland, what does the Cheshire Cat say and what is written on the bottle?"
    subqueries = QueryDecomposer.decompose(q)
    assert len(subqueries) >= 3
    # Both subqueries must carry the scope
    assert all("Alice in Wonderland" in sq for sq in subqueries)


def test_query_decomposer_numbered_and_multi_sentence():
    """Test decomposition of multi-sentence or numbered inquiries."""
    q1 = "What is the primary methodology? How were the results validated?"
    subqueries1 = QueryDecomposer.decompose(q1)
    assert len(subqueries1) >= 2

    q2 = "1. Who is the author 2. What tools were used"
    subqueries2 = QueryDecomposer.decompose(q2)
    assert len(subqueries2) >= 2


def test_query_decomposer_compound_title_protection():
    """Test that compound titles with 'and' are not erroneously split."""
    q = "Tell me about The Old Man and the Sea"
    subqueries = QueryDecomposer.decompose(q)
    # Should not split the title
    assert len(subqueries) == 1
    assert subqueries[0] == "Tell me about The Old Man and the Sea"


# --- Retrieval Behavior Tests across General Categories ---

def test_single_fact_retrieval(retriever):
    """Category 1: Single-fact query retrieves targeted, relevant chunks with high precision."""
    q = "How to compile a .tex file to a .pdf file?"
    chunks = retriever.retrieve(q, top_k=5)
    assert len(chunks) > 0
    # Top chunk should come from sample.pdf containing pdflatex compilation instructions
    assert chunks[0].metadata.get("filename") == "sample.pdf"
    assert any("pdflatex" in c.text.lower() for c in chunks)


def test_multi_part_same_document_retrieval(retriever):
    """Category 2: Multi-part query requiring two distinct facts from the same document (with comma)."""
    q = "What is the main character name in The Old Man and the Sea, and how long had he gone without catching a fish?"
    chunks = retriever.retrieve(q, top_k=5)
    assert len(chunks) == 5

    has_character_name = any("santiago" in c.text.lower() for c in chunks)
    has_fishing_duration = any("eighty-four" in c.text.lower() or "84" in c.text.lower() for c in chunks)

    assert has_character_name, "Retrieval must include the chunk mentioning the character name (Santiago)."
    assert has_fishing_duration, "Retrieval must include the chunk mentioning the fishing duration (84 days)."


def test_multi_part_same_document_retrieval_no_comma(retriever):
    """Category 2b: Multi-part query requiring two distinct facts from the same document without punctuation."""
    q = "What was the old man's name and how many days had he gone without catching a fish?"
    chunks = retriever.retrieve(q, top_k=5)
    assert len(chunks) == 5

    has_character_name = any("santiago" in c.text.lower() for c in chunks)
    has_fishing_duration = any("eighty-four" in c.text.lower() or "84" in c.text.lower() for c in chunks)

    assert has_character_name, "Retrieval must include the chunk mentioning the character name (Santiago)."
    assert has_fishing_duration, "Retrieval must include the chunk mentioning the fishing duration (84 days)."


def test_same_document_multi_chunk_retrieval(retriever):
    """Category 3: Query requiring information across multiple pages/sections of the same document."""
    q = "What tools and capabilities for LaTeX and PDF are described in the sample document?"
    chunks = retriever.retrieve(q, top_k=5)

    sample_chunks = [c for c in chunks if c.metadata.get("filename") == "sample.pdf"]
    assert len(sample_chunks) >= 2, "Must retrieve multiple chunks from sample.pdf."

    pages = {c.metadata.get("page_number") for c in sample_chunks}
    assert len(pages) >= 2, "Retrieved chunks must span multiple distinct pages of the document."


def test_same_document_distant_multi_part_retrieval(retriever):
    """Category 3b: Multi-part query requiring facts from distant chapters/sections of a long document."""
    q = "In Alice in Wonderland, what does the Cheshire Cat say about madness and what is written on the bottle Alice drinks from?"
    chunks = retriever.retrieve(q, top_k=5)
    assert len(chunks) == 5

    has_cat = any("cheshire" in c.text.lower() or "mad" in c.text.lower() for c in chunks)
    has_bottle = any("bottle" in c.text.lower() or "drink me" in c.text.lower() for c in chunks)

    assert has_cat, "Must retrieve chunk concerning the Cheshire Cat / madness."
    assert has_bottle, "Must retrieve chunk concerning the bottle Alice drank from."


def test_cross_document_retrieval(retriever):
    """Category 4: Cross-document comparative query retrieves chunks from both target documents."""
    q = "Compare the protagonist in Alice in Wonderland with the protagonist in The Old Man and the Sea."
    chunks = retriever.retrieve(q, top_k=5)

    filenames = {c.metadata.get("filename") for c in chunks}
    assert "Alice_in_Wonderland.pdf" in filenames, "Must include chunks from Alice_in_Wonderland.pdf."
    assert "oldmansea.pdf" in filenames, "Must include chunks from oldmansea.pdf."


def test_irrelevant_document_exclusion(retriever):
    """Category 5: Highly specific document query should not be contaminated by unrelated documents."""
    q = "What does Alice notice when she falls down the rabbit hole in Alice in Wonderland?"
    chunks = retriever.retrieve(q, top_k=5)

    alice_chunks = [c for c in chunks if c.metadata.get("filename") == "Alice_in_Wonderland.pdf"]
    assert len(alice_chunks) >= 4, "Specific query should predominantly retrieve chunks from target document."


def test_lexical_entity_matching(retriever):
    """Category 6: Exact keyword/entity queries prioritize matching authors/entities."""
    q = "Grzegorz Grudzinski Robert Maron"
    chunks = retriever.retrieve(q, top_k=3)
    assert len(chunks) > 0
    # Authors are on page 1 of sample.pdf
    assert chunks[0].metadata.get("filename") == "sample.pdf"
    assert "maron" in chunks[0].text.lower()


def test_semantic_matching(retriever):
    """Category 7: Semantic/paraphrased query without exact title keywords matches relevant theme."""
    q = "A solitary fisherman struggling against misfortune on the ocean waters"
    chunks = retriever.retrieve(q, top_k=5)
    assert len(chunks) > 0
    assert chunks[0].metadata.get("filename") == "oldmansea.pdf"
    assert any("skiff" in c.text.lower() or "fish" in c.text.lower() for c in chunks)
