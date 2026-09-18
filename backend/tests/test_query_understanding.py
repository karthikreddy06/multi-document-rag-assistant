"""
Comprehensive Unit Tests for Generic Query Understanding and Context Resolution.
Validates natural language intent detection, aspect extraction, ordinal slicing,
and conversation follow-up resolution with ZERO document-specific hardcoding.
"""

import pytest
from app.retrieval.query_understanding import (
    QueryAnalyzer,
    QueryIntent,
    ConversationContextResolver,
    OrdinalSpec,
)


def test_intent_factual_lookup():
    """Specific factual inquiries classify as FACTUAL."""
    q = "What is the author's name?"
    analysis = QueryAnalyzer.analyze(q)
    assert analysis.intent == QueryIntent.FACTUAL
    assert len(analysis.aspects) >= 1

    q2 = "What is the total marks and score?"
    analysis2 = QueryAnalyzer.analyze(q2)
    assert analysis2.intent in (QueryIntent.FACTUAL, QueryIntent.MULTI_ASPECT)


def test_intent_section_or_concept():
    """Explanatory or methodological questions classify as SECTION_OR_CONCEPT."""
    q = "Explain the methodology used in this project."
    analysis = QueryAnalyzer.analyze(q)
    assert analysis.intent == QueryIntent.SECTION_OR_CONCEPT

    q2 = "Describe the architecture and system workflow."
    analysis2 = QueryAnalyzer.analyze(q2)
    assert analysis2.intent in (QueryIntent.SECTION_OR_CONCEPT, QueryIntent.MULTI_ASPECT)


def test_intent_exhaustive():
    """Queries asking for 'all', 'every', 'complete list' classify as EXHAUSTIVE."""
    queries = [
        "give me all programs",
        "list every program",
        "what are all the sections in this document?",
        "tell me everything about this document",
        "give me the complete list of items",
    ]
    for q in queries:
        analysis = QueryAnalyzer.analyze(q)
        assert analysis.intent == QueryIntent.EXHAUSTIVE, f"Failed for query: {q}"


def test_intent_multi_aspect():
    """Queries asking for distinct attributes classify as MULTI_ASPECT and decompose."""
    q = "What are the education, skills, projects, and certifications?"
    analysis = QueryAnalyzer.analyze(q)
    assert analysis.intent == QueryIntent.MULTI_ASPECT
    assert len(analysis.aspects) >= 3
    assert any("education" in a.lower() for a in analysis.aspects)
    assert any("skills" in a.lower() for a in analysis.aspects)
    assert any("projects" in a.lower() for a in analysis.aspects)


def test_intent_comparison():
    """Comparative queries classify as COMPARISON and extract subjects."""
    q = "Compare Document A and Document B."
    analysis = QueryAnalyzer.analyze(q)
    assert analysis.intent == QueryIntent.COMPARISON
    assert len(analysis.subjects) == 2
    assert "Document A" in analysis.subjects
    assert "Document B" in analysis.subjects

    q2 = "How does Python differ from Java in this text?"
    analysis2 = QueryAnalyzer.analyze(q2)
    assert analysis2.intent == QueryIntent.COMPARISON
    assert len(analysis2.subjects) >= 2


def test_intent_page_targeted():
    """Questions specifying a page number classify as PAGE_TARGETED."""
    q = "What does page 15 say?"
    analysis = QueryAnalyzer.analyze(q)
    assert analysis.intent == QueryIntent.PAGE_TARGETED
    assert analysis.target_page == 15

    q2 = "Explain page 3."
    analysis2 = QueryAnalyzer.analyze(q2)
    assert analysis2.intent == QueryIntent.PAGE_TARGETED
    assert analysis2.target_page == 3


def test_intent_ordinal_reference():
    """Positional references classify as ORDINAL."""
    q = "Tell me about the second one."
    analysis = QueryAnalyzer.analyze(q)
    assert analysis.intent == QueryIntent.ORDINAL
    assert analysis.ordinal_spec is not None
    assert analysis.ordinal_spec.indices == [1]

    q_last = "Explain the last section."
    analysis_last = QueryAnalyzer.analyze(q_last)
    assert analysis_last.intent == QueryIntent.ORDINAL
    assert analysis_last.ordinal_spec.is_last is True


def test_ordinal_exclusions_and_ranges():
    """Exclusions like 'all except first and second' and ranges are parsed accurately."""
    q_excl = "Tell me all programs except the first and second"
    analysis_excl = QueryAnalyzer.analyze(q_excl)
    assert analysis_excl.intent == QueryIntent.ORDINAL
    assert 0 in analysis_excl.ordinal_spec.exclude_indices
    assert 1 in analysis_excl.ordinal_spec.exclude_indices

    q_range = "Give me items 3 to 10"
    analysis_range = QueryAnalyzer.analyze(q_range)
    assert analysis_range.intent == QueryIntent.ORDINAL
    assert analysis_range.ordinal_spec.slice_start == 2
    assert analysis_range.ordinal_spec.slice_end == 9


def test_conversation_follow_up_resolution():
    """Resolves 'explain the third one' given recent assistant message containing a list."""
    history = [
        {"role": "user", "content": "What are the sections?"},
        {
            "role": "assistant",
            "content": (
                "Here are the sections:\n"
                "1. Introduction\n"
                "2. System Architecture\n"
                "3. Evaluation Results\n"
                "4. Conclusion"
            )
        }
    ]

    q = "Explain the third one."
    analysis = QueryAnalyzer.analyze(q, recent_messages=history)
    assert analysis.is_follow_up is True
    assert "Evaluation Results" in analysis.resolved_query


def test_corpus_search_intent():
    """Search for mentions of a technical term or corpus search."""
    q = "Find all mentions of MongoDB in the document."
    analysis = QueryAnalyzer.analyze(q)
    assert analysis.intent == QueryIntent.CORPUS_SEARCH


def test_follow_up_third_program_code_for_it():
    """Follow-up: 'tell me about the third program' -> 'give the code for it'."""
    history = [
        {"role": "user", "content": "Tell me about the third program"},
        {
            "role": "assistant",
            "content": "The third program is **Largest of Two Numbers** (Item 3).",
            "sources_json": [{"section": "Item 3 - Largest of Two Numbers", "filename": "sample_programs.pdf"}]
        }
    ]
    q = "give the code for it"
    analysis = QueryAnalyzer.analyze(q, recent_messages=history)
    assert analysis.is_follow_up is True
    assert analysis.intent == QueryIntent.ORDINAL
    assert analysis.ordinal_spec is not None
    assert analysis.ordinal_spec.indices == [2]
    assert "Item 3: Largest of Two Numbers" in analysis.resolved_query


def test_follow_up_second_program_explain_it():
    """Follow-up: 'tell me about the second program' -> 'explain it'."""
    history = [
        {"role": "user", "content": "Tell me about the second program"},
        {
            "role": "assistant",
            "content": "The second program is **Positive, Negative or Zero** (Item 2).",
            "sources_json": [{"section": "Item 2 - Positive, Negative or Zero", "filename": "sample_programs.pdf"}]
        }
    ]
    q = "explain it"
    analysis = QueryAnalyzer.analyze(q, recent_messages=history)
    assert analysis.is_follow_up is True
    assert analysis.intent == QueryIntent.ORDINAL
    assert analysis.ordinal_spec is not None
    assert analysis.ordinal_spec.indices == [1]
    assert "Item 2: Positive, Negative or Zero" in analysis.resolved_query


def test_follow_up_last_program_what_does_it_do():
    """Follow-up: 'what is the last program?' -> 'what does it do?'."""
    history = [
        {"role": "user", "content": "What is the last program?"},
        {
            "role": "assistant",
            "content": "The last program is **Square of Numbers** (Item 20).",
            "sources_json": [{"section": "Item 20 - Square of Numbers", "filename": "sample_programs.pdf"}]
        }
    ]
    q = "what does it do?"
    analysis = QueryAnalyzer.analyze(q, recent_messages=history)
    assert analysis.is_follow_up is True
    assert analysis.intent == QueryIntent.ORDINAL
    assert analysis.ordinal_spec is not None
    assert (analysis.ordinal_spec.indices == [19] or analysis.ordinal_spec.is_last is True)
    assert "Square of Numbers" in analysis.resolved_query


def test_follow_up_first_program_give_an_example():
    """Follow-up: 'tell me about the first program' -> 'give an example'."""
    history = [
        {"role": "user", "content": "Tell me about the first program"},
        {
            "role": "assistant",
            "content": "The first program is **Even or Odd Number** (Item 1).",
            "sources_json": [{"section": "Item 1 - Even or Odd Number", "filename": "sample_programs.pdf"}]
        }
    ]
    q = "give an example"
    analysis = QueryAnalyzer.analyze(q, recent_messages=history)
    assert analysis.is_follow_up is True
    assert analysis.intent == QueryIntent.ORDINAL
    assert analysis.ordinal_spec is not None
    assert analysis.ordinal_spec.indices == [0]
    assert "Even or Odd Number" in analysis.resolved_query


def test_follow_up_relative_shifts():
    """Follow-up with relative shifts: 'previous one', 'next one', 'same one'."""
    history = [
        {"role": "user", "content": "Tell me about the third program"},
        {
            "role": "assistant",
            "content": "The third program is **Largest of Two Numbers** (Item 3).",
            "sources_json": [{"section": "Item 3 - Largest of Two Numbers", "filename": "sample_programs.pdf"}]
        }
    ]

    # Previous one (item 3 -> item 2)
    q_prev = "what about the previous one?"
    analysis_prev = QueryAnalyzer.analyze(q_prev, recent_messages=history)
    assert analysis_prev.is_follow_up is True
    assert analysis_prev.intent == QueryIntent.ORDINAL
    assert analysis_prev.ordinal_spec.indices == [1]

    # Next one (item 3 -> item 4)
    q_next = "what about the next one?"
    analysis_next = QueryAnalyzer.analyze(q_next, recent_messages=history)
    assert analysis_next.is_follow_up is True
    assert analysis_next.intent == QueryIntent.ORDINAL
    assert analysis_next.ordinal_spec.indices == [3]

    # Same one (item 3 -> item 3)
    q_same = "what about the same one?"
    analysis_same = QueryAnalyzer.analyze(q_same, recent_messages=history)
    assert analysis_same.is_follow_up is True
    assert analysis_same.intent == QueryIntent.ORDINAL
    assert analysis_same.ordinal_spec.indices == [2]


def test_follow_up_arbitrary_document_entity():
    """Follow-up on non-numbered document preserving entity and document hint."""
    history = [
        {"role": "user", "content": "Tell me about Santiago in the novel."},
        {
            "role": "assistant",
            "content": "Santiago is an aging Cuban fisherman who has gone eighty-four days without catching a fish.",
            "sources_json": [{"section": "Chapter 1", "filename": "oldmansea.pdf"}]
        }
    ]

    q_pronoun = "how long has he gone without catching a fish?"
    analysis_pronoun = QueryAnalyzer.analyze(q_pronoun, recent_messages=history)
    assert analysis_pronoun.is_follow_up is True
    assert "Santiago" in analysis_pronoun.resolved_query
    assert "Santiago" in analysis_pronoun.subjects
    assert analysis_pronoun.target_document_hint == "oldmansea.pdf"

    q_details = "give me more details"
    analysis_details = QueryAnalyzer.analyze(q_details, recent_messages=history)
    assert analysis_details.is_follow_up is True
    assert "Santiago" in analysis_details.resolved_query
    assert "Santiago" in analysis_details.subjects
    assert analysis_details.target_document_hint == "oldmansea.pdf"

