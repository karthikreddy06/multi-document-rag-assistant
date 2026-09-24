"""
Generic Query Understanding and Conversation Context Resolution Module.
Analyzes user queries without document-specific or domain-specific hardcoding.
Extracts intent, structural query types, information aspects, ordinal/slice constraints,
and resolves follow-up references using recent conversation history.
"""

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Set, Tuple


class QueryIntent(str, Enum):
    """Generic classification of information inquiry types."""
    FACTUAL = "factual"                      # Specific single fact, entity, date, number, or attribute lookup
    SECTION_OR_CONCEPT = "section_concept"  # Topic, methodology, section, or narrative explanation
    MULTI_ASPECT = "multi_aspect"            # Multiple distinct inquiries/attributes in a single query
    EXHAUSTIVE = "exhaustive"                # "all", "every", "complete list", "all sections", "everything"
    COMPARISON = "comparison"                # Comparing two or more entities or documents
    ORDINAL = "ordinal"                      # Referring to items by position or offset ("second one", "last one")
    PAGE_TARGETED = "page_targeted"          # Explicit page mentions ("page 5", "pages 2-4")
    CORPUS_SEARCH = "corpus_search"          # Search for mentions of a term across documents
    SUMMARIZATION = "summarization"          # Overall document summarization
    CODE_EXTRACTION = "code_extraction"      # Code, implementation, syntax, function request
    TABLE_LOOKUP = "table_lookup"            # Table, row, column, marks, tabular metrics
    PROCEDURAL = "procedural"                # Step-by-step instructions, procedure, workflow
    EXACT_TEXT = "exact_text"                # Exact quote, verbatim phrasing, definition
    IMAGE_UNDERSTANDING = "image_understanding" # Visual content description / image understanding
    OCR_QUERY = "ocr_query"                  # Explicit text extraction from image via OCR
    MULTI_DOCUMENT_QUERY = "multi_document_query" # Document-wide request spanning all/each attached files
    DOCUMENT_SUMMARY = "document_summary"    # Summary requested per attached document
    DOCUMENT_LIST_QUERY = "document_list_query"   # Query asking what files are attached to the session


@dataclass
class OrdinalSpec:
    """Specifies positional selection or slicing of items."""
    indices: List[int] = field(default_factory=list)      # 0-based indices to include (e.g. [1] for second)
    exclude_indices: List[int] = field(default_factory=list) # 0-based indices to exclude (e.g. [0, 1] for "except first and second")
    is_last: bool = False                                 # Target last item
    slice_start: Optional[int] = None                     # Start of range (inclusive, 0-based)
    slice_end: Optional[int] = None                       # End of range (inclusive, 0-based)
    raw_reference: str = ""                               # Original textual reference


@dataclass
class QueryAnalysis:
    """Structured understanding of a user query."""
    raw_query: str
    cleaned_query: str
    intent: QueryIntent
    aspects: List[str] = field(default_factory=list)
    subjects: List[str] = field(default_factory=list)
    target_page: Optional[int] = None
    ordinal_spec: Optional[OrdinalSpec] = None
    target_document_hint: Optional[str] = None
    is_follow_up: bool = False
    resolved_query: Optional[str] = None  # Enriched or rewritten query if follow-up was resolved
    is_code_request: bool = False
    is_table_request: bool = False
    lines_per_doc: Optional[int] = None


class QueryAnalyzer:
    """
    General-purpose natural language query analyzer.
    Determines query intent, structural constraints, and decomposed sub-needs.
    """

    EXHAUSTIVE_INDICATORS = (
        r"\b(?:all|every|each|everything|entire|complete\s+list|whole|full\s+list|"
        r"complete\s+summary|full\s+summary|entire\s+document|whole\s+document|"
        r"all\s+sections|all\s+items|all\s+entries|all\s+pages|all\s+of\s+the)\b"
    )

    COMPARISON_PREFIXES = (
        r"^(?:what\s+is\s+the\s+difference\s+between|"
        r"what\s+are\s+the\s+differences\s+between|"
        r"compare\s+and\s+contrast|"
        r"compare|"
        r"contrast|"
        r"similarities\s+and\s+differences\s+between)\s+"
    )

    DIFFER_PATTERN = r"^how\s+does\s+(.+?)\s+differ\s+from\s+(.+)$"

    ORDINAL_WORDS = {
        "first": 0, "1st": 0,
        "second": 1, "2nd": 1,
        "third": 2, "3rd": 2,
        "fourth": 3, "4th": 3,
        "fifth": 4, "5th": 4,
        "sixth": 5, "6th": 5,
        "seventh": 6, "7th": 6,
        "eighth": 7, "8th": 7,
        "ninth": 8, "8th": 8,
        "tenth": 9, "10th": 9,
        "eleventh": 10, "11th": 10,
        "twelfth": 11, "12th": 11,
        "last": -1, "final": -1,
    }

    PAGE_PATTERN = r"\b(?:page|pages|p\.)\s*(\d+)\b"

    SUMMARIZATION_PATTERNS = (
        r"\b(?:summarize|summary\s+of|give\s+(?:me\s+)?(?:a\s+)?(?:full\s+|complete\s+|detailed\s+|brief\s+|comprehensive\s+)?summary\s+of|"
        r"(?:complete|full|entire|detailed|comprehensive|brief|executive)\s+summary(?:\s+of)?|"
        r"overview\s+of|briefly\s+describe|provide\s+(?:a\s+)?summary\s+of)\b"
    )

    SEARCH_PATTERNS = (
        r"\b(?:find\s+(?:all\s+)?mentions?\s+of|search\s+for|where\s+is\s+.*mentioned|does\s+.*contain)\b"
    )

    CODE_PATTERNS = (
        r"\b(?:code|source\s+code|python\s+code|script|implementation|function\s+definition|function\s+implementation|syntax)\b"
    )

    TABLE_PATTERNS = (
        r"\b(?:table|columns?|rows?|marks?|scores?|grades?|percentages?|tabular|gpa|totals?)\b"
    )

    PROCEDURAL_PATTERNS = (
        r"\b(?:steps?|procedure|instructions?|how\s+to\s+(?:do|perform|install|run|execute|setup|configure)|algorithm\s+steps)\b"
    )

    EXACT_TEXT_PATTERNS = (
        r"\b(?:exact\s+(?:words?|text|quote|phrasing)|verbatim|exact\s+definition|quote)\b"
    )

    IMAGE_UNDERSTANDING_PATTERNS = (
        r"\b(?:tell\s+me\s+about\s+(?:that\s+|the\s+)?image|what\s+is\s+in\s+the\s+image|describe\s+the\s+image|explain\s+the\s+image|what\s+does\s+the\s+image\s+show|image\s+content|picture\s+content|about\s+that\s+photo)\b"
    )

    OCR_PATTERNS = (
        r"\b(?:what\s+text\s+is\s+in\s+the\s+image|extract\s+text\s+from\s+the\s+image|read\s+text\s+(?:in|from)\s+the\s+image|ocr\s+text|text\s+in\s+image|extract\s+text)\b"
    )

    MULTI_DOCUMENT_PATTERNS = (
        r"\b(?:explain\s+(?:about\s+)?each\s+file|explain\s+(?:about\s+)?each\s+document|summarize\s+all\s+documents|summarize\s+all\s+files|summary\s+of\s+all\s+files|explain\s+all\s+files|what\s+is\s+in\s+these\s+documents|give\s+me\s+a\s+summary\s+of\s+every\s+attached\s+document|compare\s+all\s+uploaded\s+files|what\s+does\s+each\s+document\s+contain|summary\s+of\s+all\s+the\s+files|each\s+file\s+in\s+\d+\s+lines)\b"
    )

    DOCUMENT_LIST_PATTERNS = (
        r"\b(?:what\s+files\s+are\s+attached|list\s+attached\s+files|what\s+documents\s+are\s+in\s+this\s+chat|which\s+documents\s+are\s+attached|files\s+attached\s+to\s+this\s+chat)\b"
    )

    @classmethod
    def analyze(
        cls,
        query: str,
        recent_messages: Optional[List[Dict[str, Any]]] = None,
        available_documents: Optional[List[str]] = None,
    ) -> QueryAnalysis:
        clean_q = query.strip()
        q_lower = clean_q.lower()

        lines_per_doc = None
        lines_m = re.search(r"\b(?:in|with|using|for|each\s+file\s+in|each\s+document\s+in)\s+(\d+)\s+lines?\b|\b(\d+)\s+lines?\s+(?:per|for|each)\b", q_lower)
        if lines_m:
            num_str = lines_m.group(1) or lines_m.group(2)
            if num_str and num_str.isdigit():
                lines_per_doc = int(num_str)

        # Step 0a: Check Image Understanding vs OCR Query
        if re.search(cls.OCR_PATTERNS, q_lower):
            return QueryAnalysis(
                raw_query=query,
                cleaned_query=clean_q,
                intent=QueryIntent.OCR_QUERY,
                aspects=[clean_q],
                lines_per_doc=lines_per_doc,
            )
        if re.search(cls.IMAGE_UNDERSTANDING_PATTERNS, q_lower):
            return QueryAnalysis(
                raw_query=query,
                cleaned_query=clean_q,
                intent=QueryIntent.IMAGE_UNDERSTANDING,
                aspects=[clean_q],
                lines_per_doc=lines_per_doc,
            )

        # Step 0b: Check Document List & Multi-Document Query
        if re.search(cls.DOCUMENT_LIST_PATTERNS, q_lower):
            return QueryAnalysis(
                raw_query=query,
                cleaned_query=clean_q,
                intent=QueryIntent.DOCUMENT_LIST_QUERY,
                aspects=[clean_q],
                lines_per_doc=lines_per_doc,
            )
        if re.search(cls.MULTI_DOCUMENT_PATTERNS, q_lower) or lines_per_doc is not None:
            return QueryAnalysis(
                raw_query=query,
                cleaned_query=clean_q,
                intent=QueryIntent.MULTI_DOCUMENT_QUERY,
                aspects=[clean_q],
                lines_per_doc=lines_per_doc,
            )

        # Step 1: Check for conversation follow-up resolution first
        resolved_q = None
        is_follow_up = False
        resolved_ordinal = cls._extract_ordinal_spec(clean_q)
        carried_subjects: List[str] = []
        carried_doc_hint: Optional[str] = None

        if recent_messages:
            fu_res = ConversationContextResolver.resolve(
                query=clean_q,
                messages=recent_messages,
                ordinal_spec=resolved_ordinal,
            )
            if fu_res and fu_res.is_follow_up:
                is_follow_up = True
                resolved_q = fu_res.resolved_query
                clean_q = fu_res.resolved_query
                q_lower = clean_q.lower()
                if fu_res.carried_ordinal and not resolved_ordinal:
                    resolved_ordinal = fu_res.carried_ordinal
                if fu_res.target_subjects:
                    carried_subjects = fu_res.target_subjects
                if fu_res.target_document_hint:
                    carried_doc_hint = fu_res.target_document_hint

        # Step 2: Check page targeting ("What does page 15 say?")
        page_m = re.search(cls.PAGE_PATTERN, q_lower)
        target_page = int(page_m.group(1)) if page_m else None
        if target_page is not None and len(clean_q.split()) <= 10:
            return QueryAnalysis(
                raw_query=query,
                cleaned_query=clean_q,
                intent=QueryIntent.PAGE_TARGETED,
                target_page=target_page,
                aspects=[clean_q],
                is_follow_up=is_follow_up,
                resolved_query=resolved_q,
            )

        # Step 3: Check comparative questions
        comp_subjects, comp_aspects = cls._extract_comparison(clean_q)
        if comp_subjects and len(comp_subjects) >= 2:
            return QueryAnalysis(
                raw_query=query,
                cleaned_query=clean_q,
                intent=QueryIntent.COMPARISON,
                subjects=comp_subjects,
                aspects=comp_aspects or comp_subjects,
                is_follow_up=is_follow_up,
                resolved_query=resolved_q,
            )

        # Step 4: Check ordinal / positional references ("tell me the second one", "items 3 to 10", follow-ups)
        is_code = bool(re.search(cls.CODE_PATTERNS, q_lower))
        is_table = bool(re.search(cls.TABLE_PATTERNS, q_lower))

        if resolved_ordinal:
            return QueryAnalysis(
                raw_query=query,
                cleaned_query=clean_q,
                intent=QueryIntent.ORDINAL,
                ordinal_spec=resolved_ordinal,
                aspects=[clean_q],
                subjects=carried_subjects,
                target_document_hint=carried_doc_hint,
                is_follow_up=is_follow_up,
                resolved_query=resolved_q,
                is_code_request=is_code,
                is_table_request=is_table,
            )

        # Step 5: Check corpus search / mentions ("find all mentions of X", "search for X")
        if re.search(cls.SEARCH_PATTERNS, q_lower):
            return QueryAnalysis(
                raw_query=query,
                cleaned_query=clean_q,
                intent=QueryIntent.CORPUS_SEARCH,
                aspects=[clean_q],
                is_follow_up=is_follow_up,
                resolved_query=resolved_q,
                is_code_request=is_code,
                is_table_request=is_table,
            )

        # Step 6: Check exhaustive / document-wide questions ("all", "every", "complete list")
        if re.search(cls.EXHAUSTIVE_INDICATORS, q_lower):
            aspects = cls._extract_exhaustive_aspects(clean_q)
            return QueryAnalysis(
                raw_query=query,
                cleaned_query=clean_q,
                intent=QueryIntent.EXHAUSTIVE,
                aspects=aspects,
                is_follow_up=is_follow_up,
                resolved_query=resolved_q,
                is_code_request=is_code,
                is_table_request=is_table,
            )

        # Step 7: Check summarization
        if re.search(cls.SUMMARIZATION_PATTERNS, q_lower):
            return QueryAnalysis(
                raw_query=query,
                cleaned_query=clean_q,
                intent=QueryIntent.SUMMARIZATION,
                aspects=[clean_q],
                is_follow_up=is_follow_up,
                resolved_query=resolved_q,
                is_code_request=is_code,
                is_table_request=is_table,
            )

        # Step 8: Check multi-aspect inquiries ("education, skills, projects, and certifications")
        multi_aspects = cls._extract_multi_aspects(clean_q)
        if len(multi_aspects) > 1:
            return QueryAnalysis(
                raw_query=query,
                cleaned_query=clean_q,
                intent=QueryIntent.MULTI_ASPECT,
                aspects=multi_aspects,
                is_follow_up=is_follow_up,
                resolved_query=resolved_q,
                is_code_request=is_code,
                is_table_request=is_table,
            )

        # Step 9: Check code extraction requests
        if is_code:
            return QueryAnalysis(
                raw_query=query,
                cleaned_query=clean_q,
                intent=QueryIntent.CODE_EXTRACTION,
                aspects=[clean_q],
                subjects=carried_subjects,
                target_document_hint=carried_doc_hint,
                is_follow_up=is_follow_up,
                resolved_query=resolved_q,
                is_code_request=True,
                is_table_request=False,
            )

        # Step 10: Check table / tabular requests
        if is_table:
            return QueryAnalysis(
                raw_query=query,
                cleaned_query=clean_q,
                intent=QueryIntent.TABLE_LOOKUP,
                aspects=[clean_q],
                subjects=carried_subjects,
                target_document_hint=carried_doc_hint,
                is_follow_up=is_follow_up,
                resolved_query=resolved_q,
                is_code_request=False,
                is_table_request=True,
            )

        # Step 11: Check procedural step requests
        if re.search(cls.PROCEDURAL_PATTERNS, q_lower):
            return QueryAnalysis(
                raw_query=query,
                cleaned_query=clean_q,
                intent=QueryIntent.PROCEDURAL,
                aspects=[clean_q],
                subjects=carried_subjects,
                target_document_hint=carried_doc_hint,
                is_follow_up=is_follow_up,
                resolved_query=resolved_q,
                is_code_request=is_code,
                is_table_request=is_table,
            )

        # Step 12: Check exact text requests
        if re.search(cls.EXACT_TEXT_PATTERNS, q_lower):
            return QueryAnalysis(
                raw_query=query,
                cleaned_query=clean_q,
                intent=QueryIntent.EXACT_TEXT,
                aspects=[clean_q],
                subjects=carried_subjects,
                target_document_hint=carried_doc_hint,
                is_follow_up=is_follow_up,
                resolved_query=resolved_q,
                is_code_request=is_code,
                is_table_request=is_table,
            )

        # Step 13: Section / concept inquiry vs. simple factual lookup
        if cls._is_concept_or_section(q_lower):
            intent = QueryIntent.SECTION_OR_CONCEPT
        else:
            intent = QueryIntent.FACTUAL

        return QueryAnalysis(
            raw_query=query,
            cleaned_query=clean_q,
            intent=intent,
            aspects=[clean_q],
            subjects=carried_subjects,
            target_document_hint=carried_doc_hint,
            is_follow_up=is_follow_up,
            resolved_query=resolved_q,
            is_code_request=is_code,
            is_table_request=is_table,
        )

    @classmethod
    def _extract_ordinal_spec(cls, query: str) -> Optional[OrdinalSpec]:
        """Parse ordinal references, exclusions, and ranges."""
        q_lower = query.lower()

        # 1. Check exclusions: "all except the first and second", "except item 4", "everything except the last"
        excl_m = re.search(r"\bexcept\s+(?:the\s+)?(.+)$", q_lower)
        if excl_m:
            excl_body = excl_m.group(1).strip()
            excl_indices = []
            words = re.findall(r"\b\w+\b", excl_body)
            for w in words:
                if w in cls.ORDINAL_WORDS:
                    idx = cls.ORDINAL_WORDS[w]
                    excl_indices.append(idx)
                elif w.isdigit():
                    excl_indices.append(int(w) - 1)
            if excl_indices:
                return OrdinalSpec(exclude_indices=excl_indices, raw_reference=excl_m.group(0))

        # 2. Check ranges: "items 3 to 10", "milestones 4 through 8", "programs 3 to 7", "3 through 7"
        range_m = re.search(r"\b(?:[a-zA-Z]{3,15}\s+)?(\d+)\s*(?:to|-|through)\s*(\d+)\b", q_lower)
        if range_m:
            start_i = max(0, int(range_m.group(1)) - 1)
            end_i = max(0, int(range_m.group(2)) - 1)
            return OrdinalSpec(slice_start=start_i, slice_end=end_i, raw_reference=range_m.group(0))

        # 3. Check "first N" or "first two", "first five"
        first_n_m = re.search(r"\bfirst\s+(\d+|two|three|four|five|six|seven|eight|nine|ten)\b", q_lower)
        word_to_num = {
            "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
        }
        if first_n_m:
            token = first_n_m.group(1)
            n = int(token) if token.isdigit() else word_to_num.get(token, 2)
            return OrdinalSpec(slice_start=0, slice_end=n - 1, raw_reference=first_n_m.group(0))

        # 4. Check specific ordinal: "second one", "3rd section", "last program", "explain the fifth"
        for word, idx in cls.ORDINAL_WORDS.items():
            pattern = r"\b" + re.escape(word) + r"\b(?:\s+(?:one|item|program|section|page|entry|part))?"
            if re.search(pattern, q_lower):
                # Ensure it's not part of an exclusion already handled
                if idx == -1:
                    return OrdinalSpec(is_last=True, raw_reference=word)
                return OrdinalSpec(indices=[idx], raw_reference=word)

        return None

    @classmethod
    def _extract_comparison(cls, query: str) -> Tuple[List[str], List[str]]:
        """Extract comparison subjects and shared attributes."""
        q_lower = query.lower()

        # Check "How does X differ from Y"
        differ_m = re.match(cls.DIFFER_PATTERN, query.strip(), re.IGNORECASE)
        if differ_m:
            side_a = differ_m.group(1).strip(" ?,.")
            side_b = differ_m.group(2).strip(" ?,.")
            if len(side_a) > 2 and len(side_b) > 2:
                return [side_a, side_b], [f"{side_a}", f"{side_b}"]

        # Check comparison prefixes
        pref_m = re.match(cls.COMPARISON_PREFIXES, query.strip(), re.IGNORECASE)
        if pref_m:
            body = query.strip()[pref_m.end():].strip(" ?,.")
            # Check "Compare X and Y based on / in terms of Z"
            based_on_m = re.search(r"\s+\b(?:based\s+on|in\s+terms\s+of|regarding|with\s+respect\s+to)\s+(.+)$", body, re.IGNORECASE)
            if based_on_m:
                subj_str = body[:based_on_m.start()].strip()
                attr_str = based_on_m.group(1).strip()
                subjects = cls._split_subjects(subj_str)
                if len(subjects) >= 2:
                    aspects = [f"{attr_str} in {s}" for s in subjects]
                    return subjects, aspects

            # Check "Compare [attribute] in/of/for [subjects]"
            prep_m = re.search(r"\b(in|of|for|between|across)\b", body, re.IGNORECASE)
            if prep_m:
                attr_cand = body[:prep_m.start()].strip()
                subj_cand = body[prep_m.end():].strip()
                if attr_cand and len(attr_cand.split()) <= 4:
                    subjects = cls._split_subjects(subj_cand)
                    if len(subjects) >= 2:
                        aspects = [f"{attr_cand} of {s}" for s in subjects]
                        return subjects, aspects

            subjects = cls._split_subjects(body)
            if len(subjects) >= 2:
                return subjects, subjects

        # Check "X vs Y" or "X versus Y"
        vs_m = re.search(r"^(.*?)\s+\b(?:versus|vs\.?)\b\s+(.+)$", query.strip(), re.IGNORECASE)
        if vs_m:
            side_a = vs_m.group(1).strip(" ?,.")
            side_b = vs_m.group(2).strip(" ?,.")
            if len(side_a) > 1 and len(side_b) > 1:
                return [side_a, side_b], [side_a, side_b]

        return [], []

    @classmethod
    def _clean_token(cls, t: str) -> str:
        s = t.strip(" ?,.:; \t\n")
        return re.sub(r"^(?:and\s+)+", "", s, flags=re.IGNORECASE).strip(" ?,.:; \t\n")

    @classmethod
    def _split_subjects(cls, s: str) -> List[str]:
        """Split conjoined subjects cleanly, protecting titles with 'and'."""
        s = s.strip(" ?,.")

        # Check 'with', 'versus', 'vs'
        vs_m = re.split(r"\s+\b(?:with|versus|vs\.?)\b\s+", s, flags=re.IGNORECASE)
        if len(vs_m) > 1:
            return [cls._clean_token(p) for p in vs_m if cls._clean_token(p)]

        # Check comma-separated lists
        if "," in s:
            parts = re.split(r",\s*(?:and\s+)?", s, flags=re.IGNORECASE)
            cleaned = [cls._clean_token(p) for p in parts if cls._clean_token(p)]
            if len(cleaned) > 1:
                return cleaned

        # Find all occurrences of " and "
        and_positions = [m.start() for m in re.finditer(r"\s+\band\b\s+", s, re.IGNORECASE)]
        if not and_positions:
            cleaned = cls._clean_token(s)
            return [cleaned] if cleaned else []

        if len(and_positions) == 1:
            idx = and_positions[0]
            m = re.search(r"\s+\band\b\s+", s[idx:], re.IGNORECASE)
            left = cls._clean_token(s[:idx])
            right = cls._clean_token(s[idx + m.end():])
            if re.match(r"^The\s+[A-Z]", left) and re.match(r"^(?:the\s+)?[A-Z]", right):
                return [cls._clean_token(s)]
            return [left, right]

        best_split = None
        best_score = -999

        for pos in and_positions:
            m = re.search(r"\s+\band\b\s+", s[pos:], re.IGNORECASE)
            left = cls._clean_token(s[:pos])
            right = cls._clean_token(s[pos + m.end():])

            score = 0
            left_is_title = bool(re.search(r"\bThe\s+[A-Z].*\s+and\s+(?:the\s+)?[A-Z]", left))
            right_is_title = bool(re.search(r"\bThe\s+[A-Z].*\s+and\s+(?:the\s+)?[A-Z]", right))

            breaks_title = (
                re.match(r"^the\s+[A-Z]", right) and not re.match(r"^The\s+[A-Z]", right)
            ) or left.endswith(" The") or left.endswith(" the")

            if breaks_title:
                score -= 20
            if left_is_title:
                score += 10
            if right_is_title:
                score += 10
            if left and left[0].isupper():
                score += 2
            if right and right[0].isupper():
                score += 2

            if score > best_score:
                best_score = score
                best_split = [left, right]

        if best_split:
            return best_split

        return [cls._clean_token(s)]

    @classmethod
    def _extract_multi_aspects(cls, query: str) -> List[str]:
        """Extract multi-aspect questions (e.g. 'What are the education, skills, projects, and certifications?')."""
        clean_q = query.strip()

        # Multi-question split ("What is X? What is Y?", semicolons)
        multi_q = re.split(r"(?:\?\s+|\;\s*)", clean_q)
        multi_q = [q.strip(" ?,.") for q in multi_q if len(q.strip(" ?,.")) > 3]
        if len(multi_q) > 1:
            return multi_q

        # Coordinated attributes: "What are the X, Y, and Z?" or "Tell me about X and Y"
        match = re.match(
            r"^(?:what\s+(?:is|are)\s+(?:the\s+)?(?:all\s+)?(?:my\s+)?|tell\s+me\s+about\s+(?:the\s+)?|give\s+me\s+(?:the\s+)?)(.+)$",
            clean_q,
            re.IGNORECASE
        )
        if match:
            body = match.group(1).strip(" ?.,")
            # If body has commas or 'and'
            if "," in body:
                parts = re.split(r",\s*(?:and\s+)?", body, flags=re.IGNORECASE)
                cleaned_parts = [p.strip() for p in parts if len(p.strip()) > 1]
                if len(cleaned_parts) >= 2:
                    return cleaned_parts

            subjects = cls._split_subjects(body)
            if len(subjects) >= 2:
                return subjects

        return [clean_q]

    @classmethod
    def _extract_exhaustive_aspects(cls, query: str) -> List[str]:
        """Extract core subject from exhaustive inquiry."""
        # e.g. "give me all programs" -> "all programs", "programs"
        # "explain every section" -> "every section"
        return [query.strip()]

    @classmethod
    def _is_concept_or_section(cls, q_lower: str) -> bool:
        """Heuristic distinguishing explanatory/methodological questions from specific factual lookups."""
        concept_starters = (
            "explain", "describe", "how does", "how do", "why is", "why does",
            "methodology", "architecture", "overview", "workflow", "process",
            "background", "discussion"
        )
        return any(q_lower.startswith(s) or f" {s} " in f" {q_lower} " for s in concept_starters)


@dataclass
class FollowUpResolution:
    """Outcome of resolving conversational dependencies from message history."""
    is_follow_up: bool = False
    resolved_query: str = ""
    carried_ordinal: Optional[OrdinalSpec] = None
    target_subjects: List[str] = field(default_factory=list)
    target_document_hint: Optional[str] = None


class ConversationContextResolver:
    """
    Resolves conversational dependencies (pronouns, ordinals, relative item references,
    elliptical action requests) using recent conversation history.
    Operates generically across any document type without hardcoded names or queries.
    """

    PRONOUN_PATTERN = re.compile(
        r"\b(?:it|its|this|that|these|those|them|their|he|him|his|she|her)\b",
        re.IGNORECASE
    )

    RELATIVE_ITEM_PATTERN = re.compile(
        r"\b(?:the\s+)?(?:previous|earlier|prior|next|following|same|above|mentioned|first|second|third|last)\s+(?:one|program|step|rule|item|section|algorithm|method|topic|chapter|part|function|example|code|problem|question)\b|"
        r"\b(?:what\s+about\s+(?:the\s+)?)?(?:previous\s+one|next\s+one|same\s+one|last\s+one|first\s+one|the\s+above)\b",
        re.IGNORECASE
    )

    ACTION_ELLIPTICAL_PATTERN = re.compile(
        r"^(?:(?:please\s+)?(?:give|show|what\s+is|write|display|print)\s+(?:me\s+)?(?:the\s+|more\s+)?(?:code|output|example|details|inputs?|implementation)(?:\s+(?:for|of)\s+(?:it|this|that))?\??|"
        r"(?:explain|elaborate|clarify)(?:\s+(?:it|this|that|further|more))?\??|"
        r"what\s+does\s+(?:it|this|that)\s+do\??|"
        r"how\s+does\s+(?:it|this|that)\s+work\??|"
        r"give\s+(?:me\s+)?(?:an\s+)?example(?:\s+of\s+(?:it|this|that))?\??|"
        r"show\s+(?:its\s+)?output\??|"
        r"what\s+are\s+(?:its\s+)?inputs\??|"
        r"(?:give\s+me\s+|tell\s+me\s+)?(?:more\s+(?:details|info|information)|further\s+details)(?:\s+(?:on|about|for)\s+(?:it|this|that))?\??|"
        r"tell\s+me\s+more\??)$",
        re.IGNORECASE
    )

    @classmethod
    def resolve(
        cls,
        query: str,
        messages: List[Dict[str, Any]],
        ordinal_spec: Optional[OrdinalSpec] = None,
    ) -> FollowUpResolution:
        """
        Resolve conversational references (pronouns, relative positions, elliptical requests)
        against previous turns and sources.
        """
        clean_q = query.strip()
        if not messages or not clean_q:
            return FollowUpResolution(is_follow_up=False, resolved_query=clean_q)

        q_lower = clean_q.lower()

        # Find the last assistant message and last user message
        last_assistant_msg = next((m for m in reversed(messages) if m.get("role") == "assistant" and m.get("content")), None)
        last_user_msg = next((m for m in reversed(messages) if m.get("role") == "user" and m.get("content")), None)

        if not last_assistant_msg and not last_user_msg:
            return FollowUpResolution(is_follow_up=False, resolved_query=clean_q)

        prev_asst_text = last_assistant_msg.get("content", "") if last_assistant_msg else ""
        prev_user_text = last_user_msg.get("content", "") if last_user_msg else ""
        prev_sources = last_assistant_msg.get("sources_json") or [] if last_assistant_msg else []

        prev_doc_hint = prev_sources[0].get("filename") if (prev_sources and isinstance(prev_sources, list) and prev_sources[0].get("filename")) else None

        # Check follow-up indicators
        has_pronoun = bool(cls.PRONOUN_PATTERN.search(clean_q))
        has_relative = bool(cls.RELATIVE_ITEM_PATTERN.search(clean_q))
        has_elliptical = bool(cls.ACTION_ELLIPTICAL_PATTERN.search(clean_q))
        is_follow_up_candidate = has_pronoun or has_relative or has_elliptical or (ordinal_spec is not None)

        if not is_follow_up_candidate:
            return FollowUpResolution(is_follow_up=False, resolved_query=clean_q)

        # ----------------------------------------------------------------------
        # 1. Detect prior item context (e.g. Item 3: Largest of Two Numbers)
        # ----------------------------------------------------------------------
        item_num: Optional[int] = None
        item_title = ""
        is_last = False
        prev_idx: Optional[int] = None

        # Scan recent messages in reverse to find the most recent item context and doc hint
        for m in reversed(messages):
            if m.get("role") == "assistant":
                m_sources = m.get("sources_json") or []
                if isinstance(m_sources, str):
                    try:
                        m_sources = json.loads(m_sources)
                    except Exception:
                        m_sources = []
                if isinstance(m_sources, list) and m_sources:
                    if not prev_doc_hint and isinstance(m_sources[0], dict):
                        prev_doc_hint = m_sources[0].get("filename")
                    for s in m_sources:
                        if isinstance(s, dict):
                            sec = str(s.get("section") or "")
                            sec_m = re.search(r"Item\s+(\d+)(?:\s*[\:\-]\s*([^\n\r]+))?", sec, re.IGNORECASE)
                            if sec_m:
                                item_num = int(sec_m.group(1))
                                if sec_m.group(2):
                                    item_title = sec_m.group(2).strip()
                                break
                    if item_num is not None:
                        break

                # Also inspect assistant text content
                c_text = m.get("content", "")
                if not item_num and c_text:
                    m_asst = re.search(r"\(Item\s+(\d+)[\:\-]\s*([^\)]+)\)", c_text)
                    if m_asst:
                        item_num = int(m_asst.group(1))
                        item_title = m_asst.group(2).strip()
                        break
                    m_asst2 = re.search(r"\bItem\s+(\d+)[\:\-]\s*([^\n\r\,\.\(]+)", c_text, re.IGNORECASE)
                    if m_asst2:
                        item_num = int(m_asst2.group(1))
                        item_title = m_asst2.group(2).strip()
                        break
                    m_asst3 = re.search(r"\b(?:program|step|rule|section)\s+(\d+)\b", c_text, re.IGNORECASE)
                    if m_asst3:
                        item_num = int(m_asst3.group(1))
                        break

            elif m.get("role") == "user":
                u_ord = QueryAnalyzer._extract_ordinal_spec(m.get("content", ""))
                if u_ord:
                    if u_ord.is_last:
                        is_last = True
                        break
                    elif u_ord.indices:
                        prev_idx = u_ord.indices[0]
                        if not item_num:
                            item_num = prev_idx + 1
                        break

        if item_num is not None and prev_idx is None:
            prev_idx = item_num - 1

        # ----------------------------------------------------------------------
        # 2. Handle relative item shifts ("what about the previous one?", "next one")
        # ----------------------------------------------------------------------
        if re.search(r"\b(?:previous|earlier|prior)\s+(?:one|program|item|step|rule)\b", q_lower):
            curr_idx = prev_idx if prev_idx is not None else ((item_num - 1) if item_num else None)
            if curr_idx is not None and curr_idx > 0:
                target_idx = curr_idx - 1
                target_num = target_idx + 1
                ord_spec = OrdinalSpec(indices=[target_idx], raw_reference=f"Item {target_num}")
                resolved = f"Tell me about item {target_num} based on the document."
                return FollowUpResolution(
                    is_follow_up=True,
                    resolved_query=resolved,
                    carried_ordinal=ord_spec,
                    target_subjects=[f"Item {target_num}"],
                    target_document_hint=prev_doc_hint,
                )

        if re.search(r"\b(?:same)\s+(?:one|program|item|step|rule)\b", q_lower):
            curr_idx = prev_idx if prev_idx is not None else ((item_num - 1) if item_num else None)
            if curr_idx is not None:
                target_num = curr_idx + 1
                ord_spec = OrdinalSpec(indices=[curr_idx], raw_reference=f"Item {target_num}")
                label = f"Item {target_num}: {item_title}" if item_title else f"Item {target_num}"
                resolved = f"Tell me more about {label} based on the document."
                return FollowUpResolution(
                    is_follow_up=True,
                    resolved_query=resolved,
                    carried_ordinal=ord_spec,
                    target_subjects=[label],
                    target_document_hint=prev_doc_hint,
                )

        if re.search(r"\b(?:next|following)\s+(?:one|program|item|step|rule)\b", q_lower):
            curr_idx = prev_idx if prev_idx is not None else ((item_num - 1) if item_num else None)
            if curr_idx is not None:
                target_idx = curr_idx + 1
                target_num = target_idx + 1
                ord_spec = OrdinalSpec(indices=[target_idx], raw_reference=f"Item {target_num}")
                resolved = f"Tell me about item {target_num} based on the document."
                return FollowUpResolution(
                    is_follow_up=True,
                    resolved_query=resolved,
                    carried_ordinal=ord_spec,
                    target_subjects=[f"Item {target_num}"],
                    target_document_hint=prev_doc_hint,
                )

        # ----------------------------------------------------------------------
        # 3. Handle list queries with ordinal follow-up ("what about the second one?")
        # ----------------------------------------------------------------------
        items_in_asst = cls._extract_ordered_items_from_text(prev_asst_text)
        if ordinal_spec and items_in_asst:
            target_item_label = None
            if ordinal_spec.is_last and items_in_asst:
                target_item_label = items_in_asst[-1]
            elif ordinal_spec.indices:
                idx = ordinal_spec.indices[0]
                if 0 <= idx < len(items_in_asst):
                    target_item_label = items_in_asst[idx]

            if target_item_label:
                resolved = f"Explain and provide full details for '{target_item_label}' based on the document."
                return FollowUpResolution(
                    is_follow_up=True,
                    resolved_query=resolved,
                    carried_ordinal=ordinal_spec,
                    target_subjects=[target_item_label],
                    target_document_hint=prev_doc_hint,
                )

        # ----------------------------------------------------------------------
        # 4. Handle item-specific follow-ups ("give the code for it", "explain it")
        # ----------------------------------------------------------------------
        if ordinal_spec and not has_relative and not has_pronoun:
            # Standalone ordinal inquiry: user is explicitly asking for a specific item (e.g. "first program")
            return FollowUpResolution(
                is_follow_up=False,
                resolved_query=clean_q,
                target_document_hint=prev_doc_hint,
            )

        if (item_num is not None or is_last) and not ordinal_spec:
            label = (f"Item {item_num}: {item_title}" if item_title else f"Item {item_num}") if item_num else "the last item"
            target_idx = (item_num - 1) if item_num is not None else 19
            carried_ord = (
                OrdinalSpec(is_last=True, raw_reference="the last item")
                if is_last
                else OrdinalSpec(indices=[target_idx], raw_reference=label)
            )

            # Rewrite query to explicitly include the target item label
            resolved = clean_q
            if re.search(r"\b(?:for|of|about)\s+(?:it|this|that|the\s+above)\b", q_lower):
                resolved = re.sub(
                    r"\b(for|of|about)\s+(?:it|this|that|the\s+above)\b",
                    f"\\1 {label}",
                    clean_q,
                    flags=re.IGNORECASE
                )
            elif re.search(r"\b(?:it|this|that|this\s+program|that\s+program|the\s+above)\b", q_lower):
                resolved = re.sub(
                    r"\b(?:it|this|that|this\s+program|that\s+program|the\s+above)\b",
                    label,
                    clean_q,
                    flags=re.IGNORECASE
                )
            elif re.search(r"\b(?:its|their)\b", q_lower):
                resolved = re.sub(
                    r"\b(?:its|their)\b",
                    f"{label}'s",
                    clean_q,
                    flags=re.IGNORECASE
                )
            else:
                resolved = f"{clean_q} for {label}"

            return FollowUpResolution(
                is_follow_up=True,
                resolved_query=resolved,
                carried_ordinal=carried_ord,
                target_subjects=[label],
                target_document_hint=prev_doc_hint,
            )

        # ----------------------------------------------------------------------
        # 5. Handle general entity follow-ups for non-numbered documents
        # ----------------------------------------------------------------------
        if ordinal_spec or not (has_pronoun or has_relative or has_elliptical):
            return FollowUpResolution(
                is_follow_up=False,
                resolved_query=clean_q,
                target_document_hint=prev_doc_hint,
            )

        prev_subj = None
        # Check proper noun / character / entity from previous user or assistant message
        subj_matches = re.findall(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b", prev_user_text)
        # Filter out common stop words
        common_words = {"What", "Where", "When", "Which", "Who", "Tell", "Give", "Explain", "Describe", "The"}
        clean_subjs = [s for s in subj_matches if s not in common_words]
        if clean_subjs:
            prev_subj = clean_subjs[0]
        elif prev_asst_text:
            asst_subjs = re.findall(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b", prev_asst_text)
            clean_asst_subjs = [s for s in asst_subjs if s not in common_words]
            if clean_asst_subjs:
                prev_subj = clean_asst_subjs[0]

        if prev_subj:
            resolved = clean_q
            if re.search(r"\b(?:his|her|their|its)\b", q_lower):
                resolved = re.sub(
                    r"\b(?:his|her|their|its)\b",
                    f"{prev_subj}'s",
                    resolved,
                    flags=re.IGNORECASE
                )
            if re.search(r"\b(?:he|him|she|it|this|that|they|them)\b", resolved.lower()):
                resolved = re.sub(
                    r"\b(?:he|him|she|it|this|that|they|them)\b",
                    prev_subj,
                    resolved,
                    flags=re.IGNORECASE
                )
            if resolved == clean_q:
                resolved = f"{clean_q} regarding {prev_subj}"

            return FollowUpResolution(
                is_follow_up=True,
                resolved_query=resolved,
                target_subjects=[prev_subj],
                target_document_hint=prev_doc_hint,
            )

        # Fallback if follow-up detected but specific entity could not be isolated
        return FollowUpResolution(
            is_follow_up=True,
            resolved_query=f"{clean_q} (continuing from previous turn)",
            target_document_hint=prev_doc_hint,
        )

    @classmethod
    def _extract_ordered_items_from_text(cls, text: str) -> List[str]:
        """Extract item labels from numbered lists, bullet points, or markdown headers."""
        items = []

        # 1. Numbered lists: "1. Program Name" or "1) Title"
        numbered = re.findall(r"(?:^|\n)\s*(?:\d+[\.\)]\s+)([^\n\:\.\(\)]+)", text)
        for n in numbered:
            cleaned = n.strip(" *_-#`")
            if cleaned and len(cleaned.split()) <= 8:
                items.append(cleaned)

        if items:
            return items

        # 2. Bullet points: "- Title" or "• Title"
        bullets = re.findall(r"(?:^|\n)\s*[\-\*•]\s+([^\n\:\.\(\)]+)", text)
        for b in bullets:
            cleaned = b.strip(" *_-#`")
            if cleaned and len(cleaned.split()) <= 8:
                items.append(cleaned)

        return items
