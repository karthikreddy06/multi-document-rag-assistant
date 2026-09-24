"""
Hybrid and Adaptive Retrieval Module.
Combines dense vector similarity with lexical term matching, generic query understanding,
document resolution, retrieval strategy planning, document-locality context propagation,
aspect-guaranteed candidate fusion, and multi-pass coverage verification.
Handles single-fact, multi-aspect, document-wide, ordinal, and cross-document questions.
"""

from collections import defaultdict
from dataclasses import dataclass, field
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from enum import Enum
from app.config import settings
from app.embeddings.service import EmbeddingService
from app.models import RetrievedChunk
from app.retrieval.query_understanding import QueryAnalyzer, QueryAnalysis, QueryIntent, OrdinalSpec
from app.retrieval.document_resolver import DocumentResolver, DocumentResolutionResult, ResolvedDocument
from app.retrieval.planner import RetrievalPlanner, RetrievalPlan, RetrievalStrategy, GenerationBudget
from app.retrieval.ordered_sequence import OrderedSequenceResolver, DocumentItem
from app.utils.logger import setup_logger
from app.vectorstore.store import VectorStore

logger = setup_logger("retrieval.retriever")


class EvidenceStatus(str, Enum):
    """Assessment of factual grounding in retrieved evidence."""
    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    ABSENT = "absent"
    CONFLICTING = "conflicting"


@dataclass
class EvidenceAssessment:
    """Detailed evaluation of evidence completeness."""
    status: EvidenceStatus = EvidenceStatus.SUFFICIENT
    confidence: float = 1.0
    missing_aspects: List[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class CoverageInfo:
    """Tracks retrieval coverage across requested aspects, subjects, documents, and pages."""
    aspects_requested: List[str] = field(default_factory=list)
    aspects_covered: List[str] = field(default_factory=list)
    missing_aspects: List[str] = field(default_factory=list)
    documents_covered: List[str] = field(default_factory=list)
    pages_covered: List[int] = field(default_factory=list)
    coverage_ratio: float = 1.0
    secondary_pass_triggered: bool = False
    evidence_assessment: Optional[EvidenceAssessment] = None


class EvidenceEvaluator:
    """
    Evaluates whether retrieved evidence is sufficient, partial, or absent
    prior to invoking the generation layer, preventing hallucination.
    """
    STOPWORDS = {
        "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
        "tell", "give", "show", "describe", "explain", "does", "have", "with",
        "from", "about", "this", "that", "these", "those", "document", "file", "text",
        "the", "is", "are", "was", "were", "and", "for", "please", "can", "you"
    }

    @classmethod
    def evaluate(
        cls,
        query: str,
        chunks: List[RetrievedChunk],
        plan: Optional[RetrievalPlan] = None,
    ) -> EvidenceAssessment:
        if not chunks:
            return EvidenceAssessment(
                status=EvidenceStatus.ABSENT,
                confidence=0.0,
                reason="No chunks retrieved.",
            )

        # Determine effective query for token matching (e.g. resolve follow-ups)
        eval_query = plan.query if (plan and getattr(plan, "is_follow_up", False) and getattr(plan, "query", None)) else query
        combined_text = " ".join(c.text.lower() for c in chunks)
        q_tokens = [w for w in re.findall(r"\b[a-zA-Z0-9_]{3,}\b", eval_query.lower())]
        substantive_tokens = [t for t in q_tokens if t not in cls.STOPWORDS]

        if not substantive_tokens:
            return EvidenceAssessment(status=EvidenceStatus.SUFFICIENT, confidence=1.0)

        matched_tokens = [t for t in substantive_tokens if t in combined_text]
        match_ratio = len(matched_tokens) / len(substantive_tokens)

        # Code verification check across programming languages (Python, JS, C, Java, Bash)
        is_code_request = bool(
            (plan and getattr(plan, "is_code_request", False)) or
            re.search(r"\b(?:code|source\s+code|script|implementation|syntax)\b", eval_query.lower())
        )
        if is_code_request:
            has_code_syntax = any(
                c.metadata.get("content_type") == "code" or
                "```" in c.text or
                bool(re.search(r"\b(?:def\s+\w+|class\s+\w+|import\s+\w+|from\s+\w+|input\(|print\(|return\b|while\b|for\s+\w+\s+in|console\.log|printf|System\.out)\b", c.text))
                for c in chunks
            )
            if not has_code_syntax:
                return EvidenceAssessment(
                    status=EvidenceStatus.ABSENT,
                    confidence=0.0,
                    reason="Code was requested but no source code syntax found in retrieved chunks.",
                )

        if match_ratio == 0.0:
            return EvidenceAssessment(
                status=EvidenceStatus.ABSENT,
                confidence=0.0,
                reason="None of the substantive query terms appear in the retrieved evidence.",
            )
        elif match_ratio < 0.35 and len(substantive_tokens) >= 3:
            return EvidenceAssessment(
                status=EvidenceStatus.PARTIAL,
                confidence=match_ratio,
                missing_aspects=[t for t in substantive_tokens if t not in matched_tokens],
                reason=f"Only {int(match_ratio * 100)}% of query terms found in evidence.",
            )

        return EvidenceAssessment(status=EvidenceStatus.SUFFICIENT, confidence=match_ratio)


class QueryDecomposer:
    """
    Analyzes queries and decomposes complex, multi-part, or comparative questions
    into distinct information needs to maximize retrieval coverage across chunks and documents.
    Protects compound titles and entities from false splitting and preserves global scope.
    """

    CLAUSE_INDICATORS = (
        r"what", r"when", r"where", r"which", r"who", r"whom", r"whose",
        r"why", r"how", r"describe", r"explain", r"tell\s+me", r"list",
        r"details?\s+of", r"information\s+about", r"give\s+me"
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

    COMMON_PREPOSITIONS = ("in", "of", "for", "between", "across", "from", "with", "regarding")

    @classmethod
    def _clean_token(cls, t: str) -> str:
        s = t.strip(" ?,.:; \t\n")
        return re.sub(r"^(?:and\s+)+", "", s, flags=re.IGNORECASE).strip(" ?,.:; \t\n")

    @classmethod
    def _split_subjects(cls, s: str) -> List[str]:
        s = s.strip(" ?,.")

        vs_m = re.split(r"\s+\b(?:with|versus|vs\.?)\b\s+", s, flags=re.IGNORECASE)
        if len(vs_m) > 1:
            return [cls._clean_token(p) for p in vs_m if cls._clean_token(p)]

        if "," in s:
            parts = re.split(r",\s*(?:and\s+)?", s, flags=re.IGNORECASE)
            cleaned = [cls._clean_token(p) for p in parts if cls._clean_token(p)]
            if len(cleaned) > 1:
                return cleaned

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
    def _decompose_comparisons(cls, core: str, clean_q: str, scope: str = "") -> Optional[List[str]]:
        differ_m = re.match(cls.DIFFER_PATTERN, core, re.IGNORECASE)
        if differ_m:
            side_a = differ_m.group(1).strip(" ?,.")
            side_b = differ_m.group(2).strip(" ?,.")
            if len(side_a) > 2 and len(side_b) > 2:
                prep_regex = r"\s+\b(" + "|".join(cls.COMMON_PREPOSITIONS) + r")\b\s+"
                prep_m = re.search(prep_regex, side_a, flags=re.IGNORECASE)
                if prep_m:
                    attr_candidate = side_a[:prep_m.start()].strip()
                    prep_candidate = prep_m.group(1).lower()
                    is_proper_name = len(attr_candidate.split()) == 1 and attr_candidate[0].isupper()
                    if attr_candidate and not is_proper_name:
                        if not side_b.lower().startswith(attr_candidate.lower()):
                            cleaned_b = side_b
                            if cleaned_b.lower().startswith(prep_candidate + " "):
                                cleaned_b = cleaned_b[len(prep_candidate):].strip()
                            side_b = f"{attr_candidate} {prep_candidate} {cleaned_b}"
                sq1 = f"{scope}, {side_a}" if scope else side_a
                sq2 = f"{scope}, {side_b}" if scope else side_b
                return [sq1, sq2, clean_q]

        pref_m = re.match(cls.COMPARISON_PREFIXES, core, re.IGNORECASE)
        if not pref_m:
            vs_m = re.search(r"^(.*?)\s+\b(?:versus|vs\.?)\b\s+(.+)$", core, re.IGNORECASE)
            if vs_m:
                side_a = vs_m.group(1).strip(" ?,.")
                side_b = vs_m.group(2).strip(" ?,.")
                if len(side_a) > 2 and len(side_b) > 2:
                    sq1 = f"{scope}, {side_a}" if scope else side_a
                    sq2 = f"{scope}, {side_b}" if scope else side_b
                    return [sq1, sq2, clean_q]
            return None

        body = core[pref_m.end():].strip()

        based_on_m = re.search(
            r"\s+\b(?:based\s+on|in\s+terms\s+of|regarding|with\s+respect\s+to)\s+(.+)$",
            body,
            re.IGNORECASE
        )
        if based_on_m:
            subj_str = body[:based_on_m.start()].strip()
            attr_str = based_on_m.group(1).strip()
            subjects = cls._split_subjects(subj_str)
            if len(subjects) >= 2:
                subqueries = []
                for s in subjects:
                    sq = f"{attr_str} in {s}"
                    if scope:
                        sq = f"{scope}, {sq}"
                    subqueries.append(sq)
                subqueries.append(clean_q)
                return subqueries

        prep_regex = r"\b(" + "|".join(cls.COMMON_PREPOSITIONS) + r")\b"
        prep_matches = list(re.finditer(prep_regex, body, re.IGNORECASE))

        for pm in prep_matches:
            potential_attr = body[:pm.start()].strip()
            prep = pm.group(1).lower()
            potential_subj_str = body[pm.end():].strip()

            if not potential_attr or len(potential_attr.split()) < 1:
                continue

            if len(potential_attr.split()) == 1 and potential_attr[0].isupper() and not potential_attr.lower().startswith("the"):
                continue

            subjects = cls._split_subjects(potential_subj_str)
            if len(subjects) >= 2:
                subqueries = []
                for s in subjects:
                    sq = f"{potential_attr} {prep} {s}"
                    if scope:
                        sq = f"{scope}, {sq}"
                    subqueries.append(sq)
                subqueries.append(clean_q)
                return subqueries

        subjects = cls._split_subjects(body)
        if len(subjects) >= 2:
            subqueries = []
            for s in subjects:
                sq = f"{scope}, {s}" if scope else s
                subqueries.append(sq)
            subqueries.append(clean_q)
            return subqueries

        return None

    @classmethod
    def decompose(cls, query: str) -> List[str]:
        clean_q = query.strip()
        if not clean_q:
            return []

        scope = cls._extract_scope(clean_q)
        working_text = clean_q
        if scope:
            working_text = re.sub(
                r"^(?:in|for|according\s+to|regarding)\s+[^,]+,\s*",
                "",
                clean_q,
                flags=re.IGNORECASE
            ).strip()

        core = working_text.rstrip("?. ")

        comp_subqueries = cls._decompose_comparisons(core, clean_q, scope)
        if comp_subqueries:
            return cls._deduplicate(comp_subqueries)

        multi_q = re.split(r"(?:\?\s+|\;\s*|\b\d+[\.\)]\s+)", working_text)
        multi_q = [q.strip(" ?,.") for q in multi_q if len(q.strip(" ?,.")) > 3]
        if len(multi_q) > 1:
            subqueries = []
            for i, part in enumerate(multi_q):
                if scope and not cls._has_scope(part, scope):
                    subqueries.append(f"{scope}, {part}")
                else:
                    subqueries.append(part)
            subqueries.append(clean_q)
            return cls._deduplicate(subqueries)

        indicators_regex = "|".join(cls.CLAUSE_INDICATORS)
        split_pattern = (
            r"(?:,\s*(?:and|as\s+well\s+as)\s+|"
            r"\s+as\s+well\s+as\s+|"
            r"\s+and\s+(?=(?:" + indicators_regex + r")\b))"
        )

        parts = re.split(split_pattern, working_text, flags=re.IGNORECASE)
        if len(parts) > 1:
            subqueries = []
            main_subject = cls._extract_subject(parts[0])
            for i, part in enumerate(parts):
                part_cleaned = part.strip(" ?,.")
                if not part_cleaned or len(part_cleaned.split()) < 2:
                    continue

                if scope:
                    enriched = f"{scope}, {part_cleaned}"
                elif i > 0 and main_subject and main_subject.lower() not in part_cleaned.lower():
                    enriched = f"{part_cleaned} ({main_subject})"
                else:
                    enriched = part_cleaned

                subqueries.append(enriched)

            if len(subqueries) > 1:
                subqueries.append(clean_q)
                return cls._deduplicate(subqueries)

        coord_match = re.match(
            r"^(what\s+(?:is|are)\s+(?:all\s+)?(?:my\s+)?|tell\s+me\s+about\s+|describe\s+)(.+?)\s+and\s+(.+)$",
            working_text,
            re.IGNORECASE,
        )
        if coord_match:
            prefix = coord_match.group(1).strip()
            item_a = coord_match.group(2).strip(" ?,.")
            item_b = coord_match.group(3).strip(" ?,.")
            if re.match(r"^The\s+[A-Z]", item_a) and re.match(r"^(?:the\s+)?[A-Z]", item_b):
                return [clean_q]

            if 0 < len(item_a.split()) <= 5 and 0 < len(item_b.split()) <= 5:
                sq1 = f"{prefix} {item_a}"
                sq2 = f"{prefix} {item_b}"
                if scope:
                    sq1 = f"{scope}, {sq1}"
                    sq2 = f"{scope}, {sq2}"
                return cls._deduplicate([sq1, sq2, clean_q])

        return [clean_q]

    @staticmethod
    def _extract_scope(text: str) -> str:
        match = re.match(r"^(in|for|according\s+to|regarding)\s+([^,]+),", text, re.IGNORECASE)
        if match:
            return match.group(0).rstrip(",")
        return ""

    @staticmethod
    def _has_scope(text: str, scope: str) -> bool:
        scope_tokens = set(re.findall(r"\w+", scope.lower())) - {"in", "for", "according", "to", "regarding"}
        text_tokens = set(re.findall(r"\w+", text.lower()))
        return bool(scope_tokens & text_tokens)

    @staticmethod
    def _extract_subject(clause: str) -> str:
        in_match = re.search(r"\bin\s+([A-Z][a-zA-Z0-9_\s\.\-]{2,})", clause)
        if in_match:
            return in_match.group(1).strip(" ?,.")

        cap_seq = re.findall(r"\b[A-Z][a-zA-Z0-9_\-]+\b(?:\s+[A-Z][a-zA-Z0-9_\-]+)*", clause)
        if cap_seq:
            filtered = [
                s for s in cap_seq
                if s.lower() not in {"what", "who", "how", "why", "where", "when", "can", "is", "are"}
            ]
            if filtered:
                return filtered[-1]

        noun_match = re.search(r"\b(the\s+[a-z]+(?:\s+[a-z]+)?)\b", clause, re.IGNORECASE)
        if noun_match:
            candidate = noun_match.group(1).strip()
            if candidate.lower() not in {"the what", "the difference", "the following"}:
                return candidate

        return ""

    @staticmethod
    def _deduplicate(items: List[str]) -> List[str]:
        seen = set()
        result = []
        for item in items:
            norm = item.strip().lower()
            if norm and norm not in seen:
                seen.add(norm)
                result.append(item.strip())
        return result


class HybridRetriever:
    """
    Adaptive Hybrid Retriever executing dynamic retrieval plans.
    Combines dense embeddings with lexical keyword scoring, sequential document-wide traversal,
    independent cross-document comparison, aspect-guaranteed reciprocal rank fusion,
    and iterative multi-pass coverage verification.
    """

    STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
        "to", "was", "were", "will", "with", "what", "which", "who", "tell",
        "me", "about", "all", "my", "your", "do", "i", "how", "have", "or"
    }

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_service: Optional[EmbeddingService] = None,
        top_k: Optional[int] = None,
    ):
        self.vector_store = vector_store or VectorStore()
        self.embedding_service = embedding_service or EmbeddingService()
        self.top_k = top_k or settings.top_k

    def _tokenize(self, text: str) -> List[str]:
        words = re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower())
        return [w for w in words if w not in self.STOPWORDS]

    def _calculate_lexical_score(
        self, query: str, text: str, section: str, filename: str = ""
    ) -> float:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return 0.0

        unique_q_tokens = set(query_tokens)
        doc_lower = text.lower()
        sec_lower = section.lower()
        fn_lower = filename.lower()

        matched_tokens = 0
        freq_score = 0.0

        for token in unique_q_tokens:
            pattern = r"\b" + re.escape(token) + r"\b"
            in_fn = bool(re.search(pattern, fn_lower))
            in_sec = bool(re.search(pattern, sec_lower))
            counts = len(re.findall(pattern, doc_lower))

            if in_fn:
                freq_score += 1.5
            if in_sec:
                freq_score += 2.0
            if counts > 0:
                matched_tokens += 1
                freq_score += min(math.log1p(counts), 2.0)
            elif in_fn or in_sec:
                matched_tokens += 1

        coverage = matched_tokens / len(unique_q_tokens)
        clean_q = " ".join(query_tokens)
        phrase_bonus = 0.0
        if len(query_tokens) >= 2 and clean_q in doc_lower:
            phrase_bonus = 1.0

        # Numerical / exact identifier matching bonus
        number_tokens = [t for t in unique_q_tokens if re.search(r"\d", t)]
        num_bonus = 0.0
        for nt in number_tokens:
            if re.search(r"\b" + re.escape(nt) + r"\b", doc_lower):
                num_bonus += 0.5

        raw_score = (freq_score / len(unique_q_tokens)) * (0.5 + 0.5 * coverage) + phrase_bonus + num_bonus
        return min(raw_score, 3.0) / 3.0

    def _match_target_document(self, query: str) -> Optional[str]:
        try:
            indexed_files = list(self.vector_store.get_indexed_files().keys())
        except Exception:
            indexed_files = []

        if not indexed_files:
            return None

        q_lower = query.lower()
        q_words = [w for w in re.findall(r"\b[a-z0-9]+\b", q_lower) if w not in self.STOPWORDS]
        if not q_words:
            return None

        matches = []
        for fn in indexed_files:
            base = re.sub(r"\.[a-zA-Z0-9]+$", "", fn).lower()
            base_clean = re.sub(r"[^a-z0-9]", "", base)
            base_parts = set(re.findall(r"[a-z0-9]+", base)) - self.STOPWORDS

            common_parts = {w for w in q_words if w in base_parts and len(w) >= 3}
            concat_match_len = 0
            for i in range(len(q_words)):
                for j in range(i + 2, min(i + 6, len(q_words) + 1)):
                    joined = "".join(q_words[i:j])
                    if len(joined) >= 4 and (joined in base_clean or base_clean in joined):
                        if len(joined) > concat_match_len:
                            concat_match_len = len(joined)

            score = len(common_parts) * 10 + concat_match_len
            if score >= 6:
                matches.append((fn, score))

        if len(matches) == 1:
            return matches[0][0]
        return None

    def _retrieve_for_subquery(
        self,
        subquery: str,
        k: int,
        total_chunks: int,
        context_query: Optional[str] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        query_embedding = self.embedding_service.embed_text(subquery)
        if not query_embedding:
            logger.error(f"Failed to generate query embedding for subquery: '{subquery}'")
            return []

        target_doc = self._match_target_document(subquery)
        if target_doc and where:
            if "$and" in where and isinstance(where["$and"], list):
                where_filter = {"$and": [*where["$and"], {"filename": target_doc}]}
            else:
                where_filter = {"$and": [where, {"filename": target_doc}]}
        elif target_doc:
            where_filter = {"filename": target_doc}
        elif where:
            where_filter = where
        else:
            where_filter = None

        candidate_k = min(max(k * 15, 120), total_chunks)
        raw_results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=candidate_k,
            where=where_filter
        )

        doc_list = list(raw_results.get("documents", [[]])[0])
        meta_list = list(raw_results.get("metadatas", [[]])[0])
        dist_list = list(raw_results.get("distances", [[]])[0]) if "distances" in raw_results else [1.0] * len(doc_list)

        if target_doc and not doc_list:
            raw_results = self.vector_store.query(
                query_embedding=query_embedding,
                top_k=candidate_k,
                where=where
            )
            doc_list = list(raw_results.get("documents", [[]])[0])
            meta_list = list(raw_results.get("metadatas", [[]])[0])
            dist_list = list(raw_results.get("distances", [[]])[0]) if "distances" in raw_results else [1.0] * len(doc_list)

        if not doc_list:
            return []

        candidates: List[RetrievedChunk] = []
        scoring_query = context_query or subquery

        for doc_text, meta, dist in zip(doc_list, meta_list, dist_list):
            dense_similarity = 1.0 / (1.0 + max(dist, 0.0))
            section_name = str(meta.get("section", "General"))
            filename = str(meta.get("filename", ""))

            lex_sq = self._calculate_lexical_score(subquery, doc_text, section_name, filename)
            lex_full = (
                self._calculate_lexical_score(scoring_query, doc_text, section_name, filename)
                if context_query else lex_sq
            )
            lexical_score = max(lex_sq, lex_full)
            hybrid_score = (0.65 * dense_similarity) + (0.35 * lexical_score)

            candidates.append(RetrievedChunk(
                text=doc_text,
                metadata=meta,
                score=hybrid_score
            ))

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    @staticmethod
    def _are_sections_contiguous(sec1: str, sec2: str) -> bool:
        s1, s2 = sec1.lower(), sec2.lower()
        if s1 == s2:
            return True
        if "general" in s1 and "general" in s2:
            return True
        base1 = s1.split(":")[0].strip()
        base2 = s2.split(":")[0].strip()
        return bool(base1 and base1 == base2)

    def _fuse_with_aspect_guarantee_and_locality(
        self,
        per_need_candidates: List[List[RetrievedChunk]],
        k: int,
        k_rrf: int = 60
    ) -> List[RetrievedChunk]:
        if not per_need_candidates:
            return []

        num_needs = len(per_need_candidates)
        rrf_scores: Dict[str, float] = defaultdict(float)
        chunk_map: Dict[str, RetrievedChunk] = {}
        chunk_file: Dict[str, str] = {}
        chunk_idx_map: Dict[str, int] = {}
        top_anchors: List[Tuple[str, int, str, float]] = []

        for need_idx, cands in enumerate(per_need_candidates):
            weight = 0.8 if (num_needs > 1 and need_idx == num_needs - 1) else 1.0
            for rank, c in enumerate(cands):
                cid = c.chunk_id or f"{c.metadata.get('filename')}_{c.metadata.get('chunk_index')}"
                chunk_map[cid] = c
                fn = str(c.metadata.get("filename", ""))
                chunk_file[cid] = fn
                sec = str(c.metadata.get("section", "General"))

                raw_cidx = c.metadata.get("chunk_index")
                if raw_cidx is not None:
                    try:
                        cidx = int(raw_cidx)
                        chunk_idx_map[cid] = cidx
                    except (ValueError, TypeError):
                        pass

                pts = weight * (1.0 / (k_rrf + rank + 1))
                rrf_scores[cid] += pts

                if rank < 2 and raw_cidx is not None:
                    try:
                        top_anchors.append((fn, int(raw_cidx), sec, pts))
                    except (ValueError, TypeError):
                        pass

        for anchor_fn, anchor_cidx, anchor_sec, anchor_pts in top_anchors:
            for cid, c in chunk_map.items():
                if chunk_file.get(cid) == anchor_fn and cid in chunk_idx_map:
                    dist = abs(chunk_idx_map[cid] - anchor_cidx)
                    if 1 <= dist <= 2:
                        target_sec = str(c.metadata.get("section", "General"))
                        if self._are_sections_contiguous(anchor_sec, target_sec):
                            bonus = (0.3 / dist) * anchor_pts
                            rrf_scores[cid] += bonus

        selected: List[RetrievedChunk] = []
        selected_ids: Set[str] = set()

        def add_chunk(c: RetrievedChunk) -> bool:
            cid = c.chunk_id or f"{c.metadata.get('filename')}_{c.metadata.get('chunk_index')}"
            if cid not in selected_ids:
                selected_ids.add(cid)
                selected.append(c)
                return True
            return False

        # Step 1: Aspect Coverage Guarantee
        for need_idx in range(num_needs):
            if len(selected) >= k:
                break
            cands = per_need_candidates[need_idx]
            for cand in cands:
                if add_chunk(cand):
                    break

        # Step 2: Fill remaining slots by overall RRF score descending
        sorted_by_rrf = sorted(
            chunk_map.keys(),
            key=lambda cid: rrf_scores[cid],
            reverse=True
        )
        for cid in sorted_by_rrf:
            if len(selected) >= k:
                break
            add_chunk(chunk_map[cid])

        # Step 3: Set normalized score
        for c in selected:
            cid = c.chunk_id or f"{c.metadata.get('filename')}_{c.metadata.get('chunk_index')}"
            c.score = max(c.score, rrf_scores[cid] * 25.0)

        selected.sort(key=lambda c: c.score, reverse=True)
        return selected[:k]

    # --------------------------------------------------------------------------
    # ADAPTIVE STRATEGY IMPLEMENTATIONS
    # --------------------------------------------------------------------------

    def _retrieve_document_wide(
        self,
        plan: RetrievalPlan,
    ) -> List[RetrievedChunk]:
        """
        Exhaustive / Document-Wide Retrieval Strategy:
        Scans across targeted document(s) in sequential document order (page -> chunk_index).
        Does not truncate to arbitrary top-k; returns sequential document evidence.
        """
        logger.info(f"Executing DOCUMENT_WIDE retrieval for query: '{plan.query}'")
        raw_chunks: List[RetrievedChunk] = []

        try:
            # If where_filter is specified, pull all chunks matching the filter
            if plan.where_filter:
                res = self.vector_store.collection.get(
                    where=plan.where_filter,
                    include=["metadatas", "documents"],
                )
            elif plan.target_documents:
                doc_filenames = [d.filename for d in plan.target_documents]
                if len(doc_filenames) == 1:
                    where_cond = {"filename": doc_filenames[0]}
                else:
                    where_cond = {"$or": [{"filename": fn} for fn in doc_filenames]}
                res = self.vector_store.collection.get(
                    where=where_cond,
                    include=["metadatas", "documents"],
                )
            else:
                # Fallback to query with large candidate pool
                q_emb = self.embedding_service.embed_text(plan.query)
                res = self.vector_store.query(
                    query_embedding=q_emb,
                    top_k=min(plan.candidate_k, self.vector_store.count()),
                    where=plan.where_filter,
                )

            docs = res.get("documents", [])
            metas = res.get("metadatas", [])
            if docs and isinstance(docs[0], list):
                docs = docs[0]
                metas = metas[0] if metas else []

            for doc_text, meta in zip(docs, metas):
                raw_chunks.append(RetrievedChunk(
                    text=doc_text,
                    metadata=meta or {},
                    score=1.0,
                ))
        except Exception as e:
            logger.error(f"Error fetching document-wide chunks: {e}")
            return []

        unique_docs = {str(c.metadata.get("filename", "")) for c in raw_chunks}
        if len(unique_docs) <= 1:
            raw_chunks.sort(key=lambda c: (
                int(c.metadata.get("page_number", 1)),
                int(c.metadata.get("chunk_index", 0)),
            ))
        else:
            raw_chunks.sort(key=lambda c: c.score, reverse=True)

        # Check if the document contains an ordered sequence of items
        # (e.g. numbered programs, steps, rules, sections)
        items = OrderedSequenceResolver.extract_sequence(raw_chunks)
        if items and len(items) >= 2:
            plan.resolved_items = items
            target_fn = plan.target_documents[0].filename if plan.target_documents else ""
            evidence = OrderedSequenceResolver.build_item_evidence_chunks(items, target_fn)
            if evidence:
                logger.info(f"DOCUMENT_WIDE extracted {len(items)} items, assembled {len(evidence)} evidence chunks.")
                return evidence[:plan.final_top_k]

        logger.info(f"DOCUMENT_WIDE retrieval found {len(raw_chunks)} sequential chunks across pages.")
        return raw_chunks[:plan.final_top_k]

    def _retrieve_comparison(
        self,
        plan: RetrievalPlan,
    ) -> List[RetrievedChunk]:
        """
        Comparison Retrieval Strategy:
        Performs independent candidate retrieval for each compared subject or document.
        Guarantees balanced representation so neither subject is crowded out.
        """
        logger.info(f"Executing COMPARISON retrieval for query: '{plan.query}'")
        total_chunks = self.vector_store.count()

        subqueries = QueryDecomposer.decompose(plan.query)
        if len(subqueries) <= 1:
            subqueries = [f"{s} in {plan.query}" for s in (plan.subjects or [d.filename for d in plan.target_documents])]
            if not subqueries:
                subqueries = [plan.query]

        per_need_candidates: List[List[RetrievedChunk]] = []
        for sq in subqueries:
            cands = self._retrieve_for_subquery(
                subquery=sq,
                k=plan.final_top_k * 3,
                total_chunks=total_chunks,
                context_query=plan.query,
                where=plan.where_filter,
            )
            per_need_candidates.append(cands)

        # Ensure balanced representation across distinct information needs
        num_needs = len(per_need_candidates)
        quota_per_need = max(plan.final_top_k // max(num_needs, 1), 2)

        selected: List[RetrievedChunk] = []
        selected_ids: Set[str] = set()

        def add_chunk(c: RetrievedChunk) -> bool:
            cid = c.chunk_id or f"{c.metadata.get('filename')}_{c.metadata.get('chunk_index')}"
            if cid not in selected_ids:
                selected_ids.add(cid)
                selected.append(c)
                return True
            return False

        # Pass 1: Add balanced quota from each need/document
        for cands in per_need_candidates:
            added = 0
            for c in cands:
                if add_chunk(c):
                    added += 1
                    if added >= quota_per_need:
                        break

        # Pass 2: Fill remaining slots via RRF
        if len(selected) < plan.final_top_k:
            fused = self._fuse_with_aspect_guarantee_and_locality(per_need_candidates, plan.final_top_k)
            for c in fused:
                if len(selected) >= plan.final_top_k:
                    break
                add_chunk(c)

        logger.info(f"COMPARISON retrieval assembled {len(selected)} balanced chunks across {num_needs} needs.")
        return selected[:plan.final_top_k]

    def _retrieve_multi_aspect(
        self,
        plan: RetrievalPlan,
    ) -> Tuple[List[RetrievedChunk], CoverageInfo]:
        """
        Multi-Aspect Retrieval Strategy with Iterative Coverage Verification:
        Decomposes query into information needs, retrieves independent candidates per aspect,
        verifies aspect coverage, and triggers a secondary targeted pass if evidence is missing.
        """
        logger.info(f"Executing MULTI_ASPECT retrieval for aspects: {plan.aspects}")
        total_chunks = self.vector_store.count()
        per_need_candidates: List[List[RetrievedChunk]] = []
        aspects_covered: List[str] = []
        missing_aspects: List[str] = []

        for aspect in plan.aspects:
            cands = self._retrieve_for_subquery(
                subquery=aspect,
                k=plan.candidate_k,
                total_chunks=total_chunks,
                context_query=plan.query,
                where=plan.where_filter,
            )
            per_need_candidates.append(cands)
            if cands and cands[0].score >= 0.25:
                aspects_covered.append(aspect)
            else:
                missing_aspects.append(aspect)

        secondary_triggered = False

        # Iterative Retrieval Loop: if any requested aspect has weak/missing coverage, trigger secondary pass
        if missing_aspects:
            logger.info(f"Coverage analysis identified missing aspects: {missing_aspects}. Triggering secondary retrieval pass...")
            secondary_triggered = True
            for idx, aspect in enumerate(plan.aspects):
                if aspect in missing_aspects:
                    # Relax search: use pure lexical query tokens
                    tokens = self._tokenize(aspect)
                    relaxed_query = " ".join(tokens) if tokens else aspect
                    refined_cands = self._retrieve_for_subquery(
                        subquery=relaxed_query,
                        k=plan.candidate_k * 2,
                        total_chunks=total_chunks,
                        context_query=plan.query,
                        where=plan.where_filter,
                    )
                    if refined_cands:
                        per_need_candidates[idx].extend(refined_cands)
                        aspects_covered.append(aspect)
                        missing_aspects.remove(aspect)

        # Fuse candidates with aspect guarantee
        fused = self._fuse_with_aspect_guarantee_and_locality(per_need_candidates, plan.final_top_k)

        # Build coverage telemetry
        docs_covered = list({str(c.metadata.get("filename", "")) for c in fused if c.metadata.get("filename")})
        pages_covered = sorted(list({int(c.metadata.get("page_number", 1)) for c in fused if c.metadata.get("page_number")}))
        ratio = len(aspects_covered) / max(len(plan.aspects), 1)

        coverage = CoverageInfo(
            aspects_requested=plan.aspects,
            aspects_covered=aspects_covered,
            missing_aspects=missing_aspects,
            documents_covered=docs_covered,
            pages_covered=pages_covered,
            coverage_ratio=ratio,
            secondary_pass_triggered=secondary_triggered,
        )

        return fused, coverage

    def _retrieve_ordinal(
        self,
        plan: RetrievalPlan,
    ) -> List[RetrievedChunk]:
        """
        Ordinal and Positional Retrieval Strategy:
        Retrieves sequential document chunks, extracts structured items generically,
        applies OrdinalSpec (exclusions, slices, indices), verifies coverage,
        and returns item-level evidence chunks without excluded items.
        """
        logger.info(f"Executing ORDINAL retrieval for spec: {plan.ordinal_spec}")
        sequential_chunks = self._retrieve_document_wide(plan)
        if not sequential_chunks:
            return []

        spec = plan.ordinal_spec
        if not spec:
            return sequential_chunks[:plan.final_top_k]

        # Extract sequence items generically across chunks
        items = OrderedSequenceResolver.extract_sequence(sequential_chunks)
        if items:
            selected = OrderedSequenceResolver.resolve_selection(items, spec)
            plan.resolved_items = selected
            target_fn = plan.target_documents[0].filename if plan.target_documents else ""
            evidence = OrderedSequenceResolver.build_item_evidence_chunks(selected, target_fn)
            is_complete, missing = OrderedSequenceResolver.verify_coverage(selected, evidence)
            logger.info(
                f"ORDINAL sequence resolved: {len(items)} total -> {len(selected)} selected items. "
                f"Coverage complete: {is_complete}, chunks: {len(evidence)}"
            )
            if evidence:
                return evidence[:plan.final_top_k]

        # Fallback if no multi-item list detected in document text
        total_items = len(sequential_chunks)
        if spec.exclude_indices:
            excl_set = set(spec.exclude_indices)
            selected_chunks = [c for idx, c in enumerate(sequential_chunks) if idx not in excl_set]
            return selected_chunks[:plan.final_top_k]

        if spec.slice_start is not None or spec.slice_end is not None:
            start_i = spec.slice_start if spec.slice_start is not None else 0
            end_i = spec.slice_end if spec.slice_end is not None else total_items - 1
            selected_chunks = [c for idx, c in enumerate(sequential_chunks) if start_i <= idx <= end_i]
            return selected_chunks[:plan.final_top_k]

        if spec.is_last and sequential_chunks:
            return [sequential_chunks[-1]]

        if spec.indices and sequential_chunks:
            target_idx = spec.indices[0]
            if 0 <= target_idx < len(sequential_chunks):
                return [sequential_chunks[target_idx]]

        return sequential_chunks[:plan.final_top_k]

    def _retrieve_page_targeted(
        self,
        plan: RetrievalPlan,
    ) -> List[RetrievedChunk]:
        """Scoped retrieval targeting a specific page number."""
        logger.info(f"Executing PAGE_TARGETED retrieval for page {plan.target_page}")
        where_cond = {"page_number": plan.target_page}
        if plan.where_filter:
            if "$and" in plan.where_filter and isinstance(plan.where_filter["$and"], list):
                where_cond = {"$and": [*plan.where_filter["$and"], where_cond]}
            else:
                where_cond = {"$and": [plan.where_filter, where_cond]}

        raw_results = self.vector_store.collection.get(
            where=where_cond,
            include=["metadatas", "documents"]
        )
        docs = raw_results.get("documents", [])
        metas = raw_results.get("metadatas", [])
        if not docs:
            # Fallback to query if page_number where metadata is not integer indexed
            q_emb = self.embedding_service.embed_text(plan.query)
            return self._retrieve_for_subquery(
                subquery=plan.query,
                k=plan.final_top_k,
                total_chunks=self.vector_store.count(),
                where=plan.where_filter,
            )

        chunks = []
        for doc_text, meta in zip(docs, metas):
            chunks.append(RetrievedChunk(text=doc_text, metadata=meta or {}, score=1.0))
        chunks.sort(key=lambda c: int(c.metadata.get("chunk_index", 0)))
        return chunks[:plan.final_top_k]

    def _retrieve_multi_document_summary(
        self,
        plan: RetrievalPlan,
        available_documents: Optional[List[Dict[str, Any]]] = None,
    ) -> List[RetrievedChunk]:
        """
        Multi-Document Summary Strategy:
        Retrieves balanced evidence chunks for EVERY distinct attached document in scope.
        Guarantees 100% document coverage so no attached document is starved or omitted.
        """
        logger.info(f"Executing MULTI_DOCUMENT_SUMMARY retrieval for query: '{plan.query}'")

        target_doc_names = []
        if available_documents:
            target_doc_names = [d.get("filename") for d in available_documents if d.get("filename")]
        elif plan.target_documents:
            target_doc_names = [d.filename for d in plan.target_documents]
        else:
            try:
                indexed = self.vector_store.get_indexed_files()
                target_doc_names = list(indexed.keys())
            except Exception:
                target_doc_names = []

        if not target_doc_names:
            return self._retrieve_for_subquery(
                subquery=plan.query,
                k=plan.final_top_k,
                total_chunks=self.vector_store.count(),
                where=plan.where_filter,
            )

        chunks_per_doc = max(3, plan.final_top_k // max(len(target_doc_names), 1))
        all_chunks: List[RetrievedChunk] = []

        for fn in target_doc_names:
            doc_where = {"filename": fn}
            if plan.where_filter:
                if "$and" in plan.where_filter and isinstance(plan.where_filter["$and"], list):
                    doc_where = {"$and": [*plan.where_filter["$and"], {"filename": fn}]}
                else:
                    doc_where = {"$and": [plan.where_filter, {"filename": fn}]}

            doc_chunks = self._retrieve_for_subquery(
                subquery=plan.query,
                k=chunks_per_doc,
                total_chunks=self.vector_store.count(),
                where=doc_where,
            )
            if not doc_chunks:
                try:
                    res = self.vector_store.collection.get(where=doc_where, limit=chunks_per_doc)
                    docs = res.get("documents", [])
                    metas = res.get("metadatas", [])
                    for d_text, m in zip(docs, metas):
                        doc_chunks.append(RetrievedChunk(text=d_text, metadata=m or {}, score=0.5))
                except Exception:
                    pass
            all_chunks.extend(doc_chunks[:chunks_per_doc])

        logger.info(f"MULTI_DOCUMENT_SUMMARY assembled {len(all_chunks)} chunks across {len(target_doc_names)} documents.")
        return all_chunks

    # --------------------------------------------------------------------------
    # MAIN ADAPTIVE ENTRYPOINTS
    # --------------------------------------------------------------------------

    def retrieve_adaptive(
        self,
        query: str,
        where: Optional[Dict[str, Any]] = None,
        recent_messages: Optional[List[Dict[str, Any]]] = None,
        available_documents: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[RetrievedChunk], RetrievalPlan, CoverageInfo]:
        """
        Execute full Adaptive RAG retrieval:
        1. Query Understanding (Intent, Aspects, Ordinals, Follow-ups)
        2. Document Resolution (Chat-scoped, Titular, Singular references)
        3. Retrieval Planning (Strategy, Candidate Depth, Generation Budget)
        4. Adaptive Retrieval Execution
        5. Coverage Verification & Telemetry
        """
        if not query or not query.strip():
            empty_plan = RetrievalPlanner.create_plan(
                QueryAnalyzer.analyze(""),
                DocumentResolutionResult(),
                default_top_k=self.top_k
            )
            return [], empty_plan, CoverageInfo()

        clean_query = query.strip()
        logger.info(f"Starting adaptive retrieval for query: '{clean_query}'")

        # Step 1: Query Understanding
        analysis = QueryAnalyzer.analyze(
            query=clean_query,
            recent_messages=recent_messages,
        )

        # Step 2: Document Resolution
        docs_pool = available_documents
        if not docs_pool:
            try:
                indexed = self.vector_store.get_indexed_files()
                docs_pool = [{"filename": fn, "file_hash": fh} for fn, fh in indexed.items()]
            except Exception as e:
                logger.warning(f"Failed to query indexed files for document resolution: {e}")
                docs_pool = []

        doc_resolution = DocumentResolver.resolve(
            query=analysis.cleaned_query,
            available_documents=docs_pool or [],
            base_where_filter=where,
            is_comparison=(analysis.intent == QueryIntent.COMPARISON),
        )

        # Step 3: Retrieval Planning
        plan = RetrievalPlanner.create_plan(
            analysis=analysis,
            doc_resolution=doc_resolution,
            default_top_k=self.top_k,
        )

        # Step 4: Execute Planned Strategy
        coverage = CoverageInfo()

        if plan.strategy == RetrievalStrategy.MULTI_DOCUMENT_SUMMARY:
            chunks = self._retrieve_multi_document_summary(plan, available_documents=docs_pool)
        elif plan.strategy == RetrievalStrategy.DOCUMENT_WIDE:
            chunks = self._retrieve_document_wide(plan)
        elif plan.strategy == RetrievalStrategy.COMPARISON:
            chunks = self._retrieve_comparison(plan)
        elif plan.strategy == RetrievalStrategy.MULTI_ASPECT:
            chunks, coverage = self._retrieve_multi_aspect(plan)
        elif plan.strategy == RetrievalStrategy.ORDINAL:
            chunks = self._retrieve_ordinal(plan)
        elif plan.strategy == RetrievalStrategy.PAGE_TARGETED:
            chunks = self._retrieve_page_targeted(plan)
        else:
            # NORMAL Strategy: Hybrid RRF
            total_chunks = self.vector_store.count()
            subqueries = QueryDecomposer.decompose(plan.query)
            if len(subqueries) <= 1:
                chunks = self._retrieve_for_subquery(
                    subquery=plan.query,
                    k=plan.final_top_k,
                    total_chunks=total_chunks,
                    where=plan.where_filter,
                )[:plan.final_top_k]
            else:
                per_need: List[List[RetrievedChunk]] = []
                for sq in subqueries:
                    sq_cands = self._retrieve_for_subquery(
                        subquery=sq,
                        k=plan.final_top_k,
                        total_chunks=total_chunks,
                        context_query=plan.query,
                        where=plan.where_filter,
                    )
                    per_need.append(sq_cands)
                chunks = self._fuse_with_aspect_guarantee_and_locality(per_need, plan.final_top_k)

        # Final coverage update
        docs_covered = list({str(c.metadata.get("filename", "")) for c in chunks if c.metadata.get("filename")})
        pages_covered = sorted(list({int(c.metadata.get("page_number", 1)) for c in chunks if c.metadata.get("page_number")}))
        coverage.documents_covered = docs_covered
        # Evidence evaluation
        assessment = EvidenceEvaluator.evaluate(query=clean_query, chunks=chunks, plan=plan)
        coverage.evidence_assessment = assessment

        logger.info(
            f"Adaptive retrieval complete. Strategy: {plan.strategy.value}, "
            f"Retrieved: {len(chunks)} chunks, Pages: {pages_covered}, Docs: {docs_covered}, "
            f"Evidence: {assessment.status.value}"
        )
        return chunks, plan, coverage

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """
        Backward-compatible retrieval entrypoint for existing tests and endpoints.
        Delegates to retrieve_adaptive() while preserving explicit caller parameters.
        """
        chunks, plan, _ = self.retrieve_adaptive(query=query, where=where)
        # If caller explicitly asked for a top_k, respect it unless it was a document-wide exhaustive scan
        if top_k is not None:
            if plan.strategy in (RetrievalStrategy.DOCUMENT_WIDE, RetrievalStrategy.ORDINAL):
                # Ensure exhaustive queries aren't arbitrarily clamped down to small top_k
                return chunks[:max(top_k, len(chunks))]
            return chunks[:top_k]
        return chunks
