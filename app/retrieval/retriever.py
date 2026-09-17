"""
Hybrid and Query-Aware Retrieval Module.
Combines dense vector similarity with lexical term matching, generic query decomposition,
document-locality context propagation, and aspect-guaranteed reciprocal rank fusion.
Reliably handles single-fact, multi-part, same-document, and cross-document questions.
"""

from collections import defaultdict
from dataclasses import dataclass, field
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from app.config import settings
from app.embeddings.service import EmbeddingService
from app.models import RetrievedChunk
from app.utils.logger import setup_logger
from app.vectorstore.store import VectorStore

logger = setup_logger("retrieval.retriever")


class QueryDecomposer:
    """
    Analyzes queries and decomposes complex, multi-part, or comparative questions
    into distinct information needs to maximize retrieval coverage across chunks and documents.
    Protects compound titles and entities from false splitting and preserves global scope.
    """

    # Clause indicators signaling a new inquiry or information requirement
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

        # Check 'with', 'versus', 'vs'
        vs_m = re.split(r"\s+\b(?:with|versus|vs\.?)\b\s+", s, flags=re.IGNORECASE)
        if len(vs_m) > 1:
            return [cls._clean_token(p) for p in vs_m if cls._clean_token(p)]

        # Check comma-separated lists, e.g. "A, B, and C" or "A, B and C"
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
            # If entire string is "The X and the Y" (single title), do not split
            if re.match(r"^The\s+[A-Z]", left) and re.match(r"^(?:the\s+)?[A-Z]", right):
                return [cls._clean_token(s)]
            return [left, right]

        # Multiple "and"s: evaluate each candidate split point to protect compound titles
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
        """
        Decompose comparative questions across multiple entities/documents.
        Generates equivalent information needs for EACH subject when comparing shared attributes.
        """
        # 1. Check "How does X differ from Y"
        differ_m = re.match(cls.DIFFER_PATTERN, core, re.IGNORECASE)
        if differ_m:
            side_a = differ_m.group(1).strip(" ?,.")
            side_b = differ_m.group(2).strip(" ?,.")
            if len(side_a) > 2 and len(side_b) > 2:
                # Check if side_a has an attribute + preposition + subject
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

        # 2. Check standard comparison prefixes
        pref_m = re.match(cls.COMPARISON_PREFIXES, core, re.IGNORECASE)
        if not pref_m:
            # Check "X vs Y" or "X versus Y"
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

        # 3. Check "Compare [SUBJECTS] based on / in terms of / regarding [ATTRIBUTES]"
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

        # 4. Check "Compare [ATTRIBUTE] in/of/for/between/across [SUBJECTS]"
        prep_regex = r"\b(" + "|".join(cls.COMMON_PREPOSITIONS) + r")\b"
        prep_matches = list(re.finditer(prep_regex, body, re.IGNORECASE))

        for pm in prep_matches:
            potential_attr = body[:pm.start()].strip()
            prep = pm.group(1).lower()
            potential_subj_str = body[pm.end():].strip()

            if not potential_attr or len(potential_attr.split()) < 1:
                continue

            # Don't split if potential_attr is just a single capitalized word like "Alice" in "Alice in Wonderland"
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

        # 5. Bare comparison: "Compare [SUBJECTS]" (e.g. "Compare Document A and Document B")
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
        """
        Decompose a compound or comparative query into distinct subqueries.
        Returns a list containing the focused subqueries and the full query.
        """
        clean_q = query.strip()
        if not clean_q:
            return []

        # 1. Extract introductory scope first if present ("In <Scope>, ...")
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

        # 2. Check comparative queries (extracting shared attributes across subjects)
        comp_subqueries = cls._decompose_comparisons(core, clean_q, scope)
        if comp_subqueries:
            return cls._deduplicate(comp_subqueries)

        # 3. Check multiple sentences or questions: "What is X? What is Y?", semicolons, or numbered lists
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

        # 4. Multi-clause conjunction splitting:
        # Splits on:
        # - ", and "
        # - " as well as "
        # - " and " ONLY when followed by a clause indicator (what, how, who, etc.)
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

        # 5. Coordinated noun phrases: "Tell me about X and Y", "What are X and Y?"
        coord_match = re.match(
            r"^(what\s+(?:is|are)\s+(?:all\s+)?(?:my\s+)?|tell\s+me\s+about\s+|describe\s+)(.+?)\s+and\s+(.+)$",
            working_text,
            re.IGNORECASE,
        )
        if coord_match:
            prefix = coord_match.group(1).strip()
            item_a = coord_match.group(2).strip(" ?,.")
            item_b = coord_match.group(3).strip(" ?,.")
            # Check if this is a capitalized title phrase like "The Old Man and the Sea"
            if re.match(r"^The\s+[A-Z]", item_a) and re.match(r"^(?:the\s+)?[A-Z]", item_b):
                return [clean_q]

            # Verify items are distinct inquiries and not a single compound domain phrase
            if 0 < len(item_a.split()) <= 5 and 0 < len(item_b.split()) <= 5:
                sq1 = f"{prefix} {item_a}"
                sq2 = f"{prefix} {item_b}"
                if scope:
                    sq1 = f"{scope}, {sq1}"
                    sq2 = f"{scope}, {sq2}"
                return cls._deduplicate([sq1, sq2, clean_q])

        # Single information need
        return [clean_q]

    @staticmethod
    def _extract_scope(text: str) -> str:
        """Extract introductory prepositional scope such as 'In Alice in Wonderland' or 'According to Chapter 1'."""
        match = re.match(r"^(in|for|according\s+to|regarding)\s+([^,]+),", text, re.IGNORECASE)
        if match:
            return match.group(0).rstrip(",")
        return ""

    @staticmethod
    def _has_scope(text: str, scope: str) -> bool:
        """Check if clause already contains key tokens from the extracted scope."""
        scope_tokens = set(re.findall(r"\w+", scope.lower())) - {"in", "for", "according", "to", "regarding"}
        text_tokens = set(re.findall(r"\w+", text.lower()))
        return bool(scope_tokens & text_tokens)

    @staticmethod
    def _extract_subject(clause: str) -> str:
        """Extract primary subject phrase from an introductory clause to enrich dependent subqueries."""
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
    Production retriever pairing dense embeddings with lexical term matching,
    document-locality propagation, and aspect-guaranteed candidate fusion.
    """

    # Common English stopwords to ignore during keyword scoring
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
        """Extract alphanumeric tokens excluding short tokens and stopwords."""
        words = re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower())
        return [w for w in words if w not in self.STOPWORDS]

    def _calculate_lexical_score(
        self, query: str, text: str, section: str, filename: str = ""
    ) -> float:
        """
        Compute lexical score using regex word boundaries, token coverage ratio,
        and exact phrase bonus across document text, section title, and filename.
        """
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

        # Token coverage ratio (fraction of distinct query tokens covered by chunk)
        coverage = matched_tokens / len(unique_q_tokens)

        # Exact multi-token phrase bonus
        clean_q = " ".join(query_tokens)
        phrase_bonus = 0.0
        if len(query_tokens) >= 2 and clean_q in doc_lower:
            phrase_bonus = 1.0

        raw_score = (freq_score / len(unique_q_tokens)) * (0.5 + 0.5 * coverage) + phrase_bonus
        return min(raw_score, 3.0) / 3.0

    def _match_target_document(self, query: str) -> Optional[str]:
        """
        Identify if a query or subquery explicitly refers to a specific indexed document.
        Uses generic constituent word and token sequence matching against available document filenames.
        Zero hardcoding of specific filenames or book titles.
        """
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
    ) -> List[RetrievedChunk]:
        """
        Retrieve and score candidates for an individual information need.
        Uses document-aware targeting and an expanded candidate pool.
        """
        query_embedding = self.embedding_service.embed_text(subquery)
        if not query_embedding:
            logger.error(f"Failed to generate query embedding for subquery: '{subquery}'")
            return []

        # Document/Title Awareness: preferentially target specific document if subquery references it
        target_doc = self._match_target_document(subquery)
        where_filter = {"filename": target_doc} if target_doc else None

        # Pull enough candidates for robust hybrid re-ranking across multi-document collections
        candidate_k = min(max(k * 8, 40), total_chunks)
        raw_results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=candidate_k,
            where=where_filter
        )

        doc_list = list(raw_results.get("documents", [[]])[0])
        meta_list = list(raw_results.get("metadatas", [[]])[0])
        dist_list = list(raw_results.get("distances", [[]])[0]) if "distances" in raw_results else [1.0] * len(doc_list)

        # Fallback to unfiltered query if filtered query yielded nothing
        if target_doc and not doc_list:
            raw_results = self.vector_store.query(
                query_embedding=query_embedding,
                top_k=candidate_k,
                where=None
            )
            doc_list = list(raw_results.get("documents", [[]])[0])
            meta_list = list(raw_results.get("metadatas", [[]])[0])
            dist_list = list(raw_results.get("distances", [[]])[0]) if "distances" in raw_results else [1.0] * len(doc_list)

        # If a target document was matched and subquery has an attribute prefix ("X in Document"),
        # also retrieve candidates for the attribute to capture internal document semantics
        if target_doc:
            attr_part = re.sub(r"\s+\b(in|of|for|between|across|from)\b\s+.*$", "", subquery, flags=re.IGNORECASE).strip()
            if attr_part and attr_part.lower() != subquery.lower() and len(attr_part.split()) >= 1:
                attr_emb = self.embedding_service.embed_text(attr_part)
                if attr_emb:
                    attr_res = self.vector_store.query(
                        query_embedding=attr_emb,
                        top_k=candidate_k,
                        where=where_filter
                    )
                    existing_cids = {m.get("chunk_id") for m in meta_list}
                    attr_docs = attr_res.get("documents", [[]])[0]
                    attr_metas = attr_res.get("metadatas", [[]])[0]
                    attr_dists = attr_res.get("distances", [[]])[0] if "distances" in attr_res else [1.0] * len(attr_docs)
                    for adoc, ameta, adist in zip(attr_docs, attr_metas, attr_dists):
                        cid = ameta.get("chunk_id")
                        if cid and cid not in existing_cids:
                            existing_cids.add(cid)
                            doc_list.append(adoc)
                            meta_list.append(ameta)
                            dist_list.append(adist)

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

            # Combined hybrid score (65% dense semantic similarity, 35% lexical match)
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
        """Check if two section titles represent continuous prose or related sub-sections."""
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
        """
        Fuse candidate lists using Reciprocal Rank Fusion (RRF) with:
        1. Aspect Guarantee: Guarantees top hit(s) from each information need in the context.
        2. Document-Locality Propagation: Anchors top hits to boost immediate neighbor chunks in the same document.
        3. Multi-Need Consensus: Chunks addressing multiple aspects receive cumulative score boosts.
        """
        if not per_need_candidates:
            return []

        num_needs = len(per_need_candidates)
        rrf_scores: Dict[str, float] = defaultdict(float)
        chunk_map: Dict[str, RetrievedChunk] = {}
        chunk_file: Dict[str, str] = {}
        chunk_idx_map: Dict[str, int] = {}

        # Collect high-confidence anchor chunks across all needs to propagate locality
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

                # Top hits serve as spatial anchors for adjacent co-located facts
                if rank < 2 and raw_cidx is not None:
                    try:
                        top_anchors.append((fn, int(raw_cidx), sec, pts))
                    except (ValueError, TypeError):
                        pass

        # Generic Document-Locality Propagation:
        # Neighboring chunks (distance 1-2) within contiguous text receive a locality bonus
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
        # Guarantees that every distinct information need has its top relevant chunk represented
        for need_idx in range(num_needs):
            if len(selected) >= k:
                break
            cands = per_need_candidates[need_idx]
            for cand in cands:
                if add_chunk(cand):
                    break

        # Step 2: Fill remaining slots by overall RRF score descending
        # Chunks with multi-need consensus and locality bonuses populate remaining capacity
        sorted_by_rrf = sorted(
            chunk_map.keys(),
            key=lambda cid: rrf_scores[cid],
            reverse=True
        )
        for cid in sorted_by_rrf:
            if len(selected) >= k:
                break
            add_chunk(chunk_map[cid])

        # Step 3: Normalize informative score for output formatting
        for c in selected:
            cid = c.chunk_id or f"{c.metadata.get('filename')}_{c.metadata.get('chunk_index')}"
            c.score = max(c.score, rrf_scores[cid] * 25.0)

        selected.sort(key=lambda c: c.score, reverse=True)
        return selected[:k]

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        """
        Retrieve the top_k most relevant chunks for the given query.
        Decomposes compound queries, retrieves candidates per need, and fuses via RRF + Locality.
        """
        if not query or not query.strip():
            logger.warning("Empty query submitted for retrieval.")
            return []

        k = top_k or self.top_k
        total_chunks = self.vector_store.count()

        if total_chunks == 0:
            logger.warning("Vector database has 0 chunks. Ingestion required before querying.")
            return []

        clean_query = query.strip()
        logger.info(f"Retrieving top {k} chunks for query: '{clean_query}'")

        # Decompose query into constituent information needs
        subqueries = QueryDecomposer.decompose(clean_query)

        # Single information need: direct hybrid retrieval
        if len(subqueries) <= 1:
            candidates = self._retrieve_for_subquery(clean_query, k, total_chunks)
            selected = candidates[:k]
            if selected:
                logger.info(f"Retrieved {len(selected)} chunks (top score: {selected[0].score:.3f}).")
            return selected

        logger.info(f"Query decomposed into {len(subqueries)} information needs: {subqueries}")

        # Multi-need retrieval: retrieve candidate lists for each subquery
        per_need_candidates: List[List[RetrievedChunk]] = []
        for sq in subqueries:
            sq_cands = self._retrieve_for_subquery(sq, k, total_chunks, context_query=clean_query)
            per_need_candidates.append(sq_cands)

        # Aspect-guaranteed reciprocal rank fusion with document locality
        selected = self._fuse_with_aspect_guarantee_and_locality(per_need_candidates, k)
        if selected:
            logger.info(
                f"Fused {len(selected)} chunks from {len(subqueries)} needs (top score: {selected[0].score:.3f})."
            )
        return selected
