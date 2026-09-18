"""
Ordered Sequence Resolver Module.
Extracts complete ordered item sequences (numbered lists, steps, programs, rules, etc.)
from document evidence, resolves ordinal exclusions and slices, assembles item-complete
evidence chunks, and verifies 100% item coverage before generation.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from app.models import RetrievedChunk
from app.retrieval.query_understanding import OrdinalSpec
from app.utils.logger import setup_logger

logger = setup_logger("retrieval.ordered_sequence")


@dataclass
class DocumentItem:
    """Represents an atomic ordered item extracted from a document."""
    sequence_index: int               # 0-indexed position in sequence
    identifier: str                   # e.g. "1", "2", "A", "Step 1"
    numeric_value: Optional[int]      # Integer value if numeric (e.g. 1, 2)
    title: str                        # Headline / item title
    body: str                         # Content text of the item (clean body)
    chunk_index: int = 0              # Source chunk index
    page_number: int = 1              # Source page number
    filename: str = ""                # Source document filename
    sequence_header: str = ""         # Preamble / document context before first item
    raw_text: str = ""                # Complete text including header and body


class OrderedSequenceResolver:
    """
    Production-grade sequence resolver for ordered / list documents.
    Operates generically across any arbitrary PDF or text document.
    """

    # Generic patterns matching list item headers:
    # 1. Numbered with optional prefix: "1. Title", "1) Title", "Program 1: Title", "Step 1 - Title"
    NUMERIC_PATTERN = re.compile(
        r"(?:^|\n)[ \t]*(?:(?:item|step|problem|program|question|rule|section|case|part|point|algorithm|example)\s+)?(\d+)[ \t]*[\.\)\:\-][ \t]+([^\n\r]+)",
        re.IGNORECASE
    )

    # 2. Lettered or roman numeral items: "a. Title", "A) Title", "(i) Title"
    ALPHA_ROMAN_PATTERN = re.compile(
        r"(?:^|\n)[ \t]*(?:\(?([a-zA-Z]|\b(?:[ivxldcmIVXLDCM]+)\b)[\.\)])[ \t]+([^\n\r]+)"
    )

    @classmethod
    def extract_ordered_sequence(
        cls,
        chunks: List[RetrievedChunk],
    ) -> List[DocumentItem]:
        """
        Scan full document chunks in reading order to extract complete, ordered list items.
        Returns empty list if no clear structured sequence is found.
        """
        return cls._extract_ordered_sequence_impl(chunks)

    # Backward compatibility alias
    extract_sequence = extract_ordered_sequence

    @classmethod
    def _extract_ordered_sequence_impl(
        cls,
        chunks: List[RetrievedChunk],
    ) -> List[DocumentItem]:
        if not chunks:
            return []

        # Sort chunks strictly by reading order: filename -> page_number -> chunk_index
        sorted_chunks = sorted(
            chunks,
            key=lambda c: (
                str(c.metadata.get("filename", "")),
                int(c.metadata.get("page_number", 1)),
                int(c.metadata.get("chunk_index", 0))
            )
        )

        # Build unified text with chunk boundary markers to accurately track provenance
        chunk_offsets: List[Tuple[int, int, RetrievedChunk]] = []
        full_text_parts = []
        current_offset = 0

        for c in sorted_chunks:
            c_text = c.text.strip()
            if not c_text:
                continue
            start_off = current_offset
            full_text_parts.append(c_text)
            current_offset += len(c_text) + 1  # newline separator
            end_off = current_offset
            chunk_offsets.append((start_off, end_off, c))

        if not full_text_parts:
            return []

        full_text = "\n".join(full_text_parts)

        # Attempt numeric pattern first
        matches = list(cls.NUMERIC_PATTERN.finditer(full_text))
        is_numeric = True

        if len(matches) < 2:
            # Fall back to alpha/roman pattern
            matches = list(cls.ALPHA_ROMAN_PATTERN.finditer(full_text))
            is_numeric = False

        if len(matches) < 2:
            logger.debug("No multi-item ordered sequence detected in chunks.")
            return []

        # Extract sequence preamble / document context before the first item
        preamble = full_text[:matches[0].start()].strip() if matches else ""
        header_text = ""
        if preamble:
            p_lines = [l.strip() for l in preamble.splitlines() if l.strip()]
            header_text = " - ".join(p_lines[:2]) if p_lines else ""
            if header_text:
                header_text = re.sub(r"^(?:(?:Document\s+Context|Document\s+Title|Document)\s*:\s*)+", "", header_text, flags=re.IGNORECASE).strip()

        # Helper to find source chunk for a character offset in full_text
        def get_source_chunk(offset: int) -> RetrievedChunk:
            for s_off, e_off, ch in chunk_offsets:
                if s_off <= offset < e_off:
                    return ch
            return chunk_offsets[-1][2]

        raw_items: List[DocumentItem] = []
        seen_identifiers: Set[str] = set()

        for i, m in enumerate(matches):
            ident = m.group(1).strip()
            title = m.group(2).strip()
            start_pos = m.start()

            # End position is the start of the next different item, or end of text
            end_pos = len(full_text)
            for next_m in matches[i + 1:]:
                next_ident = next_m.group(1).strip()
                if next_ident != ident:
                    end_pos = next_m.start()
                    break

            raw_item_text = full_text[start_pos:end_pos].strip()
            content_after_header = full_text[m.end():end_pos].strip()
            item_body = content_after_header if content_after_header else title

            src_chunk = get_source_chunk(start_pos)
            meta = src_chunk.metadata or {}

            num_val = int(ident) if (is_numeric and ident.isdigit()) else None

            # Deduplicate consecutive identical items (caused by chunk overlap)
            # If same identifier was seen recently, merge body if longer
            if raw_items and raw_items[-1].identifier == ident:
                if len(item_body) > len(raw_items[-1].body):
                    raw_items[-1].body = item_body
                    raw_items[-1].raw_text = raw_item_text
                continue

            # If this identifier was seen earlier in non-consecutive position, check if sequence reset
            item_obj = DocumentItem(
                sequence_index=len(raw_items),
                identifier=ident,
                numeric_value=num_val,
                title=title,
                body=item_body,
                chunk_index=int(meta.get("chunk_index", 0)),
                page_number=int(meta.get("page_number", 1)),
                filename=str(meta.get("filename", meta.get("source_file", ""))),
                sequence_header=header_text,
                raw_text=raw_item_text,
            )
            raw_items.append(item_obj)

        logger.info(f"Extracted {len(raw_items)} ordered items from document sequence (identifiers: {[it.identifier for it in raw_items[:5]]}...)")
        return raw_items

    @classmethod
    def resolve_selection(
        cls,
        items: List[DocumentItem],
        spec: Optional[OrdinalSpec]
    ) -> List[DocumentItem]:
        """
        Resolve ordinal exclusions, slices, or specific positions against the complete sequence.
        """
        if not items:
            return []

        if not spec:
            return items

        total = len(items)

        # 1. Exclusions (e.g. "all except the first and second" -> exclude_indices=[0, 1])
        if spec.exclude_indices:
            excl_set = set(spec.exclude_indices)
            selected = [it for idx, it in enumerate(items) if idx not in excl_set]
            numeric_excl = getattr(spec, "numeric_exclusions", None)
            if numeric_excl:
                num_excl_set = set(numeric_excl)
                selected = [it for it in selected if it.numeric_value not in num_excl_set]
            logger.info(f"Ordinal exclusion resolved: {len(items)} items -> {len(selected)} requested items.")
            return selected

        # 2. Slices (e.g. "items 3 to 10", "first 5")
        if spec.slice_start is not None or spec.slice_end is not None:
            start_i = spec.slice_start if spec.slice_start is not None else 0
            end_i = spec.slice_end if spec.slice_end is not None else total - 1
            start_i = max(0, min(start_i, total - 1))
            end_i = max(0, min(end_i, total - 1))
            selected = [it for idx, it in enumerate(items) if start_i <= idx <= end_i]
            logger.info(f"Ordinal slice [{start_i}..{end_i}] resolved: {len(selected)} items.")
            return selected

        # 3. Last item
        if spec.is_last and items:
            return [items[-1]]

        # 4. Specific index ("second one" -> idx 1)
        if spec.indices and items:
            target_idx = spec.indices[0]
            if 0 <= target_idx < total:
                return [items[target_idx]]

        return items

    @classmethod
    def build_item_evidence_chunks(
        cls,
        requested_items: List[DocumentItem],
        target_filename: str = "",
    ) -> List[RetrievedChunk]:
        """
        Assemble retrieved chunks containing ONLY the requested items.
        Groups items logically by page/document to preserve provenance while guaranteeing
        that excluded items are omitted from the LLM context.
        """
        if not requested_items:
            return []

        chunks: List[RetrievedChunk] = []

        def _format_item_body(body_text: str, doc_hdr: str) -> Tuple[str, str, bool]:
            """Returns (formatted_body, detected_lang, is_code)."""
            trimmed = body_text.strip()
            if not trimmed:
                return "", "", False

            is_code = (
                "```" in trimmed or
                trimmed.startswith("def ") or
                trimmed.startswith("class ") or
                bool(re.search(r"\b(?:def|class|import|from|input|print|return|while|for|if|else|elif)\b", trimmed)) or
                bool(re.search(r"^\s*[a-zA-Z_]\w*\s*\(.*\)\s*$", trimmed, re.MULTILINE))
            )
            detected_lang = ""
            hdr_lower = (doc_hdr or "").lower()
            if "python" in hdr_lower or re.search(r"\b(?:input\(|print\(|elif\b|def\s+\w+)\b", trimmed):
                detected_lang = "python"
            elif "javascript" in hdr_lower or "node" in hdr_lower or "console.log" in trimmed:
                detected_lang = "javascript"
            elif "java" in hdr_lower or "system.out" in trimmed.lower():
                detected_lang = "java"
            elif "c++" in hdr_lower or "cpp" in hdr_lower or "#include" in trimmed:
                detected_lang = "cpp"
            elif "bash" in hdr_lower or "shell" in hdr_lower:
                detected_lang = "bash"

            if is_code and "```" not in trimmed:
                fenced = f"```{detected_lang}\n{trimmed}\n```" if detected_lang else f"```\n{trimmed}\n```"
                return fenced, detected_lang, True
            return trimmed, detected_lang, is_code

        # Group items by page_number
        items_by_page: Dict[int, List[DocumentItem]] = {}
        for it in requested_items:
            p = it.page_number
            if p not in items_by_page:
                items_by_page[p] = []
            items_by_page[p].append(it)

        for page_num in sorted(items_by_page.keys()):
            page_items = items_by_page[page_num]
            raw_hdr = page_items[0].sequence_header or ""
            clean_hdr = re.sub(r"^(?:(?:Document\s+Context|Document\s+Title|Document)\s*:\s*)+", "", raw_hdr, flags=re.IGNORECASE).strip()
            header_prefix = f"Document: {clean_hdr}\n\n" if clean_hdr else ""

            chunk_is_code = False
            if len(requested_items) == 1:
                it = page_items[0]
                body_fmt, lang, is_code = _format_item_body(it.body, clean_hdr)
                chunk_is_code = is_code
                lang_line = f"Language: {lang.capitalize()}\n" if lang else ""
                combined_text = f"{header_prefix}Item {it.identifier}: {it.title}\n{lang_line}{body_fmt}"
                sec_name = f"Item {it.identifier} - {it.title}"
            else:
                formatted_item_texts = []
                for it in page_items:
                    body_fmt, lang, is_code = _format_item_body(it.body, clean_hdr)
                    if is_code:
                        chunk_is_code = True
                        formatted_item_texts.append(f"{it.identifier}. {it.title}\n{body_fmt}")
                    else:
                        formatted_item_texts.append(it.raw_text)
                combined_text = header_prefix + "\n\n".join(formatted_item_texts)
                start_id = page_items[0].identifier
                end_id = page_items[-1].identifier
                sec_name = f"Items {start_id} to {end_id}" if start_id != end_id else f"Item {start_id}"
            fn = page_items[0].filename or target_filename

            chunk = RetrievedChunk(
                text=combined_text,
                metadata={
                    "filename": fn,
                    "page_number": page_num,
                    "section": sec_name,
                    "chunk_index": page_items[0].chunk_index,
                    "content_type": "code" if chunk_is_code else "text",
                },
                score=1.0,
            )
            chunks.append(chunk)

        logger.info(f"Assembled {len(chunks)} evidence chunks covering {len(requested_items)} requested items.")
        return chunks

    @classmethod
    def verify_coverage(
        cls,
        requested_items: List[DocumentItem],
        chunks: List[RetrievedChunk]
    ) -> Tuple[bool, List[DocumentItem]]:
        """
        Pre-generation validation:
        Verifies that every requested item is represented in the assembled evidence chunks.
        Returns (is_complete, missing_items).
        """
        if not requested_items:
            return True, []

        combined_text = "\n".join(c.text.lower() for c in chunks)
        missing: List[DocumentItem] = []

        for it in requested_items:
            # Check title (or clean words from title) or exact identifier
            title_norm = it.title.lower().strip()
            # Title words check (at least 2 words or whole title)
            words = [w for w in re.split(r"\W+", title_norm) if len(w) > 2]
            title_found = False
            if title_norm in combined_text:
                title_found = True
            elif words and sum(1 for w in words if w in combined_text) >= min(len(words), 2):
                title_found = True

            # Also check identifier format like "3." or "3)"
            ident_marker = f"{it.identifier}." in combined_text or f"{it.identifier})" in combined_text

            if not (title_found or ident_marker):
                missing.append(it)

        is_complete = (len(missing) == 0)
        if not is_complete:
            logger.warning(f"Coverage check identified {len(missing)} missing items: {[m.title for m in missing]}")
        else:
            logger.info(f"Coverage check PASSED: all {len(requested_items)} requested items verified in evidence.")

        return is_complete, missing
