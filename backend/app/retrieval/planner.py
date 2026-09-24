"""
Adaptive Retrieval Strategy Planner.
Translates QueryAnalysis and DocumentResolution into an actionable RetrievalPlan
with dynamic candidate depth, retrieval strategy, and adaptive generation budget.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from app.retrieval.query_understanding import QueryAnalysis, QueryIntent, OrdinalSpec
from app.retrieval.document_resolver import DocumentResolutionResult, ResolvedDocument
from app.utils.logger import setup_logger

logger = setup_logger("retrieval.planner")


class RetrievalStrategy(str, Enum):
    """Execution strategy for information retrieval."""
    NORMAL = "normal"                      # Standard hybrid dense + lexical RRF
    MULTI_ASPECT = "multi_aspect"          # Independent parallel aspect retrieval with aspect guarantee
    COMPARISON = "comparison"              # Independent per-subject/document retrieval with balanced quotas
    DOCUMENT_WIDE = "document_wide"        # Sequential document-order traversal across entire document
    ORDINAL = "ordinal"                    # Item-aware ordinal/slice selection from document evidence
    PAGE_TARGETED = "page_targeted"        # Scoped to specific page number(s)
    MULTI_DOCUMENT_SUMMARY = "multi_document_summary" # Balanced round-robin retrieval per attached document


@dataclass
class GenerationBudget:
    """Dynamic LLM parameters tailored to query complexity."""
    num_predict: int = 120
    num_ctx: int = 1536
    temperature: float = 0.0


@dataclass
class RetrievalPlan:
    """Actionable execution blueprint for the retrieval engine."""
    strategy: RetrievalStrategy
    query: str
    target_documents: List[ResolvedDocument] = field(default_factory=list)
    aspects: List[str] = field(default_factory=list)
    subjects: List[str] = field(default_factory=list)
    target_page: Optional[int] = None
    ordinal_spec: Optional[OrdinalSpec] = None
    candidate_k: int = 10
    final_top_k: int = 5
    where_filter: Optional[Dict[str, Any]] = None
    generation_budget: GenerationBudget = field(default_factory=GenerationBudget)
    requires_coverage_validation: bool = False
    resolved_items: List[Any] = field(default_factory=list)
    is_follow_up: bool = False
    is_code_request: bool = False
    is_table_request: bool = False
    intent: Optional[QueryIntent] = None
    lines_per_doc: Optional[int] = None


class RetrievalPlanner:
    """
    Plans retrieval depth, execution strategy, and generation budget
    dynamically based on the query understanding and document resolution.
    """

    @classmethod
    def create_plan(
        cls,
        analysis: QueryAnalysis,
        doc_resolution: DocumentResolutionResult,
        default_top_k: int = 5,
    ) -> RetrievalPlan:
        plan = cls._create_plan_internal(analysis, doc_resolution, default_top_k)
        plan.intent = analysis.intent
        plan.lines_per_doc = analysis.lines_per_doc
        plan.is_follow_up = analysis.is_follow_up
        plan.is_code_request = analysis.is_code_request
        plan.is_table_request = analysis.is_table_request
        if analysis.is_code_request:
            plan.generation_budget.num_predict = max(plan.generation_budget.num_predict, 512)
        return plan

    @classmethod
    def _create_plan_internal(
        cls,
        analysis: QueryAnalysis,
        doc_resolution: DocumentResolutionResult,
        default_top_k: int = 5,
    ) -> RetrievalPlan:
        """
        Synthesize query analysis and document resolution into a concrete RetrievalPlan.
        """
        query_text = analysis.resolved_query or analysis.cleaned_query
        target_docs = doc_resolution.resolved_documents
        where = doc_resolution.chroma_where_filter

        # 0. MULTI-DOCUMENT / EACH-FILE STRATEGY
        if analysis.intent in (QueryIntent.MULTI_DOCUMENT_QUERY, QueryIntent.DOCUMENT_SUMMARY, QueryIntent.DOCUMENT_LIST_QUERY):
            return RetrievalPlan(
                strategy=RetrievalStrategy.MULTI_DOCUMENT_SUMMARY,
                query=query_text,
                target_documents=target_docs,
                aspects=analysis.aspects,
                candidate_k=40,
                final_top_k=8,
                where_filter=where,
                generation_budget=GenerationBudget(num_predict=384, num_ctx=2048),
                requires_coverage_validation=True,
            )

        # 1. PAGE TARGETED STRATEGY
        if analysis.intent == QueryIntent.PAGE_TARGETED and analysis.target_page is not None:
            return RetrievalPlan(
                strategy=RetrievalStrategy.PAGE_TARGETED,
                query=query_text,
                target_documents=target_docs,
                aspects=analysis.aspects,
                target_page=analysis.target_page,
                candidate_k=20,
                final_top_k=8,
                where_filter=where,
                generation_budget=GenerationBudget(num_predict=256, num_ctx=2048),
            )

        # 2. COMPARISON STRATEGY
        if analysis.intent == QueryIntent.COMPARISON:
            return RetrievalPlan(
                strategy=RetrievalStrategy.COMPARISON,
                query=query_text,
                target_documents=target_docs,
                aspects=analysis.aspects,
                subjects=analysis.subjects,
                candidate_k=30,
                final_top_k=12,
                where_filter=where,
                generation_budget=GenerationBudget(num_predict=512, num_ctx=3072),
                requires_coverage_validation=True,
            )

        # 3. ORDINAL / LIST SELECTION STRATEGY
        if analysis.intent == QueryIntent.ORDINAL and analysis.ordinal_spec:
            return RetrievalPlan(
                strategy=RetrievalStrategy.ORDINAL,
                query=query_text,
                target_documents=target_docs,
                aspects=analysis.aspects,
                ordinal_spec=analysis.ordinal_spec,
                candidate_k=50,
                final_top_k=25,
                where_filter=where,
                generation_budget=GenerationBudget(num_predict=512, num_ctx=3072),
                requires_coverage_validation=True,
            )

        # 4. EXHAUSTIVE / DOCUMENT-WIDE STRATEGY
        # Only select DOCUMENT_WIDE if target document is specifically resolved or singular in scope
        if analysis.intent in (QueryIntent.EXHAUSTIVE, QueryIntent.SUMMARIZATION):
            def _has_doc_constraint(filt: Optional[Dict[str, Any]]) -> bool:
                if not filt:
                    return False
                if "filename" in filt or "document_id" in filt or "file_hash" in filt:
                    return True
                if "$and" in filt and isinstance(filt["$and"], list):
                    return any(_has_doc_constraint(sub) for sub in filt["$and"])
                if "$or" in filt and isinstance(filt["$or"], list):
                    return any(_has_doc_constraint(sub) for sub in filt["$or"])
                return False

            is_single_target = (
                doc_resolution.is_strictly_targeted
                or (target_docs and len(target_docs) == 1)
                or _has_doc_constraint(where)
            )
            if is_single_target:
                return RetrievalPlan(
                    strategy=RetrievalStrategy.DOCUMENT_WIDE,
                    query=query_text,
                    target_documents=target_docs,
                    aspects=analysis.aspects,
                    candidate_k=100,
                    final_top_k=50,
                    where_filter=where,
                    generation_budget=GenerationBudget(num_predict=768, num_ctx=3584),
                    requires_coverage_validation=True,
                )
            # If no target document is known, check if it has multiple aspects
            if len(analysis.aspects) > 1:
                return RetrievalPlan(
                    strategy=RetrievalStrategy.MULTI_ASPECT,
                    query=query_text,
                    target_documents=target_docs,
                    aspects=analysis.aspects,
                    candidate_k=max(len(analysis.aspects) * 8, 24),
                    final_top_k=max(len(analysis.aspects) * 3, 10),
                    where_filter=where,
                    generation_budget=GenerationBudget(num_predict=512, num_ctx=3072),
                    requires_coverage_validation=True,
                )
            # Otherwise, use NORMAL hybrid search with generous candidate depth
            return RetrievalPlan(
                strategy=RetrievalStrategy.NORMAL,
                query=query_text,
                target_documents=target_docs,
                aspects=analysis.aspects,
                candidate_k=max(default_top_k * 6, 30),
                final_top_k=default_top_k * 2,
                where_filter=where,
                generation_budget=GenerationBudget(num_predict=256, num_ctx=2048),
                requires_coverage_validation=False,
            )

        # 5. MULTI-ASPECT STRATEGY
        if analysis.intent == QueryIntent.MULTI_ASPECT and len(analysis.aspects) > 1:
            return RetrievalPlan(
                strategy=RetrievalStrategy.MULTI_ASPECT,
                query=query_text,
                target_documents=target_docs,
                aspects=analysis.aspects,
                candidate_k=max(len(analysis.aspects) * 8, 24),
                final_top_k=max(len(analysis.aspects) * 3, 10),
                where_filter=where,
                generation_budget=GenerationBudget(num_predict=512, num_ctx=3072),
                requires_coverage_validation=True,
            )

        # 6. CODE EXTRACTION STRATEGY
        if analysis.intent == QueryIntent.CODE_EXTRACTION:
            return RetrievalPlan(
                strategy=RetrievalStrategy.NORMAL,
                query=query_text,
                target_documents=target_docs,
                aspects=analysis.aspects,
                candidate_k=max(default_top_k * 4, 20),
                final_top_k=min(default_top_k, 5),
                where_filter=where,
                generation_budget=GenerationBudget(num_predict=512, num_ctx=3072),
                requires_coverage_validation=True,
            )

        # 7. TABLE LOOKUP STRATEGY
        if analysis.intent == QueryIntent.TABLE_LOOKUP:
            return RetrievalPlan(
                strategy=RetrievalStrategy.NORMAL,
                query=query_text,
                target_documents=target_docs,
                aspects=analysis.aspects,
                candidate_k=max(default_top_k * 5, 25),
                final_top_k=max(default_top_k, 6),
                where_filter=where,
                generation_budget=GenerationBudget(num_predict=512, num_ctx=3072),
                requires_coverage_validation=True,
            )

        # 8. PROCEDURAL STRATEGY
        if analysis.intent == QueryIntent.PROCEDURAL:
            return RetrievalPlan(
                strategy=RetrievalStrategy.NORMAL,
                query=query_text,
                target_documents=target_docs,
                aspects=analysis.aspects,
                candidate_k=max(default_top_k * 5, 25),
                final_top_k=max(default_top_k, 6),
                where_filter=where,
                generation_budget=GenerationBudget(num_predict=512, num_ctx=3072),
                requires_coverage_validation=True,
            )

        # 9. NORMAL STRATEGY (FAST FACTUAL / SECTION LOOKUP)
        return RetrievalPlan(
            strategy=RetrievalStrategy.NORMAL,
            query=query_text,
            target_documents=target_docs,
            aspects=analysis.aspects,
            candidate_k=max(default_top_k * 4, 15),
            final_top_k=default_top_k,
            where_filter=where,
            generation_budget=GenerationBudget(num_predict=120, num_ctx=1536),
            requires_coverage_validation=False,
        )
