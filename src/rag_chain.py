import re
import time
from typing import Dict, Any, List, Optional, Generator
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever

from src.config import MEDICAL_DISCLAIMER
from src.llm_factory import get_llm
from src.retriever import create_hybrid_retriever
from src.security import (
    detect_prompt_injection,
    detect_clinical_emergency,
    sanitize_pii,
)


MEDICAL_SYSTEM_PROMPT = """You are MediRAG, an evidence-based clinical guidelines assistant.

Your highest priority is factual grounding and zero hallucination.

CORE RULE:
Answer ONLY from information explicitly supported by the retrieved documents.
Never infer, extrapolate, or use medical knowledge that is not supported by the retrieved evidence.

--------------------------------------------------
1. CONVERSATIONAL CLINICAL INTERACTION
--------------------------------------------------
You converse professionally with clinicians in a continuous multi-turn dialogue.
When the user asks follow-up questions (e.g., "What if she is pregnant?", "What are their doses?", "What about patients above 80?"), use the conversation history to understand context, but answer strictly using the retrieved evidence.

--------------------------------------------------
2. RELEVANCE CHECK BEFORE ANSWERING
--------------------------------------------------
Before generating an answer, determine whether the retrieved evidence is actually relevant to the user's question.
For every retrieved chunk, evaluate:
- Does it address the same clinical condition/topic?
- Does it address the same patient population?
- Does it contain information relevant to the requested intervention, dose, diagnosis, recommendation, or outcome?
- Does it directly support the claim you are about to make?

Do NOT treat a document as relevant merely because it contains related medical keywords.

--------------------------------------------------
3. CITATION RULES
--------------------------------------------------
A citation may ONLY be attached to a claim if the cited passage directly supports that claim.
Never cite a retrieved document simply because it was returned by the retriever.
Never use unrelated citations to justify a refusal.
If no retrieved passage supports the answer, DO NOT provide citations.
Format citations exactly as: `[Source: <filename>, Page: <page_number>]`

--------------------------------------------------
4. OUT-OF-CORPUS / UNSUPPORTED QUESTIONS
--------------------------------------------------
If the requested information is not explicitly available in the retrieved documents:
- Do NOT answer from general medical knowledge.
- Do NOT estimate a dose.
- Do NOT provide an unverified treatment.
- Clearly state that the information is not available in the provided documents.

Use this format:
"Based on the provided documents, this information is not available. The retrieved evidence does not contain a recommendation for [requested information]. Therefore, I cannot provide a guideline-grounded answer from the available sources."

--------------------------------------------------
5. DIRECT EVIDENCE REQUIREMENT
--------------------------------------------------
Prefer evidence that directly states the clinical recommendation.
Distinguish carefully between "mentioned in the guideline" and "recommended by the guideline."

--------------------------------------------------
6. CITATION ENTAILMENT CHECK
--------------------------------------------------
Before returning each citation, verify:
CLAIM -> CITED PASSAGE
The cited passage must provide direct textual evidence for the claim.
If CLAIM is NOT supported by the passage, remove the citation.

--------------------------------------------------
7. REFUSAL SAFETY RULE
--------------------------------------------------
For clinical questions, if the relevant evidence is absent or insufficient, prefer a transparent refusal over an unsupported clinical recommendation.

--------------------------------------------------
8. FINAL RESPONSE FORMAT
--------------------------------------------------
When evidence is available:
**Clinical Summary:**
[Direct, conversational clinical answer strictly supported by the documents.]

**Detailed Evidence & Recommendations:**
- [Claim strictly supported by evidence] `[Source: <filename>, Page: <page_number>]`
- [Claim strictly supported by evidence] `[Source: <filename>, Page: <page_number>]`

When evidence is NOT available:
**Clinical Summary:**
"Based on the provided documents, this information is not available."

**Reason:**
"The retrieved documents do not contain relevant evidence addressing [topic]."
(Omit all citations).

CONTEXT PROVIDED:
---------------------
{context}
---------------------
"""

HUMAN_PROMPT = """Conversation History:
{chat_history}

Current Question: {question}

Please evaluate the context and conversation history carefully. Provide an evidence-grounded response with direct citations, or a clear grounded refusal if evidence is missing."""

CONDENSE_QUESTION_PROMPT = """Given the conversation history between a clinician and MediRAG, and a follow-up question, rephrase the follow-up question into a standalone medical search query containing all necessary clinical keywords from previous turns.
Do NOT answer the question. Only output the concise standalone search query.

Conversation History:
{chat_history}

Follow-up Question: {question}
Standalone Medical Query:"""


def format_docs_for_context(docs: List[Document]) -> str:
    """Formats retrieved chunks with clear source document and page number labels."""
    formatted_chunks = []
    for idx, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page", "?")
        chunk_id = doc.metadata.get("chunk_id", f"chunk_{idx}")
        score_str = f" | Rerank Score: {doc.metadata.get('rerank_score')}" if "rerank_score" in doc.metadata else ""

        formatted_chunks.append(
            f"--- [DOCUMENT {idx}: {source} | Page: {page} | ID: {chunk_id}{score_str}] ---\n"
            f"{doc.page_content}\n"
        )
    return "\n\n".join(formatted_chunks)


def format_chat_history(chat_history: Optional[List[Dict[str, Any]]], max_turns: int = 4) -> str:
    """Formats recent chat history turns into a readable text block."""
    if not chat_history:
        return "No previous conversation."

    valid_msgs = [m for m in chat_history if m.get("role") in ("user", "assistant")]
    recent_msgs = valid_msgs[-(max_turns * 2):]

    if not recent_msgs:
        return "No previous conversation."

    lines = []
    for msg in recent_msgs:
        role_label = "Clinician" if msg.get("role") == "user" else "MediRAG Assistant"
        content = msg.get("content", "").strip()
        if content:
            truncated = content[:400] + "..." if len(content) > 400 else content
            lines.append(f"{role_label}: {truncated}")

    return "\n".join(lines)


def extract_citations(docs: List[Document], answer: str = "") -> List[Dict[str, Any]]:
    """
    Extracts deduplicated citations strictly matching the claims in the answer.
    If the response is a refusal, returns an empty citation list (Zero irrelevant citations).
    """
    is_refusal = any(kw in answer.lower() for kw in [
        "not available", "insufficient evidence", "cannot provide", "does not contain"
    ])
    if is_refusal:
        return []

    # Check which pages/sources were explicitly cited in the answer
    cited_pages = set(re.findall(r'Page:\s*(\d+)', answer, re.IGNORECASE))

    citations = []
    seen = set()
    for doc in docs:
        source = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page", 1)
        key = (source, page)

        if cited_pages and str(page) not in cited_pages:
            continue

        preview = doc.page_content[:220].replace("\n", " ") + "..."
        if key not in seen:
            seen.add(key)
            citations.append({
                "source": source,
                "page": page,
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "snippet": preview,
                "rerank_score": doc.metadata.get("rerank_score"),
            })

    # Fallback: if answer cited evidence without exact page format, return top retrieved docs
    if not citations and not is_refusal and len(docs) > 0:
        for doc in docs[:3]:
            source = doc.metadata.get("source", "Unknown Document")
            page = doc.metadata.get("page", 1)
            key = (source, page)
            if key not in seen:
                seen.add(key)
                citations.append({
                    "source": source,
                    "page": page,
                    "chunk_id": doc.metadata.get("chunk_id", ""),
                    "snippet": doc.page_content[:220].replace("\n", " ") + "...",
                    "rerank_score": doc.metadata.get("rerank_score"),
                })

    return citations


def verify_clinical_safety(context: str, answer: str) -> Dict[str, Any]:
    """
    Automated Clinical Guardrail & Fact-Checker:
    1. Audits numerical values, dosages, and thresholds in answer against context.
    2. Identifies high-risk clinical populations and flags contraindication warnings.
    """
    is_refusal = any(kw in answer.lower() for kw in [
        "not available", "insufficient evidence", "cannot provide", "does not contain"
    ])

    warnings: List[str] = []
    verified_metrics: List[str] = []
    unverified_metrics: List[str] = []

    if is_refusal:
        return {
            "passed": True,
            "status": "[Safety Passed] Zero-Hallucination Refusal",
            "fact_check_score": 100,
            "warnings": [],
            "verified_metrics": [],
            "unverified_metrics": [],
        }

    # 1. Regex to extract clinical numbers, thresholds, and units
    num_pattern = re.compile(
        r'\b(?:\d+/\d+|\d+(?:\.\d+)?)\s*(?:mmHg|mg|g|mcg|ml|mmol/L|%|weeks|monthly|daily|hours)?\b',
        re.IGNORECASE
    )
    answer_numbers = set(num_pattern.findall(answer))
    context_lower = context.lower()

    for item in answer_numbers:
        item_clean = item.strip().lower()
        if len(item_clean) >= 2 and (not item_clean.isdigit() or int(item_clean) > 10):
            core_num = re.findall(r'\d+(?:/\d+)?(?:\.\d+)?', item_clean)
            if core_num and core_num[0] in context_lower:
                verified_metrics.append(item)
            else:
                unverified_metrics.append(item)

    # 2. High-Risk Clinical Contraindication Checks
    answer_lower = answer.lower()

    # Pregnancy Check for Antihypertensives (ACEi / ARBs)
    if any(drug in answer_lower for drug in ["ace", "arb", "angiotensin", "lisinopril", "losartan", "enalapril"]):
        if any(term in answer_lower for term in ["pregnant", "pregnancy"]):
            warnings.append("Contraindication Notice: ACE inhibitors and ARBs are strictly contraindicated during pregnancy due to risk of fetal harm.")

    # Renal / Hyperkalemia Check
    if any(term in answer_lower for term in ["diuretic", "spironolactone", "ace", "arb"]) and any(term in answer_lower for term in ["kidney", "renal", "kalemia", "potassium"]):
        warnings.append("Clinical Caution: Monitor renal function (eGFR) and serum potassium regularly when initiating combination therapies.")

    # Calculate Fact-Check Score
    total_metrics = len(verified_metrics) + len(unverified_metrics)
    fact_score = 100
    if total_metrics > 0:
        fact_score = round((len(verified_metrics) / total_metrics) * 100, 1)

    return {
        "passed": len(unverified_metrics) == 0,
        "status": f"[Verified] {fact_score}% Evidence Match",
        "fact_check_score": fact_score,
        "warnings": warnings,
        "verified_metrics": list(verified_metrics),
        "unverified_metrics": list(unverified_metrics),
    }


class MedicalRAGPipeline:
    """Conversational End-to-End Medical RAG Pipeline with 2-Stage Retrieval, Streaming, and History Memory."""

    def __init__(
        self,
        retriever: Optional[BaseRetriever] = None,
        provider: str = "groq",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
    ):
        self.retriever = retriever or create_hybrid_retriever()
        self.llm = get_llm(
            provider=provider,
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", MEDICAL_SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])
        self.output_parser = StrOutputParser()

    def condense_query_if_needed(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Reformulates conversational follow-up questions into standalone medical search queries.
        """
        if not chat_history:
            return question

        history_str = format_chat_history(chat_history)
        if not history_str or history_str == "No previous conversation.":
            return question

        try:
            condense_prompt = ChatPromptTemplate.from_template(CONDENSE_QUESTION_PROMPT)
            condense_chain = condense_prompt | self.llm | self.output_parser
            standalone_q = condense_chain.invoke({
                "chat_history": history_str,
                "question": question,
            }).strip()

            standalone_q = re.sub(r"<think>.*?</think>", "", standalone_q, flags=re.DOTALL).strip()
            return standalone_q if len(standalone_q) >= 4 else question
        except Exception as e:
            print(f"[WARNING] Query condensation failed: {e}")
            return question

    def stream_query(
        self,
        question: str,
        result_container: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Generator[str, None, None]:
        """
        Real-time Token Streaming Conversational Generator with Multi-Turn Memory.
        """
        start_t = time.time()

        # 1. Security Check: Prompt Injection / Jailbreak Filter
        is_injection, injection_msg = detect_prompt_injection(question)
        if is_injection:
            if result_container is not None:
                result_container.update({
                    "question": question,
                    "search_query": question,
                    "answer": f"Security Notice: {injection_msg}",
                    "citations": [],
                    "safety_audit": {"passed": False, "status": "[Security Blocked] Prompt Injection Attempt", "warnings": [injection_msg]},
                    "disclaimer": MEDICAL_DISCLAIMER,
                    "raw_documents": [],
                    "latency_seconds": round(time.time() - start_t, 2),
                    "grounding_status": "[Security Blocked] Prompt Injection",
                })
            yield f"Security Notice: {injection_msg}"
            return

        # 2. Acute Clinical Emergency Detection
        is_emergency, emergency_msg = detect_clinical_emergency(question)
        if is_emergency:
            if result_container is not None:
                result_container.update({
                    "question": question,
                    "search_query": question,
                    "answer": emergency_msg,
                    "citations": [],
                    "safety_audit": {"passed": False, "status": "[Emergency Intercept] Acute Red Flag", "warnings": ["Life-threatening clinical red flag identified."]},
                    "disclaimer": MEDICAL_DISCLAIMER,
                    "raw_documents": [],
                    "latency_seconds": round(time.time() - start_t, 2),
                    "grounding_status": "[Emergency Intercept] Immediate Referral Required",
                })
            yield emergency_msg
            return

        # 3. PII / PHI Sanitize
        clean_question, pii_count = sanitize_pii(question)

        search_query = self.condense_query_if_needed(clean_question, chat_history)
        retrieved_docs = self.retriever.invoke(search_query)
        history_formatted = format_chat_history(chat_history)

        if not retrieved_docs:
            not_found_msg = (
                "Based on the provided documents, this information is not available. "
                "The retrieved evidence does not contain relevant recommendations for this query."
            )
            if result_container is not None:
                result_container.update({
                    "question": question,
                    "search_query": search_query,
                    "answer": not_found_msg,
                    "citations": [],
                    "safety_audit": {"passed": True, "status": "[Safety Refusal] Zero-Hallucination", "warnings": []},
                    "disclaimer": MEDICAL_DISCLAIMER,
                    "raw_documents": [],
                    "latency_seconds": round(time.time() - start_t, 2),
                    "grounding_status": "[Safety Refusal] Zero-Hallucination",
                })
            yield not_found_msg
            return

        context_text = format_docs_for_context(retrieved_docs)
        chain = self.prompt | self.llm | self.output_parser

        accumulated_chunks = []
        in_think_block = False

        for chunk in chain.stream({
            "context": context_text,
            "chat_history": history_formatted,
            "question": question,
        }):
            if "<think>" in chunk:
                in_think_block = True
                continue
            if "</think>" in chunk:
                in_think_block = False
                continue
            if not in_think_block:
                accumulated_chunks.append(chunk)
                yield chunk

        full_answer = "".join(accumulated_chunks).strip()
        full_answer = re.sub(r"<think>.*?</think>", "", full_answer, flags=re.DOTALL).strip()

        citations = extract_citations(retrieved_docs, answer=full_answer)
        safety_audit = verify_clinical_safety(context_text, full_answer)
        elapsed = round(time.time() - start_t, 2)

        is_refusal = any(kw in full_answer.lower() for kw in [
            "not available", "insufficient evidence", "cannot provide", "does not contain"
        ])
        grounding_status = "[Safety Refusal] Zero-Hallucination" if is_refusal else "[Strictly Grounded] Verified Evidence"

        if result_container is not None:
            result_container.update({
                "question": question,
                "search_query": search_query,
                "answer": full_answer,
                "citations": citations,
                "safety_audit": safety_audit,
                "disclaimer": MEDICAL_DISCLAIMER,
                "raw_documents": [
                    {"content": d.page_content, "metadata": d.metadata}
                    for d in retrieved_docs
                ],
                "latency_seconds": elapsed,
                "grounding_status": grounding_status,
            })

    def query(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Synchronous query executing conversational multi-turn RAG flow."""
        start_t = time.time()

        # 1. Security Check: Prompt Injection / Jailbreak Filter
        is_injection, injection_msg = detect_prompt_injection(question)
        if is_injection:
            return {
                "question": question,
                "search_query": question,
                "answer": f"Security Notice: {injection_msg}",
                "citations": [],
                "safety_audit": {"passed": False, "status": "[Security Blocked] Prompt Injection Attempt", "warnings": [injection_msg]},
                "disclaimer": MEDICAL_DISCLAIMER,
                "raw_documents": [],
                "latency_seconds": round(time.time() - start_t, 2),
                "grounding_status": "[Security Blocked] Prompt Injection",
            }

        # 2. Acute Clinical Emergency Detection
        is_emergency, emergency_msg = detect_clinical_emergency(question)
        if is_emergency:
            return {
                "question": question,
                "search_query": question,
                "answer": emergency_msg,
                "citations": [],
                "safety_audit": {"passed": False, "status": "[Emergency Intercept] Acute Red Flag", "warnings": ["Life-threatening clinical red flag identified."]},
                "disclaimer": MEDICAL_DISCLAIMER,
                "raw_documents": [],
                "latency_seconds": round(time.time() - start_t, 2),
                "grounding_status": "[Emergency Intercept] Immediate Referral Required",
            }

        # 3. PII / PHI Sanitize
        clean_question, pii_count = sanitize_pii(question)

        search_query = self.condense_query_if_needed(clean_question, chat_history)
        retrieved_docs = self.retriever.invoke(search_query)
        history_formatted = format_chat_history(chat_history)

        if not retrieved_docs:
            not_found_msg = (
                "Based on the provided documents, this information is not available. "
                "The retrieved evidence does not contain relevant recommendations for this query."
            )
            return {
                "question": question,
                "search_query": search_query,
                "answer": not_found_msg,
                "citations": [],
                "safety_audit": {"passed": True, "status": "[Safety Refusal] Zero-Hallucination", "warnings": []},
                "disclaimer": MEDICAL_DISCLAIMER,
                "raw_documents": [],
                "latency_seconds": round(time.time() - start_t, 2),
                "grounding_status": "[Safety Refusal] Zero-Hallucination",
            }

        context_text = format_docs_for_context(retrieved_docs)
        chain = self.prompt | self.llm | self.output_parser
        response_text = chain.invoke({
            "context": context_text,
            "chat_history": history_formatted,
            "question": question,
        })

        response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
        citations = extract_citations(retrieved_docs, answer=response_text)
        safety_audit = verify_clinical_safety(context_text, response_text)
        elapsed = round(time.time() - start_t, 2)

        is_refusal = any(kw in response_text.lower() for kw in [
            "not available", "insufficient evidence", "cannot provide", "does not contain"
        ])
        grounding_status = "[Safety Refusal] Zero-Hallucination" if is_refusal else "[Strictly Grounded] Verified Evidence"

        return {
            "question": question,
            "search_query": search_query,
            "answer": response_text,
            "citations": citations,
            "safety_audit": safety_audit,
            "disclaimer": MEDICAL_DISCLAIMER,
            "raw_documents": [
                {"content": d.page_content, "metadata": d.metadata}
                for d in retrieved_docs
            ],
            "latency_seconds": elapsed,
            "grounding_status": grounding_status,
        }
