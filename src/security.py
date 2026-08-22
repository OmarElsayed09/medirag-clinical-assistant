"""
MediRAG Security & Clinical Guardrails Engine.
Implements:
1. Context-Aware Prompt Injection & Adversarial Attack Defense.
2. Context-Aware Acute Emergency Interceptor (Distinguishes active patient crises from academic/guideline questions).
3. Medical PII/PHI (Protected Health Information) Redaction & Anonymization.
4. Secure File Upload Validator (Magic Bytes, Max File Size, Path Traversal Prevention).
5. Sliding Window Rate Limiter for API Protection (Redis-ready abstraction).
"""

import os
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


# --- 1. PROMPT INJECTION & JAILBREAK PATTERNS ---
PROMPT_INJECTION_PATTERNS = [
    r"(?i)\bignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)\b",
    r"(?i)\bdisregard\s+(all\s+)?(previous|prior|system)\s+(instructions|guidelines)\b",
    r"(?i)\byou\s+are\s+now\s+(in\s+)?(developer|dan|unrestricted|god|jailbroken)\s+mode\b",
    r"(?i)\b(system\s+prompt|system\s+instructions)\s*[:=]",
    r"(?i)\bpretend\s+you\s+(are\s+not|have\s+no)\s+(rules|guidelines|medical\s+limitations)\b",
    r"(?i)\bdo\s+anything\s+now\b",
    r"(?i)\bbypass\s+(all\s+)?(safety|medical|content)\s+(filters|guardrails)\b",
    r"(?i)\bact\s+as\s+an\s+unfiltered\s+ai\b",
    r"(?i)\brepeat\s+the\s+words\s+above\b",
    r"(?i)\bshow\s+(me\s+)?(your|the)\s+(hidden|internal|system)\s+(prompt|instructions)\b",
]

# --- 2. ACUTE CLINICAL EMERGENCY RED FLAGS ---
ACUTE_EMERGENCY_PATTERNS = [
    (
        r"(?i)\b(crushing|severe|radiating|substernal)\s+(chest\s+pain|chest\s+pressure)\b",
        "Suspected Acute Coronary Syndrome / Myocardial Infarction. Immediate emergency evaluation and ECG required."
    ),
    (
        r"(?i)\b(sudden\s+weakness|facial\s+droop|slurred\s+speech|hemiparesis|loss\s+of\s+speech)\b",
        "Suspected Acute Stroke / Cerebrovascular Event (FAST criteria positive). Immediate emergency transfer required."
    ),
    (
        r"(?i)\b(anaphylaxis|swollen\s+tongue|throat\s+tightness|stridor|difficulty\s+breathing\s+after\s+(sting|peanut|medication|injection))\b",
        "Suspected Severe Anaphylaxis. Immediate intramuscular Epinephrine and emergency airway management required."
    ),
    (
        r"(?i)\b(blood\s+pressure\s+(above|>|>=)\s*200/\d+|hypertensive\s+crisis\s+with\s+(headache|vision\s+loss|chest\s+pain|seizures))\b",
        "Suspected Hypertensive Emergency with potential acute target-organ damage. Immediate emergency care required."
    ),
]

# Markers indicating academic / clinical guideline query (NOT an active patient crisis)
ACADEMIC_GUIDELINE_MARKERS = [
    r"(?i)\b(what\s+is|what\s+are|how\s+does|how\s+do|according\s+to|protocol\s+for|definition\s+of)\b",
    r"(?i)\b(guidelines?|recommendations?|criteria|classification|management\s+algorithm|first-line)\b",
    r"(?i)\b(who\b|nice\b|acc/aha\b|esh\b|triaging\s+protocol|diagnostic\s+criteria)\b",
    r"(?i)\b(does\s+the\s+guideline|what\s+do\s+the\s+guidelines)\b",
]

# Markers indicating active patient distress / personal acute emergency
ACTIVE_DISTRESS_MARKERS = [
    r"(?i)\b(i\s+am|i'm|my\s+father|my\s+mother|my\s+child|help\s+me|right\s+now|just\s+started|since\s+\d+\s+min)\b",
    r"(?i)\b(i\s+feel|i\s+have|she\s+has|he\s+has|collapsed|unresponsive|passing\s+out)\b",
    r"(?i)\b(what\s+should\s+i\s+do\s+now|hurry|emergency)\b",
]

# --- 3. PII / PHI PATTERNS FOR SANITIZATION ---
PII_PATTERNS = [
    # Email addresses
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b", "[REDACTED_EMAIL]"),
    # Phone numbers
    (r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[REDACTED_PHONE]"),
    # Social Security / National ID numbers (generic formats)
    (r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED_SSN]"),
    (r"\b\d{14}\b", "[REDACTED_NATIONAL_ID]"),
    # Dates of Birth markers
    (r"(?i)(?:dob|date of birth|born on)\s*[:=]?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", "[REDACTED_DOB]"),
    # Patient Name indicators
    (r"(?i)(?:patient\s+name|pt\s+name)\s*[:=]\s*([A-Za-z]+(\s+[A-Za-z]+){1,3})", "[REDACTED_PATIENT_NAME]"),
]


def detect_prompt_injection(text: str) -> Tuple[bool, Optional[str]]:
    """
    Scans input text for known prompt injection, system prompt extraction, and adversarial jailbreak vectors.
    Returns (is_violation, reason).
    """
    if not text:
        return False, None

    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, text):
            return True, "Adversarial prompt injection pattern detected. Query rejected for security compliance."

    # Check for excessive delimiter repetition / adversarial payloads
    if re.search(r"(?:={15,}|-{15,}|\*{15,}|_{15,})", text) and ("system" in text.lower() or "prompt" in text.lower()):
        return True, "Suspicious payload structure detected. Query rejected for security compliance."

    return False, None


def detect_clinical_emergency(text: str) -> Tuple[bool, Optional[str]]:
    """
    Context-aware Clinical Emergency Interceptor:
    1. Identifies acute life-threatening medical red flags.
    2. Distinguishes active patient crises from educational guideline inquiries.
    Returns (is_emergency, emergency_protocol_message).
    """
    if not text:
        return False, None

    # Check if text contains any acute red flag pattern
    found_warning = None
    for pattern, warning_msg in ACUTE_EMERGENCY_PATTERNS:
        if re.search(pattern, text):
            found_warning = warning_msg
            break

    if not found_warning:
        return False, None

    # Evaluate context: Is this an educational/guideline query or active patient distress?
    is_active_distress = any(re.search(pat, text) for pat in ACTIVE_DISTRESS_MARKERS)
    is_academic_query = any(re.search(pat, text) for pat in ACADEMIC_GUIDELINE_MARKERS)

    # If it's clearly an academic / guideline question without active distress markers, allow RAG retrieval
    if is_academic_query and not is_active_distress:
        return False, None

    # Otherwise, intercept for immediate emergency redirection
    return True, (
        f"CRITICAL CLINICAL ALERT: {found_warning} "
        "This query indicates an acute, potentially life-threatening clinical situation. "
        "Do NOT rely on routine chatbot answers. Immediately activate emergency medical services (call 911 / 123 / 999) "
        "or transfer to the nearest Emergency Department."
    )


def sanitize_pii(text: str) -> Tuple[str, int]:
    """
    Redacts Protected Health Information (PHI) and PII before forwarding to cloud LLMs.
    Returns (sanitized_text, redaction_count).
    """
    if not text:
        return text, 0

    sanitized = text
    total_redactions = 0

    for pattern, replacement in PII_PATTERNS:
        matches = re.findall(pattern, sanitized)
        if matches:
            total_redactions += len(matches)
            sanitized = re.sub(pattern, replacement, sanitized)

    return sanitized, total_redactions


def validate_pdf_upload(file_bytes: bytes, filename: str, max_size_mb: int = 30) -> Tuple[bool, Optional[str]]:
    """
    Validates uploaded document files:
    1. Checks filename for path traversal / dangerous extensions.
    2. Enforces maximum byte size.
    3. Validates PDF magic header (%PDF-).
    """
    if not filename:
        return False, "Filename cannot be empty."

    # Prevent Path Traversal
    safe_name = Path(filename).name
    if safe_name != filename or ".." in filename or "/" in filename or "\\" in filename:
        return False, "Invalid filename. Path traversal characters detected."

    # Extension Check
    if not safe_name.lower().endswith(".pdf"):
        return False, "Invalid file format. Only official PDF clinical documents are permitted."

    # File Size Check
    max_bytes = max_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        return False, f"File size ({round(len(file_bytes) / (1024*1024), 1)} MB) exceeds maximum limit of {max_size_mb} MB."

    if len(file_bytes) < 10:
        return False, "Uploaded file is empty or corrupted."

    # Magic Bytes Validation for PDF (%PDF-)
    if not file_bytes.startswith(b"%PDF-"):
        return False, "File header verification failed. The file is not a valid PDF document."

    return True, None


# --- 5. SLIDING WINDOW RATE LIMITER ---
class InMemoryRateLimiter:
    """
    Lightweight sliding-window rate limiter per client IP address.
    Prevents API denial of service and LLM quota exhaustion.
    """

    def __init__(self, requests_per_minute: int = 40):
        self.limit = requests_per_minute
        self.window_seconds = 60
        self.history: Dict[str, List[float]] = {}

    def is_allowed(self, client_ip: str) -> Tuple[bool, int]:
        """Returns (is_allowed, remaining_requests)."""
        now = time.time()
        client_history = self.history.get(client_ip, [])

        # Filter out timestamps older than the sliding window
        valid_timestamps = [t for t in client_history if now - t < self.window_seconds]

        if len(valid_timestamps) >= self.limit:
            self.history[client_ip] = valid_timestamps
            return False, 0

        valid_timestamps.append(now)
        self.history[client_ip] = valid_timestamps
        remaining = self.limit - len(valid_timestamps)
        return True, remaining

    def reset(self):
        """Clears all rate limiting history."""
        self.history.clear()


# Global Security Limiter Instance
security_rate_limiter = InMemoryRateLimiter(requests_per_minute=40)
