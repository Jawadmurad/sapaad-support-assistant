import glob
import json
import os
import re
import urllib.error
import urllib.request
import streamlit as st
from groq import Groq
from huggingface_hub import snapshot_download
from rank_bm25 import BM25Okapi

# ================= CONFIGURATION =================
st.set_page_config(
    page_title="Sapaad Support AI Assistant",
    page_icon="📩",
    layout="wide",
)


def get_secret(name: str, default: str = "") -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return os.environ.get(name, default).strip()


GROQ_API_KEY = get_secret("GROQ_API_KEY")
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
HF_TOKEN = get_secret("HF_TOKEN")
HF_DATASET_REPO = get_secret("HF_DATASET_REPO")
TEAM_PASSWORD = get_secret("TEAM_PASSWORD")
LOCAL_DATA_DIR = "data"

MAX_CONTEXT_CHARS = 10800

# Model choices (provider, model_id)
MODELS = {
    "Groq: GPT-OSS 120B (Best Quality)": ("groq", "openai/gpt-oss-120b"),
    "Groq: Qwen 3.8 27B (Strong Arabic/EN)": ("groq", "qwen/qwen3.8-27b"),
    "Groq: GPT-OSS 20B (Ultra-Fast)": ("groq", "openai/gpt-oss-20b"),
    "Gemini: 2.5 Flash (Strong Fallback / Arabic)": ("gemini", "gemini-2.5-flash"),
    "Gemini: 2.5 Flash-Lite (Fast Fallback)": ("gemini", "gemini-2.5-flash-lite"),
}

# Automatic fallback order for main generation
FALLBACK_CHAIN = [
    ("groq", "openai/gpt-oss-120b"),
    ("groq", "qwen/qwen3.8-27b"),
    ("groq", "openai/gpt-oss-20b"),
    ("gemini", "gemini-2.5-flash"),
    ("gemini", "gemini-2.5-flash-lite"),
]

# Query extractor fallback order (uses fast/separate bucket first)
EXTRACTOR_CHAIN = [
    ("groq", "openai/gpt-oss-20b"),
    ("groq", "qwen/qwen3.8-27b"),
    ("gemini", "gemini-2.5-flash-lite"),
    ("gemini", "gemini-2.5-flash"),
]

OVERVIEW_KB_FILES = {"subscription.md", "general.md", "about.md"}

SYSTEM_PROMPT = """You are a Sapaad Customer Support email assistant. Your job is to draft customer replies, refine rough drafts, and write internal Groove ticket notes using the provided Sapaad Knowledge Base (KB) and historical Groove Support Tickets as your primary references.

1. SOURCE PRIORITY & ACCURACY
- Always ground your answer in the provided REFERENCE CONTEXT. Never invent Sapaad features, menu paths, pricing, third-party provider names, API keys, or policies.
- Prioritize Sapaad KB articles for official step-by-step navigation paths and product rules.
- Use historical Groove tickets for real-world edge cases, troubleshooting workarounds, and Sapaad's standard support phrasing.
- Prioritize recent tickets (2025–2026) over older tickets (2023–2024). Always recommend AnyDesk (never TeamViewer) for remote support, and use the current Cloud Print Agent link (prints.sapaad.com/getcpa).
- NEVER invent or guess Sapaad menu paths (`>`), settings, or button names—not even for prerequisite steps (e.g., if the context mentions creating a surcharge without giving its exact menu path, state the prerequisite clearly without inventing a `>` path). ONLY include a `>` menu path if it explicitly appears in the REFERENCE CONTEXT below.
- Always keep Sapaad UI menu paths and product names in their exact original English (e.g., Setup > Location Setup > Self Ordering Systems > Sapaad Direct) even when replying in Arabic, with ZERO mixed-script typos.
- Do NOT assume the customer is on a "Lite" plan unless they explicitly say so; provide the standard setup steps first and mention plan tiers only as a brief note if relevant.
- Factual accuracy ALWAYS overrides "guide alone" instructions: if a feature is unsupported or requires Sapaad backend action (such as open APIs or backend integrations), state that limitation clearly instead of inventing self-serve steps.
- If the context does not clearly confirm an answer or requires account-level checking, add a short "[Agent Note: ...]" line at the very top BEFORE the email greeting.

2. AGENT INSTRUCTIONS PRIORITY & TONE SAFETY
- Strictly obey any AGENT INSTRUCTIONS provided by the support agent (e.g., do not offer a meeting, do not promise to check their account).
- Follow agent instructions SILENTLY and NATURALLY—never leak abrupt internal instructions into the customer email (e.g., never tell the customer "without needing a meeting or review from our side"; instead say warmly "Below are the direct steps so you can configure each item easily").

3. PRIVACY & PLACEHOLDER SAFETY
- NEVER copy redacted placeholders ([PHONE_OR_ID], [EMAIL]), other customers' names, restaurant names, passwords, or expired Zoom/WeTransfer links from past tickets into a new draft.
- Use clean bracketed placeholders like [Zoom Link] or [Date & Time] only if the agent hasn't provided them.

4. TONE, LANGUAGE & FORMATTING
- Always write in the "we" tone (representing the Sapaad Support Team), polite, clear, and concise.
- Match the requested or customer's language (English or Arabic).
- In Arabic replies, use consistent respectful plural addressing throughout (e.g., افتحوا، اختاروا، انتقلوا، اضغطوا، حرّروا، احفظوا، قوموا — never mix singular and plural verbs, and spell واختاروا correctly).
- Greeting:
  - English: "Dear [Customer Name]," or "Hi [Customer Name]," (use "Hi," if no name is provided).
  - Arabic: Always start with "السلام عليكم،" followed by a blank line and a brief 1-sentence warm opening before numbered points.
- Mandatory Closing (Customer Emails Only — include ONCE, with NO duplicate closing sentence right above it):
  - English: End every customer email with exactly:
    Please feel free to write to us for any further assistance.
  - Arabic: End every customer email with exactly:
    لا تترددوا بالتواصل معنا لأي مساعدة إضافية.
- NO sign-off or signature: Never add "Best regards,", "Thanks,", or an agent name at the bottom (Groove appends the signature automatically).
- For multi-part customer inquiries, use clean numbered points (1., 2., 3.) with consistent spacing and blank lines between points.
- PLAIN TEXT ONLY: Never use Markdown formatting. Do NOT use **bold**, *italics*, ### headers, or backticks.
"""

REFINE_SYSTEM_PROMPT = """You are a senior Sapaad Customer Support editor. Your job is to polish and elevate an agent's rough draft into a clear, warm, professional Sapaad Support email ready to paste into Groove.

RULES:
- Preserve EVERY fact, date, time, link, thank-you sentence, and commitment from the agent's rough draft. Do NOT drop sentences or invent new promises.
- Genuinely improve grammar, clarity, flow, and professional "we" tone (representing the Sapaad Support Team).
- Keep Sapaad UI names and menu paths in English even when refining an Arabic email.
- In Arabic, use consistent respectful plural addressing (e.g., تفضلوا، يرجى، يسعدنا، يمكنكم).
- Apply proper greeting ("Hi [Name]," / "Dear [Name]," / "السلام عليكم،") and the exact mandatory closing line once at the end:
  - English: Please feel free to write to us for any further assistance.
  - Arabic: لا تترددوا بالتواصل معنا لأي مساعدة إضافية.
- NO sign-off ("Best regards,", agent name) and NO Markdown formatting (**bold**, ###, backticks). Plain text only.
"""


# ================= SMART BM25+ CLEANING & SYNONYMS =================
BOILERPLATE_PATTERNS = [
    r"Did you know Sapaad is integrated with.*?(?=\n|$)",
    r"Global Support Numbers:.*?(?=\n|$)",
    r"NOTICE:\s*This electronic mail message.*$",
    r"CONFIDENTIALITY CAUTION:.*$",
    r"This email(?: and any attachments)? may contain confidential.*$",
    r"قد تحتوي هذه الرسالة الإلكترونية على معلومات سرية.*$",
    r"Sent from (?:my iPhone|my iPad|Outlook for \w+|Yahoo Mail|Mail for Windows|Samsung).*",
    r"Get Outlook for (?:iOS|Android).*",
    r"Please reconsider not printing out this email.*",
    r"\[PHONE_OR_ID\]|\[EMAIL\]",
]

GENERIC_RESOLUTION_PATTERNS = [
    r"could you please (?:further )?elaborate",
    r"share the (?:name of your restaurant|registered email)",
    r"forwarded your (?:details|request) to our sales team",
    r"remotely connected to your (?:system|device) via anydesk and assisted",
    r"contacted you and assisted you with your (?:query|requirement)",
    r"tried (?:reaching|calling) you.*unable to get through",
    r"kindly share your anydesk id",
]

# Static local Sapaad synonym dictionary (0 tokens, <1ms)
SAPAAD_SYNONYMS = {
    "receipt machine": "thermal printer cloud print agent",
    "receipt printer": "thermal printer cloud print agent printing rules",
    "bill printer": "thermal printer receipt print",
    "kitchen printer": "kot kitchen slip printing rules cloud print agent",
    "not printing": "cloud print agent cpa printing rules printer",
    "kitchen screen": "kds kitchen display system",
    "cancel item": "void item cancel order",
    "delete item": "void remove menu item",
    "delivery apps": "aggregators urbanpiper atlas sapaad connect",
    "talabat": "urbanpiper atlas aggregator talabat",
    "deliveroo": "urbanpiper atlas sapaad connect deliveroo",
    "noon": "urbanpiper atlas noon food",
    "careem": "urbanpiper atlas careem",
    "online store": "sapaad.online online ordering",
    "website ordering": "sapaad.online online ordering",
    "qr menu": "sapaad direct qr code table qr",
    "table ordering": "sapaad direct qr code dine in",
    "waiter button": "waiter hailing sapaad direct",
    "whatsapp otp": "email otp sms otp guest checkout",
    "no otp": "guest checkout disable otp checkout mode",
    "cash on delivery": "pay at counter cash on delivery payment types",
    "cod": "pay at counter cash on delivery",
    "zatca": "zatca phase 2 e-invoicing fatoora",
    "fatoora": "zatca phase 2 e-invoicing",
    "add branch": "add location subscription upgrade",
    "new branch": "add location subscription upgrade",
    "loyalty": "loyalty points rewards expiry",
    "inventory": "inventory track stock ingredients recipes stock take",
    "quickbooks": "quickbooks accounting sync integration",
    "xero": "xero accounting sync integration",
    "delivery fee": "delivery areas surcharge default delivery fee",
    "scheduled order": "accept scheduled orders pre-order sapaad.online",
    "api": "sapaad connect api integration logistics partners",
}


def clean_for_index(text: str) -> str:
    """Remove noisy signatures, disclaimers, and normalize common Sapaad typos."""
    cleaned = text
    for pat in BOILERPLATE_PATTERNS:
        cleaned = re.sub(pat, " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"\b(saapad|sappad|spaad|sapad)\b", "sapaad", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bquickbook\b", "quickbooks", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\burban\s+piper\b", "urbanpiper", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bany\s+desk\b", "anydesk", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def expand_synonyms(query: str) -> str:
    """Append Sapaad-native synonyms locally without using LLM tokens."""
    q_lower = query.lower()
    additions = []
    for trigger, expansion in SAPAAD_SYNONYMS.items():
        if trigger in q_lower:
            additions.append(expansion)
    if additions:
        return f"{query} " + " ".join(additions)
    return query


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w\u0600-\u06FF]+", text.lower())


def split_ticket_parts(body: str) -> tuple[str, str]:
    """Split a ticket body into (customer_part, resolution_part)."""
    match = re.search(r"\*\*Support Resolution:\*\*|Support Resolution:", body, re.IGNORECASE)
    if not match:
        return body.strip(), ""
    customer_part = body[:match.start()].strip()
    resolution_part = body[match.end():].strip()
    customer_part = re.sub(r"^\*\*Customer:\*\*\s*|Customer:\s*", "", customer_part, flags=re.IGNORECASE).strip()
    return customer_part, resolution_part


def compute_ticket_weight(resolution_part: str, year: int) -> float:
    """Score ticket reliability based on recency and resolution substance."""
    if not resolution_part or len(resolution_part) < 35:
        return 0.0  # Skip unresolved or empty-resolution tickets

    base = 1.25 if year >= 2025 else 0.85
    if len(resolution_part) < 70:
        return base * 0.40

    if len(resolution_part) < 350:
        for pat in GENERIC_RESOLUTION_PATTERNS:
            if re.search(pat, resolution_part, re.IGNORECASE):
                return base * 0.45

    return base


def format_ticket_for_context(header: str, customer_part: str, resolution_part: str) -> str:
    """Context Diet: keep customer summary concise so the full resolution is never cut off."""
    cust_short = clean_for_index(customer_part)[:420]
    res_clean = resolution_part[:1600].strip()
    return f"{header}\nCustomer Issue: {cust_short}\nSupport Resolution:\n{res_clean}"


# ================= DATA LOADING & INDEXING =================
def load_data_directory() -> str:
    """Use local ./data folder if present, otherwise download from Private HF Dataset."""
    if os.path.isdir(LOCAL_DATA_DIR) and glob.glob(os.path.join(LOCAL_DATA_DIR, "*.md")):
        return LOCAL_DATA_DIR

    if HF_DATASET_REPO and HF_TOKEN:
        cache_dir = "/tmp/sapaad_data"
        snapshot_download(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            token=HF_TOKEN,
            local_dir=cache_dir,
            allow_patterns=["*.md"],
        )
        return cache_dir

    return ""


def chunk_kb_article(title: str, body: str, max_chunk: int = 1900, overlap: int = 250) -> list[str]:
    """Split long KB articles into overlapping chunks so bottom sections aren't lost."""
    body = body.strip()
    if len(body) <= max_chunk:
        return [body]
    chunks = []
    start = 0
    while start < len(body):
        end = start + max_chunk
        chunks.append(body[start:end].strip())
        if end >= len(body):
            break
        start = end - overlap
    return chunks


@st.cache_resource(show_spinner="Indexing Sapaad KB & Support Tickets (Smart BM25+)...")
def build_search_index():
    data_dir = load_data_directory()
    if not data_dir:
        return None, [], 0, 0, 0

    docs = []
    corpus = []
    kb_count = 0
    ticket_count = 0
    skipped_tickets = 0

    for filepath in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        filename = os.path.basename(filepath)
        is_ticket_file = filename.lower().startswith("tickets_")

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        if is_ticket_file:
            parts = re.split(r"(?m)^#{3,5}\s+(Ticket\s+#\d+[^\n]*)", content)
            for i in range(1, len(parts), 2):
                header = parts[i].strip()
                body = parts[i + 1].strip() if i + 1 < len(parts) else ""
                if len(body) < 40:
                    skipped_tickets += 1
                    continue

                customer_part, resolution_part = split_ticket_parts(body)
                year_match = re.search(r"\((\d{4})-", header)
                year = int(year_match.group(1)) if year_match else 2024
                weight = compute_ticket_weight(resolution_part, year)

                if weight <= 0.0:
                    skipped_tickets += 1
                    continue

                clean_title = clean_for_index(header)
                clean_cust = clean_for_index(customer_part)[:600]
                clean_res = clean_for_index(resolution_part)[:1600]

                # BM25F-style field boosting: Title 3x, Customer Issue 2x, Resolution 1x
                indexed_text = f"{clean_title} {clean_title} {clean_title} {clean_cust} {clean_cust} {clean_res}"

                docs.append({
                    "source": filename,
                    "title": header,
                    "type": "Ticket",
                    "year": year,
                    "weight": weight,
                    "text": format_ticket_for_context(header, customer_part, resolution_part),
                })
                corpus.append(tokenize(indexed_text))
                ticket_count += 1
        else:
            kb_weight = 1.05 if filename.lower() in OVERVIEW_KB_FILES else 1.40
            parts = re.split(r"(?m)^#{2,4}\s+([^\n]+)", content)
            if len(parts) > 2:
                for i in range(1, len(parts), 2):
                    title = parts[i].strip()
                    body = parts[i + 1].strip() if i + 1 < len(parts) else ""
                    if len(body) < 35:
                        continue
                    # Skip file-level table of contents / metadata blocks
                    if "Total Articles:" in body[:250] or "Consolidated Knowledge Base" in body[:250]:
                        continue

                    for c_idx, chunk in enumerate(chunk_kb_article(title, body)):
                        chunk_title = title if c_idx == 0 else f"{title} (Part {c_idx + 1})"
                        clean_title = clean_for_index(title)
                        clean_body = clean_for_index(chunk)
                        # BM25F-style boost: KB Title 3x
                        indexed_text = f"{clean_title} {clean_title} {clean_title} {clean_body}"

                        docs.append({
                            "source": filename,
                            "title": chunk_title,
                            "type": "KB",
                            "year": 2026,
                            "weight": kb_weight,
                            "text": f"KB Article: {chunk_title}\n{chunk}",
                        })
                        corpus.append(tokenize(indexed_text))
                        kb_count += 1
            elif len(content.strip()) > 35:
                clean_body = clean_for_index(content[:2500])
                docs.append({
                    "source": filename,
                    "title": filename,
                    "type": "KB",
                    "year": 2026,
                    "weight": kb_weight,
                    "text": content[:2500],
                })
                corpus.append(tokenize(f"{filename} {filename} {clean_body}"))
                kb_count += 1

    if not docs:
        return None, [], 0, 0, skipped_tickets

    bm25 = BM25Okapi(corpus)
    return bm25, docs, kb_count, ticket_count, skipped_tickets


# ================= MULTI-PROVIDER LLM ENGINE (GROQ + GEMINI) =================
def _call_groq_raw(model_id: str, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not configured")
    client = Groq(api_key=GROQ_API_KEY)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    resp = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = (resp.choices[0].message.content or "").strip()
    if not content:
        raise RuntimeError(f"Empty response from Groq model {model_id}")
    return content


def _call_gemini_raw(model_id: str, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
    """Zero-dependency Gemini REST API call using Python stdlib urllib."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not configured")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    if system_prompt:
        payload["system_instruction"] = {"parts": [{"text": system_prompt}]}

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Gemini HTTP {e.code}: {err_body[:300]}") from e

    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {str(data)[:200]}")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts).strip()
    if not text:
        raise RuntimeError(f"Empty text from Gemini model {model_id}")
    return text


def call_provider(provider: str, model_id: str, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
    if provider == "groq":
        return _call_groq_raw(model_id, system_prompt, user_prompt, temperature, max_tokens)
    if provider == "gemini":
        return _call_gemini_raw(model_id, system_prompt, user_prompt, temperature, max_tokens)
    raise ValueError(f"Unknown provider: {provider}")


# ================= SMART BYPASS & QUERY EXTRACTION =================
def should_bypass_llm_extraction(text: str) -> bool:
    """Skip LLM query extraction for short, single-topic English emails (saves ~35% tokens)."""
    stripped = text.strip()
    if not stripped or len(stripped) > 350:
        return False
    # Never bypass if Arabic characters are present (needs cross-language translation)
    if re.search(r"[\u0600-\u06FF]", stripped):
        return False
    # Never bypass multi-part numbered/bulleted emails or multiple questions
    bullet_lines = re.findall(r"(?m)^\s*(?:\d+[\.\)]|[-•*])\s+", stripped)
    if len(bullet_lines) >= 2 or stripped.count("?") >= 2:
        return False
    return True


@st.cache_data(ttl=1800, show_spinner=False)
def extract_english_subqueries(text: str, draft_model_id: str) -> tuple[list[str], str]:
    """Translate & split Arabic/multi-topic emails into Sapaad-native English queries with Groq->Gemini fallback."""
    if should_bypass_llm_extraction(text):
        cleaned = expand_synonyms(clean_for_index(text))
        return [cleaned], "Smart Bypass (0 tokens)"

    # Prefer a model different from the drafting model to avoid Groq TPM collisions
    chain = [item for item in EXTRACTOR_CHAIN if item[1] != draft_model_id] + [
        item for item in EXTRACTOR_CHAIN if item[1] == draft_model_id
    ]

    extractor_prompt = (
        "You are a search query extractor for Sapaad POS support documentation.\n"
        "Read the customer email below (Arabic or English) and extract EACH distinct support topic "
        "as a separate short ENGLISH search query (3 to 7 English keywords per line). "
        "Do NOT merge two different questions into one line.\n"
        "Use exact Sapaad terminology where relevant:\n"
        "- QR table ordering -> Sapaad Direct QR code setup\n"
        "- Waiter call button -> waiter hailing Sapaad Direct\n"
        "- Disable OTP / no OTP -> guest checkout Checkout Mode disable OTP\n"
        "- WhatsApp OTP alternative -> enable Email OTP SMS\n"
        "- Cash on delivery -> Pay at Counter Cash on Delivery payment types\n"
        "- Scheduled / pre-orders -> Accept Scheduled Orders Sapaad.Online\n"
        "- Delivery API / third-party courier -> Sapaad Connect logistics partners API integration\n"
        "- Delivery fee per area -> Delivery Areas Delivery Fee surcharge\n"
        "Output ONLY the English lines—no Arabic, no numbering, no bullets, no explanations. Max 10 lines.\n\n"
        f"{text[:2600]}"
    )

    last_err = None
    for provider, model_id in chain:
        if provider == "groq" and not GROQ_API_KEY:
            continue
        if provider == "gemini" and not GEMINI_API_KEY:
            continue
        try:
            raw = call_provider(provider, model_id, "", extractor_prompt, temperature=0.0, max_tokens=1200)
            lines = []
            for ln in raw.splitlines():
                cleaned = re.sub(r"^[\d\.\-\•\*\s]+", "", ln).strip()
                # Filter out broken 1-word fragments and non-English lines
                if cleaned and len(cleaned.split()) >= 2 and re.search(r"[a-zA-Z]{2,}", cleaned):
                    lines.append(expand_synonyms(cleaned))
            if lines:
                return lines[:10], f"{provider}:{model_id}"
        except Exception as e:
            last_err = e
            continue

    fallback_q = expand_synonyms(clean_for_index(text))
    return [fallback_q], f"Fallback Raw ({str(last_err)[:60]})"


# ================= SMART BM25+ WITH RRF RETRIEVAL =================
def retrieve_context(bm25, docs, query: str, draft_model_id: str, top_k: int = 6):
    """Smart BM25+ with Per-Topic Guarantee + Reciprocal Rank Fusion (RRF) + Context Budget."""
    if not bm25 or not query.strip():
        return [], [], "None"

    subqueries, extractor_used = extract_english_subqueries(query, draft_model_id)
    is_multi_topic = len(subqueries) > 1
    max_docs = max(top_k, min(11, len(subqueries) * 2)) if is_multi_topic else top_k

    rrf_scores = {}
    k_rrf = 60
    per_topic_priority = []
    seen_priority = set()

    for q in subqueries:
        tokens = tokenize(q)
        if not tokens:
            continue
        raw_scores = bm25.get_scores(tokens)
        scored = [
            (score * docs[idx]["weight"], idx)
            for idx, score in enumerate(raw_scores)
            if score > 0 and docs[idx]["weight"] > 0
        ]
        scored.sort(reverse=True)

        # Accumulate RRF score across all subqueries
        for rank, (_, idx) in enumerate(scored[:25], start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (k_rrf + rank))

        # For multi-topic emails, guarantee the #1 match for EACH distinct topic
        if is_multi_topic and scored:
            for _, idx in scored[:5]:
                if idx not in seen_priority:
                    per_topic_priority.append(idx)
                    seen_priority.add(idx)
                    break

    rrf_ranked = [idx for idx, _ in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)]

    # Assemble final list: per-topic top matches first, then RRF-ranked backfill with KB/Ticket balance
    selected = []
    seen = set()
    total_chars = 0
    kb_count = 0
    ticket_count = 0
    kb_cap = max(4, max_docs // 2 + 1)

    def try_add(idx: int, enforce_kb_cap: bool = True) -> bool:
        nonlocal total_chars, kb_count, ticket_count
        if idx in seen:
            return False
        doc = docs[idx]
        if enforce_kb_cap and doc["type"] == "KB" and kb_count >= kb_cap and ticket_count == 0:
            return False
        doc_len = len(doc["text"])
        if total_chars + doc_len > MAX_CONTEXT_CHARS and len(selected) >= 3:
            return False
        selected.append(doc)
        seen.add(idx)
        total_chars += doc_len
        if doc["type"] == "KB":
            kb_count += 1
        else:
            ticket_count += 1
        return True

    # Pass 1: Guaranteed top match per topic (for multi-point emails)
    for idx in per_topic_priority:
        if len(selected) >= max_docs:
            break
        try_add(idx, enforce_kb_cap=False)

    # Pass 2: RRF ranked fill (balanced KB + Tickets)
    for idx in rrf_ranked:
        if len(selected) >= max_docs:
            break
        try_add(idx, enforce_kb_cap=True)

    # Pass 3: Backfill without KB cap if still under top_k
    if len(selected) < top_k:
        for idx in rrf_ranked:
            if len(selected) >= top_k:
                break
            try_add(idx, enforce_kb_cap=False)

    # Clean display queries (hide appended synonym tails in UI)
    display_queries = [q[:90] + ("..." if len(q) > 90 else "") for q in subqueries]
    return selected, display_queries, extractor_used


# ================= OUTPUT CLEANING & AGENT NOTE SEPARATION =================
def extract_agent_note(text: str) -> tuple[str, str]:
    """Separate any [Agent Note: ...] lines so Ready to Copy is 100% customer-safe."""
    notes = re.findall(r"(?i)\[Agent Note:\s*([^\]]+)\]", text)
    cleaned = re.sub(r"(?i)^\s*\[Agent Note:[^\]]+\]\s*\n*", "", text, flags=re.MULTILINE)
    cleaned = re.sub(r"(?i)\[Agent Note:[^\]]+\]", "", cleaned)
    return "\n".join(n.strip() for n in notes), cleaned.strip()


def clean_for_groove(text: str, mode: str) -> str:
    """Strip Markdown, fix mixed-script/Arabic slips, and guarantee clean Groove spacing."""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"(?m)^#{1,6}\s+", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # Fix common mixed-script and Arabic spelling slips
    text = re.sub(r"Sapa[\u0600-\u06FF]+", "Sapaad", text, flags=re.IGNORECASE)
    text = text.replace("واختروا", "واختاروا")
    text = text.replace("حرّرو ", "حرّروا ").replace("قموا ", "قوموا ")

    # Normalize spacing
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    # Ensure Arabic customer replies start with السلام عليكم،
    if mode in ("Draft Reply", "Refine Draft") and re.search(r"[\u0600-\u06FF]", text):
        first_line = text.splitlines()[0].strip() if text else ""
        if "السلام عليكم" not in first_line and "مرحباً" not in first_line and "مرحبا" not in first_line:
            text = f"السلام عليكم،\n\n{text}"

    return text


def generate_with_fallback(
    preferred_provider: str,
    preferred_model: str,
    system_prompt: str,
    user_prompt: str,
    mode: str,
    temperature: float = 0.25,
    max_tokens: int = 2800,
) -> tuple[str, str, str]:
    """Try preferred model first, then automatically cascade through Groq -> Gemini fallback chain."""
    chain = [(preferred_provider, preferred_model)]
    for item in FALLBACK_CHAIN:
        if item not in chain:
            chain.append(item)

    errors = []
    for idx, (provider, model_id) in enumerate(chain):
        if provider == "groq" and not GROQ_API_KEY:
            continue
        if provider == "gemini" and not GEMINI_API_KEY:
            continue
        try:
            raw_output = call_provider(provider, model_id, system_prompt, user_prompt, temperature, max_tokens)
            agent_note, body = extract_agent_note(raw_output)
            clean_body = clean_for_groove(body, mode)
            label = f"{provider}:{model_id}" if idx == 0 else f"{provider}:{model_id} (auto-fallback)"
            return clean_body, agent_note, label
        except Exception as e:
            errors.append(f"{provider}:{model_id} -> {str(e)[:120]}")
            continue

    raise RuntimeError("All configured AI providers failed:\n" + "\n".join(errors))


# ================= PASSWORD GATE =================
if TEAM_PASSWORD:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("🔒 Sapaad Support AI Assistant")
        pwd = st.text_input("Enter Team Password", type="password")
        if st.button("Login", type="primary"):
            if pwd == TEAM_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        st.stop()


# ================= MAIN UI =================
if not GROQ_API_KEY and not GEMINI_API_KEY:
    st.error("Missing API keys. Add `GROQ_API_KEY` and/or `GEMINI_API_KEY` in Streamlit Cloud Settings → Secrets.")
    st.stop()

bm25, docs, kb_count, ticket_count, skipped_tickets = build_search_index()

with st.sidebar:
    st.title("⚙️ Sapaad Support AI")
    mode = st.radio(
        "Mode",
        ["Draft Reply", "Refine Draft", "FAQ / Template", "Internal Note / Summary"],
        index=0,
    )
    language = st.selectbox("Output Language", ["Auto (Match Customer)", "English", "Arabic"])
    model_label = st.selectbox("Primary AI Model", list(MODELS.keys()), index=0)
    top_k = st.slider("Reference Sources to Pull", min_value=3, max_value=8, value=5)

    verify_kb_in_refine = False
    if mode == "Refine Draft":
        verify_kb_in_refine = st.checkbox("Verify Sapaad steps against KB/Tickets", value=False)

    st.divider()
    if docs:
        st.success(f"Indexed **{kb_count:,}** KB chunks & **{ticket_count:,}** resolved tickets")
        if skipped_tickets:
            st.caption(f"Filtered out {skipped_tickets:,} unresolved/empty tickets")
    else:
        st.warning("No `.md` files found in `./data` or `HF_DATASET_REPO`.")

    active_providers = []
    if GROQ_API_KEY:
        active_providers.append("Groq")
    if GEMINI_API_KEY:
        active_providers.append("Gemini")
    st.caption(f"Active Providers (Auto-Fallback): **{' → '.join(active_providers)}**")

st.header(f"📩 {mode}")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    if mode == "Draft Reply":
        customer_email = st.text_area(
            "Paste Customer Email",
            height=240,
            placeholder="Paste the full customer email from Groove here...",
        )
        agent_notes = st.text_area(
            "Agent Instructions / Key Points (Optional)",
            height=100,
            placeholder="e.g., Do not offer a meeting / Ask for AnyDesk ID / Confirm they are on iOS...",
        )
    elif mode == "Refine Draft":
        agent_notes = st.text_area(
            "Your Rough Draft to Refine",
            height=200,
            placeholder="Paste your rough draft here...",
        )
        customer_email = st.text_area(
            "Customer Email (Optional context)",
            height=140,
            placeholder="Optional: paste the customer's email for context...",
        )
    elif mode == "FAQ / Template":
        customer_email = st.text_area(
            "Topic / Scenario for Reusable Template",
            height=200,
            placeholder="e.g., How to install Cloud Print Agent and find the API key...",
        )
        agent_notes = st.text_area("Extra Requirements (Optional)", height=100)
    else:
        customer_email = st.text_area(
            "Paste Ticket Thread or Foreign-Language Email",
            height=260,
            placeholder="Paste the email or thread to summarize into an English internal note...",
        )
        agent_notes = st.text_area("Focus / Specific Question (Optional)", height=80)

    generate_btn = st.button("✨ Generate", type="primary", use_container_width=True)

with col2:
    if generate_btn:
        has_input = bool(customer_email.strip() or agent_notes.strip())
        if not has_input:
            st.warning("Please enter an email or draft first.")
        else:
            pref_provider, pref_model = MODELS[model_label]
            with st.spinner("Searching Sapaad KB & Tickets (Smart BM25+) and generating..."):
                run_search = (
                    mode in ("Draft Reply", "FAQ / Template")
                    or (mode == "Refine Draft" and verify_kb_in_refine)
                )
                search_source = customer_email.strip() or agent_notes.strip()

                if run_search:
                    matches, used_queries, extractor_used = retrieve_context(
                        bm25, docs, search_source, pref_model, top_k=top_k
                    )
                else:
                    matches, used_queries, extractor_used = [], [], "Skipped (Not needed for mode)"

                context_block = "\n\n---\n\n".join(
                    f"[{m['type']} | {m['source']}]\n{m['text']}" for m in matches
                )

                lang_rule = (
                    "Match the customer's language (English or Arabic)."
                    if language == "Auto (Match Customer)"
                    else f"Write the output in {language}."
                )

                active_sys_prompt = SYSTEM_PROMPT
                temp = 0.25

                if mode == "Draft Reply":
                    prompt = (
                        f"TASK: Draft a ready-to-send Sapaad Support email reply.\n"
                        f"LANGUAGE: {lang_rule}\n\n"
                        f"CUSTOMER EMAIL:\n{customer_email}\n\n"
                        f"AGENT INSTRUCTIONS (obey strictly and naturally):\n{agent_notes or 'None'}\n\n"
                        f"REFERENCE CONTEXT (Sapaad KB & Past Tickets):\n{context_block or 'No matching context found.'}"
                    )
                elif mode == "Refine Draft":
                    active_sys_prompt = REFINE_SYSTEM_PROMPT
                    temp = 0.35
                    prompt = (
                        f"TASK: Polish and elevate the agent's rough draft below into a clear, warm, professional Sapaad Support email.\n"
                        f"LANGUAGE: {lang_rule}\n\n"
                        f"AGENT ROUGH DRAFT:\n{agent_notes}\n\n"
                        f"CUSTOMER EMAIL (Optional context):\n{customer_email or 'Not provided'}\n\n"
                        f"REFERENCE CONTEXT (for verifying Sapaad terms/steps only):\n{context_block or 'None'}"
                    )
                elif mode == "FAQ / Template":
                    prompt = (
                        f"TASK: Create a concise, reusable Sapaad Support email template with clean placeholders like [Customer Name].\n"
                        f"LANGUAGE: {lang_rule}\n\n"
                        f"TOPIC:\n{customer_email}\n"
                        f"EXTRA NOTES:\n{agent_notes or 'None'}\n\n"
                        f"REFERENCE CONTEXT:\n{context_block or 'None'}"
                    )
                else:
                    prompt = (
                        f"TASK: Write a concise English internal Groove note summarizing this ticket for teammates (translate if needed). Do NOT include a customer greeting or customer closing line.\n\n"
                        f"TICKET CONTENT:\n{customer_email}\n"
                        f"FOCUS:\n{agent_notes or 'Key issue, account details, and next action needed.'}"
                    )

                try:
                    output, agent_note, used_model = generate_with_fallback(
                        pref_provider, pref_model, active_sys_prompt, prompt, mode, temperature=temp
                    )
                    st.session_state.last_output = output
                    st.session_state.last_agent_note = agent_note
                    st.session_state.last_model = used_model
                    st.session_state.last_matches = matches
                    st.session_state.last_queries = used_queries
                    st.session_state.last_extractor = extractor_used
                except Exception as e:
                    st.error(f"API Error: {e}")

    if "last_output" in st.session_state:
        if st.session_state.get("last_agent_note"):
            st.warning(f"📝 **Agent Note (Not included in copy box):**\n\n{st.session_state.last_agent_note}")

        st.subheader("Ready to Copy")
        st.code(st.session_state.last_output, language=None, wrap_lines=True)
        st.caption(
            f"Model used: `{st.session_state.last_model}` | "
            f"Query engine: `{st.session_state.get('last_extractor', 'N/A')}` — "
            "Click the copy icon in the top-right corner of the box above."
        )

        if st.session_state.get("last_matches"):
            with st.expander(f"📚 Referenced Sources ({len(st.session_state.last_matches)})"):
                if st.session_state.get("last_queries"):
                    st.info("Search queries used: " + " | ".join(st.session_state.last_queries))
                for m in st.session_state.last_matches:
                    st.markdown(f"**[{m['type']}] {m['title']}** *({m['source']} | weight: {m['weight']:.2f})*")
                    st.caption(m["text"][:450] + "...")
                    st.divider()
