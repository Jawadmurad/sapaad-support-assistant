import html
import json
import os
import re
import urllib.error
import urllib.request

import streamlit as st
import streamlit.components.v1 as components
from groq import Groq

from attachments import analyze_uploaded_files
from groove_client import fetch_groove_ticket
from privacy import restore_pii, scrub_pii
from retrieval import (
    build_search_index as _build_retrieval_index,
    retrieve_context as _smart_retrieve_context,
)
from verifier import verify_and_repair_draft

# ================= CONFIGURATION =================
st.set_page_config(
    page_title="Sapaad Support AI Assistant",
    page_icon="📩",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stButton"] > button {
        border-radius: 8px;
        font-weight: 500;
        padding: 0.42rem 0.65rem;
        transition: all 0.15s ease;
    }
    div[data-testid="stButton"] > button p {
        font-size: 0.88rem;
        white-space: nowrap;
    }
    .quick-actions-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        opacity: 0.72;
        margin: 0.35rem 0 0.4rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_secret(name: str, default: str = "") -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default)


GROQ_API_KEY = get_secret("GROQ_API_KEY")
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
OPENROUTER_API_KEY = get_secret("OPENROUTER_API_KEY")
HF_TOKEN = get_secret("HF_TOKEN")
HF_DATASET_REPO = get_secret("HF_DATASET_REPO")
TEAM_PASSWORD = get_secret("TEAM_PASSWORD")
CF_ACCOUNT_ID = get_secret("CF_ACCOUNT_ID")
CF_API_TOKEN = get_secret("CF_API_TOKEN")
GROOVE_API_TOKEN = get_secret("GROOVE_API_TOKEN")
GROOVE_BEARER_TOKEN = get_secret("GROOVE_BEARER_TOKEN")
LOCAL_DATA_DIR = "data"

MAX_CONTEXT_CHARS = 10800


def build_budgeted_context(matches: list[dict], customer_email: str, agent_notes: str, language: str) -> str:
    """Dynamically cap retrieved context based on Arabic vs English token density (~2.2 vs ~3.8 chars/token)."""
    combined_in = f"{customer_email}\n{agent_notes}"
    has_arabic = language == "Arabic" or bool(re.search(r"[\u0600-\u06FF]", combined_in))

    char_budget = 6800 if has_arabic else MAX_CONTEXT_CHARS
    per_doc_cap = 1350 if has_arabic else 2200

    chunks = []
    used_chars = 0
    for m in matches:
        body = m["text"][:per_doc_cap]
        snippet = f"[{m['type']} | {m['source']}]\n{body}"
        if used_chars + len(snippet) > char_budget:
            remaining = char_budget - used_chars
            if remaining > 350:
                chunks.append(snippet[:remaining] + "...")
            break
        chunks.append(snippet)
        used_chars += len(snippet) + 8
    return "\n\n---\n\n".join(chunks)


MODELS = {
    "Auto (Smart Route: EN→120B, AR→Qwen)": ("auto", "smart-route"),
    "Groq: GPT-OSS 120B (Best English)": ("groq", "openai/gpt-oss-120b"),
    "Groq: Qwen 3.8 27B (Best Arabic/EN)": ("groq", "qwen/qwen3.8-27b"),
    "Groq: GPT-OSS 20B (Ultra-Fast)": ("groq", "openai/gpt-oss-20b"),
    "Gemini: 2.5 Flash (High Quality)": ("gemini", "gemini-2.5-flash"),
    "Gemini: 2.5 Flash-Lite (Fast Fallback)": ("gemini", "gemini-2.5-flash-lite"),
    "OpenRouter: Free Auto-Router": ("openrouter", "openrouter/free"),
}

SYSTEM_PROMPT = """You are a Sapaad Customer Support email assistant. Your job is to draft customer replies, refine rough drafts, and write internal Groove ticket notes using the provided Sapaad Knowledge Base (KB) and historical Groove Support Tickets as your primary references.

1. SOURCE PRIORITY & ACCURACY
- Always ground your answer in the provided REFERENCE CONTEXT. Never invent Sapaad features, menu paths, buttons, pricing, API-key steps, third-party providers, or policies.
- Prioritize Official Policies (if present) and Sapaad KB articles for step-by-step navigation paths and product rules.
- Use historical Groove tickets for real-world edge cases, troubleshooting workarounds, and Sapaad's standard support phrasing.
- Prioritize recent tickets (2025–2026) over older tickets (2023–2024). Always recommend AnyDesk (never TeamViewer) for remote support, and use the current Cloud Print Agent link (prints.sapaad.com/getcpa).
- NEVER invent or guess Sapaad menu paths, settings, or button names—not even for prerequisite steps (such as adding a surcharge). ONLY include a `>` menu path if that exact path appears in the REFERENCE CONTEXT below.
- Always keep Sapaad UI menu paths in their original English (e.g., **Setup > Location Setup > Self Ordering Systems > Sapaad Direct**) even when replying in Arabic, because the Sapaad interface is in English. Never mix Arabic and English characters inside the same word (e.g., never write "Sapaاد").
- If PREVIOUS REPLIES / ALREADY TRIED steps are provided, NEVER ask the customer to repeat troubleshooting steps they have already completed—acknowledge them and move straight to the next step or remote assistance.
- If ATTACHED CUSTOMER FILES analysis is provided (screenshots, PDFs, Excel, or CSV inspection), reference the exact findings (such as specific error messages, `.xlsx` vs `.csv` format issues, non-UTF-8 encoding, blank mandatory columns, or disallowed special characters) accurately in your reply.
- Obey AGENT INSTRUCTIONS strictly regarding how to handle the customer (e.g., whether to promise a callback/account check or guide them step-by-step), while keeping the tone warm and polite—never leak harsh internal phrasing to the customer. However, factual accuracy ALWAYS overrides self-serve instructions: if a detail is not in the REFERENCE CONTEXT, do NOT fabricate steps.
- If the context does not clearly confirm an answer or requires account-level checking, add a short `[Agent Note: ...]` line at the very top BEFORE the greeting explaining what the agent should verify internally, and keep the email body polite and accurate.

2. PRIVACY & PLACEHOLDER SAFETY
- Preserve any `[CUST_EMAIL_1]` or `[CUST_PHONE_1]` tokens from the customer's email verbatim if referencing their contact details (they are restored locally after generation).
- NEVER copy redacted dataset placeholders ([PHONE_OR_ID], [EMAIL]), other customers' names, restaurant names, passwords, or expired Zoom/WeTransfer links from past tickets into a new draft.
- Use clean bracketed placeholders like [Zoom Link] or [Date & Time] only if the agent hasn't provided them.

3. TONE, LANGUAGE & GROOVE FORMATTING
- Always write in the "we" tone (representing the Sapaad Support Team), polite, clear, and concise.
- Match the requested or customer's language (English or Arabic).
- For Arabic replies, consistently use respectful plural addressing throughout (e.g., قوموا، اختاروا، اضغطوا، حرّروا، تواصلوا) and avoid singular/plural mixing or spelling slips (use واختاروا, not واختروا).
- Greeting (Customer Emails Only):
  - English: "Dear [Customer Name]," or "Hi [Customer Name]," (use "Hi," if no name is provided).
  - Arabic: "السلام عليكم،"
- Follow the greeting with a brief, polite 1-sentence opening line before jumping into numbered points.
- Mandatory Closing (Customer Emails Only — include ONCE only, never add a second similar closing sentence above it):
  - English: End every customer email with exactly:
    Please feel free to write to us for any further assistance.
  - Arabic: End every customer email with exactly:
    لا تترددوا بالتواصل معنا لأي مساعدة إضافية.
- NO sign-off or signature: Never add "Best regards,", "Thanks,", or an agent name at the bottom (Groove appends the signature automatically).
- GROOVE-READY FORMATTING:
  - Use **bold** sparingly and cleanly for Sapaad menu paths, key settings, button names, and short numbered point titles.
  - Use clean numbered lists (1., 2., 3.) for multi-part inquiries or sequential steps, and simple hyphen bullets (- ) for sub-items.
  - Separate paragraphs and numbered sections with a single blank line.
  - NEVER use Markdown headers (###), code blocks, backticks (`), or tables.
"""

REFINE_SYSTEM_PROMPT = """You are a senior Sapaad Customer Support editor refining or transforming an agent's email draft for Groove.

RULES:
- Preserve EVERY fact, date, time, link, placeholder, Sapaad UI path, and commitment unless explicitly asked to shorten or convert to an internal note. Do NOT invent new Sapaad steps or promises.
- Significantly elevate clarity, grammar, flow, and polite Sapaad Support "we" tone.
- Keep Sapaad UI menu paths in their original English (e.g., **Setup > Location Setup**) even when translating to Arabic.
- For Arabic drafts, use natural, professional business Arabic with consistent respectful plural addressing (قوموا، اختاروا، تواصلوا).
- For customer emails, apply Sapaad greeting rules ("Hi [Name]," / "Dear [Name]," / "السلام عليكم،") and the exact mandatory closing line once ("Please feel free to write to us for any further assistance." or "لا تترددوا بالتواصل معنا لأي مساعدة إضافية."). Never add a sign-off or agent signature.
- Use clean **bold** for key paths/dates/actions and clean numbered or bulleted lists where helpful. Never use ### headers, backticks, or tables.
"""

TRANSLATE_AR_SYSTEM_PROMPT = """You are an elite English-to-Arabic SaaS & Restaurant POS localization specialist translating Sapaad Customer Support emails into flawless Modern Standard Business Arabic.

STRICT QUALITY RULES:
1. ZERO MISTAKES: Deliver natural, fluent, executive-grade Arabic with flawless grammar, spelling, punctuation, and sentence flow—never produce stiff or literal word-for-word translations.
2. RESPECTFUL PLURAL ADDRESSING: Always address the customer in the respectful Arabic plural throughout (e.g., يُرجى منكم، قوموا، اختاروا، اضغطوا، افتحوا، احفظوا، تواصلوا). NEVER mix singular (قم، اختر) and plural, and never write typos like "واختروا" or "قموا".
3. KEEP SAPAAD TERMS & UI PATHS IN ENGLISH: Keep all Sapaad product names, UI navigation paths, button labels, URLs, codes, and placeholders in their exact original English (e.g., **Sapaad Direct**, **Sapaad.Online**, **Cloud Print Agent**, **AnyDesk**, **Setup > Location Setup > Delivery Settings**). Never mix Arabic and English letters inside the same word.
4. 100% FIDELITY: Preserve every single fact, step, date, time, amount, URL, and `[CUST_EMAIL_1]` / `[CUST_PHONE_1]` token accurately. Never omit details and never add unstated promises.
5. GROOVE FORMATTING: Use **bold** cleanly for Sapaad menu paths, button names, and key headings, clean numbered lists (1., 2., 3.) or hyphen bullets (- ), and single blank lines between paragraphs. Never use ### headers, backticks, or tables. Never add a sign-off ("مع أطيب التحيات" or agent name).
6. Output ONLY the final Arabic translation—zero commentary or explanations.
"""

PROOFREAD_AR_SYSTEM_PROMPT = """You are a senior Arabic linguist and Sapaad Support QA editor performing a final zero-mistake proofread of an Arabic email translation against its English source.

Check and silently fix:
- Any grammar, gender agreement, Hamza (أ/إ/آ), Taa Marbuta (ة/ه), or spelling mistake.
- Any awkward literal phrasing (make it read like a native Arab support specialist wrote it).
- Any singular/plural inconsistency (enforce respectful plural: قوموا، اختاروا، اضغطوا، تواصلوا).
- Any accidentally translated Sapaad UI path or mixed-script word (keep Sapaad UI paths and product names in exact English).
- Any missing fact, link, date, or number compared to the source text.

Output ONLY the final perfected Arabic email text—no notes, no explanations, no Markdown headers or sign-offs.
"""


# ================= INDEX LOADING (DELEGATED TO retrieval.py) =================
@st.cache_resource(show_spinner="Indexing Sapaad KB & Support Tickets (Smart BM25+)...")
def load_index():
    return _build_retrieval_index(LOCAL_DATA_DIR, HF_DATASET_REPO, HF_TOKEN)


# ================= MULTI-PROVIDER LLM LAYER (GROQ -> GEMINI -> OPENROUTER) =================
def resolve_primary_model(selected_pair: tuple[str, str], text_hint: str, language: str) -> tuple[str, str]:
    prov, mod = selected_pair
    if prov != "auto":
        return prov, mod

    is_arabic = language == "Arabic" or (
        language == "Auto (Match Customer)" and bool(re.search(r"[\u0600-\u06FF]", text_hint))
    )
    if GROQ_API_KEY:
        return ("groq", "qwen/qwen3.8-27b") if is_arabic else ("groq", "openai/gpt-oss-120b")
    if GEMINI_API_KEY:
        return ("gemini", "gemini-2.5-flash")
    return ("openrouter", "openrouter/free")


def _call_groq_raw(model_id: str, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
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
    return (resp.choices[0].message.content or "").strip()


def _call_gemini_raw(model_id: str, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_id}:generateContent?key={GEMINI_API_KEY}"
    )
    payload: dict = {
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    if system_prompt:
        payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")[:300]
        raise RuntimeError(f"Gemini HTTP {e.code}: {err_body}") from e

    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {data}")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if "text" in p).strip()
    if not text:
        raise RuntimeError("Gemini returned empty text.")
    return text


def _call_openrouter_raw(model_id: str, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://sapaad-support-assistant.streamlit.app",
            "X-Title": "Sapaad Support AI Assistant",
            "User-Agent": "SapaadSupportAI/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=50) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")[:300]
        raise RuntimeError(f"OpenRouter HTTP {e.code}: {err_body}") from e

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenRouter returned no choices: {data}")
    text = ((choices[0].get("message") or {}).get("content") or "").strip()
    if not text:
        raise RuntimeError("OpenRouter returned empty text.")
    return text


def build_fallback_chain(primary_provider: str, primary_model: str) -> list[tuple[str, str]]:
    candidates = [
        (primary_provider, primary_model),
        ("groq", "openai/gpt-oss-120b"),
        ("groq", "qwen/qwen3.8-27b"),
        ("groq", "openai/gpt-oss-20b"),
        ("gemini", "gemini-2.5-flash"),
        ("gemini", "gemini-2.5-flash-lite"),
        ("openrouter", "openrouter/free"),
    ]
    chain = []
    seen = set()
    for prov, mod in candidates:
        if (prov, mod) in seen:
            continue
        if prov == "groq" and not GROQ_API_KEY:
            continue
        if prov == "gemini" and not GEMINI_API_KEY:
            continue
        if prov == "openrouter" and not OPENROUTER_API_KEY:
            continue
        seen.add((prov, mod))
        chain.append((prov, mod))
    return chain


def call_llm(
    primary_provider: str,
    primary_model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.25,
    max_tokens: int = 3000,
) -> tuple[str, str]:
    chain = build_fallback_chain(primary_provider, primary_model)
    if not chain:
        raise RuntimeError(
            "No API keys configured. Add GROQ_API_KEY, GEMINI_API_KEY, or OPENROUTER_API_KEY in Secrets."
        )

    errors = []
    for idx, (prov, mod) in enumerate(chain):
        try:
            if prov == "groq":
                out = _call_groq_raw(mod, system_prompt, user_prompt, temperature, max_tokens)
            elif prov == "gemini":
                out = _call_gemini_raw(mod, system_prompt, user_prompt, temperature, max_tokens)
            else:
                out = _call_openrouter_raw(mod, system_prompt, user_prompt, temperature, max_tokens)
            if not out:
                raise RuntimeError("Model returned empty output.")
            label = f"{prov}:{mod}" if idx == 0 else f"{prov}:{mod} (auto-fallback)"
            return out, label
        except Exception as e:
            errors.append(f"{prov}:{mod} -> {e}")
            continue

    raise RuntimeError("All configured LLM providers failed:\n" + "\n".join(errors))


# ================= MULTI-TOPIC ENGLISH QUERY EXTRACTOR =================
def extract_english_subqueries(
    text: str,
    draft_provider: str,
    draft_model: str,
) -> tuple[list[str], str]:
    if draft_provider == "groq":
        if draft_model == "openai/gpt-oss-20b":
            q_candidates = [
                ("groq", "qwen/qwen3.8-27b"),
                ("gemini", "gemini-2.5-flash-lite"),
                ("openrouter", "openrouter/free"),
            ]
        else:
            q_candidates = [
                ("groq", "openai/gpt-oss-20b"),
                ("gemini", "gemini-2.5-flash-lite"),
                ("openrouter", "openrouter/free"),
            ]
    else:
        q_candidates = [
            ("gemini", "gemini-2.5-flash-lite"),
            ("groq", "openai/gpt-oss-20b"),
            ("openrouter", "openrouter/free"),
        ]

    prompt = (
        "You are a search query generator for Sapaad POS support documentation.\n"
        "Read the customer email below (Arabic or English) and extract EVERY distinct support question or topic "
        "as its own short ENGLISH search query (3 to 7 English keywords per line). NEVER merge two different "
        "questions onto the same line.\n"
        "Use exact Sapaad terminology where relevant:\n"
        "- QR code for tables -> Sapaad Direct QR code setup generate table QR\n"
        "- Call waiter button -> Sapaad Direct waiter hailing button\n"
        "- Disable OTP / order without login -> Sapaad Direct guest checkout Checkout Mode disable OTP\n"
        "- WhatsApp / SMS OTP alternative -> Sapaad Direct enable Email OTP SMS\n"
        "- Cash on delivery / pay at counter -> Sapaad Direct Pay at Counter Cash on Delivery payment types\n"
        "- Scheduled / pre-orders -> Sapaad.Online accept scheduled orders\n"
        "- Delivery company API / external integration -> Sapaad Open API integration Sapaad Connect delivery partner\n"
        "- Delivery fees per area / zone -> Sapaad.Online delivery areas delivery fee surcharge\n"
        "Output ONLY the English query lines (max 10 lines). No Arabic, no numbering, no bullets, no explanations.\n\n"
        f"{text[:2600]}"
    )

    for prov, mod in q_candidates:
        if prov == "groq" and not GROQ_API_KEY:
            continue
        if prov == "gemini" and not GEMINI_API_KEY:
            continue
        if prov == "openrouter" and not OPENROUTER_API_KEY:
            continue
        try:
            if prov == "groq":
                raw = _call_groq_raw(mod, "", prompt, temperature=0.0, max_tokens=1600)
            elif prov == "gemini":
                raw = _call_gemini_raw(mod, "", prompt, temperature=0.0, max_tokens=500)
            else:
                raw = _call_openrouter_raw(mod, "", prompt, temperature=0.0, max_tokens=500)

            lines = []
            for ln in raw.splitlines():
                cleaned = re.sub(r"^[\d\.\-\•\*\s]+", "", ln).strip()
                if len(cleaned.split()) >= 2 and re.search(r"[a-zA-Z]{2,}", cleaned):
                    lines.append(cleaned)
            if lines:
                return lines[:10], f"{prov}:{mod}"
        except Exception:
            continue

    return [text], "Fallback (raw text)"


# ================= POST-PROCESSING & GROOVE HTML CONVERTER =================
def split_agent_note_and_clean(text: str, mode: str, language_choice: str, enforce_ar_greeting: bool = True) -> tuple[str, str]:
    text = text.strip()
    agent_notes = []

    def _collect_note(match):
        note_body = match.group(1).strip()
        if note_body:
            agent_notes.append(note_body)
        return ""

    text = re.sub(r"(?im)^\s*\[?Agent Note\s*:\s*([^\n\]]+)\]?\s*\n*", _collect_note, text)

    text = re.sub(r"(?m)^#{1,6}\s+", "", text)
    text = re.sub(r"```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"__(.*?)__", r"**\1**", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    text = re.sub(r"Sapa[\u0600-\u06FF]+", "Sapaad", text)
    text = text.replace("واختروا", "واختاروا").replace("قموا", "قوموا")
    text = text.strip()

    if enforce_ar_greeting and mode != "Internal Note / Summary":
        is_arabic_output = (
            language_choice == "Arabic"
            or (language_choice == "Auto (Match Customer)" and bool(re.search(r"[\u0600-\u06FF]", text)))
        )
        if is_arabic_output and not re.match(r"^(\*\*)?السلام عليكم", text):
            text = f"السلام عليكم،\n\n{text}"

    return "\n".join(agent_notes).strip(), text


def _format_inline_html(escaped_text: str) -> str:
    formatted = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped_text)

    def _linkify(match: re.Match) -> str:
        url = match.group(1)
        trailing = ""
        while url and url[-1] in ".,;:!?)":
            trailing = url[-1] + trailing
            url = url[:-1]
        return (
            f'<a href="{url}" dir="ltr" style="unicode-bidi: isolate; color: #2563eb; text-decoration: underline;" '
            f'target="_blank" rel="noopener noreferrer">{url}</a>{trailing}'
        )

    formatted = re.sub(r"(https?://[^\s<]+)", _linkify, formatted)
    return formatted


def markdown_to_groove_html(text: str) -> tuple[str, str, bool]:
    is_rtl = bool(re.search(r"[\u0600-\u06FF]", text))
    lines = text.splitlines()
    html_parts = []
    in_list = None

    def close_list():
        nonlocal in_list
        if in_list:
            html_parts.append(f"</{in_list}>")
            in_list = None

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            close_list()
            html_parts.append("<div><br></div>")
            continue

        ol_match = re.match(r"^(\d+)[\.\)]\s+(.+)$", stripped)
        ul_match = re.match(r"^[-•\*]\s+(.+)$", stripped)

        if ol_match:
            if in_list != "ol":
                close_list()
                html_parts.append("<ol>")
                in_list = "ol"
            item_html = _format_inline_html(html.escape(ol_match.group(2)))
            html_parts.append(f"<li>{item_html}</li>")
        elif ul_match:
            if in_list != "ul":
                close_list()
                html_parts.append("<ul>")
                in_list = "ul"
            item_html = _format_inline_html(html.escape(ul_match.group(1)))
            html_parts.append(f"<li>{item_html}</li>")
        else:
            close_list()
            line_html = _format_inline_html(html.escape(stripped))
            html_parts.append(f"<div>{line_html}</div>")

    close_list()

    dir_attr = ' dir="rtl"' if is_rtl else ""
    align_style = "text-align: right; " if is_rtl else ""
    groove_html = (
        f'<div{dir_attr} style="font-family: Arial, Helvetica, sans-serif; '
        f'font-size: 14px; line-height: 1.65; color: #1f2937; {align_style}">'
        + "".join(html_parts)
        + "</div>"
    )

    plain_text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    return groove_html, plain_text, is_rtl


def render_groove_preview_and_copy(draft_text: str):
    groove_html, plain_text, is_rtl = markdown_to_groove_html(draft_text)
    groove_html_js = json.dumps(groove_html)
    plain_text_js = json.dumps(plain_text)

    dir_css = "rtl" if is_rtl else "ltr"
    align_css = "right" if is_rtl else "left"
    line_count = max(10, draft_text.count("\n") + len(draft_text) // 75)
    box_height = min(560, max(260, line_count * 24 + 92))

    component_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8" />
      <style>
        * {{ box-sizing: border-box; }}
        body {{
          margin: 0;
          padding: 2px;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
          background: transparent;
        }}
        .toolbar {{
          display: flex;
          gap: 8px;
          align-items: center;
          margin-bottom: 10px;
          flex-wrap: wrap;
        }}
        .btn-primary {{
          background: #ff4b4b;
          color: #ffffff;
          border: none;
          border-radius: 8px;
          padding: 9px 16px;
          font-size: 13.5px;
          font-weight: 600;
          cursor: pointer;
          transition: background 0.15s ease, transform 0.1s ease;
        }}
        .btn-primary:hover {{ background: #e03e3e; }}
        .btn-primary:active {{ transform: scale(0.98); }}
        .btn-secondary {{
          background: #f3f4f6;
          color: #1f2937;
          border: 1px solid #d1d5db;
          border-radius: 8px;
          padding: 8.5px 14px;
          font-size: 13px;
          font-weight: 500;
          cursor: pointer;
          transition: background 0.15s ease;
        }}
        .btn-secondary:hover {{ background: #e5e7eb; }}
        .status {{
          font-size: 13px;
          color: #10b981;
          font-weight: 600;
        }}
        .preview-card {{
          background: #ffffff;
          color: #1f2937;
          border: 1px solid #d1d5db;
          border-radius: 12px;
          padding: 20px 22px;
          direction: {dir_css};
          text-align: {align_css};
          font-family: Arial, Helvetica, sans-serif;
          font-size: 14px;
          line-height: 1.7;
          max-height: {box_height - 65}px;
          overflow-y: auto;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        }}
        .preview-card::-webkit-scrollbar {{ width: 7px; }}
        .preview-card::-webkit-scrollbar-track {{ background: transparent; }}
        .preview-card::-webkit-scrollbar-thumb {{ background: #cbd5e1; border-radius: 999px; }}
        .preview-card::-webkit-scrollbar-thumb:hover {{ background: #94a3b8; }}
        .preview-card ol, .preview-card ul {{ margin: 6px 0 10px 0; padding-inline-start: 24px; }}
        .preview-card li {{ margin-bottom: 6px; }}
        .preview-card strong {{ color: #111827; }}
      </style>
    </head>
    <body>
      <div class="toolbar">
        <button class="btn-primary" id="copyRichBtn">📋 Copy Formatted for Groove</button>
        <button class="btn-secondary" id="copyPlainBtn">📄 Copy Plain Text</button>
        <span class="status" id="copyStatus"></span>
      </div>
      <div class="preview-card" id="previewBox">{groove_html}</div>

      <script>
        const GROOVE_HTML = {groove_html_js};
        const PLAIN_TEXT = {plain_text_js};
        const statusEl = document.getElementById("copyStatus");

        function showStatus(msg) {{
          statusEl.textContent = msg;
          setTimeout(() => {{ statusEl.textContent = ""; }}, 3000);
        }}

        document.getElementById("copyRichBtn").addEventListener("click", async () => {{
          try {{
            if (navigator.clipboard && window.ClipboardItem) {{
              const htmlBlob = new Blob([GROOVE_HTML], {{ type: "text/html" }});
              const textBlob = new Blob([PLAIN_TEXT], {{ type: "text/plain" }});
              await navigator.clipboard.write([
                new ClipboardItem({{ "text/html": htmlBlob, "text/plain": textBlob }})
              ]);
              showStatus("✅ Copied with Groove formatting!");
              return;
            }}
          }} catch (err) {{}}

          const range = document.createRange();
          const box = document.getElementById("previewBox");
          range.selectNodeContents(box);
          const sel = window.getSelection();
          sel.removeAllRanges();
          sel.addRange(range);
          document.execCommand("copy");
          sel.removeAllRanges();
          showStatus("✅ Copied with Groove formatting!");
        }});

        document.getElementById("copyPlainBtn").addEventListener("click", async () => {{
          try {{
            await navigator.clipboard.writeText(PLAIN_TEXT);
            showStatus("✅ Plain text copied!");
          }} catch (err) {{
            const ta = document.createElement("textarea");
            ta.value = PLAIN_TEXT;
            document.body.appendChild(ta);
            ta.select();
            document.execCommand("copy");
            document.body.removeChild(ta);
            showStatus("✅ Plain text copied!");
          }}
        }});
      </script>
    </body>
    </html>
    """
    components.html(component_html, height=box_height, scrolling=False)


# ================= ONE-CLICK POST-DRAFT ACTION HELPER =================
def apply_quick_action(action_key: str, selected_pair: tuple[str, str]):
    current_draft = st.session_state.get("last_output", "")
    if not current_draft:
        return

    is_currently_arabic = bool(re.search(r"[\u0600-\u06FF]", current_draft))
    target_mode = "Draft Reply"
    target_lang = "Auto (Match Customer)"
    sys_prompt = REFINE_SYSTEM_PROMPT

    if action_key == "shorten":
        instruction = (
            "Make this email significantly shorter, punchier, and easier to scan—keep only the essential steps, "
            "Sapaad UI paths, and links without losing factual accuracy."
        )
    elif action_key == "deescalate":
        instruction = (
            "Rewrite with warm empathy and calm de-escalation at the opening—acknowledge the customer's frustration "
            "sincerely while keeping every technical step and commitment accurate."
        )
    elif action_key == "anydesk":
        instruction = (
            "Keep the existing reply intact and add a polite paragraph offering remote assistance via **AnyDesk** "
            "(include download link [https://anydesk.com/en/downloads/windows](https://anydesk.com/en/downloads/windows) and ask for their AnyDesk ID and preferred time)."
        )
    elif action_key == "translate":
        if is_currently_arabic:
            target_lang = "English"
            instruction = (
                "Translate this Sapaad Support reply into polished, natural English while keeping all Sapaad UI paths, "
                "links, and facts identical."
            )
        else:
            target_lang = "Arabic"
            sys_prompt = TRANSLATE_AR_SYSTEM_PROMPT
            instruction = (
                "Translate this Sapaad Support reply into flawless, natural business Arabic using respectful plural "
                "addressing, keeping all Sapaad UI paths in English."
            )
    else:
        target_mode = "Internal Note / Summary"
        target_lang = "English"
        instruction = (
            "Convert this draft into a concise 3-5 bullet English internal Groove note for teammates summarizing "
            "what was explained/promised and any next action needed. Remove customer greeting and customer closing."
        )

    primary_provider, primary_model = resolve_primary_model(selected_pair, current_draft, target_lang)
    scrubbed_list, pii_map = scrub_pii([current_draft])
    prompt = f"TASK: {instruction}\n\nCURRENT DRAFT:\n{scrubbed_list[0]}"

    with st.spinner("Applying quick refinement..."):
        raw_out, used_model = call_llm(
            primary_provider=primary_provider,
            primary_model=primary_model,
            system_prompt=sys_prompt,
            user_prompt=prompt,
            temperature=0.22,
            max_tokens=2400,
        )
        _, cleaned = split_agent_note_and_clean(raw_out, target_mode, target_lang)
        audit_report = verify_and_repair_draft(
            draft_text=cleaned,
            context_block=st.session_state.get("last_context_block", ""),
            mode=target_mode,
            language_choice=target_lang,
            customer_email=st.session_state.get("last_customer_email", ""),
            agent_notes=st.session_state.get("last_agent_notes", ""),
        )
        st.session_state.last_output = restore_pii(audit_report["draft"], pii_map)
        st.session_state.last_audit = audit_report
        st.session_state.last_model = used_model


# ================= BOOKMARKLET HANDSHAKE (AUTO-CONNECT GROOVE) =================
st.session_state.setdefault("groove_runtime_bearer", "")
st.session_state.setdefault("pending_groove_ticket", "")

if "gtoken" in st.query_params or "ticket" in st.query_params:
    incoming_tok = str(st.query_params.get("gtoken", "")).strip()
    incoming_tkt = str(st.query_params.get("ticket", "")).strip()
    if incoming_tok:
        st.session_state.groove_runtime_bearer = incoming_tok
    if incoming_tkt:
        st.session_state.pending_groove_ticket = incoming_tkt
    st.query_params.clear()


# ================= PASSWORD GATE =================
if TEAM_PASSWORD:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.markdown(
            """
            <style>
            div[data-testid="stTextInput"] button[aria-label*="password"],
            div[data-testid="stTextInput"] button[title*="password"] {
                opacity: 1 !important;
                color: #4b5563 !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        _, center_col, _ = st.columns([1, 1.3, 1])
        with center_col:
            st.title("🔒 Sapaad Support AI Assistant")
            if st.session_state.get("pending_groove_ticket"):
                st.info(
                    f"🔗 Groove Ticket **#{st.session_state.pending_groove_ticket}** queued — "
                    "log in to load it automatically."
                )
            with st.form("login_form", clear_on_submit=False):
                st.text_input("Team Account", value="sapaad-support-team", autocomplete="username")
                pwd = st.text_input(
                    "Enter Team Password",
                    type="password",
                    autocomplete="current-password",
                    placeholder="Type team password and press Enter...",
                )
                submitted = st.form_submit_button("Login", type="primary", use_container_width=True)
                if submitted:
                    if pwd == TEAM_PASSWORD:
                        st.session_state.authenticated = True
                        st.rerun()
                    else:
                        st.error("Incorrect password.")
        st.stop()


# ================= SESSION STATE INIT =================
st.session_state.setdefault("customer_email_box", "")
st.session_state.setdefault("prev_replies_box", "")
st.session_state.setdefault("loaded_ticket_banner", "")
st.session_state.setdefault("attachment_cache", {})

if not GROQ_API_KEY and not GEMINI_API_KEY and not OPENROUTER_API_KEY:
    st.error(
        "Missing LLM API keys. Add at least one (`GROQ_API_KEY`, `GEMINI_API_KEY`, or `OPENROUTER_API_KEY`) "
        "in Streamlit Cloud Settings → Secrets."
    )
    st.stop()

index_bundle = load_index()
stats = index_bundle.get("stats", {})

available_models = {
    label: pair
    for label, pair in MODELS.items()
    if pair[0] == "auto"
    or (pair[0] == "groq" and GROQ_API_KEY)
    or (pair[0] == "gemini" and GEMINI_API_KEY)
    or (pair[0] == "openrouter" and OPENROUTER_API_KEY)
}

# ================= SIDEBAR =================
with st.sidebar:
    st.title("⚙️ Sapaad Support AI")
    mode = st.radio(
        "Mode",
        [
            "Draft Reply",
            "🌍 Arabic Translator",
            "Refine Draft",
            "FAQ / Template",
            "Internal Note / Summary",
        ],
        index=0,
    )
    language = (
        "Arabic"
        if mode == "🌍 Arabic Translator"
        else st.selectbox("Output Language", ["Auto (Match Customer)", "English", "Arabic"])
    )
    model_label = st.selectbox("AI Model", list(available_models.keys()), index=0)
    if mode in ("Draft Reply", "Refine Draft", "FAQ / Template"):
        top_k = st.slider("Reference Sources to Pull", min_value=3, max_value=8, value=5)
    else:
        top_k = 5

    st.divider()
    with st.expander("🔗 Groove Connection (Ticket Fetch)"):
        if GROOVE_API_TOKEN:
            st.success("Groove REST API Token active (permanent).")
        elif st.session_state.groove_runtime_bearer:
            st.success("Connected via Groove 1-Click Bookmarklet.")
        elif GROOVE_BEARER_TOKEN:
            st.success("Groove GraphQL Bearer Token active.")
        else:
            st.info("Click your **✨ Sapaad AI** bookmark while on Groove to connect automatically.")
        st.text_input(
            "Manual Session Bearer Token (Optional)",
            key="groove_runtime_bearer",
            type="password",
            placeholder="Auto-filled by bookmarklet...",
            help="Filled automatically by the Groove bookmarklet, or paste manually if needed.",
        )

    if index_bundle.get("docs"):
        policy_note = f" + **{stats.get('policies', 0)}** policies" if stats.get("policies") else ""
        st.success(
            f"Indexed **{stats.get('kb', 0):,}** KB chunks{policy_note} & "
            f"**{stats.get('tickets', 0):,}** resolved tickets "
            f"*(skipped {stats.get('skipped_tickets', 0):,} empty/unresolved)*"
        )
        if index_bundle.get("hybrid_ready") and CF_ACCOUNT_ID and CF_API_TOKEN:
            st.caption("Search Mode: **Hybrid (Smart BM25+ & bge-m3)**")
        else:
            st.caption("Search Mode: **Smart BM25+ (Zero-RAM)**")
    else:
        st.warning("No `.md` files found in `./data` or `HF_DATASET_REPO`.")

    active_provs = []
    if GROQ_API_KEY:
        active_provs.append("Groq")
    if GEMINI_API_KEY:
        active_provs.append("Gemini")
    if OPENROUTER_API_KEY:
        active_provs.append("OpenRouter")
    st.caption(f"Active Providers: **{' → '.join(active_provs)}**")
    st.caption("Privacy Shield: **Client-Side PII Scrubber ON**")

header_icon = "" if mode.startswith("🌍") else "📩 "
st.header(f"{header_icon}{mode}")

col1, col2 = st.columns([0.95, 1.05], gap="large")

with col1:
    verify_kb_in_refine = False
    uploaded_files = []
    proofread_ar = True
    include_groove_envelope = True
    previous_replies = ""
    agent_notes = ""
    customer_email = ""

    # -------- GROOVE TICKET FETCHER BAR (Draft Reply & Internal Note) --------
    if mode in ("Draft Reply", "Internal Note / Summary"):
        g1, g2, g3 = st.columns([1.8, 1.15, 1.0], gap="small")
        ticket_num_input = g1.text_input(
            "Groove Ticket #",
            placeholder="Ticket # (e.g., 114201)",
            label_visibility="collapsed",
        )
        fetch_clicked = g2.button("📥 Fetch", use_container_width=True, help="Fetch full ticket thread from Groove")
        clear_clicked = g3.button("🧹 Clear", use_container_width=True, help="Clear loaded ticket fields")
        
        if clear_clicked:
            st.session_state.customer_email_box = ""
            st.session_state.prev_replies_box = ""
            st.session_state.loaded_ticket_banner = ""
            st.rerun()

        if fetch_clicked:
            active_bearer = st.session_state.groove_runtime_bearer.strip() or GROOVE_BEARER_TOKEN
            with st.spinner(f"Fetching Ticket #{ticket_num_input} from Groove..."):
                try:
                    t_data = fetch_groove_ticket(
                        ticket_input=ticket_num_input,
                        groove_api_token=GROOVE_API_TOKEN,
                        groove_bearer_token=active_bearer,
                    )
                    if mode == "Internal Note / Summary":
                        st.session_state.customer_email_box = t_data["full_thread"]
                        st.session_state.prev_replies_box = ""
                    else:
                        st.session_state.customer_email_box = t_data["latest_customer_email"]
                        st.session_state.prev_replies_box = t_data["previous_thread"]

                    subj_tag = f" — *{t_data['subject']}*" if t_data.get("subject") else ""
                    st.session_state.loaded_ticket_banner = (
                        f"Loaded **Ticket #{t_data['ticket_number']}**{subj_tag} "
                        f"({t_data['message_count']} message(s) pulled from Groove)"
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Groove Fetch Error: {e}")

        # Auto-fetch ticket if opened via the Groove 1-Click Bookmarklet
        if st.session_state.get("pending_groove_ticket"):
            auto_tkt = st.session_state.pop("pending_groove_ticket")
            active_bearer = st.session_state.groove_runtime_bearer.strip() or GROOVE_BEARER_TOKEN
            with st.spinner(f"Auto-loading Ticket #{auto_tkt} from Groove..."):
                try:
                    t_data = fetch_groove_ticket(
                        ticket_input=auto_tkt,
                        groove_api_token=GROOVE_API_TOKEN,
                        groove_bearer_token=active_bearer,
                    )
                    if mode == "Internal Note / Summary":
                        st.session_state.customer_email_box = t_data["full_thread"]
                        st.session_state.prev_replies_box = ""
                    else:
                        st.session_state.customer_email_box = t_data["latest_customer_email"]
                        st.session_state.prev_replies_box = t_data["previous_thread"]

                    subj_tag = f" — *{t_data['subject']}*" if t_data.get("subject") else ""
                    st.session_state.loaded_ticket_banner = (
                        f"Auto-loaded **Ticket #{t_data['ticket_number']}**{subj_tag} "
                        f"({t_data['message_count']} message(s) pulled from Groove)"
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Groove Auto-Fetch Error: {e}")

        if st.session_state.loaded_ticket_banner:
            st.info(st.session_state.loaded_ticket_banner)

    # -------- MODE-SPECIFIC INPUTS --------
    if mode == "Draft Reply":
        customer_email = st.text_area(
            "Paste Customer Email (or fetch by Ticket # above)",
            key="customer_email_box",
            height=210,
            placeholder="Paste the full customer email from Groove here, or enter a Ticket # above...",
        )
        uploaded_files = st.file_uploader(
            "📎 Attach Customer Media / Files (Optional: Photo, PDF, Excel, CSV)",
            type=["png", "jpg", "jpeg", "webp", "pdf", "xlsx", "xls", "csv"],
            accept_multiple_files=True,
            help="Screenshots are read via Gemini Vision; Excel/CSV files are inspected locally for Sapaad upload errors.",
        )
        with st.expander(
            "💬 Previous Replies / Already Tried (Optional)",
            expanded=bool(st.session_state.prev_replies_box.strip()),
        ):
            previous_replies = st.text_area(
                "Earlier thread messages or troubleshooting steps already tried",
                key="prev_replies_box",
                height=115,
                placeholder="Auto-filled when fetching a Groove ticket, or paste earlier replies here...",
            )
        agent_notes = st.text_area(
            "Agent Instructions / Key Points (Optional)",
            height=95,
            placeholder="e.g., Ask if they are using Android or iOS / Tell them we tried calling and need AnyDesk ID...",
        )
        btn_label = "✨ Generate Reply"

    elif mode == "🌍 Arabic Translator":
        customer_email = st.text_area(
            "Write or Paste Email to Translate to Arabic",
            height=260,
            placeholder="Paste an English email (or rough draft) here to translate into flawless Sapaad Arabic in one click...",
        )
        t_opt1, t_opt2 = st.columns(2)
        proofread_ar = t_opt1.checkbox(
            "✨ 0-Mistake Proofread Pass",
            value=True,
            help="Runs a 2nd Arabic linguistic QA pass to eliminate grammar, plural, or phrasing slips.",
        )
        include_groove_envelope = t_opt2.checkbox(
            "Include Groove Greeting & Closing",
            value=True,
            help="Ensures السلام عليكم، and لا تترددوا بالتواصل معنا لأي مساعدة إضافية.",
        )
        agent_notes = st.text_area(
            "Optional Tone / Extra Note",
            height=75,
            placeholder="e.g., Keep it extra warm and apologetic / Keep the product name in English...",
        )
        btn_label = "🌍 Translate to Arabic"

    elif mode == "Refine Draft":
        agent_notes = st.text_area(
            "Your Rough Draft to Refine",
            height=210,
            placeholder="Paste your rough draft here...",
        )
        customer_email = st.text_area(
            "Customer Email (Optional context)",
            height=130,
            placeholder="Optional: paste the customer's email so the AI checks tone & context...",
        )
        verify_kb_in_refine = st.checkbox(
            "Verify technical Sapaad steps against KB & Tickets",
            value=False,
        )
        btn_label = "✨ Refine Draft"

    elif mode == "FAQ / Template":
        customer_email = st.text_area(
            "Topic / Scenario for Reusable Template",
            height=210,
            placeholder="e.g., How to install Cloud Print Agent and find the API key...",
        )
        agent_notes = st.text_area("Extra Requirements (Optional)", height=100)
        btn_label = "✨ Generate Template"

    else:
        customer_email = st.text_area(
            "Paste Ticket Thread or Fetch by Ticket # Above",
            key="customer_email_box",
            height=250,
            placeholder="Paste the email/thread or fetch a Groove Ticket # above to summarize into an English internal note...",
        )
        uploaded_files = st.file_uploader(
            "📎 Attach Files to Include in Summary (Optional)",
            type=["png", "jpg", "jpeg", "webp", "pdf", "xlsx", "xls", "csv"],
            accept_multiple_files=True,
        )
        agent_notes = st.text_area("Focus / Specific Question (Optional)", height=80)
        btn_label = "✨ Generate Internal Note"

    generate_btn = st.button(btn_label, type="primary", use_container_width=True)

with col2:
    selected_model_pair = available_models[model_label]

    if generate_btn:
        attachment_summary, attachment_labels = ("", [])
        if uploaded_files:
            with st.spinner("Analyzing attached media / files..."):
                attachment_summary, attachment_labels = analyze_uploaded_files(
                    uploaded_files, GEMINI_API_KEY, st.session_state.attachment_cache
                )

        if mode == "Refine Draft":
            raw_search_source = f"{agent_notes}\n{customer_email}".strip()
        else:
            raw_search_source = "\n".join(
                p for p in [customer_email.strip(), attachment_summary.strip(), agent_notes.strip()] if p
            ).strip()

        if not raw_search_source:
            st.warning("Please enter an email, ticket number, or attach a file first.")
        else:
            primary_provider, primary_model = resolve_primary_model(
                selected_model_pair, raw_search_source, language
            )
            [safe_email, safe_notes, safe_prev, safe_attach, safe_search], pii_map = scrub_pii(
                [customer_email, agent_notes, previous_replies, attachment_summary, raw_search_source]
            )

            # -------- DEDICATED 1-CLICK ARABIC TRANSLATOR PIPELINE --------
            if mode == "🌍 Arabic Translator":
                if selected_model_pair[0] == "auto":
                    if GROQ_API_KEY:
                        primary_provider, primary_model = ("groq", "qwen/qwen3.8-27b")
                    elif GEMINI_API_KEY:
                        primary_provider, primary_model = ("gemini", "gemini-2.5-flash")

                envelope_rule = (
                    "Include the Sapaad Arabic greeting ('السلام عليكم،') and the exact single closing line "
                    "('لا تترددوا بالتواصل معنا لأي مساعدة إضافية.') with NO sign-off."
                    if include_groove_envelope
                    else "Translate only the provided text cleanly without adding extra greetings or closings."
                )
                trans_prompt = (
                    f"TASK: Translate the email/text below into flawless, natural Sapaad Business Arabic.\n"
                    f"ENVELOPE RULE: {envelope_rule}\n"
                    f"EXTRA NOTE: {safe_notes or 'None'}\n\n"
                    f"SOURCE TEXT:\n{safe_email}"
                )
                with st.spinner("Translating to professional Arabic..."):
                    try:
                        ar_draft, used_model = call_llm(
                            primary_provider=primary_provider,
                            primary_model=primary_model,
                            system_prompt=TRANSLATE_AR_SYSTEM_PROMPT,
                            user_prompt=trans_prompt,
                            temperature=0.15,
                            max_tokens=2800,
                        )
                        if proofread_ar:
                            qa_prompt = (
                                f"SOURCE TEXT:\n{safe_email}\n\n"
                                f"ARABIC DRAFT TO PROOFREAD:\n{ar_draft}\n\n"
                                f"ENVELOPE RULE: {envelope_rule}"
                            )
                            ar_draft, qa_model = call_llm(
                                primary_provider=primary_provider,
                                primary_model=primary_model,
                                system_prompt=PROOFREAD_AR_SYSTEM_PROMPT,
                                user_prompt=qa_prompt,
                                temperature=0.10,
                                max_tokens=2800,
                            )
                            used_model = f"{used_model} + QA({qa_model})"

                        verify_mode = "Draft Reply" if include_groove_envelope else "Internal Note / Summary"
                        _, clean_draft = split_agent_note_and_clean(
                            ar_draft, verify_mode, "Arabic", enforce_ar_greeting=include_groove_envelope
                        )
                        audit_report = verify_and_repair_draft(
                            draft_text=clean_draft,
                            context_block="",
                            mode=verify_mode,
                            language_choice="Arabic",
                            customer_email=safe_email,
                            agent_notes=safe_notes,
                        )
                        st.session_state.last_output = restore_pii(audit_report["draft"], pii_map)
                        st.session_state.last_agent_note = ""
                        st.session_state.last_audit = audit_report
                        st.session_state.last_pii_count = len(pii_map)
                        st.session_state.last_model = used_model
                        st.session_state.last_matches = []
                        st.session_state.last_search_meta = {}
                        st.session_state.last_attachment_labels = []
                        st.session_state.last_attachment_summary = ""
                        st.session_state.last_context_block = ""
                    except Exception as e:
                        st.error(f"Translation Error: {e}")

            # -------- STANDARD SUPPORT / RAG MODES --------
            else:
                should_search = (
                    mode in ("Draft Reply", "FAQ / Template")
                    or (mode == "Refine Draft" and verify_kb_in_refine)
                )

                with st.spinner("Searching Sapaad KB & Tickets and generating..."):
                    if should_search:
                        matches, search_meta = _smart_retrieve_context(
                            index_bundle=index_bundle,
                            query=safe_search,
                            extract_subqueries_fn=lambda txt: extract_english_subqueries(
                                txt, primary_provider, primary_model
                            ),
                            top_k=top_k,
                            cf_account_id=CF_ACCOUNT_ID,
                            cf_api_token=CF_API_TOKEN,
                        )
                    else:
                        matches, search_meta = [], {}

                    context_block = build_budgeted_context(matches, safe_email, safe_notes, language)

                    lang_rule = (
                        "Match the customer's language (English or Arabic)."
                        if language == "Auto (Match Customer)"
                        else f"Write the output in {language}."
                    )

                    active_system_prompt = SYSTEM_PROMPT
                    temp = 0.22
                    attach_section = (
                        f"ATTACHED CUSTOMER FILES ANALYSIS (screenshots/PDF/Excel/CSV):\n{safe_attach}\n\n"
                        if safe_attach.strip()
                        else ""
                    )

                    if mode == "Draft Reply":
                        prev_section = (
                            f"PREVIOUS REPLIES / ALREADY TRIED (do NOT repeat these steps):\n{safe_prev}\n\n"
                            if safe_prev.strip()
                            else ""
                        )
                        prompt = (
                            f"TASK: Draft a ready-to-send Sapaad Support email reply.\n"
                            f"LANGUAGE: {lang_rule}\n\n"
                            f"CUSTOMER EMAIL:\n{safe_email or 'See attached file(s) below.'}\n\n"
                            f"{attach_section}"
                            f"{prev_section}"
                            f"AGENT INSTRUCTIONS (follow strictly if provided, without exposing harsh internal phrasing):\n"
                            f"{safe_notes or 'None'}\n\n"
                            f"REFERENCE CONTEXT (Sapaad Policies, KB & Past Tickets):\n"
                            f"{context_block or 'No matching context found.'}"
                        )
                    elif mode == "Refine Draft":
                        active_system_prompt = REFINE_SYSTEM_PROMPT
                        temp = 0.35
                        prompt = (
                            f"TASK: Refine and polish the agent's rough draft below.\n"
                            f"LANGUAGE: {lang_rule}\n\n"
                            f"AGENT ROUGH DRAFT:\n{safe_notes}\n\n"
                            f"CUSTOMER EMAIL (Optional context):\n{safe_email or 'Not provided'}\n\n"
                            f"REFERENCE CONTEXT:\n{context_block or 'None'}"
                        )
                    elif mode == "FAQ / Template":
                        prompt = (
                            f"TASK: Create a concise, reusable Sapaad Support email template with clean placeholders like [Customer Name].\n"
                            f"LANGUAGE: {lang_rule}\n\n"
                            f"TOPIC:\n{safe_email}\n"
                            f"EXTRA NOTES:\n{safe_notes or 'None'}\n\n"
                            f"REFERENCE CONTEXT:\n{context_block or 'None'}"
                        )
                    else:
                        prompt = (
                            f"TASK: Write a concise English internal Groove note summarizing this ticket for teammates (translate if needed). "
                            f"Do NOT include a customer greeting or customer closing line.\n\n"
                            f"TICKET CONTENT:\n{safe_email}\n\n"
                            f"{attach_section}"
                            f"FOCUS:\n{safe_notes or 'Key issue, account details, what was tried, and next action needed.'}"
                        )

                    try:
                        raw_output, used_model = call_llm(
                            primary_provider=primary_provider,
                            primary_model=primary_model,
                            system_prompt=active_system_prompt,
                            user_prompt=prompt,
                            temperature=temp,
                            max_tokens=3000,
                        )
                        agent_note, clean_draft = split_agent_note_and_clean(raw_output, mode, language)
                        audit_report = verify_and_repair_draft(
                            draft_text=clean_draft,
                            context_block=context_block,
                            mode=mode,
                            language_choice=language,
                            customer_email=safe_email,
                            agent_notes=safe_notes,
                        )
                        final_draft = restore_pii(audit_report["draft"], pii_map)
                        final_agent_note = restore_pii(agent_note, pii_map)

                        st.session_state.last_output = final_draft
                        st.session_state.last_agent_note = final_agent_note
                        st.session_state.last_audit = audit_report
                        st.session_state.last_pii_count = len(pii_map)
                        st.session_state.last_model = used_model
                        st.session_state.last_matches = matches
                        st.session_state.last_search_meta = search_meta
                        st.session_state.last_attachment_labels = attachment_labels
                        st.session_state.last_attachment_summary = attachment_summary
                        st.session_state.last_context_block = context_block
                        st.session_state.last_customer_email = customer_email
                        st.session_state.last_agent_notes = agent_notes
                    except Exception as e:
                        st.error(f"API Error: {e}")

    if "last_output" in st.session_state:
        st.subheader("Ready for Groove")

        if st.session_state.get("last_agent_note"):
            st.warning(f"**Internal Agent Note (not copied into email):** {st.session_state.last_agent_note}")

        audit = st.session_state.get("last_audit", {})
        if audit.get("errors"):
            for err in audit["errors"]:
                st.error(f"🚨 **Audit Alert:** {err}")
        elif audit:
            pii_count = st.session_state.get("last_pii_count", 0)
            pii_tag = f" • 🔒 {pii_count} PII item(s) masked" if pii_count else ""
            st.success(f"✅ **Audit Passed:** Menu paths verified • Zero leaks • Groove format valid{pii_tag}")

        if audit.get("auto_fixes"):
            st.caption("🛠️ **Auto-repaired:** " + " • ".join(audit["auto_fixes"]))

        if st.session_state.get("last_attachment_labels"):
            st.caption("📎 **Attachments Analyzed:** " + " • ".join(st.session_state.last_attachment_labels))

        render_groove_preview_and_copy(st.session_state.last_output)

        st.markdown('<div class="quick-actions-label">⚡ Quick Refinements</div>', unsafe_allow_html=True)
        r1c1, r1c2, r1c3 = st.columns(3, gap="small")
        if r1c1.button("✂️ Shorten", use_container_width=True, help="Make the reply shorter and easier to scan"):
            apply_quick_action("shorten", selected_model_pair)
            st.rerun()
        if r1c2.button("🤝 De-escalate", use_container_width=True, help="Add calm empathy for an upset customer"):
            apply_quick_action("deescalate", selected_model_pair)
            st.rerun()
        if r1c3.button("🖥️ + AnyDesk", use_container_width=True, help="Append a polite AnyDesk remote support request"):
            apply_quick_action("anydesk", selected_model_pair)
            st.rerun()

        r2c1, r2c2 = st.columns(2, gap="small")
        if r2c1.button("🌐 Switch EN ↔ AR", use_container_width=True, help="Translate draft between English and Arabic"):
            apply_quick_action("translate", selected_model_pair)
            st.rerun()
        if r2c2.button("📝 Convert to Internal Note", use_container_width=True, help="Summarize into an English Groove internal note"):
            apply_quick_action("internal_note", selected_model_pair)
            st.rerun()

        meta = st.session_state.get("last_search_meta", {})
        engine_caption = f" | Search: `{meta.get('engine')}` | Grounding: `{meta.get('confidence')}`" if meta else ""
        st.caption(f"Model used: `{st.session_state.last_model}`{engine_caption}")

        if st.session_state.get("last_attachment_summary"):
            with st.expander("📎 View Extracted Attachment Details"):
                st.text(st.session_state.last_attachment_summary)

        if st.session_state.get("last_matches"):
            with st.expander(f"📚 Referenced Sources ({len(st.session_state.last_matches)})"):
                if meta:
                    cf_err_note = f"\n\n⚠️ **Cloudflare Hybrid Note:** `{meta['cf_error']}`" if meta.get("cf_error") else ""
                    st.info(
                        f"**Engine:** {meta.get('engine')} | "
                        f"**Grounding Confidence:** {meta.get('confidence')} "
                        f"(BM25: {meta.get('top_bm25', 0)}, Cosine: {meta.get('top_cosine', 0)})\n\n"
                        f"**Queries:** {' | '.join(meta.get('queries', []))}{cf_err_note}"
                    )
                for m in st.session_state.last_matches:
                    matched_tag = f" — *matched: {m['matched_query']}*" if m.get("matched_query") else ""
                    st.markdown(f"**[{m['type']}] {m['title']}** *({m['source']})*{matched_tag}")
                    st.caption(m["text"][:450] + ("..." if len(m["text"]) > 450 else ""))
                    st.divider()

