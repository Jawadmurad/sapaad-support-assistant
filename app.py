import html
import json
import os
import re
import urllib.error
import urllib.request

import streamlit as st
import streamlit.components.v1 as components
from groq import Groq

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


def get_secret(name: str, default: str = "") -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default)


GROQ_API_KEY = get_secret("GROQ_API_KEY")
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
HF_TOKEN = get_secret("HF_TOKEN")
HF_DATASET_REPO = get_secret("HF_DATASET_REPO")
TEAM_PASSWORD = get_secret("TEAM_PASSWORD")
CF_ACCOUNT_ID = get_secret("CF_ACCOUNT_ID")
CF_API_TOKEN = get_secret("CF_API_TOKEN")
LOCAL_DATA_DIR = "data"

MAX_CONTEXT_CHARS = 10800

MODELS = {
    "Groq: GPT-OSS 120B (Best Quality)": ("groq", "openai/gpt-oss-120b"),
    "Groq: Qwen 3.8 27B (Strong Arabic/EN)": ("groq", "qwen/qwen3.8-27b"),
    "Groq: GPT-OSS 20B (Ultra-Fast)": ("groq", "openai/gpt-oss-20b"),
    "Gemini: 2.5 Flash (High Quality)": ("gemini", "gemini-2.5-flash"),
    "Gemini: 2.5 Flash-Lite (Fast Fallback)": ("gemini", "gemini-2.5-flash-lite"),
}

SYSTEM_PROMPT = """You are a Sapaad Customer Support email assistant. Your job is to draft customer replies, refine rough drafts, and write internal Groove ticket notes using the provided Sapaad Knowledge Base (KB) and historical Groove Support Tickets as your primary references.

1. SOURCE PRIORITY & ACCURACY
- Always ground your answer in the provided REFERENCE CONTEXT. Never invent Sapaad features, menu paths, buttons, pricing, API-key steps, third-party providers, or policies.
- Prioritize Official Policies (if present) and Sapaad KB articles for step-by-step navigation paths and product rules.
- Use historical Groove tickets for real-world edge cases, troubleshooting workarounds, and Sapaad's standard support phrasing.
- Prioritize recent tickets (2025–2026) over older tickets (2023–2024). Always recommend AnyDesk (never TeamViewer) for remote support, and use the current Cloud Print Agent link (prints.sapaad.com/getcpa).
- NEVER invent or guess Sapaad menu paths, settings, or button names—not even for prerequisite steps (such as adding a surcharge). ONLY include a `>` menu path if that exact path appears in the REFERENCE CONTEXT below.
- Always keep Sapaad UI menu paths in their original English (e.g., **Setup > Location Setup > Self Ordering Systems > Sapaad Direct**) even when replying in Arabic, because the Sapaad interface is in English. Never mix Arabic and English characters inside the same word (e.g., never write "Sapaاد").
- Obey AGENT INSTRUCTIONS strictly regarding how to handle the customer (e.g., whether to promise a callback/account check or guide them step-by-step), while keeping the tone warm and polite—never leak harsh internal phrasing to the customer. However, factual accuracy ALWAYS overrides self-serve instructions: if a detail is not in the REFERENCE CONTEXT, do NOT fabricate steps.
- If the context does not clearly confirm an answer or requires account-level checking, add a short `[Agent Note: ...]` line at the very top BEFORE the greeting explaining what the agent should verify internally, and keep the email body polite and accurate.

2. PRIVACY & PLACEHOLDER SAFETY
- NEVER copy redacted placeholders ([PHONE_OR_ID], [EMAIL]), other customers' names, restaurant names, passwords, or expired Zoom/WeTransfer links from past tickets into a new draft.
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

REFINE_SYSTEM_PROMPT = """You are a senior Sapaad Customer Support editor refining an agent's rough email draft for Groove.

RULES:
- Preserve EVERY fact, date, time, link, placeholder, thank-you line, and commitment from the agent's rough draft. Do NOT drop details or add new promises.
- Significantly elevate clarity, grammar, flow, and polite Sapaad Support "we" tone.
- Match the language of the rough draft (or the requested output language).
- For Arabic drafts, use natural, professional business Arabic with consistent respectful plural addressing (قوموا، اختاروا، تواصلوا).
- Apply Sapaad greeting rules ("Hi [Name]," / "Dear [Name]," / "السلام عليكم،") and the exact mandatory closing line once ("Please feel free to write to us for any further assistance." or "لا تترددوا بالتواصل معنا لأي مساعدة إضافية."). Never add a sign-off or agent signature.
- Use clean **bold** for key paths/dates/actions and clean numbered or bulleted lists where helpful. Never use ### headers, backticks, or tables.
- If REFERENCE CONTEXT is provided, verify Sapaad UI menu paths against it and keep UI paths in English.
"""


# ================= INDEX LOADING (DELEGATED TO retrieval.py) =================
@st.cache_resource(show_spinner="Indexing Sapaad KB & Support Tickets (Smart BM25+)...")
def load_index():
    return _build_retrieval_index(LOCAL_DATA_DIR, HF_DATASET_REPO, HF_TOKEN)


# ================= MULTI-PROVIDER LLM LAYER (GROQ -> GEMINI) =================
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


def build_fallback_chain(primary_provider: str, primary_model: str) -> list[tuple[str, str]]:
    candidates = [
        (primary_provider, primary_model),
        ("groq", "openai/gpt-oss-20b"),
        ("groq", "qwen/qwen3.8-27b"),
        ("gemini", "gemini-2.5-flash"),
        ("gemini", "gemini-2.5-flash-lite"),
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
        raise RuntimeError("No API keys configured. Add GROQ_API_KEY and/or GEMINI_API_KEY in Secrets.")

    errors = []
    for idx, (prov, mod) in enumerate(chain):
        try:
            if prov == "groq":
                out = _call_groq_raw(mod, system_prompt, user_prompt, temperature, max_tokens)
            else:
                out = _call_gemini_raw(mod, system_prompt, user_prompt, temperature, max_tokens)
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
    """Translate and split Arabic or multi-point emails into focused English Sapaad search queries."""
    if draft_provider == "groq":
        if draft_model == "openai/gpt-oss-20b":
            q_candidates = [("groq", "qwen/qwen3.8-27b"), ("gemini", "gemini-2.5-flash-lite"), ("gemini", "gemini-2.5-flash")]
        else:
            q_candidates = [("groq", "openai/gpt-oss-20b"), ("gemini", "gemini-2.5-flash-lite"), ("gemini", "gemini-2.5-flash")]
    else:
        q_candidates = [("gemini", "gemini-2.5-flash-lite"), ("groq", "openai/gpt-oss-20b"), ("gemini", "gemini-2.5-flash")]

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
        try:
            if prov == "groq":
                raw = _call_groq_raw(mod, "", prompt, temperature=0.0, max_tokens=1600)
            else:
                raw = _call_gemini_raw(mod, "", prompt, temperature=0.0, max_tokens=500)

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
def split_agent_note_and_clean(text: str, mode: str, language_choice: str, customer_email: str) -> tuple[str, str]:
    """Separate [Agent Note: ...] from the email body and apply Groove formatting safety guards."""
    text = text.strip()
    agent_notes = []

    def _collect_note(match):
        note_body = match.group(1).strip()
        if note_body:
            agent_notes.append(note_body)
        return ""

    text = re.sub(r"(?im)^\s*\[?Agent Note\s*:\s*([^\n\]]+)\]?\s*\n*", _collect_note, text)

    # Strip unwanted Markdown syntax while preserving **bold** and lists for Groove TinyMCE
    text = re.sub(r"(?m)^#{1,6}\s+", "", text)
    text = re.sub(r"```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"__(.*?)__", r"**\1**", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Catch rare mixed-script & Arabic spelling slips
    text = re.sub(r"Sapa[\u0600-\u06FF]+", "Sapaad", text)
    text = text.replace("واختروا", "واختاروا").replace("قموا", "قوموا")
    text = text.strip()

    # Guarantee Arabic greeting on customer-facing Arabic drafts
    if mode != "Internal Note / Summary":
        is_arabic_output = (
            language_choice == "Arabic"
            or (language_choice == "Auto (Match Customer)" and bool(re.search(r"[\u0600-\u06FF]", text)))
        )
        if is_arabic_output and not re.match(r"^(\*\*)?السلام عليكم", text):
            text = f"السلام عليكم،\n\n{text}"

    return "\n".join(agent_notes).strip(), text


def _format_inline_html(escaped_text: str) -> str:
    """Convert **bold** and bare URLs inside already-escaped text to Groove TinyMCE HTML."""
    formatted = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped_text)
    formatted = re.sub(
        r"(https?://[^\s<]+)",
        r'<a href="\1" target="_blank" rel="noopener noreferrer">\1</a>',
        formatted,
    )
    return formatted


def markdown_to_groove_html(text: str) -> tuple[str, str, bool]:
    """Convert draft into TinyMCE-compatible HTML (<div>, <div><br></div>, <strong>, <ol>, <ul>) + clean plain text."""
    is_rtl = bool(re.search(r"[\u0600-\u06FF]", text))
    lines = text.splitlines()
    html_parts = []
    in_list = None  # "ol" or "ul"

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
        f'font-size: 14px; line-height: 1.6; color: #1f2937; {align_style}">'
        + "".join(html_parts)
        + "</div>"
    )

    plain_text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    return groove_html, plain_text, is_rtl


def render_groove_preview_and_copy(draft_text: str):
    """Render white-background email preview + one-click TinyMCE HTML & Plain Text copy buttons."""
    groove_html, plain_text, is_rtl = markdown_to_groove_html(draft_text)
    groove_html_js = json.dumps(groove_html)
    plain_text_js = json.dumps(plain_text)

    dir_css = "rtl" if is_rtl else "ltr"
    align_css = "right" if is_rtl else "left"
    line_count = max(10, draft_text.count("\n") + len(draft_text) // 75)
    box_height = min(560, max(260, line_count * 24 + 90))

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
          font-size: 14px;
          font-weight: 600;
          cursor: pointer;
          transition: background 0.15s ease;
        }}
        .btn-primary:hover {{ background: #e03e3e; }}
        .btn-secondary {{
          background: #f3f4f6;
          color: #1f2937;
          border: 1px solid #d1d5db;
          border-radius: 8px;
          padding: 9px 14px;
          font-size: 13px;
          font-weight: 500;
          cursor: pointer;
        }}
        .btn-secondary:hover {{ background: #e5e7eb; }}
        .status {{
          font-size: 13px;
          color: #059669;
          font-weight: 600;
        }}
        .preview-card {{
          background: #ffffff;
          color: #1f2937;
          border: 1px solid #d1d5db;
          border-radius: 10px;
          padding: 18px 20px;
          direction: {dir_css};
          text-align: {align_css};
          font-family: Arial, Helvetica, sans-serif;
          font-size: 14px;
          line-height: 1.65;
          max-height: {box_height - 65}px;
          overflow-y: auto;
          box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
        }}
        .preview-card ol, .preview-card ul {{
          margin: 6px 0 8px 0;
          padding-inline-start: 24px;
        }}
        .preview-card li {{
          margin-bottom: 5px;
        }}
        .preview-card strong {{
          color: #111827;
        }}
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


# ================= PASSWORD GATE (ENTER SUBMIT + PASSWORD MANAGER + EYE ICON) =================
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
            with st.form("login_form", clear_on_submit=False):
                st.text_input(
                    "Team Account",
                    value="sapaad-support-team",
                    autocomplete="username",
                )
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


# ================= MAIN UI =================
if not GROQ_API_KEY and not GEMINI_API_KEY:
    st.error("Missing `GROQ_API_KEY` or `GEMINI_API_KEY` secret. Add at least one in Streamlit Cloud Settings → Secrets.")
    st.stop()

index_bundle = load_index()
stats = index_bundle.get("stats", {})

available_models = {
    label: pair
    for label, pair in MODELS.items()
    if (pair[0] == "groq" and GROQ_API_KEY) or (pair[0] == "gemini" and GEMINI_API_KEY)
}

with st.sidebar:
    st.title("⚙️ Sapaad Support AI")
    mode = st.radio(
        "Mode",
        ["Draft Reply", "Refine Draft", "FAQ / Template", "Internal Note / Summary"],
        index=0,
    )
    language = st.selectbox("Output Language", ["Auto (Match Customer)", "English", "Arabic"])
    model_label = st.selectbox("AI Model", list(available_models.keys()), index=0)
    top_k = st.slider("Reference Sources to Pull", min_value=3, max_value=8, value=5)

    st.divider()
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
    st.caption(f"Active Providers (Auto-Fallback): **{' → '.join(active_provs)}**")

st.header(f"📩 {mode}")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    verify_kb_in_refine = False
    if mode == "Draft Reply":
        customer_email = st.text_area(
            "Paste Customer Email",
            height=250,
            placeholder="Paste the full customer email from Groove here...",
        )
        agent_notes = st.text_area(
            "Agent Instructions / Key Points (Optional)",
            height=105,
            placeholder="e.g., Ask if they are using Android or iOS / Tell them we tried calling and need AnyDesk ID...",
        )
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
    elif mode == "FAQ / Template":
        customer_email = st.text_area(
            "Topic / Scenario for Reusable Template",
            height=210,
            placeholder="e.g., How to install Cloud Print Agent and find the API key...",
        )
        agent_notes = st.text_area("Extra Requirements (Optional)", height=100)
    else:
        customer_email = st.text_area(
            "Paste Ticket Thread or Foreign-Language Email",
            height=270,
            placeholder="Paste the email or thread to summarize into an English internal note...",
        )
        agent_notes = st.text_area("Focus / Specific Question (Optional)", height=80)

    generate_btn = st.button("✨ Generate", type="primary", use_container_width=True)

with col2:
    if generate_btn:
        search_source = (
            (customer_email.strip() or agent_notes.strip())
            if mode != "Refine Draft"
            else f"{agent_notes}\n{customer_email}".strip()
        )
        if not search_source:
            st.warning("Please enter an email or draft first.")
        else:
            primary_provider, primary_model = available_models[model_label]
            should_search = (
                mode in ("Draft Reply", "FAQ / Template")
                or (mode == "Refine Draft" and verify_kb_in_refine)
            )

            with st.spinner("Searching Sapaad KB & Tickets and generating..."):
                if should_search:
                    matches, search_meta = _smart_retrieve_context(
                        index_bundle=index_bundle,
                        query=search_source,
                        extract_subqueries_fn=lambda txt: extract_english_subqueries(
                            txt, primary_provider, primary_model
                        ),
                        top_k=top_k,
                        cf_account_id=CF_ACCOUNT_ID,
                        cf_api_token=CF_API_TOKEN,
                    )
                else:
                    matches, search_meta = [], {}

                context_chunks = []
                used_chars = 0
                for m in matches:
                    snippet = f"[{m['type']} | {m['source']}]\n{m['text']}"
                    if used_chars + len(snippet) > MAX_CONTEXT_CHARS:
                        remaining = MAX_CONTEXT_CHARS - used_chars
                        if remaining > 350:
                            context_chunks.append(snippet[:remaining] + "...")
                        break
                    context_chunks.append(snippet)
                    used_chars += len(snippet) + 8
                context_block = "\n\n---\n\n".join(context_chunks)

                lang_rule = (
                    "Match the customer's language (English or Arabic)."
                    if language == "Auto (Match Customer)"
                    else f"Write the output in {language}."
                )

                active_system_prompt = SYSTEM_PROMPT
                temp = 0.22

                if mode == "Draft Reply":
                    prompt = (
                        f"TASK: Draft a ready-to-send Sapaad Support email reply.\n"
                        f"LANGUAGE: {lang_rule}\n\n"
                        f"CUSTOMER EMAIL:\n{customer_email}\n\n"
                        f"AGENT INSTRUCTIONS (follow strictly if provided, without exposing harsh internal phrasing):\n"
                        f"{agent_notes or 'None'}\n\n"
                        f"REFERENCE CONTEXT (Sapaad Policies, KB & Past Tickets):\n"
                        f"{context_block or 'No matching context found.'}"
                    )
                elif mode == "Refine Draft":
                    active_system_prompt = REFINE_SYSTEM_PROMPT
                    temp = 0.35
                    prompt = (
                        f"TASK: Refine and polish the agent's rough draft below.\n"
                        f"LANGUAGE: {lang_rule}\n\n"
                        f"AGENT ROUGH DRAFT:\n{agent_notes}\n\n"
                        f"CUSTOMER EMAIL (Optional context):\n{customer_email or 'Not provided'}\n\n"
                        f"REFERENCE CONTEXT:\n{context_block or 'None'}"
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
                        f"TASK: Write a concise English internal Groove note summarizing this ticket for teammates (translate if needed). "
                        f"Do NOT include a customer greeting or customer closing line.\n\n"
                        f"TICKET CONTENT:\n{customer_email}\n"
                        f"FOCUS:\n{agent_notes or 'Key issue, account details, and next action needed.'}"
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
                    agent_note, clean_draft = split_agent_note_and_clean(
                        raw_output, mode, language, customer_email
                    )
                    audit_report = verify_and_repair_draft(
                        draft_text=clean_draft,
                        context_block=context_block,
                        mode=mode,
                        language_choice=language,
                        customer_email=customer_email,
                        agent_notes=agent_notes,
                    )
                    st.session_state.last_output = audit_report["draft"]
                    st.session_state.last_agent_note = agent_note
                    st.session_state.last_audit = audit_report
                    st.session_state.last_model = used_model
                    st.session_state.last_matches = matches
                    st.session_state.last_search_meta = search_meta
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
            st.success("✅ **Audit Passed:** Menu paths verified • Zero leaks • Groove format valid")

        if audit.get("auto_fixes"):
            st.caption("🛠️ **Auto-repaired:** " + " • ".join(audit["auto_fixes"]))

        render_groove_preview_and_copy(st.session_state.last_output)

        meta = st.session_state.get("last_search_meta", {})
        engine_caption = f" | Search: `{meta.get('engine')}` | Grounding: `{meta.get('confidence')}`" if meta else ""
        st.caption(f"Model used: `{st.session_state.last_model}`{engine_caption}")

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
