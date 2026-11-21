import glob
import os
import re
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
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default)


GROQ_API_KEY = get_secret("GROQ_API_KEY")
HF_TOKEN = get_secret("HF_TOKEN")
HF_DATASET_REPO = get_secret("HF_DATASET_REPO")
TEAM_PASSWORD = get_secret("TEAM_PASSWORD")
LOCAL_DATA_DIR = "data"

MODELS = {
    "GPT-OSS 120B (Best Quality)": "openai/gpt-oss-120b",
    "Qwen 3.8 27B (Strong Arabic/EN)": "qwen/qwen3.8-27b",
    "GPT-OSS 20B (Ultra-Fast)": "openai/gpt-oss-20b",
}
FALLBACK_MODEL = "openai/gpt-oss-20b"

# Keep total reference context safely under Groq's 8,000 TPM free-tier limit
MAX_CONTEXT_CHARS = 11000
MAX_DOC_CHARS = 1100

SYSTEM_PROMPT = """You are a Sapaad Customer Support email assistant. Your job is to draft customer replies, refine rough drafts, and write internal Groove ticket notes using the provided Sapaad Knowledge Base (KB) and historical Groove Support Tickets as your primary references.

1. SOURCE PRIORITY & ACCURACY
- Always ground your answer in the provided REFERENCE CONTEXT. Never invent Sapaad features, menu paths, pricing, or policies.
- Prioritize Sapaad KB articles for official step-by-step navigation paths and product rules.
- Use historical Groove tickets for real-world edge cases, troubleshooting workarounds, and Sapaad's standard support phrasing.
- Prioritize recent tickets (2025–2026) over older tickets (2023–2024). Always recommend AnyDesk (never TeamViewer) for remote support, and use the current Cloud Print Agent link (prints.sapaad.com/getcpa).
- If the context does not clearly confirm an answer or requires account-level checking, add a short "[Agent Note: ...]" line at the very top before the email draft, and draft a polite clarification or AnyDesk request.
- NEVER invent or guess Sapaad menu paths, settings, or button names. ONLY include a menu path if it appears in the REFERENCE CONTEXT below.
- Always keep Sapaad UI menu paths in their original English (e.g., Setup > Location Setup > Self Ordering Systems > Sapaad Direct) even when replying in Arabic, because the Sapaad interface is in English.
- If a specific point in a multi-part email is NOT covered in the REFERENCE CONTEXT, do NOT guess—state politely that we will review/confirm that item during the meeting or check their account settings.

2. PRIVACY & PLACEHOLDER SAFETY
- NEVER copy redacted placeholders ([PHONE_OR_ID], [EMAIL]), other customers' names, restaurant names, passwords, or expired Zoom/WeTransfer links from past tickets into a new draft.
- Use clean bracketed placeholders like [Zoom Link] or [Date & Time] only if the agent hasn't provided them.

3. TONE, LANGUAGE & FORMATTING
- Always write in the "we" tone (representing the Sapaad Support Team), polite, clear, and concise.
- Match the requested or customer's language (English or Arabic).
- Greeting: Use "Dear [Customer Name]," or "Hi [Customer Name]," (use "Hi," if no name is provided; use "السلام عليكم،" for Arabic).
- Mandatory Closing (Customer Emails Only):
  - English: End every customer email with exactly:
    Please feel free to write to us for any further assistance.
  - Arabic: End every customer email with exactly:
    لا تترددوا بالتواصل معنا لأي مساعدة إضافية.
- NO sign-off or signature: Never add "Best regards,", "Thanks,", or an agent name at the bottom (Groove appends the signature automatically).
- For multi-part customer inquiries, use clean numbered points so each question is clearly answered.
- PLAIN TEXT ONLY: Never use Markdown formatting. Do NOT use **bold**, *italics*, ### headers, or backticks. Use plain text, clean line breaks, and simple numbered lists (1., 2., 3.) or hyphens (-) so the output can be pasted directly into Groove.
"""


# ================= TOKENIZATION & INDEXING =================
def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w\u0600-\u06FF]+", text.lower())


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


@st.cache_resource(show_spinner="Indexing Sapaad KB & Support Tickets...")
def build_search_index():
    data_dir = load_data_directory()
    if not data_dir:
        return None, [], 0, 0

    docs = []
    kb_count = 0
    ticket_count = 0

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
                    continue
                year_match = re.search(r"\((\d{4})-", header)
                year = int(year_match.group(1)) if year_match else 2024
                weight = 1.20 if year >= 2025 else 0.85
                docs.append({
                    "source": filename,
                    "title": header,
                    "type": "Ticket",
                    "year": year,
                    "weight": weight,
                    "text": f"{header}\n{body[:1800]}",
                })
                ticket_count += 1
        else:
            # Try splitting KB files by Markdown headings (##, ###, ####)
            parts = re.split(r"(?m)^#{2,4}\s+([^\n]+)", content)
            if len(parts) > 2:
                for i in range(1, len(parts), 2):
                    title = parts[i].strip()
                    body = parts[i + 1].strip() if i + 1 < len(parts) else ""
                    if len(body) < 30:
                        continue
                    docs.append({
                        "source": filename,
                        "title": title,
                        "type": "KB",
                        "year": 2026,
                        "weight": 1.40,
                        "text": f"KB Article: {title}\n{body[:1800]}",
                    })
                    kb_count += 1
            else:
                # Fallback: split by horizontal rules (---) or Source: URL blocks
                chunks = re.split(r"(?m)^-{3,}\s*$", content)
                added_any = False
                for chunk in chunks:
                    chunk = chunk.strip()
                    if len(chunk) < 40:
                        continue
                    first_line = chunk.splitlines()[0].strip("#* ")
                    docs.append({
                        "source": filename,
                        "title": first_line[:120] or filename,
                        "type": "KB",
                        "year": 2026,
                        "weight": 1.40,
                        "text": chunk[:1800],
                    })
                    kb_count += 1
                    added_any = True
                if not added_any and len(content.strip()) > 30:
                    docs.append({
                        "source": filename,
                        "title": filename,
                        "type": "KB",
                        "year": 2026,
                        "weight": 1.40,
                        "text": content[:1800],
                    })
                    kb_count += 1

    if not docs:
        return None, [], 0, 0

    corpus = [tokenize(d["text"]) for d in docs]
    bm25 = BM25Okapi(corpus)
    return bm25, docs, kb_count, ticket_count


def extract_english_subqueries(text: str, model_id: str) -> list[str]:
    """Translate and split any Arabic or English email into clean English Sapaad search queries."""
    client = Groq(api_key=GROQ_API_KEY)
    models_to_try = [model_id] if model_id == FALLBACK_MODEL else [model_id, FALLBACK_MODEL]
    last_error = None

    for m in models_to_try:
        try:
            resp = client.chat.completions.create(
                model=m,
                messages=[{
                    "role": "user",
                    "content": (
                        "Extract each distinct Sapaad POS / Sapaad.Online / Sapaad Direct support topic "
                        "from the customer email below as a short ENGLISH search query (3 to 6 English keywords per line).\n"
                        "Use standard Sapaad terms where relevant (e.g., Sapaad Direct, Sapaad.Online, QR code, "
                        "waiter hailing, delivery areas, delivery fee, OTP, Email OTP, SMS, scheduled orders, cash on delivery, API).\n"
                        "Output ONLY the English lines—no Arabic, no numbering, no bullets, no intro text. Max 8 lines.\n\n"
                        f"{text[:2500]}"
                    ),
                }],
                temperature=0.0,
                max_tokens=1000,
            )
            raw = (resp.choices[0].message.content or "").strip()
            lines = []
            for ln in raw.splitlines():
                cleaned = re.sub(r"^[\d\.\-\•\*\s]+", "", ln).strip()
                if cleaned and re.search(r"[a-zA-Z]{2,}", cleaned):
                    lines.append(cleaned)
            if lines:
                return lines[:8]
        except Exception as e:
            last_error = e

    if last_error:
        st.warning(f"English query translation failed ({last_error}); falling back to original text.")
    return [text]


def retrieve_context(bm25, docs, query: str, model_id: str, top_k: int = 6):
    """Search English KB & Tickets per topic using round-robin, capped to stay within Groq TPM limits."""
    if not bm25 or not query.strip():
        return [], []

    subqueries = extract_english_subqueries(query, model_id)

    selected = []
    seen_ids = set()
    max_docs = max(top_k, min(10, len(subqueries) + 2))

    # First pass: grab the best KB article for each topic
    for q in subqueries:
        tokens = tokenize(q)
        if not tokens:
            continue
        raw_scores = bm25.get_scores(tokens)
        scored = sorted(
            [(score * docs[idx]["weight"], idx) for idx, score in enumerate(raw_scores) if score > 0],
            reverse=True,
        )
        for _, idx in scored[:15]:
            if idx not in seen_ids and docs[idx]["type"] == "KB":
                selected.append(docs[idx])
                seen_ids.add(idx)
                break
        if len(selected) >= max_docs:
            break

    # Second pass: fill remaining slots with top matching Tickets per topic
    if len(selected) < max_docs:
        for q in subqueries:
            tokens = tokenize(q)
            if not tokens:
                continue
            raw_scores = bm25.get_scores(tokens)
            scored = sorted(
                [(score * docs[idx]["weight"], idx) for idx, score in enumerate(raw_scores) if score > 0],
                reverse=True,
            )
            for _, idx in scored[:15]:
                if idx not in seen_ids and docs[idx]["type"] == "Ticket":
                    selected.append(docs[idx])
                    seen_ids.add(idx)
                    break
            if len(selected) >= max_docs:
                break

    # Backfill if fewer than top_k were found
    if len(selected) < top_k:
        combined_tokens = tokenize(" ".join(subqueries))
        raw_scores = bm25.get_scores(combined_tokens)
        scored = sorted(
            [(score * docs[idx]["weight"], idx) for idx, score in enumerate(raw_scores) if score > 0],
            reverse=True,
        )
        for _, idx in scored:
            if idx not in seen_ids:
                selected.append(docs[idx])
                seen_ids.add(idx)
            if len(selected) >= top_k:
                break

    return selected[:max_docs], subqueries


def build_context_block(matches: list[dict]) -> str:
    """Build a token-safe reference context block that never exceeds MAX_CONTEXT_CHARS."""
    parts = []
    total_chars = 0
    for m in matches:
        snippet = m["text"][:MAX_DOC_CHARS]
        block = f"[{m['type']} | {m['source']}]\n{snippet}"
        if total_chars + len(block) > MAX_CONTEXT_CHARS:
            break
        parts.append(block)
        total_chars += len(block) + 7
    return "\n\n---\n\n".join(parts)


# ================= GROQ LLM CALL =================
def clean_for_groove(text: str) -> str:
    """Strip any stray Markdown symbols so the draft is 100% plain-text ready for Groove."""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"(?m)^#{1,6}\s+", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def call_groq(model_id: str, user_prompt: str, temperature: float = 0.25) -> tuple[str, str]:
    client = Groq(api_key=GROQ_API_KEY)
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=3000,
        )
        return clean_for_groove(resp.choices[0].message.content), model_id
    except Exception as e:
        err_str = str(e).lower()
        if "429" in err_str or "413" in err_str or "rate_limit" in err_str:
            # Trim prompt slightly and retry on fallback model
            trimmed_prompt = user_prompt[:12000]
            retry_model = FALLBACK_MODEL if model_id != FALLBACK_MODEL else model_id
            resp = client.chat.completions.create(
                model=retry_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": trimmed_prompt},
                ],
                temperature=temperature,
                max_tokens=3000,
            )
            label = f"{retry_model} (fallback)" if retry_model != model_id else f"{retry_model} (trimmed)"
            return clean_for_groove(resp.choices[0].message.content), label
        raise


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
if not GROQ_API_KEY:
    st.error("Missing `GROQ_API_KEY` secret. Add it in Streamlit Cloud Settings → Secrets.")
    st.stop()

bm25, docs, kb_count, ticket_count = build_search_index()

with st.sidebar:
    st.title("⚙️ Sapaad Support AI")
    mode = st.radio(
        "Mode",
        ["Draft Reply", "Refine Draft", "FAQ / Template", "Internal Note / Summary"],
        index=0,
    )
    language = st.selectbox("Output Language", ["Auto (Match Customer)", "English", "Arabic"])
    model_label = st.selectbox("AI Model", list(MODELS.keys()), index=0)
    top_k = st.slider("Reference Sources to Pull", min_value=3, max_value=8, value=5)

    st.divider()
    if docs:
        st.success(f"Indexed **{kb_count:,}** KB articles & **{ticket_count:,}** tickets")
    else:
        st.warning("No `.md` files found in `./data` or `HF_DATASET_REPO`.")

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
            placeholder="e.g., Ask if they are using Android or iOS / Tell them we tried calling and need AnyDesk ID...",
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
            placeholder="Optional: paste the customer's email so the AI checks accuracy...",
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
        search_query = f"{customer_email}\n{agent_notes}".strip()
        if not search_query:
            st.warning("Please enter an email or draft first.")
        else:
            with st.spinner("Searching Sapaad KB & Tickets and generating..."):
                search_source = customer_email.strip() or agent_notes.strip()
                if mode != "Internal Note / Summary":
                    matches, used_queries = retrieve_context(
                        bm25, docs, search_source, MODELS[model_label], top_k=top_k
                    )
                else:
                    matches, used_queries = [], []

                context_block = build_context_block(matches)

                lang_rule = (
                    "Match the customer's language (English or Arabic)."
                    if language == "Auto (Match Customer)"
                    else f"Write the output in {language}."
                )

                if mode == "Draft Reply":
                    prompt = (
                        f"TASK: Draft a ready-to-send Sapaad Support email reply.\n"
                        f"LANGUAGE: {lang_rule}\n\n"
                        f"CUSTOMER EMAIL:\n{customer_email}\n\n"
                        f"AGENT INSTRUCTIONS (follow strictly if provided):\n{agent_notes or 'None'}\n\n"
                        f"REFERENCE CONTEXT (Sapaad KB & Past Tickets):\n{context_block or 'No matching context found.'}"
                    )
                elif mode == "Refine Draft":
                    prompt = (
                        f"TASK: Refine the agent's rough draft. Polish grammar, clarity, structure, and Sapaad Support tone without changing the core meaning or adding extra promises. Apply greeting and mandatory closing rules.\n"
                        f"LANGUAGE: {lang_rule}\n\n"
                        f"AGENT ROUGH DRAFT:\n{agent_notes}\n\n"
                        f"CUSTOMER EMAIL (if any):\n{customer_email or 'Not provided'}\n\n"
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
                    output, used_model = call_groq(MODELS[model_label], prompt)
                    st.session_state.last_output = output
                    st.session_state.last_model = used_model
                    st.session_state.last_matches = matches
                    st.session_state.last_queries = used_queries
                except Exception as e:
                    st.error(f"API Error: {e}")

    if "last_output" in st.session_state:
        st.subheader("Ready to Copy")
        st.code(st.session_state.last_output, language=None, wrap_lines=True)
        st.caption(f"Model used: `{st.session_state.last_model}` — Click the copy icon in the top-right corner of the box above.")

        if st.session_state.get("last_matches"):
            with st.expander(f"📚 Referenced Sources ({len(st.session_state.last_matches)})"):
                if st.session_state.get("last_queries"):
                    st.info("English search queries used: " + " | ".join(st.session_state.last_queries))
                for m in st.session_state.last_matches:
                    st.markdown(f"**[{m['type']}] {m['title']}** *({m['source']})*")
                    st.caption(m["text"][:450] + "...")
                    st.divider()
