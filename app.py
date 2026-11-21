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
    "Llama 3.3 70B (High Limit / Fast)": "llama-3.3-70b-versatile",
    "GPT-OSS 20B (Ultra-Fast)": "openai/gpt-oss-20b",
}
FALLBACK_MODEL = "llama-3.3-70b-versatile"

# Token-safe limits for Groq Free Tier (8,000 TPM on GPT-OSS models)
MAX_CONTEXT_CHARS = 10800
MAX_DOC_CHARS = 1050

SYSTEM_PROMPT = """You are a Sapaad Customer Support email assistant. Your job is to draft customer replies, refine rough drafts, and write internal Groove ticket notes using the provided Sapaad Knowledge Base (KB) and historical Groove Support Tickets as your primary references.

1. SOURCE PRIORITY & ACCURACY (HIGHEST PRIORITY — NEVER COMPROMISE)
- Always ground your answer in the provided REFERENCE CONTEXT. Never invent Sapaad features, menu paths, buttons, API-key workflows, third-party provider names, pricing, or policies.
- Factual accuracy ALWAYS overrides self-service preferences: even if the agent asks to let the customer configure everything alone, NEVER invent steps, API keys, or menu paths that do not exist in the REFERENCE CONTEXT. If a feature is an optional add-on, not self-service, or not supported via open API, state that clearly and honestly.
- Prioritize Sapaad KB articles for official step-by-step navigation paths and product rules.
- Use historical Groove tickets for real-world edge cases, troubleshooting workarounds, policy clarifications (e.g., Guest Checkout, no custom open delivery API), and Sapaad's standard support phrasing.
- Prioritize recent tickets (2025–2026) over older tickets (2023–2024). Always recommend AnyDesk (never TeamViewer) if remote support is needed, and use prints.sapaad.com/getcpa for Cloud Print Agent.
- NEVER invent or guess Sapaad menu paths, settings, or button names. ONLY write a ">" menu path if that exact path appears in the REFERENCE CONTEXT.
- If a KB article mentions a prerequisite (such as "add a surcharge called Delivery Fee") without giving its menu path, state the prerequisite plainly WITHOUT adding any ">" menu path to it.
- Never assume the customer is on a limited plan (such as Sapaad Direct Lite) unless the customer explicitly mentions it; provide the standard setup steps first and only mention plan differences as a brief note if relevant.
- Always keep Sapaad UI menu paths and product names in clean, exact English (e.g., Setup > Sapaad Add-Ons > Sapaad Direct) even when replying in Arabic, and NEVER mix English and Arabic letters inside the same word.
- Whenever providing Sapaad.Online configuration steps (delivery areas, delivery fees, scheduled orders, payment methods, timings, or menu changes), remind the customer to sync changes via: Setup > Sapaad Add-Ons > Sapaad.Online Settings > Synchronize > Save.

2. AGENT INSTRUCTIONS & MISSING INFO HANDLING
- Follow AGENT INSTRUCTIONS strictly for workflow decisions (e.g., do NOT schedule a meeting, do NOT offer AnyDesk, and do NOT promise checking/reviewing the customer's account if the agent instructs against it).
- NEVER expose, quote, or mirror internal Agent Instructions harshly to the customer (never say "without needing any meeting or review from our side"). Keep the customer-facing tone warm, helpful, and natural.
- If a specific point is NOT covered in the REFERENCE CONTEXT:
  * Never invent or guess a menu path or technical workflow (such as API keys).
  * If the agent gave no restriction, politely state that we can check their account settings or review it together.
  * If the agent instructed you not to promise a review or meeting, state concisely what is supported or requires an add-on/partner integration without guessing fake steps, and flag the gap inside a short "[Agent Note: ...]" line at the very top of your output (the UI hides this automatically from the copyable email).
- Write "[Agent Note: ...]" strictly as an internal note to the support agent (never address the customer inside it). Only include it if something genuinely needs the agent's attention.

3. PRIVACY & PLACEHOLDER SAFETY
- NEVER copy redacted placeholders ([PHONE_OR_ID], [EMAIL]), other customers' names, restaurant names, passwords, or expired Zoom/WeTransfer links from past tickets into a new draft.
- Use clean bracketed placeholders like [Zoom Link] or [Date & Time] only if needed and not provided.

4. TONE, LANGUAGE & FORMATTING
- Always write in the "we" tone (representing the Sapaad Support Team), polite, clear, and concise.
- Match the requested or customer's language (English or Arabic).
- For Arabic replies, ALWAYS use respectful plural addressing with accurate spelling in EVERY verb and step (e.g., افتحوا، اختاروا، انتقلوا، اضغطوا، ضعوا، حرّروا، احفظوا، كرروا). NEVER switch to singular verbs or misspell them (never write واختروا or اختر).
- MANDATORY GREETING (Customer Emails Only): ALWAYS start line 1 with the greeting on its own line, followed by a blank line, THEN a brief 1-sentence polite opening line:
  - English: "Dear [Customer Name]," or "Hi [Customer Name]," (or "Hi," if no name is provided)
  - Arabic: "السلام عليكم،"
- For multi-part customer inquiries, place each numbered heading on its own line ("1. Title"), followed by simple "- " bullet lines underneath, and leave a blank line between every numbered section. NEVER put "- " before a number.
- Mandatory Closing (Customer Emails Only) on its own separate paragraph at the very end:
  - English:
    Please feel free to write to us for any further assistance.
  - Arabic:
    لا تترددوا بالتواصل معنا لأي مساعدة إضافية.
- NO duplicate closing sentence right before the mandatory closing line.
- NO sign-off or signature: Never add "Best regards,", "Thanks,", or an agent name at the bottom (Groove appends the signature automatically).
- PLAIN TEXT ONLY: Never use Markdown formatting. Do NOT use **bold**, *italics*, ### headers, or backticks.
"""


# ================= TOKENIZATION & INDEXING =================
def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w\u0600-\u06FF]+", text.lower())


def deduplicate_paragraphs(text: str) -> str:
    """Remove identical repeated paragraphs often scraped in Groove ticket threads."""
    seen = set()
    cleaned = []
    for para in re.split(r"\n+", text):
        p = para.strip()
        if not p:
            continue
        key = re.sub(r"\s+", " ", p.lower())
        if key not in seen:
            seen.add(key)
            cleaned.append(p)
    return "\n".join(cleaned)


def chunk_long_text(text: str, max_chars: int = 1350, overlap: int = 200) -> list[str]:
    """Split long KB articles into overlapping chunks so bottom sections are never lost."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            split_pos = text.rfind("\n", start + max_chars // 2, end)
            if split_pos == -1:
                split_pos = text.rfind(". ", start + max_chars // 2, end)
            if split_pos != -1:
                end = split_pos + 1
        chunk = text[start:end].strip()
        if len(chunk) >= 40:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


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
        # Give high-level subscription/overview blurbs a lower weight than step-by-step KB guides & 2025-2026 tickets
        kb_weight = 1.05 if filename.lower() in {"subscription.md", "pricing.md"} else 1.40

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        if is_ticket_file:
            parts = re.split(r"(?m)^#{3,5}\s+(Ticket\s+#\d+[^\n]*)", content)
            for i in range(1, len(parts), 2):
                header = parts[i].strip()
                body = parts[i + 1].strip() if i + 1 < len(parts) else ""
                if len(body) < 40:
                    continue

                res_parts = re.split(r"\*{0,2}Support Resolution:\*{0,2}", body)
                if len(res_parts) < 2:
                    continue

                cust_text = deduplicate_paragraphs(res_parts[0])
                cust_text = re.sub(r"^\*{0,2}Customer:\*{0,2}\s*", "", cust_text).strip()
                res_text = deduplicate_paragraphs("\n".join(res_parts[1:]))
                if len(res_text) < 35:
                    continue

                year_match = re.search(r"\((\d{4})-", header)
                year = int(year_match.group(1)) if year_match else 2024
                weight = 1.25 if year >= 2025 else 0.85

                llm_text = (
                    f"{header}\n"
                    f"Customer Issue: {cust_text[:300]}\n"
                    f"Support Resolution: {res_text[:1150]}"
                )
                search_text = f"{header} {header}\n{cust_text[:450]}\n{res_text[:1400]}"

                docs.append({
                    "source": filename,
                    "title": header,
                    "type": "Ticket",
                    "year": year,
                    "weight": weight,
                    "text": llm_text,
                    "search_text": search_text,
                })
                ticket_count += 1
        else:
            parts = re.split(r"(?m)^#{2,4}\s+([^\n]+)", content)
            if len(parts) > 2:
                for i in range(1, len(parts), 2):
                    title = parts[i].strip()
                    body = parts[i + 1].strip() if i + 1 < len(parts) else ""
                    if len(body) < 35:
                        continue
                    if "Consolidated Knowledge Base documentation" in body and "Total Articles:" in body:
                        continue

                    for idx_chunk, chunk in enumerate(chunk_long_text(body)):
                        chunk_title = title if idx_chunk == 0 else f"{title} (Part {idx_chunk + 1})"
                        docs.append({
                            "source": filename,
                            "title": chunk_title,
                            "type": "KB",
                            "year": 2026,
                            "weight": kb_weight,
                            "text": f"KB Article: {chunk_title}\n{chunk}",
                            "search_text": f"{title} {title}\n{chunk}",
                        })
                        kb_count += 1
            else:
                chunks = re.split(r"(?m)^-{3,}\s*$", content)
                for raw_chunk in chunks:
                    raw_chunk = raw_chunk.strip()
                    if len(raw_chunk) < 40:
                        continue
                    if "Consolidated Knowledge Base documentation" in raw_chunk and "Total Articles:" in raw_chunk:
                        continue
                    first_line = raw_chunk.splitlines()[0].strip("#* ")
                    for idx_sub, sub in enumerate(chunk_long_text(raw_chunk)):
                        chunk_title = first_line[:120] or filename
                        if idx_sub > 0:
                            chunk_title += f" (Part {idx_sub + 1})"
                        docs.append({
                            "source": filename,
                            "title": chunk_title,
                            "type": "KB",
                            "year": 2026,
                            "weight": kb_weight,
                            "text": sub,
                            "search_text": f"{chunk_title} {chunk_title}\n{sub}",
                        })
                        kb_count += 1

    if not docs:
        return None, [], 0, 0

    corpus = [tokenize(d["search_text"]) for d in docs]
    bm25 = BM25Okapi(corpus)
    return bm25, docs, kb_count, ticket_count


# ================= MULTI-TOPIC ENGLISH QUERY EXTRACTION =================
def get_extractor_models(primary_model: str) -> list[str]:
    """Use a separate Groq model for query extraction so it does not eat the drafting model's TPM limit."""
    candidates = ["openai/gpt-oss-20b", "llama-3.3-70b-versatile", "openai/gpt-oss-120b"]
    return [m for m in candidates if m != primary_model] + [primary_model]


def extract_english_subqueries(text: str, primary_model: str) -> tuple[list[str], str]:
    """Translate and split any Arabic or English email into Sapaad-native English search queries."""
    client = Groq(api_key=GROQ_API_KEY)
    last_error = None

    extraction_prompt = (
        "You are a search query translator for Sapaad POS Knowledge Base & Support Tickets.\n"
        "Read the customer email below (Arabic or English) and extract EVERY distinct question or requirement "
        "as its OWN separate line (NEVER merge two different points into one line).\n"
        "Each line must be a short ENGLISH search query (3 to 7 keywords) mapped to exact Sapaad terminology:\n"
        "- QR code setup / activation -> how to set up Sapaad Direct Generate QR Table QRs\n"
        "- waiter button / call waiter -> Enable Waiter Hailing Sapaad Direct Pro Open Tab\n"
        "- disable OTP / remove OTP / without OTP -> guest checkout Checkout Mode instead of OTP Sapaad Direct\n"
        "- OTP without WhatsApp / alternative OTP -> enable email OTP Sapaad Direct Sapaad Online\n"
        "- scheduled orders / advance orders -> Accept Scheduled Orders Sapaad.Online Make Arrangements\n"
        "- disable cash on delivery / pay at counter -> disable Pay at Counter Cash on Delivery Payment Types Sapaad Direct Sapaad.Online\n"
        "- delivery API / third party delivery integration -> open API third party delivery partner integration Sapaad Connect\n"
        "- delivery areas / map / country filter -> delivery areas Google Map Integration upload delivery areas\n"
        "- delivery fee per zone -> add surcharge Delivery Fee different delivery fee per area\n"
        "Output ONLY the complete English query lines—no Arabic, no numbering, no bullets, no explanations. Max 10 lines.\n\n"
        f"{text[:2500]}"
    )

    for m in get_extractor_models(primary_model):
        try:
            resp = client.chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": extraction_prompt}],
                temperature=0.0,
                max_tokens=2000,
            )
            raw = (resp.choices[0].message.content or "").strip()
            lines = []
            for ln in raw.splitlines():
                cleaned = re.sub(r"^[\d\.\-\•\*\s]+", "", ln).strip()
                if cleaned and len(cleaned.split()) >= 2 and re.search(r"[a-zA-Z]{2,}", cleaned):
                    lines.append(cleaned)
            if lines:
                return lines[:10], m
        except Exception as e:
            last_error = e

    if last_error:
        st.warning(f"English query translation failed ({last_error}); falling back to original text.")
    return [text], "fallback-raw"


def retrieve_context(bm25, docs, query: str, primary_model: str, top_k: int = 6):
    """Balanced per-topic retrieval: guarantees EVERY customer point gets its best KB or Ticket match."""
    if not bm25 or not query.strip():
        return [], [], ""

    subqueries, extractor_used = extract_english_subqueries(query, primary_model)

    selected = []
    seen_ids = set()
    max_docs = max(top_k, min(12, len(subqueries) + 2))

    per_query_ranked = []
    for q in subqueries:
        tokens = tokenize(q)
        if not tokens:
            per_query_ranked.append((q, []))
            continue
        raw_scores = bm25.get_scores(tokens)
        scored = sorted(
            [(score * docs[idx]["weight"], idx) for idx, score in enumerate(raw_scores) if score > 0],
            reverse=True,
        )
        per_query_ranked.append((q, scored[:20]))

    # Pass 1: Every single sub-query gets its #1 highest-scoring match (whether KB or Ticket)
    first_pass_types = {}
    for q, scored in per_query_ranked:
        for weighted_score, idx in scored:
            if idx not in seen_ids:
                d = dict(docs[idx])
                d["matched_query"] = q
                selected.append(d)
                seen_ids.add(idx)
                first_pass_types[q] = d["type"]
                break
        if len(selected) >= max_docs:
            break

    # Pass 2: Add complementary match (KB <-> Ticket) for remaining slots if relevant
    if len(selected) < max_docs:
        for q, scored in per_query_ranked:
            wanted_type = "Ticket" if first_pass_types.get(q) == "KB" else "KB"
            for weighted_score, idx in scored:
                if weighted_score < 4.0:
                    break
                if idx not in seen_ids and docs[idx]["type"] == wanted_type:
                    d = dict(docs[idx])
                    d["matched_query"] = q
                    selected.append(d)
                    seen_ids.add(idx)
                    break
            if len(selected) >= max_docs:
                break

    # Pass 3: Backfill up to top_k if it's a single-topic email
    if len(selected) < top_k:
        combined_tokens = tokenize(" ".join(subqueries))
        raw_scores = bm25.get_scores(combined_tokens)
        scored = sorted(
            [(score * docs[idx]["weight"], idx) for idx, score in enumerate(raw_scores) if score > 0],
            reverse=True,
        )
        for _, idx in scored:
            if idx not in seen_ids:
                d = dict(docs[idx])
                d["matched_query"] = "combined"
                selected.append(d)
                seen_ids.add(idx)
            if len(selected) >= top_k:
                break

    return selected[:max_docs], subqueries, extractor_used


def build_context_block(matches: list[dict]) -> str:
    """Build a token-safe reference context block that never exceeds MAX_CONTEXT_CHARS."""
    if not matches:
        return ""
    per_doc_limit = min(MAX_DOC_CHARS, max(750, MAX_CONTEXT_CHARS // max(1, len(matches))))
    parts = []
    total_chars = 0
    for m in matches:
        snippet = m["text"][:per_doc_limit]
        topic_tag = f" | Topic: {m.get('matched_query', '')}" if m.get("matched_query") else ""
        block = f"[{m['type']} | {m['source']}{topic_tag}]\n{snippet}"
        if total_chars + len(block) > MAX_CONTEXT_CHARS:
            break
        parts.append(block)
        total_chars += len(block) + 7
    return "\n\n---\n\n".join(parts)


# ================= GROQ LLM CALL & OUTPUT CLEANING =================
def clean_for_groove(text: str, is_customer_email: bool = True) -> str:
    """Strip Markdown symbols, fix minor typos, and enforce Groove-ready greeting & spacing."""
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"(?m)^#{1,6}\s+", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Fix rare mixed-script brand typos & common Arabic verb typos
    text = re.sub(r"Sapa[\u0600-\u06FF]+", "Sapaad", text)
    text = re.sub(r"\bواختروا\b", "واختاروا", text)
    text = re.sub(r"\bاختروا\b", "اختاروا", text)
    # Remove any stray bullet glued before a numbered heading (e.g. "- 2. Title" -> "2. Title")
    text = re.sub(r"(?m)^[ \t]*-\s+(\d+\.\s+)", r"\1", text)
    # Guarantee a blank line before numbered headings
    text = re.sub(r"(?m)(?<!\n)\n(\d+\.\s+)", r"\n\n\1", text)
    text = text.strip()

    if is_customer_email:
        # Ensure Arabic customer emails always begin with السلام عليكم،
        if re.search(r"[\u0600-\u06FF]", text):
            if not re.match(r"^(السلام عليكم|مرحب[ااً]|عزيزي|السادة)", text):
                text = f"السلام عليكم،\n\n{text}"
            else:
                text = re.sub(r"^(السلام عليكم[،,]?)[ \t]*\n*", r"\1\n\n", text)

        # Guarantee the mandatory closing sits cleanly on its own paragraph
        for closing in (
            "لا تترددوا بالتواصل معنا لأي مساعدة إضافية.",
            "Please feel free to write to us for any further assistance.",
        ):
            text = re.sub(rf"\s*{re.escape(closing)}\s*$", f"\n\n{closing}", text)

    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_agent_note(text: str) -> tuple[str, str]:
    """Extract optional leading [Agent Note: ...] so it displays separately from the copyable email."""
    match = re.match(r"^\s*\[Agent Note:\s*(.*?)\]\s*\n+", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        note = re.sub(r"\*\*(.*?)\*\*", r"\1", match.group(1)).strip()
        return note, text[match.end():].strip()
    return "", text.strip()


def call_groq(model_id: str, user_prompt: str, temperature: float = 0.2) -> tuple[str, str]:
    client = Groq(api_key=GROQ_API_KEY)
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=2800,
        )
        return (resp.choices[0].message.content or "").strip(), model_id
    except Exception as e:
        err_str = str(e).lower()
        if "429" in err_str or "413" in err_str or "rate_limit" in err_str:
            trimmed_prompt = user_prompt[:11000]
            retry_models = [m for m in [FALLBACK_MODEL, "openai/gpt-oss-20b"] if m != model_id] or [model_id]
            last_err = e
            for retry_model in retry_models:
                try:
                    resp = client.chat.completions.create(
                        model=retry_model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": trimmed_prompt},
                        ],
                        temperature=temperature,
                        max_tokens=2800,
                    )
                    return (resp.choices[0].message.content or "").strip(), f"{retry_model} (fallback)"
                except Exception as retry_e:
                    last_err = retry_e
            raise last_err
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
    top_k = st.slider("Reference Sources to Pull", min_value=3, max_value=10, value=6)

    st.divider()
    if docs:
        st.success(f"Indexed **{kb_count:,}** KB sections & **{ticket_count:,}** resolved tickets")
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
            placeholder="e.g., Guide the customer step-by-step without scheduling a meeting...",
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
                    matches, used_queries, extractor_used = retrieve_context(
                        bm25, docs, search_source, MODELS[model_label], top_k=top_k
                    )
                else:
                    matches, used_queries, extractor_used = [], [], ""

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
                        f"AGENT INSTRUCTIONS (follow strictly for workflow, without inventing paths/features or exposing internal instructions verbatim):\n{agent_notes or 'None'}\n\n"
                        f"REFERENCE CONTEXT (Sapaad KB & Past Resolved Tickets):\n{context_block or 'No matching context found.'}"
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
                    raw_output, used_model = call_groq(MODELS[model_label], prompt)
                    agent_note, body_text = split_agent_note(raw_output)
                    clean_email = clean_for_groove(
                        body_text, is_customer_email=(mode != "Internal Note / Summary")
                    )
                    st.session_state.last_agent_note = agent_note
                    st.session_state.last_output = clean_email
                    st.session_state.last_model = used_model
                    st.session_state.last_matches = matches
                    st.session_state.last_queries = used_queries
                    st.session_state.last_extractor = extractor_used
                except Exception as e:
                    st.error(f"API Error: {e}")

    if "last_output" in st.session_state:
        if st.session_state.get("last_agent_note"):
            st.warning(f"**Agent Note (not included in copy box):** {st.session_state.last_agent_note}")

        st.subheader("Ready to Copy")
        st.code(st.session_state.last_output, language=None, wrap_lines=True)
        st.caption(f"Model used: `{st.session_state.last_model}` — Use the copy icon in the top-right corner of the box above (avoid manual drag-select to preserve line breaks).")

        if st.session_state.get("last_matches"):
            with st.expander(f"📚 Referenced Sources ({len(st.session_state.last_matches)})"):
                if st.session_state.get("last_queries"):
                    ext_tag = f" ({st.session_state.get('last_extractor', '')})" if st.session_state.get("last_extractor") else ""
                    st.info(f"English search queries used{ext_tag}: " + " | ".join(st.session_state.last_queries))
                for m in st.session_state.last_matches:
                    topic_label = f" — *Matched: {m['matched_query']}*" if m.get("matched_query") else ""
                    st.markdown(f"**[{m['type']}] {m['title']}** *({m['source']})*{topic_label}")
                    st.caption(m["text"][:450] + "...")
                    st.divider()
