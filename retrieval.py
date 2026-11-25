import glob
import json
import os
import re
import urllib.request
from typing import Callable, Optional

import numpy as np
from huggingface_hub import snapshot_download
from rank_bm25 import BM25Okapi

# ================= 1. NOISE & BOILERPLATE CLEANING =================
BOILERPLATE_PATTERNS = [
    r"Did you know Sapaad is integrated with.*?(?=\n|$)",
    r"Global Support Numbers:.*?(?=\n|$)",
    r"NOTICE:\s*This electronic mail message.*$",
    r"CONFIDENTIALITY (?:CAUTION|NOTE):.*$",
    r"Privileged/confidential information may be contained.*$",
    r"This email(?: and any attachments)? may contain confidential.*$",
    r"قد تحتوي هذه الرسالة الإلكترونية على معلومات سرية.*$",
    r"Please consider the environment before printing.*?(?=\n|$)",
    r"Sent (?:from|securely from) (?:my iPhone|my Huawei|Outlook for \w+|Yahoo Mail|Mail for Windows|Spacemail).*?(?=\n|$)",
    r"Get Outlook for (?:iOS|Android).*?(?=\n|$)",
    r"مُرسل من هاتف.*?(?=\n|$)",
    r"\[PHONE_OR_ID\]|\[EMAIL\]",
]

# Common support greetings/filler words that dilute BM25 IDF
STOPWORDS = {
    "hi", "hello", "dear", "team", "sapaad", "support", "please", "kindly",
    "thanks", "thank", "regards", "best", "hope", "doing", "well", "writing",
    "assist", "assistance", "help", "need", "want", "would", "like", "could",
    "can", "you", "your", "our", "we", "the", "and", "for", "with", "this",
    "that", "from", "have", "has", "are", "was", "were", "been", "not", "but",
    "السلام", "عليكم", "مرحبا", "اهلا", "تحيه", "طيبه", "شكرا", "جزيلا",
    "يرجي", "الرجاء", "نرجو", "فريق", "الدعم", "ساباد", "من", "في", "علي",
    "الي", "عن", "مع", "هذا", "هذه", "تم", "ان", "او", "لا", "ما",
}

# Ticket resolutions that carry little troubleshooting knowledge
GENERIC_RESOLUTION_PATTERNS = [
    r"could you please (?:further )?elaborate",
    r"provide more details about your query",
    r"share the (?:name of your restaurant|registered email)",
    r"forwarded your (?:details|request|suggestion) to our (?:sales|development|product) team",
    r"remotely connected to your (?:system|device) via anydesk and assisted you with your query",
    r"contacted you and assisted you with your (?:query|requirement)",
    r"share (?:the|your) anydesk",
]

# Fast, zero-token Sapaad domain synonym & typo expander
SYNONYM_MAP = {
    r"\b(?:saapad|sappad|spaad|sapad|saapaad)\b": "sapaad",
    r"\bquickbook\b": "quickbooks",
    r"\burban\s+piper\b": "urbanpiper",
    r"\breceipt\s+machine\b": "thermal printer receipt printer",
    r"\bkitchen\s+slip\b": "kot kitchen printer",
    r"\bcancel\s+item\b": "void item cancel order",
    r"\bdelivery\s+apps?\b": "aggregators urbanpiper talabat deliveroo",
    r"\bqr\s+menu\b": "sapaad direct qr code table qr",
    r"\bcash\s+on\s+delivery\b": "pay at counter cod cash on delivery",
    r"\bprint\s+agent\b": "cloud print agent cpa",
    r"\bkitchen\s+screen\b": "kds kitchen display system",
    r"\bcustomer\s+screen\b": "cds customer display screen",
    r"\breset\s+order\s+number": "token numbers reset invoice sequence",
    r"\bweight\s+barcode": "barcode reader weight embedded barcode",
    r"\bmenu\s+not\s+updating": "menu flush sync publish atlas urbanpiper",
    r"\bauto\s+reconcile": "stock take auto reconcile wastage",
    r"\btax\s+inclusive": "tax inclusive item price vat",
    r"\bdelivery\s+charge": "delivery fee surcharge delivery areas",
    r"\bwaiter\s+call": "waiter hailing sapaad direct pro",
    r"\bguest\s+order": "guest checkout checkout mode otp",
    r"\bswipe\s+card": "swipe card rfid login staff",
    r"\bcall\s+center": "pabx caller id callcenter url",
}

# Marketing/brochure KB files that shouldn't outrank resolved 2025-2026 tickets
LOW_PRIORITY_KB_FILES = {"subscription.md", "pricing.md", "referrals.md", "miscellaneous.md"}


def normalize_arabic(text: str) -> str:
    """Unify Arabic letter variants and strip diacritics for higher lexical recall."""
    text = re.sub(r"[\u064B-\u065F\u0670\u0640]", "", text)  # Strip tashkeel & tatweel
    text = re.sub(r"[أإآ]", "ا", text)
    text = text.replace("ة", "ه").replace("ى", "ي")
    return text


def clean_for_index(text: str) -> str:
    cleaned = text
    for pat in BOILERPLATE_PATTERNS:
        cleaned = re.sub(pat, " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
    # Normalize frequent typos right at index time
    cleaned = re.sub(r"\b(?:saapad|sappad|spaad|sapad|saapaad)\b", "sapaad", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bquickbook\b", "quickbooks", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\burban\s+piper\b", "urbanpiper", cleaned, flags=re.IGNORECASE)
    return cleaned


def expand_synonyms(query: str) -> str:
    """Append Sapaad-official terminology for common customer phrasing (0 tokens)."""
    extras = []
    q_low = query.lower()
    for pattern, expansion in SYNONYM_MAP.items():
        if re.search(pattern, q_low, flags=re.IGNORECASE):
            extras.append(expansion)
    return f"{query} {' '.join(extras)}".strip() if extras else query


def tokenize(text: str) -> list[str]:
    norm = normalize_arabic(text.lower())
    tokens = re.findall(r"[\w\u0600-\u06FF]+", norm)
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


# ================= 2. QUALITY SCORING & CONTEXT DIET =================
def split_ticket_sections(body: str) -> tuple[str, str]:
    """Separate Customer issue from Support Resolution cleanly."""
    if "**Support Resolution:**" in body:
        cust, res = body.split("**Support Resolution:**", 1)
    elif "Support Resolution:" in body:
        cust, res = body.split("Support Resolution:", 1)
    else:
        return body.strip(), ""
    cust = re.sub(r"^\*\*Customer:\*\*\s*", "", cust.strip())
    return cust.strip(), res.strip()


def compute_ticket_weight(res_part: str, year: int) -> float:
    """Drop unresolved tickets; down-weight generic replies; boost 2025-2026 fixes."""
    if not res_part or len(res_part) < 45:
        return 0.0  # Exclude unresolved/empty tickets completely

    base = 1.25 if year >= 2025 else 0.80
    if len(res_part) < 90:
        return base * 0.55

    if any(re.search(p, res_part, re.IGNORECASE) for p in GENERIC_RESOLUTION_PATTERNS):
        return base * 0.45  # Keep as fallback, never crowd out detailed fixes
    return base


def format_ticket_for_llm(header: str, cust_part: str, res_part: str) -> str:
    """Context Diet: compact customer problem + full resolution so fixes never get cut off."""
    clean_cust = clean_for_index(cust_part)[:450].strip()
    clean_res = clean_for_index(res_part)[:1800].strip()
    return f"{header}\nCustomer Issue: {clean_cust}\nSupport Resolution: {clean_res}"


# ================= 3. DATA LOADING & DETERMINISTIC PARSING =================
def load_data_directory(local_dir: str, hf_repo: str, hf_token: str) -> str:
    """Use local ./data if present, otherwise download .md + optional .npy/.json from Private HF Dataset."""
    if os.path.isdir(local_dir) and glob.glob(os.path.join(local_dir, "*.md")):
        return local_dir

    if hf_repo and hf_token:
        cache_dir = "/tmp/sapaad_data"
        snapshot_download(
            repo_id=hf_repo,
            repo_type="dataset",
            token=hf_token,
            local_dir=cache_dir,
            allow_patterns=["*.md", "sapaad_embeddings.npy", "sapaad_embeddings_meta.json"],
        )
        return cache_dir
    return ""


def load_and_parse_docs(data_dir: str) -> tuple[list[dict], dict]:
    """Parse KB, Policies, and Tickets deterministically (shared by app.py and build_embeddings.py)."""
    docs = []
    stats = {"kb": 0, "tickets": 0, "skipped_tickets": 0, "policies": 0}

    for filepath in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        filename = os.path.basename(filepath)
        fname_low = filename.lower()

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Golden Truth policies.md (ready for Enhancement #3)
        if fname_low == "policies.md":
            parts = re.split(r"(?m)^#{2,4}\s+([^\n]+)", content)
            for i in range(1, len(parts), 2):
                title = parts[i].strip()
                body = parts[i + 1].strip() if i + 1 < len(parts) else ""
                if len(body) < 20:
                    continue
                docs.append({
                    "doc_id": f"{filename}::{title}",
                    "source": filename,
                    "title": title,
                    "type": "Policy",
                    "year": 2026,
                    "weight": 2.00,
                    "text": f"Official Policy — {title}\n{body[:2000]}",
                    "index_text": f"{title} {title} {title}\n{clean_for_index(body[:2000])}",
                })
                stats["policies"] += 1
            continue

        # Groove Ticket Files
        if fname_low.startswith("tickets_"):
            parts = re.split(r"(?m)^#{3,5}\s+(Ticket\s+#\d+[^\n]*)", content)
            for i in range(1, len(parts), 2):
                header = parts[i].strip()
                body = parts[i + 1].strip() if i + 1 < len(parts) else ""
                cust_part, res_part = split_ticket_sections(body)
                year_match = re.search(r"\((\d{4})-", header)
                year = int(year_match.group(1)) if year_match else 2024
                weight = compute_ticket_weight(res_part, year)

                if weight <= 0.0:
                    stats["skipped_tickets"] += 1
                    continue

                # BM25F-style boosting: Title 3x + Customer Issue 2x + Resolution 1x
                clean_title = clean_for_index(header)
                clean_cust = clean_for_index(cust_part[:900])
                clean_res = clean_for_index(res_part[:1600])
                index_text = f"{clean_title} {clean_title} {clean_title}\n{clean_cust} {clean_cust}\n{clean_res}"

                docs.append({
                    "doc_id": f"{filename}::{header}",
                    "source": filename,
                    "title": header,
                    "type": "Ticket",
                    "year": year,
                    "weight": weight,
                    "text": format_ticket_for_llm(header, cust_part, res_part),
                    "index_text": index_text,
                })
                stats["tickets"] += 1
            continue

        # Official Sapaad KB Files (split by ##/###/#### headings)
        kb_weight = 1.05 if fname_low in LOW_PRIORITY_KB_FILES else 1.45
        parts = re.split(r"(?m)^#{2,4}\s+([^\n]+)", content)
        if len(parts) > 2:
            for i in range(1, len(parts), 2):
                title = parts[i].strip()
                body = parts[i + 1].strip() if i + 1 < len(parts) else ""
                if len(body) < 35:
                    continue
                clean_title = clean_for_index(title)
                clean_body = clean_for_index(body[:2400])
                docs.append({
                    "doc_id": f"{filename}::{title}",
                    "source": filename,
                    "title": title,
                    "type": "KB",
                    "year": 2026,
                    "weight": kb_weight,
                    "text": f"KB Article: {title}\n{body[:2400]}",
                    "index_text": f"{clean_title} {clean_title} {clean_title}\n{clean_body}",
                })
                stats["kb"] += 1
        elif len(content.strip()) > 35:
            docs.append({
                "doc_id": f"{filename}::{filename}",
                "source": filename,
                "title": filename,
                "type": "KB",
                "year": 2026,
                "weight": kb_weight,
                "text": content[:2400],
                "index_text": f"{filename} {filename}\n{clean_for_index(content[:2400])}",
            })
            stats["kb"] += 1

    return docs, stats


def build_search_index(local_dir: str, hf_repo: str, hf_token: str) -> dict:
    """Build Smart BM25+ index and auto-load optional precomputed bge-m3 embeddings."""
    data_dir = load_data_directory(local_dir, hf_repo, hf_token)
    if not data_dir:
        return {"bm25": None, "docs": [], "stats": {}, "hybrid_ready": False}

    docs, stats = load_and_parse_docs(data_dir)
    if not docs:
        return {"bm25": None, "docs": [], "stats": stats, "hybrid_ready": False}

    corpus = [tokenize(d["index_text"]) for d in docs]
    bm25 = BM25Okapi(corpus, k1=1.4, b=0.65)

    # Optional Zero-RAM Hybrid Embeddings (aligned strictly by doc_id)
    emb_matrix = None
    emb_doc_indices = None
    npy_path = os.path.join(data_dir, "sapaad_embeddings.npy")
    meta_path = os.path.join(data_dir, "sapaad_embeddings_meta.json")

    if os.path.isfile(npy_path) and os.path.isfile(meta_path):
        try:
            raw_vecs = np.load(npy_path).astype(np.float32)
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_ids = json.load(f).get("doc_ids", [])
            id_to_idx = {d["doc_id"]: idx for idx, d in enumerate(docs)}
            valid_rows, mapped_doc_idxs = [], []
            for row_i, did in enumerate(meta_ids):
                if did in id_to_idx and row_i < len(raw_vecs):
                    valid_rows.append(row_i)
                    mapped_doc_idxs.append(id_to_idx[did])
            if valid_rows:
                emb_matrix = raw_vecs[valid_rows]
                emb_doc_indices = np.array(mapped_doc_idxs, dtype=np.int32)
        except Exception:
            emb_matrix = None
            emb_doc_indices = None

    return {
        "bm25": bm25,
        "docs": docs,
        "stats": stats,
        "emb_matrix": emb_matrix,
        "emb_doc_indices": emb_doc_indices,
        "hybrid_ready": emb_matrix is not None,
    }


# ================= 4. SMART BYPASS & CLOUDFLARE EMBEDDING =================
def should_bypass_llm_query_extraction(text: str) -> bool:
    """Skip LLM query extraction for short, single-topic English emails (saves 30-40% tokens)."""
    has_arabic = bool(re.search(r"[\u0600-\u06FF]", text))
    if has_arabic:
        return False
    if len(text.strip()) > 380:
        return False
    # Check if customer wrote a numbered/bulleted multi-question list
    list_items = re.findall(r"(?m)^\s*(?:\d+[\.\)]|[-•\*])\s+\S+", text)
    if len(list_items) >= 2:
        return False
    if text.count("?") >= 3:
        return False
    return True


def embed_query_cloudflare(text: str, account_id: str, api_token: str) -> Optional[np.ndarray]:
    """Embed only the incoming customer email via Cloudflare Workers AI @cf/baai/bge-m3 (free tier)."""
    if not account_id or not api_token:
        return None
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/baai/bge-m3"
    payload = json.dumps({"text": [text[:2000]], "truncate_inputs": True}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        vec = np.array(data["result"]["data"][0], dtype=np.float32)
        norm = np.linalg.norm(vec)
        return (vec / norm) if norm > 0 else None
    except Exception:
        return None


# ================= 5. SMART BM25+ / HYBRID RRF RETRIEVAL =================
def retrieve_context(
    index_bundle: dict,
    query: str,
    extract_subqueries_fn: Callable[[str], tuple[list[str], str]],
    top_k: int = 6,
    cf_account_id: str = "",
    cf_api_token: str = "",
) -> tuple[list[dict], dict]:
    """Retrieve grounded docs via Smart BM25+ RRF + optional Cloudflare bge-m3 Hybrid fusion."""
    bm25 = index_bundle.get("bm25")
    docs = index_bundle.get("docs", [])
    if not bm25 or not query.strip():
        return [], {"queries": [], "engine": "None", "confidence": "Low"}

    # 1. Decide: Smart Bypass (0 tokens) vs LLM Multi-Topic Extraction
    if should_bypass_llm_query_extraction(query):
        subqueries = [clean_for_index(query)]
        query_engine = "Smart Bypass (0 tokens)"
    else:
        subqueries, query_engine = extract_subqueries_fn(query)
        if not subqueries:
            subqueries = [clean_for_index(query)]

    # 2. Run Smart BM25+ across all sub-queries & fuse via Reciprocal Rank Fusion (RRF)
    rrf_scores: dict[int, float] = {}
    matched_by: dict[int, str] = {}
    k_rrf = 60
    top_raw_bm25 = 0.0
    per_topic_reserved: list[int] = []

    for q in subqueries:
        expanded_q = expand_synonyms(clean_for_index(q))
        tokens = tokenize(expanded_q)
        if not tokens:
            continue
        raw_scores = bm25.get_scores(tokens)
        scored = [
            (float(score) * docs[idx]["weight"], idx)
            for idx, score in enumerate(raw_scores)
            if score > 0
        ]
        scored.sort(reverse=True)
        if scored:
            top_raw_bm25 = max(top_raw_bm25, scored[0][0])

        # Guarantee top-1 KB/Policy or Ticket for each distinct sub-topic in multi-point emails
        if len(subqueries) > 1 and scored:
            top_idx = scored[0][1]
            if top_idx not in per_topic_reserved:
                per_topic_reserved.append(top_idx)
                matched_by.setdefault(top_idx, q)

        for rank, (_, idx) in enumerate(scored[:25], start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (k_rrf + rank))
            matched_by.setdefault(idx, q)

    # 3. Optional Hybrid Semantic Fusion (Cloudflare bge-m3 + precomputed .npy)
    used_hybrid = False
    top_cosine = 0.0
    if index_bundle.get("hybrid_ready") and cf_account_id and cf_api_token:
        q_vec = embed_query_cloudflare(clean_for_index(query), cf_account_id, cf_api_token)
        if q_vec is not None:
            used_hybrid = True
            sims = index_bundle["emb_matrix"] @ q_vec  # Fast vectorized dot product (~4ms)
            doc_indices = index_bundle["emb_doc_indices"]
            # Weight semantic similarity gently by doc priority so 2025-2026 & KB stay on top
            weighted_sims = [
                (float(sims[i]) * (0.85 + 0.15 * docs[int(doc_indices[i])]["weight"]), int(doc_indices[i]), float(sims[i]))
                for i in range(len(sims))
                if sims[i] > 0.42
            ]
            weighted_sims.sort(reverse=True)
            if weighted_sims:
                top_cosine = weighted_sims[0][2]
            # Semantic weight 1.25x in RRF so paraphrased & Arabic matches shine
            for rank, (_, idx, _) in enumerate(weighted_sims[:30], start=1):
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.25 / (k_rrf + rank))
                matched_by.setdefault(idx, "Semantic (bge-m3)")

    # 4. Final Selection: Reserved multi-topic anchors + top RRF-ranked docs (balanced mix)
    max_docs = max(top_k, min(11, len(subqueries) * 2))
    selected_indices: list[int] = []
    seen: set[int] = set()

    for idx in per_topic_reserved:
        if idx not in seen and len(selected_indices) < max_docs:
            selected_indices.append(idx)
            seen.add(idx)

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    kb_count = sum(1 for i in selected_indices if docs[i]["type"] in ("KB", "Policy"))

    for idx, _ in ranked:
        if idx in seen:
            continue
        doc_type = docs[idx]["type"]
        # Ensure tickets aren't completely starved by KB on single-topic queries
        if len(subqueries) == 1 and doc_type == "KB" and kb_count >= max(3, top_k - 2):
            continue
        selected_indices.append(idx)
        seen.add(idx)
        if doc_type in ("KB", "Policy"):
            kb_count += 1
        if len(selected_indices) >= max_docs:
            break

    results = []
    for idx in selected_indices:
        doc_copy = dict(docs[idx])
        doc_copy["matched_query"] = matched_by.get(idx, "")
        results.append(doc_copy)

    # 5. Simple Grounding Confidence Signal
    if top_raw_bm25 >= 14.0 or top_cosine >= 0.68:
        confidence = "High"
    elif top_raw_bm25 >= 7.0 or top_cosine >= 0.55:
        confidence = "Medium"
    else:
        confidence = "Low"

    debug_info = {
        "queries": subqueries,
        "engine": f"{query_engine} + {'Hybrid (BM25+ & bge-m3)' if used_hybrid else 'Smart BM25+'}",
        "confidence": confidence,
        "top_bm25": round(top_raw_bm25, 2),
        "top_cosine": round(top_cosine, 3),
    }
    return results, debug_info
