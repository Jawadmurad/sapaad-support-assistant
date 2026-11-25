import base64
import csv
import hashlib
import io
import json
import re
import urllib.error
import urllib.request

IMAGE_EXTS = {"png", "jpg", "jpeg", "webp"}
MIME_BY_EXT = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "pdf": "application/pdf",
}
# Sapaad ingredient/menu CSVs disallow special chars except '&' and '.'
DISALLOWED_SPECIAL_RE = re.compile(r"[\\/|\*\?<>~`^]")


def _call_gemini_multimodal(
    raw_bytes: bytes,
    mime_type: str,
    prompt: str,
    gemini_api_key: str,
) -> str:
    """Analyze an image or scanned PDF using Gemini 2.5 Flash REST API (no extra SDK)."""
    if not gemini_api_key:
        return "(Gemini API key missing — add GEMINI_API_KEY in Secrets to enable image/scanned-PDF reading.)"

    b64_data = base64.b64encode(raw_bytes).decode("ascii")
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inlineData": {"mimeType": mime_type, "data": b64_data}},
            ]
        }],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1200},
    }

    last_err = ""
    for model_id in ("gemini-2.5-flash", "gemini-2.5-flash-lite"):
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_id}:generateContent?key={gemini_api_key}"
        )
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            candidates = data.get("candidates") or []
            if candidates:
                parts = (candidates[0].get("content") or {}).get("parts") or []
                text = "".join(p.get("text", "") for p in parts if "text" in p).strip()
                if text:
                    return text
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}"
        except Exception as e:
            last_err = str(e)

    return f"(Could not analyze visual attachment: {last_err})"


def _inspect_tabular_rows(filename: str, rows: list[list[str]], extra_notes: list[str]) -> str:
    """Inspect CSV/Excel tables for Sapaad upload issues (blank cells, multi-emails, special chars)."""
    if not rows:
        return f"Spreadsheet `{filename}` appears to be empty."

    headers = [str(c).strip() if c is not None else "" for c in rows[0]]
    data_rows = rows[1:]
    total_rows = len(data_rows)

    blank_counts = {h or f"Col_{i+1}": 0 for i, h in enumerate(headers)}
    special_char_hits = []
    multi_email_hits = []

    for r_idx, row in enumerate(data_rows[:500], start=2):
        padded = list(row) + [""] * max(0, len(headers) - len(row))
        for c_idx, col_name in enumerate(headers):
            key = col_name or f"Col_{c_idx+1}"
            val = str(padded[c_idx]).strip() if padded[c_idx] is not None else ""
            if not val:
                blank_counts[key] += 1
            else:
                # Check for multiple emails packed into one cell
                if "@" in val and (val.count("@") > 1 or "," in val or ";" in val):
                    if len(multi_email_hits) < 5:
                        multi_email_hits.append(f"Row {r_idx} ({key})")
                # Check for disallowed special characters (skip URLs)
                if not val.lower().startswith(("http://", "https://")) and DISALLOWED_SPECIAL_RE.search(val):
                    if len(special_char_hits) < 5:
                        special_char_hits.append(f"Row {r_idx} ({key}: {val[:35]})")

    lines = [
        f"File: {filename}",
        f"Total Data Rows: {total_rows} | Columns ({len(headers)}): {', '.join(headers[:20])}",
    ]
    lines.extend(extra_notes)

    cols_with_blanks = [f"{k}: {v} blank" for k, v in blank_counts.items() if v > 0]
    if cols_with_blanks:
        lines.append("Blank Cells Detected: " + "; ".join(cols_with_blanks[:12]))
    if multi_email_hits:
        lines.append("Multiple Emails in One Cell: " + ", ".join(multi_email_hits))
    if special_char_hits:
        lines.append("Disallowed Special Characters (e.g., '/', '\\', '*'): " + ", ".join(special_char_hits))

    # Compact sample of first 8 rows
    lines.append("Sample Rows (first 8):")
    for r_idx, row in enumerate(data_rows[:8], start=2):
        preview = " | ".join(str(c).strip()[:40] if c is not None else "" for c in row[:12])
        lines.append(f"  Row {r_idx}: {preview}")

    return "\n".join(lines)[:3200]


def _parse_csv(filename: str, raw_bytes: bytes) -> str:
    extra_notes = []
    try:
        decoded = raw_bytes.decode("utf-8-sig")
        extra_notes.append("Encoding Check: Valid UTF-8 CSV.")
    except UnicodeDecodeError:
        decoded = raw_bytes.decode("cp1256", errors="replace")
        extra_notes.append(
            "Encoding Warning: File is NOT UTF-8 compliant (Arabic characters may display as junk/question marks in Sapaad)."
        )

    reader = csv.reader(io.StringIO(decoded))
    rows = [r for r in reader if any(cell.strip() for cell in r)]
    return _inspect_tabular_rows(filename, rows, extra_notes)


def _parse_excel(filename: str, raw_bytes: bytes) -> str:
    extra_notes = [
        "Format Note: Uploaded as Excel (.xlsx) — Sapaad bulk Menu/Ingredient uploads require UTF-8 CSV (.csv) format."
    ]
    try:
        import openpyxl
    except ImportError:
        return (
            f"File: {filename} (Excel spreadsheet). "
            "Install `openpyxl` in requirements.txt to inspect Excel columns and rows automatically."
        )

    try:
        wb = openpyxl.load_workbook(io.BytesIO(raw_bytes), data_only=True, read_only=True)
        sheet_names = wb.sheetnames
        ws = wb[sheet_names[0]]
        extra_notes.append(f"Sheets: {', '.join(sheet_names)} (inspecting '{sheet_names[0]}')")
        rows = []
        for idx, row in enumerate(ws.iter_rows(values_only=True)):
            if idx > 505:
                break
            if any(c is not None and str(c).strip() for c in row):
                rows.append(["" if c is None else str(c) for c in row])
        wb.close()
        return _inspect_tabular_rows(filename, rows, extra_notes)
    except Exception as e:
        return f"Could not parse Excel file `{filename}` ({e}). Note: legacy `.xls` files must be saved as `.xlsx` or UTF-8 `.csv`."


def _parse_pdf(filename: str, raw_bytes: bytes, gemini_api_key: str) -> str:
    extracted = ""
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(raw_bytes))
        pages_text = []
        for idx, page in enumerate(reader.pages[:6], start=1):
            t = (page.extract_text() or "").strip()
            if t:
                pages_text.append(f"[Page {idx}]\n{t}")
        extracted = "\n\n".join(pages_text).strip()
    except ImportError:
        extracted = ""
    except Exception:
        extracted = ""

    if len(extracted) >= 40:
        return f"PDF Document: {filename}\n{extracted[:3200]}"

    # Fallback for scanned / image-based PDFs via Gemini Vision (under 6 MB)
    if len(raw_bytes) <= 6 * 1024 * 1024 and gemini_api_key:
        prompt = (
            "You are inspecting a PDF attached to a Sapaad POS customer support email. "
            "Summarize the document accurately: list all error messages, invoice/order numbers, "
            "dates, Sapaad screens, or key customer details shown. Be concise and factual."
        )
        vision_text = _call_gemini_multimodal(raw_bytes, "application/pdf", prompt, gemini_api_key)
        return f"Scanned PDF Document: {filename}\n{vision_text}"

    return f"PDF Document `{filename}` contained no selectable text (likely scanned)."


def analyze_uploaded_files(uploaded_files: list, gemini_api_key: str, cache: dict) -> tuple[str, list[str]]:
    """
    Analyze uploaded images, PDFs, Excel, and CSV files.
    Uses SHA-256 session caching so repeated clicks don't re-call Vision APIs.
    Returns (combined_summary_text, list_of_short_status_labels).
    """
    if not uploaded_files:
        return "", []

    summaries = []
    labels = []

    for uf in uploaded_files:
        raw_bytes = uf.getvalue()
        file_hash = hashlib.sha256(raw_bytes).hexdigest()[:16]
        ext = uf.name.rsplit(".", 1)[-1].lower() if "." in uf.name else ""
        cache_key = f"{uf.name}:{file_hash}"

        if cache_key in cache:
            text_summary, label = cache[cache_key]
        else:
            if ext in IMAGE_EXTS:
                mime = MIME_BY_EXT.get(ext, "image/png")
                vision_prompt = (
                    "You are analyzing a screenshot or photo attached to a Sapaad POS support ticket. "
                    "Transcribe EVERY visible error message, alert popup, URL, order/invoice ID, printer/KDS status, "
                    "or Sapaad menu screen verbatim, and briefly describe what issue is visible. "
                    "Do not guess anything that is not clearly visible in the image."
                )
                vision_out = _call_gemini_multimodal(raw_bytes, mime, vision_prompt, gemini_api_key)
                text_summary = f"Attached Image ({uf.name}):\n{vision_out}"
                label = f"🖼️ {uf.name} (Vision analyzed)"
            elif ext == "pdf":
                text_summary = _parse_pdf(uf.name, raw_bytes, gemini_api_key)
                label = f"📄 {uf.name} (PDF extracted)"
            elif ext == "csv":
                text_summary = _parse_csv(uf.name, raw_bytes)
                label = f"📊 {uf.name} (CSV inspected)"
            elif ext in ("xlsx", "xls"):
                text_summary = _parse_excel(uf.name, raw_bytes)
                label = f"📊 {uf.name} (Excel inspected)"
            else:
                text_summary = f"Unsupported attachment type: {uf.name}"
                label = f"📎 {uf.name} (unsupported)"

            cache[cache_key] = (text_summary, label)

        summaries.append(text_summary)
        labels.append(label)

    return "\n\n---\n\n".join(summaries), labels
