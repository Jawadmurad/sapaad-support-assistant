import re

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# Matches international/local phone numbers and 9-10 digit AnyDesk IDs (avoiding short numbers/prices)
PHONE_OR_ANYDESK_RE = re.compile(
    r"(?<!\d)(?:\+?\d{1,3}[\s\-\.]?)?(?:\(?\d{2,4}\)?[\s\-\.]?)\d{3,4}[\s\-\.]?\d{3,4}(?!\d)"
)


def scrub_pii(texts: list[str]) -> tuple[list[str], dict[str, str]]:
    """
    Mask emails and phone/AnyDesk numbers across a list of input strings before sending to external LLMs.
    Returns (scrubbed_texts, token_to_original_map).
    """
    token_map: dict[str, str] = {}
    reverse_map: dict[str, str] = {}
    email_counter = 1
    phone_counter = 1

    def _replace_email(match: re.Match) -> str:
        nonlocal email_counter
        raw = match.group(0)
        # Keep official Sapaad support addresses unmasked so context stays clear
        if raw.lower().endswith("@sapaad.com"):
            return raw
        if raw not in reverse_map:
            tok = f"[CUST_EMAIL_{email_counter}]"
            reverse_map[raw] = tok
            token_map[tok] = raw
            email_counter += 1
        return reverse_map[raw]

    def _replace_phone(match: re.Match) -> str:
        nonlocal phone_counter
        raw = match.group(0).strip()
        digits = re.sub(r"\D", "", raw)
        if len(digits) < 8 or len(digits) > 15:
            return match.group(0)
        if raw not in reverse_map:
            tok = f"[CUST_PHONE_{phone_counter}]"
            reverse_map[raw] = tok
            token_map[tok] = raw
            phone_counter += 1
        return reverse_map[raw]

    scrubbed_list = []
    for t in texts:
        if not t:
            scrubbed_list.append(t)
            continue
        s = EMAIL_RE.sub(_replace_email, t)
        s = PHONE_OR_ANYDESK_RE.sub(_replace_phone, s)
        scrubbed_list.append(s)

    return scrubbed_list, token_map


def restore_pii(text: str, token_map: dict[str, str]) -> str:
    """Swap local PII tokens back to the customer's real details after LLM generation."""
    if not text or not token_map:
        return text
    restored = text
    for tok, original in token_map.items():
        restored = restored.replace(tok, original)
    return restored
