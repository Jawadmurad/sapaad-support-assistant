import re

# ================= 1. CURATED VERIFIED SAPAAD UI PATHS =================
# Core paths that are 100% real in Sapaad (prevents false-positive warnings)
VERIFIED_SAPAAD_PATHS = {
    "setup > location setup",
    "setup > menu setup",
    "setup > staff",
    "setup > surcharges",
    "setup > global settings > others",
    "setup > global settings > promotions",
    "setup > sapaad add-ons > sapaad direct",
    "setup > sapaad add-ons > sapaad kds",
    "setup > sapaad add-ons > sapaad.online",
    "setup > partner integrations > online platform aggregators",
    "setup > location setup > self ordering systems > sapaad direct",
    "setup > location setup > delivery settings > delivery areas",
    "setup > location setup > printer & peripheral settings > kds stations",
    "setup > location setup > printer & peripheral settings > print settings > secondary printers",
    "setup > location setup > online platform aggregators > urbanpiper menu",
    "profile > edit profile",
}

# Known hallucination traps the LLM has guessed in the past
KNOWN_BAD_PATHS = {
    "setup > location setup > surcharges",
    "location setup > surcharges",
    "location setup > delivery settings > surcharges",
    "sapaad direct > primary settings > cash on delivery",
}

MANDATORY_EN_CLOSING = "Please feel free to write to us for any further assistance."
MANDATORY_AR_CLOSING = "لا تترددوا بالتواصل معنا لأي مساعدة إضافية."

LEGACY_CPA_URL_RE = re.compile(
    r"https?://(?:www\.)?sapaad\.com/software-updates/cloud-print-agent[^\s\)\]]*(?:\s+[a-f0-9]{1,8})?",
    flags=re.IGNORECASE,
)

TRAILING_SIGNOFF_RE = re.compile(
    r"(?im)^\s*(?:best regards|kind regards|warm regards|regards|thanks|thank you|sincerely|"
    r"sapaad support(?: team)?|مع أطيب التحيات|أطيب التحيات|تحياتنا|فريق دعم ساباد)[,\.\!\s]*$"
)

REDUNDANT_CLOSING_RE = re.compile(
    r"(?i)^(?:if\s+(?:you\s+(?:have|need|require)|after\s+these\s+steps)|"
    r"should\s+you\s+(?:have|need|require)|"
    r"feel\s+free\s+to\s+(?:reach\s+out|contact|let\s+us\s+know)|"
    r"please\s+(?:don'?t\s+hesitate|do\s+not\s+hesitate|let\s+us\s+know)|"
    r"إذا\s+(?:احتجتم|كان\s+لديكم)|في\s+حال\s+(?:وجود|احتجتم))"
)

REDUNDANT_CLOSING_KEYWORDS_RE = re.compile(
    r"(?i)(?:further\s+assistance|any\s+(?:other\s+|further\s+)?questions|let\s+us\s+know|"
    r"reach\s+out|investigate\s+further|happy\s+to\s+(?:help|assist|investigate)|"
    r"مساعدة|استفسار|توضيح|تواصل)"
)


def _normalize_path(path: str) -> str:
    p = path.lower().replace("&", "and").replace("–", "-").replace("—", "-")
    p = re.sub(r"\s*>\s*", " > ", p)
    p = re.sub(r"\s+", " ", p).strip(" .,:;\"'()[]")
    return p


def _extract_menu_paths(text: str) -> list[str]:
    """Extract Sapaad UI breadcrumbs like 'Setup > Location Setup > Delivery Areas'."""
    plain = re.sub(r"\*\*|__", "", text)
    # Match sequences of 2 to 6 short segments joined by '>'
    raw_matches = re.findall(
        r"([A-Za-z][A-Za-z0-9&\.\-\/\s]{1,32}(?:\s*>\s*[A-Za-z][A-Za-z0-9&\.\-\/\s]{1,32}){1,5})",
        plain,
    )
    paths = []
    # Trim leading/trailing lowercase prose words attached to the breadcrumb
    lead_trim = re.compile(
        r"^(?:(?:go|visit|open|navigate|click|select|choose|switch|under|in|on|to|from|then|and|the|your|our|a)\s+)+",
        flags=re.IGNORECASE,
    )
    trail_trim = re.compile(
        r"(?:\s+(?:and|then|to|in|on|for|where|which|that|click|select|open|choose|tick|check|scroll|button|tab|page|screen|menu|section|option|field|box))+$",
        flags=re.IGNORECASE,
    )

    for m in raw_matches:
        segs = [s.strip() for s in m.split(">")]
        if len(segs) < 2:
            continue
        segs[0] = lead_trim.sub("", segs[0]).strip()
        segs[-1] = trail_trim.sub("", segs[-1]).strip()
        # Keep only plausible UI breadcrumb segments (1-5 words each)
        if not segs[0] or not segs[-1]:
            continue
        if any(len(s.split()) > 6 or len(s) < 2 for s in segs):
            continue
        candidate = " > ".join(segs)
        if candidate not in paths:
            paths.append(candidate)
    return paths


def verify_and_repair_draft(
    draft_text: str,
    context_block: str,
    mode: str,
    language_choice: str,
    customer_email: str = "",
    agent_notes: str = "",
) -> dict:
    """
    Deterministic post-generation auditor & auto-repair gate.
    Returns:
      {
        "draft": str,          # Cleaned & auto-repaired draft
        "errors": list[str],   # Critical issues (unverified paths, leaks, TeamViewer)
        "warnings": list[str], # Soft warnings
        "auto_fixes": list[str]# Transparent log of silent repairs applied
      }
    """
    text = draft_text.strip()
    errors: list[str] = []
    warnings: list[str] = []
    auto_fixes: list[str] = []

    is_customer_email = mode != "Internal Note / Summary"
    is_arabic = (
        language_choice == "Arabic"
        or (language_choice == "Auto (Match Customer)" and bool(re.search(r"[\u0600-\u06FF]", text)))
    )

    # 1. Auto-repair legacy / corrupted Cloud Print Agent .msi links -> prints.sapaad.com/getcpa
    if LEGACY_CPA_URL_RE.search(text):
        text = LEGACY_CPA_URL_RE.sub("https://prints.sapaad.com/getcpa", text)
        auto_fixes.append("Replaced legacy/corrupted CPA `.msi` link with `https://prints.sapaad.com/getcpa`.")

    # 2. Auto-repair mixed-script & common Arabic spelling slips
    fixed_script = re.sub(r"Sapa[\u0600-\u06FF]+", "Sapaad", text)
    if fixed_script != text:
        text = fixed_script
        auto_fixes.append("Fixed mixed-script `Sapaad` spelling slip.")
    if "واختروا" in text or "قموا" in text:
        text = text.replace("واختروا", "واختاروا").replace("قموا", "قوموا")
        auto_fixes.append("Fixed Arabic plural verb spelling.")

    # 3. Customer-email Groove formatting guard (greeting, single closing, zero sign-offs)
    if is_customer_email:
        # Remove trailing sign-offs (e.g., "Best regards,", "Sapaad Support Team")
        lines = text.splitlines()
        removed_signoff = False
        while lines and (not lines[-1].strip() or TRAILING_SIGNOFF_RE.match(lines[-1].strip())):
            if lines[-1].strip() and TRAILING_SIGNOFF_RE.match(lines[-1].strip()):
                removed_signoff = True
            lines.pop()
        if removed_signoff:
            text = "\n".join(lines).strip()
            auto_fixes.append("Stripped trailing sign-off (Groove appends signature automatically).")

        # Enforce exact mandatory closing ONCE and remove redundant closing right above it
        mandatory = MANDATORY_AR_CLOSING if is_arabic else MANDATORY_EN_CLOSING
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

        # Strip existing copies of the mandatory closing from the tail so we can inspect the paragraph above
        while paragraphs and (
            paragraphs[-1].rstrip(".") == mandatory.rstrip(".")
            or mandatory.lower() in paragraphs[-1].lower()
        ):
            paragraphs.pop()

        # Check if the last remaining paragraph is a duplicate generic closing sentence
        if paragraphs and len(paragraphs[-1]) < 190:
            if REDUNDANT_CLOSING_RE.search(paragraphs[-1]) and REDUNDANT_CLOSING_KEYWORDS_RE.search(paragraphs[-1]):
                paragraphs.pop()
                auto_fixes.append("Removed duplicate closing sentence above mandatory Groove closing.")

        paragraphs.append(mandatory)
        text = "\n\n".join(paragraphs)

        # Ensure Arabic greeting is present
        if is_arabic and not re.match(r"^(\*\*)?السلام عليكم", text):
            text = f"السلام عليكم،\n\n{text}"
            auto_fixes.append("Added missing `السلام عليكم،` greeting.")

    # 4. Privacy & Legacy Leak Blocker (Critical Errors)
    combined_input = f"{customer_email}\n{agent_notes}".lower()

    if re.search(r"\[PHONE_OR_ID\]|\[EMAIL\]", text):
        errors.append("Leaked dataset redaction token (`[PHONE_OR_ID]` or `[EMAIL]`) found in draft.")

    if re.search(r"\bteamviewer\b|get\.teamviewer\.com", text, flags=re.IGNORECASE):
        errors.append("Outdated `TeamViewer` reference detected — Sapaad now uses **AnyDesk** (`https://anydesk.com/en/downloads/windows`).")

    # Flag Zoom links unless the agent/customer actually supplied that Zoom link
    zoom_links = re.findall(r"https?://[^\s\)]*zoom\.us/[^\s\)]+", text, flags=re.IGNORECASE)
    for zl in zoom_links:
        if zl.lower() not in combined_input:
            errors.append(f"Possible expired Zoom link copied from a past ticket: `{zl}`.")

    # Flag hardcoded sample passwords or 32-char hex API keys leaked from historical tickets
    if re.search(r"\$ap@ad123|Kfrw93@t", text):
        errors.append("Hardcoded password from an old ticket leaked into the draft.")
    hex_keys = re.findall(r"\b[a-f0-9]{32}\b", text)
    for hk in hex_keys:
        if hk.lower() not in combined_input:
            errors.append(f"Possible hardcoded API/Vendor key leaked from a past ticket (`{hk[:8]}...`).")

    # 5. Menu-Path Hallucination Auditor (Only when KB/Ticket context was retrieved)
    if context_block.strip() and mode in ("Draft Reply", "FAQ / Template", "Refine Draft"):
        norm_context = _normalize_path(re.sub(r"\*\*|__", "", context_block))
        norm_whitelist = {_normalize_path(p) for p in VERIFIED_SAPAAD_PATHS}
        norm_bad = {_normalize_path(p) for p in KNOWN_BAD_PATHS}

        extracted_paths = _extract_menu_paths(text)
        unverified_paths = []

        for raw_path in extracted_paths:
            np_path = _normalize_path(raw_path)

            # Immediate trap check
            if any(bad in np_path for bad in norm_bad):
                unverified_paths.append(raw_path)
                continue

            # Exact match in context or curated whitelist
            if np_path in norm_context or np_path in norm_whitelist:
                continue

            # Check adjacent pairs (handles combined paths like A > B + B > C)
            segs = [s.strip() for s in np_path.split(" > ")]
            all_pairs_grounded = True
            for i in range(len(segs) - 1):
                pair = f"{segs[i]} > {segs[i + 1]}"
                if pair in norm_context or pair in norm_whitelist:
                    continue
                # Fallback: both segment names exist verbatim in the retrieved context
                if not (segs[i] in norm_context and segs[i + 1] in norm_context):
                    all_pairs_grounded = False
                    break

            if not all_pairs_grounded:
                unverified_paths.append(raw_path)

        if unverified_paths:
            formatted_list = ", ".join(f"`{p}`" for p in unverified_paths)
            errors.append(
                f"Unverified Sapaad menu path(s) not found in retrieved sources: {formatted_list}. "
                "Verify in Sapaad before sending."
            )

    return {
        "draft": text,
        "errors": errors,
        "warnings": warnings,
        "auto_fixes": auto_fixes,
    }
