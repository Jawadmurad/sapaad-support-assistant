import html
import json
import re
import urllib.error
import urllib.request

GROOVE_GQL_URL = "https://api.groovehq.com/v2/graphql"
GROOVE_REST_URL = "https://api.groovehq.com/v1"

# Verified working Groove GraphQL v2 query for Sapaad's workspace
THREAD_QUERY = """
query TicketConversationQuery($eventGroupsFilter: EventGroupsFilterInput!, $last: Int, $before: String) {
  eventGroups(filter: $eventGroupsFilter, last: $last, before: $before) {
    edges {
      node {
        isNote
        events {
          edges {
            node {
              createdAt
              change {
                __typename
                ... on EmailMessage {
                  id
                  body
                  isNote
                  createdAt
                }
                ... on Reply {
                  id
                  body
                  isNote
                  createdAt
                }
              }
            }
          }
        }
      }
    }
    pageInfo {
      hasPreviousPage
      startCursor
    }
  }
}
"""

# Optional lightweight metadata query (subject + contact name)
META_QUERY = """
query TicketMeta($id: ID!) {
  conversation(id: $id) {
    number
    subject
    state
    contact { name email }
  }
}
"""


def clean_email_html(body_html: str) -> str:
    """Convert Groove HTML email body into clean plain text and strip repeated quoted reply chains."""
    if not body_html:
        return ""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(body_html, "html.parser")
        for tag in soup(["script", "style", "blockquote"]):
            tag.decompose()
        for br in soup.find_all("br"):
            br.replace_with("\n")
        for p in soup.find_all(["p", "div", "li"]):
            p.append("\n")
        text = soup.get_text()
    except ImportError:
        text = re.sub(r"(?i)<br\s*/?>", "\n", body_html)
        text = re.sub(r"(?i)</(p|div|li)>", "\n", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = html.unescape(text)

    # Strip quoted history below common email reply dividers
    cut_patterns = [
        r"(?im)^\s*On\s+.+?wrote:\s*$",
        r"(?im)^\s*-{3,}\s*Original Message\s*-{3,}",
        r"(?im)^\s*From:\s+.+\n\s*Sent:\s+",
        r"(?im)^\s*_{10,}\s*$",
    ]
    for pat in cut_patterns:
        m = re.search(pat, text)
        if m and m.start() > 25:
            text = text[: m.start()]

    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _gql_post(token: str, query_name: str, payload: dict) -> dict:
    url = f"{GROOVE_GQL_URL}?_{query_name}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "accept": "*/*",
            "authorization": f"Bearer {token.strip()}",
            "content-type": "application/json",
            "origin": "https://sapaad.groovehq.com",
            "referer": "https://sapaad.groovehq.com/",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise RuntimeError(
                "Groove session token expired (HTTP 401). Grab a fresh Bearer token from DevTools "
                "and paste it in the sidebar 'Groove Connection' box or update GROOVE_BEARER_TOKEN in Secrets."
            ) from e
        err_body = e.read().decode("utf-8", errors="ignore")[:250]
        raise RuntimeError(f"Groove GraphQL HTTP {e.code}: {err_body}") from e


def _fetch_via_graphql(ticket_num: str, bearer_token: str) -> dict:
    conv_id = f"cnv_{ticket_num}"
    subject = ""
    customer_name = ""
    state = ""

    # 1. Best-effort metadata fetch (never fails the ticket if schema differs)
    try:
        meta_res = _gql_post(
            bearer_token,
            "TicketMeta",
            {"query": META_QUERY, "_method": "POST", "variables": {"id": conv_id}},
        )
        conv = (meta_res.get("data") or {}).get("conversation") or {}
        subject = conv.get("subject") or ""
        state = conv.get("state") or ""
        contact = conv.get("contact") or {}
        customer_name = contact.get("name") or ""
    except Exception:
        pass

    # 2. Fetch full chronological thread via verified THREAD_QUERY
    all_messages = []
    before = None
    for _ in range(5):  # Up to 250 events safety cap
        variables = {
            "eventGroupsFilter": {"conversationId": conv_id},
            "last": 50,
        }
        if before:
            variables["before"] = before

        data = _gql_post(
            bearer_token,
            "TicketConversationQuery",
            {"query": THREAD_QUERY, "_method": "POST", "variables": variables},
        )
        if "errors" in data:
            raise RuntimeError(f"Groove GraphQL Error: {data['errors'][0].get('message', data['errors'])}")

        eg = (data.get("data") or {}).get("eventGroups") or {}
        batch = []
        for edge in eg.get("edges") or []:
            group = edge.get("node") or {}
            is_group_note = bool(group.get("isNote"))
            for ev_edge in (group.get("events") or {}).get("edges") or []:
                ev = ev_edge.get("node") or {}
                change = ev.get("change") or {}
                ctype = change.get("__typename")
                if ctype in ("EmailMessage", "Reply"):
                    is_note = is_group_note or bool(change.get("isNote"))
                    role = "note" if is_note else ("customer" if ctype == "EmailMessage" else "agent")
                    body_clean = clean_email_html(change.get("body") or "")
                    if body_clean:
                        batch.append({
                            "role": role,
                            "created_at": (change.get("createdAt") or ev.get("createdAt") or "")[:16].replace("T", " "),
                            "text": body_clean,
                        })

        all_messages = batch + all_messages
        page_info = eg.get("pageInfo") or {}
        if page_info.get("hasPreviousPage") and page_info.get("startCursor"):
            before = page_info["startCursor"]
        else:
            break

    return _package_ticket(ticket_num, subject, customer_name, state, all_messages)


def _rest_get(api_token: str, path: str) -> dict:
    url = f"{GROOVE_REST_URL}{path}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_token.strip()}",
            "Accept": "application/json",
            "User-Agent": "SapaadSupportAI/1.0",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise RuntimeError(f"Ticket #{ticket_num_from_path(path)} not found in Groove.") from e
        err_body = e.read().decode("utf-8", errors="ignore")[:250]
        raise RuntimeError(f"Groove REST HTTP {e.code}: {err_body}") from e


def ticket_num_from_path(path: str) -> str:
    m = re.search(r"/tickets/(\d+)", path)
    return m.group(1) if m else "?"


def _fetch_via_rest_v1(ticket_num: str, api_token: str) -> dict:
    t_data = _rest_get(api_token, f"/tickets/{ticket_num}")
    ticket = t_data.get("ticket") or {}
    subject = ticket.get("title") or ticket.get("subject") or ""
    state = ticket.get("state") or ""

    m_data = _rest_get(api_token, f"/tickets/{ticket_num}/messages?per_page=50")
    raw_msgs = m_data.get("messages") or []
    # REST API often returns newest first; sort chronologically by created_at
    raw_msgs.sort(key=lambda m: m.get("created_at") or "")

    messages = []
    for m in raw_msgs:
        is_note = bool(m.get("note"))
        author_href = ((m.get("links") or {}).get("author") or {}).get("href") or ""
        role = "note" if is_note else ("agent" if "/agents/" in author_href else "customer")
        body_clean = clean_email_html(m.get("body") or m.get("plain_text_body") or "")
        if body_clean:
            messages.append({
                "role": role,
                "created_at": (m.get("created_at") or "")[:16].replace("T", " "),
                "text": body_clean,
            })

    return _package_ticket(ticket_num, subject, "", state, messages)


def _package_ticket(
    ticket_num: str,
    subject: str,
    customer_name: str,
    state: str,
    messages: list[dict],
) -> dict:
    if not messages:
        raise RuntimeError(
            f"Ticket #{ticket_num} returned 0 messages. Check the ticket number and your Groove token."
        )

    # Separate public email thread from internal notes
    public_msgs = [m for m in messages if m["role"] in ("customer", "agent")]
    if not public_msgs:
        public_msgs = messages

    # Latest customer message goes into the main Customer Email box
    latest_cust_idx = None
    for idx in range(len(public_msgs) - 1, -1, -1):
        if public_msgs[idx]["role"] == "customer":
            latest_cust_idx = idx
            break
    if latest_cust_idx is None:
        latest_cust_idx = len(public_msgs) - 1

    latest_msg = public_msgs[latest_cust_idx]
    header_parts = [f"Ticket #{ticket_num}"]
    if subject:
        header_parts.append(f"Subject: {subject}")
    if customer_name:
        header_parts.append(f"Customer: {customer_name}")
    header_line = " | ".join(header_parts)

    latest_customer_email = f"{header_line}\n\n{latest_msg['text']}"

    # Earlier replies go into Previous Replies / Already Tried
    earlier_blocks = []
    for idx, m in enumerate(public_msgs):
        if idx == latest_cust_idx:
            continue
        who = "Customer" if m["role"] == "customer" else "Sapaad Agent"
        earlier_blocks.append(f"[{who} — {m['created_at']}]\n{m['text']}")
    previous_thread = "\n\n---\n\n".join(earlier_blocks)

    # Full chronological thread (including internal notes) for Internal Note / Summary mode
    full_blocks = [header_line]
    for m in messages:
        who = {"customer": "Customer", "agent": "Sapaad Agent", "note": "Internal Note"}.get(m["role"], "Message")
        full_blocks.append(f"[{who} — {m['created_at']}]\n{m['text']}")
    full_thread = "\n\n---\n\n".join(full_blocks)

    return {
        "ticket_number": ticket_num,
        "subject": subject,
        "customer_name": customer_name,
        "state": state,
        "message_count": len(messages),
        "latest_customer_email": latest_customer_email,
        "previous_thread": previous_thread,
        "full_thread": full_thread,
    }


def fetch_groove_ticket(
    ticket_input: str,
    groove_api_token: str = "",
    groove_bearer_token: str = "",
) -> dict:
    """Fetch a full Groove ticket by ticket number using REST v1 (permanent token) or GraphQL v2 (Bearer token)."""
    num_match = re.search(r"\d+", str(ticket_input or ""))
    if not num_match:
        raise ValueError("Please enter a valid numeric Groove ticket number (e.g., 114201).")
    ticket_num = num_match.group(0)

    if groove_api_token.strip():
        return _fetch_via_rest_v1(ticket_num, groove_api_token)
    if groove_bearer_token.strip():
        return _fetch_via_graphql(ticket_num, groove_bearer_token)

    raise RuntimeError(
        "No Groove token configured. Add `GROOVE_BEARER_TOKEN` (or `GROOVE_API_TOKEN`) in Streamlit Secrets, "
        "or paste your session Bearer token in the left sidebar under '🔗 Groove Connection'."
    )
