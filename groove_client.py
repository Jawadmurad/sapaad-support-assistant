import html
import json
import re
import urllib.error
import urllib.request

GROOVE_GQL_URL = "https://api.groovehq.com/v2/graphql"
GROOVE_REST_URL = "https://api.groovehq.com/v1"

# Verified Groove GraphQL v2 query capturing full EmailMessage, Reply, and summary.bodyPlainText
THREAD_QUERY = """
query TicketConversationQuery($eventGroupsFilter: EventGroupsFilterInput!, $last: Int, $before: String) {
  eventGroups(filter: $eventGroupsFilter, last: $last, before: $before) {
    edges {
      node {
        isNote
        summary {
          id
          body
          bodyPlainText
          createdAt
          author {
            __typename
            ... on Agent { name }
            ... on Contact { name }
          }
        }
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


def _strip_quoted_history(text: str) -> str:
    """Remove repeated quoted email reply chains below standard dividers."""
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


def clean_email_html(body_html: str) -> str:
    """Convert Groove HTML email body into clean plain text."""
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

    return _strip_quoted_history(text)


def _pick_best_body(*candidates: tuple[str, bool]) -> str:
    """
    Evaluate (text, is_html) candidates and return the longest, most complete cleaned message.
    Prevents truncated summary.body HTML from overriding complete summary.bodyPlainText.
    """
    best = ""
    for raw, is_html in candidates:
        if not raw:
            continue
        cleaned = clean_email_html(raw) if is_html else _strip_quoted_history(html.unescape(raw))
        if len(cleaned) > len(best):
            best = cleaned
    return best


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
                "Groove session token expired (HTTP 401). Open Groove or click your ✨ Sapaad AI bookmark/extension to refresh."
            ) from e
        err_body = e.read().decode("utf-8", errors="ignore")[:250]
        raise RuntimeError(f"Groove GraphQL HTTP {e.code}: {err_body}") from e


def _fetch_via_graphql(
    ticket_num: str,
    bearer_token: str,
    selected_ids: list[str] | None = None,
    selected_indices: list[int] | None = None,
) -> dict:
    conv_id = f"cnv_{ticket_num}"
    subject = ""
    customer_name = ""
    state = ""

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

    all_messages = []
    before = None
    for _ in range(5):
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
            grp_summary = group.get("summary") or {}
            grp_author = grp_summary.get("author") or {}
            author_name = grp_author.get("name") or ""
            author_type = grp_author.get("__typename") or ""

            found_in_events = False
            for ev_edge in (group.get("events") or {}).get("edges") or []:
                ev = ev_edge.get("node") or {}
                change = ev.get("change") or {}
                ctype = change.get("__typename")
                if ctype in ("EmailMessage", "Reply"):
                    found_in_events = True
                    is_note = is_group_note or bool(change.get("isNote"))
                    role = "note" if is_note else ("agent" if ctype == "Reply" else "customer")

                    body_clean = _pick_best_body(
                        (change.get("body") or "", True),
                        (grp_summary.get("body") or "", True),
                        (grp_summary.get("bodyPlainText") or "", False),
                    )
                    if body_clean:
                        batch.append({
                            "id": str(change.get("id") or grp_summary.get("id") or ""),
                            "role": role,
                            "author": author_name,
                            "created_at": (change.get("createdAt") or ev.get("createdAt") or "")[:16].replace("T", " "),
                            "text": body_clean,
                        })

            if not found_in_events and (grp_summary.get("body") or grp_summary.get("bodyPlainText")):
                role = "note" if is_group_note else ("agent" if author_type == "Agent" else "customer")
                body_clean = _pick_best_body(
                    (grp_summary.get("bodyPlainText") or "", False),
                    (grp_summary.get("body") or "", True),
                )
                if body_clean:
                    batch.append({
                        "id": str(grp_summary.get("id") or ""),
                        "role": role,
                        "author": author_name,
                        "created_at": (grp_summary.get("createdAt") or "")[:16].replace("T", " "),
                        "text": body_clean,
                    })

        all_messages = batch + all_messages
        page_info = eg.get("pageInfo") or {}
        if page_info.get("hasPreviousPage") and page_info.get("startCursor"):
            before = page_info["startCursor"]
        else:
            break

    return _package_ticket(
        ticket_num,
        subject,
        customer_name,
        state,
        all_messages,
        selected_ids=selected_ids,
        selected_indices=selected_indices,
    )


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
            m = re.search(r"/tickets/(\d+)", path)
            raise RuntimeError(f"Ticket #{m.group(1) if m else '?'} not found in Groove.") from e
        err_body = e.read().decode("utf-8", errors="ignore")[:250]
        raise RuntimeError(f"Groove REST HTTP {e.code}: {err_body}") from e


def _fetch_via_rest_v1(
    ticket_num: str,
    api_token: str,
    selected_ids: list[str] | None = None,
    selected_indices: list[int] | None = None,
) -> dict:
    t_data = _rest_get(api_token, f"/tickets/{ticket_num}")
    ticket = t_data.get("ticket") or {}
    subject = ticket.get("title") or ticket.get("subject") or ""
    state = ticket.get("state") or ""

    m_data = _rest_get(api_token, f"/tickets/{ticket_num}/messages?per_page=50")
    raw_msgs = m_data.get("messages") or []
    raw_msgs.sort(key=lambda m: m.get("created_at") or "")

    messages = []
    for m in raw_msgs:
        is_note = bool(m.get("note"))
        author_href = ((m.get("links") or {}).get("author") or {}).get("href") or ""
        role = "note" if is_note else ("agent" if "/agents/" in author_href else "customer")
        body_clean = _pick_best_body(
            (m.get("body") or "", True),
            (m.get("plain_text_body") or "", False),
        )
        if body_clean:
            messages.append({
                "id": str(m.get("id") or ""),
                "role": role,
                "author": "",
                "created_at": (m.get("created_at") or "")[:16].replace("T", " "),
                "text": body_clean,
            })

    return _package_ticket(
        ticket_num,
        subject,
        "",
        state,
        messages,
        selected_ids=selected_ids,
        selected_indices=selected_indices,
    )


def _package_ticket(
    ticket_num: str,
    subject: str,
    customer_name: str,
    state: str,
    messages: list[dict],
    selected_ids: list[str] | None = None,
    selected_indices: list[int] | None = None,
) -> dict:
    if not messages:
        raise RuntimeError(
            f"Ticket #{ticket_num} returned 0 messages. Check the ticket number and your Groove token."
        )

    total_message_count = len(messages)
    filtered_messages = messages
    explicit_selection = False

    if selected_ids:
        id_set = {s.strip() for s in selected_ids if s.strip()}
        matched = [m for m in messages if m.get("id") and m["id"] in id_set]
        if matched:
            filtered_messages = matched
            explicit_selection = len(matched) < total_message_count
    elif selected_indices is not None:
        idx_set = {i for i in selected_indices if 0 <= i < total_message_count}
        matched = [messages[i] for i in sorted(idx_set)]
        if matched:
            filtered_messages = matched
            explicit_selection = len(matched) < total_message_count

    if explicit_selection:
        draft_msgs = filtered_messages
    else:
        draft_msgs = [m for m in filtered_messages if m["role"] in ("customer", "agent")]
        if not draft_msgs:
            draft_msgs = filtered_messages

    latest_cust_idx = None
    for idx in range(len(draft_msgs) - 1, -1, -1):
        if draft_msgs[idx]["role"] == "customer":
            latest_cust_idx = idx
            break
    if latest_cust_idx is None:
        latest_cust_idx = len(draft_msgs) - 1

    latest_msg = draft_msgs[latest_cust_idx]
    header_parts = [f"Ticket #{ticket_num}"]
    if subject:
        header_parts.append(f"Subject: {subject}")
    if customer_name:
        header_parts.append(f"Customer: {customer_name}")
    header_line = " | ".join(header_parts)

    latest_customer_email = f"{header_line}\n\n{latest_msg['text']}"

    earlier_blocks = []
    for idx, m in enumerate(draft_msgs):
        if idx == latest_cust_idx:
            continue
        base_who = {"customer": "Customer", "agent": "Sapaad Agent", "note": "Internal Note"}.get(
            m["role"], "Message"
        )
        who = f"{base_who} ({m['author']})" if m.get("author") else base_who
        earlier_blocks.append(f"[{who} — {m['created_at']}]\n{m['text']}")
    previous_thread = "\n\n---\n\n".join(earlier_blocks)

    full_blocks = [header_line]
    for m in filtered_messages:
        base_who = {"customer": "Customer", "agent": "Sapaad Agent", "note": "Internal Note"}.get(
            m["role"], "Message"
        )
        who = f"{base_who} ({m['author']})" if m.get("author") else base_who
        full_blocks.append(f"[{who} — {m['created_at']}]\n{m['text']}")
    full_thread = "\n\n---\n\n".join(full_blocks)

    return {
        "ticket_number": ticket_num,
        "subject": subject,
        "customer_name": customer_name,
        "state": state,
        "message_count": len(filtered_messages),
        "total_message_count": total_message_count,
        "selected_subset": explicit_selection,
        "latest_customer_email": latest_customer_email,
        "previous_thread": previous_thread,
        "full_thread": full_thread,
    }


def fetch_groove_ticket(
    ticket_input: str,
    groove_api_token: str = "",
    groove_bearer_token: str = "",
    selected_ids: list[str] | None = None,
    selected_indices: list[int] | None = None,
) -> dict:
    """Fetch a full Groove ticket by ticket number using REST v1 (permanent token) or GraphQL v2 (Bearer token)."""
    num_match = re.search(r"\d+", str(ticket_input or ""))
    if not num_match:
        raise ValueError("Please enter a valid numeric Groove ticket number (e.g., 114201).")
    ticket_num = num_match.group(0)

    if groove_api_token.strip():
        return _fetch_via_rest_v1(ticket_num, groove_api_token, selected_ids=selected_ids, selected_indices=selected_indices)
    if groove_bearer_token.strip():
        return _fetch_via_graphql(ticket_num, groove_bearer_token, selected_ids=selected_ids, selected_indices=selected_indices)

    raise RuntimeError(
        "No Groove token configured. Open Groove with the extension or click your ✨ Sapaad AI bookmark to connect automatically."
    )
