"""
Sapaad Support AI — Golden Regression Suite (eval.py)
Run fast 0-token checks (retrieval + verifier + privacy):
    python eval.py
"""
import os
import sys

from privacy import restore_pii, scrub_pii
from retrieval import (
    build_search_index,
    retrieve_context,
    should_bypass_llm_query_extraction,
)
from verifier import verify_and_repair_draft

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = os.environ.get("HF_DATASET_REPO", "JawadMurad/sapaad-support-data")
CF_ACCOUNT_ID = os.environ.get("CF_ACCOUNT_ID", "")
CF_API_TOKEN = os.environ.get("CF_API_TOKEN", "")
LOCAL_DATA_DIR = "data"


def run_eval() -> int:
    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal passed, failed
        if condition:
            print(f"[PASS] {name}")
            passed += 1
        else:
            print(f"[FAIL] {name} -> {detail}")
            failed += 1

    print("=== 1. Privacy Scrubber Tests ===")
    raw = "Hi, email owner@mycafe.ae, phone +971 50 123 4567, support@sapaad.com"
    [scrubbed], pii_map = scrub_pii([raw])
    check("Masks customer email & phone", "owner@mycafe.ae" not in scrubbed and "123 4567" not in scrubbed, scrubbed)
    check("Keeps @sapaad.com unmasked", "support@sapaad.com" in scrubbed, scrubbed)
    check("Restores PII cleanly", restore_pii(scrubbed, pii_map) == raw, restore_pii(scrubbed, pii_map))

    print("\n=== 2. Verifier & Auto-Repair Tests ===")
    bad_cpa = (
        "Hi,\n\nDownload CPA here: "
        "[https://www.sapaad.com/software-updates/cloud-print-agent-v3/updates/dl/latest-build/CloudPrintSetup/CloudPrintSetup.msi?cache=236842](https://www.sapaad.com/software-updates/cloud-print-agent-v3/updates/dl/latest-build/CloudPrintSetup/CloudPrintSetup.msi?cache=236842) d2\n\n"
        "If you have any other questions, feel free to let us know.\n\n"
        "Please feel free to write to us for any further assistance.\n\nBest regards,\nSapaad Support Team"
    )
    rep1 = verify_and_repair_draft(bad_cpa, "Cloud Print Agent setup", "Draft Reply", "English")
    check("Replaces broken .msi CPA link", "[https://prints.sapaad.com/getcpa](https://prints.sapaad.com/getcpa)" in rep1["draft"], rep1["draft"])
    check("Strips duplicate closing & sign-off", "Best regards" not in rep1["draft"] and "other questions" not in rep1["draft"], rep1["draft"])

    trap_draft = "Hi,\n\nVisit Setup > Location Setup > Surcharges and install TeamViewer.\n\nPlease feel free to write to us for any further assistance."
    rep2 = verify_and_repair_draft(trap_draft, "Setup > Surcharges", "Draft Reply", "English")
    check("Flags hallucinated Surcharges path & TeamViewer", len(rep2["errors"]) >= 2, str(rep2["errors"]))

    ar_draft = "واختروا Setup > Location Setup ثم قموا بالحفظ."
    rep3 = verify_and_repair_draft(ar_draft, "Setup > Location Setup", "Draft Reply", "Arabic")
    check("Fixes Arabic greeting & plural typos", rep3["draft"].startswith("السلام عليكم،") and "واختاروا" in rep3["draft"], rep3["draft"])

    print("\n=== 3. Smart Bypass & Retrieval Tests ===")
    check("Smart Bypass ON for short single EN email", should_bypass_llm_query_extraction("My receipt machine is not printing kitchen slips"))
    check("Smart Bypass OFF for inline 5-point list", not should_bypass_llm_query_extraction("Help with: 1) QR codes, 2) Call waiter, 3) Guest checkout, 4) Scheduled orders, 5) Delivery fees."))
    check("Smart Bypass OFF for Arabic email", not should_bypass_llm_query_extraction("السلام عليكم، ما هي طريقة تقييم المخزون؟"))

    idx = build_search_index(LOCAL_DATA_DIR, HF_REPO, HF_TOKEN)
    if not idx.get("docs"):
        print("[SKIP] Retrieval index tests skipped (set HF_TOKEN or add ./data/*.md to run).")
    else:
        print(f"Loaded index: {idx['stats']} | Hybrid ready: {idx['hybrid_ready']}")
        tests = [
            ("Synonym & typo: receipt machine + kitchen slips", "My receipt machine is not printing kitchen slips on saapad", ["Printing.md", "Tickets_Printing_KDS_Hardware.md"]),
            ("Talabat menu flush (no signature hijack)", "How do we flush our menu on Talabat?", ["Tickets_Aggregators_UrbanPiper.md"]),
            ("Policy priority: Open API & Vantage Pro", "Can we integrate our custom website with Sapaad via API, and how much is Sapaad Vantage Pro?", ["policies.md"]),
        ]
        for label, q, expected_sources in tests:
            matches, meta = retrieve_context(
                idx,
                q,
                extract_subqueries_fn=lambda t: ([t], "eval-mock"),
                top_k=5,
                cf_account_id=CF_ACCOUNT_ID,
                cf_api_token=CF_API_TOKEN,
            )
            sources = [m["source"] for m in matches]
            hit = any(es in sources for es in expected_sources)
            check(f"Retrieval: {label}", hit, f"Got sources={sources[:4]}, meta={meta}")

    print(f"\n=== SUMMARY: {passed} passed, {failed} failed ===")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_eval())
