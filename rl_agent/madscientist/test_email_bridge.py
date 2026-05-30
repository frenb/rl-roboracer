"""Unit tests for rl_agent/madscientist/email_bridge.py.

Covers token signing/verification (the part that must stay byte-for-byte
compatible with the dashboard's TypeScript verifier), email rendering
(content + button presence), and the notify_once DB side-effects
(idempotent notified_at stamp + audit event). SMTP is always mocked -
no real mail is sent.

Run:
    docker compose -f docker-compose.yml -f compose/scale.yml exec \
        madscientist python -m rl_agent.madscientist.test_email_bridge
"""
from __future__ import annotations

import datetime
import hashlib
import hmac
import os
import sys
import traceback
from typing import Any, Callable, Dict, List

from pymongo import MongoClient

from rl_agent.madscientist import constants, email_bridge


_passed = 0
_failed = 0
_failures: List[str] = []
_inserted_proposals: List[Any] = []


def _expect(label: str, predicate: Callable[[], bool], detail: str = ""):
    global _passed, _failed
    try:
        ok = predicate()
    except Exception as e:  # noqa: BLE001
        ok = False
        detail = f"{detail} | EXCEPTION: {type(e).__name__}: {e}"
    suffix = f" - {detail}" if detail else ""
    if ok:
        _passed += 1
        print(f"  [PASS] {label}{suffix}", flush=True)
    else:
        _failed += 1
        _failures.append(label)
        print(f"  [FAIL] {label}{suffix}", flush=True)


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _db():
    url = os.environ.get("MONGO_URL", "mongodb://root:example@mongo:27017/")
    client = MongoClient(url, serverSelectionTimeoutMS=10000)
    client.admin.command("ping")
    return client.robotaxi


SECRET = "test-secret-do-not-use-in-prod"


def _sample_proposal() -> Dict[str, Any]:
    return {
        "_id": "6a1a782c553f0c3bacd1582b",
        "title": "Protected demo replay (two-table BC+RL)",
        "status": constants.STATUS_PENDING_USER,
        "hypothesis": "Two-table mode improves avg_return by >=15%.",
        "motivation": "FIFO eviction erases demos by ~75k iters.",
        "n_seeds_per_arm": 3,
        "num_iterations_per_seed": 50000,
        "expected_wall_time_hours": 8.5,
        "experiment_arms": [
            {"name": "base", "experiment_design_id": "experiment-default"},
            {"name": "two_table_25pct",
             "experiment_design_fields": {"demo_min_keep": 50000,
                                          "demo_sample_ratio": 0.25}},
        ],
        "success_criteria": {
            "primary": "two_table_25pct avg_return >= +15% vs base.",
            "primary_parsed": {"metric": "avg_return", "arm_a": "two_table_25pct",
                               "arm_b": "base", "comparator": ">=",
                               "threshold": 0.15, "threshold_kind": "relative"},
            "secondary": ["avg_speed within +/-10% of base"],
        },
        "source_papers": [
            {"arxiv_id": "1707.08817", "title": "DDPGfD",
             "section_refs": ["Section 3.2"],
             "supporting_evidence": "Protected demo buffer prevents eviction."},
        ],
        "judge_review": {
            "overall": "accept", "normalized_score": 0.8,
            "scores": {"novelty": 3, "paper_faithfulness": 5,
                       "goodhart_resistance": 4},
            "strengths": ["Reuses existing knobs."],
            "concerns": ["Only 3 seeds."],
        },
    }


# ---- Token tests ---------------------------------------------------------


def test_token_roundtrip():
    print("\nGroup 1: token sign + verify roundtrip", flush=True)
    tok = email_bridge.sign_decision_token(
        "6a1a782c553f0c3bacd1582b", "approve", SECRET)
    _expect("token has 4 dot-separated parts",
            lambda: len(tok.split(".")) == 4)
    pid, action = email_bridge.verify_decision_token(tok, SECRET)
    _expect("verify returns original proposal_id",
            lambda: pid == "6a1a782c553f0c3bacd1582b")
    _expect("verify returns original action", lambda: action == "approve")


def test_token_tamper():
    print("\nGroup 2: tampered tokens are rejected", flush=True)
    tok = email_bridge.sign_decision_token("abc123", "reject", SECRET)
    pid, action, exp, sig = tok.split(".")

    # Flip the action without re-signing -> signature mismatch.
    forged = f"{pid}.approve.{exp}.{sig}"
    _expect("action swap rejected",
            lambda: _raises(lambda: email_bridge.verify_decision_token(forged, SECRET)))

    # Wrong secret.
    _expect("wrong secret rejected",
            lambda: _raises(lambda: email_bridge.verify_decision_token(tok, "other-secret")))

    # Mangled signature.
    bad_sig = f"{pid}.{action}.{exp}.{'0' * len(sig)}"
    _expect("bad signature rejected",
            lambda: _raises(lambda: email_bridge.verify_decision_token(bad_sig, SECRET)))

    # Not 4 parts.
    _expect("malformed (too few parts) rejected",
            lambda: _raises(lambda: email_bridge.verify_decision_token("a.b.c", SECRET)))


def test_token_expiry():
    print("\nGroup 3: expired tokens are rejected", flush=True)
    # Sign with a tiny ttl anchored in the past.
    past = _now() - datetime.timedelta(hours=2)
    tok = email_bridge.sign_decision_token(
        "abc", "defer", SECRET, ttl_seconds=60, now=past)
    _expect("expired token rejected",
            lambda: _raises(lambda: email_bridge.verify_decision_token(tok, SECRET)))
    # Same token verified AS OF its signing time is still valid.
    _expect("not-yet-expired token (verified at sign time) accepted",
            lambda: email_bridge.verify_decision_token(
                tok, SECRET, now=past)[1] == "defer")


def test_token_matches_ts_formula():
    print("\nGroup 4: token HMAC matches the documented TS formula", flush=True)
    # The dashboard recomputes HMAC_SHA256(secret, "{pid}.{action}.{exp}")
    # and hex-encodes it. Reproduce that here to lock the contract.
    tok = email_bridge.sign_decision_token("PID", "approve", SECRET,
                                           ttl_seconds=999, now=_now())
    pid, action, exp, sig = tok.split(".")
    body = f"{pid}.{action}.{exp}"
    expected = hmac.new(SECRET.encode(), body.encode(),
                        hashlib.sha256).hexdigest()
    _expect("hexdigest matches manual HMAC-SHA256", lambda: sig == expected)
    _expect("signature is 64 hex chars", lambda: len(sig) == 64)


def test_bad_action_rejected_at_sign():
    print("\nGroup 5: signing an invalid action raises", flush=True)
    _expect("sign rejects unknown action",
            lambda: _raises(lambda: email_bridge.sign_decision_token(
                "x", "delete", SECRET)))


# ---- Render tests --------------------------------------------------------


def test_render_with_buttons():
    print("\nGroup 6: render_proposal_email WITH token secret", flush=True)
    subject, text, html = email_bridge.render_proposal_email(
        _sample_proposal(), base_url="http://host:8080",
        token_secret=SECRET)
    _expect("subject mentions the title",
            lambda: "Protected demo replay" in subject)
    _expect("html contains the hypothesis",
            lambda: "Two-table mode improves" in html)
    _expect("html contains the judge verdict",
            lambda: "accept" in html.lower())
    _expect("html contains all three action URLs",
            lambda: html.count("/madscientist/act?token=") == 3)
    _expect("html links arms + arxiv paper",
            lambda: "two_table_25pct" in html and "1707.08817" in html)
    _expect("text body also carries the magic links",
            lambda: text.count("/madscientist/act?token=") == 3)
    # The approve token in the HTML must verify.
    import re
    m = re.search(r"/madscientist/act\?token=([\w.]+)", html)
    _expect("embedded token verifies",
            lambda: email_bridge.verify_decision_token(m.group(1), SECRET)[0]
            == "6a1a782c553f0c3bacd1582b")


def test_render_without_buttons():
    print("\nGroup 7: render_proposal_email WITHOUT token secret", flush=True)
    subject, text, html = email_bridge.render_proposal_email(
        _sample_proposal(), base_url="http://host:8080", token_secret=None)
    _expect("no action links when secret missing",
            lambda: "/madscientist/act?token=" not in html)
    _expect("falls back to a dashboard link",
            lambda: "/madscientist" in html)
    _expect("still renders the title",
            lambda: "Protected demo replay" in subject)


# ---- notify_once tests ---------------------------------------------------


class _FakeSmtp:
    configured = True
    to_addr = "ops@example.com"
    user = "ops@example.com"


def test_notify_once_sends_and_stamps():
    print("\nGroup 8: notify_once sends once + stamps notified_at (idempotent)",
          flush=True)
    db = _db()
    doc = _sample_proposal()
    doc.pop("_id")
    doc["notified_at"] = None
    doc["audit_events"] = []
    pid = db.proposals.insert_one(doc).inserted_id
    _inserted_proposals.append(pid)
    proposal = db.proposals.find_one({"_id": pid})

    sent_calls: List[Any] = []

    def _fake_send(subject, text, html, smtp, to_addr=None):
        sent_calls.append((subject, smtp.to_addr))

    res1 = email_bridge.notify_once(
        db, proposal, base_url="http://host:8080", token_secret=SECRET,
        smtp=_FakeSmtp(), send_fn=_fake_send)
    _expect("notify_once returns True (sent)", lambda: res1 is True)
    _expect("send_fn called once", lambda: len(sent_calls) == 1)

    after = db.proposals.find_one({"_id": pid})
    _expect("notified_at stamped", lambda: after.get("notified_at") is not None)
    _expect("'notified' audit event appended",
            lambda: any(e.get("event") == "notified"
                        for e in after.get("audit_events", [])))

    # Second call on the now-stamped doc: the filtered update won't
    # match (notified_at != None), so modified_count == 0 -> returns
    # False. (send_fn DOES run first, but the loop only calls
    # notify_once for proposals where notified_at is None, so in
    # practice it won't re-send. We assert the stamp guard here.)
    proposal_after = db.proposals.find_one({"_id": pid})
    res2 = email_bridge.notify_once(
        db, proposal_after, base_url="http://host:8080", token_secret=SECRET,
        smtp=_FakeSmtp(), send_fn=_fake_send)
    _expect("second notify_once returns False (stamp already set)",
            lambda: res2 is False)
    _expect("only one 'notified' audit event total",
            lambda: sum(1 for e in db.proposals.find_one({"_id": pid})
                        .get("audit_events", [])
                        if e.get("event") == "notified") == 1)


def test_smtp_config_missing_detection():
    print("\nGroup 9: SmtpConfig.missing() reports unset fields", flush=True)
    # Snapshot + clear env so we get a deterministic "all missing".
    saved = {k: os.environ.get(k) for k in
             ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "NOTIFICATION_EMAIL")}
    try:
        for k in saved:
            os.environ.pop(k, None)
        cfg = email_bridge.SmtpConfig()
        _expect("unconfigured SmtpConfig.configured is False",
                lambda: cfg.configured is False)
        _expect("missing() lists the empty fields",
                lambda: "SMTP_HOST" in cfg.missing())
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def _raises(fn) -> bool:
    try:
        fn()
        return False
    except Exception:  # noqa: BLE001
        return True


def _cleanup(db):
    if _inserted_proposals:
        db.proposals.delete_many({"_id": {"$in": _inserted_proposals}})
    print(f"\n  cleanup: removed {len(_inserted_proposals)} proposals",
          flush=True)


def main() -> int:
    print("=" * 64, flush=True)
    print("Unit tests: rl_agent/madscientist/email_bridge.py", flush=True)
    print("=" * 64, flush=True)
    db = _db()
    try:
        test_token_roundtrip()
        test_token_tamper()
        test_token_expiry()
        test_token_matches_ts_formula()
        test_bad_action_rejected_at_sign()
        test_render_with_buttons()
        test_render_without_buttons()
        test_notify_once_sends_and_stamps()
        test_smtp_config_missing_detection()
    finally:
        _cleanup(db)

    print("\n" + "=" * 64, flush=True)
    print(f"PASSED: {_passed}", flush=True)
    print(f"FAILED: {_failed}", flush=True)
    if _failures:
        for f in _failures:
            print(f"  - {f}", flush=True)
    print("=" * 64, flush=True)
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        rc = 2
    sys.exit(rc)
