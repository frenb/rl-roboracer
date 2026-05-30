"""MadScientist email bridge - outbound proposal notifications.

When the Judge promotes a proposal to ``pending_user`` this worker emails
the operator a fully-formatted summary of the proposal (hypothesis, judge
review + per-axis scores, experiment arms, success criteria, cited
papers) plus three signed "magic-link" buttons - Approve / Reject /
Defer - that point at the dashboard's GET /madscientist/act endpoint.

Design notes
------------
* Outbound only. Inbound replies are NOT parsed here; the response path
  is the magic-link buttons, which reuse the dashboard's existing
  decision logic. (A future Gmail-API reply-parser could add a second
  inbound channel; the schema's UserDecision.source already enumerates
  'email'.)

* Idempotent. The loop only emails proposals in ``pending_user`` that
  don't yet carry a ``notified_at`` stamp; sending stamps it (+ an audit
  event) so a proposal is never emailed twice.

* Magic-link tokens are HMAC-SHA256 signed over
  ``{proposal_id}.{action}.{exp_unix}`` with MADSCIENTIST_TOKEN_SECRET.
  The dashboard server (dashboard/src/server.ts) recomputes the same
  HMAC to verify - keep the two implementations in sync. Tokens carry
  an expiry; "single use" is enforced server-side by gating the decision
  on the proposal still being in pending_user/deferred.

* SMTP via smtplib using the same SMTP_* env vars proven out by the
  outbound smoke test. If SMTP isn't configured the loop logs once and
  no-ops (never crashes the process).
"""
from __future__ import annotations

import datetime
import hashlib
import hmac
import html
import os
import smtplib
import ssl
import time
import traceback
from email.message import EmailMessage
from typing import Any, Dict, List, Optional, Tuple

from . import constants

# Default poll cadence for the notify loop. New pending_user proposals
# appear at most ~1/day (rate-gated), so a 60s poll is plenty responsive
# without hammering Mongo.
DEFAULT_NOTIFY_POLL_INTERVAL_SECONDS = 60

# Magic-link token lifetime. Generous so a link doesn't expire before
# the operator gets around to clicking it; overridable via env.
DEFAULT_DECISION_TOKEN_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days

# The three actions a magic-link button can carry. Mirrors the
# dashboard's MS_VALID_DECISIONS (minus approve_with_revisions, which
# isn't a one-click action).
_VALID_ACTIONS = ("approve", "reject", "defer")

# Verdict -> (bg, fg) inline-style colors for the email's verdict pill.
# Inline hex (not Tailwind classes) because email clients strip <style>.
_VERDICT_COLORS = {
    "strong_accept": ("#065f46", "#d1fae5"),
    "accept":        ("#047857", "#d1fae5"),
    "weak_accept":   ("#92400e", "#fef3c7"),
    "weak_reject":   ("#9a3412", "#ffedd5"),
    "reject":        ("#991b1b", "#fee2e2"),
}


# ---- Token signing (must stay in sync with server.ts) --------------------


def sign_decision_token(
    proposal_id: str,
    action: str,
    secret: str,
    *,
    ttl_seconds: int = DEFAULT_DECISION_TOKEN_TTL_SECONDS,
    now: Optional[datetime.datetime] = None,
) -> str:
    """Return a signed magic-link token for (proposal_id, action).

    Format: ``{pid}.{action}.{exp}.{hexsig}`` where hexsig =
    HMAC_SHA256(secret, "{pid}.{action}.{exp}"). All segments are
    URL-safe (hex / lowercase ascii / digits) so no extra encoding is
    needed.
    """
    if action not in _VALID_ACTIONS:
        raise ValueError(f"action must be one of {_VALID_ACTIONS}; got {action!r}")
    now = now or datetime.datetime.now(datetime.timezone.utc)
    exp = int(now.timestamp()) + int(ttl_seconds)
    body = f"{proposal_id}.{action}.{exp}"
    sig = hmac.new(secret.encode("utf-8"), body.encode("utf-8"),
                   hashlib.sha256).hexdigest()
    return f"{body}.{sig}"


def verify_decision_token(
    token: str,
    secret: str,
    *,
    now: Optional[datetime.datetime] = None,
) -> Tuple[str, str]:
    """Verify a token + return (proposal_id, action).

    Raises ValueError on malformed / tampered / expired tokens. This is
    the Python-side verifier (used in tests + any future Python inbound
    path); the live dashboard verifies in TypeScript.
    """
    parts = (token or "").split(".")
    if len(parts) != 4:
        raise ValueError("malformed token (expected 4 dot-separated parts)")
    pid, action, exp_str, sig = parts
    body = f"{pid}.{action}.{exp_str}"
    expected = hmac.new(secret.encode("utf-8"), body.encode("utf-8"),
                        hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        raise ValueError("bad signature")
    try:
        exp = int(exp_str)
    except ValueError:
        raise ValueError("bad expiry")
    now = now or datetime.datetime.now(datetime.timezone.utc)
    if int(now.timestamp()) > exp:
        raise ValueError("token expired")
    if action not in _VALID_ACTIONS:
        raise ValueError(f"bad action {action!r}")
    return pid, action


def build_action_url(
    base_url: str,
    proposal_id: str,
    action: str,
    secret: str,
    *,
    ttl_seconds: int = DEFAULT_DECISION_TOKEN_TTL_SECONDS,
    now: Optional[datetime.datetime] = None,
) -> str:
    """Full dashboard URL for a one-click decision."""
    token = sign_decision_token(
        proposal_id, action, secret, ttl_seconds=ttl_seconds, now=now)
    base = base_url.rstrip("/")
    return f"{base}/madscientist/act?token={token}"


# ---- Email rendering -----------------------------------------------------


def _esc(s: Any) -> str:
    return html.escape("" if s is None else str(s))


def _get(obj: Any, name: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _btn(url: str, label: str, bg: str) -> str:
    return (
        f'<a href="{_esc(url)}" '
        f'style="display:inline-block;padding:11px 22px;margin:0 6px 0 0;'
        f'background:{bg};color:#ffffff;text-decoration:none;'
        f'border-radius:8px;font-weight:600;font-size:14px;'
        f'font-family:Arial,sans-serif">{_esc(label)}</a>'
    )


def render_proposal_email(
    proposal: Dict[str, Any],
    *,
    base_url: str,
    token_secret: Optional[str],
    token_ttl_seconds: int = DEFAULT_DECISION_TOKEN_TTL_SECONDS,
    now: Optional[datetime.datetime] = None,
) -> Tuple[str, str, str]:
    """Render (subject, text_body, html_body) for a proposal.

    When ``token_secret`` is falsy the action buttons are omitted and
    the email falls back to a "review on the dashboard" link only (the
    notify loop still sends - the operator can act from the dashboard).
    """
    pid = str(_get(proposal, "_id", ""))
    title = _get(proposal, "title", "(untitled proposal)")
    hypothesis = _get(proposal, "hypothesis", "")
    motivation = _get(proposal, "motivation", "")
    arms = _get(proposal, "experiment_arms") or []
    n_seeds = _get(proposal, "n_seeds_per_arm", "?")
    n_iters = _get(proposal, "num_iterations_per_seed", "?")
    wall = _get(proposal, "expected_wall_time_hours", None)
    sc = _get(proposal, "success_criteria") or {}
    papers = _get(proposal, "source_papers") or []
    judge = _get(proposal, "judge_review") or {}

    verdict = _get(judge, "overall", "pending")
    norm = _get(judge, "normalized_score", None)
    norm_txt = f" ({norm * 100:.0f}%)" if isinstance(norm, (int, float)) else ""
    scores = _get(judge, "scores") or {}
    concerns = _get(judge, "concerns") or []
    strengths = _get(judge, "strengths") or []

    base = (base_url or "").rstrip("/")
    dashboard_link = f"{base}/madscientist"

    # ---- subject ----
    subject = f"[MadScientist] Review needed: {title}"

    # ---- plain-text body (fallback for non-HTML clients) ----
    tlines: List[str] = []
    tlines.append(f"NEW PROPOSAL AWAITING YOUR DECISION")
    tlines.append("=" * 60)
    tlines.append(f"Title: {title}")
    tlines.append(f"Judge verdict: {verdict}{norm_txt}")
    tlines.append("")
    tlines.append("Hypothesis:")
    tlines.append(f"  {hypothesis}")
    if motivation:
        tlines.append("")
        tlines.append("Motivation:")
        tlines.append(f"  {motivation}")
    if scores:
        tlines.append("")
        tlines.append("Judge scores (0-5):")
        for ax in sorted(scores.keys()):
            tlines.append(f"  {ax.replace('_', ' ')}: {scores[ax]}/5")
    if concerns:
        tlines.append("")
        tlines.append("Concerns:")
        for c in concerns:
            tlines.append(f"  - {c}")
    tlines.append("")
    tlines.append("Experiment arms:")
    for a in arms:
        fields = _get(a, "experiment_design_fields") or {}
        ftxt = ", ".join(f"{k}={v}" for k, v in fields.items()) \
            or (_get(a, "experiment_design_id") or "base / no overrides")
        tlines.append(f"  [{_get(a, 'name', '?')}] {ftxt}")
    tlines.append(f"Sizing: {n_seeds} seeds x {n_iters} iters"
                  + (f" (~{wall}h)" if wall else ""))
    primary = _get(sc, "primary", "")
    if primary:
        tlines.append("")
        tlines.append(f"Success criterion: {primary}")
    if papers:
        tlines.append("")
        tlines.append("Source papers:")
        for p in papers:
            aid = _get(p, "arxiv_id", "")
            tlines.append(f"  - {_get(p, 'title', '')} "
                          f"(arXiv:{aid}) https://arxiv.org/abs/{aid}")
    tlines.append("")
    tlines.append("-" * 60)
    if token_secret:
        approve_url = build_action_url(base, pid, "approve", token_secret,
                                       ttl_seconds=token_ttl_seconds, now=now)
        reject_url = build_action_url(base, pid, "reject", token_secret,
                                      ttl_seconds=token_ttl_seconds, now=now)
        defer_url = build_action_url(base, pid, "defer", token_secret,
                                     ttl_seconds=token_ttl_seconds, now=now)
        tlines.append(f"APPROVE: {approve_url}")
        tlines.append(f"REJECT:  {reject_url}")
        tlines.append(f"DEFER:   {defer_url}")
    else:
        approve_url = reject_url = defer_url = None
    tlines.append(f"View on dashboard: {dashboard_link}")
    tlines.append(f"Proposal id: {pid}")
    text_body = "\n".join(tlines)

    # ---- HTML body ----
    vbg, vfg = _VERDICT_COLORS.get(verdict, ("#334155", "#e2e8f0"))
    sans = "font-family:Arial,Helvetica,sans-serif"

    def section(label: str, inner: str) -> str:
        return (
            f'<tr><td style="padding:14px 0 4px 0;{sans};font-size:11px;'
            f'letter-spacing:0.05em;text-transform:uppercase;color:#64748b;'
            f'font-weight:700;border-bottom:1px solid #e2e8f0">{_esc(label)}'
            f'</td></tr>'
            f'<tr><td style="padding:8px 0;{sans};font-size:14px;'
            f'color:#1e293b;line-height:1.55">{inner}</td></tr>'
        )

    # arms table
    arm_rows = ""
    for a in arms:
        fields = _get(a, "experiment_design_fields") or {}
        ftxt = ", ".join(f"{_esc(k)}={_esc(v)}" for k, v in fields.items()) \
            or _esc(_get(a, "experiment_design_id") or "base / no overrides")
        arm_rows += (
            f'<tr><td style="padding:3px 8px 3px 0;vertical-align:top">'
            f'<span style="font-family:monospace;font-size:12px;'
            f'background:#eef2ff;color:#4338ca;padding:2px 7px;'
            f'border-radius:4px">{_esc(_get(a, "name", "?"))}</span></td>'
            f'<td style="padding:3px 0;font-family:monospace;font-size:12px;'
            f'color:#475569">{ftxt}</td></tr>'
        )
    arms_html = f'<table cellpadding="0" cellspacing="0">{arm_rows}</table>'

    # judge scores table
    scores_html = ""
    if scores:
        rows = ""
        for ax in sorted(scores.keys()):
            s = scores[ax]
            color = "#b91c1c" if s == 0 else ("#047857" if s == 5 else "#475569")
            rows += (
                f'<tr><td style="padding:2px 12px 2px 0;font-size:12px;'
                f'color:#64748b">{_esc(ax.replace("_", " "))}</td>'
                f'<td style="padding:2px 0;font-size:12px;font-family:monospace;'
                f'font-weight:700;color:{color};text-align:right">'
                f'{_esc(s)}/5</td></tr>'
            )
        scores_html = f'<table cellpadding="0" cellspacing="0">{rows}</table>'

    def bullet_list(items: List[Any], color: str) -> str:
        if not items:
            return ""
        lis = "".join(
            f'<li style="margin:2px 0;color:{color}">{_esc(i)}</li>'
            for i in items)
        return f'<ul style="margin:4px 0;padding-left:18px;font-size:13px">{lis}</ul>'

    papers_html = ""
    for p in papers:
        aid = str(_get(p, "arxiv_id", ""))
        url = f"https://arxiv.org/abs/{aid}" if aid else "#"
        refs = _get(p, "section_refs") or []
        ref_pills = "".join(
            f'<span style="display:inline-block;font-family:monospace;'
            f'font-size:10px;background:#f1f5f9;color:#475569;padding:1px 6px;'
            f'border-radius:3px;margin:0 4px 4px 0">{_esc(r)}</span>'
            for r in refs)
        evidence = _get(p, "supporting_evidence", "")
        ev_html = (
            f'<div style="font-size:12px;font-style:italic;color:#64748b;'
            f'border-left:2px solid #cbd5e1;padding-left:8px;margin-top:4px">'
            f'{_esc(evidence)}</div>') if evidence else ""
        papers_html += (
            f'<div style="padding:8px 0;border-bottom:1px solid #f1f5f9">'
            f'<a href="{_esc(url)}" style="color:#4f46e5;text-decoration:none;'
            f'font-weight:600;font-size:13px">{_esc(_get(p, "title", "(untitled)"))}</a>'
            f' <span style="font-family:monospace;font-size:10px;color:#94a3b8">'
            f'arXiv:{_esc(aid)}</span><div style="margin-top:4px">{ref_pills}</div>'
            f'{ev_html}</div>'
        )

    buttons_html = ""
    if token_secret:
        buttons_html = (
            _btn(approve_url, "Approve", "#4f46e5") +
            _btn(reject_url, "Reject", "#dc2626") +
            _btn(defer_url, "Defer", "#64748b")
        )
    else:
        buttons_html = (
            '<div style="font-size:13px;color:#b45309">One-click buttons '
            'are disabled (MADSCIENTIST_TOKEN_SECRET not set). Use the '
            'dashboard link below to decide.</div>')

    primary_html = ""
    if primary:
        pp = _get(sc, "primary_parsed") or {}
        pp_txt = ""
        if pp:
            pp_txt = (
                f'<div style="font-family:monospace;font-size:12px;'
                f'color:#475569;margin-top:4px">'
                f'{_esc(_get(pp, "metric"))}({_esc(_get(pp, "arm_a"))}) '
                f'{_esc(_get(pp, "comparator"))} {_esc(_get(pp, "threshold"))} '
                f'({_esc(_get(pp, "threshold_kind"))} vs '
                f'{_esc(_get(pp, "arm_b"))})</div>')
        secondary = _get(sc, "secondary") or []
        primary_html = (
            f'<div>{_esc(primary)}</div>{pp_txt}'
            + (bullet_list(secondary, "#475569") if secondary else ""))

    wall_txt = f" &middot; ~{_esc(wall)}h est" if wall else ""
    html_body = f"""\
<!DOCTYPE html>
<html><body style="margin:0;padding:0;background:#f1f5f9">
<table cellpadding="0" cellspacing="0" width="100%" style="background:#f1f5f9;padding:24px 0">
<tr><td align="center">
<table cellpadding="0" cellspacing="0" width="640" style="max-width:640px;background:#ffffff;border-radius:14px;overflow:hidden;border:1px solid #e2e8f0">
  <tr><td style="padding:20px 24px;background:#0f172a">
    <div style="{sans};font-size:11px;letter-spacing:0.08em;text-transform:uppercase;color:#818cf8;font-weight:700">MadScientist &middot; proposal awaiting your decision</div>
    <div style="{sans};font-size:19px;font-weight:700;color:#f8fafc;margin-top:6px">{_esc(title)}</div>
    <div style="margin-top:10px">
      <span style="display:inline-block;{sans};font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;background:{vbg};color:{vfg};padding:4px 10px;border-radius:999px">{_esc(verdict.replace("_", " "))}{_esc(norm_txt)}</span>
    </div>
  </td></tr>
  <tr><td style="padding:8px 24px 20px 24px">
    <table cellpadding="0" cellspacing="0" width="100%">
      {section("Hypothesis", f'<div>{_esc(hypothesis)}</div>')}
      {section("Motivation", f'<div style="color:#475569">{_esc(motivation)}</div>') if motivation else ""}
      {section("Judge scores", scores_html) if scores_html else ""}
      {section("Strengths", bullet_list(strengths, "#047857")) if strengths else ""}
      {section("Concerns", bullet_list(concerns, "#b45309")) if concerns else ""}
      {section("Experiment arms", arms_html + f'<div style="margin-top:6px;font-size:12px;color:#64748b">Sizing: {_esc(n_seeds)} seeds &times; {_esc(n_iters)} iters{wall_txt}</div>')}
      {section("Success criteria", primary_html) if primary_html else ""}
      {section("Source papers", papers_html) if papers_html else ""}
    </table>
    <div style="margin-top:22px;padding-top:18px;border-top:1px solid #e2e8f0">
      {buttons_html}
    </div>
    <div style="margin-top:18px;{sans};font-size:12px;color:#94a3b8">
      <a href="{_esc(dashboard_link)}" style="color:#6366f1">View full details on the dashboard</a>
      &middot; proposal id <span style="font-family:monospace">{_esc(pid)}</span>
    </div>
  </td></tr>
</table>
</td></tr>
</table>
</body></html>"""

    return subject, text_body, html_body


# ---- SMTP send -----------------------------------------------------------


class SmtpConfig:
    """Resolved SMTP settings read from the environment."""

    def __init__(self):
        self.host = os.environ.get("SMTP_HOST", "").strip()
        self.port = int(os.environ.get("SMTP_PORT", "587") or "587")
        self.user = os.environ.get("SMTP_USER", "").strip()
        self.password = os.environ.get("SMTP_PASSWORD", "")
        self.to_addr = (os.environ.get("NOTIFICATION_EMAIL", "").strip()
                        or self.user)

    @property
    def configured(self) -> bool:
        return bool(self.host and self.user and self.password and self.to_addr)

    def missing(self) -> List[str]:
        return [n for n, v in [
            ("SMTP_HOST", self.host), ("SMTP_USER", self.user),
            ("SMTP_PASSWORD", self.password),
            ("NOTIFICATION_EMAIL/To", self.to_addr)] if not v]


def send_email(
    subject: str,
    text_body: str,
    html_body: str,
    smtp: SmtpConfig,
    *,
    to_addr: Optional[str] = None,
) -> None:
    """Send a multipart (text + HTML) email. Raises on SMTP failure."""
    to_addr = to_addr or smtp.to_addr
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp.user
    msg["To"] = to_addr
    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")

    ctx = ssl.create_default_context()
    if smtp.port == 465:
        with smtplib.SMTP_SSL(smtp.host, smtp.port, timeout=30, context=ctx) as s:
            s.login(smtp.user, smtp.password)
            s.send_message(msg)
    else:
        with smtplib.SMTP(smtp.host, smtp.port, timeout=30) as s:
            s.ehlo()
            s.starttls(context=ctx)
            s.ehlo()
            s.login(smtp.user, smtp.password)
            s.send_message(msg)


# ---- Notify (one proposal) -----------------------------------------------


def notify_once(
    db,
    proposal: Dict[str, Any],
    *,
    base_url: str,
    token_secret: Optional[str],
    smtp: SmtpConfig,
    token_ttl_seconds: int = DEFAULT_DECISION_TOKEN_TTL_SECONDS,
    send_fn=send_email,
    now: Optional[datetime.datetime] = None,
) -> bool:
    """Email a single proposal + stamp notified_at. Returns True if sent.

    Idempotent at the DB level: the notified_at stamp is written with a
    filtered updateOne so a concurrent notifier can't double-send. send
    happens BEFORE the stamp; if send raises, notified_at stays unset
    and the next loop retries.
    """
    now = now or datetime.datetime.now(datetime.timezone.utc)
    pid = proposal["_id"]

    subject, text_body, html_body = render_proposal_email(
        proposal, base_url=base_url, token_secret=token_secret,
        token_ttl_seconds=token_ttl_seconds, now=now)

    send_fn(subject, text_body, html_body, smtp)

    audit_event = {
        "at": now,
        "by_agent": constants.AGENT_EMAIL_BRIDGE,
        "event": "notified",
        "detail": {
            "channel": "email",
            "to": smtp.to_addr,
            "buttons": bool(token_secret),
        },
    }
    # Only stamp if still unset - keeps it idempotent under races.
    res = db[constants.COLL_PROPOSALS].update_one(
        {"_id": pid, "notified_at": None},
        {"$set": {"notified_at": now},
         "$push": {"audit_events": audit_event}},
    )
    return res.modified_count > 0


# ---- Notify loop ---------------------------------------------------------


def notify_loop(
    db,
    *,
    base_url: str,
    token_secret: Optional[str],
    poll_interval_seconds: int = DEFAULT_NOTIFY_POLL_INTERVAL_SECONDS,
    token_ttl_seconds: int = DEFAULT_DECISION_TOKEN_TTL_SECONDS,
    should_stop_fn=None,
) -> None:
    """Poll for un-notified pending_user proposals and email each once.

    Resilient: an SMTP failure on one proposal is logged + retried next
    cycle (notified_at stays unset). If SMTP isn't configured the loop
    logs once and idles (so the rest of the worker still runs).
    """
    smtp = SmtpConfig()
    print(
        f"email_bridge: starting notify loop, poll interval = "
        f"{poll_interval_seconds}s, to = {smtp.to_addr or '<unset>'}, "
        f"buttons = {bool(token_secret)}, base_url = {base_url!r}",
        flush=True)
    warned_unconfigured = False

    while True:
        if should_stop_fn is not None and should_stop_fn():
            print("email_bridge: stop signal received; exiting.", flush=True)
            return
        try:
            if not smtp.configured:
                if not warned_unconfigured:
                    print(
                        f"email_bridge: SMTP not fully configured "
                        f"(missing {smtp.missing()}); notifications disabled "
                        f"until set. Idling.",
                        flush=True)
                    warned_unconfigured = True
            else:
                warned_unconfigured = False
                cursor = db[constants.COLL_PROPOSALS].find({
                    "status": constants.STATUS_PENDING_USER,
                    "notified_at": None,
                })
                for proposal in cursor:
                    try:
                        sent = notify_once(
                            db, proposal,
                            base_url=base_url,
                            token_secret=token_secret,
                            smtp=smtp,
                            token_ttl_seconds=token_ttl_seconds)
                        if sent:
                            print(
                                f"email_bridge: emailed proposal "
                                f"{proposal.get('_id')} "
                                f"('{(proposal.get('title') or '')[:60]}') "
                                f"to {smtp.to_addr}.",
                                flush=True)
                    except Exception as e:  # noqa: BLE001
                        tb = traceback.format_exc()
                        print(
                            f"email_bridge: send failed for "
                            f"{proposal.get('_id')}: {type(e).__name__}: {e}\n"
                            f"{tb}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"email_bridge: loop error: {e}", flush=True)
        time.sleep(poll_interval_seconds)
