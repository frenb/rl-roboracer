"""Cursor SDK code-writing agent for proposals that need new SCHEMA fields.

When a proposal carries `proposed_schema_extensions`, the orchestrator
delegates the code edits to a Cursor cloud agent via the official
`cursor-sdk` Python package. The agent:

  1. Receives a structured implementation prompt (the proposal JSON +
     a checklist of expected edits).
  2. Clones the repo into a Cursor-hosted VM.
  3. Edits rl_agent/experiment_designs.py SCHEMA + adds the trainer
     kwarg plumbing in rl_agent/robotaxi.py::main() to honor each
     proposed_schema_extensions entry.
  4. Commits to a branch named xp-<proposal-id-short>/<slug>.
  5. Opens a PR via auto_create_pr=True.
  6. Returns (status, pr_url, agent_id, run_id).

Streams progress events into proposal.implementation_log so the
dashboard's Mad Scientist Lab tab can tail them.

This module guards the cursor-sdk import behind a lazy try/except so
the rest of the system runs cleanly even without the package installed.
If the SDK is missing AND a proposal needs the Cursor path, the
orchestrator marks the proposal failed with a clear "install cursor-sdk
+ set CURSOR_API_KEY" reason.

Tests mock the cursor-sdk module via sys.modules so they don't require
the real package to be installed.
"""
from __future__ import annotations

import datetime
import os
import re
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import constants


# ---- Configuration -------------------------------------------------------

# Model used by the Cursor agent for code edits. composer-2.5 is the
# Cursor SDK's default; override via env if needed.
CURSOR_AGENT_MODEL = os.environ.get(
    "CURSOR_AGENT_MODEL", "composer-2.5")

# GitHub repo the Cursor cloud agent clones from. Format: "owner/repo".
# Required when MADSCIENTIST_ENABLED=true AND we hit the Cursor path.
# Defaults match this codebase's public origin; override for forks /
# private repos via the env var.
CURSOR_TARGET_REPO = os.environ.get(
    "CURSOR_TARGET_REPO", "frenb/rl-roboracer")

# Branch base name (the rest is appended per proposal). Cursor agents
# create branches with this prefix off main.
CURSOR_BRANCH_PREFIX = os.environ.get(
    "CURSOR_BRANCH_PREFIX", "xp")

# How many implementation-log chunks we keep in proposal.implementation_log.
# The dashboard tails these; older chunks rotate out so the proposal
# document doesn't grow unboundedly during long agent runs.
_MAX_LOG_CHUNKS = 200


# ---- Exception types -----------------------------------------------------


class CursorSdkNotInstalled(RuntimeError):
    """Raised when cursor-sdk is requested but isn't importable."""


# ---- Lazy SDK import -----------------------------------------------------


def _import_cursor_sdk():
    """Lazy import that returns (Agent, CloudAgentOptions, CursorAgentError).

    Raises CursorSdkNotInstalled if the cursor-sdk package isn't
    available. Tests can monkey-patch this function to inject a mock.
    """
    try:
        from cursor_sdk import (  # type: ignore
            Agent, CloudAgentOptions, CursorAgentError)
    except ImportError as e:
        raise CursorSdkNotInstalled(
            f"cursor-sdk not installed: {e}. Run "
            f"`pip install cursor-sdk` inside the madscientist container, "
            f"or rebuild the image after adding `cursor-sdk` to "
            f"docker/madscientist/requirements.txt.")
    return Agent, CloudAgentOptions, CursorAgentError


# ---- Implementation prompt builder ---------------------------------------


_IMPLEMENTATION_PROMPT_TEMPLATE = """You are implementing experiment-design SCHEMA additions for the rl-roboracer codebase, on behalf of the MadScientist autonomous research agent.

Proposal ID: {proposal_id}
Proposal title: {title}

The Researcher and Judge have approved a proposal that needs these NEW fields added to `rl_agent/experiment_designs.py::SCHEMA`:

```json
{schema_extensions_json}
```

YOUR JOB (in this exact order):

1. For each entry in the JSON above, add a new dict entry to SCHEMA in
   `rl_agent/experiment_designs.py`. Place it under the section
   matching the entry's `section` field (sections are demarcated with
   `_section_*` keys; preserve the existing section ordering).
   Schema entry shape:
       "<name>": {{
           "type": "<type>",
           "default": <default>,
           "min": <min_value or None>,
           "max": <max_value or None>,
           "doc": "<doc>",
           "paper_ref": "<paper_ref or None>",
           "kwarg": "<name>_val",
       }}
   The "kwarg" field is `<name>_val` by convention.

2. In `rl_agent/robotaxi.py::main()`, add the new kwarg(s) to main()'s
   signature with the same default as in SCHEMA. Place the kwarg near
   similar existing knobs (e.g., new BC knobs go near
   bc_pretrain_steps_val).

3. In `rl_agent/robotaxi.py::main()`, add a 1-line coercion of the new
   kwarg to its expected type (int/float/bool) near the top of main()
   where the existing kwargs are similarly coerced. Pattern:
       new_knob = <type>(<name>_val) if <name>_val is not None else <default>

4. If the new knob affects training behavior, IMPLEMENT the behavior.
   For example, if it's `aux_bc_loss_weight`, add the aux BC step in
   the training loop. Cite the paper (paper_ref field) in the code
   comment for the new behavior.

5. Do NOT modify any safety-critical code paths. Forbidden patterns:
   _emergency_pause_handler, _get_job_lifecycle_state,
   _restore_paused_active_dir, move_all_jobs_data,
   RpcClient.__init__, virtual_endpoint/virtual.py, anything under
   dashboard/.

6. Commit the changes to a new branch named
   `{branch_name}` with a clear commit message referencing this
   proposal id.

7. Open a PR titled "exp/{proposal_short_id}: {title}". The PR body
   should:
       - cite this proposal id
       - list the new SCHEMA fields
       - briefly describe the training-loop behavior changes (if any)
       - note that MadScientist's Outcome Ingester will queue training
         jobs against the new fields once the PR is merged

PROPOSAL CONTEXT (for understanding the motivation):

```json
{proposal_summary_json}
```

Begin now. Stream your work so the dashboard can tail it.
"""


def build_implementation_prompt(proposal_doc: Dict[str, Any]) -> Tuple[str, str]:
    """Construct the Cursor agent's prompt + the proposed branch name.

    Returns (prompt, branch_name). branch_name is what the agent should
    push to; included separately so the orchestrator can record it on
    the proposal even before the agent reports back.
    """
    import json

    proposal_id = str(proposal_doc.get("_id", "?"))
    proposal_short_id = proposal_id[-8:] if len(proposal_id) >= 8 else proposal_id
    title = str(proposal_doc.get("title") or "untitled")
    slug = _slugify(title)
    branch_name = f"{CURSOR_BRANCH_PREFIX}-{proposal_short_id}/{slug}"

    schema_extensions = proposal_doc.get("proposed_schema_extensions") or []
    schema_extensions_json = json.dumps(
        schema_extensions, indent=2, default=str)

    # Trim the proposal context to the scientifically-relevant bits -
    # the Cursor agent doesn't need audit_events / cost / etc.
    proposal_summary = {
        "title": proposal_doc.get("title"),
        "hypothesis": proposal_doc.get("hypothesis"),
        "motivation": proposal_doc.get("motivation"),
        "code_changes_summary": proposal_doc.get("code_changes_summary"),
        "source_papers": proposal_doc.get("source_papers"),
        "experiment_arms": proposal_doc.get("experiment_arms"),
        "success_criteria": proposal_doc.get("success_criteria"),
    }
    proposal_summary_json = json.dumps(
        proposal_summary, indent=2, default=str)

    prompt = _IMPLEMENTATION_PROMPT_TEMPLATE.format(
        proposal_id=proposal_id,
        proposal_short_id=proposal_short_id,
        title=title,
        branch_name=branch_name,
        schema_extensions_json=schema_extensions_json,
        proposal_summary_json=proposal_summary_json,
    )
    return prompt, branch_name


def _slugify(s: str) -> str:
    """Cheap slug for branch names: lowercase, alnum + hyphens, max 40."""
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()
    return s[:40] or "untitled"


# ---- Run-result extraction -----------------------------------------------


_PR_URL_RE = re.compile(
    r"https://github\.com/[\w./-]+/pull/\d+", re.IGNORECASE)


def _extract_pr_url(result_obj, agent_messages: List[str]) -> Optional[str]:
    """Best-effort PR URL extraction from a cursor-sdk RunResult.

    The exact API surface of `RunResult` varies; we try several common
    field names + fall back to scanning the agent's transcript for a
    GitHub PR URL. If nothing matches, returns None - the operator can
    still find the PR by branch name.
    """
    # Try common attribute names first.
    for attr in ("pr_url", "pull_request_url", "github_pr_url"):
        v = getattr(result_obj, attr, None)
        if isinstance(v, str) and v:
            return v
    # Try a metadata-style dict.
    for attr in ("metadata", "outputs", "result"):
        meta = getattr(result_obj, attr, None)
        if isinstance(meta, dict):
            for k in ("pr_url", "pull_request_url", "github_pr_url"):
                v = meta.get(k)
                if isinstance(v, str) and v:
                    return v
    # Scan the assistant transcript for a GitHub PR URL.
    for msg in reversed(agent_messages):  # walk newest -> oldest
        m = _PR_URL_RE.search(msg or "")
        if m:
            return m.group(0)
    return None


# ---- Streaming + Mongo write helpers -------------------------------------


def _append_log_chunk(
    db, proposal_id, chunk: str, *,
    by_agent: str = constants.AGENT_ORCHESTRATOR,
    max_chunk_len: int = 2000,
):
    """Push one chunk into proposal.implementation_log. Caps the array
    via $slice so the document doesn't bloat during long agent runs."""
    if not chunk:
        return
    chunk = chunk[:max_chunk_len]
    now = datetime.datetime.now(datetime.timezone.utc)
    formatted = f"[{now.isoformat(timespec='seconds')}] [{by_agent}] {chunk}"
    try:
        db[constants.COLL_PROPOSALS].update_one(
            {"_id": proposal_id},
            {"$push": {
                "implementation_log": {
                    "$each": [formatted],
                    "$slice": -_MAX_LOG_CHUNKS,
                },
            }})
    except Exception:  # noqa: BLE001
        # Logging failure shouldn't crash the agent loop.
        pass


# ---- Main entry point ----------------------------------------------------


def spawn_cursor_agent_for_proposal(
    db,
    proposal_doc: Dict[str, Any],
    *,
    api_key: Optional[str] = None,
    target_repo: Optional[str] = None,
    sdk_import: Optional[Callable] = None,
    on_message: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Spawn a Cursor cloud agent that implements this proposal's
    proposed_schema_extensions + opens a PR.

    Args:
        db: pymongo database (for streaming log chunks).
        proposal_doc: the proposal we're implementing. Must have
            non-empty proposed_schema_extensions.
        api_key: Cursor API key. Defaults to env CURSOR_API_KEY.
        target_repo: "owner/repo" string. Defaults to CURSOR_TARGET_REPO.
        sdk_import: test hook - overrides _import_cursor_sdk.
        on_message: optional callback per streamed message (also gets
            logged to Mongo). Tests can use this to capture events.

    Returns dict with:
        status: "finished" | "error" | "did_not_start"
        pr_url: extracted PR URL or None
        branch_name: the branch we asked the agent to push to
        agent_id: cursor agent id (useful for resume / lookup)
        run_id: cursor run id
        error: error message on failure, else None

    Does NOT advance proposal.status itself - the caller in
    orchestrator.py handles status transitions based on the returned
    dict.
    """
    proposal_id = proposal_doc["_id"]
    prompt, branch_name = build_implementation_prompt(proposal_doc)

    api_key = api_key or os.environ.get("CURSOR_API_KEY", "").strip()
    if not api_key:
        return _make_failure_result(
            branch_name,
            error=(
                "CURSOR_API_KEY missing. Set it in .env + "
                "'docker compose restart madscientist' to enable the "
                "Cursor SDK code-writing agent."),
            status="did_not_start")

    target_repo = (
        target_repo
        or os.environ.get("CURSOR_TARGET_REPO", CURSOR_TARGET_REPO))

    importer = sdk_import or _import_cursor_sdk
    try:
        Agent, CloudAgentOptions, CursorAgentError = importer()
    except CursorSdkNotInstalled as e:
        return _make_failure_result(
            branch_name, error=str(e), status="did_not_start")

    _append_log_chunk(
        db, proposal_id,
        f"spawning Cursor cloud agent (model={CURSOR_AGENT_MODEL}, "
        f"repo={target_repo}, branch={branch_name}).")

    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    transcript: List[str] = []

    try:
        with Agent.create(
            api_key=api_key,
            model=CURSOR_AGENT_MODEL,
            cloud=CloudAgentOptions(
                repos=[target_repo],
                auto_create_pr=True,
                # Phase 1D will add reviewer-request notifications;
                # for now skip so the operator just gets a PR with
                # no extra noise.
                skip_reviewer_request=True,
            ),
        ) as agent:
            agent_id = getattr(agent, "agent_id", None) or getattr(
                agent, "agentId", None)
            _append_log_chunk(
                db, proposal_id,
                f"agent created (agent_id={agent_id}); sending prompt.")
            run = agent.send(prompt)
            run_id = getattr(run, "id", None) or getattr(run, "run_id", None)
            _append_log_chunk(
                db, proposal_id,
                f"run started (run_id={run_id}); streaming events.")

            # Stream events. The Python SDK exposes run.messages() to
            # yield typed messages. We extract text content where
            # available + push every chunk into Mongo for the dashboard.
            try:
                for msg in run.messages():
                    text = _coerce_message_to_text(msg)
                    if text:
                        transcript.append(text)
                        _append_log_chunk(db, proposal_id, text)
                        if on_message is not None:
                            try:
                                on_message(text)
                            except Exception:  # noqa: BLE001
                                pass
            except Exception as stream_err:  # noqa: BLE001
                _append_log_chunk(
                    db, proposal_id,
                    f"stream interrupted: {type(stream_err).__name__}: {stream_err}")

            # Always call wait() to materialize the terminal state.
            result = run.wait()
            status = getattr(result, "status", "unknown")
            _append_log_chunk(
                db, proposal_id, f"run.wait() returned status={status!r}")

            if status == "error":
                return _make_failure_result(
                    branch_name,
                    error=getattr(result, "error", None) or "RunResult.status=error",
                    status="error",
                    agent_id=agent_id,
                    run_id=run_id)

            pr_url = _extract_pr_url(result, transcript)
            return {
                "status": "finished",
                "pr_url": pr_url,
                "branch_name": branch_name,
                "agent_id": agent_id,
                "run_id": run_id,
                "error": None,
            }
    except CursorAgentError as e:
        # Didn't start: auth / config / network. is_retryable hint
        # surfaces on the exception.
        _append_log_chunk(
            db, proposal_id,
            f"CursorAgentError (did not start): {e}")
        return _make_failure_result(
            branch_name,
            error=f"CursorAgentError: {e}",
            status="did_not_start",
            agent_id=agent_id,
            run_id=run_id)
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        _append_log_chunk(
            db, proposal_id,
            f"unexpected error: {type(e).__name__}: {e}\n{tb[-500:]}")
        return _make_failure_result(
            branch_name,
            error=f"{type(e).__name__}: {e}",
            status="error",
            agent_id=agent_id,
            run_id=run_id)


def _coerce_message_to_text(msg) -> str:
    """Best-effort: extract printable text from one streamed SDK message.

    The SDK's message shapes evolve. We accept:
      * raw strings
      * objects with .text or .content[].text attributes
      * dicts with "text" or "content" keys
    """
    if msg is None:
        return ""
    if isinstance(msg, str):
        return msg

    # Common: message.message.content is a list of blocks
    sub = getattr(msg, "message", None)
    if sub is not None:
        content = getattr(sub, "content", None)
        if isinstance(content, list):
            parts = []
            for block in content:
                if hasattr(block, "text"):
                    parts.append(getattr(block, "text", "") or "")
                elif isinstance(block, dict) and "text" in block:
                    parts.append(str(block["text"]))
            if parts:
                return "".join(parts)

    # Direct .text attribute
    t = getattr(msg, "text", None)
    if isinstance(t, str):
        return t

    # Dict-shaped message
    if isinstance(msg, dict):
        if "text" in msg:
            return str(msg["text"])
        content = msg.get("content")
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    parts.append(str(block["text"]))
            if parts:
                return "".join(parts)
        return str(msg)

    return ""


def _make_failure_result(
    branch_name: str,
    *,
    error: str,
    status: str,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "status": status,
        "pr_url": None,
        "branch_name": branch_name,
        "agent_id": agent_id,
        "run_id": run_id,
        "error": error,
    }
