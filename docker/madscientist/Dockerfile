# MadScientist agent container.
#
# Hosts the four worker scripts under rl_agent/madscientist/:
#   * researcher.py - drafts proposals from new papers (Phase 1)
#   * judge.py - applies the rubric to draft proposals (Phase 1)
#   * orchestrator.py - spawns Cursor SDK agents in /worktrees/ (Phase 1)
#   * outcome_ingester.py - watches db.jobs for completed TRAIN runs (Phase 1)
#
# Phase 0 (this commit) ships only main.py (a heartbeat stub) + seed.py.
# The container still builds end-to-end so the rest of the system can be
# scaffolded around it; the workers light up once MADSCIENTIST_ENABLED=true
# is set and Phase 1 lands their implementations.
#
# Why a slim Python base instead of full tensorflow-gpu image: this agent
# does no training. It only needs:
#   - pymongo (read/write proposals)
#   - anthropic SDK (LLM calls)
#   - arxiv + bs4 (literature retrieval)
#   - google-api-python-client (Gmail inbound webhook polling)
#   - @cursor/sdk via npm (Node-only) for spawning implementation agents
#   - git (for `git clone --reference /repo /worktrees/<id>/`)
#
# Total image ~600 MB vs the trainer's multi-GB tensorflow-gpu base, so
# rebuilds during development are fast.

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps:
#   git    - worktree clones for the orchestrator
#   curl   - Node.js installer + arxiv/web fetches (Python httpx also fine)
#   gnupg  - apt-key for the NodeSource repo
#   ca-certificates - HTTPS to anthropic.com, arxiv.org, googleapis.com
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        gnupg \
    && rm -rf /var/lib/apt/lists/*

# Node.js for the Cursor SDK (TypeScript / Node-only). Pinned to v20 LTS.
# Installs `node` + `npm`; we then npm-i `@cursor/sdk` globally so the
# orchestrator can `import { Agent } from "@cursor/sdk"` without per-
# worktree node_modules churn.
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @cursor/sdk

# Python deps. Layer-cached separately from the source bind mount so
# code edits don't bust the pip-install layer.
WORKDIR /madscientist
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Entrypoint script handles MADSCIENTIST_ENABLED gating + seed-on-boot.
# CRLF stripping with sed because this repo is typically edited on
# Windows + bind-mounted from there; without it bash chokes on the
# shebang line ("#!/bin/bash\r" - "no such file or directory" because
# /bin/bash\r isn't a real binary).
COPY entrypoint.sh /usr/local/bin/madscientist-entrypoint
RUN sed -i 's/\r$//' /usr/local/bin/madscientist-entrypoint \
    && chmod +x /usr/local/bin/madscientist-entrypoint

# The source itself lives at /repo (bind-mounted read-only from the host
# repo root in docker-compose.yml). PYTHONPATH adds /repo so
# `python -m rl_agent.madscientist.main` resolves the package.
ENV PYTHONPATH=/repo

ENTRYPOINT ["/usr/local/bin/madscientist-entrypoint"]
