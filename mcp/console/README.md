# roboracer-console MCP server

A [Model Context Protocol](https://modelcontextprotocol.io) server that lets a coding agent do anything the web console can: queue and manage jobs, evaluate and compare models, edit reward and experiment designs, read the weakness map, and publish gym builds. It talks to the dashboard's HTTP API, so it applies the same rules the console UI does (job state transitions, demo requirements per course, model eval rules), and destructive tools require `confirm: true`.

## Setup

Requires Node 22+ and a running stack (dashboard on http://localhost).

```bash
cd mcp/console && npm install
```

Cursor picks it up from [`.cursor/mcp.json`](../../.cursor/mcp.json). Other MCP clients: run `node mcp/console/src/index.js` over stdio, with `ROBORACER_DASHBOARD_URL` if the dashboard is not on http://localhost. Publishing gyms needs the GitHub CLI (`gh`) signed in.

## Tools

| Area | Tools |
|---|---|
| Jobs | `list_jobs`, `get_job`, `create_job`, `update_jobs`, `delete_jobs`, `get_queue_state`, `set_queue_paused`, `tail_trainer_log` |
| Weakness | `get_weakness_map`, `get_weakness_at`, `get_difficult_sections`, `suggest_start_gates` |
| Models | `list_models`, `get_model`, `get_model_reward_source`, `delete_model`, `eval_models`, `compare_models`, `compare_in_tensorboard`, `get_leaderboard`, `get_env_specs` |
| Designs | `list_reward_designs`, `get_reward_design`, `lint_reward_design`, `create_reward_design`, `update_reward_design`, `set_reward_design_archived`, `get_experiment_design_schema`, `list_experiment_designs`, `get_experiment_design`, `create_experiment_design`, `update_experiment_design`, `set_experiment_design_archived` |
| Gyms and demos | `list_gyms`, `create_gym`, `update_gym`, `set_gym_archived`, `list_demos`, `create_demo`, `update_demo`, `set_demo_archived` |
| Mad Scientist | `list_proposals`, `get_proposal`, `get_madscientist_activity`, `decide_proposal`, `edit_proposal` |
| Releases | `list_releases`, `list_gym_builds`, `package_gym`, `add_gym_to_release`, `delete_release_asset` |

`get_difficult_sections` and `suggest_start_gates` need the dashboard's `/weakness_by_gate` endpoint, added alongside them.

## Test

```bash
node test/smoke.js
```

Runs every read tool and the create/update/archive paths against the live stack. What it creates is named `mcp-smoke-*`: test jobs are created as DONE (so the trainer never runs them) and deleted; designs, gyms and demos are archived.
