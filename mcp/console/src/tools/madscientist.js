import { z } from 'zod';
import { get, post } from '../api.js';
import { defineTool, defined } from '../util.js';

// STATUSES_ALL in rl_agent/madscientist/constants.py.
const STATUSES = ['pending_judge', 'pending_user', 'approved', 'rejected', 'deferred', 'implementing', 'pr_open',
  'training', 'done', 'failed', 'cancelled'];
const objectId = z.string().regex(/^[a-fA-F0-9]{24}$/, 'a 24-character proposal id');

export function register(server) {
  defineTool(server, 'list_proposals', {
    title: 'List MadScientist proposals',
    description: 'MadScientist tab: research proposals, most recently updated first. By default everything that needs or had a '
      + 'decision, excluding rejected ones.',
    readOnly: true,
    input: {
      status: z.array(z.enum(STATUSES)).optional().describe('e.g. ["pending_user"] for proposals awaiting your decision.'),
      limit: z.number().int().min(1).max(200).optional().describe('Default 50.'),
    },
  }, async ({ status, limit }) => get('/madscientist/proposals', defined({ status: status && status.join(','), limit })));

  defineTool(server, 'get_proposal', {
    title: 'Get proposal',
    description: 'Full proposal: hypothesis, experiment arms, judge review and audit timeline.',
    readOnly: true,
    input: { id: objectId },
  }, async ({ id }) => get(`/madscientist/proposals/${id}`));

  defineTool(server, 'get_madscientist_activity', {
    title: 'MadScientist activity feed',
    description: 'Recent audit events across all proposals, newest first, with the status of each proposal\'s training jobs.',
    readOnly: true,
    input: {
      limit: z.number().int().min(1).max(200).optional().describe('Default 30.'),
      offset: z.number().int().min(0).optional(),
    },
  }, async ({ limit, offset }) => get('/madscientist/activity', defined({ limit, offset })));

  defineTool(server, 'decide_proposal', {
    title: 'Decide proposal',
    description: 'Approve, approve with revisions, reject or defer a proposal. Approval lets the orchestrator queue its training jobs.',
    input: {
      id: objectId,
      action: z.enum(['approve', 'approve_with_revisions', 'reject', 'defer']),
      note: z.string().max(4000).optional(),
    },
  }, async ({ id, action, note }) => post('/madscientist/decide', defined({ proposal_id: id, action, note })));

  defineTool(server, 'edit_proposal', {
    title: 'Edit proposal',
    description: 'Adjust a proposal still awaiting a decision (pending_user or deferred). Editable keys: title, hypothesis, '
      + 'motivation, code_changes_summary, n_seeds_per_arm, num_iterations_per_seed, expected_wall_time_hours, '
      + 'success_criteria.primary, and success_criteria.primary_parsed.{metric, arm_a, arm_b, comparator, threshold, '
      + 'threshold_kind}. Other keys are ignored. Every edit is recorded in the audit trail.',
    input: { id: objectId, patch: z.record(z.string(), z.any()) },
  }, async ({ id, patch }) => post(`/madscientist/proposals/${id}/edit`, { patch }));
}
