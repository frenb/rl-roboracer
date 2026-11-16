// Leaderboard eval trials of roboracer models (/leaderboard_scores rows).

// Per-trial arrays on a leaderboard_scores row, by metric name.
export const METRICS = {
  score: 'scores',
  goals_per_episode: 'avg_goals_per_episode',
  speed: 'avg_speeds',
  episode_length: 'episode_lengths',
};

/** Eval runs for a model: linked by model_id, or by path for rows older than that link. */
export function runsFor(model, scores) {
  const id = String(model._id);
  return scores.filter((s) => String(s.model_id || '') === id || (!s.model_id && model.location && s.path === model.location));
}

export function pooled(runs, metric) {
  const key = METRICS[metric];
  return runs.flatMap((r) => (Array.isArray(r[key]) ? r[key] : [])).map(Number).filter(Number.isFinite);
}
