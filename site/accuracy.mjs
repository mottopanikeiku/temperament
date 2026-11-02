const percent = value => `${(value * 100).toFixed(1)}%`;

export function accuracySummary(summary) {
  const example = summary.rows.filter(row => row.model === 'qwen7' && row.task === 'hellaswag');
  const accuracy = variant => percent(example.find(row => row.variant === variant).accuracy);
  return `Qwen2.5-7B-Instruct HellaSwag accuracy goes from ${accuracy(summary.baseline)} in BF16 to ${accuracy('tile')} in plain FP8 and ${accuracy('rotate')} with rotation. Centering keys brings it back to ${accuracy('rotate_smooth_k')}.`;
}
