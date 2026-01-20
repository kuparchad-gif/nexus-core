export function learningSignals(text: string) {
  const t = text.toLowerCase();
  const curiosity = /(why|how|what if|learn|study|explore)/.test(t) ? 1 : 0;
  const reflection = /(i noticed|i realized|lesson|next time)/.test(t) ? 1 : 0;
  return { curiosity, reflection };
}
