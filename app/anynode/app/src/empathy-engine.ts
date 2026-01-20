export function empathyScore(text: string) {
  // tiny heuristic + token length; replace with your TS module if needed
  const t = text.toLowerCase();
  let score = 0;
  if (/(sorry|i hear you|i understand|with you)/.test(t)) score += 0.4;
  if (/(angry|sad|fear|anxious|hurt)/.test(t)) score += 0.2;
  score += Math.min(0.4, Math.max(0, text.length / 5000));
  return Number(score.toFixed(3));
}
