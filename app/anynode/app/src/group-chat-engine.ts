export function groupSignals(text: string) {
  const t = text.toLowerCase();
  const collaboration = /(let's|together|team|collaborate|pair)/.test(t) ? 1 : 0;
  const conflict = /(blame|fault|us vs them|you people)/.test(t) ? 1 : 0;
  return { collaboration, conflict };
}
