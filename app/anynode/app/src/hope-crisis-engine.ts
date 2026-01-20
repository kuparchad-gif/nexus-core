export function hopeSignals(text: string) {
  const t = text.toLowerCase();
  const crisis = /(hopeless|suicid|self-harm|no way out|panic attack)/.test(t) ? 1 : 0;
  const hope = /(can do|we got this|possible|try again|believe)/.test(t) ? 1 : 0;
  return { crisis, hope };
}
