
// Simple DuckDuckGo JSON query
export async function ddgJson(input) {
  const q = typeof input === "string" ? input : (input?.q || "");
  if (!q) return { query: "", results: [] };
  const url = "https://api.duckduckgo.com/?format=json&no_redirect=1&no_html=1&q=" + encodeURIComponent(q);
  const res = await fetch(url);
  if (!res.ok) throw new Error("duckduckgo api error " + res.status);
  const data = await res.json();
  // Very light shaping
  const out = {
    heading: data.Heading || "",
    abstract: data.Abstract || "",
    results: (data.RelatedTopics || []).slice(0, 8).map((t) => ({
      text: t.Text || "",
      url: t.FirstURL || ""
    }))
  };
  return out;
}
