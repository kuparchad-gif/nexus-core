import fetch from "node-fetch";

export async function pushCannon({lokiUrl, orgId, triops="Nexus TriOps", module="gateway", cluster="nexus_gateway", job="viren", envelope}){
  try{
    const ts = `${Date.now()}000000`; // ns
    const body = {
      streams: [{
        stream: { triops, module, cluster, job },
        values: [[ts, JSON.stringify(envelope)]]
      }]
    };
    await fetch(`${lokiUrl}/loki/api/v1/push`, {
      method: "POST",
      headers: { "content-type":"application/json", "X-Scope-OrgID": orgId || "fake" },
      body: JSON.stringify(body)
    });
  }catch(_e){ /* never throw */ }
}

export function mkEnvelope({subject, payload, origin="gateway", band="ops", strength=1.0}){
  return {
    signal_id: crypto.randomUUID(),
    origin, band, strength, subject,
    context: { user: "Chad", trace: crypto.randomUUID(), policies: { retain: "30y" } },
    payload
  };
}
