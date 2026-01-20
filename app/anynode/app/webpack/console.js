function getToken(){ return (document.getElementById('admintoken').value || localStorage.getItem('ADMIN_TOKEN') || '').trim(); }
function setToken(t){ localStorage.setItem('ADMIN_TOKEN', t); }

document.getElementById('admintoken').addEventListener('change', e => setToken(e.target.value));

async function goAlive(){
  const t = getToken(); if(!t){ alert('Admin token required'); return; }
  const res = await fetch('/api/viren/alive', { headers: { 'X-Admin-Token': t }});
  const txt = await res.text();
  document.getElementById('virenOut').textContent = txt;
}
document.getElementById('btnAlive').addEventListener('click', goAlive);

async function goAuthorize(){
  const t = getToken(); if(!t){ alert('Admin token required'); return; }
  let body; try{ body = JSON.parse(document.getElementById('virenBody').value); }catch(e){ alert('Bad JSON'); return; }
  const res = await fetch('/api/viren/authorize', { method:'POST', headers: { 'Content-Type':'application/json', 'X-Admin-Token': t }, body: JSON.stringify(body)});
  const txt = await res.text();
  document.getElementById('virenOut').textContent = txt;
}
document.getElementById('btnAuthorize').addEventListener('click', goAuthorize);

async function publishNats(){
  const t = getToken(); if(!t){ alert('Admin token required'); return; }
  const subject = document.getElementById('natsSubject').value.trim();
  let body; try{ body = JSON.parse(document.getElementById('natsBody').value); }catch(e){ alert('Bad JSON'); return; }
  const res = await fetch('/api/nats/publish', { method:'POST', headers:{ 'Content-Type':'application/json', 'X-Admin-Token': t }, body: JSON.stringify({ subject, data: body })});
  document.getElementById('natsOut').textContent = await res.text();
}
document.getElementById('btnPublish').addEventListener('click', publishNats);

let ws;
function subscribeNats(){
  const t = getToken(); if(!t){ alert('Admin token required'); return; }
  const pat = encodeURIComponent(document.getElementById('natsPattern').value.trim());
  if(ws){ ws.close(); }
  ws = new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws/nats?token=${encodeURIComponent(t)}&subject=${pat}`);
  const out = document.getElementById('natsOut');
  ws.onopen = ()=>{ out.textContent += '[ws] connected\n'; };
  ws.onmessage = (e)=>{ out.textContent += e.data + '\n'; out.scrollTop = out.scrollHeight; };
  ws.onerror = ()=>{ out.textContent += '[ws] error\n'; };
  ws.onclose = ()=>{ out.textContent += '[ws] closed\n'; };
}
document.getElementById('btnSubscribe').addEventListener('click', subscribeNats);
